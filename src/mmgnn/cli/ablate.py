#!/usr/bin/env python3
"""Leave-one-out ablation of GAE latent dimensions for reconstruction importance."""

import argparse
import coloredlogs
import json
import logging
from pathlib import Path
from typing import Any

import torch

from mmgnn.explain.runner import run_ablation
from mmgnn.explain.utils import (
    get_ablation_config_help,
    load_subject_ids,
    prepare_graphs_for_explanation,
    validate_ablation_config,
)
from mmgnn.utils.utils import (
    get_dataset_config_help,
    get_model_config_help,
    resolve_normalization_dir,
)


LOGGER = logging.getLogger(__name__)


def argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Ablate each latent embedding dimension (mean-replace) and measure "
            "the change in frozen-decoder reconstruction loss."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        required=True,
        help="Path to the dataset configuration JSON file.\n%s" % get_dataset_config_help(),
    )
    parser.add_argument(
        "--model_config",
        type=str,
        required=True,
        help="Path to the GAE configuration JSON file.\n%s" % get_model_config_help(),
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the saved GAE checkpoint (.pt file).",
    )
    parser.add_argument(
        "--embeddings_dir",
        type=str,
        required=True,
        help=(
            "Directory containing node embeddings from mmgnn.cli.inference "
            "(expects subjects/<id>/node_embeddings.npy)."
        ),
    )
    parser.add_argument(
        "--subjects_file",
        type=str,
        required=True,
        help="Path to a text file listing subject IDs to ablate (one per line).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where ablation outputs will be saved.",
    )
    parser.add_argument(
        "--ablation_config",
        type=str,
        default=None,
        help=(
            "Optional JSON file with ablation settings.\n%s"
            % get_ablation_config_help()
        ),
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help=(
            "Directory containing normalization_stats.json. "
            "If omitted and normalization is enabled, the directory next to the "
            "checkpoint or its parent is used when that file is present; otherwise "
            "the training run directory (two levels above the checkpoint) is used."
        ),
    )
    parser.add_argument(
        "--greedy",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Whether to run sequential greedy multi-dim ablation (O(D^2) per subject). "
            "Default is enabled. Use --no-greedy to skip."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="If set, will print detailed ablation progress.",
    )
    return parser


def main() -> None:
    parser = argparser()
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)
        coloredlogs.install(level=logging.INFO)

    with Path(args.dataset_config).open("r", encoding="utf-8") as handle:
        dataset_config = json.load(handle)
    with Path(args.model_config).open("r", encoding="utf-8") as handle:
        model_config = json.load(handle)

    ablation_config_input = None
    if args.ablation_config is not None:
        ablation_config_path = Path(args.ablation_config)
        if not ablation_config_path.exists():
            raise FileNotFoundError(f"Ablation config not found: {ablation_config_path}")
        with open(ablation_config_path, "r", encoding="utf-8") as handle:
            ablation_config_input = json.load(handle)

    ablation_config = validate_ablation_config(ablation_config_input)
    # CLI flag wins over ablation_config JSON.
    ablation_config["run_greedy"] = bool(args.greedy)

    checkpoint_path = Path(args.checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    embeddings_dir = Path(args.embeddings_dir)
    if not embeddings_dir.exists():
        raise FileNotFoundError(f"Embeddings directory not found: {embeddings_dir}")

    subjects_file = Path(args.subjects_file)
    if not subjects_file.exists():
        raise FileNotFoundError(f"Subjects file not found: {subjects_file}")

    if model_config.get("normalize_data", True):
        checkpoint_dir = resolve_normalization_dir(checkpoint_path, args.checkpoint_dir)
    else:
        checkpoint_dir = (
            Path(args.checkpoint_dir) if args.checkpoint_dir is not None else checkpoint_path.parent.parent
        )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    subject_ids = load_subject_ids(subjects_file)
    max_subjects = ablation_config["max_subjects"]
    if max_subjects is not None:
        subject_ids = subject_ids[: int(max_subjects)]
    logging.info("Ablating %d subject(s) from %s", len(subject_ids), subjects_file)

    dataset_kwargs, validated_model_config, graphs = prepare_graphs_for_explanation(
        dataset_config=dataset_config,
        model_config=model_config,
        checkpoint_dir=checkpoint_dir,
        subject_ids=subject_ids,
    )
    resolved_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(output_dir / "dataset_config.json", "w", encoding="utf-8") as handle:
        json.dump(dataset_kwargs, handle, indent=4)
    with open(output_dir / "model_config.json", "w", encoding="utf-8") as handle:
        json.dump(validated_model_config, handle, indent=4)
    with open(output_dir / "ablation_config.json", "w", encoding="utf-8") as handle:
        json.dump(ablation_config, handle, indent=4)

    subject_summaries, inferred_presence_decoder_type, checkpoint_meta = run_ablation(
        graphs=graphs,
        checkpoint_path=checkpoint_path,
        validated_model_config=validated_model_config,
        embeddings_dir=embeddings_dir,
        ablation_config=ablation_config,
        output_dir=output_dir,
        device=resolved_device,
    )

    with open(output_dir / "subject_summaries.json", "w", encoding="utf-8") as handle:
        json.dump(subject_summaries, handle, indent=4)

    metadata: dict[str, Any] = {
        "checkpoint_path": str(checkpoint_path),
        "embeddings_dir": str(embeddings_dir),
        "subjects_file": str(subjects_file),
        "num_subjects": len(subject_ids),
        "seed": int(ablation_config["seed"]),
        "greedy_topk": int(ablation_config["greedy_topk"]),
        "run_greedy": bool(ablation_config["run_greedy"]),
        "fill_mode": "mean",
        "edge_presence_decoder_type": inferred_presence_decoder_type,
        "device": str(resolved_device),
        "method": "leave_one_out_mean_replace",
    }
    metadata.update(checkpoint_meta)
    with open(output_dir / "ablation_metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=4)

    logging.info(
        "Saved latent-dimension ablation for %d subject(s) to %s",
        len(subject_ids),
        output_dir,
    )


if __name__ == "__main__":
    main()
