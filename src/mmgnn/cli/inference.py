#!/usr/bin/env python3
"""Extract graph-level embeddings from a trained GAE checkpoint for a set of subjects."""

import argparse
import coloredlogs
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from mmgnn.data import BrainGraphDataset, NormalizationStats
from mmgnn.inference.embeddings import (
    extract_embeddings,
    save_subject_node_embeddings_batch,
)
from mmgnn.models.utils import build_encoder, build_model
from mmgnn.training.features import FeatureSelector
from mmgnn.utils.utils import (
    get_dataset_config_help,
    get_model_config_help,
    infer_gae_presence_decoder_type,
    overlay_checkpoint_architecture,
    resolve_normalization_dir,
    validate_dataset_config,
    validate_model_config,
)
from mmgnn.explain.utils import select_graphs


LOGGER = logging.getLogger(__name__)


def argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract graph-level embeddings from a trained GAE checkpoint.",
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
        help=(
            "Path to the GAE configuration JSON file. "
            "\nNote: for inference, this config must include `node_feature_keys` and `edge_feature_keys`."
            "\n%s"
            % get_model_config_help()
        ),
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the saved GAE checkpoint (.pt file).",
    )
    parser.add_argument(
        "--subjects_file",
        type=str,
        required=True,
        help="Path to a text file listing subject IDs to embed (one per line).",
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
        "--output_dir",
        type=str,
        required=True,
        help="Directory where embedding outputs will be saved.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="If set, will print detailed inference progress.",
    )
    return parser


def main() -> None:
    parser = argparser()
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)
        coloredlogs.install(level=logging.INFO)

    with Path(args.dataset_config).open("r", encoding="utf-8") as f:
        dataset_config = json.load(f)
    with Path(args.model_config).open("r", encoding="utf-8") as f:
        model_config = json.load(f)

    dataset_kwargs = validate_dataset_config(dataset_config)
    model_config = validate_model_config(model_config)

    checkpoint_path = Path(args.checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    subjects_file = Path(args.subjects_file)
    if not subjects_file.exists():
        raise FileNotFoundError(f"Subjects file not found: {subjects_file}")

    if model_config["normalize_data"]:
        checkpoint_dir = resolve_normalization_dir(checkpoint_path, args.checkpoint_dir)
    else:
        checkpoint_dir = (
            Path(args.checkpoint_dir) if args.checkpoint_dir is not None else checkpoint_path.parent.parent
        )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(subjects_file, "r", encoding="utf-8") as f:
        subject_ids = [line.strip() for line in f if line.strip()]
    if not subject_ids:
        raise ValueError(f"No subject IDs found in {subjects_file}")
    logging.info("Loaded %d subject IDs from %s", len(subject_ids), subjects_file)

    with open(output_dir / "dataset_config.json", "w", encoding="utf-8") as f:
        json.dump(dataset_kwargs, f, indent=4)
    with open(output_dir / "model_config.json", "w", encoding="utf-8") as f:
        json.dump(model_config, f, indent=4)

    logging.info("Loading the dataset...")
    dataset = BrainGraphDataset(**dataset_kwargs)

    feature_selector = FeatureSelector(
        dataset=dataset,
        node_feature_keys=model_config["node_feature_keys"],
        edge_feature_keys=model_config.get("edge_feature_keys"),
    )
    dataset = feature_selector.subset_dataset()

    graphs = select_graphs(dataset, subject_ids)
    if model_config["normalize_data"]:
        normalizer_path = checkpoint_dir / "normalization_stats.json"
        normalizer = NormalizationStats.load(normalizer_path)
        graphs = [normalizer.transform(graph) for graph in graphs]
        logging.info("Applied normalization statistics from %s", normalizer_path)

    sample_graph = graphs[0]
    num_node_features = sample_graph.x.size(-1)
    sample_edge_attr = getattr(sample_graph, "edge_attr", None)
    num_edge_features = sample_edge_attr.size(-1) if sample_edge_attr is not None else 0

    resolved_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inferred_presence_decoder_type = infer_gae_presence_decoder_type(checkpoint_path, device=resolved_device)

    checkpoint = torch.load(checkpoint_path, map_location=resolved_device, weights_only=False)
    gae_model_config = overlay_checkpoint_architecture(
        model_config,
        checkpoint.get("model_config") if isinstance(checkpoint.get("model_config"), dict) else None,
    )
    gae_model_config["edge_presence_decoder_type"] = inferred_presence_decoder_type

    encoder = build_encoder(
        num_node_features=num_node_features,
        num_edge_features=num_edge_features,
        model_config=gae_model_config,
    )
    model = build_model(
        encoder=encoder,
        num_node_features=num_node_features,
        num_edge_features=num_edge_features,
        model_config=gae_model_config,
    ).to(resolved_device)

    model.load_state_dict(checkpoint["model_state_dict"])
    logging.info("Loaded checkpoint from %s", checkpoint_path)

    embeddings, node_embeddings_by_subject, produced_subject_ids, node_logvar_by_subject, node_sigma_by_subject = (
        extract_embeddings(
            model,
            graphs,
            device=resolved_device,
            batch_size=model_config["batch_size"],
        )
    )

    np.save(output_dir / "embeddings.npy", embeddings)
    with open(output_dir / "subject_ids.json", "w", encoding="utf-8") as f:
        json.dump(produced_subject_ids, f, indent=4)

    subject_summaries = save_subject_node_embeddings_batch(
        output_dir,
        node_embeddings_by_subject,
        node_logvar_by_subject=node_logvar_by_subject,
        node_sigma_by_subject=node_sigma_by_subject,
    )
    with open(output_dir / "subject_summaries.json", "w", encoding="utf-8") as f:
        json.dump(subject_summaries, f, indent=4)

    embedding_columns = [f"embedding_{idx}" for idx in range(embeddings.shape[1])]
    embeddings_df = pd.DataFrame(embeddings, columns=embedding_columns)
    embeddings_df.insert(0, "subject_id", produced_subject_ids)
    embeddings_df.to_csv(output_dir / "embeddings.csv", index=False)

    metadata: dict[str, Any] = {
        "checkpoint_path": str(checkpoint_path),
        "subjects_file": str(subjects_file),
        "num_subjects": len(produced_subject_ids),
        "embedding_dim": int(embeddings.shape[1]),
        "node_embedding_dim": int(next(iter(node_embeddings_by_subject.values())).shape[1]),
        "edge_presence_decoder_type": inferred_presence_decoder_type,
        "model_type": str(gae_model_config.get("model_type", "gae")),
        "saved_uncertainty_maps": node_logvar_by_subject is not None and node_sigma_by_subject is not None,
        "device": str(resolved_device),
    }
    if "epoch" in checkpoint:
        metadata["checkpoint_epoch"] = int(checkpoint["epoch"])
    with open(output_dir / "inference_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    logging.info(
        "Saved graph-level and node embeddings for %d subjects (dim=%d) to %s",
        len(produced_subject_ids),
        embeddings.shape[1],
        output_dir,
    )


if __name__ == "__main__":
    main()
