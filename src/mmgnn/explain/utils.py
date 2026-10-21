"""
Shared utilities for explanation workflows.

The goal of this module is to keep `mmgnn.cli.ablate` thin: CLI should parse args
and delegate to these functions.
"""

from pathlib import Path
from typing import Any, Mapping

import torch
from torch_geometric.data import Data

from mmgnn.data import BrainGraphDataset, NormalizationStats
from mmgnn.models.utils import build_encoder, build_model
from mmgnn.training.features import FeatureSelector
from mmgnn.utils.utils import (
    infer_gae_presence_decoder_type,
    overlay_checkpoint_architecture,
    type_to_help_str,
    validate_dataset_config,
    validate_model_config,
)


ABLATION_DEFAULTS: dict[str, Any] = {
    "seed": 0,
    "max_subjects": None,
    "greedy_topk": 8,
    "run_greedy": True,
}

ABLATION_CONFIG_TYPES: dict[str, object] = {
    "seed": int,
    "max_subjects": "int|None",
    "greedy_topk": int,
    "run_greedy": bool,
}


def get_ablation_config_help() -> str:
    """Return formatted help text for `--ablation_config` JSON files."""
    lines: list[str] = []
    for key in sorted(ABLATION_DEFAULTS):
        type_hint = ABLATION_CONFIG_TYPES.get(key, type(ABLATION_DEFAULTS[key]))
        default = ABLATION_DEFAULTS[key]
        default_repr = "null" if default is None else repr(default)
        lines.append(f"{key}: {type_to_help_str(type_hint)} (default={default_repr})")
    return "Keys are:\n    %s" % "\n    ".join(lines)


def validate_ablation_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    """Validate leave-one-out ablation settings and fill in defaults."""
    resolved = dict(ABLATION_DEFAULTS)
    if config is None:
        return resolved

    unknown_keys = set(config) - set(ABLATION_DEFAULTS)
    if unknown_keys:
        raise ValueError(f"Unknown ablation config keys: {sorted(unknown_keys)}")

    resolved.update(config)

    seed = int(resolved["seed"])
    if seed < 0:
        raise ValueError("seed must be >= 0")
    resolved["seed"] = seed

    max_subjects = resolved["max_subjects"]
    if max_subjects is not None:
        max_subjects = int(max_subjects)
        if max_subjects < 1:
            raise ValueError("max_subjects must be >= 1 when provided")
        resolved["max_subjects"] = max_subjects

    greedy_topk = int(resolved["greedy_topk"])
    if greedy_topk < 1:
        raise ValueError("greedy_topk must be >= 1")
    resolved["greedy_topk"] = greedy_topk

    resolved["run_greedy"] = bool(resolved["run_greedy"])

    return resolved


def load_subject_ids(subjects_file: Path) -> list[str]:
    """Load subject IDs from newline-delimited text file."""
    with open(subjects_file, "r", encoding="utf-8") as handle:
        subject_ids = [line.strip() for line in handle if line.strip()]
    if not subject_ids:
        raise ValueError(f"No subject IDs found in {subjects_file}")
    return subject_ids


def select_graphs(dataset: BrainGraphDataset, subject_ids: list[str]) -> list[Data]:
    """Select graphs for a list of subject IDs (preserves subject_ids order)."""
    subject_to_index = {dataset[i].subject_id: i for i in range(len(dataset))}
    missing = [subject_id for subject_id in subject_ids if subject_id not in subject_to_index]
    if missing:
        preview = ", ".join(missing[:5])
        suffix = "..." if len(missing) > 5 else ""
        raise ValueError(f"Subjects not found in dataset ({len(missing)}): {preview}{suffix}")
    return [dataset[subject_to_index[subject_id]].clone() for subject_id in subject_ids]


def load_frozen_gae(
    *,
    checkpoint_path: Path,
    model_config: Mapping[str, Any],
    sample_graph: Data,
    device: torch.device,
) -> tuple[torch.nn.Module, str, dict[str, Any]]:
    """Load a GAE/VGAE checkpoint, freeze parameters, and return inferred decoder type + checkpoint metadata."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    inferred_presence_decoder_type = infer_gae_presence_decoder_type(checkpoint_path, device=device)

    num_node_features = int(sample_graph.x.size(-1))
    sample_edge_attr = getattr(sample_graph, "edge_attr", None)
    num_edge_features = int(sample_edge_attr.size(-1)) if sample_edge_attr is not None else 0

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
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    metadata: dict[str, Any] = {
        "model_type": str(gae_model_config.get("model_type", "gae")),
    }
    if "epoch" in checkpoint:
        metadata["checkpoint_epoch"] = int(checkpoint["epoch"])
    return model, inferred_presence_decoder_type, metadata


def prepare_graphs_for_explanation(
    *,
    dataset_config: Mapping[str, Any],
    model_config: Mapping[str, Any],
    checkpoint_dir: Path,
    subject_ids: list[str],
) -> tuple[dict[str, Any], dict[str, Any], list[Data]]:
    """Load dataset and return graphs subset with feature selection + optional normalization applied."""
    dataset_kwargs = validate_dataset_config(dataset_config)
    validated_model_config = validate_model_config(model_config)

    dataset = BrainGraphDataset(**dataset_kwargs)
    feature_selector = FeatureSelector(
        dataset=dataset,
        node_feature_keys=validated_model_config["node_feature_keys"],
        edge_feature_keys=validated_model_config.get("edge_feature_keys"),
    )
    dataset = feature_selector.subset_dataset()

    graphs = select_graphs(dataset, subject_ids)
    if validated_model_config["normalize_data"]:
        normalizer_path = checkpoint_dir / "normalization_stats.json"
        normalizer = NormalizationStats.load(normalizer_path)
        graphs = [normalizer.transform(graph) for graph in graphs]

    return dataset_kwargs, validated_model_config, graphs
