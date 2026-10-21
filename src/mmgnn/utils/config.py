"""
Configuration utilities for dataset and model configuration.
"""
from typing import Any


# Config for model architecture and training
SUPPORTED_CONV_TYPES = {"gcn", "gin", "gine", "gat", "gine_edge_mlp"}
SUPPORTED_NORM_TYPES = {"batch", "layer", "graph", "pair", "diff_group"}
SUPPORTED_EDGE_PRESENCE_DECODER_TYPES = {"inner_product", "mlp"}
SUPPORTED_EDGE_ATTR_DECODER_TYPES = {"mlp", None}
SUPPORTED_RECON_LOSSES = {"mse", "mae", "l1", "huber", "smooth_l1"}
SUPPORTED_SCHEDULERS = {None, "reduce_on_plateau", "step"}

# Config for dataset
DATASET_REQUIRED_KEYS = {
    "subjects_file": str,
    "dmri_folder": str,
    "fc_folder": str,
    "fc_metric": str,
    "anatomical_df": str | list[str],
    "primary_dmri_key": str,
}

DATASET_OPTIONAL_KEYS = {
    "targets_df": str,
    "edge_attributes": list[str],
    "threshold": float,
    "density": float,
    "top_k": int,
    "log_transform": bool,
    "target_columns": list[str],
    "cache_dir": str,
    "force_reload": bool,
    "num_workers": int,
    "write_cache_payload_debug": bool
}

SUPPORTED_EVALUATION_RECONSTRUCTION_SPLITS = {"train", "val", "test"}
SUPPORTED_EVALUATION_EDGE_PRESENCE_DECODER_TYPES = {"auto", "inner_product", "mlp"}

EVALUATION_DEFAULTS: dict[str, Any] = {
    "save_reconstruction_artifacts": True,
    "reconstruction_split": "test",
    "reconstruction_sample_index": 0,
    "reconstruction_sample_count": 5,
    "reconstruction_sample_seed": 1234,
    "reconstruction_output_subdir": "reconstruction_maps",
    "reconstruction_presence_threshold": 0.5,
    "edge_presence_decoder_type": "auto",
}