"""
Utility functions for validating and normalizing configuration.
"""

import inspect
from pathlib import Path
from typing import Mapping, Any

import torch
from mmgnn.data import BrainGraphDataset
from mmgnn.data.dataset import _require_primary_edge_attribute
from mmgnn.models import Encoder, GAE
from mmgnn.models.blocks import ACT_FUNC
from mmgnn.utils.config import (
    DATASET_OPTIONAL_KEYS,
    DATASET_REQUIRED_KEYS,
    SUPPORTED_CONV_TYPES,
    SUPPORTED_NORM_TYPES,
    SUPPORTED_EDGE_PRESENCE_DECODER_TYPES,
    SUPPORTED_EDGE_ATTR_DECODER_TYPES,
    SUPPORTED_RECON_LOSSES,
    SUPPORTED_SCHEDULERS,
    SUPPORTED_EVALUATION_RECONSTRUCTION_SPLITS,
    SUPPORTED_EVALUATION_EDGE_PRESENCE_DECODER_TYPES,
    EVALUATION_DEFAULTS,
)


def _signature_defaults(callable_obj: Any) -> dict[str, Any]:
    """Return a `name -> default` mapping for a callable signature."""
    defaults: dict[str, Any] = {}
    for name, param in inspect.signature(callable_obj).parameters.items():
        if name == "self":
            continue
        if param.default is not inspect._empty:
            defaults[name] = param.default
    return defaults


def _allowed_keys(callable_obj: Any, extra_keys: set[str] | None = None) -> set[str]:
    allowed = set(inspect.signature(callable_obj).parameters)
    allowed.discard("self")
    if extra_keys:
        allowed.update(extra_keys)
    return allowed


def _validate_known_keys(
    raw_config: Mapping[str, Any],
    allowed_keys: set[str],
    section_name: str,
) -> None:
    unknown_keys = sorted(set(raw_config) - allowed_keys)
    if unknown_keys:
        raise KeyError(
            f"Unsupported {section_name} config key(s): {unknown_keys}. "
            f"Supported keys are: {sorted(allowed_keys)}"
        )


def _coerce_mapping(
    raw_config: Mapping[str, Any],
    defaults: Mapping[str, Any],
    *,
    required_keys: set[str] | None = None,
    section_name: str,
) -> dict[str, Any]:
    required_keys = required_keys or set()
    missing = sorted(key for key in required_keys if key not in raw_config or raw_config[key] is None)
    if missing:
        raise KeyError(f"Missing required {section_name} config key(s): {missing}")

    resolved = dict(defaults)
    resolved.update(raw_config)
    return resolved


def get_dataset_config_options() -> dict[str, Any]:
    """Return all dataset constructor options with their defaults."""
    return dict(_signature_defaults(BrainGraphDataset.__init__))


def get_evaluation_config_options() -> dict[str, Any]:
    """Return evaluation configuration options with their defaults."""
    return dict(EVALUATION_DEFAULTS)


def get_encoder_config_options() -> dict[str, Any]:
    """Return all encoder constructor options with their defaults."""
    defaults = _signature_defaults(Encoder.__init__)
    defaults.pop("num_node_features", None)
    defaults.pop("num_edge_features", None)
    return defaults


def get_gae_config_options() -> dict[str, Any]:
    """Return all GAE constructor options with their defaults."""
    defaults = _signature_defaults(GAE.__init__)
    defaults.pop("encoder", None)
    return defaults


def validate_dataset_config(raw_config: Mapping[str, Any]) -> dict[str, Any]:
    """Split and validate dataset-related configuration.

    The returned tuple contains:
    - dataset_kwargs: arguments passed to `BrainGraphDataset`
    """
    dataset_defaults = get_dataset_config_options()
    dataset_allowed = _allowed_keys(BrainGraphDataset.__init__)
    _validate_known_keys(raw_config, dataset_allowed, "dataset")

    dataset_kwargs = {
        key: raw_config[key]
        for key in raw_config
        if key in dataset_allowed
    }
    dataset_kwargs = _coerce_mapping(
        dataset_kwargs,
        dataset_defaults,
        required_keys=DATASET_REQUIRED_KEYS,
        section_name="dataset",
    )
    _require_primary_edge_attribute(
        dataset_kwargs["primary_dmri_key"],
        dataset_kwargs.get("edge_attributes"),
    )

    return dataset_kwargs


def validate_model_config(raw_config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the GAE/training configuration and fill in defaults."""
    encoder_defaults = get_encoder_config_options()
    gae_defaults = get_gae_config_options()

    training_defaults: dict[str, Any] = {
        "normalize_data": True,
        "normalize_nodes": True,
        "normalize_edges": True,
        "normalize_edge_weights": True,
        "batch_size": 32,
        "num_epochs": 100,
        "learning_rate": 1e-3,
        "weight_decay": 1e-5,
        "scheduler": None,
        "patience": 20,
        "delta": 0.0,
        "optimizer_param_groups": None,
        "neg_sampling_ratio": 1.0,
        "recon_edge_presence_weight": 1.0,
        "recon_node_attr_weight": 1.0,
        "recon_edge_attr_weight": 1.0,
        "recon_node_loss_type": "mse",
        "recon_edge_attr_loss_type": "mse",
        "model_type": "gae",
        "kl_weight": 1.0,
    }

    allowed_keys = (
        (_allowed_keys(Encoder.__init__) - {"num_node_features", "num_edge_features"})
        | (_allowed_keys(GAE.__init__) - {"encoder"})
        | set(training_defaults)
        | {"edge_attr_out_dim"}
        | {"edge_feature_keys"}
        | {"node_feature_keys"}
        | {"test_subjects"}
        | {"val_subjects"}
        | {"train_subjects"}
        | {"model_type"}
        | {"kl_weight"}
    )

    _validate_known_keys(raw_config, allowed_keys, "model")

    config = dict(training_defaults)
    config.update(gae_defaults)
    config.update(encoder_defaults)
    config.update({key: raw_config[key] for key in raw_config if key in allowed_keys})

    required_keys = {"node_feature_keys", "edge_feature_keys"}
    missing_required = sorted(key for key in required_keys if key not in raw_config or raw_config[key] is None)
    if missing_required:
        raise KeyError(f"Missing required model config key(s): {missing_required}")

    def _coerce_string_list(value: Any, *, config_key: str) -> list[str]:
        if isinstance(value, str):
            # Allow comma-separated strings for convenience.
            items = [v.strip() for v in value.split(",")]
            return [v for v in items if v]
        if isinstance(value, (list, tuple)):
            items = [str(v).strip() for v in value]
            return [v for v in items if v]
        raise TypeError(f"Expected `{config_key}` to be a list[str] (or comma-separated string), got {type(value)}")

    config["node_feature_keys"] = _coerce_string_list(config["node_feature_keys"], config_key="node_feature_keys")
    config["edge_feature_keys"] = _coerce_string_list(config["edge_feature_keys"], config_key="edge_feature_keys")
    if not config["node_feature_keys"]:
        raise ValueError("node_feature_keys must be a non-empty list")
    if not config["edge_feature_keys"]:
        raise ValueError("edge_feature_keys must be a non-empty list")

    # Validation and normalization of supported values.
    if "hidden_dim" not in config:
        raise KeyError("Missing required model config key: ['hidden_dim']")
    config["hidden_dim"] = int(config["hidden_dim"])
    if config["hidden_dim"] <= 0:
        raise ValueError("hidden_dim must be a positive integer")
    config["num_layers"] = int(config["num_layers"])
    if config["num_layers"] <= 0:
        raise ValueError("num_layers must be a positive integer")
    config["batch_size"] = int(config["batch_size"])
    if config["batch_size"] <= 0:
        raise ValueError("batch_size must be a positive integer")
    config["num_epochs"] = int(config["num_epochs"])
    if config["num_epochs"] <= 0:
        raise ValueError("num_epochs must be a positive integer")
    config["learning_rate"] = float(config["learning_rate"])
    if config["learning_rate"] <= 0:
        raise ValueError("learning_rate must be positive")
    config["weight_decay"] = float(config["weight_decay"])
    if config["weight_decay"] < 0:
        raise ValueError("weight_decay must be non-negative")
    config["patience"] = int(config["patience"])
    if config["patience"] <= 0:
        raise ValueError("patience must be a positive integer")
    config["delta"] = float(config["delta"])
    if config["delta"] < 0:
        raise ValueError("delta must be non-negative")
    config["recon_edge_presence_weight"] = float(config["recon_edge_presence_weight"])
    if config["recon_edge_presence_weight"] < 0:
        raise ValueError("recon_edge_presence_weight must be non-negative")
    config["recon_node_attr_weight"] = float(config["recon_node_attr_weight"])
    if config["recon_node_attr_weight"] < 0:
        raise ValueError("recon_node_attr_weight must be non-negative")
    config["recon_edge_attr_weight"] = float(config["recon_edge_attr_weight"])
    if config["recon_edge_attr_weight"] < 0:
        raise ValueError("recon_edge_attr_weight must be non-negative")

    conv_type = str(config["conv_type"]).lower()
    if conv_type not in SUPPORTED_CONV_TYPES:
        raise ValueError(f"Unsupported conv_type '{conv_type}'. Supported values: {sorted(SUPPORTED_CONV_TYPES)}")
    config["conv_type"] = conv_type

    act_func = str(config["act_func"]).lower()
    if act_func not in ACT_FUNC:
        raise ValueError(f"Unsupported act_func '{act_func}'. Supported values: {sorted(ACT_FUNC)}")
    config["act_func"] = act_func

    norm_type = str(config["norm_type"]).lower()
    if norm_type not in SUPPORTED_NORM_TYPES:
        raise ValueError(f"Unsupported norm_type '{norm_type}'. Supported values: {sorted(SUPPORTED_NORM_TYPES)}")
    config["norm_type"] = norm_type

    if config["scheduler"] not in SUPPORTED_SCHEDULERS:
        raise ValueError(
            f"Unsupported scheduler '{config['scheduler']}'. Supported values: {sorted(x for x in SUPPORTED_SCHEDULERS if x is not None)}"
        )

    edge_presence_decoder_type = str(config["edge_presence_decoder_type"]).lower()
    if edge_presence_decoder_type not in SUPPORTED_EDGE_PRESENCE_DECODER_TYPES:
        raise ValueError(
            f"Unsupported edge_presence_decoder_type '{edge_presence_decoder_type}'. "
            f"Supported values: {sorted(SUPPORTED_EDGE_PRESENCE_DECODER_TYPES)}"
        )
    config["edge_presence_decoder_type"] = edge_presence_decoder_type

    edge_attr_decoder_type = config["edge_attr_decoder_type"]
    if edge_attr_decoder_type is not None:
        edge_attr_decoder_type = str(edge_attr_decoder_type).lower()
    if edge_attr_decoder_type not in SUPPORTED_EDGE_ATTR_DECODER_TYPES:
        raise ValueError(
            f"Unsupported edge_attr_decoder_type '{edge_attr_decoder_type}'. "
            f"Supported values: {sorted(x for x in SUPPORTED_EDGE_ATTR_DECODER_TYPES if x is not None)} or null"
        )
    config["edge_attr_decoder_type"] = edge_attr_decoder_type

    for key in ("recon_node_loss_type", "recon_edge_attr_loss_type"):
        loss_type = str(config[key]).lower()
        if loss_type not in SUPPORTED_RECON_LOSSES:
            raise ValueError(f"Unsupported {key} '{loss_type}'. Supported values: {sorted(SUPPORTED_RECON_LOSSES)}")
        config[key] = loss_type

    model_type = str(config.get("model_type", "gae")).lower()
    if model_type not in {"gae", "vgae"}:
        raise ValueError(f"Unsupported model_type '{model_type}'. Supported values: 'gae', 'vgae'")
    config["model_type"] = model_type

    config["kl_weight"] = float(config.get("kl_weight", 1.0))
    if config["kl_weight"] < 0:
        raise ValueError("kl_weight must be non-negative")

    if config.get("num_heads") is not None:
        config["num_heads"] = int(config["num_heads"])
        if config["num_heads"] <= 0:
            raise ValueError("num_heads must be positive when provided")
    elif conv_type == "gat":
        config["num_heads"] = 4

    if config.get("edge_dropout") is None:
        config["edge_dropout"] = 0.0
    else:
        config["edge_dropout"] = float(config["edge_dropout"])
    if config.get("dropout") is None:
        config["dropout"] = 0.0
    else:
        config["dropout"] = float(config["dropout"])

    node_out_dim = config.get("node_out_dim")
    if node_out_dim is not None:
        config["node_out_dim"] = int(node_out_dim)
        if config["node_out_dim"] <= 0:
            raise ValueError("node_out_dim must be positive when provided")

    edge_attr_out_dim = config.get("edge_attr_out_dim")
    if edge_attr_out_dim is not None:
        config["edge_attr_out_dim"] = int(edge_attr_out_dim)
        if config["edge_attr_out_dim"] <= 0:
            raise ValueError("edge_attr_out_dim must be positive when provided")

    decoder_hidden_dim = config.get("decoder_hidden_dim")
    if decoder_hidden_dim is not None:
        config["decoder_hidden_dim"] = int(decoder_hidden_dim)
        if config["decoder_hidden_dim"] <= 0:
            raise ValueError("decoder_hidden_dim must be positive when provided")

    return config


def validate_evaluation_config(raw_config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate evaluation-specific configuration and fill in defaults."""
    _validate_known_keys(raw_config, set(EVALUATION_DEFAULTS), "evaluation")

    config = dict(EVALUATION_DEFAULTS)
    config.update({key: raw_config[key] for key in raw_config if key in EVALUATION_DEFAULTS})

    reconstruction_split = str(config["reconstruction_split"]).lower().strip()
    if reconstruction_split not in SUPPORTED_EVALUATION_RECONSTRUCTION_SPLITS:
        raise ValueError(
            f"Unsupported reconstruction_split '{reconstruction_split}'. "
            f"Supported values: {sorted(SUPPORTED_EVALUATION_RECONSTRUCTION_SPLITS)}"
        )
    config["reconstruction_split"] = reconstruction_split

    edge_presence_decoder_type = str(config["edge_presence_decoder_type"]).lower()
    if edge_presence_decoder_type not in SUPPORTED_EVALUATION_EDGE_PRESENCE_DECODER_TYPES:
        raise ValueError(
            f"Unsupported edge_presence_decoder_type '{edge_presence_decoder_type}'. "
            f"Supported values: {sorted(SUPPORTED_EVALUATION_EDGE_PRESENCE_DECODER_TYPES)}"
        )
    config["edge_presence_decoder_type"] = edge_presence_decoder_type

    config["reconstruction_sample_index"] = int(config["reconstruction_sample_index"])
    config["reconstruction_sample_count"] = int(config["reconstruction_sample_count"])
    if config["reconstruction_sample_count"] < 1:
        raise ValueError("reconstruction_sample_count must be at least 1")

    seed = config["reconstruction_sample_seed"]
    config["reconstruction_sample_seed"] = int(seed) if seed is not None else None

    config["reconstruction_presence_threshold"] = float(config["reconstruction_presence_threshold"])
    if not 0.0 <= config["reconstruction_presence_threshold"] <= 1.0:
        raise ValueError("reconstruction_presence_threshold must be between 0 and 1")

    config["save_reconstruction_artifacts"] = bool(config["save_reconstruction_artifacts"])
    config["reconstruction_output_subdir"] = str(config["reconstruction_output_subdir"])

    return config


def type_to_help_str(type_hint: object) -> str:
    """Best-effort human-readable type formatting for CLI help text."""
    if type_hint is None:
        return "None"
    if isinstance(type_hint, type):
        return type_hint.__name__
    return str(type_hint)


def _default_repr(value: Any) -> str:
    if value is None:
        return "null"
    return repr(value)


def _infer_type_hint(default: Any) -> object:
    if default is None:
        return "None"
    if isinstance(default, list) and all(isinstance(item, str) for item in default):
        return "list[str]"
    if isinstance(default, type):
        return default
    return type(default)


def _config_help_line(
    key: str,
    type_hint: object,
    default: Any,
    *,
    required_without_default: bool = False,
) -> str:
    if required_without_default:
        return f"{key}: {type_to_help_str(type_hint)} (default=required)"
    return f"{key}: {type_to_help_str(type_hint)} (default={_default_repr(default)})"


def get_dataset_config_help() -> str:
    """Generate required/optional dataset config help with types and defaults."""
    dataset_defaults = get_dataset_config_options()
    required_key_set = set(DATASET_REQUIRED_KEYS)

    required_lines = [
        _config_help_line(
            key,
            DATASET_REQUIRED_KEYS[key],
            dataset_defaults.get(key),
            required_without_default=True,
        )
        for key in sorted(DATASET_REQUIRED_KEYS)
    ]

    optional_lines = []
    for key in sorted(dataset_defaults):
        if key in required_key_set:
            continue
        default = dataset_defaults[key]
        type_hint = DATASET_OPTIONAL_KEYS.get(key, _infer_type_hint(default))
        optional_lines.append(_config_help_line(key, type_hint, default))

    return "Required keys are:\n    %s\nOptional keys are:\n    %s" % (
        "\n    ".join(required_lines),
        "\n    ".join(optional_lines),
    )


def get_model_config_help() -> str:
    """Generate required/optional model config help with types and defaults."""
    validated_defaults = validate_model_config(
        {"hidden_dim": 1, "node_feature_keys": ["_"], "edge_feature_keys": ["_"]}
    )
    required_types = {
        "hidden_dim": int,
        "node_feature_keys": "list[str]",
        "edge_feature_keys": "list[str]",
    }

    required_lines = [
        _config_help_line(key, required_types[key], None, required_without_default=True)
        for key in sorted(required_types)
    ]

    optional_lines = []
    for key in sorted(validated_defaults):
        if key in required_types:
            continue
        default = validated_defaults[key]
        optional_lines.append(_config_help_line(key, _infer_type_hint(default), default))

    return "Required keys are:\n    %s\nOptional keys are:\n    %s" % (
        "\n    ".join(required_lines),
        "\n    ".join(optional_lines),
    )

def infer_gae_presence_decoder_type(
    checkpoint_path: Path,
    device: torch.device = torch.device("cpu"),
) -> str:
    """Infer the edge-presence decoder architecture stored in a GAE checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_config = checkpoint.get("model_config")
    if isinstance(model_config, dict) and "edge_presence_decoder_type" in model_config:
        return str(model_config["edge_presence_decoder_type"])

    state_dict = checkpoint.get("model_state_dict", {})
    if any(key.startswith("edge_presence_decoder.mlp") for key in state_dict):
        return "mlp"
    return "inner_product"


def overlay_checkpoint_architecture(
    model_config: Mapping[str, Any],
    architecture: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Overlay rebuild-critical architecture fields from a checkpoint onto a model config."""
    resolved = dict(model_config)
    if not isinstance(architecture, Mapping):
        return resolved
    for key in (
        "model_type",
        "kl_weight",
        "edge_presence_decoder_type",
        "edge_attr_out_dim",
        "edge_attr_decoder_type",
        "decode_node_features",
        "node_out_dim",
    ):
        if key in architecture and architecture[key] is not None:
            resolved[key] = architecture[key]
    return resolved


def resolve_normalization_dir(checkpoint_path: Path, checkpoint_dir: str | Path | None) -> Path:
    """Locate the directory that holds ``normalization_stats.json``.

    An explicit directory is returned unchanged. Otherwise the checkpoint's
    parent is preferred over its grandparent when exactly one of them contains
    the stats file. If neither does, the grandparent is returned so the caller
    still fails on the historical default path. Both matches require
    ``--checkpoint_dir``.
    """
    if checkpoint_dir is not None:
        return Path(checkpoint_dir)

    candidates = [checkpoint_path.parent, checkpoint_path.parent.parent]
    found = [path for path in candidates if (path / "normalization_stats.json").is_file()]
    if len(found) > 1:
        raise ValueError(
            "normalization_stats.json was found next to the checkpoint and in the "
            "training run directory. Pass --checkpoint_dir to choose one."
        )
    if len(found) == 1:
        return found[0]
    return checkpoint_path.parent.parent
