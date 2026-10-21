"""Unit tests for mmgnn.utils.utils validation helpers."""

from pathlib import Path
from typing import Any

import pytest
import torch

from mmgnn.utils.utils import (
    get_dataset_config_help,
    get_model_config_help,
    infer_gae_presence_decoder_type,
    overlay_checkpoint_architecture,
    resolve_normalization_dir,
    type_to_help_str,
    validate_dataset_config,
    validate_evaluation_config,
    validate_model_config,
)


def minimal_dataset_cfg(**overrides: Any) -> dict[str, Any]:
    cfg = {
        "subjects_file": "subjects.txt",
        "dmri_folder": "/tmp/dmri",
        "fc_folder": "/tmp/fc",
        "fc_metric": "pearsoncorrelation",
        "anatomical_df": "/tmp/anat.csv",
        "primary_dmri_key": "fa",
        "edge_attributes": ["fa"],
    }
    cfg.update(overrides)
    return cfg


def minimal_model_cfg(**overrides: Any) -> dict[str, Any]:
    cfg = {
        "hidden_dim": 8,
        "node_feature_keys": ["n0"],
        "edge_feature_keys": ["e0"],
    }
    cfg.update(overrides)
    return cfg


class TestValidateDatasetConfig:
    def test_happy_path_fills_defaults(self):
        resolved = validate_dataset_config(minimal_dataset_cfg())
        assert resolved["subjects_file"] == "subjects.txt"
        assert resolved["fc_metric"] == "pearsoncorrelation"
        assert "force_reload" in resolved
        assert "num_workers" in resolved

    def test_missing_required_raises(self):
        cfg = minimal_dataset_cfg()
        del cfg["fc_metric"]
        with pytest.raises(KeyError, match="Missing required dataset"):
            validate_dataset_config(cfg)

    def test_unknown_key_raises(self):
        with pytest.raises(KeyError, match="Unsupported dataset config"):
            validate_dataset_config(minimal_dataset_cfg(not_a_key=1))

    def test_primary_key_must_appear_in_edge_attributes(self):
        with pytest.raises(ValueError, match="must appear in edge_attributes"):
            validate_dataset_config(minimal_dataset_cfg(edge_attributes=[]))
        with pytest.raises(ValueError, match="must appear in edge_attributes"):
            validate_dataset_config(minimal_dataset_cfg(edge_attributes=["md"]))


class TestValidateModelConfig:
    def test_happy_path(self):
        resolved = validate_model_config(minimal_model_cfg())
        assert resolved["hidden_dim"] == 8
        assert resolved["node_feature_keys"] == ["n0"]
        assert resolved["conv_type"] == "gine"
        assert resolved["model_type"] == "gae"

    def test_coerce_comma_separated_keys(self):
        resolved = validate_model_config(
            minimal_model_cfg(
                node_feature_keys="a, b",
                edge_feature_keys="x,y",
            )
        )
        assert resolved["node_feature_keys"] == ["a", "b"]
        assert resolved["edge_feature_keys"] == ["x", "y"]

    def test_bad_conv_type_raises(self):
        with pytest.raises(ValueError, match="Unsupported conv_type"):
            validate_model_config(minimal_model_cfg(conv_type="not_a_conv"))

    def test_empty_node_keys_raises(self):
        with pytest.raises(ValueError, match="node_feature_keys must be a non-empty"):
            validate_model_config(minimal_model_cfg(node_feature_keys=[]))

    def test_missing_hidden_dim_raises(self):
        cfg = minimal_model_cfg()
        del cfg["hidden_dim"]
        with pytest.raises(KeyError, match="hidden_dim"):
            validate_model_config(cfg)

    def test_gat_defaults_num_heads(self):
        resolved = validate_model_config(minimal_model_cfg(conv_type="gat", num_heads=None))
        assert resolved["num_heads"] == 4


class TestValidateEvaluationConfig:
    def test_defaults_from_empty(self):
        resolved = validate_evaluation_config({})
        assert resolved["reconstruction_split"] == "test"
        assert resolved["edge_presence_decoder_type"] == "auto"

    def test_bad_split_raises(self):
        with pytest.raises(ValueError, match="reconstruction_split"):
            validate_evaluation_config({"reconstruction_split": "holdout"})

    def test_threshold_out_of_range_raises(self):
        with pytest.raises(ValueError, match="reconstruction_presence_threshold"):
            validate_evaluation_config({"reconstruction_presence_threshold": 1.5})

    def test_sample_count_too_small_raises(self):
        with pytest.raises(ValueError, match="reconstruction_sample_count"):
            validate_evaluation_config({"reconstruction_sample_count": 0})


class TestHelpText:
    def test_dataset_help(self):
        help_text = get_dataset_config_help()
        assert "Required keys are:" in help_text
        assert "subjects_file" in help_text

    def test_model_help(self):
        help_text = get_model_config_help()
        assert "Required keys are:" in help_text
        assert "hidden_dim" in help_text


class TestTypeToHelpStr:
    def test_none_and_type(self):
        assert type_to_help_str(None) == "None"
        assert type_to_help_str(int) == "int"


class TestInferPresenceDecoder:
    def test_from_model_config(self, tmp_path: Path):
        path = tmp_path / "ckpt.pt"
        torch.save({"model_config": {"edge_presence_decoder_type": "mlp"}}, path)
        assert infer_gae_presence_decoder_type(path) == "mlp"

    def test_from_state_dict_mlp_keys(self, tmp_path: Path):
        path = tmp_path / "ckpt.pt"
        torch.save(
            {"model_state_dict": {"edge_presence_decoder.mlp.0.weight": torch.zeros(1)}},
            path,
        )
        assert infer_gae_presence_decoder_type(path) == "mlp"

    def test_defaults_to_inner_product(self, tmp_path: Path):
        path = tmp_path / "ckpt.pt"
        torch.save({"model_state_dict": {"encoder.layers.0.conv.lin.weight": torch.zeros(1)}}, path)
        assert infer_gae_presence_decoder_type(path) == "inner_product"


class TestOverlayArchitecture:
    def test_overlays_non_none_keys(self):
        base = {"model_type": "gae", "kl_weight": 1.0, "node_out_dim": 3}
        arch = {"model_type": "vgae", "kl_weight": 0.5, "node_out_dim": None}
        resolved = overlay_checkpoint_architecture(base, arch)
        assert resolved["model_type"] == "vgae"
        assert resolved["kl_weight"] == 0.5
        assert resolved["node_out_dim"] == 3  # None in arch is skipped

    def test_none_architecture_is_noop(self):
        base = {"model_type": "gae"}
        assert overlay_checkpoint_architecture(base, None) == base


def _write_stats(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "normalization_stats.json").write_text("{}", encoding="utf-8")


class TestResolveNormalizationDir:
    def test_explicit_directory(self, tmp_path: Path):
        checkpoint = tmp_path / "checkpoints" / "model.pt"
        checkpoint.parent.mkdir()
        explicit = tmp_path / "given"
        assert resolve_normalization_dir(checkpoint, explicit) == explicit

    def test_stats_next_to_checkpoint(self, tmp_path: Path):
        checkpoint = tmp_path / "run" / "model.pt"
        checkpoint.parent.mkdir()
        _write_stats(checkpoint.parent)
        assert resolve_normalization_dir(checkpoint, None) == checkpoint.parent

    def test_stats_in_training_run_directory(self, tmp_path: Path):
        checkpoint = tmp_path / "run" / "checkpoints" / "model.pt"
        checkpoint.parent.mkdir(parents=True)
        _write_stats(tmp_path / "run")
        assert resolve_normalization_dir(checkpoint, None) == tmp_path / "run"

    def test_both_locations_raise(self, tmp_path: Path):
        checkpoint = tmp_path / "run" / "checkpoints" / "model.pt"
        checkpoint.parent.mkdir(parents=True)
        _write_stats(checkpoint.parent)
        _write_stats(tmp_path / "run")
        with pytest.raises(ValueError, match="--checkpoint_dir"):
            resolve_normalization_dir(checkpoint, None)

    def test_missing_stats_keeps_run_directory(self, tmp_path: Path):
        checkpoint = tmp_path / "run" / "checkpoints" / "model.pt"
        checkpoint.parent.mkdir(parents=True)
        assert resolve_normalization_dir(checkpoint, None) == tmp_path / "run"
