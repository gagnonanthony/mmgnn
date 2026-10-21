"""Unit tests for mmgnn.explain.utils."""

from pathlib import Path

import pytest
import torch

from torch_geometric.data import Data

from mmgnn.data.normalization import NormalizationStats
from mmgnn.explain.utils import (
    ABLATION_DEFAULTS,
    get_ablation_config_help,
    load_frozen_gae,
    load_subject_ids,
    prepare_graphs_for_explanation,
    select_graphs,
    validate_ablation_config,
)
from mmgnn.models.gae import GAE
from mmgnn.models.utils import build_encoder, build_model
from tests.explain.helpers import make_graph
from tests.models.conftest import model_config, tiny_graph


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


class FakeDataset:
    """Minimal dataset stand-in for select_graphs."""

    def __init__(self, graphs: list) -> None:
        self._graphs = list(graphs)

    def __len__(self) -> int:
        return len(self._graphs)

    def __getitem__(self, index: int):
        return self._graphs[index]


def _save_gae_checkpoint(path: Path, *, epoch: int | None) -> tuple[dict, Data]:
    torch.manual_seed(0)
    graph = tiny_graph()
    saved_config = model_config(model_type="gae", edge_presence_decoder_type="mlp")
    num_node_features = int(graph.x.size(-1))
    num_edge_features = int(graph.edge_attr.size(-1))
    encoder = build_encoder(
        num_node_features=num_node_features,
        num_edge_features=num_edge_features,
        model_config=saved_config,
    )
    model = build_model(
        encoder=encoder,
        num_node_features=num_node_features,
        num_edge_features=num_edge_features,
        model_config=saved_config,
    )
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": saved_config,
    }
    if epoch is not None:
        checkpoint["epoch"] = epoch
    torch.save(checkpoint, path)
    return checkpoint, graph


def _assert_state_dict_matches(loaded: torch.nn.Module, checkpoint: dict) -> None:
    loaded_state = loaded.state_dict()
    saved_state = checkpoint["model_state_dict"]
    assert loaded_state.keys() == saved_state.keys()
    for name, value in saved_state.items():
        torch.testing.assert_close(loaded_state[name], value)


# ---------------------------------------------------------------------------
# config help / validate
# ---------------------------------------------------------------------------


class TestGetAblationConfigHelp:
    def test_lists_all_defaults(self):
        help_text = get_ablation_config_help()
        assert help_text.startswith("Keys are:")
        for key in ABLATION_DEFAULTS:
            assert f"{key}:" in help_text
        assert "default=null" in help_text
        assert "default=0" in help_text
        assert "default=8" in help_text
        assert "default=True" in help_text


class TestValidateAblationConfig:
    def test_none_returns_defaults(self):
        resolved = validate_ablation_config(None)
        assert resolved == ABLATION_DEFAULTS
        assert resolved is not ABLATION_DEFAULTS

    def test_merges_overrides(self):
        resolved = validate_ablation_config(
            {"seed": 7, "greedy_topk": 3, "run_greedy": False, "max_subjects": 10}
        )
        assert resolved["seed"] == 7
        assert resolved["greedy_topk"] == 3
        assert resolved["run_greedy"] is False
        assert resolved["max_subjects"] == 10

    def test_unknown_key_raises(self):
        with pytest.raises(ValueError, match="Unknown ablation config keys"):
            validate_ablation_config({"seed": 0, "not_a_key": 1})

    def test_seed_must_be_non_negative(self):
        with pytest.raises(ValueError, match="seed must be >= 0"):
            validate_ablation_config({"seed": -1})

    def test_max_subjects_must_be_positive(self):
        with pytest.raises(ValueError, match="max_subjects must be >= 1"):
            validate_ablation_config({"max_subjects": 0})

    def test_greedy_topk_must_be_positive(self):
        with pytest.raises(ValueError, match="greedy_topk must be >= 1"):
            validate_ablation_config({"greedy_topk": 0})


# ---------------------------------------------------------------------------
# subject IDs / graph selection
# ---------------------------------------------------------------------------


class TestLoadSubjectIds:
    def test_reads_and_skips_blank_lines(self, tmp_path: Path):
        path = tmp_path / "subjects.txt"
        path.write_text("sub-001\n\n  sub-002  \n\n", encoding="utf-8")
        assert load_subject_ids(path) == ["sub-001", "sub-002"]

    def test_empty_file_raises(self, tmp_path: Path):
        path = tmp_path / "subjects.txt"
        path.write_text("\n\n", encoding="utf-8")
        with pytest.raises(ValueError, match="No subject IDs"):
            load_subject_ids(path)


class TestSelectGraphs:
    def test_preserves_requested_order(self):
        g1 = make_graph(subject_id="sub-001")
        g2 = make_graph(subject_id="sub-002")
        dataset = FakeDataset([g1, g2])
        selected = select_graphs(dataset, ["sub-002", "sub-001"])
        assert [g.subject_id for g in selected] == ["sub-002", "sub-001"]
        assert selected[0] is not g2  # clone

    def test_missing_subjects_raises(self):
        dataset = FakeDataset([make_graph(subject_id="sub-001")])
        with pytest.raises(ValueError, match="Subjects not found in dataset \\(2\\)"):
            select_graphs(dataset, ["missing-a", "missing-b"])


# ---------------------------------------------------------------------------
# load_frozen_gae
# ---------------------------------------------------------------------------


class TestLoadFrozenGae:
    def test_loads_freezes_and_returns_metadata(self, tmp_path: Path):
        path = tmp_path / "ckpt.pt"
        checkpoint, graph = _save_gae_checkpoint(path, epoch=12)
        runtime_config = model_config(edge_presence_decoder_type="inner_product")

        loaded, decoder_type, metadata = load_frozen_gae(
            checkpoint_path=path,
            model_config=runtime_config,
            sample_graph=graph,
            device=torch.device("cpu"),
        )

        assert isinstance(loaded, GAE)
        assert any(name.startswith("edge_presence_decoder.mlp") for name in loaded.state_dict())
        _assert_state_dict_matches(loaded, checkpoint)
        assert decoder_type == "mlp"
        assert metadata == {"model_type": "gae", "checkpoint_epoch": 12}
        assert loaded.training is False
        assert all(not parameter.requires_grad for parameter in loaded.parameters())

    def test_omits_epoch_when_absent(self, tmp_path: Path):
        path = tmp_path / "ckpt.pt"
        checkpoint, graph = _save_gae_checkpoint(path, epoch=None)
        runtime_config = model_config(edge_presence_decoder_type="inner_product")

        loaded, decoder_type, metadata = load_frozen_gae(
            checkpoint_path=path,
            model_config=runtime_config,
            sample_graph=graph,
            device=torch.device("cpu"),
        )

        _assert_state_dict_matches(loaded, checkpoint)
        assert decoder_type == "mlp"
        assert metadata == {"model_type": "gae"}
        assert "checkpoint_epoch" not in metadata


# ---------------------------------------------------------------------------
# prepare_graphs_for_explanation
# ---------------------------------------------------------------------------


def _explanation_configs(
    brain_graph_fixture: dict,
    cache_dir: Path,
    *,
    normalize_data: bool,
) -> tuple[dict, dict]:
    dataset_config = {
        "subjects_file": str(brain_graph_fixture["subjects_file"]),
        "dmri_folder": str(brain_graph_fixture["dmri_folder"]),
        "fc_folder": str(brain_graph_fixture["fc_folder"]),
        "fc_metric": "pearsoncorrelation",
        "anatomical_df": str(brain_graph_fixture["anatomical_df"]),
        "primary_dmri_key": str(brain_graph_fixture["primary_dmri_key"]),
        "edge_attributes": list(brain_graph_fixture["edge_attributes"]),
        "threshold": 0.0,
        "cache_dir": str(cache_dir),
        "num_workers": 1,
    }
    model_config = {
        "hidden_dim": 8,
        "node_feature_keys": ["volume"],
        "edge_feature_keys": ["fa"],
        "normalize_data": normalize_data,
    }
    return dataset_config, model_config


class TestPrepareGraphsForExplanation:
    def test_without_normalization(self, tmp_path: Path, brain_graph_fixture: dict):
        dataset_config, model_config = _explanation_configs(
            brain_graph_fixture,
            tmp_path / "cache",
            normalize_data=False,
        )
        _, out_cfg, out_graphs = prepare_graphs_for_explanation(
            dataset_config=dataset_config,
            model_config=model_config,
            checkpoint_dir=tmp_path,
            subject_ids=["sub-002", "sub-001"],
        )

        assert out_cfg["node_feature_keys"] == ["volume"]
        assert out_cfg["edge_feature_keys"] == ["fa"]
        assert out_cfg["normalize_data"] is False
        assert [graph.subject_id for graph in out_graphs] == ["sub-002", "sub-001"]
        assert all(graph.edge_attr.shape[1] == 1 for graph in out_graphs)
        assert not (tmp_path / "normalization_stats.json").exists()

    def test_with_normalization(self, tmp_path: Path, brain_graph_fixture: dict):
        cache_dir = tmp_path / "cache"
        dataset_config, raw_model_config = _explanation_configs(
            brain_graph_fixture,
            cache_dir,
            normalize_data=False,
        )
        _, _, raw_graphs = prepare_graphs_for_explanation(
            dataset_config=dataset_config,
            model_config=raw_model_config,
            checkpoint_dir=tmp_path,
            subject_ids=["sub-002", "sub-001"],
        )
        normalizer = NormalizationStats().fit(raw_graphs)
        normalizer.save(tmp_path / "normalization_stats.json")

        _, normalized_model_config = _explanation_configs(
            brain_graph_fixture,
            cache_dir,
            normalize_data=True,
        )
        _, _, normalized_graphs = prepare_graphs_for_explanation(
            dataset_config=dataset_config,
            model_config=normalized_model_config,
            checkpoint_dir=tmp_path,
            subject_ids=["sub-002", "sub-001"],
        )

        assert [graph.subject_id for graph in normalized_graphs] == ["sub-002", "sub-001"]
        for raw_graph, normalized_graph in zip(raw_graphs, normalized_graphs):
            expected = normalizer.transform(raw_graph)
            torch.testing.assert_close(normalized_graph.x, expected.x)
            torch.testing.assert_close(normalized_graph.edge_attr, expected.edge_attr)
            torch.testing.assert_close(normalized_graph.edge_weight, expected.edge_weight)
