"""Unit tests for mmgnn.data.normalization.NormalizationStats."""

from pathlib import Path

import pytest
import torch
from torch_geometric.data import Data

from mmgnn.data.normalization import NormalizationStats


def make_graph(
    *,
    x: torch.Tensor | None = None,
    edge_attr: torch.Tensor | None = None,
    edge_weight: torch.Tensor | None = None,
    num_nodes: int | None = None,
) -> Data:
    """Build a minimal PyG Data object for normalization tests."""
    if x is None and num_nodes is None:
        raise ValueError("Provide x or num_nodes")
    if x is None:
        x = torch.zeros((num_nodes, 1), dtype=torch.float32)
    data = Data(x=x, num_nodes=int(x.shape[0]))
    if edge_attr is not None:
        data.edge_attr = edge_attr
        data.edge_index = torch.zeros((2, edge_attr.shape[0]), dtype=torch.long)
    if edge_weight is not None:
        data.edge_weight = edge_weight
        if not hasattr(data, "edge_index") or data.edge_index is None:
            data.edge_index = torch.zeros((2, edge_weight.shape[0]), dtype=torch.long)
    return data


# ---------------------------------------------------------------------------
# fit
# ---------------------------------------------------------------------------


class TestFit:
    def test_computes_node_edge_and_weight_stats(self):
        graphs = [
            make_graph(
                x=torch.tensor([[0.0, 2.0], [2.0, 4.0]], dtype=torch.float32),
                edge_attr=torch.tensor([[1.0, 3.0], [3.0, 5.0]], dtype=torch.float32),
                edge_weight=torch.tensor([1.0, 3.0], dtype=torch.float32),
            ),
            make_graph(
                x=torch.tensor([[4.0, 6.0], [6.0, 8.0]], dtype=torch.float32),
                edge_attr=torch.tensor([[5.0, 7.0], [7.0, 9.0]], dtype=torch.float32),
                edge_weight=torch.tensor([5.0, 7.0], dtype=torch.float32),
            ),
        ]
        normalizer = NormalizationStats().fit(graphs)

        all_x = torch.cat([g.x for g in graphs], dim=0)
        all_e = torch.cat([g.edge_attr for g in graphs], dim=0)
        all_w = torch.cat([g.edge_weight for g in graphs], dim=0)

        torch.testing.assert_close(normalizer.node_mean, all_x.mean(dim=0, keepdim=True))
        torch.testing.assert_close(normalizer.node_std, all_x.std(dim=0, keepdim=True))
        torch.testing.assert_close(normalizer.edge_mean, all_e.mean(dim=0, keepdim=True))
        torch.testing.assert_close(normalizer.edge_std, all_e.std(dim=0, keepdim=True))
        assert normalizer.edge_weight_mean == pytest.approx(float(all_w.mean()))
        assert normalizer.edge_weight_std == pytest.approx(float(all_w.std()))

    def test_constant_feature_clamps_std_to_eps(self):
        graphs = [
            make_graph(x=torch.ones((3, 1), dtype=torch.float32)),
            make_graph(x=torch.ones((2, 1), dtype=torch.float32)),
        ]
        eps = 1e-5
        normalizer = NormalizationStats(eps=eps).fit(graphs, normalize_edges=False, normalize_edge_weights=False)
        assert float(normalizer.node_std) == pytest.approx(eps)

    def test_flag_gating(self):
        graphs = [
            make_graph(
                x=torch.randn(4, 2),
                edge_attr=torch.randn(3, 2),
                edge_weight=torch.randn(3),
            )
        ]
        normalizer = NormalizationStats().fit(
            graphs,
            normalize_nodes=False,
            normalize_edges=False,
            normalize_edge_weights=False,
        )
        assert normalizer.node_mean is None
        assert normalizer.edge_mean is None
        assert normalizer.edge_weight_mean is None

    def test_missing_fields_leave_stats_unset(self):
        graphs = [make_graph(x=torch.randn(3, 1))]
        normalizer = NormalizationStats().fit(graphs)
        assert normalizer.node_mean is not None
        assert normalizer.edge_mean is None
        assert normalizer.edge_weight_mean is None

    def test_fit_returns_self(self):
        normalizer = NormalizationStats()
        result = normalizer.fit([make_graph(x=torch.randn(2, 1))], normalize_edges=False, normalize_edge_weights=False)
        assert result is normalizer


# ---------------------------------------------------------------------------
# transform / __call__
# ---------------------------------------------------------------------------


class TestTransform:
    def test_zscore_formula(self):
        x = torch.tensor([[0.0, 10.0], [2.0, 30.0]], dtype=torch.float32)
        edge_attr = torch.tensor([[1.0], [5.0]], dtype=torch.float32)
        edge_weight = torch.tensor([2.0, 8.0], dtype=torch.float32)
        graph = make_graph(x=x, edge_attr=edge_attr, edge_weight=edge_weight)

        normalizer = NormalizationStats().fit([graph])
        out = normalizer.transform(graph)

        expected_x = (x - normalizer.node_mean) / normalizer.node_std
        expected_e = (edge_attr - normalizer.edge_mean) / normalizer.edge_std
        expected_w = (edge_weight - normalizer.edge_weight_mean) / normalizer.edge_weight_std
        torch.testing.assert_close(out.x, expected_x)
        torch.testing.assert_close(out.edge_attr, expected_e)
        torch.testing.assert_close(out.edge_weight, expected_w)

    def test_clones_input(self):
        graph = make_graph(x=torch.tensor([[1.0], [3.0]], dtype=torch.float32))
        original_x = graph.x.clone()
        normalizer = NormalizationStats().fit([graph], normalize_edges=False, normalize_edge_weights=False)
        out = normalizer.transform(graph)
        assert out is not graph
        torch.testing.assert_close(graph.x, original_x)
        assert not torch.equal(out.x, graph.x)

    def test_call_equals_transform(self):
        graph = make_graph(
            x=torch.randn(4, 2),
            edge_attr=torch.randn(5, 1),
            edge_weight=torch.randn(5),
        )
        normalizer = NormalizationStats().fit([graph])
        via_transform = normalizer.transform(graph)
        via_call = normalizer(graph)
        torch.testing.assert_close(via_call.x, via_transform.x)
        torch.testing.assert_close(via_call.edge_attr, via_transform.edge_attr)
        torch.testing.assert_close(via_call.edge_weight, via_transform.edge_weight)

    def test_noop_when_stats_unset(self):
        graph = make_graph(
            x=torch.tensor([[1.0], [2.0]], dtype=torch.float32),
            edge_attr=torch.tensor([[3.0], [4.0]], dtype=torch.float32),
            edge_weight=torch.tensor([5.0, 6.0], dtype=torch.float32),
        )
        out = NormalizationStats().transform(graph)
        torch.testing.assert_close(out.x, graph.x)
        torch.testing.assert_close(out.edge_attr, graph.edge_attr)
        torch.testing.assert_close(out.edge_weight, graph.edge_weight)


# ---------------------------------------------------------------------------
# inverse
# ---------------------------------------------------------------------------


class TestInverse:
    def test_edge_attr_roundtrip(self):
        edge_attr = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.float32)
        graph = make_graph(x=torch.randn(2, 1), edge_attr=edge_attr)
        normalizer = NormalizationStats().fit([graph], normalize_nodes=False, normalize_edge_weights=False)
        normalized = (edge_attr - normalizer.edge_mean) / normalizer.edge_std
        restored = normalizer.inverse_transform_edge_attr(normalized)
        torch.testing.assert_close(restored, edge_attr)

    def test_edge_attr_none_passthrough(self):
        assert NormalizationStats().inverse_transform_edge_attr(None) is None

    def test_edge_weight_roundtrip(self):
        edge_weight = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
        graph = make_graph(x=torch.randn(2, 1), edge_weight=edge_weight)
        normalizer = NormalizationStats().fit([graph], normalize_nodes=False, normalize_edges=False)
        normalized = (edge_weight - normalizer.edge_weight_mean) / normalizer.edge_weight_std
        restored = normalizer.inverse_transform_edge_weight(normalized)
        torch.testing.assert_close(restored, edge_weight)

    def test_edge_weight_unchanged_when_unset(self):
        weights = torch.tensor([1.0, 2.0], dtype=torch.float32)
        restored = NormalizationStats().inverse_transform_edge_weight(weights)
        torch.testing.assert_close(restored, weights)


# ---------------------------------------------------------------------------
# persistence
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_save_load_roundtrip(self, tmp_path: Path):
        graphs = [
            make_graph(
                x=torch.randn(3, 2),
                edge_attr=torch.randn(4, 3),
                edge_weight=torch.randn(4),
            )
        ]
        normalizer = NormalizationStats(eps=1e-6).fit(graphs)
        path = tmp_path / "stats.json"
        normalizer.save(path)

        loaded = NormalizationStats.load(path)
        assert loaded.eps == pytest.approx(1e-6)
        torch.testing.assert_close(loaded.node_mean, normalizer.node_mean)
        torch.testing.assert_close(loaded.node_std, normalizer.node_std)
        torch.testing.assert_close(loaded.edge_mean, normalizer.edge_mean)
        torch.testing.assert_close(loaded.edge_std, normalizer.edge_std)
        assert loaded.edge_weight_mean == pytest.approx(normalizer.edge_weight_mean)
        assert loaded.edge_weight_std == pytest.approx(normalizer.edge_weight_std)

        original_out = normalizer.transform(graphs[0])
        loaded_out = loaded.transform(graphs[0])
        torch.testing.assert_close(loaded_out.x, original_out.x)
        torch.testing.assert_close(loaded_out.edge_attr, original_out.edge_attr)
        torch.testing.assert_close(loaded_out.edge_weight, original_out.edge_weight)

    def test_load_missing_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            NormalizationStats.load(tmp_path / "missing.json")


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------


class TestSummary:
    def test_get_summary_enabled_and_disabled(self):
        graphs = [
            make_graph(
                x=torch.randn(4, 2),
                edge_attr=torch.randn(3, 1),
                edge_weight=torch.randn(3),
            )
        ]
        fitted = NormalizationStats().fit(graphs).get_summary()
        assert fitted["node_features"]["enabled"] is True
        assert fitted["node_features"]["num_features"] == 2
        assert fitted["edge_attributes"]["enabled"] is True
        assert fitted["edge_attributes"]["num_features"] == 1
        assert fitted["edge_weights"]["enabled"] is True

        empty = NormalizationStats().get_summary()
        assert empty["node_features"]["enabled"] is False
        assert empty["edge_attributes"]["enabled"] is False
        assert empty["edge_weights"]["enabled"] is False

    def test_print_summary_smoke(self):
        graphs = [make_graph(x=torch.randn(2, 1), edge_weight=torch.randn(2))]
        NormalizationStats().fit(graphs, normalize_edges=False).print_summary()
