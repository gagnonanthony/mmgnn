"""Unit tests for mmgnn.explain.ablation helpers and subject-level ablation."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch

from mmgnn.explain.ablation import (
    LOSS_KEYS,
    ablate_latent_column,
    ablate_latent_columns,
    ablate_subject,
    aggregate_dimension_importance,
    aggregate_greedy_ablation,
    aggregate_joint_topk_ablation,
    greedy_ablate_subject,
    importance_from_deltas,
    joint_ablate_dims,
    latent_edge_diagnostics,
    recon_weights_from_model_config,
    reconstruction_components,
    save_subject_ablation,
    select_loo_importance_topk,
    subject_ablation_seed,
)
from tests.explain.helpers import (
    LATENT_DIM,
    NUM_NODES,
    SUBJECT_ID,
    FakeModel,
    make_graph,
    make_z,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def loss_kwargs() -> dict[str, Any]:
    return {
        "edge_presence_weight": 1.0,
        "node_attr_weight": 1.0,
        "edge_attr_weight": 1.0,
        "node_loss_type": "mse",
        "edge_attr_loss_type": "mse",
    }


def make_subject_result(
    *,
    subject_id: str = SUBJECT_ID,
    latent_dim: int = LATENT_DIM,
    num_nodes: int = NUM_NODES,
    include_greedy: bool = True,
    include_joint_inputs: bool = False,
    delta_scale: float = 1.0,
) -> dict[str, Any]:
    """Minimal subject result dict for aggregation tests."""
    loss_keys = list(LOSS_KEYS)
    n_heads = len(loss_keys)
    delta_losses = np.zeros((latent_dim, n_heads), dtype=float)
    for dim in range(latent_dim):
        # Decreasing importance by dim index
        delta_losses[dim, :] = (latent_dim - dim) * delta_scale
    importance = importance_from_deltas(delta_losses)
    baseline = {key: 1.0 for key in loss_keys}
    result: dict[str, Any] = {
        "subject_id": subject_id,
        "latent_dim": latent_dim,
        "num_nodes": num_nodes,
        "num_pos_edges": num_nodes * 2,
        "num_neg_edges": num_nodes * 2,
        "seed": 0,
        "loss_keys": loss_keys,
        "baseline_losses": baseline,
        "delta_losses": delta_losses,
        "importance": importance,
        "node_variance": np.linspace(1.0, 0.1, latent_dim),
        "mean_edge_abs_diff": np.linspace(0.5, 0.05, latent_dim),
    }
    if include_greedy:
        # Greedy picks dims in order 0, 1, 2, ...
        selected = np.arange(latent_dim, dtype=int)
        greedy_order = np.arange(latent_dim, dtype=int)
        incr = np.zeros((latent_dim, n_heads), dtype=float)
        cumul = np.zeros((latent_dim, n_heads), dtype=float)
        for step, dim in enumerate(selected):
            incr[dim, :] = float(latent_dim - step)
            cumul[dim, :] = float((step + 1) * (latent_dim - step / 2.0))
        result["greedy_order"] = greedy_order
        result["greedy_selected"] = selected
        result["greedy_incremental_delta"] = incr
        result["greedy_cumulative_delta"] = cumul
    if include_joint_inputs:
        z = make_z(num_nodes, latent_dim)
        graph = make_graph(num_nodes, subject_id)
        model = FakeModel(z)
        result["model"] = model
        result["z"] = z
        result["graph"] = graph
        result["neg_edge_index"] = torch.tensor([[0], [2]], dtype=torch.long)
        result["loss_kwargs"] = loss_kwargs()
    return result


# ---------------------------------------------------------------------------
# ablate latent columns
# ---------------------------------------------------------------------------


class TestAblateLatent:
    def test_single_column_mean_replace(self):
        z = make_z()
        ablated = ablate_latent_column(z, 0)
        expected_mean = float(z[:, 0].mean())
        assert torch.allclose(ablated[:, 0], torch.full((NUM_NODES,), expected_mean))
        assert torch.allclose(ablated[:, 1:], z[:, 1:])

    def test_multi_column_mean_replace(self):
        z = make_z()
        ablated = ablate_latent_columns(z, [0, 2])
        assert torch.allclose(ablated[:, 0], torch.full((NUM_NODES,), float(z[:, 0].mean())))
        assert torch.allclose(ablated[:, 2], torch.full((NUM_NODES,), float(z[:, 2].mean())))
        assert torch.allclose(ablated[:, 1], z[:, 1])

    def test_returns_clone(self):
        z = make_z()
        ablated = ablate_latent_columns(z, [1])
        ablated[0, 1] = -999.0
        assert z[0, 1].item() != -999.0

    def test_rejects_non_2d(self):
        with pytest.raises(ValueError, match="shape"):
            ablate_latent_columns(torch.zeros(3), [0])

    def test_rejects_out_of_range_dim(self):
        z = make_z()
        with pytest.raises(ValueError, match="out of range"):
            ablate_latent_columns(z, [LATENT_DIM])
        with pytest.raises(ValueError, match="out of range"):
            ablate_latent_column(z, -1)


# ---------------------------------------------------------------------------
# subject seed
# ---------------------------------------------------------------------------


class TestSubjectAblationSeed:
    def test_deterministic(self):
        a = subject_ablation_seed(42, SUBJECT_ID)
        b = subject_ablation_seed(42, SUBJECT_ID)
        assert a == b

    def test_subject_dependent(self):
        a = subject_ablation_seed(42, "sub-001")
        b = subject_ablation_seed(42, "sub-002")
        assert a != b

    def test_in_valid_range(self):
        seed = subject_ablation_seed(0, SUBJECT_ID)
        assert 0 <= seed <= 2**31 - 2


# ---------------------------------------------------------------------------
# latent edge diagnostics
# ---------------------------------------------------------------------------


class TestLatentEdgeDiagnostics:
    def test_shapes_and_values(self):
        z = make_z()
        edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        out = latent_edge_diagnostics(z, edge_index)
        assert out["node_variance"].shape == (LATENT_DIM,)
        assert out["mean_edge_abs_diff"].shape == (LATENT_DIM,)
        np.testing.assert_allclose(out["node_variance"], z.numpy().var(axis=0))
        # Constant column → zero edge abs diff
        assert out["mean_edge_abs_diff"][2] == pytest.approx(0.0)

    def test_empty_edges_zero_diff(self):
        z = make_z()
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        out = latent_edge_diagnostics(z, edge_index)
        assert np.allclose(out["mean_edge_abs_diff"], 0.0)

    def test_rejects_bad_shapes(self):
        with pytest.raises(ValueError, match="shape"):
            latent_edge_diagnostics(torch.zeros(3), torch.zeros((2, 1), dtype=torch.long))
        with pytest.raises(ValueError, match="edge_index"):
            latent_edge_diagnostics(make_z(), torch.zeros((3, 2), dtype=torch.long))


# ---------------------------------------------------------------------------
# importance
# ---------------------------------------------------------------------------


class TestImportanceFromDeltas:
    def test_l1_normalize_positives(self):
        deltas = np.array([[2.0, 1.0], [2.0, -1.0], [0.0, 3.0]], dtype=float)
        importance = importance_from_deltas(deltas)
        # Column 0: [2, 2, 0] → [0.5, 0.5, 0]
        np.testing.assert_allclose(importance[:, 0], [0.5, 0.5, 0.0])
        # Column 1: max → [1, 0, 3] → [0.25, 0, 0.75]
        np.testing.assert_allclose(importance[:, 1], [0.25, 0.0, 0.75])
        assert np.allclose(importance.sum(axis=0), 1.0)

    def test_all_zero_or_negative_column(self):
        deltas = np.array([[-1.0], [0.0], [-2.0]], dtype=float)
        importance = importance_from_deltas(deltas)
        assert np.allclose(importance, 0.0)

    def test_rejects_non_2d(self):
        with pytest.raises(ValueError, match="shape"):
            importance_from_deltas(np.array([1.0, 2.0]))


# ---------------------------------------------------------------------------
# recon weights + topk select
# ---------------------------------------------------------------------------


class TestReconWeights:
    def test_extracts_required_keys_and_default_ratio(self):
        config = {
            "recon_edge_presence_weight": 1.5,
            "recon_node_attr_weight": 0.5,
            "recon_edge_attr_weight": 2.0,
            "recon_node_loss_type": "mse",
            "recon_edge_attr_loss_type": "huber",
        }
        weights = recon_weights_from_model_config(config)
        assert weights["edge_presence_weight"] == 1.5
        assert weights["node_attr_weight"] == 0.5
        assert weights["edge_attr_weight"] == 2.0
        assert weights["node_loss_type"] == "mse"
        assert weights["edge_attr_loss_type"] == "huber"
        assert weights["neg_sampling_ratio"] == 1.0

    def test_custom_neg_sampling_ratio(self):
        config = {
            "recon_edge_presence_weight": 1.0,
            "recon_node_attr_weight": 1.0,
            "recon_edge_attr_weight": 1.0,
            "recon_node_loss_type": "mse",
            "recon_edge_attr_loss_type": "mse",
            "neg_sampling_ratio": 2.5,
        }
        assert recon_weights_from_model_config(config)["neg_sampling_ratio"] == 2.5


class TestSelectLooImportanceTopk:
    def test_ranks_by_mean_importance(self):
        table = pd.DataFrame(
            {
                "dim": [0, 1, 2],
                "mean_importance_total_loss": [0.1, 0.7, 0.2],
            }
        )
        assert select_loo_importance_topk(table, greedy_topk=2) == [1, 2]

    def test_clamps_k_to_n_dims(self):
        table = pd.DataFrame(
            {"dim": [0, 1], "mean_importance_total_loss": [0.9, 0.1]}
        )
        assert select_loo_importance_topk(table, greedy_topk=10) == [0, 1]

    def test_missing_column_raises(self):
        with pytest.raises(ValueError, match="mean_importance_total_loss"):
            select_loo_importance_topk(pd.DataFrame({"dim": [0]}), greedy_topk=1)

    def test_k_less_than_one_raises(self):
        table = pd.DataFrame({"dim": [0], "mean_importance_total_loss": [1.0]})
        with pytest.raises(ValueError, match="greedy_topk"):
            select_loo_importance_topk(table, greedy_topk=0)


# ---------------------------------------------------------------------------
# reconstruction + greedy + ablate_subject + joint + save
# ---------------------------------------------------------------------------


class TestReconstructionComponents:
    def test_returns_float_loss_keys(self):
        z = make_z()
        model = FakeModel(z)
        graph = make_graph()
        neg = torch.tensor([[0], [2]], dtype=torch.long)
        components = reconstruction_components(model, z, graph, neg, **loss_kwargs())
        assert set(components) == set(LOSS_KEYS)
        assert all(isinstance(v, float) for v in components.values())
        assert components["total_loss"] == pytest.approx(0.0)

    def test_ablating_varying_dim_raises_loss(self):
        z = make_z()
        model = FakeModel(z)
        graph = make_graph()
        neg = torch.tensor([[0], [2]], dtype=torch.long)
        baseline = reconstruction_components(model, z, graph, neg, **loss_kwargs())
        ablated = reconstruction_components(
            model, ablate_latent_column(z, 0), graph, neg, **loss_kwargs()
        )
        assert ablated["total_loss"] > baseline["total_loss"]


class TestGreedyAblateSubject:
    def test_selects_all_dims_and_orders_by_variance(self):
        z = make_z()
        model = FakeModel(z)
        graph = make_graph()
        neg = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
        baseline = reconstruction_components(model, z, graph, neg, **loss_kwargs())
        out = greedy_ablate_subject(
            model,
            z,
            graph,
            neg,
            baseline,
            list(LOSS_KEYS),
            loss_kwargs=loss_kwargs(),
        )
        selected = out["greedy_selected"]
        assert sorted(selected.tolist()) == list(range(LATENT_DIM))
        # Highest variance columns first (col 0 > col 1 > col 2 constant)
        assert int(selected[0]) == 0
        assert int(selected[1]) == 1
        assert int(selected[2]) == 2
        # greedy_order is inverse: order[dim] = step
        for step, dim in enumerate(selected):
            assert int(out["greedy_order"][dim]) == step

    def test_cumulative_matches_baseline_delta(self):
        z = make_z()
        model = FakeModel(z)
        graph = make_graph()
        neg = torch.tensor([[0], [2]], dtype=torch.long)
        baseline = reconstruction_components(model, z, graph, neg, **loss_kwargs())
        out = greedy_ablate_subject(
            model,
            z,
            graph,
            neg,
            baseline,
            list(LOSS_KEYS),
            loss_kwargs=loss_kwargs(),
        )
        # After all dims selected, cumulative on last selected dim equals full ablation
        last_dim = int(out["greedy_selected"][-1])
        z_all = ablate_latent_columns(z, list(range(LATENT_DIM)))
        full = reconstruction_components(model, z_all, graph, neg, **loss_kwargs())
        total_idx = list(LOSS_KEYS).index("total_loss")
        assert out["greedy_cumulative_delta"][last_dim, total_idx] == pytest.approx(
            full["total_loss"] - baseline["total_loss"]
        )

    def test_requires_total_loss_in_baseline(self):
        z = make_z()
        model = FakeModel(z)
        graph = make_graph()
        neg = torch.tensor([[0], [2]], dtype=torch.long)
        with pytest.raises(ValueError, match="total_loss"):
            greedy_ablate_subject(
                model,
                z,
                graph,
                neg,
                {"edge_presence_loss": 1.0},
                ["edge_presence_loss"],
                loss_kwargs=loss_kwargs(),
            )


class TestAblateSubject:
    def test_shapes_keys_and_importance(self):
        z = make_z()
        model = FakeModel(z)
        graph = make_graph()
        result = ablate_subject(model, z, graph, seed=7, run_greedy=True, **loss_kwargs())
        assert result["subject_id"] == SUBJECT_ID
        assert result["latent_dim"] == LATENT_DIM
        assert result["delta_losses"].shape == (LATENT_DIM, len(LOSS_KEYS))
        assert result["importance"].shape == (LATENT_DIM, len(LOSS_KEYS))
        assert result["node_variance"].shape == (LATENT_DIM,)
        assert "greedy_selected" in result
        # Positive total-loss deltas → importance column sums to 1
        total_idx = list(LOSS_KEYS).index("total_loss")
        if result["delta_losses"][:, total_idx].max() > 0:
            assert result["importance"][:, total_idx].sum() == pytest.approx(1.0)

    def test_run_greedy_false_omits_greedy_keys(self):
        z = make_z()
        model = FakeModel(z)
        graph = make_graph()
        result = ablate_subject(model, z, graph, seed=1, run_greedy=False, **loss_kwargs())
        assert "greedy_selected" not in result
        assert "greedy_order" not in result

    def test_node_count_mismatch(self):
        z = make_z(num_nodes=NUM_NODES)
        model = FakeModel(z)
        graph = make_graph(num_nodes=NUM_NODES - 1)
        with pytest.raises(ValueError, match="Node count mismatch"):
            ablate_subject(model, z, graph, seed=0, run_greedy=False, **loss_kwargs())

    def test_latent_dim_mismatch(self):
        z = make_z(latent_dim=LATENT_DIM)
        model = FakeModel(make_z(latent_dim=LATENT_DIM + 1))
        graph = make_graph()
        with pytest.raises(ValueError, match="does not match model latent dim"):
            ablate_subject(model, z, graph, seed=0, run_greedy=False, **loss_kwargs())

    def test_same_seed_reuses_negative_edges(self):
        z = make_z()
        model = FakeModel(z)
        graph = make_graph()
        a = ablate_subject(model, z, graph, seed=123, run_greedy=False, **loss_kwargs())
        b = ablate_subject(model, z, graph, seed=123, run_greedy=False, **loss_kwargs())
        assert torch.equal(a["neg_edge_index"], b["neg_edge_index"])


class TestJointAblateDims:
    def test_joint_delta_vs_baseline(self):
        z = make_z()
        model = FakeModel(z)
        graph = make_graph()
        neg = torch.tensor([[0], [2]], dtype=torch.long)
        baseline = reconstruction_components(model, z, graph, neg, **loss_kwargs())
        single = joint_ablate_dims(
            model, z, graph, neg, [0], baseline, loss_kwargs=loss_kwargs()
        )
        all_dims = joint_ablate_dims(
            model, z, graph, neg, list(range(LATENT_DIM)), baseline, loss_kwargs=loss_kwargs()
        )
        assert single["total_loss"] > 0
        assert all_dims["total_loss"] >= single["total_loss"]


class TestSaveSubjectAblation:
    def test_writes_arrays_and_summary(self, tmp_path: Path):
        z = make_z()
        model = FakeModel(z)
        graph = make_graph()
        result = ablate_subject(model, z, graph, seed=3, run_greedy=True, **loss_kwargs())
        summary = save_subject_ablation(tmp_path, SUBJECT_ID, result)
        subject_dir = tmp_path / "subjects" / SUBJECT_ID
        assert (subject_dir / "delta_losses.npy").is_file()
        assert (subject_dir / "importance.npy").is_file()
        assert (subject_dir / "node_variance.npy").is_file()
        assert (subject_dir / "mean_edge_abs_diff.npy").is_file()
        assert (subject_dir / "greedy_order.npy").is_file()
        assert (subject_dir / "greedy_selected.npy").is_file()
        assert (subject_dir / "baseline_losses.json").is_file()
        assert (subject_dir / "loss_keys.json").is_file()
        assert (subject_dir / "summary.json").is_file()
        assert summary["subject_id"] == SUBJECT_ID
        assert summary["latent_dim"] == LATENT_DIM
        assert summary["mean_delta_total"] is not None

    def test_omits_greedy_files_without_greedy_keys(self, tmp_path: Path):
        z = make_z()
        model = FakeModel(z)
        graph = make_graph()
        result = ablate_subject(model, z, graph, seed=3, run_greedy=False, **loss_kwargs())
        save_subject_ablation(tmp_path, SUBJECT_ID, result)
        subject_dir = tmp_path / "subjects" / SUBJECT_ID
        assert not (subject_dir / "greedy_order.npy").exists()
        assert not (subject_dir / "greedy_selected.npy").exists()


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------


class TestAggregateDimensionImportance:
    def test_means_medians_and_topk_flags(self):
        results = [
            make_subject_result(subject_id="sub-001", delta_scale=1.0),
            make_subject_result(subject_id="sub-002", delta_scale=3.0),
        ]
        table = aggregate_dimension_importance(results, topk_dims=[0, 2])
        assert len(table) == LATENT_DIM
        assert "mean_delta_total_loss" in table.columns
        assert "mean_importance_total_loss" in table.columns
        assert "mean_greedy_rank" in table.columns
        assert table.loc[table["dim"] == 0, "in_loo_importance_topk"].item() is True
        assert table.loc[table["dim"] == 1, "in_loo_importance_topk"].item() is False
        assert table.loc[table["dim"] == 2, "in_loo_importance_topk"].item() is True
        expected_mean = (results[0]["delta_losses"][0, 0] + results[1]["delta_losses"][0, 0]) / 2
        assert table.loc[table["dim"] == 0, "mean_delta_total_loss"].item() == pytest.approx(
            expected_mean
        )

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="No subject"):
            aggregate_dimension_importance([])

    def test_inconsistent_keys_raises(self):
        a = make_subject_result(subject_id="a")
        b = make_subject_result(subject_id="b")
        b["loss_keys"] = ["total_loss"]
        with pytest.raises(ValueError, match="Inconsistent loss keys"):
            aggregate_dimension_importance([a, b])

    def test_inconsistent_dims_raises(self):
        a = make_subject_result(subject_id="a", latent_dim=3)
        b = make_subject_result(subject_id="b", latent_dim=2)
        with pytest.raises(ValueError, match="Inconsistent latent dims"):
            aggregate_dimension_importance([a, b])


class TestAggregateGreedyAblation:
    def test_one_row_per_step(self):
        results = [
            make_subject_result(subject_id="sub-001"),
            make_subject_result(subject_id="sub-002"),
        ]
        table = aggregate_greedy_ablation(results)
        assert list(table["step"]) == list(range(LATENT_DIM))
        assert "mean_incremental_delta_total_loss" in table.columns
        assert "mean_cumulative_delta_total_loss" in table.columns

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="No subject"):
            aggregate_greedy_ablation([])

    def test_missing_total_loss_raises(self):
        result = make_subject_result()
        result["loss_keys"] = ["edge_presence_loss"]
        with pytest.raises(ValueError, match="total_loss"):
            aggregate_greedy_ablation([result])


class TestAggregateJointTopk:
    def test_summary_fields_and_redundancy(self):
        results = [
            make_subject_result(subject_id="sub-001", include_joint_inputs=True),
            make_subject_result(subject_id="sub-002", include_joint_inputs=True),
        ]
        summary = aggregate_joint_topk_ablation(results, topk_dims=[0, 1])
        assert summary["topk_dims"] == [0, 1]
        assert summary["k"] == 2
        assert summary["num_subjects"] == 2
        assert "mean_joint_delta_total_loss" in summary
        assert "mean_sum_loo_delta_total_loss" in summary
        assert "redundancy_ratio" in summary

    def test_empty_subjects_raises(self):
        with pytest.raises(ValueError, match="No subject"):
            aggregate_joint_topk_ablation([], topk_dims=[0])

    def test_empty_topk_raises(self):
        with pytest.raises(ValueError, match="topk_dims"):
            aggregate_joint_topk_ablation([make_subject_result(include_joint_inputs=True)], topk_dims=[])

    def test_missing_joint_inputs_raises(self):
        result = make_subject_result(include_joint_inputs=False)
        with pytest.raises(ValueError, match="joint-ablation inputs"):
            aggregate_joint_topk_ablation([result], topk_dims=[0])
