"""Unit tests for mmgnn.data.dataset helpers and graph construction."""

from pathlib import Path

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from mmgnn.data.dataset import (
    BrainGraphDataset,
    apply_density_threshold,
    apply_top_k_sparsification,
    build_node_features,
    create_pyg_data,
    find_matrix_file,
    gather_edge_attributes,
    load_dataset,
    matrix_to_edge_index,
    save_dataset,
)
from tests.helpers import (
    EDGE_ATTRIBUTES,
    NUM_NODES,
    PRIMARY_DMRI_KEY,
    SUBJECT_IDS,
    dmri_filename,
    sym_matrix,
    write_subject_matrices,
)


# ---------------------------------------------------------------------------
# build_node_features
# ---------------------------------------------------------------------------


class TestBuildNodeFeatures:
    def test_anatomical_2d_passthrough(self):
        anat = np.arange(NUM_NODES * 2, dtype=np.float64).reshape(NUM_NODES, 2)
        features = build_node_features(NUM_NODES, anatomical_features=anat)
        assert features.shape == (NUM_NODES, 2)
        assert features.dtype == np.float32
        np.testing.assert_allclose(features, anat.astype(np.float32))

    def test_anatomical_1d_reshaped(self):
        anat = np.arange(NUM_NODES, dtype=np.float32)
        features = build_node_features(NUM_NODES, anatomical_features=anat)
        assert features.shape == (NUM_NODES, 1)
        np.testing.assert_allclose(features[:, 0], anat)

    def test_none_returns_ones(self):
        features = build_node_features(NUM_NODES, anatomical_features=None)
        assert features.shape == (NUM_NODES, 1)
        np.testing.assert_allclose(features, np.ones((NUM_NODES, 1), dtype=np.float32))

    def test_nan_inf_sanitized(self):
        anat = np.array([[1.0, np.nan], [np.inf, -np.inf], [2.0, 3.0], [4.0, 5.0]], dtype=np.float32)
        features = build_node_features(NUM_NODES, anatomical_features=anat)
        assert np.isfinite(features).all()
        assert features[0, 1] == 0.0
        assert features[1, 0] == 0.0
        assert features[1, 1] == 0.0

    def test_row_mismatch_raises(self):
        anat = np.ones((NUM_NODES - 1, 1), dtype=np.float32)
        with pytest.raises(ValueError, match="anatomical_features has"):
            build_node_features(NUM_NODES, anatomical_features=anat)


# ---------------------------------------------------------------------------
# matrix_to_edge_index
# ---------------------------------------------------------------------------


class TestMatrixToEdgeIndex:
    def test_known_matrix_edges_and_weights(self):
        matrix = np.array(
            [
                [0.0, 0.5, 0.0],
                [0.5, 0.0, 0.2],
                [0.0, 0.2, 0.0],
            ],
            dtype=np.float32,
        )
        edge_index, edge_weight = matrix_to_edge_index(matrix, threshold=0.0)

        expected_pairs = {(0, 1), (1, 0), (1, 2), (2, 1)}
        pairs = {
            (int(edge_index[0, i]), int(edge_index[1, i]))
            for i in range(edge_index.shape[1])
        }
        assert pairs == expected_pairs
        assert edge_index.shape == (2, 4)
        assert edge_weight.shape == (4,)
        for i in range(edge_index.shape[1]):
            src, dst = int(edge_index[0, i]), int(edge_index[1, i])
            assert edge_weight[i].item() == pytest.approx(matrix[src, dst])

    def test_threshold_excludes_weak_edges(self):
        # This thresholding option is not really used, but left in case it is needed in the future.
        matrix = np.array(
            [
                [0.0, 0.5, 0.05],
                [0.5, 0.0, 0.2],
                [0.05, 0.2, 0.0],
            ],
            dtype=np.float32,
        )
        edge_index, edge_weight = matrix_to_edge_index(matrix, threshold=0.1)
        pairs = {
            (int(edge_index[0, i]), int(edge_index[1, i]))
            for i in range(edge_index.shape[1])
        }
        assert (0, 2) not in pairs
        assert (2, 0) not in pairs
        assert (0, 1) in pairs
        assert edge_weight.numel() == 4

    def test_empty_above_threshold(self):
        matrix = np.zeros((3, 3), dtype=np.float32)
        edge_index, edge_weight = matrix_to_edge_index(matrix, threshold=0.0)
        assert edge_index.shape == (2, 0)
        assert edge_weight.shape == (0,)


# ---------------------------------------------------------------------------
# gather_edge_attributes
# ---------------------------------------------------------------------------


class TestGatherEdgeAttributes:
    def test_sorted_key_order_and_values(self):
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        matrices = {
            "md": np.array([[0.0, 2.0], [2.0, 0.0]], dtype=np.float32),
            "fa": np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32),
        }
        edge_attr = gather_edge_attributes(edge_index, matrices)
        # Keys sorted: fa, md
        assert edge_attr.shape == (2, 2)
        torch.testing.assert_close(edge_attr[0], torch.tensor([1.0, 2.0]))
        torch.testing.assert_close(edge_attr[1], torch.tensor([1.0, 2.0]))

    def test_nan_sanitized(self):
        edge_index = torch.tensor([[0], [1]], dtype=torch.long)
        matrices = {"fa": np.array([[0.0, np.nan], [np.nan, 0.0]], dtype=np.float32)}
        edge_attr = gather_edge_attributes(edge_index, matrices)
        assert edge_attr.shape == (1, 1)
        assert edge_attr[0, 0].item() == 0.0


# ---------------------------------------------------------------------------
# apply_top_k_sparsification / apply_density_threshold
# ---------------------------------------------------------------------------


class TestSparsification:
    def test_top_k_keeps_strongest_and_is_symmetric(self):
        matrix = np.array(
            [
                [0.0, 0.9, 0.2, 0.1],
                [0.9, 0.0, 0.8, 0.05],
                [0.2, 0.8, 0.0, 0.7],
                [0.1, 0.05, 0.7, 0.0],
            ],
            dtype=np.float32,
        )
        sparsified = apply_top_k_sparsification(matrix, top_k=1)
        np.testing.assert_allclose(sparsified, sparsified.T)
        np.testing.assert_allclose(np.diag(sparsified), 0.0)
        # Each row should have at most top_k nonzero after symmetrization may add more
        # but original strongest neighbor of node 0 is node 1.
        assert sparsified[0, 1] == pytest.approx(0.9)
        assert sparsified[0, 2] == 0.0 or sparsified[0, 3] == 0.0

    def test_top_k_large_keeps_all_nonzero(self):
        matrix = sym_matrix(seed=3) # Create a full 4x4 matrix
        sparsified = apply_top_k_sparsification(matrix, top_k=NUM_NODES) # Keep all edges since top_k is NUM_NODES
        np.testing.assert_allclose(sparsified, matrix)

    def test_density_extremes(self):
        matrix = sym_matrix(seed=5) # Same here, create a full 4x4 matrix
        assert apply_density_threshold(matrix, density=0.0) == np.inf # No edges should be kept
        assert apply_density_threshold(matrix, density=1.0) == 0.0 # All edges should be kept

    def test_density_mid_yields_target_edge_count(self):
        matrix = sym_matrix(seed=11) # Create a full 4x4 matrix
        density = 0.5 # Keep 50% of the edges
        n = matrix.shape[0]
        max_edges = n * (n - 1) // 2
        target_edges = int(max_edges * density)
        threshold = apply_density_threshold(matrix, density)
        upper = np.abs(matrix[np.triu_indices(n, k=1)])
        kept = int(np.sum(upper > threshold))
        # Threshold is the value at index target_edges in descending sort, so
        # edges strictly above it are at most target_edges.
        assert kept <= target_edges


# ---------------------------------------------------------------------------
# find_matrix_file
# ---------------------------------------------------------------------------


class TestFindMatrixFile:
    def test_finds_alternate_naming(self, tmp_path: Path):
        subject_id = SUBJECT_IDS[0]
        path = tmp_path / f"{subject_id}_atlas-Schaefer_metric-fa.npy"
        np.save(path, sym_matrix(seed=1))
        found = find_matrix_file(tmp_path, subject_id, "fa")
        assert found == path

    def test_finds_brainnetome_style(self, single_subject_dirs):
        subject_id = str(single_subject_dirs["subject_id"])
        found = find_matrix_file(single_subject_dirs["dmri"], subject_id, "fa")
        assert found.name == dmri_filename(subject_id, "fa")

    def test_short_metric_does_not_match_afd(self, tmp_path: Path):
        subject_id = SUBJECT_IDS[0]
        np.save(tmp_path / f"{subject_id}_stat-afd.npy", sym_matrix(seed=2))
        with pytest.raises(FileNotFoundError, match="No .npy file"):
            find_matrix_file(tmp_path, subject_id, "fa")

    def test_missing_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="No .npy file"):
            find_matrix_file(tmp_path, SUBJECT_IDS[0], "fa")

    def test_ambiguous_raises(self, tmp_path: Path):
        subject_id = SUBJECT_IDS[0]
        np.save(tmp_path / f"{subject_id}_run-01_stat-fa.npy", sym_matrix(seed=3))
        np.save(tmp_path / f"{subject_id}_run-02_stat-fa.npy", sym_matrix(seed=4))
        with pytest.raises(ValueError, match="Ambiguous matrix files"):
            find_matrix_file(tmp_path, subject_id, "fa")


# ---------------------------------------------------------------------------
# create_pyg_data
# ---------------------------------------------------------------------------


class TestCreatePygData:
    def test_basic_graph_shapes(self, single_subject_dirs):
        data = create_pyg_data(
            dmri_path=str(single_subject_dirs["dmri"]),
            primary_dmri_key=PRIMARY_DMRI_KEY,
            fc_path=str(single_subject_dirs["fc"]),
            fc_metric="pearsoncorrelation",
            edge_attributes=list(EDGE_ATTRIBUTES),
            subject_id=str(single_subject_dirs["subject_id"]),
            threshold=0.0,
        )
        assert isinstance(data, Data)
        assert data.num_nodes == NUM_NODES
        assert data.x.shape == (NUM_NODES, 1)  # constant ones
        assert data.edge_index.shape[0] == 2
        assert data.edge_weight.shape[0] == data.edge_index.shape[1]
        # sorted(edge_attributes) + fc
        assert data.edge_attr.shape == (data.edge_index.shape[1], len(EDGE_ATTRIBUTES) + 1)
        assert data.subject_id == single_subject_dirs["subject_id"]

    def test_anatomical_features(self, single_subject_dirs):
        anat = np.arange(NUM_NODES, dtype=np.float32).reshape(-1, 1) + 1.0
        data = create_pyg_data(
            dmri_path=str(single_subject_dirs["dmri"]),
            primary_dmri_key=PRIMARY_DMRI_KEY,
            fc_path=str(single_subject_dirs["fc"]),
            fc_metric="pearsoncorrelation",
            edge_attributes=list(EDGE_ATTRIBUTES),
            anatomical_features=anat,
            subject_id=str(single_subject_dirs["subject_id"]),
            threshold=0.0,
        )
        torch.testing.assert_close(data.x, torch.tensor(anat))

    def test_threshold_reduces_edges(self, single_subject_dirs):
        kwargs = dict(
            dmri_path=str(single_subject_dirs["dmri"]),
            primary_dmri_key=PRIMARY_DMRI_KEY,
            fc_path=str(single_subject_dirs["fc"]),
            fc_metric="pearsoncorrelation",
            edge_attributes=list(EDGE_ATTRIBUTES),
            subject_id=str(single_subject_dirs["subject_id"]),
        )
        dense = create_pyg_data(**kwargs, threshold=0.0)
        sparse = create_pyg_data(**kwargs, threshold=0.5)
        assert sparse.edge_index.shape[1] < dense.edge_index.shape[1]

    def test_top_k_reduces_edges(self, single_subject_dirs):
        kwargs = dict(
            dmri_path=str(single_subject_dirs["dmri"]),
            primary_dmri_key=PRIMARY_DMRI_KEY,
            fc_path=str(single_subject_dirs["fc"]),
            fc_metric="pearsoncorrelation",
            edge_attributes=list(EDGE_ATTRIBUTES),
            subject_id=str(single_subject_dirs["subject_id"]),
        )
        dense = create_pyg_data(**kwargs, threshold=0.0)
        sparse = create_pyg_data(**kwargs, top_k=1)
        assert sparse.edge_index.shape[1] <= dense.edge_index.shape[1]

    def test_log_transform(self, single_subject_dirs):
        data = create_pyg_data(
            dmri_path=str(single_subject_dirs["dmri"]),
            primary_dmri_key=PRIMARY_DMRI_KEY,
            fc_path=str(single_subject_dirs["fc"]),
            fc_metric="pearsoncorrelation",
            edge_attributes=list(EDGE_ATTRIBUTES),
            subject_id=str(single_subject_dirs["subject_id"]),
            threshold=0.0,
            log_transform=True,
        )
        primary = np.load(
            Path(single_subject_dirs["dmri"])
            / dmri_filename(str(single_subject_dirs["subject_id"]), PRIMARY_DMRI_KEY)
        )
        # Spot-check one edge weight equals log1p of primary matrix entry
        src = int(data.edge_index[0, 0])
        dst = int(data.edge_index[1, 0])
        expected = float(np.log1p(abs(primary[src, dst])) * np.sign(primary[src, dst]))
        assert data.edge_weight[0].item() == pytest.approx(expected, rel=1e-5)

    def test_missing_primary_key_raises(self, single_subject_dirs):
        with pytest.raises(ValueError, match="must appear in edge_attributes"):
            create_pyg_data(
                dmri_path=str(single_subject_dirs["dmri"]),
                primary_dmri_key="missing",
                fc_path=str(single_subject_dirs["fc"]),
                fc_metric="pearsoncorrelation",
                edge_attributes=list(EDGE_ATTRIBUTES),
                subject_id=str(single_subject_dirs["subject_id"]),
                threshold=0.0,
            )

    def test_shape_mismatch_raises(self, tmp_path: Path):
        dmri_dir = tmp_path / "dmri"
        fc_dir = tmp_path / "fc"
        subject_id = SUBJECT_IDS[0]
        write_subject_matrices(dmri_dir, fc_dir, subject_id, seed=1)
        # Overwrite md with wrong shape
        np.save(dmri_dir / dmri_filename(subject_id, "md"), np.ones((3, 3), dtype=np.float32))
        with pytest.raises(ValueError, match="Shape mismatch"):
            create_pyg_data(
                dmri_path=str(dmri_dir),
                primary_dmri_key=PRIMARY_DMRI_KEY,
                fc_path=str(fc_dir),
                fc_metric="pearsoncorrelation",
                edge_attributes=list(EDGE_ATTRIBUTES),
                subject_id=subject_id,
                threshold=0.0,
            )

    def test_alternate_naming_scheme(self, tmp_path: Path):
        subject_id = SUBJECT_IDS[0]
        dmri_dir = tmp_path / "dmri"
        fc_dir = tmp_path / "fc"
        dmri_dir.mkdir()
        fc_dir.mkdir()
        np.save(dmri_dir / f"{subject_id}_atlas-Schaefer_metric-fa.npy", sym_matrix(seed=21))
        np.save(dmri_dir / f"{subject_id}_atlas-Schaefer_metric-md.npy", sym_matrix(seed=22))
        np.save(fc_dir / f"{subject_id}_atlas-Schaefer_metric-pearson.npy", sym_matrix(seed=23))

        data = create_pyg_data(
            dmri_path=str(dmri_dir),
            primary_dmri_key="fa",
            fc_path=str(fc_dir),
            edge_attributes=["fa", "md"],
            subject_id=subject_id,
            threshold=0.0,
            fc_metric="pearson",
        )
        assert data.num_nodes == NUM_NODES
        assert data.edge_attr.shape[1] == 3
        assert data.subject_id == subject_id


# ---------------------------------------------------------------------------
# BrainGraphDataset + save/load
# ---------------------------------------------------------------------------


class TestBrainGraphDataset:
    def test_builds_two_subject_dataset(self, brain_graph_fixture):
        dataset = BrainGraphDataset(
            subjects_file=str(brain_graph_fixture["subjects_file"]),
            dmri_folder=str(brain_graph_fixture["dmri_folder"]),
            fc_folder=str(brain_graph_fixture["fc_folder"]),
            fc_metric="pearsoncorrelation",
            anatomical_df=str(brain_graph_fixture["anatomical_df"]),
            primary_dmri_key=str(brain_graph_fixture["primary_dmri_key"]),
            edge_attributes=list(brain_graph_fixture["edge_attributes"]),
            threshold=0.0,
            cache_dir=str(brain_graph_fixture["cache_dir"]),
            num_workers=1,
        )
        assert len(dataset) == 2
        assert "volume" in dataset.node_feature_indices
        assert "degree" not in dataset.node_feature_indices
        assert set(dataset.edge_attribute_indices) == {"fa", "md", "fc"}
        assert dataset.edge_attribute_indices["fa"] < dataset.edge_attribute_indices["md"]
        assert dataset.edge_attribute_indices["fc"] == 2

        graph = dataset[0]
        assert graph.x.shape == (NUM_NODES, 1)
        assert graph.num_nodes == NUM_NODES
        assert graph.subject_id in SUBJECT_IDS

    def test_primary_key_missing_from_edge_attributes_raises(self, tmp_path: Path):
        subjects_file = tmp_path / "subjects.txt"
        subjects_file.write_text("sub-001\n", encoding="utf-8")
        with pytest.raises(ValueError, match="must appear in edge_attributes"):
            BrainGraphDataset(
                subjects_file=str(subjects_file),
                dmri_folder=str(tmp_path),
                fc_folder=str(tmp_path),
                fc_metric="pearsoncorrelation",
                anatomical_df=None,
                primary_dmri_key="fa",
                edge_attributes=[],
            )

    def test_subject_failure_does_not_save_partial_dataset(self, brain_graph_fixture):
        root = Path(brain_graph_fixture["cache_dir"]).parent
        subjects_file = root / "subjects_with_missing.txt"
        subjects_file.write_text("\n".join([*SUBJECT_IDS, "sub-missing"]) + "\n", encoding="utf-8")
        cache_dir = root / "partial_cache"

        with pytest.raises(FileNotFoundError, match="sub-missing"):
            BrainGraphDataset(
                subjects_file=str(subjects_file),
                dmri_folder=str(brain_graph_fixture["dmri_folder"]),
                fc_folder=str(brain_graph_fixture["fc_folder"]),
                fc_metric="pearsoncorrelation",
                anatomical_df=None,
                primary_dmri_key=str(brain_graph_fixture["primary_dmri_key"]),
                edge_attributes=list(brain_graph_fixture["edge_attributes"]),
                threshold=0.0,
                cache_dir=str(cache_dir),
                num_workers=2,
            )

        assert list(cache_dir.rglob("data.pt")) == []


class TestSaveLoadDataset:
    def test_roundtrip_with_metadata(self, tmp_path: Path):
        graphs = [
            Data(
                x=torch.ones((NUM_NODES, 1)),
                edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
                edge_weight=torch.tensor([0.5, 0.5]),
                num_nodes=NUM_NODES,
            )
        ]
        graphs[0].subject_id = "sub-001"
        path = tmp_path / "graphs.pt"
        metadata = {"primary_dmri_key": "fa"}
        save_dataset(graphs, str(path), metadata=metadata)

        loaded, loaded_meta = load_dataset(str(path), return_metadata=True)
        assert len(loaded) == 1
        assert loaded[0].subject_id == "sub-001"
        assert loaded_meta == metadata
        torch.testing.assert_close(loaded[0].x, graphs[0].x)

    def test_load_missing_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_dataset(str(tmp_path / "missing.pt"))
