"""Unit tests for mmgnn.inference.embeddings."""

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool

from mmgnn.inference.embeddings import (
    extract_embeddings,
    save_subject_node_embeddings,
    save_subject_node_embeddings_batch,
    load_subject_node_embeddings
)
from tests.explain.helpers import NUM_NODES, make_graph

LATENT_DIM = 3
DEVICE = torch.device("cpu")


# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------


class FakeGAE(nn.Module):
    """Deterministic encode + mean-pool graph embeddings."""

    def encode(self, x, edge_index, edge_attr=None, batch=None) -> torch.Tensor:
        # Row i is filled with float(i) across latent dims.
        vals = torch.arange(x.size(0), device=x.device, dtype=torch.float32).unsqueeze(1)
        return vals.expand(-1, LATENT_DIM).contiguous()

    def forward_embeddings(self, data):
        z = self.encode(
            data.x,
            data.edge_index,
            getattr(data, "edge_attr", None),
            getattr(data, "batch", None),
        )
        batch = getattr(data, "batch", None)
        if batch is None:
            graph_emb = z.mean(dim=0, keepdim=True)
        else:
            graph_emb = global_mean_pool(z, batch)
        return None, graph_emb


class FakeVGAE(FakeGAE):
    def encode_stats(self, x, edge_index, edge_attr=None, batch=None):
        mu = torch.ones(x.size(0), LATENT_DIM, device=x.device)
        logvar = torch.full((x.size(0), LATENT_DIM), -2.0, device=x.device)
        return mu, logvar


# ---------------------------------------------------------------------------
# extract_embeddings
# ---------------------------------------------------------------------------


class TestExtractEmbeddings:
    def test_gae_single_and_batch(self):
        graphs = [
            make_graph(subject_id="sub-001"),
            make_graph(subject_id="sub-002"),
        ]
        model = FakeGAE()

        for batch_size in (1, 2):
            graph_emb, node_by_sid, sids, logvar, sigma = extract_embeddings(
                model, graphs, device=DEVICE, batch_size=batch_size
            )
            assert graph_emb.shape == (2, LATENT_DIM)
            assert sids == ["sub-001", "sub-002"]
            assert set(node_by_sid) == {"sub-001", "sub-002"}
            assert node_by_sid["sub-001"].shape == (NUM_NODES, LATENT_DIM)
            assert logvar is None and sigma is None

    def test_vgae_returns_uncertainty(self):
        graphs = [make_graph(subject_id="sub-001")]
        graph_emb, node_by_sid, sids, logvar, sigma = extract_embeddings(
            FakeVGAE(), graphs, device=DEVICE, batch_size=1
        )
        assert graph_emb.shape[0] == 1
        assert sids == ["sub-001"]
        assert node_by_sid["sub-001"].shape == (NUM_NODES, LATENT_DIM)
        assert logvar is not None and sigma is not None
        np.testing.assert_allclose(logvar["sub-001"], -2.0)
        np.testing.assert_allclose(sigma["sub-001"], np.exp(-1.0))

    def test_empty_graphs_raises(self):
        with pytest.raises(ValueError, match="No embeddings"):
            extract_embeddings(FakeGAE(), [], device=DEVICE, batch_size=1)


# ---------------------------------------------------------------------------
# load / save
# ---------------------------------------------------------------------------


class TestLoadSubjectNodeEmbeddings:
    def test_round_trip(self, tmp_path: Path):
        arr = np.arange(NUM_NODES * LATENT_DIM, dtype=np.float32).reshape(NUM_NODES, LATENT_DIM)
        save_subject_node_embeddings(tmp_path, "sub-001", node_embeddings=arr)
        loaded = load_subject_node_embeddings(tmp_path, "sub-001")
        np.testing.assert_array_equal(loaded, arr)

    def test_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="Node embeddings not found"):
            load_subject_node_embeddings(tmp_path, "missing")

    def test_non_2d_raises(self, tmp_path: Path):
        subject_dir = tmp_path / "subjects" / "sub-001"
        subject_dir.mkdir(parents=True)
        np.save(subject_dir / "node_embeddings.npy", np.arange(3, dtype=np.float32))
        with pytest.raises(ValueError, match="shape"):
            load_subject_node_embeddings(tmp_path, "sub-001")


class TestSaveSubjectNodeEmbeddings:
    def test_writes_embeddings_and_summary(self, tmp_path: Path):
        arr = np.ones((NUM_NODES, LATENT_DIM), dtype=np.float32)
        summary = save_subject_node_embeddings(tmp_path, "sub-001", node_embeddings=arr)
        subject_dir = tmp_path / "subjects" / "sub-001"
        assert (subject_dir / "node_embeddings.npy").is_file()
        assert (subject_dir / "summary.json").is_file()
        assert summary["has_uncertainty"] is False
        assert summary["num_nodes"] == NUM_NODES
        assert summary["embedding_dim"] == LATENT_DIM
        assert not (subject_dir / "node_logvar.npy").exists()

    def test_writes_uncertainty_maps(self, tmp_path: Path):
        arr = np.ones((NUM_NODES, LATENT_DIM), dtype=np.float32)
        logvar = np.full_like(arr, -1.0)
        sigma = np.exp(0.5 * logvar)
        summary = save_subject_node_embeddings(
            tmp_path,
            "sub-001",
            node_embeddings=arr,
            node_logvar=logvar,
            node_sigma=sigma,
        )
        subject_dir = tmp_path / "subjects" / "sub-001"
        assert (subject_dir / "node_logvar.npy").is_file()
        assert (subject_dir / "node_sigma.npy").is_file()
        assert summary["has_uncertainty"] is True


class TestSaveSubjectNodeEmbeddingsBatch:
    def test_saves_multiple_subjects(self, tmp_path: Path):
        by_subject = {
            "sub-001": np.ones((NUM_NODES, LATENT_DIM), dtype=np.float32),
            "sub-002": np.zeros((NUM_NODES, LATENT_DIM), dtype=np.float32),
        }
        logvar = {sid: np.full_like(arr, -2.0) for sid, arr in by_subject.items()}
        sigma = {sid: np.exp(0.5 * lv) for sid, lv in logvar.items()}

        summaries = save_subject_node_embeddings_batch(
            tmp_path,
            by_subject,
            node_logvar_by_subject=logvar,
            node_sigma_by_subject=sigma,
        )
        assert len(summaries) == 2
        assert all(s["has_uncertainty"] for s in summaries)
        for sid in by_subject:
            assert (tmp_path / "subjects" / sid / "node_embeddings.npy").is_file()
            assert (tmp_path / "subjects" / sid / "node_logvar.npy").is_file()
