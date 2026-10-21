"""Unit tests for training plot helpers."""

from pathlib import Path

import numpy as np
import pytest

from mmgnn.training.plots import (
    _compute_embedding_tsne,
    plot_connectivity_heatmap,
    plot_connectivity_triptych,
    plot_contingency_heatmap,
    plot_embeddings,
    plot_training_history,
)


class TestComputeEmbeddingTsne:
    def test_shape(self):
        emb = np.random.randn(20, 4)
        coords, indices, mask = _compute_embedding_tsne(
            emb, perplexity=5.0, random_state=0, max_iter=250
        )
        assert coords.shape == (20, 2)
        assert indices.shape == (20,)
        assert mask.shape == (20,)
        assert mask.all()

    def test_too_few_rows_raises(self):
        with pytest.raises(ValueError, match="at least 2 finite"):
            _compute_embedding_tsne(
                np.ones((1, 3)), perplexity=0.5, random_state=0, max_iter=100
            )

    def test_bad_perplexity_raises(self):
        emb = np.random.randn(10, 3)
        with pytest.raises(ValueError, match="perplexity must be > 0"):
            _compute_embedding_tsne(emb, perplexity=0.0, random_state=0, max_iter=100)
        with pytest.raises(ValueError, match="smaller than the number of samples"):
            _compute_embedding_tsne(emb, perplexity=10.0, random_state=0, max_iter=100)


class TestPlotTrainingHistory:
    def test_empty_history_raises(self):
        with pytest.raises(ValueError, match="any training epochs"):
            plot_training_history({"train_loss": []})

    def test_non_finite_only_raises(self):
        with pytest.raises(ValueError, match="finite GAE reconstruction"):
            plot_training_history(
                {
                    "train_loss": [np.nan],
                    "val_loss": [np.nan],
                    "train_edge_presence_loss": [np.nan],
                    "val_edge_presence_loss": [np.nan],
                    "train_node_attr_loss": [np.nan],
                    "val_node_attr_loss": [np.nan],
                    "train_edge_attr_loss": [np.nan],
                    "val_edge_attr_loss": [np.nan],
                }
            )

    def test_writes_save_path(self, tmp_path: Path):
        history = {
            "train_loss": [1.0, 0.8],
            "val_loss": [1.1, 0.9],
            "train_edge_presence_loss": [0.5, 0.4],
            "val_edge_presence_loss": [0.6, 0.5],
            "train_node_attr_loss": [0.2, 0.1],
            "val_node_attr_loss": [0.3, 0.2],
            "train_edge_attr_loss": [0.1, 0.05],
            "val_edge_attr_loss": [0.15, 0.1],
        }
        out = tmp_path / "history.png"
        plot_training_history(history, save_path=str(out), best_epoch=2)
        assert out.is_file()


class TestHeatmapValidation:
    def test_connectivity_heatmap_rejects_bad_input(self):
        with pytest.raises(ValueError, match="2D"):
            plot_connectivity_heatmap(np.arange(3), title="t")
        with pytest.raises(ValueError, match="non-empty"):
            plot_connectivity_heatmap(np.zeros((0, 0)), title="t")
        with pytest.raises(ValueError, match="finite"):
            plot_connectivity_heatmap(np.array([[np.nan, 1.0], [1.0, 0.0]]), title="t")

    def test_contingency_heatmap_rejects_bad_input(self):
        with pytest.raises(ValueError, match="2D"):
            plot_contingency_heatmap(np.arange(4))

    def test_triptych_rejects_shape_mismatch(self):
        a = np.eye(3)
        b = np.eye(2)
        with pytest.raises(ValueError, match="Matrix shape mismatch"):
            plot_connectivity_triptych(a, b, title="t")


class TestPlotEmbeddings:
    def test_too_few_samples_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            plot_embeddings(np.ones((1, 4)), perplexity=0.5)

    def test_save_smoke(self, tmp_path: Path):
        emb = np.random.randn(15, 4)
        out = tmp_path / "emb.png"
        coords = plot_embeddings(emb, save_path=str(out), perplexity=5.0, max_iter=250)
        assert out.is_file()
        assert coords.shape == (15, 2)
