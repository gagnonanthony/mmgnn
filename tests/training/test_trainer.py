"""Unit tests for EarlyStopping and Trainer."""

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.optim import Adam

from mmgnn.training.trainer import EarlyStopping, Trainer
from tests.models.conftest import EDGE_IN, NUM_NODES, tiny_graph
from tests.training.conftest import FakeTrainerModel, make_loaders


def _make_trainer(tmp_path: Path, model: nn.Module | None = None) -> Trainer:
    train_loader, val_loader, _ = make_loaders()
    model = model or FakeTrainerModel()
    return Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=Adam,
        patience=5,
        checkpoint_dir=tmp_path / "ckpts",
        device=torch.device("cpu"),
        recon_edge_presence_weight=1.0,
        recon_node_attr_weight=1.0,
        recon_edge_attr_weight=1.0,
    )


class TestEarlyStopping:
    def test_improvement_resets_counter(self):
        stopper = EarlyStopping(patience=2, mode="min")
        assert stopper(1.0) is False  # baseline does not consume patience
        assert stopper.counter == 0
        assert stopper.best_score == 1.0
        assert stopper(0.5) is False  # improvement
        assert stopper.counter == 0
        assert stopper.best_score == 0.5

    def test_plateau_stops_after_patience(self):
        stopper = EarlyStopping(patience=2, mode="min")
        assert stopper(1.0) is False  # baseline, counter stays 0
        assert stopper(1.0) is False  # counter=1
        assert stopper(1.0) is True  # counter=2 >= patience
        assert stopper.should_stop is True

    def test_patience_one_survives_baseline(self):
        stopper = EarlyStopping(patience=1, mode="min")
        assert stopper(1.0) is False
        assert stopper.counter == 0
        assert stopper(1.0) is True


class TestDenseFromEdgeIndex:
    def test_scalar_and_multichannel(self):
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        values = torch.tensor([2.0, 4.0])
        dense = Trainer._dense_from_edge_index(edge_index, values, num_nodes=2, symmetrize=True)
        assert dense.shape == (2, 2)
        assert dense[0, 1] == pytest.approx(3.0)  # (2+4)/2
        assert dense[1, 0] == pytest.approx(3.0)

        multi = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        dense_c = Trainer._dense_from_edge_index(edge_index, multi, num_nodes=2, symmetrize=False)
        assert dense_c.shape == (2, 2, 2)
        np.testing.assert_allclose(dense_c[0, 1], [1.0, 2.0])


class TestTrainerInit:
    def test_no_trainable_params_raises(self, tmp_path: Path):
        class Frozen(nn.Module):
            def __init__(self):
                super().__init__()
                self.w = nn.Parameter(torch.ones(1), requires_grad=False)

            def encode(self, *args, **kwargs):
                return torch.zeros(1, 1)

            def recon_total_loss(self, *args, **kwargs):
                return {"total_loss": torch.tensor(0.0)}

        train_loader, val_loader, _ = make_loaders()
        with pytest.raises(ValueError, match="No trainable parameters"):
            Trainer(
                model=Frozen(),
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=Adam,
                checkpoint_dir=tmp_path / "ckpts",
                device=torch.device("cpu"),
            )

    def test_bad_scheduler_raises(self, tmp_path: Path):
        train_loader, val_loader, _ = make_loaders()
        with pytest.raises(ValueError, match="Unsupported scheduler"):
            Trainer(
                model=FakeTrainerModel(),
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=Adam,
                scheduler="cosine",
                checkpoint_dir=tmp_path / "ckpts",
                device=torch.device("cpu"),
            )


class TestTrainerLoop:
    def test_one_epoch_and_evaluate(self, tmp_path: Path):
        trainer = _make_trainer(tmp_path)
        trainer.train(num_epochs=1, verbose=False)
        assert len(trainer.history["train_loss"]) == 1
        assert np.isfinite(trainer.history["train_loss"][0])
        assert np.isfinite(trainer.history["val_loss"][0])

        metrics = trainer.evaluate(trainer.val_loader)
        assert np.isfinite(metrics["loss"])

    def test_checkpoint_roundtrip(self, tmp_path: Path):
        trainer = _make_trainer(tmp_path)
        trainer.train(num_epochs=1, verbose=False)
        ckpt = trainer.checkpoint_dir / "checkpoint_epoch_1.pt"
        assert ckpt.is_file()

        trainer2 = _make_trainer(tmp_path / "other")
        trainer2.load_checkpoint(ckpt)
        assert trainer2.best_val_loss == pytest.approx(trainer.best_val_loss)
        assert len(trainer2.history["train_loss"]) == 1


class TestMatrixAPIs:
    def test_presence_recon_and_empirical_shapes(self, tmp_path: Path):
        trainer = _make_trainer(tmp_path)
        graph = tiny_graph()
        presence = trainer.predict_edge_presence_matrix(graph)
        assert presence.shape == (NUM_NODES, NUM_NODES)

        recon = trainer.reconstruct_edge_matrices(graph, feature_names=["fa", "md"])
        assert set(recon) == {"fa", "md"}
        assert recon["fa"].shape == (NUM_NODES, NUM_NODES)

        empirical = trainer.get_empirical_edge_matrices(graph, feature_names=["fa", "md"])
        assert set(empirical) == {"fa", "md"}
        assert empirical["fa"].shape == (NUM_NODES, NUM_NODES)
        assert EDGE_IN == 2
