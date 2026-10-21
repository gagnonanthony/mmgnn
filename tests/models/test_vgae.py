"""Unit tests for VGAE-specific behavior."""

import pytest
import torch

from mmgnn.models.gae import Encoder
from mmgnn.models.vgae import VGAE
from tests.models.conftest import (
    EDGE_IN,
    HIDDEN,
    NODE_IN,
    NUM_LAYERS,
    pos_neg_edges,
)


class TestVGAE:
    def test_kl_loss_zero_at_standard_normal(self):
        mu = torch.zeros(4, HIDDEN)
        logvar = torch.zeros(4, HIDDEN)
        assert float(VGAE.kl_loss(mu, logvar)) == pytest.approx(0.0, abs=1e-6)

    def test_reparameterize_scales_noise_by_exp_half_logvar(self, monkeypatch):
        mu = torch.tensor([[0.0, 1.0]])
        logvar = torch.tensor([[0.0, 2.0]])

        def ones_like(tensor):
            return torch.ones_like(tensor)

        monkeypatch.setattr(torch, "randn_like", ones_like)
        z = VGAE.reparameterize(mu, logvar)
        assert torch.allclose(z, mu + torch.exp(0.5 * logvar))

    def test_kl_weight_must_be_non_negative(self):
        encoder = Encoder(
            num_node_features=NODE_IN,
            num_edge_features=EDGE_IN,
            hidden_dim=HIDDEN,
            num_layers=NUM_LAYERS,
            dropout=0.0,
        )
        with pytest.raises(ValueError, match="kl_weight"):
            VGAE(encoder, kl_weight=-0.1)

    def test_encode_eval_is_mu_train_is_sample(self, built_vgae: VGAE, graph):
        built_vgae.eval()
        mu, _ = built_vgae.encode_stats(graph.x, graph.edge_index, graph.edge_attr)
        z_eval = built_vgae.encode(graph.x, graph.edge_index, graph.edge_attr)
        assert torch.allclose(z_eval, mu)

        built_vgae.train()
        torch.manual_seed(0)
        z_train = built_vgae.encode(graph.x, graph.edge_index, graph.edge_attr)
        assert z_train.shape == mu.shape
        assert not torch.allclose(z_train, mu)

    def test_recon_total_loss_adds_kl_and_clears_cache(self, built_vgae: VGAE, graph):
        built_vgae.train()
        z = built_vgae.encode(graph.x, graph.edge_index, graph.edge_attr)
        assert built_vgae._last_mu is not None
        assert built_vgae._last_logvar is not None

        pos, neg = pos_neg_edges(graph)
        components = built_vgae.recon_total_loss(
            z,
            pos_edge_index=pos,
            neg_edge_index=neg,
            target_x=graph.x,
            edge_index_for_attr=pos,
            target_edge_attr=graph.edge_attr,
        )
        assert "kl_loss" in components
        assert torch.isfinite(components["total_loss"])
        assert built_vgae._last_mu is None
        assert built_vgae._last_logvar is None
