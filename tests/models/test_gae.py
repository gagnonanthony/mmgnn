"""Unit tests for Encoder, decoders, and GAE."""

import pytest
import torch

from mmgnn.models.gae import (
    EdgeMLPDecoder,
    Encoder,
    GAE,
    InnerProductEdgeDecoder,
    NodeMLPDecoder,
)
from tests.models.conftest import (
    EDGE_IN,
    HIDDEN,
    NODE_IN,
    NUM_LAYERS,
    NUM_NODES,
    pos_neg_edges,
    tiny_graph,
)


class TestEncoder:
    def test_forward_shape_with_projection(self):
        # NODE_IN != HIDDEN → proj is used
        enc = Encoder(
            num_node_features=NODE_IN,
            num_edge_features=EDGE_IN,
            hidden_dim=HIDDEN,
            num_layers=NUM_LAYERS,
            conv_type="gine",
            dropout=0.0,
        )
        assert enc.proj is not None
        graph = tiny_graph()
        z = enc(graph.x, graph.edge_index, graph.edge_attr)
        assert z.shape == (NUM_NODES, HIDDEN)

    def test_no_projection_when_dims_match(self):
        enc = Encoder(
            num_node_features=HIDDEN,
            num_edge_features=EDGE_IN,
            hidden_dim=HIDDEN,
            num_layers=NUM_LAYERS,
            conv_type="gine",
            dropout=0.0,
        )
        assert enc.proj is None
        graph = tiny_graph(node_in=HIDDEN)
        z = enc(graph.x, graph.edge_index, graph.edge_attr)
        assert z.shape == (NUM_NODES, HIDDEN)


class TestDecoders:
    def test_inner_product_and_mlp_shapes(self):
        z = torch.randn(NUM_NODES, HIDDEN)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
        e = edge_index.shape[1]

        ip = InnerProductEdgeDecoder()
        assert ip(z, edge_index).shape == (e,)

        edge_mlp = EdgeMLPDecoder(latent_dim=HIDDEN, out_dim=EDGE_IN)
        assert edge_mlp(z, edge_index).shape == (e, EDGE_IN)

        node_mlp = NodeMLPDecoder(latent_dim=HIDDEN, out_dim=NODE_IN)
        assert node_mlp(z).shape == (NUM_NODES, NODE_IN)


class TestReconLoss:
    def test_total_loss_keys_and_finite(self, built_gae: GAE, graph):
        built_gae.eval()
        z = built_gae.encode(graph.x, graph.edge_index, graph.edge_attr)
        pos, neg = pos_neg_edges(graph)
        components = built_gae.recon_total_loss(
            z,
            pos_edge_index=pos,
            neg_edge_index=neg,
            target_x=graph.x,
            edge_index_for_attr=pos,
            target_edge_attr=graph.edge_attr,
        )
        assert "total_loss" in components
        assert "edge_presence_loss" in components
        assert "node_attr_loss" in components
        assert "edge_attr_loss" in components
        assert torch.isfinite(components["total_loss"])

    def test_zero_weight_skips_presence(self, built_gae: GAE, graph):
        built_gae.eval()
        z = built_gae.encode(graph.x, graph.edge_index, graph.edge_attr)
        components = built_gae.recon_total_loss(
            z,
            edge_presence_weight=0.0,
            target_x=graph.x,
            edge_attr_weight=0.0,
        )
        assert "edge_presence_loss" not in components
        assert "node_attr_loss" in components
        assert torch.isfinite(components["total_loss"])

    def test_missing_edges_when_weight_positive_raises(self, built_gae: GAE, graph):
        built_gae.eval()
        z = built_gae.encode(graph.x, graph.edge_index, graph.edge_attr)
        with pytest.raises(ValueError, match="pos_edge_index"):
            built_gae.recon_total_loss(z, edge_presence_weight=1.0)


class TestGAE:
    def test_encode_and_forward(self, built_gae: GAE, graph):
        built_gae.eval()
        z = built_gae.encode(graph.x, graph.edge_index, graph.edge_attr)
        assert z.shape == (NUM_NODES, HIDDEN)

        out = built_gae(graph)
        assert out["z"].shape == (NUM_NODES, HIDDEN)
        assert out["edge_presence_logits"].shape[0] == graph.edge_index.shape[1]
        assert out["edge_attr_logits"].shape == (graph.edge_index.shape[1], EDGE_IN)
        assert out["x_hat"].shape == graph.x.shape

    def test_forward_embeddings(self, built_gae: GAE, graph):
        built_gae.eval()
        _, graph_emb = built_gae.forward_embeddings(graph)
        assert graph_emb.shape == (1, HIDDEN)

    def test_architecture_config(self, built_gae: GAE):
        cfg = built_gae.get_architecture_config()
        assert cfg["model_type"] == "gae"
        assert cfg["edge_presence_decoder_type"] == "inner_product"
        assert cfg["decode_node_features"] is True
        assert cfg["edge_attr_out_dim"] == EDGE_IN
