"""Unit tests for model builders."""

import pytest

from mmgnn.models.gae import GAE
from mmgnn.models.utils import build_encoder, build_model
from mmgnn.models.vgae import VGAE
from tests.models.conftest import EDGE_IN, HIDDEN, NODE_IN, encoder_config, model_config, tiny_graph


class TestBuildEncoder:
    def test_clears_num_heads_for_non_gat(self):
        cfg = encoder_config(conv_type="gine", num_heads=4)
        enc = build_encoder(
            num_node_features=NODE_IN,
            num_edge_features=EDGE_IN,
            model_config=cfg,
        )
        # Heads are forced to None for non-GAT; encoder still builds.
        assert enc.hidden_dim == HIDDEN
        assert enc.layers[0].conv.__class__.__name__ != "GATConv"

    def test_keeps_num_heads_for_gat(self):
        cfg = encoder_config(conv_type="gat", num_heads=2, residual=False)
        enc = build_encoder(
            num_node_features=NODE_IN,
            num_edge_features=EDGE_IN,
            model_config=cfg,
        )
        assert enc.layers[0].conv.__class__.__name__ == "GATConv"


class TestBuildModel:
    def test_returns_gae_and_vgae(self):
        enc = build_encoder(
            num_node_features=NODE_IN,
            num_edge_features=EDGE_IN,
            model_config=encoder_config(),
        )
        gae = build_model(
            encoder=enc,
            num_node_features=NODE_IN,
            num_edge_features=EDGE_IN,
            model_config=model_config(model_type="gae"),
        )
        assert isinstance(gae, GAE)
        assert not isinstance(gae, VGAE)

        enc2 = build_encoder(
            num_node_features=NODE_IN,
            num_edge_features=EDGE_IN,
            model_config=encoder_config(),
        )
        vgae = build_model(
            encoder=enc2,
            num_node_features=NODE_IN,
            num_edge_features=EDGE_IN,
            model_config=model_config(model_type="vgae"),
        )
        assert isinstance(vgae, VGAE)

    def test_defaults_out_dims(self):
        enc = build_encoder(
            num_node_features=NODE_IN,
            num_edge_features=EDGE_IN,
            model_config=encoder_config(),
        )
        model = build_model(
            encoder=enc,
            num_node_features=NODE_IN,
            num_edge_features=EDGE_IN,
            model_config=model_config(),
        )
        assert model.edge_attr_out_dim == EDGE_IN
        assert model.node_out_dim == NODE_IN

    def test_invalid_model_type_raises(self):
        enc = build_encoder(
            num_node_features=NODE_IN,
            num_edge_features=EDGE_IN,
            model_config=encoder_config(),
        )
        with pytest.raises(ValueError, match="Unsupported model_type"):
            build_model(
                encoder=enc,
                num_node_features=NODE_IN,
                num_edge_features=EDGE_IN,
                model_config=model_config(model_type="not_a_model"),
            )

    def test_smoke_encode(self):
        graph = tiny_graph()
        cfg = model_config()
        enc = build_encoder(
            num_node_features=NODE_IN,
            num_edge_features=EDGE_IN,
            model_config=cfg,
        )
        model = build_model(
            encoder=enc,
            num_node_features=NODE_IN,
            num_edge_features=EDGE_IN,
            model_config=cfg,
        )
        model.eval()
        z = model.encode(graph.x, graph.edge_index, graph.edge_attr)
        assert z.shape == (graph.num_nodes, HIDDEN)
