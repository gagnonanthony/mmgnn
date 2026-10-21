"""Unit tests for mmgnn.models.blocks (covers gine_edge_mlp / modules)."""

import pytest
import torch

from mmgnn.models.blocks import (
    GraphConvBlock,
    add_act_func,
    add_conv_layer,
    add_norm_layer,
    add_readout,
)
from tests.models.conftest import EDGE_IN, HIDDEN, NODE_IN, tiny_graph


class TestFactories:
    def test_act_norm_readout_happy_path(self):
        assert callable(add_act_func("relu"))
        assert add_norm_layer("batch", HIDDEN) is not None
        assert callable(add_readout("mean"))

    def test_unknown_names_raise(self):
        with pytest.raises(ValueError, match="Unsupported activation"):
            add_act_func("not_an_act")
        with pytest.raises(ValueError, match="Unsupported normalization"):
            add_norm_layer("not_a_norm", HIDDEN)
        with pytest.raises(ValueError, match="Unsupported convolution"):
            add_conv_layer("not_a_conv", NODE_IN, HIDDEN)
        with pytest.raises(ValueError, match="Unsupported readout"):
            add_readout("not_a_pool")

    @pytest.mark.parametrize("edge_dim", [None, 0])
    def test_gine_edge_mlp_requires_edge_dim(self, edge_dim):
        with pytest.raises(ValueError, match="positive edge_dim"):
            add_conv_layer("gine_edge_mlp", NODE_IN, HIDDEN, edge_dim=edge_dim)


class TestGraphConvBlock:
    @pytest.mark.parametrize(
        "conv_type,num_heads,needs_edge_attr",
        [
            ("gcn", None, False),
            ("gine", None, True),
            ("gine_edge_mlp", None, True),
            ("gat", 2, True),
        ],
    )
    def test_forward_shape(self, conv_type, num_heads, needs_edge_attr):
        graph = tiny_graph()
        block = GraphConvBlock(
            in_channels=NODE_IN,
            out_channels=HIDDEN,
            conv_type=conv_type,
            edge_dim=EDGE_IN if needs_edge_attr else None,
            num_heads=num_heads,
        )
        edge_attr = graph.edge_attr if needs_edge_attr else None
        if needs_edge_attr:
            out = block(graph.x, graph.edge_index, edge_attr)
        else:
            out = block(graph.x, graph.edge_index)
        assert out.shape == (graph.num_nodes, HIDDEN)

    def test_residual_same_channels(self):
        graph = tiny_graph(node_in=HIDDEN)
        block = GraphConvBlock(
            in_channels=HIDDEN,
            out_channels=HIDDEN,
            conv_type="gcn",
            residual=True,
        )
        out = block(graph.x, graph.edge_index)
        assert out.shape == (graph.num_nodes, HIDDEN)

    def test_edge_dropout_train_mode(self):
        graph = tiny_graph()
        block = GraphConvBlock(
            in_channels=NODE_IN,
            out_channels=HIDDEN,
            conv_type="gine",
            edge_dim=EDGE_IN,
            edge_dropout=0.5,
        )
        block.train()
        out = block(graph.x, graph.edge_index, graph.edge_attr)
        assert out.shape == (graph.num_nodes, HIDDEN)
        assert torch.isfinite(out).all()
