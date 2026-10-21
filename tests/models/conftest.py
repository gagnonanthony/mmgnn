"""Shared fixtures for mmgnn.models unit tests."""

from typing import Any

import pytest
import torch
from torch_geometric.data import Data

from mmgnn.models.gae import GAE
from mmgnn.models.utils import build_encoder, build_model
from mmgnn.models.vgae import VGAE

NUM_NODES = 4
NODE_IN = 4
EDGE_IN = 2
HIDDEN = 8
NUM_LAYERS = 2


def tiny_graph(
    *,
    num_nodes: int = NUM_NODES,
    node_in: int = NODE_IN,
    edge_in: int = EDGE_IN,
) -> Data:
    """Bidirectional cycle with dense node/edge features."""
    src = list(range(num_nodes))
    dst = [(i + 1) % num_nodes for i in range(num_nodes)]
    edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long)
    x = torch.randn(num_nodes, node_in, dtype=torch.float32)
    edge_attr = torch.randn(edge_index.shape[1], edge_in, dtype=torch.float32)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=num_nodes)


def pos_neg_edges(graph: Data) -> tuple[torch.Tensor, torch.Tensor]:
    """Positive edges from the graph; a few non-edges as negatives."""
    pos = graph.edge_index
    n = int(graph.num_nodes)
    existing = {(int(s), int(d)) for s, d in zip(pos[0].tolist(), pos[1].tolist())}
    neg_pairs: list[tuple[int, int]] = []
    for i in range(n):
        for j in range(n):
            if i == j or (i, j) in existing:
                continue
            neg_pairs.append((i, j))
            if len(neg_pairs) >= pos.shape[1]:
                break
        if len(neg_pairs) >= pos.shape[1]:
            break
    if not neg_pairs:
        raise RuntimeError("Could not find negative edges for tiny graph")
    neg = torch.tensor(neg_pairs, dtype=torch.long).t().contiguous()
    return pos, neg


def encoder_config(**overrides: Any) -> dict[str, Any]:
    config = {
        "hidden_dim": HIDDEN,
        "num_heads": 4,
        "num_layers": NUM_LAYERS,
        "conv_type": "gine",
        "act_func": "relu",
        "norm_type": "batch",
        "dropout": 0.0,
        "edge_dropout": 0.0,
        "residual": False,
    }
    config.update(overrides)
    return config


def model_config(**overrides: Any) -> dict[str, Any]:
    config = {
        **encoder_config(),
        "model_type": "gae",
        "edge_presence_decoder_type": "inner_product",
        "edge_attr_decoder_type": "mlp",
        "decode_node_features": True,
        "dropout": 0.0,
        "kl_weight": 1.0,
    }
    config.update(overrides)
    return config


@pytest.fixture
def graph() -> Data:
    torch.manual_seed(0)
    return tiny_graph()


@pytest.fixture
def built_gae(graph: Data) -> GAE:
    cfg = model_config(model_type="gae")
    encoder = build_encoder(
        num_node_features=graph.x.size(-1),
        num_edge_features=graph.edge_attr.size(-1),
        model_config=cfg,
    )
    return build_model(
        encoder=encoder,
        num_node_features=graph.x.size(-1),
        num_edge_features=graph.edge_attr.size(-1),
        model_config=cfg,
    )


@pytest.fixture
def built_vgae(graph: Data) -> VGAE:
    cfg = model_config(model_type="vgae", kl_weight=0.5)
    encoder = build_encoder(
        num_node_features=graph.x.size(-1),
        num_edge_features=graph.edge_attr.size(-1),
        model_config=cfg,
    )
    model = build_model(
        encoder=encoder,
        num_node_features=graph.x.size(-1),
        num_edge_features=graph.edge_attr.size(-1),
        model_config=cfg,
    )
    assert isinstance(model, VGAE)
    return model
