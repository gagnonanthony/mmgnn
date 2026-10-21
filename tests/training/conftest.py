"""Shared fixtures for mmgnn.training unit tests."""

from typing import Any, Mapping, Sequence

import matplotlib
import pytest
import torch
import torch.nn as nn
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool

from tests.models.conftest import EDGE_IN, HIDDEN, NODE_IN, tiny_graph

matplotlib.use("Agg")


@pytest.fixture(autouse=True)
def _agg_backend():
    """Ensure plot tests never open a GUI backend."""
    matplotlib.use("Agg")


class TinyInMemoryDataset(InMemoryDataset):
    """Minimal InMemoryDataset built from a list of Data objects."""

    def __init__(
        self,
        graphs: Sequence[Data],
        *,
        node_feature_indices: Mapping[str, int] | None = None,
        edge_attribute_indices: Mapping[str, int] | None = None,
        target_indices: Mapping[str, int] | None = None,
    ):
        super().__init__(root="/tmp/mmgnn_tiny_dataset", transform=None, pre_transform=None)
        data, slices = self.collate(list(graphs))
        self._data = data
        self.slices = slices
        self._data_list = None
        self.node_feature_indices = dict(node_feature_indices or {})
        self.edge_attribute_indices = dict(edge_attribute_indices or {})
        self.target_indices = dict(target_indices or {})

    def _download(self):
        return

    def _process(self):
        return


def make_memory_dataset(
    graphs: Sequence[Data] | None = None,
    *,
    node_feature_indices: Mapping[str, int] | None = None,
    edge_attribute_indices: Mapping[str, int] | None = None,
    target_indices: Mapping[str, int] | None = None,
) -> TinyInMemoryDataset:
    if graphs is None:
        g0 = tiny_graph()
        g0.subject_id = "sub-001"
        g0.y = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
        g1 = tiny_graph()
        g1.subject_id = "sub-002"
        g1.y = torch.tensor([[3.0, 4.0]], dtype=torch.float32)
        graphs = [g0, g1]
        if node_feature_indices is None:
            node_feature_indices = {f"n{i}": i for i in range(NODE_IN)}
        if edge_attribute_indices is None:
            edge_attribute_indices = {f"e{i}": i for i in range(EDGE_IN)}
        if target_indices is None:
            target_indices = {"t0": 0, "t1": 1}
    return TinyInMemoryDataset(
        graphs,
        node_feature_indices=node_feature_indices,
        edge_attribute_indices=edge_attribute_indices,
        target_indices=target_indices,
    )


class FakeTrainerModel(nn.Module):
    """Lightweight model satisfying Trainer encode / recon / decode contracts."""

    def __init__(self, latent_dim: int = HIDDEN, edge_attr_dim: int = EDGE_IN):
        super().__init__()
        self.latent_dim = latent_dim
        self.edge_attr_dim = edge_attr_dim
        self.probe = nn.Parameter(torch.zeros(1))

    def encode(self, x, edge_index, edge_attr=None, batch=None) -> torch.Tensor:
        return torch.ones(x.size(0), self.latent_dim, device=x.device) + self.probe

    def recon_total_loss(self, z, **kwargs) -> dict[str, torch.Tensor]:
        # Differentiable w.r.t. z and probe so Trainer.backward works.
        presence = (z ** 2).mean() + self.probe.pow(2)
        node = presence * 0.3
        edge = presence * 0.2
        total = presence + node + edge
        return {
            "total_loss": total,
            "edge_presence_loss": presence,
            "node_attr_loss": node,
            "edge_attr_loss": edge,
        }

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

    def decode_edge_presence(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index
        return (z[src] * z[dst]).sum(dim=-1)

    def decode_edge_attrs(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index
        base = (z[src] * z[dst]).sum(dim=-1, keepdim=True)
        return base.expand(-1, self.edge_attr_dim)

    def get_architecture_config(self) -> dict[str, Any]:
        return {"model_type": "fake", "latent_dim": self.latent_dim}


def make_loaders(batch_size: int = 1) -> tuple[DataLoader, DataLoader, list[Data]]:
    graphs = []
    for sid in ("sub-001", "sub-002"):
        g = tiny_graph()
        g.subject_id = sid
        g.y = torch.tensor([[1.0]], dtype=torch.float32)
        graphs.append(g)
    train_loader = DataLoader(graphs[:1], batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(graphs[1:], batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, graphs
