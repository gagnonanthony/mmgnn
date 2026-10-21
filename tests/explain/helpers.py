"""Shared fixtures for explain-module unit tests."""

import torch
from torch_geometric.data import Data

NUM_NODES = 4
LATENT_DIM = 3
SUBJECT_ID = "sub-001"


class FakeEncoder:
    def __init__(self, hidden_dim: int) -> None:
        self.hidden_dim = hidden_dim


class FakeModel(torch.nn.Module):
    """MSE-to-target recon loss so mean-replacing a varying column raises loss."""

    def __init__(self, target_z: torch.Tensor) -> None:
        super().__init__()
        self.encoder = FakeEncoder(int(target_z.shape[1]))
        self.register_buffer("target_z", target_z.clone().detach())

    def recon_total_loss(
        self,
        z: torch.Tensor,
        pos_edge_index: torch.Tensor | None = None,
        neg_edge_index: torch.Tensor | None = None,
        target_x: torch.Tensor | None = None,
        edge_index_for_attr: torch.Tensor | None = None,
        target_edge_attr: torch.Tensor | None = None,
        edge_presence_weight: float = 1.0,
        node_attr_weight: float = 1.0,
        edge_attr_weight: float = 1.0,
        node_loss_type: str = "mse",
        edge_attr_loss_type: str = "mse",
    ) -> dict[str, torch.Tensor]:
        mse = ((z - self.target_z) ** 2).mean()
        return {
            "total_loss": mse,
            "edge_presence_loss": mse * 0.5,
            "node_attr_loss": mse * 0.3,
            "edge_attr_loss": mse * 0.2,
        }


def make_z(num_nodes: int = NUM_NODES, latent_dim: int = LATENT_DIM) -> torch.Tensor:
    """Fixed embeddings with distinct, known column variances."""
    # Col 0: high variance, col 1: medium, col 2: constant (zero variance)
    rows = [[float(i), float(i) * 0.5, 1.0] for i in range(num_nodes)]
    z = torch.tensor(rows, dtype=torch.float32)
    if latent_dim > LATENT_DIM:
        pad = torch.zeros((num_nodes, latent_dim - LATENT_DIM), dtype=torch.float32)
        z = torch.cat([z, pad], dim=1)
    elif latent_dim < LATENT_DIM:
        z = z[:, :latent_dim]
    return z


def make_graph(
    num_nodes: int = NUM_NODES,
    subject_id: str = SUBJECT_ID,
) -> Data:
    """Minimal undirected cycle graph with node/edge features."""
    src = list(range(num_nodes))
    dst = [(i + 1) % num_nodes for i in range(num_nodes)]
    edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long)
    x = torch.ones((num_nodes, 1), dtype=torch.float32)
    edge_attr = torch.ones((edge_index.shape[1], 2), dtype=torch.float32)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=num_nodes)
    data.subject_id = subject_id
    return data
