"""
Bit of custom modules.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv


ACT_LAYER = {
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "elu": nn.ELU,
    "gelu": nn.GELU,
    "prelu": nn.PReLU,
    "tanh": nn.Tanh,
    "tanhshrink": nn.Tanhshrink,
}


def add_act_layer(name: str) -> nn.Module:
    """Return the matching activation layer module for use inside nn.Sequential."""
    if name not in ACT_LAYER:
        raise ValueError(
            f"Unsupported activation function: {name}. Supported functions are: {list(ACT_LAYER.keys())}"
        )
    return ACT_LAYER[name]()


class GINEConvEdgeMLP(nn.Module):
    """
    GINEConv but the edge attributes are passed through an MLP before being used in the convolution.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 edge_attr_dim: int,
                 activation: str = "leaky_relu",
                 hidden_dim: int = 32):
        super(GINEConvEdgeMLP, self).__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_attr_dim, hidden_dim),
            add_act_layer(activation),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.conv = GINEConv(
            nn.Sequential(
                nn.Linear(in_channels, out_channels),
                add_act_layer(activation),
                nn.Linear(out_channels, out_channels)
            ),
            edge_dim=hidden_dim
        )

    def forward(self,
                x: torch.Tensor,
                edge_index: torch.Tensor,
                edge_attr: torch.Tensor = None):
        if edge_attr is not None:
            edge_attr = self.edge_mlp(edge_attr)
        return self.conv(x, edge_index, edge_attr)
