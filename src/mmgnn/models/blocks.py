"""
Model building blocks for Multimodal Graph Neural Networks (MMGNN) for Brain Connectome Analysis.

Each block includes:
    1. Convolutional layers for processing graph data.
    2. Normalization layers to stabilize training.
    3. Activation functions to introduce non-linearity.
    4. Dropout for regularization. (optional)
    5. Residual connections to facilitate gradient flow and improve performance. (optional)
"""

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv,
    GINConv,
    GINEConv,
    GATConv,
    BatchNorm,
    LayerNorm,
    GraphNorm,
    PairNorm,
    DiffGroupNorm,
    global_mean_pool,
    global_max_pool,
    global_add_pool
)
from torch_geometric.nn.aggr import AttentionalAggregation
from torch_geometric.utils import dropout_edge

from .modules import GINEConvEdgeMLP


ACT_FUNC = {
    "relu": F.relu,
    "leaky_relu": F.leaky_relu,
    "elu": F.elu,
    "gelu": F.gelu,
    "prelu": F.prelu,
    "tanh": F.tanh,
    "tanhshrink": F.tanhshrink,
}

ACT_LAYER = {
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "elu": nn.ELU,
    "gelu": nn.GELU,
    "prelu": nn.PReLU,
    "tanh": nn.Tanh,
    "tanhshrink": nn.Tanhshrink,
}

def add_act_func(name: str):
    """
    Return the matching activation function from the ACT_FUNC dictionary.

    Args:
        name (str): The name of the activation function to retrieve.

    Returns:
        function: The activation function corresponding to the provided name.
    """
    if name not in ACT_FUNC:
        raise ValueError(f"Unsupported activation function: {name}. Supported functions are: {list(ACT_FUNC.keys())}")
    return ACT_FUNC[name]

def add_act_layer(name: str) -> nn.Module:
    """
    Return the matching activation layer module for use inside nn.Sequential.

    Args:
        name (str): Activation layer name.

    Returns:
        nn.Module: Instantiated activation layer.
    """
    if name not in ACT_LAYER:
        raise ValueError(f"Unsupported activation function: {name}. Supported functions are: {list(ACT_LAYER.keys())}")
    return ACT_LAYER[name]()

def add_norm_layer(name: str,
                   num_features: int):
    """
    Return the matching normalization layer based on the provided name.

    Args:
        name (str): The name of the normalization layer to retrieve.
        num_features (int): The number of features for the normalization layer.

    Returns:
        nn.Module: The normalization layer corresponding to the provided name.
    """
    if name == "batch":
        return BatchNorm(num_features)
    elif name == "layer":
        return LayerNorm(num_features)
    elif name == "graph":
        return GraphNorm(num_features)
    elif name == "pair":
        return PairNorm(num_features)
    elif name == "diff_group":
        return DiffGroupNorm(num_features, groups=10)
    else:
        raise ValueError(f"Unsupported normalization type: {name}. Supported types are: 'batch', 'layer', 'graph'.")

def add_conv_layer(name: str,
                   in_channels: int,
                   out_channels: int,
                   edge_dim: int = None,
                   activation: str = "leaky_relu",
                   num_heads: int = None):
    """
    Return the matching convolutional layer based on the provided name.

    Args:
        name (str): The name of the convolutional layer to retrieve.
        in_channels (int): The number of input channels for the convolutional layer.
        out_channels (int): The number of output channels for the convolutional layer.
        edge_dim (int, optional): The dimensionality of edge features (required for GINEConv). Defaults to None.
        num_heads (int, optional): The number of attention heads (required for GATConv). Defaults to None.

    Returns:
        nn.Module: The convolutional layer corresponding to the provided name.
    """
    if name == "gcn":
        return GCNConv(in_channels, out_channels)
    elif name == "gin":
        return GINConv(nn.Sequential(
            nn.Linear(in_channels, out_channels),
            add_act_layer(activation),
            nn.Linear(out_channels, out_channels)
        ))
    elif name == "gine":
        return GINEConv(nn.Sequential(
            nn.Linear(in_channels, out_channels),
            add_act_layer(activation),
            nn.Linear(out_channels, out_channels)
        ), edge_dim=edge_dim if edge_dim is not None and edge_dim > 0 else None)
    elif name == "gine_edge_mlp":
        if edge_dim is None or edge_dim <= 0:
            raise ValueError("gine_edge_mlp requires a positive edge_dim")
        hidden_dim = max(out_channels // 2, 16)
        return GINEConvEdgeMLP(in_channels, out_channels, edge_attr_dim=edge_dim, hidden_dim=hidden_dim, activation=activation)
    elif name == "gat":
        return GATConv(in_channels, out_channels // num_heads, heads=num_heads,
                       concat=True,
                       edge_dim=edge_dim if edge_dim is not None and edge_dim > 0 else None)
    else:
        raise ValueError(f"Unsupported convolution type: {name}. Supported types are: 'gcn', 'gin', 'gine', 'gat', 'gine_edge_mlp'.")

class GraphConvBlock(nn.Module):
    """
    Single block of graph convolutional layers, normalization, activation, and optional dropout and residual connection.
    
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        conv_type (str): Type of convolutional layer to use ('gcn', 'gin', 'gine', 'gat', 'gine_edge_mlp').
        norm_type (str): Type of normalization layer to use ('batch', 'layer', 'graph').
        act_type (str): Type of activation function to use ('relu', 'leaky_relu', 'elu', 'gelu').
        dropout (float, optional): Dropout rate. Defaults to 0.0.
        residual (bool, optional): Whether to include a residual connection. Defaults to False.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 conv_type: str,
                 norm_type: str = "batch",
                 act_type: str = "relu",
                 edge_dim: int = None,
                 num_heads: int = None,
                 dropout: float = 0.0,
                 edge_dropout: float = 0.0,
                 residual: bool = False):
        super(GraphConvBlock, self).__init__()
        self.conv = add_conv_layer(conv_type, in_channels, out_channels, edge_dim=edge_dim, num_heads=num_heads,
                                   activation=act_type)
        self.norm = add_norm_layer(norm_type, out_channels)
        self.act = add_act_layer(act_type)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.edge_dropout = edge_dropout
        self.residual = residual

    def forward(self, x, edge_index, edge_attr=None):
        if self.edge_dropout > 0:
            edge_index, edge_mask = dropout_edge(edge_index, p=self.edge_dropout, training=self.training)
            if edge_attr is not None:
                edge_attr = edge_attr[edge_mask]
        out = self.conv(x, edge_index, edge_attr) if edge_attr is not None else self.conv(x, edge_index)
        out = self.norm(out)
        out = self.act(out)
        if self.dropout is not None:
            out = self.dropout(out)
        if self.residual:
            out = out + x
        return out

def add_readout(name: str, hidden_dim: int = 64):
    """
    Return the matching readout function based on the provided name.

    Args:
        name (str): The name of the readout function to retrieve.
        hidden_dim (int): The hidden dimension for the attention mechanism.

    Returns:
        function: The readout function corresponding to the provided name.
    """
    if name == "mean":
        return global_mean_pool
    elif name == "max":
        return global_max_pool
    elif name == "add":
        return global_add_pool
    elif name == "attention":
        gate_nn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        return AttentionalAggregation(gate_nn=gate_nn)
    else:
        raise ValueError(f"Unsupported readout type: {name}. Supported types are: 'mean', 'max', 'add', 'attention'.")
