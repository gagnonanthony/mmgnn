"""
Builds the main architecture.
"""

import torch.nn as nn
from torch_geometric.nn import JumpingKnowledge
from .blocks import GraphConvBlock


class GNN(nn.Module):
    """
    Main GNN architecture for multimodal graph neural networks (MMGNN) for brain connectome analysis.
    """
    def __init__(self,
                 num_node_features: int,
                 num_edge_features: int,
                 hidden_dim: int,
                 num_heads: int = None,
                 num_layers: int = 3,
                 conv_type: str = "gine",
                 act_func: str = "relu",
                 norm_type: str = "batch",
                 dropout: float = 0.2,
                 edge_dropout: float = 0.0,
                 residual: bool = False,
                 jumping_knowledge_mode: str = None):
        super(GNN, self).__init__()
        self.in_channels = num_node_features
        self.edge_dim = num_edge_features
        self._hidden_dim = hidden_dim
        self.jumping_knowledge_mode = jumping_knowledge_mode
        self.num_layers = num_layers

        # Projection layer.
        self.proj = nn.Linear(num_node_features, hidden_dim) \
            if num_node_features != hidden_dim else None

        # Build graph convolutional layers.
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            # If projection is enabled, the first layer receives hidden_dim features.
            in_channels = hidden_dim if (i > 0 or self.proj is not None) else num_node_features
            self.layers.append(
                GraphConvBlock(
                    in_channels=in_channels,
                    out_channels=hidden_dim,
                    conv_type=conv_type,
                    act_type=act_func,
                    norm_type=norm_type,
                    edge_dim=num_edge_features,
                    edge_dropout=edge_dropout,
                    dropout=dropout,
                    num_heads=num_heads,
                    residual=residual
                )
            )

        # Initialize JumpingKnowledge if mode is specified
        if jumping_knowledge_mode is not None:
            if jumping_knowledge_mode != "lstm":
                raise ValueError(f"Only 'lstm' mode is supported for JumpingKnowledge. Got: {jumping_knowledge_mode}")
            self.jk = JumpingKnowledge(mode="lstm", channels=hidden_dim, num_layers=num_layers)
        else:
            self.jk = None

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        if self.proj is not None:
            x = self.proj(x)
        
        if self.jk is not None:
            # Collect outputs from all layers for JumpingKnowledge
            layer_outputs = []
            for layer in self.layers:
                x = layer(x, edge_index, edge_attr if edge_attr is not None else None)
                layer_outputs.append(x)
            # Apply JumpingKnowledge aggregation
            x = self.jk(layer_outputs)
        else:
            # Standard forward: only use final layer output
            for layer in self.layers:
                x = layer(x, edge_index, edge_attr if edge_attr is not None else None)
        
        return x

    @property
    def hidden_dim(self):
        return self._hidden_dim
