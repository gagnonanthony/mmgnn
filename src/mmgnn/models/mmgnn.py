"""
Main model architecture for Multimodal Graph Neural Networks (MMGNN) for Brain Connectome Analysis.
"""

import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Data
import torch.nn.functional as F

from .blocks import add_readout, add_act_layer
from .backbone import GNN


def initialize_model_weights(
    model: nn.Module,
    mode: str,
    nonlinearity: str = "relu",
) -> None:
    """Initialize model parameters with a supported strategy.

    Supported modes:
        - "xavier": Xavier/Glorot uniform for linear layers
        - "he": Kaiming/He uniform for linear layers
    """
    mode_normalized = mode.lower().strip()
    if mode_normalized not in {"xavier", "he"}:
        raise ValueError("init_mode must be one of: 'xavier', 'he'")

    for module in model.modules():
        if isinstance(module, nn.Linear):
            if mode_normalized == "xavier":
                nn.init.xavier_uniform_(module.weight)
            else:
                nn.init.kaiming_uniform_(module.weight, nonlinearity=nonlinearity)

            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.BatchNorm1d, nn.LayerNorm)):
            if module.weight is not None:
                nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)


def prediction_head(in_dim: int,
                    out_dim: int,
                    hidden_dim: int,
                    num_layers: int,
                    activation: str,
                    dropout: float):
    """
    Builds a simple MLP prediction head for graph-level tasks.
    """
    if num_layers < 1:
        raise ValueError("num_layers must be at least 1")

    layers = []
    layers.extend([
        nn.Linear(in_dim, hidden_dim),
        add_act_layer(activation),
        nn.Dropout(dropout)
    ])
    for _ in range(num_layers - 1):
        layers.extend([
            nn.Linear(hidden_dim, hidden_dim),
            add_act_layer(activation),
            nn.Dropout(dropout)
        ])
    layers.append(nn.Linear(hidden_dim, out_dim))
    return nn.Sequential(*layers)


class MMGNN(nn.Module):
    """
    Main architecture for Multimodal Graph Neural Networks (MMGNN) for Brain Connectome Analysis.
    """
    def __init__(self,
                 backbone: GNN,
                 out_dim: int,
                 hidden_dim: int,
                 readout_type: str,
                 num_layers: int,
                 activation: str,
                 dropout: float):
        super().__init__()
        self.backbone = backbone
        self.hidden_dim = backbone.hidden_dim
        # When using JumpingKnowledge LSTM mode, output dimension is still hidden_dim
        # (LSTM produces a representation of size hidden_dim)
        self.readout = add_readout(readout_type, hidden_dim=self.hidden_dim)
        self.prediction_head = prediction_head(
            in_dim=self.hidden_dim,
            out_dim=out_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            activation=activation,
            dropout=dropout
        )

    def forward(self, data: Data) -> Tensor:
        """Return raw predictions/logits from the prediction head."""
        x = data.x
        edge_index = data.edge_index
        edge_attr = getattr(data, 'edge_attr', None)
        batch = getattr(data, 'batch', None)

        x = self.backbone(x, edge_index, edge_attr, batch=batch)
        x = self.readout(x, batch)
        out = self.prediction_head(x)
        return out

    def forward_embeddings(self, data: Data) -> Tensor:
        """
        Forward pass that returns the graph embeddings before the prediction head.
        """
        x = data.x
        edge_index = data.edge_index
        edge_attr = getattr(data, 'edge_attr', None)
        batch = getattr(data, 'batch', None)

        x = self.backbone(x, edge_index, edge_attr, batch=batch)
        x = self.readout(x, batch)
        pred = self.prediction_head(x)
        return pred, x

    @property
    def nb_params(self) -> int:
        """
        Returns the total number of trainable parameters in the model.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self):
        return f"MMGNN(backbone={self.backbone.__class__.__name__}, params={self.nb_params:,})"
