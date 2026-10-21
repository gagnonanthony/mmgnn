"""
Main class for graph autoencoder (GAE) architecture to encode multimodel brain connectome data.
"""

import torch
import torch.nn.functional as F
import torch.nn as nn

from .blocks import GraphConvBlock
from torch_geometric.nn import global_mean_pool


class Encoder(nn.Module):
    """
    Ecoder module for graph autoencoder (GAE) architecture.
    Uses GINE layers with batch normalization and ReLU activation.

    Args:
        in_dim: Input feature dimension (number of node features).
        hidden_dim: Hidden dimension for GNN layers.
        num_layers: Number of GNN layers in the encoder.
        dropout: Dropout rate for regularization.
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
                 residual: bool = False):
        super().__init__()
        self.in_channels = num_node_features
        self.edge_dim = num_edge_features
        self._hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Projection layer if needed
        self.proj = nn.Linear(num_node_features, hidden_dim) \
            if num_node_features != hidden_dim else None

        # Build graph convolutional layers.
        self.layers = nn.ModuleList()
        for i in range(num_layers):
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

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        if self.proj is not None:
            x = self.proj(x)
        
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)
        
        return x

    @property
    def hidden_dim(self):
        return self._hidden_dim


class InnerProductEdgeDecoder(nn.Module):
    """Decode edge existence from latent node embeddings with an inner product."""

    def __init__(self, apply_sigmoid: bool = False):
        super().__init__()
        self.apply_sigmoid = apply_sigmoid

    def forward(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]
        logits = (z[src] * z[dst]).sum(dim=-1)
        if self.apply_sigmoid:
            return torch.sigmoid(logits)
        return logits


class EdgeMLPDecoder(nn.Module):
    """Decode edge scores or edge attributes from pairs of latent node embeddings."""

    def __init__(
        self,
        latent_dim: int,
        out_dim: int = 1,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        hidden_dim = hidden_dim or latent_dim
        in_dim = latent_dim * 4  # [z_i, z_j, |z_i-z_j|, z_i*z_j]
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]
        zi = z[src]
        zj = z[dst]
        pair = torch.cat([zi, zj, torch.abs(zi - zj), zi * zj], dim=-1)
        return self.mlp(pair)


class NodeMLPDecoder(nn.Module):
    """Decode latent node embeddings back to node feature space."""

    def __init__(self, latent_dim: int, out_dim: int, hidden_dim: int | None = None, dropout: float = 0.0):
        super().__init__()
        hidden_dim = hidden_dim or latent_dim
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.mlp(z)


class GAE(nn.Module):
    """Graph autoencoder with separate edge-presence and edge-attribute decoders."""

    def __init__(
        self,
        encoder: Encoder,
        edge_presence_decoder_type: str = "inner_product",
        edge_attr_out_dim: int | None = None,
        edge_attr_decoder_type: str = "mlp",
        decode_node_features: bool = True,
        node_out_dim: int | None = None,
        decoder_hidden_dim: int | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.encoder = encoder

        # Auto-infer decoder output shapes from the encoder's input spaces.
        inferred_node_out_dim = encoder.in_channels
        inferred_edge_attr_dim = encoder.edge_dim if encoder.edge_dim is not None and encoder.edge_dim > 0 else None

        if edge_presence_decoder_type == "inner_product":
            self.edge_presence_decoder = InnerProductEdgeDecoder(apply_sigmoid=False)
        elif edge_presence_decoder_type == "mlp":
            self.edge_presence_decoder = EdgeMLPDecoder(
                latent_dim=encoder.hidden_dim,
                out_dim=1,
                hidden_dim=decoder_hidden_dim,
                dropout=dropout,
            )
        else:
            raise ValueError("edge_presence_decoder_type must be one of: 'inner_product', 'mlp'")

        if edge_attr_out_dim is None:
            edge_attr_out_dim = inferred_edge_attr_dim
        self.edge_attr_out_dim = edge_attr_out_dim

        self.edge_attr_decoder = None
        if edge_attr_out_dim is not None:
            if edge_attr_decoder_type != "mlp":
                raise ValueError("edge_attr_decoder_type must be 'mlp' when edge_attr_out_dim is provided")
            self.edge_attr_decoder = EdgeMLPDecoder(
                latent_dim=encoder.hidden_dim,
                out_dim=edge_attr_out_dim,
                hidden_dim=decoder_hidden_dim,
                dropout=dropout,
            )

        self.node_decoder = None
        self.node_out_dim = None
        if decode_node_features:
            node_out_dim = inferred_node_out_dim if node_out_dim is None else node_out_dim
            self.node_out_dim = node_out_dim
            self.node_decoder = NodeMLPDecoder(
                latent_dim=encoder.hidden_dim,
                out_dim=node_out_dim,
                hidden_dim=decoder_hidden_dim,
                dropout=dropout,
            )

    def encode(self, x, edge_index, edge_attr=None, batch=None) -> torch.Tensor:
        return self.encoder(x, edge_index, edge_attr=edge_attr, batch=batch)

    def decode_edge_presence(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.edge_presence_decoder(z, edge_index)

    def decode_edge_attrs(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor | None:
        if self.edge_attr_decoder is None:
            return None
        return self.edge_attr_decoder(z, edge_index)

    def decode_edges(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor | None:
        """Backward-compatible alias for edge-attribute decoding."""
        return self.decode_edge_attrs(z, edge_index)

    def decode_nodes(self, z: torch.Tensor) -> torch.Tensor | None:
        if self.node_decoder is None:
            return None
        return self.node_decoder(z)

    @staticmethod
    def _reconstruction_loss(
        preds: torch.Tensor,
        targets: torch.Tensor,
        loss_type: str = "mse",
    ) -> torch.Tensor:
        loss_name = loss_type.lower()
        if loss_name == "mse":
            return F.mse_loss(preds, targets)
        if loss_name in {"mae", "l1"}:
            return F.l1_loss(preds, targets)
        if loss_name in {"huber", "smooth_l1"}:
            return F.smooth_l1_loss(preds, targets)
        raise ValueError("loss_type must be one of: 'mse', 'mae'/'l1', 'huber'/'smooth_l1'")

    def recon_edge_loss(
        self,
        z: torch.Tensor,
        pos_edge_index: torch.Tensor,
        neg_edge_index: torch.Tensor,
    ) -> torch.Tensor:
        pos_logits_raw = self.decode_edge_presence(z, pos_edge_index)
        neg_logits_raw = self.decode_edge_presence(z, neg_edge_index)

        if (pos_logits_raw.ndim > 1 and pos_logits_raw.shape[-1] != 1) or (
            neg_logits_raw.ndim > 1 and neg_logits_raw.shape[-1] != 1
        ):
            raise ValueError(
                "recon_edge_loss expects scalar edge logits per edge. "
                "Use an edge-presence decoder with scalar outputs."
            )

        pos_logits = pos_logits_raw.view(-1)
        neg_logits = neg_logits_raw.view(-1)

        pos_target = torch.ones_like(pos_logits)
        neg_target = torch.zeros_like(neg_logits)

        loss_pos = F.binary_cross_entropy_with_logits(pos_logits, pos_target)
        loss_neg = F.binary_cross_entropy_with_logits(neg_logits, neg_target)
        return loss_pos + loss_neg

    def recon_node_loss(
        self,
        z: torch.Tensor,
        target_x: torch.Tensor,
        loss_type: str = "mse",
    ) -> torch.Tensor:
        x_hat = self.decode_nodes(z)
        if x_hat is None:
            raise ValueError("Node decoder is disabled. Set decode_node_features=True to use recon_node_loss.")

        if x_hat.shape != target_x.shape:
            raise ValueError(
                f"Node reconstruction shape mismatch: predicted {x_hat.shape} vs target {target_x.shape}."
            )

        target_x = target_x.to(device=x_hat.device, dtype=x_hat.dtype)
        return self._reconstruction_loss(x_hat, target_x, loss_type=loss_type)

    def recon_edge_attr_loss(
        self,
        z: torch.Tensor,
        edge_index: torch.Tensor,
        target_edge_attr: torch.Tensor,
        loss_type: str = "mse",
    ) -> torch.Tensor:
        edge_attr_hat = self.decode_edge_attrs(z, edge_index)
        if edge_attr_hat is None:
            raise ValueError("Edge attribute decoder is disabled. Provide edge_attr_out_dim to enable it.")

        if edge_attr_hat.ndim == 1:
            edge_attr_hat = edge_attr_hat.unsqueeze(-1)
        if target_edge_attr.ndim == 1:
            target_edge_attr = target_edge_attr.unsqueeze(-1)

        if edge_attr_hat.shape != target_edge_attr.shape:
            raise ValueError(
                "Edge attribute reconstruction shape mismatch: "
                f"predicted {edge_attr_hat.shape} vs target {target_edge_attr.shape}. "
                "For scalar edge-presence decoding, use recon_edge_loss instead."
            )

        target_edge_attr = target_edge_attr.to(device=edge_attr_hat.device, dtype=edge_attr_hat.dtype)
        return self._reconstruction_loss(edge_attr_hat, target_edge_attr, loss_type=loss_type)

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
        """Compute a weighted sum of available reconstruction losses.

        Returns:
            Dict with total_loss and per-component losses for easy logging.
        """
        if edge_presence_weight < 0 or node_attr_weight < 0 or edge_attr_weight < 0:
            raise ValueError("All reconstruction weights must be >= 0")

        components: dict[str, torch.Tensor] = {}
        total_loss: torch.Tensor | None = None

        if edge_presence_weight > 0:
            if pos_edge_index is None or neg_edge_index is None:
                raise ValueError(
                    "pos_edge_index and neg_edge_index are required when edge_presence_weight > 0"
                )
            edge_presence_loss = self.recon_edge_loss(z, pos_edge_index, neg_edge_index)
            components["edge_presence_loss"] = edge_presence_loss
            weighted = edge_presence_weight * edge_presence_loss
            total_loss = weighted if total_loss is None else total_loss + weighted

        if node_attr_weight > 0 and target_x is not None:
            node_attr_loss = self.recon_node_loss(z, target_x, loss_type=node_loss_type)
            components["node_attr_loss"] = node_attr_loss
            weighted = node_attr_weight * node_attr_loss
            total_loss = weighted if total_loss is None else total_loss + weighted

        if edge_attr_weight > 0 and target_edge_attr is not None:
            if edge_index_for_attr is None:
                raise ValueError(
                    "edge_index_for_attr is required when edge_attr_weight > 0 and target_edge_attr is provided"
                )
            edge_attr_loss = self.recon_edge_attr_loss(
                z,
                edge_index_for_attr,
                target_edge_attr,
                loss_type=edge_attr_loss_type,
            )
            components["edge_attr_loss"] = edge_attr_loss
            weighted = edge_attr_weight * edge_attr_loss
            total_loss = weighted if total_loss is None else total_loss + weighted

        if total_loss is None:
            raise ValueError(
                "No reconstruction component was enabled. Provide at least one valid target/index pair with non-zero weight."
            )

        components["total_loss"] = total_loss
        return components

    def forward(self, data) -> dict[str, torch.Tensor | None]:
        x = data.x
        edge_index = data.edge_index
        edge_attr = getattr(data, "edge_attr", None)
        batch = getattr(data, "batch", None)

        z = self.encode(x, edge_index, edge_attr=edge_attr, batch=batch)
        edge_presence_logits = self.decode_edge_presence(z, edge_index)
        edge_attr_logits = self.decode_edge_attrs(z, edge_index)
        x_hat = self.decode_nodes(z)
        return {
            "z": z,
            "edge_presence_logits": edge_presence_logits,
            "edge_attr_logits": edge_attr_logits,
            "x_hat": x_hat,
        }

    def get_embedding(self, data) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            z = self.encode(data.x,
                            data.edge_index,
                            edge_attr=getattr(data, "edge_attr", None),
                            batch=getattr(data, "batch", None))
        return z

    def forward_embeddings(self, data):
        """Return (None, graph_level_embeddings) to match MMGNN.forward_embeddings signature.

        Graph-level embedding is computed by mean-pooling node latents per graph when `data.batch` is present.
        """
        x = data.x
        edge_index = data.edge_index
        edge_attr = getattr(data, "edge_attr", None)
        batch = getattr(data, "batch", None)

        z = self.encode(x, edge_index, edge_attr=edge_attr, batch=batch)
        if batch is None:
            # If no batch provided, treat whole graph as single sample: pool to a single vector
            graph_emb = z.mean(dim=0, keepdim=True)
        else:
            graph_emb = global_mean_pool(z, batch)

        return None, graph_emb

    def get_architecture_config(self) -> dict[str, object]:
        """Return the decoder/encoder configuration needed to rebuild this model."""
        return {
            "model_type": "gae",
            "edge_presence_decoder_type": (
                "mlp" if isinstance(self.edge_presence_decoder, EdgeMLPDecoder) else "inner_product"
            ),
            "edge_attr_out_dim": self.edge_attr_out_dim,
            "edge_attr_decoder_type": "mlp" if self.edge_attr_decoder is not None else None,
            "decode_node_features": self.node_decoder is not None,
            "node_out_dim": self.node_out_dim,
        }
