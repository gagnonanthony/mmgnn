"""
Variational graph autoencoder (VGAE) built on top of the deterministic GAE.

Reuses GAE decoders and reconstruction losses. Adds mu/logvar heads, reparameterized
sampling during training, and a KL term folded into ``recon_total_loss``.
"""

import torch
import torch.nn as nn

from mmgnn.models.gae import GAE, Encoder


class VGAE(GAE):
    """Variational graph autoencoder with the same decoder stack as ``GAE``.

    ``encode`` still returns a single latent tensor:
    - training: reparameterized sample from ``q(z|G)``
    - eval: ``mu`` (deterministic)

    Variational parameters from the latest ``encode`` call are cached so
    ``recon_total_loss`` can add ``kl_weight * KL`` without changing the trainer API.
    """

    def __init__(
        self,
        encoder: Encoder,
        *,
        kl_weight: float = 1.0,
        edge_presence_decoder_type: str = "inner_product",
        edge_attr_out_dim: int | None = None,
        edge_attr_decoder_type: str = "mlp",
        decode_node_features: bool = True,
        node_out_dim: int | None = None,
        decoder_hidden_dim: int | None = None,
        dropout: float = 0.0,
    ):
        super().__init__(
            encoder,
            edge_presence_decoder_type=edge_presence_decoder_type,
            edge_attr_out_dim=edge_attr_out_dim,
            edge_attr_decoder_type=edge_attr_decoder_type,
            decode_node_features=decode_node_features,
            node_out_dim=node_out_dim,
            decoder_hidden_dim=decoder_hidden_dim,
            dropout=dropout,
        )
        latent_dim = encoder.hidden_dim
        self.mu = nn.Linear(latent_dim, latent_dim)
        self.logvar = nn.Linear(latent_dim, latent_dim)
        nn.init.zeros_(self.mu.weight)
        nn.init.constant_(self.logvar.bias, -5.0)
        self.kl_weight = float(kl_weight)
        if self.kl_weight < 0:
            raise ValueError("kl_weight must be >= 0")
        self._last_mu: torch.Tensor | None = None
        self._last_logvar: torch.Tensor | None = None

    def encode_stats(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
        batch: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return variational parameters ``(mu, logvar)`` for each node."""
        h = self.encoder(x, edge_index, edge_attr=edge_attr, batch=batch)
        return self.mu(h), self.logvar(h).clamp(max=2.0)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample ``z ~ N(mu, diag(exp(logvar)))`` via the reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
        batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        mu, logvar = self.encode_stats(x, edge_index, edge_attr=edge_attr, batch=batch)
        self._last_mu = mu
        self._last_logvar = logvar.clamp(max=2.0)
        if self.training:
            return self.reparameterize(mu, logvar)
        return mu

    @staticmethod
    def kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Analytic KL divergence to ``N(0, I)``, averaged over nodes."""
        # 0.5 * sum_d (mu^2 + var - 1 - logvar), then mean over nodes
        logvar = logvar.clamp(max=2.0)
        kl_per_node = -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        return kl_per_node.mean()

    def clear_variational_cache(self) -> None:
        """Drop cached variational parameters from the last ``encode`` call."""
        self._last_mu = None
        self._last_logvar = None

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
        """Reconstruction ELBO: GAE recon losses plus weighted KL."""
        components = super().recon_total_loss(
            z=z,
            pos_edge_index=pos_edge_index,
            neg_edge_index=neg_edge_index,
            target_x=target_x,
            edge_index_for_attr=edge_index_for_attr,
            target_edge_attr=target_edge_attr,
            edge_presence_weight=edge_presence_weight,
            node_attr_weight=node_attr_weight,
            edge_attr_weight=edge_attr_weight,
            node_loss_type=node_loss_type,
            edge_attr_loss_type=edge_attr_loss_type,
        )
        if self._last_mu is not None and self._last_logvar is not None:
            kl = self.kl_loss(self._last_mu, self._last_logvar)
            components["kl_loss"] = kl
            components["total_loss"] = components["total_loss"] + self.kl_weight * kl
        self.clear_variational_cache()
        return components

    def get_architecture_config(self) -> dict[str, object]:
        config = super().get_architecture_config()
        config["model_type"] = "vgae"
        config["kl_weight"] = float(self.kl_weight)
        return config
