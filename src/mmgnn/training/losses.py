"""
Losses for training the MMPNN model.
"""

from typing import Optional, Sequence, Union

import torch
import torch.nn as nn
from torch.nn import functional as F


class QuantileLoss(nn.Module):
    """Pinball/quantile loss with fixed quantiles configured at init."""

    def __init__(
        self,
        quantiles: Union[float, Sequence[float], torch.Tensor],
        reduction: str = "mean",
    ):
        super().__init__()

        q = torch.as_tensor(quantiles, dtype=torch.float32)
        if q.ndim == 0:
            q = q.unsqueeze(0)
        if torch.any(q < 0) or torch.any(q > 1):
            raise ValueError("Quantiles should be between 0 and 1")
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction must be one of: 'mean', 'sum', 'none'")

        self.register_buffer("quantiles", q)
        self.reduction = reduction

    def forward(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        sample_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if targets.ndim == 1:
            targets = targets.unsqueeze(-1)

        # Broadcast targets over quantile dimension when needed.
        if targets.shape[-1] == 1 and preds.shape[-1] > 1:
            targets = targets.expand_as(preds)

        if preds.shape != targets.shape:
            raise ValueError(
                f"QuantileLoss expects matching preds/targets shapes after broadcast, got {preds.shape} vs {targets.shape}"
            )

        q = self.quantiles.to(device=preds.device, dtype=preds.dtype)
        if q.numel() == 1 and preds.shape[-1] > 1:
            q = q.expand(preds.shape[-1])
        if q.numel() != preds.shape[-1]:
            raise ValueError(
                f"Number of quantiles ({q.numel()}) must match output dimension ({preds.shape[-1]})"
            )

        errors = targets - preds
        loss = torch.maximum(q * errors, (q - 1) * errors)

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class FocalLoss(nn.Module):
    """Focal loss for binary classification with logits input."""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        if alpha < 0 or alpha > 1:
            raise ValueError("alpha must be in [0, 1] for class-specific binary focal loss")
        if gamma < 0:
            raise ValueError("gamma must be non-negative")
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction must be one of: 'mean', 'sum', 'none'")

        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        if logits.ndim == 1:
            logits = logits.unsqueeze(-1)
        if targets.ndim == 1:
            targets = targets.unsqueeze(-1)

        if logits.shape != targets.shape:
            raise ValueError(f"FocalLoss expects preds and targets to have the same shape, got {logits.shape} vs {targets.shape}")

        targets = targets.to(device=logits.device, dtype=logits.dtype)
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = torch.exp(-bce_loss)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        loss = alpha_t * (1 - p_t) ** self.gamma * bce_loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class LossSelector(nn.Module):
    """Factory wrapper for loss functions with optional per-loss kwargs."""

    def __init__(self, loss_name: str, **loss_kwargs):
        super().__init__()

        name = loss_name.lower()
        self.loss_name = name
        if name == "mse":
            self.loss_fn = nn.MSELoss(**loss_kwargs)
        elif name == "mae":
            self.loss_fn = nn.L1Loss(**loss_kwargs)
        elif name == "huber":
            self.loss_fn = nn.SmoothL1Loss(**loss_kwargs)
        elif name in {"cross_entropy", "ce"}:
            self.loss_fn = nn.CrossEntropyLoss(**loss_kwargs)
        elif name in {
            "binary_cross_entropy",
            "binary_cross_entropy_with_logits",
            "binarycrossentropylogits",
            "bce",
        }:
            self.loss_fn = nn.BCEWithLogitsLoss(**loss_kwargs)
        elif name == "focal":
            alpha = loss_kwargs.pop("alpha", 1.0)
            gamma = loss_kwargs.pop("gamma", 2.0)
            reduction = loss_kwargs.pop("reduction", "mean")
            if loss_kwargs:
                raise ValueError(f"Unsupported kwargs for focal loss: {sorted(loss_kwargs.keys())}")
            self.loss_fn = FocalLoss(alpha=alpha, gamma=gamma, reduction=reduction)
        elif name == "quantile":
            quantiles = loss_kwargs.pop("quantiles", loss_kwargs.pop("quantile", None))
            if quantiles is None:
                raise ValueError("quantile loss requires 'quantiles' (or 'quantile') argument")
            reduction = loss_kwargs.pop("reduction", "mean")
            if loss_kwargs:
                raise ValueError(f"Unsupported kwargs for quantile loss: {sorted(loss_kwargs.keys())}")
            self.loss_fn = QuantileLoss(quantiles=quantiles, reduction=reduction)
        else:
            raise ValueError(f"Unsupported loss function: {loss_name}")

    def forward(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        if self.loss_name in {
            "binary_cross_entropy",
            "binary_cross_entropy_with_logits",
            "binarycrossentropylogits",
            "bce",
        }:
            if preds.ndim == 1:
                preds = preds.unsqueeze(-1)
            if targets.ndim == 1:
                targets = targets.unsqueeze(-1)

            if preds.shape != targets.shape:
                raise ValueError(
                    "Binary cross-entropy expects preds and targets to have matching shapes, "
                    f"got {preds.shape} vs {targets.shape}."
                )

            targets = targets.to(device=preds.device, dtype=preds.dtype)
            return self.loss_fn(preds, targets)

        if self.loss_name == "focal":
            if preds.ndim == 1:
                preds = preds.unsqueeze(-1)
            if targets.ndim == 1:
                targets = targets.unsqueeze(-1)

            if preds.shape != targets.shape:
                raise ValueError(
                    "Focal loss expects preds and targets to have matching shapes, "
                    f"got {preds.shape} vs {targets.shape}."
                )

            targets = targets.to(device=preds.device, dtype=preds.dtype)
            return self.loss_fn(preds, targets)

        if self.loss_name in {"cross_entropy", "ce"}:
            if preds.ndim < 2:
                raise ValueError(
                    f"CrossEntropy expects logits with shape [B, C], got {preds.shape}"
                )

            # Soft-label path: targets already provided as per-class probabilities [B, C].
            if targets.ndim == 2 and targets.shape == preds.shape:
                probs = targets.to(device=preds.device, dtype=preds.dtype)
                if torch.any(probs < 0):
                    raise ValueError("CrossEntropy soft targets must be non-negative")

                row_sums = probs.sum(dim=1, keepdim=True)
                if torch.any(row_sums <= 0):
                    raise ValueError("CrossEntropy soft targets must have strictly positive row sums")

                probs = probs / row_sums
                return self.loss_fn(preds, probs)

            # For class-index supervision, target should be [B] (or [B, 1]).
            if targets.ndim == 2 and targets.shape[-1] == 1:
                targets = targets.squeeze(-1)
            elif targets.ndim != 1:
                raise ValueError(
                    "CrossEntropy expects class-index targets with shape [B] or [B, 1], "
                    "or soft-label targets with shape [B, C] matching logits. "
                    f"Got {targets.shape}."
                )

            targets = targets.to(dtype=torch.long)

        return self.loss_fn(preds, targets)
