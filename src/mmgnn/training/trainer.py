"""
Training utilities for the MMGNN framework.
"""

from pathlib import Path
from typing import Dict, Optional
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from scipy import stats


class EarlyStopping:
    """
    Early stopping utility to halt training when validation performance does not improve for a specified number of epochs.
    """
    def __init__(self, patience: int = 10, delta: float = 0.0, mode: str = "max"):
        self.patience = patience
        self.delta = delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.should_stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
        if (self.mode == "max" and score > self.best_score + self.delta) or \
           (self.mode == "min" and score < self.best_score - self.delta):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


class Trainer:
    """
    Trainer class to handle the training loop, validation, and early stopping for the MMGNN model.
    """
    def __init__(self,
                 model: nn.Module,
                 task_type: str,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 optimizer: Adam,
                 criterion: nn.Module,
                 scheduler: str = None,
                 patience: int = 20,
                 delta: float = 0.0,
                 learning_rate: float = 1e-3,
                 weight_decay: float = 1e-5,
                 optimizer_param_groups: Optional[list[dict]] = None,
                 checkpoint_dir: Path = Path("./checkpoints"),
                 device: torch.device = None):

        self.task_type = task_type.lower()
        if self.task_type not in {"regression", "classification"}:
            raise ValueError("task_type must be either 'regression' or 'classification'")

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.device = device
        self.checkpoint_dir = checkpoint_dir

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.model = model.to(self.device)

        # Set optimizer and loss function.
        if optimizer_param_groups is not None:
            self.optimizer = optimizer(optimizer_param_groups, lr=learning_rate, weight_decay=weight_decay)
        else:
            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            if len(trainable_params) == 0:
                raise ValueError("No trainable parameters found for optimizer initialization")
            self.optimizer = optimizer(trainable_params, lr=learning_rate, weight_decay=weight_decay)
        self.loss = criterion.to(self.device)
        self.early_stopping = EarlyStopping(patience=patience, delta=delta, mode="min")

        # Setup scheduler if specified
        if scheduler is not None:
            if scheduler == "reduce_on_plateau":
                self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", factor=0.5, patience=10)
            elif scheduler == "step":
                self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.5)
            else:
                raise ValueError(f"Unsupported scheduler type: {scheduler}. Supported types are: 'reduce_on_plateau', 'step'.")
        else:
            self.scheduler = None

        # Checkpoints
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # History
        self.history: Dict[str, list] = {
            "train_loss": [],
            "train_mae": [],
            "train_pearson": [],
            "train_accuracy": [],
            "train_f1": [],
            "val_loss": [],
            "val_mae": [],
            "val_pearson": [],
            "val_accuracy": [],
            "val_f1": [],
            "lr": []
        }
        self.best_val_loss = float("inf")
        self.best_epoch = 0

    @staticmethod
    def _safe_pearson(preds: np.ndarray, targets: np.ndarray) -> float:
        """Compute Pearson correlation safely for flattened arrays."""
        if preds.size < 2 or targets.size < 2:
            return float("nan")
        if np.allclose(preds, preds.flat[0]) or np.allclose(targets, targets.flat[0]):
            return float("nan")
        corr, _ = stats.pearsonr(preds.flatten(), targets.flatten())
        return float(corr)

    @staticmethod
    def _extract_class_indices(targets: torch.Tensor) -> torch.Tensor:
        """Normalize class targets to shape [B] for classification metrics/losses."""
        if targets.ndim == 1:
            return targets.to(dtype=torch.long)
        if targets.ndim == 2 and targets.shape[-1] == 1:
            return targets.squeeze(-1).to(dtype=torch.long)
        if targets.ndim == 2 and targets.shape[-1] > 1:
            # Accept one-hot or membership vectors by reducing to hard class labels.
            return torch.argmax(targets, dim=-1).to(dtype=torch.long)
        raise ValueError(
            f"Unsupported classification target shape {tuple(targets.shape)}; expected [B], [B,1], or [B,C]."
        )

    @staticmethod
    def _classification_metrics(logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """Compute accuracy and macro-F1 from logits and class indices."""
        if logits.ndim == 1:
            logits = logits.unsqueeze(-1)
        if logits.ndim != 2:
            raise ValueError(f"Expected logits shape [B, C] or [B], got {tuple(logits.shape)}")

        y_true = Trainer._extract_class_indices(targets)

        if logits.shape[1] == 1:
            probs = torch.sigmoid(logits.squeeze(-1))
            y_pred = (probs >= 0.5).to(dtype=torch.long)
            num_classes = 2
        else:
            y_pred = torch.argmax(logits, dim=1)
            num_classes = int(logits.shape[1])

        if y_pred.numel() == 0:
            return {"accuracy": float("nan"), "f1": float("nan")}

        accuracy = (y_pred == y_true).float().mean().item()

        conf = torch.bincount(
            y_true * num_classes + y_pred,
            minlength=num_classes * num_classes,
        ).reshape(num_classes, num_classes).to(dtype=torch.float32)

        tp = conf.diag()
        fp = conf.sum(dim=0) - tp
        fn = conf.sum(dim=1) - tp
        denom = 2 * tp + fp + fn
        f1_per_class = torch.where(denom > 0, (2 * tp) / denom, torch.zeros_like(denom))
        macro_f1 = f1_per_class.mean().item()

        return {"accuracy": float(accuracy), "f1": float(macro_f1)}

    def _run_epoch(self, loader: DataLoader, training: bool = True) -> Dict[str, float]:
        """Runs a single epoch of training or validation."""
        self.model.train() if training else self.model.eval()
        total_loss, total_samples = 0.0, 0
        all_preds, all_targets = [], []

        with torch.set_grad_enabled(training):
            for batch in loader:
                batch = batch.to(self.device)
                out = self.model(batch)
                loss = self.loss(out, batch.y)

                if training:
                    self.optimizer.zero_grad()
                    loss.backward()
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                    self.optimizer.step()

                total_loss += loss.item() * batch.num_graphs
                total_samples += batch.num_graphs

                all_preds.append(out.cpu())
                all_targets.append(batch.y.cpu())

        average_loss = total_loss / max(1, total_samples)
        preds = torch.cat(all_preds).detach()
        targets = torch.cat(all_targets).detach()

        if self.task_type == "classification":
            cls_metrics = self._classification_metrics(preds, targets)
            return {
                "lr": self.optimizer.param_groups[0]['lr'],
                "loss": average_loss,
                "mae": float("nan"),
                "pearson": float("nan"),
                "accuracy": cls_metrics["accuracy"],
                "f1": cls_metrics["f1"],
            }

        preds_np = preds.numpy()
        targets_np = targets.numpy()
        average_mae = float(np.mean(np.abs(preds_np - targets_np)))
        pearson_corr = self._safe_pearson(preds_np, targets_np)
        return {
            "lr": self.optimizer.param_groups[0]['lr'],
            "loss": average_loss,
            "mae": average_mae,
            "pearson": pearson_corr,
            "accuracy": float("nan"),
            "f1": float("nan"),
        }

    def _run_validation(self, loader: DataLoader) -> Dict[str, float]:
        """Runs validation and computes Pearson correlation."""
        return self._run_epoch(loader, training=False)

    def train(self, num_epochs: int = 100, verbose: bool = True):
        """Main training loop."""
        for epoch in range(1, num_epochs + 1):
            start_time = time.time()
            train_metrics = self._run_epoch(self.train_loader, training=True)
            val_metrics = self._run_validation(self.val_loader)

            # Update scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics["loss"])
                else:
                    self.scheduler.step()

            # Update history
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["train_mae"].append(train_metrics["mae"])
            self.history["train_pearson"].append(train_metrics["pearson"])
            self.history["train_accuracy"].append(train_metrics["accuracy"])
            self.history["train_f1"].append(train_metrics["f1"])
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_mae"].append(val_metrics["mae"])
            self.history["val_pearson"].append(val_metrics["pearson"])
            self.history["val_accuracy"].append(val_metrics["accuracy"])
            self.history["val_f1"].append(val_metrics["f1"])
            self.history["lr"].append(self.optimizer.param_groups[0]['lr'])

            # Check for improvement
            if val_metrics["loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["loss"]
                self.best_epoch = epoch
                self.save_checkpoint(epoch)

            # Early stopping check
            if self.early_stopping(val_metrics["loss"]):
                if verbose:
                    print(f"Early stopping at epoch {epoch}. Best validation loss: {self.best_val_loss:.4f} at epoch {self.best_epoch}.")
                break

            if verbose:
                elapsed = time.time() - start_time
                if self.task_type == "classification":
                    print(
                        f"Epoch {epoch}/{num_epochs} - "
                        f"LR: {train_metrics['lr']:.2e} - "
                        f"Train Loss: {train_metrics['loss']:.4f}, "
                        f"Train Acc: {train_metrics['accuracy']:.4f}, "
                        f"Train F1: {train_metrics['f1']:.4f} - "
                        f"Val Loss: {val_metrics['loss']:.4f}, "
                        f"Val Acc: {val_metrics['accuracy']:.4f}, "
                        f"Val F1: {val_metrics['f1']:.4f} - "
                        f"Time: {elapsed:.2f}s"
                    )
                else:
                    print(
                        f"Epoch {epoch}/{num_epochs} - "
                        f"LR: {train_metrics['lr']:.2e} - "
                        f"Train Loss: {train_metrics['loss']:.4f}, "
                        f"Train MAE: {train_metrics['mae']:.4f}, "
                        f"Train Pearson: {train_metrics['pearson']:.4f} - "
                        f"Val Loss: {val_metrics['loss']:.4f}, "
                        f"Val MAE: {val_metrics['mae']:.4f}, "
                        f"Val Pearson: {val_metrics['pearson']:.4f} - "
                        f"Time: {elapsed:.2f}s"
                    )

    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        """Evaluates the model on a given data loader."""
        self.model.eval()
        all_preds, all_targets = [], []

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device, non_blocking=True)
                out = self.model(batch)
                all_preds.append(out.cpu())
                all_targets.append(batch.y.cpu())

        preds_t = torch.cat(all_preds).detach()
        targets_t = torch.cat(all_targets).detach()

        if self.task_type == "classification":
            metrics = self._classification_metrics(preds_t, targets_t)
            true_classes = self._extract_class_indices(targets_t)

            if preds_t.ndim == 1:
                preds_t = preds_t.unsqueeze(-1)

            if preds_t.shape[1] == 1:
                positive_prob = torch.sigmoid(preds_t.squeeze(-1))
                pred_classes = (positive_prob >= 0.5).to(dtype=torch.long)
                probabilities = torch.stack([1.0 - positive_prob, positive_prob], dim=1)
                target_memberships = torch.nn.functional.one_hot(true_classes, num_classes=2).to(dtype=torch.float32)
            else:
                pred_classes = torch.argmax(preds_t, dim=1)
                probabilities = torch.softmax(preds_t, dim=1)
                target_memberships = None
                if targets_t.ndim == 2 and targets_t.shape[-1] > 1:
                    target_memberships = targets_t.to(dtype=torch.float32)

            return {
                "accuracy": metrics["accuracy"],
                "f1": metrics["f1"],
                "logits": preds_t.numpy(),
                "probabilities": probabilities.numpy(),
                "predictions": pred_classes.numpy(),
                "targets": true_classes.numpy(),
                "target_memberships": target_memberships,
            }

        preds = preds_t.numpy()
        targets = targets_t.numpy()
        mae = float(np.mean(np.abs(preds - targets)))
        mse = float(np.mean((preds - targets) ** 2))
        pearson_corr = self._safe_pearson(preds, targets)

        return {
            "mae": mae,
            "mse": mse,
            "pearson": pearson_corr,
            "predictions": preds,
            "targets": targets,
        }

    def get_embeddings(self, loader: DataLoader) -> np.ndarray:
        """Extracts graph embeddings from the model for a given data loader."""
        self.model.eval()
        all_embeddings = []
        all_targets = []

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device, non_blocking=True)
                _, embeddings = self.model.forward_embeddings(batch)
                all_embeddings.append(embeddings.cpu())
                all_targets.append(batch.y.cpu())

        return torch.cat(all_embeddings).numpy(), torch.cat(all_targets).numpy()

    def get_contingency_matrix(self, loader: DataLoader) -> np.ndarray:
        """Computes the contingency matrix of predicted vs true classes for a classification task."""
        if self.task_type != "classification":
            raise ValueError("Contingency matrix can only be computed for classification tasks.")

        eval_results = self.evaluate(loader)
        pred_classes = eval_results["predictions"]
        true_classes = eval_results["targets"]

        num_classes = max(pred_classes.max(), true_classes.max()) + 1
        contingency_matrix = np.zeros((num_classes, num_classes), dtype=int)

        for t, p in zip(true_classes, pred_classes):
            contingency_matrix[t, p] += 1

        return contingency_matrix

    def save_checkpoint(self, epoch: int):
        """Saves a checkpoint of the current model state."""
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": epoch,
            "best_val_loss": self.best_val_loss,
            "history": self.history
        }, checkpoint_path)

    def load_checkpoint(self, checkpoint_path: Path):
        """Loads a checkpoint and restores the model and optimizer state."""
        # Local checkpoints include optimizer/history metadata, so disable weights_only
        # for compatibility with PyTorch 2.6+ default behavior.
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.history = checkpoint.get("history", self.history)
        self.best_val_loss = checkpoint.get("best_val_loss", self.best_val_loss)
        if "epoch" in checkpoint:
            print(f"Loaded checkpoint from epoch {checkpoint['epoch']} with best validation loss {self.best_val_loss:.4f}.")


def set_seed(seed: int):
    """Utility function to set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False