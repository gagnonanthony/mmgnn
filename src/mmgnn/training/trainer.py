"""
Training utilities for GAE reconstruction training.
"""

from pathlib import Path
import logging
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import negative_sampling


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
        elif (self.mode == "max" and score > self.best_score + self.delta) or \
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
    Trainer class for GAE reconstruction training, validation, and checkpointing.
    """
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 optimizer: Adam,
                 scheduler: str = None,
                 patience: int = 20,
                 delta: float = 0.0,
                 learning_rate: float = 1e-3,
                 weight_decay: float = 1e-5,
                 optimizer_param_groups: list[dict] | None = None,
                 checkpoint_dir: Path = Path("./checkpoints"),
                 device: torch.device = None,
                 neg_sampling_ratio: float = 1.0,
                 recon_edge_presence_weight: float = 1.0,
                 recon_node_attr_weight: float = 1.0,
                 recon_edge_attr_weight: float = 1.0,
                 recon_node_loss_type: str = "mse",
                 recon_edge_attr_loss_type: str = "mse"):

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.neg_sampling_ratio = neg_sampling_ratio

        self.recon_edge_presence_weight = float(recon_edge_presence_weight)
        self.recon_node_attr_weight = float(recon_node_attr_weight)
        self.recon_edge_attr_weight = float(recon_edge_attr_weight)
        self.recon_node_loss_type = recon_node_loss_type
        self.recon_edge_attr_loss_type = recon_edge_attr_loss_type

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
        self.history: dict[str, list] = {
            "train_loss": [],
            "train_edge_presence_loss": [],
            "train_node_attr_loss": [],
            "train_edge_attr_loss": [],
            "train_kl_loss": [],
            "val_loss": [],
            "val_edge_presence_loss": [],
            "val_node_attr_loss": [],
            "val_edge_attr_loss": [],
            "val_kl_loss": [],
            "lr": [],
        }
        self.best_val_loss = float("inf")
        self.best_epoch = 0

    def _run_epoch(self, loader: DataLoader, training: bool = True) -> dict[str, float]:
        """Runs a single epoch of GAE reconstruction training or validation."""
        self.model.train() if training else self.model.eval()
        total_loss, total_samples = 0.0, 0
        total_edge_presence_loss = 0.0
        total_node_attr_loss = 0.0
        total_edge_attr_loss = 0.0
        total_kl_loss = 0.0

        with torch.set_grad_enabled(training):
            for batch in loader:
                batch = batch.to(self.device)

                if hasattr(self.model, "encode"):
                    z = self.model.encode(
                        batch.x,
                        batch.edge_index,
                        edge_attr=getattr(batch, "edge_attr", None),
                        batch=getattr(batch, "batch", None),
                    )
                else:
                    out = self.model(batch)
                    z = out.get("z") if isinstance(out, dict) else None

                if z is None:
                    raise RuntimeError("Model does not expose an encoder 'encode' method or return 'z' in forward().")

                pos_ei = batch.edge_index
                num_pos = pos_ei.size(1)
                num_neg = int(num_pos * max(1.0, self.neg_sampling_ratio))
                neg_ei = negative_sampling(edge_index=pos_ei, num_nodes=batch.num_nodes, num_neg_samples=num_neg)

                components = self.model.recon_total_loss(
                    z=z,
                    pos_edge_index=pos_ei,
                    neg_edge_index=neg_ei,
                    target_x=getattr(batch, "x", None),
                    edge_index_for_attr=pos_ei,
                    target_edge_attr=getattr(batch, "edge_attr", None),
                    edge_presence_weight=self.recon_edge_presence_weight,
                    node_attr_weight=self.recon_node_attr_weight,
                    edge_attr_weight=self.recon_edge_attr_weight,
                    node_loss_type=self.recon_node_loss_type,
                    edge_attr_loss_type=self.recon_edge_attr_loss_type,
                )

                batch_total_loss = components["total_loss"]
                if training:
                    self.optimizer.zero_grad()
                    batch_total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                    self.optimizer.step()

                total_loss += batch_total_loss.item() * batch.num_graphs
                total_samples += batch.num_graphs
                total_edge_presence_loss += components.get("edge_presence_loss", torch.tensor(0.0)).item() * batch.num_graphs
                total_node_attr_loss += components.get("node_attr_loss", torch.tensor(0.0)).item() * batch.num_graphs
                total_edge_attr_loss += components.get("edge_attr_loss", torch.tensor(0.0)).item() * batch.num_graphs
                if "kl_loss" in components:
                    total_kl_loss += components["kl_loss"].item() * batch.num_graphs

        denom = max(1, total_samples)
        return {
            "lr": self.optimizer.param_groups[0]["lr"],
            "loss": total_loss / denom,
            "edge_presence_loss": total_edge_presence_loss / denom,
            "node_attr_loss": total_node_attr_loss / denom,
            "edge_attr_loss": total_edge_attr_loss / denom,
            "kl_loss": total_kl_loss / denom,
        }

    def _run_validation(self, loader: DataLoader) -> dict[str, float]:
        """Runs validation on a data loader."""
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
            self.history["train_edge_presence_loss"].append(train_metrics["edge_presence_loss"])
            self.history["train_node_attr_loss"].append(train_metrics["node_attr_loss"])
            self.history["train_edge_attr_loss"].append(train_metrics["edge_attr_loss"])
            self.history["train_kl_loss"].append(train_metrics.get("kl_loss", 0.0))
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_edge_presence_loss"].append(val_metrics["edge_presence_loss"])
            self.history["val_node_attr_loss"].append(val_metrics["node_attr_loss"])
            self.history["val_edge_attr_loss"].append(val_metrics["edge_attr_loss"])
            self.history["val_kl_loss"].append(val_metrics.get("kl_loss", 0.0))
            self.history["lr"].append(self.optimizer.param_groups[0]["lr"])

            # Check for improvement
            if val_metrics["loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["loss"]
                self.best_epoch = epoch
                self.save_checkpoint(epoch)

            # Early stopping check
            if self.early_stopping(val_metrics["loss"]):
                if verbose:
                    logging.info(
                        "Early stopping at epoch %d. Best validation loss: %.4f at epoch %s.",
                        epoch,
                        self.best_val_loss,
                        self.best_epoch,
                    )
                break

            if verbose:
                elapsed = time.time() - start_time
                kl_train = train_metrics.get("kl_loss", 0.0)
                kl_val = val_metrics.get("kl_loss", 0.0)
                logging.info(
                    "Epoch %d/%d - "
                    "LR: %.2e - "
                    "Train Loss: %.4f, "
                    "Edge Presence: %.4f, "
                    "Node Attr: %.4f, "
                    "Edge Attr: %.4f, "
                    "KL: %.4f - "
                    "Val Loss: %.4f, "
                    "Edge Presence: %.4f, "
                    "Node Attr: %.4f, "
                    "Edge Attr: %.4f, "
                    "KL: %.4f - "
                    "Time: %.2fs",
                    epoch,
                    num_epochs,
                    train_metrics["lr"],
                    train_metrics["loss"],
                    train_metrics["edge_presence_loss"],
                    train_metrics["node_attr_loss"],
                    train_metrics["edge_attr_loss"],
                    kl_train,
                    val_metrics["loss"],
                    val_metrics["edge_presence_loss"],
                    val_metrics["node_attr_loss"],
                    val_metrics["edge_attr_loss"],
                    kl_val,
                    elapsed,
                )

    def evaluate(self, loader: DataLoader) -> dict[str, float]:
        """Evaluates GAE reconstruction loss on a given data loader."""
        self.model.eval()
        total_loss, total_samples = 0.0, 0
        total_edge_presence_loss = 0.0
        total_node_attr_loss = 0.0
        total_edge_attr_loss = 0.0
        total_kl_loss = 0.0

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device, non_blocking=True)

                if hasattr(self.model, "encode"):
                    z = self.model.encode(
                        batch.x,
                        batch.edge_index,
                        edge_attr=getattr(batch, "edge_attr", None),
                        batch=getattr(batch, "batch", None),
                    )
                else:
                    out = self.model(batch)
                    z = out.get("z") if isinstance(out, dict) else None

                if z is None:
                    raise RuntimeError("Model does not expose an encoder 'encode' method or return 'z' in forward().")

                pos_ei = batch.edge_index
                num_pos = pos_ei.size(1)
                num_neg = int(num_pos * max(1.0, self.neg_sampling_ratio))
                neg_ei = negative_sampling(edge_index=pos_ei, num_nodes=batch.num_nodes, num_neg_samples=num_neg)

                components = self.model.recon_total_loss(
                    z=z,
                    pos_edge_index=pos_ei,
                    neg_edge_index=neg_ei,
                    target_x=getattr(batch, "x", None),
                    edge_index_for_attr=pos_ei,
                    target_edge_attr=getattr(batch, "edge_attr", None),
                    edge_presence_weight=self.recon_edge_presence_weight,
                    node_attr_weight=self.recon_node_attr_weight,
                    edge_attr_weight=self.recon_edge_attr_weight,
                    node_loss_type=self.recon_node_loss_type,
                    edge_attr_loss_type=self.recon_edge_attr_loss_type,
                )

                batch_loss = components["total_loss"]
                total_loss += batch_loss.item() * batch.num_graphs
                total_samples += batch.num_graphs
                total_edge_presence_loss += components.get("edge_presence_loss", torch.tensor(0.0)).item() * batch.num_graphs
                total_node_attr_loss += components.get("node_attr_loss", torch.tensor(0.0)).item() * batch.num_graphs
                total_edge_attr_loss += components.get("edge_attr_loss", torch.tensor(0.0)).item() * batch.num_graphs
                if "kl_loss" in components:
                    total_kl_loss += components["kl_loss"].item() * batch.num_graphs

        denom = max(1, total_samples)
        return {
            "loss": total_loss / denom,
            "edge_presence_loss": total_edge_presence_loss / denom,
            "node_attr_loss": total_node_attr_loss / denom,
            "edge_attr_loss": total_edge_attr_loss / denom,
            "kl_loss": total_kl_loss / denom,
        }

    def get_embeddings(self, loader: DataLoader) -> tuple[np.ndarray, np.ndarray | None]:
        """Extract graph-level embeddings and optional targets from a data loader."""
        self.model.eval()
        all_embeddings = []
        all_targets = []

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device, non_blocking=True)
                _, embeddings = self.model.forward_embeddings(batch)
                all_embeddings.append(embeddings.cpu())
                if getattr(batch, "y", None) is not None:
                    all_targets.append(batch.y.cpu())

        embeddings = torch.cat(all_embeddings).numpy()
        targets = torch.cat(all_targets).numpy() if all_targets else None
        return embeddings, targets

    @staticmethod
    def _dense_from_edge_index(
        edge_index: torch.Tensor,
        values: torch.Tensor,
        num_nodes: int,
        symmetrize: bool = True,
        include_self_loops: bool = False,
    ) -> np.ndarray:
        """Scatter sparse edge values into a dense adjacency-like array."""
        if values.ndim == 2 and values.shape[-1] == 1:
            values = values.squeeze(-1)

        if values.ndim == 1:
            dense = torch.zeros((num_nodes, num_nodes), dtype=values.dtype, device=values.device)
            src = edge_index[0]
            dst = edge_index[1]
            if not include_self_loops:
                mask = src != dst
                src = src[mask]
                dst = dst[mask]
                values = values[mask]
            dense[src, dst] = values
            if symmetrize:
                dense = 0.5 * (dense + dense.T)
            return dense.detach().cpu().numpy()

        if values.ndim == 2:
            dense = torch.zeros((num_nodes, num_nodes, values.shape[-1]), dtype=values.dtype, device=values.device)
            src = edge_index[0]
            dst = edge_index[1]
            if not include_self_loops:
                mask = src != dst
                src = src[mask]
                dst = dst[mask]
                values = values[mask]
            dense[src, dst, :] = values
            if symmetrize:
                dense = 0.5 * (dense + dense.transpose(0, 1))
            return dense.detach().cpu().numpy()

        raise ValueError(f"Unsupported edge value shape {tuple(values.shape)}. Expected [E] or [E, C].")

    def get_empirical_edge_matrices(
        self,
        data: Data,
        feature_names: list[str] | None = None,
        symmetrize: bool = True,
        include_self_loops: bool = False,
    ) -> dict[str, np.ndarray]:
        """Convert sparse edge attributes on a graph into dense per-feature matrices."""
        edge_attr = getattr(data, "edge_attr", None)
        if edge_attr is None:
            raise ValueError("data.edge_attr is required to build empirical edge matrices")

        if edge_attr.ndim == 1:
            edge_attr = edge_attr.unsqueeze(-1)

        num_features = int(edge_attr.shape[-1])
        if feature_names is None:
            feature_names = [f"feature_{idx}" for idx in range(num_features)]
        if len(feature_names) != num_features:
            raise ValueError(
                f"feature_names length ({len(feature_names)}) does not match edge_attr width ({num_features})"
            )

        matrices: dict[str, np.ndarray] = {}
        for feature_idx, feature_name in enumerate(feature_names):
            matrices[feature_name] = self._dense_from_edge_index(
                edge_index=data.edge_index,
                values=edge_attr[:, feature_idx],
                num_nodes=int(data.num_nodes),
                symmetrize=symmetrize,
                include_self_loops=include_self_loops,
            )
        return matrices

    def predict_edge_presence_matrix(
        self,
        data: Data,
        apply_sigmoid: bool = True,
        symmetrize: bool = True,
        include_self_loops: bool = False,
    ) -> np.ndarray:
        """Predict a dense edge-presence matrix from a single graph."""
        if not hasattr(self.model, "encode") or not hasattr(self.model, "decode_edge_presence"):
            raise AttributeError(
                "Model must expose both 'encode' and 'decode_edge_presence' to predict edge presence."
            )

        self.model.eval()
        data = data.to(self.device, non_blocking=True)

        batch = getattr(data, "batch", None)
        if batch is not None and int(batch.max().item()) + 1 > 1:
            raise ValueError(
                "predict_edge_presence_matrix expects a single graph. "
                "Split batched inputs and call this method on each graph individually."
            )

        with torch.no_grad():
            z = self.model.encode(
                data.x,
                data.edge_index,
                edge_attr=getattr(data, "edge_attr", None),
                batch=batch,
            )

            num_nodes = int(data.num_nodes)
            node_ids = torch.arange(num_nodes, device=self.device)
            src = node_ids.repeat_interleave(num_nodes)
            dst = node_ids.repeat(num_nodes)

            if not include_self_loops:
                mask = src != dst
                src = src[mask]
                dst = dst[mask]

            edge_index = torch.stack([src, dst], dim=0)
            edge_scores = self.model.decode_edge_presence(z, edge_index)

            if apply_sigmoid:
                edge_scores = torch.sigmoid(edge_scores)
            return self._dense_from_edge_index(
                edge_index=edge_index,
                values=edge_scores,
                num_nodes=num_nodes,
                symmetrize=symmetrize,
                include_self_loops=include_self_loops,
            )

    def reconstruct_edge_matrix(
        self,
        data: Data,
        apply_sigmoid: bool = False,
        symmetrize: bool = True,
        include_self_loops: bool = False,
    ) -> np.ndarray:
        """Reconstruct a dense edge-attribute matrix from a single graph.

        This is intended for GAE-style models that expose both ``encode`` and ``decode_edge_attrs``.
        For multi-graph batches, call this once per graph rather than on the whole batch.
        """
        if not hasattr(self.model, "encode") or not hasattr(self.model, "decode_edge_attrs"):
            raise AttributeError(
                "Model must expose both 'encode' and 'decode_edge_attrs' to reconstruct edge attributes."
            )

        self.model.eval()
        data = data.to(self.device, non_blocking=True)

        batch = getattr(data, "batch", None)
        if batch is not None and int(batch.max().item()) + 1 > 1:
            raise ValueError(
                "reconstruct_edge_matrix expects a single graph. "
                "Split batched inputs and call this method on each graph individually."
            )

        with torch.no_grad():
            z = self.model.encode(
                data.x,
                data.edge_index,
                edge_attr=getattr(data, "edge_attr", None),
                batch=batch,
            )

            num_nodes = int(data.num_nodes)
            node_ids = torch.arange(num_nodes, device=self.device)
            src = node_ids.repeat_interleave(num_nodes)
            dst = node_ids.repeat(num_nodes)

            if not include_self_loops:
                mask = src != dst
                src = src[mask]
                dst = dst[mask]

            edge_index = torch.stack([src, dst], dim=0)
            edge_scores = self.model.decode_edge_attrs(z, edge_index)
            if edge_scores is None:
                raise ValueError("Model does not expose an edge attribute decoder.")

            if apply_sigmoid:
                edge_scores = torch.sigmoid(edge_scores)
            return self._dense_from_edge_index(
                edge_index=edge_index,
                values=edge_scores,
                num_nodes=num_nodes,
                symmetrize=symmetrize,
                include_self_loops=include_self_loops,
            )

    def reconstruct_edge_matrices(
        self,
        data: Data,
        feature_names: list[str] | None = None,
        apply_sigmoid: bool = False,
        symmetrize: bool = True,
        include_self_loops: bool = False,
    ) -> dict[str, np.ndarray]:
        """Reconstruct one dense matrix per decoded edge attribute feature."""
        reconstructed = self.reconstruct_edge_matrix(
            data=data,
            apply_sigmoid=apply_sigmoid,
            symmetrize=symmetrize,
            include_self_loops=include_self_loops,
        )

        if reconstructed.ndim == 2:
            feature_name = feature_names[0] if feature_names else "edge_reconstruction"
            return {feature_name: reconstructed}

        num_features = reconstructed.shape[-1]
        if feature_names is None:
            feature_names = [f"feature_{idx}" for idx in range(num_features)]
        if len(feature_names) != num_features:
            raise ValueError(
                f"feature_names length ({len(feature_names)}) does not match reconstructed edge width ({num_features})"
            )

        return {feature_name: reconstructed[:, :, idx] for idx, feature_name in enumerate(feature_names)}

    def save_checkpoint(self, epoch: int):
        """Saves a checkpoint of the current model state."""
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        model_config = self.model.get_architecture_config() if hasattr(self.model, "get_architecture_config") else None
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": epoch,
            "best_val_loss": self.best_val_loss,
            "history": self.history,
            "model_config": model_config,
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
            logging.info(
                "Loaded checkpoint from epoch %s with best validation loss %.4f.",
                checkpoint["epoch"],
                self.best_val_loss,
            )
