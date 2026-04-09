"""
Plotting functions for training and evaluation metrics in the mmgnn framework.
Includes functions to visualize training/validation loss, MAE, and Pearson correlation over epochs.
"""

from pathlib import Path
import os
import shutil

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import torch
import torchlens as tl
from torch_geometric.data import Data


def plot_training_history(history: dict, save_path: str = None, best_epoch: int = None):
    """
    Plots the training history of loss, MAE, and Pearson correlation over epochs.
    
    Args:
        history (dict): A dictionary containing lists of training and validation metrics per epoch.
        save_path (str, optional): If provided, saves the plot to the specified path.
        best_epoch (int, optional): If provided, highlights the best epoch on the plots.
    """

    num_epochs = len(history.get("train_loss", []))
    epochs = range(1, num_epochs + 1)

    def _series(key: str) -> np.ndarray:
        values = history.get(key, [])
        arr = np.asarray(values, dtype=float)
        if arr.size < num_epochs:
            pad = np.full(num_epochs - arr.size, np.nan, dtype=float)
            arr = np.concatenate([arr, pad])
        elif arr.size > num_epochs:
            arr = arr[:num_epochs]
        return arr

    train_loss = _series("train_loss")
    val_loss = _series("val_loss")
    train_mae = _series("train_mae")
    val_mae = _series("val_mae")
    train_pearson = _series("train_pearson")
    val_pearson = _series("val_pearson")
    train_accuracy = _series("train_accuracy")
    val_accuracy = _series("val_accuracy")
    train_f1 = _series("train_f1")
    val_f1 = _series("val_f1")

    has_classification_metrics = (
        np.isfinite(train_accuracy).any()
        or np.isfinite(val_accuracy).any()
        or np.isfinite(train_f1).any()
        or np.isfinite(val_f1).any()
    )

    plt.figure(figsize=(15, 5))

    # Plot Loss
    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    if best_epoch is not None:
        # Circle the best epoch on the loss plot
        if 1 <= best_epoch <= num_epochs and np.isfinite(val_loss[best_epoch - 1]):
            plt.scatter(best_epoch, val_loss[best_epoch - 1], color='red', s=100, edgecolors='black', label=f'Best Epoch: {best_epoch}')
    plt.legend()

    # Plot MAE or Accuracy
    plt.subplot(1, 3, 2)
    if has_classification_metrics:
        plt.plot(epochs, train_accuracy, label="Train Accuracy")
        plt.plot(epochs, val_accuracy, label="Val Accuracy")
    else:
        plt.plot(epochs, train_mae, label="Train MAE")
        plt.plot(epochs, val_mae, label="Val MAE")
    plt.xlabel("Epoch")
    if has_classification_metrics:
        plt.ylabel("Accuracy")
        plt.title("Training and Validation Accuracy")
    else:
        plt.ylabel("MAE")
        plt.title("Training and Validation MAE")
    if best_epoch is not None:
        if 1 <= best_epoch <= num_epochs:
            marker_value = val_accuracy[best_epoch - 1] if has_classification_metrics else val_mae[best_epoch - 1]
            if np.isfinite(marker_value):
                plt.scatter(best_epoch, marker_value, color='red', s=100, edgecolors='black', label=f'Best Epoch: {best_epoch}')
    plt.legend()

    # Plot Pearson or F1
    plt.subplot(1, 3, 3)
    if has_classification_metrics:
        plt.plot(epochs, train_f1, label="Train F1")
        plt.plot(epochs, val_f1, label="Val F1")
    else:
        plt.plot(epochs, train_pearson, label="Train Pearson")
        plt.plot(epochs, val_pearson, label="Val Pearson")
    plt.xlabel("Epoch")
    if has_classification_metrics:
        plt.ylabel("Macro F1")
        plt.title("Training and Validation F1")
    else:
        plt.ylabel("Pearson Correlation")
        plt.title("Training and Validation Pearson Correlation")
    if best_epoch is not None:
        if 1 <= best_epoch <= num_epochs:
            marker_value = val_f1[best_epoch - 1] if has_classification_metrics else val_pearson[best_epoch - 1]
            if np.isfinite(marker_value):
                plt.scatter(best_epoch, marker_value, color='red', s=100, edgecolors='black', label=f'Best Epoch: {best_epoch}')
    plt.legend()

    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path)
    
    plt.show()


def plot_predictions(preds: np.ndarray, targets: np.ndarray, save_path: str = None):
    """
    Plots predicted vs. true values for a regression task.
    
    Args:
        preds (np.ndarray): An array of predicted values.
        targets (np.ndarray): An array of true target values.
        save_path (str, optional): If provided, saves the plot to the specified path.
    """

    preds = np.asarray(preds)
    targets = np.asarray(targets)

    # Ensure 2D arrays: [N, D]
    if preds.ndim == 1:
        preds = preds.reshape(-1, 1)
    elif preds.ndim > 2:
        preds = preds.reshape(preds.shape[0], -1)

    if targets.ndim == 1:
        targets = targets.reshape(-1, 1)
    elif targets.ndim > 2:
        targets = targets.reshape(targets.shape[0], -1)

    # Align samples and target dimensions to the shared subset.
    num_samples = min(preds.shape[0], targets.shape[0])
    num_targets = min(preds.shape[1], targets.shape[1])

    if num_samples == 0 or num_targets == 0:
        raise ValueError("preds and targets must contain at least one sample and one target dimension")

    if preds.shape != targets.shape:
        print(
            "Warning: preds and targets have different shapes; plotting the shared subset "
            f"[samples={num_samples}, targets={num_targets}]"
        )

    preds = preds[:num_samples, :num_targets]
    targets = targets[:num_samples, :num_targets]

    # Detect subplots based on the number of target dimensions
    fig, axes = plt.subplots(1, num_targets, figsize=(5 * num_targets, 5))
    if num_targets == 1:
        axes = [axes]  # Ensure axes is iterable

    for i in range(num_targets):
        ax = axes[i]
        ax.scatter(targets[:, i], preds[:, i], alpha=0.5)
        ax.plot([targets[:, i].min(), targets[:, i].max()], [targets[:, i].min(), targets[:, i].max()], 'r--')
        ax.set_xlabel("True Values")
        ax.set_ylabel("Predicted Values")
        ax.set_title(f"Predicted vs. True Values (Target {i + 1})")

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()


def plot_classification_probabilities(
    probabilities: np.ndarray,
    target_memberships: np.ndarray,
    save_path: str = None,
):
    """
    Plot predicted class probabilities against target memberships per class.

    This mirrors the regression scatter view but constrains both axes to [0, 1]
    because values represent probabilities/memberships.
    """
    probs = np.asarray(probabilities)
    targets = np.asarray(target_memberships)

    if probs.ndim == 1:
        probs = probs.reshape(-1, 1)
    elif probs.ndim > 2:
        probs = probs.reshape(probs.shape[0], -1)

    if targets.ndim == 1:
        targets = targets.reshape(-1, 1)
    elif targets.ndim > 2:
        targets = targets.reshape(targets.shape[0], -1)

    num_samples = min(probs.shape[0], targets.shape[0])
    num_classes = min(probs.shape[1], targets.shape[1])

    if num_samples == 0 or num_classes == 0:
        raise ValueError("probabilities and target_memberships must contain at least one sample and one class")

    if probs.shape != targets.shape:
        print(
            "Warning: probabilities and target_memberships have different shapes; plotting shared subset "
            f"[samples={num_samples}, classes={num_classes}]"
        )

    probs = probs[:num_samples, :num_classes]
    targets = targets[:num_samples, :num_classes]

    fig, axes = plt.subplots(1, num_classes, figsize=(5 * num_classes, 5))
    if num_classes == 1:
        axes = [axes]

    for i in range(num_classes):
        ax = axes[i]
        ax.scatter(targets[:, i], probs[:, i], alpha=0.5)
        ax.plot([0.0, 1.0], [0.0, 1.0], "r--")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("True Membership")
        ax.set_ylabel("Predicted Probability")
        ax.set_title(f"Class {i + 1}: Pred Prob vs True Membership")

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()


def save_model_graph_torchlens(
    model: torch.nn.Module,
    batch: Data,
    save_path: str = "./model_graph.png",
    vis_mode: str = "rolled",
):
    """
    Save model graph visualization as PNG using TorchLens.

    Args:
        model (torch.nn.Module): Model to visualize.
        batch (torch_geometric.data.Data): Sample graph used to trace the forward pass.
        save_path (str): Output PNG path.
        vis_mode (str): TorchLens graph mode ("rolled" or "unrolled").

    Returns:
        pathlib.Path: Path to the saved PNG file.
    """
    save_png = Path(save_path)
    save_png.parent.mkdir(parents=True, exist_ok=True)

    # Resolve Graphviz binary for notebook/kernel environments with stale PATH.
    dot_path = shutil.which("dot")
    if dot_path is None:
        for candidate in ("/opt/homebrew/bin/dot", "/usr/local/bin/dot"):
            if Path(candidate).exists():
                dot_path = candidate
                dot_dir = str(Path(candidate).parent)
                current_path = os.environ.get("PATH", "")
                if dot_dir not in current_path.split(":"):
                    os.environ["PATH"] = f"{dot_dir}:{current_path}" if current_path else dot_dir
                break

    if dot_path is not None:
        os.environ["GRAPHVIZ_DOT"] = dot_path

    # Wrap MMGNN so TorchLens can trace tensor arguments and rebuild Data internally.
    class _TraceWrapper(torch.nn.Module):
        def __init__(self, wrapped_model: torch.nn.Module):
            super().__init__()
            self.wrapped_model = wrapped_model

        def forward(self, x, edge_index, edge_attr, batch_index):
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch_index)
            return self.wrapped_model(data)

    model_was_training = model.training
    model.eval()

    model_device = next(model.parameters()).device
    x = batch.x.to(model_device)
    edge_index = batch.edge_index.to(model_device)

    edge_attr = getattr(batch, "edge_attr", None)
    if edge_attr is None:
        edge_attr = torch.zeros((edge_index.size(1), 1), dtype=x.dtype, device=model_device)
    else:
        edge_attr = edge_attr.to(model_device)

    batch_index = getattr(batch, "batch", None)
    if batch_index is None:
        batch_index = torch.zeros(x.size(0), dtype=torch.long, device=model_device)
    else:
        batch_index = batch_index.to(model_device)

    trace_model = _TraceWrapper(model).to(model_device).eval()

    vis_mode = vis_mode.lower().strip()
    if vis_mode not in {"rolled", "unrolled"}:
        raise ValueError("vis_mode must be 'rolled' or 'unrolled'")

    try:
        try:
            tl.show_model_graph(
                trace_model,
                (x, edge_index, edge_attr, batch_index),
                vis_mode=vis_mode,
                vis_outpath=str(save_png),
                vis_save_only=True,
                vis_fileformat="png",
                vis_direction="bottomup",
                vis_renderer="graphviz",
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                "TorchLens could not find Graphviz 'dot'. Install Graphviz and ensure dot is visible to this "
                "Python process. On macOS: brew install graphviz, then restart the notebook kernel."
            ) from exc
    finally:
        model.train(model_was_training)

    return save_png


def plot_embeddings(
    embeddings: np.ndarray,
    targets: np.ndarray = None,
    save_path: str = None,
    perplexity: float = 30.0,
    random_state: int = 42,
    max_iter: int = 1000,
    cmap: str = "viridis",
):
    """
    Plot graph-level embeddings in 2D using t-SNE, with points colored by target value.

    Args:
        embeddings (np.ndarray): Embedding matrix, shape [num_samples, embedding_dim].
        targets (np.ndarray, optional): Target values used for coloring, shape
            [num_samples] or [num_samples, num_targets].
        save_path (str, optional): If provided, saves the plot to the specified path.
        perplexity (float): t-SNE perplexity. Must be > 0 and < number of samples.
        random_state (int): Random seed for deterministic t-SNE output.
        max_iter (int): Number of optimization iterations for t-SNE.
        cmap (str): Matplotlib colormap name for target coloring.

    Returns:
        np.ndarray: 2D t-SNE coordinates with shape [num_samples_used, 2].
    """
    if targets is None:
        raise ValueError("targets must be provided for target-colored embedding plots")

    emb = np.asarray(embeddings, dtype=float)
    y = np.asarray(targets, dtype=float)

    if emb.ndim == 1:
        emb = emb.reshape(-1, 1)
    elif emb.ndim > 2:
        emb = emb.reshape(emb.shape[0], -1)

    if y.ndim == 1:
        y = y.reshape(-1, 1)
    elif y.ndim > 2:
        y = y.reshape(y.shape[0], -1)

    num_samples = min(emb.shape[0], y.shape[0])
    if num_samples < 2:
        raise ValueError("embeddings and targets must contain at least 2 samples")

    if emb.shape[0] != y.shape[0]:
        print(
            "Warning: embeddings and targets have different numbers of samples; plotting the shared subset "
            f"[samples={num_samples}]"
        )

    emb = emb[:num_samples]
    y = y[:num_samples]

    # Remove rows where embedding coordinates are not finite before running t-SNE.
    finite_embedding_mask = np.isfinite(emb).all(axis=1)
    emb = emb[finite_embedding_mask]
    y = y[finite_embedding_mask]

    if emb.shape[0] < 2:
        raise ValueError("need at least 2 finite embedding rows to compute t-SNE")

    if perplexity <= 0:
        raise ValueError("perplexity must be > 0")
    if perplexity >= emb.shape[0]:
        raise ValueError(
            f"perplexity ({perplexity}) must be smaller than the number of samples ({emb.shape[0]})"
        )

    reducer = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=random_state,
        max_iter=max_iter,
        init="pca",
        learning_rate="auto",
        verbose=0,
    )
    # Subsample to 1000 samples if more than 1000 to speed up t-SNE and avoid overcrowded plots.
    if emb.shape[0] > 1000:
        sample_indices = np.random.RandomState(seed=random_state).choice(emb.shape[0], size=1000, replace=False)
        emb = emb[sample_indices]
        y = y[sample_indices]
    else:
        sample_indices = np.arange(emb.shape[0])
        emb = emb[sample_indices]
        y = y[sample_indices]
    embeddings_2d = reducer.fit_transform(emb)

    num_targets = y.shape[1]
    n_cols = min(3, num_targets)
    n_rows = (num_targets + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    for target_idx in range(num_targets):
        ax = axes[target_idx]
        target_values = y[:, target_idx]
        valid_target_mask = np.isfinite(target_values)

        if not np.any(valid_target_mask):
            ax.text(0.5, 0.5, "No finite values", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"Target {target_idx + 1}")
            ax.set_axis_off()
            continue

        scatter = ax.scatter(
            embeddings_2d[valid_target_mask, 0],
            embeddings_2d[valid_target_mask, 1],
            c=target_values[valid_target_mask],
            cmap=cmap,
            alpha=0.75,
            s=45,
            edgecolors="black",
            linewidth=0.3,
        )
        ax.set_xlabel("t-SNE Component 1")
        ax.set_ylabel("t-SNE Component 2")
        ax.set_title(f"Target {target_idx + 1} (n={valid_target_mask.sum()})")
        ax.grid(True, alpha=0.25, linestyle="--")
        plt.colorbar(scatter, ax=ax, label=f"Target {target_idx + 1}")

    for idx in range(num_targets, len(axes)):
        axes[idx].axis("off")

    fig.suptitle("Graph Embeddings Colored by Target Values", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.98])

    if save_path is not None:
        save_file = Path(save_path)
        save_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_file, dpi=300, bbox_inches="tight")

    plt.show()
    return embeddings_2d


def plot_contingency_heatmap(
    contingency_matrix: np.ndarray,
    x_labels: list = None,
    y_labels: list = None,
    title: str = "Contingency Heatmap",
    save_path: str = None,
    cmap: str = "Blues",
):
    """
    Plots a contingency heatmap given a 2D matrix of counts or values.

    Args:
        contingency_matrix (np.ndarray): 2D array representing the contingency matrix.
        x_labels (list, optional): Labels for the x-axis categories.
        y_labels (list, optional): Labels for the y-axis categories.
        title (str, optional): Title of the heatmap.
        save_path (str, optional): If provided, saves the plot to the specified path.
        cmap (str, optional): Matplotlib colormap name for the heatmap.
    """
    matrix = np.asarray(contingency_matrix)

    if matrix.ndim != 2:
        raise ValueError("contingency_matrix must be a 2D array")
    if matrix.size == 0 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("contingency_matrix must be non-empty")
    if not np.issubdtype(matrix.dtype, np.number):
        raise ValueError("contingency_matrix must contain numeric values")
    if not np.isfinite(matrix).all():
        raise ValueError("contingency_matrix must contain only finite values")

    n_rows, n_cols = matrix.shape

    if x_labels is None:
        x_labels = [f"Pred {idx + 1}" for idx in range(n_cols)]
    if y_labels is None:
        y_labels = [f"True {idx + 1}" for idx in range(n_rows)]

    if len(x_labels) != n_cols:
        raise ValueError(
            f"x_labels length ({len(x_labels)}) must match number of columns ({n_cols})"
        )
    if len(y_labels) != n_rows:
        raise ValueError(
            f"y_labels length ({len(y_labels)}) must match number of rows ({n_rows})"
        )

    is_integer_like = np.allclose(matrix, np.round(matrix))
    annot_values = matrix.astype(int) if is_integer_like else matrix
    annot_fmt = "d" if is_integer_like else ".2f"

    fig_width = max(6, min(14, 1.2 * n_cols + 3))
    fig_height = max(5, min(12, 1.0 * n_rows + 2.5))
    plt.figure(figsize=(fig_width, fig_height))

    ax = sns.heatmap(
        matrix,
        annot=annot_values,
        fmt=annot_fmt,
        cmap=cmap,
        xticklabels=x_labels,
        yticklabels=y_labels,
        linewidths=0.5,
        linecolor="white",
        cbar=True,
        square=False,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    plt.tight_layout()

    if save_path is not None:
        save_file = Path(save_path)
        save_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_file, dpi=300, bbox_inches="tight")

    plt.show()
