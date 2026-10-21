"""
Plotting functions for GAE training and evaluation.
"""

from pathlib import Path
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE


def plot_training_history(history: dict, save_path: str = None, best_epoch: int = None):
    """
    Plot GAE training history: total loss and reconstruction component losses.

    Args:
        history: Dictionary containing per-epoch train/val metrics.
        save_path: Optional path to save the figure.
        best_epoch: Optional epoch to highlight on the plots.
    """
    num_epochs = len(history.get("train_loss", []))
    if num_epochs == 0:
        raise ValueError("history does not contain any training epochs to plot")

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

    loss_panels = [
        {
            "title": "Total Reconstruction Loss",
            "series": [
                ("Train", _series("train_loss")),
                ("Val", _series("val_loss")),
            ],
            "mark_best": True,
        },
        {
            "title": "Edge Presence Loss",
            "series": [
                ("Train", _series("train_edge_presence_loss")),
                ("Val", _series("val_edge_presence_loss")),
            ],
            "mark_best": False,
        },
        {
            "title": "Node Attribute Loss",
            "series": [
                ("Train", _series("train_node_attr_loss")),
                ("Val", _series("val_node_attr_loss")),
            ],
            "mark_best": False,
        },
        {
            "title": "Edge Attribute Loss",
            "series": [
                ("Train", _series("train_edge_attr_loss")),
                ("Val", _series("val_edge_attr_loss")),
            ],
            "mark_best": False,
        },
    ]

    active_panels = [
        panel
        for panel in loss_panels
        if any(np.isfinite(values).any() for _, values in panel["series"])
    ]
    if not active_panels:
        raise ValueError("history does not contain any finite GAE reconstruction losses to plot")

    n_cols = min(2, len(active_panels))
    n_rows = (len(active_panels) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    val_loss = _series("val_loss")
    for ax, panel in zip(axes, active_panels):
        for label, values in panel["series"]:
            if np.isfinite(values).any():
                ax.plot(epochs, values, label=label)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(panel["title"])
        if panel["mark_best"] and best_epoch is not None and 1 <= best_epoch <= num_epochs:
            if np.isfinite(val_loss[best_epoch - 1]):
                ax.scatter(
                    best_epoch,
                    val_loss[best_epoch - 1],
                    color="red",
                    s=100,
                    edgecolors="black",
                    label=f"Best Epoch: {best_epoch}",
                )
        ax.legend()
        ax.grid(True, alpha=0.25, linestyle="--")

    for ax in axes[len(active_panels):]:
        ax.axis("off")

    fig.tight_layout()

    if save_path is not None:
        save_file = Path(save_path)
        save_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_file, dpi=300, bbox_inches="tight")
        plt.close(fig)


def _compute_embedding_tsne(
    embeddings: np.ndarray,
    *,
    perplexity: float,
    random_state: int,
    max_iter: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    emb = np.asarray(embeddings, dtype=float)
    if emb.ndim == 1:
        emb = emb.reshape(-1, 1)
    elif emb.ndim > 2:
        emb = emb.reshape(emb.shape[0], -1)

    finite_embedding_mask = np.isfinite(emb).all(axis=1)
    emb = emb[finite_embedding_mask]

    if emb.shape[0] < 2:
        raise ValueError("need at least 2 finite embedding rows to compute t-SNE")
    if perplexity <= 0:
        raise ValueError("perplexity must be > 0")
    if perplexity >= emb.shape[0]:
        raise ValueError(
            f"perplexity ({perplexity}) must be smaller than the number of samples ({emb.shape[0]})"
        )

    if emb.shape[0] > 1000:
        sample_indices = np.random.RandomState(seed=random_state).choice(emb.shape[0], size=1000, replace=False)
        emb = emb[sample_indices]
    else:
        sample_indices = np.arange(emb.shape[0])

    reducer = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=random_state,
        max_iter=max_iter,
        init="pca",
        learning_rate="auto",
        verbose=0,
    )
    embeddings_2d = reducer.fit_transform(emb)
    return embeddings_2d, sample_indices, finite_embedding_mask


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
    Plot graph-level embeddings in 2D using t-SNE.

    When targets are provided, points are colored by target value. Otherwise a
    single unlabeled scatter plot is produced.

    Args:
        embeddings: Embedding matrix, shape [num_samples, embedding_dim].
        targets: Optional target values for coloring, shape [num_samples] or
            [num_samples, num_targets].
        save_path: Optional path to save the plot.
        perplexity: t-SNE perplexity. Must be > 0 and < number of samples.
        random_state: Random seed for deterministic t-SNE output.
        max_iter: Number of optimization iterations for t-SNE.
        cmap: Matplotlib colormap name for target coloring.

    Returns:
        np.ndarray: 2D t-SNE coordinates with shape [num_samples_used, 2].
    """
    embeddings_2d, sample_indices, finite_embedding_mask = _compute_embedding_tsne(
        embeddings,
        perplexity=perplexity,
        random_state=random_state,
        max_iter=max_iter,
    )

    if targets is None:
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(
            embeddings_2d[:, 0],
            embeddings_2d[:, 1],
            alpha=0.75,
            s=45,
            edgecolors="black",
            linewidth=0.3,
        )
        ax.set_xlabel("t-SNE Component 1")
        ax.set_ylabel("t-SNE Component 2")
        ax.set_title(f"Graph Embeddings (n={embeddings_2d.shape[0]})")
        ax.grid(True, alpha=0.25, linestyle="--")
        fig.tight_layout()

        if save_path is not None:
            save_file = Path(save_path)
            save_file.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_file, dpi=300, bbox_inches="tight")
            plt.close(fig)

        return embeddings_2d

    emb = np.asarray(embeddings, dtype=float)
    if emb.ndim == 1:
        emb = emb.reshape(-1, 1)
    elif emb.ndim > 2:
        emb = emb.reshape(emb.shape[0], -1)

    y = np.asarray(targets, dtype=float)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    elif y.ndim > 2:
        y = y.reshape(y.shape[0], -1)

    if emb.shape[0] != y.shape[0]:
        num_samples = min(emb.shape[0], y.shape[0])
        logging.warning(
            "Warning: embeddings and targets have different numbers of samples; plotting the shared subset "
            f"[samples={num_samples}]"
        )
        emb = emb[:num_samples]
        y = y[:num_samples]

    if y.shape[0] < 2:
        raise ValueError("embeddings and targets must contain at least 2 samples")

    y = y[finite_embedding_mask][sample_indices]

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
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    if save_path is not None:
        save_file = Path(save_path)
        save_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_file, dpi=300, bbox_inches="tight")
        plt.close(fig)

    return embeddings_2d


def plot_connectivity_heatmap(
    matrix: np.ndarray,
    title: str,
    save_path: str = None,
    cmap: str = "viridis",
):
    """Plot a single dense connectivity matrix."""
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2:
        raise ValueError("matrix must be 2D")
    if values.size == 0:
        raise ValueError("matrix must be non-empty")
    if not np.isfinite(values).all():
        raise ValueError("matrix must contain only finite values")

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(values, ax=ax, cmap=cmap, square=True, cbar=True, xticklabels=False, yticklabels=False)
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.tight_layout()

    if save_path is not None:
        save_file = Path(save_path)
        save_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_file, dpi=300, bbox_inches="tight")


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


def plot_connectivity_triptych(
    empirical_matrix: np.ndarray,
    reconstructed_matrix: np.ndarray,
    title: str,
    save_path: str = None,
    cmap: str = "viridis",
    diff_cmap: str = "coolwarm"
):
    """
    Plot empirical, reconstructed, residual, and empirical-vs-reconstructed correlation panels.

    Args:
        empirical_matrix: Ground-truth dense connectivity matrix.
        reconstructed_matrix: Model-reconstructed dense connectivity matrix.
        title: Figure title.
        save_path: Optional path to save the figure.
        cmap: Colormap for the empirical and reconstructed matrices.
        diff_cmap: Colormap for the residual matrix.
    """
    empirical = np.asarray(empirical_matrix, dtype=float)
    reconstructed = np.asarray(reconstructed_matrix, dtype=float)

    if empirical.ndim != 2 or reconstructed.ndim != 2:
        raise ValueError("empirical_matrix and reconstructed_matrix must be 2D arrays")
    if empirical.shape != reconstructed.shape:
        raise ValueError(
            f"Matrix shape mismatch: empirical {empirical.shape} vs reconstructed {reconstructed.shape}"
        )
    if empirical.size == 0:
        raise ValueError("Connectivity matrices must be non-empty")
    if not np.isfinite(empirical).all() or not np.isfinite(reconstructed).all():
        raise ValueError("Connectivity matrices must contain only finite values")

    residual = reconstructed - empirical
    finite_values = np.concatenate([empirical.ravel(), reconstructed.ravel()])
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        raise ValueError("Connectivity matrices must contain at least one finite value")

    value_min = float(np.min(finite_values))
    value_max = float(np.max(finite_values))
    if value_min == value_max:
        value_min -= 1.0
        value_max += 1.0

    diff_limit = float(np.max(np.abs(residual)))
    if diff_limit == 0.0:
        diff_limit = 1.0

    upper_mask = np.triu(np.ones(empirical.shape, dtype=bool), k=1)
    nonzero_mask = (empirical != 0.0) & (reconstructed != 0.0)
    finite_mask = np.isfinite(empirical) & np.isfinite(reconstructed)
    corr_mask = upper_mask & nonzero_mask & finite_mask
    if not np.any(corr_mask):
        raise ValueError("Connectivity matrices must contain at least one shared non-zero upper-triangular value")

    empirical_vals = empirical[corr_mask]
    reconstructed_vals = reconstructed[corr_mask]

    if empirical_vals.size > 1 and np.std(empirical_vals) > 0.0 and np.std(reconstructed_vals) > 0.0:
        corr = float(np.corrcoef(empirical_vals, reconstructed_vals)[0, 1])
    else:
        corr = float("nan")

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.ravel()
    panels = [
        (empirical, "Empirical", value_min, value_max, cmap),
        (reconstructed, "Reconstructed", value_min, value_max, cmap),
        (residual, "Residual", -diff_limit, diff_limit, diff_cmap),
    ]

    for ax, (matrix, panel_title, vmin, vmax, panel_cmap) in zip(axes, panels):
        sns.heatmap(
            matrix,
            ax=ax,
            cmap=panel_cmap,
            vmin=vmin,
            vmax=vmax,
            square=True,
            cbar=True,
            xticklabels=False,
            yticklabels=False,
        )
        ax.set_title(panel_title)
        ax.set_xlabel("")
        ax.set_ylabel("")

    corr_ax = axes[3]
    scatter_min = float(min(np.min(empirical_vals), np.min(reconstructed_vals)))
    scatter_max = float(max(np.max(empirical_vals), np.max(reconstructed_vals)))
    if scatter_min == scatter_max:
        scatter_min -= 1.0
        scatter_max += 1.0

    corr_ax.scatter(
        empirical_vals,
        reconstructed_vals,
        s=10,
        alpha=0.35,
        edgecolors="none",
        color="tab:blue",
    )
    corr_ax.plot([scatter_min, scatter_max], [scatter_min, scatter_max], linestyle="--", color="black", linewidth=1.2)
    corr_ax.set_xlim(scatter_min, scatter_max)
    corr_ax.set_ylim(scatter_min, scatter_max)
    corr_ax.set_aspect("equal", adjustable="box")
    corr_ax.set_xlabel("Empirical edge attributes")
    corr_ax.set_ylabel("Predicted edge attributes")
    corr_ax.set_title("Empirical vs Predicted")
    corr_text = f"Pearson r = {corr:.3f}" if np.isfinite(corr) else "Pearson r = undefined"
    corr_ax.text(
        0.05,
        0.95,
        corr_text,
        transform=corr_ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="none"),
    )
    corr_ax.grid(True, alpha=0.25, linestyle="--")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path is not None:
        save_file = Path(save_path)
        save_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_file, dpi=300, bbox_inches="tight")
