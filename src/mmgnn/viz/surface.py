"""
Connectome visualization utilities.

This module loads node embeddings from inference outputs and renders cortical
surface maps via yabplot.
"""

from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import yabplot as yab

from mmgnn.inference.embeddings import load_subject_node_embeddings
from mmgnn.viz.custom_atlas import (
    get_parcellation_region_names,
    resolve_atlas_argument,
)


AggregateMode = Literal["mean_abs", "mean_signed"]


def build_subject_node_embeddings(
    *,
    data_dir: Path,
    subject_id: str,
) -> np.ndarray:
    """Return node embeddings with shape (num_nodes, embedding_dim)."""
    return load_subject_node_embeddings(data_dir, subject_id)


def build_group_node_embeddings(
    *,
    data_dir: Path,
    subject_ids: list[str],
    aggregate: AggregateMode = "mean_signed",
) -> np.ndarray:
    """Average node embeddings across subjects, returning shape (num_nodes, embedding_dim)."""
    embeddings = [load_subject_node_embeddings(data_dir, subject_id) for subject_id in subject_ids]
    node_counts = {array.shape[0] for array in embeddings}
    if len(node_counts) != 1:
        raise ValueError(f"Inconsistent node counts across subjects: {sorted(node_counts)}")

    embedding_dims = {array.shape[1] for array in embeddings}
    if len(embedding_dims) != 1:
        raise ValueError(f"Inconsistent embedding dimensions across subjects: {sorted(embedding_dims)}")

    return _aggregate(embeddings, aggregate)


def _embedding_surface_path(output_path: Path, dim: int) -> Path:
    if output_path.suffix.lower() in {".png", ".pdf", ".svg", ".jpg", ".jpeg"}:
        return output_path.with_name(f"{output_path.stem}_dim{dim:03d}{output_path.suffix}")
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path / f"embedding_dim_{dim:03d}.png"


def plot_node_embedding_surfaces(
    *,
    node_embeddings: np.ndarray,
    atlas: str,
    output_path: Path,
    title_prefix: str | None = None,
    views: list[str] | None = None,
    cmap: str = "viridis",
) -> list[Path]:
    """Render one cortical surface figure per embedding dimension."""
    if node_embeddings.ndim != 2:
        raise ValueError(f"node_embeddings must be 2D (num_nodes, dim); got shape {node_embeddings.shape}")

    saved_paths: list[Path] = []
    for dim in range(node_embeddings.shape[1]):
        dim_path = _embedding_surface_path(output_path, dim)
        title = f"{title_prefix} (dim {dim})" if title_prefix else f"embedding dim {dim}"
        plot_node_scores_surface(
            node_scores=node_embeddings[:, dim],
            atlas=atlas,
            output_path=dim_path,
            title=title,
            views=views,
            cmap=cmap,
        )
        saved_paths.append(dim_path)
    return saved_paths


def load_subject_ids(subjects_file: Path) -> list[str]:
    """Load subject IDs from newline-delimited text file."""
    with open(subjects_file, "r", encoding="utf-8") as handle:
        subject_ids = [line.strip() for line in handle if line.strip()]
    if not subject_ids:
        raise ValueError(f"No subject IDs found in {subjects_file}")
    return subject_ids


def _aggregate(values: list[np.ndarray], mode: AggregateMode) -> np.ndarray:
    stacked = np.stack(values, axis=0)
    if mode == "mean_abs":
        return np.mean(np.abs(stacked), axis=0)
    if mode == "mean_signed":
        return np.mean(stacked, axis=0)
    raise ValueError(f"Unsupported aggregate mode: {mode}")


def plot_node_scores_surface(
    *,
    node_scores: np.ndarray,
    atlas: str,
    output_path: Path,
    title: str | None = None,
    views: list[str] | None = None,
    cmap: str = "viridis",
) -> None:
    """Render parcel-level node scores on the cortical surface with yabplot.

    Uses ``yabplot.plot_cortical`` to paint each cortical vertex with its
    parcellation score. Subcortical parcels are ignored on the surface mesh.
    """
    views = views or ["left_lateral", "left_medial", "right_lateral", "right_medial", "anterior", "superior"]
    resolved_atlas = resolve_atlas_argument(atlas)
    region_names = get_parcellation_region_names(atlas)

    if len(node_scores) != len(region_names):
        raise ValueError(
            f"node_scores length ({len(node_scores)}) does not match atlas regions "
            f"({len(region_names)})."
        )

    surface_data = {
        name: float(score) for name, score in zip(region_names, node_scores, strict=True)
    }

    fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=200)
    yab.plot_cortical(
        data=surface_data,
        ax=ax,
        views=views,
        cmap=cmap,
        **resolved_atlas.cortical_plot_kwargs(),
    )
    if title:
        ax.set_title(title)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
