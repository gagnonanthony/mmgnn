"""Helpers for building tiny synthetic connectome fixtures."""

from pathlib import Path

import numpy as np


NUM_NODES = 4
SUBJECT_IDS = ("sub-001", "sub-002")
EDGE_ATTRIBUTES = ("fa", "md")
PRIMARY_DMRI_KEY = "fa"


def sym_matrix(seed: int, size: int = NUM_NODES) -> np.ndarray:
    """Build a small symmetric connectome with zero diagonal."""
    rng = np.random.default_rng(seed)
    mat = rng.uniform(0.1, 1.0, size=(size, size)).astype(np.float32)
    mat = (mat + mat.T) / 2.0
    np.fill_diagonal(mat, 0.0)
    return mat


def dmri_filename(subject_id: str, key: str) -> str:
    return f"{subject_id}_seg-BrainnetomeChild_stat-{key}.npy"


def fc_filename(subject_id: str) -> str:
    return f"{subject_id}_seg-BrainnetomeChild_stat-pearsoncorrelation_relmat.npy"


def write_subject_matrices(
    dmri_dir: Path,
    fc_dir: Path,
    subject_id: str,
    *,
    seed: int,
    edge_attributes: tuple[str, ...] = EDGE_ATTRIBUTES,
) -> None:
    """Write synthetic dMRI + FC matrices for one subject."""
    dmri_dir.mkdir(parents=True, exist_ok=True)
    fc_dir.mkdir(parents=True, exist_ok=True)
    for offset, key in enumerate(edge_attributes):
        np.save(dmri_dir / dmri_filename(subject_id, key), sym_matrix(seed + offset))
    np.save(fc_dir / fc_filename(subject_id), sym_matrix(seed + 100))
