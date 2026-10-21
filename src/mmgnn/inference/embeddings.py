"""Inference utilities for extracting and saving GAE/VGAE embeddings."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


def _batch_subject_ids(batch: Data) -> list[str]:
    subject_ids = batch.subject_id
    if isinstance(subject_ids, str):
        return [subject_ids]
    if isinstance(subject_ids, (list, tuple)):
        return [str(subject_id) for subject_id in subject_ids]
    return [str(subject_ids)]


def _node_embeddings_from_latents(latents: torch.Tensor, _graph: Data) -> np.ndarray:
    """Convert per-node latents to a numpy array."""
    return latents.detach().cpu().numpy()


@torch.no_grad()
def extract_embeddings(
    model: torch.nn.Module,
    graphs: list[Data],
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[
    np.ndarray,
    dict[str, np.ndarray],
    list[str],
    dict[str, np.ndarray] | None,
    dict[str, np.ndarray] | None,
]:
    """Extract graph-level and per-node embeddings for a list of subject graphs.

    For VGAE models (with ``encode_stats``), also returns per-subject ``logvar`` and
    ``sigma`` maps. For deterministic GAE models those two dicts are ``None``.
    """
    model.eval()
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)
    has_variational = hasattr(model, "encode_stats")

    graph_embeddings_chunks: list[np.ndarray] = []
    node_embeddings_by_subject: dict[str, np.ndarray] = {}
    node_logvar_by_subject: dict[str, np.ndarray] = {}
    node_sigma_by_subject: dict[str, np.ndarray] = {}
    subject_ids: list[str] = []

    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        batch_subject_ids = _batch_subject_ids(batch)

        _, graph_embeddings = model.forward_embeddings(batch)
        graph_embeddings_chunks.append(graph_embeddings.cpu().numpy())

        encode_kwargs = {
            "edge_attr": getattr(batch, "edge_attr", None),
            "batch": getattr(batch, "batch", None),
        }
        if has_variational:
            mu, logvar = model.encode_stats(
                batch.x,
                batch.edge_index,
                **encode_kwargs,
            )
            node_latents = mu
            sigma = torch.exp(0.5 * logvar)
        else:
            node_latents = model.encode(
                batch.x,
                batch.edge_index,
                **encode_kwargs,
            )
            logvar = None
            sigma = None

        if getattr(batch, "batch", None) is None:
            if len(batch_subject_ids) != 1:
                raise RuntimeError("Expected a single graph when batch vector is missing.")
            subject_id = batch_subject_ids[0]
            node_embeddings_by_subject[subject_id] = _node_embeddings_from_latents(node_latents, batch)
            if has_variational:
                node_logvar_by_subject[subject_id] = _node_embeddings_from_latents(logvar, batch)
                node_sigma_by_subject[subject_id] = _node_embeddings_from_latents(sigma, batch)
            subject_ids.append(subject_id)
            continue

        batch_vector = batch.batch.detach().cpu()
        unique_batch_ids = sorted(int(value) for value in batch_vector.unique())
        if len(unique_batch_ids) != len(batch_subject_ids):
            raise RuntimeError("Batch size does not match number of subject IDs in batch.")

        for batch_index, subject_id in zip(unique_batch_ids, batch_subject_ids):
            node_mask = batch_vector == batch_index
            graph = batch.get_example(batch_index)
            node_embeddings_by_subject[subject_id] = _node_embeddings_from_latents(
                node_latents[node_mask],
                graph,
            )
            if has_variational:
                node_logvar_by_subject[subject_id] = _node_embeddings_from_latents(
                    logvar[node_mask],
                    graph,
                )
                node_sigma_by_subject[subject_id] = _node_embeddings_from_latents(
                    sigma[node_mask],
                    graph,
                )
            subject_ids.append(subject_id)

    if not graph_embeddings_chunks:
        raise ValueError("No embeddings were produced for the requested subjects")

    graph_embeddings = np.concatenate(graph_embeddings_chunks, axis=0)
    if len(subject_ids) != graph_embeddings.shape[0]:
        raise RuntimeError(
            "Subject ID count does not match graph embedding count: "
            f"{len(subject_ids)} vs {graph_embeddings.shape[0]}"
        )

    if has_variational:
        return (
            graph_embeddings,
            node_embeddings_by_subject,
            subject_ids,
            node_logvar_by_subject,
            node_sigma_by_subject,
        )
    return graph_embeddings, node_embeddings_by_subject, subject_ids, None, None


def load_subject_node_embeddings(embeddings_dir: Path, subject_id: str) -> np.ndarray:
    """Load per-node GAE embeddings saved by ``mmgnn.cli.inference``.

    Expects ``embeddings_dir/subjects/<subject_id>/node_embeddings.npy`` with
    shape ``(num_nodes, embedding_dim)``.
    """
    embeddings_path = embeddings_dir / "subjects" / subject_id / "node_embeddings.npy"
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Node embeddings not found: {embeddings_path}")
    embeddings = np.load(embeddings_path)
    if embeddings.ndim != 2:
        raise ValueError(
            f"Expected node_embeddings.npy to have shape (num_nodes, dim); got {embeddings.shape}"
        )
    return embeddings


def save_subject_node_embeddings(
    output_dir: Path,
    subject_id: str,
    *,
    node_embeddings: np.ndarray,
    node_logvar: np.ndarray | None = None,
    node_sigma: np.ndarray | None = None,
) -> dict[str, Any]:
    """Write per-subject node embeddings and summary JSON under ``subjects/<id>/``."""
    subject_dir = output_dir / "subjects" / subject_id
    subject_dir.mkdir(parents=True, exist_ok=True)
    np.save(subject_dir / "node_embeddings.npy", node_embeddings)

    summary: dict[str, Any] = {
        "subject_id": subject_id,
        "node_embedding_shape": list(node_embeddings.shape),
        "num_nodes": int(node_embeddings.shape[0]),
        "embedding_dim": int(node_embeddings.shape[1]) if node_embeddings.ndim == 2 else 0,
        "has_uncertainty": node_logvar is not None and node_sigma is not None,
    }
    if node_logvar is not None:
        np.save(subject_dir / "node_logvar.npy", node_logvar)
        summary["node_logvar_shape"] = list(node_logvar.shape)
    if node_sigma is not None:
        np.save(subject_dir / "node_sigma.npy", node_sigma)
        summary["node_sigma_shape"] = list(node_sigma.shape)

    with open(subject_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=4)
    return summary


def save_subject_node_embeddings_batch(
    output_dir: Path,
    node_embeddings_by_subject: dict[str, np.ndarray],
    *,
    node_logvar_by_subject: dict[str, np.ndarray] | None = None,
    node_sigma_by_subject: dict[str, np.ndarray] | None = None,
) -> list[dict[str, Any]]:
    """Persist node embeddings (and optional VGAE uncertainty maps) for all subjects."""
    summaries: list[dict[str, Any]] = []
    for subject_id, node_embeddings in node_embeddings_by_subject.items():
        logvar = None if node_logvar_by_subject is None else node_logvar_by_subject.get(subject_id)
        sigma = None if node_sigma_by_subject is None else node_sigma_by_subject.get(subject_id)
        summaries.append(
            save_subject_node_embeddings(
                output_dir,
                subject_id,
                node_embeddings=node_embeddings,
                node_logvar=logvar,
                node_sigma=sigma,
            )
        )
    return summaries
