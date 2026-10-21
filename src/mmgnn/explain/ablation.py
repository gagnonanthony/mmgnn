"""
Leave-one-dimension-out and greedy ablation of GAE latent embeddings.

Computes reconstruction Δloss importance, edge-pair diagnostics, greedy
sequential mean-replace ablation over all dims, and joint top-K ablation.
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling

LOSS_KEYS = (
    "total_loss",
    "edge_presence_loss",
    "node_attr_loss",
    "edge_attr_loss",
)


def ablate_latent_column(z: torch.Tensor, dim: int) -> torch.Tensor:
    """Return a copy of ``z`` with column ``dim`` replaced by its node-wise mean."""
    return ablate_latent_columns(z, [dim])


def ablate_latent_columns(z: torch.Tensor, dims: Sequence[int]) -> torch.Tensor:
    """Return a copy of ``z`` with the given columns replaced by their node-wise means."""
    if z.ndim != 2:
        raise ValueError(f"Expected z to have shape (num_nodes, dim); got {tuple(z.shape)}")
    ablated = z.clone()
    for dim in dims:
        dim_i = int(dim)
        if dim_i < 0 or dim_i >= z.shape[1]:
            raise ValueError(f"dim={dim_i} is out of range for latent size {z.shape[1]}")
        ablated[:, dim_i] = z[:, dim_i].mean()
    return ablated


def subject_ablation_seed(base_seed: int, subject_id: str) -> int:
    """Deterministic per-subject seed derived from the ablation config seed."""
    digest = hashlib.sha256(subject_id.encode("utf-8")).hexdigest()
    subject_offset = int(digest[:8], 16)
    return int((base_seed + subject_offset) % (2**31 - 1))


def _components_to_floats(components: Mapping[str, torch.Tensor]) -> dict[str, float]:
    return {key: float(value.detach().cpu().item()) for key, value in components.items()}


def latent_edge_diagnostics(z: torch.Tensor, edge_index: torch.Tensor) -> dict[str, np.ndarray]:
    """Per-dimension node variance and mean |z_i - z_j| on positive edges."""
    if z.ndim != 2:
        raise ValueError(f"Expected z to have shape (num_nodes, dim); got {tuple(z.shape)}")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(f"Expected edge_index shape (2, E); got {tuple(edge_index.shape)}")

    z_np = z.detach().cpu().numpy().astype(float)
    node_variance = z_np.var(axis=0)

    if edge_index.numel() == 0:
        mean_edge_abs_diff = np.zeros(z_np.shape[1], dtype=float)
    else:
        src = edge_index[0].detach().cpu().numpy().astype(int)
        dst = edge_index[1].detach().cpu().numpy().astype(int)
        mean_edge_abs_diff = np.abs(z_np[src] - z_np[dst]).mean(axis=0)

    return {
        "node_variance": node_variance,
        "mean_edge_abs_diff": mean_edge_abs_diff,
    }


@torch.no_grad()
def reconstruction_components(
    model: torch.nn.Module,
    z: torch.Tensor,
    graph: Data,
    neg_edge_index: torch.Tensor,
    *,
    edge_presence_weight: float,
    node_attr_weight: float,
    edge_attr_weight: float,
    node_loss_type: str,
    edge_attr_loss_type: str,
) -> dict[str, float]:
    """Compute reconstruction loss components for latent ``z`` on one graph."""
    pos_edge_index = graph.edge_index
    components = model.recon_total_loss(
        z=z,
        pos_edge_index=pos_edge_index,
        neg_edge_index=neg_edge_index,
        target_x=getattr(graph, "x", None),
        edge_index_for_attr=pos_edge_index,
        target_edge_attr=getattr(graph, "edge_attr", None),
        edge_presence_weight=edge_presence_weight,
        node_attr_weight=node_attr_weight,
        edge_attr_weight=edge_attr_weight,
        node_loss_type=node_loss_type,
        edge_attr_loss_type=edge_attr_loss_type,
    )
    return _components_to_floats(components)


def importance_from_deltas(deltas: np.ndarray) -> np.ndarray:
    """Convert raw Δloss to non-negative, L1-normalized importance scores.

    For each loss head (column), applies ``max(δ, 0)`` then divides by the sum.
    If a column sums to zero, returns zeros for that column.
    """
    if deltas.ndim != 2:
        raise ValueError(f"Expected deltas shape (D, n_heads); got {deltas.shape}")
    positive = np.maximum(deltas.astype(float), 0.0)
    totals = positive.sum(axis=0, keepdims=True)
    importance = np.zeros_like(positive)
    nonzero = totals > 0
    importance[:, nonzero.ravel()] = positive[:, nonzero.ravel()] / totals[:, nonzero.ravel()]
    return importance


def _sample_negative_edges(
    pos_edge_index: torch.Tensor,
    *,
    num_nodes: int,
    num_neg: int,
    seed: int,
) -> torch.Tensor:
    """Sample negative edges with a temporary RNG seed, then restore global state."""
    cpu_state = torch.random.get_rng_state()
    cuda_states = None
    if torch.cuda.is_available():
        cuda_states = torch.cuda.get_rng_state_all()
    try:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available() and pos_edge_index.is_cuda:
            torch.cuda.manual_seed_all(int(seed))
        return negative_sampling(
            edge_index=pos_edge_index,
            num_nodes=int(num_nodes),
            num_neg_samples=num_neg,
        )
    finally:
        torch.random.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


@torch.no_grad()
def greedy_ablate_subject(
    model: torch.nn.Module,
    z: torch.Tensor,
    graph: Data,
    neg_edge_index: torch.Tensor,
    baseline: Mapping[str, float],
    active_keys: Sequence[str],
    *,
    loss_kwargs: Mapping[str, Any],
) -> dict[str, Any]:
    """Greedy sequential mean-replace ablation over all latent dimensions.

    At each step, selects the remaining dim maximizing incremental ``total_loss``.
    """
    n_dims = int(z.shape[1])
    if "total_loss" not in baseline:
        raise ValueError("greedy ablation requires total_loss in baseline components")

    remaining = set(range(n_dims))
    selected: list[int] = []
    greedy_order = np.full(n_dims, -1, dtype=int)
    incremental_delta = np.zeros((n_dims, len(active_keys)), dtype=float)
    cumulative_delta = np.zeros((n_dims, len(active_keys)), dtype=float)

    current_loss = dict(baseline)
    selected_so_far: list[int] = []

    for step in range(n_dims):
        best_dim: int | None = None
        best_incr_total = float("-inf")
        best_components: dict[str, float] | None = None

        for dim in sorted(remaining):
            z_candidate = ablate_latent_columns(z, selected_so_far + [dim])
            candidate = reconstruction_components(
                model, z_candidate, graph, neg_edge_index, **loss_kwargs
            )
            incr_total = float(candidate["total_loss"]) - float(current_loss["total_loss"])
            if incr_total > best_incr_total:
                best_incr_total = incr_total
                best_dim = dim
                best_components = candidate

        if best_dim is None or best_components is None:
            raise RuntimeError("Greedy ablation failed to select a dimension")

        remaining.remove(best_dim)
        selected_so_far.append(best_dim)
        selected.append(best_dim)
        greedy_order[best_dim] = step

        for key_index, key in enumerate(active_keys):
            incremental_delta[best_dim, key_index] = float(best_components[key]) - float(current_loss[key])
            cumulative_delta[best_dim, key_index] = float(best_components[key]) - float(baseline[key])

        current_loss = best_components

    return {
        "greedy_order": greedy_order,
        "greedy_selected": np.asarray(selected, dtype=int),
        "greedy_incremental_delta": incremental_delta,
        "greedy_cumulative_delta": cumulative_delta,
    }


@torch.no_grad()
def ablate_subject(
    model: torch.nn.Module,
    z: torch.Tensor,
    graph: Data,
    *,
    seed: int,
    edge_presence_weight: float,
    node_attr_weight: float,
    edge_attr_weight: float,
    node_loss_type: str,
    edge_attr_loss_type: str,
    neg_sampling_ratio: float = 1.0,
    run_greedy: bool = True,
) -> dict[str, Any]:
    """Leave-one-out and optional greedy latent ablation for a single subject.

    Negative edges are sampled once and reused for LOO, greedy, and joint ablations.
    """
    if z.ndim != 2:
        raise ValueError(f"Expected z to have shape (num_nodes, dim); got {tuple(z.shape)}")
    if int(z.shape[0]) != int(graph.num_nodes):
        raise ValueError(
            f"Node count mismatch for subject {graph.subject_id}: "
            f"embeddings have {z.shape[0]} nodes but graph has {graph.num_nodes}"
        )

    latent_dim = int(getattr(model.encoder, "hidden_dim", z.shape[1]))
    if int(z.shape[1]) != latent_dim:
        raise ValueError(
            f"Embedding dim {z.shape[1]} does not match model latent dim {latent_dim}"
        )

    pos_edge_index = graph.edge_index
    num_pos = int(pos_edge_index.size(1))
    num_neg = int(num_pos * max(1.0, float(neg_sampling_ratio)))
    neg_edge_index = _sample_negative_edges(
        pos_edge_index,
        num_nodes=int(graph.num_nodes),
        num_neg=num_neg,
        seed=seed,
    )

    loss_kwargs = {
        "edge_presence_weight": edge_presence_weight,
        "node_attr_weight": node_attr_weight,
        "edge_attr_weight": edge_attr_weight,
        "node_loss_type": node_loss_type,
        "edge_attr_loss_type": edge_attr_loss_type,
    }

    baseline = reconstruction_components(model, z, graph, neg_edge_index, **loss_kwargs)
    active_keys = [key for key in LOSS_KEYS if key in baseline]
    diagnostics = latent_edge_diagnostics(z, pos_edge_index)

    n_dims = int(z.shape[1])
    delta_losses = np.zeros((n_dims, len(active_keys)), dtype=float)
    ablated_losses: dict[str, list[float]] = {key: [] for key in active_keys}

    for dim in range(n_dims):
        z_ablated = ablate_latent_column(z, dim)
        ablated = reconstruction_components(model, z_ablated, graph, neg_edge_index, **loss_kwargs)
        for key_index, key in enumerate(active_keys):
            value = float(ablated[key])
            ablated_losses[key].append(value)
            delta_losses[dim, key_index] = value - float(baseline[key])

    importance = importance_from_deltas(delta_losses)
    result: dict[str, Any] = {
        "subject_id": str(graph.subject_id),
        "latent_dim": n_dims,
        "num_nodes": int(graph.num_nodes),
        "num_pos_edges": num_pos,
        "num_neg_edges": num_neg,
        "seed": int(seed),
        "loss_keys": active_keys,
        "baseline_losses": baseline,
        "delta_losses": delta_losses,
        "importance": importance,
        "ablated_losses": {key: np.asarray(values, dtype=float) for key, values in ablated_losses.items()},
        "node_variance": diagnostics["node_variance"],
        "mean_edge_abs_diff": diagnostics["mean_edge_abs_diff"],
        "neg_edge_index": neg_edge_index,
        "z": z,
        "graph": graph,
        "loss_kwargs": loss_kwargs,
        "run_greedy": bool(run_greedy),
    }
    if run_greedy:
        result.update(
            greedy_ablate_subject(
                model,
                z,
                graph,
                neg_edge_index,
                baseline,
                active_keys,
                loss_kwargs=loss_kwargs,
            )
        )
    return result


@torch.no_grad()
def joint_ablate_dims(
    model: torch.nn.Module,
    z: torch.Tensor,
    graph: Data,
    neg_edge_index: torch.Tensor,
    dims: Sequence[int],
    baseline: Mapping[str, float],
    *,
    loss_kwargs: Mapping[str, Any],
) -> dict[str, float]:
    """Mean-replace ``dims`` jointly and return Δloss vs baseline for each active head."""
    z_ablated = ablate_latent_columns(z, dims)
    ablated = reconstruction_components(model, z_ablated, graph, neg_edge_index, **loss_kwargs)
    return {key: float(ablated[key]) - float(baseline[key]) for key in baseline}


def save_subject_ablation(
    output_dir: Path,
    subject_id: str,
    result: Mapping[str, Any],
) -> dict[str, Any]:
    """Write per-subject ablation arrays and summary JSON."""
    subject_dir = output_dir / "subjects" / subject_id
    subject_dir.mkdir(parents=True, exist_ok=True)

    delta_losses = np.asarray(result["delta_losses"], dtype=float)
    importance = np.asarray(result["importance"], dtype=float)
    loss_keys = list(result["loss_keys"])

    np.save(subject_dir / "delta_losses.npy", delta_losses)
    np.save(subject_dir / "importance.npy", importance)
    np.save(subject_dir / "node_variance.npy", np.asarray(result["node_variance"], dtype=float))
    np.save(subject_dir / "mean_edge_abs_diff.npy", np.asarray(result["mean_edge_abs_diff"], dtype=float))
    if "greedy_order" in result:
        np.save(subject_dir / "greedy_order.npy", np.asarray(result["greedy_order"], dtype=int))
        np.save(subject_dir / "greedy_selected.npy", np.asarray(result["greedy_selected"], dtype=int))
        np.save(
            subject_dir / "greedy_incremental_delta.npy",
            np.asarray(result["greedy_incremental_delta"], dtype=float),
        )
        np.save(
            subject_dir / "greedy_cumulative_delta.npy",
            np.asarray(result["greedy_cumulative_delta"], dtype=float),
        )
    with open(subject_dir / "baseline_losses.json", "w", encoding="utf-8") as handle:
        json.dump(result["baseline_losses"], handle, indent=4)
    with open(subject_dir / "loss_keys.json", "w", encoding="utf-8") as handle:
        json.dump(loss_keys, handle, indent=4)

    summary: dict[str, Any] = {
        "subject_id": subject_id,
        "latent_dim": int(result["latent_dim"]),
        "num_nodes": int(result["num_nodes"]),
        "num_pos_edges": int(result["num_pos_edges"]),
        "num_neg_edges": int(result["num_neg_edges"]),
        "seed": int(result["seed"]),
        "loss_keys": loss_keys,
        "baseline_losses": result["baseline_losses"],
        "delta_losses_shape": list(delta_losses.shape),
        "mean_delta_total": float(delta_losses[:, loss_keys.index("total_loss")].mean())
        if "total_loss" in loss_keys
        else None,
    }
    with open(subject_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=4)
    return summary


def aggregate_greedy_ablation(
    subject_results: list[Mapping[str, Any]],
) -> pd.DataFrame:
    """Mean greedy incremental/cumulative Δloss curves aligned by step index."""
    if not subject_results:
        raise ValueError("No subject ablation results to aggregate")

    loss_keys = list(subject_results[0]["loss_keys"])
    n_dims = int(subject_results[0]["latent_dim"])
    if "total_loss" not in loss_keys:
        raise ValueError("total_loss is required to aggregate greedy ablation")
    total_index = loss_keys.index("total_loss")

    rows: list[dict[str, Any]] = []
    for step in range(n_dims):
        incr_vals: list[float] = []
        cumul_vals: list[float] = []
        for result in subject_results:
            selected = np.asarray(result["greedy_selected"], dtype=int)
            dim = int(selected[step])
            incr = np.asarray(result["greedy_incremental_delta"], dtype=float)
            cumul = np.asarray(result["greedy_cumulative_delta"], dtype=float)
            incr_vals.append(float(incr[dim, total_index]))
            cumul_vals.append(float(cumul[dim, total_index]))
        rows.append(
            {
                "step": step,
                "mean_incremental_delta_total_loss": float(np.mean(incr_vals)),
                "median_incremental_delta_total_loss": float(np.median(incr_vals)),
                "mean_cumulative_delta_total_loss": float(np.mean(cumul_vals)),
                "median_cumulative_delta_total_loss": float(np.median(cumul_vals)),
            }
        )
    return pd.DataFrame(rows)


def aggregate_joint_topk_ablation(
    subject_results: list[Mapping[str, Any]],
    *,
    topk_dims: Sequence[int],
) -> dict[str, Any]:
    """Jointly ablate ``topk_dims`` per subject and summarize redundancy vs LOO sum."""
    if not subject_results:
        raise ValueError("No subject ablation results for joint top-K ablation")
    if not topk_dims:
        raise ValueError("topk_dims must be non-empty")

    loss_keys = list(subject_results[0]["loss_keys"])
    if "total_loss" not in loss_keys:
        raise ValueError("total_loss is required for joint top-K ablation")
    total_index = loss_keys.index("total_loss")
    topk_list = [int(d) for d in topk_dims]

    joint_deltas: list[float] = []
    sum_loo_deltas: list[float] = []
    for result in subject_results:
        model_ref, z_ref, graph_ref, neg_ref, loss_kwargs = _joint_inputs_from_result(result)
        joint = joint_ablate_dims(
            model_ref,
            z_ref,
            graph_ref,
            neg_ref,
            topk_list,
            result["baseline_losses"],
            loss_kwargs=loss_kwargs,
        )
        joint_deltas.append(float(joint["total_loss"]))
        loo = np.asarray(result["delta_losses"], dtype=float)
        sum_loo_deltas.append(float(loo[topk_list, total_index].sum()))

    joint_arr = np.asarray(joint_deltas, dtype=float)
    sum_loo_arr = np.asarray(sum_loo_deltas, dtype=float)
    mean_joint = float(joint_arr.mean())
    mean_sum_loo = float(sum_loo_arr.mean())
    redundancy_ratio = float(mean_joint / mean_sum_loo) if mean_sum_loo > 0 else None

    return {
        "topk_dims": topk_list,
        "k": len(topk_list),
        "mean_joint_delta_total_loss": mean_joint,
        "median_joint_delta_total_loss": float(np.median(joint_arr)),
        "mean_sum_loo_delta_total_loss": mean_sum_loo,
        "median_sum_loo_delta_total_loss": float(np.median(sum_loo_arr)),
        "redundancy_ratio": redundancy_ratio,
        "num_subjects": len(subject_results),
    }


def _joint_inputs_from_result(
    result: Mapping[str, Any],
) -> tuple[torch.nn.Module, torch.Tensor, Data, torch.Tensor, Mapping[str, Any]]:
    required = ("model", "z", "graph", "neg_edge_index", "loss_kwargs")
    missing = [key for key in required if key not in result]
    if missing:
        raise ValueError(f"Subject result missing joint-ablation inputs: {missing}")
    return (
        result["model"],
        result["z"],
        result["graph"],
        result["neg_edge_index"],
        result["loss_kwargs"],
    )


def select_loo_importance_topk(
    importance_table: pd.DataFrame,
    *,
    greedy_topk: int,
) -> list[int]:
    """Select top-K dim indices by mean LOO importance on total_loss."""
    if "mean_importance_total_loss" not in importance_table.columns:
        raise ValueError("importance_table must contain mean_importance_total_loss")
    n_dims = len(importance_table)
    k = min(int(greedy_topk), n_dims)
    if k < 1:
        raise ValueError("greedy_topk must be >= 1")
    ranked = importance_table.sort_values("mean_importance_total_loss", ascending=False)
    return [int(d) for d in ranked["dim"].iloc[:k].tolist()]


def aggregate_dimension_importance(
    subject_results: list[Mapping[str, Any]],
    *,
    topk_dims: Sequence[int] | None = None,
) -> pd.DataFrame:
    """Aggregate per-subject Δloss, diagnostics, and greedy ranks into a dim table."""
    if not subject_results:
        raise ValueError("No subject ablation results to aggregate")

    loss_keys = list(subject_results[0]["loss_keys"])
    n_dims = int(subject_results[0]["latent_dim"])
    for result in subject_results:
        if list(result["loss_keys"]) != loss_keys:
            raise ValueError("Inconsistent loss keys across subjects")
        if int(result["latent_dim"]) != n_dims:
            raise ValueError("Inconsistent latent dims across subjects")

    topk_set = {int(d) for d in (topk_dims or [])}
    total_index = loss_keys.index("total_loss") if "total_loss" in loss_keys else None
    has_greedy = all("greedy_order" in result for result in subject_results)

    rows: list[dict[str, Any]] = []
    for dim in range(n_dims):
        row: dict[str, Any] = {"dim": dim}
        for key_index, key in enumerate(loss_keys):
            deltas = np.asarray([result["delta_losses"][dim, key_index] for result in subject_results], dtype=float)
            importances = np.asarray(
                [result["importance"][dim, key_index] for result in subject_results], dtype=float
            )
            row[f"mean_delta_{key}"] = float(deltas.mean())
            row[f"median_delta_{key}"] = float(np.median(deltas))
            row[f"mean_importance_{key}"] = float(importances.mean())
            row[f"median_importance_{key}"] = float(np.median(importances))

        node_vars = np.asarray([result["node_variance"][dim] for result in subject_results], dtype=float)
        edge_diffs = np.asarray([result["mean_edge_abs_diff"][dim] for result in subject_results], dtype=float)
        row["mean_node_variance"] = float(node_vars.mean())
        row["median_node_variance"] = float(np.median(node_vars))
        row["mean_edge_abs_diff"] = float(edge_diffs.mean())
        row["median_edge_abs_diff"] = float(np.median(edge_diffs))

        if has_greedy:
            ranks = np.asarray([result["greedy_order"][dim] for result in subject_results], dtype=float)
            row["mean_greedy_rank"] = float(ranks.mean())
            if total_index is not None:
                incr = np.asarray(
                    [result["greedy_incremental_delta"][dim, total_index] for result in subject_results],
                    dtype=float,
                )
                cumul = np.asarray(
                    [result["greedy_cumulative_delta"][dim, total_index] for result in subject_results],
                    dtype=float,
                )
                row["mean_greedy_incremental_delta_total_loss"] = float(incr.mean())
                row["mean_greedy_cumulative_delta_total_loss"] = float(cumul.mean())

        row["in_loo_importance_topk"] = bool(dim in topk_set)
        rows.append(row)

    return pd.DataFrame(rows)


def recon_weights_from_model_config(model_config: Mapping[str, Any]) -> dict[str, Any]:
    """Extract reconstruction loss settings used by training/evaluation."""
    return {
        "edge_presence_weight": float(model_config["recon_edge_presence_weight"]),
        "node_attr_weight": float(model_config["recon_node_attr_weight"]),
        "edge_attr_weight": float(model_config["recon_edge_attr_weight"]),
        "node_loss_type": str(model_config["recon_node_loss_type"]),
        "edge_attr_loss_type": str(model_config["recon_edge_attr_loss_type"]),
        "neg_sampling_ratio": float(model_config.get("neg_sampling_ratio", 1.0)),
    }
