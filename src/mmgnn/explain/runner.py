"""
Batch orchestration for latent ablations.

Loads the frozen GAE once, then processes subjects sequentially.
"""

import json
from pathlib import Path
from typing import Any, Mapping

import torch
from torch_geometric.data import Data
from tqdm import tqdm

from mmgnn.explain.ablation import (
    ablate_subject,
    aggregate_dimension_importance,
    aggregate_greedy_ablation,
    aggregate_joint_topk_ablation,
    recon_weights_from_model_config,
    save_subject_ablation,
    select_loo_importance_topk,
    subject_ablation_seed,
)
from mmgnn.explain.utils import load_frozen_gae
from mmgnn.inference.embeddings import load_subject_node_embeddings


def run_ablation(
    *,
    graphs: list[Data],
    checkpoint_path: Path,
    validated_model_config: Mapping[str, Any],
    embeddings_dir: Path,
    ablation_config: Mapping[str, Any],
    output_dir: Path,
    device: torch.device | None = None,
) -> tuple[list[dict[str, Any]], str, dict[str, Any]]:
    """Run leave-one-dimension-out ablation for all subjects and write artifacts.

    Returns subject summaries, inferred presence decoder type, and checkpoint metadata.
    """
    if not graphs:
        raise ValueError("No graphs were provided for ablation.")

    resolved_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, inferred_presence_decoder_type, checkpoint_meta = load_frozen_gae(
        checkpoint_path=checkpoint_path,
        model_config=validated_model_config,
        sample_graph=graphs[0],
        device=resolved_device,
    )
    recon_weights = recon_weights_from_model_config(validated_model_config)
    base_seed = int(ablation_config["seed"])
    latent_dim = int(model.encoder.hidden_dim)

    subject_results: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []

    for graph in tqdm(graphs, desc="Ablating latent dimensions"):
        subject_id = str(graph.subject_id)
        node_embeddings = load_subject_node_embeddings(embeddings_dir, subject_id)
        if node_embeddings.shape[1] != latent_dim:
            raise ValueError(
                f"Subject {subject_id}: embedding dim {node_embeddings.shape[1]} "
                f"does not match model latent dim {latent_dim}"
            )
        if node_embeddings.shape[0] != int(graph.num_nodes):
            raise ValueError(
                f"Subject {subject_id}: embeddings have {node_embeddings.shape[0]} nodes "
                f"but graph has {graph.num_nodes}"
            )

        z = torch.as_tensor(node_embeddings, dtype=torch.float32, device=resolved_device)
        graph = graph.to(resolved_device)
        seed = subject_ablation_seed(base_seed, subject_id)
        result = ablate_subject(
            model,
            z,
            graph,
            seed=seed,
            edge_presence_weight=recon_weights["edge_presence_weight"],
            node_attr_weight=recon_weights["node_attr_weight"],
            edge_attr_weight=recon_weights["edge_attr_weight"],
            node_loss_type=recon_weights["node_loss_type"],
            edge_attr_loss_type=recon_weights["edge_attr_loss_type"],
            neg_sampling_ratio=recon_weights["neg_sampling_ratio"],
            run_greedy=bool(ablation_config.get("run_greedy", True)),
        )
        result["model"] = model
        summary = save_subject_ablation(output_dir, subject_id, result)
        subject_results.append(result)
        summaries.append(summary)

    # First pass without top-K flags so we can rank by mean LOO importance.
    preliminary_table = aggregate_dimension_importance(subject_results, topk_dims=None)
    topk_dims = select_loo_importance_topk(
        preliminary_table,
        greedy_topk=int(ablation_config["greedy_topk"]),
    )
    importance_table = aggregate_dimension_importance(subject_results, topk_dims=topk_dims)
    importance_table.to_csv(output_dir / "dimension_importance.csv", index=False)

    if bool(ablation_config.get("run_greedy", True)):
        greedy_table = aggregate_greedy_ablation(subject_results)
        greedy_table.to_csv(output_dir / "greedy_ablation.csv", index=False)

    joint_summary = aggregate_joint_topk_ablation(subject_results, topk_dims=topk_dims)
    with open(output_dir / "joint_topk_ablation.json", "w", encoding="utf-8") as handle:
        json.dump(joint_summary, handle, indent=4)

    return summaries, inferred_presence_decoder_type, checkpoint_meta
