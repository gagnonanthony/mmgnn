#!/usr/bin/env python3
"""Evaluate a saved GAE checkpoint on train, validation, and test splits."""

import argparse
import coloredlogs
import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from mmgnn.data import BrainGraphDataset, NormalizationStats
from mmgnn.models.utils import build_encoder, build_model
from mmgnn.training.features import FeatureSelector
from mmgnn.training.plots import (
    plot_connectivity_heatmap,
    plot_connectivity_triptych
)
from mmgnn.training.splits import DataSplitter
from mmgnn.training.trainer import Trainer
from mmgnn.utils.utils import (
    get_dataset_config_help,
    get_evaluation_config_options,
    get_model_config_help,
    infer_gae_presence_decoder_type,
    overlay_checkpoint_architecture,
    resolve_normalization_dir,
    validate_dataset_config,
    validate_evaluation_config,
    validate_model_config,
)


LOGGER = logging.getLogger(__name__)


def argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved GAE checkpoint on train, validation, and test splits.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        required=True,
        help="Path to the dataset configuration JSON file.\n%s" % get_dataset_config_help(),
    )
    parser.add_argument(
        "--model_config",
        type=str,
        required=True,
        help=(
            "Path to the GAE configuration JSON file. "
            "\nNote: for evaluation, this config must include `node_feature_keys` and `edge_feature_keys`."
            "\n%s"
            % get_model_config_help()
        ),
    )
    parser.add_argument(
        "--train_subjects",
        type=str,
        required=True,
        help="Path to the file containing the list of train subjects.",
    )
    parser.add_argument(
        "--val_subjects",
        type=str,
        required=True,
        help="Path to the file containing the list of val subjects.",
    )
    parser.add_argument(
        "--test_subjects",
        type=str,
        required=True,
        help="Path to the file containing the list of test subjects.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the saved GAE checkpoint (.pt file).",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help=(
            "Directory containing normalization_stats.json. "
            "If omitted and normalization is enabled, the directory next to the "
            "checkpoint or its parent is used when that file is present; otherwise "
            "the training run directory (two levels above the checkpoint) is used."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help=(
            "Directory where evaluation metrics and embedding plots are saved. "
            "Defaults to the parent of --checkpoint_dir."
        ),
    )
    parser.add_argument(
        "--save_reconstruction_artifacts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to save reconstruction maps and related visualization artifacts.",
    )
    parser.add_argument(
        "--reconstruction_split",
        type=str,
        default="test",
        help="Split to sample from for reconstruction artifacts: train, val, or test.",
    )
    parser.add_argument(
        "--reconstruction_sample_index",
        type=int,
        default=0,
        help="Sample index to visualize when reconstruction_sample_count is 1.",
    )
    parser.add_argument(
        "--reconstruction_sample_count",
        type=int,
        default=5,
        help="Number of graphs to sample for reconstruction artifact export.",
    )
    parser.add_argument(
        "--reconstruction_sample_seed",
        type=int,
        default=1234,
        help="Random seed used when sampling multiple reconstruction examples.",
    )
    parser.add_argument(
        "--reconstruction_output_subdir",
        type=str,
        default="reconstruction_maps",
        help="Subdirectory under output_dir where reconstruction artifacts are written.",
    )
    parser.add_argument(
        "--reconstruction_presence_threshold",
        type=float,
        default=0.5,
        help="Threshold applied to predicted edge-presence probabilities when masking reconstructions.",
    )
    parser.add_argument(
        "--edge_presence_decoder_type",
        type=str,
        default="auto",
        help="Edge-presence decoder type to expect when loading the checkpoint: auto, inner_product, or mlp.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="If set, will print detailed evaluation progress.",
    )
    return parser


def main() -> None:
    parser = argparser()
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)
        coloredlogs.install(level=logging.INFO)

    with Path(args.dataset_config).open("r", encoding="utf-8") as f:
        dataset_config = json.load(f)
    with Path(args.model_config).open("r", encoding="utf-8") as f:
        raw_model_config = json.load(f)

    dataset_kwargs = validate_dataset_config(dataset_config)
    evaluation_config_keys = set(get_evaluation_config_options())
    model_config = validate_model_config(
        {key: value for key, value in raw_model_config.items() if key not in evaluation_config_keys}
    )
    evaluation_config = validate_evaluation_config(
        {
            "save_reconstruction_artifacts": args.save_reconstruction_artifacts,
            "reconstruction_split": args.reconstruction_split,
            "reconstruction_sample_index": args.reconstruction_sample_index,
            "reconstruction_sample_count": args.reconstruction_sample_count,
            "reconstruction_sample_seed": args.reconstruction_sample_seed,
            "reconstruction_output_subdir": args.reconstruction_output_subdir,
            "reconstruction_presence_threshold": args.reconstruction_presence_threshold,
            "edge_presence_decoder_type": args.edge_presence_decoder_type,
        }
    )

    checkpoint_path = Path(args.checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir is not None else checkpoint_path.parent.parent
    output_dir = Path(args.output_dir) if args.output_dir is not None else checkpoint_dir.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "dataset_config.json", "w", encoding="utf-8") as f:
        json.dump(dataset_kwargs, f, indent=4)
    with open(output_dir / "model_config.json", "w", encoding="utf-8") as f:
        json.dump(model_config, f, indent=4)
    with open(output_dir / "evaluation_config.json", "w", encoding="utf-8") as f:
        json.dump(evaluation_config, f, indent=4)

    logging.info("Loading the dataset...")
    dataset = BrainGraphDataset(**dataset_kwargs)

    resolved_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Evaluating checkpoint %s on device %s", checkpoint_path, resolved_device)

    feature_selector = FeatureSelector(
        dataset=dataset,
        node_feature_keys=model_config["node_feature_keys"],
        edge_feature_keys=model_config.get("edge_feature_keys"),
    )
    dataset = feature_selector.subset_dataset()

    data_splitter = DataSplitter(
        dataset=dataset,
        train_subjects=args.train_subjects,
        val_subjects=args.val_subjects,
        test_subjects=args.test_subjects
    )
    train_indices, val_indices, test_indices = data_splitter.get_split_indices()

    train_data = [dataset[i].clone() for i in train_indices.values()]
    val_data = (
        [dataset[i].clone() for i in val_indices.values()]
        if val_indices is not None and len(val_indices) > 0
        else None
    )
    test_data = (
        [dataset[i].clone() for i in test_indices.values()]
        if test_indices is not None and len(test_indices) > 0
        else None
    )

    normalizer = None
    if model_config["normalize_data"]:
        normalizer_path = checkpoint_dir / "normalization_stats.json"
        normalizer = NormalizationStats.load(normalizer_path)
        train_data = [normalizer.transform(data) for data in train_data]
        if val_data is not None:
            val_data = [normalizer.transform(data) for data in val_data]
        if test_data is not None:
            test_data = [normalizer.transform(data) for data in test_data]

    batch_size = model_config["batch_size"]
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False) if val_data is not None else None
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False) if test_data is not None else None

    inferred_presence_decoder_type = infer_gae_presence_decoder_type(checkpoint_path, device=resolved_device)
    requested_presence_decoder_type = evaluation_config["edge_presence_decoder_type"]
    if (
        requested_presence_decoder_type != "auto"
        and requested_presence_decoder_type != inferred_presence_decoder_type
    ):
        LOGGER.warning(
            "Requested edge_presence_decoder_type=%s but checkpoint appears to use %s; using checkpoint architecture.",
            requested_presence_decoder_type,
            inferred_presence_decoder_type,
        )

    sample_graph = train_data[0] if train_data else test_data[0]
    num_node_features = sample_graph.x.size(-1)
    sample_edge_attr = getattr(sample_graph, "edge_attr", None)
    num_edge_features = sample_edge_attr.size(-1) if sample_edge_attr is not None else 0

    checkpoint_preview = torch.load(checkpoint_path, map_location=resolved_device, weights_only=False)
    gae_model_config = overlay_checkpoint_architecture(
        model_config,
        checkpoint_preview.get("model_config")
        if isinstance(checkpoint_preview.get("model_config"), dict)
        else None,
    )
    gae_model_config["edge_presence_decoder_type"] = inferred_presence_decoder_type

    encoder = build_encoder(
        num_node_features=num_node_features,
        num_edge_features=num_edge_features,
        model_config=gae_model_config,
    )
    model = build_model(
        encoder=encoder,
        num_node_features=num_node_features,
        num_edge_features=num_edge_features,
        model_config=gae_model_config,
    ).to(resolved_device)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader if val_loader is not None else train_loader,
        optimizer=torch.optim.AdamW,
        scheduler=None,
        patience=1,
        delta=0.0,
        learning_rate=model_config["learning_rate"],
        weight_decay=model_config["weight_decay"],
        checkpoint_dir=checkpoint_dir,
        device=resolved_device,
        neg_sampling_ratio=model_config["neg_sampling_ratio"],
        recon_edge_presence_weight=model_config["recon_edge_presence_weight"],
        recon_node_attr_weight=model_config["recon_node_attr_weight"],
        recon_edge_attr_weight=model_config["recon_edge_attr_weight"],
        recon_node_loss_type=model_config["recon_node_loss_type"],
        recon_edge_attr_loss_type=model_config["recon_edge_attr_loss_type"],
    )
    trainer.load_checkpoint(checkpoint_path)

    edge_feature_keys = model_config.get("edge_feature_keys")

    def _resolve_reconstruction_data() -> tuple[str, list[tuple[int, Data]]]:
        split = evaluation_config["reconstruction_split"]
        candidates = {
            "train": train_data,
            "val": val_data,
            "test": test_data,
        }
        if split not in candidates:
            raise ValueError("reconstruction_split must be one of: 'train', 'val', 'test'")

        data_list = candidates[split]
        if data_list is None or len(data_list) == 0:
            for fallback in ("test", "val", "train"):
                data_list = candidates[fallback]
                if data_list is not None and len(data_list) > 0:
                    split = fallback
                    break
        if data_list is None or len(data_list) == 0:
            raise ValueError("No graphs are available for reconstruction visualization")

        sample_count = evaluation_config["reconstruction_sample_count"]
        if sample_count == 1:
            sample_indices = [evaluation_config["reconstruction_sample_index"]]
        else:
            if sample_count > len(data_list):
                raise ValueError(
                    f"reconstruction_sample_count={sample_count} exceeds the number of graphs "
                    f"in split '{split}' ({len(data_list)})"
                )
            rng = np.random.default_rng(evaluation_config["reconstruction_sample_seed"])
            sample_indices = rng.choice(len(data_list), size=sample_count, replace=False).tolist()

        samples: list[tuple[int, Data]] = []
        for sample_index in sample_indices:
            if sample_index < 0 or sample_index >= len(data_list):
                raise IndexError(
                    f"reconstruction_sample_index={sample_index} is out of range for split '{split}' "
                    f"with {len(data_list)} graphs"
                )
            samples.append((sample_index, data_list[sample_index].clone()))
        return split, samples

    def _feature_names_for_sample(sample: Data) -> list[str]:
        if edge_feature_keys is not None:
            return list(edge_feature_keys)
        edge_map = getattr(dataset, "edge_attribute_indices", {})
        if edge_map:
            return [name for name, _ in sorted(edge_map.items(), key=lambda item: item[1])]
        edge_attr = getattr(sample, "edge_attr", None)
        if edge_attr is None:
            return []
        return [f"feature_{idx}" for idx in range(edge_attr.shape[-1])]

    def _inverse_scale_dense_matrices(
        matrices: dict[str, np.ndarray],
        feature_names: list[str],
    ) -> dict[str, np.ndarray]:
        if normalizer is None or normalizer.edge_mean is None or normalizer.edge_std is None:
            return matrices
        inverse: dict[str, np.ndarray] = {}
        for idx, name in enumerate(feature_names):
            if idx >= normalizer.edge_mean.shape[-1]:
                inverse[name] = matrices[name]
                continue
            mean = float(normalizer.edge_mean[0, idx].item())
            std = float(normalizer.edge_std[0, idx].item())
            inverse[name] = matrices[name] * std + mean
        return inverse

    def _dense_edge_support_matrix(sample: Data, symmetrize: bool, include_self_loops: bool) -> np.ndarray:
        edge_index = getattr(sample, "edge_index", None)
        if edge_index is None:
            raise ValueError("data.edge_index is required to build a dense support mask")

        support_values = torch.ones(edge_index.shape[1], dtype=torch.float32, device=edge_index.device)
        support = trainer._dense_from_edge_index(
            edge_index=edge_index,
            values=support_values,
            num_nodes=int(sample.num_nodes),
            symmetrize=symmetrize,
            include_self_loops=include_self_loops,
        )
        return (support > 0).astype(np.float32)

    if evaluation_config["save_reconstruction_artifacts"]:
        recon_split_name, sample_graphs = _resolve_reconstruction_data()
        reconstruction_dir = output_dir / evaluation_config["reconstruction_output_subdir"]
        reconstruction_dir.mkdir(parents=True, exist_ok=True)
        presence_threshold = evaluation_config["reconstruction_presence_threshold"]

        for sample_index, sample_graph in sample_graphs:
            feature_names = _feature_names_for_sample(sample_graph)

            if not feature_names:
                LOGGER.info("Skipping reconstruction artifact export because no edge features are available")
                continue

            if getattr(sample_graph, "edge_attr", None) is None:
                raise ValueError("Selected sample graph does not contain edge_attr for reconstruction comparison")

            presence_prob_matrix = trainer.predict_edge_presence_matrix(
                sample_graph,
                apply_sigmoid=True,
                symmetrize=True,
                include_self_loops=False,
            )
            presence_mask_matrix = (presence_prob_matrix >= presence_threshold).astype(np.float32)

            empirical_matrices = trainer.get_empirical_edge_matrices(
                sample_graph,
                feature_names=feature_names,
                symmetrize=True,
                include_self_loops=False,
            )
            reconstructed_matrices = trainer.reconstruct_edge_matrices(
                sample_graph,
                feature_names=feature_names,
                apply_sigmoid=False,
                symmetrize=True,
                include_self_loops=False,
            )
            empirical_support_matrix = _dense_edge_support_matrix(
                sample_graph,
                symmetrize=True,
                include_self_loops=False,
            )

            empirical_matrices = _inverse_scale_dense_matrices(empirical_matrices, feature_names)
            reconstructed_matrices = _inverse_scale_dense_matrices(reconstructed_matrices, feature_names)
            empirical_matrices = {
                name: matrix * empirical_support_matrix for name, matrix in empirical_matrices.items()
            }
            masked_reconstructed_matrices = {
                name: reconstructed_matrices[name] * presence_mask_matrix for name in feature_names
            }

            subject_name = getattr(sample_graph, "subject_id", f"{recon_split_name}_{sample_index}")

            plot_connectivity_heatmap(
                matrix=presence_prob_matrix,
                title=f"Predicted edge existence probability for {subject_name}",
                save_path=reconstruction_dir / f"{subject_name}_edge_presence_probability.png",
                cmap="magma",
            )

            for metric_name in feature_names:
                plot_connectivity_triptych(
                    empirical_matrix=empirical_matrices[metric_name],
                    reconstructed_matrix=masked_reconstructed_matrices[metric_name],
                    title=(
                        f"{metric_name} connectivity reconstruction masked by predicted existence "
                        f"for {subject_name}"
                    ),
                    save_path=reconstruction_dir / f"{subject_name}_{metric_name}_masked_triptych.png",
                )

        logging.info("Saved reconstruction artifacts to %s", reconstruction_dir)

    train_results = trainer.evaluate(train_loader)
    val_results = trainer.evaluate(val_loader) if val_loader is not None else None
    test_results = trainer.evaluate(test_loader) if test_loader is not None else None

    train_embeddings, _ = trainer.get_embeddings(train_loader)
    val_embeddings = val_targets = None
    if val_loader is not None:
        val_embeddings, val_targets = trainer.get_embeddings(val_loader)
        if args.verbose:
            logging.info("Extracted embeddings for validation set with shape: %s", val_embeddings.shape)
    test_embeddings = test_targets = None
    if test_loader is not None:
        test_embeddings, test_targets = trainer.get_embeddings(test_loader)
        if args.verbose:
            logging.info("Extracted embeddings for test set with shape: %s", test_embeddings.shape)

    logging.info(
        "Train reconstruction: loss=%.4f, edge_presence=%.4f, node_attr=%.4f, edge_attr=%.4f",
        train_results["loss"],
        train_results.get("edge_presence_loss", float("nan")),
        train_results.get("node_attr_loss", float("nan")),
        train_results.get("edge_attr_loss", float("nan")),
    )
    if args.verbose and val_results is not None:
        logging.info(
            "Validation reconstruction: loss=%.4f, edge_presence=%.4f, node_attr=%.4f, edge_attr=%.4f",
            val_results["loss"],
            val_results.get("edge_presence_loss", float("nan")),
            val_results.get("node_attr_loss", float("nan")),
            val_results.get("edge_attr_loss", float("nan")),
        )
    if args.verbose and test_results is not None:
        logging.info(
            "Test reconstruction: loss=%.4f, edge_presence=%.4f, node_attr=%.4f, edge_attr=%.4f",
            test_results["loss"],
            test_results.get("edge_presence_loss", float("nan")),
            test_results.get("node_attr_loss", float("nan")),
            test_results.get("edge_attr_loss", float("nan")),
        )

    metrics = {
        "checkpoint_path": str(checkpoint_path),
        "edge_presence_decoder_type": inferred_presence_decoder_type,
        "train": train_results,
        "val": val_results,
        "test": test_results,
    }
    with open(output_dir / "evaluation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
    logging.info("Saved evaluation metrics to %s", output_dir / "evaluation_metrics.json")

if __name__ == "__main__":
    main()
