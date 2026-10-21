#!/usr/bin/env python3
"""Train a graph autoencoder on a brain graph dataset."""

import argparse
import coloredlogs
import json
import logging
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

from mmgnn.data import BrainGraphDataset, NormalizationStats
from mmgnn.models.utils import build_encoder, build_model
from mmgnn.training.features import FeatureSelector
from mmgnn.training.splits import DataSplitter
from mmgnn.training.trainer import Trainer
from mmgnn.training.plots import plot_embeddings, plot_training_history
from mmgnn.utils.utils import (
    get_dataset_config_help,
    get_model_config_help,
    validate_dataset_config,
    validate_model_config,
)


LOGGER = logging.getLogger(__name__)


def argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a GAE model on a brain graph dataset.",
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
            "\nNote: for training, this config must include `node_feature_keys` and `edge_feature_keys`."
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
        "--output_dir",
        type=str,
        required=True,
        help="Path to the output directory where checkpoints and plots will be saved.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="If set, will print detailed training progress.",
    )
    return parser


def main() -> None:
    parser = argparser()
    args = parser.parse_args()

    # Set logger
    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)
        coloredlogs.install(level=logging.INFO)

    # Load config files
    with Path(args.dataset_config).open("r", encoding="utf-8") as f:
        dataset_config = json.load(f)
    with Path(args.model_config).open("r", encoding="utf-8") as f:
        model_config = json.load(f)

    # Validate the config parameters
    dataset_kwargs = validate_dataset_config(dataset_config)
    model_config = validate_model_config(model_config)
    node_feature_keys = model_config["node_feature_keys"]
    edge_feature_keys = model_config["edge_feature_keys"]

    # Once validated, save the configs to the output directory for reproducibility
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save the validated configs to the output directory
    with open(output_dir / "dataset_config.json", "w", encoding="utf-8") as f:
        json.dump(dataset_kwargs, f, indent=4)
    with open(output_dir / "model_config.json", "w", encoding="utf-8") as f:
        json.dump(model_config, f, indent=4)

    # Loading the data
    logging.info("Loading the dataset...")
    dataset = BrainGraphDataset(**dataset_kwargs)

    # Feature selection and Normalization
    logging.info("Performing feature selection and normalization...")
    feature_selector = FeatureSelector(
        dataset=dataset,
        node_feature_keys=node_feature_keys,
        edge_feature_keys=edge_feature_keys,
    )
    dataset = feature_selector.subset_dataset()

    data_splitter = DataSplitter(
        dataset=dataset,
        train_subjects=args.train_subjects,
        val_subjects=args.val_subjects,
        test_subjects=args.test_subjects,
    )
    train_indices, val_indices, test_indices = data_splitter.get_split_indices()

    train_data = [dataset[i].clone() for i in train_indices.values()]
    val_data = [dataset[i].clone() for i in val_indices.values()] if val_indices else None
    test_data = [dataset[i].clone() for i in test_indices.values()] if test_indices else None

    if not train_data:
        raise ValueError("Training split is empty after feature selection and split resolution")

    normalizer = None
    if model_config["normalize_data"]:
        normalizer = NormalizationStats().fit(
            train_data,
            normalize_nodes=model_config["normalize_nodes"],
            normalize_edges=model_config["normalize_edges"],
            normalize_edge_weights=model_config["normalize_edge_weights"],
        )
        train_data = [normalizer.transform(data) for data in train_data]
        if val_data is not None:
            val_data = [normalizer.transform(data) for data in val_data]
        if test_data is not None:
            test_data = [normalizer.transform(data) for data in test_data]
        normalizer.save(output_dir / "normalization_stats.json")

    train_loader = DataLoader(train_data, batch_size=model_config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_data, batch_size=model_config["batch_size"], shuffle=False) if val_data is not None else None
    test_loader = DataLoader(test_data, batch_size=model_config["batch_size"], shuffle=False) if test_data is not None else None

    sample_graph = train_data[0]
    num_node_features = sample_graph.x.size(-1)
    sample_edge_attr = getattr(sample_graph, "edge_attr", None)
    num_edge_features = sample_edge_attr.size(-1) if sample_edge_attr is not None else 0

    logging.info("Dataset loaded with %d samples", len(dataset))
    logging.info("Number of node features: %d", num_node_features)
    logging.info("Number of edge features: %d", num_edge_features)

    encoder = build_encoder(
        num_node_features=num_node_features,
        num_edge_features=num_edge_features,
        model_config=model_config,
    )
    resolved_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        encoder=encoder,
        num_node_features=num_node_features,
        num_edge_features=num_edge_features,
        model_config=model_config,
    ).to(resolved_device)

    trainable_params_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info("Model initialized with %d trainable parameters", trainable_params_count)

    # Training the model
    logging.info("Starting training...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader if val_loader is not None else train_loader,
        optimizer=torch.optim.AdamW,
        scheduler=model_config["scheduler"],
        patience=model_config["patience"],
        delta=model_config["delta"],
        learning_rate=model_config["learning_rate"],
        weight_decay=model_config["weight_decay"],
        optimizer_param_groups=model_config["optimizer_param_groups"],
        checkpoint_dir=checkpoint_dir,
        device=resolved_device,
        neg_sampling_ratio=model_config["neg_sampling_ratio"],
        recon_edge_presence_weight=model_config["recon_edge_presence_weight"],
        recon_node_attr_weight=model_config["recon_node_attr_weight"],
        recon_edge_attr_weight=model_config["recon_edge_attr_weight"],
        recon_node_loss_type=model_config["recon_node_loss_type"],
        recon_edge_attr_loss_type=model_config["recon_edge_attr_loss_type"],
    )

    trainer.train(num_epochs=model_config["num_epochs"], verbose=args.verbose)

    logging.info("Training completed. Best epoch: %d with validation loss: %.4f", trainer.best_epoch, trainer.best_val_loss)
    logging.info("Plotting diagnostics plots...")
    plot_training_history(
        trainer.history,
        save_path=output_dir / "training_history.png",
        best_epoch=trainer.best_epoch,
    )

    # Plot embeddings for train, val, and test sets
    train_embeddings, train_targets = trainer.get_embeddings(train_loader)
    plot_embeddings(train_embeddings, train_targets, save_path=output_dir / "train_embeddings.png")
    if val_loader is not None:
        val_embeddings, val_targets = trainer.get_embeddings(val_loader)
        plot_embeddings(val_embeddings, val_targets, save_path=output_dir / "val_embeddings.png")
    if test_loader is not None:
        test_embeddings, test_targets = trainer.get_embeddings(test_loader)
        plot_embeddings(test_embeddings, test_targets, save_path=output_dir / "test_embeddings.png")

    best_checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{trainer.best_epoch}.pt"
    if best_checkpoint_path.exists():
        trainer.load_checkpoint(best_checkpoint_path)
        logging.info("Loaded best checkpoint from %s", best_checkpoint_path)
    else:
        logging.warning("Best checkpoint not found at %s; using the current model state.", best_checkpoint_path)

    if val_loader is not None:
        val_results = trainer.evaluate(val_loader)
        logging.info(
            "Validation reconstruction results: loss=%.4f, edge_presence=%.4f, node_attr=%.4f, edge_attr=%.4f",
            val_results["loss"],
            val_results.get("edge_presence_loss", float("nan")),
            val_results.get("node_attr_loss", float("nan")),
            val_results.get("edge_attr_loss", float("nan")),
        )

    test_results = None
    if test_loader is not None:
        test_results = trainer.evaluate(test_loader)
        logging.info(
            "Test reconstruction results: loss=%.4f, edge_presence=%.4f, node_attr=%.4f, edge_attr=%.4f",
            test_results["loss"],
            test_results.get("edge_presence_loss", float("nan")),
            test_results.get("node_attr_loss", float("nan")),
            test_results.get("edge_attr_loss", float("nan")),
        )


if __name__ == "__main__":
    main()
