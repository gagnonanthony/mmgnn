"""
Contains all steps to train a model, including the main training loop, validation, and checkpointing.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch_geometric.loader import DataLoader

from .trainer import Trainer
from .plots import (
    plot_training_history,
    plot_predictions,
    plot_classification_probabilities,
    save_model_graph_torchlens,
    plot_embeddings,
    plot_contingency_heatmap
)
from .features import FeatureSelector, ILRTransform
from ..data.normalization import NormalizationStats
from .splits import DataSplitter
from ..models.mmgnn import MMGNN, initialize_model_weights
from ..models.backbone import GNN
from ..models import GAE, Encoder
from .losses import LossSelector
from .sampler import BalancedSampler


def _normalize_finetune_mode(finetune_mode: str) -> str:
    """Validate and normalize finetuning mode configuration."""
    mode = finetune_mode.lower().strip()
    if mode not in {"full", "head_only", "head_slow_backbone"}:
        raise ValueError(
            "finetune_mode must be one of: 'full', 'head_only', 'head_slow_backbone'"
        )
    return mode


def _load_pretrained_weights(
    model: torch.nn.Module,
    weights_path: Path,
    device: torch.device,
    logger: logging.Logger,
    reuse_pretrained_head: bool,
) -> None:
    """Load compatible pretrained weights, optionally excluding prediction head parameters."""
    if not weights_path.exists():
        raise FileNotFoundError(f"Pretrained weights file not found: {weights_path}")

    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state_dict, dict):
        raise ValueError("Pretrained checkpoint must be a state-dict or contain 'model_state_dict'")

    current_state = model.state_dict()
    filtered_state: dict[str, torch.Tensor] = {}
    skipped_incompatible = []
    skipped_head = []

    for key, value in state_dict.items():
        if not reuse_pretrained_head and key.startswith("prediction_head"):
            skipped_head.append(key)
            continue
        if key in current_state and current_state[key].shape == value.shape:
            filtered_state[key] = value
        else:
            skipped_incompatible.append(key)

    if not filtered_state:
        raise ValueError(
            "No compatible parameters were found in pretrained weights. "
            "Check architecture compatibility and checkpoint contents."
        )

    missing_keys, unexpected_keys = model.load_state_dict(filtered_state, strict=False)
    logger.info(
        "Loaded pretrained weights from %s: loaded=%d, skipped_incompatible=%d, skipped_head=%d, missing_after_load=%d, unexpected=%d",
        weights_path,
        len(filtered_state),
        len(skipped_incompatible),
        len(skipped_head),
        len(missing_keys),
        len(unexpected_keys),
    )


def _build_finetune_param_groups(
    model: MMGNN,
    finetune_mode: str,
    learning_rate: float,
    backbone_lr_scale: float,
) -> Optional[list[dict[str, Any]]]:
    """Configure trainable parameters and optional optimizer parameter groups for finetuning."""
    mode = _normalize_finetune_mode(finetune_mode)

    if mode == "full":
        for p in model.parameters():
            p.requires_grad = True
        return None

    readout_params = list(model.readout.parameters()) if isinstance(model.readout, torch.nn.Module) else []
    head_params = readout_params + list(model.prediction_head.parameters())
    backbone_params = list(model.backbone.parameters())

    if mode == "head_only":
        for p in backbone_params:
            p.requires_grad = False
        for p in head_params:
            p.requires_grad = True

        trainable_head = [p for p in head_params if p.requires_grad]
        if len(trainable_head) == 0:
            raise ValueError("finetune_mode='head_only' resulted in zero trainable head parameters")

        return [{"params": trainable_head, "lr": learning_rate}]

    if backbone_lr_scale <= 0:
        raise ValueError("backbone_lr_scale must be > 0 when finetune_mode='head_slow_backbone'")

    for p in backbone_params:
        p.requires_grad = True
    for p in head_params:
        p.requires_grad = True

    if len(head_params) == 0:
        raise ValueError("No trainable head parameters found for head_slow_backbone mode")
    if len(backbone_params) == 0:
        raise ValueError("No trainable backbone parameters found for head_slow_backbone mode")

    return [
        {"params": head_params, "lr": learning_rate},
        {"params": backbone_params, "lr": learning_rate * backbone_lr_scale},
    ]


def _normalize_task_config(
    task_type: str,
    classification_target_mode: str,
    loss_function: str,
    ilr_transform: bool,
) -> tuple[str, str, str]:
    """Validate and normalize task-related configuration values."""
    task = task_type.lower().strip()
    mode = classification_target_mode.lower().strip()
    loss = loss_function.lower().strip()
    binary_losses = {
        "binary_cross_entropy",
        "binary_cross_entropy_with_logits",
        "binarycrossentropylogits",
        "bce",
        "focal",
    }

    if task not in {"regression", "classification"}:
        raise ValueError("task_type must be either 'regression' or 'classification'")

    if task == "classification" and mode not in {"argmax", "soft"}:
        raise ValueError("classification_target_mode must be one of: 'argmax', 'soft'")

    if task == "classification" and ilr_transform:
        raise ValueError("ILR transform is only compatible with regression targets")

    if task == "classification" and loss not in {"cross_entropy", "ce", *binary_losses}:
        raise ValueError(
            "Classification mode currently requires 'cross_entropy' (or 'ce') or "
            "'binary_cross_entropy' (or 'bce') loss"
        )

    if task == "regression" and loss in {"cross_entropy", "ce", *binary_losses}:
        raise ValueError("Classification losses are only valid for classification mode")

    return task, mode, loss


def _squeeze_graph_target(y: torch.Tensor) -> torch.Tensor:
    """Convert a graph-level target to a compact shape while preserving class vectors."""
    if y.ndim == 0:
        return y
    if y.ndim == 1:
        return y
    if y.ndim == 2 and y.shape[0] == 1:
        return y.squeeze(0)
    if y.ndim == 2 and y.shape[1] == 1:
        return y.squeeze(-1)
    raise ValueError(
        f"Unsupported graph-level target shape {tuple(y.shape)}. Expected scalar/class-index or [1, C] membership vector."
    )


def _infer_num_classes(
    train_data: list,
    val_data: Optional[list],
    test_data: Optional[list],
) -> int:
    """Infer class cardinality from membership vectors or class-index targets."""
    datasets = [train_data]
    if val_data is not None:
        datasets.append(val_data)
    if test_data is not None:
        datasets.append(test_data)

    vector_dim: Optional[int] = None
    max_class = -1

    for split in datasets:
        for data in split:
            y = _squeeze_graph_target(data.y)

            if y.ndim == 1 and y.numel() > 1:
                current_dim = int(y.numel())
                if vector_dim is None:
                    vector_dim = current_dim
                elif vector_dim != current_dim:
                    raise ValueError(
                        f"Inconsistent class-vector sizes found across samples: {vector_dim} vs {current_dim}"
                    )
                continue

            class_index = int(torch.as_tensor(y).reshape(-1)[0].item())
            max_class = max(max_class, class_index)

    if vector_dim is not None:
        return vector_dim
    if max_class >= 0:
        return max_class + 1
    raise ValueError("Could not infer number of classes from dataset targets")


def _extract_class_targets_tensor(data_list: list, num_classes: int) -> torch.Tensor:
    """Extract train targets as [N] class ids or [N, C] soft labels."""
    rows = []
    for data in data_list:
        y = _squeeze_graph_target(data.y)
        if y.ndim == 1 and y.numel() == num_classes:
            rows.append(y.to(dtype=torch.float32))
        elif y.ndim == 1 and y.numel() == 1:
            rows.append(y.to(dtype=torch.long))
        elif y.ndim == 0:
            rows.append(y.reshape(1).to(dtype=torch.long))
        else:
            raise ValueError(
                f"Unsupported classification target shape {tuple(y.shape)} while extracting class weights"
            )

    first = rows[0]
    if first.dtype == torch.float32:
        stacked = torch.stack([r.to(dtype=torch.float32) for r in rows], dim=0)
        if stacked.shape[1] != num_classes:
            raise ValueError(
                f"Soft label shape mismatch: expected {num_classes} classes, got {stacked.shape[1]}"
            )
        return stacked

    return torch.cat([r.reshape(-1).to(dtype=torch.long) for r in rows], dim=0)


def _compute_class_weights(
    train_data: list,
    num_classes: int,
    mode: str = "balanced",
) -> torch.Tensor:
    """Compute per-class weights for cross-entropy from train split targets."""
    mode_normalized = mode.lower().strip()
    if mode_normalized != "balanced":
        raise ValueError("class_weight_mode must be 'balanced'")

    targets = _extract_class_targets_tensor(train_data, num_classes=num_classes)

    if targets.ndim == 2:
        # Soft labels: use expected class counts.
        class_counts = targets.sum(dim=0)
    else:
        class_counts = torch.bincount(targets, minlength=num_classes).to(dtype=torch.float32)

    if torch.any(class_counts <= 0):
        missing = torch.where(class_counts <= 0)[0].tolist()
        raise ValueError(
            "Cannot compute class weights because at least one class is missing in train split. "
            f"Missing class indices: {missing}"
        )

    total = class_counts.sum()
    weights = total / (num_classes * class_counts)

    # Keep average weight at 1.0 for stable learning-rate behavior.
    weights = weights / weights.mean()
    return weights.to(dtype=torch.float32)


def _convert_membership_targets_to_class_indices(data_list: list) -> list:
    """Convert per-graph membership vectors to hard class-index targets using argmax."""
    converted = []
    for data in data_list:
        updated = data.clone()
        y = _squeeze_graph_target(updated.y)

        if y.ndim == 1 and y.numel() > 1:
            updated.y = torch.argmax(y).to(dtype=torch.long)
        else:
            updated.y = torch.as_tensor(y).reshape(-1)[0].to(dtype=torch.long)

        converted.append(updated)

    return converted


def _convert_binary_targets_to_scalars(data_list: list) -> list:
    """Convert per-graph binary targets to scalar 0/1 labels for BCE training."""
    converted = []
    for data in data_list:
        updated = data.clone()
        y = _squeeze_graph_target(updated.y)

        if y.ndim == 1 and y.numel() > 1:
            if y.numel() != 2:
                raise ValueError(
                    "binary_cross_entropy requires binary targets with exactly two membership values "
                    "or scalar 0/1 labels"
                )
            label = torch.argmax(y).to(dtype=torch.float32)
        else:
            label = torch.as_tensor(y).reshape(-1)[0].to(dtype=torch.float32)

        if float(label.item()) not in {0.0, 1.0}:
            raise ValueError(
                "binary_cross_entropy requires scalar targets in {0, 1} or two-class membership vectors"
            )

        updated.y = label
        converted.append(updated)

    return converted


def _compute_binary_pos_weight(train_data: list, mode: str = "balanced") -> torch.Tensor:
    """Compute the BCE positive-class weight from scalar binary targets."""
    mode_normalized = mode.lower().strip()
    if mode_normalized not in {"balanced", "sqrt_balanced", "log_balanced"}:
        raise ValueError("class_weight_mode must be one of: 'balanced', 'sqrt_balanced', 'log_balanced'")

    labels = []
    for data in train_data:
        y = _squeeze_graph_target(data.y)
        if y.ndim == 1 and y.numel() > 1:
            if y.numel() != 2:
                raise ValueError("binary_cross_entropy class weighting requires binary two-class targets")
            label = int(torch.argmax(y).item())
        else:
            label = int(torch.as_tensor(y).reshape(-1)[0].item())

        if label not in {0, 1}:
            raise ValueError("binary_cross_entropy requires scalar targets in {0, 1}")
        labels.append(label)

    labels_tensor = torch.tensor(labels, dtype=torch.long)
    positive_count = int((labels_tensor == 1).sum().item())
    negative_count = int((labels_tensor == 0).sum().item())

    if positive_count == 0 or negative_count == 0:
        raise ValueError(
            "Cannot compute binary class weights because one class is missing in the train split"
        )

    ratio = negative_count / positive_count
    if mode_normalized == "balanced":
        pos_weight = ratio
    elif mode_normalized == "sqrt_balanced":
        pos_weight = ratio ** 0.5
    else:
        pos_weight = torch.log1p(torch.tensor(ratio, dtype=torch.float32)).item()

    return torch.tensor([pos_weight], dtype=torch.float32)


def run_experiment(
    dataset,
    conv_type: str,
    hidden_dim: int,
    num_layers: int,
    act_func: str,
    readout_type: str,
    norm_type: str,
    dropout: float,
    edge_dropout: float,
    residual: bool,
    train_subjects: str,
    scheduler: str = None,
    num_heads: int = 4,
    jumping_knowledge_mode: str = None,
    val_subjects: str = None,
    test_subjects: str = None,
    task_type: str = None,
    classification_target_mode: str = "argmax",
    class_weighting: bool = False,
    class_weight_mode: str = "balanced",
    node_feature_keys: Optional[list] = None,
    edge_feature_keys: Optional[list] = None,
    target_keys: Optional[list] = None,
    add_virtual_node: bool = False,
    add_node_type_feature: bool = True,
    virtual_edge_strategy: str = "zeros",
    normalize_data: bool = True,
    normalize_nodes: bool = True,
    normalize_edges: bool = True,
    normalize_edge_weights: bool = True,
    ilr_transform: bool = False,
    use_custom_sampler: bool = False,
    loss_function: str = "mae",
    loss_kwargs: Optional[dict] = None,
    sampler_bins: int = 10,
    sampler_tail_fraction: float = 0.2,
    sampler_tail_ratio_in_batch: float = 0.2,
    num_epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    pretrained_weights_path: Optional[Path] = None,
    finetune_mode: str = "full",
    backbone_lr_scale: float = 0.1,
    reuse_pretrained_head: bool = False,
    init_mode: Optional[str] = None,
    init_nonlinearity: str = "relu",
    patience: int = 20,
    delta: float = 0.0,
    checkpoint_dir: Path = Path("./checkpoints"),
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    verbose: bool = True,
    generate_artifacts: bool = True,
    # Model selection: 'mmgnn' (default) or 'gae'
    model_type: str = "mmgnn",
    # GAE-specific training knobs
    neg_sampling_ratio: float = 1.0,
    recon_edge_presence_weight: float = 1.0,
    recon_node_attr_weight: float = 1.0,
    recon_edge_attr_weight: float = 1.0,
    recon_node_loss_type: str = "mse",
    recon_edge_attr_loss_type: str = "mse",
) -> Dict[str, Any]:
    """
    Runs a full training experiment for the MMGNN model, including feature selection, data splitting, training, and validation.
    """
    # Set logger configuration
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    if task_type is not None:
        task_type, classification_target_mode, loss_function = _normalize_task_config(
            task_type=task_type,
            classification_target_mode=classification_target_mode,
            loss_function=loss_function,
            ilr_transform=ilr_transform,
        )

    logger.info("Starting experiment with configuration: %s", {
        "conv_type": conv_type,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "act_func": act_func,
        "norm_type": norm_type,
        "dropout": dropout,
        "edge_dropout": edge_dropout,
        "residual": residual,
        "train_subjects": train_subjects,
        "val_subjects": val_subjects,
        "test_subjects": test_subjects,
        "task_type": task_type,
        "scheduler": scheduler,
        "classification_target_mode": classification_target_mode,
        "class_weighting": class_weighting,
        "class_weight_mode": class_weight_mode,
        "node_feature_keys": node_feature_keys,
        "edge_feature_keys": edge_feature_keys,
        "target_keys": target_keys,
        "add_virtual_node": add_virtual_node,
        "add_node_type_feature": add_node_type_feature,
        "virtual_edge_strategy": virtual_edge_strategy,
        "normalize_data": normalize_data,
        "normalize_nodes": normalize_nodes,
        "normalize_edges": normalize_edges,
        "normalize_edge_weights": normalize_edge_weights,
        "ilr_transform": ilr_transform,
        "loss_function": loss_function,
        "loss_kwargs": loss_kwargs,
        "sampler_bins": sampler_bins,
        "sampler_tail_fraction": sampler_tail_fraction,
        "sampler_tail_ratio_in_batch": sampler_tail_ratio_in_batch,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "pretrained_weights_path": str(pretrained_weights_path) if pretrained_weights_path is not None else None,
        "finetune_mode": finetune_mode,
        "backbone_lr_scale": backbone_lr_scale,
        "reuse_pretrained_head": reuse_pretrained_head,
        "init_mode": init_mode,
        "init_nonlinearity": init_nonlinearity,
        "patience": patience,
        "delta": delta,
        "checkpoint_dir": checkpoint_dir,
        "device": device,
        "generate_artifacts": generate_artifacts,
    })
    logger.info("Start time: %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    is_binary_bce = task_type == "classification" and loss_function in {
        "binary_cross_entropy",
        "binary_cross_entropy_with_logits",
        "binarycrossentropylogits",
        "bce",
        "focal",
    }

    # Step 1: Feature Selection
    feature_selector = FeatureSelector(
        dataset=dataset,
        node_feature_keys=node_feature_keys,
        edge_feature_keys=edge_feature_keys,
        target_keys=target_keys
    )
    dataset = feature_selector.subset_dataset()

    # Step 2: Data Splitting
    data_splitter = DataSplitter(
        dataset=dataset,
        train_subjects=train_subjects,
        val_subjects=val_subjects,
        test_subjects=test_subjects
    )
    train_indices, val_indices, test_indices = data_splitter.get_split_indices()

    # Materialize split datasets first so we can fit/apply normalization on train stats only.
    train_data = [dataset[i].clone() for i in train_indices.values()]
    val_data = [dataset[i].clone() for i in val_indices.values()] if val_indices is not None and len(val_indices) > 0 else None
    test_data = [dataset[i].clone() for i in test_indices.values()] if test_indices is not None and len(test_indices) > 0 else None

    normalizer = None
    if normalize_data:
        normalizer = NormalizationStats().fit(
            train_data,
            normalize_nodes=normalize_nodes,
            normalize_edges=normalize_edges,
            normalize_edge_weights=normalize_edge_weights,
        )

        train_data = [normalizer.transform(data) for data in train_data]
        if val_data is not None:
            val_data = [normalizer.transform(data) for data in val_data]
        if test_data is not None:
            test_data = [normalizer.transform(data) for data in test_data]

        normalizer.save(checkpoint_dir / "normalization_stats.json")
        logger.info("Saved normalization statistics to %s", checkpoint_dir / "normalization_stats.json")

    # Run ILR transform on targets if specified.
    if ilr_transform:
        ilr_transformer = ILRTransform().fit(train_data)
        train_data = [ilr_transformer.transform(data) for data in train_data]
        ilr_transformer.save(checkpoint_dir / "ilr_transform_params.json")
        if val_data is not None:
            val_data = [ilr_transformer.transform(data) for data in val_data]
        if test_data is not None:
            test_data = [ilr_transformer.transform(data) for data in test_data]

        # Save ILR transform parameters for later inverse transformation.
        logger.info("Saved ILR transform parameters to %s", checkpoint_dir / "ilr_transform_params.json")

    if is_binary_bce:
        train_data = _convert_binary_targets_to_scalars(train_data)
        if val_data is not None:
            val_data = _convert_binary_targets_to_scalars(val_data)
        if test_data is not None:
            test_data = _convert_binary_targets_to_scalars(test_data)
    elif task_type == "classification" and classification_target_mode == "argmax":
        train_data = _convert_membership_targets_to_class_indices(train_data)
        if val_data is not None:
            val_data = _convert_membership_targets_to_class_indices(val_data)
        if test_data is not None:
            test_data = _convert_membership_targets_to_class_indices(test_data)

    # Create DataLoaders for training and evaluation.
    # Use balanced oversampling only on the training split.
    if use_custom_sampler:
        logging.info("Using custom BalancedSampler for training data with %d bins, tail_fraction=%.2f, tail_ratio_in_batch=%.2f", sampler_bins, sampler_tail_fraction, sampler_tail_ratio_in_batch)
        train_batch_sampler = BalancedSampler(
            data_source=train_data,
            batch_size=batch_size,
            target_attr="y",
            num_bins=sampler_bins,
            tail_fraction=sampler_tail_fraction,
            tail_ratio_in_batch=sampler_tail_ratio_in_batch,
            seed=42,
        )
        train_loader = DataLoader(train_data, batch_sampler=train_batch_sampler)
    else:
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False) if val_data is not None else None
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False) if test_data is not None else None

    # Step 3: Model Initialization
    sample_graph = train_data[0]
    sample_has_virtual_node = bool(getattr(sample_graph, "has_virtual_node", False))
    if add_virtual_node and not sample_has_virtual_node:
        logger.warning(
            "add_virtual_node=True but dataset samples do not advertise has_virtual_node=True. "
            "Make sure your dataset loader enables virtual-node preprocessing."
        )
    if not add_virtual_node and sample_has_virtual_node:
        logger.info(
            "Dataset contains virtual-node metadata while add_virtual_node=False. "
            "Continuing with dataset-provided tensors."
        )

    num_node_features = sample_graph.x.size(-1)
    sample_edge_attr = getattr(sample_graph, "edge_attr", None)
    num_edge_features = sample_edge_attr.size(-1) if sample_edge_attr is not None else 0
    if is_binary_bce:
        out_dim = 1
    elif task_type == "classification":
        out_dim = _infer_num_classes(train_data=train_data, val_data=val_data, test_data=test_data)
    else:
        out_dim = sample_graph.y.size(-1) if getattr(sample_graph, "y", None) is not None else 1

    # Log basic dataset info
    logger.info("Dataset loaded with %d samples", len(dataset))
    logger.info("Number of node features: %d", num_node_features)
    logger.info("Number of edge features: %d", num_edge_features)
    if task_type == "classification":
        logger.info("Number of classes: %d", out_dim)
    else:
        logger.info("Number of targets: %d", out_dim)

    backbone = GNN(
        num_node_features=num_node_features,
        num_edge_features=num_edge_features,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads if conv_type == "gat" else None,
        conv_type=conv_type,
        act_func=act_func,
        norm_type=norm_type,
        dropout=dropout,
        edge_dropout=edge_dropout,
        residual=residual,
        jumping_knowledge_mode=jumping_knowledge_mode
    )
    logger.info("Backbone architecture:\n%s", backbone)
    if model_type.lower() == "mmgnn":
        model = MMGNN(
            backbone=backbone,
            out_dim=out_dim,
            hidden_dim=hidden_dim,
            readout_type=readout_type,
            num_layers=num_layers,
            activation=act_func,
            dropout=dropout,
        ).to(device)
    elif model_type.lower() == "gae":
        # Build Encoder matching backbone config
        encoder = Encoder(
            num_node_features=num_node_features,
            num_edge_features=num_edge_features,
            hidden_dim=hidden_dim,
            num_heads=num_heads if conv_type == "gat" else None,
            num_layers=num_layers,
            conv_type=conv_type,
            act_func=act_func,
            norm_type=norm_type,
            dropout=dropout,
            edge_dropout=edge_dropout,
            residual=residual,
        )

        edge_decoder_type = "mlp" if num_edge_features and num_edge_features > 0 else "inner_product"
        if edge_decoder_type == "mlp":
            recon_edge_presence_weight = 0.0
            if recon_edge_attr_weight <= 0:
                recon_edge_attr_weight = 1.0
        else:
            if recon_edge_presence_weight <= 0:
                recon_edge_presence_weight = 1.0

        model = GAE(
            encoder=encoder,
            edge_decoder_type=edge_decoder_type,
            edge_out_dim=None,
            decode_node_features=True,
            node_out_dim=num_node_features,
            decoder_hidden_dim=None,
            dropout=dropout,
        ).to(device)
    else:
        raise ValueError("model_type must be one of: 'mmgnn', 'gae'")

    if pretrained_weights_path is not None:
        _load_pretrained_weights(
            model=model,
            weights_path=Path(pretrained_weights_path),
            device=device,
            logger=logger,
            reuse_pretrained_head=reuse_pretrained_head,
        )

    optimizer_param_groups = _build_finetune_param_groups(
        model=model,
        finetune_mode=finetune_mode,
        learning_rate=learning_rate,
        backbone_lr_scale=backbone_lr_scale,
    )
    trainable_params_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Finetuning mode '%s' configured with %d trainable parameters", finetune_mode, trainable_params_count)

    if init_mode is not None:
        if pretrained_weights_path is not None:
            logger.warning(
                "init_mode was provided together with pretrained_weights_path; skipping initialization to preserve loaded pretrained weights"
            )
        else:
            initialize_model_weights(model, mode=init_mode, nonlinearity=init_nonlinearity)
            logger.info("Applied model weight initialization: mode=%s, nonlinearity=%s", init_mode, init_nonlinearity)

    logger.info("Model initialized with %d parameters", sum(p.numel() for p in model.parameters()))
    if generate_artifacts:
        save_model_graph_torchlens(model=model, batch=sample_graph, save_path=checkpoint_dir / "model_graph.png")

    # Step 5: Trainer Initialization
    # If no validation split is provided, train metrics are used as a fallback monitor.
    monitor_loader = val_loader if val_loader is not None else train_loader

    # Select loss function based on the provided string identifier.
    criterion = LossSelector(loss_function, **(loss_kwargs or {}))

    if task_type == "classification" and loss_function in {"cross_entropy", "ce"} and class_weighting:
        prepared_loss_kwargs = dict(loss_kwargs or {})
        if "weight" in prepared_loss_kwargs:
            raise ValueError(
                "loss_kwargs already contains 'weight'. Use either class_weighting=True or provide explicit weight, not both."
            )

        class_weights = _compute_class_weights(
            train_data=train_data,
            num_classes=out_dim,
            mode=class_weight_mode,
        )
        prepared_loss_kwargs["weight"] = class_weights
        logger.info("Using class-weighted cross-entropy with weights: %s", class_weights.tolist())
        criterion = LossSelector(loss_function, **prepared_loss_kwargs)

    if (
        is_binary_bce
        and class_weighting
        and loss_function in {
            "binary_cross_entropy",
            "binary_cross_entropy_with_logits",
            "binarycrossentropylogits",
            "bce",
        }
    ):
        prepared_loss_kwargs = dict(loss_kwargs or {})
        if "pos_weight" in prepared_loss_kwargs or "weight" in prepared_loss_kwargs:
            raise ValueError(
                "loss_kwargs already contains 'pos_weight' or 'weight'. Use either class_weighting=True or explicit weighting kwargs, not both."
            )

        pos_weight = _compute_binary_pos_weight(train_data=train_data, mode=class_weight_mode)
        prepared_loss_kwargs["pos_weight"] = pos_weight
        logger.info("Using class-weighted BCEWithLogitsLoss with pos_weight=%s", pos_weight.tolist())
        criterion = LossSelector(loss_function, **prepared_loss_kwargs)

    if loss_function == "focal":
        if "gamma" not in (loss_kwargs or {}):
            raise ValueError("Focal loss requires 'gamma' parameter in loss_kwargs")
        logger.info("Using focal loss with gamma=%.2f", loss_kwargs["gamma"])
        if "alpha" not in (loss_kwargs or {}):
            raise ValueError("Focal loss requires 'alpha' parameter in loss_kwargs")
        logger.info("Focal loss alpha parameter set to %.2f", loss_kwargs["alpha"])

    trainer = Trainer(
        model=model,
        task_type=task_type,
        train_loader=train_loader,
        val_loader=monitor_loader,
        optimizer=torch.optim.AdamW,
        scheduler=scheduler if scheduler is not None else None,
        criterion=criterion,
        patience=patience,
        delta=delta,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        optimizer_param_groups=optimizer_param_groups,
        checkpoint_dir=checkpoint_dir,
        device=device,
        training_mode=("gae_reconstruction" if model_type.lower() == "gae" else "supervised"),
        neg_sampling_ratio=neg_sampling_ratio,
        recon_edge_presence_weight=recon_edge_presence_weight,
        recon_node_attr_weight=recon_node_attr_weight,
        recon_edge_attr_weight=recon_edge_attr_weight,
        recon_node_loss_type=recon_node_loss_type,
        recon_edge_attr_loss_type=recon_edge_attr_loss_type,
    )

    # Step 6: Run Training
    trainer.train(num_epochs=num_epochs)

    if generate_artifacts:
        # Plot training history
        plot_training_history(trainer.history, save_path=checkpoint_dir / "training_history.png",
                              best_epoch=trainer.best_epoch)

        # Plot prediction versus targets for last epoch's validation results and test results if available.
        val_results_last_epoch = trainer.evaluate(val_loader) if val_loader is not None else None
        if val_results_last_epoch is not None and task_type == "regression":
            plot_predictions(
                val_results_last_epoch["predictions"],
                val_results_last_epoch["targets"],
                save_path=checkpoint_dir / "val_predictions_vs_true_last_epoch.png",
            )
        test_results_last_epoch = trainer.evaluate(test_loader) if test_loader is not None else None
        if test_results_last_epoch is not None and task_type == "regression":
            plot_predictions(
                test_results_last_epoch["predictions"],
                test_results_last_epoch["targets"],
                save_path=checkpoint_dir / "test_predictions_vs_true_last_epoch.png",
            )

        # If classification, plot the contingency heatmap of predicted vs true classes
        if task_type == "classification":
            train_cont_mat = trainer.get_contingency_matrix(train_loader)
            plot_contingency_heatmap(train_cont_mat, save_path=checkpoint_dir / "train_contingency_heatmap.png")
            if val_loader is not None:
                val_cont_mat = trainer.get_contingency_matrix(val_loader)
                plot_contingency_heatmap(val_cont_mat, save_path=checkpoint_dir / "val_contingency_heatmap.png")
            if test_loader is not None:
                test_cont_mat = trainer.get_contingency_matrix(test_loader)
                plot_contingency_heatmap(test_cont_mat, save_path=checkpoint_dir / "test_contingency_heatmap.png")

        # Extract embeddings from the last epoch and plot them.
        train_embeddings, train_targets = trainer.get_embeddings(train_loader)
        plot_embeddings(train_embeddings, train_targets, save_path=checkpoint_dir / "train_embeddings.png")
        if val_loader is not None:
            val_embeddings, val_targets = trainer.get_embeddings(val_loader)
            plot_embeddings(val_embeddings, val_targets, save_path=checkpoint_dir / "val_embeddings.png")
        if test_loader is not None:
            test_embeddings, test_targets = trainer.get_embeddings(test_loader)
            plot_embeddings(test_embeddings, test_targets, save_path=checkpoint_dir / "test_embeddings.png")

    # Save last epoch checkpoint for potential later analysis.
    last_checkpoint = checkpoint_dir / f"checkpoint_epoch_{num_epochs}.pt"
    trainer.save_checkpoint(num_epochs)
    logger.info("Saved last epoch checkpoint to %s", last_checkpoint)

    # Step 7: Evaluate
    best_checkpoint = checkpoint_dir / f"checkpoint_epoch_{trainer.best_epoch}.pt"
    if best_checkpoint.exists():
        trainer.load_checkpoint(best_checkpoint)
    else:
        logger.warning("Best checkpoint not found at %s. Skipping checkpoint reload.", best_checkpoint)

    # Plot predictions vs true values for validation set if available, otherwise use training set.
    val_results = trainer.evaluate(val_loader) if val_loader is not None else None
    if val_results is not None and task_type == "classification":
        logger.info(
            "Validation results: accuracy=%.4f, f1=%.4f",
            val_results["accuracy"],
            val_results["f1"],
        )
    elif val_results is not None and model_type.lower() == "gae":
        logger.info(
            "Validation reconstruction: loss=%.4f, edge_presence=%.4f, node_attr=%.4f, edge_attr=%.4f",
            val_results["loss"],
            val_results.get("edge_presence_loss", float("nan")),
            val_results.get("node_attr_loss", float("nan")),
            val_results.get("edge_attr_loss", float("nan")),
        )
    elif val_results is not None:
        logger.info(
            "Validation results: mae=%.4f, pearson=%.4f",
            val_results["mae"],
            val_results["pearson"],
        )

    if generate_artifacts:
        if val_results is not None and task_type == "regression":
            plot_predictions(val_results["predictions"], val_results["targets"], save_path=checkpoint_dir / "val_predictions_vs_true.png")
        if (
            val_results is not None
            and task_type == "classification"
            and val_results.get("target_memberships") is not None
        ):
            plot_classification_probabilities(
                val_results["probabilities"],
                val_results["target_memberships"],
                save_path=checkpoint_dir / "val_probabilities_vs_memberships.png",
            )

    test_results = trainer.evaluate(test_loader) if test_loader is not None else None
    if test_results is not None and task_type == "classification":
        logger.info(
            "Test results: accuracy=%.4f, f1=%.4f",
            test_results["accuracy"],
            test_results["f1"],
        )
    elif test_results is not None and model_type.lower() == "gae":
        logger.info(
            "Test reconstruction: loss=%.4f, edge_presence=%.4f, node_attr=%.4f, edge_attr=%.4f",
            test_results["loss"],
            test_results.get("edge_presence_loss", float("nan")),
            test_results.get("node_attr_loss", float("nan")),
            test_results.get("edge_attr_loss", float("nan")),
        )
    elif test_results is not None:
        logger.info(
            "Test results: mae=%.4f, pearson=%.4f",
            test_results["mae"],
            test_results["pearson"],
        )

    if generate_artifacts:
        if test_results is not None and task_type == "regression":
            plot_predictions(test_results["predictions"], test_results["targets"], save_path=checkpoint_dir / "test_predictions_vs_true.png")
        if (
            test_results is not None
            and task_type == "classification"
            and test_results.get("target_memberships") is not None
        ):
            plot_classification_probabilities(
                test_results["probabilities"],
                test_results["target_memberships"],
                save_path=checkpoint_dir / "test_probabilities_vs_memberships.png",
            )

    logger.info("End time: %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    return {
        "model": model,
        "trainer": trainer,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "normalizer": normalizer,
        "best_epoch": trainer.best_epoch,
        "best_val_loss": trainer.best_val_loss,
        "best_checkpoint": checkpoint_dir / f"checkpoint_epoch_{trainer.best_epoch}.pt",
        "last_checkpoint": checkpoint_dir / f"checkpoint_epoch_{num_epochs}.pt",
        "history": trainer.history,
        "checkpoint_dir": checkpoint_dir,
        "val_results": val_results,
        "test_results": test_results
    }


def evaluate_saved_gae_checkpoint(
    dataset,
    checkpoint_path: Path,
    conv_type: str,
    hidden_dim: int,
    num_layers: int,
    act_func: str,
    norm_type: str,
    dropout: float,
    edge_dropout: float,
    residual: bool,
    train_subjects: str,
    val_subjects: str = None,
    test_subjects: str = None,
    node_feature_keys: Optional[list] = None,
    edge_feature_keys: Optional[list] = None,
    target_keys: Optional[list] = None,
    add_virtual_node: bool = False,
    add_node_type_feature: bool = True,
    virtual_edge_strategy: str = "zeros",
    normalize_data: bool = True,
    normalize_nodes: bool = True,
    normalize_edges: bool = True,
    normalize_edge_weights: bool = True,
    batch_size: int = 32,
    num_heads: int = 4,
    checkpoint_dir: Path = None,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    neg_sampling_ratio: float = 1.0,
    recon_edge_presence_weight: float = 1.0,
    recon_node_attr_weight: float = 1.0,
    recon_edge_attr_weight: float = 1.0,
    recon_node_loss_type: str = "mse",
    recon_edge_attr_loss_type: str = "mse",
    verbose: bool = True,
) -> Dict[str, Any]:
    """Load a saved GAE checkpoint and evaluate it on the held-out test split without retraining."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    checkpoint_path = Path(checkpoint_path)
    if checkpoint_dir is None:
        checkpoint_dir = checkpoint_path.parent
    else:
        checkpoint_dir = Path(checkpoint_dir)

    feature_selector = FeatureSelector(
        dataset=dataset,
        node_feature_keys=node_feature_keys,
        edge_feature_keys=edge_feature_keys,
        target_keys=target_keys,
    )
    dataset = feature_selector.subset_dataset()

    data_splitter = DataSplitter(
        dataset=dataset,
        train_subjects=train_subjects,
        val_subjects=val_subjects,
        test_subjects=test_subjects,
    )
    train_indices, val_indices, test_indices = data_splitter.get_split_indices()

    train_data = [dataset[i].clone() for i in train_indices.values()]
    val_data = [dataset[i].clone() for i in val_indices.values()] if val_indices is not None and len(val_indices) > 0 else None
    test_data = [dataset[i].clone() for i in test_indices.values()] if test_indices is not None and len(test_indices) > 0 else None

    normalizer = None
    if normalize_data:
        normalizer_path = checkpoint_dir / "normalization_stats.json"
        normalizer = NormalizationStats.load(normalizer_path)
        train_data = [normalizer.transform(data) for data in train_data]
        if val_data is not None:
            val_data = [normalizer.transform(data) for data in val_data]
        if test_data is not None:
            test_data = [normalizer.transform(data) for data in test_data]

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False) if val_data is not None else None
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False) if test_data is not None else None

    sample_graph = train_data[0] if len(train_data) > 0 else test_data[0]
    num_node_features = sample_graph.x.size(-1)
    sample_edge_attr = getattr(sample_graph, "edge_attr", None)
    num_edge_features = sample_edge_attr.size(-1) if sample_edge_attr is not None else 0

    encoder = Encoder(
        num_node_features=num_node_features,
        num_edge_features=num_edge_features,
        hidden_dim=hidden_dim,
        num_heads=num_heads if conv_type == "gat" else None,
        num_layers=num_layers,
        conv_type=conv_type,
        act_func=act_func,
        norm_type=norm_type,
        dropout=dropout,
        edge_dropout=edge_dropout,
        residual=residual,
    )

    edge_decoder_type = "mlp" if num_edge_features and num_edge_features > 0 else "inner_product"
    model = GAE(
        encoder=encoder,
        edge_decoder_type=edge_decoder_type,
        edge_out_dim=None,
        decode_node_features=True,
        node_out_dim=num_node_features,
        decoder_hidden_dim=None,
        dropout=dropout,
    ).to(device)

    trainer = Trainer(
        model=model,
        task_type=None,
        train_loader=train_loader,
        val_loader=val_loader if val_loader is not None else train_loader,
        optimizer=torch.optim.AdamW,
        scheduler=None,
        criterion=LossSelector("mse"),
        patience=1,
        delta=0.0,
        learning_rate=1e-3,
        weight_decay=1e-5,
        checkpoint_dir=checkpoint_dir,
        device=device,
        training_mode="gae_reconstruction",
        neg_sampling_ratio=neg_sampling_ratio,
        recon_edge_presence_weight=recon_edge_presence_weight,
        recon_node_attr_weight=recon_node_attr_weight,
        recon_edge_attr_weight=recon_edge_attr_weight,
        recon_node_loss_type=recon_node_loss_type,
        recon_edge_attr_loss_type=recon_edge_attr_loss_type,
    )

    trainer.load_checkpoint(checkpoint_path)

    train_results = trainer.evaluate(train_loader)
    val_results = trainer.evaluate(val_loader) if val_loader is not None else None
    test_results = trainer.evaluate(test_loader) if test_loader is not None else None

    # Get embeddings for validation set and test set.
    train_embeddings, train_targets = trainer.get_embeddings(train_loader)
    if val_loader is not None:
        val_embeddings, val_targets = trainer.get_embeddings(val_loader)
        if verbose:
            logger.info("Extracted embeddings for validation set with shape: %s", val_embeddings.shape)
    if test_loader is not None:
        test_embeddings, test_targets = trainer.get_embeddings(test_loader)
        if verbose:
            logger.info("Extracted embeddings for test set with shape: %s", test_embeddings.shape)

    logger.info(
        "Train reconstruction: loss=%.4f, edge_presence=%.4f, node_attr=%.4f, edge_attr=%.4f",
        train_results["loss"],
        train_results.get("edge_presence_loss", float("nan")),
        train_results.get("node_attr_loss", float("nan")),
        train_results.get("edge_attr_loss", float("nan")),
    )
    if verbose and val_results is not None:
        logger.info(
            "Validation reconstruction: loss=%.4f, edge_presence=%.4f, node_attr=%.4f, edge_attr=%.4f",
            val_results["loss"],
            val_results.get("edge_presence_loss", float("nan")),
            val_results.get("node_attr_loss", float("nan")),
            val_results.get("edge_attr_loss", float("nan")),
        )
    if verbose and test_results is not None:
        logger.info(
            "Test reconstruction: loss=%.4f, edge_presence=%.4f, node_attr=%.4f, edge_attr=%.4f",
            test_results["loss"],
            test_results.get("edge_presence_loss", float("nan")),
            test_results.get("node_attr_loss", float("nan")),
            test_results.get("edge_attr_loss", float("nan")),
        )

    return {
        "model": model,
        "trainer": trainer,
        "normalizer": normalizer,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "checkpoint_path": checkpoint_path,
        "checkpoint_dir": checkpoint_dir,
        "train_results": train_results,
        "val_results": val_results,
        "test_results": test_results,
        "train_embeddings": train_embeddings,
        "train_targets": train_targets,
        "val_embeddings": val_embeddings if val_loader is not None else None,
        "val_targets": val_targets if val_loader is not None else None,
        "test_embeddings": test_embeddings if test_loader is not None else None,
        "test_targets": test_targets if test_loader is not None else None,
    }