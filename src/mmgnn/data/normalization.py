"""
Normalization utilities for brain graph data.
"""

import logging
import json
import torch
from pathlib import Path
from torch_geometric.data import Data


class NormalizationStats:
    """
    Store and manage normalization statistics for brain graph features.
    
    Computes mean and standard deviation from training data and applies
    z-score normalization: (x - mean) / std
    """
    
    def __init__(self, eps: float = 1e-8):
        """
        Initialize normalization statistics.
        
        Args:
            eps: Small constant to avoid division by zero
        """
        self.eps = eps
        
        # Node feature statistics
        self.node_mean: torch.Tensor | None = None
        self.node_std: torch.Tensor | None = None
        
        # Edge attribute statistics  
        self.edge_mean: torch.Tensor | None = None
        self.edge_std: torch.Tensor | None = None
        
        # Edge weight statistics
        self.edge_weight_mean: float | None = None
        self.edge_weight_std: float | None = None
    
    def fit(self, dataset, normalize_nodes: bool = True, 
            normalize_edges: bool = True, normalize_edge_weights: bool = True):
        """
        Compute normalization statistics from a dataset (typically training set).
        
        Args:
            dataset: PyTorch Geometric dataset or list of Data objects
            normalize_nodes: Whether to compute node feature statistics
            normalize_edges: Whether to compute edge attribute statistics
            normalize_edge_weights: Whether to compute edge weight statistics
        """
        # Collect all features across the dataset
        all_node_features = []
        all_edge_attrs = []
        all_edge_weights = []
        
        for data in dataset:
            if normalize_nodes and data.x is not None:
                all_node_features.append(data.x)
            
            if normalize_edges and hasattr(data, 'edge_attr') and data.edge_attr is not None:
                all_edge_attrs.append(data.edge_attr)
            
            if normalize_edge_weights and hasattr(data, 'edge_weight') and data.edge_weight is not None:
                all_edge_weights.append(data.edge_weight)
        
        # Compute node feature statistics
        if normalize_nodes and all_node_features:
            all_node_features = torch.cat(all_node_features, dim=0)  # [total_nodes, num_features]
            self.node_mean = all_node_features.mean(dim=0, keepdim=True)  # [1, num_features]
            self.node_std = all_node_features.std(dim=0, keepdim=True)  # [1, num_features]
            # Avoid division by zero
            self.node_std = torch.clamp(self.node_std, min=self.eps)
        
        # Compute edge attribute statistics
        if normalize_edges and all_edge_attrs:
            all_edge_attrs = torch.cat(all_edge_attrs, dim=0)  # [total_edges, num_edge_features]
            self.edge_mean = all_edge_attrs.mean(dim=0, keepdim=True)  # [1, num_edge_features]
            self.edge_std = all_edge_attrs.std(dim=0, keepdim=True)  # [1, num_edge_features]
            self.edge_std = torch.clamp(self.edge_std, min=self.eps)
        
        # Compute edge weight statistics
        if normalize_edge_weights and all_edge_weights:
            all_edge_weights = torch.cat(all_edge_weights, dim=0)  # [total_edges]
            self.edge_weight_mean = float(all_edge_weights.mean())
            self.edge_weight_std = float(all_edge_weights.std())
            self.edge_weight_std = max(self.edge_weight_std, self.eps)
        
        return self
    
    def transform(self, data: Data) -> Data:
        """
        Apply normalization to a Data object using stored statistics.
        
        Args:
            data: PyTorch Geometric Data object
            
        Returns:
            Data object with normalized features (creates a copy)
        """
        data = data.clone()
        
        # Normalize node features
        if self.node_mean is not None and data.x is not None:
            data.x = (data.x - self.node_mean) / self.node_std
            data.x = torch.nan_to_num(data.x, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Normalize edge attributes
        if self.edge_mean is not None and hasattr(data, 'edge_attr') and data.edge_attr is not None:
            data.edge_attr = (data.edge_attr - self.edge_mean) / self.edge_std
            data.edge_attr = torch.nan_to_num(data.edge_attr, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Normalize edge weights
        if self.edge_weight_mean is not None and hasattr(data, 'edge_weight') and data.edge_weight is not None:
            data.edge_weight = (data.edge_weight - self.edge_weight_mean) / self.edge_weight_std
            data.edge_weight = torch.nan_to_num(data.edge_weight, nan=0.0, posinf=0.0, neginf=0.0)
        
        return data

    def inverse_transform_edge_attr(self, edge_attr: torch.Tensor) -> torch.Tensor:
        """Undo edge-attribute normalization for a tensor shaped [E, F] or [F]."""
        if edge_attr is None:
            return edge_attr

        if self.edge_mean is not None:
            mean = self.edge_mean.to(device=edge_attr.device, dtype=edge_attr.dtype)
            std = self.edge_std.to(device=edge_attr.device, dtype=edge_attr.dtype)
            return edge_attr * std + mean

        return edge_attr

    def inverse_transform_edge_weight(self, edge_weight: torch.Tensor) -> torch.Tensor:
        """Undo edge-weight normalization."""
        if edge_weight is None or self.edge_weight_mean is None:
            return edge_weight
        return edge_weight * self.edge_weight_std + self.edge_weight_mean
    
    def __call__(self, data: Data) -> Data:
        """
        Allow using NormalizationStats as a transform.
        
        Args:
            data: PyTorch Geometric Data object
            
        Returns:
            Normalized Data object
        """
        return self.transform(data)
    
    def save(self, path: str | Path):
        """
        Save normalization statistics to a JSON file.
        
        Args:
            path: Path to save statistics
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        stats_dict = {
            'eps': self.eps,
            'node_mean': self.node_mean.tolist() if self.node_mean is not None else None,
            'node_std': self.node_std.tolist() if self.node_std is not None else None,
            'edge_mean': self.edge_mean.tolist() if self.edge_mean is not None else None,
            'edge_std': self.edge_std.tolist() if self.edge_std is not None else None,
            'edge_weight_mean': self.edge_weight_mean,
            'edge_weight_std': self.edge_weight_std
        }
        
        with open(path, 'w') as f:
            json.dump(stats_dict, f, indent=2)
        
        logging.info("Saved normalization statistics to %s", path)
    
    @classmethod
    def load(cls, path: str | Path) -> 'NormalizationStats':
        """
        Load normalization statistics from a JSON file.
        
        Args:
            path: Path to load statistics from
            
        Returns:
            NormalizationStats object with loaded statistics
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Normalization stats file not found: {path}")
        
        with open(path, 'r') as f:
            stats_dict = json.load(f)
        
        normalizer = cls(eps=stats_dict.get('eps', 1e-8))
        
        if stats_dict['node_mean'] is not None:
            normalizer.node_mean = torch.tensor(stats_dict['node_mean'])
            normalizer.node_std = torch.tensor(stats_dict['node_std'])
        
        if stats_dict['edge_mean'] is not None:
            normalizer.edge_mean = torch.tensor(stats_dict['edge_mean'])
            normalizer.edge_std = torch.tensor(stats_dict['edge_std'])
        
        normalizer.edge_weight_mean = stats_dict['edge_weight_mean']
        normalizer.edge_weight_std = stats_dict['edge_weight_std']
        
        logging.info("Loaded normalization statistics from %s", path)
        return normalizer
    
    def get_summary(self) -> dict[str, any]:
        """
        Get a summary of the normalization statistics.
        
        Returns:
            Dictionary with statistics summary
        """
        summary = {
            'eps': self.eps,
            'node_features': {
                'enabled': self.node_mean is not None,
                'num_features': self.node_mean.shape[-1] if self.node_mean is not None else 0,
                'mean_range': (float(self.node_mean.min()), float(self.node_mean.max())) if self.node_mean is not None else None,
                'std_range': (float(self.node_std.min()), float(self.node_std.max())) if self.node_std is not None else None
            },
            'edge_attributes': {
                'enabled': self.edge_mean is not None,
                'num_features': self.edge_mean.shape[-1] if self.edge_mean is not None else 0,
                'mean_range': (float(self.edge_mean.min()), float(self.edge_mean.max())) if self.edge_mean is not None else None,
                'std_range': (float(self.edge_std.min()), float(self.edge_std.max())) if self.edge_std is not None else None
            },
            'edge_weights': {
                'enabled': self.edge_weight_mean is not None,
                'mean': self.edge_weight_mean,
                'std': self.edge_weight_std
            }
        }
        return summary
    
    def print_summary(self):
        """Print a human-readable summary of normalization statistics."""
        summary = self.get_summary()
        
        logging.debug("\n" + "="*60)
        logging.debug("Normalization Statistics Summary")
        logging.debug("="*60)
        logging.debug("Epsilon (for numerical stability): %s", summary["eps"])

        logging.debug("\n--- Node Features ---")
        if summary["node_features"]["enabled"]:
            logging.debug("  Number of features: %s", summary["node_features"]["num_features"])
            logging.debug(
                "  Mean range: [%.4f, %.4f]",
                summary["node_features"]["mean_range"][0],
                summary["node_features"]["mean_range"][1],
            )
            logging.debug(
                "  Std range: [%.4f, %.4f]",
                summary["node_features"]["std_range"][0],
                summary["node_features"]["std_range"][1],
            )
        else:
            logging.debug("  Not normalized")
        
        logging.debug("\n--- Edge Attributes ---")
        if summary["edge_attributes"]["enabled"]:
            logging.debug("  Number of features: %s", summary["edge_attributes"]["num_features"])
            logging.debug(
                "  Mean range: [%.4f, %.4f]",
                summary["edge_attributes"]["mean_range"][0],
                summary["edge_attributes"]["mean_range"][1],
            )
            logging.debug(
                "  Std range: [%.4f, %.4f]",
                summary["edge_attributes"]["std_range"][0],
                summary["edge_attributes"]["std_range"][1],
            )
        else:
            logging.debug("  Not normalized")
        
        logging.debug("\n--- Edge Weights ---")
        if summary["edge_weights"]["enabled"]:
            logging.debug("  Mean: %.4f", summary["edge_weights"]["mean"])
            logging.debug("  Std: %.4f", summary["edge_weights"]["std"])
        else:
            logging.debug("  Not normalized")
        
        logging.debug("="*60 + "\n")
