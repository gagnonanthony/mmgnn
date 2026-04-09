"""
Normalization utilities for brain graph data.
"""

import logging
import json
import torch
from pathlib import Path
from typing import Optional, Dict, Union
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
        self.node_mean: Optional[torch.Tensor] = None
        self.node_std: Optional[torch.Tensor] = None
        
        # Edge attribute statistics  
        self.edge_mean: Optional[torch.Tensor] = None
        self.edge_std: Optional[torch.Tensor] = None

        # Dual-branch edge attribute statistics
        self.edge_sc_mean: Optional[torch.Tensor] = None
        self.edge_sc_std: Optional[torch.Tensor] = None
        self.edge_fc_mean: Optional[torch.Tensor] = None
        self.edge_fc_std: Optional[torch.Tensor] = None
        
        # Edge weight statistics
        self.edge_weight_mean: Optional[float] = None
        self.edge_weight_std: Optional[float] = None
    
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
        all_edge_attrs_sc = []
        all_edge_attrs_fc = []
        all_edge_weights = []
        
        for data in dataset:
            if normalize_nodes and data.x is not None:
                all_node_features.append(data.x)
            
            if normalize_edges and hasattr(data, 'edge_attr') and data.edge_attr is not None:
                all_edge_attrs.append(data.edge_attr)

            if normalize_edges and hasattr(data, 'edge_attr_sc') and data.edge_attr_sc is not None:
                all_edge_attrs_sc.append(data.edge_attr_sc)

            if normalize_edges and hasattr(data, 'edge_attr_fc') and data.edge_attr_fc is not None:
                all_edge_attrs_fc.append(data.edge_attr_fc)
            
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

        # Compute dual structural edge attribute statistics
        if normalize_edges and all_edge_attrs_sc:
            all_edge_attrs_sc = torch.cat(all_edge_attrs_sc, dim=0)
            self.edge_sc_mean = all_edge_attrs_sc.mean(dim=0, keepdim=True)
            self.edge_sc_std = all_edge_attrs_sc.std(dim=0, keepdim=True)
            self.edge_sc_std = torch.clamp(self.edge_sc_std, min=self.eps)

        # Compute dual functional edge attribute statistics
        if normalize_edges and all_edge_attrs_fc:
            all_edge_attrs_fc = torch.cat(all_edge_attrs_fc, dim=0)
            self.edge_fc_mean = all_edge_attrs_fc.mean(dim=0, keepdim=True)
            self.edge_fc_std = all_edge_attrs_fc.std(dim=0, keepdim=True)
            self.edge_fc_std = torch.clamp(self.edge_fc_std, min=self.eps)
        
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

        # Normalize dual structural edge attributes
        if self.edge_sc_mean is not None and hasattr(data, 'edge_attr_sc') and data.edge_attr_sc is not None:
            data.edge_attr_sc = (data.edge_attr_sc - self.edge_sc_mean) / self.edge_sc_std
            data.edge_attr_sc = torch.nan_to_num(data.edge_attr_sc, nan=0.0, posinf=0.0, neginf=0.0)

        # Normalize dual functional edge attributes
        if self.edge_fc_mean is not None and hasattr(data, 'edge_attr_fc') and data.edge_attr_fc is not None:
            data.edge_attr_fc = (data.edge_attr_fc - self.edge_fc_mean) / self.edge_fc_std
            data.edge_attr_fc = torch.nan_to_num(data.edge_attr_fc, nan=0.0, posinf=0.0, neginf=0.0)

        # Keep combined edge_attr consistent with dual branch tensors if present.
        if (
            hasattr(data, 'edge_attr_sc') and data.edge_attr_sc is not None
            and hasattr(data, 'edge_attr_fc') and data.edge_attr_fc is not None
        ):
            data.edge_attr = torch.cat([data.edge_attr_sc, data.edge_attr_fc], dim=1)
        
        # Normalize edge weights
        if self.edge_weight_mean is not None and hasattr(data, 'edge_weight') and data.edge_weight is not None:
            data.edge_weight = (data.edge_weight - self.edge_weight_mean) / self.edge_weight_std
            data.edge_weight = torch.nan_to_num(data.edge_weight, nan=0.0, posinf=0.0, neginf=0.0)
        
        return data
    
    def __call__(self, data: Data) -> Data:
        """
        Allow using NormalizationStats as a transform.
        
        Args:
            data: PyTorch Geometric Data object
            
        Returns:
            Normalized Data object
        """
        return self.transform(data)
    
    def save(self, path: Union[str, Path]):
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
            'edge_sc_mean': self.edge_sc_mean.tolist() if self.edge_sc_mean is not None else None,
            'edge_sc_std': self.edge_sc_std.tolist() if self.edge_sc_std is not None else None,
            'edge_fc_mean': self.edge_fc_mean.tolist() if self.edge_fc_mean is not None else None,
            'edge_fc_std': self.edge_fc_std.tolist() if self.edge_fc_std is not None else None,
            'edge_weight_mean': self.edge_weight_mean,
            'edge_weight_std': self.edge_weight_std
        }
        
        with open(path, 'w') as f:
            json.dump(stats_dict, f, indent=2)
        
        print(f"Saved normalization statistics to {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'NormalizationStats':
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

        if stats_dict.get('edge_sc_mean') is not None:
            normalizer.edge_sc_mean = torch.tensor(stats_dict['edge_sc_mean'])
            normalizer.edge_sc_std = torch.tensor(stats_dict['edge_sc_std'])

        if stats_dict.get('edge_fc_mean') is not None:
            normalizer.edge_fc_mean = torch.tensor(stats_dict['edge_fc_mean'])
            normalizer.edge_fc_std = torch.tensor(stats_dict['edge_fc_std'])
        
        normalizer.edge_weight_mean = stats_dict['edge_weight_mean']
        normalizer.edge_weight_std = stats_dict['edge_weight_std']
        
        print(f"Loaded normalization statistics from {path}")
        return normalizer
    
    def get_summary(self) -> Dict[str, any]:
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
            'edge_attributes_sc': {
                'enabled': self.edge_sc_mean is not None,
                'num_features': self.edge_sc_mean.shape[-1] if self.edge_sc_mean is not None else 0,
                'mean_range': (float(self.edge_sc_mean.min()), float(self.edge_sc_mean.max())) if self.edge_sc_mean is not None else None,
                'std_range': (float(self.edge_sc_std.min()), float(self.edge_sc_std.max())) if self.edge_sc_std is not None else None
            },
            'edge_attributes_fc': {
                'enabled': self.edge_fc_mean is not None,
                'num_features': self.edge_fc_mean.shape[-1] if self.edge_fc_mean is not None else 0,
                'mean_range': (float(self.edge_fc_mean.min()), float(self.edge_fc_mean.max())) if self.edge_fc_mean is not None else None,
                'std_range': (float(self.edge_fc_std.min()), float(self.edge_fc_std.max())) if self.edge_fc_std is not None else None
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
        logging.debug(f"Epsilon (for numerical stability): {summary['eps']}")
        
        logging.debug("\n--- Node Features ---")
        if summary['node_features']['enabled']:
            logging.debug(f"  Number of features: {summary['node_features']['num_features']}")
            logging.debug(f"  Mean range: [{summary['node_features']['mean_range'][0]:.4f}, {summary['node_features']['mean_range'][1]:.4f}]")
            logging.debug(f"  Std range: [{summary['node_features']['std_range'][0]:.4f}, {summary['node_features']['std_range'][1]:.4f}]")
        else:
            logging.debug("  Not normalized")
        
        logging.debug("\n--- Edge Attributes ---")
        if summary['edge_attributes']['enabled']:
            logging.debug(f"  Number of features: {summary['edge_attributes']['num_features']}")
            logging.debug(f"  Mean range: [{summary['edge_attributes']['mean_range'][0]:.4f}, {summary['edge_attributes']['mean_range'][1]:.4f}]")
            logging.debug(f"  Std range: [{summary['edge_attributes']['std_range'][0]:.4f}, {summary['edge_attributes']['std_range'][1]:.4f}]")
        else:
            logging.debug("  Not normalized")

        logging.debug("\n--- Edge Attributes (Structural Branch) ---")
        if summary['edge_attributes_sc']['enabled']:
            logging.debug(f"  Number of features: {summary['edge_attributes_sc']['num_features']}")
            logging.debug(f"  Mean range: [{summary['edge_attributes_sc']['mean_range'][0]:.4f}, {summary['edge_attributes_sc']['mean_range'][1]:.4f}]")
            logging.debug(f"  Std range: [{summary['edge_attributes_sc']['std_range'][0]:.4f}, {summary['edge_attributes_sc']['std_range'][1]:.4f}]")
        else:
            logging.debug("  Not normalized")

        logging.debug("\n--- Edge Attributes (Functional Branch) ---")
        if summary['edge_attributes_fc']['enabled']:
            logging.debug(f"  Number of features: {summary['edge_attributes_fc']['num_features']}")
            logging.debug(f"  Mean range: [{summary['edge_attributes_fc']['mean_range'][0]:.4f}, {summary['edge_attributes_fc']['mean_range'][1]:.4f}]")
            logging.debug(f"  Std range: [{summary['edge_attributes_fc']['std_range'][0]:.4f}, {summary['edge_attributes_fc']['std_range'][1]:.4f}]")
        else:
            logging.debug("  Not normalized")
        
        logging.debug("\n--- Edge Weights ---")
        if summary['edge_weights']['enabled']:
            logging.debug(f"  Mean: {summary['edge_weights']['mean']:.4f}")
            logging.debug(f"  Std: {summary['edge_weights']['std']:.4f}")
        else:
            logging.debug("  Not normalized")
        
        logging.debug("="*60 + "\n")
