"""  Graph preprocessing utilities for brain connectivity data.

Converts connectivity matrices to PyTorch Geometric graphs with:
- Graph topology: Derived by binarizing dMRI matrices (structural connectivity)
- Node features: Anatomical volumes + graph metrics (degree, strength, clustering, etc.)
- Edge weights: Primary dMRI metric values at edge positions
- Edge attributes: All dMRI metrics + FC sampled only at existing edges

Based on cs224w-brain-graphs preprocessing approach.
"""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset
from typing import Dict, List, Optional, Tuple, Union
from tqdm import tqdm


def z_score_normalize(
    features: Union[np.ndarray, torch.Tensor],
    eps: float = 1e-8
) -> Union[np.ndarray, torch.Tensor]:
    """
    Apply z-score normalization (standardization) to features.
    
    Z-score: (x - mean) / std
    Normalizes each feature dimension independently to have mean=0, std=1.
    
    Args:
        features: Feature array/tensor [N, F] where N=nodes/edges, F=features
        eps: Small constant to avoid division by zero
        
    Returns:
        Normalized features with same shape and type as input
    """
    is_torch = isinstance(features, torch.Tensor)
    
    if is_torch:
        mean = features.mean(dim=0, keepdim=True)
        std = features.std(dim=0, keepdim=True)
        normalized = (features - mean) / (std + eps)
    else:
        mean = features.mean(axis=0, keepdims=True)
        std = features.std(axis=0, keepdims=True)
        normalized = (features - mean) / (std + eps)

    # Handle NaN/Inf that might arise from normalization
    if is_torch:
        normalized = torch.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
    
    return normalized


def compute_node_features(
    adjacency: np.ndarray,
    anatomical_features: Optional[np.ndarray] = None,
    threshold: float = 0.0,
    use_bct: bool = False
) -> np.ndarray:
    """
    Compute node features from connectivity matrix and anatomical data.
    
    Features computed (similar to cs224w-brain-graphs):
    1. Degree
    2. Strength (weighted degree)
    3. Clustering coefficient
    4. Eigenvector centrality (if BCT available)
    5. Participation coefficient (if BCT available)
    
    Args:
        adjacency: Connectivity matrix [N, N]
        anatomical_features: Anatomical features (e.g., volumes) [N, F], optional
        threshold: Threshold for binary adjacency (for degree/clustering)
        use_bct: Whether to compute BCT metrics (requires bctpy)
        
    Returns:
        Node features [N, num_features]
    """
    n = adjacency.shape[0]
    abs_adj = np.abs(adjacency).astype(np.float64)
    
    # Threshold for binary metrics
    binary_adj = (abs_adj > threshold).astype(float)
    
    # 1. Degree (binary)
    degree = binary_adj.sum(axis=1).astype(np.float32)
    
    # 2. Strength (weighted)
    strength = adjacency.sum(axis=1).astype(np.float32)
    
    # Initialize placeholders for optional metrics
    clustering = np.zeros(n, dtype=np.float32)
    eigenvec = np.zeros(n, dtype=np.float32)
    participation = np.zeros(n, dtype=np.float32)
    
    # 4-5. BCT metrics (if available)
    if use_bct:
        try:
            import bct  # type: ignore

            # Clustering coefficient
            try:
                clustering = bct.clustering_coef_wu(abs_adj).astype(np.float32)
            except Exception:
                pass

            # Eigenvector centrality
            try:
                eigenvec = bct.eigenvector_centrality_und(abs_adj).flatten().astype(np.float32)
            except Exception:
                pass
            
            # Participation coefficient (requires community detection)
            try:
                community, _ = bct.modularity_und(abs_adj)
                # Silence runtimeWarning for invalid division in participation coefficient when degree=0
                with np.errstate(divide='ignore', invalid='ignore'):
                    participation = bct.participation_coef(abs_adj, community).astype(np.float32)
            except Exception:
                pass
        except ImportError:
            pass
    
    # Stack graph features
    graph_features = np.stack([degree, strength, clustering, eigenvec, participation], axis=1)
    
    # Combine with anatomical features if provided
    if anatomical_features is not None:
        # Ensure anatomical features are 2D
        if anatomical_features.ndim == 1:
            anatomical_features = anatomical_features.reshape(-1, 1)
        features = np.concatenate([anatomical_features, graph_features], axis=1)
    else:
        features = graph_features
    
    # Handle NaN/Inf values
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    
    return features.astype(np.float32)


def matrix_to_edge_index(
    matrix: np.ndarray,
    threshold: float = 0.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert adjacency matrix to edge_index and edge_weight tensors.
    
    Args:
        matrix: Adjacency matrix [N, N]
        threshold: Minimum absolute value for edge inclusion
        
    Returns:
        Tuple of (edge_index [2, E], edge_weight [E])
    """
    rows, cols = np.where(np.abs(matrix) > threshold)
    edge_index = torch.from_numpy(np.stack([rows, cols]).astype(np.int64))
    edge_weight = torch.from_numpy(matrix[rows, cols].astype(np.float32))
    
    return edge_index, edge_weight


def gather_edge_attributes(
    edge_index: torch.Tensor,
    matrices: Dict[str, np.ndarray]
) -> torch.Tensor:
    """
    Gather edge attributes from multiple connectivity matrices.
    
    Args:
        edge_index: Edge indices [2, E]
        matrices: Dictionary mapping feature name to connectivity matrix
        
    Returns:
        Edge attributes [E, num_features]
    """
    src, dst = edge_index[0].numpy(), edge_index[1].numpy()
    
    attr_list = []
    for key in sorted(matrices.keys()):  # Sort for consistent ordering
        matrix = matrices[key]
        attr_list.append(matrix[src, dst])
    
    edge_attr = np.stack(attr_list, axis=1).astype(np.float32)
    edge_attr = torch.tensor(np.nan_to_num(edge_attr, nan=0.0, posinf=0.0, neginf=0.0))
    
    return edge_attr


def apply_top_k_sparsification(
    matrix: np.ndarray,
    top_k: int
) -> np.ndarray:
    """
    Keep only top-K strongest connections per node.
    
    Args:
        matrix: Connectivity matrix [N, N]
        top_k: Number of strongest neighbors to keep per node
        
    Returns:
        Sparsified matrix [N, N]
    """
    n = matrix.shape[0]
    abs_matrix = np.abs(matrix)
    mask = np.zeros_like(matrix, dtype=bool)
    
    for i in range(n):
        row = abs_matrix[i, :].copy()
        row[i] = -np.inf  # Exclude self-loops
        
        if top_k >= n - 1:
            top_idx = np.where(row > 0)[0]
        else:
            top_idx = np.argpartition(row, -top_k)[-top_k:]
            top_idx = top_idx[row[top_idx] > 0]
        
        mask[i, top_idx] = True
    
    # Make symmetric
    mask = mask | mask.T
    result = matrix.copy()
    result[~mask] = 0.0
    
    return result


def apply_density_threshold(
    matrix: np.ndarray,
    density: float
) -> float:
    """
    Compute threshold value to achieve target density.
    
    Args:
        matrix: Connectivity matrix [N, N]
        density: Target density (fraction of edges to keep, 0-1)
        
    Returns:
        Threshold value (float)
    """
    n = matrix.shape[0]
    max_edges = n * (n - 1) // 2  # Exclude diagonal
    upper_tri = np.abs(matrix[np.triu_indices(n, k=1)])
    target_edges = int(max_edges * density)

    if target_edges <= 0:
        return np.inf
    if target_edges >= max_edges:
        return 0.0
    
    # Sort edges in descending order
    sorted_edges = np.sort(upper_tri)[::-1]
    
    return sorted_edges[target_edges] if target_edges < len(sorted_edges) else 0.0


def create_pyg_data(
    dmri_path: str,
    primary_dmri_key: str,
    fc_path: str,
    edge_attributes: Optional[List[str]] = None,
    anatomical_features: Optional[np.ndarray] = None,
    target: Optional[np.ndarray] = None,
    subject_id: Optional[str] = None,
    threshold: float = 0.0,
    density: Optional[float] = None,
    top_k: Optional[int] = None,
    use_bct: bool = False,
    log_transform: bool = False,
    dual_mode: bool = False,
) -> Data:
    """
    Create PyTorch Geometric Data object from brain connectivity data.
    
    Graph topology is derived by binarizing dMRI matrices (structural connectivity).
    All dMRI metrics + FC are sampled at edge positions as edge attributes.
    
    Args:
        dmri_matrices: Dictionary of dMRI metrics {name: matrix [N, N]}
        primary_dmri_key: Which dMRI metric to use for graph topology and edge weights
        edge_attributes: List of dMRI metric names to include as edge attributes (optional)
        functional_connectivity: FC matrix [N, N] (sampled at edge positions only)
        anatomical_features: Anatomical features [N, F] (e.g., volumes)
        target: Target values (e.g., behavioral scores)
        subject_id: Subject identifier
        threshold: Edge weight threshold for graph topology
        density: Target graph density (overrides threshold)
        top_k: Keep top-K edges per node (overrides density and threshold)
        use_bct: Whether to compute BCT metrics for node features
        log_transform: Whether to apply log transform to edge weights (optional)
        
    Returns:
        PyTorch Geometric Data object
    """
    # Load subject-specific matrices
    dmri_matrices = {}
    expected_shape = None
    for key in edge_attributes:
        mat = np.load(f"{dmri_path}/{subject_id}_seg-BrainnetomeChild_stat-{key}.npy")

        if expected_shape is None:
            expected_shape = mat.shape
        if mat.shape != expected_shape:
            raise ValueError(f"Shape mismatch for dMRI matrix '{key}' (expected {expected_shape}, got {mat.shape})")
        dmri_matrices[key] = mat

    # Get primary dMRI matrix for defining graph structure
    if primary_dmri_key not in dmri_matrices:
        raise ValueError(f"Primary dMRI key '{primary_dmri_key}' not found in dmri_matrices")

    # Get primary matrix for the current subject
    primary_matrix = dmri_matrices[primary_dmri_key]
    num_nodes = primary_matrix.shape[0]
    
    # Step 2: Optionally apply additional sparsification (top-k or density)
    if top_k is not None and top_k > 0:
        primary_matrix = apply_top_k_sparsification(primary_matrix, top_k)
        effective_threshold = 0.0
    elif density is not None and 0 < density < 1:
        effective_threshold = apply_density_threshold(primary_matrix, density)
        primary_matrix[np.abs(primary_matrix) <= effective_threshold] = 0.0
        effective_threshold = 0.0
    else:
        effective_threshold = threshold

    # Log transform edge weights (optional)
    if log_transform:
        nonzero = primary_matrix != 0
        signs = np.sign(primary_matrix)
        primary_matrix = np.where(nonzero, np.log1p(np.abs(primary_matrix)) * signs, 0.0)

    # Extract edges
    edge_index, edge_weight = matrix_to_edge_index(primary_matrix, threshold=effective_threshold)
    edge_weight = torch.nan_to_num(edge_weight, nan=0.0, posinf=0.0, neginf=0.0)

    # Gather edge attributes: all dMRI metrics + FC sampled at edge positions
    sc_attr_dict = {key: matrix for key, matrix in dmri_matrices.items()}
    edge_attr_sc = gather_edge_attributes(edge_index, sc_attr_dict) if sc_attr_dict else None

    edge_attr_fc = None
    if fc_path is not None:
        functional_connectivity = np.load(f"{fc_path}/{subject_id}_seg-BrainnetomeChild_stat-pearsoncorrelation_relmat.npy")
        edge_attr_fc = gather_edge_attributes(edge_index, {"fc": functional_connectivity})

    combined_edge_attrs = [attr for attr in [edge_attr_sc, edge_attr_fc] if attr is not None]
    edge_attr = torch.cat(combined_edge_attrs, dim=1) if combined_edge_attrs else None

    # Compute node features from the graph structure
    thresholded = primary_matrix.copy()
    thresholded[np.abs(thresholded) <= effective_threshold] = 0.0

    x = compute_node_features(
        thresholded,
        anatomical_features=anatomical_features,
        threshold=0.0,  # Already thresholded above
        use_bct=use_bct
    )
    x = torch.tensor(x, dtype=torch.float32)

    # Set target variable if provided
    if target is not None:
        if isinstance(target, np.ndarray):
            y = torch.tensor(target, dtype=torch.float32).reshape(1, -1)
        else:
            y = torch.tensor([target], dtype=torch.float32).reshape(1, -1)

    # Create Data object
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_weight=edge_weight,
        num_nodes=num_nodes,
        y=y if target is not None else None
    )

    if edge_attr is not None:
        data.edge_attr = edge_attr

    if dual_mode:
        data.edge_index_sc = edge_index
        data.edge_weight_sc = edge_weight
        data.edge_index_fc = edge_index
        if edge_attr_sc is not None:
            data.edge_attr_sc = edge_attr_sc
        if edge_attr_fc is not None:
            data.edge_attr_fc = edge_attr_fc

    if subject_id is not None:
        data.subject_id = subject_id

    return data


class BrainGraphDataset(InMemoryDataset):
    """
    In-memory dataset for biologically constrained brain graphs.
    Adapted from here: https://github.com/cl-tang/cs224w-brain-graphs/blob/main/UKB/data/dataset.py
    """

    def __init__(
        self,
        subjects_file=None,
        dmri_folder=None,
        fc_folder=None,
        anatomical_df=None,
        targets_df=None,
        primary_dmri_key=None,
        edge_attributes=None,
        threshold=None,
        density=None,
        top_k=None,
        use_bct=False,
        log_transform=False,
        split_name=None,
        target_columns=None,
        cache_dir="cache",
        transform=None,
        pre_transform=None,
        pre_filter=None,
        force_reload=False,
        num_workers=4,
        dual_mode: bool = False,
    ):
        if edge_attributes is None:
            edge_attributes = []

        self.subjects_file = Path(subjects_file)
        self.dmri_folder = Path(dmri_folder)
        self.fc_folder = Path(fc_folder)
        self.anatomical_df = Path(anatomical_df) if anatomical_df else None
        self.targets_df = Path(targets_df) if targets_df else None
        self.primary_dmri_key = primary_dmri_key
        self.edge_attributes = edge_attributes
        self.threshold = threshold
        self.density = density
        self.top_k = top_k
        self.use_bct = use_bct
        self.log_transform = log_transform
        self.split_name = split_name
        self.target_columns = target_columns
        self.cache_dir = cache_dir
        self.transform = transform
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter
        self._force_reload = force_reload
        self.num_workers = num_workers
        self.dual_mode = dual_mode

        # Load subject, dmri mats, fc mats, anatomical features, and targets into memory
        self._subject_ids = self._load_subject_ids()
        self._anatomical_features = self._load_anatomical_features() if self.anatomical_df else None
        self._targets = self._load_targets() if self.targets_df else None

        root = Path(self.cache_dir) / self._get_cache_name()

        super().__init__(str(root), transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])

    def _load_subject_ids(self) -> List[str]:
        if not self.subjects_file.exists():
            raise FileNotFoundError(f"Subjects file not found: {self.subjects_file}")
        
        with open(self.subjects_file, 'r') as f:
            subject_ids = [line.strip() for line in f if line.strip()]
        
        return subject_ids

    def _load_anatomical_features(self) -> Dict[str, np.ndarray]:
        # Implement loading of anatomical features for all subjects
        if not self.anatomical_df.exists():
            raise FileNotFoundError(f"Anatomical features file not found: {self.anatomical_df}")

        df = pd.read_csv(self.anatomical_df)

        # Small filtering to ensure we only keep subjects in our subject list
        return df[df['sid'].isin(self._subject_ids)]

    def _load_targets(self) -> Dict[str, np.ndarray]:
        # Implement loading of target values for all subjects
        if not self.targets_df.exists():
            raise FileNotFoundError(f"Targets file not found: {self.targets_df}")

        df = pd.read_excel(self.targets_df)
        df['subjectkey'] = df["subjectkey"].apply(
            lambda x: f"sub-{x.replace('_', '')}" if isinstance(x, str) else f"sub-{x}"
        )

        # Small filtering to ensure we only keep subjects in our subject list
        return df[df['subjectkey'].isin(self._subject_ids)]

    def _get_cache_name(self) -> str:
        atlas = "brainnetome"
        parts = [atlas, self.primary_dmri_key]

        if self.top_k is not None and self.top_k > 0:
            parts.append(f"topk{self.top_k}")
        elif self.density is not None and 0 < self.density < 1:
            parts.append(f"density{self.density:.2f}")
        else:
            parts.append(f"thr{self.threshold:.2f}")

        if self.log_transform:
            parts.append("log")
        if self.edge_attributes:
            parts.append(f"attrs_{'_'.join(self.edge_attributes)}")
        if self.dual_mode:
            parts.append("dual")
        
        # Add split identifier to distinguish train/val/test caches
        if self.split_name:
            parts.append(self.split_name)
        
        # Add target identifier if specific targets selected
        if self.target_columns:
            if isinstance(self.target_columns, list):
                parts.append(f"target_{'_'.join(map(str, self.target_columns))}")
            else:
                parts.append(f"target_{self.target_columns}")

        return "_".join(parts)

    @property
    def processed_file_names(self) -> List[str]:
        return ["data.pt"]

    # Attribute mapping feature names to indices for easier access in transforms/models
    @property
    def node_feature_indices(self) -> Dict[str, int]:
        indices = {}
        idx = 0

        # Only volume is fetched from anatomical features, so we can hardcode it here if anatomical_df is provided
        if self.anatomical_df is not None:
            feature_name = "volume"
            indices[feature_name] = idx

        # Graph features are always in the same order: degree, strength, clustering, eigenvec, participation
        graph_feature_names = ['degree', 'strength', 'clustering', 'eigenvector_centrality', 'participation_coefficient']
        for name in graph_feature_names:
            indices[name] = idx
            idx += 1

        return indices

    @property
    def edge_attribute_indices(self) -> Dict[str, int]:
        indices = {}
        idx = 0

        for key in sorted(self.edge_attributes):
            indices[key] = idx
            idx += 1

        # FC attributes are added after dMRI attributes
        if self.fc_folder is not None:
            indices['fc'] = idx

        return indices

    @property
    def target_indices(self) -> Dict[str, int]:
        indices = {}
        if self.targets_df is not None:
            if self.target_columns is None:
                selected_labels = list(self._targets.columns[-4:])
            else:
                raw_cols = self.target_columns if isinstance(self.target_columns, list) else [self.target_columns]
                selected_labels = []

                for col in raw_cols:
                    if isinstance(col, int):
                        selected_labels.append(self._targets.columns[col])
                    elif isinstance(col, str):
                        selected_labels.append(col)
                    else:
                        raise TypeError(
                            f"Unsupported target column type: {type(col).__name__}. "
                            "target_columns entries must be int or str."
                        )

            # Map to local positions inside data.y (not dataframe absolute positions).
            for local_idx, label in enumerate(selected_labels):
                indices[label] = local_idx

        return indices

    def process(self):
        print(f"Processing {len(self._subject_ids)} brain graphs...")

        def load_single(args):
            sid, dmri_path, fc_path = args
            try:
                # Extract anatomical features for this subject if available
                anat_feats = None
                if self._anatomical_features is not None:
                    anat_row = self._anatomical_features[self._anatomical_features['sid'] == sid]
                    if not anat_row.empty:
                        # Exclude the 'sid' column and get numeric features
                        anat_feats = anat_row.drop(columns=['sid']).values.flatten()
                
                # Extract targets for this subject if available
                target_vals = None
                if self._targets is not None:
                    target_row = self._targets[self._targets['subjectkey'] == sid]
                    if not target_row.empty:
                        # Select target columns
                        if self.target_columns is not None:
                            if isinstance(self.target_columns, (int, str)):
                                # Single target
                                target_vals = target_row.iloc[:, [self.target_columns]].values.flatten()
                            else:
                                # Multiple targets (list of indices or column names)
                                target_vals = target_row.iloc[:, self.target_columns].values.flatten()
                        else:
                            # Default: get last 4 columns (assuming they are the targets)
                            target_vals = target_row.iloc[:, -4:].values.flatten()
                
                data = create_pyg_data(
                    dmri_path=str(dmri_path),
                    primary_dmri_key=self.primary_dmri_key,
                    fc_path=str(fc_path),
                    edge_attributes=self.edge_attributes,
                    anatomical_features=anat_feats,
                    target=target_vals,
                    subject_id=sid,
                    threshold=self.threshold,
                    density=self.density,
                    top_k=self.top_k,
                    use_bct=self.use_bct,
                    log_transform=self.log_transform,
                    dual_mode=self.dual_mode,
                )
                return (sid, data, None)
            except Exception as e:
                return (sid, None, str(e))

        data_list = []
        errors = []
        print(f"Using {self.num_workers} workers for parallel processing")

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {executor.submit(load_single, args): args[0] for args in [(sid, self.dmri_folder, self.fc_folder) for sid in self._subject_ids]}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing brain graphs"):
                sid, data, error = future.result()
                if error:
                    errors.append(f"Subject {sid}: {error}")
                else:
                    data_list.append(data) 

        if errors:
            print(f"Encountered {len(errors)} errors during processing:")
            for err in errors[:5]:
                print(err)

        if not data_list:
            raise ValueError("No valid brain graphs were processed successfully.")

        data_list.sort(key=lambda d: d.subject_id)  # Sort by subject ID for consistency

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]

        self.save(data_list, self.processed_paths[0])
        print(f"Saved processed dataset with {len(data_list)} graphs to {self.processed_paths[0]}")


def prepare_brain_graph_dataset(
    subject_list: List[str],
    dmri_matrices: Dict[str, Dict[str, np.ndarray]],
    primary_dmri_key: str,
    fc_matrices: Optional[Dict[str, np.ndarray]] = None,
    anatomical_features: Optional[Dict[str, np.ndarray]] = None,
    targets: Optional[Dict[str, np.ndarray]] = None,
    threshold: float = 0.0,
    density: Optional[float] = None,
    top_k: Optional[int] = None,
    use_bct: bool = False,
    normalize: bool = True
) -> List[Data]:
    """
    Prepare complete dataset of brain graphs from connectivity data.
    
    Graph topology derived from binarized dMRI (structural connectivity).
    FC and all dMRI metrics sampled as edge attributes at edge positions.
    
    Args:
        subject_list: List of subject IDs
        dmri_matrices: dMRI features {subject_id: {metric: matrix}}
        primary_dmri_key: Which dMRI metric defines graph structure (e.g., 'afd_fixel')
        fc_matrices: Functional connectivity {subject_id: matrix}
        anatomical_features: Anatomical features {subject_id: features}
        targets: Target values {subject_id: target}
        threshold: Edge weight threshold for graph topology
        density: Target graph density
        top_k: Keep top-K edges per node
        use_bct: Whether to compute BCT metrics
        normalize: Whether to apply z-score normalization to features
        
    Returns:
        List of PyTorch Geometric Data objects
    """
    from tqdm import tqdm
    
    dataset = []
    
    for subject_id in tqdm(subject_list, desc="Creating brain graphs"):
        try:
            # Get dMRI matrices (required)
            dmri_mats = dmri_matrices.get(subject_id)
            if dmri_mats is None or primary_dmri_key not in dmri_mats:
                continue
            
            # Get functional connectivity (optional)
            fc = fc_matrices.get(subject_id) if fc_matrices else None
            
            # Get anatomical features (optional)
            anat_feats = anatomical_features.get(subject_id) if anatomical_features else None
            
            # Get target (optional)
            target = targets.get(subject_id) if targets else None
            
            # Create graph
            data = create_pyg_data(
                dmri_matrices=dmri_mats,
                primary_dmri_key=primary_dmri_key,
                functional_connectivity=fc,
                anatomical_features=anat_feats,
                target=target,
                subject_id=subject_id,
                threshold=threshold,
                density=density,
                top_k=top_k,
                use_bct=use_bct,
                normalize=normalize
            )
            
            dataset.append(data)
            
        except Exception as e:
            print(f"Error processing subject {subject_id}: {e}")
            continue
    
    print(f"Created dataset with {len(dataset)} brain graphs")
    return dataset


def save_dataset(
    dataset: List[Data],
    save_path: str,
    metadata: Optional[Dict] = None
) -> None:
    """
    Save PyTorch Geometric dataset to disk for faster loading.
    
    Args:
        dataset: List of PyG Data objects
        save_path: Path to save file (.pt extension recommended)
        metadata: Optional metadata dictionary (preprocessing parameters, etc.)
    """
    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    
    save_dict = {
        'dataset': dataset,
        'num_graphs': len(dataset),
        'metadata': metadata or {}
    }
    
    torch.save(save_dict, save_path)
    print(f"Saved {len(dataset)} graphs to {save_path}")


def load_dataset(
    load_path: str,
    return_metadata: bool = False
) -> List[Data] | Tuple[List[Data], Dict]:
    """
    Load PyTorch Geometric dataset from disk.
    
    Args:
        load_path: Path to saved dataset file
        return_metadata: Whether to also return metadata
        
    Returns:
        List of PyG Data objects, or (dataset, metadata) if return_metadata=True
    """
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Dataset file not found: {load_path}")
    
    save_dict = torch.load(load_path, weights_only=False)
    dataset = save_dict['dataset']
    
    print(f"Loaded {len(dataset)} graphs from {load_path}")
    
    if return_metadata:
        metadata = save_dict.get('metadata', {})
        return dataset, metadata
    else:
        return dataset


def get_cache_filename(
    primary_dmri_key: str,
    threshold: float = 0.0,
    density: Optional[float] = None,
    top_k: Optional[int] = None,
    use_bct: bool = False,
    normalize: bool = True,
    suffix: str = ""
) -> str:
    """
    Generate standardized cache filename based on preprocessing parameters.
    
    Args:
        primary_dmri_key: Primary dMRI metric used for graph topology
        threshold: Threshold value
        density: Density value (if used)
        top_k: Top-K value (if used)
        use_bct: Whether BCT metrics were used
        normalize: Whether features are normalized
        suffix: Optional suffix (e.g., '_train', '_val')
        
    Returns:
        Filename string
    """
    parts = [f"brain_graphs_{primary_dmri_key}"]
    
    if top_k is not None and top_k > 0:
        parts.append(f"topk{top_k}")
    elif density is not None:
        parts.append(f"density{density:.3f}")
    else:
        parts.append(f"thr{threshold:.3f}")
    
    if use_bct:
        parts.append("bct")
    
    if normalize:
        parts.append("norm")
    
    if suffix:
        parts.append(suffix)
    
    return "_".join(parts) + ".pt"


def prepare_or_load_dataset(
    cache_path: str,
    force_recompute: bool = False,
    **prepare_kwargs
) -> List[Data]:
    """
    Load dataset from cache if available, otherwise compute and save.
    
    Args:
        cache_path: Path to cache file
        force_recompute: If True, recompute even if cache exists
        **prepare_kwargs: Arguments for prepare_brain_graph_dataset()
        
    Returns:
        List of PyG Data objects
    """
    if os.path.exists(cache_path) and not force_recompute:
        print(f"Loading dataset from cache: {cache_path}")
        dataset = load_dataset(cache_path)
    else:
        print("Computing dataset from scratch...")
        dataset = prepare_brain_graph_dataset(**prepare_kwargs)
        
        # Save metadata for reproducibility
        metadata = {
            'primary_dmri_key': prepare_kwargs.get('primary_dmri_key'),
            'threshold': prepare_kwargs.get('threshold', 0.0),
            'density': prepare_kwargs.get('density'),
            'top_k': prepare_kwargs.get('top_k'),
            'use_bct': prepare_kwargs.get('use_bct', False),
            'normalize': prepare_kwargs.get('normalize', True),
            'num_subjects': len(prepare_kwargs.get('subject_list', []))
        }
        
        save_dataset(dataset, cache_path, metadata=metadata)
    
    return dataset
