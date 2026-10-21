"""  Graph preprocessing utilities for brain connectivity data.

Converts connectivity matrices to PyTorch Geometric graphs with:
- Graph topology: Derived by binarizing dMRI matrices (structural connectivity)
- Node features: Anatomical volumes (or a constant ones channel if none are provided)
- Edge weights: Primary dMRI metric values at edge positions
- Edge attributes: All dMRI metrics + FC sampled only at existing edges

Based on cs224w-brain-graphs preprocessing approach.
"""

import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import logging

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm


def _stem_contains_tag(stem: str, tag: str) -> bool:
    """Return True if ``tag`` appears in ``stem`` with non-alphanumeric boundaries."""
    if not tag:
        return False
    pattern = rf"(?<![A-Za-z0-9]){re.escape(tag)}(?![A-Za-z0-9])"
    return re.search(pattern, stem) is not None


def find_matrix_file(folder: Path | str, subject_id: str, metric: str) -> Path:
    """Return the unique ``*.npy`` in ``folder`` matching subject id and metric tag.

    Matching is delimiter-aware so short metrics like ``fa`` do not match ``afd``.
    """
    directory = Path(folder)
    if not directory.is_dir():
        raise FileNotFoundError(f"Matrix folder not found: {directory}")
    if not subject_id:
        raise ValueError("subject_id is required to resolve matrix files")
    if not metric:
        raise ValueError("metric is required to resolve matrix files")

    matches = [
        path
        for path in sorted(directory.glob("*.npy"))
        if _stem_contains_tag(path.stem, subject_id) and _stem_contains_tag(path.stem, metric)
    ]
    if not matches:
        raise FileNotFoundError(
            f"No .npy file in {directory} matched subject_id={subject_id!r} and metric={metric!r}"
        )
    if len(matches) > 1:
        preview = ", ".join(str(path.name) for path in matches[:5])
        suffix = "..." if len(matches) > 5 else ""
        raise ValueError(
            f"Ambiguous matrix files in {directory} for subject_id={subject_id!r} "
            f"metric={metric!r}: {preview}{suffix}"
        )
    return matches[0]


def build_node_features(
    num_nodes: int,
    anatomical_features: np.ndarray | None = None,
) -> np.ndarray:
    """Build node feature matrix from anatomical data, or a constant ones channel.

    Args:
        num_nodes: Number of graph nodes.
        anatomical_features: Anatomical features (e.g., volumes) [N, F], optional.

    Returns:
        Node features [N, num_features].
    """
    if anatomical_features is not None:
        features = np.asarray(anatomical_features, dtype=np.float32)
        if features.ndim == 1:
            features = features.reshape(-1, 1)
        if features.shape[0] != num_nodes:
            raise ValueError(
                f"anatomical_features has {features.shape[0]} rows but graph has {num_nodes} nodes"
            )
    else:
        features = np.ones((num_nodes, 1), dtype=np.float32)

    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    return features.astype(np.float32)


def matrix_to_edge_index(
    matrix: np.ndarray,
    threshold: float = 0.0
) -> tuple[torch.Tensor, torch.Tensor]:
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
    matrices: dict[str, np.ndarray]
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



def _require_primary_edge_attribute(primary_dmri_key: str, edge_attributes: list[str] | None) -> None:
    """Fail before I/O when the topology metric is not an edge attribute."""
    if primary_dmri_key not in (edge_attributes or []):
        raise ValueError(
            f"primary_dmri_key {primary_dmri_key!r} must appear in edge_attributes"
        )


def create_pyg_data(
    dmri_path: str,
    primary_dmri_key: str,
    fc_path: str,
    fc_metric: str,
    edge_attributes: list[str] | None = None,
    anatomical_features: np.ndarray | None = None,
    target: np.ndarray | None = None,
    subject_id: str | None = None,
    threshold: float = 0.0,
    density: float | None = None,
    top_k: int | None = None,
    log_transform: bool = False,
) -> Data:
    """
    Create PyTorch Geometric Data object from brain connectivity data.
    
    Graph topology is derived by binarizing dMRI matrices (structural connectivity).
    All dMRI metrics + FC are sampled at edge positions as edge attributes.
    Node features are anatomical volumes when provided, otherwise a constant ones channel.

    Matrix files are resolved by matching ``subject_id`` and each metric tag inside
    filenames under ``dmri_path`` / ``fc_path`` (segmentation naming is not hardcoded).
    
    Args:
        dmri_path: Folder containing subject diffusion matrices (``*.npy``).
        primary_dmri_key: Which dMRI metric to use for graph topology and edge weights
        fc_path: Folder containing subject FC matrices (``*.npy``)
        fc_metric: Metric tag used to discover the functional connectivity matrix
        edge_attributes: dMRI metric names stored as edge attributes; must include ``primary_dmri_key``
        anatomical_features: Anatomical features [N, F] (e.g., volumes)
        target: Target values (e.g., behavioral scores)
        subject_id: Subject identifier (required for on-disk matrix discovery)
        threshold: Edge weight threshold for graph topology
        density: Target graph density (overrides threshold)
        top_k: Keep top-K edges per node (overrides density and threshold)
        log_transform: Whether to apply log transform to edge weights (optional)
        
    Returns:
        PyTorch Geometric Data object
    """
    if edge_attributes is None:
        edge_attributes = []
    _require_primary_edge_attribute(primary_dmri_key, edge_attributes)
    if not subject_id:
        raise ValueError("subject_id is required to load subject matrix files")

    # Load subject-specific matrices
    dmri_matrices = {}
    expected_shape = None
    for key in edge_attributes:
        mat = np.load(find_matrix_file(dmri_path, subject_id, key))

        if expected_shape is None:
            expected_shape = mat.shape
        if mat.shape != expected_shape:
            raise ValueError(f"Shape mismatch for dMRI matrix '{key}' (expected {expected_shape}, got {mat.shape})")
        dmri_matrices[key] = mat

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
    dmri_edge_attr = gather_edge_attributes(edge_index, sc_attr_dict) if sc_attr_dict else None

    functional_connectivity = np.load(find_matrix_file(fc_path, subject_id, fc_metric))
    fc_edge_attr = gather_edge_attributes(edge_index, {"fc": functional_connectivity})

    combined_edge_attrs = [attr for attr in [dmri_edge_attr, fc_edge_attr] if attr is not None]
    edge_attr = torch.cat(combined_edge_attrs, dim=1) if combined_edge_attrs else None

    x = torch.tensor(
        build_node_features(num_nodes, anatomical_features=anatomical_features),
        dtype=torch.float32,
    )

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
        subjects_file: str,
        dmri_folder: str,
        fc_folder: str,
        fc_metric: str,
        anatomical_df: str | list[str] | None,
        primary_dmri_key: str,
        targets_df: str = None,
        edge_attributes: list[str] = None,
        threshold: float = None,
        density: float = None,
        top_k: int = None,
        log_transform: bool = False,
        target_columns: list[str] = None,
        cache_dir: str = "cache",
        force_reload: bool = False,
        num_workers: int = 4,
        write_cache_payload_debug: bool = False,
    ):
        if edge_attributes is None:
            edge_attributes = []
        _require_primary_edge_attribute(primary_dmri_key, edge_attributes)

        self.subjects_file = Path(subjects_file)
        self.dmri_folder = Path(dmri_folder)
        self.fc_folder = Path(fc_folder)
        if anatomical_df is None:
            self.anatomical_dfs: list[Path] = []
        elif isinstance(anatomical_df, (str, Path)):
            self.anatomical_dfs = [Path(anatomical_df)]
        else:
            self.anatomical_dfs = [Path(p) for p in anatomical_df]
        # Keep legacy attribute for backward compatibility.
        self.anatomical_df = self.anatomical_dfs[0] if len(self.anatomical_dfs) > 0 else None
        self.targets_df = Path(targets_df) if targets_df else None
        self.primary_dmri_key = primary_dmri_key
        self.edge_attributes = edge_attributes
        self.threshold = threshold
        self.density = density
        self.top_k = top_k
        self.log_transform = log_transform
        self.target_columns = target_columns
        self.cache_dir = cache_dir
        self._force_reload = force_reload
        self.num_workers = num_workers
        self.write_cache_payload_debug = write_cache_payload_debug
        self.fc_metric = fc_metric
        self._anatomical_feature_names: list[str] = []

        # Load subject, dmri mats, fc mats, anatomical features, and targets into memory
        self._subject_ids = self._load_subject_ids()
        self._anatomical_features = self._load_anatomical_features() if self.anatomical_dfs else None
        self._targets = self._load_targets() if self.targets_df else None

        self._cache_payload = self._build_cache_payload()
        root = Path(self.cache_dir) / self._get_cache_name(payload=self._cache_payload)

        super().__init__(str(root))
        self.load(self.processed_paths[0])

        if self.write_cache_payload_debug:
            self._write_cache_payload_debug_file(root)

    def _load_subject_ids(self) -> list[str]:
        if not self.subjects_file.exists():
            raise FileNotFoundError(f"Subjects file not found: {self.subjects_file}")
        
        with open(self.subjects_file, 'r') as f:
            subject_ids = [line.strip() for line in f if line.strip()]
        
        return subject_ids

    @staticmethod
    def _load_dataframe(path: Path) -> pd.DataFrame:
        suffix = path.suffix.lower()
        if suffix in {".xlsx", ".xls"}:
            return pd.read_excel(path)
        if suffix == ".tsv":
            return pd.read_csv(path, sep="\t")
        return pd.read_csv(path)

    def _load_anatomical_features(self) -> dict[str, pd.DataFrame]:
        """Load anatomical files as per-node feature channels keyed by file stem.

        Each file must have `sid` and node columns. For a subject row, node values
        are interpreted as a vector of length num_nodes. Multiple files are stacked
        as multiple channels: [num_nodes, num_anatomical_features].
        """
        loaded: dict[str, pd.DataFrame] = {}
        feature_names: list[str] = []

        for anat_path in self.anatomical_dfs:
            if not anat_path.exists():
                raise FileNotFoundError(f"Anatomical features file not found: {anat_path}")

            df = self._load_dataframe(anat_path)
            if "sid" not in df.columns:
                raise ValueError(f"Anatomical file must contain 'sid' column: {anat_path}")

            df = df.copy()
            df["sid"] = df["sid"].astype(str)
            df = df[df["sid"].isin(self._subject_ids)].copy()

            node_cols = [c for c in df.columns if c != "sid"]
            if len(node_cols) == 0:
                raise ValueError(f"Anatomical file has no node columns: {anat_path}")

            df[node_cols] = df[node_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

            # Find out which feature is being loaded by matching token to supported
            # feature names.
            supported_features = ["volume", "thickness", "surface"]
            feature_name = None
            for token in supported_features:
                if token in anat_path.stem.lower().split("_"):
                    feature_name = token
                    break
                else:
                    feature_name = anat_path.stem.lower()

            loaded[feature_name] = df
            feature_names.append(feature_name)

        self._anatomical_feature_names = feature_names
        return loaded

    def _load_targets(self) -> dict[str, np.ndarray]:
        # Implement loading of target values for all subjects
        if not self.targets_df.exists():
            raise FileNotFoundError(f"Targets file not found: {self.targets_df}")

        df = pd.read_excel(self.targets_df)
        df['subjectkey'] = df["subjectkey"].apply(
            lambda x: f"sub-{x.replace('_', '')}" if isinstance(x, str) else f"sub-{x}"
        )

        # Small filtering to ensure we only keep subjects in our subject list
        return df[df['subjectkey'].isin(self._subject_ids)]


    def _build_cache_payload(self) -> dict[str, object]:
        """Collect all configuration fields that define cached graph materialization."""
        return {
            "primary_dmri_key": self.primary_dmri_key,
            "threshold": self.threshold,
            "density": self.density,
            "top_k": self.top_k,
            "log_transform": self.log_transform,
            "edge_attributes": sorted(self.edge_attributes),
            "target_columns": self.target_columns,
            "subjects_file": str(self.subjects_file),
            "subject_ids": self._subject_ids,
            "anatomical_df": [str(p) for p in self.anatomical_dfs] if self.anatomical_dfs else None,
            "targets_df": str(self.targets_df) if self.targets_df is not None else None,
            "fc_folder": str(self.fc_folder),
            "fc_metric": self.fc_metric,
        }

    def _get_cache_name(self, payload: dict[str, object] | None = None) -> str:
        """Build a short, deterministic cache directory name.

        Verbose feature lists can exceed filesystem path limits on some systems.
        We keep a compact human-readable prefix and append an 8-char checksum
        from all relevant dataset configuration fields.
        """
        if payload is None:
            payload = self._build_cache_payload()

        payload_text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        checksum = hashlib.blake2s(payload_text.encode("utf-8"), digest_size=4).hexdigest()

        primary = str(self.primary_dmri_key).replace(" ", "")
        return f"{primary}_cfg_{checksum}"

    def _write_cache_payload_debug_file(self, root: Path) -> None:
        """Persist unhashed cache payload next to processed outputs for traceability."""
        try:
            root.mkdir(parents=True, exist_ok=True)
            debug_path = root / "cache_payload.json"
            with open(debug_path, "w", encoding="utf-8") as handle:
                json.dump(self._cache_payload, handle, indent=2, sort_keys=True)
        except OSError as error:
            # Debug export should never break dataset loading.
            logging.warning("Could not write cache payload debug file: %s", error)

    @property
    def processed_file_names(self) -> list[str]:
        return ["data.pt"]

    # Attribute mapping feature names to indices for easier access in transforms/models
    @property
    def node_feature_indices(self) -> dict[str, int]:
        indices = {}
        idx = 0

        if self._anatomical_feature_names:
            for feature_name in self._anatomical_feature_names:
                indices[feature_name] = idx
                idx += 1
        else:
            indices["constant"] = idx
            idx += 1

        return indices

    @property
    def edge_attribute_indices(self) -> dict[str, int]:
        indices = {}
        idx = 0

        for key in sorted(self.edge_attributes):
            indices[key] = idx
            idx += 1

        # FC attributes are added after dMRI attributes
        indices['fc'] = idx

        return indices

    @property
    def target_indices(self) -> dict[str, int]:
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
        logging.info("Processing %d brain graphs...", len(self._subject_ids))

        def load_single(args):
            sid, dmri_path, fc_path = args
            # Extract anatomical features for this subject if available
            anat_feats = None
            if self._anatomical_features is not None:
                anat_vectors: list[np.ndarray] = []
                for feature_name in self._anatomical_feature_names:
                    df = self._anatomical_features[feature_name]
                    anat_row = df[df['sid'] == sid]
                    if anat_row.empty:
                        raise ValueError(
                            f"Missing anatomical feature '{feature_name}' for subject {sid}"
                        )

                    node_vector = anat_row.drop(columns=['sid']).values.flatten().astype(np.float32)
                    anat_vectors.append(node_vector)

                if len(anat_vectors) == 1:
                    anat_feats = anat_vectors[0]
                elif len(anat_vectors) > 1:
                    lengths = {vec.shape[0] for vec in anat_vectors}
                    if len(lengths) != 1:
                        raise ValueError(
                            "All anatomical channels must contain the same number of nodes"
                        )
                    anat_feats = np.stack(anat_vectors, axis=1)

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

            return create_pyg_data(
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
                log_transform=self.log_transform,
                fc_metric=self.fc_metric,
            )

        data_list = []
        logging.info("Using %d workers for parallel processing", self.num_workers)

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            jobs = [(sid, self.dmri_folder, self.fc_folder) for sid in self._subject_ids]
            futures = [executor.submit(load_single, args) for args in jobs]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing brain graphs"):
                data_list.append(future.result())

        if not data_list:
            raise ValueError("No valid brain graphs were processed successfully.")

        data_list.sort(key=lambda d: d.subject_id)  # Sort by subject ID for consistency

        self.save(data_list, self.processed_paths[0])
        logging.info(
            "Saved processed dataset with %d graphs to %s",
            len(data_list),
            self.processed_paths[0],
        )


def save_dataset(
    dataset: list[Data],
    save_path: str,
    metadata: dict | None = None
) -> None:
    """
    Save PyTorch Geometric dataset to disk for faster loading.
    
    Args:
        dataset: List of PyG Data objects
        save_path: Path to save file (.pt extension recommended)
        metadata: Optional metadata dictionary (preprocessing parameters, etc.)
    """
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    save_dict = {
        'dataset': dataset,
        'num_graphs': len(dataset),
        'metadata': metadata or {}
    }

    torch.save(save_dict, path)
    logging.info("Saved %d graphs to %s", len(dataset), save_path)


def load_dataset(
    load_path: str,
    return_metadata: bool = False
) -> list[Data] | tuple[list[Data], dict]:
    """
    Load PyTorch Geometric dataset from disk.
    
    Args:
        load_path: Path to saved dataset file
        return_metadata: Whether to also return metadata
        
    Returns:
        List of PyG Data objects, or (dataset, metadata) if return_metadata=True
    """
    path = Path(load_path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {load_path}")

    save_dict = torch.load(path, weights_only=False)
    dataset = save_dict['dataset']
    
    logging.info("Loaded %d graphs from %s", len(dataset), load_path)
    
    if return_metadata:
        metadata = save_dict.get('metadata', {})
        return dataset, metadata
    else:
        return dataset
