"""
Utilities to select features from an InMemoryDataset for training, validation, and testing.
This provides flexibility to specify which features to use for the node attributes (x), edge attributes (edge_attr),
and target labels (y) during training. The selected features are then used to create DataLoaders for the training, validation, and testing phases.
This will also subset the dataset to only include the selected features, which can help reduce memory usage and improve training efficiency.
"""

import json
from pathlib import Path
from typing import List, Optional, Union

import torch
from torch_geometric.data import InMemoryDataset, Data

class FeatureSelector:
    """
    Utility class to select features from an InMemoryDataset for training, validation, and testing.
    """
    def __init__(self,
                 dataset: InMemoryDataset,
                 node_feature_keys: Optional[List[str]] = None,
                 edge_feature_keys: Optional[List[str]] = None,
                 target_keys: Optional[List[str]] = None):
        self.dataset = dataset
        self.node_feature_keys = node_feature_keys
        self.edge_feature_keys = edge_feature_keys
        self.target_keys = target_keys

    @staticmethod
    def _resolve_indices(
        keys: List,
        name_to_index: dict,
        feature_type: str,
    ) -> List[int]:
        """Resolve a list of string/int keys into positional indices."""
        indices: List[int] = []

        for key in keys:
            if isinstance(key, int):
                indices.append(key)
                continue

            if isinstance(key, str):
                if key not in name_to_index:
                    available = sorted([k for k in name_to_index.keys() if isinstance(k, str)])
                    raise KeyError(
                        f"Unknown {feature_type} key '{key}'. Available {feature_type} keys: {available}"
                    )
                indices.append(name_to_index[key])
                continue

            raise TypeError(
                f"Unsupported {feature_type} key type: {type(key).__name__}. "
                "Keys must be int or str."
            )

        return indices

    def subset_dataset(self) -> InMemoryDataset:
        """
        Subset the dataset to only include the selected features for node attributes, edge attributes, and target labels.
        """

        # Work on a copy so repeated run_experiment calls do not modify the
        # original dataset instance held by the notebook.
        subset = self.dataset.copy()

        node_indices = None
        edge_indices = None
        target_indices = None

        if self.node_feature_keys is not None:
            node_map = getattr(subset, "node_feature_indices", {})
            node_indices = self._resolve_indices(self.node_feature_keys, node_map, "node")

        if self.edge_feature_keys is not None:
            edge_map = getattr(subset, "edge_attribute_indices", {})
            edge_indices = self._resolve_indices(self.edge_feature_keys, edge_map, "edge")

        if self.target_keys is not None:
            target_map = getattr(subset, "target_indices", {})
            target_indices = self._resolve_indices(self.target_keys, target_map, "target")

        # NOTE: Iterating over an InMemoryDataset returns detached Data objects.
        # Rebuild the backing storage to persist any feature/target subsetting.
        selected_data_list = []

        for i in range(len(subset)):
            data = subset[i].clone()
            if node_indices is not None:
                data.x = data.x[:, node_indices]

            if edge_indices is not None:
                if getattr(data, "edge_attr", None) is None:
                    raise ValueError("edge_feature_keys were provided but data.edge_attr is missing")
                data.edge_attr = data.edge_attr[:, edge_indices]

            if target_indices is not None:
                if getattr(data, "y", None) is None:
                    raise ValueError("target_keys were provided but data.y is missing")

                if data.y.ndim == 1:
                    data.y = data.y.unsqueeze(0)

                data.y = data.y[:, target_indices]

            selected_data_list.append(data)

        if selected_data_list:
            data, slices = subset.collate(selected_data_list)
            subset._data = data
            subset.slices = slices
            # Ensure __getitem__ does not return stale cached Data objects.
            subset._data_list = None

        return subset


class ILRTransform:
    """
    Isometric Log-Ratio (ILR) transform for compositional target variables.

    Compositional data live on the *simplex* (D values that sum to 1, all > 0)
    which is a curved subspace of ℝ^D.  Standard Euclidean losses (MAE, MSE)
    and regressors implicitly assume flat Euclidean space, so operating directly
    on simplex values introduces geometric distortions.

    The ILR transform maps the simplex isometrically onto an unconstrained
    Euclidean subspace ℝ^{D-1} using a Helmert orthonormal basis:

        z = log(y) · H        (forward,  simplex → ℝ^{D-1})
        y = softmax(z · Hᵀ)   (inverse,  ℝ^{D-1} → simplex)

    where H ∈ ℝ^{D × D-1} is the orthonormal Helmert basis.

    Key properties
    --------------
    - Distances are preserved: d_Aitchison(y_i, y_j) = ‖z_i - z_j‖₂
    - Output is unconstrained, so MAE/MSE losses are well-specified
    - Output dimension is D-1 (one less than the simplex dimension)
    - The transform is bijective: ``inverse_transform(transform(y)) == y``

    Usage notes
    -----------
    - After applying this transform, set ``out_channels = D - 1`` in the model.
    - At inference, call ``inverse_transform`` on model outputs to recover
      probabilities.  ``bGINE.predict(apply_ilr_inverse=True)`` handles this
      automatically.
    - **Not** compatible with ``dirichlet_nll`` or ``kl_div`` losses (those
      expect raw simplex targets).  Use with ``mae`` or ``mse``.

    Args:
        eps: Small constant added before log for numerical stability (default: 1e-12)
    """
    def __init__(self, eps: float = 1e-12):
        self.eps = eps
        self._D: Optional[int] = None
        self._basis: Optional[torch.Tensor] = None

    @staticmethod
    def _build_basis(D: int) -> torch.Tensor:
        """Construct the Helmert orthonormal basis for D-dimensional simplex."""
        H = torch.zeros(D, D - 1, dtype=torch.float64)
        for k in range(D - 1):
            remaining = D - k          # number of nonzero rows (k .. D-1)
            H[k, k] = (remaining - 1) / remaining
            H[k, k] = torch.sqrt(H[k, k].clone())
            neg_val = -1.0 / ((remaining - 1) * remaining) ** 0.5
            for j in range(k + 1, D):
                H[j, k] = neg_val
        return H.float()

    def fit(self, dataset: InMemoryDataset) -> "ILRTransform":
        """
        Infer the simplex dimension D from the first sample in the dataset
        and pre-compute the Helmert basis matrix.

        Args:
            dataset: Iterable of ``torch_geometric.data.Data`` objects whose
                     ``.y`` attribute holds compositional targets of shape
                     ``[1, D]`` or ``[D]``.

        Returns:
            self (for chaining)
        """
        # Infer target dimension D from the first data sample.
        for data in dataset:
            if data.y is not None:
                D = data.y.shape[-1]
                if D < 2:
                    raise ValueError(
                        f"ILR requires at least 2 compositional parts, got D={D}."
                    )
                self._D = D
                self._basis = self._build_basis(D)   # [D, D-1]
                return self

        raise ValueError(
            "Dataset contains no target values (data.y is None for all samples)."
        )

    def transform(self, data: Data) -> Data:
        """
        Apply the ILR transform to targets in a Data object.

        The output shape changes from ``[N, D]`` to ``[N, D-1]``.

        Args:
            data: PyTorch Geometric Data object with ``.y`` of shape ``[N, D]``.

        Returns:
            A cloned Data object whose ``.y`` is in ILR space, shape ``[N, D-1]``.
        """
        if self._basis is None:
            raise RuntimeError("ILRTransform has not been fitted yet. Call fit() first.")

        data = data.clone()
        if data.y is not None:
            y = data.y
            squeezed = False
            if y.dim() == 1:
                y = y.unsqueeze(0)
                squeezed = True

            basis = self._basis.to(y.device)
            y_safe = torch.clamp(y, min=self.eps)
            z = torch.log(y_safe) @ basis          # [N, D-1]

            data.y = z.squeeze(0) if squeezed else z
        return data

    def inverse_transform(self, z: torch.Tensor) -> torch.Tensor:
        """
        Map ILR coordinates back to the probability simplex.

        Args:
            z: ILR coordinates ``[N, D-1]`` or ``[D-1]``.

        Returns:
            Compositional probabilities ``[N, D]`` or ``[D]``, summing to 1.
        """
        if self._basis is None:
            raise RuntimeError("ILRTransform has not been fitted yet. Call fit() first.")

        basis = self._basis.to(z.device)           # [D, D-1]
        squeezed = z.dim() == 1
        if squeezed:
            z = z.unsqueeze(0)

        log_y = z @ basis.T                        # [N, D]
        y = torch.exp(log_y)
        y = y / y.sum(dim=-1, keepdim=True)        # normalize to simplex

        return y.squeeze(0) if squeezed else y

    def __call__(self, data: Data) -> Data:
        """Convenience method to apply transform directly."""
        return self.transform(data)

    def save(self, path: Union[str, Path]) -> None:
        """Save the fitted basis matrix to a file."""
        if self._basis is None:
            raise RuntimeError("ILRTransform has not been fitted yet. Call fit() first.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "D": self._D,
            "eps": self.eps,
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ILRTransform":
        """
        Load a previously saved ILRTransform from a file.
        
        Args:
            path: Path to the saved ILRTransform file.

        Returns:
            An ILRTransform instance with the basis matrix loaded.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"ILRTransform file not found: {path}")

        with path.open("r") as f:
            payload = json.load(f)

        D = payload["D"]
        basis = torch.tensor(payload["basis"])
        eps = payload.get("eps", 1e-12)

        transform = cls(eps=eps)
        transform._D = D
        transform._basis = basis
        return transform

    def __repr__(self) -> str:
        return f"ILRTransform(D={self._D}, eps={self.eps}, fitted={self._basis is not None})"
