"""
Utilities to select features from an InMemoryDataset for training, validation, and testing.
This provides flexibility to specify which features to use for the node attributes (x), edge attributes (edge_attr),
and target labels (y) during training. The selected features are then used to create DataLoaders for the training, validation, and testing phases.
This will also subset the dataset to only include the selected features, which can help reduce memory usage and improve training efficiency.
"""

from torch_geometric.data import InMemoryDataset

class FeatureSelector:
    """
    Utility class to select features from an InMemoryDataset for training, validation, and testing.
    """
    def __init__(self,
                 dataset: InMemoryDataset,
                 node_feature_keys: list[str] | None = None,
                 edge_feature_keys: list[str] | None = None,
                 target_keys: list[str] | None = None):
        self.dataset = dataset
        self.node_feature_keys = node_feature_keys
        self.edge_feature_keys = edge_feature_keys
        self.target_keys = target_keys

    @staticmethod
    def _resolve_indices(
        keys: list,
        name_to_index: dict,
        feature_type: str,
    ) -> list[int]:
        """Resolve a list of string/int keys into positional indices."""
        indices: list[int] = []

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
