"""Unit tests for FeatureSelector."""

import pytest
import torch

from mmgnn.training.features import FeatureSelector
from tests.models.conftest import EDGE_IN, tiny_graph
from tests.training.conftest import make_memory_dataset


class TestResolveIndices:
    def test_int_and_str_keys(self):
        mapping = {"a": 0, "b": 2}
        assert FeatureSelector._resolve_indices([0, "b"], mapping, "node") == [0, 2]

    def test_unknown_key_raises(self):
        with pytest.raises(KeyError, match="Unknown node key"):
            FeatureSelector._resolve_indices(["missing"], {"a": 0}, "node")

    def test_bad_type_raises(self):
        with pytest.raises(TypeError, match="Unsupported node key type"):
            FeatureSelector._resolve_indices([1.5], {"a": 0}, "node")


class TestSubsetDataset:
    def test_slices_features_and_preserves_original(self):
        dataset = make_memory_dataset()
        original_x_width = dataset[0].x.shape[1]
        selector = FeatureSelector(
            dataset,
            node_feature_keys=["n0", "n2"],
            edge_feature_keys=["e1"],
            target_keys=["t0"],
        )
        subset = selector.subset_dataset()

        assert subset[0].x.shape[1] == 2
        assert subset[0].edge_attr.shape[1] == 1
        assert subset[0].y.shape[-1] == 1
        # Original unchanged
        assert dataset[0].x.shape[1] == original_x_width
        assert dataset[0].edge_attr.shape[1] == EDGE_IN

    def test_missing_edge_attr_raises(self):
        g = tiny_graph()
        g.subject_id = "sub-001"
        del g.edge_attr
        dataset = make_memory_dataset(
            [g],
            edge_attribute_indices={"e0": 0},
        )
        selector = FeatureSelector(dataset, edge_feature_keys=["e0"])
        with pytest.raises(ValueError, match="edge_attr is missing"):
            selector.subset_dataset()

    def test_missing_y_raises(self):
        g = tiny_graph()
        g.subject_id = "sub-001"
        dataset = make_memory_dataset([g], target_indices={"t0": 0})
        selector = FeatureSelector(dataset, target_keys=["t0"])
        with pytest.raises(ValueError, match="data.y is missing"):
            selector.subset_dataset()
