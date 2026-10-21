"""Unit tests for DataSplitter."""

from pathlib import Path

import pytest

from mmgnn.training.splits import DataSplitter
from tests.training.conftest import make_memory_dataset


class TestLoadSubjectIds:
    def test_skips_blank_lines(self, tmp_path: Path):
        path = tmp_path / "train.txt"
        path.write_text("sub-001\n\n  sub-002  \n", encoding="utf-8")
        assert DataSplitter._load_subject_ids(path) == ["sub-001", "sub-002"]


class TestGetSplitIndices:
    def test_maps_known_subjects(self, tmp_path: Path):
        dataset = make_memory_dataset()
        train = tmp_path / "train.txt"
        val = tmp_path / "val.txt"
        train.write_text("sub-001\n", encoding="utf-8")
        val.write_text("sub-002\n", encoding="utf-8")

        splitter = DataSplitter(dataset, str(train), val_subjects=str(val))
        train_idx, val_idx, test_idx = splitter.get_split_indices()

        assert train_idx == {"sub-001": 0}
        assert val_idx == {"sub-002": 1}
        assert test_idx is None

    def test_unknown_subject_raises(self, tmp_path: Path):
        dataset = make_memory_dataset()
        train = tmp_path / "train.txt"
        train.write_text("sub-001\nmissing\n", encoding="utf-8")
        splitter = DataSplitter(dataset, str(train))
        with pytest.raises(ValueError, match=r"Subjects not found in dataset \(1\): missing"):
            splitter.get_split_indices()

    def test_omitted_val_test_are_none(self, tmp_path: Path):
        dataset = make_memory_dataset()
        train = tmp_path / "train.txt"
        train.write_text("sub-001\nsub-002\n", encoding="utf-8")
        splitter = DataSplitter(dataset, str(train))
        train_idx, val_idx, test_idx = splitter.get_split_indices()
        assert set(train_idx) == {"sub-001", "sub-002"}
        assert val_idx is None
        assert test_idx is None
