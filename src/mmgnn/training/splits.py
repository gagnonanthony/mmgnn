"""
Utilities to manage training/validation/test splits for the mmgnn framework.
Splits are pre-defined and stored as txt file containing the subject IDs
for each split. The utility functions will read these files and create the corresponding
DataLoaders for training, validation, and testing based on a InMemoryDataset.
"""

from pathlib import Path

from torch_geometric.data import InMemoryDataset


class DataSplitter:
    """
    Utility class to handle data splits for training, validation, and testing.
    """
    def __init__(self,
                 dataset: InMemoryDataset,
                 train_subjects: str,
                 val_subjects: str = None,
                 test_subjects: str = None):
        self.dataset = dataset
        self.train_subjects = Path(train_subjects)
        if val_subjects is not None:
            self.val_subjects = Path(val_subjects)
        if test_subjects is not None:
            self.test_subjects = Path(test_subjects)

    @staticmethod
    def _load_subject_ids(file_path: Path) -> list[str]:
        with open(file_path, 'r') as f:
            subject_ids = [line.strip() for line in f if line.strip()]
        return subject_ids

    def _get_indices_for_subjects(self, subject_ids: list[str]) -> dict[str, int]:
        subject_to_index = {self.dataset[i].subject_id: i for i in range(len(self.dataset))}
        missing = [sid for sid in subject_ids if sid not in subject_to_index]
        if missing:
            preview = ", ".join(missing[:5])
            suffix = "..." if len(missing) > 5 else ""
            raise ValueError(f"Subjects not found in dataset ({len(missing)}): {preview}{suffix}")
        return {sid: subject_to_index[sid] for sid in subject_ids if sid in subject_to_index}

    def get_split_indices(self) -> tuple[dict[str, int], dict[str, int] | None, dict[str, int] | None]:
        train_subject_ids = self._load_subject_ids(self.train_subjects)
        train_indices = self._get_indices_for_subjects(train_subject_ids)

        val_indices = None
        if hasattr(self, 'val_subjects'):
            val_subject_ids = self._load_subject_ids(self.val_subjects)
            val_indices = self._get_indices_for_subjects(val_subject_ids)

        test_indices = None
        if hasattr(self, 'test_subjects'):
            test_subject_ids = self._load_subject_ids(self.test_subjects)
            test_indices = self._get_indices_for_subjects(test_subject_ids)

        return train_indices, val_indices, test_indices
