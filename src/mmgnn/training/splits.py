"""
Utilities to manage training/validation/test splits for the mmgnn framework.
Splits are pre-defined and stored as txt file containing the subject IDs
for each split. The utility functions will read these files and create the corresponding
DataLoaders for training, validation, and testing based on a InMemoryDataset.
"""

from pathlib import Path
from typing import Tuple, List, Optional, Dict

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
    def _load_subject_ids(file_path: Path) -> List[str]:
        with open(file_path, 'r') as f:
            subject_ids = [line.strip() for line in f if line.strip()]
        return subject_ids

    def _get_indices_for_subjects(self, subject_ids: List[str]) -> Dict[str, int]:
        subject_to_index = {self.dataset[i].subject_id: i for i in range(len(self.dataset))}
        indices = {sid: subject_to_index[sid] for sid in subject_ids if sid in subject_to_index}
        return indices

    def get_split_indices(self) -> Tuple[Dict[str, int], Optional[Dict[str, int]], Optional[Dict[str, int]]]:
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
