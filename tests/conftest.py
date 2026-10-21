"""Shared fixtures for mmgnn tests."""

from pathlib import Path

import pandas as pd
import pytest

from tests.helpers import (
    EDGE_ATTRIBUTES,
    NUM_NODES,
    PRIMARY_DMRI_KEY,
    SUBJECT_IDS,
    write_subject_matrices,
)


@pytest.fixture
def connectome_dirs(tmp_path: Path) -> dict[str, Path]:
    """Write connectomes for two subjects under temporary dMRI/FC folders."""
    dmri_dir = tmp_path / "dmri"
    fc_dir = tmp_path / "fc"
    for index, subject_id in enumerate(SUBJECT_IDS):
        write_subject_matrices(dmri_dir, fc_dir, subject_id, seed=10 + index * 20)
    return {"root": tmp_path, "dmri": dmri_dir, "fc": fc_dir}


@pytest.fixture
def single_subject_dirs(tmp_path: Path) -> dict[str, Path | str]:
    """Write connectomes for a single subject used by create_pyg_data tests."""
    dmri_dir = tmp_path / "dmri"
    fc_dir = tmp_path / "fc"
    subject_id = SUBJECT_IDS[0]
    write_subject_matrices(dmri_dir, fc_dir, subject_id, seed=7)
    return {
        "root": tmp_path,
        "dmri": dmri_dir,
        "fc": fc_dir,
        "subject_id": subject_id,
    }


@pytest.fixture
def brain_graph_fixture(connectome_dirs: dict[str, Path]) -> dict[str, Path | str | list[str]]:
    """Subjects file + anatomical CSV ready for BrainGraphDataset."""
    root = connectome_dirs["root"]
    subjects_file = root / "subjects.txt"
    subjects_file.write_text("\n".join(SUBJECT_IDS) + "\n", encoding="utf-8")

    node_cols = [f"node_{i}" for i in range(NUM_NODES)]
    rows = []
    for sid_index, subject_id in enumerate(SUBJECT_IDS):
        row = {"sid": subject_id}
        for node_index, col in enumerate(node_cols):
            row[col] = float(sid_index + 1) + 0.1 * node_index
        rows.append(row)
    anatomical_df = root / "subject_volume_features.csv"
    pd.DataFrame(rows).to_csv(anatomical_df, index=False)

    return {
        "subjects_file": subjects_file,
        "dmri_folder": connectome_dirs["dmri"],
        "fc_folder": connectome_dirs["fc"],
        "anatomical_df": anatomical_df,
        "cache_dir": root / "cache",
        "edge_attributes": list(EDGE_ATTRIBUTES),
        "primary_dmri_key": PRIMARY_DMRI_KEY,
    }
