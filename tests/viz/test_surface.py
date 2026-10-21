"""Unit tests for mmgnn.viz.surface (yabplot plotting stubbed)."""

from pathlib import Path
from unittest.mock import patch

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from mmgnn.viz.surface import (
    _embedding_surface_path,
    build_group_node_embeddings,
    build_subject_node_embeddings,
    load_subject_ids,
    load_subject_node_embeddings,
    plot_node_embedding_surfaces,
    plot_node_scores_surface,
)


def _write_embeddings(data_dir: Path, subject_id: str, arr: np.ndarray) -> None:
    subject_dir = data_dir / "subjects" / subject_id
    subject_dir.mkdir(parents=True, exist_ok=True)
    np.save(subject_dir / "node_embeddings.npy", arr)


class TestEmbeddingsIO:
    def test_round_trip(self, tmp_path: Path):
        arr = np.arange(12, dtype=np.float32).reshape(4, 3)
        _write_embeddings(tmp_path, "sub-001", arr)
        loaded = load_subject_node_embeddings(tmp_path, "sub-001")
        np.testing.assert_array_equal(loaded, arr)
        np.testing.assert_array_equal(
            build_subject_node_embeddings(data_dir=tmp_path, subject_id="sub-001"),
            arr,
        )

    def test_missing_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="Node embeddings not found"):
            load_subject_node_embeddings(tmp_path, "missing")

    def test_non_2d_raises(self, tmp_path: Path):
        _write_embeddings(tmp_path, "sub-001", np.arange(3, dtype=np.float32))
        with pytest.raises(ValueError, match="shape"):
            load_subject_node_embeddings(tmp_path, "sub-001")


class TestGroupAggregate:
    def test_mean_signed_and_abs(self, tmp_path: Path):
        a = np.array([[1.0, -2.0], [3.0, -4.0]], dtype=np.float32)
        b = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
        _write_embeddings(tmp_path, "sub-001", a)
        _write_embeddings(tmp_path, "sub-002", b)

        signed = build_group_node_embeddings(
            data_dir=tmp_path, subject_ids=["sub-001", "sub-002"], aggregate="mean_signed"
        )
        np.testing.assert_allclose(signed, (a + b) / 2.0)

        abs_mean = build_group_node_embeddings(
            data_dir=tmp_path, subject_ids=["sub-001", "sub-002"], aggregate="mean_abs"
        )
        np.testing.assert_allclose(abs_mean, (np.abs(a) + np.abs(b)) / 2.0)

    def test_inconsistent_shapes_raise(self, tmp_path: Path):
        _write_embeddings(tmp_path, "sub-001", np.ones((4, 2), dtype=np.float32))
        _write_embeddings(tmp_path, "sub-002", np.ones((3, 2), dtype=np.float32))
        with pytest.raises(ValueError, match="Inconsistent node counts"):
            build_group_node_embeddings(
                data_dir=tmp_path, subject_ids=["sub-001", "sub-002"]
            )

        _write_embeddings(tmp_path, "sub-002", np.ones((4, 3), dtype=np.float32))
        with pytest.raises(ValueError, match="Inconsistent embedding dimensions"):
            build_group_node_embeddings(
                data_dir=tmp_path, subject_ids=["sub-001", "sub-002"]
            )


class TestEmbeddingSurfacePath:
    def test_file_suffix(self, tmp_path: Path):
        out = tmp_path / "map.png"
        assert _embedding_surface_path(out, 2) == tmp_path / "map_dim002.png"

    def test_directory(self, tmp_path: Path):
        out = tmp_path / "maps"
        assert _embedding_surface_path(out, 1) == out / "embedding_dim_001.png"
        assert out.is_dir()


class TestLoadSubjectIds:
    def test_skips_blanks(self, tmp_path: Path):
        path = tmp_path / "subjects.txt"
        path.write_text("sub-001\n\n  sub-002  \n", encoding="utf-8")
        assert load_subject_ids(path) == ["sub-001", "sub-002"]

    def test_empty_raises(self, tmp_path: Path):
        path = tmp_path / "subjects.txt"
        path.write_text("\n\n", encoding="utf-8")
        with pytest.raises(ValueError, match="No subject IDs"):
            load_subject_ids(path)


class TestPlotting:
    def test_score_length_mismatch(self, tmp_path: Path):
        with patch("yabplot.data.get_atlas_regions", return_value=["a", "b"]):
            with pytest.raises(ValueError, match="does not match atlas regions"):
                plot_node_scores_surface(
                    node_scores=np.array([1.0]),
                    atlas="schaefer",
                    output_path=tmp_path / "out.png",
                )

    def test_plot_node_embedding_surfaces(self, tmp_path: Path):
        embeddings = np.ones((3, 2), dtype=np.float32)
        out = tmp_path / "surfaces"
        with (
            patch("yabplot.data.get_atlas_regions", return_value=["r0", "r1", "r2"]),
            patch("mmgnn.viz.surface.yab.plot_cortical") as plot_cortical,
        ):
            paths = plot_node_embedding_surfaces(
                node_embeddings=embeddings,
                atlas="schaefer",
                output_path=out,
                title_prefix="test",
            )

        assert len(paths) == 2
        assert plot_cortical.call_count == 2
        for path in paths:
            assert path.is_file()
