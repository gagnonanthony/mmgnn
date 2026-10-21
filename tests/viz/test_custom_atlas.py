"""Unit tests for mmgnn.viz.custom_atlas helpers (no full CIFTI pipeline)."""

from pathlib import Path

import numpy as np
import pytest

from mmgnn.viz.custom_atlas import (
    ResolvedAtlas,
    _atlas_name_from_dlabel,
    _clean_region_name,
    _cortical_output_dir,
    _cortical_region_ids,
    _find_dlabel_file,
    _find_label_tsv,
    _read_label_tsv,
    _rgb_to_hex,
    _write_cortical_lut_from_tsv,
    cortical_atlas_dir,
    resolve_atlas_argument,
)


class TestResolvedAtlas:
    def test_builtin_kwargs(self):
        atlas = ResolvedAtlas(builtin_name="schaefer")
        assert atlas.is_custom is False
        assert atlas.plot_kwargs() == {"atlas": "schaefer"}
        assert atlas.cortical_plot_kwargs() == {"atlas": "schaefer"}

    def test_custom_kwargs(self, tmp_path: Path):
        atlas = ResolvedAtlas(custom_atlas_dir=tmp_path)
        assert atlas.is_custom is True
        with pytest.raises(ValueError, match="Built-in atlas name"):
            atlas.plot_kwargs()
        assert atlas.cortical_plot_kwargs() == {
            "custom_atlas_path": str(tmp_path / "yabplot" / "cortical")
        }

    def test_empty_raises(self):
        atlas = ResolvedAtlas()
        with pytest.raises(ValueError, match="builtin_name or custom_atlas_dir"):
            atlas.cortical_plot_kwargs()


class TestPureHelpers:
    def test_clean_region_name(self):
        assert _clean_region_name("Left / Frontal") == "Left_-_Frontal"

    def test_rgb_to_hex(self):
        assert _rgb_to_hex(255, 0, 16) == "#ff0010"

    def test_atlas_name_from_dlabel(self):
        path = Path("sub-01_atlas-Schaefer2018_den-32k.dlabel.nii")
        assert _atlas_name_from_dlabel(path) == "Schaefer2018"
        # Fallback uses parent.parent.name when no _atlas- token
        nested = Path("/data/MyAtlas/parcels/labels.dlabel.nii")
        assert _atlas_name_from_dlabel(nested) == "MyAtlas"


class TestCorticalPaths:
    def test_prefers_cortical_subdir(self, tmp_path: Path):
        cortical = tmp_path / "yabplot" / "cortical"
        cortical.mkdir(parents=True)
        assert _cortical_output_dir(tmp_path) == cortical
        assert cortical_atlas_dir(tmp_path) == cortical

    def test_legacy_yabplot_fallback(self, tmp_path: Path):
        legacy = tmp_path / "yabplot"
        legacy.mkdir()
        (legacy / "atlas.csv").write_text("1\n", encoding="utf-8")
        assert _cortical_output_dir(tmp_path) == legacy


class TestDiscovery:
    def test_find_single_dlabel(self, tmp_path: Path):
        dlabel = tmp_path / "atlas.dlabel.nii"
        dlabel.write_bytes(b"nii")
        assert _find_dlabel_file(tmp_path) == dlabel

    def test_find_dlabel_missing(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="No .dlabel.nii"):
            _find_dlabel_file(tmp_path)

    def test_find_dlabel_prefers_unique_32k(self, tmp_path: Path):
        a = tmp_path / "atlas_den-32k.dlabel.nii"
        b = tmp_path / "atlas_den-164k.dlabel.nii"
        a.write_bytes(b"a")
        b.write_bytes(b"b")
        assert _find_dlabel_file(tmp_path) == a

    def test_find_dlabel_ambiguous(self, tmp_path: Path):
        (tmp_path / "a.dlabel.nii").write_bytes(b"a")
        (tmp_path / "b.dlabel.nii").write_bytes(b"b")
        with pytest.raises(ValueError, match="Multiple .dlabel.nii"):
            _find_dlabel_file(tmp_path)

    def test_find_label_tsv_sidecar(self, tmp_path: Path):
        dlabel = tmp_path / "atlas.dlabel.nii"
        tsv = tmp_path / "atlas.tsv"
        dlabel.write_bytes(b"nii")
        tsv.write_text("index\tlabel\n", encoding="utf-8")
        assert _find_label_tsv(dlabel) == tsv

    def test_find_label_tsv_missing(self, tmp_path: Path):
        dlabel = tmp_path / "atlas.dlabel.nii"
        dlabel.write_bytes(b"nii")
        with pytest.raises(FileNotFoundError, match="Could not find label TSV"):
            _find_label_tsv(dlabel)


class TestTsvAndLut:
    def test_read_label_tsv_requires_columns(self, tmp_path: Path):
        path = tmp_path / "labels.tsv"
        path.write_text("index\tlabel\n1\tA\n", encoding="utf-8")
        with pytest.raises(ValueError, match="must contain columns"):
            _read_label_tsv(path)

    def test_write_cortical_lut(self, tmp_path: Path):
        tsv = tmp_path / "labels.tsv"
        tsv.write_text(
            "index\tlabel\tcolor_red\tcolor_green\tcolor_blue\n"
            "1\tLeft A\t10\t20\t30\n"
            "2\tRight/B\t1\t2\t3\n"
            "3\tUnused\t0\t0\t0\n",
            encoding="utf-8",
        )
        lut = tmp_path / "atlas.txt"
        vertex_labels = np.array([0, 1, 1, 2], dtype=int)
        _write_cortical_lut_from_tsv(tsv, vertex_labels, lut)
        text = lut.read_text(encoding="utf-8")
        assert "1  Left_A  10  20  30  0" in text
        assert "2  Right-B  1  2  3  0" in text
        assert "Unused" not in text

    def test_cortical_region_ids(self, tmp_path: Path):
        csv_path = tmp_path / "atlas.csv"
        np.savetxt(csv_path, np.array([0, 1, 2, 2]), fmt="%i")
        assert _cortical_region_ids(tmp_path) == {1, 2}


class TestResolveAtlasArgument:
    def test_builtin_name(self):
        resolved = resolve_atlas_argument("schaefer")
        assert resolved.builtin_name == "schaefer"
        assert resolved.custom_atlas_dir is None

    def test_directory_missing_dlabel_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="No .dlabel.nii file found"):
            resolve_atlas_argument(str(tmp_path))
