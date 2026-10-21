"""
Custom atlas preparation for yabplot connectome visualization.

yabplot's built-in ``plot_connectome`` supports cortical (``.csv`` + ``.txt``) or
subcortical (``.vtk``) atlases, but not both at once. For fsLR ``.dlabel.nii``
atlases we materialize cortical surface files and compute subcortical centroids
from CIFTI voxel coordinates, then merge both into a single node layout.
"""

from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nibabel.affines import apply_affine


@dataclass(frozen=True)
class ResolvedAtlas:
    """Resolved atlas target for yabplot plotting."""

    builtin_name: str | None = None
    custom_atlas_dir: Path | None = None

    @property
    def is_custom(self) -> bool:
        return self.custom_atlas_dir is not None

    def plot_kwargs(self) -> dict[str, str]:
        if self.builtin_name is not None:
            return {"atlas": self.builtin_name}
        raise ValueError("Built-in atlas name is required for plot_kwargs().")

    def cortical_plot_kwargs(self) -> dict[str, str]:
        if self.custom_atlas_dir is not None:
            return {"custom_atlas_path": str(cortical_atlas_dir(self.custom_atlas_dir))}
        if self.builtin_name is not None:
            return {"atlas": self.builtin_name}
        raise ValueError("ResolvedAtlas must specify either builtin_name or custom_atlas_dir.")


def cortical_atlas_dir(atlas_dir: Path) -> Path:
    """Return the yabplot cortical atlas directory for ``plot_cortical``."""
    return _cortical_output_dir(atlas_dir)


def get_parcellation_region_names(atlas: str) -> list[str]:
    """Return region names in the same order used for node score vectors."""
    resolved = resolve_atlas_argument(atlas)
    if resolved.custom_atlas_dir is not None:
        dlabel_path = _find_dlabel_file(resolved.custom_atlas_dir)
        df = _read_label_tsv(_find_label_tsv(dlabel_path))
        return [_clean_region_name(str(label)) for label in df["label"]]

    try:
        from yabplot.data import get_atlas_regions
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Atlas region lookup requires `yabplot` to be installed.") from exc

    if resolved.builtin_name is None:
        raise ValueError("Built-in atlas name is required.")
    return get_atlas_regions(resolved.builtin_name, "cortical")


def resolve_atlas_argument(atlas: str) -> ResolvedAtlas:
    """Resolve the CLI ``--atlas`` argument for yabplot.

    - Directory path: prepare custom atlas artifacts and return the atlas root.
    - Otherwise: treat ``atlas`` as a built-in yabplot atlas name.
    """
    atlas_path = Path(atlas)
    if atlas_path.is_dir():
        prepare_custom_atlas(atlas_path)
        return ResolvedAtlas(custom_atlas_dir=atlas_path)
    return ResolvedAtlas(builtin_name=atlas)


def prepare_custom_atlas(atlas_dir: Path) -> Path:
    """Create yabplot cortical atlas files from fsLR ``.dlabel.nii`` surface data."""
    dlabel_path = _find_dlabel_file(atlas_dir)
    tsv_path = _find_label_tsv(dlabel_path)
    atlas_name = _atlas_name_from_dlabel(dlabel_path)

    cortical_dir = atlas_dir / "yabplot" / "cortical"
    cortical_dir.mkdir(parents=True, exist_ok=True)
    csv_path = cortical_dir / f"{atlas_name}.csv"
    lut_path = cortical_dir / f"{atlas_name}.txt"

    sources_mtime = max(dlabel_path.stat().st_mtime, tsv_path.stat().st_mtime)
    if csv_path.exists() and lut_path.exists():
        if min(csv_path.stat().st_mtime, lut_path.stat().st_mtime) >= sources_mtime:
            return atlas_dir

    vertex_labels = _dlabel_to_yabplot_vertex_map(dlabel_path)
    np.savetxt(csv_path, vertex_labels, fmt="%i")
    _write_cortical_lut_from_tsv(tsv_path, vertex_labels, lut_path)
    return atlas_dir


def extract_combined_atlas(
    atlas_dir: Path,
    *,
    bmesh_type: str = "midthickness",
) -> tuple[dict[str, np.ndarray], list[str], dict[str, str]]:
    """Return merged cortical + subcortical region centroids in TSV index order."""
    prepare_custom_atlas(atlas_dir)

    dlabel_path = _find_dlabel_file(atlas_dir)
    tsv_path = _find_label_tsv(dlabel_path)
    df = _read_label_tsv(tsv_path)

    cortical_dir = _cortical_output_dir(atlas_dir)
    cortical_centroids, _, cortical_colors = _extract_yabplot_centroids(
        "cortical", cortical_dir, bmesh_type
    )
    cortical_region_ids = _cortical_region_ids(cortical_dir)
    subcortical_centroids = _subcortical_centroids_from_dlabel(
        dlabel_path,
        df,
        cortical_region_ids=cortical_region_ids,
    )

    centroids: dict[str, np.ndarray] = {}
    region_names: list[str] = []
    atlas_colors: dict[str, str] = {}

    for _, row in df.iterrows():
        region_id = int(row["index"])
        name = _clean_region_name(str(row["label"]))
        atlas_colors[name] = _rgb_to_hex(
            int(row["color_red"]),
            int(row["color_green"]),
            int(row["color_blue"]),
        )

        if name in cortical_centroids:
            centroid = cortical_centroids[name]
        elif region_id in subcortical_centroids:
            centroid = subcortical_centroids[region_id]
        else:
            centroid = np.array([np.nan, np.nan, np.nan], dtype=float)

        centroids[name] = centroid
        region_names.append(name)

    atlas_colors.update(cortical_colors)
    return centroids, region_names, atlas_colors


def _cortical_region_ids(cortical_dir: Path) -> set[int]:
    csv_candidates = sorted(cortical_dir.glob("*.csv"))
    if not csv_candidates:
        raise FileNotFoundError(f"No cortical .csv file found in {cortical_dir}")
    labels = np.loadtxt(csv_candidates[0], dtype=int)
    return {int(region_id) for region_id in np.unique(labels) if int(region_id) != 0}


def _cortical_output_dir(atlas_dir: Path) -> Path:
    cortical_dir = atlas_dir / "yabplot" / "cortical"
    if cortical_dir.exists():
        return cortical_dir

    # Backward compatibility with the first-pass layout (<atlas_dir>/yabplot/*.csv).
    legacy_dir = atlas_dir / "yabplot"
    if legacy_dir.exists() and any(legacy_dir.glob("*.csv")):
        return legacy_dir
    return cortical_dir


def _extract_yabplot_centroids(
    category: str,
    atlas_dir: Path,
    bmesh_type: str,
) -> tuple[dict[str, np.ndarray], list[str], dict[str, str]]:
    try:
        from yabplot.plotting import _extract_centroids
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Custom atlas support requires `yabplot` to be installed.") from exc

    return _extract_centroids(category, None, str(atlas_dir), bmesh_type)


def _subcortical_centroids_from_dlabel(
    dlabel_path: Path,
    label_table: pd.DataFrame,
    *,
    cortical_region_ids: set[int],
) -> dict[int, np.ndarray]:
    """Compute MNI-space centroids for non-cortical CIFTI structures."""
    target_ids = {
        int(region_id)
        for region_id in label_table["index"].astype(int)
        if int(region_id) not in cortical_region_ids and int(region_id) != 0
    }

    img = nib.load(str(dlabel_path))
    data = np.asarray(img.dataobj[0], dtype=int)
    pooled: dict[int, list[np.ndarray]] = {region_id: [] for region_id in target_ids}

    for structure_name, structure_slice, brain_model in img.header.get_axis(1).iter_structures():
        if "CORTEX" in structure_name:
            continue

        labels = data[structure_slice]
        voxels = brain_model.voxel
        valid = voxels[:, 0] >= 0
        if not np.any(valid):
            continue

        world_coords = apply_affine(brain_model.affine, voxels[valid])
        structure_labels = labels[valid]
        for region_id in np.unique(structure_labels):
            region_id = int(region_id)
            if region_id == 0 or region_id not in pooled:
                continue
            pooled[region_id].append(world_coords[structure_labels == region_id])

    centroids: dict[int, np.ndarray] = {}
    for region_id, chunks in pooled.items():
        if chunks:
            centroids[region_id] = np.vstack(chunks).mean(axis=0)
    return centroids


def _find_dlabel_file(atlas_dir: Path) -> Path:
    candidates = sorted(atlas_dir.rglob("*.dlabel.nii"))
    if not candidates:
        raise FileNotFoundError(
            f"No .dlabel.nii file found under custom atlas directory: {atlas_dir}"
        )
    if len(candidates) > 1:
        den32k = [path for path in candidates if "den-32k" in path.name or "32k" in path.name]
        if len(den32k) == 1:
            return den32k[0]
        names = [path.name for path in candidates]
        raise ValueError(f"Multiple .dlabel.nii files found under {atlas_dir}: {names}")
    return candidates[0]


def _find_label_tsv(dlabel_path: Path) -> Path:
    tsv_path = dlabel_path.with_name(dlabel_path.name.replace(".dlabel.nii", ".tsv"))
    if tsv_path.exists():
        return tsv_path

    stem = dlabel_path.name.replace(".dlabel.nii", "")
    for candidate in dlabel_path.parent.glob("*.tsv"):
        if stem in candidate.name or "atlas-" in candidate.name:
            return candidate

    raise FileNotFoundError(
        f"Could not find label TSV for dlabel file {dlabel_path}. "
        "Expected a matching .tsv sidecar in the same directory."
    )


def _atlas_name_from_dlabel(dlabel_path: Path) -> str:
    stem = dlabel_path.name.replace(".dlabel.nii", "")
    if "_atlas-" in stem:
        remainder = stem.split("_atlas-", 1)[1]
        return remainder.split("_", 1)[0]
    return dlabel_path.parent.parent.name if dlabel_path.parent.name != "." else stem


def _read_label_tsv(tsv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(tsv_path, sep="\t")
    required = {"index", "label", "color_red", "color_green", "color_blue"}
    if not required.issubset(df.columns):
        raise ValueError(f"Label TSV {tsv_path} must contain columns: {sorted(required)}")
    return df


def _clean_region_name(name: str) -> str:
    return name.replace(" ", "_").replace("/", "-")


def _rgb_to_hex(red: int, green: int, blue: int) -> str:
    return f"#{red:02x}{green:02x}{blue:02x}"


def _dlabel_to_yabplot_vertex_map(dlabel_path: Path) -> np.ndarray:
    """Map CIFTI cortex labels onto yabplot's fsLR32k LH+RH vertex ordering."""
    try:
        from yabplot.data import get_surface_paths
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Custom atlas support requires `yabplot` to be installed.") from exc

    img = nib.load(str(dlabel_path))
    cortex_labels: dict[str, np.ndarray] = {}
    for structure_name, structure_slice, _ in img.header.get_axis(1).iter_structures():
        if "CORTEX" not in structure_name:
            continue
        cortex_labels[structure_name] = np.asarray(img.dataobj[0, structure_slice], dtype=int).ravel()

    missing = {"CIFTI_STRUCTURE_CORTEX_LEFT", "CIFTI_STRUCTURE_CORTEX_RIGHT"} - cortex_labels.keys()
    if missing:
        raise ValueError(
            f"dlabel file {dlabel_path} is missing required cortical structures: {sorted(missing)}"
        )

    lh_mask_path, rh_mask_path = get_surface_paths("nomedialwall", "label")
    lh_mask = nib.load(lh_mask_path).darrays[0].data.astype(bool).ravel()
    rh_mask = nib.load(rh_mask_path).darrays[0].data.astype(bool).ravel()

    if lh_mask.sum() != len(cortex_labels["CIFTI_STRUCTURE_CORTEX_LEFT"]):
        raise ValueError(
            "Left cortex label count does not match yabplot fsLR32k nomedialwall mask: "
            f"{len(cortex_labels['CIFTI_STRUCTURE_CORTEX_LEFT'])} vs {lh_mask.sum()}"
        )
    if rh_mask.sum() != len(cortex_labels["CIFTI_STRUCTURE_CORTEX_RIGHT"]):
        raise ValueError(
            "Right cortex label count does not match yabplot fsLR32k nomedialwall mask: "
            f"{len(cortex_labels['CIFTI_STRUCTURE_CORTEX_RIGHT'])} vs {rh_mask.sum()}"
        )

    lh_vertices = np.zeros(len(lh_mask), dtype=int)
    rh_vertices = np.zeros(len(rh_mask), dtype=int)
    lh_vertices[lh_mask] = cortex_labels["CIFTI_STRUCTURE_CORTEX_LEFT"]
    rh_vertices[rh_mask] = cortex_labels["CIFTI_STRUCTURE_CORTEX_RIGHT"]
    return np.concatenate([lh_vertices, rh_vertices])


def _write_cortical_lut_from_tsv(
    tsv_path: Path,
    vertex_labels: np.ndarray,
    lut_path: Path,
) -> None:
    """Write a yabplot-compatible cortical LUT from a BIDS-style label TSV."""
    df = _read_label_tsv(tsv_path)
    surviving_ids = set(np.unique(vertex_labels).astype(int)) - {0}
    lines: list[str] = []
    for _, row in df.iterrows():
        region_id = int(row["index"])
        if region_id not in surviving_ids:
            continue
        clean_name = _clean_region_name(str(row["label"]))
        red = int(row["color_red"])
        green = int(row["color_green"])
        blue = int(row["color_blue"])
        lines.append(f"{region_id}  {clean_name}  {red}  {green}  {blue}  0")

    if not lines:
        raise ValueError(
            f"No regions from {tsv_path} survived mapping onto the fsLR32k surface."
        )

    lut_path.write_text("\n".join(lines), encoding="utf-8")
