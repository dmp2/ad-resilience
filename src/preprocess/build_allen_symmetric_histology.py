"""Build canonical bilateral Allen histology by exact physical reflection.

This creates the canonical ``histology_symmetric`` source layer from the
observed prepared-left derivative. It does not create EM-LDDMM views and does
not modify the observed derivative.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import shutil
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

LOG = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LEFT_DATASET = Path(
    "data/derivatives/allen/specimen_708424/histology_linear_nissl"
)
DEFAULT_ANNOTATIONS = Path(
    "data/derivatives/allen/specimen_708424/annotations_ome_zarr"
)
DEFAULT_OUTPUT = Path("data/derivatives/allen/specimen_708424/histology_symmetric")
DEFAULT_MASK_ROOT = Path("data/raw/allen/specimen_708424")
MRI_PROVENANCE = Path(
    "data/derivatives/allen/specimen_708424/mri_7t_whole/mri_provenance.json"
)
RAW_STRUCTURES = Path("data/raw/allen/specimen_708424/metadata/structures.tsv")
EXPECTED_COUNTS = {"nissl": 641, "pv": 287}
EXPECTED_ROWS = 2846
OBSERVED_LEFT = 1
SYNTHETIC_RIGHT = 2
PV_ENTROPY_RADIUS = 5
PV_CLOSING_RADIUS = 3
PV_HOLE_AREA = 64
PV_COMPONENT_FRACTION = 0.01
PV_COMPONENT_FLOOR = 25
PV_MIN_AREA_FRACTION = 0.01
PV_MAX_AREA_FRACTION = 0.85
PV_FIDUCIAL_SATURATION = 0.5
PV_FIDUCIAL_SATURATED_FRACTION = 0.9
PV_FIDUCIAL_DARK_CORE_FRACTION = 0.1
PV_NISSL_EDGE_TOLERANCE_PX = 2
PV_RGB_FIDUCIAL_MAX_AREA = 512
PV_RGB_FIDUCIAL_BOUNDARY_DISTANCE_PX = 3.0
PV_RGB_FIDUCIAL_NEIGHBORHOOD_RADIUS_PX = 6
PV_RGB_FIDUCIAL_ANNULUS_OUTER_RADIUS_PX = 10
PV_RGB_FIDUCIAL_COLOR_DISTANCE = 24.0


@dataclass(frozen=True)
class PVMaskResult:
    mask: np.ndarray
    entropy_image: np.ndarray
    texture_mask: np.ndarray
    darkness_mask: np.ndarray
    combined_mask: np.ndarray
    entropy_threshold: float
    darkness_threshold: float
    component_areas: tuple[int, ...]
    retained_component_areas: tuple[int, ...]
    retained_area: int
    retained_area_fraction: float
    medial_column: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    absolute = path.resolve()
    try:
        return absolute.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(absolute)


def _json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def apply_tissue_mask(
    image: np.ndarray, tissue_mask: np.ndarray, *, background: int = 0
) -> np.ndarray:
    """Replace all non-tissue RGB pixels with the prepared canvas background."""

    rgb = np.asarray(image)
    mask = np.asarray(tissue_mask, dtype=bool)
    if rgb.ndim != 3 or rgb.shape[:2] != mask.shape:
        raise ValueError(
            f"RGB and tissue-mask shapes differ: {rgb.shape} versus {mask.shape}"
        )
    cleaned = np.full_like(rgb, background)
    cleaned[mask] = rgb[mask]
    return cleaned


def remove_pv_fiducial_rgb(
    image: np.ndarray, tissue_mask: np.ndarray
) -> tuple[np.ndarray, int]:
    """Remove only small saturated components in the tissue boundary band."""

    from scipy.ndimage import binary_dilation, distance_transform_edt
    from skimage.measure import label
    from skimage.morphology import disk

    cleaned = np.asarray(image).copy()
    mask = np.asarray(tissue_mask, dtype=bool)
    saturated = (_rgb_saturation(cleaned) >= PV_FIDUCIAL_SATURATION) & mask
    labels = label(saturated, connectivity=2)
    distance = distance_transform_edt(mask)
    removed_components = 0
    for component_label in range(1, int(labels.max()) + 1):
        component = labels == component_label
        area = int(np.count_nonzero(component))
        if (
            area <= PV_RGB_FIDUCIAL_MAX_AREA
            and float(np.median(distance[component]))
            <= PV_RGB_FIDUCIAL_BOUNDARY_DISTANCE_PX
        ):
            neighborhood = binary_dilation(
                component, structure=disk(PV_RGB_FIDUCIAL_NEIGHBORHOOD_RADIUS_PX)
            ) & mask
            annulus = (
                binary_dilation(
                    component,
                    structure=disk(PV_RGB_FIDUCIAL_ANNULUS_OUTER_RADIUS_PX),
                )
                & mask
                & ~neighborhood
            )
            local_background = np.median(cleaned[annulus], axis=0) if np.any(annulus) else np.zeros(cleaned.shape[2], dtype=np.float64)
            color_distance = np.linalg.norm(
                cleaned.astype(np.float32) - local_background, axis=2
            )
            candidates = neighborhood & (
                color_distance >= PV_RGB_FIDUCIAL_COLOR_DISTANCE
            )
            candidates |= component
            candidate_labels = label(candidates, connectivity=2)
            seeded_labels = np.unique(candidate_labels[component])
            removal_mask = np.isin(candidate_labels, seeded_labels[seeded_labels != 0])
            cleaned[removal_mask] = np.rint(local_background).astype(cleaned.dtype)
            removed_components += 1
    return cleaned, removed_components


def shift_to_medial_edge(
    array: np.ndarray, shift_px: int, *, background: int = 0
) -> np.ndarray:
    """Shift an array left without interpolation and fill lateral space."""

    source = np.asarray(array)
    if source.ndim not in (2, 3) or source.shape[1] < 1:
        raise ValueError(f"Unsupported section array shape: {source.shape}")
    if not 0 <= shift_px < source.shape[1]:
        raise ValueError("Section shift is outside the source grid")
    shifted = np.full_like(source, background)
    if shift_px == 0:
        shifted[...] = source
    else:
        shifted[:, : source.shape[1] - shift_px, ...] = source[:, shift_px:, ...]
    return shifted


def bilateral_union(shifted_left: np.ndarray) -> np.ndarray:
    """Reflect a shifted fixed-size half and concatenate exactly."""

    array = np.asarray(shifted_left)
    if array.ndim not in (2, 3) or array.shape[1] < 1:
        raise ValueError(f"Unsupported section array shape: {array.shape}")
    return np.concatenate((array[:, ::-1, ...], array), axis=1)


def hemisphere_origin_mask(shape_yx: tuple[int, int]) -> np.ndarray:
    height, width = shape_yx
    if width % 2:
        raise ValueError("Between-column symmetry requires an even output width")
    mask = np.empty((height, width), dtype=np.uint8)
    mask[:, : width // 2] = OBSERVED_LEFT
    mask[:, width // 2 :] = SYNTHETIC_RIGHT
    return mask



def _rgb_saturation(rgb: np.ndarray) -> np.ndarray:
    values = np.asarray(rgb, dtype=np.float32) / 255.0
    maximum = values.max(axis=2)
    minimum = values.min(axis=2)
    saturation = np.zeros_like(maximum)
    nonzero = maximum > 0
    saturation[nonzero] = (maximum[nonzero] - minimum[nonzero]) / maximum[nonzero]
    return saturation


def generate_pv_mask(image: np.ndarray, *, section: int) -> PVMaskResult:
    """Segment PV tissue on the prepared RGB grid using entropy and darkness."""

    from skimage.color import rgb2gray
    from skimage.filters import rank, threshold_otsu
    from skimage.measure import label
    from skimage.morphology import closing, disk, remove_small_holes

    rgb = np.asarray(image)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"PV section {section} is not RGB: {rgb.shape}")
    gray = rgb2gray(rgb)
    gray_uint8 = np.rint(np.clip(gray, 0.0, 1.0) * 255.0).astype(np.uint8)
    darkness = 1.0 - gray
    entropy_image = rank.entropy(gray_uint8, disk(PV_ENTROPY_RADIUS))
    entropy_threshold = float(threshold_otsu(entropy_image))
    darkness_threshold = float(threshold_otsu(darkness))
    valid_canvas = np.any(rgb != 0, axis=2)
    texture_mask = (entropy_image > entropy_threshold) & valid_canvas
    darkness_mask = (darkness > darkness_threshold) & valid_canvas
    combined_mask = texture_mask | darkness_mask
    closed = closing(combined_mask, disk(PV_CLOSING_RADIUS))
    try:
        filled = remove_small_holes(closed, max_size=PV_HOLE_AREA - 1)
    except TypeError:  # scikit-image < 0.26
        filled = remove_small_holes(closed, area_threshold=PV_HOLE_AREA)
    filled &= valid_canvas

    labels = label(filled, connectivity=2)
    sizes = np.bincount(labels.ravel())[1:]
    if sizes.size == 0:
        raise ValueError(f"PV section {section} mask is empty")
    largest = int(sizes.max())
    minimum_area = max(PV_COMPONENT_FLOOR, math.ceil(PV_COMPONENT_FRACTION * largest))
    retained_labels = np.flatnonzero(sizes >= minimum_area) + 1
    saturation = _rgb_saturation(rgb)
    largest_label = int(np.argmax(sizes)) + 1
    fiducial_labels = []
    for component_label in retained_labels:
        component = labels == component_label
        dark_core = component & darkness_mask
        saturated_fraction = (
            float(np.mean(saturation[dark_core] >= PV_FIDUCIAL_SATURATION))
            if np.any(dark_core)
            else 0.0
        )
        dark_core_fraction = float(
            np.count_nonzero(dark_core) / sizes[component_label - 1]
        )
        if (
            saturated_fraction >= PV_FIDUCIAL_SATURATED_FRACTION
            and dark_core_fraction >= PV_FIDUCIAL_DARK_CORE_FRACTION
        ):
            if component_label == largest_label:
                raise ValueError(
                    f"PV section {section} mask consists only of a colored fiducial"
                )
            fiducial_labels.append(int(component_label))
    retained_labels = np.asarray(
        [value for value in retained_labels if value not in fiducial_labels]
    )
    mask = np.isin(labels, retained_labels)
    retained_areas = tuple(int(sizes[index - 1]) for index in retained_labels)
    retained_area = int(mask.sum())
    retained_fraction = retained_area / mask.size
    if not PV_MIN_AREA_FRACTION <= retained_fraction <= PV_MAX_AREA_FRACTION:
        raise ValueError(
            f"PV section {section} mask area is implausible: {retained_area} pixels "
            f"({retained_fraction:.6f} of prepared canvas; expected "
            f"{PV_MIN_AREA_FRACTION:.2f}-{PV_MAX_AREA_FRACTION:.2f})"
        )

    _, xx = np.nonzero(mask)
    if not xx.size:
        raise ValueError(f"PV section {section} mask has no usable medial column")
    return PVMaskResult(
        mask=mask,
        entropy_image=entropy_image,
        texture_mask=texture_mask,
        darkness_mask=darkness_mask,
        combined_mask=combined_mask,
        entropy_threshold=entropy_threshold,
        darkness_threshold=darkness_threshold,
        component_areas=tuple(sorted((int(value) for value in sizes), reverse=True)),
        retained_component_areas=tuple(sorted(retained_areas, reverse=True)),
        retained_area=retained_area,
        retained_area_fraction=retained_fraction,
        medial_column=int(xx.min()),
    )


def _nissl_edges_by_block(
    *,
    rows: list[dict[str, str]],
    left_dataset: Path,
    canvas_shape_yx: tuple[int, int],
) -> dict[str, list[tuple[int, int]]]:
    edges: dict[str, list[tuple[int, int]]] = {}
    for row in rows:
        if row["image_present"] != "true" or row["stain"] != "nissl":
            continue
        edge, _ = _section_shift_px(
            row=row, left_dataset=left_dataset, canvas_shape_yx=canvas_shape_yx
        )
        edges.setdefault(row["block_id"], []).append(
            (int(row["allen_section_number"]), edge)
        )
    for values in edges.values():
        values.sort()
    return edges


def _pv_nissl_lower_bound(
    *,
    section: int,
    block_id: str,
    nissl_edges: dict[str, list[tuple[int, int]]],
) -> int:
    values = nissl_edges.get(block_id, [])
    if not values:
        return 0
    before = [item for item in values if item[0] <= section]
    after = [item for item in values if item[0] >= section]
    if before and after:
        left = before[-1]
        right = after[0]
        if left[0] == right[0]:
            interpolated = float(left[1])
        else:
            fraction = (section - left[0]) / (right[0] - left[0])
            interpolated = left[1] + fraction * (right[1] - left[1])
    else:
        nearest = min(values, key=lambda item: abs(item[0] - section))
        interpolated = float(nearest[1])
    return max(0, math.floor(interpolated) - PV_NISSL_EDGE_TOLERANCE_PX)


def _constrain_pv_medial_support(
    result: PVMaskResult, *, lower_bound_px: int, section: int
) -> tuple[PVMaskResult, int]:
    from skimage.measure import label

    mask = result.mask.copy()
    before_area = int(mask.sum())
    mask[:, :lower_bound_px] = False
    labels = label(mask, connectivity=2)
    sizes = np.bincount(labels.ravel())[1:]
    _, xx = np.nonzero(mask)
    if not xx.size:
        raise ValueError(
            f"PV section {section} has no support after Nissl edge constraint"
        )
    retained_area = int(mask.sum())
    corrected = PVMaskResult(
        mask=mask,
        entropy_image=result.entropy_image,
        texture_mask=result.texture_mask,
        darkness_mask=result.darkness_mask,
        combined_mask=result.combined_mask,
        entropy_threshold=result.entropy_threshold,
        darkness_threshold=result.darkness_threshold,
        component_areas=result.component_areas,
        retained_component_areas=tuple(
            sorted((int(value) for value in sizes if value), reverse=True)
        ),
        retained_area=retained_area,
        retained_area_fraction=retained_area / mask.size,
        medial_column=int(xx.min()),
    )
    return corrected, before_area - retained_area


def _pv_metrics(section: int, result: PVMaskResult) -> dict[str, Any]:
    return {
        "section_number": section,
        "entropy_threshold": result.entropy_threshold,
        "darkness_threshold": result.darkness_threshold,
        "component_areas": list(result.component_areas),
        "retained_component_areas": list(result.retained_component_areas),
        "retained_area": result.retained_area,
        "retained_area_fraction": result.retained_area_fraction,
        "medial_column": result.medial_column,
        "qc_passed": True,
    }


def _qc_rgb(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array)
    if values.dtype == bool:
        gray = values.astype(np.uint8) * 255
    else:
        finite = values[np.isfinite(values)]
        if not finite.size or float(finite.max()) == float(finite.min()):
            gray = np.zeros(values.shape, dtype=np.uint8)
        else:
            scaled = (values - finite.min()) / (finite.max() - finite.min())
            gray = np.rint(np.clip(scaled, 0.0, 1.0) * 255.0).astype(np.uint8)
    return np.repeat(gray[..., None], 3, axis=2)


def _write_pv_qc_montage(
    path: Path,
    samples: list[tuple[int, np.ndarray, PVMaskResult, np.ndarray]],
) -> None:
    import matplotlib.pyplot as plt

    titles = (
        "Original PV", "Entropy", "Entropy threshold", "Darkness threshold",
        "Combined mask", "Cleaned mask over RGB", "Final symmetric image",
    )
    figure, axes = plt.subplots(
        len(samples), len(titles), figsize=(21, 3.4 * len(samples)), squeeze=False
    )
    for row_index, (section, original, result, bilateral) in enumerate(samples):
        overlay = original.copy()
        overlay[~result.mask] = np.rint(0.25 * overlay[~result.mask]).astype(np.uint8)
        panels = (
            original, _qc_rgb(result.entropy_image), _qc_rgb(result.texture_mask),
            _qc_rgb(result.darkness_mask), _qc_rgb(result.combined_mask), overlay, bilateral,
        )
        for column, (axis, title, panel) in enumerate(zip(axes[row_index], titles, panels, strict=True)):
            axis.imshow(panel)
            axis.set_axis_off()
            axis.set_title(title if row_index == 0 else "")
            if column == 0:
                axis.set_ylabel(f"PV section {section}")
    figure.tight_layout()
    figure.savefig(path, dpi=120, facecolor="white")
    plt.close(figure)

def _read_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        return list(reader), list(reader.fieldnames or ())


def _write_rows(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=fieldnames,
            delimiter="\t",
            lineterminator="\n",
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def _sidecar_geometry(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = ("Sizes", "SpaceDirections", "SpaceOrigin", "SpaceUnits")
    if any(key not in payload for key in required):
        raise ValueError(f"Incomplete loader sidecar: {path}")
    return payload


def _registered_left_half_space(
    source_shape_yx: tuple[int, int],
    source_geometry: dict[str, Any],
    affine: np.ndarray,
    mri_midline_um: float,
) -> dict[str, Any]:
    """Select the established observed-left side of the registered grid."""

    height, width = source_shape_yx
    directions = source_geometry["directions"]
    spacing = float(np.linalg.norm(directions[1]))
    transform = np.asarray(affine, dtype=np.float64)
    if (
        transform.shape != (4, 4)
        or not np.allclose(transform[2, 1:3], 0.0, atol=1e-6)
    ):
        raise ValueError("MRI midsagittal plane is not column-aligned in the restack")
    if spacing <= 0.0 or not np.allclose(
        directions[1], [spacing, 0.0, 0.0]
    ):
        raise ValueError("Corrected restack column direction is not increasing")
    origin = float(source_geometry["origin_xy"][0])
    centers = origin + np.arange(width, dtype=np.float64) * spacing
    edges = origin + (np.arange(width + 1, dtype=np.float64) - 0.5) * spacing
    midline = float(transform[2, 0] * mri_midline_um + transform[2, 3])
    boundary_index = int(np.argmin(np.abs(edges - midline)))
    discrepancy = float(edges[boundary_index] - midline)
    if abs(discrepancy) >= spacing / 2.0:
        raise ValueError(
            "MRI midline is not within half a source column of a cell edge: "
            f"{discrepancy:.6g} um ({discrepancy / spacing:.6g} pixels)"
        )
    if not 0 < boundary_index < width:
        raise ValueError("MRI midline leaves no valid bilateral source half")
    # Existing Allen/bilateral_union convention retains the source unchanged in
    # the increasing-column half and prepends its exact reflected copy.
    retained = centers[boundary_index:]
    return {
        "first_column": boundary_index,
        "column_coordinates_um": retained,
        "boundary_um": float(edges[boundary_index]),
        "midline_um": midline,
        "boundary_discrepancy_um": discrepancy,
        "source_shape_yx": (height, len(retained)),
    }


def _validate_left_source(
    left_dataset: Path,
) -> tuple[list[dict[str, str]], list[str], dict[str, Any]]:
    dataset = json.loads((left_dataset / "dataset.json").read_text())
    rows, fields = _read_rows(left_dataset / "metadata" / "physical_sections.tsv")
    if len(rows) != EXPECTED_ROWS:
        raise ValueError(f"Observed derivative has {len(rows)} positions")
    corrected = dataset.get("space_name") == "HIST_LINEAR_NISSL"
    present = [row for row in rows if row["image_present"] == "true"]
    counts = Counter(row["stain"] for row in present)
    expected_counts = (
        {"nissl": EXPECTED_COUNTS["nissl"]} if corrected else EXPECTED_COUNTS
    )
    if counts != Counter(expected_counts):
        raise ValueError(f"Observed derivative stain counts differ: {counts}")
    if [int(row["allen_section_number"]) for row in rows] != list(range(36, 2882)):
        raise ValueError("Observed derivative does not use the 2,846-row lattice")
    if not corrected and any(float(row["serial_pitch_um"]) != 50.0 for row in rows):
        raise ValueError("Observed derivative does not use 50-um serial pitch")

    expected_shape = tuple(dataset["prepared_canvas_shape_yx"])
    reference: dict[str, Any] | None = None
    for row in present:
        image_path = left_dataset / row["prepared_relative_path"]
        if sha256_file(image_path) != row["prepared_sha256"]:
            raise ValueError(f"Observed image checksum mismatch: {image_path}")
        with Image.open(image_path) as image:
            if image.size != (expected_shape[1], expected_shape[0]):
                raise ValueError(f"Observed image is off the common grid: {image_path}")
        sidecar = _sidecar_geometry(image_path.with_suffix(".json"))
        geometry = {
            "sizes": sidecar["Sizes"][:3],
            "directions": sidecar["SpaceDirections"][:3],
            "origin_xy": sidecar["SpaceOrigin"][:2],
            "units": sidecar["SpaceUnits"],
        }
        if reference is None:
            reference = geometry
        elif geometry != reference:
            raise ValueError(f"Observed sidecar grid differs: {image_path}")
    if reference is None:
        raise ValueError("Observed derivative contains no images")
    if corrected:
        directions = reference["directions"]
        spacing = float(np.linalg.norm(directions[1]))
        if (
            not np.allclose(directions[1], [spacing, 0.0, 0.0])
            or not np.allclose(directions[2], [0.0, spacing, 0.0])
        ):
            raise ValueError("Corrected restack is not on an LR-aligned section grid")
        audit = json.loads(
            (left_dataset / "metadata" / "linear_restack.json").read_text()
        )
        affine = np.asarray(
            audit["global_affine_mri_um_to_histology_um"], dtype=float
        )
        mri = json.loads((PROJECT_ROOT / MRI_PROVENANCE).read_text())
        half_space = _registered_left_half_space(
            expected_shape,
            reference,
            affine,
            float(mri["physical_center_mm"][0]) * 1000.0,
        )
    else:
        half_space = None
    return rows, fields, {
        "dataset": dataset,
        "geometry": reference,
        "corrected": corrected,
        "half_space": half_space,
    }


def _symmetric_geometry(
    source_shape_yx: tuple[int, int],
    source_geometry: dict[str, Any],
    *,
    reflection_plane_um: float = 0.0,
) -> dict[str, Any]:
    height, half_width = source_shape_yx
    x_direction = source_geometry["directions"][1]
    y_direction = source_geometry["directions"][2]
    spacing_x = float(np.linalg.norm(x_direction))
    spacing_y = float(np.linalg.norm(y_direction))
    if not np.isclose(spacing_x, spacing_y) or spacing_x <= 0:
        raise ValueError("Observed source requires an isotropic valid in-plane grid")
    width = 2 * half_width
    origin_x = reflection_plane_um - (half_width - 0.5) * spacing_x
    return {
        "source_shape_yx": [height, half_width],
        "bilateral_shape_yx": [height, width],
        "pixel_size_um": spacing_x,
        "bilateral_origin_xy_um": [
            origin_x,
            float(source_geometry["origin_xy"][1]),
        ],
        "reflection_plane": {
            "axis": "x",
            "coordinate_um": reflection_plane_um,
            "location": "between_columns",
            "adjacent_column_indices": [half_width - 1, half_width],
        },
        "source_space": "HIST_OBSERVED_PREPARED_LEFT",
        "symmetric_space": "HIST_SYMMETRIC",
        "reflection_matrix_um": [
            [-1.0, 0.0, 0.0, 2.0 * reflection_plane_um],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "hemisphere_origin_values": {
            "1": "observed_left",
            "2": "synthetically_reflected_right",
        },
    }


def _write_image_sidecar(
    path: Path,
    *,
    shape_yx: tuple[int, int],
    origin_xy_um: tuple[float, float],
    z_um: float,
    pixel_size_um: float,
) -> None:
    height, width = shape_yx
    _json(
        path.with_suffix(".json"),
        {
            "DataFile": path.name,
            "Type": "uint8",
            "Dimension": 4,
            "Sizes": [3, width, height, 1],
            "Endian": "big",
            "Space": "right-inferior-posterior",
            "SpaceDimension": 3,
            "SpaceUnits": ["um", "um", "um"],
            "SpaceDirections": [
                "none",
                [pixel_size_um, 0.0, 0.0],
                [0.0, pixel_size_um, 0.0],
                [0.0, 0.0, 50.0],
            ],
            "SpaceOrigin": [*origin_xy_um, z_um],
        },
    )


def _read_zarr_v3_uint32(array_dir: Path) -> np.ndarray:
    import zstandard

    metadata = json.loads((array_dir / "zarr.json").read_text())
    if metadata.get("zarr_format") != 3 or metadata.get("data_type") != "uint32":
        raise ValueError(f"Unsupported categorical Zarr array: {array_dir}")
    shape = tuple(int(value) for value in metadata["shape"])
    chunks = tuple(
        int(value) for value in metadata["chunk_grid"]["configuration"]["chunk_shape"]
    )
    result = np.zeros(shape, dtype=np.uint32)
    decompressor = zstandard.ZstdDecompressor()
    for cy in range(math.ceil(shape[0] / chunks[0])):
        for cx in range(math.ceil(shape[1] / chunks[1])):
            chunk_path = array_dir / "c" / str(cy) / str(cx)
            if not chunk_path.is_file():
                continue
            y0, x0 = cy * chunks[0], cx * chunks[1]
            chunk_shape = (
                min(chunks[0], shape[0] - y0),
                min(chunks[1], shape[1] - x0),
            )
            decoded = decompressor.decompress(
                chunk_path.read_bytes(),
                max_output_size=int(np.prod(chunks)) * 4,
            )
            values = np.frombuffer(decoded, dtype="<u4")
            if values.size == int(np.prod(chunks)):
                chunk_array = values.reshape(chunks)[: chunk_shape[0], : chunk_shape[1]]
            elif values.size == int(np.prod(chunk_shape)):
                chunk_array = values.reshape(chunk_shape)
            else:
                raise ValueError(f"Unexpected Zarr chunk shape: {chunk_path}")
            result[y0 : y0 + chunk_shape[0], x0 : x0 + chunk_shape[1]] = chunk_array
    return result


def _project_annotation(
    source: np.ndarray,
    transform: dict[str, Any],
    canvas_shape_yx: tuple[int, int],
) -> np.ndarray:
    matrix = np.asarray(transform["prepared_to_source_pixel"], dtype=np.float64)
    content_height, content_width = transform["prepared_content_shape_yx"]
    yy = np.arange(content_height, dtype=np.float64)
    xx = np.arange(content_width, dtype=np.float64)
    source_y = np.rint(matrix[1, 1] * yy + matrix[1, 2]).astype(np.int64)
    source_x = np.rint(matrix[0, 0] * xx + matrix[0, 2]).astype(np.int64)
    if (
        source_y.min(initial=0) < 0
        or source_x.min(initial=0) < 0
        or source_y.max(initial=0) >= source.shape[0]
        or source_x.max(initial=0) >= source.shape[1]
    ):
        raise ValueError("Annotation nearest-neighbor projection leaves source grid")
    left = np.zeros(canvas_shape_yx, dtype=np.uint32)
    left[:content_height, :content_width] = source[np.ix_(source_y, source_x)]
    return left


def _section_shift_px(
    *,
    row: dict[str, str],
    left_dataset: Path,
    canvas_shape_yx: tuple[int, int],
) -> tuple[int, np.ndarray]:
    """Project the established raw tissue mask and return its medial edge."""

    section = int(row["allen_section_number"])
    series_root = "nissl" if row["stain"] == "nissl" else "ihc"
    mask_path = PROJECT_ROOT / DEFAULT_MASK_ROOT / series_root / "masks_orig" / (
        f"mask_{section:04d}.png"
    )
    if not mask_path.is_file():
        raise FileNotFoundError(f"Missing established tissue mask: {mask_path}")
    prepared_path = left_dataset / row["prepared_relative_path"]
    transform = json.loads(
        prepared_path.with_name(
            f"{prepared_path.stem}_prepared-to-source.json"
        ).read_text(encoding="utf-8")
    )
    with Image.open(mask_path) as image:
        source_mask = np.asarray(image) != 0
    prepared_mask = (
        _project_annotation(source_mask.astype(np.uint32), transform, canvas_shape_yx)
        != 0
    )
    _, xx = np.nonzero(prepared_mask)
    if not xx.size:
        raise ValueError(f"Tissue mask is empty after projection: {mask_path}")
    return int(xx.min()), prepared_mask


def _build_annotations(
    *,
    annotation_source: Path,
    left_dataset: Path,
    staging: Path,
    rows_by_section: dict[int, dict[str, str]],
    canvas_shape_yx: tuple[int, int],
    section_shifts_px: dict[int, int],
) -> dict[str, Any]:
    import tifffile

    annotation_dataset = json.loads(
        (annotation_source / "dataset.json").read_text(encoding="utf-8")
    )
    manifest_rows, _ = _read_rows(annotation_source / "metadata" / "manifest.tsv")
    inventory: list[dict[str, Any]] = []
    group_counts: Counter[int] = Counter()
    annotations_root = staging / "annotations"
    annotations_root.mkdir()
    for manifest_row in manifest_rows:
        section = int(manifest_row["section_number"])
        histology_row = rows_by_section.get(section)
        if histology_row is None or histology_row["stain"] != "nissl":
            raise ValueError(
                f"Annotation section lacks matching Nissl image: {section}"
            )
        prepared_path = left_dataset / histology_row["prepared_relative_path"]
        transform_path = prepared_path.with_name(
            f"{prepared_path.stem}_prepared-to-source.json"
        )
        transform = json.loads(transform_path.read_text(encoding="utf-8"))
        section_root = annotation_source / manifest_row["path"] / "labels"
        group_ids = json.loads(manifest_row["graphic_groups_present"])
        for group_id in group_ids:
            source_array_path = section_root / f"group-{group_id}" / "0"
            source_labels = _read_zarr_v3_uint32(source_array_path)
            projected = _project_annotation(
                source_labels, transform, canvas_shape_yx
            )
            shifted = shift_to_medial_edge(
                projected, section_shifts_px[section], background=0
            )
            labels = bilateral_union(shifted)
            group_dir = annotations_root / f"group-{group_id}"
            group_dir.mkdir(exist_ok=True)
            output_path = group_dir / f"section-{section:04d}.tif"
            tifffile.imwrite(output_path, labels, compression="deflate")
            with tifffile.TiffFile(output_path) as image:
                reloaded = image.asarray()
            if not np.array_equal(reloaded, labels):
                raise ValueError(
                    f"Categorical TIFF round trip changed labels: {output_path}"
                )
            inventory.append(
                {
                    "section_number": section,
                    "graphic_group_id": int(group_id),
                    "path": output_path.relative_to(staging).as_posix(),
                    "sha256": sha256_file(output_path),
                    "sampling": "categorical_nearest_neighbor",
                    "right_origin": "synthetically_reflected",
                }
            )
            group_counts[int(group_id)] += 1
    fields = list(inventory[0]) if inventory else []
    with (staging / "metadata" / "annotations.tsv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(
            stream, fieldnames=fields, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(inventory)
    structures = PROJECT_ROOT / RAW_STRUCTURES
    if sha256_file(structures) != annotation_dataset["source"]["raw_structures_sha256"]:
        raise ValueError("Allen structure hierarchy checksum mismatch")
    shutil.copy2(structures, annotations_root / "structures.tsv")
    return {
        "annotation_section_count": len(manifest_rows),
        "annotation_image_count": len(inventory),
        "graphic_group_counts": {
            str(key): value for key, value in sorted(group_counts.items())
        },
        "structures_sha256": sha256_file(annotations_root / "structures.tsv"),
        "sampling": "categorical_nearest_neighbor",
        "ontology_ids_changed": False,
        "annotations_cogridded": True,
    }


def build_symmetric_histology(
    *,
    left_dataset: Path,
    output_dir: Path,
    annotation_source: Path | None = DEFAULT_ANNOTATIONS,
    overwrite: bool = False,
) -> dict[str, Any]:
    left_dataset = left_dataset.resolve()
    output_dir = output_dir.resolve()
    rows, fields, source = _validate_left_source(left_dataset)
    corrected = source["corrected"]
    full_source_shape = tuple(source["dataset"]["prepared_canvas_shape_yx"])
    half_space = source["half_space"]
    source_shape = (
        tuple(half_space["source_shape_yx"])
        if corrected else full_source_shape
    )
    source_geometry = dict(source["geometry"])
    if corrected:
        source_geometry["origin_xy"] = [
            float(half_space["column_coordinates_um"][0]),
            float(source_geometry["origin_xy"][1]),
        ]
    rows_by_section = {int(row["allen_section_number"]): row for row in rows}
    nissl_edges_by_block = {} if corrected else _nissl_edges_by_block(
        rows=rows, left_dataset=left_dataset, canvas_shape_yx=source_shape
    )
    symmetry = _symmetric_geometry(
        source_shape,
        source_geometry,
        reflection_plane_um=(
            float(half_space["boundary_um"]) if corrected else 0.0
        ),
    )
    bilateral_shape = tuple(symmetry["bilateral_shape_yx"])
    pv_sections = [
        int(row["allen_section_number"])
        for row in rows
        if row["image_present"] == "true" and row["stain"] == "pv"
    ]
    qc_sections = (
        {pv_sections[0], pv_sections[len(pv_sections) // 2], pv_sections[-1]}
        if pv_sections else set()
    )
    if output_dir.exists() and not overwrite:
        raise FileExistsError(f"Symmetric source exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        (staging / "metadata").mkdir()
        images_root = staging / "images"
        images_root.mkdir()
        output_rows: list[dict[str, str]] = []
        section_shifts_px: dict[int, int] = {}
        pv_section_metrics: list[dict[str, Any]] = []
        pv_before_shifts: list[int] = []
        pv_after_shifts: list[int] = []
        pv_before_areas: list[int] = []
        pv_removed_areas: list[int] = []
        pv_fiducial_sections_before = 0
        pv_fiducial_sections_after = 0
        pv_qc_samples: list[tuple[int, np.ndarray, PVMaskResult, np.ndarray]] = []
        for original_row in rows:
            row = dict(original_row)
            if row["image_present"] == "true":
                source_path = left_dataset / row["prepared_relative_path"]
                with Image.open(source_path) as image:
                    observed = np.asarray(image.convert("RGB"))
                if corrected:
                    observed = observed[:, half_space["first_column"] :, :]
                    if observed.shape[:2] != source_shape:
                        raise ValueError("Corrected half-space slice changed grid shape")
                section = int(row["allen_section_number"])
                pv_result: PVMaskResult | None = None
                if corrected:
                    cleaned = observed
                    shift_px = 0
                    tissue_mask = np.any(observed != 0, axis=2)
                elif row["stain"] == "pv":
                    unconstrained = generate_pv_mask(observed, section=section)
                    lower_bound = _pv_nissl_lower_bound(
                        section=section,
                        block_id=row["block_id"],
                        nissl_edges=nissl_edges_by_block,
                    )
                    pv_result, removed_area = _constrain_pv_medial_support(
                        unconstrained,
                        lower_bound_px=lower_bound,
                        section=section,
                    )
                    shift_px, tissue_mask = pv_result.medial_column, pv_result.mask
                    pv_before_shifts.append(unconstrained.medial_column)
                    pv_after_shifts.append(shift_px)
                    pv_before_areas.append(unconstrained.retained_area)
                    pv_removed_areas.append(removed_area)
                    pv_section_metrics.append(_pv_metrics(section, pv_result))
                else:
                    shift_px, tissue_mask = _section_shift_px(
                        row=row,
                        left_dataset=left_dataset,
                        canvas_shape_yx=source_shape,
                    )
                if not corrected:
                    cleaned = apply_tissue_mask(
                        observed, tissue_mask, background=0
                    )
                if pv_result is not None:
                    cleaned, removed_components = remove_pv_fiducial_rgb(
                        cleaned, tissue_mask
                    )
                    pv_fiducial_sections_before += int(removed_components > 0)
                    _, remaining_components = remove_pv_fiducial_rgb(
                        cleaned, tissue_mask
                    )
                    pv_fiducial_sections_after += int(remaining_components > 0)
                shifted = cleaned if corrected else shift_to_medial_edge(
                    cleaned, shift_px, background=0
                )
                shifted_mask = tissue_mask if corrected else shift_to_medial_edge(
                    tissue_mask, shift_px, background=0
                )
                if not corrected and (
                    np.count_nonzero(shifted_mask) != np.count_nonzero(tissue_mask)
                    or not np.any(shifted_mask[:, 0])
                    or np.any(shifted[~shifted_mask] != 0)
                ):
                    raise ValueError(
                        f"Tissue support changed during shift for section {section}"
                    )
                section_shifts_px[section] = shift_px
                bilateral = bilateral_union(shifted)
                if bilateral.shape[:2] != bilateral_shape:
                    raise ValueError("Derived bilateral image shape is inconsistent")
                if corrected and not np.array_equal(
                    bilateral[:, source_shape[1]:], observed
                ):
                    raise ValueError("Corrected observed pixels changed during union")
                if pv_result is not None and section in qc_sections:
                    pv_qc_samples.append((section, observed, pv_result, bilateral))
                stain_dir = images_root / row["stain"]
                stain_dir.mkdir(exist_ok=True)
                output_path = stain_dir / source_path.name
                Image.fromarray(bilateral).save(
                    output_path, format="TIFF", compression="tiff_deflate"
                )
                with Image.open(output_path) as reloaded:
                    if not np.array_equal(np.asarray(reloaded), bilateral):
                        raise ValueError(
                            f"Bilateral TIFF changed pixels: {output_path}"
                        )
                _write_image_sidecar(
                    output_path,
                    shape_yx=bilateral_shape,
                    origin_xy_um=tuple(symmetry["bilateral_origin_xy_um"]),
                    z_um=float(row["serial_z_center_mm"]) * 1000.0,
                    pixel_size_um=float(symmetry["pixel_size_um"]),
                )
                row["prepared_relative_path"] = output_path.relative_to(
                    staging
                ).as_posix()
                row["prepared_sha256"] = sha256_file(output_path)
            output_rows.append(row)

        if not corrected:
            LOG.info(
                "Section medial shifts: min=%d px max=%d px n=%d",
                min(section_shifts_px.values()),
                max(section_shifts_px.values()),
                len(section_shifts_px),
            )
            LOG.info(
                "PV medial shifts before=%d..%d px after=%d..%d px; "
                "column-0 masks before=%d after=%d",
                min(pv_before_shifts),
                max(pv_before_shifts),
                min(pv_after_shifts),
                max(pv_after_shifts),
                sum(value == 0 for value in pv_before_shifts),
                sum(value == 0 for value in pv_after_shifts),
            )
            LOG.info(
                "PV retained-mask area removed=%d of %d pixels (%.6f)",
                sum(pv_removed_areas),
                sum(pv_before_areas),
                sum(pv_removed_areas) / sum(pv_before_areas),
            )
            LOG.info(
                "PV sections with fiducial-like RGB components before=%d after=%d",
                pv_fiducial_sections_before,
                pv_fiducial_sections_after,
            )

        _write_rows(staging / "metadata" / "physical_sections.tsv", output_rows, fields)
        mask = hemisphere_origin_mask(bilateral_shape)
        Image.fromarray(mask).save(
            staging / "metadata" / "hemisphere_origin_mask.tif",
            format="TIFF",
            compression="tiff_deflate",
        )
        pv_segmentation = None
        if pv_sections:
            qc_root = staging / "qc"
            qc_root.mkdir()
            qc_path = qc_root / "pv_segmentation_montage.png"
            pv_qc_samples.sort(key=lambda item: item[0])
            if [item[0] for item in pv_qc_samples] != sorted(qc_sections):
                raise ValueError("PV QC representatives were not generated")
            _write_pv_qc_montage(qc_path, pv_qc_samples)
            pv_segmentation = {
                "method": "prepared_grid_local_entropy_and_darkness_otsu",
                "nissl_mask_method_changed": False,
                "parameters": {
                    "exclude_exact_black_canvas": True,
                    "entropy_radius": PV_ENTROPY_RADIUS,
                    "closing_radius": PV_CLOSING_RADIUS,
                    "hole_area_threshold": PV_HOLE_AREA,
                    "component_fraction_of_largest": PV_COMPONENT_FRACTION,
                    "component_area_floor": PV_COMPONENT_FLOOR,
                    "minimum_mask_area_fraction": PV_MIN_AREA_FRACTION,
                    "maximum_mask_area_fraction": PV_MAX_AREA_FRACTION,
                    "fiducial_saturation_threshold": PV_FIDUCIAL_SATURATION,
                    "fiducial_saturated_fraction": PV_FIDUCIAL_SATURATED_FRACTION,
                },
                "section_metrics": pv_section_metrics,
                "qc_montage": {
                    "path": qc_path.relative_to(staging).as_posix(),
                    "sha256": sha256_file(qc_path),
                    "section_numbers": sorted(qc_sections),
                    "panels": [
                        "original_pv", "entropy_image", "entropy_threshold",
                        "darkness_threshold", "combined_mask",
                        "cleaned_mask_over_rgb", "final_symmetric_image",
                    ],
                },
            }
        symmetry.update(
            {
                "schema_version": 1,
                "specimen_id": "708424",
                "source_dataset": _display_path(left_dataset),
                "source_dataset_json_sha256": sha256_file(
                    left_dataset / "dataset.json"
                ),
                "source_physical_sections_sha256": sha256_file(
                    left_dataset / "metadata" / "physical_sections.tsv"
                ),
                "hemisphere_origin_mask": "metadata/hemisphere_origin_mask.tif",
                "hemisphere_origin_mask_sha256": sha256_file(
                    staging / "metadata" / "hemisphere_origin_mask.tif"
                ),
                "image_operation": "exact_reflection_and_union_without_interpolation",
                "reflected_side_is_independent_evidence": False,
                "pv_segmentation": pv_segmentation,
            }
        )
        _json(staging / "metadata" / "symmetry.json", symmetry)
        annotation_summary = (
            _build_annotations(
                annotation_source=annotation_source.resolve(),
                left_dataset=left_dataset,
                staging=staging,
                rows_by_section=rows_by_section,
                canvas_shape_yx=source_shape,
                section_shifts_px=section_shifts_px,
            )
            if annotation_source is not None and not corrected
            else {"annotations_cogridded": False}
        )
        compatibility = {
            "schema_version": 1,
            "compatible_with_emlddmm_preparation": True,
            "source_grid_preservable": True,
            "requires_preparation_resampling": False,
            "physical_position_count": len(output_rows),
            "present_image_count": sum(
                row["image_present"] == "true" for row in output_rows
            ),
            "nissl_count": sum(
                row["stain"] == "nissl" and row["image_present"] == "true"
                for row in output_rows
            ),
            "pv_count": sum(
                row["stain"] == "pv" and row["image_present"] == "true"
                for row in output_rows
            ),
            "common_bilateral_grid": True,
            "annotations_cogridded": annotation_summary["annotations_cogridded"],
            "intended_registration_source": "symmetric_whole_brain_histology",
            "intended_registration_target": "whole_brain_7T_MRI",
            "recommended_registration_strategy": (
                "direct_symmetric_histology_to_whole_brain_7T"
            ),
        }
        _json(staging / "metadata" / "compatibility.json", compatibility)
        dataset = {
            "schema_version": 1,
            "dataset": "Allen specimen 708424 canonical symmetric histology source",
            "specimen_id": 708424,
            "space_name": "HIST_SYMMETRIC",
            "source_layer": _display_path(left_dataset),
            "physical_position_count": EXPECTED_ROWS,
            "present_image_count": sum(
                row["image_present"] == "true" for row in output_rows
            ),
            "nissl_count": compatibility["nissl_count"],
            "pv_count": compatibility["pv_count"],
            "serial_pitch_um": 50.0,
            "pixel_size_um": symmetry["pixel_size_um"],
            "bilateral_shape_yx": symmetry["bilateral_shape_yx"],
            "reflection_plane_coordinate_um": symmetry["reflection_plane"]["coordinate_um"],
            "pixels_resampled": False,
            "images_reflected_once": True,
            "annotation_summary": annotation_summary,
        }
        _json(staging / "dataset.json", dataset)
        if output_dir.exists():
            backup = staging.with_name(f"{staging.name}.previous")
            os.replace(output_dir, backup)
            try:
                os.replace(staging, output_dir)
            except BaseException:
                os.replace(backup, output_dir)
                raise
            shutil.rmtree(backup)
        else:
            os.replace(staging, output_dir)
        return dataset
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def verify_symmetric_histology(dataset_root: Path) -> dict[str, Any]:
    dataset_root = dataset_root.resolve()
    dataset = json.loads((dataset_root / "dataset.json").read_text())
    symmetry = json.loads((dataset_root / "metadata" / "symmetry.json").read_text())
    compatibility = json.loads(
        (dataset_root / "metadata" / "compatibility.json").read_text()
    )
    if (
        dataset.get("space_name") != "HIST_SYMMETRIC"
        or dataset.get("physical_position_count") != EXPECTED_ROWS
        or dataset.get("present_image_count")
        != dataset.get("nissl_count", 0) + dataset.get("pv_count", 0)
        or compatibility.get("source_grid_preservable") is not True
        or compatibility.get("requires_preparation_resampling") is not False
    ):
        raise ValueError("Symmetric source compatibility metadata is invalid")
    rows, _ = _read_rows(dataset_root / "metadata" / "physical_sections.tsv")
    if len(rows) != EXPECTED_ROWS:
        raise ValueError("Symmetric source does not retain 2,846 serial positions")
    expected_shape = tuple(symmetry["bilateral_shape_yx"])
    counts: Counter[str] = Counter()
    for row in rows:
        if row["image_present"] != "true":
            continue
        counts[row["stain"]] += 1
        path = dataset_root / row["prepared_relative_path"]
        if sha256_file(path) != row["prepared_sha256"]:
            raise ValueError(f"Symmetric image checksum mismatch: {path}")
        with Image.open(path) as image:
            if image.size != (expected_shape[1], expected_shape[0]):
                raise ValueError(f"Symmetric image is off-grid: {path}")
    expected_counts = Counter(
        nissl=dataset.get("nissl_count", 0), pv=dataset.get("pv_count", 0)
    )
    expected_counts += Counter()
    if counts != expected_counts:
        raise ValueError(f"Symmetric source stain counts differ: {counts}")
    mask_path = dataset_root / symmetry["hemisphere_origin_mask"]
    if sha256_file(mask_path) != symmetry["hemisphere_origin_mask_sha256"]:
        raise ValueError("Hemisphere-origin mask checksum mismatch")
    with Image.open(mask_path) as image:
        mask = np.asarray(image)
    if not np.array_equal(mask, hemisphere_origin_mask(expected_shape)):
        raise ValueError("Hemisphere-origin mask semantics changed")
    if dataset.get("pv_count", 0):
        pv_segmentation = symmetry.get("pv_segmentation")
        if not isinstance(pv_segmentation, dict):
            raise ValueError("PV segmentation metadata is missing")
        qc = pv_segmentation.get("qc_montage", {})
        qc_path = dataset_root / qc.get("path", "")
        if not qc_path.is_file() or sha256_file(qc_path) != qc.get("sha256"):
            raise ValueError("PV segmentation QC montage checksum mismatch")
        if len(pv_segmentation.get("section_metrics", [])) != dataset["pv_count"]:
            raise ValueError("PV segmentation metrics do not cover every PV section")
    inventory = dataset_root / "metadata" / "annotations.tsv"
    if inventory.is_file():
        annotation_rows, _ = _read_rows(inventory)
        if len(annotation_rows) != dataset["annotation_summary"]["annotation_image_count"]:
            raise ValueError("Symmetric annotation inventory count changed")
        for row in annotation_rows:
            path = dataset_root / row["path"]
            if sha256_file(path) != row["sha256"]:
                raise ValueError(f"Symmetric annotation checksum mismatch: {path}")
    return compatibility


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-dataset", type=Path, default=DEFAULT_LEFT_DATASET)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verify-existing", action="store_true")
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    logging.basicConfig(level=args.log_level, format="%(levelname)s %(message)s")
    if args.verify_existing:
        compatibility = verify_symmetric_histology(args.output_dir)
        LOG.info(
            "Validated symmetric source for %s",
            compatibility["recommended_registration_strategy"],
        )
    else:
        dataset = build_symmetric_histology(
            left_dataset=args.left_dataset,
            output_dir=args.output_dir,
            annotation_source=args.annotations,
            overwrite=args.overwrite,
        )
        LOG.info(
            "Built %s bilateral sections on %s",
            dataset["present_image_count"],
            dataset["bilateral_shape_yx"],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
