"""Render compact QC for original or prepared Allen mixed-stain stacks."""

from __future__ import annotations

import argparse
import csv
import io
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np
from PIL import Image

from preprocess.prepare_allen_emlddmm_inputs import (
    CUTTING_LATTICE_SPACING_UM,
    DEFAULT_DATA_DIR,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_TARGET_PIXEL_SIZE_UM,
    EXPECTED_PRESENT,
    MINIMUM_SLOT,
    NUMBER_OF_SLOTS,
    PHYSICAL_FIELDS,
    SAMPLE_PROVENANCE_FIELDS,
    SECONDARY_BLOCK_ENVELOPES,
    canonical_z_axis_um,
    canonical_z_um,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FIGURE = Path(
    "results/qc/allen_emlddmm_prepared_stack_overview.png"
)
DEFAULT_ORIGINAL_FIGURE = Path(
    "results/qc/allen_emlddmm_original_stack_overview.png"
)
ABSENT_COLOR = "#eeeeee"
NISSL_COLOR = "#1f77b4"
PV_COLOR = "#d95f02"
OCCUPANCY_CODES = {"absent": 0, "nissl": 1, "pv": 2}
FIGURE_FOOTERS = {
    "prepared": (
        "Preparation QC only: the provisional 200-um MRI assumption does not "
        "clear the registration provenance gate."
    ),
    "original": (
        "Original source-frame QC: profiles are origin-aligned at source "
        "pixel x=0; no centering or registration transform is applied."
    ),
}


@dataclass(frozen=True)
class StackRow:
    section_number: int
    grid_index: int
    z_um: float
    status: str
    stain: str
    sample_id: str
    image_path: Path | None
    width_px: int | None
    height_px: int | None
    pixel_size_um: float | None
    x_origin_um: float | None
    block_id: str
    secondary_envelope_context: str


@dataclass(frozen=True)
class StackInputs:
    dataset: Path
    stage: str
    rows: tuple[StackRow, ...]
    z_um: np.ndarray
    x_um: np.ndarray
    occupancy: np.ndarray
    stain_counts: Mapping[str, int]
    prepared_canvas_center_extent_yx_mm: tuple[float, float]
    prepared_canvas_outer_extent_yx_mm: tuple[float, float]
    serial_to_canvas_x_ratio: float
    tissue_support_extent_status: str
    direct_7t_comparison_status: str
    openneuro_comparison_status: str


def comparison_statuses(
    *,
    mri_config: Path = PROJECT_ROOT / "configs" / "allen_708424_mri_source.json",
    openneuro_root: Path = PROJECT_ROOT / "data" / "raw" / "openneuro" / "ds003590",
) -> tuple[str, str]:
    if not mri_config.is_file():
        direct = "unavailable_provenance_record_absent"
    else:
        mri = json.loads(mri_config.read_text(encoding="utf-8"))
        direct = (
            "available_not_yet_validated"
            if mri.get("geometry_status") == "verified"
            else "unavailable_provenance_gate_blocked"
        )
    if not openneuro_root.exists():
        openneuro = "unavailable_local_checkout_absent"
    else:
        payloads = list(openneuro_root.rglob("*.nii.gz"))
        openneuro = (
            "available_not_yet_validated"
            if any(path.is_file() for path in payloads)
            else "unavailable_local_annex_content_absent"
        )
    return direct, openneuro


def _read_tsv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        rows = list(reader)
        fields = list(reader.fieldnames or ())
    return rows, fields


def accepted_canvas_axis_length(
    canvas_audit: Mapping[str, Any], axis: str
) -> int:
    """Read an accepted canvas dimension by its axis label, not its position."""

    prefix = "accepted_canvas_shape_"
    keys = [key for key in canvas_audit if key.startswith(prefix)]
    if len(keys) != 1:
        raise RuntimeError(
            "Canvas audit must contain exactly one accepted_canvas_shape_<axes> key"
        )
    key = keys[0]
    labels = key.removeprefix(prefix)
    shape = canvas_audit[key]
    if (
        not isinstance(shape, list)
        or len(labels) != len(shape)
        or len(set(labels)) != len(labels)
        or axis not in labels
    ):
        raise RuntimeError(f"Invalid labeled canvas shape in {key}")
    length = int(shape[labels.index(axis)])
    if length < 1:
        raise RuntimeError(f"Canvas {axis} length must be positive")
    return length


def load_stack_inputs(
    dataset: Path,
    *,
    stage: str = "prepared",
    data_dir: Path = DEFAULT_DATA_DIR,
    expected_slots: int = NUMBER_OF_SLOTS,
    expected_stain_counts: Mapping[str, int] = EXPECTED_PRESENT,
) -> StackInputs:
    """Cross-validate canonical rows, HIST_ALL samples, images, and axes."""

    dataset = dataset.expanduser().resolve()
    data_dir = data_dir.expanduser().resolve()
    if stage not in {"prepared", "original"}:
        raise RuntimeError("stage must be 'prepared' or 'original'")
    physical_path = dataset / "metadata" / "physical_sections.tsv"
    samples_path = dataset / "inputs" / "views" / "HIST_ALL" / "samples.tsv"
    canvas_path = dataset / "metadata" / "loader_canvas_audit.json"
    physical, physical_fields = _read_tsv(physical_path)
    samples, sample_fields = _read_tsv(samples_path)
    if physical_fields != PHYSICAL_FIELDS:
        raise RuntimeError("physical_sections.tsv has an unexpected canonical schema")
    expected_sample_fields = [
        "sample_id",
        "participant_id",
        "species",
        "status",
        *SAMPLE_PROVENANCE_FIELDS,
    ]
    if sample_fields != expected_sample_fields:
        raise RuntimeError("HIST_ALL samples.tsv has an unexpected schema")
    if len(physical) != expected_slots or len(samples) != expected_slots:
        raise RuntimeError(
            "Canonical table and HIST_ALL samples must contain exactly "
            f"{expected_slots} positionally matching rows"
        )

    canvas = json.loads(canvas_path.read_text(encoding="utf-8"))
    spacing = float(canvas.get("target_spacing_um", float("nan")))
    if not np.isclose(spacing, DEFAULT_TARGET_PIXEL_SIZE_UM):
        raise RuntimeError("Canvas audit does not use 200-um prepared spacing")
    x_um: np.ndarray | None = None
    if stage == "prepared":
        if canvas.get("sectionwise_centering") is not False:
            raise RuntimeError(
                "Prepared QC requires one globally translated common canvas"
            )
        x_length = accepted_canvas_axis_length(canvas, "x")
        x_um = (
            np.arange(x_length, dtype=np.float64) * spacing
            + float(canvas["global_translation_xy_um"][0])
        )
    y_length = accepted_canvas_axis_length(canvas, "y")
    canvas_center_yx = (
        (y_length - 1) * spacing / 1000.0,
        (accepted_canvas_axis_length(canvas, "x") - 1) * spacing / 1000.0,
    )
    canvas_outer_yx = (
        y_length * spacing / 1000.0,
        accepted_canvas_axis_length(canvas, "x") * spacing / 1000.0,
    )

    raw_by_image_id: dict[str, dict[str, str]] = {}
    if stage == "original":
        manifest_path = data_dir / "metadata" / "manifest.tsv"
        manifest, _ = _read_tsv(manifest_path)
        raw_by_image_id = {
            row["allen_section_image_id"]: row
            for row in manifest
            if row.get("kind") == "histology_jpeg"
        }

    stack_rows: list[StackRow] = []
    occupancy = np.zeros(expected_slots, dtype=np.uint8)
    stain_counts: Counter[str] = Counter()
    view_dir = samples_path.parent
    expected_full = expected_slots == NUMBER_OF_SLOTS
    for index, (physical_row, sample_row) in enumerate(
        zip(physical, samples, strict=True)
    ):
        section = int(physical_row["allen_section_number"])
        grid_index = int(physical_row["physical_index"])
        z_um = float(physical_row["serial_z_center_mm"]) * 1000.0
        status = "present" if physical_row["image_present"] == "true" else "absent"
        stain = physical_row["stain"]
        if section != MINIMUM_SLOT + index or grid_index != index:
            raise RuntimeError(f"Noncanonical physical row at position {index}")
        if not np.isclose(z_um, canonical_z_um(section), atol=1e-8):
            raise RuntimeError(f"Noncanonical z coordinate for section {section}")
        if not np.isclose(
            float(physical_row["serial_pitch_um"]),
            CUTTING_LATTICE_SPACING_UM,
        ):
            raise RuntimeError(f"Invalid cutting-lattice spacing at row {index}")
        if sample_row["status"] != status:
            raise RuntimeError(f"Status mismatch at physical row {index}")
        if (
            sample_row["participant_id"] != "708424"
            or sample_row["species"] != "Homo sapiens"
        ):
            raise RuntimeError(f"Unexpected participant/species at row {index}")
        if any(
            sample_row[field] != physical_row[field]
            for field in SAMPLE_PROVENANCE_FIELDS
        ):
            raise RuntimeError(f"Sample provenance mismatch at physical row {index}")

        image_path: Path | None = None
        width: int | None = None
        height: int | None = None
        pixel_size: float | None = None
        x_origin_um: float | None = None
        sample_id = sample_row["sample_id"]
        if status == "present":
            if stain not in {"nissl", "pv"}:
                raise RuntimeError(f"Invalid present stain at row {index}: {stain!r}")
            expected_sample = Path(physical_row["prepared_relative_path"]).name
            expected_named_sample = (
                f"allen_708424_{stain}_{grid_index + 1:04d}.tif"
            )
            if sample_id != expected_sample or sample_id != expected_named_sample:
                raise RuntimeError(f"Sample/stain identity mismatch at row {index}")
            if stage == "prepared":
                image_path = view_dir / sample_id
                sidecar = json.loads(
                    image_path.with_suffix(".json").read_text(encoding="utf-8")
                )
                sizes = sidecar["Sizes"]
                directions = sidecar["SpaceDirections"]
                width = int(sizes[1])
                height = int(sizes[2])
                pixel_size = float(directions[1][0])
                x_origin_um = float(sidecar["SpaceOrigin"][0])
                if not np.isclose(pixel_size, DEFAULT_TARGET_PIXEL_SIZE_UM):
                    raise RuntimeError(f"Prepared spacing is not 200 um at row {index}")
                image_description = "Prepared HIST_ALL"
            else:
                raw = raw_by_image_id.get(physical_row["allen_section_image_id"])
                if raw is None:
                    raise RuntimeError(
                        f"Allen raw provenance is missing at row {index}"
                    )
                pixel_size = float(raw["pixel_size_um"])
                width = int(raw["width_px"])
                height = int(raw["height_px"])
                image_path = data_dir / physical_row["source_relative_path"]
                image_description = "Original source"
                x_origin_um = 0.0
                if pixel_size <= 0:
                    raise RuntimeError(
                        f"Original pixel spacing is invalid at row {index}"
                    )
            if not image_path.is_file():
                raise RuntimeError(
                    f"{image_description} image is missing: {image_path}"
                )
            with Image.open(image_path) as image:
                if image.mode != "RGB" or image.size != (width, height):
                    raise RuntimeError(
                        f"{image_description} image shape/mode mismatch: {image_path}"
                    )
            occupancy[index] = OCCUPANCY_CODES[stain]
            stain_counts[stain] += 1
        elif status == "absent":
            expected_sample = f"allen_708424_absent_{grid_index + 1:04d}.tif"
            if (
                stain
                or physical_row["prepared_relative_path"]
                or sample_id != expected_sample
            ):
                raise RuntimeError(f"Invalid absent row at position {index}")
            if (view_dir / sample_id).exists():
                raise RuntimeError(f"Absent row has a placeholder image: {sample_id}")
        else:
            raise RuntimeError(f"Invalid status at row {index}: {status!r}")

        stack_rows.append(
            StackRow(
                section_number=section,
                grid_index=grid_index,
                z_um=z_um,
                status=status,
                stain=stain,
                sample_id=sample_id,
                image_path=image_path,
                width_px=width,
                height_px=height,
                pixel_size_um=pixel_size,
                x_origin_um=x_origin_um,
                block_id=physical_row["block_id"],
                secondary_envelope_context=physical_row[
                    "secondary_envelope_context"
                ],
            )
        )

    actual_counts = dict(stain_counts)
    if actual_counts != dict(expected_stain_counts):
        raise RuntimeError(
            f"Expected stain counts {dict(expected_stain_counts)}, "
            f"found {actual_counts}"
        )
    z_axis = np.array([row.z_um for row in stack_rows], dtype=np.float64)
    if expected_full and not np.allclose(
        z_axis, canonical_z_axis_um(), atol=1e-9, rtol=0.0
    ):
        raise RuntimeError("Full HIST_ALL z coordinates differ from the canonical axis")
    if stage == "original":
        maximum_extent_um = max(
            (int(row.width_px) - 1) * float(row.pixel_size_um)
            for row in stack_rows
            if row.status == "present"
        )
        display_spacing_um = DEFAULT_TARGET_PIXEL_SIZE_UM
        x_length = int(np.ceil(maximum_extent_um / display_spacing_um)) + 1
        x_um = np.arange(x_length, dtype=np.float64) * display_spacing_um
    if x_um is None:
        raise RuntimeError("Could not construct the stack x axis")
    direct_status, openneuro_status = comparison_statuses()
    serial_extent_mm = (z_axis[-1] - z_axis[0]) / 1000.0
    return StackInputs(
        dataset=dataset,
        stage=stage,
        rows=tuple(stack_rows),
        z_um=z_axis,
        x_um=x_um,
        occupancy=occupancy,
        stain_counts=actual_counts,
        prepared_canvas_center_extent_yx_mm=canvas_center_yx,
        prepared_canvas_outer_extent_yx_mm=canvas_outer_yx,
        serial_to_canvas_x_ratio=serial_extent_mm / canvas_center_yx[1],
        tissue_support_extent_status=(
            "not_computed_no_accepted_full_stack_support"
        ),
        direct_7t_comparison_status=direct_status,
        openneuro_comparison_status=openneuro_status,
    )


def representative_rows(inputs: StackInputs) -> tuple[StackRow, StackRow]:
    def select(stain: str) -> StackRow:
        candidates = [row for row in inputs.rows if row.stain == stain]
        if not candidates:
            raise RuntimeError(f"No present {stain} section is available")
        return min(
            candidates,
            key=lambda row: (abs(row.z_um), row.section_number),
        )

    return select("nissl"), select("pv")


def load_rgb(row: StackRow) -> np.ndarray:
    if row.image_path is None:
        raise RuntimeError(f"Section {row.section_number} has no source image")
    with Image.open(row.image_path) as image:
        return np.asarray(image, dtype=np.uint8)


def build_side_profile(inputs: StackInputs) -> np.ndarray:
    """Build slot-by-x maximum darkness without constructing a 3-D volume."""

    profile = np.full(
        (len(inputs.rows), inputs.x_um.size), np.nan, dtype=np.float32
    )
    for index, row in enumerate(inputs.rows):
        if row.status != "present":
            continue
        rgb = load_rgb(row).astype(np.float32)
        # Optical darkness retains the prepared RGB appearance as a scalar.
        # Max over y makes the sparse side-on tissue silhouette legible.
        darkness = 1.0 - np.mean(rgb, axis=2) / 255.0
        if inputs.stage == "prepared":
            valid = rgb[..., 0] > 0.0
            valid_columns = np.any(valid, axis=0)
            darkness_x = np.full(
                darkness.shape[1], np.nan, dtype=np.float32
            )
            if np.any(valid_columns):
                masked = np.where(
                    valid[:, valid_columns],
                    darkness[:, valid_columns],
                    -np.inf,
                )
                darkness_x[valid_columns] = np.max(masked, axis=0)
            source_x = (
                np.arange(int(row.width_px), dtype=np.float64)
                * float(row.pixel_size_um)
                + float(row.x_origin_um)
            )
        else:
            darkness_x = np.max(darkness, axis=0)
            source_x = (
                np.arange(int(row.width_px), dtype=np.float64)
                * float(row.pixel_size_um)
            )
        inside = (inputs.x_um >= source_x[0]) & (inputs.x_um <= source_x[-1])
        profile[index, inside] = np.interp(
            inputs.x_um[inside], source_x, darkness_x
        )
    return profile


def cell_edges(centers: np.ndarray) -> np.ndarray:
    centers = np.asarray(centers, dtype=np.float64)
    if centers.ndim != 1 or centers.size < 2:
        raise RuntimeError("Coordinate centers must be a one-dimensional axis")
    steps = np.diff(centers)
    if not np.allclose(steps, steps[0]) or steps[0] <= 0:
        raise RuntimeError("Coordinate centers must be uniformly increasing")
    edges = np.empty(centers.size + 1, dtype=np.float64)
    edges[1:-1] = (centers[:-1] + centers[1:]) / 2.0
    edges[0] = centers[0] - steps[0] / 2.0
    edges[-1] = centers[-1] + steps[0] / 2.0
    return edges



def _coordinate_cell_edges(centers: np.ndarray) -> np.ndarray:
    """Return physical cell edges for a strictly monotone coordinate vector."""
    centers = np.asarray(centers, dtype=np.float64)
    if centers.ndim != 1 or centers.size < 2:
        raise RuntimeError("Coordinate centers must be a one-dimensional axis")
    steps = np.diff(centers)
    if not (np.all(steps > 0.0) or np.all(steps < 0.0)):
        raise RuntimeError("Coordinate centers must be strictly monotone")
    edges = np.empty(centers.size + 1, dtype=np.float64)
    edges[1:-1] = (centers[:-1] + centers[1:]) / 2.0
    edges[0] = centers[0] - steps[0] / 2.0
    edges[-1] = centers[-1] + steps[-1] / 2.0
    return edges


def _uniform_representative_positions(count: int, number: int = 9) -> np.ndarray:
    """Select stable, uniformly spaced positions including both endpoints."""
    if count < 1 or number < 1:
        raise RuntimeError("Representative selection requires positive sizes")
    return np.unique(
        np.rint(np.linspace(0, count - 1, min(count, number))).astype(np.int64)
    )


def _normalize_supported_projection(
    numerator: np.ndarray, denominator: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize an RGB numerator and return its scalar positive-support mask."""
    numerator = np.asarray(numerator, dtype=np.float32)
    denominator = np.asarray(denominator, dtype=np.float32)
    if numerator.ndim != 3 or numerator.shape[0] != 3:
        raise RuntimeError("Projection numerator must have shape (3, serial, spatial)")
    if denominator.shape != numerator.shape[1:]:
        raise RuntimeError("Projection denominator shape does not match numerator")
    if not np.all(np.isfinite(numerator)) or not np.all(np.isfinite(denominator)):
        raise RuntimeError("Projection inputs contain nonfinite values")
    if np.any(denominator < 0.0):
        raise RuntimeError("Projection support must be nonnegative")
    positive = denominator > 0.0
    output = np.zeros_like(numerator, dtype=np.float32)
    output[:, positive] = numerator[:, positive] / denominator[positive]
    return output, positive


def _render_saved_transform_stack_overview(
    *,
    representative_original: Sequence[np.ndarray],
    representative_transformed: Sequence[np.ndarray],
    representative_titles: Sequence[str],
    projections: Mapping[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    serial_um: np.ndarray,
    trace_serial_um: np.ndarray,
    traces: Mapping[str, np.ndarray],
    marked_serial_um: Mapping[str, float],
    output: Path,
) -> tuple[Path, Mapping[str, list[int]]]:
    """Render a generic original/transformed stack overview from prepared data."""
    if len(representative_original) != len(representative_transformed):
        raise RuntimeError("Original and transformed representative counts differ")
    if len(representative_original) != len(representative_titles):
        raise RuntimeError("Representative titles do not match image counts")
    serial_edges = _coordinate_cell_edges(serial_um)
    figure = plt.figure(figsize=(24.0, 17.0), dpi=160, facecolor="white")
    grid = figure.add_gridspec(
        5, 18, height_ratios=(1.3, 1.3, 3.0, 1.25, 1.25),
        hspace=0.48, wspace=0.35,
    )
    for column, (original, transformed, title) in enumerate(
        zip(representative_original, representative_transformed,
            representative_titles, strict=True)
    ):
        start = 2 * column
        for row, image, label in (
            (0, original, "original"),
            (1, transformed, "atlas-free transformed"),
        ):
            axis = figure.add_subplot(grid[row, start:start + 2])
            axis.imshow(
                np.clip(np.moveaxis(image, 0, -1), 0.0, 1.0), origin="lower"
            )
            axis.set_title(f"{title}\n{label}", fontsize=7)
            axis.set_axis_off()

    projection_shapes: dict[str, list[int]] = {}
    projection_order = (
        ("original_serial_row", "Original serial-row"),
        ("transformed_serial_row", "Atlas-free transformed serial-row"),
        ("original_serial_column", "Original serial-column"),
        ("transformed_serial_column", "Atlas-free transformed serial-column"),
    )
    for panel, (key, title) in enumerate(projection_order):
        numerator, denominator, spatial_um = projections[key]
        normalized, positive = _normalize_supported_projection(
            numerator, denominator
        )
        projection_shapes[key] = list(normalized.shape[1:])
        rgb = np.moveaxis(normalized, 0, -1).transpose(1, 0, 2)
        rgba = np.concatenate(
            (np.clip(rgb, 0.0, 1.0), positive.T[..., None]), axis=2
        )
        spatial_edges = _coordinate_cell_edges(spatial_um)
        axis = figure.add_subplot(grid[2, panel * 4:panel * 4 + 4])
        axis.set_facecolor("#e6e6e6")
        axis.imshow(
            rgba, origin="lower", aspect="auto", interpolation="nearest",
            extent=(
                serial_edges[0] / 1000.0, serial_edges[-1] / 1000.0,
                spatial_edges[0] / 1000.0, spatial_edges[-1] / 1000.0,
            ),
        )
        for position in marked_serial_um.values():
            axis.axvline(
                position / 1000.0, color="#d62728", lw=0.45, alpha=0.75
            )
        axis.set_title(title, fontsize=9)
        axis.set_xlabel("Anterior → posterior serial position (mm)")
        axis.set_ylabel(
            "Row position (mm)" if "row" in key else "Column position (mm)"
        )

    trace_order = (
        ("row_translation_um", "Residual row translation (um)"),
        ("column_translation_um", "Residual column translation (um)"),
        ("rotation_deg", "Residual rotation (degrees)"),
    )
    for panel, (key, label) in enumerate(trace_order):
        axis = figure.add_subplot(grid[3:, panel * 6:panel * 6 + 6])
        axis.plot(
            trace_serial_um / 1000.0, traces[key],
            lw=0.75, color="#1f77b4",
        )
        for position in marked_serial_um.values():
            axis.axvline(
                position / 1000.0, color="#d62728", lw=0.5, alpha=0.7
            )
        axis.set_xlabel("Anterior → posterior serial position (mm)")
        axis.set_ylabel(label)
        axis.grid(alpha=0.15)
    figure.suptitle(
        "Allen 708424 Nissl atlas-free section alignment — "
        "saved transforms, 800-um in-plane",
        fontsize=14,
    )
    figure.text(
        0.5, 0.012,
        "Neutral projection background denotes zero accumulated tissue support; "
        "red lines mark reviewed outliers/jump endpoints.",
        ha="center", fontsize=8,
    )
    figure.subplots_adjust(
        top=0.95, bottom=0.06, left=0.045, right=0.99
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        figure.savefig(
            output, format="png", dpi=160, facecolor="white",
            metadata={"Software": "ad-resilience saved-transform stack QC"},
        )
    finally:
        plt.close(figure)
    return output, projection_shapes


def section_gap_counts(
    inputs: StackInputs, stain: str | None = None
) -> Counter[int]:
    sections = sorted(
        row.section_number
        for row in inputs.rows
        if row.status == "present" and (stain is None or row.stain == stain)
    )
    return Counter(int(delta) for delta in np.diff(sections))


def _plot_section_gaps(
    axis: plt.Axes,
    counts: Counter[int],
    *,
    title: str,
    color: str,
) -> None:
    increments = np.array(sorted(counts), dtype=np.int64)
    frequencies = np.array([counts[int(value)] for value in increments])
    axis.set_title(title, fontsize=9)
    axis.set_xlabel("Physical-section increment")
    axis.set_ylabel("Count (log)")
    if not counts:
        axis.text(0.5, 0.5, "Fewer than two observations", ha="center", va="center")
        return
    axis.bar(increments, frequencies, width=0.8, color=color, alpha=0.85)
    axis.set_yscale("log")
    maximum = int(increments[-1])
    axis.text(
        0.98,
        0.95,
        f"max: {maximum} sections = {maximum * 0.05:.2f} mm",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=7,
    )
    secondary = axis.secondary_xaxis(
        "top", functions=(lambda value: value * 0.05, lambda value: value / 0.05)
    )
    secondary.set_xlabel("Center gap (mm)", fontsize=8)


def render_stack_overview(
    inputs: StackInputs,
) -> tuple[plt.Figure, tuple[str, ...]]:
    nissl_row, pv_row = representative_rows(inputs)
    nissl_rgb = load_rgb(nissl_row)
    pv_rgb = load_rgb(pv_row)
    if inputs.stage == "prepared":
        nissl_rgb = nissl_rgb.copy()
        pv_rgb = pv_rgb.copy()
        nissl_rgb[nissl_rgb[..., 0] == 0] = 255
        pv_rgb[pv_rgb[..., 0] == 0] = 255
    profile = build_side_profile(inputs)
    z_edges_mm = cell_edges(inputs.z_um) / 1000.0
    x_edges_mm = cell_edges(inputs.x_um) / 1000.0

    figure = plt.figure(figsize=(14.0, 11.0), dpi=240)
    grid = figure.add_gridspec(
        3,
        6,
        height_ratios=(1.45, 3.0, 1.35),
        hspace=0.48,
        wspace=0.65,
    )
    nissl_axis = figure.add_subplot(grid[0, :3])
    pv_axis = figure.add_subplot(grid[0, 3:])
    profile_axis = figure.add_subplot(grid[1, :5])
    occupancy_axis = figure.add_subplot(grid[1, 5], sharey=profile_axis)
    combined_gap_axis = figure.add_subplot(grid[2, :2])
    nissl_gap_axis = figure.add_subplot(grid[2, 2:4])
    pv_gap_axis = figure.add_subplot(grid[2, 4:])

    stage_label = "original " if inputs.stage == "original" else ""
    spacing_label = (
        f", {nissl_row.pixel_size_um:.3f} um/px"
        if inputs.stage == "original"
        else ""
    )
    pv_spacing_label = (
        f", {pv_row.pixel_size_um:.3f} um/px"
        if inputs.stage == "original"
        else ""
    )
    representative_titles = (
        f"Central {stage_label}Nissl (section {nissl_row.section_number}, "
        f"z={nissl_row.z_um / 1000.0:.3f} mm{spacing_label})",
        f"Central {stage_label}PV (section {pv_row.section_number}, "
        f"z={pv_row.z_um / 1000.0:.3f} mm{pv_spacing_label})",
    )
    for axis, rgb, title in zip(
        (nissl_axis, pv_axis),
        (nissl_rgb, pv_rgb),
        representative_titles,
        strict=True,
    ):
        axis.imshow(rgb)
        axis.set_title(title, fontsize=10)
        axis.set_axis_off()

    darkness_cmap = plt.colormaps["gray_r"].copy()
    darkness_cmap.set_bad("white")
    profile_axis.imshow(
        np.ma.masked_invalid(profile),
        origin="lower",
        aspect="equal",
        interpolation="nearest",
        extent=(
            x_edges_mm[0],
            x_edges_mm[-1],
            z_edges_mm[0],
            z_edges_mm[-1],
        ),
        cmap=darkness_cmap,
        vmin=0.0,
        vmax=1.0,
    )
    if inputs.stage == "prepared":
        profile_title = (
            "Globally translated x-z profile (equal mm aspect; no sectionwise "
            "centering)"
        )
        x_label = "Globally translated prepared-canvas x (mm)"
        figure_title = (
            "Allen 708424 prepared histology — 50-um physical pitch; "
            "200/400-um nominal Nissl/PV sampling"
        )
    else:
        profile_title = (
            "Original x-z side profile (source x=0; maximum darkness over y)"
        )
        x_label = "Source-frame x from pixel 0 (mm)"
        figure_title = (
            "Allen 708424 original histology: native images, 50-um lattice"
        )
    profile_axis.set_title(profile_title, fontsize=10)
    profile_axis.set_xlabel(x_label)
    profile_axis.set_ylabel("Nominal serial z center (mm)")
    for block_id, (first, last) in SECONDARY_BLOCK_ENVELOPES.items():
        low = canonical_z_um(first) / 1000.0 - 0.025
        high = canonical_z_um(last) / 1000.0 + 0.025
        if high < z_edges_mm[0] or low > z_edges_mm[-1]:
            continue
        profile_axis.axhspan(low, high, color="#756bb1", alpha=0.045, zorder=0)
        profile_axis.text(
            x_edges_mm[-1],
            (low + high) / 2.0,
            block_id,
            ha="right",
            va="center",
            fontsize=7,
            color="#54278f",
        )

    occupancy_axis.imshow(
        inputs.occupancy[:, None],
        aspect="auto",
        interpolation="nearest",
        origin="lower",
        extent=(0.0, 1.0, z_edges_mm[0], z_edges_mm[-1]),
        cmap=ListedColormap([ABSENT_COLOR, NISSL_COLOR, PV_COLOR]),
        vmin=-0.5,
        vmax=2.5,
    )
    occupancy_axis.set_xticks([])
    occupancy_axis.tick_params(labelleft=False)
    occupancy_axis.set_title("Observed\nsections", fontsize=9)
    occupancy_axis.legend(
        handles=[
            Patch(
                facecolor=ABSENT_COLOR,
                edgecolor="#999999",
                label=(
                    "Absent "
                    f"({len(inputs.rows) - sum(inputs.stain_counts.values())})"
                ),
            ),
            Patch(
                facecolor=NISSL_COLOR,
                label=f"Nissl ({inputs.stain_counts['nissl']})",
            ),
            Patch(
                facecolor=PV_COLOR,
                label=f"PV ({inputs.stain_counts['pv']})",
            ),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=1,
        frameon=False,
        fontsize=8,
    )

    _plot_section_gaps(
        combined_gap_axis,
        section_gap_counts(inputs),
        title="Combined observed gaps",
        color="#636363",
    )
    _plot_section_gaps(
        nissl_gap_axis,
        section_gap_counts(inputs, "nissl"),
        title="Nissl observed gaps",
        color=NISSL_COLOR,
    )
    _plot_section_gaps(
        pv_gap_axis,
        section_gap_counts(inputs, "pv"),
        title="PV observed gaps",
        color=PV_COLOR,
    )

    canvas_text = (
        "prepared canvas center y/x "
        f"{inputs.prepared_canvas_center_extent_yx_mm[0]:.1f}/"
        f"{inputs.prepared_canvas_center_extent_yx_mm[1]:.1f} mm; "
        "outer y/x "
        f"{inputs.prepared_canvas_outer_extent_yx_mm[0]:.1f}/"
        f"{inputs.prepared_canvas_outer_extent_yx_mm[1]:.1f} mm; "
        f"serial/canvas-x {inputs.serial_to_canvas_x_ratio:.2f}"
    )
    figure.suptitle(f"{figure_title}\n{canvas_text}", fontsize=12)
    if inputs.openneuro_comparison_status == "unavailable_local_annex_content_absent":
        openneuro_text = (
            "Local OpenNeuro comparison unavailable: annexed image content is "
            "not present."
        )
    else:
        openneuro_text = f"OpenNeuro comparison: {inputs.openneuro_comparison_status}."
    figure.text(
        0.5,
        0.012,
        f"{FIGURE_FOOTERS[inputs.stage]}  {openneuro_text}  "
        f"Direct 7T: {inputs.direct_7t_comparison_status}.  "
        f"Tissue support: {inputs.tissue_support_extent_status}.",
        ha="center",
        fontsize=7,
    )
    figure.subplots_adjust(top=0.91, bottom=0.09, left=0.07, right=0.98)
    titles = representative_titles + (
        profile_title,
        "Stack occupancy",
        "Combined observed gaps",
        "Nissl observed gaps",
        "PV observed gaps",
    )
    return figure, titles


def write_stack_overview(
    dataset: Path,
    output: Path | None = None,
    *,
    stage: str = "prepared",
    data_dir: Path = DEFAULT_DATA_DIR,
    overwrite: bool = False,
    expected_slots: int = NUMBER_OF_SLOTS,
    expected_stain_counts: Mapping[str, int] = EXPECTED_PRESENT,
) -> tuple[Path, tuple[str, ...]]:
    default_output = DEFAULT_FIGURE if stage == "prepared" else DEFAULT_ORIGINAL_FIGURE
    destination = (output or default_output).expanduser().resolve()
    if destination.suffix.lower() != ".png":
        raise RuntimeError(f"Output must be a PNG path: {destination}")
    if destination.exists() and not overwrite:
        raise RuntimeError(
            f"Output already exists: {destination}; use --overwrite to replace it"
        )
    inputs = load_stack_inputs(
        dataset,
        stage=stage,
        data_dir=data_dir,
        expected_slots=expected_slots,
        expected_stain_counts=expected_stain_counts,
    )
    figure, titles = render_stack_overview(inputs)
    buffer = io.BytesIO()
    try:
        figure.savefig(
            buffer,
            format="png",
            dpi=240,
            facecolor="white",
            metadata={"Software": f"ad-resilience {stage}-stack QC"},
        )
    finally:
        plt.close(figure)
    destination.parent.mkdir(parents=True, exist_ok=True)
    mode = "wb" if overwrite else "xb"
    try:
        with destination.open(mode) as stream:
            stream.write(buffer.getvalue())
    except FileExistsError as exc:
        raise RuntimeError(
            f"Output already exists: {destination}; use --overwrite to replace it"
        ) from exc
    return destination, titles


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--dataset", type=Path, default=DEFAULT_OUTPUT_DIR)
    result.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    result.add_argument("--stage", choices=("prepared", "original"), default="prepared")
    result.add_argument("--output", type=Path)
    result.add_argument("--overwrite", action="store_true")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    destination, _ = write_stack_overview(
        args.dataset,
        args.output,
        stage=args.stage,
        data_dir=args.data_dir,
        overwrite=args.overwrite,
    )
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
