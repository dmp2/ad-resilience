"""Prepare mixed Allen 708424 histology inputs for pinned EM-LDDMM.

Raw Allen JPEGs remain immutable.  This command creates 200-um RGB TIFF
derivatives, upstream-compatible JSON sidecars, a canonical physical-section
table, and deterministic HIST_ALL/HIST_NISSL/HIST_PV views.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import importlib
import io
import json
import logging
import math
import os
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, map_coordinates


LOG = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = Path("data/raw/allen/specimen_708424")
DEFAULT_OUTPUT_DIR = Path(
    "data/derivatives/allen/specimen_708424/emlddmm_7t"
)
DEFAULT_SYMMETRIC_OUTPUT_DIR = Path(
    "data/derivatives/allen/specimen_708424/emlddmm_7t_symmetric"
)
PIN_FILE = PROJECT_ROOT / "configs" / "emlddmm-upstream-commit.txt"
MINIMUM_SLOT = 36
MAXIMUM_SLOT = 2881
NUMBER_OF_SLOTS = MAXIMUM_SLOT - MINIMUM_SLOT + 1
CUTTING_LATTICE_SPACING_UM = 50.0
PHYSICAL_SECTION_THICKNESS_UM = 50.0
DEFAULT_TARGET_PIXEL_SIZE_UM = 200.0
EXPECTED_PRESENT = {"nissl": 641, "pv": 287}
NOMINAL_SERIES_INTERVAL_UM = {"nissl": 200.0, "pv": 400.0}
SPECIMEN_ID = "708424"
PREPARED_METADATA_SCHEMA = "allen-emlddmm-serial-v3"
SERIAL_CENTER_EXTENT_MM = (NUMBER_OF_SLOTS - 1) * CUTTING_LATTICE_SPACING_UM / 1000.0
SERIAL_OUTER_FACE_EXTENT_MM = NUMBER_OF_SLOTS * CUTTING_LATTICE_SPACING_UM / 1000.0
SECONDARY_BLOCK_REFERENCE_REVISION = (
    "f979cb205031335ed2647cff904b4bbf5f68b14f"
)
SECONDARY_BLOCK_ENVELOPES = {
    "B1": (36, 738),
    "B2": (771, 1429),
    "B3": (1444, 1765),
    "B4": (1811, 2122),
    "B5": (2162, 2482),
    "B6": (2529, 2881),
}
VIEW_STAINS = {
    "HIST_ALL": frozenset({"nissl", "pv"}),
    "HIST_NISSL": frozenset({"nissl"}),
    "HIST_PV": frozenset({"pv"}),
}
PHYSICAL_FIELDS = [
    "physical_index",
    "specimen_id",
    "allen_section_number",
    "serial_z_center_mm",
    "section_thickness_um",
    "serial_pitch_um",
    "image_present",
    "stain",
    "allen_section_image_id",
    "allen_data_set_id",
    "block_id",
    "block_assignment_source",
    "block_assignment_status",
    "secondary_envelope_context",
    "source_relative_path",
    "prepared_relative_path",
    "source_sha256",
    "prepared_sha256",
    "nominal_series_interval_um",
    "observation_class",
]
SAMPLE_PROVENANCE_FIELDS = [
    "physical_index",
    "specimen_id",
    "allen_section_number",
    "serial_z_center_mm",
    "section_thickness_um",
    "serial_pitch_um",
    "image_present",
    "stain",
    "allen_section_image_id",
    "allen_data_set_id",
    "block_id",
    "block_assignment_source",
    "block_assignment_status",
    "secondary_envelope_context",
    "source_relative_path",
    "prepared_relative_path",
    "nominal_series_interval_um",
    "observation_class",
]
# These fields are used only while preparing pixels and validating the existing
# derivative. They deliberately do not broaden the canonical physical-lattice
# table, whose public schema is PHYSICAL_FIELDS above.
INTERNAL_FIELDS = [
    "section_number",
    "physical_serial_slot",
    "grid_index",
    "z_um",
    "status",
    "source_path",
    "source_image_id",
    "source_width_px",
    "source_height_px",
    "source_pixel_size_um",
    "prepared_path",
    "prepared_width_px",
    "prepared_height_px",
    "prepared_pixel_size_um",
    "prepared_to_source_transform",
    "first_channel_zero_count",
    "valid_pixel_count",
    "first_channel_zero_fraction",
    "physical_section_thickness_um",
    "cutting_lattice_spacing_um",
]


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_z_um(section_number: int) -> float:
    grid_index = section_number - MINIMUM_SLOT
    return (
        grid_index - (NUMBER_OF_SLOTS - 1) / 2.0
    ) * CUTTING_LATTICE_SPACING_UM


def canonical_z_axis_um() -> np.ndarray:
    axis = np.arange(NUMBER_OF_SLOTS, dtype=np.float64)
    axis *= CUTTING_LATTICE_SPACING_UM
    axis -= np.mean(axis)
    return axis


def secondary_envelope(section_number: int) -> tuple[str, str] | None:
    """Return the secondary observation-envelope label and context."""

    for block_id, (first, last) in SECONDARY_BLOCK_ENVELOPES.items():
        if first <= section_number <= last:
            return block_id, f"inside_{block_id}_observation_envelope"
    return None


def parse_section_range(value: str | None) -> tuple[int, int]:
    if value is None:
        return MINIMUM_SLOT, MAXIMUM_SLOT
    pieces = value.split(":", maxsplit=1)
    if len(pieces) != 2:
        raise argparse.ArgumentTypeError("section range must be FIRST:LAST")
    first, last = (int(piece) for piece in pieces)
    if not MINIMUM_SLOT <= first <= last <= MAXIMUM_SLOT:
        raise argparse.ArgumentTypeError(
            f"section range must lie within {MINIMUM_SLOT}:{MAXIMUM_SLOT}"
        )
    return first, last


def _manifest_histology_rows(
    manifest: Path,
    *,
    validate_counts: bool = True,
) -> dict[int, dict[str, str]]:
    with manifest.open(encoding="utf-8", newline="") as stream:
        candidates = [
            row
            for row in csv.DictReader(stream, delimiter="\t")
            if row.get("kind") == "histology_jpeg"
            and row.get("series_or_layer") in EXPECTED_PRESENT
        ]

    counts = Counter(row["series_or_layer"] for row in candidates)
    if validate_counts and counts != Counter(EXPECTED_PRESENT):
        raise ValueError(
            f"Expected histology counts {EXPECTED_PRESENT}, found {dict(counts)}"
        )

    by_section: dict[int, dict[str, str]] = {}
    for row in candidates:
        stain = row["series_or_layer"]
        section = int(row["section_number"])
        if not MINIMUM_SLOT <= section <= MAXIMUM_SLOT:
            raise ValueError(f"Section {section} lies outside the validated lattice")
        if section in by_section:
            raise ValueError(
                f"Multiple images occupy section {section}; mixed-view semantics "
                "require one observed image per cutting slot"
            )
        expected_prefix = "nissl" if stain == "nissl" else "ihc"
        expected_path = (
            f"{expected_prefix}/images_orig/image_{section:04d}.jpg"
        )
        if row["path"] != expected_path:
            raise ValueError(
                f"Filename/section mismatch for {stain} {section}: {row['path']!r}"
            )
        by_section[section] = row
    return by_section


def build_physical_rows(
    manifest: Path,
    *,
    section_range: tuple[int, int] = (MINIMUM_SLOT, MAXIMUM_SLOT),
    validate_counts: bool = True,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Build the canonical mixed-stain physical cutting table."""

    by_section = _manifest_histology_rows(
        manifest, validate_counts=validate_counts
    )
    first, last = section_range
    rows: list[dict[str, str]] = []
    for section in range(first, last + 1):
        source = by_section.get(section)
        stain = source["series_or_layer"] if source else ""
        status = "present" if source else "absent"
        row = {field: "" for field in PHYSICAL_FIELDS + INTERNAL_FIELDS}
        envelope = secondary_envelope(section)
        if source:
            observation_class = f"observed_{stain}"
            if envelope is None:
                block_id = ""
                block_status = "unresolved"
                envelope_context = "unresolved"
            else:
                block_id, envelope_context = envelope
                block_status = "matched_secondary_observation"
        else:
            observation_class = "unobserved"
            if envelope is None:
                block_id = ""
                block_status = "outside_secondary_envelopes"
                envelope_context = "between_observation_envelopes"
            else:
                block_id, envelope_context = envelope
                block_status = "inferred_within_secondary_envelope"
        row.update(
            {
                "physical_index": str(section - MINIMUM_SLOT),
                "specimen_id": SPECIMEN_ID,
                "allen_section_number": str(section),
                "serial_z_center_mm": f"{canonical_z_um(section) / 1000.0:.3f}",
                "section_thickness_um": f"{PHYSICAL_SECTION_THICKNESS_UM:.1f}",
                "serial_pitch_um": f"{CUTTING_LATTICE_SPACING_UM:.1f}",
                "image_present": "true" if source else "false",
                "block_id": block_id,
                "block_assignment_source": "secondary_reconstruction_table",
                "block_assignment_status": block_status,
                "secondary_envelope_context": envelope_context,
                "observation_class": observation_class,
                "section_number": str(section),
                "physical_serial_slot": str(section),
                "grid_index": str(section - MINIMUM_SLOT),
                "z_um": f"{canonical_z_um(section):.1f}",
                "status": status,
                "stain": stain,
                "cutting_lattice_spacing_um": (
                    f"{CUTTING_LATTICE_SPACING_UM:.1f}"
                ),
                "physical_section_thickness_um": (
                    f"{PHYSICAL_SECTION_THICKNESS_UM:.1f}"
                ),
            }
        )
        if source:
            row.update(
                {
                    "allen_section_image_id": source["allen_section_image_id"],
                    "allen_data_set_id": source.get("allen_data_set_id", ""),
                    "source_relative_path": source["path"],
                    "source_path": source["path"],
                    "source_image_id": source["allen_section_image_id"],
                    "source_width_px": source["width_px"],
                    "source_height_px": source["height_px"],
                    "source_pixel_size_um": source["pixel_size_um"],
                    "source_sha256": source["sha256"],
                    "nominal_series_interval_um": (
                        f"{NOMINAL_SERIES_INTERVAL_UM[stain]:.1f}"
                    ),
                }
            )
        rows.append(row)

    present = [row for row in rows if row["status"] == "present"]
    summary = {
        "minimum_slot": first,
        "maximum_slot": last,
        "row_count": len(rows),
        "present_count": len(present),
        "absent_count": len(rows) - len(present),
        "stain_counts": dict(Counter(row["stain"] for row in present)),
        "full_lattice_minimum_slot": MINIMUM_SLOT,
        "full_lattice_maximum_slot": MAXIMUM_SLOT,
        "full_lattice_count": NUMBER_OF_SLOTS,
    }
    return rows, summary


def _output_size(n_source: int, source_spacing: float, target_spacing: float) -> int:
    if n_source < 1:
        raise ValueError("source image dimensions must be positive")
    if source_spacing <= 0 or target_spacing <= 0:
        raise ValueError("pixel spacings must be positive")
    if n_source == 1:
        return 1
    source_extent = (n_source - 1) * source_spacing
    return max(1, int(math.floor(source_extent / target_spacing + 1e-12)) + 1)


def origin_preserving_prepared_to_source_transform(
    source_shape_yx: tuple[int, int],
    source_spacing_um: float,
    prepared_shape_yx: tuple[int, int],
    prepared_spacing_um: float,
) -> np.ndarray:
    """Map a common-origin prepared grid to the source pixel grid."""

    if min(*source_shape_yx, *prepared_shape_yx) < 1:
        raise ValueError("source and prepared image dimensions must be positive")
    if source_spacing_um <= 0 or prepared_spacing_um <= 0:
        raise ValueError("source and prepared pixel spacings must be positive")
    scale = prepared_spacing_um / source_spacing_um
    transform = np.eye(3, dtype=np.float64)
    transform[0, 0] = scale
    transform[1, 1] = scale
    return transform


def resample_origin_preserving_rgb(
    source_rgb: np.ndarray,
    source_spacing_um: float,
    target_spacing_um: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Antialias RGB data without translating an individual section."""

    source = np.asarray(source_rgb)
    if source.ndim != 3 or source.shape[2] < 3:
        raise ValueError(f"Expected an RGB image, found shape {source.shape}")
    source = source[..., :3].astype(np.float32)
    output_height = _output_size(
        source.shape[0], source_spacing_um, target_spacing_um
    )
    output_width = _output_size(
        source.shape[1], source_spacing_um, target_spacing_um
    )
    transform = origin_preserving_prepared_to_source_transform(
        source.shape[:2],
        source_spacing_um,
        (output_height, output_width),
        target_spacing_um,
    )

    down_factor = target_spacing_um / source_spacing_um
    sigma = max(0.0, 0.5 * math.sqrt(max(down_factor**2 - 1.0, 0.0)))
    filtered = gaussian_filter(
        source, sigma=(sigma, sigma, 0.0), mode="nearest"
    )
    yy, xx = np.meshgrid(
        np.arange(output_height, dtype=np.float64),
        np.arange(output_width, dtype=np.float64),
        indexing="ij",
    )
    source_x = transform[0, 0] * xx + transform[0, 2]
    source_y = transform[1, 1] * yy + transform[1, 2]
    channels = [
        map_coordinates(
            filtered[..., channel],
            [source_y, source_x],
            order=1,
            mode="nearest",
            prefilter=False,
        )
        for channel in range(3)
    ]
    prepared = np.stack(channels, axis=-1)
    return np.clip(np.rint(prepared), 0, 255).astype(np.uint8), transform


def embed_content_in_common_canvas(
    content_rgb: np.ndarray,
    canvas_shape_yx: tuple[int, int],
) -> np.ndarray:
    """Place section content at the shared pixel origin without translation."""

    content = np.asarray(content_rgb)
    if content.ndim != 3 or content.shape[2] != 3:
        raise ValueError(f"Expected RGB section content, found {content.shape}")
    canvas_height, canvas_width = canvas_shape_yx
    content_height, content_width = content.shape[:2]
    if (
        canvas_height < content_height
        or canvas_width < content_width
        or min(canvas_height, canvas_width) < 1
    ):
        raise ValueError(
            f"Content {content.shape[:2]} does not fit canvas {canvas_shape_yx}"
        )
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=content.dtype)
    canvas[:content_height, :content_width] = content
    return canvas


def _pinned_checkout() -> Path:
    configured = os.environ.get("EMLDDMM_REPO")
    checkout = (
        Path(configured).expanduser().resolve()
        if configured
        else PROJECT_ROOT.parent / "emlddmm"
    )
    if not (checkout / "histsetup.py").is_file():
        raise FileNotFoundError(f"EM-LDDMM checkout is unavailable: {checkout}")
    expected = PIN_FILE.read_text(encoding="utf-8").strip()
    actual = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if actual != expected:
        raise RuntimeError(
            f"EM-LDDMM checkout is {actual}; required pinned commit is {expected}"
        )
    return checkout


def _generate_and_validate_sidecars(
    image_dir: Path,
    rows: Iterable[dict[str, str]],
    target_spacing_um: float,
) -> None:
    checkout = _pinned_checkout()
    checkout_text = str(checkout)
    if checkout_text not in sys.path:
        sys.path.insert(0, checkout_text)
    histsetup = importlib.import_module("histsetup")
    with contextlib.redirect_stdout(io.StringIO()):
        histsetup.generate_sidecars(
            str(image_dir),
            ext=".tif",
            max_slice=NUMBER_OF_SLOTS,
            dtype="uint8",
            dv=[
                float(target_spacing_um),
                float(target_spacing_um),
                CUTTING_LATTICE_SPACING_UM,
            ],
            slice_downfactor=1,
            sep="_",
            fnumidx=-1,
            space="right-inferior-posterior",
        )

    for row in rows:
        if row["status"] != "present":
            continue
        image_path = image_dir / Path(row["prepared_path"]).name
        sidecar_path = image_path.with_suffix(".json")
        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
        width = int(row["prepared_width_px"])
        height = int(row["prepared_height_px"])
        expected_origin = [
            -(width - 1) * target_spacing_um / 2.0,
            -(height - 1) * target_spacing_um / 2.0,
            float(row["z_um"]),
        ]
        if payload["DataFile"] != image_path.name:
            raise ValueError(f"Sidecar DataFile mismatch: {sidecar_path}")
        if payload["Sizes"] != [3, width, height, 1]:
            raise ValueError(f"Sidecar Sizes mismatch: {sidecar_path}")
        expected_directions = [
            "none",
            [target_spacing_um, 0.0, 0.0],
            [0.0, target_spacing_um, 0.0],
            [0.0, 0.0, CUTTING_LATTICE_SPACING_UM],
        ]
        if payload["SpaceDirections"] != expected_directions:
            raise ValueError(f"Sidecar SpaceDirections mismatch: {sidecar_path}")
        if not np.allclose(payload["SpaceOrigin"], expected_origin, atol=1e-8):
            raise ValueError(f"Sidecar SpaceOrigin mismatch: {sidecar_path}")


def _write_tsv(path: Path, rows: Iterable[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=PHYSICAL_FIELDS,
            delimiter="\t",
            lineterminator="\n",
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_samples(
    view_dir: Path,
    rows: list[dict[str, str]],
    stains: frozenset[str],
    *,
    include_provenance: bool = False,
) -> dict[str, int]:
    counts = {"present": 0, "absent": 0}
    fields = ["sample_id", "participant_id", "species", "status"]
    if include_provenance:
        fields.extend(SAMPLE_PROVENANCE_FIELDS)
    with (view_dir / "samples.tsv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=fields,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            present = row["status"] == "present" and row["stain"] in stains
            sample_id = (
                Path(row["prepared_path"]).name
                if present
                else f"allen_708424_absent_{int(row['grid_index']) + 1:04d}.tif"
            )
            status = "present" if present else "absent"
            sample = {
                "sample_id": sample_id,
                "participant_id": SPECIMEN_ID,
                "species": "Homo sapiens",
                "status": status,
            }
            if include_provenance:
                sample.update({field: row[field] for field in SAMPLE_PROVENANCE_FIELDS})
            writer.writerow(sample)
            counts[status] += 1
    return counts


def calculate_canvas_audit(
    rows: list[dict[str, str]],
    target_spacing_um: float,
) -> dict[str, Any]:
    """Audit one common-origin canvas and its single global translation."""

    heights = np.zeros(len(rows), dtype=np.int64)
    widths = np.zeros(len(rows), dtype=np.int64)
    present_rows: list[dict[str, str]] = []
    for index, row in enumerate(rows):
        if row["prepared_path"]:
            heights[index] = int(row["prepared_height_px"])
            widths[index] = int(row["prepared_width_px"])
            present_rows.append(row)
    if not present_rows:
        raise ValueError("Cannot audit a loader canvas with no prepared images")

    default_height = int(np.quantile(heights, 0.95) * 1.01)
    default_width = int(np.quantile(widths, 0.95) * 1.01)
    maximum_height = int(np.max(heights))
    maximum_width = int(np.max(widths))
    default_extent_y = (default_height - 1) * target_spacing_um
    default_extent_x = (default_width - 1) * target_spacing_um
    section_audits: list[dict[str, Any]] = []
    for row in present_rows:
        extent_y = (int(row["prepared_height_px"]) - 1) * target_spacing_um
        extent_x = (int(row["prepared_width_px"]) - 1) * target_spacing_um
        section_audits.append(
            {
                "section_number": int(row["section_number"]),
                "stain": row["stain"],
                "content_origin_yx_px": [0, 0],
                "content_shape_yx": [
                    int(row["prepared_height_px"]),
                    int(row["prepared_width_px"]),
                ],
                "common_origin_extent_y_um": extent_y,
                "common_origin_extent_x_um": extent_x,
                "margin_y_um": default_extent_y - extent_y,
                "margin_x_um": default_extent_x - extent_x,
                "fits_default": (
                    extent_y <= default_extent_y + 1e-8
                    and extent_x <= default_extent_x + 1e-8
                ),
            }
        )
    fits = all(item["fits_default"] for item in section_audits)
    accepted_height = default_height if fits else maximum_height
    accepted_width = default_width if fits else maximum_width
    global_translation_xy_um = [
        -(accepted_width - 1) * target_spacing_um / 2.0,
        -(accepted_height - 1) * target_spacing_um / 2.0,
    ]
    for item in section_audits:
        item["globally_translated_bounds_xy_um"] = [
            global_translation_xy_um[0],
            global_translation_xy_um[0] + item["common_origin_extent_x_um"],
            global_translation_xy_um[1],
            global_translation_xy_um[1] + item["common_origin_extent_y_um"],
        ]
    return {
        "algorithm": (
            "common source-pixel origin; audit pinned quantile(size, 0.95) "
            "* 1.01 canvas and expand to uncropped maximum"
        ),
        "in_plane_placement_model": (
            "common_source_pixel_origin_single_global_translation"
        ),
        "sectionwise_centering": False,
        "target_spacing_um": target_spacing_um,
        "default_canvas_shape_yx": [default_height, default_width],
        "maximum_canvas_shape_yx": [maximum_height, maximum_width],
        "accepted_canvas_shape_yx": [accepted_height, accepted_width],
        "content_origin_yx_px": [0, 0],
        "global_translation_xy_um": global_translation_xy_um,
        "automatic_canvas_accepted": fits,
        "worst_default_margin_y_um": min(
            item["margin_y_um"] for item in section_audits
        ),
        "worst_default_margin_x_um": min(
            item["margin_x_um"] for item in section_audits
        ),
        "section_bounds": section_audits,
    }


def full_mixed_canvas_rows(
    rows: list[dict[str, str]], target_spacing_um: float
) -> list[dict[str, str]]:
    """Represent every observed stain's prepared bounds for one shared canvas."""

    canvas_rows: list[dict[str, str]] = []
    for row in rows:
        canvas_row = dict(row)
        if row["status"] != "present":
            canvas_row["prepared_path"] = ""
        elif not row["prepared_path"]:
            source_spacing = float(row["source_pixel_size_um"])
            canvas_row["prepared_path"] = "__domain_audit_only__"
            canvas_row["prepared_height_px"] = str(
                _output_size(
                    int(row["source_height_px"]),
                    source_spacing,
                    target_spacing_um,
                )
            )
            canvas_row["prepared_width_px"] = str(
                _output_size(
                    int(row["source_width_px"]),
                    source_spacing,
                    target_spacing_um,
                )
            )
        canvas_rows.append(canvas_row)
    return canvas_rows


def centered_axis(n: int, spacing: float) -> np.ndarray:
    """Return a centered regular axis for non-sectionwise coordinate domains."""

    axis = np.arange(n, dtype=np.float64) * spacing
    axis -= np.mean(axis)
    return axis


def accepted_loader_axes(
    rows: list[dict[str, str]],
    canvas_audit: dict[str, Any],
) -> list[np.ndarray]:
    if not rows:
        raise ValueError("rows must not be empty")
    sections = np.array(
        [
            int(row.get("allen_section_number") or row.get("section_number", ""))
            for row in rows
        ]
    )
    if not np.all(np.diff(sections) == 1):
        raise ValueError("loader rows must be consecutive cutting slots")
    z = canonical_z_axis_um()[sections - MINIMUM_SLOT]
    height, width = canvas_audit["accepted_canvas_shape_yx"]
    spacing = float(canvas_audit["target_spacing_um"])
    translation_x, translation_y = canvas_audit["global_translation_xy_um"]
    y = np.arange(height, dtype=np.float64) * spacing + translation_y
    x = np.arange(width, dtype=np.float64) * spacing + translation_x
    return [z, y, x]


def _relative_symlink(target: Path, link: Path) -> None:
    link.symlink_to(os.path.relpath(target, start=link.parent))


def _atomic_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def dataset_metadata(
    *,
    series: str,
    section_range: tuple[int, int],
    summary: dict[str, Any],
    view_counts: dict[str, dict[str, int]],
    target_pixel_size_um: float,
    manifest: Path,
    canvas_audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the machine-readable serial-geometry and provenance sidecar."""

    center_min = canonical_z_um(section_range[0]) / 1000.0
    center_max = canonical_z_um(section_range[1]) / 1000.0
    half_thickness_mm = PHYSICAL_SECTION_THICKNESS_UM / 2000.0
    result = {
        "dataset": "Allen specimen 708424 mixed-stain EM-LDDMM input",
        "prepared_metadata_schema": PREPARED_METADATA_SCHEMA,
        "specimen_id": int(SPECIMEN_ID),
        "series": series,
        "section_range": list(section_range),
        "summary": summary,
        "view_counts": view_counts,
        "target_pixel_size_um": target_pixel_size_um,
        "serial_geometry_model": "50um_physical_section_lattice",
        "physical_section_thickness_um": PHYSICAL_SECTION_THICKNESS_UM,
        "physical_serial_pitch_um": CUTTING_LATTICE_SPACING_UM,
        "cutting_lattice_spacing_um": CUTTING_LATTICE_SPACING_UM,
        "nominal_nissl_sampling_um": NOMINAL_SERIES_INTERVAL_UM["nissl"],
        "nominal_pv_sampling_um": NOMINAL_SERIES_INTERVAL_UM["pv"],
        "nominal_series_interval_um": NOMINAL_SERIES_INTERVAL_UM,
        "physical_position_count": summary["row_count"],
        "present_image_count": summary["present_count"],
        "nissl_count": summary["stain_counts"].get("nissl", 0),
        "pv_count": summary["stain_counts"].get("pv", 0),
        "serial_center_min_mm": center_min,
        "serial_center_max_mm": center_max,
        "serial_center_extent_mm": center_max - center_min,
        "serial_outer_face_min_mm": center_min - half_thickness_mm,
        "serial_outer_face_max_mm": center_max + half_thickness_mm,
        "serial_outer_face_extent_mm": center_max - center_min + 2 * half_thickness_mm,
        "serial_z_status": "nominal_cutting_center_coordinate",
        "anatomical_z_status": "not_established",
        "block_ids": list(SECONDARY_BLOCK_ENVELOPES),
        "secondary_block_reference": {
            "purpose": "Observation-envelope assignment only",
            "semantics": (
                "First and last matched observed sections; not physical slab faces"
            ),
            "publication_doi": "10.1016/j.media.2021.102265",
            "repository_revision": SECONDARY_BLOCK_REFERENCE_REVISION,
            "status": "secondary_non_allen_provenance",
        },
        "source_manifest": str(manifest),
        "source_manifest_sha256": sha256_file(manifest),
        "emlddmm_commit": PIN_FILE.read_text(encoding="utf-8").strip(),
        "coordinate_convention": (
            "serial_z_center_mm = (allen_section_number - 1458.5) * 0.05"
        ),
        "w0_behavior": (
            "Technical support versus padding, operationally generated at "
            "the pinned commit from first_loaded_channel > 0"
        ),
        "external_tissue_mask": None,
        "tissue_support_extent_status": (
            "not_computed_no_accepted_full_stack_support"
        ),
    }
    if canvas_audit is not None:
        result.update(
            {
                "in_plane_placement_model": canvas_audit[
                    "in_plane_placement_model"
                ],
                "sectionwise_in_plane_centering": canvas_audit[
                    "sectionwise_centering"
                ],
                "global_in_plane_translation_xy_um": canvas_audit[
                    "global_translation_xy_um"
                ],
                "prepared_canvas_shape_yx": canvas_audit[
                    "accepted_canvas_shape_yx"
                ],
            }
        )
    return result


def _copy_preserved_entry(source: Path, destination: Path) -> None:
    if source.is_symlink():
        destination.symlink_to(os.readlink(source))
    elif source.is_dir():
        shutil.copytree(source, destination, symlinks=True)
    else:
        shutil.copy2(source, destination)


def _preserve_unmanaged_entries(existing: Path, staging: Path) -> None:
    """Copy derivative content not owned by histology preparation."""

    for source in existing.iterdir():
        destination = staging / source.name
        if source.name in {"dataset.json", "metadata", "inputs"} or destination.exists():
            continue
        _copy_preserved_entry(source, destination)

    managed_metadata = {
        "physical_sections.tsv",
        "loader_canvas_audit.json",
        "first_channel_zero_diagnostic.json",
    }
    existing_metadata = existing / "metadata"
    if existing_metadata.is_dir():
        for source in existing_metadata.iterdir():
            destination = staging / "metadata" / source.name
            if source.name in managed_metadata or destination.exists():
                continue
            _copy_preserved_entry(source, destination)

    existing_inputs = existing / "inputs"
    if existing_inputs.is_dir():
        for source in existing_inputs.iterdir():
            destination = staging / "inputs" / source.name
            if source.name in {"sections", "views"} or destination.exists():
                continue
            _copy_preserved_entry(source, destination)


def _replace_prepared_dataset(staging: Path, output_dir: Path) -> None:
    """Install a staged derivative with rollback if directory replacement fails."""

    if not output_dir.exists():
        os.replace(staging, output_dir)
        return
    backup = staging.with_name(f"{staging.name}.previous")
    if backup.exists():
        raise FileExistsError(f"Unexpected derivative backup path: {backup}")
    os.replace(output_dir, backup)
    try:
        os.replace(staging, output_dir)
    except BaseException:
        os.replace(backup, output_dir)
        raise
    shutil.rmtree(backup)


def prepare_dataset(
    *,
    data_dir: Path,
    output_dir: Path,
    series: str,
    target_pixel_size_um: float,
    section_range: tuple[int, int],
    overwrite: bool = False,
) -> dict[str, Any]:
    """Create a complete prepared derivative using a sibling staging directory."""

    if series not in {"nissl", "pv", "all"}:
        raise ValueError("series must be nissl, pv, or all")
    if target_pixel_size_um <= 0:
        raise ValueError("target_pixel_size_um must be positive")
    manifest = data_dir / "metadata" / "manifest.tsv"
    rows, summary = build_physical_rows(
        manifest, section_range=section_range, validate_counts=True
    )
    selected_stains = (
        frozenset({"nissl", "pv"})
        if series == "all"
        else frozenset({series})
    )

    output_dir = output_dir.resolve()
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output exists: {output_dir}; use --verify-existing or --overwrite"
            )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        metadata_dir = staging / "metadata"
        sections_root = staging / "inputs" / "sections"
        views_root = staging / "inputs" / "views"
        metadata_dir.mkdir(parents=True)
        sections_root.mkdir(parents=True)
        views_root.mkdir(parents=True)

        # Every source image uses the same source-pixel origin. Determine one
        # uncropped canvas for the full mixed series, then translate that canvas
        # once. Individual sections are never recentered.
        canvas_input_rows = full_mixed_canvas_rows(rows, target_pixel_size_um)
        canvas_audit = calculate_canvas_audit(
            canvas_input_rows, target_pixel_size_um
        )
        canvas_height, canvas_width = canvas_audit[
            "accepted_canvas_shape_yx"
        ]

        rows_by_stain: dict[str, list[dict[str, str]]] = {
            stain: [] for stain in selected_stains
        }
        for row in rows:
            if row["status"] != "present" or row["stain"] not in selected_stains:
                continue
            stain = row["stain"]
            source_path = data_dir / row["source_path"]
            if sha256_file(source_path) != row["source_sha256"]:
                raise ValueError(f"Source checksum mismatch: {source_path}")
            with Image.open(source_path) as source_image:
                source_rgb = np.asarray(source_image.convert("RGB"))
            prepared_content, transform = resample_origin_preserving_rgb(
                source_rgb,
                float(row["source_pixel_size_um"]),
                target_pixel_size_um,
            )
            prepared = embed_content_in_common_canvas(
                prepared_content,
                (canvas_height, canvas_width),
            )
            content_height, content_width = prepared_content.shape[:2]

            stain_dir = sections_root / stain
            stain_dir.mkdir(exist_ok=True)
            frame = int(row["grid_index"]) + 1
            prepared_name = f"allen_708424_{stain}_{frame:04d}.tif"
            prepared_path = stain_dir / prepared_name
            Image.fromarray(prepared).save(
                prepared_path, format="TIFF", compression="tiff_deflate"
            )
            transform_name = f"{prepared_path.stem}_prepared-to-source.json"
            transform_path = stain_dir / transform_name
            transform_payload = {
                "coordinate_order": "x_y_homogeneous",
                "prepared_to_source_pixel": transform.tolist(),
                "source_to_prepared_pixel": np.linalg.inv(transform).tolist(),
                "source_shape_yx": list(source_rgb.shape[:2]),
                "prepared_content_shape_yx": list(prepared_content.shape[:2]),
                "prepared_canvas_shape_yx": list(prepared.shape[:2]),
                "prepared_content_origin_yx_px": [0, 0],
                "prepared_shape_yx": list(prepared.shape[:2]),
                "source_pixel_size_um": float(row["source_pixel_size_um"]),
                "prepared_pixel_size_um": target_pixel_size_um,
                "centered_pixel_domain": False,
                "sectionwise_centering": False,
                "global_canvas_centered": True,
                "global_translation_xy_um": canvas_audit[
                    "global_translation_xy_um"
                ],
            }
            _atomic_json(transform_path, transform_payload)

            zero_count = int(np.count_nonzero(prepared[..., 0] == 0))
            valid_count = int(prepared.shape[0] * prepared.shape[1])
            row.update(
                {
                    "prepared_relative_path": str(
                        Path("inputs")
                        / "sections"
                        / stain
                        / prepared_name
                    ),
                    "prepared_path": str(
                        Path("inputs") / "sections" / stain / prepared_name
                    ),
                    "prepared_width_px": str(prepared.shape[1]),
                    "prepared_height_px": str(prepared.shape[0]),
                    "prepared_pixel_size_um": f"{target_pixel_size_um:.6g}",
                    "prepared_sha256": sha256_file(prepared_path),
                    "prepared_to_source_transform": str(
                        Path("inputs")
                        / "sections"
                        / stain
                        / transform_name
                    ),
                    "first_channel_zero_count": str(zero_count),
                    "valid_pixel_count": str(valid_count),
                    "first_channel_zero_fraction": (
                        f"{zero_count / valid_count:.12g}"
                    ),
                }
            )
            rows_by_stain[stain].append(row)

        for stain, stain_rows in rows_by_stain.items():
            _generate_and_validate_sidecars(
                sections_root / stain, stain_rows, target_pixel_size_um
            )

        view_names = (
            ["HIST_ALL", "HIST_NISSL", "HIST_PV"]
            if series == "all"
            else ["HIST_NISSL" if series == "nissl" else "HIST_PV"]
        )
        view_counts: dict[str, dict[str, int]] = {}
        for view_name in view_names:
            view_dir = views_root / view_name
            view_dir.mkdir()
            stains = VIEW_STAINS[view_name]
            for row in rows:
                if not row["prepared_path"] or row["stain"] not in stains:
                    continue
                image_path = staging / row["prepared_path"]
                _relative_symlink(image_path, view_dir / image_path.name)
                sidecar_path = image_path.with_suffix(".json")
                _relative_symlink(sidecar_path, view_dir / sidecar_path.name)
            view_counts[view_name] = _write_samples(
                view_dir,
                rows,
                stains,
                include_provenance=view_name == "HIST_ALL",
            )

        _write_tsv(metadata_dir / "physical_sections.tsv", rows)
        _atomic_json(metadata_dir / "loader_canvas_audit.json", canvas_audit)

        zero_diagnostics: dict[str, Any] = {"by_stain": {}}
        for stain in sorted(selected_stains):
            stain_rows = rows_by_stain[stain]
            zero = sum(int(row["first_channel_zero_count"]) for row in stain_rows)
            valid = sum(int(row["valid_pixel_count"]) for row in stain_rows)
            zero_diagnostics["by_stain"][stain] = {
                "first_channel_zero_count": zero,
                "valid_pixel_count": valid,
                "fraction": zero / valid if valid else None,
            }
        total_zero = sum(
            item["first_channel_zero_count"]
            for item in zero_diagnostics["by_stain"].values()
        )
        total_valid = sum(
            item["valid_pixel_count"]
            for item in zero_diagnostics["by_stain"].values()
        )
        zero_diagnostics["overall"] = {
            "first_channel_zero_count": total_zero,
            "valid_pixel_count": total_valid,
            "fraction": total_zero / total_valid if total_valid else None,
            "interpretation": (
                "Diagnostic for pinned W0 first-channel-positive behavior; "
                "not an anatomical tissue mask"
            ),
        }
        _atomic_json(
            metadata_dir / "first_channel_zero_diagnostic.json",
            zero_diagnostics,
        )

        dataset = dataset_metadata(
            series=series,
            section_range=section_range,
            summary=summary,
            view_counts=view_counts,
            target_pixel_size_um=target_pixel_size_um,
            manifest=manifest,
            canvas_audit=canvas_audit,
        )
        _atomic_json(staging / "dataset.json", dataset)
        if output_dir.exists():
            _preserve_unmanaged_entries(output_dir, staging)
        _replace_prepared_dataset(staging, output_dir)
        return dataset
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _hydrate_existing_prepared(
    rows: list[dict[str, str]],
    data_dir: Path,
    output_dir: Path,
    canvas_audit: dict[str, Any],
) -> None:
    """Validate immutable payloads and attach their existing derivative metadata."""

    for row in rows:
        if row["status"] != "present":
            continue
        source_path = data_dir / row["source_relative_path"]
        if sha256_file(source_path) != row["source_sha256"]:
            raise ValueError(f"Source checksum mismatch: {source_path}")
        frame = int(row["physical_index"]) + 1
        prepared_name = f"allen_708424_{row['stain']}_{frame:04d}.tif"
        prepared_relative = Path("inputs") / "sections" / row["stain"] / prepared_name
        prepared_path = output_dir / prepared_relative
        if not prepared_path.is_file():
            raise FileNotFoundError(f"Missing prepared image: {prepared_path}")
        sidecar_path = prepared_path.with_suffix(".json")
        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
        if payload.get("DataFile") != prepared_name:
            raise ValueError(f"Sidecar DataFile mismatch: {sidecar_path}")
        if payload.get("SpaceDirections", [None])[-1] != [
            0.0,
            0.0,
            CUTTING_LATTICE_SPACING_UM,
        ]:
            raise ValueError(f"Sidecar serial pitch mismatch: {sidecar_path}")
        expected_z_um = float(row["serial_z_center_mm"]) * 1000.0
        if not np.isclose(
            payload.get("SpaceOrigin", [0, 0, float("nan")])[-1],
            expected_z_um,
        ):
            raise ValueError(f"Sidecar serial origin mismatch: {sidecar_path}")
        sizes = payload.get("Sizes", [])
        if len(sizes) != 4 or sizes[0] != 3 or sizes[-1] != 1:
            raise ValueError(f"Invalid prepared sidecar sizes: {sidecar_path}")
        expected_height, expected_width = canvas_audit[
            "accepted_canvas_shape_yx"
        ]
        if sizes[1:3] != [expected_width, expected_height]:
            raise ValueError(f"Prepared image is not on the common canvas: {sidecar_path}")
        if not np.allclose(
            payload["SpaceOrigin"][:2],
            canvas_audit["global_translation_xy_um"],
            atol=1e-8,
        ):
            raise ValueError(f"Prepared image lacks the global translation: {sidecar_path}")
        transform_path = prepared_path.with_name(
            f"{prepared_path.stem}_prepared-to-source.json"
        )
        transform = json.loads(transform_path.read_text(encoding="utf-8"))
        if (
            transform.get("sectionwise_centering") is not False
            or transform.get("prepared_content_origin_yx_px") != [0, 0]
            or transform.get("prepared_canvas_shape_yx")
            != [expected_height, expected_width]
        ):
            raise ValueError(f"Invalid global-canvas transform: {transform_path}")
        prepared_sha256 = sha256_file(prepared_path)
        row.update(
            {
                "prepared_relative_path": str(prepared_relative),
                "prepared_sha256": prepared_sha256,
                "prepared_path": str(prepared_relative),
                "prepared_width_px": str(sizes[1]),
                "prepared_height_px": str(sizes[2]),
                "prepared_pixel_size_um": f"{DEFAULT_TARGET_PIXEL_SIZE_UM:.1f}",
            }
        )


def _validate_canonical_rows(
    rows: list[dict[str, str]],
    *,
    section_range: tuple[int, int],
) -> dict[str, Any]:
    first, last = section_range
    expected_sections = list(range(first, last + 1))
    sections = [int(row["allen_section_number"]) for row in rows]
    if sections != expected_sections or len(set(sections)) != len(sections):
        raise ValueError("physical_sections.tsv is not the unique consecutive range")
    for position, row in enumerate(rows):
        section = sections[position]
        expected_index = section - MINIMUM_SLOT
        if int(row["physical_index"]) != expected_index:
            raise ValueError(f"Invalid physical_index for section {section}")
        if row["specimen_id"] != SPECIMEN_ID:
            raise ValueError(f"Invalid specimen_id for section {section}")
        if not np.isclose(
            float(row["serial_z_center_mm"]),
            canonical_z_um(section) / 1000.0,
            atol=1e-12,
        ):
            raise ValueError(f"Invalid serial z center for section {section}")
        if float(row["section_thickness_um"]) != PHYSICAL_SECTION_THICKNESS_UM:
            raise ValueError(f"Invalid section thickness for section {section}")
        if float(row["serial_pitch_um"]) != CUTTING_LATTICE_SPACING_UM:
            raise ValueError(f"Invalid serial pitch for section {section}")
        present = row["image_present"] == "true"
        expected_class = f"observed_{row['stain']}" if present else "unobserved"
        if row["observation_class"] != expected_class:
            raise ValueError(f"Invalid observation class for section {section}")
        image_fields = (
            "stain",
            "allen_section_image_id",
            "allen_data_set_id",
            "source_relative_path",
            "prepared_relative_path",
            "source_sha256",
            "prepared_sha256",
            "nominal_series_interval_um",
        )
        if present:
            if row["stain"] not in EXPECTED_PRESENT or any(
                not row[field] for field in image_fields
            ):
                raise ValueError(f"Incomplete observed provenance at section {section}")
            if float(row["nominal_series_interval_um"]) != (
                NOMINAL_SERIES_INTERVAL_UM[row["stain"]]
            ):
                raise ValueError(f"Invalid nominal interval at section {section}")
        elif any(row[field] for field in image_fields):
            raise ValueError(f"Unobserved row has image metadata at section {section}")
    present_rows = [row for row in rows if row["image_present"] == "true"]
    summary = {
        "minimum_slot": first,
        "maximum_slot": last,
        "row_count": len(rows),
        "present_count": len(present_rows),
        "absent_count": len(rows) - len(present_rows),
        "stain_counts": dict(Counter(row["stain"] for row in present_rows)),
        "full_lattice_minimum_slot": MINIMUM_SLOT,
        "full_lattice_maximum_slot": MAXIMUM_SLOT,
        "full_lattice_count": NUMBER_OF_SLOTS,
    }
    if section_range == (MINIMUM_SLOT, MAXIMUM_SLOT):
        if summary["present_count"] != sum(EXPECTED_PRESENT.values()):
            raise ValueError("Full lattice does not retain all 928 observations")
        if summary["stain_counts"] != EXPECTED_PRESENT:
            raise ValueError(
                "Full lattice stain counts do not match local Allen metadata"
            )
        unmatched = [
            row
            for row in rows
            if row["allen_section_number"] == "2130"
        ][0]
        if not (
            unmatched["allen_section_image_id"] == "146699677"
            and unmatched["observation_class"] == "observed_pv"
            and unmatched["block_assignment_status"] == "unresolved"
            and unmatched["block_id"] == ""
        ):
            raise ValueError(
                "PV section 2130 must remain observed and block-unresolved"
            )
    return summary


def _view_counts(rows: list[dict[str, str]]) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for view_name, stains in VIEW_STAINS.items():
        present = sum(
            row["image_present"] == "true" and row["stain"] in stains
            for row in rows
        )
        result[view_name] = {"present": present, "absent": len(rows) - present}
    return result


def refresh_metadata(*, data_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Refresh metadata through a staged, rollback-capable file transaction."""

    output_dir = output_dir.expanduser().resolve()
    manifest = data_dir / "metadata" / "manifest.tsv"
    rows, _ = build_physical_rows(manifest, validate_counts=True)
    canvas_audit = json.loads(
        (output_dir / "metadata" / "loader_canvas_audit.json").read_text(
            encoding="utf-8"
        )
    )
    if canvas_audit.get("sectionwise_centering") is not False:
        raise ValueError("Metadata refresh requires the global-canvas derivative")
    _hydrate_existing_prepared(rows, data_dir, output_dir, canvas_audit)
    summary = _validate_canonical_rows(
        rows, section_range=(MINIMUM_SLOT, MAXIMUM_SLOT)
    )
    view_counts = _view_counts(rows)
    dataset = dataset_metadata(
        series="all",
        section_range=(MINIMUM_SLOT, MAXIMUM_SLOT),
        summary=summary,
        view_counts=view_counts,
        target_pixel_size_um=DEFAULT_TARGET_PIXEL_SIZE_UM,
        manifest=manifest,
        canvas_audit=canvas_audit,
    )

    staging = Path(
        tempfile.mkdtemp(prefix=".metadata-refresh.", dir=output_dir.parent)
    )
    try:
        physical_candidate = staging / "physical_sections.tsv"
        samples_dir = staging / "HIST_ALL"
        samples_dir.mkdir()
        dataset_candidate = staging / "dataset.json"
        _write_tsv(physical_candidate, rows)
        _write_samples(
            samples_dir,
            rows,
            VIEW_STAINS["HIST_ALL"],
            include_provenance=True,
        )
        _atomic_json(dataset_candidate, dataset)

        with physical_candidate.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream, delimiter="\t")
            staged_rows = list(reader)
            if list(reader.fieldnames or ()) != PHYSICAL_FIELDS:
                raise ValueError("Staged physical_sections.tsv schema mismatch")
        _validate_canonical_rows(
            staged_rows, section_range=(MINIMUM_SLOT, MAXIMUM_SLOT)
        )
        samples_candidate = samples_dir / "samples.tsv"
        with samples_candidate.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream, delimiter="\t")
            sample_rows = list(reader)
            expected_fields = [
                "sample_id",
                "participant_id",
                "species",
                "status",
                *SAMPLE_PROVENANCE_FIELDS,
            ]
            if list(reader.fieldnames or ()) != expected_fields:
                raise ValueError("Staged HIST_ALL samples.tsv schema mismatch")
        if len(sample_rows) != len(staged_rows):
            raise ValueError("Staged HIST_ALL samples are not positionally complete")

        targets = [
            (physical_candidate, output_dir / "metadata" / "physical_sections.tsv"),
            (
                samples_candidate,
                output_dir / "inputs" / "views" / "HIST_ALL" / "samples.tsv",
            ),
            (dataset_candidate, output_dir / "dataset.json"),
        ]
        backups = staging / "backups"
        backups.mkdir()
        before_hashes: dict[Path, str] = {}
        for index, (_, target) in enumerate(targets):
            before_hashes[target] = sha256_file(target)
            shutil.copy2(target, backups / str(index))
        LOG.info(
            "Validated staged metadata; replacing %d files with rollback backups",
            len(targets),
        )
        replaced: list[tuple[int, Path]] = []
        try:
            for index, (candidate, target) in enumerate(targets):
                os.replace(candidate, target)
                replaced.append((index, target))
        except BaseException:
            for index, target in reversed(replaced):
                os.replace(backups / str(index), target)
            raise
        LOG.info(
            "Metadata refresh complete; previous hashes: %s",
            {str(path): digest for path, digest in before_hashes.items()},
        )
        return dataset
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _validated_preserved_reflection_plane(
    symmetry: dict[str, Any],
    shape_yx: tuple[int, int],
    origin_xy_um: tuple[float, float],
    spacing_um: float,
) -> float:
    plane = symmetry.get("reflection_plane", {})
    width = shape_yx[1]
    expected_adjacent = [width // 2 - 1, width // 2]
    try:
        coordinate = float(plane["coordinate_um"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Canonical reflection coordinate is invalid") from exc
    grid_coordinate = origin_xy_um[0] + (width / 2.0 - 0.5) * spacing_um
    if (
        width % 2
        or plane.get("axis") != "x"
        or plane.get("location") != "between_columns"
        or plane.get("adjacent_column_indices") != expected_adjacent
        or not np.isfinite(coordinate)
        or not np.isclose(coordinate, grid_coordinate, atol=1e-6, rtol=0.0)
    ):
        raise ValueError("Canonical reflection plane is inconsistent with its grid")
    return coordinate


def prepare_preserved_source_grid(
    *,
    source_dataset: Path,
    output_dir: Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Package a validated canonical bilateral source without pixel operations."""

    source_dataset = source_dataset.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if source_dataset == output_dir:
        raise ValueError("Source and prepared symmetric derivatives must be distinct")
    source_metadata = json.loads(
        (source_dataset / "dataset.json").read_text(encoding="utf-8")
    )
    compatibility = json.loads(
        (source_dataset / "metadata" / "compatibility.json").read_text(
            encoding="utf-8"
        )
    )
    symmetry = json.loads(
        (source_dataset / "metadata" / "symmetry.json").read_text(
            encoding="utf-8"
        )
    )
    if (
        source_metadata.get("space_name") != "HIST_SYMMETRIC"
        or compatibility.get("source_grid_preservable") is not True
        or compatibility.get("requires_preparation_resampling") is not False
    ):
        raise ValueError(
            "Canonical source does not authorize interpolation-free preparation"
        )
    if source_metadata.get("pixels_resampled") is not False:
        raise ValueError("Canonical symmetric source reports prior resampling")
    expected_shape = tuple(int(value) for value in symmetry["bilateral_shape_yx"])
    expected_origin = tuple(
        float(value) for value in symmetry["bilateral_origin_xy_um"]
    )
    expected_spacing = float(symmetry["pixel_size_um"])
    reflection_coordinate = _validated_preserved_reflection_plane(
        symmetry, expected_shape, expected_origin, expected_spacing
    )

    physical_path = source_dataset / "metadata" / "physical_sections.tsv"
    with physical_path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        source_rows = list(reader)
        fields = list(reader.fieldnames or ())
    if fields != PHYSICAL_FIELDS or len(source_rows) != NUMBER_OF_SLOTS:
        raise ValueError("Symmetric source physical lattice schema is incompatible")
    counts = Counter(
        row["stain"]
        for row in source_rows
        if row["image_present"] == "true"
    )
    metadata_counts = Counter(
        {
            stain: int(source_metadata.get(f"{stain}_count", 0))
            for stain in EXPECTED_PRESENT
        }
    )
    metadata_counts += Counter()
    if counts != metadata_counts:
        raise ValueError(
            f"Symmetric source rows {counts} differ from dataset metadata "
            f"{metadata_counts}"
        )
    source_coordinates = [float(row["serial_z_center_mm"]) for row in source_rows]
    source_pitches = [float(row["serial_pitch_um"]) for row in source_rows]
    if not np.all(np.isfinite(source_coordinates + source_pitches)):
        raise ValueError("Symmetric source serial metadata is nonfinite")

    if output_dir.exists() and not overwrite:
        raise FileExistsError(
            f"Output exists: {output_dir}; use --verify-existing or --overwrite"
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        metadata_dir = staging / "metadata"
        sections_root = staging / "inputs" / "sections"
        views_root = staging / "inputs" / "views"
        metadata_dir.mkdir(parents=True)
        sections_root.mkdir(parents=True)
        views_root.mkdir(parents=True)
        rows: list[dict[str, str]] = []
        for source_row in source_rows:
            row = dict(source_row)
            row.update(
                {
                    "section_number": row["allen_section_number"],
                    "grid_index": row["physical_index"],
                    "status": (
                        "present" if row["image_present"] == "true" else "absent"
                    ),
                    "z_um": str(float(row["serial_z_center_mm"]) * 1000.0),
                }
            )
            if row["status"] == "present":
                source_path = source_dataset / row["prepared_relative_path"]
                if sha256_file(source_path) != row["prepared_sha256"]:
                    raise ValueError(f"Symmetric source checksum mismatch: {source_path}")
                with Image.open(source_path) as image:
                    if image.size != (expected_shape[1], expected_shape[0]):
                        raise ValueError(f"Symmetric source image is off-grid: {source_path}")
                sidecar_path = source_path.with_suffix(".json")
                sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
                if (
                    sidecar.get("Sizes") != [3, expected_shape[1], expected_shape[0], 1]
                    or not np.allclose(sidecar.get("SpaceOrigin", [])[:2], expected_origin)
                    or sidecar.get("SpaceDirections", [None, None])[1]
                    != [expected_spacing, 0.0, 0.0]
                    or sidecar.get("SpaceDirections", [None, None, None])[2]
                    != [0.0, expected_spacing, 0.0]
                ):
                    raise ValueError(f"Symmetric source sidecar is off-grid: {sidecar_path}")
                stain_dir = sections_root / row["stain"]
                stain_dir.mkdir(exist_ok=True)
                prepared_path = stain_dir / source_path.name
                shutil.copy2(source_path, prepared_path)
                shutil.copy2(sidecar_path, prepared_path.with_suffix(".json"))
                if sha256_file(prepared_path) != row["prepared_sha256"]:
                    raise ValueError("Preserve-source-grid copy changed image bytes")
                row["prepared_relative_path"] = prepared_path.relative_to(staging).as_posix()
                row["prepared_path"] = row["prepared_relative_path"]
                row["prepared_width_px"] = str(expected_shape[1])
                row["prepared_height_px"] = str(expected_shape[0])
                row["prepared_pixel_size_um"] = str(expected_spacing)
            else:
                row["prepared_path"] = ""
            rows.append(row)

        view_counts: dict[str, dict[str, int]] = {}
        for view_name, stains in VIEW_STAINS.items():
            view_dir = views_root / view_name
            view_dir.mkdir()
            for row in rows:
                if row["status"] != "present" or row["stain"] not in stains:
                    continue
                image_path = staging / row["prepared_path"]
                _relative_symlink(image_path, view_dir / image_path.name)
                _relative_symlink(
                    image_path.with_suffix(".json"),
                    view_dir / image_path.with_suffix(".json").name,
                )
            view_counts[view_name] = _write_samples(
                view_dir,
                rows,
                stains,
                include_provenance=view_name == "HIST_ALL",
            )
        expected_view_counts = {}
        for view_name, stains in VIEW_STAINS.items():
            present = sum(counts[stain] for stain in stains)
            expected_view_counts[view_name] = {
                "present": present,
                "absent": NUMBER_OF_SLOTS - present,
            }
        if view_counts != expected_view_counts:
            raise ValueError(f"Preserved view counts differ: {view_counts}")

        _write_tsv(metadata_dir / "physical_sections.tsv", rows)
        for name in (
            "symmetry.json",
            "compatibility.json",
            "hemisphere_origin_mask.tif",
        ):
            shutil.copy2(source_dataset / "metadata" / name, metadata_dir / name)
        annotations = source_dataset / "annotations"
        if annotations.is_dir():
            shutil.copytree(annotations, staging / "annotations")
            inventory = source_dataset / "metadata" / "annotations.tsv"
            if inventory.is_file():
                shutil.copy2(inventory, metadata_dir / "annotations.tsv")
        canvas_audit = {
            "accepted_canvas_shape_yx": list(expected_shape),
            "target_spacing_um": expected_spacing,
            "global_translation_xy_um": list(expected_origin),
            "in_plane_placement_model": "preserved_canonical_symmetric_source_grid",
            "sectionwise_centering": False,
            "preserve_source_grid": True,
            "reflection_plane_coordinate_um": reflection_coordinate,
            "pixel_operations": [],
        }
        _atomic_json(metadata_dir / "loader_canvas_audit.json", canvas_audit)
        dataset = {
            **source_metadata,
            "dataset": "Allen specimen 708424 symmetric EM-LDDMM package",
            "prepared_metadata_schema": PREPARED_METADATA_SCHEMA,
            "preparation_mode": "preserve_source_grid",
            "source_dataset": str(source_dataset),
            "source_dataset_json_sha256": sha256_file(source_dataset / "dataset.json"),
            "source_symmetry_sha256": sha256_file(
                source_dataset / "metadata" / "symmetry.json"
            ),
            "view_counts": view_counts,
            "prepared_canvas_shape_yx": list(expected_shape),
            "sectionwise_in_plane_centering": False,
            "images_reflected_during_preparation": False,
            "images_resampled_during_preparation": False,
            "images_cropped_during_preparation": False,
        }
        _atomic_json(staging / "dataset.json", dataset)
        if output_dir.exists():
            _preserve_unmanaged_entries(output_dir, staging)
        _replace_prepared_dataset(staging, output_dir)
        return dataset
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _required_preserved_view_counts(
    rows: list[dict[str, str]],
) -> dict[str, dict[str, int]]:
    present_stains = Counter(
        row["stain"] for row in rows if row["image_present"] == "true"
    )
    if present_stains.get("nissl", 0) != 641:
        raise ValueError("Preserved symmetric source must contain 641 Nissl sections")
    return {
        view_name: {
            "present": sum(present_stains.get(stain, 0) for stain in stains),
            "absent": len(rows)
            - sum(present_stains.get(stain, 0) for stain in stains),
        }
        for view_name, stains in VIEW_STAINS.items()
        if any(present_stains.get(stain, 0) for stain in stains)
    }


def verify_preserved_source_grid(output_dir: Path) -> dict[str, Any]:
    dataset = json.loads((output_dir / "dataset.json").read_text(encoding="utf-8"))
    if dataset.get("preparation_mode") != "preserve_source_grid":
        raise ValueError("Prepared derivative is not preserve-source-grid mode")
    symmetry_path = output_dir / "metadata" / "symmetry.json"
    if sha256_file(symmetry_path) != dataset["source_symmetry_sha256"]:
        raise ValueError("Prepared reflection metadata differs from canonical source")
    symmetry = json.loads(symmetry_path.read_text(encoding="utf-8"))
    expected_shape = symmetry["bilateral_shape_yx"]
    audit = json.loads(
        (output_dir / "metadata" / "loader_canvas_audit.json").read_text(
            encoding="utf-8"
        )
    )
    reflection_coordinate = _validated_preserved_reflection_plane(
        symmetry,
        tuple(int(value) for value in expected_shape),
        tuple(float(value) for value in audit["global_translation_xy_um"]),
        float(audit["target_spacing_um"]),
    )
    if not np.isclose(
        float(audit["reflection_plane_coordinate_um"]),
        reflection_coordinate,
        atol=1e-6,
        rtol=0.0,
    ):
        raise ValueError("Prepared loader audit changed the reflection coordinate")
    rows_path = output_dir / "metadata" / "physical_sections.tsv"
    with rows_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    if len(rows) != NUMBER_OF_SLOTS:
        raise ValueError("Prepared symmetric lattice does not contain 2,846 rows")
    for row in rows:
        if row["image_present"] != "true":
            continue
        path = output_dir / row["prepared_relative_path"]
        if sha256_file(path) != row["prepared_sha256"]:
            raise ValueError(f"Prepared symmetric image checksum mismatch: {path}")
        with Image.open(path) as image:
            if list(image.size[::-1]) != expected_shape:
                raise ValueError(f"Prepared symmetric image dimensions changed: {path}")
    required_views = _required_preserved_view_counts(rows)
    declared_views = dataset.get("view_counts", {})
    for view_name, expected in required_views.items():
        if declared_views.get(view_name) != expected:
            raise ValueError(
                f"Prepared symmetric {view_name} declaration changed"
            )
        samples = output_dir / "inputs" / "views" / view_name / "samples.tsv"
        with samples.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream, delimiter="\t")
            sample_rows = list(reader)
            fields = list(reader.fieldnames or ())
        if fields[:4] != ["sample_id", "participant_id", "species", "status"]:
            raise ValueError(f"Invalid samples.tsv loader prefix for {view_name}")
        if len(sample_rows) != len(rows):
            raise ValueError(f"TSV/z row mismatch for {view_name}")
        stains = VIEW_STAINS[view_name]
        actual_counts = {"present": 0, "absent": 0}
        for source_row, sample_row in zip(rows, sample_rows, strict=True):
            present = (
                source_row["image_present"] == "true"
                and source_row["stain"] in stains
            )
            expected_status = "present" if present else "absent"
            if sample_row["status"] != expected_status:
                raise ValueError(f"Prepared symmetric {view_name} inventory changed")
            actual_counts[expected_status] += 1
            if present:
                expected_name = Path(source_row["prepared_relative_path"]).name
                if sample_row["sample_id"] != expected_name:
                    raise ValueError(
                        f"Prepared symmetric {view_name} sample identity changed"
                    )
                view_image = samples.parent / expected_name
                if not view_image.is_file() or not view_image.with_suffix(".json").is_file():
                    raise FileNotFoundError(
                        f"Missing prepared symmetric {view_name} sample: {view_image}"
                    )
        if actual_counts != expected:
            raise ValueError(f"Prepared symmetric {view_name} counts changed")
    return dataset


def verify_existing(output_dir: Path) -> dict[str, Any]:
    dataset_path = output_dir / "dataset.json"
    physical_path = output_dir / "metadata" / "physical_sections.tsv"
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    if dataset.get("preparation_mode") == "preserve_source_grid":
        return verify_preserved_source_grid(output_dir)
    if dataset.get("prepared_metadata_schema") != PREPARED_METADATA_SCHEMA:
        raise ValueError("Prepared metadata schema is not current")
    canvas_audit = json.loads(
        (output_dir / "metadata" / "loader_canvas_audit.json").read_text(
            encoding="utf-8"
        )
    )
    if (
        canvas_audit.get("sectionwise_centering") is not False
        or dataset.get("sectionwise_in_plane_centering") is not False
    ):
        raise ValueError("Prepared derivative must use one global translation")
    with physical_path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        rows = list(reader)
        if list(reader.fieldnames or ()) != PHYSICAL_FIELDS:
            raise ValueError("physical_sections.tsv has an unexpected schema")
    expected_first, expected_last = dataset["section_range"]
    expected_sections = list(range(expected_first, expected_last + 1))
    actual_sections = [int(row["allen_section_number"]) for row in rows]
    if actual_sections != expected_sections:
        raise ValueError("physical_sections.tsv is not a consecutive requested range")
    _validate_canonical_rows(rows, section_range=(expected_first, expected_last))
    expected_height, expected_width = canvas_audit["accepted_canvas_shape_yx"]
    for row in rows:
        if row["prepared_relative_path"]:
            path = output_dir / row["prepared_relative_path"]
            if sha256_file(path) != row["prepared_sha256"]:
                raise ValueError(f"Prepared checksum mismatch: {path}")
            sidecar = path.with_suffix(".json")
            if not sidecar.is_file():
                raise FileNotFoundError(f"Missing sidecar: {sidecar}")
            sidecar_payload = json.loads(sidecar.read_text(encoding="utf-8"))
            if sidecar_payload["Sizes"][1:3] != [
                expected_width,
                expected_height,
            ] or not np.allclose(
                sidecar_payload["SpaceOrigin"][:2],
                canvas_audit["global_translation_xy_um"],
                atol=1e-8,
            ):
                raise ValueError(f"Prepared section is not globally placed: {path}")
            transform_path = path.with_name(
                f"{path.stem}_prepared-to-source.json"
            )
            transform = json.loads(transform_path.read_text(encoding="utf-8"))
            if (
                transform.get("sectionwise_centering") is not False
                or transform.get("prepared_content_origin_yx_px") != [0, 0]
            ):
                raise ValueError(
                    f"Prepared transform uses sectionwise centering: {transform_path}"
                )
    for view_name, counts in dataset["view_counts"].items():
        samples = output_dir / "inputs" / "views" / view_name / "samples.tsv"
        with samples.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream, delimiter="\t")
            sample_rows = list(reader)
            sample_fields = list(reader.fieldnames or ())
        required_prefix = ["sample_id", "participant_id", "species", "status"]
        if sample_fields[:4] != required_prefix:
            raise ValueError(f"Invalid samples.tsv loader prefix for {view_name}")
        if view_name == "HIST_ALL" and sample_fields != [
            *required_prefix,
            *SAMPLE_PROVENANCE_FIELDS,
        ]:
            raise ValueError("HIST_ALL samples.tsv lacks canonical provenance")
        if len(sample_rows) != len(rows):
            raise ValueError(f"TSV/z row mismatch for {view_name}")
        actual_counts = Counter(row["status"] for row in sample_rows)
        if dict(actual_counts) != counts:
            raise ValueError(f"View count mismatch for {view_name}")
        if any(
            row["participant_id"] != "708424"
            or row["species"] != "Homo sapiens"
            or row["status"] not in {"present", "absent"}
            for row in sample_rows
        ):
            raise ValueError(f"Invalid samples.tsv metadata for {view_name}")
    return dataset


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--source-dataset",
        type=Path,
        help="Canonical symmetric source to package without pixel operations",
    )
    parser.add_argument(
        "--preserve-source-grid",
        action="store_true",
        help="Copy a validated bilateral source grid without crop or resampling",
    )
    parser.add_argument("--series", choices=("nissl", "pv", "all"), default="all")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--target-pixel-size-um",
        type=float,
        default=DEFAULT_TARGET_PIXEL_SIZE_UM,
    )
    parser.add_argument("--section-range")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--overwrite", action="store_true")
    mode.add_argument("--verify-existing", action="store_true")
    mode.add_argument(
        "--refresh-metadata",
        action="store_true",
        help="Refresh only canonical TSV/JSON metadata with staged rollback",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    logging.basicConfig(level=args.log_level, format="%(levelname)s %(message)s")
    section_range = parse_section_range(args.section_range)
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = (
            DEFAULT_SYMMETRIC_OUTPUT_DIR
            if args.source_dataset is not None
            else DEFAULT_OUTPUT_DIR
        )
        if (
            args.source_dataset is None
            and section_range != (MINIMUM_SLOT, MAXIMUM_SLOT)
        ):
            output_dir = (
                output_dir
                / "pilot"
                / f"slots-{section_range[0]:04d}-{section_range[1]:04d}"
            )

    if (args.source_dataset is None) != (not args.preserve_source_grid):
        raise ValueError(
            "--source-dataset and --preserve-source-grid must be used together"
        )
    if args.source_dataset is not None and (
        section_range != (MINIMUM_SLOT, MAXIMUM_SLOT)
        or args.series != "all"
        or args.refresh_metadata
    ):
        raise ValueError(
            "Symmetric preserve-source-grid preparation requires full --series all"
        )

    if args.verify_existing:
        dataset = verify_existing(output_dir)
    elif args.source_dataset is not None:
        dataset = prepare_preserved_source_grid(
            source_dataset=args.source_dataset,
            output_dir=output_dir,
            overwrite=args.overwrite,
        )
    elif args.refresh_metadata:
        if section_range != (MINIMUM_SLOT, MAXIMUM_SLOT) or args.series != "all":
            raise ValueError(
                "--refresh-metadata requires the full --series all dataset"
            )
        dataset = refresh_metadata(data_dir=args.data_dir, output_dir=output_dir)
    else:
        dataset = prepare_dataset(
            data_dir=args.data_dir,
            output_dir=output_dir,
            series=args.series,
            target_pixel_size_um=args.target_pixel_size_um,
            section_range=section_range,
            overwrite=args.overwrite,
        )
    summary = dataset.get("summary", dataset)
    LOG.info(
        "Validated %s: %d rows, %d present",
        output_dir,
        summary.get("row_count", dataset.get("physical_position_count")),
        summary.get("present_count", dataset.get("present_image_count")),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
