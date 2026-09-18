#!/usr/bin/env python3
"""Dense registered-histology Allen annotations on the canonical section lattice.

The implemented baseline is two-sided Nissl-driven diffeomorphic interpolation
of categorical Allen annotations using two endpoint-conditioned WSI trajectories.
Only already section-aligned symmetric inputs and the accepted final A2d placement
are consumed; this module does not estimate section placement or transfer data to
MRI space.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import resource
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import tifffile
import zarr

from preprocess import run_allen_emlddmm_full_coarse_nissl as coarse
from preprocess.run_allen_emlddmm_full_resolution_nissl import (
    warp_categorical_section,
)
from preprocess.visualize_allen_annotations import _combined_display_map

PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = PROJECT / (
    "data/derivatives/allen/specimen_708424/"
    "histology_symmetric_nissl_native_200um_section_aligned"
)
DEFAULT_REGISTRATION = PROJECT / (
    "results/allen/specimen_708424/emlddmm/native-200um-clean/"
    "HIST_NISSL_SYMMETRIC_SECTION_ALIGNED_to_MRI_7T_WHOLE"
)
DEFAULT_OUTPUT = PROJECT / (
    "data/derivatives/allen/specimen_708424/"
    "annotations_dense_registered_histology_200um"
)
DEFAULT_PAIR_CONFIG = PROJECT / "configs/allen_dense_pairwise.json"
DEFAULT_WSI_REPOSITORY = PROJECT.parent / "wsi-tissue-pipeline"
DEFAULT_EMLDDMM_REPOSITORY = PROJECT.parent / "emlddmm"
WSI_PIN = "d4d118a47d08700c8c30cf852b855e14e411bbdf"

SCHEMA = "allen-dense-registered-histology-v3"
ANCHOR_SCHEMA = "allen-final-registered-annotation-anchors-v1"
PAIR_SCHEMA = "allen-dense-pair-v3"
TIFF_STORE_SCHEMA = "allen-dense-tiff-store-v1"

UNSUPPORTED = 0
OBSERVED_LABELS = 1
OBSERVED_VALID_EMPTY = 2
INFERRED = 3
STATE_NAME = {
    UNSUPPORTED: "UNSUPPORTED",
    OBSERVED_LABELS: "OBSERVED_LABELS",
    OBSERVED_VALID_EMPTY: "OBSERVED_VALID_EMPTY",
    INFERRED: "INFERRED",
}
SEMANTIC_LABELED = "LABELED"
SEMANTIC_VALID_EMPTY = "VALID_EMPTY"
SEMANTIC_UNAVAILABLE = "UNAVAILABLE"


@dataclass(frozen=True)
class SourceContext:
    dataset: Path
    annotations: Path
    registration: Path
    numerical: Path
    rows: tuple[dict[str, str], ...]
    annotation_sections: tuple[int, ...]
    annotation_inventory: Mapping[tuple[int, int], dict[str, str]]
    physical_by_section: Mapping[int, int]
    observed_nissl: np.ndarray
    axes: tuple[np.ndarray, np.ndarray, np.ndarray]
    final_a2d: np.ndarray
    registered_axes: tuple[np.ndarray, np.ndarray]
    source_hashes: Mapping[str, str]
    registration_identifier: str
    graphic_groups: tuple[int, ...]
    shape: tuple[int, int]
    canonical_count: int
    serial_spacing_um: float
    pixel_size_um: float


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def _write_tsv(
    path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=fields, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _copy_file_transactionally(source: Path, destination: Path) -> None:
    """Copy immutable metadata without ever exposing a partial destination."""
    payload = source.read_bytes()
    if destination.is_file():
        if destination.read_bytes() != payload:
            raise RuntimeError(
                f"Existing metadata conflicts with {source}: {destination}"
            )
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, destination)


def _recorded_path(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _validate_uniform_axis(axis: np.ndarray, expected_step: float, name: str) -> None:
    if axis.ndim != 1 or len(axis) < 2 or not np.all(np.isfinite(axis)):
        raise RuntimeError(f"{name} is not a finite one-dimensional coordinate axis")
    if not np.allclose(np.diff(axis), expected_step, atol=1e-6, rtol=0.0):
        raise RuntimeError(
            f"{name} does not preserve the authoritative {expected_step:g}-um step"
        )


def _positive_metadata_number(
    metadata: Mapping[str, Any], key: str, *, integer: bool = False
) -> int | float:
    value = metadata.get(key)
    try:
        number = int(value) if integer else float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"dataset.json has invalid {key}") from exc
    if number <= 0 or (integer and number != value):
        raise RuntimeError(f"dataset.json has invalid {key}")
    return number


def _metadata_shape(metadata: Mapping[str, Any], key: str) -> tuple[int, int]:
    value = metadata.get(key)
    if not isinstance(value, list) or len(value) != 2:
        raise RuntimeError(f"dataset.json has invalid {key}")
    try:
        shape = tuple(int(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"dataset.json has invalid {key}") from exc
    if any(item <= 0 for item in shape) or list(shape) != value:
        raise RuntimeError(f"dataset.json has invalid {key}")
    return shape


def _annotation_graphic_groups(
    annotations: Path, annotation_meta: Mapping[str, Any]
) -> tuple[int, ...]:
    source_value = annotation_meta.get("annotations_ome_zarr_source")
    if not isinstance(source_value, str):
        raise RuntimeError(
            "Annotation derivative does not identify its source graphic-group catalog"
        )
    source_root = _recorded_path(annotations, source_value)
    catalog_path = source_root / "dataset.json"
    if not catalog_path.is_file() or catalog_path.is_symlink():
        raise RuntimeError(f"Missing source graphic-group catalog: {catalog_path}")
    catalog = json.loads(catalog_path.read_text()).get("graphic_groups")
    if not isinstance(catalog, list) or not catalog:
        raise RuntimeError("Source annotation dataset has no graphic-group catalog")
    try:
        groups = tuple(int(item["id"]) for item in catalog)
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "Source annotation graphic-group catalog is invalid"
        ) from exc
    if len(groups) != len(set(groups)):
        raise RuntimeError("Source annotation graphic-group catalog has duplicate IDs")
    return groups


def _discover_annotation_derivative(dataset: Path) -> Path:
    """Find the unique sibling annotation derivative linked to ``dataset``."""
    candidates: list[Path] = []
    for metadata_path in sorted(dataset.parent.glob("*/dataset.json")):
        if metadata_path.is_symlink():
            continue
        try:
            metadata = json.loads(metadata_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        parent = metadata.get("parent_nissl_derivative")
        if not isinstance(parent, str):
            continue
        candidate = metadata_path.parent.resolve()
        if _recorded_path(candidate, parent) != dataset:
            continue
        required = (
            candidate / "metadata/annotations.tsv",
            candidate / "metadata/source_annotation_manifest.tsv",
        )
        if all(path.is_file() and not path.is_symlink() for path in required):
            candidates.append(candidate)
    if len(candidates) != 1:
        formatted = ", ".join(str(path) for path in candidates) or "none"
        raise RuntimeError(
            "Could not identify one annotation derivative for the selected Nissl "
            f"dataset (found {formatted}); pass --annotations explicitly"
        )
    return candidates[0]


def discover_inputs(
    dataset: Path, registration: Path, annotations: Path | None = None
) -> SourceContext:
    """Validate and freeze authoritative geometry, derivatives, and final A2d."""
    dataset = dataset.expanduser().resolve()
    registration = registration.expanduser().resolve()
    annotations = (
        annotations.expanduser().resolve()
        if annotations is not None
        else _discover_annotation_derivative(dataset)
    )
    for required in (dataset / "dataset.json", annotations / "dataset.json"):
        if not required.is_file() or required.is_symlink():
            raise RuntimeError(f"Missing authoritative derivative metadata: {required}")

    nissl_meta = json.loads((dataset / "dataset.json").read_text())
    annotation_meta = json.loads((annotations / "dataset.json").read_text())
    canonical_count = int(
        _positive_metadata_number(nissl_meta, "physical_serial_positions", integer=True)
    )
    shape = _metadata_shape(nissl_meta, "prepared_canvas_shape_yx")
    serial_spacing_um = float(
        _positive_metadata_number(nissl_meta, "serial_spacing_um")
    )
    pixel_size_um = float(_positive_metadata_number(nissl_meta, "pixel_size_um"))
    if nissl_meta.get("preparation_mode") != "preserve_source_grid":
        raise RuntimeError(
            "Nissl derivative did not preserve its authoritative source grid"
        )
    recorded_parent = annotation_meta.get("parent_nissl_derivative")
    if (
        not isinstance(recorded_parent, str)
        or _recorded_path(annotations, recorded_parent) != dataset
    ):
        raise RuntimeError(
            "Annotation derivative does not name the selected Nissl parent"
        )
    if annotation_meta.get("parent_nissl_dataset_json_sha256") != _sha256(
        dataset / "dataset.json"
    ):
        raise RuntimeError("Annotation derivative parent checksum differs")
    if _metadata_shape(annotation_meta, "prepared_canvas_shape_yx") != shape:
        raise RuntimeError("Annotation and Nissl grids differ")
    annotation_pixel_size = float(
        _positive_metadata_number(annotation_meta, "pixel_size_um")
    )
    if not np.isclose(annotation_pixel_size, pixel_size_um, atol=1e-9, rtol=0.0):
        raise RuntimeError("Annotation and Nissl pixel sizes differ")
    graphic_groups = _annotation_graphic_groups(annotations, annotation_meta)

    physical_path = dataset / "metadata/physical_sections.tsv"
    rows = _read_tsv(physical_path)
    if len(rows) != canonical_count:
        raise RuntimeError(
            "Canonical physical-section table length differs from dataset metadata"
        )
    physical = np.asarray([int(row["physical_index"]) for row in rows], dtype=np.int64)
    if not np.array_equal(physical, np.arange(canonical_count)):
        raise RuntimeError("physical_index is not a consecutive zero-based lattice")
    z_from_table = np.asarray(
        [float(row["serial_z_center_mm"]) * 1000.0 for row in rows], dtype=np.float64
    )
    _validate_uniform_axis(z_from_table, serial_spacing_um, "physical_z_um")
    section_numbers = [int(row["allen_section_number"]) for row in rows]
    if len(section_numbers) != len(set(section_numbers)):
        raise RuntimeError(
            "Canonical physical-section table has duplicate section numbers"
        )
    physical_by_section = {
        section_number: physical_index
        for physical_index, section_number in enumerate(section_numbers)
    }

    source_manifest_path = annotations / "metadata/source_annotation_manifest.tsv"
    source_manifest = _read_tsv(source_manifest_path)
    annotation_sections = tuple(int(row["section_number"]) for row in source_manifest)
    recorded_annotation_count = int(
        _positive_metadata_number(
            annotation_meta, "annotation_section_count", integer=True
        )
    )
    if len(annotation_sections) != recorded_annotation_count or len(
        set(annotation_sections)
    ) != len(annotation_sections):
        raise RuntimeError("Annotation-level identity/count differs from metadata")
    unknown_sections = set(annotation_sections) - set(physical_by_section)
    if unknown_sections:
        raise RuntimeError(
            f"Annotation sections are absent from the physical lattice: {sorted(unknown_sections)}"
        )

    inventory_path = annotations / "metadata/annotations.tsv"
    inventory_rows = _read_tsv(inventory_path)
    inventory: dict[tuple[int, int], dict[str, str]] = {}
    for row in inventory_rows:
        key = (int(row["section_number"]), int(row["graphic_group_id"]))
        if key in inventory or key[1] not in graphic_groups:
            raise RuntimeError(
                f"Invalid duplicate/unknown annotation inventory row: {key}"
            )
        if key[0] not in set(annotation_sections):
            raise RuntimeError(f"Annotation inventory has an unknown section: {key[0]}")
        if row.get("sampling") != "categorical_nearest_neighbor":
            raise RuntimeError(f"Anchor {key} lacks categorical placement provenance")
        inventory[key] = row
    recorded_image_count = int(
        _positive_metadata_number(
            annotation_meta, "annotation_image_count", integer=True
        )
    )
    if len(inventory) != recorded_image_count:
        raise RuntimeError("Annotation inventory length differs from dataset metadata")
    actual_group_counts = {
        str(group): sum(key[1] == group for key in inventory)
        for group in graphic_groups
    }
    try:
        recorded_group_counts = {
            str(int(group)): int(count)
            for group, count in annotation_meta["graphic_group_counts"].items()
        }
    except (KeyError, AttributeError, TypeError, ValueError) as exc:
        raise RuntimeError("Annotation graphic-group counts are invalid") from exc
    if recorded_group_counts != actual_group_counts:
        raise RuntimeError("Annotation graphic-group counts differ from the inventory")
    for row in source_manifest:
        section = int(row["section_number"])
        try:
            declared = tuple(
                int(group) for group in json.loads(row["graphic_groups_present"])
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Invalid graphic_groups_present for section {section}"
            ) from exc
        if len(declared) != len(set(declared)) or not set(declared).issubset(
            graphic_groups
        ):
            raise RuntimeError(
                f"Invalid graphic-group availability for section {section}"
            )
        inventoried = {group for candidate, group in inventory if candidate == section}
        if set(declared) != inventoried:
            raise RuntimeError(
                f"Graphic-group availability differs for section {section}"
            )

    checkpoint_path = registration / "checkpoints/registration.json"
    checkpoint = json.loads(checkpoint_path.read_text())
    if checkpoint.get("status") not in {"complete", "complete_through_scale"}:
        raise RuntimeError("Selected registration checkpoint is not accepted/completed")
    numerical_value = checkpoint.get("numerical")
    if not isinstance(numerical_value, str):
        raise RuntimeError("Registration checkpoint does not record numerical outputs")
    numerical = _recorded_path(registration, numerical_value)
    if not numerical.is_file() or numerical.is_symlink():
        raise RuntimeError(f"Missing accepted numerical package: {numerical}")
    provenance_value = checkpoint.get("provenance")
    if not isinstance(provenance_value, str):
        raise RuntimeError("Registration checkpoint does not record provenance")
    provenance_path = _recorded_path(registration, provenance_value)
    provenance = json.loads(provenance_path.read_text())
    expected = provenance.get("checksums", {}).get(str(numerical))
    if expected is not None and expected != _sha256(numerical):
        raise RuntimeError(
            "Accepted numerical package checksum differs from provenance"
        )
    source_dataset = provenance.get("source_dataset")
    if (
        not isinstance(source_dataset, str)
        or _recorded_path(registration, source_dataset) != dataset
    ):
        raise RuntimeError(
            "Accepted registration did not act on the selected section-aligned stack"
        )
    section_initializer = provenance.get("section_initializer")
    if (
        not isinstance(section_initializer, dict)
        or section_initializer.get("purpose") != "already_materialized_section_stack"
        or section_initializer.get("left_atlas_free_A2d_reapplied") is not False
    ):
        raise RuntimeError(
            "Accepted registration does not freeze the no-reapplied-atlas-free-A2d convention"
        )

    with np.load(numerical) as saved:
        required = {"A2d", "xJ0", "xJ1", "xJ2", "observed"}
        if not required.issubset(saved.files):
            raise RuntimeError(
                "Accepted numerical package lacks final placement arrays"
            )
        final_a2d = np.asarray(saved["A2d"], dtype=np.float64)
        axes = tuple(np.asarray(saved[f"xJ{i}"], dtype=np.float64) for i in range(3))
        observed = np.asarray(saved["observed"], dtype=np.int64)
    if final_a2d.shape != (canonical_count, 3, 3):
        raise RuntimeError("Accepted final A2d does not span the physical lattice")
    if tuple(map(len, axes)) != (canonical_count, *shape):
        raise RuntimeError("Accepted registration histology axes differ from metadata")
    if not np.allclose(axes[0], z_from_table, atol=1e-9, rtol=0.0):
        raise RuntimeError("Accepted xJ[0] differs from canonical physical coordinates")
    _validate_uniform_axis(axes[0], serial_spacing_um, "accepted xJ[0]")
    _validate_uniform_axis(axes[1], pixel_size_um, "accepted xJ[1]")
    _validate_uniform_axis(axes[2], pixel_size_um, "accepted xJ[2]")
    table_observed = np.flatnonzero(
        [row["image_present"] == "true" and row["stain"] == "nissl" for row in rows]
    )
    if not np.array_equal(observed, table_observed):
        raise RuntimeError(
            "Accepted observed Nissl indices differ from the canonical table"
        )
    unsupported = np.ones(canonical_count, dtype=bool)
    unsupported[observed] = False
    if not np.any(unsupported):
        raise RuntimeError(
            "No unsupported A2d row is available to define the accepted output frame"
        )
    baseline = final_a2d[unsupported][0]
    if not np.array_equal(
        final_a2d[unsupported], np.broadcast_to(baseline, final_a2d[unsupported].shape)
    ):
        raise RuntimeError(
            "Final unsupported A2d rows do not define one accepted frame"
        )
    registered_axes = tuple(
        np.asarray(a) for a in coarse._registered_frame_axes(list(axes), baseline)
    )
    if tuple(map(len, registered_axes)) != shape:
        raise RuntimeError("Registered-histology grid shape differs from metadata")

    source_paths = (
        physical_path,
        dataset / "dataset.json",
        annotations / "dataset.json",
        inventory_path,
        source_manifest_path,
        checkpoint_path,
        provenance_path,
        numerical,
    )
    hashes = {str(path): _sha256(path) for path in source_paths}
    identifier = (
        f"{registration.name}:{checkpoint.get('status')}:{hashes[str(numerical)][:12]}"
    )
    return SourceContext(
        dataset=dataset,
        annotations=annotations,
        registration=registration,
        numerical=numerical,
        rows=tuple(rows),
        annotation_sections=annotation_sections,
        annotation_inventory=inventory,
        physical_by_section=physical_by_section,
        observed_nissl=observed,
        axes=axes,
        final_a2d=final_a2d,
        registered_axes=registered_axes,
        source_hashes=hashes,
        registration_identifier=identifier,
        graphic_groups=graphic_groups,
        shape=shape,
        canonical_count=canonical_count,
        serial_spacing_um=serial_spacing_um,
        pixel_size_um=pixel_size_um,
    )


def semantic_state(context: SourceContext, section: int, group: int) -> str:
    """Return metadata-derived availability without inspecting label pixels."""
    return (
        SEMANTIC_LABELED
        if (section, group) in context.annotation_inventory
        else SEMANTIC_UNAVAILABLE
    )


def _zarr_array(
    group: Any,
    name: str,
    *,
    shape: tuple[int, ...],
    chunks: tuple[int, ...],
    dtype: str,
    fill: int | float = 0,
) -> Any:
    if name in group:
        array = group[name]
        if tuple(array.shape) != shape or np.dtype(array.dtype) != np.dtype(dtype):
            raise RuntimeError(
                f"Existing Zarr array {array.path} has incompatible shape/dtype"
            )
        return array
    return group.create_array(
        name, shape=shape, chunks=chunks, dtype=dtype, fill_value=fill
    )


def _anchor_paths(output: Path) -> tuple[Path, Path]:
    return output / "anchors.zarr", output / "metadata/anchors.tsv"


def materialize_anchors(context: SourceContext, output: Path) -> dict[str, Any]:
    """Place every real annotation level once into final registered histology."""
    output = output.resolve()
    anchor_path, manifest_path = _anchor_paths(output)
    complete = output / "metadata/anchors.json"
    if complete.is_file():
        report = json.loads(complete.read_text())
        if (
            report.get("source_hashes") != context.source_hashes
            or tuple(report.get("graphic_groups", ())) != context.graphic_groups
        ):
            raise RuntimeError(
                "Existing anchors were made from different authoritative inputs"
            )
        if not anchor_path.is_dir() or not manifest_path.is_file():
            raise RuntimeError("Anchor completion metadata exists without its products")
        return report
    if anchor_path.exists() or manifest_path.exists():
        raise RuntimeError(
            "Incomplete anchor product exists; move it aside before retrying"
        )

    output.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".anchors.", dir=output))
    staged_zarr = stage / "anchors.zarr"
    root = zarr.open_group(str(staged_zarr), mode="w")
    count = len(context.annotation_sections)
    nissl = _zarr_array(
        root,
        "nissl",
        shape=(count, 3, *context.shape),
        chunks=(1, 3, *context.shape),
        dtype="float32",
    )
    weight = _zarr_array(
        root,
        "nissl_weight",
        shape=(count, *context.shape),
        chunks=(1, *context.shape),
        dtype="float32",
    )
    groups_root = root.require_group("groups")
    group_arrays = {
        group: _zarr_array(
            groups_root,
            str(group),
            shape=(count, *context.shape),
            chunks=(1, *context.shape),
            dtype="uint32",
        )
        for group in context.graphic_groups
    }
    rows: list[dict[str, Any]] = []
    vocabularies: dict[int, set[int]] = {group: {0} for group in context.graphic_groups}
    y, x = context.registered_axes
    source_y, source_x = context.axes[1:]
    try:
        for ordinal, section in enumerate(context.annotation_sections):
            physical_index = context.physical_by_section[section]
            if physical_index not in set(map(int, context.observed_nissl)):
                raise RuntimeError(
                    f"Annotated section {section} lacks accepted Nissl placement"
                )
            prepared_relative = context.rows[physical_index].get(
                "prepared_relative_path", ""
            )
            if not prepared_relative:
                raise RuntimeError(
                    f"Annotated section {section} has no prepared Nissl path"
                )
            image_path = (context.dataset / prepared_relative).resolve()
            image_path.relative_to(context.dataset)
            stain = context.rows[physical_index].get("stain", "")
            if not stain:
                raise RuntimeError(f"Annotated section {section} has no declared stain")
            weight_path = context.dataset / "support" / stain / image_path.name
            image = tifffile.imread(image_path)
            raw_weight = tifffile.imread(weight_path).astype(np.float32)
            if image.shape != (*context.shape, 3) or raw_weight.shape != context.shape:
                raise RuntimeError(
                    f"Nissl/weight geometry differs for section {section}"
                )
            placed_image, placed_weight = coarse._warp_saved_section(
                image.transpose(2, 0, 1).astype(np.float32) / 255.0,
                raw_weight,
                context.final_a2d[physical_index],
                y,
                x,
                source_row_um=source_y,
                source_column_um=source_x,
            )
            nissl[ordinal] = placed_image
            weight[ordinal] = placed_weight
            for group in context.graphic_groups:
                state = semantic_state(context, section, group)
                record = context.annotation_inventory.get((section, group))
                source_path = ""
                source_sha = ""
                if state == SEMANTIC_LABELED:
                    assert record is not None
                    label_path = (context.annotations / record["path"]).resolve()
                    label_path.relative_to(context.annotations)
                    if _sha256(label_path) != record["sha256"]:
                        raise RuntimeError(
                            f"Authoritative annotation checksum differs: {label_path}"
                        )
                    labels = tifffile.imread(label_path)
                    if labels.shape != context.shape or not np.issubdtype(
                        labels.dtype, np.integer
                    ):
                        raise RuntimeError(
                            f"Invalid authoritative label raster: {label_path}"
                        )
                    vocabularies[group].update(map(int, np.unique(labels)))
                    placed = warp_categorical_section(
                        labels,
                        context.final_a2d[physical_index],
                        y,
                        x,
                        source_row_um=source_y,
                        source_column_um=source_x,
                    )
                    group_arrays[group][ordinal] = placed
                    source_path = str(label_path)
                    source_sha = record["sha256"]
                rows.append(
                    {
                        "physical_index": physical_index,
                        "physical_z_um": f"{context.axes[0][physical_index]:.9g}",
                        "allen_section_number": section,
                        "source_annotation_identifier": f"section-{section:04d}/group-{group}",
                        "source_annotation_path": source_path,
                        "source_annotation_sha256": source_sha,
                        "graphic_group": group,
                        "semantic_state": state,
                        "final_rows": context.shape[0],
                        "final_columns": context.shape[1],
                        "registration_identifier": context.registration_identifier,
                        "placement": "accepted final A2d physical pullback exactly once",
                    }
                )
        staged_manifest = stage / "anchors.tsv"
        fields = list(rows[0])
        _write_tsv(staged_manifest, rows, fields)
        os.replace(staged_zarr, anchor_path)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staged_manifest, manifest_path)
    finally:
        try:
            stage.rmdir()
        except OSError:
            pass

    report = {
        "schema": ANCHOR_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "accepted final per-section A2d physical pullback; no interpolation through z",
        "source_hashes": dict(context.source_hashes),
        "registration_identifier": context.registration_identifier,
        "annotation_levels": count,
        "graphic_groups": list(context.graphic_groups),
        "semantic_counts": {
            state: sum(row["semantic_state"] == state for row in rows)
            for state in (SEMANTIC_LABELED, SEMANTIC_VALID_EMPTY, SEMANTIC_UNAVAILABLE)
        },
        "unavailable": [
            {
                "allen_section_number": row["allen_section_number"],
                "graphic_group": row["graphic_group"],
            }
            for row in rows
            if row["semantic_state"] == SEMANTIC_UNAVAILABLE
        ],
        "group_vocabularies": {
            str(group): sorted(values) for group, values in vocabularies.items()
        },
        "zarr": str(anchor_path),
        "manifest": str(manifest_path),
    }
    _json(complete, report)
    return report


def build_endpoint_sequences(
    anchor_rows: Sequence[Mapping[str, str]],
    graphic_groups: Sequence[int] | None = None,
) -> tuple[dict[int, list[int]], dict[tuple[int, int], tuple[int, ...]]]:
    """Build per-group semantic anchors and the compact unique-pair table."""
    groups = (
        tuple(graphic_groups)
        if graphic_groups is not None
        else tuple(dict.fromkeys(int(row["graphic_group"]) for row in anchor_rows))
    )
    by_group: dict[int, list[int]] = {group: [] for group in groups}
    for row in anchor_rows:
        group = int(row["graphic_group"])
        if row["semantic_state"] in {SEMANTIC_LABELED, SEMANTIC_VALID_EMPTY}:
            by_group[group].append(int(row["physical_index"]))
    pair_groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for group in groups:
        sequence = sorted(set(by_group[group]))
        by_group[group] = sequence
        for left, right in zip(sequence[:-1], sequence[1:], strict=False):
            pair_groups[(left, right)].append(group)
    return by_group, {
        pair: tuple(groups) for pair, groups in sorted(pair_groups.items())
    }


def _load_pair_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text())
    required = {
        "slice_matching": [False],
        "eA": [0.0],
        "eA2d": [0.0],
        "Amode": 0,
        "downI": [[1, 1]],
        "downJ": [[1, 1]],
    }
    for key, value in required.items():
        if config.get(key) != value:
            raise RuntimeError(f"Pair configuration must keep {key}={value!r}")
    nt = int(config.get("nt", 0))
    if nt < 1:
        raise RuntimeError("Pair configuration nt must be positive")
    return config


def pair_solver_config(
    config: Mapping[str, Any], nt_override: int | None = None
) -> dict[str, Any]:
    """Return stable WSI arguments with an optional expert integration nt."""
    solver = dict(config)
    if nt_override is not None:
        if nt_override < 1:
            raise ValueError("An nt override must be positive")
        solver["nt"] = int(nt_override)
    return solver


def _verify_wsi_repository(repository: Path) -> str:
    repository = repository.expanduser().resolve()
    actual = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if actual != WSI_PIN:
        raise RuntimeError(
            f"WSI checkout is {actual}; required pinned commit is {WSI_PIN}"
        )
    src = repository / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    return actual


def _load_wsi(
    repository: Path,
) -> tuple[Callable[..., Mapping[str, Any]], Any, Any, Any, Any, str]:
    commit = _verify_wsi_repository(repository)
    emlddmm_repository = (
        Path(os.environ.get("EMLDDMM_REPO", str(DEFAULT_EMLDDMM_REPOSITORY)))
        .expanduser()
        .resolve()
    )
    expected_emlddmm = (
        (PROJECT / "configs/emlddmm-upstream-commit.txt").read_text().strip()
    )
    actual_emlddmm = subprocess.run(
        ["git", "-C", str(emlddmm_repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if actual_emlddmm != expected_emlddmm:
        raise RuntimeError(
            f"EM-LDDMM checkout is {actual_emlddmm}; required pin is {expected_emlddmm}"
        )
    if str(emlddmm_repository) not in sys.path:
        sys.path.insert(0, str(emlddmm_repository))
    try:
        import torch
        from wsi_pipeline.registration.symmetric import (
            _integrate_inverse_flow,
            _resample_transform_to_domain,
            _resolve_emlddmm_module,
            emlddmm_multiscale_symmetric_N,
        )
    except ImportError as exc:
        raise RuntimeError(
            "The selected Python environment cannot import the pinned WSI registration modules"
        ) from exc
    return (
        emlddmm_multiscale_symmetric_N,
        _resolve_emlddmm_module(),
        torch,
        _integrate_inverse_flow,
        _resample_transform_to_domain,
        commit,
    )


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value)


@dataclass
class SourceFlowEvaluator:
    """Evaluate a fitted WSI source-side inverse flow at arbitrary pseudotime."""

    velocity_axes: tuple[Any, ...]
    image_axes: tuple[Any, ...]
    velocity: Any
    boundary_maps: Any
    em: Any
    torch: Any
    resample_transform: Any

    @property
    def nt(self) -> int:
        return int(self.velocity.shape[0])

    def evaluate(self, t: float) -> np.ndarray:
        if not np.isfinite(t) or not 0.0 <= t <= 1.0:
            raise ValueError("Pseudotime must be finite and in [0,1]")
        scaled = float(t) * self.nt
        nearest = int(round(scaled))
        if abs(scaled - nearest) <= 1e-10:
            phi_velocity = self.boundary_maps[nearest]
        else:
            interval = int(np.floor(scaled))
            partial_dt = float(t) - interval / self.nt
            mesh = self.torch.stack(
                self.torch.meshgrid(self.velocity_axes, indexing="ij")
            )
            sample_points = mesh - self.velocity[interval] * partial_dt
            phi_velocity = (
                self.em.interp(
                    self.velocity_axes,
                    self.boundary_maps[interval] - mesh,
                    sample_points,
                    interp2d=True,
                )
                + sample_points
            )
        phi_image = self.resample_transform(
            self.velocity_axes,
            phi_velocity[None],
            self.image_axes,
            emlddmm_module=self.em,
            interp2d=True,
        )[0]
        return _as_numpy(phi_image).astype(np.float32, copy=False)


def build_source_flow_evaluator(
    output: Mapping[str, Any],
    image_axes: Sequence[np.ndarray],
    em: Any,
    torch: Any,
    integrate_inverse_flow: Any,
    resample_transform: Any,
) -> SourceFlowEvaluator:
    """Reintegrate v_symmetric using WSI's exact identity-inclusive convention."""
    velocity = torch.as_tensor(output["v_symmetric"], dtype=torch.float32)
    forward = output["forward"]
    forward_last = forward[-1] if isinstance(forward, list) else forward
    velocity_axes = tuple(
        torch.as_tensor(axis, device=velocity.device, dtype=velocity.dtype)
        for axis in forward_last["xv"][-2:]
    )
    image_axes_t = tuple(
        torch.as_tensor(axis, device=velocity.device, dtype=velocity.dtype)
        for axis in image_axes
    )
    boundary_maps = integrate_inverse_flow(
        velocity_axes,
        velocity,
        emlddmm_module=em,
        interp2d=True,
    )
    return SourceFlowEvaluator(
        velocity_axes=velocity_axes,
        image_axes=image_axes_t,
        velocity=velocity,
        boundary_maps=boundary_maps,
        em=em,
        torch=torch,
        resample_transform=resample_transform,
    )


def validate_arbitrary_time_evaluator(
    evaluator: SourceFlowEvaluator,
    output: Mapping[str, Any],
    source: np.ndarray,
    axes: Sequence[np.ndarray],
) -> dict[str, float]:
    """At k/nt, reproduce WSI phi_I and its corresponding warped source state."""
    stored_phi = _as_numpy(output["phi_I"])
    if len(stored_phi) != evaluator.nt + 1:
        raise RuntimeError("Stored source trajectory length differs from evaluator nt")
    stored_images = _as_numpy(output["ItAll"])
    map_error = 0.0
    image_error = 0.0
    image_axes = tuple(
        evaluator.torch.as_tensor(axis, dtype=evaluator.torch.float32) for axis in axes
    )
    source_tensor = evaluator.torch.as_tensor(source, dtype=evaluator.torch.float32)
    for k in range(evaluator.nt + 1):
        evaluated = evaluator.evaluate(k / evaluator.nt)
        map_error = max(map_error, float(np.max(np.abs(evaluated - stored_phi[k]))))
        reproduced = evaluator.em.interp(
            image_axes,
            source_tensor,
            evaluator.torch.as_tensor(evaluated, dtype=evaluator.torch.float32),
            interp2d=True,
        )
        image_error = max(
            image_error,
            float(np.max(np.abs(_as_numpy(reproduced) - stored_images[k]))),
        )
    if map_error > 2e-5 or image_error > 2e-5:
        raise RuntimeError(
            "Arbitrary-time flow regression failed: "
            f"map={map_error:g}, image={image_error:g}"
        )
    return {
        "stored_state_map_max_abs_error": map_error,
        "stored_state_nissl_max_abs_error": image_error,
    }


def validate_source_map(
    output: Mapping[str, Any],
    source: np.ndarray,
    axes: Sequence[np.ndarray],
    emlddmm_module: Any,
    torch_module: Any,
    *,
    state: int | None = None,
) -> float:
    """Prove phi_I is the pullback that reproduces the stored source trajectory."""
    phi = _as_numpy(output["phi_I"])
    stored = _as_numpy(output["ItAll"])
    if phi.ndim != 4 or stored.ndim != 4 or len(phi) != len(stored):
        raise RuntimeError("WSI source trajectory shapes changed")
    index = len(phi) // 2 if state is None else int(state)
    reproduced = emlddmm_module.interp(
        tuple(
            torch_module.as_tensor(axis, dtype=torch_module.float32) for axis in axes
        ),
        torch_module.as_tensor(source, dtype=torch_module.float32),
        torch_module.as_tensor(phi[index], dtype=torch_module.float32),
        interp2d=True,
    )
    error = float(np.max(np.abs(_as_numpy(reproduced) - stored[index])))
    if error > 2e-5:
        raise RuntimeError(
            f"phi_I map-convention regression failed: max error {error:g}"
        )
    return error


def fit_pair_trajectories(
    left_image: np.ndarray,
    right_image: np.ndarray,
    left_weight: np.ndarray,
    right_weight: np.ndarray,
    axes: Sequence[np.ndarray],
    config: Mapping[str, Any],
    *,
    wsi_repository: Path,
    device: str = "cpu",
) -> tuple[SourceFlowEvaluator, SourceFlowEvaluator, dict[str, Any], Any, Any]:
    """Run WSI's two outer calls and retain source-flow evaluators."""
    solver, em, torch, integrate, resample, commit = _load_wsi(wsi_repository)
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {device}, but CUDA is unavailable")
    tensors = [
        torch.as_tensor(value, dtype=torch.float32, device=device)
        for value in (left_image, right_image, left_weight, right_weight)
    ]
    x = tuple(
        torch.as_tensor(axis, dtype=torch.float32, device=device) for axis in axes
    )
    left, right, w_left, w_right = tensors
    source_map_errors: list[float] = []
    evaluator_reports: list[dict[str, float]] = []
    evaluators: list[SourceFlowEvaluator] = []
    for source, target, weight in ((left, right, w_left), (right, left, w_right)):
        result = solver(xI=x, I=source, xJ=x, J=target, W0=weight, **dict(config))
        source_map_errors.append(
            validate_source_map(result, _as_numpy(source), axes, em, torch)
        )
        evaluator = build_source_flow_evaluator(
            result,
            axes,
            em,
            torch,
            integrate,
            resample,
        )
        evaluator_reports.append(
            validate_arbitrary_time_evaluator(
                evaluator, result, _as_numpy(source), axes
            )
        )
        evaluators.append(evaluator)
        del result
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    expected_nt = int(config["nt"])
    expected_map_shape = (2, *left_image.shape[-2:])
    if any(
        value.nt != expected_nt or value.evaluate(0.0).shape != expected_map_shape
        for value in evaluators
    ):
        raise RuntimeError(
            "WSI source-flow evaluator has an unexpected temporal or image-domain shape"
        )
    report = {
        "wsi_commit": commit,
        "outer_calls": 2,
        "inner_symmetric_optimizations_per_outer_call": 2,
        "source_flow": "v_symmetric integrated as WSI phi_I inverse pullback",
        "map_regression_max_abs_errors": source_map_errors,
        "arbitrary_time_evaluator_regression": evaluator_reports,
        "time_sampling": (
            "canonical z fraction evaluated by WSI Euler/composition convention; "
            "right source at complementary 1-t"
        ),
        "device": device,
    }
    return evaluators[0], evaluators[1], report, em, torch


def _warp_channels(
    channels: np.ndarray,
    phi: np.ndarray,
    axes: Sequence[np.ndarray],
    em: Any,
    torch: Any,
) -> np.ndarray:
    result = em.interp(
        tuple(torch.as_tensor(axis, dtype=torch.float32) for axis in axes),
        torch.as_tensor(channels, dtype=torch.float32),
        torch.as_tensor(phi, dtype=torch.float32),
        interp2d=True,
    )
    return _as_numpy(result).astype(np.float32, copy=False)


def categorical_pair_plane(
    left: np.ndarray,
    right: np.ndarray,
    vocabulary: Sequence[int],
    p: float,
    phi_left: np.ndarray,
    phi_right: np.ndarray,
    axes: Sequence[np.ndarray],
    em: Any,
    torch: Any,
    *,
    channel_batch: int = 16,
) -> np.ndarray:
    """Transport one-hot memberships, fuse, and harden with an explicit tie rule."""
    if not (
        np.issubdtype(left.dtype, np.integer) and np.issubdtype(right.dtype, np.integer)
    ):
        raise ValueError("Categorical endpoints must be integer rasters")
    ids = np.asarray(sorted(set(map(int, vocabulary))), dtype=np.uint32)
    if len(ids) == 0 or ids[0] != 0:
        raise RuntimeError("Pair vocabulary must explicitly contain background ID 0")
    source_ids = set(map(int, np.unique(left))) | set(map(int, np.unique(right)))
    if not source_ids.issubset(set(map(int, ids))):
        raise RuntimeError("Pair vocabulary omits an endpoint Allen ID")
    shape = left.shape
    best = np.full(shape, -np.inf, dtype=np.float32)
    best_left = np.full(shape, -np.inf, dtype=np.float32)
    best_right = np.full(shape, -np.inf, dtype=np.float32)
    winner = np.full(shape, np.iinfo(np.uint32).max, dtype=np.uint32)
    for start in range(0, len(ids), channel_batch):
        batch_ids = ids[start : start + channel_batch]
        left_one_hot = (left[None] == batch_ids[:, None, None]).astype(np.float32)
        right_one_hot = (right[None] == batch_ids[:, None, None]).astype(np.float32)
        transported_left = _warp_channels(left_one_hot, phi_left, axes, em, torch)
        transported_right = _warp_channels(right_one_hot, phi_right, axes, em, torch)
        scores = (1.0 - p) * transported_left + p * transported_right
        for offset, label_id in enumerate(batch_ids):
            score = scores[offset]
            left_score = transported_left[offset]
            right_score = transported_right[offset]
            better = score > best
            tied = score == best
            better |= tied & (left_score > best_left)
            tied &= left_score == best_left
            better |= tied & (right_score > best_right)
            tied &= right_score == best_right
            better |= tied & (label_id < winner)
            best[better] = score[better]
            best_left[better] = left_score[better]
            best_right[better] = right_score[better]
            winner[better] = label_id
    if not set(map(int, np.unique(winner))).issubset(set(map(int, ids))):
        raise RuntimeError("Categorical hardening emitted an unknown Allen ID")
    return winner


class DenseAnnotationStore:
    """Narrow storage boundary for already-hardened uint32 label planes."""

    output_format: str

    def __init__(
        self,
        output: Path,
        shape: tuple[int, int],
        canonical_z_um: np.ndarray,
        groups: Sequence[int],
    ) -> None:
        self.output = output
        self.shape = tuple(shape)
        self.canonical_z_um = np.asarray(canonical_z_um, dtype=np.float64)
        self.groups = tuple(groups)
        self.group_index = {group: index for index, group in enumerate(self.groups)}

    def write_group_plane(
        self, group: int, canonical_index: int, plane: np.ndarray, *, overwrite: bool
    ) -> bool:
        raise NotImplementedError

    def read_group_plane(self, group: int, canonical_index: int) -> np.ndarray:
        raise NotImplementedError

    def write_combined_plane(
        self, canonical_index: int, plane: np.ndarray, *, overwrite: bool
    ) -> bool:
        raise NotImplementedError

    def plane_exists(
        self, canonical_index: int, *, group: int | None = None, combined: bool = False
    ) -> bool:
        raise NotImplementedError

    def verify_plane(
        self,
        canonical_index: int,
        *,
        group: int | None = None,
        combined: bool = False,
        expected: np.ndarray | None = None,
    ) -> bool:
        raise NotImplementedError

    def state(self, group: int, canonical_index: int) -> int:
        return int(self.states[self.group_index[group], canonical_index])

    def set_state(self, group: int, canonical_index: int, value: int) -> None:
        self.states[self.group_index[group], canonical_index] = value

    def finalize(self, *, include_combined: bool) -> Path | None:
        return None

    def products(self) -> dict[str, str]:
        raise NotImplementedError


def _validated_uint32_plane(plane: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    value = np.asarray(plane)
    if value.dtype != np.dtype("uint32") or value.shape != shape:
        raise RuntimeError(
            f"Dense label plane must have dtype uint32 and shape {shape}; "
            f"got {value.dtype} {value.shape}"
        )
    return value


def _dense_products(
    output: Path,
    shape: tuple[int, int],
    canonical_z_um: np.ndarray,
    groups: Sequence[int],
) -> tuple[Any, dict[int, Any], Any]:
    """Open the validated legacy Zarr dense product."""
    canonical_count = len(canonical_z_um)
    root = zarr.open_group(str(output / "dense.zarr"), mode="a")
    coordinate_exists = "z_um" in root
    coordinate = _zarr_array(
        root,
        "z_um",
        shape=(canonical_count,),
        chunks=(canonical_count,),
        dtype="float64",
    )
    if coordinate_exists:
        if not np.array_equal(np.asarray(coordinate), canonical_z_um):
            raise RuntimeError(
                "Existing dense z-axis differs from the authoritative canonical lattice"
            )
    else:
        coordinate[:] = canonical_z_um
    root.attrs["axis_0"] = "physical_z_um"
    root.attrs["axis_0_sampling"] = "authoritative_canonical_physical_section_lattice"
    group_root = root.require_group("groups")
    arrays = {
        group: _zarr_array(
            group_root,
            str(group),
            shape=(canonical_count, *shape),
            chunks=(1, *shape),
            dtype="uint32",
        )
        for group in groups
    }
    states = _zarr_array(
        root,
        "semantic_state",
        shape=(len(groups), canonical_count),
        chunks=(1, canonical_count),
        dtype="uint8",
    )
    return root, arrays, states


class ZarrDenseAnnotationStore(DenseAnnotationStore):
    """Adapter around the previously validated dense Zarr representation."""

    output_format = "zarr"

    def __init__(
        self,
        output: Path,
        shape: tuple[int, int],
        canonical_z_um: np.ndarray,
        groups: Sequence[int],
    ) -> None:
        super().__init__(output, shape, canonical_z_um, groups)
        self.root, self.arrays, self.states = _dense_products(
            output, shape, canonical_z_um, groups
        )

    def write_group_plane(
        self, group: int, canonical_index: int, plane: np.ndarray, *, overwrite: bool
    ) -> bool:
        self.arrays[group][canonical_index] = _validated_uint32_plane(plane, self.shape)
        return True

    def read_group_plane(self, group: int, canonical_index: int) -> np.ndarray:
        return np.asarray(self.arrays[group][canonical_index], dtype=np.uint32)

    def write_combined_plane(
        self, canonical_index: int, plane: np.ndarray, *, overwrite: bool
    ) -> bool:
        combined = _zarr_array(
            self.root,
            "combined",
            shape=(len(self.canonical_z_um), *self.shape),
            chunks=(1, *self.shape),
            dtype="uint32",
        )
        combined[canonical_index] = _validated_uint32_plane(plane, self.shape)
        return True

    def plane_exists(
        self, canonical_index: int, *, group: int | None = None, combined: bool = False
    ) -> bool:
        if combined:
            return "combined" in self.root
        return group in self.arrays

    def verify_plane(
        self,
        canonical_index: int,
        *,
        group: int | None = None,
        combined: bool = False,
        expected: np.ndarray | None = None,
    ) -> bool:
        if combined:
            if "combined" not in self.root:
                return False
            value = np.asarray(self.root["combined"][canonical_index])
        else:
            if group is None or group not in self.arrays:
                return False
            value = self.read_group_plane(group, canonical_index)
        _validated_uint32_plane(value, self.shape)
        return expected is None or np.array_equal(value, expected)

    def products(self) -> dict[str, str]:
        return {
            "dense_z_coordinate": str(self.output / "dense.zarr/z_um"),
            "dense_per_group": str(self.output / "dense.zarr/groups"),
            "semantic_state": str(self.output / "dense.zarr/semantic_state"),
            "combined": str(self.output / "dense.zarr/combined"),
        }


class TiffDenseAnnotationStore(DenseAnnotationStore):
    """Transactional, one-file-per-plane uint32 TIFF dense representation."""

    output_format = "tiff"

    def __init__(
        self,
        output: Path,
        shape: tuple[int, int],
        canonical_z_um: np.ndarray,
        *,
        compression: str = "deflate",
        groups: Sequence[int],
    ) -> None:
        super().__init__(output, shape, canonical_z_um, groups)
        if compression not in {"deflate", "none"}:
            raise ValueError(f"Unsupported TIFF compression: {compression}")
        self.compression = compression
        self.root = output / "dense_tiff"
        self.states = np.zeros(
            (len(self.groups), len(self.canonical_z_um)), dtype=np.uint8
        )
        descriptor_path = output / "metadata/dense_tiff_store.json"
        coordinate_sha256 = hashlib.sha256(
            self.canonical_z_um.astype("<f8", copy=False).tobytes()
        ).hexdigest()
        descriptor = {
            "schema": TIFF_STORE_SCHEMA,
            "output_format": self.output_format,
            "compression": self.compression,
            "dtype": "uint32",
            "shape_yx": list(self.shape),
            "canonical_positions": len(self.canonical_z_um),
            "canonical_z_float64_sha256": coordinate_sha256,
            "graphic_groups": list(self.groups),
        }
        if descriptor_path.is_file():
            if json.loads(descriptor_path.read_text()) != descriptor:
                raise RuntimeError(
                    f"Existing TIFF product conflicts with this run: {descriptor_path}"
                )
        else:
            if self.root.exists() and any(self.root.rglob("*.tif")):
                raise RuntimeError(
                    "Existing TIFF planes lack compatible store metadata; move the "
                    f"incomplete product aside: {self.root}"
                )
            _json(descriptor_path, descriptor)
        for group in self.groups:
            (self.root / "groups" / str(group)).mkdir(parents=True, exist_ok=True)
        (self.root / "combined").mkdir(parents=True, exist_ok=True)

    def _path(self, canonical_index: int, *, group: int | None, combined: bool) -> Path:
        if not 0 <= canonical_index < len(self.canonical_z_um):
            raise IndexError(f"Canonical position is out of range: {canonical_index}")
        filename = f"{canonical_index:06d}.tif"
        if combined:
            if group is not None:
                raise ValueError("A combined plane cannot have a graphic group")
            return self.root / "combined" / filename
        if group not in self.group_index:
            raise ValueError(f"Unknown graphic group: {group}")
        return self.root / "groups" / str(group) / filename

    def _write(self, path: Path, plane: np.ndarray, *, overwrite: bool) -> bool:
        value = _validated_uint32_plane(plane, self.shape)
        if path.is_file():
            existing = tifffile.imread(path)
            _validated_uint32_plane(existing, self.shape)
            if np.array_equal(existing, value):
                return False
            if not overwrite:
                raise RuntimeError(
                    f"Existing TIFF conflicts with requested dense plane: {path}"
                )
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        os.close(file_descriptor)
        temporary = Path(temporary_name)
        try:
            tifffile.imwrite(
                temporary,
                value,
                compression=None if self.compression == "none" else "deflate",
                photometric="minisblack",
                metadata=None,
            )
            written = tifffile.imread(temporary)
            _validated_uint32_plane(written, self.shape)
            if not np.array_equal(written, value):
                raise RuntimeError(f"TIFF verification failed before commit: {path}")
            os.replace(temporary, path)
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
        return True

    def write_group_plane(
        self, group: int, canonical_index: int, plane: np.ndarray, *, overwrite: bool
    ) -> bool:
        return self._write(
            self._path(canonical_index, group=group, combined=False),
            plane,
            overwrite=overwrite,
        )

    def read_group_plane(self, group: int, canonical_index: int) -> np.ndarray:
        path = self._path(canonical_index, group=group, combined=False)
        return np.asarray(tifffile.imread(path))

    def write_combined_plane(
        self, canonical_index: int, plane: np.ndarray, *, overwrite: bool
    ) -> bool:
        return self._write(
            self._path(canonical_index, group=None, combined=True),
            plane,
            overwrite=overwrite,
        )

    def plane_exists(
        self, canonical_index: int, *, group: int | None = None, combined: bool = False
    ) -> bool:
        return self._path(canonical_index, group=group, combined=combined).is_file()

    def verify_plane(
        self,
        canonical_index: int,
        *,
        group: int | None = None,
        combined: bool = False,
        expected: np.ndarray | None = None,
    ) -> bool:
        path = self._path(canonical_index, group=group, combined=combined)
        if not path.is_file():
            return False
        value = np.asarray(tifffile.imread(path))
        _validated_uint32_plane(value, self.shape)
        return expected is None or np.array_equal(value, expected)

    @staticmethod
    def _manifest_state(value: int) -> tuple[str, str]:
        evidence = {
            UNSUPPORTED: "UNAVAILABLE",
            OBSERVED_LABELS: "OBSERVED",
            OBSERVED_VALID_EMPTY: "OBSERVED",
            INFERRED: "INFERRED",
        }
        return STATE_NAME[value], evidence[value]

    def finalize(self, *, include_combined: bool) -> Path:
        rows: list[dict[str, Any]] = []
        for group in self.groups:
            for canonical_index, z_um in enumerate(self.canonical_z_um):
                path = self._path(canonical_index, group=group, combined=False)
                if not self.verify_plane(canonical_index, group=group):
                    raise RuntimeError(f"Missing completed TIFF plane: {path}")
                semantic, evidence = self._manifest_state(
                    self.state(group, canonical_index)
                )
                rows.append(
                    {
                        "relative_path": str(path.relative_to(self.output)),
                        "canonical_index": canonical_index,
                        "z_um": f"{z_um:.17g}",
                        "graphic_group_id": group,
                        "product_type": "group",
                        "semantic_state": semantic,
                        "evidence_state": evidence,
                        "dtype": "uint32",
                        "height": self.shape[0],
                        "width": self.shape[1],
                        "sha256": _sha256(path),
                    }
                )
        if include_combined:
            for canonical_index, z_um in enumerate(self.canonical_z_um):
                path = self._path(canonical_index, group=None, combined=True)
                if not self.verify_plane(canonical_index, combined=True):
                    raise RuntimeError(f"Missing completed TIFF plane: {path}")
                state_values = [
                    self.state(group, canonical_index) for group in self.groups
                ]
                if all(value == UNSUPPORTED for value in state_values):
                    semantic, evidence = "UNSUPPORTED", "UNAVAILABLE"
                else:
                    semantic, evidence = "SEE_SECTION_GROUP_PROVENANCE", "COMBINED"
                rows.append(
                    {
                        "relative_path": str(path.relative_to(self.output)),
                        "canonical_index": canonical_index,
                        "z_um": f"{z_um:.17g}",
                        "graphic_group_id": "",
                        "product_type": "combined",
                        "semantic_state": semantic,
                        "evidence_state": evidence,
                        "dtype": "uint32",
                        "height": self.shape[0],
                        "width": self.shape[1],
                        "sha256": _sha256(path),
                    }
                )
        path = self.output / "metadata/dense_tiff_manifest.tsv"
        _write_tsv(path, rows, list(rows[0]))
        return path

    def products(self) -> dict[str, str]:
        return {
            "dense_per_group": str(self.root / "groups"),
            "semantic_state": str(
                self.output / "metadata/section_group_provenance.tsv"
            ),
            "combined": str(self.root / "combined"),
            "tiff_manifest": str(self.output / "metadata/dense_tiff_manifest.tsv"),
        }


def create_dense_store(
    output: Path,
    shape: tuple[int, int],
    canonical_z_um: np.ndarray,
    *,
    output_format: str,
    tiff_compression: str,
    groups: Sequence[int],
) -> DenseAnnotationStore:
    if output_format == "tiff":
        return TiffDenseAnnotationStore(
            output,
            shape,
            canonical_z_um,
            compression=tiff_compression,
            groups=groups,
        )
    if output_format == "zarr":
        return ZarrDenseAnnotationStore(output, shape, canonical_z_um, groups)
    raise ValueError(f"Unsupported dense output format: {output_format}")


def initialize_dense(
    context: SourceContext,
    output: Path,
    store: DenseAnnotationStore,
    recoverable_planes: set[tuple[int, int]],
) -> dict[int, int]:
    anchor_root = zarr.open_group(str(output / "anchors.zarr"), mode="r")
    anchor_rows = _read_tsv(output / "metadata/anchors.tsv")
    ordinal_by_physical: dict[int, int] = {}
    for ordinal, section in enumerate(context.annotation_sections):
        ordinal_by_physical[context.physical_by_section[section]] = ordinal
    anchor_groups = anchor_root["groups"]
    for row in anchor_rows:
        state = row["semantic_state"]
        if state not in {SEMANTIC_LABELED, SEMANTIC_VALID_EMPTY}:
            continue
        physical = int(row["physical_index"])
        group = int(row["graphic_group"])
        ordinal = ordinal_by_physical[physical]
        observed = np.asarray(anchor_groups[str(group)][ordinal], dtype=np.uint32)
        store.write_group_plane(group, physical, observed, overwrite=False)
        store.set_state(
            group,
            physical,
            OBSERVED_LABELS if state == SEMANTIC_LABELED else OBSERVED_VALID_EMPTY,
        )
    if isinstance(store, TiffDenseAnnotationStore):
        _restore_tiff_checkpoint_states(store, output)
        zero = np.zeros(store.shape, dtype=np.uint32)
        for group in store.groups:
            for physical in range(len(store.canonical_z_um)):
                if not store.plane_exists(physical, group=group):
                    store.write_group_plane(group, physical, zero, overwrite=False)
                elif (
                    store.state(group, physical) == UNSUPPORTED
                    and (group, physical) not in recoverable_planes
                    and np.any(store.read_group_plane(group, physical))
                ):
                    raise RuntimeError(
                        "Existing TIFF plane has no matching observed or pair "
                        f"provenance: group {group}, canonical index {physical}"
                    )
    return ordinal_by_physical


def _pair_id(pair: tuple[int, int]) -> str:
    return f"{pair[0]:04d}-{pair[1]:04d}"


def _pair_status_path(output: Path, pair: tuple[int, int], output_format: str) -> Path:
    return output / "metadata/pairs" / output_format / f"{_pair_id(pair)}.json"


def _existing_pair_status_path(
    output: Path, pair: tuple[int, int], output_format: str
) -> Path | None:
    path = _pair_status_path(output, pair, output_format)
    if path.is_file():
        return path
    legacy = output / "metadata/pairs" / f"{_pair_id(pair)}.json"
    if output_format == "zarr" and legacy.is_file():
        return legacy
    return None


def _restore_tiff_checkpoint_states(
    store: TiffDenseAnnotationStore, output: Path
) -> None:
    directory = output / "metadata/pairs/tiff"
    if not directory.is_dir():
        return
    for path in sorted(directory.glob("*.json")):
        status = json.loads(path.read_text())
        if status.get("status") != "complete" or status.get("output_format") != "tiff":
            continue
        left, right = map(int, status["pair"])
        for group in map(int, status["groups"]):
            for canonical_index in range(left + 1, right):
                if not store.verify_plane(canonical_index, group=group):
                    raise RuntimeError(
                        f"TIFF pair checkpoint exists without a valid plane: {path}"
                    )
                store.set_state(group, canonical_index, INFERRED)


def _parse_pair(value: str) -> tuple[int, int]:
    try:
        left, right = value.split("-", 1)
        pair = (int(left), int(right))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            "pair must be PHYSICAL_INDEX-PHYSICAL_INDEX"
        ) from exc
    if not 0 <= pair[0] < pair[1]:
        raise argparse.ArgumentTypeError(
            "pair indices must be nonnegative and strictly increasing"
        )
    return pair


def _peak_rss_kib() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def process_pair(
    context: SourceContext,
    output: Path,
    store: DenseAnnotationStore,
    pair: tuple[int, int],
    groups: Sequence[int],
    config: Mapping[str, Any],
    *,
    nt_source: str,
    wsi_repository: Path,
    device: str,
    overwrite: bool,
) -> dict[str, Any]:
    nt = int(config["nt"])
    configuration_sha256 = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    left_index, right_index = pair
    existing_status_path = _existing_pair_status_path(output, pair, store.output_format)
    status_path = _pair_status_path(output, pair, store.output_format)
    if existing_status_path is not None and not overwrite:
        status = json.loads(existing_status_path.read_text())
        if status.get("status") != "complete":
            raise RuntimeError(
                f"Existing pair status is incomplete: {existing_status_path}"
            )
        recorded_format = status.get("output_format")
        if recorded_format is None and existing_status_path == (
            output / "metadata/pairs" / f"{_pair_id(pair)}.json"
        ):
            recorded_format = "zarr"
        if recorded_format != store.output_format:
            raise RuntimeError(
                f"Existing pair status belongs to {recorded_format}, not "
                f"{store.output_format}: {existing_status_path}"
            )
        if (
            status.get("nt") != nt
            or status.get("configuration_sha256") != configuration_sha256
        ):
            raise RuntimeError(
                f"Existing pair status uses another nt/configuration: "
                f"{existing_status_path}"
            )
        for group in groups:
            for physical in range(left_index + 1, right_index):
                if store.state(group, physical) != INFERRED or not store.verify_plane(
                    physical, group=group
                ):
                    raise RuntimeError(
                        "Pair checkpoint exists without its completed "
                        f"{store.output_format} output: {existing_status_path}"
                    )
        if existing_status_path != status_path:
            status = {**status, "output_format": store.output_format}
            _json(status_path, status)
        return status
    anchor_root = zarr.open_group(str(output / "anchors.zarr"), mode="r")
    ordinal_by_physical = {
        context.physical_by_section[section]: ordinal
        for ordinal, section in enumerate(context.annotation_sections)
    }
    left_ordinal = ordinal_by_physical[left_index]
    right_ordinal = ordinal_by_physical[right_index]
    z0_um = float(context.axes[0][left_index])
    z1_um = float(context.axes[0][right_index])
    # rgb imaging 
    left_image = np.asarray(anchor_root["nissl"][left_ordinal], dtype=np.float32)
    right_image = np.asarray(anchor_root["nissl"][right_ordinal], dtype=np.float32)
    left_weight = np.asarray(
        anchor_root["nissl_weight"][left_ordinal], dtype=np.float32
    )
    right_weight = np.asarray(
        anchor_root["nissl_weight"][right_ordinal], dtype=np.float32
    )
    # # annotations
    # left_labels = np.asarray(anchor_root["groups"][str(group)][left_ordinal], dtype=np.uint32)
    # right_labels = np.asarray(anchor_root["groups"][str(group)][right_ordinal], dtype=np.uint32)

    # left_image = render_labels_to_rgb(left_labels).astype(np.float32)
    # right_image = render_labels_to_rgb(right_labels).astype(np.float32)

    # left_weight = (left_labels != 0).astype(np.float32)
    # right_weight = (right_labels != 0).astype(np.float32)
    start_rss = _peak_rss_kib()
    started = time.monotonic()
    left_flow, right_flow, map_report, em, torch = fit_pair_trajectories(
        left_image,
        right_image,
        left_weight,
        right_weight,
        context.registered_axes,
        config,
        wsi_repository=wsi_repository,
        device=device,
    )
    group_reports: dict[str, Any] = {}
    for group in groups:
        endpoint_left = np.asarray(
            anchor_root["groups"][str(group)][left_ordinal], dtype=np.uint32
        )
        endpoint_right = np.asarray(
            anchor_root["groups"][str(group)][right_ordinal], dtype=np.uint32
        )
        vocabulary = sorted(
            set(map(int, np.unique(endpoint_left)))
            | set(map(int, np.unique(endpoint_right)))
            | {0}
        )
        written = 0
        for physical in range(left_index + 1, right_index):
            current = store.state(group, physical)
            if current in {OBSERVED_LABELS, OBSERVED_VALID_EMPTY}:
                continue
            t = (float(context.axes[0][physical]) - z0_um) / (z1_um - z0_um)
            phi_left = left_flow.evaluate(t)
            phi_right = right_flow.evaluate(1.0 - t)
            plane = categorical_pair_plane(
                endpoint_left,
                endpoint_right,
                vocabulary,
                t,
                phi_left,
                phi_right,
                context.registered_axes,
                em,
                torch,
            )
            replace_placeholder = False
            if isinstance(store, TiffDenseAnnotationStore) and current == UNSUPPORTED:
                existing = store.read_group_plane(group, physical)
                replace_placeholder = not np.any(existing)
            store.write_group_plane(
                group,
                physical,
                plane,
                overwrite=overwrite or replace_placeholder,
            )
            store.set_state(group, physical, INFERRED)
            written += 1
        group_reports[str(group)] = {
            "pair_local_vocabulary": vocabulary,
            "inferred_canonical_planes_written": written,
        }
    elapsed = time.monotonic() - started
    cuda_peak = None
    if torch.cuda.is_available():
        cuda_peak = int(torch.cuda.max_memory_allocated())
    status = {
        "schema": PAIR_SCHEMA,
        "status": "complete",
        "output_format": store.output_format,
        "pair": list(pair),
        "pair_id": _pair_id(pair),
        "endpoint_z_um": [z0_um, z1_um],
        "delta_z_um": z1_um - z0_um,
        "nt": nt,
        "nt_role": "LDDMM temporal integration discretization",
        "nt_source": nt_source,
        "canonical_output_planes_between_endpoints": right_index - left_index - 1,
        "output_sampling": (
            "authoritative canonical z; t=(z-z0)/(z1-z0); arbitrary-t flow integration"
        ),
        "groups": list(groups),
        "configuration_sha256": configuration_sha256,
        "weight_construction": (
            "accepted final-A2d placement of the prepared Nissl mask channel; "
            "left and right endpoint planes supplied separately as W0"
        ),
        "map_convention": map_report,
        "categorical_estimator": "two-sided unweighted one-hot membership fusion; no Jacobian",
        "tie_rule": (
            "maximum fused membership, then maximum left membership, then maximum right "
            "membership, then smallest original Allen ID"
        ),
        "group_reports": group_reports,
        "wall_time_seconds": elapsed,
        "peak_process_rss_kib": max(start_rss, _peak_rss_kib()),
        "peak_cuda_allocated_bytes": cuda_peak,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _json(status_path, status)
    del left_flow, right_flow
    gc.collect()
    return status


def assert_observed_immutable(
    context: SourceContext, output: Path, store: DenseAnnotationStore
) -> dict[str, Any]:
    anchor_root = zarr.open_group(str(output / "anchors.zarr"), mode="r")
    rows = _read_tsv(output / "metadata/anchors.tsv")
    ordinal_by_section = {
        section: i for i, section in enumerate(context.annotation_sections)
    }
    checked = 0
    for row in rows:
        if row["semantic_state"] not in {SEMANTIC_LABELED, SEMANTIC_VALID_EMPTY}:
            continue
        section = int(row["allen_section_number"])
        physical = int(row["physical_index"])
        group = int(row["graphic_group"])
        observed = np.asarray(
            anchor_root["groups"][str(group)][ordinal_by_section[section]],
            dtype=np.uint32,
        )
        if not store.verify_plane(physical, group=group, expected=observed):
            raise RuntimeError(
                f"Observed-evidence immutability failed for section {section}, group {group}"
            )
        checked += 1
    return {
        "observed_section_group_arrays_checked": checked,
        "pixel_identical": True,
        "endpoints_generated_from_trajectories": False,
    }


def _write_state_table(context: SourceContext, output: Path, states: Any) -> Path:
    rows = []
    for group_index, group in enumerate(context.graphic_groups):
        values = np.asarray(states[group_index], dtype=np.uint8)
        for physical, value in enumerate(values):
            rows.append(
                {
                    "physical_index": physical,
                    "physical_z_um": f"{context.axes[0][physical]:.9g}",
                    "allen_section_number": context.rows[physical][
                        "allen_section_number"
                    ],
                    "graphic_group": group,
                    "provenance_state": STATE_NAME[int(value)],
                }
            )
    path = output / "metadata/section_group_provenance.tsv"
    _write_tsv(path, rows, list(rows[0]))
    return path


def finalize_dense(
    context: SourceContext,
    output: Path,
    store: DenseAnnotationStore,
    pair_groups: Mapping[tuple[int, int], Sequence[int]],
    anchor_report: Mapping[str, Any],
    config: Mapping[str, Any],
    pair_reports: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    missing_status = [
        pair
        for pair in pair_groups
        if _existing_pair_status_path(output, pair, store.output_format) is None
    ]
    if missing_status:
        raise RuntimeError(
            f"Cannot finalize: {len(missing_status)} endpoint pairs are incomplete"
        )
    for (left, right), groups in pair_groups.items():
        for group in groups:
            values = np.asarray(
                store.states[store.group_index[group], left + 1 : right]
            )
            if np.any(values != INFERRED):
                raise RuntimeError(
                    f"Pair {_pair_id((left, right))}, group {group} has missing inferred planes"
                )
    source_vocab = {
        int(value)
        for values in anchor_report["group_vocabularies"].values()
        for value in values
    }
    for physical in range(context.canonical_count):
        plane = _combined_display_map(
            [
                store.read_group_plane(group, physical)
                for group in context.graphic_groups
            ]
        )
        plane = np.asarray(plane, dtype=np.uint32)
        if not set(map(int, np.unique(plane))).issubset(source_vocab):
            raise RuntimeError("Combined product contains a non-authoritative Allen ID")
        store.write_combined_plane(physical, plane, overwrite=False)
    immutability = assert_observed_immutable(context, output, store)
    state_path = _write_state_table(context, output, store.states)
    manifest_path = store.finalize(include_combined=True)
    after = {path: _sha256(Path(path)) for path in context.source_hashes}
    if after != context.source_hashes:
        raise RuntimeError("An authoritative input changed during dense assembly")
    coordinate_product = (
        str(output / "dense.zarr/z_um")
        if store.output_format == "zarr"
        else str(output / "metadata/physical_sections.tsv")
    )
    report = {
        "schema": SCHEMA,
        "status": "complete",
        "output_format": store.output_format,
        "tiff_compression": (
            store.compression if isinstance(store, TiffDenseAnnotationStore) else None
        ),
        "method": (
            "two-sided Nissl-driven diffeomorphic interpolation of categorical Allen "
            "annotations using two endpoint-conditioned WSI trajectories"
        ),
        "canonical_lattice": {
            "positions": context.canonical_count,
            "physical_z_um_source": str(
                context.dataset / "metadata/physical_sections.tsv"
            ),
            "coordinate_product": coordinate_product,
            **(
                {"coordinate_array": coordinate_product}
                if store.output_format == "zarr"
                else {}
            ),
            "xJ0_verified_equal": True,
            "role": "both real-anchor locations and dense output sampling",
        },
        "authoritative_inputs": {
            "nissl": str(context.dataset),
            "annotations": str(context.annotations),
            "registration": str(context.registration),
            "numerical": str(context.numerical),
            "source_hashes": dict(context.source_hashes),
        },
        "graphic_groups": list(context.graphic_groups),
        "graphic_group_combination": (
            "existing ordered nonzero overwrite via visualize_allen_annotations._combined_display_map"
        ),
        "semantic_state_codes": {str(key): value for key, value in STATE_NAME.items()},
        "unsupported_zero_distinguished_by": (
            str(output / "dense.zarr/semantic_state")
            if store.output_format == "zarr"
            else str(state_path)
        ),
        "time_convention": (
            "for canonical z, t=(z-z0)/(z1-z0); source velocities are integrated "
            "to arbitrary t and complementary 1-t using WSI's Euler/composition convention"
        ),
        "lddmm_temporal_configuration": dict(config),
        "pair_count": len(pair_groups),
        "observed_evidence_immutability": immutability,
        "categorical_integrity": {
            "integer_ids_interpolated": False,
            "one_hot_scalar_memberships": True,
            "output_ids_checked_against_authoritative_vocabularies": True,
            "availability_inferred_from_label_nonzero": False,
        },
        "products": {
            "observed_anchors": str(output / "anchors.zarr"),
            "anchor_manifest": str(output / "metadata/anchors.tsv"),
            "physical_sections": str(output / "metadata/physical_sections.tsv"),
            "semantic_provenance_tsv": str(state_path),
            "pair_provenance": str(output / "metadata/pairs" / store.output_format),
            **store.products(),
        },
        "per_plane_sha256": manifest_path is not None,
        "capacity_planning": [
            {
                key: report.get(key)
                for key in (
                    "pair_id",
                    "delta_z_um",
                    "nt",
                    "canonical_output_planes_between_endpoints",
                    "wall_time_seconds",
                    "peak_process_rss_kib",
                    "peak_cuda_allocated_bytes",
                )
            }
            for report in pair_reports
        ],
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _json(output / "dataset.json", report)
    _json(output / "provenance.json", report)
    return report


def _write_output_readme(output: Path) -> None:
    text = """# Dense registered-histology Allen annotations

The default scientific raster representation is a lossless plain TIFF series:
one uint32 file per canonical z position and graphic group under `dense_tiff/`.
Files are named by six-digit canonical position index; physical z coordinates
are authoritative in `metadata/physical_sections.tsv`.

TIFF value 0 alone does not establish annotation availability. Consumers must
consult `metadata/section_group_provenance.tsv` and the TIFF manifest to
distinguish valid categorical background from unsupported or unavailable
evidence. These files are categorical scientific rasters, not display images.
Lossy compression is forbidden. The optional Zarr backend is selected explicitly
with `--output-format zarr` and remains in `dense.zarr/` when present.
"""
    path = output / "README.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text)
    os.replace(temporary, path)


def run(
    *,
    dataset: Path,
    registration: Path,
    annotations: Path | None,
    output: Path,
    pair: tuple[int, int] | None,
    pair_config: Path,
    wsi_repository: Path,
    device: str,
    overwrite: bool,
    nt_override: int | None = None,
    anchors_only: bool = False,
    output_format: str = "tiff",
    tiff_compression: str = "deflate",
) -> dict[str, Any]:
    context = discover_inputs(dataset, registration, annotations)
    _copy_file_transactionally(
        context.dataset / "metadata/physical_sections.tsv",
        output / "metadata/physical_sections.tsv",
    )
    _write_output_readme(output)
    anchor_report = materialize_anchors(context, output)
    anchor_rows = _read_tsv(output / "metadata/anchors.tsv")
    sequences, pair_groups = build_endpoint_sequences(
        anchor_rows, context.graphic_groups
    )
    if pair is not None:
        if pair[1] >= context.canonical_count:
            raise RuntimeError(
                f"Requested pair {_pair_id(pair)} is outside the canonical lattice"
            )
        if pair not in pair_groups:
            raise RuntimeError(
                f"Requested pair {_pair_id(pair)} is not in the required endpoint table"
            )
    if nt_override is not None and pair is None:
        raise RuntimeError("An nt override requires one explicitly selected pair")
    base_config = _load_pair_config(pair_config.resolve())

    pair_table = []
    for endpoint_pair, groups in pair_groups.items():
        left, right = endpoint_pair
        z0_um = float(context.axes[0][left])
        z1_um = float(context.axes[0][right])
        selected_override = nt_override if endpoint_pair == pair else None
        solver_config = pair_solver_config(base_config, selected_override)
        left_row, right_row = context.rows[left], context.rows[right]
        pair_table.append(
            {
                "pair_id": _pair_id(endpoint_pair),
                "left_physical_index": left,
                "right_physical_index": right,
                "left_z_um": f"{z0_um:.17g}",
                "right_z_um": f"{z1_um:.17g}",
                "delta_z_um": f"{z1_um - z0_um:.17g}",
                "canonical_index_gap": right - left,
                "canonical_interior_output_planes": right - left - 1,
                "lddmm_nt": int(solver_config["nt"]),
                "nt_source": (
                    "command_line_override"
                    if selected_override is not None
                    else "configured_temporal_discretization"
                ),
                "left_block_id": left_row["block_id"],
                "right_block_id": right_row["block_id"],
                "graphic_groups": json.dumps(list(groups), separators=(",", ":")),
            }
        )
    pair_table_path = output / "metadata/endpoint_pairs.tsv"
    _write_tsv(pair_table_path, pair_table, list(pair_table[0]))
    _json(
        output / "metadata/endpoint_sequences.json",
        {
            "semantic_anchor_physical_indices": {
                str(group): values for group, values in sequences.items()
            },
            "unique_endpoint_pair_count": len(pair_groups),
            "canonical_output_positions": context.canonical_count,
            "canonical_z_coordinate_source": str(
                output / "metadata/physical_sections.tsv"
            ),
            "configured_lddmm_nt": int(base_config["nt"]),
            "output_time_rule": "t=(z-z0)/(z1-z0) at each canonical physical z",
            "output_format": output_format,
        },
    )
    planning = {
        "unique_endpoint_pairs": len(pair_groups),
        "canonical_output_positions": context.canonical_count,
        "endpoint_pair_table": str(pair_table_path),
        "configured_lddmm_nt": int(base_config["nt"]),
        "output_time_rule": "t=(z-z0)/(z1-z0)",
        "output_format": output_format,
    }
    if anchors_only:
        return {
            "status": "anchors_and_plan_complete",
            **anchor_report,
            "planning": planning,
        }

    selected = list(pair_groups) if pair is None else [pair]
    recoverable_planes = {
        (group, physical)
        for endpoint_pair in selected
        for group in pair_groups[endpoint_pair]
        for physical in range(endpoint_pair[0] + 1, endpoint_pair[1])
    }
    store = create_dense_store(
        output,
        context.shape,
        context.axes[0],
        output_format=output_format,
        tiff_compression=tiff_compression,
        groups=context.graphic_groups,
    )
    initialize_dense(context, output, store, recoverable_planes)
    reports = []
    for endpoint_pair in selected:
        selected_override = nt_override if endpoint_pair == pair else None
        solver_config = pair_solver_config(base_config, selected_override)
        reports.append(
            process_pair(
                context,
                output,
                store,
                endpoint_pair,
                pair_groups[endpoint_pair],
                solver_config,
                nt_source=(
                    "command_line_override"
                    if selected_override is not None
                    else "configured_temporal_discretization"
                ),
                wsi_repository=wsi_repository,
                device=device,
                overwrite=overwrite,
            )
        )
    immutability = assert_observed_immutable(context, output, store)
    if pair is not None:
        state_path = _write_state_table(context, output, store.states)
        manifest_path = store.finalize(include_combined=False)
        return {
            "status": "pair_complete",
            "output_format": store.output_format,
            "pair": list(pair),
            "report": reports[0],
            "planning": planning,
            "observed_evidence_immutability": immutability,
            "semantic_provenance_tsv": str(state_path),
            "tiff_manifest": str(manifest_path) if manifest_path else None,
            "output": str(output),
        }
    return finalize_dense(
        context,
        output,
        store,
        pair_groups,
        anchor_report,
        base_config,
        reports,
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Densify sparse Allen annotations after accepted final registered-histology placement"
        )
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--annotations",
        type=Path,
        default=None,
        help="annotation derivative (default: unique derivative linked to --dataset)",
    )
    parser.add_argument(
        "--registration-run",
        type=Path,
        default=None,
        help="accepted registration run (required for a non-default --dataset)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="output root (required for a non-default --dataset)",
    )
    parser.add_argument(
        "--output-format",
        choices=("tiff", "zarr"),
        default="tiff",
        help="primary dense raster backend (default: tiff)",
    )
    parser.add_argument(
        "--tiff-compression",
        choices=("deflate", "none"),
        default="deflate",
        help="lossless TIFF compression (used only for TIFF output)",
    )
    parser.add_argument(
        "--pair",
        type=_parse_pair,
        default=None,
        help="one required endpoint pair as LEFT_PHYSICAL_INDEX-RIGHT_PHYSICAL_INDEX",
    )
    parser.add_argument(
        "--nt",
        type=int,
        default=None,
        help="expert nt override for one explicit --pair (use a dedicated output)",
    )
    parser.add_argument("--pair-config", type=Path, default=DEFAULT_PAIR_CONFIG)
    parser.add_argument("--wsi-repository", type=Path, default=DEFAULT_WSI_REPOSITORY)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="recompute a selected completed pair; observed anchors remain immutable",
    )
    parser.add_argument("--anchors-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    dataset = args.dataset.expanduser().resolve()
    using_default_dataset = dataset == DEFAULT_DATASET.resolve()
    if args.registration_run is None and not using_default_dataset:
        parser.error("--registration-run is required with a non-default --dataset")
    if args.output is None and not using_default_dataset:
        parser.error("--output is required with a non-default --dataset")
    registration = args.registration_run or DEFAULT_REGISTRATION
    output = args.output or DEFAULT_OUTPUT
    if args.overwrite and args.pair is None:
        parser.error("--overwrite is limited to an explicitly selected --pair")
    if args.nt is not None and args.pair is None:
        parser.error("--nt requires an explicitly selected --pair")
    report = run(
        dataset=dataset,
        registration=registration,
        annotations=args.annotations,
        output=output.expanduser().resolve(),
        pair=args.pair,
        pair_config=args.pair_config,
        wsi_repository=args.wsi_repository,
        device=args.device,
        overwrite=args.overwrite,
        nt_override=args.nt,
        anchors_only=args.anchors_only,
        output_format=args.output_format,
        tiff_compression=args.tiff_compression,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
