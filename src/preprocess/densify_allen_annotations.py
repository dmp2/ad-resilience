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
DEFAULT_ANNOTATIONS = PROJECT / (
    "data/derivatives/allen/specimen_708424/"
    "annotations_symmetric_nissl_native_200um_section_aligned"
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
GROUP_ORDER = (31, 113753816, 141667008, 265297118)
EXPECTED_SHAPE = (522, 730)
EXPECTED_SECTION_COUNT = 2846
EXPECTED_ANNOTATION_LEVELS = 106

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


def discover_inputs(
    dataset: Path, registration: Path, annotations: Path | None = None
) -> SourceContext:
    """Validate and freeze authoritative geometry, derivatives, and final A2d."""
    dataset = dataset.expanduser().resolve()
    registration = registration.expanduser().resolve()
    annotations = (
        annotations.expanduser().resolve()
        if annotations is not None
        else DEFAULT_ANNOTATIONS.resolve()
    )
    for required in (dataset / "dataset.json", annotations / "dataset.json"):
        if not required.is_file() or required.is_symlink():
            raise RuntimeError(f"Missing authoritative derivative metadata: {required}")

    nissl_meta = json.loads((dataset / "dataset.json").read_text())
    annotation_meta = json.loads((annotations / "dataset.json").read_text())
    if nissl_meta.get("physical_serial_positions") != EXPECTED_SECTION_COUNT:
        raise RuntimeError(
            "Nissl derivative is not the canonical 2,846-position product"
        )
    if nissl_meta.get("prepared_canvas_shape_yx") != list(EXPECTED_SHAPE):
        raise RuntimeError(
            "Nissl derivative is not the authoritative 522x730 native grid"
        )
    if nissl_meta.get("preparation_mode") != "preserve_source_grid":
        raise RuntimeError(
            "Nissl derivative did not preserve its authoritative source grid"
        )
    recorded_parent = annotation_meta.get("parent_nissl_derivative")
    if (
        not isinstance(recorded_parent, str)
        or Path(recorded_parent).resolve() != dataset
    ):
        raise RuntimeError(
            "Annotation derivative does not name the selected Nissl parent"
        )
    if annotation_meta.get("parent_nissl_dataset_json_sha256") != _sha256(
        dataset / "dataset.json"
    ):
        raise RuntimeError("Annotation derivative parent checksum differs")
    if annotation_meta.get("prepared_canvas_shape_yx") != list(EXPECTED_SHAPE):
        raise RuntimeError("Annotation and Nissl grids differ")
    if annotation_meta.get("graphic_group_counts") != {
        "31": 106,
        "113753816": 106,
        "141667008": 102,
        "265297118": 106,
    }:
        raise RuntimeError("Authoritative graphic-group coverage changed")

    physical_path = dataset / "metadata/physical_sections.tsv"
    rows = _read_tsv(physical_path)
    if len(rows) != EXPECTED_SECTION_COUNT:
        raise RuntimeError("Canonical physical-section table length changed")
    physical = np.asarray([int(row["physical_index"]) for row in rows], dtype=np.int64)
    if not np.array_equal(physical, np.arange(EXPECTED_SECTION_COUNT)):
        raise RuntimeError(
            "physical_index is not the authoritative consecutive lattice"
        )
    z_from_table = np.asarray(
        [float(row["serial_z_center_mm"]) * 1000.0 for row in rows], dtype=np.float64
    )
    _validate_uniform_axis(z_from_table, 50.0, "physical_z_um")
    physical_by_section = {
        int(row["allen_section_number"]): i for i, row in enumerate(rows)
    }

    source_manifest_path = annotations / "metadata/source_annotation_manifest.tsv"
    source_manifest = _read_tsv(source_manifest_path)
    annotation_sections = tuple(int(row["section_number"]) for row in source_manifest)
    if len(annotation_sections) != EXPECTED_ANNOTATION_LEVELS or len(
        set(annotation_sections)
    ) != len(annotation_sections):
        raise RuntimeError("Authoritative annotation-level identity/count changed")
    inventory_path = annotations / "metadata/annotations.tsv"
    inventory_rows = _read_tsv(inventory_path)
    inventory: dict[tuple[int, int], dict[str, str]] = {}
    for row in inventory_rows:
        key = (int(row["section_number"]), int(row["graphic_group_id"]))
        if key in inventory or key[1] not in GROUP_ORDER:
            raise RuntimeError(
                f"Invalid duplicate/unknown annotation inventory row: {key}"
            )
        if row.get("sampling") != "categorical_nearest_neighbor":
            raise RuntimeError(f"Anchor {key} lacks categorical placement provenance")
        inventory[key] = row
    expected_missing = {
        (111, 141667008),
        (179, 141667008),
        (2737, 141667008),
        (2797, 141667008),
    }
    actual_missing = {
        (section, group)
        for section in annotation_sections
        for group in GROUP_ORDER
        if (section, group) not in inventory
    }
    if actual_missing != expected_missing:
        raise RuntimeError(
            f"Unresolved graphic-group availability changed: {sorted(actual_missing)}"
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
    if final_a2d.shape != (EXPECTED_SECTION_COUNT, 3, 3):
        raise RuntimeError("Accepted final A2d does not span the physical lattice")
    if tuple(map(len, axes)) != (EXPECTED_SECTION_COUNT, *EXPECTED_SHAPE):
        raise RuntimeError("Accepted registration histology axes changed")
    if not np.allclose(axes[0], z_from_table, atol=1e-9, rtol=0.0):
        raise RuntimeError("Accepted xJ[0] differs from canonical physical coordinates")
    _validate_uniform_axis(axes[0], 50.0, "accepted xJ[0]")
    _validate_uniform_axis(axes[1], 200.0, "accepted xJ[1]")
    _validate_uniform_axis(axes[2], 200.0, "accepted xJ[2]")
    table_observed = np.flatnonzero(
        [row["image_present"] == "true" and row["stain"] == "nissl" for row in rows]
    )
    if not np.array_equal(observed, table_observed):
        raise RuntimeError(
            "Accepted observed Nissl indices differ from the canonical table"
        )
    unsupported = np.ones(EXPECTED_SECTION_COUNT, dtype=bool)
    unsupported[observed] = False
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
    if tuple(map(len, registered_axes)) != EXPECTED_SHAPE:
        raise RuntimeError("Registered-histology grid shape changed")

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
        if report.get("source_hashes") != context.source_hashes:
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
        shape=(count, 3, *EXPECTED_SHAPE),
        chunks=(1, 3, *EXPECTED_SHAPE),
        dtype="float32",
    )
    weight = _zarr_array(
        root,
        "nissl_weight",
        shape=(count, *EXPECTED_SHAPE),
        chunks=(1, *EXPECTED_SHAPE),
        dtype="float32",
    )
    groups_root = root.require_group("groups")
    group_arrays = {
        group: _zarr_array(
            groups_root,
            str(group),
            shape=(count, *EXPECTED_SHAPE),
            chunks=(1, *EXPECTED_SHAPE),
            dtype="uint32",
        )
        for group in GROUP_ORDER
    }
    rows: list[dict[str, Any]] = []
    vocabularies: dict[int, set[int]] = {group: {0} for group in GROUP_ORDER}
    y, x = context.registered_axes
    source_y, source_x = context.axes[1:]
    try:
        for ordinal, section in enumerate(context.annotation_sections):
            physical_index = context.physical_by_section[section]
            if physical_index not in set(map(int, context.observed_nissl)):
                raise RuntimeError(
                    f"Annotated section {section} lacks accepted Nissl placement"
                )
            image_name = f"allen_708424_nissl_{section:04d}.tif"
            image_path = context.dataset / "inputs/views/HIST_NISSL" / image_name
            weight_path = context.dataset / "support/nissl" / image_name
            image = tifffile.imread(image_path)
            raw_weight = tifffile.imread(weight_path).astype(np.float32)
            if (
                image.shape != (*EXPECTED_SHAPE, 3)
                or raw_weight.shape != EXPECTED_SHAPE
            ):
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
            for group in GROUP_ORDER:
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
                    if labels.shape != EXPECTED_SHAPE or not np.issubdtype(
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
                        "final_rows": EXPECTED_SHAPE[0],
                        "final_columns": EXPECTED_SHAPE[1],
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
        "graphic_groups": list(GROUP_ORDER),
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
) -> tuple[dict[int, list[int]], dict[tuple[int, int], tuple[int, ...]]]:
    """Build per-group semantic anchors and the compact unique-pair table."""
    by_group: dict[int, list[int]] = {group: [] for group in GROUP_ORDER}
    for row in anchor_rows:
        group = int(row["graphic_group"])
        if row["semantic_state"] in {SEMANTIC_LABELED, SEMANTIC_VALID_EMPTY}:
            by_group[group].append(int(row["physical_index"]))
    pair_groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for group in GROUP_ORDER:
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


def _dense_products(
    output: Path, shape: tuple[int, int], canonical_z_um: np.ndarray
) -> tuple[Any, dict[int, Any], Any]:
    if len(canonical_z_um) != EXPECTED_SECTION_COUNT:
        raise RuntimeError(
            "Dense output must use the authoritative canonical z lattice"
        )
    root = zarr.open_group(str(output / "dense.zarr"), mode="a")
    coordinate_exists = "z_um" in root
    coordinate = _zarr_array(
        root,
        "z_um",
        shape=(EXPECTED_SECTION_COUNT,),
        chunks=(EXPECTED_SECTION_COUNT,),
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
            shape=(EXPECTED_SECTION_COUNT, *shape),
            chunks=(1, *shape),
            dtype="uint32",
        )
        for group in GROUP_ORDER
    }
    states = _zarr_array(
        root,
        "semantic_state",
        shape=(len(GROUP_ORDER), EXPECTED_SECTION_COUNT),
        chunks=(1, EXPECTED_SECTION_COUNT),
        dtype="uint8",
    )
    return root, arrays, states


def initialize_dense(
    context: SourceContext, output: Path
) -> tuple[Any, dict[int, Any], Any, dict[int, int]]:
    anchor_root = zarr.open_group(str(output / "anchors.zarr"), mode="r")
    anchor_rows = _read_tsv(output / "metadata/anchors.tsv")
    ordinal_by_physical: dict[int, int] = {}
    for ordinal, section in enumerate(context.annotation_sections):
        ordinal_by_physical[context.physical_by_section[section]] = ordinal
    root, arrays, states = _dense_products(output, EXPECTED_SHAPE, context.axes[0])
    group_index = {group: i for i, group in enumerate(GROUP_ORDER)}
    anchor_groups = anchor_root["groups"]
    for row in anchor_rows:
        state = row["semantic_state"]
        if state not in {SEMANTIC_LABELED, SEMANTIC_VALID_EMPTY}:
            continue
        physical = int(row["physical_index"])
        group = int(row["graphic_group"])
        ordinal = ordinal_by_physical[physical]
        arrays[group][physical] = anchor_groups[str(group)][ordinal]
        states[group_index[group], physical] = (
            OBSERVED_LABELS if state == SEMANTIC_LABELED else OBSERVED_VALID_EMPTY
        )
    return root, arrays, states, ordinal_by_physical


def _pair_id(pair: tuple[int, int]) -> str:
    return f"{pair[0]:04d}-{pair[1]:04d}"


def _parse_pair(value: str) -> tuple[int, int]:
    try:
        left, right = value.split("-", 1)
        pair = (int(left), int(right))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            "pair must be PHYSICAL_INDEX-PHYSICAL_INDEX"
        ) from exc
    if not 0 <= pair[0] < pair[1] < EXPECTED_SECTION_COUNT:
        raise argparse.ArgumentTypeError(
            "pair indices are outside the canonical lattice"
        )
    return pair


def _peak_rss_kib() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def process_pair(
    context: SourceContext,
    output: Path,
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
    status_path = output / "metadata/pairs" / f"{_pair_id(pair)}.json"
    if status_path.is_file() and not overwrite:
        status = json.loads(status_path.read_text())
        if status.get("status") != "complete":
            raise RuntimeError(f"Existing pair status is incomplete: {status_path}")
        if (
            status.get("nt") != nt
            or status.get("configuration_sha256") != configuration_sha256
        ):
            raise RuntimeError(
                f"Existing pair status uses another nt/configuration: {status_path}"
            )
        return status
    anchor_root = zarr.open_group(str(output / "anchors.zarr"), mode="r")
    _, dense, states = _dense_products(output, EXPECTED_SHAPE, context.axes[0])
    ordinal_by_physical = {
        context.physical_by_section[section]: ordinal
        for ordinal, section in enumerate(context.annotation_sections)
    }
    left_index, right_index = pair
    left_ordinal = ordinal_by_physical[left_index]
    right_ordinal = ordinal_by_physical[right_index]
    z0_um = float(context.axes[0][left_index])
    z1_um = float(context.axes[0][right_index])
    left_image = np.asarray(anchor_root["nissl"][left_ordinal], dtype=np.float32)
    right_image = np.asarray(anchor_root["nissl"][right_ordinal], dtype=np.float32)
    left_weight = np.asarray(
        anchor_root["nissl_weight"][left_ordinal], dtype=np.float32
    )
    right_weight = np.asarray(
        anchor_root["nissl_weight"][right_ordinal], dtype=np.float32
    )
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
    group_index = {group: i for i, group in enumerate(GROUP_ORDER)}
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
            current = int(states[group_index[group], physical])
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
            dense[group][physical] = plane
            states[group_index[group], physical] = INFERRED
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


def assert_observed_immutable(context: SourceContext, output: Path) -> dict[str, Any]:
    anchor_root = zarr.open_group(str(output / "anchors.zarr"), mode="r")
    dense_root = zarr.open_group(str(output / "dense.zarr"), mode="r")
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
            anchor_root["groups"][str(group)][ordinal_by_section[section]]
        )
        emitted = np.asarray(dense_root["groups"][str(group)][physical])
        if not np.array_equal(observed, emitted):
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
    for group_index, group in enumerate(GROUP_ORDER):
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
    pair_groups: Mapping[tuple[int, int], Sequence[int]],
    anchor_report: Mapping[str, Any],
    config: Mapping[str, Any],
    pair_reports: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    root, dense, states = _dense_products(output, EXPECTED_SHAPE, context.axes[0])
    missing_status = [
        pair
        for pair in pair_groups
        if not (output / "metadata/pairs" / f"{_pair_id(pair)}.json").is_file()
    ]
    if missing_status:
        raise RuntimeError(
            f"Cannot finalize: {len(missing_status)} endpoint pairs are incomplete"
        )
    for (left, right), groups in pair_groups.items():
        for group in groups:
            values = np.asarray(states[GROUP_ORDER.index(group), left + 1 : right])
            if np.any(values != INFERRED):
                raise RuntimeError(
                    f"Pair {_pair_id((left, right))}, group {group} has missing inferred planes"
                )
    combined = _zarr_array(
        root,
        "combined",
        shape=(EXPECTED_SECTION_COUNT, *EXPECTED_SHAPE),
        chunks=(1, *EXPECTED_SHAPE),
        dtype="uint32",
    )
    source_vocab = {
        int(value)
        for values in anchor_report["group_vocabularies"].values()
        for value in values
    }
    for physical in range(EXPECTED_SECTION_COUNT):
        plane = _combined_display_map(
            [
                np.asarray(dense[group][physical], dtype=np.uint32)
                for group in GROUP_ORDER
            ]
        )
        if not set(map(int, np.unique(plane))).issubset(source_vocab):
            raise RuntimeError("Combined product contains a non-authoritative Allen ID")
        combined[physical] = plane
    immutability = assert_observed_immutable(context, output)
    state_path = _write_state_table(context, output, states)
    after = {path: _sha256(Path(path)) for path in context.source_hashes}
    if after != context.source_hashes:
        raise RuntimeError("An authoritative input changed during dense assembly")
    report = {
        "schema": SCHEMA,
        "status": "complete",
        "method": (
            "two-sided Nissl-driven diffeomorphic interpolation of categorical Allen "
            "annotations using two endpoint-conditioned WSI trajectories"
        ),
        "canonical_lattice": {
            "positions": EXPECTED_SECTION_COUNT,
            "physical_z_um_source": str(
                context.dataset / "metadata/physical_sections.tsv"
            ),
            "coordinate_array": str(output / "dense.zarr/z_um"),
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
        "graphic_groups": list(GROUP_ORDER),
        "graphic_group_combination": (
            "existing ordered nonzero overwrite via visualize_allen_annotations._combined_display_map"
        ),
        "semantic_state_codes": {str(key): value for key, value in STATE_NAME.items()},
        "unsupported_zero_distinguished_by": "dense.zarr/semantic_state",
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
            "dense_z_coordinate": str(output / "dense.zarr/z_um"),
            "dense_per_group": str(output / "dense.zarr/groups"),
            "semantic_state": str(output / "dense.zarr/semantic_state"),
            "semantic_provenance_tsv": str(state_path),
            "combined": str(output / "dense.zarr/combined"),
            "pair_provenance": str(output / "metadata/pairs"),
        },
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
    return report


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
) -> dict[str, Any]:
    context = discover_inputs(dataset, registration, annotations)
    anchor_report = materialize_anchors(context, output)
    anchor_rows = _read_tsv(output / "metadata/anchors.tsv")
    sequences, pair_groups = build_endpoint_sequences(anchor_rows)
    if pair is not None and pair not in pair_groups:
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
            "canonical_output_positions": EXPECTED_SECTION_COUNT,
            "canonical_z_coordinate_source": str(
                context.dataset / "metadata/physical_sections.tsv"
            ),
            "configured_lddmm_nt": int(base_config["nt"]),
            "output_time_rule": "t=(z-z0)/(z1-z0) at each canonical physical z",
        },
    )
    planning = {
        "unique_endpoint_pairs": len(pair_groups),
        "canonical_output_positions": EXPECTED_SECTION_COUNT,
        "endpoint_pair_table": str(pair_table_path),
        "configured_lddmm_nt": int(base_config["nt"]),
        "output_time_rule": "t=(z-z0)/(z1-z0)",
    }
    if anchors_only:
        return {
            "status": "anchors_and_plan_complete",
            **anchor_report,
            "planning": planning,
        }

    initialize_dense(context, output)
    selected = list(pair_groups) if pair is None else [pair]
    reports = []
    for endpoint_pair in selected:
        selected_override = nt_override if endpoint_pair == pair else None
        solver_config = pair_solver_config(base_config, selected_override)
        reports.append(
            process_pair(
                context,
                output,
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
    immutability = assert_observed_immutable(context, output)
    if pair is not None:
        return {
            "status": "pair_complete",
            "pair": list(pair),
            "report": reports[0],
            "planning": planning,
            "observed_evidence_immutability": immutability,
            "output": str(output),
        }
    return finalize_dense(
        context,
        output,
        pair_groups,
        anchor_report,
        base_config,
        reports,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Densify sparse Allen annotations after accepted final registered-histology placement"
        )
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--annotations", type=Path, default=None)
    parser.add_argument("--registration-run", type=Path, default=DEFAULT_REGISTRATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
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
    args = parser.parse_args(argv)
    if args.overwrite and args.pair is None:
        parser.error("--overwrite is limited to an explicitly selected --pair")
    if args.nt is not None and args.pair is None:
        parser.error("--nt requires an explicitly selected --pair")
    report = run(
        dataset=args.dataset,
        registration=args.registration_run,
        annotations=args.annotations,
        output=args.output.expanduser().resolve(),
        pair=args.pair,
        pair_config=args.pair_config,
        wsi_repository=args.wsi_repository,
        device=args.device,
        overwrite=args.overwrite,
        nt_override=args.nt,
        anchors_only=args.anchors_only,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
