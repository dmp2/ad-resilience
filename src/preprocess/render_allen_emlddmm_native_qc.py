"""Render isolated QC from the saved 400-um native Allen EM-LDDMM state.

This module is deliberately an adapter around the pinned upstream QC writer.  It
does not register images, apply transforms, or reproduce EM-LDDMM QC math.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
import torch

from preprocess.prepare_allen_emlddmm_inputs import accepted_loader_axes
from preprocess.run_allen_emlddmm import (
    load_physical_rows,
    load_pinned_mri_image,
    mri_physical_axes_from_provenance,
    pinned_emlddmm,
    sha256_file,
)
from preprocess import run_allen_emlddmm_full_coarse_nissl as coarse


PROJECT = Path(__file__).resolve().parents[2]
PIN = "864990e0619fcdfb3e22e05298291f439f1b6f3d"
PIN_FILE = PROJECT / "configs/emlddmm-upstream-commit.txt"
REGISTRATION_ROOT = PROJECT / (
    "results/allen/specimen_708424/emlddmm/native-200um-clean/"
    "HIST_NISSL_SYMMETRIC_SECTION_ALIGNED_to_MRI_7T_WHOLE"
)
CHECKPOINT = REGISTRATION_ROOT / "checkpoints/registration.json"
DATASET = PROJECT / (
    "data/derivatives/allen/specimen_708424/"
    "histology_symmetric_nissl_native_200um_section_aligned"
)
VIEW = DATASET / "inputs/views/HIST_NISSL"
SUPPORT = DATASET / "support/nissl"
MRI = PROJECT / (
    "data/derivatives/allen/specimen_708424/mri_7t_whole/"
    "T1_rot_space-MRI_7T_WHOLE_desc-header-corrected.nii"
)
MRI_PROVENANCE = PROJECT / (
    "data/derivatives/allen/specimen_708424/mri_7t_whole/mri_provenance.json"
)
POSTPROCESSED_QC = REGISTRATION_ROOT / "postprocessed_qc"
MEMORY_CEILING_BYTES = 50 * 1024**3
MEMORY_MARGIN_BYTES = 8 * 1024**3
QC_MEMORY_MULTIPLIER = 13
EXPECTED_COMMIT = PIN
EXPECTED_SCALE_INDEX = 1
EXPECTED_SCALE_NUMBER = 2
EXPECTED_EFFECTIVE_INPLANE_UM = 400.0
EXPECTED_DOWN_I = [2, 2, 2]
EXPECTED_DOWN_J = [1, 2, 2]
EXPECTED_ROWS = 2846
EXPECTED_OBSERVED = 641
EXPECTED_SOURCE_SHAPE = (1, 474, 568, 509)
EXPECTED_TARGET_SHAPE = (3, 2846, 261, 365)
REQUIRED_STATE = ("A", "A2d", "v", "xv0", "xv1", "xv2")
EXPECTED_QC_OUTPUTS = (
    Path("HIST_NISSL/MRI_7T_WHOLE_to_HIST_NISSL/qc/")
    / "MRI_7T_WHOLE_7T_T1_to_HIST_NISSL.jpg",
    Path("HIST_NISSL/MRI_7T_WHOLE_to_HIST_NISSL/qc/")
    / "HIST_NISSL_HIST_NISSL.jpg",
    Path("HIST_NISSL_registered/")
    / "MRI_7T_WHOLE_to_HIST_NISSL_registered/qc/"
    / "HIST_NISSL_HIST_NISSL_registered.jpg",
    Path("HIST_NISSL_registered/")
    / "MRI_7T_WHOLE_to_HIST_NISSL_registered/qc/"
    / "MRI_7T_WHOLE_7T_T1_to_HIST_NISSL_registered.jpg",
    Path("MRI_7T_WHOLE/")
    / "HIST_NISSL_registered_to_MRI_7T_WHOLE/qc/"
    / "HIST_NISSL_HIST_NISSL_to_MRI_7T_WHOLE.jpg",
    Path("MRI_7T_WHOLE/")
    / "HIST_NISSL_registered_to_MRI_7T_WHOLE/qc/"
    / "MRI_7T_WHOLE_7T_T1.jpg",
)


@dataclass(frozen=True)
class ImageFacade:
    """Minimal interface consumed by pinned ``write_qc_outputs``."""

    space: str
    name: str
    data: np.ndarray
    x: list[np.ndarray]
    title: str
    names: list[str]

    def fnames(self) -> list[str]:
        return self.names


@dataclass(frozen=True)
class HistologyContext:
    rows: list[dict[str, str]]
    samples: list[dict[str, str]]
    axes: list[np.ndarray]
    observed: np.ndarray


@dataclass(frozen=True)
class ResolvedState:
    checkpoint: dict[str, Any]
    scale_manifest: dict[str, Any]
    provenance: dict[str, Any]
    numerical_path: Path
    arrays: dict[str, np.ndarray]
    native_xi: list[np.ndarray]
    native_xj: list[np.ndarray]
    histology: HistologyContext
    mri_provenance: dict[str, Any]
    down_i: list[int]
    down_j: list[int]
    source_shape: tuple[int, ...]
    target_shape: tuple[int, ...]
    source_dtype: np.dtype[Any]
    target_dtype: np.dtype[Any]
    estimated_memory_bytes: int
    mem_available_bytes: int
    stage_dir: Path
    final_dir: Path


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot read JSON: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return value


def _recorded_path(value: Any, *, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"Checkpoint field {field} is not a path")
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT / path
    if not path.is_file():
        raise RuntimeError(f"Recorded {field} is missing: {path}")
    return path.resolve()


def _recorded_directory(value: Any, *, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"Checkpoint field {field} is not a directory")
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT / path
    if not path.is_dir():
        raise RuntimeError(f"Recorded {field} is missing: {path}")
    return path.resolve()


def _require_equal(name: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise RuntimeError(f"{name} is {actual!r}, expected {expected!r}")


def _load_histology_context() -> HistologyContext:
    rows = load_physical_rows(DATASET)
    samples_path = VIEW / "samples.tsv"
    with samples_path.open(newline="", encoding="utf-8") as stream:
        samples = list(csv.DictReader(stream, delimiter="\t"))
    if len(rows) != EXPECTED_ROWS or len(samples) != len(rows):
        raise RuntimeError(
            f"Expected {EXPECTED_ROWS:,} physical and slice-manifest rows"
        )
    canvas = _json(DATASET / "metadata/loader_canvas_audit.json")
    axes = [np.asarray(axis) for axis in accepted_loader_axes(rows, canvas)]
    observed = np.asarray(
        [
            index
            for index, (row, sample) in enumerate(zip(rows, samples, strict=True))
            if sample["status"] == "present" and row["stain"] == "nissl"
        ],
        dtype=np.int64,
    )
    if observed.size != EXPECTED_OBSERVED:
        raise RuntimeError(
            f"Expected {EXPECTED_OBSERVED} observed Nissl sections, "
            f"found {observed.size}"
        )
    return HistologyContext(rows, samples, axes, observed)


def _load_numerical(
    path: Path, scale_manifest: dict[str, Any]
) -> tuple[dict[str, np.ndarray], list[np.ndarray], list[np.ndarray]]:
    expected_keys = set(REQUIRED_STATE) | {
        "xI0", "xI1", "xI2", "xJ0", "xJ1", "xJ2"
    }
    try:
        with np.load(path, allow_pickle=False) as saved:
            missing = expected_keys.difference(saved.files)
            if missing:
                raise RuntimeError(f"Numerical product lacks {sorted(missing)}")
            arrays = {key: np.asarray(saved[key]).copy() for key in REQUIRED_STATE}
            native_xi = [np.asarray(saved[f"xI{i}"]).copy() for i in range(3)]
            native_xj = [np.asarray(saved[f"xJ{i}"]).copy() for i in range(3)]
    except (OSError, ValueError, KeyError) as exc:
        raise RuntimeError(f"Cannot load numerical product: {path}") from exc
    shapes = scale_manifest.get("state_shapes", {})
    dtypes = scale_manifest.get("state_dtypes", {})
    for key, array in arrays.items():
        if list(array.shape) != shapes.get(key):
            raise RuntimeError(f"Saved {key} shape differs from scale manifest")
        if array.dtype.name != dtypes.get(key):
            raise RuntimeError(f"Saved {key} dtype differs from scale manifest")
        if not np.all(np.isfinite(array)):
            raise RuntimeError(f"Saved {key} contains nonfinite values")
    return arrays, native_xi, native_xj


def _available_memory_bytes() -> int:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError) as exc:
        raise RuntimeError("Cannot determine MemAvailable from /proc/meminfo") from exc
    raise RuntimeError("MemAvailable is absent from /proc/meminfo")


def _downsampled_shape(
    native_shape: list[int], factors: list[int]
) -> tuple[int, ...]:
    if len(native_shape) != len(factors) + 1:
        raise RuntimeError("Image shape and spatial downsampling factors differ")
    if any(type(factor) is not int or factor < 1 for factor in factors):
        raise RuntimeError("Downsampling factors must be positive integers")
    return (native_shape[0],) + tuple(
        size // factor
        for size, factor in zip(native_shape[1:], factors, strict=True)
    )


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _assert_axes_equal(
    name: str, actual: list[np.ndarray], expected: list[np.ndarray]
) -> None:
    if len(actual) != len(expected) or any(
        not np.array_equal(np.asarray(left), np.asarray(right))
        for left, right in zip(actual, expected, strict=True)
    ):
        raise RuntimeError(f"{name} native axes differ from the numerical product")


def _section_paths(
    context: HistologyContext, index: int
) -> tuple[Path, Path]:
    sample = context.samples[index]
    image = VIEW / sample["sample_id"]
    support = SUPPORT / image.name
    if not image.is_file() or not support.is_file():
        raise RuntimeError(f"Missing image/support pair for physical index {index}")
    return image, support


def _load_native_section(
    context: HistologyContext, index: int, *, image: bool
) -> tuple[np.ndarray | None, np.ndarray]:
    image_path, support_path = _section_paths(context, index)
    support = tifffile.imread(support_path).astype(np.float32)
    expected_shape = tuple(map(len, context.axes[1:]))
    if support.shape != expected_shape or not np.all(np.isfinite(support)):
        raise RuntimeError(f"Invalid support at physical index {index}")
    if np.min(support) < 0.0 or np.max(support) != 1.0:
        raise RuntimeError(f"Support range changed at physical index {index}")
    if not image:
        return None, support
    raw = tifffile.imread(image_path)
    data = raw[..., :3].astype(np.float32) / 255.0
    if data.shape[:2] != expected_shape:
        raise RuntimeError(f"Invalid Nissl shape at physical index {index}")
    return data.transpose(2, 0, 1), support


def _weighted_section_downsample(
    em: Any,
    axes: list[np.ndarray],
    image: np.ndarray,
    support: np.ndarray,
    factors: list[int],
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    down_axes, down_image, down_support = em.downsample_image_domain(
        axes,
        image[:, None],
        factors,
        W=support[None],
    )
    down_axes = [np.asarray(axis) for axis in down_axes]
    down_image = np.asarray(down_image)
    down_support = np.asarray(down_support)
    if float(np.max(down_support)) != 1.0:
        raise RuntimeError(
            "Section support does not reach the shared full-support maximum; "
            "sectionwise pinned downsampling is not equivalent to the full stack"
        )
    return down_axes, down_image, down_support


def _audit_weighted_section_equivalence(
    em: Any, context: HistologyContext, down_j: list[int]
) -> list[np.ndarray]:
    expected_axes: list[np.ndarray] | None = None
    native_shape = tuple(map(len, context.axes[1:]))
    zero = np.zeros((1, *native_shape), dtype=np.float32)
    for index in context.observed:
        _, support = _load_native_section(context, int(index), image=False)
        section_axes = [context.axes[0][index : index + 1], *context.axes[1:]]
        down_axes, _, _ = _weighted_section_downsample(
            em, section_axes, zero, support, down_j
        )
        inplane = down_axes[1:]
        if expected_axes is None:
            expected_axes = [context.axes[0], *inplane]
        elif any(
            not np.array_equal(actual, expected)
            for actual, expected in zip(inplane, expected_axes[1:], strict=True)
        ):
            raise RuntimeError("Sectionwise pinned downsampling produced varying axes")
    if expected_axes is None:
        raise RuntimeError("No observed Nissl sections were available")
    return expected_axes


def _output_paths(scale_um: float) -> tuple[Path, Path]:
    if not float(scale_um).is_integer():
        raise RuntimeError(f"Nonintegral completed in-plane scale: {scale_um}")
    name = f"emlddmm_upstream_qc_{int(scale_um)}um"
    return POSTPROCESSED_QC / f".{name}.tmp", POSTPROCESSED_QC / name


def _ensure_output_paths_available(stage_dir: Path, final_dir: Path) -> None:
    for path in (stage_dir, final_dir):
        if path.exists():
            raise FileExistsError(f"Refusing existing QC output path: {path}")


def resolve_state(em: Any) -> ResolvedState:
    _require_equal(
        "configured EM-LDDMM pin",
        PIN_FILE.read_text(encoding="utf-8").strip(),
        EXPECTED_COMMIT,
    )
    checkpoint = _json(CHECKPOINT)
    expected_checkpoint = {
        "status": "complete_through_scale",
        "stage": "symmetric-registration",
        "completed_scale_index": EXPECTED_SCALE_INDEX,
        "completed_scale_number": EXPECTED_SCALE_NUMBER,
        "effective_inplane_um": EXPECTED_EFFECTIVE_INPLANE_UM,
        "emlddmm_commit": EXPECTED_COMMIT,
    }
    for key, expected in expected_checkpoint.items():
        _require_equal(f"registration checkpoint {key}", checkpoint.get(key), expected)
    checkpoint_dataset = _recorded_directory(
        checkpoint.get("source_dataset"), field="source_dataset"
    )
    _require_equal("registration source dataset", checkpoint_dataset, DATASET.resolve())

    scale_path = (
        CHECKPOINT.parent
        / f"registration_scale-{checkpoint['completed_scale_number']:02d}.json"
    )
    scale = _json(scale_path)
    for key in (
        "completed_scale_index",
        "completed_scale_number",
        "emlddmm_commit",
        "source_dataset",
    ):
        _require_equal(f"scale manifest {key}", scale.get(key), checkpoint.get(key))
    _require_equal("scale manifest status", scale.get("status"), "complete")
    down_i = scale.get("downI")
    down_j = scale.get("downJ")
    _require_equal("completed downI", down_i, EXPECTED_DOWN_I)
    _require_equal("completed downJ", down_j, EXPECTED_DOWN_J)
    _require_equal(
        "completed effective resolution",
        scale.get("effective_resolution_um"),
        {"I": [400.0, 400.0, 400.0], "J": [50.0, 400.0, 400.0]},
    )
    if "native_shapes" in checkpoint:
        _require_equal(
            "scale/checkpoint native shapes",
            scale.get("native_shapes"),
            checkpoint["native_shapes"],
        )

    profile_name = checkpoint.get("profile")
    profile = coarse.resolve_registration_execution(profile_name)
    _require_equal(
        "profile downI", profile["downI"][EXPECTED_SCALE_INDEX], down_i
    )
    _require_equal(
        "profile downJ", profile["downJ"][EXPECTED_SCALE_INDEX], down_j
    )

    provenance_path = _recorded_path(checkpoint.get("provenance"), field="provenance")
    provenance = _json(provenance_path)
    numerical_path = _recorded_path(checkpoint.get("numerical"), field="numerical")
    for key, expected in {
        "status": "complete_through_scale",
        "completed_scale_index": EXPECTED_SCALE_INDEX,
        "completed_scale_number": EXPECTED_SCALE_NUMBER,
        "effective_inplane_um": EXPECTED_EFFECTIVE_INPLANE_UM,
    }.items():
        _require_equal(f"through-scale provenance {key}", provenance.get(key), expected)
    _require_equal(
        "through-scale provenance commit",
        provenance.get("lineage", {}).get("emlddmm_commit"),
        EXPECTED_COMMIT,
    )
    _require_equal(
        "through-scale provenance source dataset",
        Path(provenance.get("source_dataset", "")).resolve(),
        DATASET.resolve(),
    )
    _require_equal(
        "through-scale provenance numerical path",
        Path(provenance.get("numerical", "")).resolve(),
        numerical_path,
    )
    recorded_digest = provenance.get("checksums", {}).get(str(numerical_path))
    _require_equal(
        "numerical product checksum", sha256_file(numerical_path), recorded_digest
    )

    arrays, native_xi, native_xj = _load_numerical(numerical_path, scale)
    if not MRI.is_file():
        raise RuntimeError(f"MRI input is missing: {MRI}")
    mri_provenance = _json(MRI_PROVENANCE)
    loader_xi = [np.asarray(axis) for axis in mri_physical_axes_from_provenance(mri_provenance)]
    histology = _load_histology_context()
    _assert_axes_equal("MRI loader", loader_xi, native_xi)
    _assert_axes_equal("histology loader", histology.axes, native_xj)

    native_shapes = scale.get("native_shapes", {})
    source_shape = _downsampled_shape(native_shapes.get("I", []), down_i)
    target_shape = _downsampled_shape(native_shapes.get("J", []), down_j)
    _require_equal("source QC shape", source_shape, EXPECTED_SOURCE_SHAPE)
    _require_equal("target QC shape", target_shape, EXPECTED_TARGET_SHAPE)
    if target_shape[1] != arrays["A2d"].shape[0]:
        raise RuntimeError("Target serial lattice differs from saved A2d")
    if tuple(down_j[1:]) == (1, 1) or target_shape[2:] == tuple(native_shapes["J"][2:]):
        raise RuntimeError("Refusing native/full-resolution target QC sampling")

    source_dtype = np.dtype(scale.get("state_dtypes", {}).get("A", ""))
    target_dtype = source_dtype
    _require_equal("source QC dtype", source_dtype.name, "float32")
    _require_equal("target QC dtype", target_dtype.name, "float32")
    source_bytes = int(np.prod(source_shape, dtype=np.int64)) * source_dtype.itemsize
    target_bytes = int(np.prod(target_shape, dtype=np.int64)) * target_dtype.itemsize
    state_bytes = sum(array.nbytes for array in arrays.values())
    estimate = QC_MEMORY_MULTIPLIER * (source_bytes + target_bytes) + state_bytes
    if estimate >= MEMORY_CEILING_BYTES:
        raise MemoryError(
            f"Heuristic QC estimate {estimate / 1024**3:.3f} GiB reaches 50-GiB limit"
        )
    available = _available_memory_bytes()
    if available < estimate + MEMORY_MARGIN_BYTES:
        raise MemoryError(
            f"MemAvailable {available / 1024**3:.3f} GiB is below heuristic "
            f"estimate plus 8-GiB margin {(estimate + MEMORY_MARGIN_BYTES) / 1024**3:.3f} GiB"
        )

    stage_dir, final_dir = _output_paths(EXPECTED_EFFECTIVE_INPLANE_UM)
    if not POSTPROCESSED_QC.is_dir():
        raise RuntimeError(f"QC parent directory is missing: {POSTPROCESSED_QC}")
    _ensure_output_paths_available(stage_dir, final_dir)

    _audit_weighted_section_equivalence(em, histology, down_j)
    return ResolvedState(
        checkpoint=checkpoint,
        scale_manifest=scale,
        provenance=provenance,
        numerical_path=numerical_path,
        arrays=arrays,
        native_xi=native_xi,
        native_xj=native_xj,
        histology=histology,
        mri_provenance=mri_provenance,
        down_i=list(down_i),
        down_j=list(down_j),
        source_shape=source_shape,
        target_shape=target_shape,
        source_dtype=source_dtype,
        target_dtype=target_dtype,
        estimated_memory_bytes=estimate,
        mem_available_bytes=available,
        stage_dir=stage_dir,
        final_dir=final_dir,
    )


def _preflight_report(state: ResolvedState) -> dict[str, Any]:
    source_bytes = int(np.prod(state.source_shape, dtype=np.int64)) * state.source_dtype.itemsize
    target_bytes = int(np.prod(state.target_shape, dtype=np.int64)) * state.target_dtype.itemsize
    return {
        "commit": EXPECTED_COMMIT,
        "numerical_product": str(state.numerical_path),
        "completed_scale_index": EXPECTED_SCALE_INDEX,
        "completed_scale_number": EXPECTED_SCALE_NUMBER,
        "completed_scale_um": EXPECTED_EFFECTIVE_INPLANE_UM,
        "downI": state.down_i,
        "downJ": state.down_j,
        "native_source_shape": state.scale_manifest["native_shapes"]["I"],
        "native_target_shape": state.scale_manifest["native_shapes"]["J"],
        "source_qc_shape": list(state.source_shape),
        "target_qc_shape": list(state.target_shape),
        "source_dtype": state.source_dtype.name,
        "target_dtype": state.target_dtype.name,
        "source_array_gib": source_bytes / 1024**3,
        "target_array_gib": target_bytes / 1024**3,
        "persistent_arrays_gib": (source_bytes + target_bytes) / 1024**3,
        "heuristic_qc_working_gib": state.estimated_memory_bytes / 1024**3,
        "memory_estimate_is_peak_bound": False,
        "mem_available_gib": state.mem_available_bytes / 1024**3,
        "support_weighted_downsampling": True,
        "support_shared_full_scale_maximum": True,
        "staging_directory": str(state.stage_dir),
        "output_directory": str(state.final_dir),
    }


def _build_source(em: Any, state: ResolvedState) -> ImageFacade:
    native = load_pinned_mri_image(
        em, mri_path=MRI, provenance=state.mri_provenance
    )
    _assert_axes_equal(
        "loaded MRI", [np.asarray(axis) for axis in native.x], state.native_xi
    )
    metadata = (
        native.space,
        native.name,
        native.title,
        list(native.fnames()),
    )
    native_data = np.asarray(native.data, dtype=np.float32)
    del native
    gc.collect()
    x400, data400 = em.downsample_image_domain(
        state.native_xi, native_data, state.down_i
    )
    del native_data
    gc.collect()
    return ImageFacade(
        space=metadata[0],
        name=metadata[1],
        data=np.asarray(data400),
        x=[np.asarray(axis) for axis in x400],
        title=metadata[2],
        names=metadata[3],
    )


def _build_target(em: Any, state: ResolvedState) -> ImageFacade:
    data400 = np.zeros(state.target_shape, dtype=state.target_dtype)
    expected_axes: list[np.ndarray] | None = None
    for count, index_value in enumerate(state.histology.observed, 1):
        index = int(index_value)
        image, support = _load_native_section(state.histology, index, image=True)
        assert image is not None
        section_axes = [
            state.histology.axes[0][index : index + 1],
            *state.histology.axes[1:],
        ]
        down_axes, down_image, _ = _weighted_section_downsample(
            em, section_axes, image, support, state.down_j
        )
        if down_image.shape != (3, 1, *state.target_shape[2:]):
            raise RuntimeError(
                f"Downsampled section shape changed at physical index {index}"
            )
        inplane = down_axes[1:]
        if expected_axes is None:
            expected_axes = [state.histology.axes[0], *inplane]
        elif any(
            not np.array_equal(actual, expected)
            for actual, expected in zip(inplane, expected_axes[1:], strict=True)
        ):
            raise RuntimeError("Sectionwise pinned downsampling produced varying axes")
        data400[:, index] = down_image[:, 0]
        if count % 50 == 0:
            print(
                f"weighted-downsampled {count}/{len(state.histology.observed)} sections",
                flush=True,
            )
    if expected_axes is None:
        raise RuntimeError("No observed Nissl sections were reconstructed")
    return ImageFacade(
        space="HIST_NISSL",
        name="HIST_NISSL",
        data=data400,
        x=expected_axes,
        title="slice_dataset",
        names=[str(index) for index in range(len(state.histology.rows))],
    )


def _validate_qc_inputs(
    state: ResolvedState, source: ImageFacade, target: ImageFacade
) -> None:
    _require_equal("actual source QC shape", source.data.shape, state.source_shape)
    _require_equal("actual target QC shape", target.data.shape, state.target_shape)
    _require_equal("actual source QC dtype", source.data.dtype, state.source_dtype)
    _require_equal("actual target QC dtype", target.data.dtype, state.target_dtype)
    if len(target.x[0]) != state.arrays["A2d"].shape[0]:
        raise RuntimeError("Actual target serial axis differs from saved A2d")
    if any(len(axis) != size for axis, size in zip(source.x, source.data.shape[1:], strict=True)):
        raise RuntimeError("Actual source axes and data shape differ")
    if any(len(axis) != size for axis, size in zip(target.x, target.data.shape[1:], strict=True)):
        raise RuntimeError("Actual target axes and data shape differ")


def _torch_output(state: ResolvedState) -> dict[str, Any]:
    tensors = {key: torch.from_numpy(state.arrays[key]) for key in REQUIRED_STATE}
    for key, tensor in tensors.items():
        if tensor.dtype != torch.float32:
            raise RuntimeError(f"Saved {key} did not remain float32")
        if not np.array_equal(tensor.numpy(), state.arrays[key]):
            raise RuntimeError(f"Saved {key} changed during torch conversion")
    return {
        "A": tensors["A"],
        "A2d": tensors["A2d"],
        "v": tensors["v"],
        "xv": [tensors["xv0"], tensors["xv1"], tensors["xv2"]],
    }


def _verify_upstream_outputs(stage_dir: Path) -> None:
    actual = {
        path.relative_to(stage_dir)
        for path in stage_dir.rglob("*")
        if path.is_file()
    }
    expected = set(EXPECTED_QC_OUTPUTS)
    missing = expected.difference(actual)
    unexpected = actual.difference(expected)
    if missing or unexpected:
        raise RuntimeError(
            f"Unexpected upstream QC inventory; missing={sorted(map(str, missing))}, "
            f"unexpected={sorted(map(str, unexpected))}"
        )
    empty = [str(path) for path in EXPECTED_QC_OUTPUTS if not (stage_dir / path).stat().st_size]
    if empty:
        raise RuntimeError(f"Upstream QC created empty JPEGs: {empty}")


def _write_provenance(state: ResolvedState) -> None:
    scale_checkpoint = (
        CHECKPOINT.parent / f"registration_scale-{EXPECTED_SCALE_NUMBER:02d}.json"
    ).resolve()
    through_scale_provenance = _recorded_path(
        state.checkpoint.get("provenance"), field="provenance"
    )
    report = {
        "schema": "allen-emlddmm-upstream-qc-v1",
        "status": "complete",
        **_preflight_report(state),
        "registration_checkpoint": str(CHECKPOINT.resolve()),
        "registration_checkpoint_sha256": sha256_file(CHECKPOINT),
        "scale_checkpoint": str(scale_checkpoint),
        "scale_checkpoint_sha256": sha256_file(scale_checkpoint),
        "through_scale_provenance": str(through_scale_provenance),
        "through_scale_provenance_sha256": sha256_file(through_scale_provenance),
        "numerical_product_sha256": sha256_file(state.numerical_path),
        "mri_input": str(MRI.resolve()),
        "histology_input": str(DATASET.resolve()),
        "saved_state_sha256": {
            key: _array_sha256(state.arrays[key]) for key in REQUIRED_STATE
        },
        "write_qc_outputs_called_directly": True,
        "labels_passed": False,
        "upstream_outputs": [str(path) for path in EXPECTED_QC_OUTPUTS],
    }
    (state.stage_dir / "adapter_provenance.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def run(*, dry_run: bool = False) -> Path | None:
    em = pinned_emlddmm()
    state = resolve_state(em)
    print(json.dumps(_preflight_report(state), indent=2), flush=True)
    if dry_run:
        print("Dry run complete; no bulk images allocated and no output created.", flush=True)
        return None

    source = _build_source(em, state)
    target = _build_target(em, state)
    _validate_qc_inputs(state, source, target)
    print(
        json.dumps(
            {
                "source_qc_shape": list(source.data.shape),
                "target_qc_shape": list(target.data.shape),
                "source_dtype": source.data.dtype.name,
                "target_dtype": target.data.dtype.name,
                "completed_scale_um": EXPECTED_EFFECTIVE_INPLANE_UM,
                "downI": state.down_i,
                "downJ": state.down_j,
                "heuristic_qc_working_gib": state.estimated_memory_bytes / 1024**3,
                "memory_estimate_is_peak_bound": False,
            },
            indent=2,
        ),
        flush=True,
    )

    _ensure_output_paths_available(state.stage_dir, state.final_dir)
    state.stage_dir.mkdir()
    output = _torch_output(state)
    em.write_qc_outputs(str(state.stage_dir), output, source, target)
    _verify_upstream_outputs(state.stage_dir)
    _write_provenance(state)
    if state.final_dir.exists():
        raise FileExistsError(
            f"Final QC output appeared during staging: {state.final_dir}"
        )
    os.rename(state.stage_dir, state.final_dir)
    print(f"Published pinned upstream QC: {state.final_dir}", flush=True)
    return state.final_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render isolated pinned EM-LDDMM QC for the saved 400-um state."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate provenance and memory without allocating bulk image arrays.",
    )
    args = parser.parse_args(argv)
    run(dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
