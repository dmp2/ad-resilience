"""Clean native-resolution Allen Nissl EM-LDDMM workflow.

The only coarse images in this workflow are temporary inputs to atlas-free
slice-to-neighbor estimation.  Both atlas-to-slice registrations receive native
200-um MRI and histology directly.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
import torch
from PIL import Image

from preprocess.build_allen_symmetric_histology import (
    _json,
    _project_annotation,
    _read_rows,
    _read_zarr_v3_uint32,
    _write_image_sidecar,
    _write_rows,
    bilateral_union,
)
from preprocess.run_allen_emlddmm import (
    load_pinned_mri_image, pinned_emlddmm,
)
from preprocess import run_allen_emlddmm_full_coarse_nissl as coarse


PROJECT = Path(__file__).resolve().parents[2]
NATIVE_DATASET = PROJECT / "data/derivatives/allen/specimen_708424/emlddmm_7t"
NATIVE_VIEW = NATIVE_DATASET / "inputs/views/HIST_NISSL"
MRI = PROJECT / (
    "data/derivatives/allen/specimen_708424/mri_7t_whole/"
    "T1_rot_space-MRI_7T_WHOLE_desc-header-corrected.nii"
)
MRI_PROVENANCE = PROJECT / (
    "data/derivatives/allen/specimen_708424/mri_7t_whole/mri_provenance.json"
)
INITIAL_A_SYMMETRIC = PROJECT / (
    "results/qc/allen_708424_mri7t_to_symmetric_nissl_initial_similitude.txt"
)
INITIAL_A_REPORT = PROJECT / (
    "results/qc/allen_708424_mri7t_to_symmetric_nissl_initialization_report.txt"
)
INITIAL_A_SHA256 = "3a03af802d8ddfb347801a934080e7db809960c57c00e67e8366f1ec6ffd17eb"
CLEAN_ROOT = PROJECT / "results/allen/specimen_708424/emlddmm/native-200um-clean"
HEMI_ROOT = CLEAN_ROOT / "HIST_NISSL_LEFT_to_MRI_7T_WHOLE"
SYMMETRIC_ROOT = CLEAN_ROOT / (
    "HIST_NISSL_SYMMETRIC_SECTION_ALIGNED_to_MRI_7T_WHOLE"
)
ATLAS_FREE_MANIFEST = HEMI_ROOT / "checkpoints/atlas-free.json"
CLEAN_SYMMETRIC_DATASET = PROJECT / (
    "data/derivatives/allen/specimen_708424/"
    "histology_symmetric_nissl_native_200um_section_aligned"
)
SYMMETRIC_ANNOTATION_DATASET = PROJECT / (
    "data/derivatives/allen/specimen_708424/"
    "annotations_symmetric_nissl_native_200um_section_aligned"
)
THROUGH_SCALE_DIRECTORY = "registration_through_400um"
SPACING_UM = 200.0
HEMI_PROFILE = "example-standard-sigmaR5e4-a2000-dv4000-lc188-linear-no-v"
FINAL_PROFILE = "example-standard-sigmaR5e4-a2000-dv4000-lc188"
SCALE_CHECKPOINT_SCHEMA = "allen-native-emlddmm-scale-v1"


def native_source_axes(shape_yx: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """The exact physical convention used before historical x4 reduction."""
    return tuple(
        np.arange(size, dtype=np.float64) * SPACING_UM
        - (size - 1) * SPACING_UM / 2.0
        for size in shape_yx
    )  # type: ignore[return-value]


def coarse_axes_from_native(
    em: Any, axes: tuple[np.ndarray, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """Use EM-LDDMM's domain reducer; do not assume stride-four coordinates."""
    dummy = np.zeros((1, 1, len(axes[0]), len(axes[1])), np.float32)
    down_axes, _ = em.downsample_image_domain(
        [np.asarray([0.0]), *axes], dummy, [1, 4, 4]
    )
    return np.asarray(down_axes[1]), np.asarray(down_axes[2])


@contextlib.contextmanager
def _coarse_context(dataset: Path, output: Path):
    saved = (
        coarse.DATASET, coarse.VIEW, coarse.OUTPUT, coarse.CHECKPOINTS,
        coarse.ATLAS_DIR, coarse.REG_DIR, coarse.POST_DIR, coarse.ANNOTATION_DIR,
        coarse.ANNOTATION_DATASET,
    )
    coarse.DATASET = dataset
    coarse.ANNOTATION_DATASET = (
        SYMMETRIC_ANNOTATION_DATASET
        if dataset.resolve() == CLEAN_SYMMETRIC_DATASET.resolve() else dataset
    )
    coarse.VIEW = dataset / "inputs/views/HIST_NISSL"
    coarse.configure_output_root(output)
    if dataset.resolve() == CLEAN_SYMMETRIC_DATASET.resolve():
        coarse.ANNOTATION_DIR = output / "annotations_on_native_mri"
    try:
        yield
    finally:
        (
            coarse.DATASET, coarse.VIEW, coarse.OUTPUT, coarse.CHECKPOINTS,
            coarse.ATLAS_DIR, coarse.REG_DIR, coarse.POST_DIR,
            coarse.ANNOTATION_DIR, coarse.ANNOTATION_DATASET,
        ) = saved


def _read_original(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read native prepared pixels and the established W0 validity convention."""
    if NATIVE_DATASET.resolve() not in path.resolve().parents:
        raise RuntimeError("Hemisphere pixels must come from native emlddmm_7t")
    raw = tifffile.imread(path)
    image = raw[..., :3].astype(np.float32) / 255.0
    return image.transpose(2, 0, 1), (image[..., 0] > 0).astype(np.float32)


def _native_context(
    dataset: Path,
) -> tuple[Any, list[dict[str, str]], list[dict[str, str]], list[np.ndarray], np.ndarray]:
    with _coarse_context(dataset, Path("/unused/native-context")):
        return coarse.load_context()


def load_native_stack(
    dataset: Path,
) -> tuple[Any, list[dict[str, str]], np.ndarray, list[np.ndarray], np.ndarray, np.ndarray]:
    """Load native J/W0 without any preliminary downsampling."""
    em, rows, samples, axes, observed = _native_context(dataset)
    serial_count = len(axes[0])
    if dataset.resolve() == NATIVE_DATASET.resolve():
        shape = tuple(map(len, axes[1:]))
        J = np.zeros((3, serial_count, *shape), np.float32)
        W0 = np.zeros((serial_count, *shape), np.float32)
        for index in observed:
            image, support = _read_original(NATIVE_VIEW / samples[int(index)]["sample_id"])
            if image.shape[1:] != shape:
                raise RuntimeError("Original Nissl raster and native axes differ")
            J[:, index], W0[index] = image, support
    else:
        support_root = dataset / "support/nissl"
        shape = tuple(map(len, axes[1:]))
        J = np.zeros((3, serial_count, *shape), np.float32)
        W0 = np.zeros((serial_count, *shape), np.float32)
        view = dataset / "inputs/views/HIST_NISSL"
        for index in observed:
            path = view / samples[int(index)]["sample_id"]
            image = tifffile.imread(path)[..., :3].astype(np.float32) / 255.0
            support = tifffile.imread(support_root / path.name).astype(np.float32)
            if image.shape[:2] != shape or support.shape != shape:
                raise RuntimeError("Clean symmetric image/support axes differ")
            J[:, index], W0[index] = image.transpose(2, 0, 1), support
    if not np.allclose([abs(np.diff(a).mean()) for a in axes[1:]], [200.0, 200.0]):
        raise RuntimeError("Histology is not native 200-um data")
    return em, rows, observed, [np.asarray(a) for a in axes], J, W0


def estimate_slice_initializer(dataset: Path, output: Path) -> dict[str, Any]:
    """Run cheap x4 atlas-free estimation and serialize physical-coordinate A2d."""
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite initializer root: {output}")
    em, rows, observed, xJ, J, W0 = load_native_stack(dataset)
    coarse_axes = coarse_axes_from_native(em, (xJ[1], xJ[2]))
    coarse_shape = tuple(map(len, coarse_axes))
    Jc = np.zeros((3, len(observed), *coarse_shape), np.float32)
    Wc = np.zeros((len(observed), *coarse_shape), np.float32)
    for order, index in enumerate(observed):
        Jc[:, order], Wc[order] = coarse.downsample_section(J[:, index], W0[index])
    xc = [xJ[0][observed], *coarse_axes]
    result = em.atlas_free_reconstruction(
        J=Jc, xJ=xc, W=Wc, n_steps=10, eA2d=2e4,
        downI=[1, 2, 2], downJ=[1, 2, 2],
    )
    observed_A2d = coarse.finite("atlas-free A2d", result["A2d"]).astype(np.float64)
    baseline = coarse.rigid_frame(observed_A2d)
    expanded = np.repeat(baseline[None], len(xJ[0]), axis=0)
    expanded[observed] = observed_A2d
    atlas = output / "section_alignment_atlas_free"
    checkpoints = output / "checkpoints"
    atlas.mkdir(parents=True)
    checkpoints.mkdir()
    paths = {
        "observed_A2d": atlas / "observed_A2d.npy",
        "expanded_A2d": atlas / "expanded_2846_A2d.npy",
        "observed_indices": atlas / "observed_physical_indices.npy",
        "bookkeeping_frame": atlas / "common_bookkeeping_frame.txt",
    }
    np.save(paths["observed_A2d"], observed_A2d)
    np.save(paths["expanded_A2d"], expanded)
    np.save(paths["observed_indices"], observed)
    np.savetxt(paths["bookkeeping_frame"], baseline)
    report = {
        "stage": "atlas-free", "status": "complete",
        "purpose": "slice_to_neighbor_initializer_only",
        "source_dataset": str(dataset), "pre_downsample": [1, 4, 4],
        "native_spacings_um": [50.0, 200.0, 200.0],
        "coarse_spacings_um": [
            float(abs(np.diff(xc[0]).mean())),
            float(abs(np.diff(xc[1]).mean())),
            float(abs(np.diff(xc[2]).mean())),
        ],
        "outputs": {key: str(value) for key, value in paths.items()},
        "checksums": {str(path): coarse.checksum(path) for path in paths.values()},
    }
    coarse.atomic_json(checkpoints / "atlas-free.json", report)
    return report


def _load_initializer(
    output: Path, observed: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    checkpoint = json.loads(
        (output / "checkpoints/atlas-free.json").read_text(encoding="utf-8")
    )
    if checkpoint.get("status") != "complete":
        raise RuntimeError("Slice initializer checkpoint is incomplete")
    path = output / "section_alignment_atlas_free/expanded_2846_A2d.npy"
    if checkpoint["checksums"].get(str(path)) != coarse.checksum(path):
        raise RuntimeError("Slice initializer checksum mismatch")
    A2d = coarse.finite("slice initializer A2d", np.load(path)).astype(np.float64)
    saved = np.load(output / "section_alignment_atlas_free/observed_physical_indices.npy")
    if A2d.shape != (2846, 3, 3) or not np.array_equal(saved, observed):
        raise RuntimeError("Slice initializer section identity mismatch")
    return A2d, checkpoint


def _load_native_mri(em: Any) -> tuple[Any, np.ndarray, list[np.ndarray]]:
    provenance = json.loads(MRI_PROVENANCE.read_text(encoding="utf-8"))
    image = load_pinned_mri_image(em, mri_path=MRI, provenance=provenance)
    I = np.asarray(image.data, dtype=np.float32)
    xI = [np.asarray(axis, dtype=np.float64) for axis in image.x]
    if not np.allclose(
        [abs(np.diff(axis).mean()) for axis in xI], [200.0] * 3,
        atol=1e-3, rtol=0.0,
    ):
        raise RuntimeError("MRI is not native 200-um data")
    return image, I, xI


def native_multiscale_configuration(
    *, I: np.ndarray, xI: list[np.ndarray], J: np.ndarray,
    xJ: list[np.ndarray], W0: np.ndarray, A: np.ndarray, A2d: np.ndarray,
    profile: str,
) -> dict[str, Any]:
    """Construct the direct native call and preserve the preset verbatim."""
    if profile not in {HEMI_PROFILE, FINAL_PROFILE}:
        raise RuntimeError(f"Unsupported clean native profile: {profile}")
    preset = coarse.resolve_registration_execution(profile)
    unchanged = copy.deepcopy(preset)
    config = dict(
        I=I, xI=[xI], J=J, xJ=[xJ], W0=W0,
        A=np.asarray(A).copy(), A2d=np.asarray(A2d).copy(), v=None,
        dtype=torch.float32, device="cpu", **preset,
    )
    if preset != unchanged:
        raise RuntimeError("Validated profile was mutated")
    return config


def _scale_parameters(config: dict[str, Any], scale_index: int) -> dict[str, Any]:
    """Select one scale exactly as pinned ``emlddmm_multiscale`` does."""
    params: dict[str, Any] = {}
    for key in config:
        value = config[key]
        if type(value) is list:
            params[key] = value[scale_index] if len(value) > 1 else value[0]
        else:
            params[key] = value
    if "sigmaM" not in params:
        params["sigmaM"] = np.ones(config["J"].shape[0])
    if "sigmaB" not in params:
        params["sigmaB"] = np.ones(config["J"].shape[0]) * 2.0
    if "sigmaA" not in params:
        params["sigmaA"] = np.ones(config["J"].shape[0]) * 5.0
    return params


def _multiscale_count(config: dict[str, Any]) -> int:
    downI = config["downI"]
    return len(downI) if type(downI[0]) is list else 1


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _array_sha256(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _configured_dtype_name(config: dict[str, Any]) -> str:
    dtype = config.get("dtype")
    if dtype is None:
        dtype = torch.float32
    if isinstance(dtype, str):
        dtype = {"float": torch.float32, "float32": torch.float32,
                 "float64": torch.float64}[dtype]
    return torch.empty((), dtype=dtype).numpy().dtype.name


def _scale_checkpoint_paths(
    checkpoint_dir: Path, scale_index: int,
) -> tuple[Path, Path]:
    stem = f"registration_scale-{scale_index + 1:02d}"
    return checkpoint_dir / f"{stem}.npz", checkpoint_dir / f"{stem}.json"


def _effective_resolution(
    lineage: dict[str, Any], params: dict[str, Any],
) -> dict[str, list[float]]:
    return {
        name: [
            float(spacing * factor)
            for spacing, factor in zip(
                lineage["native_spacings_um"][name], params[f"down{name}"],
                strict=True,
            )
        ]
        for name in ("I", "J")
    }


def _numpy_state(
    output: dict[str, Any], history: np.ndarray, *, slice_matching: bool,
) -> dict[str, np.ndarray]:
    state = {
        "A": coarse.finite("scale A", output["A"]),
        "v": coarse.finite("scale v", output["v"]),
        "xv0": coarse.finite("scale xv0", output["xv"][0]),
        "xv1": coarse.finite("scale xv1", output["xv"][1]),
        "xv2": coarse.finite("scale xv2", output["xv"][2]),
        "Esave": coarse.finite("scale Esave", history),
    }
    if slice_matching:
        state["A2d"] = coarse.finite("scale A2d", output["A2d"])
    return {key: np.asarray(value) for key, value in state.items()}


def _validate_scale_state(
    state: dict[str, np.ndarray], *, params: dict[str, Any], config: dict[str, Any],
) -> None:
    required = {"A", "v", "xv0", "xv1", "xv2", "Esave"}
    if params.get("slice_matching"):
        required.add("A2d")
    missing = required.difference(state)
    if missing:
        raise RuntimeError(f"scale state is missing {sorted(missing)}")
    for name in required:
        if not np.issubdtype(state[name].dtype, np.number):
            raise RuntimeError(f"scale state {name} is not numerical")
        if not np.all(np.isfinite(state[name])):
            raise RuntimeError(f"scale state {name} contains nonfinite values")
    if state["A"].shape != (4, 4):
        raise RuntimeError(f"scale A shape is {state['A'].shape}")
    velocity = state["v"]
    if velocity.ndim != 5 or velocity.shape[1] != 3:
        raise RuntimeError(f"scale velocity shape is {velocity.shape}")
    for axis, expected in enumerate(velocity.shape[2:]):
        xv = state[f"xv{axis}"]
        if xv.ndim != 1 or len(xv) != expected:
            raise RuntimeError(
                f"scale xv{axis} shape {xv.shape} does not match velocity"
            )
        if len(xv) > 1 and not np.all(np.diff(xv) > 0):
            raise RuntimeError(f"scale xv{axis} is not strictly increasing")
    if params.get("slice_matching"):
        expected = (config["J"].shape[1], 3, 3)
        if state["A2d"].shape != expected:
            raise RuntimeError(
                f"scale A2d shape is {state['A2d'].shape}, expected {expected}"
            )


def _write_scale_checkpoint(
    checkpoint_dir: Path, scale_index: int, state: dict[str, np.ndarray],
    *, config: dict[str, Any], params: dict[str, Any], lineage: dict[str, Any],
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    state_path, manifest_path = _scale_checkpoint_paths(checkpoint_dir, scale_index)
    _validate_scale_state(state, params=params, config=config)
    temporary = state_path.with_suffix(state_path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **state)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, state_path)
    manifest = {
        "schema": SCALE_CHECKPOINT_SCHEMA,
        "stage": lineage["stage"],
        "status": "complete",
        "profile_name": lineage["profile_name"],
        "completed_scale_index": scale_index,
        "completed_scale_number": scale_index + 1,
        "total_scales": _multiscale_count(config),
        "downI": params["downI"],
        "downJ": params["downJ"],
        "effective_resolution_um": _effective_resolution(lineage, params),
        "source_dataset": lineage["source_dataset"],
        "native_shapes": lineage["native_shapes"],
        "native_spacings_um": lineage["native_spacings_um"],
        "initializer_lineage_sha256": lineage["initializer_lineage_sha256"],
        "initializer_checksums": lineage["initializer_checksums"],
        "initial_A_sha256": lineage["initial_A_sha256"],
        "emlddmm_commit": lineage["emlddmm_commit"],
        "dtype": state["v"].dtype.name,
        "state_file": state_path.name,
        "state_file_sha256": coarse.checksum(state_path),
        "state_shapes": {key: list(value.shape) for key, value in state.items()},
        "state_dtypes": {key: value.dtype.name for key, value in state.items()},
    }
    coarse.atomic_json(manifest_path, manifest)


def _load_scale_checkpoint(
    checkpoint_dir: Path, scale_index: int, *, config: dict[str, Any],
    lineage: dict[str, Any],
) -> tuple[dict[str, torch.Tensor], np.ndarray]:
    state_path, manifest_path = _scale_checkpoint_paths(checkpoint_dir, scale_index)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"scale manifest cannot be read: {manifest_path}") from exc
    params = _scale_parameters(config, scale_index)
    expected = {
        "schema": SCALE_CHECKPOINT_SCHEMA,
        "stage": lineage["stage"],
        "status": "complete",
        "profile_name": lineage["profile_name"],
        "completed_scale_index": scale_index,
        "completed_scale_number": scale_index + 1,
        "total_scales": _multiscale_count(config),
        "downI": params["downI"],
        "downJ": params["downJ"],
        "effective_resolution_um": _effective_resolution(lineage, params),
        "source_dataset": lineage["source_dataset"],
        "native_shapes": lineage["native_shapes"],
        "native_spacings_um": lineage["native_spacings_um"],
        "initializer_lineage_sha256": lineage["initializer_lineage_sha256"],
        "initializer_checksums": lineage["initializer_checksums"],
        "initial_A_sha256": lineage["initial_A_sha256"],
        "emlddmm_commit": lineage["emlddmm_commit"],
        "state_file": state_path.name,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise RuntimeError(f"scale manifest {key} does not match this registration")
    if not state_path.is_file():
        raise RuntimeError(f"scale state is missing: {state_path}")
    if manifest.get("state_file_sha256") != coarse.checksum(state_path):
        raise RuntimeError(f"scale state checksum mismatch: {state_path}")
    try:
        with np.load(state_path, allow_pickle=False) as saved:
            state = {key: np.asarray(saved[key]).copy() for key in saved.files}
    except (OSError, ValueError, KeyError) as exc:
        raise RuntimeError(f"scale state cannot be read: {state_path}") from exc
    _validate_scale_state(state, params=params, config=config)
    shapes = {key: list(value.shape) for key, value in state.items()}
    dtypes = {key: value.dtype.name for key, value in state.items()}
    if manifest.get("state_shapes") != shapes or manifest.get("state_dtypes") != dtypes:
        raise RuntimeError("scale state shape/dtype inventory mismatch")
    if manifest.get("dtype") != state["v"].dtype.name:
        raise RuntimeError("scale velocity dtype mismatch")
    if state["v"].dtype.name != _configured_dtype_name(config):
        raise RuntimeError("scale velocity dtype does not match the registration dtype")
    continuation = {
        key: torch.from_numpy(state[key].copy()).cpu()
        for key in ("A", "v", "A2d") if key in state
    }
    return continuation, state["Esave"]


def _resume_scale_state(
    checkpoint_dir: Path, *, config: dict[str, Any], lineage: dict[str, Any],
) -> tuple[int, dict[str, torch.Tensor] | None, list[np.ndarray]]:
    """Return the state after the highest contiguous safe non-final scale."""
    continuation = None
    histories: list[np.ndarray] = []
    start_scale = 0
    for scale_index in range(max(0, _multiscale_count(config) - 1)):
        _, manifest_path = _scale_checkpoint_paths(checkpoint_dir, scale_index)
        if not manifest_path.exists():
            later = [
                _scale_checkpoint_paths(checkpoint_dir, later_index)[1]
                for later_index in range(scale_index + 1, _multiscale_count(config))
                if _scale_checkpoint_paths(checkpoint_dir, later_index)[1].exists()
            ]
            if later:
                print(
                    f"Ignoring noncontiguous later scale checkpoints after missing "
                    f"scale {scale_index + 1}",
                    file=sys.stderr,
                    flush=True,
                )
            break
        try:
            candidate, history = _load_scale_checkpoint(
                checkpoint_dir, scale_index, config=config, lineage=lineage
            )
        except RuntimeError as exc:
            print(
                f"Ignoring ineligible scale checkpoint {scale_index + 1}: {exc}",
                file=sys.stderr,
                flush=True,
            )
            break
        continuation = candidate
        histories.append(history)
        start_scale = scale_index + 1
    return start_scale, continuation, histories


def checkpointed_multiscale(
    em: Any, *, config: dict[str, Any], checkpoint_dir: Path,
    lineage: dict[str, Any], resume: bool, stop_after_scale: int | None = None,
) -> tuple[list[dict[str, Any]], list[np.ndarray]]:
    """Expose pinned scale boundaries while preserving its call semantics."""
    working = dict(config)
    nscales = _multiscale_count(working)
    if stop_after_scale is not None and not 1 <= stop_after_scale <= nscales:
        raise ValueError(
            f"stop_after_scale must be between 1 and {nscales}, "
            f"got {stop_after_scale}"
        )
    print(f"Found {nscales} scales")
    start_scale = 0
    histories: list[np.ndarray] = []
    if resume:
        start_scale, continuation, histories = _resume_scale_state(
            checkpoint_dir, config=working, lineage=lineage
        )
        if continuation is not None:
            working.update(continuation)
    outputs: list[dict[str, Any]] = []
    for scale_index in range(start_scale, nscales):
        if stop_after_scale is not None and scale_index >= stop_after_scale:
            break
        params = _scale_parameters(working, scale_index)
        captured: list[np.ndarray] = []
        previous_profile = sys.getprofile()
        sys.setprofile(coarse.profile_capture(em.emlddmm.__code__, captured))
        try:
            output = em.emlddmm(**params)
        finally:
            sys.setprofile(previous_profile)
        if len(captured) != 1:
            raise RuntimeError(
                f"Expected one raw Esave history for scale {scale_index + 1}, "
                f"captured {len(captured)}"
            )
        history = captured[0]
        state = _numpy_state(
            output, history, slice_matching=bool(params.get("slice_matching"))
        )
        _write_scale_checkpoint(
            checkpoint_dir, scale_index, state,
            config=config, params=params, lineage=lineage,
        )
        outputs.append(output)
        histories.append(history)
        working["A"] = output["A"]
        working["v"] = output["v"]
        if params.get("slice_matching"):
            working["A2d"] = output["A2d"]
    return outputs, histories


def _load_validated_symmetric_initial_affine(
    dataset: Path,
    *,
    affine_path: Path = INITIAL_A_SYMMETRIC,
    report_path: Path = INITIAL_A_REPORT,
    mri_provenance_path: Path = MRI_PROVENANCE,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Validate that the pinned similitude targets the corrected symmetric frame."""
    if dataset.resolve() != CLEAN_SYMMETRIC_DATASET.resolve():
        raise RuntimeError("Symmetric initial A is restricted to the corrected derivative")
    symmetry = json.loads(
        (dataset / "metadata/symmetry.json").read_text(encoding="utf-8")
    )
    transforms = dataset / "metadata/transforms"
    axes = [
        np.load(transforms / name)
        for name in (
            "serial_axis_um.npy",
            "row_axis_um.npy",
            "symmetric_lr_axis_um.npy",
        )
    ]
    expected_axes = (
        (2846, -71125.0, 71125.0, 50.0),
        (522, -52100.0, 52100.0, 200.0),
        (730, -72900.0, 72900.0, 200.0),
    )
    for axis, (size, first, last, spacing) in zip(
        axes, expected_axes, strict=True
    ):
        if (
            axis.shape != (size,)
            or not np.isclose(axis[0], first, atol=1e-8, rtol=0.0)
            or not np.isclose(axis[-1], last, atol=1e-8, rtol=0.0)
            or not np.allclose(np.diff(axis), spacing, atol=1e-8, rtol=0.0)
        ):
            raise RuntimeError(
                "Corrected symmetric axes differ from the initial-A coordinate frame"
            )
    if (
        symmetry.get("bilateral_shape_yx") != [len(axes[1]), len(axes[2])]
        or symmetry.get("reflection_plane_um") != 0.0
        or symmetry.get("medial_centers_um") != [-100.0, 100.0]
        or not np.isclose(axes[2][len(axes[2]) // 2 - 1], -100.0)
        or not np.isclose(axes[2][len(axes[2]) // 2], 100.0)
    ):
        raise RuntimeError("Corrected symmetric reflection geometry is incompatible")

    if coarse.checksum(affine_path) != INITIAL_A_SHA256:
        raise RuntimeError("Symmetric initial-A checksum changed")
    affine = coarse.finite("symmetric initial A", np.loadtxt(affine_path)).astype(
        np.float64
    )
    expected_linear = np.array(
        [
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
            [1.0, 0.0, 0.0],
        ]
    )
    if (
        affine.shape != (4, 4)
        or not np.array_equal(affine[:3, :3], expected_linear)
        or not np.array_equal(affine[3], [0.0, 0.0, 0.0, 1.0])
    ):
        raise RuntimeError("Symmetric initial A has the wrong direction convention")

    mri_provenance = json.loads(
        mri_provenance_path.read_text(encoding="utf-8")
    )
    mri_center_um = np.asarray(
        mri_provenance["physical_center_mm"], dtype=np.float64
    ) * 1000.0
    mapped_center = affine @ np.r_[mri_center_um, 1.0]
    if not np.allclose(
        mapped_center, [-100.0, -100.0, 100.0, 1.0],
        atol=2e-3,
        rtol=0.0,
    ):
        raise RuntimeError("Symmetric initial A no longer maps the MRI center as audited")

    report = report_path.read_text(encoding="utf-8")
    required_report_lines = (
        "HIST axis 0: -71125.000000 .. 71125.000000; increasing",
        "HIST axis 1: -52100.000000 .. 52100.000000; increasing",
        "HIST axis 2: -72900.000000 .. 72900.000000; increasing",
        "A maps pinned-loader MRI physical coordinates in um to registered histology",
    )
    if any(line not in report for line in required_report_lines):
        raise RuntimeError("Initial-A audit report does not describe the corrected frame")

    audit = {
        "status": "compatible",
        "source": str(affine_path),
        "sha256": INITIAL_A_SHA256,
        "audit_report": str(report_path),
        "direction": "MRI physical um to histology [serial,row,LR] physical um",
        "histology_axis_sha256": [_array_sha256(axis) for axis in axes],
        "histology_axis_lengths": [len(axis) for axis in axes],
        "histology_axis_ranges_um": [
            [float(axis[0]), float(axis[-1])] for axis in axes
        ],
        "histology_spacings_um": [
            float(np.diff(axis).mean()) for axis in axes
        ],
        "reflection_plane_x_um": 0.0,
        "mapped_mri_center_histology_um": mapped_center[:3].tolist(),
        "compatibility_basis": (
            "exact audited axis ranges/spacings, signed anatomical permutation, "
            "affine checksum, zero-centered LR seam, and MRI-center mapping"
        ),
    }
    return affine, audit


def _identity_section_initializer(
    dataset: Path,
    observed: np.ndarray,
    xJ: list[np.ndarray],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Use explicit identity A2d because corrected pixels are already aligned."""
    if dataset.resolve() != CLEAN_SYMMETRIC_DATASET.resolve():
        raise RuntimeError("Identity section initialization requires corrected symmetry")
    transforms = dataset / "metadata/transforms"
    saved_axes = [
        np.load(transforms / name)
        for name in (
            "serial_axis_um.npy",
            "row_axis_um.npy",
            "symmetric_lr_axis_um.npy",
        )
    ]
    if len(xJ) != 3 or any(
        not np.array_equal(np.asarray(actual), saved)
        for actual, saved in zip(xJ, saved_axes, strict=True)
    ):
        raise RuntimeError("Corrected symmetric registration axes changed")
    provenance_path = transforms / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if (
        provenance.get("schema") != "allen-native-symmetric-section-aligned-v1"
        or provenance.get("materialization", {}).get("native_pixels_upsampled")
        is not False
        or provenance.get("materialization", {}).get("serial_coordinates_modified")
        is not False
    ):
        raise RuntimeError("Corrected section-materialization provenance is invalid")
    identities = np.asarray(observed, dtype=np.int64)
    if identities.ndim != 1 or np.any(identities < 0) or np.any(
        identities >= len(xJ[0])
    ):
        raise RuntimeError("Corrected observed indices are invalid")
    A2d = np.repeat(
        np.eye(3, dtype=np.float64)[None], len(xJ[0]), axis=0
    )
    initializer = {
        "status": "complete",
        "type": "explicit_identity_per_physical_serial_position",
        "purpose": "already_materialized_section_stack",
        "source_dataset": str(dataset),
        "source_provenance": str(provenance_path),
        "source_provenance_sha256": coarse.checksum(provenance_path),
        "physical_serial_positions": len(xJ[0]),
        "observed_sections": len(identities),
        "A2d_shape": list(A2d.shape),
        "A2d_sha256": _array_sha256(A2d),
        "pinned_semantics": (
            "equivalent to pinned emlddmm A2d=None initialization when "
            "slice_matching=True; explicit identity is recorded for auditability"
        ),
        "left_atlas_free_A2d_reapplied": False,
        "checksums": {
            str(provenance_path): coarse.checksum(provenance_path),
        },
    }
    return A2d, initializer


def _validate_loaded_axes_against_initial_a(
    xJ: list[np.ndarray], initial_A_validation: dict[str, Any] | None,
) -> None:
    if initial_A_validation is None:
        return
    actual = [_array_sha256(axis) for axis in xJ]
    if actual != initial_A_validation.get("histology_axis_sha256"):
        raise RuntimeError("Loaded histology axes differ from initial-A validation")


def _serialize_through_scale_product(
    em: Any,
    *,
    output: Path,
    final: dict[str, Any],
    histories: list[np.ndarray],
    mri: Any,
    hist: Any,
    xI: list[np.ndarray],
    xJ: list[np.ndarray],
    observed: np.ndarray,
    config: dict[str, Any],
    lineage: dict[str, Any],
    running: dict[str, Any],
    completed_scale_number: int,
) -> dict[str, Any]:
    """Publish a normal transform/numerical product for an intentional coarse stop."""
    total_scales = _multiscale_count(config)
    if not 1 <= completed_scale_number < total_scales:
        raise RuntimeError("Through-scale publication requires a non-final scale")
    product = output / THROUGH_SCALE_DIRECTORY
    if product.exists():
        raise FileExistsError(f"Refusing to overwrite through-scale product: {product}")
    product.mkdir(parents=True)
    em.write_transform_outputs(str(product), final, mri, hist)

    A = coarse.finite("through-scale A", final["A"])
    A2d = coarse.finite("through-scale A2d", final["A2d"])
    v = coarse.finite("through-scale v", final["v"])
    xv = [
        coarse.finite(f"through-scale xv{axis}", value)
        for axis, value in enumerate(final["xv"])
    ]
    numerical = product / "numerical_outputs.npz"
    np.savez_compressed(
        numerical,
        A=A,
        A2d=A2d,
        v=v,
        xv0=xv[0],
        xv1=xv[1],
        xv2=xv[2],
        xI0=xI[0],
        xI1=xI[1],
        xI2=xI[2],
        xJ0=xJ[0],
        xJ1=xJ[1],
        xJ2=xJ[2],
        observed=observed,
    )
    energy_paths = []
    for level, history in enumerate(histories, 1):
        path = product / f"raw_Esave_level-{level}.npy"
        np.save(path, coarse.finite(f"raw Esave level {level}", history))
        energy_paths.append(str(path))
    params = _scale_parameters(config, completed_scale_number - 1)
    effective = _effective_resolution(lineage, params)
    provenance_path = product / "provenance.json"
    provenance = {
        "schema": "allen-native-registration-through-scale-v1",
        "status": "complete_through_scale",
        "source_dataset": lineage["source_dataset"],
        "profile_name": lineage["profile_name"],
        "completed_scale_number": completed_scale_number,
        "completed_scale_index": completed_scale_number - 1,
        "total_configured_scales": total_scales,
        "native_input_inplane_um": 200.0,
        "native_input_spacings_um": lineage["native_spacings_um"],
        "completed_effective_resolution_um": effective,
        "effective_inplane_um": float(effective["J"][1]),
        "configured_final_scale_executed": False,
        "full_configured_profile_preserved": True,
        "numerical": str(numerical),
        "transform_output_root": str(product),
        "raw_Esave": energy_paths,
        "effective_match_weight": None,
        "effective_match_weight_note": (
            "not returned at scale 2 because the unchanged profile has "
            "full_outputs=False for configured scale index 1"
        ),
        "initial_A_validation": running.get("initial_A_validation"),
        "section_initializer": running["initializer"],
        "lineage": lineage,
        "checksums": {
            str(numerical): coarse.checksum(numerical),
            **{path: coarse.checksum(Path(path)) for path in energy_paths},
        },
    }
    coarse.atomic_json(provenance_path, provenance)
    return {
        **running,
        "status": "complete_through_scale",
        "completed_scale_number": completed_scale_number,
        "completed_scale_index": completed_scale_number - 1,
        "total_configured_scales": total_scales,
        "native_input_inplane_um": 200.0,
        "effective_inplane_um": float(effective["J"][1]),
        "final_configured_scale_executed": False,
        "through_scale_product": str(product),
        "numerical": str(numerical),
        "transform_output_root": str(product),
        "raw_Esave": energy_paths,
        "provenance": str(provenance_path),
        "effective_match_weight": None,
    }


def _registration_lineage(
    *, stage: str, profile: str, dataset: Path, I: np.ndarray, J: np.ndarray,
    W0: np.ndarray, initial_A: np.ndarray, initializer: dict[str, Any],
) -> dict[str, Any]:
    return {
        "stage": stage,
        "profile_name": profile,
        "source_dataset": str(dataset),
        "native_shapes": {
            "I": list(I.shape), "J": list(J.shape), "W0": list(W0.shape),
        },
        "native_spacings_um": {
            "I": [200.0, 200.0, 200.0],
            "J": [50.0, 200.0, 200.0],
        },
        "initializer_lineage_sha256": _json_sha256(initializer),
        "initializer_checksums": initializer.get("checksums", {}),
        "initial_A_sha256": _array_sha256(initial_A),
        "emlddmm_commit": (
            PROJECT / "configs/emlddmm-upstream-commit.txt"
        ).read_text(encoding="utf-8").strip(),
    }


def _prepare_registration_output(
    output: Path, *, restart_interrupted: bool,
) -> None:
    """Protect completed registration and optionally clear an interrupted one."""
    reg = output / "registration"
    checkpoint = output / "checkpoints/registration.json"
    numerical = reg / "full_resolution_numerical_outputs.npz"

    if numerical.exists():
        raise FileExistsError(
            f"Refusing to overwrite completed registration: {numerical} exists"
        )

    checkpoint_status = None
    if checkpoint.exists():
        try:
            checkpoint_status = json.loads(
                checkpoint.read_text(encoding="utf-8")
            ).get("status")
        except (json.JSONDecodeError, OSError, AttributeError):
            checkpoint_status = "unreadable"
    if checkpoint_status == "complete":
        raise FileExistsError(
            f"Refusing to overwrite completed registration: {checkpoint} "
            "has status='complete'"
        )

    registration_files_present = reg.exists() and any(reg.iterdir())
    scale_files_present = any(
        (output / "checkpoints").glob("registration_scale-*")
    )
    interrupted = (
        checkpoint.exists() or registration_files_present or scale_files_present
    )
    if interrupted and not restart_interrupted:
        detail = (
            f"checkpoint status={checkpoint_status!r}"
            if checkpoint.exists()
            else "partial registration files are present"
        )
        raise FileExistsError(
            f"Interrupted registration detected at {output} ({detail}); "
            "rerun with --restart-interrupted to remove only incomplete "
            "registration-stage outputs"
        )

    if interrupted:
        if reg.exists():
            shutil.rmtree(reg)
        reg.mkdir(parents=True)
        checkpoint.unlink(missing_ok=True)
        for temporary in (output / "checkpoints").glob("registration_scale-*.tmp"):
            temporary.unlink()


def _registration(
    dataset: Path,
    initializer_root: Path | None,
    output: Path,
    *,
    stage: str,
    profile: str,
    initial_A: np.ndarray,
    a2d_initialization: str = "atlas_free",
    initial_A_validation: dict[str, Any] | None = None,
    restart_interrupted: bool = False,
    stop_after_scale: int | None = None,
) -> dict[str, Any]:
    _prepare_registration_output(
        output, restart_interrupted=restart_interrupted
    )
    em, rows, observed, xJ, J, W0 = load_native_stack(dataset)
    _validate_loaded_axes_against_initial_a(xJ, initial_A_validation)
    if a2d_initialization == "identity":
        if initializer_root is not None:
            raise RuntimeError("Identity A2d initialization must not use atlas-free")
        A2d, initializer = _identity_section_initializer(
            dataset, observed, xJ
        )
    elif a2d_initialization == "atlas_free":
        if initializer_root is None:
            raise RuntimeError("Atlas-free A2d initialization requires its root")
        A2d, initializer = _load_initializer(initializer_root, observed)
    else:
        raise ValueError(f"Unknown A2d initialization: {a2d_initialization}")

    mri, I, xI = _load_native_mri(em)
    config = native_multiscale_configuration(
        I=I, xI=xI, J=J, xJ=xJ, W0=W0,
        A=initial_A, A2d=A2d, profile=profile,
    )
    lineage = _registration_lineage(
        stage=stage, profile=profile, dataset=dataset, I=I, J=J, W0=W0,
        initial_A=np.asarray(config["A"]), initializer=initializer,
    )
    reg = output / "registration"
    checkpoints = output / "checkpoints"
    reg.mkdir(parents=True, exist_ok=True)
    checkpoints.mkdir(exist_ok=True)
    running = {
        "stage": stage,
        "status": "running",
        "profile": profile,
        "source_dataset": str(dataset),
        "initializer": initializer,
        "initial_A_validation": initial_A_validation,
        "requested_stop_after_scale": stop_after_scale,
        "external_pre_downsample": {
            "I": [1, 1, 1],
            "J": [1, 1, 1],
            "W0": [1, 1, 1],
        },
        "native_shapes": {
            "I": list(I.shape),
            "J": list(J.shape),
            "W0": list(W0.shape),
        },
        "native_spacings_um": {
            "I": [200.0] * 3,
            "J": [50.0, 200.0, 200.0],
        },
        "effective_inplane_um": [800.0, 400.0, 200.0],
        "initial_velocity": "implicit_zero",
        "scale_checkpoint_schema": SCALE_CHECKPOINT_SCHEMA,
        "initializer_lineage_sha256": lineage["initializer_lineage_sha256"],
        "initial_A_sha256": lineage["initial_A_sha256"],
        "emlddmm_commit": lineage["emlddmm_commit"],
    }
    coarse.atomic_json(checkpoints / "registration.json", running)
    outputs, histories = checkpointed_multiscale(
        em,
        config=config,
        checkpoint_dir=checkpoints,
        lineage=lineage,
        resume=restart_interrupted,
        stop_after_scale=stop_after_scale,
    )
    total_scales = _multiscale_count(config)
    expected_completed = (
        total_scales if stop_after_scale is None else stop_after_scale
    )
    if not outputs or len(histories) != expected_completed:
        raise RuntimeError(
            "Missing newly executed scale output or raw Esave histories: "
            f"expected {expected_completed}, got outputs={len(outputs)}, "
            f"histories={len(histories)}"
        )
    final = outputs[-1]
    hist = coarse.LightImage(
        "HIST_NISSL",
        "HIST_NISSL",
        J,
        xJ,
        "slice_dataset",
        [str(index) for index in range(len(rows))],
    )

    if expected_completed < total_scales:
        done = _serialize_through_scale_product(
            em,
            output=output,
            final=final,
            histories=histories,
            mri=mri,
            hist=hist,
            xI=xI,
            xJ=xJ,
            observed=observed,
            config=config,
            lineage=lineage,
            running=running,
            completed_scale_number=expected_completed,
        )
        coarse.atomic_json(checkpoints / "registration.json", done)
        return done

    A_final = coarse.finite("final A", final["A"])
    A2d_final = coarse.finite("final A2d", final["A2d"])
    v = coarse.finite("final v", final["v"])
    xv = [coarse.finite(f"xv{i}", value) for i, value in enumerate(final["xv"])]
    coarse._save_final_effective_match_weight(
        final, observed, tuple(W0.shape),
        reg / "final_observed_effective_match_weight.npy",
    )
    em.write_transform_outputs(str(reg), final, mri, hist)
    numerical = reg / "full_resolution_numerical_outputs.npz"
    np.savez_compressed(
        numerical, A=A_final, A2d=A2d_final, v=v,
        xv0=xv[0], xv1=xv[1], xv2=xv[2],
        xI0=xI[0], xI1=xI[1], xI2=xI[2],
        xJ0=xJ[0], xJ1=xJ[1], xJ2=xJ[2], observed=observed,
    )
    energy_paths = []
    for level, history in enumerate(histories, 1):
        path = reg / f"raw_Esave_level-{level}.npy"
        np.save(path, history)
        energy_paths.append(str(path))
    done = {
        **running, "status": "complete", "numerical": str(numerical),
        "raw_Esave": energy_paths,
        "final_effective_match_weight": str(
            reg / "final_observed_effective_match_weight.npy"
        ),
    }
    coarse.atomic_json(checkpoints / "registration.json", done)
    return done


def hemisphere_registration(
    *, restart_interrupted: bool = False,
) -> dict[str, Any]:
    return _registration(
        NATIVE_DATASET, HEMI_ROOT, HEMI_ROOT,
        stage="hemi-registration", profile=HEMI_PROFILE,
        initial_A=np.loadtxt(INITIAL_A_SYMMETRIC),
        restart_interrupted=restart_interrupted,
    )


def symmetric_lr_axis(half_width: int, spacing_um: float) -> np.ndarray:
    """Return the historical between-column axis for a complete unilateral raster."""
    if half_width < 1 or spacing_um <= 0.0:
        raise ValueError("Symmetric histology requires a nonempty positive-spacing grid")
    return (
        np.arange(2 * half_width, dtype=np.float64) * spacing_um
        - (half_width - 0.5) * spacing_um
    )


def reflect_observed_half(
    image: np.ndarray, support: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the historical column flip/union to aligned RGB and validity."""
    aligned = np.asarray(image)
    validity = np.asarray(support)
    if aligned.ndim != 3 or validity.shape != aligned.shape[1:]:
        raise ValueError(
            f"Aligned image/support shapes differ: {aligned.shape} versus "
            f"{validity.shape}"
        )
    reflected_rgb = bilateral_union(np.moveaxis(aligned, 0, -1))
    return np.moveaxis(reflected_rgb, -1, 0), bilateral_union(validity)


def _resolve_atlas_free_output(manifest_path: Path, recorded: str) -> Path:
    path = Path(recorded)
    if not path.is_absolute():
        path = manifest_path.parent.parent / path
    return path.resolve()


def _load_completed_atlas_free_manifest(
    manifest_path: Path,
) -> tuple[dict[str, Any], dict[str, Path], dict[str, str]]:
    """Resolve and checksum every transform output recorded by atlas-free.json."""
    checkpoint = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        checkpoint.get("stage") != "atlas-free"
        or checkpoint.get("status") != "complete"
    ):
        raise RuntimeError("Atlas-free section alignment checkpoint is incomplete")
    recorded_outputs = checkpoint.get("outputs")
    recorded_checksums = checkpoint.get("checksums")
    if not isinstance(recorded_outputs, dict) or not isinstance(
        recorded_checksums, dict
    ):
        raise RuntimeError("Atlas-free checkpoint has no output/checksum inventory")
    required = {"expanded_A2d", "observed_indices", "bookkeeping_frame"}
    if not required.issubset(recorded_outputs):
        raise RuntimeError("Atlas-free checkpoint is missing materialization outputs")

    paths: dict[str, Path] = {}
    checksums: dict[str, str] = {}
    for key, recorded in recorded_outputs.items():
        if not isinstance(recorded, str):
            raise RuntimeError(f"Atlas-free output path {key!r} is invalid")
        path = _resolve_atlas_free_output(manifest_path, recorded)
        if not path.is_file():
            raise FileNotFoundError(f"Atlas-free output is missing: {path}")
        expected = recorded_checksums.get(recorded)
        if expected is None:
            expected = recorded_checksums.get(str(path))
        actual = coarse.checksum(path)
        if expected != actual:
            raise RuntimeError(f"Atlas-free output checksum mismatch: {path}")
        paths[str(key)] = path
        checksums[str(key)] = actual
    return checkpoint, paths, checksums


def _common_frame_residuals(
    A2d: np.ndarray,
    observed: np.ndarray,
    common_frame_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove the shared unsupported-row bookkeeping frame exactly once."""
    transforms = coarse.finite("atlas-free expanded A2d", A2d).astype(np.float64)
    identities = np.asarray(observed, dtype=np.int64)
    if transforms.ndim != 3 or transforms.shape[1:] != (3, 3):
        raise RuntimeError("Atlas-free expanded A2d shape is invalid")
    if (
        identities.ndim != 1
        or np.any(identities < 0)
        or np.any(identities >= len(transforms))
        or len(np.unique(identities)) != len(identities)
    ):
        raise RuntimeError("Atlas-free observed physical indices are invalid")
    unsupported = np.ones(len(transforms), dtype=bool)
    unsupported[identities] = False
    if not np.any(unsupported):
        raise RuntimeError("Atlas-free A2d has no unsupported bookkeeping rows")
    baseline = A2d[unsupported][0]
    if not np.array_equal(
        A2d[unsupported], np.broadcast_to(baseline, A2d[unsupported].shape)
    ):
        raise RuntimeError("Atlas-free unsupported rows do not share one frame")
    recorded_baseline = coarse.finite(
        "atlas-free common bookkeeping frame", np.loadtxt(common_frame_path)
    ).astype(np.float64)
    if recorded_baseline.shape != (3, 3) or not np.array_equal(
        recorded_baseline, baseline
    ):
        raise RuntimeError("Recorded atlas-free bookkeeping frame changed")
    residual = np.linalg.inv(baseline)[None] @ A2d
    return np.asarray(baseline, dtype=np.float64), residual


def _validated_native_geometry(
    rows: list[dict[str, str]],
    observed: np.ndarray,
    xJ: list[np.ndarray],
    J: np.ndarray,
    W0: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    """Validate the native physical lattice without assuming dataset dimensions."""
    axes = [np.asarray(axis, dtype=np.float64) for axis in xJ]
    if len(axes) != 3 or any(axis.ndim != 1 or len(axis) < 2 for axis in axes):
        raise RuntimeError("Native histology axes are invalid")
    serial_count, row_count, column_count = map(len, axes)
    if (
        len(rows) != serial_count
        or J.shape != (3, serial_count, row_count, column_count)
        or W0.shape != (serial_count, row_count, column_count)
    ):
        raise RuntimeError("Native histology arrays, axes, and rows differ")
    identities = np.asarray(observed, dtype=np.int64)
    if (
        identities.ndim != 1
        or np.any(identities < 0)
        or np.any(identities >= serial_count)
        or len(np.unique(identities)) != len(identities)
    ):
        raise RuntimeError("Native observed section identities are invalid")
    differences = [np.diff(axis) for axis in axes]
    if any(
        not np.allclose(delta, delta[0], atol=1e-8, rtol=0.0)
        or delta[0] <= 0.0
        for delta in differences
    ):
        raise RuntimeError("Native histology axes are not uniform and increasing")
    serial_spacing, row_spacing, column_spacing = (
        float(delta[0]) for delta in differences
    )
    if not np.allclose(
        [serial_spacing, row_spacing, column_spacing],
        [50.0, SPACING_UM, SPACING_UM],
        atol=1e-8,
        rtol=0.0,
    ):
        raise RuntimeError("Native histology physical spacing changed")
    return axes[1], axes[2], serial_spacing, row_spacing, column_spacing


def _copy_transform_provenance(
    metadata: Path,
    *,
    manifest_path: Path,
    manifest: dict[str, Any],
    source_paths: dict[str, Path],
    source_checksums: dict[str, str],
    baseline: np.ndarray,
    observed: np.ndarray,
    xJ: list[np.ndarray],
    symmetric_axis: np.ndarray,
) -> Path:
    """Make the derivative independently intelligible if results/ is cleaned."""
    transforms = metadata / "transforms"
    transforms.mkdir()
    copied: dict[str, Path] = {}
    copy_names = {
        "expanded_A2d": "atlas_free_expanded_A2d.npy",
        "observed_A2d": "atlas_free_observed_A2d.npy",
        "observed_indices": "observed_physical_indices.npy",
        "bookkeeping_frame": "atlas_free_common_bookkeeping_frame.txt",
    }
    for key, name in copy_names.items():
        if key in source_paths:
            destination = transforms / name
            shutil.copy2(source_paths[key], destination)
            copied[key] = destination
    manifest_copy = transforms / "atlas_free_manifest.json"
    shutil.copy2(manifest_path, manifest_copy)
    copied["manifest"] = manifest_copy

    axis_paths = {
        "serial_axis_um": transforms / "serial_axis_um.npy",
        "row_axis_um": transforms / "row_axis_um.npy",
        "left_lr_axis_um": transforms / "left_lr_axis_um.npy",
        "symmetric_lr_axis_um": transforms / "symmetric_lr_axis_um.npy",
    }
    for path, axis in zip(
        axis_paths.values(), [xJ[0], xJ[1], xJ[2], symmetric_axis]
    ):
        np.save(path, np.asarray(axis, dtype=np.float64))

    derivative_checksums = {
        path.relative_to(metadata.parent).as_posix(): coarse.checksum(path)
        for path in [*copied.values(), *axis_paths.values()]
    }
    provenance_path = transforms / "provenance.json"
    _json(provenance_path, {
        "schema": "allen-native-symmetric-section-aligned-v1",
        "native_source_dataset": str(NATIVE_DATASET),
        "atlas_free_manifest_source": str(manifest_path),
        "atlas_free_manifest_sha256": coarse.checksum(manifest_path),
        "atlas_free_manifest": manifest,
        "atlas_free_source_outputs": {
            key: str(path) for key, path in source_paths.items()
        },
        "atlas_free_source_checksums_sha256": source_checksums,
        "derivative_transform_checksums_sha256": derivative_checksums,
        "observed_physical_indices_count": int(len(observed)),
        "physical_serial_positions": int(len(xJ[0])),
        "common_frame_normalization": {
            "baseline_selection": "expanded_A2d[unsupported][0]",
            "unsupported_definition": (
                "all physical serial indices absent from observed_physical_indices"
            ),
            "unsupported_rows_share_baseline": "exact np.array_equal",
            "formula": "residual[index] = inv(baseline) @ expanded_A2d[index]",
            "baseline_matrix": np.asarray(baseline).tolist(),
            "coordinate_units": "micrometers",
            "translation_rescaling": "none",
        },
        "materialization": {
            "operation": "coarse._warp_saved_section",
            "target_row_axis": "row_axis_um.npy",
            "target_column_axis": "left_lr_axis_um.npy",
            "source_row_axis": "row_axis_um.npy",
            "source_column_axis": "left_lr_axis_um.npy",
            "native_pixels_upsampled": False,
            "serial_coordinates_modified": False,
        },
        "symmetry": {
            "channel_last_formula": (
                "concatenate((flip(unilateral, axis=1), unilateral), axis=1)"
            ),
            "channel_first_lr_axis": 2,
            "support_operation_identical": True,
            "high_column_half": "complete pixel-identical original unilateral",
            "low_column_half": "exact LR reversal of complete unilateral",
            "medial_column_duplicated": False,
            "reflection_plane_x_um": 0.0,
        },
    })
    return provenance_path


def construct_symmetric_histology(
    output: Path = CLEAN_SYMMETRIC_DATASET,
) -> dict[str, Any]:
    """Materialize atlas-free aligned native left sections, then reflect exactly."""
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite clean derivative: {output}")
    manifest_path = ATLAS_FREE_MANIFEST
    manifest, transform_paths, transform_checksums = (
        _load_completed_atlas_free_manifest(manifest_path)
    )
    source_dataset = Path(str(manifest.get("source_dataset", ""))).resolve()
    if source_dataset != NATIVE_DATASET.resolve():
        raise RuntimeError("Atlas-free checkpoint source is not the native Nissl stack")

    em, rows, observed, xJ, J, W0 = load_native_stack(NATIVE_DATASET)
    row_axis, observed_axis, serial_spacing, row_spacing, column_spacing = (
        _validated_native_geometry(rows, observed, xJ, J, W0)
    )
    A2d = np.load(transform_paths["expanded_A2d"])
    saved_observed = np.asarray(
        np.load(transform_paths["observed_indices"]), dtype=np.int64
    )
    if A2d.shape != (len(xJ[0]), 3, 3):
        raise RuntimeError("Atlas-free expanded A2d does not span the native lattice")
    if not np.array_equal(saved_observed, observed):
        raise RuntimeError("Atlas-free observed identities differ from native stack")
    baseline, residual = _common_frame_residuals(
        A2d, saved_observed, transform_paths["bookkeeping_frame"]
    )

    symmetric_axis = symmetric_lr_axis(len(observed_axis), column_spacing)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        view = stage / "inputs/views/HIST_NISSL"
        support_dir = stage / "support/nissl"
        metadata = stage / "metadata"
        view.mkdir(parents=True)
        support_dir.mkdir(parents=True)
        metadata.mkdir()
        provenance_path = _copy_transform_provenance(
            metadata,
            manifest_path=manifest_path,
            manifest=manifest,
            source_paths=transform_paths,
            source_checksums=transform_checksums,
            baseline=baseline,
            observed=observed,
            xJ=xJ,
            symmetric_axis=symmetric_axis,
        )
        output_rows = [dict(row) for row in rows]
        present_indices = set(map(int, observed))
        samples = []
        for index, row in enumerate(output_rows):
            present = index in present_indices
            name = f"allen_708424_nissl_{int(row['allen_section_number']):04d}.tif"
            samples.append({
                "sample_id": name, "participant_id": "708424",
                "species": "human", "status": "present" if present else "absent",
            })
            if not present:
                continue
            warped, support = coarse._warp_saved_section(
                J[:, index],
                W0[index],
                residual[index],
                row_axis,
                observed_axis,
                source_row_um=xJ[1],
                source_column_um=xJ[2],
            )
            symmetric, symmetric_support = reflect_observed_half(warped, support)
            width = len(observed_axis)
            expected_shape = (len(row_axis), 2 * width)
            if (
                warped.shape != (3, len(row_axis), width)
                or support.shape != (len(row_axis), width)
                or symmetric.shape[1:] != expected_shape
                or symmetric_support.shape != expected_shape
                or not np.array_equal(symmetric[:, :, :width], warped[:, :, ::-1])
                or not np.array_equal(symmetric[:, :, width:], warped)
                or not np.array_equal(symmetric_support[:, :width], support[:, ::-1])
                or not np.array_equal(symmetric_support[:, width:], support)
            ):
                raise RuntimeError(
                    "Symmetry did not preserve the complete aligned unilateral raster"
                )
            rgb = np.moveaxis(
                np.rint(np.clip(symmetric, 0.0, 1.0) * 255.0).astype(np.uint8),
                0, -1,
            )
            Image.fromarray(rgb).save(
                view / name, format="TIFF", compression="tiff_deflate"
            )
            tifffile.imwrite(support_dir / name, symmetric_support.astype(np.float32))
            _write_image_sidecar(
                view / name,
                shape_yx=rgb.shape[:2],
                origin_xy_um=(float(symmetric_axis[0]), float(row_axis[0])),
                z_um=float(row["serial_z_center_mm"]) * 1000.0,
                pixel_size_um=column_spacing,
            )
            row["prepared_relative_path"] = (
                Path("inputs/views/HIST_NISSL") / name
            ).as_posix()

        _write_rows(
            metadata / "physical_sections.tsv",
            output_rows,
            list(output_rows[0]),
        )
        _write_rows(
            view / "samples.tsv",
            samples,
            ["sample_id", "participant_id", "species", "status"],
        )
        _json(metadata / "loader_canvas_audit.json", {
            "accepted_canvas_shape_yx": [len(row_axis), len(symmetric_axis)],
            "target_spacing_um": column_spacing,
            "global_translation_xy_um": [
                float(symmetric_axis[0]), float(row_axis[0])
            ],
            "preserve_source_grid": True,
        })
        _json(metadata / "symmetry.json", {
            "pixel_size_um": column_spacing,
            "serial_spacing_um": serial_spacing,
            "unilateral_shape_yx": [len(row_axis), len(observed_axis)],
            "bilateral_shape_yx": [len(row_axis), len(symmetric_axis)],
            "bilateral_origin_xy_um": [
                float(symmetric_axis[0]), float(row_axis[0])
            ],
            "reflection_axis": "histology_column_lr",
            "reflection_plane_um": 0.0,
            "medial_centers_um": [
                -column_spacing / 2.0, column_spacing / 2.0
            ],
            "operation": "exact_reflection_no_medial_duplication",
            "support_semantics": "validity_missing_data_not_tissue",
            "source_dataset": str(NATIVE_DATASET),
            "source_transforms": str(manifest_path),
            "transform_provenance": str(provenance_path.relative_to(stage)),
        })
        _json(stage / "dataset.json", {
            "dataset": "native 200-um section-aligned symmetric Nissl",
            "space_name": "HIST_SYMMETRIC_NATIVE_200UM",
            "preparation_mode": "preserve_source_grid",
            "pixel_size_um": column_spacing,
            "serial_spacing_um": serial_spacing,
            "physical_serial_positions": len(xJ[0]),
            "nissl_count": len(observed),
            "unilateral_shape_yx": [len(row_axis), len(observed_axis)],
            "prepared_canvas_shape_yx": [len(row_axis), len(symmetric_axis)],
        })

        qc = stage / "native_symmetric_stack_orthogonal.png"
        J_qc = np.zeros(
            (3, len(observed), len(row_axis), len(symmetric_axis)),
            np.float32,
        )
        W_qc = np.zeros(
            (len(observed), len(row_axis), len(symmetric_axis)),
            np.float32,
        )
        for order, index in enumerate(observed):
            name = (
                f"allen_708424_nissl_"
                f"{int(rows[index]['allen_section_number']):04d}.tif"
            )
            J_qc[:, order] = (
                tifffile.imread(view / name).transpose(2, 0, 1) / 255.0
            )
            W_qc[order] = tifffile.imread(support_dir / name)
        coarse._emlddmm_stack_draw_qc(
            em,
            J_qc,
            W_qc,
            [xJ[0][observed], row_axis, symmetric_axis],
            qc,
            "Native 200-um atlas-free section-aligned symmetric Nissl",
        )
        os.replace(stage, output)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return {
        "output": str(output),
        "atlas_free_manifest": str(manifest_path),
        "physical_serial_positions": len(xJ[0]),
        "observed_sections": len(observed),
        "unilateral_shape_yx": [len(row_axis), len(observed_axis)],
        "symmetric_shape_yx": [len(row_axis), len(symmetric_axis)],
        "spacing_um": [serial_spacing, row_spacing, column_spacing],
        "reflection_plane_um": 0.0,
        "qc": str(output / "native_symmetric_stack_orthogonal.png"),
    }


def warp_categorical_section(
    labels: np.ndarray,
    transform: np.ndarray,
    row_um: np.ndarray,
    column_um: np.ndarray,
    *,
    source_row_um: np.ndarray | None = None,
    source_column_um: np.ndarray | None = None,
) -> np.ndarray:
    """Apply the Nissl physical pullback with categorical nearest-neighbor."""
    categorical = np.asarray(labels)
    if categorical.ndim != 2 or not np.issubdtype(categorical.dtype, np.integer):
        raise ValueError("Categorical section must be a 2-D integer raster")
    iy, ix = coarse._section_sample_indices(
        transform,
        row_um,
        column_um,
        source_row_um=source_row_um,
        source_column_um=source_column_um,
    )
    return coarse.ndi.map_coordinates(
        categorical,
        [iy, ix],
        order=0,
        mode="constant",
        cval=0,
        prefilter=False,
    ).astype(np.uint32)


def _prepared_projection_transform(
    source_rows: dict[int, dict[str, str]], section: int
) -> tuple[Path, dict[str, Any]]:
    row = source_rows.get(section)
    if row is None or row.get("image_present") != "true" or row.get("stain") != "nissl":
        raise RuntimeError(f"Annotated Allen section {section} has no source Nissl")
    prepared = NATIVE_DATASET / row["prepared_relative_path"]
    transform_path = prepared.with_name(
        f"{prepared.stem}_prepared-to-source.json"
    )
    if not transform_path.is_file() or transform_path.is_symlink():
        raise RuntimeError(f"Missing prepared-to-source transform: {transform_path}")
    return transform_path, json.loads(transform_path.read_text(encoding="utf-8"))


def construct_symmetric_annotations(
    output: Path = SYMMETRIC_ANNOTATION_DATASET,
) -> dict[str, Any]:
    """Project, residual-warp, and exactly reflect all Allen annotations."""
    output = output.resolve()
    parent = CLEAN_SYMMETRIC_DATASET.resolve()
    if output == parent or parent in output.parents:
        raise RuntimeError("Annotation output must not modify the parent Nissl derivative")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite annotation derivative: {output}")

    parent_transforms = parent / "metadata/transforms"
    axis_names = (
        "serial_axis_um.npy",
        "row_axis_um.npy",
        "left_lr_axis_um.npy",
        "symmetric_lr_axis_um.npy",
    )
    serial_axis, row_axis, left_axis, symmetric_axis = [
        np.load(parent_transforms / name) for name in axis_names
    ]
    parent_dataset = json.loads((parent / "dataset.json").read_text(encoding="utf-8"))
    parent_symmetry = json.loads(
        (parent / "metadata/symmetry.json").read_text(encoding="utf-8")
    )
    left_shape = (len(row_axis), len(left_axis))
    bilateral_shape = (len(row_axis), len(symmetric_axis))
    if (
        tuple(parent_dataset.get("unilateral_shape_yx", ())) != left_shape
        or tuple(parent_dataset.get("prepared_canvas_shape_yx", ())) != bilateral_shape
        or tuple(parent_symmetry.get("unilateral_shape_yx", ())) != left_shape
        or tuple(parent_symmetry.get("bilateral_shape_yx", ())) != bilateral_shape
        or len(symmetric_axis) != 2 * len(left_axis)
    ):
        raise RuntimeError("Parent Nissl axes and declared dimensions differ")

    expanded = np.load(parent_transforms / "atlas_free_expanded_A2d.npy")
    observed = np.asarray(
        np.load(parent_transforms / "observed_physical_indices.npy"), dtype=np.int64
    )
    baseline, residual = _common_frame_residuals(
        expanded,
        observed,
        parent_transforms / "atlas_free_common_bookkeeping_frame.txt",
    )
    parent_rows, _ = _read_rows(parent / "metadata/physical_sections.tsv")
    source_rows_list, _ = _read_rows(
        NATIVE_DATASET / "metadata/physical_sections.tsv"
    )
    if len(parent_rows) != len(serial_axis) or len(source_rows_list) != len(serial_axis):
        raise RuntimeError("Physical-section inventory does not span the parent axes")
    physical_by_allen = {
        int(row["allen_section_number"]): index
        for index, row in enumerate(parent_rows)
    }
    source_by_allen = {
        int(row["allen_section_number"]): row for row in source_rows_list
    }
    observed_set = set(map(int, observed))

    annotation_manifest, _ = _read_rows(
        coarse.ANNOTATION_ZARR / "metadata/manifest.tsv"
    )
    if len(annotation_manifest) != 106:
        raise RuntimeError(
            f"Expected 106 source annotation sections, found {len(annotation_manifest)}"
        )
    source_dataset_path = coarse.ANNOTATION_ZARR / "dataset.json"
    source_dataset = json.loads(source_dataset_path.read_text(encoding="utf-8"))
    source_structures = (
        PROJECT / "data/raw/allen/specimen_708424/metadata/structures.tsv"
    )
    if coarse.checksum(source_structures) != source_dataset["source"]["raw_structures_sha256"]:
        raise RuntimeError("Source annotation structure hierarchy checksum differs")

    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    inventory: list[dict[str, Any]] = []
    projection_rows: list[dict[str, Any]] = []
    graphic_group_counts: dict[int, int] = {}
    metadata_by_section: dict[str, Any] = {}
    source_label_ids: set[int] = set()
    output_label_ids: set[int] = set()
    try:
        annotations_root = stage / "annotations"
        metadata_root = stage / "metadata"
        transforms_root = metadata_root / "transforms"
        annotations_root.mkdir(parents=True)
        transforms_root.mkdir(parents=True)

        transform_copy_names = {
            "provenance.json": "parent_nissl_provenance.json",
            **{name: name for name in (
                *axis_names,
                "atlas_free_expanded_A2d.npy",
                "atlas_free_observed_A2d.npy",
                "observed_physical_indices.npy",
                "atlas_free_common_bookkeeping_frame.txt",
                "atlas_free_manifest.json",
            )},
        }
        copied_transform_checksums: dict[str, str] = {}
        for source_name, destination_name in transform_copy_names.items():
            source = parent_transforms / source_name
            if not source.is_file() or source.is_symlink():
                raise RuntimeError(f"Missing parent transform provenance: {source}")
            destination = transforms_root / destination_name
            shutil.copy2(source, destination)
            copied_transform_checksums[
                destination.relative_to(stage).as_posix()
            ] = coarse.checksum(destination)

        present_sections: list[int] = []
        for manifest_row in annotation_manifest:
            section = int(manifest_row["section_number"])
            physical_index = physical_by_allen.get(section)
            if physical_index is None or physical_index not in observed_set:
                raise RuntimeError(
                    f"Annotated Allen section {section} is not an observed parent section"
                )
            if parent_rows[physical_index].get("stain") != "nissl":
                raise RuntimeError(f"Annotated Allen section {section} is not Nissl")
            transform_path, projection_transform = _prepared_projection_transform(
                source_by_allen, section
            )
            projection_rows.append({
                "section_number": section,
                "physical_index": physical_index,
                "prepared_to_source_path": str(transform_path),
                "prepared_to_source_sha256": coarse.checksum(transform_path),
                "residual_formula": "inv(baseline) @ expanded_A2d[index]",
            })
            package = coarse.ANNOTATION_ZARR / manifest_row["path"]
            labels_metadata = json.loads(
                (package / "labels/zarr.json").read_text(encoding="utf-8")
            )
            declared_names = labels_metadata.get("attributes", {}).get("ome", {}).get(
                "labels", []
            )
            group_ids = json.loads(manifest_row["graphic_groups_present"])
            if declared_names != [f"group-{value}" for value in group_ids]:
                raise RuntimeError(
                    f"Graphic-group order differs for Allen section {section}"
                )
            section_metadata: dict[str, Any] = {
                "graphic_groups": [],
                "source_package": str(package),
                "source_package_tree_sha256": manifest_row["tree_sha256"],
            }
            for group_id in group_ids:
                group_name = f"group-{group_id}"
                group_root = package / "labels" / group_name
                group_metadata = json.loads(
                    (group_root / "zarr.json").read_text(encoding="utf-8")
                ).get("attributes", {})
                labels = _read_zarr_v3_uint32(group_root / "0")
                projected = _project_annotation(
                    labels, projection_transform, left_shape
                )
                aligned = warp_categorical_section(
                    projected,
                    residual[physical_index],
                    row_axis,
                    left_axis,
                    source_row_um=row_axis,
                    source_column_um=left_axis,
                )
                bilateral = bilateral_union(aligned)
                if (
                    aligned.shape != left_shape
                    or bilateral.shape != bilateral_shape
                    or not np.array_equal(
                        bilateral[:, : len(left_axis)], aligned[:, ::-1]
                    )
                    or not np.array_equal(
                        bilateral[:, len(left_axis) :], aligned
                    )
                ):
                    raise RuntimeError(
                        f"Exact annotation symmetry failed for Allen section {section}"
                    )
                source_ids = {int(value) for value in np.unique(labels) if value}
                aligned_ids = {int(value) for value in np.unique(aligned) if value}
                if not aligned_ids.issubset(source_ids):
                    raise RuntimeError(
                        f"Categorical warp created label IDs for Allen section {section}"
                    )
                source_label_ids.update(source_ids)
                output_label_ids.update(aligned_ids)
                group_dir = annotations_root / group_name
                group_dir.mkdir(exist_ok=True)
                destination = group_dir / f"section-{section:04d}.tif"
                tifffile.imwrite(destination, bilateral, compression="deflate")
                reloaded = tifffile.imread(destination)
                if reloaded.dtype != np.uint32 or not np.array_equal(reloaded, bilateral):
                    raise RuntimeError(f"Categorical TIFF round trip changed {destination}")
                inventory.append({
                    "section_number": section,
                    "graphic_group_id": int(group_id),
                    "path": destination.relative_to(stage).as_posix(),
                    "sha256": coarse.checksum(destination),
                    "sampling": "categorical_nearest_neighbor",
                    "right_origin": "synthetically_reflected",
                })
                graphic_group_counts[int(group_id)] = (
                    graphic_group_counts.get(int(group_id), 0) + 1
                )
                section_metadata["graphic_groups"].append({
                    "graphic_group_id": int(group_id),
                    "ome_and_allen_metadata": group_metadata,
                    "source_label_ids": sorted(source_ids),
                    "aligned_label_ids": sorted(aligned_ids),
                })
            metadata_by_section[str(section)] = section_metadata
            present_sections.append(section)

        expected_sections = sorted(int(row["section_number"]) for row in annotation_manifest)
        if sorted(present_sections) != expected_sections or len(set(present_sections)) != 106:
            raise RuntimeError("Annotation section identity/inventory changed")
        if output_label_ids != source_label_ids:
            raise RuntimeError(
                "Aligned annotation label-ID inventory differs from the source"
            )
        _write_rows(
            metadata_root / "annotations.tsv",
            inventory,
            [
                "section_number", "graphic_group_id", "path", "sha256",
                "sampling", "right_origin",
            ],
        )
        _write_rows(
            metadata_root / "source_projection_transforms.tsv",
            projection_rows,
            [
                "section_number", "physical_index", "prepared_to_source_path",
                "prepared_to_source_sha256", "residual_formula",
            ],
        )
        shutil.copy2(source_structures, annotations_root / "structures.tsv")
        shutil.copy2(
            coarse.ANNOTATION_ZARR / "metadata/manifest.tsv",
            metadata_root / "source_annotation_manifest.tsv",
        )
        shutil.copy2(source_dataset_path, metadata_root / "source_annotations_dataset.json")
        _json(metadata_root / "annotation_metadata.json", {
            "graphic_groups": source_dataset["graphic_groups"],
            "sections": metadata_by_section,
            "source_label_ids": sorted(source_label_ids),
            "aligned_label_ids": sorted(output_label_ids),
            "aligned_ids_subset_of_source": output_label_ids.issubset(source_label_ids),
        })
        _json(metadata_root / "symmetry.json", {
            "symmetric_space": "HIST_SYMMETRIC",
            "image_operation": "exact_reflection_and_union_without_interpolation",
            "operation": "exact_reflection_no_medial_duplication",
            "parent_nissl_derivative": str(parent),
            "parent_nissl_dataset_json_sha256": coarse.checksum(parent / "dataset.json"),
            "pixel_size_um": float(np.diff(left_axis).mean()),
            "unilateral_shape_yx": list(left_shape),
            "bilateral_shape_yx": list(bilateral_shape),
            "bilateral_origin_xy_um": [
                float(symmetric_axis[0]), float(row_axis[0])
            ],
            "reflection_axis": "histology_column_lr",
            "reflection_plane_um": 0.0,
            "medial_centers_um": [
                float(symmetric_axis[len(left_axis) - 1]),
                float(symmetric_axis[len(left_axis)]),
            ],
            "left_half_columns": [len(left_axis), len(symmetric_axis) - 1],
            "synthetic_reflection_columns": [0, len(left_axis) - 1],
            "concatenation": "[flipped_aligned_left | aligned_left]",
            "columns_cropped": 0,
            "categorical_interpolation": "nearest_neighbor",
            "right_origin": "synthetically_reflected",
        })
        projection_manifest = metadata_root / "source_projection_transforms.tsv"
        transform_provenance = {
            "schema": "allen-native-symmetric-annotations-v1",
            "parent_nissl_derivative": str(parent),
            "parent_nissl_dataset_json_sha256": coarse.checksum(parent / "dataset.json"),
            "parent_nissl_symmetry_sha256": coarse.checksum(
                parent / "metadata/symmetry.json"
            ),
            "parent_nissl_physical_sections_sha256": coarse.checksum(
                parent / "metadata/physical_sections.tsv"
            ),
            "source_annotations_ome_zarr": str(coarse.ANNOTATION_ZARR.resolve()),
            "source_annotations_dataset_json_sha256": coarse.checksum(source_dataset_path),
            "source_annotations_manifest_sha256": coarse.checksum(
                coarse.ANNOTATION_ZARR / "metadata/manifest.tsv"
            ),
            "source_structures_sha256": coarse.checksum(source_structures),
            "prepared_projection_manifest": str(
                projection_manifest.relative_to(stage)
            ),
            "prepared_projection_manifest_sha256": coarse.checksum(projection_manifest),
            "copied_parent_transform_checksums_sha256": copied_transform_checksums,
            "baseline_selection": "expanded_A2d[unsupported][0]",
            "residual_formula": "inv(baseline) @ expanded_A2d[index]",
            "baseline_matrix": baseline.tolist(),
            "section_transform_semantics": "coarse._section_sample_indices",
            "categorical_sampling": {
                "operation": "scipy.ndimage.map_coordinates",
                "order": 0,
                "mode": "constant",
                "cval": 0,
                "prefilter": False,
            },
            "left_atlas_free_transform_applications": 1,
            "mri_geometry_used": False,
            "axes": {
                "serial": "metadata/transforms/serial_axis_um.npy",
                "row": "metadata/transforms/row_axis_um.npy",
                "unilateral_lr": "metadata/transforms/left_lr_axis_um.npy",
                "symmetric_lr": "metadata/transforms/symmetric_lr_axis_um.npy",
                "lengths": [
                    len(serial_axis), len(row_axis), len(left_axis), len(symmetric_axis)
                ],
                "units": "micrometers",
            },
            "symmetry": {
                "formula": "concatenate((flip(aligned_left, axis=1), aligned_left), axis=1)",
                "exact": True,
                "interpolation": False,
                "columns_cropped": 0,
                "medial_column_duplicated": False,
            },
        }
        _json(transforms_root / "provenance.json", transform_provenance)
        _json(stage / "dataset.json", {
            "dataset": "native 200-um section-aligned symmetric Allen annotations",
            "space_name": "HIST_SYMMETRIC",
            "symmetric_space": "HIST_SYMMETRIC",
            "parent_nissl_derivative": str(parent),
            "parent_nissl_dataset_json_sha256": coarse.checksum(parent / "dataset.json"),
            "annotations_ome_zarr_source": str(coarse.ANNOTATION_ZARR.resolve()),
            "annotation_section_count": len(expected_sections),
            "annotation_image_count": len(inventory),
            "graphic_group_counts": {
                str(key): value for key, value in sorted(graphic_group_counts.items())
            },
            "source_label_ids": sorted(source_label_ids),
            "aligned_label_ids": sorted(output_label_ids),
            "label_id_inventory_preserved": True,
            "label_ids_changed_by_interpolation": False,
            "pixel_size_um": float(np.diff(left_axis).mean()),
            "unilateral_shape_yx": list(left_shape),
            "prepared_canvas_shape_yx": list(bilateral_shape),
            "transform_provenance": "metadata/transforms/provenance.json",
        })

        from preprocess.visualize_allen_annotations import write_bilateral_montage

        qc_root = stage / "qc"
        qc_root.mkdir()
        for section in (1532, 1616):
            write_bilateral_montage(
                stage,
                section,
                output=qc_root / f"section-{section:04d}.png",
                annotations_zarr=coarse.ANNOTATION_ZARR,
                panel_width=480,
            )
        os.replace(stage, output)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return {
        "output": str(output),
        "parent_nissl_derivative": str(parent),
        "annotation_section_count": 106,
        "annotation_image_count": len(inventory),
        "unilateral_shape_yx": list(left_shape),
        "bilateral_shape_yx": list(bilateral_shape),
        "qc": [
            str(output / "qc/section-1532.png"),
            str(output / "qc/section-1616.png"),
        ],
    }


def symmetric_registration(
    *, restart_interrupted: bool = False, stop_after_scale: int | None = None,
) -> dict[str, Any]:
    initial_A, initial_A_validation = _load_validated_symmetric_initial_affine(
        CLEAN_SYMMETRIC_DATASET
    )
    return _registration(
        CLEAN_SYMMETRIC_DATASET, None, SYMMETRIC_ROOT,
        stage="symmetric-registration", profile=FINAL_PROFILE,
        initial_A=initial_A,
        a2d_initialization="identity",
        initial_A_validation=initial_A_validation,
        restart_interrupted=restart_interrupted,
        stop_after_scale=stop_after_scale,
    )


def postprocess() -> None:
    saved_tmp = coarse.RUN_TMP
    with tempfile.TemporaryDirectory(prefix="allen-native-postprocess-") as temporary:
        coarse.RUN_TMP = Path(temporary)
        try:
            with _coarse_context(CLEAN_SYMMETRIC_DATASET, SYMMETRIC_ROOT):
                coarse.postprocess(native_resolution=True)
        finally:
            coarse.RUN_TMP = saved_tmp


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=(
        "hemi-atlas-free", "hemi-registration", "construct-symmetric",
        "construct-annotations", "symmetric-atlas-free",
        "symmetric-registration", "postprocess",
    ))
    parser.add_argument(
        "--restart-interrupted", action="store_true",
        help=(
            "restart an incomplete registration stage after removing only its "
            "checkpoint and registration outputs"
        ),
    )
    parser.add_argument(
        "--stop-after-scale",
        type=int,
        choices=(1, 2, 3),
        help="stop successfully after this one-based configured scale number",
    )
    args = parser.parse_args()
    registration_stages = {"hemi-registration", "symmetric-registration"}
    if args.restart_interrupted and args.stage not in registration_stages:
        parser.error("--restart-interrupted applies only to registration stages")
    if args.stop_after_scale is not None and args.stage != "symmetric-registration":
        parser.error(
            "--stop-after-scale applies only to symmetric-registration"
        )
    actions = {
        "hemi-atlas-free": lambda: estimate_slice_initializer(NATIVE_DATASET, HEMI_ROOT),
        "hemi-registration": lambda: hemisphere_registration(
            restart_interrupted=args.restart_interrupted
        ),
        "construct-symmetric": construct_symmetric_histology,
        "construct-annotations": construct_symmetric_annotations,
        "symmetric-atlas-free": lambda: estimate_slice_initializer(
            CLEAN_SYMMETRIC_DATASET, SYMMETRIC_ROOT
        ),
        "symmetric-registration": lambda: symmetric_registration(
            restart_interrupted=args.restart_interrupted,
            stop_after_scale=args.stop_after_scale,
        ),
        "postprocess": postprocess,
    }
    actions[args.stage]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
