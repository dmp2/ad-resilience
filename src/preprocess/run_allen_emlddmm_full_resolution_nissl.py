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
import json
import math
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
    _json, _write_image_sidecar, _write_rows,
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
INITIAL_A_HEMI = PROJECT / (
    "results/qc/allen_708424_mri7t_to_symmetric_nissl_initial_similitude.txt"
)
CLEAN_ROOT = PROJECT / "results/allen/specimen_708424/emlddmm/native-200um-clean"
HEMI_ROOT = CLEAN_ROOT / "HIST_NISSL_LEFT_to_MRI_7T_WHOLE"
SYMMETRIC_ROOT = CLEAN_ROOT / "HIST_NISSL_SYMMETRIC_to_MRI_7T_WHOLE"
CLEAN_SYMMETRIC_DATASET = PROJECT / (
    "data/derivatives/allen/specimen_708424/"
    "histology_symmetric_nissl_native_200um_clean"
)
SPACING_UM = 200.0
HEMI_PROFILE = "example-standard-sigmaR5e4-a2000-dv4000-lc188-linear-no-v"
FINAL_PROFILE = "example-standard-sigmaR5e4-a2000-dv4000-lc188"


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
    )
    coarse.DATASET = dataset
    coarse.VIEW = dataset / "inputs/views/HIST_NISSL"
    coarse.configure_output_root(output)
    try:
        yield
    finally:
        (
            coarse.DATASET, coarse.VIEW, coarse.OUTPUT, coarse.CHECKPOINTS,
            coarse.ATLAS_DIR, coarse.REG_DIR, coarse.POST_DIR,
            coarse.ANNOTATION_DIR,
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
    if dataset.resolve() == NATIVE_DATASET.resolve():
        shape = tuple(map(len, axes[1:]))
        J = np.zeros((3, 2846, *shape), np.float32)
        W0 = np.zeros((2846, *shape), np.float32)
        for index in observed:
            image, support = _read_original(NATIVE_VIEW / samples[int(index)]["sample_id"])
            if image.shape[1:] != shape:
                raise RuntimeError("Original Nissl raster and native axes differ")
            J[:, index], W0[index] = image, support
    else:
        support_root = dataset / "support/nissl"
        shape = tuple(map(len, axes[1:]))
        J = np.zeros((3, 2846, *shape), np.float32)
        W0 = np.zeros((2846, *shape), np.float32)
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
    expanded = np.repeat(baseline[None], 2846, axis=0)
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
    interrupted = checkpoint.exists() or registration_files_present
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


def _registration(
    dataset: Path, initializer_root: Path, output: Path,
    *, profile: str, initial_A: np.ndarray, restart_interrupted: bool = False,
) -> dict[str, Any]:
    _prepare_registration_output(
        output, restart_interrupted=restart_interrupted
    )
    em, rows, observed, xJ, J, W0 = load_native_stack(dataset)
    A2d, initializer = _load_initializer(initializer_root, observed)
    mri, I, xI = _load_native_mri(em)
    config = native_multiscale_configuration(
        I=I, xI=xI, J=J, xJ=xJ, W0=W0,
        A=initial_A, A2d=A2d, profile=profile,
    )
    reg = output / "registration"
    checkpoints = output / "checkpoints"
    reg.mkdir(parents=True, exist_ok=True)
    checkpoints.mkdir(exist_ok=True)
    running = {
        "stage": "registration", "status": "running", "profile": profile,
        "source_dataset": str(dataset), "initializer": initializer,
        "external_pre_downsample": {"I": [1, 1, 1], "J": [1, 1, 1], "W0": [1, 1, 1]},
        "native_shapes": {"I": list(I.shape), "J": list(J.shape), "W0": list(W0.shape)},
        "native_spacings_um": {"I": [200.0] * 3, "J": [50.0, 200.0, 200.0]},
        "effective_inplane_um": [800.0, 400.0, 200.0],
        "initial_velocity": "implicit_zero",
    }
    coarse.atomic_json(checkpoints / "registration.json", running)
    histories: list[np.ndarray] = []
    sys.setprofile(coarse.profile_capture(em.emlddmm.__code__, histories))
    try:
        outputs = em.emlddmm_multiscale(**config)
    finally:
        sys.setprofile(None)
    final = outputs[-1]
    A_final = coarse.finite("final A", final["A"])
    A2d_final = coarse.finite("final A2d", final["A2d"])
    v = coarse.finite("final v", final["v"])
    xv = [coarse.finite(f"xv{i}", value) for i, value in enumerate(final["xv"])]
    coarse._save_final_effective_match_weight(
        final, observed, tuple(W0.shape),
        reg / "final_observed_effective_match_weight.npy",
    )
    hist = coarse.LightImage(
        "HIST_NISSL", "HIST_NISSL", J, xJ, "slice_dataset",
        [str(index) for index in range(2846)],
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
        profile=HEMI_PROFILE, initial_A=np.loadtxt(INITIAL_A_HEMI),
        restart_interrupted=restart_interrupted,
    )


def _snap_axis(axis: np.ndarray) -> np.ndarray:
    first = math.floor((float(axis[0]) - 100.0) / 200.0) * 200.0 + 100.0
    last = math.ceil((float(axis[-1]) + 100.0) / 200.0) * 200.0 - 100.0
    return np.arange(first, last + 100.0, 200.0, dtype=np.float64)


def _mri_midline(A: np.ndarray) -> float:
    provenance = json.loads(MRI_PROVENANCE.read_text(encoding="utf-8"))
    center = np.asarray(provenance["physical_center_mm"], dtype=np.float64) * 1000.0
    return float((np.asarray(A) @ np.r_[center, 1.0])[2])


def symmetric_lr_axis(midline_um: float, valid_observed_max_um: float) -> np.ndarray:
    """Return p±(100+200k), retaining the increasing-column hemisphere."""
    half_count = max(
        1, math.ceil((valid_observed_max_um - (midline_um + 100.0)) / 200.0) + 1
    )
    observed = midline_um + 100.0 + np.arange(half_count) * 200.0
    return np.concatenate((2.0 * midline_um - observed[::-1], observed))


def reflect_observed_half(
    image: np.ndarray, support: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Reflect RGB and missing-data validity identically without a medial copy."""
    return (
        np.concatenate((image[:, :, ::-1], image), axis=2),
        np.concatenate((support[:, ::-1], support), axis=1),
    )


def construct_symmetric_histology(
    output: Path = CLEAN_SYMMETRIC_DATASET,
) -> dict[str, Any]:
    """Materialize the hemisphere final A/A2d once, then reflect it exactly."""
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite clean derivative: {output}")
    em, rows, observed, xJ, J, W0 = load_native_stack(NATIVE_DATASET)
    numerical = HEMI_ROOT / "registration/full_resolution_numerical_outputs.npz"
    with np.load(numerical) as saved:
        A = coarse.finite("A_hemi_final", saved["A"]).astype(np.float64)
        A2d = coarse.finite("A2d_hemi_final", saved["A2d"]).astype(np.float64)
        saved_observed = np.asarray(saved["observed"], dtype=np.int64)
    if A.shape != (4, 4) or A2d.shape != (2846, 3, 3):
        raise RuntimeError("Hemisphere final transform shapes are invalid")
    if not np.array_equal(saved_observed, observed):
        raise RuntimeError("Hemisphere final section identities changed")
    unsupported = np.ones(2846, bool)
    unsupported[observed] = False
    baseline = A2d[unsupported][0]
    if not np.array_equal(A2d[unsupported], np.broadcast_to(baseline, A2d[unsupported].shape)):
        raise RuntimeError("Hemisphere unsupported rows do not share a frame")
    registered = coarse._registered_frame_axes(xJ, baseline)
    provisional_row, provisional_column = map(_snap_axis, registered)
    residual = np.linalg.inv(baseline)[None] @ A2d
    union_support = np.zeros((len(provisional_row), len(provisional_column)), bool)
    for index in observed:
        _, warped_support = coarse._warp_saved_section(
            J[:, index], W0[index], residual[index],
            provisional_row, provisional_column,
            source_row_um=xJ[1], source_column_um=xJ[2],
        )
        union_support |= warped_support > 0
    valid_rows, valid_columns = np.where(union_support)
    if not valid_rows.size:
        raise RuntimeError("Registered hemisphere validity is empty")
    row_axis = provisional_row[valid_rows.min() : valid_rows.max() + 1]
    p = _mri_midline(A)
    observed_valid_columns = provisional_column[valid_columns]
    observed_valid_columns = observed_valid_columns[observed_valid_columns > p]
    if not observed_valid_columns.size:
        raise RuntimeError("Registered validity does not reach the observed hemisphere")
    valid_column_max = float(observed_valid_columns.max())
    symmetric_axis = symmetric_lr_axis(p, valid_column_max)
    observed_axis = symmetric_axis[len(symmetric_axis) // 2 :]
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        view = stage / "inputs/views/HIST_NISSL"
        support_dir = stage / "support/nissl"
        metadata = stage / "metadata"
        view.mkdir(parents=True)
        support_dir.mkdir(parents=True)
        metadata.mkdir()
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
                J[:, index], W0[index], residual[index], row_axis, observed_axis,
                source_row_um=xJ[1], source_column_um=xJ[2],
            )
            symmetric, symmetric_support = reflect_observed_half(warped, support)
            rgb = np.moveaxis(
                np.rint(np.clip(symmetric, 0.0, 1.0) * 255.0).astype(np.uint8),
                0, -1,
            )
            Image.fromarray(rgb).save(view / name, format="TIFF", compression="tiff_deflate")
            tifffile.imwrite(support_dir / name, symmetric_support.astype(np.float32))
            _write_image_sidecar(
                view / name, shape_yx=rgb.shape[:2],
                origin_xy_um=(float(symmetric_axis[0]), float(row_axis[0])),
                z_um=float(row["serial_z_center_mm"]) * 1000.0,
                pixel_size_um=200.0,
            )
            row["prepared_relative_path"] = (Path("inputs/views/HIST_NISSL") / name).as_posix()
        _write_rows(metadata / "physical_sections.tsv", output_rows, list(output_rows[0]))
        _write_rows(view / "samples.tsv", samples, ["sample_id", "participant_id", "species", "status"])
        canvas = {
            "accepted_canvas_shape_yx": [len(row_axis), len(symmetric_axis)],
            "target_spacing_um": 200.0,
            "global_translation_xy_um": [float(symmetric_axis[0]), float(row_axis[0])],
            "preserve_source_grid": True,
        }
        _json(metadata / "loader_canvas_audit.json", canvas)
        _json(metadata / "symmetry.json", {
            "pixel_size_um": 200.0,
            "bilateral_shape_yx": [len(row_axis), len(symmetric_axis)],
            "bilateral_origin_xy_um": [float(symmetric_axis[0]), float(row_axis[0])],
            "reflection_axis": "histology_column_lr",
            "reflection_plane_um": p,
            "medial_centers_um": [p - 100.0, p + 100.0],
            "operation": "exact_reflection_no_medial_duplication",
            "support_semantics": "validity_missing_data_not_tissue",
            "source_dataset": str(NATIVE_DATASET),
            "source_transforms": str(numerical),
        })
        _json(stage / "dataset.json", {
            "dataset": "clean native 200-um registered symmetric Nissl",
            "space_name": "HIST_SYMMETRIC_NATIVE_200UM",
            "preparation_mode": "preserve_source_grid",
            "pixel_size_um": 200.0,
            "prepared_canvas_shape_yx": [len(row_axis), len(symmetric_axis)],
        })
        qc = stage / "native_symmetric_stack_orthogonal.png"
        J_qc = np.zeros((3, len(observed), len(row_axis), len(symmetric_axis)), np.float32)
        W_qc = np.zeros((len(observed), len(row_axis), len(symmetric_axis)), np.float32)
        for order, index in enumerate(observed):
            name = f"allen_708424_nissl_{int(rows[index]['allen_section_number']):04d}.tif"
            J_qc[:, order] = tifffile.imread(view / name).transpose(2, 0, 1) / 255.0
            W_qc[order] = tifffile.imread(support_dir / name)
        coarse._emlddmm_stack_draw_qc(
            em, J_qc, W_qc, [xJ[0][observed], row_axis, symmetric_axis],
            qc, "Clean native 200-um symmetric Nissl",
        )
        os.replace(stage, output)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return {"output": str(output), "midline_um": p, "spacing_um": 200.0}


def symmetric_registration(
    *, restart_interrupted: bool = False,
) -> dict[str, Any]:
    hemi_path = HEMI_ROOT / "registration/full_resolution_numerical_outputs.npz"
    with np.load(hemi_path) as saved:
        A_hemi_final = np.asarray(saved["A"]).copy()
    return _registration(
        CLEAN_SYMMETRIC_DATASET, SYMMETRIC_ROOT, SYMMETRIC_ROOT,
        profile=FINAL_PROFILE, initial_A=A_hemi_final,
        restart_interrupted=restart_interrupted,
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
        "symmetric-atlas-free", "symmetric-registration", "postprocess",
    ))
    parser.add_argument(
        "--restart-interrupted", action="store_true",
        help=(
            "restart an incomplete registration stage after removing only its "
            "checkpoint and registration outputs"
        ),
    )
    args = parser.parse_args()
    registration_stages = {"hemi-registration", "symmetric-registration"}
    if args.restart_interrupted and args.stage not in registration_stages:
        parser.error("--restart-interrupted applies only to registration stages")
    actions = {
        "hemi-atlas-free": lambda: estimate_slice_initializer(NATIVE_DATASET, HEMI_ROOT),
        "hemi-registration": lambda: hemisphere_registration(
            restart_interrupted=args.restart_interrupted
        ),
        "construct-symmetric": construct_symmetric_histology,
        "symmetric-atlas-free": lambda: estimate_slice_initializer(
            CLEAN_SYMMETRIC_DATASET, SYMMETRIC_ROOT
        ),
        "symmetric-registration": lambda: symmetric_registration(
            restart_interrupted=args.restart_interrupted
        ),
        "postprocess": postprocess,
    }
    actions[args.stage]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
