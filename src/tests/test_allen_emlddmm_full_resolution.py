from __future__ import annotations

import ast
import copy
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import tifffile
import torch

from preprocess.run_allen_emlddmm import pinned_emlddmm
from preprocess import run_allen_emlddmm_full_resolution_nissl as native


def _write_atlas_free_fixture(
    root: Path,
    *,
    A2d: np.ndarray,
    observed: np.ndarray,
) -> Path:
    atlas = root / "section_alignment_atlas_free"
    checkpoints = root / "checkpoints"
    atlas.mkdir(parents=True)
    checkpoints.mkdir()
    paths = {
        "observed_A2d": atlas / "observed_A2d.npy",
        "expanded_A2d": atlas / "deliberately_manifest_named_A2d.npy",
        "observed_indices": atlas / "deliberately_manifest_named_indices.npy",
        "bookkeeping_frame": atlas / "deliberately_manifest_named_frame.txt",
    }
    unsupported = np.ones(len(A2d), dtype=bool)
    unsupported[observed] = False
    np.save(paths["observed_A2d"], A2d[observed])
    np.save(paths["expanded_A2d"], A2d)
    np.save(paths["observed_indices"], observed)
    np.savetxt(paths["bookkeeping_frame"], A2d[unsupported][0])
    payload = {
        "stage": "atlas-free",
        "status": "complete",
        "purpose": "slice_to_neighbor_initializer_only",
        "source_dataset": str(native.NATIVE_DATASET),
        "outputs": {key: str(path) for key, path in paths.items()},
        "checksums": {
            str(path): native.coarse.checksum(path) for path in paths.values()
        },
    }
    manifest = checkpoints / "atlas-free.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    return manifest


def _write_registration_checkpoint(root: Path, status: str) -> Path:
    checkpoint = root / "checkpoints/registration.json"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_text(json.dumps({"status": status}), encoding="utf-8")
    return checkpoint


def test_registration_allows_shared_root_with_atlas_free_outputs(
    tmp_path: Path,
) -> None:
    atlas = tmp_path / "section_alignment_atlas_free"
    atlas.mkdir()
    initializer = atlas / "expanded_2846_A2d.npy"
    initializer.write_bytes(b"initializer")
    atlas_checkpoint = _write_registration_checkpoint(tmp_path, "complete")
    atlas_checkpoint.rename(tmp_path / "checkpoints/atlas-free.json")

    native._prepare_registration_output(
        tmp_path, restart_interrupted=False
    )

    assert initializer.read_bytes() == b"initializer"
    assert (tmp_path / "checkpoints/atlas-free.json").exists()


def test_running_registration_with_empty_directory_requires_explicit_restart(
    tmp_path: Path,
) -> None:
    (tmp_path / "registration").mkdir()
    checkpoint = _write_registration_checkpoint(tmp_path, "running")

    with pytest.raises(FileExistsError, match="--restart-interrupted"):
        native._prepare_registration_output(
            tmp_path, restart_interrupted=False
        )

    assert checkpoint.exists()


def test_running_registration_with_empty_directory_restarts_with_flag(
    tmp_path: Path,
) -> None:
    registration = tmp_path / "registration"
    registration.mkdir()
    checkpoint = _write_registration_checkpoint(tmp_path, "running")

    native._prepare_registration_output(
        tmp_path, restart_interrupted=True
    )

    assert registration.is_dir()
    assert not any(registration.iterdir())
    assert not checkpoint.exists()


def test_partial_registration_files_require_explicit_restart(
    tmp_path: Path,
) -> None:
    registration = tmp_path / "registration"
    registration.mkdir()
    partial = registration / "partial.npy"
    partial.write_bytes(b"partial")

    with pytest.raises(FileExistsError, match="--restart-interrupted"):
        native._prepare_registration_output(
            tmp_path, restart_interrupted=False
        )

    assert partial.exists()


def test_explicit_restart_clears_only_incomplete_registration_products(
    tmp_path: Path,
) -> None:
    atlas = tmp_path / "section_alignment_atlas_free"
    atlas.mkdir()
    initializer = atlas / "expanded_2846_A2d.npy"
    initializer.write_bytes(b"preserve initializer and checksum source")
    atlas_checkpoint = tmp_path / "checkpoints/atlas-free.json"
    atlas_checkpoint.parent.mkdir()
    atlas_checkpoint.write_text(
        json.dumps({"status": "complete", "checksums": {"sentinel": "abc"}}),
        encoding="utf-8",
    )
    atlas_before = initializer.read_bytes()
    checkpoint_before = atlas_checkpoint.read_bytes()
    registration = tmp_path / "registration"
    registration.mkdir()
    (registration / "partial.npy").write_bytes(b"partial")
    registration_checkpoint = _write_registration_checkpoint(tmp_path, "running")
    scale_state = tmp_path / "checkpoints/registration_scale-01.npz"
    scale_manifest = tmp_path / "checkpoints/registration_scale-01.json"
    scale_temporary = tmp_path / "checkpoints/registration_scale-02.npz.tmp"
    scale_state.write_bytes(b"completed scale state")
    scale_manifest.write_text(
        json.dumps({"status": "complete"}), encoding="utf-8"
    )
    scale_temporary.write_bytes(b"incomplete scale state")

    native._prepare_registration_output(
        tmp_path, restart_interrupted=True
    )

    assert registration.is_dir()
    assert not any(registration.iterdir())
    assert not registration_checkpoint.exists()
    assert initializer.read_bytes() == atlas_before
    assert atlas_checkpoint.read_bytes() == checkpoint_before
    assert scale_state.read_bytes() == b"completed scale state"
    assert json.loads(scale_manifest.read_text(encoding="utf-8")) == {
        "status": "complete"
    }
    assert not scale_temporary.exists()


def test_intentional_partial_completion_can_resume_without_losing_scale_two(
    tmp_path: Path,
) -> None:
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    registration = tmp_path / "registration"
    registration.mkdir()
    (registration / "partial-file").write_text("replace on resume", encoding="utf-8")
    (checkpoints / "registration.json").write_text(
        json.dumps(
            {
                "status": "complete_through_scale",
                "completed_scale_number": 2,
            }
        ),
        encoding="utf-8",
    )
    scale_files = [
        checkpoints / f"registration_scale-{scale:02d}.{suffix}"
        for scale in (1, 2)
        for suffix in ("npz", "json")
    ]
    for path in scale_files:
        path.write_text("preserve", encoding="utf-8")
    product = tmp_path / native.THROUGH_SCALE_DIRECTORY
    product.mkdir()
    sentinel = product / "numerical_outputs.npz"
    sentinel.write_text("preserve published scale 2", encoding="utf-8")

    with pytest.raises(FileExistsError, match="--restart-interrupted"):
        native._prepare_registration_output(
            tmp_path, restart_interrupted=False
        )
    native._prepare_registration_output(
        tmp_path, restart_interrupted=True
    )

    assert registration.is_dir() and not any(registration.iterdir())
    assert not (checkpoints / "registration.json").exists()
    assert all(path.read_text(encoding="utf-8") == "preserve" for path in scale_files)
    assert sentinel.read_text(encoding="utf-8") == "preserve published scale 2"


@pytest.mark.parametrize("completion_evidence", ["checkpoint", "numerical"])
def test_completed_registration_is_protected(
    tmp_path: Path, completion_evidence: str,
) -> None:
    if completion_evidence == "checkpoint":
        protected = _write_registration_checkpoint(tmp_path, "complete")
    else:
        registration = tmp_path / "registration"
        registration.mkdir()
        protected = registration / "full_resolution_numerical_outputs.npz"
        protected.write_bytes(b"complete")

    with pytest.raises(FileExistsError, match="completed registration"):
        native._prepare_registration_output(
            tmp_path, restart_interrupted=True
        )

    assert protected.exists()


def test_restart_flag_is_forwarded_only_to_registration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    monkeypatch.setattr(
        native, "hemisphere_registration",
        lambda **kwargs: calls.append(kwargs),
    )
    monkeypatch.setattr(
        sys, "argv",
        ["run_allen_emlddmm_full_resolution_nissl.py",
         "hemi-registration", "--restart-interrupted"],
    )

    assert native.main() == 0
    assert calls == [{"restart_interrupted": True}]


def test_restart_flag_is_rejected_for_nonregistration_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys, "argv",
        ["run_allen_emlddmm_full_resolution_nissl.py",
         "hemi-atlas-free", "--restart-interrupted"],
    )

    with pytest.raises(SystemExit, match="2"):
        native.main()


def test_stop_after_scale_is_forwarded_only_to_symmetric_registration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    monkeypatch.setattr(
        native, "symmetric_registration",
        lambda **kwargs: calls.append(kwargs),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_allen_emlddmm_full_resolution_nissl.py",
            "symmetric-registration",
            "--stop-after-scale",
            "2",
        ],
    )

    assert native.main() == 0
    assert calls == [{"restart_interrupted": False, "stop_after_scale": 2}]


def test_stop_after_scale_is_rejected_for_other_stages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_allen_emlddmm_full_resolution_nissl.py",
            "construct-symmetric",
            "--stop-after-scale",
            "2",
        ],
    )
    with pytest.raises(SystemExit, match="2"):
        native.main()


def test_native_and_temporary_coarse_axes_share_physical_frame() -> None:
    source = next(native.NATIVE_VIEW.glob("*.tif"))
    axes = native.native_source_axes(tifffile.imread(source).shape[:2])
    reduced = native.coarse_axes_from_native(pinned_emlddmm(), axes)
    historical = (
        native.PROJECT
        / "results/allen/specimen_708424/emlddmm/full-coarse"
        / "HIST_NISSL_to_MRI_7T_WHOLE_linear-no-v/registration"
        / "full_coarse_numerical_outputs.npz"
    )
    with np.load(historical) as saved:
        np.testing.assert_allclose(reduced[0], saved["xJ1"], atol=1e-6, rtol=0)
        np.testing.assert_allclose(reduced[1], saved["xJ2"], atol=1e-6, rtol=0)
    assert not np.array_equal(reduced[0], axes[0][::4][: len(reduced[0])])


def test_original_reader_rejects_non_authoritative_derivatives(monkeypatch) -> None:
    sentinel = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
    monkeypatch.setattr(native.tifffile, "imread", lambda path: sentinel)
    image, support = native._read_original(native.NATIVE_VIEW / "sentinel.tif")
    np.testing.assert_allclose(image, sentinel.transpose(2, 0, 1) / 255.0)
    np.testing.assert_array_equal(support, sentinel[..., 0] > 0)
    with pytest.raises(RuntimeError, match="native emlddmm_7t"):
        native._read_original(
            native.PROJECT
            / "data/derivatives/allen/specimen_708424/histology_symmetric/x.tif"
        )


@pytest.mark.parametrize("profile", [native.HEMI_PROFILE, native.FINAL_PROFILE])
def test_atlas_to_slice_configuration_is_native_and_preserves_A2d(profile) -> None:
    I = np.zeros((1, 3, 4, 5), np.float32)
    J = np.zeros((3, 6, 7, 8), np.float32)
    W0 = np.ones((6, 7, 8), np.float32)
    xI = [np.arange(n) * 200.0 for n in I.shape[1:]]
    xJ = [np.arange(6) * 50.0, np.arange(7) * 200.0, np.arange(8) * 200.0]
    A = np.eye(4)
    A2d = np.zeros((6, 3, 3), np.float64)
    A2d[:, 0, 2] = 1234.5
    expected = copy.deepcopy(native.coarse.resolve_registration_execution(profile))
    config = native.native_multiscale_configuration(
        I=I, xI=xI, J=J, xJ=xJ, W0=W0, A=A, A2d=A2d, profile=profile
    )
    assert config["I"] is I and config["J"] is J and config["W0"] is W0
    assert config["xI"][0] is xI and config["xJ"][0] is xJ
    np.testing.assert_array_equal(config["A2d"], A2d)
    np.testing.assert_array_equal(config["A2d"][:, 0, 2], 1234.5)
    assert config["v"] is None
    for key, value in expected.items():
        assert config[key] == value
    assert config["downI"] == [[4, 4, 4], [2, 2, 2], [1, 1, 1]]
    assert config["downJ"] == [[1, 4, 4], [1, 2, 2], [1, 1, 1]]


def test_corrected_symmetric_profile_remains_exactly_unchanged() -> None:
    profile = native.coarse.resolve_registration_execution(native.FINAL_PROFILE)
    assert profile["Amode"] == [2, 2, 2]
    assert profile["a"] == [2000.0] * 3
    assert profile["dv"] == [4000.0] * 3
    assert profile["sigmaR"] == [50000.0] * 3
    assert profile["local_contrast"] == [[1, 8, 8]] * 3
    assert profile["eA"] == [1e6] * 3
    assert profile["eA2d"] == [1e5] * 3
    assert profile["ev"] == [0.01] * 3
    assert profile["n_iter"] == [100, 50, 40]
    assert profile["slice_matching"] == [True, True, True]
    assert profile["slice_deformation"] == [False, False, False]
    assert profile["v_start"] == [0, 0, 0]
    assert profile["downI"] == [[4, 4, 4], [2, 2, 2], [1, 1, 1]]
    assert profile["downJ"] == [[1, 4, 4], [1, 2, 2], [1, 1, 1]]
    assert profile["full_outputs"] == [False, False, True]
    assert profile["rigid_procrustes"] == [True, True, True]


def test_initializer_stage_downsamples_only_before_atlas_free(
    tmp_path: Path, monkeypatch,
) -> None:
    class EM:
        def downsample_image_domain(self, axes, image, down):
            return [
                axes[0],
                native.coarse.block_axis(axes[1], 4),
                native.coarse.block_axis(axes[2], 4),
            ], image[:, :, :4, :4]

        def atlas_free_reconstruction(self, **kwargs):
            captured.update(kwargs)
            return {"A2d": np.repeat(np.eye(3)[None], 2, axis=0)}

    captured = {}
    axes = [
        np.asarray([0.0, 50.0]),
        np.arange(8) * 200.0,
        np.arange(8) * 200.0,
    ]
    J = np.ones((3, 2846, 8, 8), np.float32)
    W0 = np.ones((2846, 8, 8), np.float32)
    rows = [{"physical_index": str(i)} for i in range(2846)]
    monkeypatch.setattr(
        native, "load_native_stack",
        lambda dataset: (EM(), rows, np.asarray([0, 1]), axes, J, W0),
    )
    native.estimate_slice_initializer(tmp_path / "native", tmp_path / "out")
    assert captured["J"].shape == (3, 2, 2, 2)
    assert captured["W"].shape == (2, 2, 2)
    assert np.diff(captured["xJ"][1]).item() == 800.0


def test_lr_symmetry_is_historical_coronal_column_flip() -> None:
    axis = native.symmetric_lr_axis(4, 200.0)
    np.testing.assert_array_equal(
        axis, [-700.0, -500.0, -300.0, -100.0, 100.0, 300.0, 500.0, 700.0]
    )
    assert len(axis) % 2 == 0
    assert np.allclose(np.diff(axis), 200.0)
    assert len(native.symmetric_lr_axis(365, 200.0)) == 730
    image = np.arange(3 * 2 * 4).reshape(3, 2, 4)
    support = np.arange(2 * 4).reshape(2, 4)
    reflected_image, reflected_support = native.reflect_observed_half(image, support)
    np.testing.assert_array_equal(reflected_image[:, :, :4], image[:, :, ::-1])
    np.testing.assert_array_equal(reflected_image[:, :, 4:], image)
    np.testing.assert_array_equal(reflected_support[:, :4], support[:, ::-1])
    np.testing.assert_array_equal(reflected_support[:, 4:], support)
    assert reflected_image.shape[1] == image.shape[1]
    assert reflected_image.shape[2] == 2 * image.shape[2]
    assert reflected_support.shape[0] == support.shape[0]
    assert reflected_support.shape[1] == 2 * support.shape[1]


def test_native_symmetry_construction_has_no_mri_or_support_canvas_selector() -> None:
    source = inspect.getsource(native.construct_symmetric_histology)
    for forbidden in (
        "full_resolution_numerical_outputs",
        "registration_scale-",
        "MRI_PROVENANCE",
        "_registered_left_half_space",
        "_mri_midline",
        "_snap_axis",
        "union_support",
        "valid_columns",
        "valid_column_max",
        "_registered_frame_axes",
    ):
        assert forbidden not in source
    assert "manifest_path = ATLAS_FREE_MANIFEST" in source
    assert 'transform_paths["expanded_A2d"]' in source
    assert "_validated_native_geometry" in source
    assert "_common_frame_residuals" in source
    assert "_warp_saved_section" in source
    assert "[xJ[0][observed], row_axis, symmetric_axis]" in source


def test_common_frame_normalization_matches_validated_formula(tmp_path: Path) -> None:
    baseline = np.array(
        [[1.0, 0.0, 120.0], [0.0, 1.0, -80.0], [0.0, 0.0, 1.0]]
    )
    observed = np.array([0, 2, 5], dtype=np.int64)
    A2d = np.repeat(baseline[None], 6, axis=0)
    local = np.array(
        [[0.0, -1.0, 200.0], [1.0, 0.0, -400.0], [0.0, 0.0, 1.0]]
    )
    A2d[2] = baseline @ local
    common = tmp_path / "common.txt"
    np.savetxt(common, baseline)

    actual_baseline, residual = native._common_frame_residuals(
        A2d, observed, common
    )

    np.testing.assert_array_equal(actual_baseline, baseline)
    np.testing.assert_allclose(
        residual, np.linalg.inv(baseline)[None] @ A2d, atol=0.0, rtol=0.0
    )
    np.testing.assert_allclose(residual[2], local, atol=1e-12, rtol=0.0)


def test_atlas_free_observed_identities_must_match_native_stack(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    serial_count, height, width = 5, 2, 3
    observed = np.array([0, 2], dtype=np.int64)
    A2d = np.repeat(np.eye(3)[None], serial_count, axis=0)
    manifest = _write_atlas_free_fixture(
        tmp_path / "initializer", A2d=A2d, observed=np.array([0, 3])
    )
    axes = [
        np.arange(serial_count) * 50.0,
        np.arange(height) * 200.0,
        np.arange(width) * 200.0,
    ]
    rows = [
        {
            "physical_index": str(index),
            "allen_section_number": str(index),
            "serial_z_center_mm": str(index * 0.05),
            "stain": "nissl" if index in observed else "absent",
        }
        for index in range(serial_count)
    ]
    monkeypatch.setattr(native, "ATLAS_FREE_MANIFEST", manifest)
    monkeypatch.setattr(
        native,
        "load_native_stack",
        lambda dataset: (
            object(),
            rows,
            observed,
            axes,
            np.zeros((3, serial_count, height, width), np.float32),
            np.zeros((serial_count, height, width), np.float32),
        ),
    )

    with pytest.raises(RuntimeError, match="observed identities differ"):
        native.construct_symmetric_histology(tmp_path / "derivative")


def test_construct_symmetric_uses_native_atlas_free_materialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    serial_count, height, width = 7, 3, 4
    observed = np.array([0, 3, 6], dtype=np.int64)
    axes = [
        np.arange(serial_count, dtype=np.float64) * 50.0 - 150.0,
        np.arange(height, dtype=np.float64) * 200.0 - 200.0,
        np.arange(width, dtype=np.float64) * 200.0 - 300.0,
    ]
    rows = [
        {
            "physical_index": str(index),
            "allen_section_number": str(100 + index),
            "serial_z_center_mm": str(axes[0][index] / 1000.0),
            "stain": "nissl" if index in observed else "absent",
            "prepared_relative_path": "",
        }
        for index in range(serial_count)
    ]
    J = np.zeros((3, serial_count, height, width), np.float32)
    W0 = np.zeros((serial_count, height, width), np.float32)
    for order, index in enumerate(observed, 1):
        J[:, index] = (
            np.arange(3 * height * width, dtype=np.float32)
            .reshape(3, height, width)
            + order
        ) / 255.0
        W0[index] = (
            np.arange(height * width, dtype=np.float32).reshape(height, width)
            + order
        ) / 20.0

    baseline = np.array(
        [[1.0, 0.0, 40.0], [0.0, 1.0, -60.0], [0.0, 0.0, 1.0]]
    )
    A2d = np.repeat(baseline[None], serial_count, axis=0)
    local_transforms = []
    for order, index in enumerate(observed):
        local = np.array(
            [[1.0, 0.0, order * 10.0], [0.0, 1.0, -order * 5.0], [0.0, 0.0, 1.0]]
        )
        local_transforms.append(local)
        A2d[index] = baseline @ local
    manifest = _write_atlas_free_fixture(
        tmp_path / "initializer", A2d=A2d, observed=observed
    )
    captured = []

    def fake_warp(
        image, support, transform, row_axis, column_axis,
        *, source_row_um, source_column_um,
    ):
        captured.append(
            (
                np.asarray(transform).copy(),
                np.asarray(row_axis).copy(),
                np.asarray(column_axis).copy(),
                np.asarray(source_row_um).copy(),
                np.asarray(source_column_um).copy(),
            )
        )
        return np.asarray(image).copy(), np.asarray(support).copy()

    def fake_qc(em, image, support, qc_axes, output, title):
        assert image.shape == (3, len(observed), height, 2 * width)
        assert support.shape == (len(observed), height, 2 * width)
        np.testing.assert_array_equal(qc_axes[0], axes[0][observed])
        output.write_bytes(b"orthogonal qc")
        return output

    monkeypatch.setattr(native, "ATLAS_FREE_MANIFEST", manifest)
    monkeypatch.setattr(
        native,
        "load_native_stack",
        lambda dataset: (object(), rows, observed, axes, J, W0),
    )
    monkeypatch.setattr(native.coarse, "_warp_saved_section", fake_warp)
    monkeypatch.setattr(native.coarse, "_emlddmm_stack_draw_qc", fake_qc)
    output = tmp_path / "authoritative"
    result = native.construct_symmetric_histology(output)

    assert len(captured) == len(observed)
    for call, expected in zip(captured, local_transforms):
        np.testing.assert_allclose(call[0], expected, atol=1e-12, rtol=0.0)
        np.testing.assert_array_equal(call[1], axes[1])
        np.testing.assert_array_equal(call[2], axes[2])
        np.testing.assert_array_equal(call[3], axes[1])
        np.testing.assert_array_equal(call[4], axes[2])

    for index in observed:
        name = f"allen_708424_nissl_{int(rows[index]['allen_section_number']):04d}.tif"
        symmetric = tifffile.imread(output / "inputs/views/HIST_NISSL" / name)
        support = tifffile.imread(output / "support/nissl" / name)
        unilateral = np.rint(J[:, index] * 255.0).astype(np.uint8).transpose(1, 2, 0)
        np.testing.assert_array_equal(symmetric[:, width:], unilateral)
        np.testing.assert_array_equal(symmetric[:, :width], unilateral[:, ::-1])
        np.testing.assert_array_equal(support[:, width:], W0[index])
        np.testing.assert_array_equal(support[:, :width], W0[index, :, ::-1])

    transforms = output / "metadata/transforms"
    expected_transform_files = {
        "atlas_free_manifest.json",
        "atlas_free_expanded_A2d.npy",
        "atlas_free_observed_A2d.npy",
        "atlas_free_common_bookkeeping_frame.txt",
        "observed_physical_indices.npy",
        "serial_axis_um.npy",
        "row_axis_um.npy",
        "left_lr_axis_um.npy",
        "symmetric_lr_axis_um.npy",
        "provenance.json",
    }
    assert expected_transform_files.issubset(
        {path.name for path in transforms.iterdir()}
    )
    np.testing.assert_array_equal(np.load(transforms / "serial_axis_um.npy"), axes[0])
    np.testing.assert_array_equal(np.load(transforms / "row_axis_um.npy"), axes[1])
    np.testing.assert_array_equal(
        np.load(transforms / "left_lr_axis_um.npy"), axes[2]
    )
    np.testing.assert_array_equal(
        np.load(transforms / "symmetric_lr_axis_um.npy"),
        native.symmetric_lr_axis(width, 200.0),
    )
    provenance = json.loads((transforms / "provenance.json").read_text())
    assert provenance["common_frame_normalization"]["translation_rescaling"] == "none"
    assert provenance["materialization"]["native_pixels_upsampled"] is False
    assert provenance["materialization"]["serial_coordinates_modified"] is False
    assert provenance["symmetry"]["medial_column_duplicated"] is False
    assert result["physical_serial_positions"] == serial_count
    assert result["observed_sections"] == len(observed)
    assert result["unilateral_shape_yx"] == [height, width]
    assert result["symmetric_shape_yx"] == [height, 2 * width]
    assert result["spacing_um"] == [50.0, 200.0, 200.0]
    assert (output / "native_symmetric_stack_orthogonal.png").is_file()


def test_authoritative_symmetric_derivative_refuses_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "exists"
    output.mkdir()
    monkeypatch.setattr(
        native,
        "_load_completed_atlas_free_manifest",
        lambda path: pytest.fail("manifest must not be read before overwrite refusal"),
    )
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        native.construct_symmetric_histology(output)


def test_symmetric_registration_uses_only_corrected_main_path() -> None:
    assert native.CLEAN_SYMMETRIC_DATASET.name == (
        "histology_symmetric_nissl_native_200um_section_aligned"
    )
    assert native.SYMMETRIC_ROOT.name == (
        "HIST_NISSL_SYMMETRIC_SECTION_ALIGNED_to_MRI_7T_WHOLE"
    )
    source = inspect.getsource(native.symmetric_registration)
    assert "CLEAN_SYMMETRIC_DATASET, None, SYMMETRIC_ROOT" in source
    assert 'a2d_initialization="identity"' in source
    for forbidden in (
        "HEMI_ROOT",
        "full_resolution_numerical_outputs",
        "registration_scale-01",
        "registration_scale-02",
        "_load_initializer",
        "estimate_slice_initializer",
    ):
        assert forbidden not in source


def test_initial_affine_is_validated_for_corrected_zero_centered_frame() -> None:
    affine, audit = native._load_validated_symmetric_initial_affine(
        native.CLEAN_SYMMETRIC_DATASET
    )
    assert audit["status"] == "compatible"
    assert audit["sha256"] == native.INITIAL_A_SHA256
    assert audit["histology_axis_lengths"] == [2846, 522, 730]
    assert audit["histology_spacings_um"] == [50.0, 200.0, 200.0]
    assert audit["reflection_plane_x_um"] == 0.0
    np.testing.assert_allclose(
        audit["mapped_mri_center_histology_um"],
        [-100.0, -100.0, 100.0],
        atol=2e-3,
        rtol=0.0,
    )
    np.testing.assert_array_equal(
        affine[:3, :3],
        [[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]],
    )


def test_corrected_stack_uses_identity_a2d_without_double_application() -> None:
    transforms = native.CLEAN_SYMMETRIC_DATASET / "metadata/transforms"
    xJ = [
        np.load(transforms / "serial_axis_um.npy"),
        np.load(transforms / "row_axis_um.npy"),
        np.load(transforms / "symmetric_lr_axis_um.npy"),
    ]
    observed = np.load(transforms / "observed_physical_indices.npy")
    A2d, initializer = native._identity_section_initializer(
        native.CLEAN_SYMMETRIC_DATASET, observed, xJ
    )
    assert A2d.shape == (len(xJ[0]), 3, 3)
    np.testing.assert_array_equal(
        A2d, np.broadcast_to(np.eye(3), A2d.shape)
    )
    assert initializer["type"] == "explicit_identity_per_physical_serial_position"
    assert initializer["left_atlas_free_A2d_reapplied"] is False
    assert "atlas_free_expanded_A2d" not in json.dumps(initializer)


def test_symmetric_initializer_is_not_raster_applied_before_final_call() -> None:
    tree = ast.parse(inspect.getsource(native.symmetric_registration))
    called = {
        node.func.id if isinstance(node.func, ast.Name) else node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, (ast.Name, ast.Attribute))
    }
    assert "_registration" in called
    assert "construct_symmetric_histology" not in called
    assert "_warp_saved_section" not in called


def test_native_postprocess_has_annotation_compatibility_gate() -> None:
    source = inspect.getsource(native.coarse.postprocess)
    assert "native_resolution and annotations_enabled" in source
    assert "bilateral_shape_yx" in source
    assert "annotations_enabled = False" in source


def _checkpoint_config() -> dict:
    I = np.zeros((1, 2, 2, 2), np.float32)
    J = np.zeros((3, 4, 2, 2), np.float32)
    return {
        "I": I,
        "xI": [[np.arange(2, dtype=np.float64)] * 3],
        "J": J,
        "xJ": [[np.arange(4, dtype=np.float64),
                 np.arange(2, dtype=np.float64),
                 np.arange(2, dtype=np.float64)]],
        "W0": np.ones((4, 2, 2), np.float32),
        "A": np.eye(4, dtype=np.float32),
        "A2d": np.repeat(np.eye(3, dtype=np.float32)[None], 4, axis=0),
        "v": None,
        "dtype": torch.float32,
        "device": "cpu",
        "downI": [[4, 4, 4], [2, 2, 2], [1, 1, 1]],
        "downJ": [[1, 4, 4], [1, 2, 2], [1, 1, 1]],
        "n_iter": [10, 20, 30],
        "slice_matching": [True, True, True],
        "scale_tag": [0, 1, 2],
    }


def _checkpoint_lineage(config: dict, **overrides) -> dict:
    lineage = {
        "stage": "hemi-registration",
        "profile_name": "unit-profile",
        "source_dataset": "/unit/source",
        "native_shapes": {
            "I": list(config["I"].shape),
            "J": list(config["J"].shape),
            "W0": list(config["W0"].shape),
        },
        "native_spacings_um": {
            "I": [200.0, 200.0, 200.0],
            "J": [50.0, 200.0, 200.0],
        },
        "initializer_lineage_sha256": "initializer-lineage",
        "initializer_checksums": {"expanded_A2d.npy": "initializer-checksum"},
        "initial_A_sha256": "initial-A",
        "emlddmm_commit": "pinned-commit",
    }
    lineage.update(overrides)
    return lineage


class _FakeEM:
    def __init__(self, fail_scale: int | None = None):
        self.fail_scale = fail_scale
        self.calls: list[dict] = []
        self.outputs: list[dict] = []

    def emlddmm(self, **kwargs):
        scale = kwargs["scale_tag"]
        self.calls.append(dict(kwargs))
        Esave = [[torch.tensor(float(scale), dtype=torch.float32)]]
        if scale == self.fail_scale:
            raise MemoryError(f"simulated scale {scale + 1} OOM")
        size = scale + 1
        output = {
            "A": torch.eye(4, dtype=torch.float32) * (scale + 1),
            "A2d": torch.eye(3, dtype=torch.float32)[None].repeat(4, 1, 1)
                    * (scale + 1),
            "v": torch.full(
                (1, 3, size, size, size), float(scale + 1), dtype=torch.float32
            ),
            "xv": [torch.arange(size, dtype=torch.float32) for _ in range(3)],
        }
        self.outputs.append(output)
        return output


def _assert_same_value(left, right) -> None:
    if isinstance(left, (np.ndarray, torch.Tensor)) or isinstance(
        right, (np.ndarray, torch.Tensor)
    ):
        np.testing.assert_array_equal(np.asarray(left), np.asarray(right))
    elif isinstance(left, list):
        assert isinstance(right, list) and len(left) == len(right)
        for lvalue, rvalue in zip(left, right, strict=True):
            _assert_same_value(lvalue, rvalue)
    else:
        assert left == right


def test_checkpointed_multiscale_matches_pinned_calls_and_propagation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _checkpoint_config()
    pinned = pinned_emlddmm()
    expected = _FakeEM()
    monkeypatch.setattr(pinned, "emlddmm", expected.emlddmm)
    pinned.emlddmm_multiscale(**copy.deepcopy(config))
    actual = _FakeEM()
    outputs, histories = native.checkpointed_multiscale(
        actual, config=copy.deepcopy(config), checkpoint_dir=tmp_path,
        lineage=_checkpoint_lineage(config), resume=False,
    )

    assert [call["scale_tag"] for call in actual.calls] == [0, 1, 2]
    assert len(histories) == 3
    for expected_call, actual_call in zip(expected.calls, actual.calls, strict=True):
        assert expected_call.keys() == actual_call.keys()
        for key in expected_call:
            _assert_same_value(expected_call[key], actual_call[key])
    assert actual.calls[1]["A"] is outputs[0]["A"]
    assert actual.calls[1]["v"] is outputs[0]["v"]
    assert actual.calls[1]["A2d"] is outputs[0]["A2d"]


def test_stop_after_scale_two_runs_original_indices_and_preserves_continuation(
    tmp_path: Path,
) -> None:
    config = _checkpoint_config()
    lineage = _checkpoint_lineage(config)
    first = _FakeEM()
    outputs, histories = native.checkpointed_multiscale(
        first,
        config=config,
        checkpoint_dir=tmp_path,
        lineage=lineage,
        resume=False,
        stop_after_scale=2,
    )

    assert [call["scale_tag"] for call in first.calls] == [0, 1]
    assert [call["n_iter"] for call in first.calls] == [10, 20]
    assert len(outputs) == 2
    assert len(histories) == 2
    assert first.calls[1]["A"] is outputs[0]["A"]
    assert first.calls[1]["v"] is outputs[0]["v"]
    assert first.calls[1]["A2d"] is outputs[0]["A2d"]
    for scale_index in (0, 1):
        state, manifest = native._scale_checkpoint_paths(tmp_path, scale_index)
        assert state.is_file()
        assert manifest.is_file()
    assert not native._scale_checkpoint_paths(tmp_path, 2)[0].exists()
    assert not native._scale_checkpoint_paths(tmp_path, 2)[1].exists()

    resumed = _FakeEM()
    _, resumed_histories = native.checkpointed_multiscale(
        resumed,
        config=config,
        checkpoint_dir=tmp_path,
        lineage=lineage,
        resume=True,
    )
    assert [call["scale_tag"] for call in resumed.calls] == [2]
    assert resumed.calls[0]["n_iter"] == 30
    assert resumed.calls[0]["A"].shape == (4, 4)
    assert resumed.calls[0]["v"].shape == (1, 3, 2, 2, 2)
    assert resumed.calls[0]["A2d"].shape == (4, 3, 3)
    assert len(resumed_histories) == 3


def test_intentional_scale_two_stop_publishes_usable_product(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _checkpoint_config()
    fake = _FakeEM()
    transform_file = (
        Path("MRI_7T_WHOLE")
        / "HIST_NISSL_registered_to_MRI_7T_WHOLE"
        / "transforms"
        / "A.txt"
    )

    def write_transform_outputs(output_dir, final, mri, hist):
        path = Path(output_dir) / transform_file
        path.parent.mkdir(parents=True)
        path.write_text("normal pinned transform output", encoding="utf-8")

    fake.write_transform_outputs = write_transform_outputs
    xJ = config["xJ"][0]
    J = config["J"]
    W0 = config["W0"]
    rows = [{"physical_index": str(index)} for index in range(J.shape[1])]
    observed = np.arange(J.shape[1], dtype=np.int64)
    initializer = {
        "status": "complete",
        "type": "explicit_identity_per_physical_serial_position",
        "checksums": {},
    }
    monkeypatch.setattr(
        native,
        "load_native_stack",
        lambda dataset: (fake, rows, observed, xJ, J, W0),
    )
    monkeypatch.setattr(
        native,
        "_identity_section_initializer",
        lambda dataset, actual_observed, axes: (config["A2d"], initializer),
    )
    monkeypatch.setattr(
        native,
        "_load_native_mri",
        lambda em: (object(), config["I"], config["xI"][0]),
    )
    monkeypatch.setattr(
        native,
        "native_multiscale_configuration",
        lambda **kwargs: config,
    )
    monkeypatch.setattr(native.coarse, "LightImage", lambda *args: object())

    output = tmp_path / "fresh-corrected-registration"
    done = native._registration(
        native.CLEAN_SYMMETRIC_DATASET,
        None,
        output,
        stage="symmetric-registration",
        profile=native.FINAL_PROFILE,
        initial_A=np.eye(4),
        a2d_initialization="identity",
        stop_after_scale=2,
    )

    checkpoint = json.loads(
        (output / "checkpoints/registration.json").read_text(encoding="utf-8")
    )
    assert done == checkpoint
    assert checkpoint["status"] == "complete_through_scale"
    assert checkpoint["completed_scale_number"] == 2
    assert checkpoint["total_configured_scales"] == 3
    assert checkpoint["native_input_inplane_um"] == 200.0
    assert checkpoint["effective_inplane_um"] == 400.0
    assert checkpoint["final_configured_scale_executed"] is False
    product = output / native.THROUGH_SCALE_DIRECTORY
    assert Path(checkpoint["through_scale_product"]) == product
    assert (product / "numerical_outputs.npz").is_file()
    assert (product / transform_file).is_file()
    assert (product / "raw_Esave_level-1.npy").is_file()
    assert (product / "raw_Esave_level-2.npy").is_file()
    assert not (output / "registration/full_resolution_numerical_outputs.npz").exists()
    provenance = json.loads((product / "provenance.json").read_text())
    assert provenance["status"] == "complete_through_scale"
    assert provenance["native_input_inplane_um"] == 200.0
    assert provenance["effective_inplane_um"] == 400.0
    assert provenance["configured_final_scale_executed"] is False
    assert provenance["full_configured_profile_preserved"] is True
    assert provenance["effective_match_weight"] is None
    with np.load(product / "numerical_outputs.npz") as saved:
        assert {"A", "A2d", "v", "xv0", "xv1", "xv2"}.issubset(saved.files)
    assert [call["scale_tag"] for call in fake.calls] == [0, 1]
    assert native._scale_checkpoint_paths(output / "checkpoints", 0)[0].is_file()
    assert native._scale_checkpoint_paths(output / "checkpoints", 1)[0].is_file()
    assert not native._scale_checkpoint_paths(output / "checkpoints", 2)[0].exists()


def test_successful_scale_checkpoint_is_complete_and_self_describing(
    tmp_path: Path,
) -> None:
    config = _checkpoint_config()
    fake = _FakeEM(fail_scale=1)
    with pytest.raises(MemoryError):
        native.checkpointed_multiscale(
            fake, config=config, checkpoint_dir=tmp_path,
            lineage=_checkpoint_lineage(config), resume=False,
        )
    state_path, manifest_path = native._scale_checkpoint_paths(tmp_path, 0)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["completed_scale_index"] == 0
    assert manifest["downI"] == [4, 4, 4]
    assert manifest["downJ"] == [1, 4, 4]
    assert manifest["effective_resolution_um"]["J"] == [50.0, 800.0, 800.0]
    assert manifest["state_file_sha256"] == native.coarse.checksum(state_path)
    with np.load(state_path, allow_pickle=False) as saved:
        assert set(saved.files) == {"A", "A2d", "v", "xv0", "xv1", "xv2", "Esave"}
        assert saved["v"].dtype == np.float32
    assert not native._scale_checkpoint_paths(tmp_path, 1)[1].exists()
    assert not list(tmp_path.glob("registration_scale-*.tmp"))


def test_restart_from_scale_one_uses_original_later_profile_entries(
    tmp_path: Path,
) -> None:
    config = _checkpoint_config()
    lineage = _checkpoint_lineage(config)
    with pytest.raises(MemoryError):
        native.checkpointed_multiscale(
            _FakeEM(fail_scale=1), config=config, checkpoint_dir=tmp_path,
            lineage=lineage, resume=False,
        )
    resumed = _FakeEM()
    _, histories = native.checkpointed_multiscale(
        resumed, config=config, checkpoint_dir=tmp_path,
        lineage=lineage, resume=True,
    )
    assert [call["scale_tag"] for call in resumed.calls] == [1, 2]
    assert [call["n_iter"] for call in resumed.calls] == [20, 30]
    assert len(histories) == 3
    for key in ("A", "v", "A2d"):
        assert isinstance(resumed.calls[0][key], torch.Tensor)
        assert resumed.calls[0][key].device.type == "cpu"
        assert resumed.calls[0][key].dtype == torch.float32


def test_restart_from_scale_two_reruns_only_final_scale(tmp_path: Path) -> None:
    config = _checkpoint_config()
    lineage = _checkpoint_lineage(config)
    with pytest.raises(MemoryError):
        native.checkpointed_multiscale(
            _FakeEM(fail_scale=2), config=config, checkpoint_dir=tmp_path,
            lineage=lineage, resume=False,
        )
    resumed = _FakeEM()
    native.checkpointed_multiscale(
        resumed, config=config, checkpoint_dir=tmp_path,
        lineage=lineage, resume=True,
    )
    assert [call["scale_tag"] for call in resumed.calls] == [2]
    assert resumed.calls[0]["v"].shape == (1, 3, 2, 2, 2)


@pytest.mark.parametrize(
    "damage", ["missing", "checksum", "corrupt", "manifest", "nonfinite"]
)
def test_damaged_scale_state_is_not_eligible(
    tmp_path: Path, damage: str,
) -> None:
    config = _checkpoint_config()
    lineage = _checkpoint_lineage(config)
    with pytest.raises(MemoryError):
        native.checkpointed_multiscale(
            _FakeEM(fail_scale=1), config=config, checkpoint_dir=tmp_path,
            lineage=lineage, resume=False,
        )
    state_path, manifest_path = native._scale_checkpoint_paths(tmp_path, 0)
    if damage == "manifest":
        manifest_path.write_text("not json", encoding="utf-8")
    elif damage == "missing":
        state_path.unlink()
    elif damage == "checksum":
        with state_path.open("ab") as stream:
            stream.write(b"changed")
    elif damage == "corrupt":
        state_path.write_bytes(b"not an npz")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["state_file_sha256"] = native.coarse.checksum(state_path)
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    else:
        with np.load(state_path, allow_pickle=False) as saved:
            state = {key: np.asarray(saved[key]).copy() for key in saved.files}
        state["A"][0, 0] = np.inf
        np.savez_compressed(state_path, **state)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["state_file_sha256"] = native.coarse.checksum(state_path)
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    start, continuation, histories = native._resume_scale_state(
        tmp_path, config=config, lineage=lineage
    )
    assert (start, continuation, histories) == (0, None, [])


@pytest.mark.parametrize(
    ("key", "value"),
    [("profile_name", "other-profile"),
     ("source_dataset", "/other/source"),
     ("initializer_lineage_sha256", "other-initializer"),
     ("initializer_checksums", {"expanded_A2d.npy": "other-checksum"})],
)
def test_scale_checkpoint_lineage_mismatch_is_not_eligible(
    tmp_path: Path, key: str, value,
) -> None:
    config = _checkpoint_config()
    lineage = _checkpoint_lineage(config)
    with pytest.raises(MemoryError):
        native.checkpointed_multiscale(
            _FakeEM(fail_scale=1), config=config, checkpoint_dir=tmp_path,
            lineage=lineage, resume=False,
        )
    incompatible = dict(lineage)
    incompatible[key] = value
    start, continuation, histories = native._resume_scale_state(
        tmp_path, config=config, lineage=incompatible
    )
    assert (start, continuation, histories) == (0, None, [])


def test_restart_with_no_valid_scale_checkpoint_starts_at_level_one(
    tmp_path: Path,
) -> None:
    config = _checkpoint_config()
    fake = _FakeEM()
    native.checkpointed_multiscale(
        fake, config=config, checkpoint_dir=tmp_path,
        lineage=_checkpoint_lineage(config), resume=True,
    )
    assert [call["scale_tag"] for call in fake.calls] == [0, 1, 2]


def test_restart_does_not_skip_a_missing_earlier_scale(tmp_path: Path) -> None:
    config = _checkpoint_config()
    lineage = _checkpoint_lineage(config)
    with pytest.raises(MemoryError):
        native.checkpointed_multiscale(
            _FakeEM(fail_scale=2), config=config, checkpoint_dir=tmp_path,
            lineage=lineage, resume=False,
        )
    native._scale_checkpoint_paths(tmp_path, 0)[1].unlink()
    resumed = _FakeEM()
    native.checkpointed_multiscale(
        resumed, config=config, checkpoint_dir=tmp_path,
        lineage=lineage, resume=True,
    )
    assert [call["scale_tag"] for call in resumed.calls] == [0, 1, 2]
