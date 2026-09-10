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

from preprocess.run_allen_emlddmm import pinned_emlddmm
from preprocess import run_allen_emlddmm_full_resolution_nissl as native


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

    native._prepare_registration_output(
        tmp_path, restart_interrupted=True
    )

    assert registration.is_dir()
    assert not any(registration.iterdir())
    assert not registration_checkpoint.exists()
    assert initializer.read_bytes() == atlas_before
    assert atlas_checkpoint.read_bytes() == checkpoint_before


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


def test_lr_symmetry_is_coronal_column_flip_about_mri_midline() -> None:
    axis = native.symmetric_lr_axis(25.0, 724.0)
    assert len(axis) % 2 == 0
    assert np.allclose(np.diff(axis), 200.0)
    mid = len(axis) // 2
    np.testing.assert_allclose(axis[mid - 1 : mid + 1], [-75.0, 125.0])
    image = np.arange(3 * 2 * 4).reshape(3, 2, 4)
    support = np.arange(2 * 4).reshape(2, 4)
    reflected_image, reflected_support = native.reflect_observed_half(image, support)
    np.testing.assert_array_equal(reflected_image[:, :, :4], image[:, :, ::-1])
    np.testing.assert_array_equal(reflected_image[:, :, 4:], image)
    np.testing.assert_array_equal(reflected_support[:, :4], support[:, ::-1])
    np.testing.assert_array_equal(reflected_support[:, 4:], support)


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
