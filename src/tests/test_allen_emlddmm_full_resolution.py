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
