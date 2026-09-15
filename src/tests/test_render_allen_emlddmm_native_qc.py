from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile

from preprocess import render_allen_emlddmm_native_qc as qc
from preprocess.run_allen_emlddmm import pinned_emlddmm


def _numerical_arrays(serial: int = 3) -> dict[str, np.ndarray]:
    return {
        "A": np.arange(16, dtype=np.float32).reshape(4, 4),
        "A2d": np.arange(serial * 9, dtype=np.float32).reshape(serial, 3, 3),
        "v": np.arange(24, dtype=np.float32).reshape(1, 3, 2, 2, 2),
        "xv0": np.asarray([-1.0, 1.0], dtype=np.float32),
        "xv1": np.asarray([-2.0, 2.0], dtype=np.float32),
        "xv2": np.asarray([-3.0, 3.0], dtype=np.float32),
    }


def _histology_context(serial: int = 3) -> qc.HistologyContext:
    rows = [
        {"stain": "nissl" if index != 1 else "", "physical_index": str(index)}
        for index in range(serial)
    ]
    samples = [
        {
            "status": "present" if index != 1 else "missing",
            "sample_id": f"section_{index}.tif" if index != 1 else "",
        }
        for index in range(serial)
    ]
    return qc.HistologyContext(
        rows=rows,
        samples=samples,
        axes=[
            np.arange(serial, dtype=np.float64),
            np.arange(4, dtype=np.float64),
            np.arange(4, dtype=np.float64),
        ],
        observed=np.asarray([0, 2], dtype=np.int64),
    )


def _write_state_tree(
    tmp_path: Path,
) -> tuple[Path, Path, Path, dict[str, np.ndarray]]:
    checkpoints = tmp_path / "checkpoints"
    product = tmp_path / "registration_through_400um"
    checkpoints.mkdir()
    product.mkdir()
    numerical = product / "numerical_outputs.npz"
    arrays = _numerical_arrays()
    xi = [np.arange(4, dtype=np.float64) for _ in range(3)]
    xj = [
        np.arange(3, dtype=np.float64),
        np.arange(4, dtype=np.float64),
        np.arange(4, dtype=np.float64),
    ]
    np.savez(numerical, **arrays, **{f"xI{i}": x for i, x in enumerate(xi)},
             **{f"xJ{i}": x for i, x in enumerate(xj)})
    provenance = product / "provenance.json"
    provenance.write_text(
        json.dumps(
            {
                "status": "complete_through_scale",
                "completed_scale_index": 1,
                "completed_scale_number": 2,
                "effective_inplane_um": 400.0,
                "numerical": str(numerical.resolve()),
                "checksums": {
                    str(numerical.resolve()): qc.sha256_file(numerical)
                },
            }
        ),
        encoding="utf-8",
    )
    source_dataset = tmp_path / "symmetric-sections"
    source_dataset.mkdir()
    checkpoint = checkpoints / "registration.json"
    checkpoint.write_text(
        json.dumps(
            {
                "status": "complete_through_scale",
                "stage": "symmetric-registration",
                "completed_scale_index": 1,
                "completed_scale_number": 2,
                "effective_inplane_um": 400.0,
                "emlddmm_commit": qc.PIN,
                "profile": "unit-profile",
                "source_dataset": str(source_dataset.resolve()),
                "native_shapes": {"I": [1, 4, 4, 4], "J": [3, 3, 4, 4]},
                "numerical": str(numerical.resolve()),
                "provenance": str(provenance.resolve()),
            }
        ),
        encoding="utf-8",
    )
    scale = {
        "status": "complete",
        "completed_scale_index": 1,
        "completed_scale_number": 2,
        "emlddmm_commit": qc.PIN,
        "source_dataset": str(source_dataset.resolve()),
        "downI": [2, 2, 2],
        "downJ": [1, 2, 2],
        "effective_resolution_um": {
            "I": [400.0, 400.0, 400.0],
            "J": [50.0, 400.0, 400.0],
        },
        "native_shapes": {"I": [1, 4, 4, 4], "J": [3, 3, 4, 4]},
        "state_shapes": {key: list(value.shape) for key, value in arrays.items()},
        "state_dtypes": {key: value.dtype.name for key, value in arrays.items()},
    }
    (checkpoints / "registration_scale-02.json").write_text(
        json.dumps(scale), encoding="utf-8"
    )
    provenance_data = json.loads(provenance.read_text(encoding="utf-8"))
    provenance_data["source_dataset"] = str(source_dataset.resolve())
    provenance_data["lineage"] = {"emlddmm_commit": qc.PIN}
    provenance.write_text(json.dumps(provenance_data), encoding="utf-8")
    return checkpoint, numerical, source_dataset, arrays


def test_resolves_recorded_completed_scale_and_native_axes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint, _, source_dataset, arrays = _write_state_tree(tmp_path)
    mri_provenance = tmp_path / "mri_provenance.json"
    mri_provenance.write_text("{}", encoding="utf-8")
    mri = tmp_path / "mri.nii"
    mri.write_bytes(b"unit MRI")
    post = tmp_path / "postprocessed_qc"
    post.mkdir()
    histology = _histology_context()
    audited: list[list[int]] = []

    monkeypatch.setattr(qc, "CHECKPOINT", checkpoint)
    monkeypatch.setattr(qc, "DATASET", source_dataset)
    monkeypatch.setattr(qc, "MRI_PROVENANCE", mri_provenance)
    monkeypatch.setattr(qc, "MRI", mri)
    monkeypatch.setattr(qc, "POSTPROCESSED_QC", post)
    monkeypatch.setattr(qc, "EXPECTED_SOURCE_SHAPE", (1, 2, 2, 2))
    monkeypatch.setattr(qc, "EXPECTED_TARGET_SHAPE", (3, 3, 2, 2))
    monkeypatch.setattr(qc, "_load_histology_context", lambda: histology)
    monkeypatch.setattr(
        qc,
        "mri_physical_axes_from_provenance",
        lambda provenance: [np.arange(4, dtype=np.float64) for _ in range(3)],
    )
    monkeypatch.setattr(
        qc.coarse,
        "resolve_registration_execution",
        lambda name: {
            "downI": [[4, 4, 4], [2, 2, 2], [1, 1, 1]],
            "downJ": [[1, 4, 4], [1, 2, 2], [1, 1, 1]],
        },
    )
    monkeypatch.setattr(qc, "_available_memory_bytes", lambda: 100 * 1024**3)
    monkeypatch.setattr(
        qc,
        "_audit_weighted_section_equivalence",
        lambda em, context, factors: audited.append(list(factors)),
    )

    state = qc.resolve_state(object())

    assert state.down_i == [2, 2, 2]
    assert state.down_j == [1, 2, 2]
    assert state.source_shape == (1, 2, 2, 2)
    assert state.target_shape == (3, 3, 2, 2)
    assert audited == [[1, 2, 2]]
    for key in qc.REQUIRED_STATE:
        np.testing.assert_array_equal(state.arrays[key], arrays[key])


def test_sectionwise_weighted_downsample_matches_full_stack() -> None:
    em = pinned_emlddmm()
    rng = np.random.default_rng(42)
    image = rng.random((3, 4, 6, 8), dtype=np.float32)
    support = np.ones((4, 6, 8), dtype=np.float32)
    support[:, :2, :2] = 0.0
    image[:, 1] = 0.0
    support[1] = 0.0
    axes = [
        np.arange(4, dtype=np.float64),
        np.arange(6, dtype=np.float64),
        np.arange(8, dtype=np.float64),
    ]
    down = [1, 2, 2]

    full_axes, full_image, full_support = em.downsample_image_domain(
        axes, image, down, W=support
    )
    streamed = np.zeros_like(full_image)
    streamed_support = np.zeros_like(full_support)
    streamed_inplane = None
    for index in (0, 2, 3):
        section_axes = [axes[0][index : index + 1], axes[1], axes[2]]
        down_axes, down_image, down_support = qc._weighted_section_downsample(
            em, section_axes, image[:, index], support[index], down
        )
        streamed[:, index] = down_image[:, 0]
        streamed_support[index] = down_support[0]
        streamed_inplane = down_axes[1:]

    np.testing.assert_array_equal(streamed, full_image)
    np.testing.assert_array_equal(streamed_support, full_support)
    np.testing.assert_array_equal(full_axes[0], axes[0])
    assert streamed_inplane is not None
    for actual, expected in zip(streamed_inplane, full_axes[1:], strict=True):
        np.testing.assert_array_equal(actual, expected)


def test_sectionwise_weighted_downsample_rejects_nonshared_support_maximum() -> None:
    em = pinned_emlddmm()
    image = np.ones((3, 4, 4), dtype=np.float32)
    support = np.zeros((4, 4), dtype=np.float32)
    support[0, 0] = 1.0
    axes = [
        np.asarray([0.0]),
        np.arange(4, dtype=np.float64),
        np.arange(4, dtype=np.float64),
    ]

    with pytest.raises(RuntimeError, match="shared full-support maximum"):
        qc._weighted_section_downsample(
            em, axes, image, support, [1, 2, 2]
        )


class _FakeEM:
    def __init__(self, upstream: object):
        self.upstream = upstream
        self.downsample_calls: list[dict[str, object]] = []
        self.qc_calls: list[tuple[object, ...]] = []

    def downsample_image_domain(self, axes, image, factors, W=None):
        self.downsample_calls.append(
            {"factors": list(factors), "weighted": W is not None}
        )
        return self.upstream.downsample_image_domain(axes, image, factors, W=W)

    def write_qc_outputs(self, *args):
        self.qc_calls.append(args)
        assert len(args) == 4
        root = Path(args[0])
        for relative in qc.EXPECTED_QC_OUTPUTS:
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"upstream jpeg")

    def emlddmm(self, *args, **kwargs):
        raise AssertionError("registration must never be invoked")

    def emlddmm_multiscale(self, *args, **kwargs):
        raise AssertionError("registration must never be invoked")

    def write_transform_outputs(self, *args, **kwargs):
        raise AssertionError("transform output must never be invoked")


def _small_resolved_state(tmp_path: Path) -> qc.ResolvedState:
    arrays = _numerical_arrays()
    histology = _histology_context()
    checkpoint = tmp_path / "registration.json"
    numerical = tmp_path / "numerical.npz"
    provenance = tmp_path / "provenance.json"
    checkpoint.write_text("{}", encoding="utf-8")
    (tmp_path / "registration_scale-02.json").write_text("{}", encoding="utf-8")
    provenance.write_text("{}", encoding="utf-8")
    numerical.write_bytes(b"unit numerical product")
    post = tmp_path / "postprocessed_qc"
    post.mkdir()
    return qc.ResolvedState(
        checkpoint={"provenance": str(provenance)},
        scale_manifest={
            "native_shapes": {"I": [1, 4, 4, 4], "J": [3, 3, 4, 4]}
        },
        provenance={},
        numerical_path=numerical,
        arrays=arrays,
        native_xi=[np.arange(4, dtype=np.float64) for _ in range(3)],
        native_xj=histology.axes,
        histology=histology,
        mri_provenance={},
        down_i=[2, 2, 2],
        down_j=[1, 2, 2],
        source_shape=(1, 2, 2, 2),
        target_shape=(3, 3, 2, 2),
        source_dtype=np.dtype("float32"),
        target_dtype=np.dtype("float32"),
        estimated_memory_bytes=4096,
        mem_available_bytes=100 * 1024**3,
        stage_dir=post / ".emlddmm_upstream_qc_400um.tmp",
        final_dir=post / "emlddmm_upstream_qc_400um",
    )


def _write_small_sections(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    view = tmp_path / "view"
    support = tmp_path / "support"
    view.mkdir()
    support.mkdir()
    for index in (0, 2):
        image = np.full((4, 4, 3), index + 1, dtype=np.uint8)
        weight = np.ones((4, 4), dtype=np.float32)
        tifffile.imwrite(view / f"section_{index}.tif", image)
        tifffile.imwrite(support / f"section_{index}.tif", weight)
    monkeypatch.setattr(qc, "VIEW", view)
    monkeypatch.setattr(qc, "SUPPORT", support)


def test_run_calls_upstream_qc_directly_and_publishes_atomically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    upstream = pinned_emlddmm()
    fake = _FakeEM(upstream)
    state = _small_resolved_state(tmp_path)
    _write_small_sections(tmp_path, monkeypatch)
    source_image = SimpleNamespace(
        space="MRI_7T_WHOLE",
        name="7T_T1",
        title="",
        data=np.arange(64, dtype=np.float32).reshape(1, 4, 4, 4),
        x=state.native_xi,
        fnames=lambda: ["mri.nii"],
    )
    before = {key: value.copy() for key, value in state.arrays.items()}
    sibling = state.stage_dir.parent / "existing_qc.png"
    sibling.write_bytes(b"preserve")

    monkeypatch.setattr(qc, "CHECKPOINT", tmp_path / "registration.json")
    monkeypatch.setattr(qc, "pinned_emlddmm", lambda: fake)
    monkeypatch.setattr(qc, "resolve_state", lambda em: state)
    monkeypatch.setattr(qc, "load_pinned_mri_image", lambda *args, **kwargs: source_image)

    result = qc.run()

    assert result == state.final_dir
    assert state.final_dir.is_dir()
    assert not state.stage_dir.exists()
    assert sibling.read_bytes() == b"preserve"
    assert len(fake.qc_calls) == 1
    output_dir, output, source, target = fake.qc_calls[0]
    assert Path(output_dir) == state.stage_dir
    assert source.space == "MRI_7T_WHOLE"
    assert target.title == "slice_dataset"
    assert target.data.shape == state.target_shape
    assert [call["factors"] for call in fake.downsample_calls] == [
        [2, 2, 2],
        [1, 2, 2],
        [1, 2, 2],
    ]
    assert [call["weighted"] for call in fake.downsample_calls] == [False, True, True]
    for key in ("A", "A2d", "v"):
        np.testing.assert_array_equal(output[key].numpy(), before[key])
    for index, key in enumerate(("xv0", "xv1", "xv2")):
        np.testing.assert_array_equal(output["xv"][index].numpy(), before[key])
    provenance = json.loads(
        (state.final_dir / "adapter_provenance.json").read_text(encoding="utf-8")
    )
    assert provenance["write_qc_outputs_called_directly"] is True
    assert provenance["labels_passed"] is False


def test_dry_run_allocates_no_bulk_images_and_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = _small_resolved_state(tmp_path)
    calls: list[str] = []
    fake = object()
    monkeypatch.setattr(qc, "pinned_emlddmm", lambda: calls.append("pin") or fake)
    monkeypatch.setattr(qc, "resolve_state", lambda em: calls.append("resolve") or state)
    monkeypatch.setattr(
        qc, "_build_source", lambda *args: pytest.fail("bulk MRI allocation")
    )
    monkeypatch.setattr(
        qc, "_build_target", lambda *args: pytest.fail("bulk target allocation")
    )

    assert qc.run(dry_run=True) is None
    assert calls == ["pin", "resolve"]
    assert not state.stage_dir.exists()
    assert not state.final_dir.exists()


def test_existing_output_paths_abort_without_upstream_calls(tmp_path: Path) -> None:
    state = _small_resolved_state(tmp_path)
    state.final_dir.mkdir()

    with pytest.raises(FileExistsError, match="Refusing existing QC output"):
        qc._ensure_output_paths_available(state.stage_dir, state.final_dir)


def test_failed_upstream_qc_leaves_only_hidden_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = _small_resolved_state(tmp_path)
    source = qc.ImageFacade(
        "MRI_7T_WHOLE",
        "7T_T1",
        np.zeros(state.source_shape, dtype=np.float32),
        [np.arange(size) for size in state.source_shape[1:]],
        "",
        ["mri.nii"],
    )
    target = qc.ImageFacade(
        "HIST_NISSL",
        "HIST_NISSL",
        np.zeros(state.target_shape, dtype=np.float32),
        [np.arange(size) for size in state.target_shape[1:]],
        "slice_dataset",
        [str(index) for index in range(state.target_shape[1])],
    )

    class FailingEM:
        def write_qc_outputs(self, output_dir, output, actual_source, actual_target):
            partial = Path(output_dir) / qc.EXPECTED_QC_OUTPUTS[0]
            partial.parent.mkdir(parents=True)
            partial.write_bytes(b"partial")
            raise MemoryError("simulated upstream failure")

    fake = FailingEM()
    monkeypatch.setattr(qc, "pinned_emlddmm", lambda: fake)
    monkeypatch.setattr(qc, "resolve_state", lambda em: state)
    monkeypatch.setattr(qc, "_build_source", lambda em, resolved: source)
    monkeypatch.setattr(qc, "_build_target", lambda em, resolved: target)

    with pytest.raises(MemoryError, match="simulated upstream failure"):
        qc.run()

    assert state.stage_dir.is_dir()
    assert not state.final_dir.exists()
