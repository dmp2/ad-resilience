from __future__ import annotations

import ast
import inspect
import json
import tarfile
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from preprocess.prepare_allen_7t_mri import (
    CORRECTED_VOXEL_SIZE_MM,
    MGH_DELTA_END,
    MGH_DELTA_OFFSET,
    _correct_mgz_header,
    _direction_cosines,
    _physical_center,
    _nifti_voxel_sha256,
    _validate_lossless_nifti,
    _write_lossless_nifti,
    inspect_archive,
    select_candidate,
    validate_archive_members,
)
from preprocess.prepare_allen_emlddmm_inputs import sha256_file
from preprocess.run_allen_emlddmm import (
    estimate_registration_bytes,
    load_pinned_mri_image,
    load_reviewed_initial_affine,
    pinned_emlddmm,
    require_unused_output_root,
    validate_loaded_support,
    validate_registration_domain,
    validate_registration_contract,
    validated_config,
)
from preprocess.run_allen_emlddmm_full_coarse_nissl import (
    _emlddmm_stack_draw_qc,
    _registration_initial_affine,
    coarse_spatial_axes,
    stream_stack,
    _sample_chain_slabs,
    _save_final_effective_match_weight,
    _validate_published_affine,
    _warp_effective_match_weight,
    resolve_registration_execution,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]




def test_preserved_source_grid_loads_directly_without_preliminary_downsample(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tifffile
    import preprocess.run_allen_emlddmm_full_coarse_nissl as runner

    view = tmp_path / "view"
    view.mkdir()
    raster = np.arange(130 * 108 * 3, dtype=np.uint32).reshape(130, 108, 3)
    raster = (raster % 254 + 1).astype(np.uint8)
    tifffile.imwrite(view / "section.tif", raster)
    monkeypatch.setattr(runner, "VIEW", view)
    row = np.arange(130, dtype=np.float64) * 813.851708 - 361346.5
    column = np.arange(108, dtype=np.float64) * 813.851708 + 436782.8
    serial = np.array([0.0])
    working = coarse_spatial_axes(
        [serial, row, column], preserve_source_grid=True
    )
    np.testing.assert_array_equal(working[0], row)
    np.testing.assert_array_equal(working[1], column)
    J, W = stream_stack(
        [{"sample_id": "section.tif"}], np.array([0]), 1,
        shape=(130, 108), spatial_axes=[row, column],
        preserve_source_grid=True,
    )
    assert J.shape == (3, 1, 130, 108)
    assert W.shape == (1, 130, 108)
    np.testing.assert_array_equal(
        J[:, 0], raster.transpose(2, 0, 1).astype(np.float32) / np.float32(255.0)
    )
    np.testing.assert_array_equal(W[0], 1.0)


def test_preserved_source_grid_accepts_inherited_linear_affine(
    tmp_path: Path,
) -> None:
    lineage = tmp_path / "histology_linear_nissl"
    (lineage / "metadata").mkdir(parents=True)
    affine = np.array([
        [0.0, -1.0, 0.0, 419900.0],
        [0.0, 0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    (lineage / "metadata/linear_restack.json").write_text(json.dumps({
        "global_affine_mri_um_to_histology_um": affine.tolist()
    }))
    dataset = tmp_path / "emlddmm_7t_symmetric"
    (dataset / "metadata").mkdir(parents=True)
    (dataset / "metadata/symmetry.json").write_text(json.dumps({
        "source_dataset": str(lineage)
    }))
    initialization = tmp_path / "full_coarse_numerical_outputs.npz"
    np.savez(initialization, A=affine, ignored=np.ones(1))
    np.testing.assert_array_equal(
        _registration_initial_affine(
            initialization, dataset, preserve_source_grid=True
        ),
        affine,
    )
    np.savez(initialization, A=affine + np.eye(4))
    with pytest.raises(RuntimeError, match="differs from symmetric-source lineage"):
        _registration_initial_affine(
            initialization, dataset, preserve_source_grid=True
        )

def test_final_effective_match_weight_retention_is_selective(
    tmp_path: Path,
) -> None:
    profile = "example-standard-sigmaR5e4-a2000-dv4000-lc188-linear-no-v"
    config = resolve_registration_execution(profile)
    assert config["full_outputs"] == [False, False, True]
    assert config["n_draw"] == [0, 0, 0]

    observed = np.arange(641, dtype=np.int64)
    matching = np.linspace(0.0, 1.0, 641 * 2 * 3, dtype=np.float32).reshape(
        641, 2, 3
    )
    support = np.linspace(1.0, 0.0, 641 * 2 * 3, dtype=np.float32).reshape(
        641, 2, 3
    )
    final = {
        "WM": matching, "W0": support,
        "WA": np.zeros_like(matching), "WB": np.zeros_like(matching),
    }
    output = tmp_path / "final_observed_effective_match_weight.npy"
    assert _save_final_effective_match_weight(
        final, observed, matching.shape, output
    ) == output
    retained = np.load(output)
    assert retained.dtype == np.float32
    np.testing.assert_array_equal(retained, (matching * support)[observed])
    assert set(final) == {"WM", "W0", "WA", "WB"}

    import preprocess.run_allen_emlddmm_full_coarse_nissl as runner
    tree = ast.parse(inspect.getsource(runner.registration))
    save_call = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "savez_compressed"
    )
    assert {keyword.arg for keyword in save_call.keywords} == {
        "A", "A2d", "v", "xv0", "xv1", "xv2",
        "xI0", "xI1", "xI2", "xJ0", "xJ1", "xJ2", "observed",
    }


def test_emlddmm_stack_draw_qc_normalizes_supported_rgb(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import matplotlib.pyplot as plt
    import preprocess.run_allen_emlddmm_full_coarse_nissl as runner

    class Drawer:
        captured = None

        @classmethod
        def draw(cls, image, **kwargs):
            cls.captured = (np.asarray(image).copy(), kwargs)
            return plt.figure(), np.empty((3, 5), dtype=object)

    published = {}

    def fake_atomic(path, figure, *, dpi=160):
        published["path"] = path
        published["title"] = figure._suptitle.get_text()
        plt.close(figure)

    monkeypatch.setattr(runner, "_atomic_figure", fake_atomic)
    support = np.array([[[1.0, 2.0], [0.0, 4.0]]], dtype=np.float32)
    normalized = np.array([
        [[[0.1, 0.2], [0.0, 0.4]]],
        [[[0.5, 0.6], [0.0, 0.8]]],
        [[[0.9, 1.0], [0.0, 0.3]]],
    ], dtype=np.float32)
    numerator = normalized * support[None]
    axes = [np.array([10.0]), np.array([20.0, 21.0]), np.array([30.0, 31.0])]
    output = tmp_path / "stack.png"
    assert _emlddmm_stack_draw_qc(
        Drawer, numerator, support, axes, output, "matched stack"
    ) == output
    image, kwargs = Drawer.captured
    np.testing.assert_array_equal(image, normalized)
    assert kwargs == {
        "xJ": axes, "n_slices": 5, "disp": False,
        "interpolation": "none", "vmin": 0, "vmax": 1,
    }
    assert published == {"path": output, "title": "matched stack"}


def test_effective_match_weight_follows_a2d_and_mri_sampling_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import preprocess.run_allen_emlddmm_full_coarse_nissl as runner

    monkeypatch.setattr(runner, "RUN_TMP", tmp_path)
    axes = [
        np.array([0.0, 1.0]),
        np.array([0.0, 1.0, 2.0]),
        np.array([0.0, 1.0, 2.0, 3.0]),
    ]
    observed = np.array([0, 1], dtype=np.int64)
    final = np.repeat(np.eye(3)[None], 2, axis=0)
    source = np.linspace(0.0, 1.0, 2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    registered = _warp_effective_match_weight(
        source, observed, axes, final, axes[1], axes[2]
    )
    np.testing.assert_allclose(registered, source)

    support = np.ones((2, 3, 4), dtype=np.float32)
    numerator = np.stack([support * value for value in (0.2, 0.4, 0.6)])
    phi = np.stack(np.meshgrid(*axes, indexing="ij"))
    reconstruction, propagated_support, propagated_effective = _sample_chain_slabs(
        np.eye(4), phi, axes, axes, axes, axes[1], axes[2],
        numerator, support, registered,
    )
    np.testing.assert_allclose(propagated_effective, source)
    np.testing.assert_allclose(propagated_support, support)
    np.testing.assert_allclose(reconstruction[..., 0], 0.2)


def test_mri_nissl_overview_uses_effective_weight_only_for_display(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    import matplotlib.pyplot as plt
    import preprocess.run_allen_emlddmm_full_coarse_nissl as runner

    captured = []

    def fake_atomic(path, figure, *, dpi=160):
        captured.append((path.name, figure))

    monkeypatch.setattr(runner, "_atomic_figure", fake_atomic)
    xI = [
        np.arange(5, dtype=np.float64) * 2.0 + 10.0,
        np.arange(11, dtype=np.float64) * 3.0 + 20.0,
        np.arange(7, dtype=np.float64) * 4.0 + 30.0,
    ]
    mri = np.linspace(0.0, 1.0, 5 * 11 * 7, dtype=np.float32).reshape(
        1, 5, 11, 7
    )
    reconstruction = np.full((5, 11, 7, 3), 0.4, dtype=np.float32)
    support = np.ones((5, 11, 7), dtype=np.float32)
    support[[0, -1], :, :] = 0.0
    support[:, :, [0, -1]] = 0.0
    effective = np.linspace(0.0, 1.0, 5 * 11 * 7, dtype=np.float32).reshape(
        5, 11, 7
    )

    runner._mri_nissl_figures(
        mri, xI, reconstruction, support, effective, mri_midline_um=13.0
    )
    assert [name for name, _ in captured] == [
        "mri_nissl_registration_overview.png",
        "registered_nissl_orthogonal_overview.png",
    ]
    overview = captured[0][1]
    assert len(overview.axes) == 36
    for row in range(4):
        for column in range(9):
            axis = overview.axes[row * 9 + column]
            assert len(axis.collections) == 0
            expected_lines = 0 if row == 2 else 1
            assert len(axis.lines) == expected_lines
            if expected_lines:
                np.testing.assert_allclose(
                    axis.lines[0].get_xdata(), [1.5, 1.5]
                )
                assert axis.lines[0].get_linestyle() == "--"
                assert axis.lines[0].get_color() == "cyan"
    for column, axis in enumerate(overview.axes[27:]):
        line_positions = [
            overview.axes[row * 9 + column].lines[0].get_xdata()
            for row in (0, 1, 3)
        ]
        np.testing.assert_allclose(line_positions, np.full((3, 2), 1.5))
        base = np.asarray(overview.axes[column].images[0].get_array())
        nissl = np.asarray(overview.axes[9 + column].images[0].get_array())
        weight = np.asarray(overview.axes[18 + column].images[0].get_array())
        overlay = np.asarray(axis.images[0].get_array())
        departure_from_white = np.sqrt(
            np.mean((1.0 - nissl) ** 2, axis=-1)
        )
        alpha = weight * departure_from_white
        np.testing.assert_allclose(
            overlay,
            (1.0 - alpha[..., None]) * base[..., None]
            + alpha[..., None] * np.array([1.0, 0.0, 0.7]),
        )
    assert [text.get_text() for text in overview.texts] == [
        "MRI",
        "Registered Nissl",
        "Effective match weight (WM × W0)",
        "MRI + weighted Nissl",
        "cyan dashed line = MRI midsagittal plane",
        "Saved-transform MRI/Nissl registration overview — manual anatomical review required",
    ]
    assert len(captured[1][1].axes) == 6
    for _, figure in captured:
        plt.close(figure)


def test_emlddmm_stack_draw_qc_rejects_bad_support(tmp_path: Path) -> None:
    numerator = np.zeros((3, 2, 3, 4), dtype=np.float32)
    axes = [np.arange(2), np.arange(3), np.arange(4)]
    with pytest.raises(RuntimeError, match="nonnegative"):
        _emlddmm_stack_draw_qc(
            object(), numerator, -np.ones((2, 3, 4)), axes,
            tmp_path / "bad.png", "bad",
        )


def test_published_restack_affine_is_the_sampling_chain_affine() -> None:
    affine = np.array([
        [0.0, -1.0, 0.0, 420000.0],
        [0.0, 0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    rows = [
        {"serial_z_center_mm": "-1.0"},
        {"serial_z_center_mm": "0.0"},
        {"serial_z_center_mm": "1.0"},
    ]
    published = _validate_published_affine(
        affine,
        rows,
        np.arange(3),
        np.array([-321000.0, -320000.0, -319000.0]),
        np.array([479000.0, 480000.0, 481000.0]),
        np.array([480000.0, 420000.0, 320000.0]),
    )
    np.testing.assert_array_equal(published, affine)

    baseline = np.array([
        [np.cos(0.01), -np.sin(0.01), 315000.0],
        [np.sin(0.01), np.cos(0.01), -483000.0],
        [0.0, 0.0, 1.0],
    ])
    reapplied = np.eye(4)
    reapplied[1:3, 1:3] = np.linalg.inv(baseline)[:2, :2]
    reapplied[1:3, 3] = np.linalg.inv(baseline)[:2, 2]
    with pytest.raises(RuntimeError, match="outside published registered stack"):
        _validate_published_affine(
            reapplied @ affine,
            rows,
            np.arange(3),
            np.array([-321000.0, -320000.0, -319000.0]),
            np.array([479000.0, 480000.0, 481000.0]),
            np.array([480000.0, 420000.0, 320000.0]),
        )


def test_multiscale_contrast_argument_is_nested() -> None:
    path = (
        PROJECT_ROOT
        / "configs/emlddmm/allen_708424_mri7t_to_hist_all_pilot.json"
    )
    config = validated_config(path, "pilot")
    assert config["slice_matching"] is True
    assert config["order"] == 1
    assert config["local_contrast"] == [[]]
    assert config["full_outputs"] is True
    assert config["n_draw"] == 0

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["local_contrast"] = []
    bad_path = path.parent / "_not_written.json"
    # Exercise validation without creating another repository file.
    with pytest.raises(ValueError, match="local_contrast"):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory) / "bad.json"
            temporary.write_text(json.dumps(payload), encoding="utf-8")
            validated_config(temporary, "pilot")
    assert not bad_path.exists()


def test_full_runner_disables_full_voxel_outputs() -> None:
    path = (
        PROJECT_ROOT
        / "configs/emlddmm/allen_708424_mri7t_to_hist_all_full.json"
    )
    config = validated_config(path, "full")
    assert config["full_outputs"] is False
    assert config["n_draw"] == 0


def test_symmetric_histology_to_whole_t1_contract_direction() -> None:
    path = (
        PROJECT_ROOT
        / "configs/emlddmm/allen_708424_hist_symmetric_to_mri7t_t1.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["logical_edge"] == {
        "source_space": "HIST_SYMMETRIC",
        "source_view": "HIST_ALL",
        "target_space": "MRI_7T_WHOLE",
        "target_contrast": "7T_T1",
        "direction": "source_to_target",
    }
    assert payload["algorithm_adapter"]["I_role"] == "graph_target"
    assert payload["algorithm_adapter"]["J_role"] == "graph_source"
    assert payload["synthetic_mri_required"] is False
    assert payload["initial_transform"]["initial_midlines_must_coincide"] is False
    assert payload["registration_execution_enabled"] is False
    assert not {
        "dimensions",
        "voxel_size_mm",
        "orientation",
        "affine",
    }.intersection(payload["inputs"])
    provenance = PROJECT_ROOT / payload["inputs"]["mri_provenance"]
    if provenance.is_file():
        validated = validate_registration_contract(path)
        assert validated["resolved_mri_provenance"]["status"] == "ready"


def test_pinned_mri_adapter_uses_provenance_affine_in_micrometers(
    tmp_path: Path,
) -> None:
    path = tmp_path / "mri.nii"
    data = np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5)
    affine = np.diag([0.2, 0.2, 0.2, 1.0])
    affine[:3, 3] = [10.0, 20.0, 30.0]
    nib.save(nib.Nifti1Image(data, affine), path)
    provenance = {
        "status": "ready",
        "units": "millimeter",
        "dimensions": [3, 4, 5],
        "voxel_to_physical_affine_mm": affine.tolist(),
    }
    module = pinned_emlddmm()
    original_load = module.nibabel.load
    image = load_pinned_mri_image(
        module, mri_path=path, provenance=provenance
    )
    assert module.nibabel.load is original_load
    assert image.data.shape == (1, 3, 4, 5)
    assert image.coordinate_units == "um"
    np.testing.assert_allclose(image.x[0], [10000.0, 10200.0, 10400.0])
    np.testing.assert_allclose(image.x[1], [20000.0, 20200.0, 20400.0, 20600.0])
    np.testing.assert_allclose(
        image.x[2], [30000.0, 30200.0, 30400.0, 30600.0, 30800.0]
    )


def test_memory_estimate_accounts_for_pilot_retention() -> None:
    hist = (30, 100, 80)
    mri = (40, 50, 60)
    pilot = estimate_registration_bytes(hist, mri, full_outputs=True)
    full = estimate_registration_bytes(hist, mri, full_outputs=False)
    assert pilot > full > 0


def test_support_validation_is_positional_and_not_anatomical() -> None:
    support = np.zeros((3, 4, 5), dtype=np.float32)
    support[0, 1:3, 1:4] = 1
    support[2, 0:2, 0:2] = 1
    rows = [
        {"sample_id": "a.tif", "status": "present"},
        {"sample_id": "absent.tif", "status": "absent"},
        {"sample_id": "b.tif", "status": "present"},
    ]
    assert validate_loaded_support(support, rows) == {
        "present": 2,
        "absent": 1,
    }
    support[1, 0, 0] = 1
    with pytest.raises(ValueError, match="nonzero W0"):
        validate_loaded_support(support, rows)


def test_reviewed_initialization_and_duplicate_edge_gate(tmp_path: Path) -> None:
    matrix_path = tmp_path / "initialization.txt"
    np.savetxt(matrix_path, np.eye(4))
    review = {
        "review_status": "accepted",
        "source_image_identifier": "flash20",
        "target_view": "HIST_ALL",
        "matrix_sha256": sha256_file(matrix_path),
    }
    matrix_path.with_suffix(".json").write_text(
        json.dumps(review), encoding="utf-8"
    )
    matrix, loaded_review = load_reviewed_initial_affine(
        matrix_path,
        source_image_identifier="flash20",
        target_view="HIST_ALL",
    )
    np.testing.assert_allclose(matrix, np.eye(4))
    assert loaded_review == review

    output = tmp_path / "registration"
    output.mkdir()
    require_unused_output_root(output)
    (output / "A.txt").write_text("existing", encoding="utf-8")
    with pytest.raises(FileExistsError, match="graph edge"):
        require_unused_output_root(output)


def test_registration_domain_is_exact_and_not_recentered() -> None:
    pilot_rows = [
        {
            "section_number": str(section),
            "status": "present" if section in range(1448, 1478, 4) else "absent",
            "stain": "nissl" if section in range(1448, 1478, 4) else "",
        }
        for section in range(1448, 1478)
    ]
    validate_registration_domain(pilot_rows, "pilot", "nissl")
    pilot_rows[0]["section_number"] = "36"
    with pytest.raises(ValueError, match="1448-1477"):
        validate_registration_domain(pilot_rows, "pilot", "nissl")


def _test_mgz(tmp_path: Path, name: str = "T1_rot.mgz") -> Path:
    source = tmp_path / name
    data = np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5)
    nib.save(nib.MGHImage(data, np.eye(4)), source)
    return source


def test_archive_rejects_unsafe_member_paths() -> None:
    member = tarfile.TarInfo("../escape.mgz")
    member.size = 1
    with pytest.raises(ValueError, match="Unsafe archive member path"):
        validate_archive_members([member])

    link = tarfile.TarInfo("7T/link.mgz")
    link.type = tarfile.SYMTYPE
    link.linkname = "/tmp/target"
    with pytest.raises(ValueError, match="Unsafe archive link"):
        validate_archive_members([link])


def test_locked_t1_selection_is_independent_of_archive_order() -> None:
    t1 = {"member_path": "7T/T1_rot.mgz"}
    pd = {"member_path": "7T/PD_rot.mgz"}
    assert select_candidate([pd, t1]) is t1
    assert select_candidate([t1, pd]) is t1
    with pytest.raises(RuntimeError, match="ambiguous"):
        select_candidate([pd, t1], selected_basename=None)


def test_corrupt_archive_fails(tmp_path: Path) -> None:
    archive = tmp_path / "corrupt.tgz"
    archive.write_bytes(b"not a tar archive")
    with pytest.raises(ValueError, match="Unreadable or corrupt"):
        inspect_archive(archive)


def test_header_geometry_correction_preserves_array_and_center(
    tmp_path: Path,
) -> None:
    source = _test_mgz(tmp_path)
    corrected = tmp_path / "corrected.mgz"
    _correct_mgz_header(source, corrected)

    source_image = nib.load(source)
    corrected_image = nib.load(corrected)
    assert source_image.shape == corrected_image.shape
    assert source_image.get_data_dtype() == corrected_image.get_data_dtype()
    assert np.array_equal(
        np.asanyarray(source_image.dataobj),
        np.asanyarray(corrected_image.dataobj),
    )
    np.testing.assert_allclose(
        corrected_image.header.get_zooms()[:3], CORRECTED_VOXEL_SIZE_MM
    )
    np.testing.assert_allclose(
        _direction_cosines(source_image), _direction_cosines(corrected_image)
    )
    np.testing.assert_allclose(
        _physical_center(source_image), _physical_center(corrected_image)
    )
    assert nib.aff2axcodes(source_image.affine) == nib.aff2axcodes(
        corrected_image.affine
    )


def test_only_header_spacing_bytes_change_before_recompression(
    tmp_path: Path,
) -> None:
    import gzip

    source = _test_mgz(tmp_path)
    corrected = tmp_path / "corrected.mgz"
    _correct_mgz_header(source, corrected)
    with gzip.open(source, "rb") as stream:
        source_payload = stream.read()
    with gzip.open(corrected, "rb") as stream:
        corrected_payload = stream.read()
    assert source_payload[:MGH_DELTA_OFFSET] == corrected_payload[:MGH_DELTA_OFFSET]
    assert source_payload[MGH_DELTA_END:] == corrected_payload[MGH_DELTA_END:]
    assert source_payload[MGH_DELTA_OFFSET:MGH_DELTA_END] != (
        corrected_payload[MGH_DELTA_OFFSET:MGH_DELTA_END]
    )


def test_lossless_nifti_normalization_preserves_corrected_voxels(
    tmp_path: Path,
) -> None:
    source = _test_mgz(tmp_path)
    corrected = tmp_path / "corrected.mgz"
    normalized = tmp_path / "corrected.nii"
    _correct_mgz_header(source, corrected)
    _write_lossless_nifti(corrected, normalized)
    voxel_hash = _nifti_voxel_sha256(normalized)
    _validate_lossless_nifti(corrected, normalized, voxel_hash)
    source_image = nib.load(corrected)
    normalized_image = nib.load(normalized)
    np.testing.assert_array_equal(
        np.asanyarray(source_image.dataobj),
        np.asanyarray(normalized_image.dataobj),
    )


def test_archive_inspection_finds_readable_t1(tmp_path: Path) -> None:
    source = _test_mgz(tmp_path)
    archive = tmp_path / "direct-7t.tgz"
    with tarfile.open(archive, "w:gz") as bundle:
        bundle.add(source, arcname="7T/T1_rot.mgz")
    candidates = inspect_archive(archive)
    assert candidates == [
        {
            "member_path": "7T/T1_rot.mgz",
            "format": "mgz",
            "dimensions": [3, 4, 5],
            "voxel_size_mm": [1.0, 1.0, 1.0],
            "datatype_code": 3,
            "frames": 1,
        }
    ]
