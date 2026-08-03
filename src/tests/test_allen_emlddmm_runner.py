from __future__ import annotations

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


PROJECT_ROOT = Path(__file__).resolve().parents[2]


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
