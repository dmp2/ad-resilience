from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")
import numpy as np
from PIL import Image

from preprocess.prepare_allen_emlddmm_inputs import (
    PHYSICAL_FIELDS,
    SAMPLE_PROVENANCE_FIELDS,
    canonical_z_um,
    secondary_envelope,
)
from preprocess import visualize_allen_emlddmm_stack as visual


SYNTHETIC_COUNTS = {"nissl": 2, "pv": 1}


def _write_tsv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=fields, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def write_synthetic_dataset(
    root: Path,
    *,
    canvas_key: str = "accepted_canvas_shape_yx",
    canvas_shape: list[int] | None = None,
) -> Path:
    canvas_shape = canvas_shape or [7, 9]
    metadata = root / "metadata"
    view = root / "inputs/views/HIST_ALL"
    metadata.mkdir(parents=True)
    view.mkdir(parents=True)

    definitions = {
        0: ("nissl", 3, 4, 255, 5, 3, 100.0),
        2: ("pv", 4, 3, 128, 4, 4, 200.0),
        4: ("nissl", 5, 2, 64, 3, 5, 300.0),
    }
    physical_rows: list[dict[str, str]] = []
    sample_rows: list[dict[str, str]] = []
    raw_manifest_rows: list[dict[str, str]] = []
    for grid_index in range(5):
        section = 36 + grid_index
        block_id, envelope_context = secondary_envelope(section) or (
            "",
            "between_observation_envelopes",
        )
        row = {field: "" for field in PHYSICAL_FIELDS}
        row.update(
            {
                "physical_index": str(grid_index),
                "specimen_id": "708424",
                "allen_section_number": str(section),
                "serial_z_center_mm": f"{canonical_z_um(section) / 1000:.3f}",
                "section_thickness_um": "50.0",
                "serial_pitch_um": "50.0",
                "block_id": block_id,
                "block_assignment_source": "secondary_reconstruction_table",
                "secondary_envelope_context": envelope_context,
            }
        )
        if grid_index in definitions:
            (
                stain,
                width,
                height,
                value,
                source_width,
                source_height,
                source_spacing,
            ) = definitions[grid_index]
            sample_id = f"allen_708424_{stain}_{grid_index + 1:04d}.tif"
            source_relative = Path(stain) / f"source_{section:04d}.jpg"
            row.update(
                {
                    "image_present": "true",
                    "stain": stain,
                    "allen_section_image_id": str(100000 + section),
                    "allen_data_set_id": str(200000 + section),
                    "block_assignment_status": "matched_secondary_observation",
                    "source_relative_path": str(source_relative),
                    "prepared_relative_path": f"inputs/sections/{stain}/{sample_id}",
                    "source_sha256": "a" * 64,
                    "prepared_sha256": "b" * 64,
                    "nominal_series_interval_um": (
                        "200.0" if stain == "nissl" else "400.0"
                    ),
                    "observation_class": f"observed_{stain}",
                }
            )
            array = np.zeros((*canvas_shape, 3), dtype=np.uint8)
            array[:height, :width] = value
            Image.fromarray(array).save(view / sample_id)
            (view / sample_id).with_suffix(".json").write_text(
                json.dumps(
                    {
                        "DataFile": sample_id,
                        "Sizes": [3, canvas_shape[1], canvas_shape[0], 1],
                        "SpaceDirections": [
                            "none",
                            [200.0, 0.0, 0.0],
                            [0.0, 200.0, 0.0],
                            [0.0, 0.0, 50.0],
                        ],
                        "SpaceOrigin": [
                            -(canvas_shape[1] - 1) * 100.0,
                            -(canvas_shape[0] - 1) * 100.0,
                            canonical_z_um(section),
                        ],
                    }
                ),
                encoding="utf-8",
            )
            source_path = root / "raw" / source_relative
            source_path.parent.mkdir(parents=True, exist_ok=True)
            source = np.full(
                (source_height, source_width, 3), value, dtype=np.uint8
            )
            Image.fromarray(source).save(source_path)
            raw_manifest_rows.append(
                {
                    "kind": "histology_jpeg",
                    "allen_section_image_id": str(100000 + section),
                    "width_px": str(source_width),
                    "height_px": str(source_height),
                    "pixel_size_um": str(source_spacing),
                }
            )
            status = "present"
        else:
            sample_id = f"allen_708424_absent_{grid_index + 1:04d}.tif"
            row.update(
                {
                    "image_present": "false",
                    "block_assignment_status": "inferred_within_secondary_envelope",
                    "observation_class": "unobserved",
                }
            )
            status = "absent"
        physical_rows.append(row)
        sample = {
            "sample_id": sample_id,
            "participant_id": "708424",
            "species": "Homo sapiens",
            "status": status,
        }
        sample.update({field: row[field] for field in SAMPLE_PROVENANCE_FIELDS})
        sample_rows.append(sample)

    _write_tsv(metadata / "physical_sections.tsv", PHYSICAL_FIELDS, physical_rows)
    _write_tsv(
        view / "samples.tsv",
        [
            "sample_id",
            "participant_id",
            "species",
            "status",
            *SAMPLE_PROVENANCE_FIELDS,
        ],
        sample_rows,
    )
    _write_tsv(
        root / "raw/metadata/manifest.tsv",
        [
            "kind",
            "allen_section_image_id",
            "width_px",
            "height_px",
            "pixel_size_um",
        ],
        raw_manifest_rows,
    )
    (metadata / "loader_canvas_audit.json").write_text(
        json.dumps(
            {
                canvas_key: canvas_shape,
                "target_spacing_um": 200.0,
                "sectionwise_centering": False,
                "global_translation_xy_um": [
                    -(canvas_shape[1] - 1) * 100.0,
                    -(canvas_shape[0] - 1) * 100.0,
                ],
            }
        ),
        encoding="utf-8",
    )
    return root


def _load_synthetic(
    dataset: Path, stage: str = "prepared"
) -> visual.StackInputs:
    return visual.load_stack_inputs(
        dataset,
        stage=stage,
        data_dir=dataset / "raw",
        expected_slots=5,
        expected_stain_counts=SYNTHETIC_COUNTS,
    )


def test_canvas_axis_is_selected_by_label_not_hard_coded_position() -> None:
    assert (
        visual.accepted_canvas_axis_length(
            {"accepted_canvas_shape_yx": [7, 11]}, "x"
        )
        == 11
    )
    assert (
        visual.accepted_canvas_axis_length(
            {"accepted_canvas_shape_xy": [11, 7]}, "x"
        )
        == 11
    )


def test_comparison_statuses_follow_local_gate_and_annex_content(
    tmp_path: Path,
) -> None:
    config = tmp_path / "mri.json"
    config.write_text(json.dumps({"geometry_status": "blocked"}), encoding="utf-8")
    openneuro = tmp_path / "openneuro"
    openneuro.mkdir()
    (openneuro / "missing.nii.gz").symlink_to("annex/object/not-present")
    assert visual.comparison_statuses(
        mri_config=config, openneuro_root=openneuro
    ) == (
        "unavailable_provenance_gate_blocked",
        "unavailable_local_annex_content_absent",
    )
    (openneuro / "present.nii.gz").write_bytes(b"representative payload")
    assert visual.comparison_statuses(
        mri_config=config, openneuro_root=openneuro
    )[1] == "available_not_yet_validated"


def test_positional_validation_and_representative_selection(tmp_path: Path) -> None:
    inputs = _load_synthetic(write_synthetic_dataset(tmp_path / "dataset"))
    assert len(inputs.rows) == 5
    assert inputs.x_um.size == 9
    np.testing.assert_allclose(
        inputs.z_um, [canonical_z_um(section) for section in range(36, 41)]
    )
    np.testing.assert_array_equal(inputs.occupancy, [1, 0, 2, 0, 1])
    assert inputs.prepared_canvas_center_extent_yx_mm == (1.2, 1.6)
    assert visual.section_gap_counts(inputs) == {2: 2}
    assert visual.section_gap_counts(inputs, "nissl") == {4: 1}
    nissl, pv = visual.representative_rows(inputs)
    assert nissl.section_number == 40
    assert pv.section_number == 38


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("status", "Status mismatch"),
        ("sample", "Sample/stain identity mismatch"),
        ("z", "Noncanonical z coordinate"),
    ],
)
def test_cross_validation_rejects_mismatches(
    tmp_path: Path, mutation: str, message: str
) -> None:
    dataset = write_synthetic_dataset(tmp_path / mutation)
    physical_path = dataset / "metadata/physical_sections.tsv"
    samples_path = dataset / "inputs/views/HIST_ALL/samples.tsv"
    physical, physical_fields = visual._read_tsv(physical_path)
    samples, sample_fields = visual._read_tsv(samples_path)
    if mutation == "status":
        samples[0]["status"] = "absent"
    elif mutation == "sample":
        samples[0]["sample_id"] = "allen_708424_pv_0001.tif"
    else:
        physical[0]["serial_z_center_mm"] = "0.0"
    _write_tsv(physical_path, physical_fields, physical)
    _write_tsv(samples_path, sample_fields, samples)
    with pytest.raises(RuntimeError, match=message):
        _load_synthetic(dataset)


def test_prepared_side_profile_preserves_absence_and_common_pixel_origin(
    tmp_path: Path,
) -> None:
    inputs = _load_synthetic(write_synthetic_dataset(tmp_path / "profile"))
    profile = visual.build_side_profile(inputs)
    assert profile.shape == (5, 9)
    assert np.all(np.isnan(profile[1]))
    assert np.all(np.isnan(profile[3]))

    finite_white = np.flatnonzero(np.isfinite(profile[0]))
    finite_gray = np.flatnonzero(np.isfinite(profile[2]))
    finite_black = np.flatnonzero(np.isfinite(profile[4]))
    np.testing.assert_array_equal(finite_white, [0, 1, 2])
    np.testing.assert_array_equal(finite_gray, [0, 1, 2, 3])
    np.testing.assert_array_equal(finite_black, [0, 1, 2, 3, 4])
    np.testing.assert_allclose(profile[0, finite_white], 0.0)
    np.testing.assert_allclose(profile[2, finite_gray], 1.0 - 128.0 / 255.0)
    np.testing.assert_allclose(profile[4, finite_black], 1.0 - 64.0 / 255.0)


def test_original_profile_uses_native_spacing_and_source_pixel_zero_origin(
    tmp_path: Path,
) -> None:
    dataset = write_synthetic_dataset(tmp_path / "original")
    inputs = _load_synthetic(dataset, stage="original")
    assert inputs.stage == "original"
    np.testing.assert_array_equal(inputs.x_um, [0.0, 200.0, 400.0, 600.0])
    profile = visual.build_side_profile(inputs)
    assert profile.shape == (5, 4)
    np.testing.assert_array_equal(np.flatnonzero(np.isfinite(profile[0])), [0, 1, 2])
    np.testing.assert_array_equal(
        np.flatnonzero(np.isfinite(profile[2])), [0, 1, 2, 3]
    )
    np.testing.assert_array_equal(
        np.flatnonzero(np.isfinite(profile[4])), [0, 1, 2, 3]
    )
    assert inputs.rows[0].image_path == (
        dataset / "raw/nissl/source_0036.jpg"
    )

    figure, titles = visual.render_stack_overview(inputs)
    try:
        assert titles[0].startswith("Central original Nissl (section 40,")
        assert titles[1].startswith("Central original PV (section 38,")
        assert titles[2].startswith("Original x-z side profile (source x=0;")
    finally:
        visual.plt.close(figure)


def test_rendering_writes_one_png_and_protects_existing_output(
    tmp_path: Path,
) -> None:
    dataset = write_synthetic_dataset(tmp_path / "render")
    inputs = _load_synthetic(dataset)
    figure, titles = visual.render_stack_overview(inputs)
    try:
        assert len(titles) == 7
        assert titles[0].startswith("Central Nissl (section 40,")
        assert titles[1].startswith("Central PV (section 38,")
    finally:
        visual.plt.close(figure)

    output = tmp_path / "qc/overview.png"
    destination, written_titles = visual.write_stack_overview(
        dataset,
        output,
        expected_slots=5,
        expected_stain_counts=SYNTHETIC_COUNTS,
    )
    assert destination == output.resolve()
    assert written_titles == titles
    with Image.open(output) as rendered:
        assert rendered.format == "PNG"
        assert rendered.size == (3360, 2640)
    assert list(output.parent.iterdir()) == [output]

    with pytest.raises(RuntimeError, match="already exists"):
        visual.write_stack_overview(
            dataset,
            output,
            expected_slots=5,
            expected_stain_counts=SYNTHETIC_COUNTS,
        )
    visual.write_stack_overview(
        dataset,
        output,
        overwrite=True,
        expected_slots=5,
        expected_stain_counts=SYNTHETIC_COUNTS,
    )


def test_original_stage_writes_only_its_png(tmp_path: Path) -> None:
    dataset = write_synthetic_dataset(tmp_path / "original-render")
    output = tmp_path / "qc/original.png"
    destination, titles = visual.write_stack_overview(
        dataset,
        output,
        stage="original",
        data_dir=dataset / "raw",
        expected_slots=5,
        expected_stain_counts=SYNTHETIC_COUNTS,
    )
    assert destination == output.resolve()
    assert titles[0].startswith("Central original Nissl")
    with Image.open(output) as rendered:
        assert rendered.size == (3360, 2640)
    assert list(output.parent.iterdir()) == [output]
