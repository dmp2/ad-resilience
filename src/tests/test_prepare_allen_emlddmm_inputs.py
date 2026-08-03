from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from preprocess.build_allen_symmetric_histology import (
    _symmetric_geometry,
    apply_tissue_mask,
    bilateral_union,
    hemisphere_origin_mask,
    generate_pv_mask,
    shift_to_medial_edge,
)
from preprocess.build_allen_emlddmm_lattice import (
    build_lattice_rows,
    verify_lattice,
)
from preprocess.prepare_allen_emlddmm_inputs import (
    INTERNAL_FIELDS,
    NUMBER_OF_SLOTS,
    PHYSICAL_FIELDS,
    PREPARED_METADATA_SCHEMA,
    SAMPLE_PROVENANCE_FIELDS,
    SERIAL_CENTER_EXTENT_MM,
    SERIAL_OUTER_FACE_EXTENT_MM,
    _generate_and_validate_sidecars,
    _write_samples,
    _write_tsv,
    accepted_loader_axes,
    build_physical_rows,
    calculate_canvas_audit,
    canonical_z_axis_um,
    canonical_z_um,
    dataset_metadata,
    embed_content_in_common_canvas,
    full_mixed_canvas_rows,
    resample_origin_preserving_rgb,
)


def _write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    fields = [
        "kind",
        "series_or_layer",
        "section_number",
        "allen_section_image_id",
        "allen_data_set_id",
        "path",
        "width_px",
        "height_px",
        "pixel_size_um",
        "sha256",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _manifest_row(stain: str, section: int) -> dict[str, object]:
    prefix = "nissl" if stain == "nissl" else "ihc"
    return {
        "kind": "histology_jpeg",
        "series_or_layer": stain,
        "section_number": section,
        "allen_section_image_id": 100000 + section,
        "allen_data_set_id": 200000 + section,
        "path": f"{prefix}/images_orig/image_{section:04d}.jpg",
        "width_px": 100,
        "height_px": 120,
        "pixel_size_um": 32,
        "sha256": "a" * 64,
    }


def test_real_manifest_builds_mixed_uncompressed_50um_lattice() -> None:
    manifest = Path("data/raw/allen/specimen_708424/metadata/manifest.tsv")
    if not manifest.exists():
        return

    rows, summary = build_lattice_rows(manifest)
    assert summary["present_count"] == 928
    assert summary["stain_counts"] == {"pv": 287, "nissl": 641}
    assert summary["row_count"] == 2846
    assert summary["absent_count"] == 1918
    assert rows[0]["allen_section_number"] == "36"
    assert rows[0]["stain"] == "pv"
    assert rows[0]["serial_z_center_mm"] == "-71.125"
    assert rows[-1]["allen_section_number"] == "2881"
    assert rows[-1]["stain"] == "nissl"
    assert rows[-1]["serial_z_center_mm"] == "71.125"
    assert Counter(row["observation_class"] for row in rows) == {
        "observed_nissl": 641,
        "observed_pv": 287,
        "unobserved": 1918,
    }
    assert SERIAL_CENTER_EXTENT_MM == 142.25
    assert SERIAL_OUTER_FACE_EXTENT_MM == 142.3
    metadata = dataset_metadata(
        series="all",
        section_range=(36, 2881),
        summary=summary,
        view_counts={
            "HIST_ALL": {"present": 928, "absent": 1918},
            "HIST_NISSL": {"present": 641, "absent": 2205},
            "HIST_PV": {"present": 287, "absent": 2559},
        },
        target_pixel_size_um=200.0,
        manifest=manifest,
    )
    assert metadata["prepared_metadata_schema"] == PREPARED_METADATA_SCHEMA
    pv_2130 = rows[2130 - 36]
    assert pv_2130["allen_section_image_id"] == "146699677"
    assert pv_2130["observation_class"] == "observed_pv"
    assert pv_2130["block_assignment_status"] == "unresolved"
    assert pv_2130["block_id"] == ""
    observed_sections = [
        int(row["allen_section_number"])
        for row in rows
        if row["image_present"] == "true"
    ]
    assert observed_sections[:5] == [36, 39, 43, 44, 47]
    assert np.diff(observed_sections[:5]).tolist() == [3, 4, 1, 3]
    np.testing.assert_allclose(
        np.diff(observed_sections[:5]) * 0.05,
        [0.15, 0.2, 0.05, 0.15],
        atol=1e-12,
        rtol=0.0,
    )


def test_physical_table_uses_global_center_for_bounded_window(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.tsv"
    _write_manifest(
        manifest,
        [_manifest_row("nissl", 1448), _manifest_row("pv", 1453)],
    )
    rows, summary = build_physical_rows(
        manifest,
        section_range=(1448, 1477),
        validate_counts=False,
    )
    expected_axis = canonical_z_axis_um()[1448 - 36 : 1477 - 36 + 1]
    np.testing.assert_array_equal(
        np.array([float(row["serial_z_center_mm"]) * 1000 for row in rows]),
        expected_axis,
    )
    assert summary["row_count"] == 30
    assert canonical_z_um(1448) != -np.ptp(expected_axis) / 2


def test_bilateral_union_is_one_observation_with_exact_reflection() -> None:
    observed = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    tissue_mask = np.array([[False, True, True], [False, True, False]])
    cleaned = apply_tissue_mask(observed, tissue_mask, background=0)
    np.testing.assert_array_equal(cleaned[~tissue_mask], 0)
    np.testing.assert_array_equal(cleaned[tissue_mask], observed[tissue_mask])
    shifted = shift_to_medial_edge(cleaned, 1, background=0)
    np.testing.assert_array_equal(shifted[:, :2], cleaned[:, 1:])
    np.testing.assert_array_equal(shifted[:, 2], 0)
    bilateral = bilateral_union(shifted)
    assert bilateral.shape == (2, 6, 3)
    np.testing.assert_array_equal(bilateral[:, :3], shifted[:, ::-1])
    np.testing.assert_array_equal(bilateral[:, 3:], shifted)
    mask = hemisphere_origin_mask(bilateral.shape[:2])
    np.testing.assert_array_equal(mask[:, :3], 1)
    np.testing.assert_array_equal(mask[:, 3:], 2)


def test_symmetric_grid_and_plane_are_derived_from_source_geometry() -> None:
    geometry = _symmetric_geometry(
        (7, 4),
        {
            "directions": ["none", [250.0, 0.0, 0.0], [0.0, 250.0, 0.0]],
            "origin_xy": [-375.0, -750.0],
        },
    )
    assert geometry["bilateral_shape_yx"] == [7, 8]
    assert geometry["bilateral_origin_xy_um"] == [-875.0, -750.0]
    assert geometry["reflection_plane"] == {
        "axis": "x",
        "coordinate_um": 0.0,
        "location": "between_columns",
        "adjacent_column_indices": [3, 4],
    }


def test_200um_resampling_preserves_source_pixel_origin() -> None:
    source = np.zeros((22, 32, 3), dtype=np.uint8)
    source[10, 15] = [255, 100, 50]
    prepared, transform = resample_origin_preserving_rgb(source, 40.0, 200.0)
    assert prepared.shape == (5, 7, 3)
    center = np.array(
        [[(prepared.shape[1] - 1) / 2], [(prepared.shape[0] - 1) / 2], [1]]
    )
    mapped = transform @ center
    np.testing.assert_allclose(mapped[:2, 0], [15.0, 10.0])
    np.testing.assert_allclose(transform[0, 0], 5.0)
    np.testing.assert_allclose(transform[1, 1], 5.0)
    np.testing.assert_array_equal(transform[:2, 2], [0.0, 0.0])
    assert not np.allclose(mapped[:2, 0], [15.5, 10.5])


def _canvas_rows(dimensions: list[tuple[int, int]]) -> list[dict[str, str]]:
    rows = []
    for index, (height, width) in enumerate(dimensions):
        row = {field: "" for field in PHYSICAL_FIELDS + INTERNAL_FIELDS}
        row.update(
            {
                "section_number": str(36 + index),
                "prepared_path": f"image_{index}.tif",
                "prepared_height_px": str(height),
                "prepared_width_px": str(width),
            }
        )
        rows.append(row)
    return rows


def test_loader_canvas_accepts_default_only_when_every_section_fits() -> None:
    uniform = _canvas_rows([(10, 12)] * 20)
    accepted = calculate_canvas_audit(uniform, 200.0)
    assert accepted["automatic_canvas_accepted"]
    assert accepted["accepted_canvas_shape_yx"] == [10, 12]
    assert accepted["sectionwise_centering"] is False
    assert accepted["global_translation_xy_um"] == [-1100.0, -900.0]

    with_outlier = _canvas_rows([(10, 12)] * 19 + [(30, 40)])
    explicit = calculate_canvas_audit(with_outlier, 200.0)
    assert not explicit["automatic_canvas_accepted"]
    assert explicit["accepted_canvas_shape_yx"] == [30, 40]
    assert explicit["global_translation_xy_um"] == [-3900.0, -2900.0]
    axes = accepted_loader_axes(with_outlier, explicit)
    assert [len(axis) for axis in axes] == [20, 30, 40]
    assert axes[1][0] == -2900.0
    assert axes[2][0] == -3900.0


def test_series_specific_canvas_retains_companion_stain_bounds() -> None:
    nissl = {field: "" for field in PHYSICAL_FIELDS + INTERNAL_FIELDS}
    nissl.update(
        {
            "section_number": "36",
            "status": "present",
            "stain": "nissl",
            "prepared_path": "nissl.tif",
            "prepared_height_px": "10",
            "prepared_width_px": "10",
        }
    )
    pv = {field: "" for field in PHYSICAL_FIELDS + INTERNAL_FIELDS}
    pv.update(
        {
            "section_number": "37",
            "status": "present",
            "stain": "pv",
            "source_height_px": "100",
            "source_width_px": "80",
            "source_pixel_size_um": "200",
        }
    )
    canvas_rows = full_mixed_canvas_rows([nissl, pv], 200.0)
    assert canvas_rows[1]["prepared_path"] == "__domain_audit_only__"
    audit = calculate_canvas_audit(canvas_rows, 200.0)
    assert audit["maximum_canvas_shape_yx"] == [100, 80]


def test_samples_tsv_is_human_and_positionally_complete(tmp_path: Path) -> None:
    rows = []
    for index in range(3):
        row = {field: "" for field in PHYSICAL_FIELDS + INTERNAL_FIELDS}
        row.update(
            {
                "grid_index": str(index),
                "section_number": str(36 + index),
                "status": "present" if index in {0, 2} else "absent",
                "stain": "pv" if index in {0, 2} else "",
                "prepared_path": (
                    f"inputs/sections/pv/allen_708424_pv_{index + 1:04d}.tif"
                    if index in {0, 2}
                    else ""
                ),
            }
        )
        rows.append(row)
    counts = _write_samples(
        tmp_path, rows, frozenset({"pv"}), include_provenance=True
    )
    with (tmp_path / "samples.tsv").open(encoding="utf-8", newline="") as stream:
        samples = list(csv.DictReader(stream, delimiter="\t"))
    assert list(samples[0]) == [
        "sample_id",
        "participant_id",
        "species",
        "status",
        *SAMPLE_PROVENANCE_FIELDS,
    ]
    assert [row["status"] for row in samples] == ["present", "absent", "present"]
    assert {row["participant_id"] for row in samples} == {"708424"}
    assert {row["species"] for row in samples} == {"Homo sapiens"}
    assert counts == {"present": 2, "absent": 1}
    assert not (tmp_path / samples[1]["sample_id"]).exists()


def test_upstream_sidecar_adapter_records_one_globally_translated_canvas(
    tmp_path: Path,
) -> None:
    for dependency in ("pandas", "matplotlib", "skimage", "h5py"):
        pytest.importorskip(
            dependency,
            reason=f"pinned histsetup imports {dependency} at module load",
        )
    image_name = "allen_708424_pv_0001.tif"
    Image.fromarray(np.full((5, 7, 3), 20, dtype=np.uint8)).save(
        tmp_path / image_name
    )
    row = {field: "" for field in PHYSICAL_FIELDS + INTERNAL_FIELDS}
    row.update(
        {
            "status": "present",
            "prepared_path": image_name,
            "prepared_width_px": "7",
            "prepared_height_px": "5",
            "z_um": "-71125.0",
        }
    )
    _generate_and_validate_sidecars(tmp_path, [row], 200.0)
    sidecar = json.loads((tmp_path / "allen_708424_pv_0001.json").read_text())
    assert sidecar["Sizes"] == [3, 7, 5, 1]
    assert sidecar["SpaceOrigin"] == [-600.0, -400.0, -71125.0]
    assert sidecar["SpaceDirections"][-1] == [0.0, 0.0, 50.0]


def test_global_canvas_keeps_section_content_at_the_common_pixel_origin() -> None:
    first = np.zeros((4, 6, 3), dtype=np.uint8)
    second = np.zeros((3, 4, 3), dtype=np.uint8)
    first[1, 2] = 255
    second[1, 2] = 255

    common = []
    for content in (first, second):
        common.append(embed_content_in_common_canvas(content, (7, 9)))

    assert np.argwhere(common[0][..., 0] > 0).tolist() == [[1, 2]]
    assert np.argwhere(common[1][..., 0] > 0).tolist() == [[1, 2]]
    global_origin_xy_um = [-(9 - 1) * 100.0, -(7 - 1) * 100.0]
    assert global_origin_xy_um == [-800.0, -600.0]
    with pytest.raises(ValueError, match="does not fit"):
        embed_content_in_common_canvas(first, (3, 5))


def test_lattice_validator_uses_preparation_owned_schema(tmp_path: Path) -> None:
    manifest = Path("data/raw/allen/specimen_708424/metadata/manifest.tsv")
    if not manifest.exists():
        return
    rows, _ = build_lattice_rows(manifest)
    output = tmp_path / "physical_sections.tsv"
    _write_tsv(output, rows)
    summary = verify_lattice(manifest, output)
    assert summary["present_count"] == 928
    text = output.read_text(encoding="utf-8")
    output.write_text(text.replace("-71.125", "-71.124", 1), encoding="utf-8")
    try:
        verify_lattice(manifest, output)
    except ValueError as error:
        assert "differs" in str(error)
    else:
        raise AssertionError("verification accepted a changed lattice")


def test_canonical_axis_has_required_length_and_spacing() -> None:
    axis = canonical_z_axis_um()
    assert axis.shape == (NUMBER_OF_SLOTS,)
    np.testing.assert_array_equal(np.diff(axis), 50.0)
    assert np.mean(axis) == 0.0



def _pv_canvas(height: int = 160, width: int = 160) -> np.ndarray:
    return np.full((height, width, 3), 255, dtype=np.uint8)


def test_pv_mask_combines_homogeneous_darkness_and_texture() -> None:
    image = _pv_canvas()
    image[25:95, 20:75] = 105
    checker = np.indices((55, 55)).sum(axis=0) % 2
    image[30:85, 95:150] = np.where(checker[..., None] == 0, 205, 245)

    result = generate_pv_mask(image, section=120)

    assert result.darkness_mask[50, 40]
    assert result.texture_mask[50, 115]
    assert result.mask[50, 40]
    assert result.mask[50, 115]
    assert result.medial_column <= 20
    assert result.retained_area == np.count_nonzero(result.mask)
    assert len(result.retained_component_areas) == 2


def test_pv_mask_keeps_textured_tissue_with_tiny_saturated_core() -> None:
    image = _pv_canvas(240, 240)
    checker = np.indices((150, 140)).sum(axis=0) % 2
    image[45:195, 40:180] = np.where(checker[..., None] == 0, 220, 240)
    image[100:103, 100:103] = [255, 0, 0]

    result = generate_pv_mask(image, section=121)

    assert result.mask[80, 80]
    assert result.retained_area > 10_000


def test_pv_mask_removes_small_isolated_colored_dot() -> None:
    image = _pv_canvas(240, 240)
    image[45:195, 40:180] = 120
    image[10:13, 220:223] = [255, 0, 0]

    result = generate_pv_mask(image, section=121)

    assert result.mask[100, 100]
    assert not result.mask[11, 221]
    assert len(result.component_areas) > len(result.retained_component_areas)


def test_pv_mask_fills_small_hole_and_closes_small_gap() -> None:
    image = _pv_canvas()
    image[25:135, 25:135] = 110
    image[75:78, 25:135] = 255
    image[70:74, 70:74] = 255

    result = generate_pv_mask(image, section=122)

    assert result.mask[76, 80]
    assert result.mask[71, 71]


@pytest.mark.parametrize(
    ("image", "message"),
    [
        (_pv_canvas(), "mask is empty"),
        (np.full((100, 100, 3), 100, dtype=np.uint8), "mask is empty"),
    ],
)
def test_pv_mask_fails_closed_for_no_usable_tissue(
    image: np.ndarray, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        generate_pv_mask(image, section=123)


def test_pv_mask_fails_closed_for_implausibly_large_area() -> None:
    image = _pv_canvas(160, 160)
    image[:, :150] = 100
    with pytest.raises(ValueError, match="mask area is implausible"):
        generate_pv_mask(image, section=124)


def test_pv_mask_discards_retained_colored_fiducial() -> None:
    image = _pv_canvas()
    image[30:125, 20:90] = 110
    image[40:60, 125:145] = [255, 0, 0]
    result = generate_pv_mask(image, section=125)
    assert result.mask[70, 50]
    assert not result.mask[50, 135]
