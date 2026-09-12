from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import tifffile

from preprocess import run_allen_emlddmm_full_coarse_nissl as coarse
from preprocess import run_allen_emlddmm_full_resolution_nissl as native


def test_categorical_and_nissl_warps_share_physical_pullback(monkeypatch):
    calls = []

    def coordinates(transform, row_um, column_um, **kwargs):
        calls.append((
            np.asarray(transform).copy(),
            np.asarray(row_um).copy(),
            np.asarray(column_um).copy(),
            kwargs,
        ))
        return np.meshgrid(
            np.arange(len(row_um), dtype=np.float64),
            np.arange(len(column_um), dtype=np.float64),
            indexing="ij",
        )

    monkeypatch.setattr(coarse, "_section_sample_indices", coordinates)
    transform = np.array([[1.0, 0.0, 3.0], [0.0, 1.0, -2.0], [0.0, 0.0, 1.0]])
    row = np.arange(3, dtype=np.float64) * 200.0
    column = np.arange(4, dtype=np.float64) * 200.0
    labels = np.arange(12, dtype=np.uint32).reshape(3, 4)
    native.warp_categorical_section(labels, transform, row, column)
    coarse._warp_saved_section(
        np.ones((3, 3, 4), np.float32),
        np.ones((3, 4), np.float32),
        transform,
        row,
        column,
    )

    assert len(calls) == 2
    for left, right in zip(calls[0][:3], calls[1][:3], strict=True):
        np.testing.assert_array_equal(left, right)
    assert calls[0][3] == calls[1][3]


def test_categorical_warp_is_nearest_neighbor(monkeypatch):
    labels = np.array([[0, 10], [20, 30]], dtype=np.uint32)
    iy = np.array([[0.1, 0.1], [0.9, 0.9]])
    ix = np.array([[0.1, 0.9], [0.1, 0.9]])
    monkeypatch.setattr(
        coarse, "_section_sample_indices", lambda *args, **kwargs: (iy, ix)
    )
    warped = native.warp_categorical_section(
        labels, np.eye(3), np.arange(2), np.arange(2)
    )
    np.testing.assert_array_equal(warped, labels)
    assert warped.dtype == np.uint32


def test_exact_categorical_bilateral_union_has_no_crop_or_interpolation():
    left = np.arange(3 * 5, dtype=np.uint32).reshape(3, 5)
    bilateral = native.bilateral_union(left)
    assert bilateral.shape == (3, 10)
    np.testing.assert_array_equal(bilateral[:, :5], left[:, ::-1])
    np.testing.assert_array_equal(bilateral[:, 5:], left)


def test_annotation_output_cannot_target_authoritative_nissl():
    with pytest.raises(RuntimeError, match="must not modify"):
        native.construct_symmetric_annotations(native.CLEAN_SYMMETRIC_DATASET)


def test_native_context_uses_sibling_annotations_and_native_output_name(tmp_path):
    original = coarse.ANNOTATION_DATASET
    with native._coarse_context(native.CLEAN_SYMMETRIC_DATASET, tmp_path):
        assert coarse.DATASET == native.CLEAN_SYMMETRIC_DATASET
        assert coarse.ANNOTATION_DATASET == native.SYMMETRIC_ANNOTATION_DATASET
        assert coarse.ANNOTATION_DIR == tmp_path / "annotations_on_native_mri"
    assert coarse.ANNOTATION_DATASET == original


def test_complete_through_scale_uses_checkpoint_recorded_paths(
    tmp_path, monkeypatch
):
    checkpoints = tmp_path / "checkpoints"
    product = tmp_path / "deliberately-named-product"
    checkpoints.mkdir()
    product.mkdir()
    numerical = product / "deliberately-named-numerical.npz"
    raw = [product / f"deliberately-named-history-{level}.npy" for level in (1, 2)]
    np.savez(numerical, sentinel=np.array([1]))
    for level, path in enumerate(raw, 1):
        np.save(path, np.array([level]))
    provenance = product / "provenance.json"
    provenance.write_text(json.dumps({
        "checksums": {
            str(numerical): coarse.checksum(numerical),
            **{str(path): coarse.checksum(path) for path in raw},
        }
    }))
    registration = {
        "status": "complete_through_scale",
        "completed_scale_number": 2,
        "numerical": str(numerical),
        "raw_Esave": [str(path) for path in raw],
        "effective_match_weight": None,
        "provenance": str(provenance),
        "transform_output_root": str(product),
    }
    checkpoint = checkpoints / "registration.json"
    checkpoint.write_text(json.dumps(registration))
    monkeypatch.setattr(coarse, "CHECKPOINTS", checkpoints)
    loaded = coarse.load_checkpoint(
        "registration", accepted_statuses=("complete", "complete_through_scale")
    )
    resolved, histories, effective, hashes, report = (
        coarse._native_postprocess_sources(loaded)
    )
    assert resolved == numerical.resolve()
    assert histories == [path.resolve() for path in raw]
    assert effective is None
    assert str(checkpoint.resolve()) in hashes
    assert report["checkpoint_status"] == "complete_through_scale"
    assert report["recorded_paths_used"] is True
    assert report["effective_match_weight_missing_is_legitimate"] is True
    assert not (tmp_path / "section_alignment_atlas_free").exists()


def test_normal_complete_registration_uses_recorded_effective_weight(
    tmp_path, monkeypatch
):
    checkpoints = tmp_path / "checkpoints"
    product = tmp_path / "normal-product"
    checkpoints.mkdir()
    product.mkdir()
    numerical = product / "custom-numerical.npz"
    history = product / "custom-history.npy"
    effective = product / "custom-effective.npy"
    np.savez(numerical, sentinel=np.array([1]))
    np.save(history, np.array([1.0]))
    np.save(effective, np.array([1.0], dtype=np.float32))
    registration = {
        "status": "complete",
        "numerical": str(numerical),
        "raw_Esave": [str(history)],
        "final_effective_match_weight": str(effective),
    }
    (checkpoints / "registration.json").write_text(json.dumps(registration))
    monkeypatch.setattr(coarse, "CHECKPOINTS", checkpoints)
    resolved, histories, effective_path, _, report = (
        coarse._native_postprocess_sources(registration)
    )
    assert resolved == numerical.resolve()
    assert histories == [history.resolve()]
    assert effective_path == effective.resolve()
    assert report["effective_match_weight"] == str(effective.resolve())
    assert report["effective_match_weight_missing_is_legitimate"] is False


def test_missing_effective_weight_uses_support_without_synthesis(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(coarse, "RUN_TMP", tmp_path)
    axes = [
        np.array([0.0, 1.0]),
        np.array([0.0, 1.0, 2.0]),
        np.array([0.0, 1.0, 2.0, 3.0]),
    ]
    support = np.ones((2, 3, 4), dtype=np.float32)
    numerator = np.stack([support * value for value in (0.2, 0.4, 0.6)])
    phi = np.stack(np.meshgrid(*axes, indexing="ij"))
    reconstruction, propagated_support, propagated_effective = (
        coarse._sample_chain_slabs(
            np.eye(4), phi, axes, axes, axes, axes[1], axes[2],
            numerator, support, None,
        )
    )
    assert propagated_effective is None
    np.testing.assert_allclose(propagated_support, support)
    np.testing.assert_allclose(reconstruction[..., 0], 0.2)
    assert not (tmp_path / "effective_match_weight_on_mri.dat").exists()


def test_missing_effective_weight_is_labeled_as_support(monkeypatch):
    captured = []
    monkeypatch.setattr(
        coarse, "_atomic_figure", lambda path, figure, **kwargs: captured.append(figure)
    )
    xI = [
        np.arange(5, dtype=np.float64),
        np.arange(11, dtype=np.float64),
        np.arange(7, dtype=np.float64),
    ]
    mri = np.linspace(0.0, 1.0, 5 * 11 * 7, dtype=np.float32).reshape(1, 5, 11, 7)
    reconstruction = np.full((5, 11, 7, 3), 0.4, dtype=np.float32)
    support = np.ones((5, 11, 7), dtype=np.float32)
    coarse._mri_nissl_figures(
        mri, xI, reconstruction, support, None, 2.0, grid_name="native"
    )
    text = " ".join(item.get_text() for item in captured[0].texts)
    assert "Propagated Nissl support" in text
    assert "support-weighted Nissl" in text
    assert "WM × W0" not in text


def test_annotation_sources_code_reads_the_sibling_root():
    import inspect

    source = inspect.getsource(coarse._annotation_sources)
    assert 'ANNOTATION_DATASET / "metadata/annotations.tsv"' in source
    assert '(ANNOTATION_DATASET / record["path"]).resolve()' in source


def test_identity_is_native_residual_comparison_initializer(tmp_path, monkeypatch):
    monkeypatch.setattr(coarse, "POST_DIR", tmp_path)
    rows = [{"allen_section_number": str(index + 36)} for index in range(2846)]
    observed = np.arange(len(coarse.FLAGGED_ALLEN), dtype=np.int64)
    for index, allen in zip(observed, coarse.FLAGGED_ALLEN, strict=True):
        rows[int(index)]["allen_section_number"] = str(allen)
    final = np.broadcast_to(np.eye(3), (2846, 3, 3)).copy()
    _, report, comparisons = coarse._final_residual_diagnostics(
        rows,
        [np.arange(2846, dtype=np.float64), np.arange(2), np.arange(2)],
        observed,
        final,
        comparison_initializer="identity",
    )
    assert report["comparison_initializer"] == "identity"
    assert report["comparison_initial_baseline"] == (
        "identity per physical serial position"
    )
    assert all(item["initial_baseline"] == report["comparison_initial_baseline"] for item in comparisons)
    assert all(item["initial"] == {
        "row_translation_um": 0.0,
        "column_translation_um": 0.0,
        "rotation_deg": 0.0,
    } for item in comparisons)


@pytest.mark.skipif(
    not native.SYMMETRIC_ANNOTATION_DATASET.is_dir(),
    reason="materialized native annotation derivative is unavailable",
)
def test_materialized_annotation_inventory_ids_groups_and_symmetry():
    derivative = native.SYMMETRIC_ANNOTATION_DATASET
    source_rows = list(csv.DictReader(
        (coarse.ANNOTATION_ZARR / "metadata/manifest.tsv").open(), delimiter="\t"
    ))
    output_rows = list(csv.DictReader(
        (derivative / "metadata/annotations.tsv").open(), delimiter="\t"
    ))
    assert len({int(row["section_number"]) for row in source_rows}) == 106
    assert {int(row["section_number"]) for row in output_rows} == {
        int(row["section_number"]) for row in source_rows
    }
    expected_pairs = {
        (int(row["section_number"]), int(group))
        for row in source_rows
        for group in json.loads(row["graphic_groups_present"])
    }
    assert {
        (int(row["section_number"]), int(row["graphic_group_id"]))
        for row in output_rows
    } == expected_pairs
    observed_ids = set()
    for row in output_rows:
        labels = tifffile.imread(derivative / row["path"])
        assert labels.shape == (522, 730)
        assert labels.dtype == np.uint32
        np.testing.assert_array_equal(labels[:, :365], labels[:, 365:][:, ::-1])
        observed_ids.update(int(value) for value in np.unique(labels) if value)
        assert row["sampling"] == "categorical_nearest_neighbor"
        assert row["right_origin"] == "synthetically_reflected"
    dataset = json.loads((derivative / "dataset.json").read_text())
    assert dataset["annotation_section_count"] == 106
    assert dataset["annotation_image_count"] == len(expected_pairs)
    assert dataset["source_label_ids"] == dataset["aligned_label_ids"]
    assert observed_ids == set(dataset["source_label_ids"])
    symmetry = json.loads((derivative / "metadata/symmetry.json").read_text())
    assert symmetry["symmetric_space"] == "HIST_SYMMETRIC"
    assert symmetry["image_operation"] == (
        "exact_reflection_and_union_without_interpolation"
    )
    assert symmetry["operation"] == "exact_reflection_no_medial_duplication"
    assert symmetry["columns_cropped"] == 0
    for section in (1532, 1616):
        assert (derivative / f"qc/section-{section:04d}.png").stat().st_size > 0

    from preprocess.visualize_allen_annotations import load_bilateral_panels

    _, panels = load_bilateral_panels(
        derivative, 1532, annotations_zarr=coarse.ANNOTATION_ZARR, panel_width=80
    )
    titles = [panel.title for panel in panels]
    assert titles[0] == "Nissl reference"
    assert "Combined modified-Brodmann atlas preview" in titles
    assert "Combined boundaries on Nissl" in titles
