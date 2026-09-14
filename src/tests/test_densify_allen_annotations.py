from __future__ import annotations

import csv

import numpy as np
import pytest

from preprocess import densify_allen_annotations as dense


class _TorchFacade:
    float32 = np.float32

    @staticmethod
    def as_tensor(value, dtype=None):
        return np.asarray(value, dtype=dtype)


class _IdentityInterpolation:
    @staticmethod
    def interp(axes, image, phi, interp2d=False):
        assert interp2d is True
        return np.asarray(image, dtype=np.float32)


def test_arbitrary_time_evaluator_reproduces_stored_wsi_states():
    _, em, torch, integrate, resample, _ = dense._load_wsi(dense.DEFAULT_WSI_REPOSITORY)
    axes = (
        np.linspace(-1.0, 1.0, 3, dtype=np.float32),
        np.linspace(-1.5, 1.5, 4, dtype=np.float32),
    )
    velocity = torch.zeros((4, 2, 3, 4), dtype=torch.float32)
    velocity[:, 0] = 0.05
    velocity[:, 1] = -0.025
    velocity_axes = tuple(torch.as_tensor(axis) for axis in axes)
    phi = integrate(
        velocity_axes,
        velocity,
        emlddmm_module=em,
        interp2d=True,
    )
    source = np.arange(12, dtype=np.float32).reshape(1, 3, 4)
    warped = torch.stack(
        [
            em.interp(
                velocity_axes,
                torch.as_tensor(source),
                state,
                interp2d=True,
            )
            for state in phi
        ]
    )
    output = {
        "v_symmetric": velocity,
        "forward": {"xv": velocity_axes},
        "phi_I": phi,
        "ItAll": warped,
    }

    evaluator = dense.build_source_flow_evaluator(
        output, axes, em, torch, integrate, resample
    )
    report = dense.validate_arbitrary_time_evaluator(evaluator, output, source, axes)

    assert report["stored_state_map_max_abs_error"] <= 2e-5
    assert report["stored_state_nissl_max_abs_error"] <= 2e-5
    assert evaluator.evaluate(0.375).shape == (2, 3, 4)


def test_semantic_sequences_skip_unavailable_without_zero_inference():
    rows = []
    for physical, state in ((10, "LABELED"), (20, "UNAVAILABLE"), (30, "LABELED")):
        rows.append(
            {
                "physical_index": str(physical),
                "graphic_group": "141667008",
                "semantic_state": state,
            }
        )
    for physical in (10, 20, 30):
        rows.append(
            {
                "physical_index": str(physical),
                "graphic_group": "31",
                "semantic_state": "LABELED",
            }
        )
    for group in (113753816, 265297118):
        rows.extend(
            [
                {
                    "physical_index": "10",
                    "graphic_group": str(group),
                    "semantic_state": "LABELED",
                },
                {
                    "physical_index": "30",
                    "graphic_group": str(group),
                    "semantic_state": "LABELED",
                },
            ]
        )

    sequences, pairs = dense.build_endpoint_sequences(rows)

    assert sequences[141667008] == [10, 30]
    assert 141667008 in pairs[(10, 30)]
    assert pairs[(10, 20)] == (31,)
    assert pairs[(20, 30)] == (31,)


def test_categorical_fusion_warps_one_hot_and_has_explicit_tie_rule():
    left = np.full((2, 3), 10, dtype=np.uint32)
    right = np.full((2, 3), 20, dtype=np.uint32)
    phi = np.zeros((2, 2, 3), dtype=np.float32)
    axes = (np.arange(2, dtype=np.float32), np.arange(3, dtype=np.float32))
    em = _IdentityInterpolation()
    torch = _TorchFacade()

    midpoint = dense.categorical_pair_plane(
        left,
        right,
        [0, 10, 20],
        0.5,
        phi,
        phi,
        axes,
        em,
        torch,
    )
    rightward = dense.categorical_pair_plane(
        left,
        right,
        [0, 10, 20],
        0.75,
        phi,
        phi,
        axes,
        em,
        torch,
    )

    # At the exact membership tie, endpoint-side evidence (left) wins before
    # the final numeric-ID fallback. This is explicit rather than channel order.
    np.testing.assert_array_equal(midpoint, left)
    np.testing.assert_array_equal(rightward, right)
    assert midpoint.dtype == np.uint32


def test_map_convention_reproduces_stored_source_trajectory():
    source = np.arange(6, dtype=np.float32).reshape(1, 2, 3)
    output = {
        "phi_I": np.zeros((3, 2, 2, 3), dtype=np.float32),
        "ItAll": np.repeat(source[None], 3, axis=0),
    }
    error = dense.validate_source_map(
        output,
        source,
        (np.arange(2, dtype=np.float32), np.arange(3, dtype=np.float32)),
        _IdentityInterpolation(),
        _TorchFacade(),
    )
    assert error == 0.0


def test_pair_config_uses_modest_configurable_temporal_nt():
    config = dense._load_pair_config(dense.DEFAULT_PAIR_CONFIG)
    assert config["nt"] == 10
    assert dense.pair_solver_config(config)["nt"] == 10
    assert dense.pair_solver_config(config, 20)["nt"] == 20
    assert config["slice_matching"] == [False]
    assert config["eA"] == [0.0]
    assert config["eA2d"] == [0.0]


def test_unknown_endpoint_id_is_rejected_before_transport():
    left = np.array([[10]], dtype=np.uint32)
    right = np.array([[20]], dtype=np.uint32)
    with pytest.raises(RuntimeError, match="omits"):
        dense.categorical_pair_plane(
            left,
            right,
            [0, 10],
            0.5,
            np.zeros((2, 1, 1), np.float32),
            np.zeros((2, 1, 1), np.float32),
            (np.arange(1), np.arange(1)),
            _IdentityInterpolation(),
            _TorchFacade(),
        )


def test_tiff_uint32_round_trip_preserves_ids_shape_and_deflate(tmp_path):
    axis = np.array([-71_125.0, -71_075.0], dtype=np.float64)
    store = dense.TiffDenseAnnotationStore(
        tmp_path, (2, 3), axis, compression="deflate", groups=(31,)
    )
    plane = np.array([[0, 10, 266441685], [20, 30, 40]], dtype=np.uint32)

    assert store.write_group_plane(31, 1, plane, overwrite=False)

    path = tmp_path / "dense_tiff/groups/31/000001.tif"
    assert path.is_file()
    emitted = store.read_group_plane(31, 1)
    np.testing.assert_array_equal(emitted, plane)
    assert emitted.dtype == np.uint32
    assert emitted.shape == (2, 3)
    with dense.tifffile.TiffFile(path) as tif:
        assert tif.pages[0].compression.name in {"DEFLATE", "ADOBE_DEFLATE"}


def test_tiff_manifest_maps_canonical_filename_to_z_and_observed_state(tmp_path):
    axis = np.array([-100.25, -50.0], dtype=np.float64)
    store = dense.TiffDenseAnnotationStore(
        tmp_path, (2, 2), axis, compression="none", groups=(31,)
    )
    observed = np.array([[0, 101], [202, 303]], dtype=np.uint32)
    unavailable = np.zeros((2, 2), dtype=np.uint32)
    store.write_group_plane(31, 0, observed, overwrite=False)
    store.write_group_plane(31, 1, unavailable, overwrite=False)
    store.set_state(31, 0, dense.OBSERVED_LABELS)

    manifest = store.finalize(include_combined=False)

    with manifest.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    assert rows[0]["relative_path"] == "dense_tiff/groups/31/000000.tif"
    assert rows[0]["canonical_index"] == "0"
    assert float(rows[0]["z_um"]) == axis[0]
    assert rows[0]["graphic_group_id"] == "31"
    assert rows[0]["semantic_state"] == "OBSERVED_LABELS"
    assert rows[0]["evidence_state"] == "OBSERVED"
    assert rows[0]["sha256"] == dense._sha256(tmp_path / rows[0]["relative_path"])
    assert rows[1]["semantic_state"] == "UNSUPPORTED"
    assert rows[1]["evidence_state"] == "UNAVAILABLE"


def test_observed_anchor_tiff_is_pixel_identical(tmp_path):
    axis = np.array([0.0], dtype=np.float64)
    store = dense.TiffDenseAnnotationStore(
        tmp_path, (2, 2), axis, compression="deflate", groups=(31,)
    )
    observed = np.array([[0, 1], [np.iinfo(np.uint32).max, 42]], dtype=np.uint32)

    store.write_group_plane(31, 0, observed, overwrite=False)

    assert store.verify_plane(0, group=31, expected=observed)
    np.testing.assert_array_equal(store.read_group_plane(31, 0), observed)


def test_cli_defaults_to_tiff_and_accepts_explicit_zarr():
    parser = dense.build_argument_parser()

    defaults = parser.parse_args([])
    zarr_args = parser.parse_args(["--output-format", "zarr"])

    assert defaults.output_format == "tiff"
    assert defaults.tiff_compression == "deflate"
    assert zarr_args.output_format == "zarr"


def test_store_factory_selects_existing_zarr_backend(monkeypatch, tmp_path):
    selected = object()

    def fake_zarr_store(output, shape, canonical_z_um):
        assert output == tmp_path
        assert shape == (2, 2)
        np.testing.assert_array_equal(canonical_z_um, [0.0])
        return selected

    monkeypatch.setattr(dense, "ZarrDenseAnnotationStore", fake_zarr_store)

    result = dense.create_dense_store(
        tmp_path,
        (2, 2),
        np.array([0.0]),
        output_format="zarr",
        tiff_compression="deflate",
    )

    assert result is selected


def test_tiff_temp_file_is_not_a_completed_plane(tmp_path):
    axis = np.array([0.0], dtype=np.float64)
    store = dense.TiffDenseAnnotationStore(
        tmp_path, (2, 2), axis, compression="deflate", groups=(31,)
    )
    temporary = tmp_path / "dense_tiff/groups/31/.000000.tif.interrupted.tmp"
    temporary.write_bytes(b"partial TIFF")

    assert not store.plane_exists(0, group=31)
    assert not store.verify_plane(0, group=31)
