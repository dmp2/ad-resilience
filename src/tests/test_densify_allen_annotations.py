from __future__ import annotations

import csv
import json

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


def _write_anchor_groups(tmp_path, values):
    root = dense.zarr.open_group(str(tmp_path / "anchors.zarr"), mode="w")
    groups = root.require_group("groups")
    for group, array in values.items():
        value = np.asarray(array, dtype=np.uint32)
        groups.create_array(str(group), data=value, chunks=(1, *value.shape[1:]))
    return root


def _minimal_context(tmp_path, groups=(31, 265297118)):
    return dense.SourceContext(
        dataset=tmp_path,
        annotations=tmp_path,
        registration=tmp_path,
        numerical=tmp_path / "numerical.npz",
        rows=tuple({"allen_section_number": str(100 + i)} for i in range(3)),
        annotation_sections=(100, 102),
        annotation_inventory={},
        physical_by_section={100: 0, 102: 2},
        observed_nissl=np.array([0, 2]),
        axes=(
            np.array([0.0, 5.0, 20.0]),
            np.arange(2, dtype=np.float32),
            np.arange(2, dtype=np.float32),
        ),
        final_a2d=np.zeros((3, 3, 3)),
        registered_axes=(
            np.arange(2, dtype=np.float32),
            np.arange(2, dtype=np.float32),
        ),
        source_hashes={"authoritative": "abc123"},
        registration_identifier="registration:test",
        graphic_groups=tuple(groups),
        shape=(2, 2),
        canonical_count=3,
        serial_spacing_um=5.0,
        pixel_size_um=200.0,
    )


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


def test_endpoint_sequences_preserve_different_group_availability_patterns():
    rows = [
        {"physical_index": "0", "graphic_group": "31", "semantic_state": "LABELED"},
        {"physical_index": "4", "graphic_group": "31", "semantic_state": "LABELED"},
        {"physical_index": "9", "graphic_group": "31", "semantic_state": "LABELED"},
        {"physical_index": "0", "graphic_group": "99", "semantic_state": "LABELED"},
        {"physical_index": "9", "graphic_group": "99", "semantic_state": "LABELED"},
    ]

    sequences, pairs = dense.build_endpoint_sequences(rows, (31, 99))

    assert sequences == {31: [0, 4, 9], 99: [0, 9]}
    assert pairs == {(0, 4): (31,), (0, 9): (99,), (4, 9): (31,)}


def test_annotation_pair_driver_has_joint_ordered_group_id_channels(tmp_path):
    root = _write_anchor_groups(
        tmp_path,
        {
            31: [
                [[5, 0], [0, 0]],
                [[0, 7], [0, 0]],
            ],
            99: [
                [[0, 5], [0, 0]],
                [[0, 0], [5, 0]],
            ],
        },
    )

    left, right, wleft, wright, keys, report = (
        dense.build_annotation_pair_driver(root, 0, 1, (31, 99))
    )

    assert keys == ((31, 5), (31, 7), (99, 5))
    assert left.dtype == right.dtype == np.float32
    assert left.shape == right.shape == (3, 2, 2)
    np.testing.assert_array_equal(left[1], 0.0)  # ID 7 is absent on the left.
    np.testing.assert_array_equal(right[0], 0.0)  # Group-31 ID 5 is absent right.
    assert not np.array_equal(left[0], left[2])  # Same Allen ID, distinct groups.
    np.testing.assert_array_equal(wleft, np.array([[1, 1], [0, 0]], np.float32))
    np.testing.assert_array_equal(wright, np.array([[0, 1], [1, 0]], np.float32))
    assert report["ordered_roi_channel_keys"] == [[31, 5], [31, 7], [99, 5]]
    assert report["driver_channel_count"] == 3


def test_annotation_pair_driver_vocabulary_changes_between_pairs(tmp_path):
    root = _write_anchor_groups(
        tmp_path,
        {
            31: [
                [[1, 0], [0, 0]],
                [[2, 0], [0, 0]],
                [[3, 0], [0, 0]],
            ]
        },
    )

    *_, keys_01, _ = dense.build_annotation_pair_driver(root, 0, 1, (31,))
    *_, keys_12, _ = dense.build_annotation_pair_driver(root, 1, 2, (31,))

    assert keys_01 == ((31, 1), (31, 2))
    assert keys_12 == ((31, 2), (31, 3))


def test_annotation_pair_driver_rejects_zero_channel_pair(tmp_path):
    root = _write_anchor_groups(
        tmp_path,
        {31: np.zeros((2, 2, 2), dtype=np.uint32)},
    )

    with pytest.raises(RuntimeError, match="no nonzero ROI channels"):
        dense.build_annotation_pair_driver(root, 0, 1, (31,))


def test_graphic_group_selection_rejects_duplicates_and_unknown_ids(tmp_path):
    context = _minimal_context(tmp_path)

    selected = dense.select_graphic_groups(context, (265297118, 31))
    assert selected.graphic_groups == (31, 265297118)
    with pytest.raises(RuntimeError, match="Duplicate"):
        dense.select_graphic_groups(context, (31, 31))
    with pytest.raises(RuntimeError, match="Unknown"):
        dense.select_graphic_groups(context, (31, 999))


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


def test_restricted_tiff_store_allocates_only_selected_groups(tmp_path):
    store = dense.TiffDenseAnnotationStore(
        tmp_path,
        (2, 2),
        np.array([0.0, 1.0]),
        compression="none",
        groups=(31, 265297118),
    )

    assert store.groups == (31, 265297118)
    assert (tmp_path / "dense_tiff/groups/31").is_dir()
    assert (tmp_path / "dense_tiff/groups/265297118").is_dir()
    assert sorted(path.name for path in (tmp_path / "dense_tiff/groups").iterdir()) == [
        "265297118",
        "31",
    ]


def test_checkpoint_identity_distinguishes_nissl_from_annotation(tmp_path):
    context = _minimal_context(tmp_path)
    nissl_keys, nissl_report = dense._nissl_driver_description()
    nissl_identity = dense._checkpoint_identity(
        context,
        driver="nissl",
        pair_groups=(31,),
        driver_keys=nissl_keys,
        driver_report=nissl_report,
        config={"nt": 10},
        wsi_commit="wsi",
    )
    annotation_report = {
        "registration_support": "endpoint foreground union across selected ROI channels"
    }
    annotation_identity = dense._checkpoint_identity(
        context,
        driver="annotation",
        pair_groups=(31,),
        driver_keys=((31, 10),),
        driver_report=annotation_report,
        config={"nt": 10},
        wsi_commit="wsi",
    )
    status = {
        "checkpoint_identity": nissl_identity,
        "checkpoint_identity_sha256": dense._stable_json_sha256(nissl_identity),
    }

    assert dense._checkpoint_matches(status, nissl_identity)
    assert not dense._checkpoint_matches(status, annotation_identity)
    assert not dense._legacy_nissl_checkpoint_matches(status, annotation_identity)


def test_process_pair_preserves_nissl_inputs_reuses_partial_outputs_and_skips(
    monkeypatch, tmp_path
):
    context = _minimal_context(tmp_path)
    root = _write_anchor_groups(
        tmp_path,
        {
            31: [
                [[10, 10], [10, 10]],
                [[20, 20], [20, 20]],
            ],
            265297118: [
                [[30, 30], [30, 30]],
                [[40, 40], [40, 40]],
            ],
        },
    )
    nissl = np.arange(24, dtype=np.float32).reshape(2, 3, 2, 2) / 24.0
    weights = np.array(
        [[[1, 0], [1, 1]], [[0, 1], [1, 1]]], dtype=np.float32
    )
    root.create_array("nissl", data=nissl, chunks=(1, 3, 2, 2))
    root.create_array("nissl_weight", data=weights, chunks=(1, 2, 2))
    store = dense.TiffDenseAnnotationStore(
        tmp_path,
        (2, 2),
        context.axes[0],
        compression="none",
        groups=context.graphic_groups,
    )
    expected = {31: np.full((2, 2), 10, np.uint32), 265297118: np.full((2, 2), 30, np.uint32)}
    mtimes = {}
    for group, plane in expected.items():
        store.write_group_plane(group, 1, plane, overwrite=False)
        path = tmp_path / f"dense_tiff/groups/{group}/000001.tif"
        mtimes[group] = path.stat().st_mtime_ns

    driver_keys, driver_report = dense._nissl_driver_description()
    identity = dense._checkpoint_identity(
        context,
        driver="nissl",
        pair_groups=context.graphic_groups,
        driver_keys=driver_keys,
        driver_report=driver_report,
        config={"nt": 10},
        wsi_commit="wsi-test",
    )
    captured = {}

    class Flow:
        def __init__(self):
            self.calls = []

        def evaluate(self, t):
            self.calls.append(t)
            return np.zeros((2, 2, 2), dtype=np.float32)

    left_flow, right_flow = Flow(), Flow()

    class Cuda:
        @staticmethod
        def is_available():
            return False

    class Torch(_TorchFacade):
        cuda = Cuda()

    def fake_fit(left, right, wleft, wright, axes, config, **kwargs):
        captured["arrays"] = tuple(value.copy() for value in (left, right, wleft, wright))
        return left_flow, right_flow, {"wsi_commit": "wsi-test"}, _IdentityInterpolation(), Torch()

    monkeypatch.setattr(dense, "fit_pair_trajectories", fake_fit)
    report = dense.process_pair(
        context,
        tmp_path,
        store,
        (0, 2),
        context.graphic_groups,
        {"nt": 10},
        driver="nissl",
        checkpoint_identity=identity,
        nt_source="test",
        wsi_repository=tmp_path,
        device="cpu",
        overwrite=False,
    )

    for actual, wanted in zip(captured["arrays"], (nissl[0], nissl[1], weights[0], weights[1])):
        np.testing.assert_array_equal(actual, wanted)
    assert left_flow.calls == [0.25]
    assert right_flow.calls == [0.75]
    for group, plane in expected.items():
        path = tmp_path / f"dense_tiff/groups/{group}/000001.tif"
        assert path.stat().st_mtime_ns == mtimes[group]
        np.testing.assert_array_equal(store.read_group_plane(group, 1), plane)
        assert report["group_reports"][str(group)]["compatible_existing_planes_reused"] == 1

    monkeypatch.setattr(
        dense,
        "fit_pair_trajectories",
        lambda *args, **kwargs: pytest.fail("compatible checkpoint should skip fitting"),
    )
    repeated = dense.process_pair(
        context,
        tmp_path,
        store,
        (0, 2),
        context.graphic_groups,
        {"nt": 10},
        driver="nissl",
        checkpoint_identity=identity,
        nt_source="test",
        wsi_repository=tmp_path,
        device="cpu",
        overwrite=False,
    )
    assert repeated["checkpoint_identity_sha256"] == dense._stable_json_sha256(identity)


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
    annotation_args = parser.parse_args(
        ["--driver", "annotation", "--graphic-groups", "31", "265297118"]
    )

    assert defaults.output_format == "tiff"
    assert defaults.tiff_compression == "deflate"
    assert defaults.driver == "nissl"
    assert zarr_args.output_format == "zarr"
    assert annotation_args.driver == "annotation"
    assert annotation_args.graphic_groups == [31, 265297118]


def test_store_factory_selects_existing_zarr_backend(monkeypatch, tmp_path):
    selected = object()

    def fake_zarr_store(output, shape, canonical_z_um, groups):
        assert output == tmp_path
        assert shape == (2, 2)
        np.testing.assert_array_equal(canonical_z_um, [0.0])
        assert groups == (19, 7)
        return selected

    monkeypatch.setattr(dense, "ZarrDenseAnnotationStore", fake_zarr_store)

    result = dense.create_dense_store(
        tmp_path,
        (2, 2),
        np.array([0.0]),
        output_format="zarr",
        tiff_compression="deflate",
        groups=(19, 7),
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


def test_zarr_store_uses_programmatic_dimensions_and_group_order(tmp_path):
    axis = np.array([-12.5, 7.25, 99.0], dtype=np.float64)
    store = dense.ZarrDenseAnnotationStore(tmp_path, (3, 5), axis, groups=(9001, 17))

    assert tuple(store.root["z_um"].shape) == (3,)
    assert tuple(store.root["groups"]["9001"].shape) == (3, 3, 5)
    assert tuple(store.root["groups"]["17"].shape) == (3, 3, 5)
    assert tuple(store.states.shape) == (2, 3)
    assert store.groups == (9001, 17)


def test_endpoint_sequences_use_supplied_programmatic_group_order():
    rows = [
        {"physical_index": "2", "graphic_group": "17", "semantic_state": "LABELED"},
        {"physical_index": "8", "graphic_group": "17", "semantic_state": "LABELED"},
        {"physical_index": "2", "graphic_group": "9001", "semantic_state": "LABELED"},
        {"physical_index": "8", "graphic_group": "9001", "semantic_state": "LABELED"},
    ]

    _, pairs = dense.build_endpoint_sequences(rows, graphic_groups=(9001, 17))

    assert pairs[(2, 8)] == (9001, 17)


def test_pair_parser_has_no_example_specific_upper_bound():
    assert dense._parse_pair("3000-4000") == (3000, 4000)


def test_annotation_derivative_is_discovered_from_selected_dataset(tmp_path):
    dataset = tmp_path / "nissl_example"
    annotations = tmp_path / "annotations_example"
    dataset.mkdir()
    annotations.joinpath("metadata").mkdir(parents=True)
    dataset.joinpath("dataset.json").write_text("{}\n")
    annotations.joinpath("dataset.json").write_text(
        json.dumps({"parent_nissl_derivative": "../nissl_example"}) + "\n"
    )
    annotations.joinpath("metadata/annotations.tsv").write_text("header\n")
    annotations.joinpath("metadata/source_annotation_manifest.tsv").write_text(
        "header\n"
    )

    assert (
        dense._discover_annotation_derivative(dataset.resolve())
        == annotations.resolve()
    )


def test_annotation_derivative_discovery_rejects_ambiguity(tmp_path):
    dataset = tmp_path / "nissl_example"
    dataset.mkdir()
    dataset.joinpath("dataset.json").write_text("{}\n")
    for name in ("annotations_a", "annotations_b"):
        candidate = tmp_path / name
        candidate.joinpath("metadata").mkdir(parents=True)
        candidate.joinpath("dataset.json").write_text(
            json.dumps({"parent_nissl_derivative": "../nissl_example"}) + "\n"
        )
        candidate.joinpath("metadata/annotations.tsv").write_text("header\n")
        candidate.joinpath("metadata/source_annotation_manifest.tsv").write_text(
            "header\n"
        )

    with pytest.raises(RuntimeError, match="pass --annotations explicitly"):
        dense._discover_annotation_derivative(dataset.resolve())


def test_custom_dataset_requires_explicit_registration_and_output(tmp_path):
    with pytest.raises(SystemExit):
        dense.main(["--dataset", str(tmp_path)])
    with pytest.raises(SystemExit):
        dense.main(
            [
                "--dataset",
                str(tmp_path),
                "--registration-run",
                str(tmp_path / "registration"),
            ]
        )


def test_default_example_paths_resolve_before_run(monkeypatch):
    received = {}

    def fake_run(**kwargs):
        received.update(kwargs)
        return {"status": "test"}

    monkeypatch.setattr(dense, "run", fake_run)

    assert dense.main([]) == 0
    assert received["dataset"] == dense.DEFAULT_DATASET.resolve()
    assert received["registration"] == dense.DEFAULT_REGISTRATION
    assert received["output"] == dense.DEFAULT_OUTPUT.resolve()
    assert received["driver"] == "nissl"


def test_annotation_default_uses_dedicated_output(monkeypatch):
    received = {}

    def fake_run(**kwargs):
        received.update(kwargs)
        return {"status": "test"}

    monkeypatch.setattr(dense, "run", fake_run)

    assert dense.main(["--driver", "annotation"]) == 0
    assert received["output"] == dense.DEFAULT_ANNOTATION_OUTPUT.resolve()
    assert received["graphic_groups"] == [31, 265297118]
