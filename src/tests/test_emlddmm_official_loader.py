from __future__ import annotations

import importlib.util
import json
import os
import subprocess
from pathlib import Path
from types import ModuleType

import matplotlib
import numpy as np
import pytest
from PIL import Image


matplotlib.use("Agg")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PIN_FILE = PROJECT_ROOT / "configs" / "emlddmm-upstream-commit.txt"


def _checkout() -> Path:
    configured = os.environ.get("EMLDDMM_REPO")
    if configured:
        return Path(configured).expanduser().resolve()
    return PROJECT_ROOT.parent / "emlddmm"


@pytest.fixture(scope="module")
def pinned_emlddmm() -> ModuleType:
    checkout = _checkout()
    if not (checkout / "emlddmm.py").exists():
        pytest.skip(f"Pinned EM-LDDMM checkout is unavailable: {checkout}")
    expected = PIN_FILE.read_text(encoding="utf-8").strip()
    actual = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert actual == expected

    spec = importlib.util.spec_from_file_location(
        "audit_pinned_emlddmm", checkout / "emlddmm.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_slice(
    directory: Path,
    name: str,
    value: int,
    *,
    z_step_um: float,
    slice_thickness_um: float,
    origin_xyz_um: tuple[float, float, float],
) -> None:
    Image.fromarray(
        np.full((4, 5, 3), value, dtype=np.uint8)
    ).save(directory / name, quality=100, subsampling=0)
    sidecar = {
        "DataFile": name,
        "Type": "uint8",
        "Dimension": 4,
        "Sizes": [3, 4, 5, 1],
        "Space": "right-inferior-posterior",
        "SpaceDimension": 3,
        "SpaceUnits": ["um", "um", "um"],
        "SpaceDirections": [
            "none",
            [2.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, 0.0, z_step_um],
        ],
        "SliceThickness": slice_thickness_um,
        "SpaceOrigin": list(origin_xyz_um),
    }
    (directory / f"{Path(name).stem}.json").write_text(
        json.dumps(sidecar), encoding="utf-8"
    )


def _write_samples(
    directory: Path, rows: list[tuple[str, str]]
) -> None:
    lines = ["sample_id\tparticipant_id\tspecies\tstatus"]
    lines.extend(f"{name}\tsynthetic\tphantom\t{status}" for name, status in rows)
    (directory / "samples.tsv").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _dataset(
    root: Path,
    *,
    origins_z: list[float],
    z_step_um: float,
    thicknesses_um: list[float],
) -> Path:
    root.mkdir()
    rows: list[tuple[str, str]] = []
    for index, (origin_z, thickness) in enumerate(
        zip(origins_z, thicknesses_um, strict=True)
    ):
        name = f"slice_{index:04d}.jpg"
        _write_slice(
            root,
            name,
            25 + index * 75,
            z_step_um=z_step_um,
            slice_thickness_um=thickness,
            origin_xyz_um=(1234.0, -987.0, origin_z),
        )
        rows.append((name, "present"))
    _write_samples(root, rows)
    return root


def test_loader_field_roles_and_axis_order(
    tmp_path: Path, pinned_emlddmm: ModuleType
) -> None:
    dataset = _dataset(
        tmp_path / "base",
        origins_z=[-200.0, 0.0, 200.0],
        z_step_um=200.0,
        thicknesses_um=[5.0, 50.0, 500.0],
    )
    axes, images, weights = pinned_emlddmm.load_slices(str(dataset))

    np.testing.assert_allclose(axes[0], [-200.0, 0.0, 200.0])
    # The loader reverses diagonal SpaceDirections: array row spacing comes
    # from the second spatial vector (3 um), and columns from the first (2 um).
    np.testing.assert_allclose(axes[1], [-4.5, -1.5, 1.5, 4.5])
    np.testing.assert_allclose(axes[2], [-4.0, -2.0, 0.0, 2.0, 4.0])
    # SpaceOrigin x/y are ignored and each in-plane image is centered.
    assert axes[1][0] != -987.0
    assert axes[2][0] != 1234.0
    assert images.shape == (3, 3, 4, 5)
    assert weights.shape == (3, 4, 5)


def test_slice_thickness_is_ignored(
    tmp_path: Path, pinned_emlddmm: ModuleType
) -> None:
    first = _dataset(
        tmp_path / "thin",
        origins_z=[-200.0, 0.0, 200.0],
        z_step_um=200.0,
        thicknesses_um=[1.0, 2.0, 3.0],
    )
    second = _dataset(
        tmp_path / "thick",
        origins_z=[-200.0, 0.0, 200.0],
        z_step_um=200.0,
        thicknesses_um=[100.0, 200.0, 300.0],
    )
    loaded_first = pinned_emlddmm.load_slices(str(first))
    loaded_second = pinned_emlddmm.load_slices(str(second))
    for left, right in zip(loaded_first, loaded_second, strict=True):
        if isinstance(left, list):
            for left_axis, right_axis in zip(left, right, strict=True):
                np.testing.assert_array_equal(left_axis, right_axis)
        else:
            np.testing.assert_array_equal(left, right)


def test_space_direction_z_controls_grid_step_and_optimization_geometry(
    tmp_path: Path, pinned_emlddmm: ModuleType
) -> None:
    coarse = _dataset(
        tmp_path / "coarse",
        origins_z=[-200.0, 0.0, 200.0],
        z_step_um=200.0,
        thicknesses_um=[50.0] * 3,
    )
    fine = _dataset(
        tmp_path / "fine",
        origins_z=[-100.0, 0.0, 100.0],
        z_step_um=100.0,
        thicknesses_um=[50.0] * 3,
    )
    coarse_axes, _, _ = pinned_emlddmm.load_slices(str(coarse))
    fine_axes, _, _ = pinned_emlddmm.load_slices(str(fine))
    assert np.diff(coarse_axes[0]).tolist() == [200.0, 200.0]
    assert np.diff(fine_axes[0]).tolist() == [100.0, 100.0]


def test_space_origin_z_controls_extent_but_tsv_controls_image_order(
    tmp_path: Path, pinned_emlddmm: ModuleType
) -> None:
    dataset = _dataset(
        tmp_path / "reverse_origins",
        origins_z=[200.0, 0.0, -200.0],
        z_step_um=200.0,
        thicknesses_um=[50.0] * 3,
    )
    axes, images, _ = pinned_emlddmm.load_slices(str(dataset))
    np.testing.assert_allclose(axes[0], [-200.0, 0.0, 200.0])
    means = images[0].mean(axis=(1, 2))
    assert means[0] < means[1] < means[2]


def test_absent_row_preserves_empty_grid_slot_without_placeholder_image(
    tmp_path: Path, pinned_emlddmm: ModuleType
) -> None:
    dataset = tmp_path / "absent"
    dataset.mkdir()
    _write_slice(
        dataset,
        "slice_0000.jpg",
        50,
        z_step_um=200.0,
        slice_thickness_um=50.0,
        origin_xyz_um=(0.0, 0.0, -200.0),
    )
    _write_slice(
        dataset,
        "slice_0002.jpg",
        200,
        z_step_um=200.0,
        slice_thickness_um=50.0,
        origin_xyz_um=(0.0, 0.0, 200.0),
    )
    _write_samples(
        dataset,
        [
            ("slice_0000.jpg", "present"),
            ("slice_0001.jpg", "absent"),
            ("slice_0002.jpg", "present"),
        ],
    )
    assert not (dataset / "slice_0001.jpg").exists()
    axes, _, weights = pinned_emlddmm.load_slices(str(dataset))
    np.testing.assert_allclose(axes[0], [-200.0, 0.0, 200.0])
    assert np.any(weights[0] > 0)
    assert not np.any(weights[1] > 0)
    assert np.any(weights[2] > 0)


def test_w0_is_operationally_first_channel_positive(
    tmp_path: Path, pinned_emlddmm: ModuleType
) -> None:
    dataset = tmp_path / "w0"
    dataset.mkdir()
    image = np.full((4, 5, 3), 100, dtype=np.uint8)
    image[1, 2, 0] = 0
    image[2, 3, :] = 0
    Image.fromarray(image).save(dataset / "slice_0000.tif")
    sidecar = {
        "DataFile": "slice_0000.tif",
        "Type": "uint8",
        "Dimension": 4,
        "Sizes": [3, 5, 4, 1],
        "Space": "right-inferior-posterior",
        "SpaceDimension": 3,
        "SpaceUnits": ["um", "um", "um"],
        "SpaceDirections": [
            "none",
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 50.0],
        ],
        "SpaceOrigin": [-4.0, -3.0, 0.0],
    }
    (dataset / "slice_0000.json").write_text(
        json.dumps(sidecar), encoding="utf-8"
    )
    _write_samples(dataset, [("slice_0000.tif", "present")])
    axes = [
        np.array([0.0]),
        np.arange(4) * 2.0 - 3.0,
        np.arange(5) * 2.0 - 4.0,
    ]
    _, _, weights = pinned_emlddmm.load_slices(str(dataset), xJ=axes)
    assert weights[0, 0, 0] == 1
    assert weights[0, 1, 2] == 0
    assert weights[0, 2, 3] == 0
    assert np.count_nonzero(weights) == 18


def test_multiscale_unwraps_nested_empty_local_contrast(
    pinned_emlddmm: ModuleType, monkeypatch
) -> None:
    captured: list[tuple[object, object, object]] = []

    def fake_core(**kwargs):
        captured.append(
            (
                kwargs["slice_matching"],
                kwargs["order"],
                kwargs["local_contrast"],
            )
        )
        return {
            "A": np.eye(4),
            "v": np.zeros((1, 3, 1, 1, 1)),
            "xv": [np.array([0.0])] * 3,
            "A2d": np.eye(3)[None],
        }

    monkeypatch.setattr(pinned_emlddmm, "emlddmm", fake_core)
    pinned_emlddmm.emlddmm_multiscale(
        downI=[[1, 1, 1]],
        downJ=[[1, 1, 1]],
        local_contrast=[[]],
        slice_matching=True,
        order=1,
        I=np.ones((1, 1, 1, 1)),
        J=np.ones((3, 1, 1, 1)),
    )
    assert captured == [(True, 1, [])]


def test_dataset_list_is_mandatory(
    tmp_path: Path, pinned_emlddmm: ModuleType
) -> None:
    with pytest.raises(FileNotFoundError):
        pinned_emlddmm.load_slices(str(tmp_path))


def test_affine_forward_reverse_and_matrix_file_direction(
    tmp_path: Path, pinned_emlddmm: ModuleType
) -> None:
    affine_zyx = np.eye(4)
    affine_zyx[:3, -1] = [30.0, -20.0, 10.0]
    points = np.array([1.0, 2.0, 3.0])[:, None, None, None]

    forward = pinned_emlddmm.Transform(affine_zyx, direction="f")
    reverse = pinned_emlddmm.Transform(affine_zyx, direction="b")
    mapped = forward(points)
    round_trip = reverse(mapped)
    np.testing.assert_allclose(
        mapped[:, 0, 0, 0], [31.0, -18.0, 13.0]
    )
    np.testing.assert_allclose(round_trip, points, atol=3e-6)

    matrix_path = tmp_path / "registered_to_input_matrix.txt"
    pinned_emlddmm.write_matrix_data(matrix_path, affine_zyx)
    reloaded = pinned_emlddmm.read_matrix_data(matrix_path)
    np.testing.assert_allclose(reloaded, affine_zyx)


def test_vtk_writer_extent_and_vector_component_round_trip(
    tmp_path: Path, pinned_emlddmm: ModuleType
) -> None:
    axes = [
        np.array([-200.0, 0.0, 200.0]),
        np.array([-3.0, 0.0, 3.0]),
        np.array([10.0, 12.0, 14.0, 16.0]),
    ]
    displacement_zyx = np.zeros((1, 3, 3, 3, 4), dtype=np.float32)
    displacement_zyx[:, 0] = 7.0
    displacement_zyx[:, 1] = 8.0
    displacement_zyx[:, 2] = 9.0
    output = tmp_path / "displacement.vtk"

    pinned_emlddmm.write_vtk_data(
        output, axes, displacement_zyx, "synthetic displacement"
    )
    read_axes, read_displacement, _, _ = pinned_emlddmm.read_vtk_data(output)
    for actual, expected in zip(read_axes, axes, strict=True):
        np.testing.assert_allclose(actual, expected)
    np.testing.assert_array_equal(read_displacement, displacement_zyx)
