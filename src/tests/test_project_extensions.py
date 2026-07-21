"""Small synthetic tests for project-only preprocessing extensions.

These tests intentionally do not test xIV-LDDMM optimization.  That requires the
configured repository and PyKeOps environment.  They test the pieces that this project
adds around xmodmap: affine-world geometry, 3-D block mass conservation, dense scalar
moments/bins, and the VTK coordinate adapter.
"""

from __future__ import annotations

import numpy as np

from dense_image_utils import IntensityTransform, construct_dense_image_particles
from particle_utils import FeatureDefinition, construct_particles
from xmodmap_compat import XmodmapAPI, write_vtk_xyz


def test_categorical_mass_conservation() -> None:
    labels = np.zeros((5, 6, 7), dtype=np.int16)
    labels[1:4, 1:5, 1:6] = 1
    labels[2:5, 3:6, 2:7] = 2
    affine = np.array(
        [[0.2, 0, 0, 10], [0, 0.3, 0, -4], [0, 0, 0.4, 7], [0, 0, 0, 1]],
        dtype=float,
    )
    features = FeatureDefinition((1, 2), ("one", "two"), ("both",), (0, 0))
    for factor in (1, 2, 5):
        _, nu_sub, nu_all, diagnostics = construct_particles(
            labels, affine, features, factor, "affine_world"
        )
        assert np.allclose(nu_all.sum(), diagnostics["expected_total_mass_mm3"])
        assert np.allclose(nu_sub.sum(), nu_all.sum())


def test_dense_histogram_mass_conservation() -> None:
    data = np.arange(5 * 6 * 7, dtype=float).reshape(5, 6, 7)
    mask = data > 20
    affine = np.array(
        [[0.2, 0, 0, 10], [0, 0.3, 0, -4], [0, 0, 0.4, 7], [0, 0, 0, 1]],
        dtype=float,
    )
    transform = IntensityTransform("fixed_range", 0.0, float(data.max()), True)
    expected = mask.sum() * abs(np.linalg.det(affine[:3, :3]))
    for factor in (1, 2, 5):
        _, nu_intensity, nu_bin, _, _ = construct_dense_image_particles(
            data,
            affine,
            mask,
            transform,
            factor,
            "affine_world",
            "block_center",
            {"enabled": True, "bins": 8, "assignment": "linear"},
        )
        assert nu_bin is not None
        assert np.allclose(nu_intensity[:, 0].sum(), expected)
        assert np.allclose(nu_bin.sum(), expected, rtol=1e-6)


def test_vtk_xyz_adapter_cancels_xmodmap_swap() -> None:
    captured: dict[str, np.ndarray] = {}

    def historical_write(points, arrays, names, filename):
        points = np.asarray(points)
        captured["vtk_points"] = points[:, [1, 0, 2]]

    api = XmodmapAPI(
        xmodmap=None,
        get_from_file=None,
        make_from_single_channel_image=None,
        make_bins_from_multichannel_image=None,
        get_entropy=None,
        get_mass_ratio=None,
        write_particle_vtk=None,
        write_vtk=historical_write,
        make_pq=None,
        resize_data=None,
    )
    xyz = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    write_vtk_xyz(api, xyz, [np.ones(2)], ["test"], "unused.vtk")
    assert np.array_equal(captured["vtk_points"], xyz)
