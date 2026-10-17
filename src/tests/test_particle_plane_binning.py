from __future__ import annotations

import numpy as np
import pytest

from preprocess.particle_utils import (
    bin_particles_2d,
    cut_plane,
    group_feature_columns,
    varifold_score_mm,
)


def _grid(spacing=0.5):
    axis = np.arange(-2.0, 2.0 + 1e-9, spacing)
    return np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)


def test_cut_plane_keeps_one_layer_and_hemisphere():
    X = _grid()
    nu = np.full((len(X), 1), 0.125)
    plane_X, (kept,), meta = cut_plane(
        X, [nu], origin_mm=[0, 0.5, 0], u_axis=[1, 0, 0], v_axis=[0, 0, 1],
        half_thickness_mm=0.25, hemisphere="left",
    )
    # One y-layer (9 x 9 points) minus x >= 0 columns -> 4 x 9 points.
    assert plane_X.shape == (36, 3)
    assert np.all(plane_X[:, 0] < 0) and np.all(plane_X[:, 2] == 0)
    np.testing.assert_allclose(kept, 0.25)  # 0.125 mm^3 / 0.5 mm
    np.testing.assert_allclose(meta["normal"], [0, -1, 0])


def test_cut_plane_basis_round_trip():
    X = _grid()
    origin, u, v = np.array([0.0, 0.5, 0.0]), np.array([1.0, 0, 0]), np.array([0, 0, 1.0])
    plane_X, _, meta = cut_plane(X, [np.ones((len(X), 1))], origin, u, v, 0.25)
    world = np.asarray(meta["origin_mm"]) + plane_X[:, :1] * u + plane_X[:, 1:2] * v
    assert np.all(np.isclose(world[:, 1], 0.5))
    original = X[np.isclose(X[:, 1], 0.5)]
    np.testing.assert_allclose(np.sort(world, axis=0), np.sort(original, axis=0))


def test_cut_plane_rejects_non_orthonormal_basis():
    with pytest.raises(ValueError):
        cut_plane(_grid(), [np.ones((125, 1))], [0, 0, 0], [1, 0, 0], [1, 1, 0], 0.25)


def test_bin_particles_conserves_mass_and_centroids():
    X = np.array([[0.1, 0.1, 0], [0.3, 0.1, 0], [0.7, 0.2, 0], [0.9, 0.9, 0]])
    nu = np.array([[1.0, 0], [0, 3.0], [2.0, 0], [0, 1.0]])
    X_bins, (binned,), counts = bin_particles_2d(X, [nu], bin_mm=0.5)
    np.testing.assert_allclose(binned.sum(axis=0), nu.sum(axis=0))
    assert sorted(counts.tolist()) == [1, 1, 2]
    first = np.argmin(X_bins[:, 0] + X_bins[:, 1])
    np.testing.assert_allclose(X_bins[first, :2], [(0.1 * 1 + 0.3 * 3) / 4, 0.1])
    assert np.all(X_bins[:, 2] == 0)


def test_group_feature_columns_sums_and_validates():
    nu = np.array([[1.0, 2.0, 3.0, 0.0]])
    labels = ["a", "b", "c", "__missing__"]
    grouped, names = group_feature_columns(nu, labels, {"ab": ["a", "b"], "c": ["c"]}, drop=["__missing__"])
    assert names == ["ab", "c"]
    np.testing.assert_allclose(grouped, [[3.0, 3.0]])
    with pytest.raises(ValueError):
        group_feature_columns(nu, labels, {"ab": ["a", "b"]}, drop=["__missing__"])
    with pytest.raises(KeyError):
        group_feature_columns(nu, labels, {"x": ["typo"], "rest": ["a", "b", "c"]}, drop=["__missing__"])


def test_varifold_score_zero_invariant_and_increasing():
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 5, size=(60, 2))
    nu = rng.uniform(0.1, 1.0, size=(60, 3))
    assert varifold_score_mm(X, nu, X, nu) == pytest.approx(0.0, abs=1e-12)

    angle = 0.7
    R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    shifted = X + np.array([0.5, 0.0])
    base = varifold_score_mm(shifted, nu, X, nu)
    moved = varifold_score_mm(shifted @ R.T + 3.0, nu, X @ R.T + 3.0, nu)
    assert moved == pytest.approx(base, rel=1e-9)

    scores = [varifold_score_mm(X + [dx, 0.0], nu, X, nu) for dx in (0.25, 0.5, 1.0, 2.0)]
    assert all(a < b for a, b in zip(scores, scores[1:]))
