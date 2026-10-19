"""Synthetic checks for QNP particles, 2-D similarity search and atlas reslicing."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(SRC), str(SRC / "preprocess")]

from preprocess.particle_utils import cut_plane  # noqa: E402
from preprocess.qnp_particles import canonical_layer, tissue_class_particles  # noqa: E402
from preprocess.section_alignment import (  # noqa: E402
    FootprintScorer,
    Similarity2D,
    label_class_masses,
    lookup_tables,
    plane_basis,
    reslice_labels,
    search,
)


@pytest.mark.parametrize("name, expected", [
    ("Layer1", "L1"), ("LAyerIII", "L3"), ("LAyer IV", "L4"), ("Layer V-VI", "L5-6"),
    ("Layer5-6", "L5-6"), ("LayerII", "L2"), ("Vessel", None),
])
def test_canonical_layer_names(name, expected):
    assert canonical_layer(name) == expected


def test_tissue_class_particles_conserve_area_and_split_by_membership():
    mask = np.zeros((40, 60), bool)
    mask[5:35, 10:50] = True
    p_wm = np.where(np.arange(60)[None, :] < 30, 0.0, 1.0) * mask
    px = (0.01, 0.01)
    X, nu, diagnostics = tissue_class_particles(px, mask, p_wm, bin_mm=0.1)
    assert nu.sum() == pytest.approx(mask.sum() * 1e-4, rel=1e-6)  # float32 archive precision
    assert nu[:, 0].sum() == pytest.approx(30 * 20 * 1e-4)  # columns 10..29 are gray
    assert np.all(X[:, 2] == 0)
    # Block centres of 10-pixel blocks (x = column, y = row): mask columns 10-49 -> blocks 1-4.
    np.testing.assert_allclose(np.unique(X[:, 0]), [0.145, 0.245, 0.345, 0.445], atol=1e-6)
    np.testing.assert_allclose(np.unique(X[:, 1]), [0.045, 0.145, 0.245, 0.345], atol=1e-6)
    assert diagnostics["downsample_factor"] == 10


def test_similarity_round_trip():
    t = Similarity2D(33.0, 1.2, True, (3.0, -2.0))
    X = np.random.default_rng(0).normal(size=(20, 2))
    np.testing.assert_allclose(t.moving_to_section(t.section_to_moving(X)), X, atol=1e-12)
    assert Similarity2D.from_dict(t.as_dict()) == t


def _two_class_shape():
    """An asymmetric 'L' of gray tissue with a white-matter bar (1 mm particles)."""

    xs, ys = np.meshgrid(np.arange(0, 20.0), np.arange(0, 14.0))
    X = np.column_stack([xs.ravel(), ys.ravel()])
    keep = (X[:, 1] < 4) | (X[:, 0] < 5)
    X = X[keep]
    white = (X[:, 1] < 2) & (X[:, 0] > 3)
    nu = np.column_stack([~white, white]).astype(float)
    return X, nu


def test_search_recovers_rotation_and_reflection():
    X, nu = _two_class_shape()
    truth = Similarity2D(70.0, 1.0, True, (8.0, 5.0))
    P = truth.section_to_moving(X)  # the moving measure is the section seen through `truth`
    scorer = FootprintScorer(X, nu, sigmas_mm=(2.0, 1.0))
    score, found = search(scorer, P, nu, thetas=np.arange(0, 360, 15), scales=[1.0], flips=[False, True],
                          scale_bounds=(1.0, 1.0), h=0.5, blur_mm=1.0, refine_top=3, max_evals=300)[0]
    assert found.flip is True
    assert abs((found.theta_deg - 70.0 + 180) % 360 - 180) < 2.0
    np.testing.assert_allclose(found.moving_to_section(P), X, atol=0.3)
    assert score < 1e-2


def test_footprint_score_is_one_without_overlap():
    X, nu = _two_class_shape()
    scorer = FootprintScorer(X, nu)
    assert scorer.score(X + 100.0, nu) == 1.0
    assert scorer.score(X, nu) == pytest.approx(0.0, abs=1e-9)


def test_reslice_matches_cut_plane_for_a_coronal_plane():
    rng = np.random.default_rng(1)
    labels = rng.integers(0, 3, size=(12, 10, 8))
    affine = np.diag([0.5, 0.5, 0.5, 1.0])
    affine[:3, 3] = [-3.0, -2.5, -2.0]
    u, v, n = plane_basis(0.0, 0.0)
    np.testing.assert_allclose(n, [0, -1, 0])
    origin = np.array([0.0, -0.5, 0.0])  # a voxel-centre plane
    grid, axis = reslice_labels(labels, affine, origin, u, v, half_width_mm=2.0, spacing_mm=0.5)
    ijk = np.argwhere(labels == 2)
    world = ijk * 0.5 + affine[:3, 3]
    plane_X, (kept,), _ = cut_plane(world, [np.ones((len(world), 1))], origin, u, v, 0.25)
    inside = (np.abs(plane_X[:, 0]) <= 2.0) & (np.abs(plane_X[:, 1]) <= 2.0)
    from_cut = {tuple(np.round(p, 6)) for p in plane_X[inside, :2]}
    bb, aa = np.nonzero(grid == 2)
    from_reslice = {(round(axis[a], 6), round(axis[b], 6)) for a, b in zip(aa, bb)}
    assert from_cut == from_reslice


def test_lookup_tables_partition_labels():
    coarse, fine = lookup_tables(10, white_matter=[5], drop=[7], fine_regions={"MTG": [3]})
    assert not coarse["gray_matter"][0] and not coarse["white_matter"][0]
    assert coarse["white_matter"][5] and not coarse["gray_matter"][5]
    assert not coarse["gray_matter"][7] and not coarse["white_matter"][7]
    assert fine["MTG"][3] and not fine["other_gray_matter"][3] and fine["other_gray_matter"][4]
    grid = np.array([[3, 5], [7, 4]])
    X, nu = label_class_masses(grid, np.array([0.0, 0.5]), 0.5, coarse)
    assert nu.sum() == pytest.approx(3 * 0.25)  # the dropped pixel carries no mass
