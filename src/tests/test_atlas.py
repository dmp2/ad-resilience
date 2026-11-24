"""Anatomical localization: label tables, feature channels, reslicing, landmark and slab prior.

The last feature test reads the real AHRAv2 volume and is skipped when it is not downloaded.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1]
REPO = SRC.parent
sys.path[:0] = [str(SRC), str(SRC / "preprocess"), str(SRC / "config_setup"), str(SRC / "scripts")]

from preprocess.particle_utils import FeatureDefinition, cut_plane, save_particle_archive  # noqa: E402
from section_mapping.atlas import (  # noqa: E402
    PROJECTIONS,
    anatomical_tables,
    build_slab_prior,
    candidate_slabs,
    check_gmwm_reproduces,
    clamp_ys,
    classify_points,
    label_channel_index,
    label_class_masses,
    landing,
    locate_landmark,
    lookup_tables,
    offset_cover,
    plane_basis,
    posterior_mm,
    project,
    projection_matrix,
    projection_metadata,
    reslice_labels,
    search_interval,
    slab_consistency,
)


# --------------------------------------------------------------------------------------
# Label tables, reslicing, landing
# --------------------------------------------------------------------------------------

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


def test_classify_and_landing_share_one_lookup():
    labels = np.zeros((4, 4, 4), np.int32)
    labels[0] = 3          # MTG gray
    labels[1] = 5          # white matter
    labels[2] = 7          # dropped
    affine = np.eye(4)
    coarse, fine = lookup_tables(10, white_matter=[5], drop=[7], fine_regions={"MTG": [3]})
    atlas = {"labels": labels, "affine": affine, "fine": fine}
    world = np.array([[0, 1, 1], [1, 1, 1], [2, 1, 1], [3, 1, 1], [9, 1, 1]], float)
    names = ["MTG", "other_gray_matter", "white_matter"]
    # MTG, white, dropped, background (outside), off-volume (outside)
    assert classify_points(atlas, world, names).tolist() == [0, 2, 4, 3, 3]

    class Identity:
        @staticmethod
        def section_to_moving(X):
            return X

    plane = {"origin": np.zeros(3), "u": np.array([1.0, 0, 0]), "v": np.array([0, 1.0, 0])}
    fractions = landing(plane, atlas, Identity, world[:, :2], np.ones(len(world)), {n: fine[n] for n in names})
    assert fractions == {"MTG": 0.2, "other_gray_matter": 0.0, "white_matter": 0.2, "outside": 0.6}


# --------------------------------------------------------------------------------------
# Anatomical feature channels and projections
# --------------------------------------------------------------------------------------

# Toy atlas codes: 1 STG, 2 MTG, 3 ITG, 4 FuG, 5 insula, 6 WM, 7 ventricle, 8 WM tract,
# 9 an unlisted gray code (like the unnamed AHRAv2 codes 32 and 248).
FINE = {"MTG": [2], "STG": [1], "ITG": [3], "FuG": [4]}
WM, DROP = [6, 8], [7]
NAMES = ["STG_gray_matter", "MTG_gray_matter", "ITG_gray_matter", "other_gray_matter", "white_matter"]


def toy_tables(max_label=9):
    return anatomical_tables(max_label, WM, DROP, FINE)


def test_channels_are_ordered_one_hot_and_keep_other_gray():
    tables = toy_tables()
    assert list(tables) == NAMES
    index = label_channel_index(tables)
    #                 0   1  2  3  4  5  6   7  8  9
    assert index.tolist() == [-1, 0, 1, 2, 3, 3, 4, -1, 4, 3]
    # FuG and the insula-like code 5 are kept, as other gray matter, not excluded.
    assert tables["other_gray_matter"][[4, 5]].all()


def test_region_codes_listed_as_white_stay_white():
    tables = anatomical_tables(9, [6, 8, 2], DROP, FINE)
    assert not tables["MTG_gray_matter"].any()
    assert tables["white_matter"][2]


def test_unknown_gray_region_is_an_error():
    with pytest.raises(KeyError):
        anatomical_tables(9, WM, DROP, FINE, gray_regions=("STG", "MTG", "TP"))


def test_projection_matrices():
    names, M = projection_matrix("gmwm", NAMES)
    assert names == ("gray_matter", "white_matter")
    assert M.tolist() == [[1, 1, 1, 1, 0], [0, 0, 0, 0, 1]]
    names, M = projection_matrix("mtg_gm", NAMES)
    assert names == ("MTG_gray_matter",) and M.tolist() == [[0, 1, 0, 0, 0]]
    names, M = projection_matrix("anatomical", NAMES)
    assert list(names) == NAMES and np.array_equal(M, np.eye(5))
    with pytest.raises(ValueError):
        projection_matrix("learned", NAMES)
    with pytest.raises(ValueError):
        projection_matrix("mtg_gm", [n for n in NAMES if n != "MTG_gray_matter"])
    assert set(projection_metadata(NAMES)) == set(PROJECTIONS)


def test_gmwm_projection_reproduces_original_tables():
    tables = toy_tables()
    coarse, _ = lookup_tables(9, WM, DROP, FINE)
    check_gmwm_reproduces(tables, coarse)
    broken = dict(tables)
    broken["white_matter"] = tables["white_matter"].copy()
    broken["white_matter"][6] = False
    with pytest.raises(ValueError):
        check_gmwm_reproduces(broken, coarse)


def test_plane_masses_project_to_the_original_features():
    rng = np.random.default_rng(0)
    grid = rng.integers(0, 10, size=(12, 12))
    axis = np.arange(12) * 0.5
    tables = toy_tables()
    coarse, fine = lookup_tables(9, WM, DROP, FINE)
    for aggregate in (1, 2, 3):
        X, nu_anat = label_class_masses(grid, axis, 0.5, tables, aggregate)
        Xc, nu_coarse = label_class_masses(grid, axis, 0.5, coarse, aggregate)
        Xf, nu_fine = label_class_masses(grid, axis, 0.5, fine, aggregate)
        assert np.array_equal(X, Xc) and np.array_equal(X, Xf)
        assert np.array_equal(project(nu_anat, projection_matrix("gmwm", NAMES)[1]), nu_coarse)
        assert np.array_equal(project(nu_anat, projection_matrix("mtg_gm", NAMES)[1]), nu_fine[:, [0]])
        assert np.array_equal(project(nu_anat, projection_matrix("anatomical", NAMES)[1]), nu_anat)
    # Pixel level: one-hot, and the argmax recovers each pixel's channel from the labels.
    X, nu = label_class_masses(grid, axis, 0.5, tables, 1)
    assert np.all((nu > 0).sum(1) == 1)
    expected = label_channel_index(tables)[grid.ravel()]
    assert np.array_equal(nu.argmax(1), expected[expected >= 0])


def test_archive_carries_anatomical_features(tmp_path):
    X = np.zeros((3, 3))
    nu_anat = np.array([[0.25, 0, 0, 0, 0], [0, 0, 0, 0.25, 0], [0, 0, 0, 0, 0.25]])
    nu_sub = project(nu_anat, projection_matrix("gmwm", NAMES)[1])
    features = FeatureDefinition((0, 1), ("gray_matter", "white_matter"), ("gray_matter", "white_matter"), (0, 1))
    path = tmp_path / "plane.npz"
    save_particle_archive(path, X, nu_sub, nu_sub, features, {"coordinate_space": "mni_icbm2009b_symmetric_plane"},
                          write_vtk=False, extra_arrays={"nu_Anat": nu_anat.astype(np.float32),
                                                         "anatomical_feature_names": np.asarray(NAMES)})
    with np.load(path) as archive:
        assert archive.files[:3] == ["X", "nu_Sub", "nu_All"]
        assert archive["anatomical_feature_names"].tolist() == NAMES
        assert np.array_equal(archive["nu_Anat"], nu_anat.astype(np.float32))
    with pytest.raises(ValueError):
        save_particle_archive(path, X, nu_sub, nu_sub, features, {}, write_vtk=False, extra_arrays={"nu_Sub": nu_sub})


ATLAS = REPO / "data/raw/allen/ahrav2/2025/ahrav2_gyral_annotations_500um.nii.gz"


@pytest.mark.skipif(not ATLAS.is_file(), reason="AHRAv2 gyral annotation not downloaded")
def test_real_atlas_channels_match_the_segmentation():
    pytest.importorskip("nibabel")
    from config_setup.project_config import load_config
    from config_setup.qnp_jobs import resolve_qnp_job
    from scripts.qnp_atlas_planes import _anatomy_checks, atlas_setup

    config = load_config(REPO / "configs/project_config.yaml")
    scan = resolve_qnp_job(config, "H20.33.034_MTG")["atlas_scan"]
    atlas = atlas_setup(config, scan)  # raises if gmwm != the original gray/white tables
    assert list(atlas["anatomical"]) == NAMES
    checks = _anatomy_checks(atlas, scan, config)
    assert checks["unassigned_nonzero_voxels"] == 0
    assert checks["gmwm_projection_voxels_differing"] == 0
    for name, channel in checks["channels"].items():
        assert channel["voxels_differing"] == 0, name
        assert channel.get("allen_parcellation_voxels", channel["voxels"]) == channel["voxels"], name
    assert checks["channels"]["MTG_gray_matter"]["voxels"] == 415406  # Allen parcellation volume, code 30
    json.dumps(checks)


# --------------------------------------------------------------------------------------
# Landmark and slab prior
# --------------------------------------------------------------------------------------

LGN = 7


def _volume(flip_j: bool = False):
    """40 x 60 x 30 voxels at 0.5 mm, RAS world; a 'landmark' blob spanning world y -22 ... -28.

    One isolated voxel sits at y -20 (anterior to the blob) to exercise min_area_mm2.
    With ``flip_j`` the voxel j axis runs posterior (negative affine y column): same world.
    """

    labels = np.zeros((40, 60, 30), np.int16)
    affine = np.array([[0.5, 0, 0, -15.0], [0, 0.5, 0, -40.0], [0, 0, 0.5, -10.0], [0, 0, 0, 1]])
    j = lambda y: int(round((y - affine[1, 3]) / 0.5))
    labels[5:10, j(-28.0):j(-22.0) + 1, 10:14] = LGN   # x < 0 (left), 5 x 4 voxels = 5 mm^2 per plane
    labels[30:32, j(-25.0), 10:12] = LGN              # right hemisphere copy, smaller
    labels[6, j(-20.0), 11] = LGN                      # stray voxel: 0.25 mm^2
    if flip_j:
        labels = labels[:, ::-1, :]
        flip = np.diag([1.0, -1.0, 1.0, 1.0])
        flip[1, 3] = labels.shape[1] - 1
        affine = affine @ flip
    return labels, affine


@pytest.mark.parametrize("flip_j", [False, True])
def test_landmark_first_appearance_uses_world_y_not_voxel_index(flip_j):
    labels, affine = _volume(flip_j)
    lm = locate_landmark(labels, affine, {"name": "LGN", "codes": [LGN], "min_area_mm2": 1.0}, "left")
    assert lm["y_mm"] == pytest.approx(-22.0)          # most anterior plane with >= 1 mm^2
    assert lm["first_any_voxel_y_mm"] == pytest.approx(-20.0)  # the stray voxel, rejected by area
    assert lm["s_mm"] == pytest.approx(22.0)            # s = -y increases posteriorly
    assert lm["area_at_landmark_mm2"] == pytest.approx(5.0)
    assert lm["centroid_mm"][0] < 0 and lm["centroid_mm"][1] == pytest.approx(-22.0)
    ys = [r[0] for r in lm["profile_y_area_mm2"]]
    assert ys == sorted(ys, reverse=True)               # reported anterior -> posterior


def test_manual_landmark_and_missing_label():
    labels, affine = _volume()
    lm = locate_landmark(labels, affine, {"name": "LGN", "codes": [LGN], "y_mm": -24.5}, "left")
    assert lm["y_mm"] == -24.5 and lm["method"].startswith("manual")
    assert lm["label_first_appearance_y_mm_for_comparison"] == pytest.approx(-20.0)
    with pytest.raises(ValueError, match="manually"):
        locate_landmark(labels, affine, {"name": "X", "codes": [99]}, "left")


def test_posterior_coordinate_direction():
    assert posterior_mm(-30.0) > posterior_mm(-20.0)   # more negative MNI y = more posterior


def test_slab_construction_4mm_and_offsets():
    s_L, h = 22.0, 4.0
    slabs = candidate_slabs(s_L, h, [-1, 0, 1], 0.5)
    assert len(slabs) == 3 * 8
    for c in slabs:
        s0, s1 = c["s_mm"]
        assert s1 - s0 == pytest.approx(h)
        assert c["y_mm"] == [pytest.approx(-s0), pytest.approx(-s1)]  # [anterior, posterior]
        assert 0.0 <= c["delta_mm"] < h
        if c["j"] == 0:
            assert s0 <= s_L <= s1                                     # the landmark slab contains s_L
        elif c["j"] == -1:
            assert s1 <= s_L + 1e-9                                    # anterior neighbour
        else:
            assert s0 >= s_L - 1e-9 and s0 > s_L - h                   # posterior neighbour
    assert offset_cover(s_L, h, 0) == (18.0, 26.0)
    assert offset_cover(s_L, h, -1) == (14.0, 22.0)
    assert offset_cover(s_L, h, 1) == (22.0, 30.0)


def test_search_interval_union_and_extension():
    iv = search_interval(22.0, 4.0, [-1, 0, 1], extension_mm=1.0)
    assert iv["s_mm"] == [13.0, 31.0]
    assert iv["y_mm"] == [-13.0, -31.0]
    assert search_interval(22.0, 4.0, [0])["y_mm"] == [-18.0, -26.0]  # narrowed: landmark slab only
    assert clamp_ys([-12.5, -13.0, -20.0, -31.0, -31.5], iv["y_mm"]) == [-13.0, -20.0, -31.0]


def test_slab_consistency_identifies_containing_slabs():
    rec = build_slab_prior({"name": "LGN", "y_mm": -22.0, "s_mm": 22.0}, {"slab_offsets": [-1, 0, 1]})
    ys = np.arange(-31.0, -12.9, 0.5)
    profile = {float(y): 0.2 + 0.01 * abs(y + 28.0) for y in ys}       # minimum at y = -28 (s = 28)
    res = slab_consistency(profile, rec["candidates"], delta_noise=0.0)
    # s = 28 lies only in posterior-neighbour slabs: j = +1 for delta <= 2 (s1 = 30 - delta >= 28)
    assert set(res["consistent"]) == {"1"}
    assert res["consistent"]["1"] == [0.0, 2.0]


def test_stacking_search_range_with_and_without_prior():
    from map_qnp_sections import StackContext

    ctx = object.__new__(StackContext)
    ctx.cfg = {"slabs": {"y_pad_mm": 5.0}}
    ctx.scan = {"fine_regions": {"MTG": {}}}
    ctx._atlas = {"region_extent_y": [-72.5, 19.5]}
    ctx.slab_cfg, ctx._slab_prior = None, None
    lo, hi, note = ctx.search_y_range()
    assert (lo, hi) == (-77.5, 24.5) and "MTG y-extent" in note   # unchanged unconstrained range
    ctx.slab_cfg = {"slab_offsets": [-1, 0, 1], "extension_mm": 1.0}
    ctx._slab_prior = build_slab_prior({"name": "LGN", "y_mm": -22.0, "s_mm": 22.0, "method": "test: x"},
                                       ctx.slab_cfg)
    lo, hi, note = ctx.search_y_range()
    assert (lo, hi) == (-31.0, -13.0) and "LGN" in note


def test_slab_figures_render(tmp_path):
    pytest.importorskip("matplotlib")
    from visualization.figures_mapping import coronal_montage, sagittal_view

    labels, affine = _volume()
    labels[labels == 0] = 1                                 # 'brain' everywhere
    labels[0:20, 20:40, 0:5] = 30                           # a left 'MTG' block
    n = int(labels.max()) + 1
    mtg = np.zeros(n, bool)
    mtg[30] = True
    atlas = {"labels": labels.astype(np.int32), "affine": affine, "fine": {"MTG": mtg},
             "centroid": np.array([-8.0, -25.0, -8.0])}
    scan = {"fine_regions": {"MTG": {}}, "hemisphere": "left"}
    cfg = {"slab_offsets": [-1, 0, 1], "extension_mm": 1.0, "figure_label_codes": {"LGN": [LGN]}}
    lm = locate_landmark(labels, affine, {"name": "LGN", "codes": [LGN], "min_area_mm2": 1.0}, "left")
    record = build_slab_prior(lm, cfg)
    template = np.full(labels.shape, 100, np.int16)
    a = coronal_montage(atlas, template, scan, cfg, record, tmp_path)
    b = sagittal_view(atlas, template, scan, cfg, record, tmp_path, windows={"NEUN": [-27.0, -28.0]}, windows_source="t")
    assert a.stat().st_size > 10_000 and b.stat().st_size > 10_000
