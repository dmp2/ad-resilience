"""Stacking building blocks (section_mapping.stacking, measures compression) on tiny synthetic inputs."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(SRC), str(SRC / "preprocess"), str(SRC / "config_setup")]

from preprocess.qnp_particles import remove_border_bands  # noqa: E402
from section_mapping.alignment import level_sigmas  # noqa: E402
from section_mapping.measures import compress, grid_initialization  # noqa: E402
from section_mapping.stacking import (  # noqa: E402
    RoleThresholds,
    assign_role,
    candidate_pose,
    distinct_minima,
    escalation_ys,
    noise_delta,
    otsu_separability,
    placement_concerns,
    placement_verdict,
    range_extension,
    search_boundary_flags,
    stripe_index,
    summarize_level,
    window,
    windows_at_bound,
)


def _section(spacing: float = 0.5):
    """Gray/white masses on a lattice: a 12 x 8 mm block with a white core."""

    xs, ys = np.meshgrid(np.arange(0, 12, spacing), np.arange(0, 8, spacing))
    X = np.column_stack([xs.ravel(), ys.ravel()])
    white = ((X[:, 0] > 4) & (X[:, 0] < 9) & (X[:, 1] > 2) & (X[:, 1] < 6)).astype(float)
    area = spacing ** 2
    return X, np.column_stack([(1 - white) * area, white * area])


def test_grid_initialization_conserves_mass_and_centroid():
    X, nu = _section()
    Z, nuZ = grid_initialization(X, nu, 1.5)
    assert len(Z) < len(X)
    np.testing.assert_allclose(nuZ.sum(0), nu.sum(0))
    np.testing.assert_allclose((Z * nuZ.sum(1, keepdims=True)).sum(0), (X * nu.sum(1, keepdims=True)).sum(0))


def test_compress_reduces_error_and_keeps_mass():
    pytest.importorskip("torch")
    X, nu = _section()
    Z, nuZ, info = compress(X, nu, 1.5, outer_steps=5)
    assert info["count"] == len(Z) < len(X)
    assert info["final_squared_error"] < info["initial_squared_error"]
    assert abs(info["total_mass_deviation_percent"]) < 2.0
    assert np.all(nuZ >= 0) and info["sigma_mm"] == 1.5


def test_otsu_separability_bimodal_vs_unimodal():
    rng = np.random.default_rng(0)
    bimodal = np.concatenate([rng.normal(0.2, 0.02, 500), rng.normal(0.8, 0.02, 500)])
    assert otsu_separability(bimodal) > 0.95
    assert otsu_separability(rng.normal(0.5, 0.1, 1000)) < 0.75
    assert otsu_separability(np.ones(10)) == 0.0


def test_stripe_index_detects_periodic_scanner_stripes_only():
    px = 0.032
    yy, xx = np.mgrid[0:600, 0:700] * px
    mask = (xx - 11) ** 2 / 100 + (yy - 9.5) ** 2 / 64 < 1
    p_wm = ((xx > 9) & (xx < 14) & (yy > 6)).astype(float)  # a straight gray/white boundary
    clean = stripe_index(p_wm, mask, px)
    striped = stripe_index(np.clip(p_wm + 0.3 * np.sin(2 * np.pi * xx / 1.0), 0, 1), mask, px)
    assert clean["max"] < 0.1
    assert striped["columns"] > 0.3 and abs(striped["columns_period_mm"] - 1.0) < 0.05


def test_assign_role_names_each_failed_check():
    t = RoleThresholds(0.75, 0.3, 0.5, ("GFAP",))
    good = {"otsu_separability": 0.85, "stripe_index": 0.02, "layer1_mean_p_wm": 0.1}
    assert assign_role("NEUN", good, t) == ("anchor", [])
    role, reasons = assign_role("GFAP", {**good, "stripe_index": 0.7, "layer1_mean_p_wm": 0.8}, t)
    assert role == "passenger" and len(reasons) == 3
    assert assign_role("LFB", {**good, "layer1_mean_p_wm": None}, t)[0] == "anchor"


def test_noise_delta_window_and_escalation():
    ys = [float(y) for y in np.arange(-20, -9.9, 0.5)]
    base = {y: 0.1 + 0.01 * (y + 15) ** 2 for y in ys}
    profiles = {"A": base, "B": {y: v + 0.002 * (y == -15.5) for y, v in base.items()},
                "C": {y: v + 0.001 for y, v in base.items()}}
    noise = noise_delta(profiles, top=5)
    assert noise["delta"] is not None and noise["delta"] >= 0
    assert noise_delta({"A": base}, 5)["delta"] is None
    w = window(base, 0.01, 0.5)
    assert w["best_y"] == -15.0 and w["window_y"] == [-14.0, -16.0] and w["contiguous"]
    assert not w["flat"]
    nxt = escalation_ys([w], 0.5)
    assert max(nxt) == -13.5 and min(nxt) == -16.5


def test_distinct_minima_and_level_sigmas():
    assert distinct_minima([0, 3, 6, 9], [0.1, 0.2, 0.05, 0.3], 2, 6) == [6.0, 0.0]
    assert level_sigmas([2, 1, 0.5], 1.5) == [2.0, 1.0]
    assert level_sigmas([2, 1, 0.5], None) == [2.0, 1.0, 0.5]


def test_remove_border_bands_only_clears_full_lines():
    mask = np.zeros((60, 40), bool)
    mask[10:55, 5:30] = True
    banded = mask.copy()
    banded[-5:, :] = True
    out, removed = remove_border_bands(banded, 0.1, 1.0)
    assert removed == {"top": 0, "bottom": 5, "left": 0, "right": 0}
    np.testing.assert_array_equal(out, mask)
    untouched, none = remove_border_bands(mask, 0.1, 1.0)
    assert not any(none.values()) and np.array_equal(untouched, mask)


def test_search_depths_recovers_the_matching_depth():
    """The driver's per-depth fresh search ranks the depth whose atlas shape matches the section."""

    torch = pytest.importorskip("torch")
    from section_mapping.alignment import Similarity2D
    from section_mapping.stacking import make_scorer, search_depths

    def shape(depth):  # an L whose arm length changes with depth (1 mm grid, gray/white)
        X, nu = [], []
        for x in range(-8, 9):
            for y in range(-8, 9):
                arm = (abs(y) <= 2 and -8 <= x <= 8) or (abs(x) <= 2 and 0 <= y <= 2 + 2 * depth)
                if arm:
                    X.append((x, y))
                    nu.append((0.0, 1.0) if abs(x) <= 1 and abs(y) <= 1 else (1.0, 0.0))
        return np.asarray(X, float), np.asarray(nu, float)

    truth = Similarity2D(30.0, 1.0, False, (0.5, -0.3))
    P, mu = shape(2)
    section = truth.moving_to_section(P)
    scorer = make_scorer(section, mu, [2.0, 1.0])
    measures = [shape(d) for d in range(5)]
    results = search_depths(scorer, measures, {"theta_step_deg": 15, "scale_grid": [1.0], "scale_bounds": [0.9, 1.1],
                                               "fft_grid_mm": 1.0, "fft_blur_mm": 1.0, "refine_top": 2, "steps": 60})
    best = int(np.argmin([r[0][0] for r in results]))
    assert best == 2


# ---- placement comparison diagnostics -------------------------------------------------

def test_boundary_and_region_distances_on_a_plane_grid():
    from section_mapping.stacking import boundary_distance_map, grid_lookup, region_distance_map

    res = 0.5
    axis = np.arange(-10, 10 + 1e-9, res)
    aa, bb = np.meshgrid(axis, axis, indexing="xy")
    disk = aa ** 2 + bb ** 2 <= 5.0 ** 2                      # 'brain' of radius 5 mm
    pts = np.array([[0.0, 0.0], [4.0, 0.0], [8.0, 0.0], [30.0, 0.0]])
    to_edge = grid_lookup(pts, boundary_distance_map(disk, res), axis, res)
    assert to_edge[0] == pytest.approx(5.0, abs=0.6) and to_edge[1] == pytest.approx(1.0, abs=0.6)
    assert to_edge[2] == pytest.approx(3.0, abs=0.6) and np.isnan(to_edge[3])   # outside the grid
    inside = grid_lookup(pts[:3], region_distance_map(disk, res), axis, res)
    assert inside[0] == 0 and inside[1] == 0 and inside[2] == pytest.approx(3.0, abs=0.6)


def test_gray_white_boundary_points_lie_on_the_split_not_the_tissue_edge():
    from section_mapping.stacking import gray_white_boundary_points

    tissue = np.zeros((60, 80), bool)
    tissue[10:50, 10:70] = True
    p = np.where(np.arange(80)[None, :] >= 40, 0.9, 0.1) * np.ones((60, 1))   # white right of column 40
    pts = gray_white_boundary_points(p, tissue, origin_mm=(1.0, 2.0), px_mm=(0.1, 0.1), smooth_px=0.5)
    assert len(pts) > 0
    np.testing.assert_allclose(pts[:, 0], 1.0 + 40 * 0.1, atol=0.15)          # x = origin + col * px
    assert pts[:, 1].min() >= 2.0 + 10 * 0.1 - 1e-9 and pts[:, 1].max() <= 2.0 + 49 * 0.1 + 1e-9


def test_placement_status_separates_prior_from_data():
    from section_mapping.stacking import placement_status

    assert placement_status(True, 5.0).startswith("prior inactive")
    assert placement_status(False, -0.2).startswith("image-supported")
    assert placement_status(False, 0.7).startswith("tie within delta")
    assert placement_status(False, 3.0).startswith("prior-driven")


def test_depth_inside_hull_separates_exposed_and_sulcal_pia():
    from section_mapping.stacking import depth_inside_hull

    xs, ys = np.meshgrid(np.arange(0, 20.1, 0.5), np.arange(0, 10.1, 0.5))
    tissue = np.column_stack([xs.ravel(), ys.ravel()])
    pia = np.array([[10.0, 10.0], [10.0, 5.0], [0.0, 5.0]])      # top edge, centre, left edge
    d = depth_inside_hull(pia, tissue)
    np.testing.assert_allclose(d, [0.0, 5.0, 0.0], atol=1e-9)


def _rows(profiles):
    return [{"representation": a, "family": "both", "y0_mm": y, "score": s, "theta_deg": 10.0, "scale": 1.0,
             "flip": True, "tau_x": 0.0, "tau_y": 0.0} for a, prof in profiles.items() for y, s in prof.items()]


def test_summarize_level_flatness_inheritance_and_shrinkage():
    ys = np.arange(-20.0, -35.01, -0.5)
    sharp = lambda c, n: {float(y): 0.1 + 0.02 * abs(y - c) + n for y in ys}
    profiles = {"A": sharp(-27.5, 0.0), "B": sharp(-28.0, 0.001), "C": {float(y): 0.2 for y in ys}}
    first = summarize_level("h1.5", _rows(profiles), None, step_mm=0.5, delta_top=20)
    assert first["flat_anchors"] == ["C"] and first["anchors"]["A"]["best_y"] == -27.5
    assert first["union_window_y"][0] >= first["union_window_y"][1]
    # A prior-limited first level inherits flatness instead of judging it.
    inherited = summarize_level("h1.5", _rows(profiles), None, step_mm=0.5, delta_top=20,
                                inherit_flat=({"B"}, "inherited (test)"))
    assert inherited["flat_anchors"] == ["B"] and inherited["anchors"]["A"]["flat_rule"] == "inherited (test)"
    second = summarize_level("h1.0", _rows(profiles), first, step_mm=0.5, delta_top=20)
    assert second["flat_anchors"] == ["C"] and second["stop_escalating"] is True


def test_range_extension_and_prior_bound():
    summary = {"anchors": {"A": {"flat": False, "window_y": [-13.0, -14.0]}, "B": {"flat": True, "window_y": [-40.0, -41.0]}}}
    assert range_extension(summary, [-13.0, -14.0, -31.0], 0.5) == [-12.5, -12.0]
    assert windows_at_bound(summary, [-13.0, -31.0], 0.5) == ["A"]


def test_search_boundary_flags_and_verdict():
    summary = {"anchors": {"A": {"window_y": [-24.5, -25.0]}}, "ys_mm": [-13.0, -31.0], "windows_at_prior_bound": []}
    assert search_boundary_flags(summary, "A", 1.0, (0.8, 1.5), 0.5) == []
    assert any("scale" in f for f in search_boundary_flags(summary, "A", 1.499, (0.8, 1.5), 0.5))
    ok = placement_verdict([[-24.0, -25.0], [-24.5, -25.5]], ["prior inactive", "image-supported"], {})
    assert ok["summary"] == "supported by image geometry" and ok["admissible_window_intersection_y_mm"] == [-24.5, -25.0]
    driven = placement_verdict([[-24.0, -25.0], [-24.5, -25.5]], ["prior-driven", "tie within delta"], {})
    assert driven["summary"] == "primarily prior-driven" and "prior-driven for 1/2 anchors" in driven["reasons"]
    apart = placement_verdict([[-13.0, -14.0], [-24.5, -25.5]], ["tie within delta", "tie within delta"], {})
    assert apart["summary"] == "unresolved" and apart["admissible_window_intersection_y_mm"] is None


def _report(flags=(), contiguous=True, windows=((-27.0, -28.0), (-27.5, -28.5))):
    sections = {f"A{k}": {"role": "anchor", "window_y_mm": list(w), "window_contiguous": True, "flags": []}
                for k, w in enumerate(windows)}
    sections["A0"].update(flags=list(flags), window_contiguous=contiguous)
    sections["P"] = {"role": "passenger", "window_y_mm": list(windows[0])}
    return {"sections": sections}


def test_placement_concerns_gate_the_xiv_handoff():
    supported = {"region_prior__landmark_slab": {"summary": "supported by image geometry", "reasons": []}}
    assert placement_concerns(_report(), "A0", supported) == []
    assert placement_concerns(_report(), "A0", {}) == []                    # no prior run: nothing to contest
    driven = {"region_prior__landmark_slab": {"summary": "primarily prior-driven", "reasons": ["prior-driven for 3/3 anchors"]}}
    assert any("prior-driven" in c for c in placement_concerns(_report(), "A0", driven))
    assert any("scale" in c for c in placement_concerns(_report(flags=["scale 1.499 at its search bound"]), "A0", {}))
    assert any("contiguous" in c for c in placement_concerns(_report(contiguous=False), "A0", {}))
    apart = _report(windows=((-13.0, -14.0), (-24.5, -25.0)))
    assert any("do not intersect" in c for c in placement_concerns(apart, "A0", {}))


def test_candidate_pose_reads_the_named_placement():
    def ev(y, theta):
        return {"y0_mm": y, "data_only_score": 0.2, "pose": {"theta_deg": theta, "scale": 1.0, "flip": True,
                                                             "tau_mm": [1.0, 2.0]}}
    cand = {"anchors": {"A": {"unconstrained": {"search_pose": ev(-15.0, 10.0)},
                              "admissible": {"search_pose": ev(-24.5, 20.0), "refined_on_full_atlas": ev(-24.5, 21.0)}}}}
    y, t, score = candidate_pose(cand, "A", "admissible-refined")
    assert (y, t.theta_deg, t.flip, t.tau, score) == (-24.5, 21.0, True, (1.0, 2.0), 0.2)
    assert candidate_pose(cand, "A", "unconstrained")[0] == -15.0
    with pytest.raises(ValueError):
        candidate_pose(cand, "A", "best")
    with pytest.raises(KeyError):
        candidate_pose(cand, "B", "admissible")
