"""Geometric alignment: transform, support-censored score, CPU search, GPU twin, pose algebra, common score.

The batched torch path must reproduce the numpy reference (runs on CPU torch too).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

SRC = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(SRC), str(SRC / "preprocess")]

from section_mapping.alignment import (  # noqa: E402
    CommonScore,
    EvaluationSettings,
    FootprintScorer,
    Similarity2D,
    compose,
    refine,
    search,
    section_frame_atlas,
    similarity_lsq,
)
from section_mapping.alignment_gpu import (  # noqa: E402
    TorchFootprintScorer,
    fft_candidates_gpu,
    pad_measures,
    refine_batch,
    search_gpu,
    search_planes_gpu,
)


@dataclass
class _Section:
    X: np.ndarray
    nu: np.ndarray
    annotation: np.ndarray | None

    @property
    def annotated(self) -> bool:
        return self.annotation is not None and bool(np.any(self.annotation > 0))


# --------------------------------------------------------------------------------------
# CPU reference
# --------------------------------------------------------------------------------------

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


def test_slide_to_local_to_plane_composition():
    """A passenger's slide frame -> QNP local frame (rigid edge M) -> atlas plane (host pose t) in one affine."""

    rng = np.random.default_rng(3)
    edge = Similarity2D(184.0, 1.0, False, (2.0, -1.0))                  # slide -> local is the inverse map
    M = np.eye(3)
    M[:2, :2] = np.linalg.inv(edge.matrix)
    M[:2, 2] = edge.tau
    t = Similarity2D(310.0, 0.995, True, (40.0, 30.0))
    X_slide = rng.uniform(0, 20, (50, 2))
    X_local = X_slide @ M[:2, :2].T + M[:2, 2]
    A, tau = t.matrix, np.asarray(t.tau)
    slide_to_plane = np.r_[np.c_[A @ M[:2, :2], A @ (M[:2, 2] - tau)], [[0, 0, 1]]]
    np.testing.assert_allclose(X_slide @ slide_to_plane[:2, :2].T + slide_to_plane[:2, 2], t.section_to_moving(X_local),
                               atol=1e-12)
    np.testing.assert_allclose(t.moving_to_section(t.section_to_moving(X_local)), X_local, atol=1e-12)


def test_similarity_row_round_trip():
    t = Similarity2D(-33.0, 1.2, True, (3.0, -2.0))
    row = t.as_row()
    assert row["theta_deg"] == pytest.approx(327.0)
    csv_row = {k: str(v) for k, v in row.items()}
    back = Similarity2D.from_row(csv_row)
    X = np.random.default_rng(0).normal(size=(20, 2))
    np.testing.assert_allclose(back.section_to_moving(X), t.section_to_moving(X), atol=1e-12)
    assert Similarity2D.from_row({f"fit_{k}": v for k, v in row.items()}, "fit_").flip is True


# --------------------------------------------------------------------------------------
# GPU twin
# --------------------------------------------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _section(seed: int = 0):
    """An asymmetric two-class 'section' on a 0.5 mm lattice (an L of gray around a white core)."""

    rng = np.random.default_rng(seed)
    a, b = np.meshgrid(np.arange(0, 14, 0.5), np.arange(0, 10, 0.5), indexing="xy")
    X = np.c_[a.ravel(), b.ravel()]
    keep = (X[:, 0] < 5) | (X[:, 1] < 4)
    X = X[keep]
    white = ((X[:, 0] > 1.5) & (X[:, 0] < 3.5)) | ((X[:, 1] > 1.0) & (X[:, 1] < 2.5))
    nu = np.c_[~white, white].astype(float) * 0.25 * (1 + 0.05 * rng.random(len(X)))[:, None]
    return X, nu


def test_gpu_score_matches_numpy_score():
    X, nu = _section()
    cpu = FootprintScorer(X, nu, (2.0, 1.0, 0.5))
    gpu = TorchFootprintScorer(cpu, device=DEVICE)
    truth = Similarity2D(30.0, 1.1, True, (2.0, -1.0))
    P = truth.section_to_moving(X)
    rng = np.random.default_rng(1)
    transforms = [truth] + [Similarity2D(float(rng.uniform(0, 360)), float(rng.uniform(0.8, 1.3)), bool(rng.integers(2)),
                                         tuple(rng.normal(0, 3, 2))) for _ in range(20)]
    mu = nu * truth.scale ** 2  # moving masses are in moving-frame mm^2
    expected = np.array([cpu.score_transform(t, P, mu) for t in transforms])
    got = gpu.score_transforms(transforms, P, mu)
    assert expected[0] < 1e-4
    np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-6)


def test_padded_batch_scores_like_individual_measures():
    X, nu = _section()
    cpu = FootprintScorer(X, nu, (2.0, 1.0))
    gpu = TorchFootprintScorer(cpu, device=DEVICE)
    t = Similarity2D(10.0, 1.0, False, (0.5, 0.5))
    measures = [(t.section_to_moving(X), nu), (t.section_to_moving(X[::2]), nu[::2])]
    P, mu = pad_measures(measures)
    to = dict(device=DEVICE, dtype=torch.float32)
    th, tau, ls, fl = gpu._params([t, t])
    with torch.no_grad():
        got = gpu.score_params(torch.as_tensor(P, **to), torch.as_tensor(mu, **to), th, tau, ls, fl).cpu().numpy()
    expected = [cpu.score_transform(t, *m) for m in measures]
    np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-6)


def test_fft_candidates_and_search_recover_a_known_pose():
    X, nu = _section()
    cpu = FootprintScorer(X, nu, (2.0, 1.0, 0.5))
    gpu = TorchFootprintScorer(cpu, device=DEVICE)
    truth = Similarity2D(90.0, 1.2, True, (1.0, 2.0))
    P, mu = truth.section_to_moving(X), nu * truth.scale ** 2
    grid = dict(thetas=np.arange(0, 360, 15.0), scales=[1.0, 1.2], flips=[False, True])
    best_value, best = fft_candidates_gpu(gpu, P, mu, h=0.5, blur_mm=1.0, **grid)[0]
    assert best.flip and abs((best.theta_deg - 90.0 + 180) % 360 - 180) < 1e-6 and best.scale == 1.2
    score, found = search_gpu(gpu, P, mu, scale_bounds=(0.8, 1.5), h=0.5, blur_mm=1.0, refine_top=2, steps=100, **grid)[0]
    assert score < 1e-3
    assert found.flip and abs((found.theta_deg - 90.0 + 180) % 360 - 180) < 0.5
    assert np.hypot(*np.subtract(found.tau, truth.tau)) < 0.1 and abs(found.scale - 1.2) < 0.01


def test_refine_batch_is_no_worse_than_its_start_and_matches_nelder_mead():
    X, nu = _section()
    cpu = FootprintScorer(X, nu, (2.0, 1.0, 0.5))
    gpu = TorchFootprintScorer(cpu, device=DEVICE)
    truth = Similarity2D(20.0, 1.0, False, (0.0, 0.0))
    P = truth.section_to_moving(X)
    start = Similarity2D(26.0, 1.0, False, (0.8, -0.6))
    nm_score, _ = refine(cpu, P, nu, start, (1.0, 1.0), 300)
    (score, t), = refine_batch(gpu, P, nu, [start], (1.0, 1.0), steps=150)
    assert score <= cpu.score_transform(start, P, nu)
    assert score <= nm_score + 1e-4
    assert abs(t.theta_deg - 20.0) < 0.5


def test_search_planes_gpu_refines_extra_starts_on_every_measure():
    X, nu = _section()
    cpu = FootprintScorer(X, nu, (2.0, 1.0))
    gpu = TorchFootprintScorer(cpu, device=DEVICE)
    truth = Similarity2D(0.0, 1.0, False, (0.0, 0.0))
    measures = [(truth.section_to_moving(X), nu), (truth.section_to_moving(X) + 0.3, nu)]
    out = search_planes_gpu(gpu, measures, thetas=[0.0, 180.0], scales=[1.0], flips=[False], scale_bounds=(1.0, 1.0),
                            h=0.5, blur_mm=1.0, refine_top=1, steps=50, extra_starts=[truth])
    assert len(out) == 2 and all(len(r) == 2 for r in out)
    assert out[0][0][0] < 1e-3 and out[1][0][0] < 1e-3


# --------------------------------------------------------------------------------------
# Pose algebra and the common post-fit score
# --------------------------------------------------------------------------------------

def _annotated_section(rng, n=150, annotated=True):
    X = rng.uniform(-8, 8, (n, 2))
    nu = np.column_stack([np.where(X[:, 1] > 0, 0.2, 0.05), np.where(X[:, 1] > 0, 0.05, 0.2)])
    ann = np.where((X[:, 0] > 0) & (X[:, 1] > 0), nu.sum(1), 0.0)[:, None] if annotated else None
    return _Section(X, nu, ann)


def test_similarity_lsq_recovers_a_rotation_scale_translation():
    rng = np.random.default_rng(0)
    P = rng.normal(size=(200, 2)) * 10
    angle = np.deg2rad(23.0)
    R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    Q = 1.13 * P @ R.T + np.array([3.0, -2.0])
    fit = similarity_lsq(P, Q)
    assert fit["scale"] == pytest.approx(1.13, rel=1e-10)
    assert fit["rotation_deg"] == pytest.approx(23.0, abs=1e-8)
    assert np.allclose(fit["b"], [3.0, -2.0])


@pytest.mark.parametrize("flip", [False, True])
def test_compose_maps_section_points_onto_the_moved_atlas(flip):
    """start places the section at p = A0(q - tau0); the fit moves the atlas x -> cRx + b onto it."""

    rng = np.random.default_rng(1)
    t_init = Similarity2D(130.0, 1.09, flip, (15.0, 14.0))
    P = rng.normal(size=(100, 2)) * 10                      # atlas plane particles
    angle = np.deg2rad(-7.0)
    R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    D = 0.92 * P @ R.T + np.array([1.5, 0.5])               # where the fit puts them (target-placed frame)
    q = t_init.moving_to_section(D)                         # the section points they now coincide with
    pose = compose(t_init, similarity_lsq(P, D))
    assert np.allclose(pose.section_to_moving(q), P, atol=1e-9)
    assert pose.scale == pytest.approx(1.09 / 0.92)
    assert pose.flip == flip


def test_common_score_is_zero_for_the_section_itself_and_frame_independent():
    """The same physical configuration reached from starts of different scale scores the same."""

    rng = np.random.default_rng(2)
    sec = _annotated_section(rng)
    score = CommonScore(sec, EvaluationSettings())
    results = []
    for scale in (0.9, 1.0, 1.25):
        pose = Similarity2D(40.0, scale, True, (1.0, -2.0))
        P = pose.section_to_moving(sec.X)                   # the atlas plane = the section, pushed into the plane
        nu_P, mtg_P = sec.nu * scale ** 2, sec.annotation * scale ** 2
        Y, (nu, mtg) = section_frame_atlas(pose, P, [nu_P, mtg_P])
        results.append(score.evaluate(Y, nu, mtg))
    for r in results:
        assert r["R_struct"] == pytest.approx(0.0, abs=1e-9)
        assert r["R_anat"] == pytest.approx(0.0, abs=1e-9)
        assert r["anat"]["atlas_self"] - r["anat"]["cross"] + 1 == pytest.approx(0.0, abs=1e-9)


def test_disjoint_atlas_mtg_scores_one_plus_its_self_term():
    rng = np.random.default_rng(3)
    sec = _annotated_section(rng)
    score = CommonScore(sec, EvaluationSettings())
    Y = sec.X + np.array([500.0, 0.0])                      # far away
    r = score.evaluate(Y, sec.nu, sec.annotation)
    assert r["anat"]["cross"] == pytest.approx(0.0, abs=1e-12)
    assert r["R_anat"] == pytest.approx(1.0 + r["anat"]["atlas_self"])
    assert r["R_struct"] == pytest.approx(1.0)              # nothing inside the footprint: atlas term 0


def test_unannotated_section_has_no_anatomical_score():
    rng = np.random.default_rng(4)
    sec = _annotated_section(rng, annotated=False)
    r = CommonScore(sec, EvaluationSettings()).evaluate(sec.X, sec.nu, sec.nu[:, :1])
    assert r["R_anat"] is None and r["R_combined"] is None and r["R_struct"] == pytest.approx(0.0, abs=1e-9)
