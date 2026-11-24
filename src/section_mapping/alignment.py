"""Geometric alignment (CPU reference): 2-D similarity, support-censored varifold score, search, poses.

Roles
=====
The *target* (fixed) is the data section measure ``mu_T = (X, nu)``; it defines the support.
The *template* (moving) is an atlas plane (or another section) ``mu_S = (P, mu)``.  Never swap
them: the support censoring exists to protect incomplete data.

Transform
=========
    p = A (x - tau),    A = s R(theta) F,    F = diag(1, -1) if flip else I,

maps section coordinates x (mm) to moving-frame coordinates p (mm).  ``s`` is moving mm per
section mm: for atlas planes it absorbs shrinkage, template size and atrophy together, so it is
a nuisance parameter, not a measurement.  Moving particles are pulled into the section frame
(``moving_to_section``) with masses divided by s^2, so area is conserved.

Score (0 = identical, 1 = no overlapping tissue)
=================================================
    w(y)  = min(1, D_T(y) / (c * median_i D_T(x_i))),   D_T(y) = sum_i |nu_i| exp(-|y - x_i|^2 / 2 sigma_f^2)
    score = mean_sigma |w mu_S - mu_T|^2_sigma / |mu_T|^2_sigma

with sigma_f = 0.5 mm, c = 0.25, moving particles with w <= 1e-3 dropped, and SDs 2, 1, 0.5 mm
(drop SDs < h/2 for particles at spacing h: ``level_sigmas``).  Moving tissue far from the
section does not count (absent tissue is censored), moving tissue missing where the section has
tissue does.  It censors *absent* tissue, not *present-but-wrong* tissue.

Search
======
FFT cross-correlation of Gaussian-blurred rasters over translation for a grid of (theta, s, flip)
gives approximate ranking values; the best distinct candidates are refined with the exact score
(Nelder-Mead here, batched Adam in ``alignment_gpu``).

Poses after a deformable fit
============================
``similarity_lsq`` reads the similarity off shot atlas particles; ``compose`` folds it into the
start pose.  ``CommonScore`` ranks fitted poses of one section on one fixed target in the section
frame (the native xIV loss is normalised per run and cannot rank across runs).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import minimize
from scipy.signal import correlate

from section_mapping.measures import gauss_inners

FOOTPRINT_CUTOFF = 1e-3   # moving particles with support weight <= this are dropped
TRUNCATE_SIGMAS = 5.0     # truncation of the footprint density kernel


# --------------------------------------------------------------------------------------
# Similarity transform
# --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Similarity2D:
    """``p = A (x - tau)`` with ``A = s R(theta) F``; x = section mm, p = moving mm."""

    theta_deg: float
    scale: float
    flip: bool
    tau: tuple[float, float]

    @property
    def matrix(self) -> np.ndarray:
        angle = np.deg2rad(self.theta_deg)
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        reflection = np.diag([1.0, -1.0]) if self.flip else np.eye(2)
        return self.scale * rotation @ reflection

    def section_to_moving(self, X: np.ndarray) -> np.ndarray:
        return (np.asarray(X, float)[:, :2] - np.asarray(self.tau)) @ self.matrix.T

    def moving_to_section(self, P: np.ndarray) -> np.ndarray:
        return np.asarray(P, float)[:, :2] @ np.linalg.inv(self.matrix).T + np.asarray(self.tau)

    def as_dict(self) -> dict[str, Any]:
        return {**asdict(self), "tau": list(self.tau), "matrix_A": self.matrix.tolist(),
                "convention": "moving = A @ (section - tau); section = A^-1 @ moving + tau; "
                              "moving masses are divided by scale^2 in the section frame"}

    def as_row(self) -> dict[str, Any]:
        """Flat table columns (theta wrapped to [0, 360))."""

        return {"theta_deg": float(self.theta_deg % 360), "scale": float(self.scale), "flip": bool(self.flip),
                "tau_x": float(self.tau[0]), "tau_y": float(self.tau[1])}

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "Similarity2D":
        return cls(float(values["theta_deg"]), float(values["scale"]), bool(values["flip"]),
                   tuple(float(v) for v in values["tau"]))

    @classmethod
    def from_row(cls, row: Mapping[str, Any], prefix: str = "") -> "Similarity2D":
        """From flat columns (CSV strings or JSON values); ``prefix`` e.g. 'fit_'."""

        return cls(float(row[prefix + "theta_deg"]), float(row[prefix + "scale"]),
                   str(row[prefix + "flip"]) in ("True", "true", "1"),
                   (float(row[prefix + "tau_x"]), float(row[prefix + "tau_y"])))


# --------------------------------------------------------------------------------------
# Support-censored score
# --------------------------------------------------------------------------------------

def _gauss_sum(X: np.ndarray, Y: np.ndarray, weights: np.ndarray, sigma: float, tree=None) -> np.ndarray:
    """sum_j weights_j exp(-|X_i - Y_j|^2 / 2 sigma^2), truncated at 5 sigma (KD-tree)."""

    from scipy.spatial import cKDTree

    tree_y = tree if tree is not None else cKDTree(Y)
    # "ndarray" output keeps zero distances (coincident points), unlike the sparse formats.
    pairs = cKDTree(X).sparse_distance_matrix(tree_y, TRUNCATE_SIGMAS * sigma, output_type="ndarray")
    values = np.exp(-pairs["v"] ** 2 / (2 * sigma * sigma)) * weights[pairs["j"]]
    return np.bincount(pairs["i"], values, minlength=len(X))


def level_sigmas(sigmas: Sequence[float], h_mm: float | None) -> list[float]:
    """Score SDs for particles at spacing h: drop SDs < h/2."""

    kept = [float(s) for s in sigmas if h_mm is None or float(s) >= float(h_mm) / 2]
    return kept or [max(float(s) for s in sigmas)]


class FootprintScorer:
    """Score moving measures against one fixed section measure (see module docstring)."""

    def __init__(self, X: np.ndarray, nu: np.ndarray, sigmas_mm: Sequence[float] = (2.0, 1.0, 0.5),
                 footprint_sigma_mm: float = 0.5, footprint_saturation: float = 0.25):
        from scipy.spatial import cKDTree

        self.X = np.asarray(X, float)[:, :2]
        self.nu = np.asarray(nu, float)
        self.sigmas = [float(s) for s in sigmas_mm]
        self.footprint_sigma = float(footprint_sigma_mm)
        area = self.nu.sum(1)
        self._area = area
        self._tree = cKDTree(self.X)
        # Saturating at a fraction of the median density gives the section's own edge and
        # corner particles full weight; outside, the weight decays over ~footprint_sigma.
        self.density_ref = footprint_saturation * float(np.median(self._density(self.X)))
        self.self_terms = self._inners(self.X, self.nu, self.X, self.nu)

    def _density(self, Y: np.ndarray) -> np.ndarray:
        return _gauss_sum(np.asarray(Y, float), self.X, self._area, self.footprint_sigma, self._tree)

    def _inners(self, X, nu, Y, mu) -> list[float]:
        """<mu_X, mu_Y> for every sigma, reusing one squared-distance matrix (float32)."""

        d2 = ((X[:, None, :].astype(np.float32) - Y[None, :, :].astype(np.float32)) ** 2).sum(-1)
        feature = (nu @ mu.T).astype(np.float32)
        return [float(np.sum(np.exp(d2 * np.float32(-0.5 / (s * s))) * feature)) for s in self.sigmas]

    def footprint(self, Y: np.ndarray) -> np.ndarray:
        """Section support weight in [0, 1] at section-frame points Y."""

        return np.minimum(self._density(Y) / self.density_ref, 1.0)

    def restricted(self, Y: np.ndarray, mu: np.ndarray, cutoff: float = FOOTPRINT_CUTOFF):
        weight = self.footprint(Y)
        keep = weight > cutoff
        return Y[keep], mu[keep] * weight[keep, None], weight

    def score(self, Y: np.ndarray, mu: np.ndarray) -> float:
        """Mean over sigma of |mu_moving - mu_section|^2 / |mu_section|^2 (section frame)."""

        Y, mu, _ = self.restricted(np.asarray(Y, float)[:, :2], np.asarray(mu, float))
        if len(Y) == 0:
            return 1.0
        aa = self._inners(Y, mu, Y, mu)
        ab = self._inners(Y, mu, self.X, self.nu)
        return float(np.mean([max(a + bb - 2.0 * b, 0.0) / bb for a, b, bb in zip(aa, ab, self.self_terms)]))

    def score_transform(self, transform: Similarity2D, P: np.ndarray, mu_P: np.ndarray) -> float:
        return self.score(transform.moving_to_section(P), np.asarray(mu_P, float) / transform.scale ** 2)


# --------------------------------------------------------------------------------------
# Search
# --------------------------------------------------------------------------------------

def _raster(points: np.ndarray, masses: np.ndarray, origin: np.ndarray, shape: tuple[int, int], h: float):
    """Sum masses into a [C, rows(y), cols(x)] grid; points outside are dropped."""

    index = np.floor((points - origin) / h).astype(int)
    keep = (index[:, 0] >= 0) & (index[:, 1] >= 0) & (index[:, 0] < shape[1]) & (index[:, 1] < shape[0])
    grid = np.zeros((masses.shape[1], *shape))
    for c in range(masses.shape[1]):
        np.add.at(grid[c], (index[keep, 1], index[keep, 0]), masses[keep, c])
    return grid


def fft_candidates(scorer: FootprintScorer, P: np.ndarray, mu_P: np.ndarray, thetas: Iterable[float],
                   scales: Iterable[float], flips: Iterable[bool], h: float = 0.5, blur_mm: float = 1.0,
                   per_combo: int = 2) -> list[tuple[float, Similarity2D]]:
    """Approximate scores over translation for every (theta, s, flip); best few per combo.

    Uses blurred rasters and the footprint applied after blurring, so values are only
    for ranking candidates; refine() recomputes the exact score.
    """

    X, nu = scorer.X, scorer.nu
    margin = 3.0 * blur_mm
    q_origin = X.min(0) - margin
    q_shape = tuple(int(v) for v in np.ceil((X.max(0) + margin - q_origin) / h)[::-1] + 1)
    sigma_px = blur_mm / h
    Q = np.stack([ndi.gaussian_filter(c, sigma_px) for c in _raster(X, nu, q_origin, q_shape, h)])
    centers = q_origin + (np.indices(q_shape)[::-1].reshape(2, -1).T + 0.5) * h
    F = scorer.footprint(centers).reshape(q_shape)
    FQ = F * Q
    norm = float((Q ** 2).sum())
    out = []
    for flip in flips:
        for scale in scales:
            for theta in thetas:
                linear = Similarity2D(theta, scale, flip, (0.0, 0.0))
                Y = linear.moving_to_section(P)  # = A^-1 p; the translation tau is searched
                mu = np.asarray(mu_P, float) / scale ** 2
                pad = np.array(q_shape[::-1]) * h
                p_origin = Y.min(0) - pad - margin
                p_shape = tuple(int(v) for v in np.ceil((Y.max(0) + pad + margin - p_origin) / h)[::-1] + 1)
                R = np.stack([ndi.gaussian_filter(c, sigma_px) for c in _raster(Y, mu, p_origin, p_shape, h)])
                cross = sum(correlate(R[c], FQ[c], mode="valid", method="fft") for c in range(R.shape[0]))
                energy = correlate((R ** 2).sum(0), F ** 2, mode="valid", method="fft")
                approx = (norm + energy - 2.0 * cross) / norm
                flat = np.argsort(approx, axis=None)[:per_combo]
                for d_row, d_col in zip(*np.unravel_index(flat, approx.shape)):
                    # Section pixel i sits on moving pixel i + d: x = y + tau.
                    tau = q_origin - (p_origin + np.array([d_col, d_row]) * h)
                    out.append((float(approx[d_row, d_col]), Similarity2D(theta, scale, flip, tuple(tau))))
    out.sort(key=lambda item: item[0])
    return out


def distinct_poses(candidates, k: int, min_shift_mm: float = 3.0, min_angle_deg: float = 20.0):
    """The first k (score, Similarity2D) that differ in flip, translation or rotation."""

    chosen: list[tuple[float, Similarity2D]] = []
    for value, t in candidates:
        if all(t.flip != c.flip or np.hypot(*np.subtract(t.tau, c.tau)) > min_shift_mm
               or abs((t.theta_deg - c.theta_deg + 180) % 360 - 180) > min_angle_deg for _, c in chosen):
            chosen.append((value, t))
        if len(chosen) == int(k):
            break
    return chosen


def refine(scorer: FootprintScorer, P: np.ndarray, mu_P: np.ndarray, start: Similarity2D,
           scale_bounds: tuple[float, float] = (1.0, 1.0), max_evals: int = 300) -> tuple[float, Similarity2D]:
    """Nelder-Mead on (theta, log s, tau) at fixed flip; s is held when bounds are equal."""

    fixed_scale = scale_bounds[0] == scale_bounds[1]

    def unpack(z):
        scale = scale_bounds[0] if fixed_scale else float(np.exp(z[3]))
        return Similarity2D(float(z[0]), scale, start.flip, (float(z[1]), float(z[2])))

    def objective(z):
        t = unpack(z)
        if not scale_bounds[0] <= t.scale <= scale_bounds[1]:
            return 1.0 + abs(np.log(t.scale / np.clip(t.scale, *scale_bounds)))
        return scorer.score_transform(t, P, mu_P)

    z0 = [start.theta_deg, *start.tau] + ([] if fixed_scale else [np.log(start.scale)])
    simplex_step = np.array([5.0, 1.0, 1.0] + ([] if fixed_scale else [0.05]))
    simplex = np.vstack([z0, z0 + np.diag(simplex_step)])
    result = minimize(objective, z0, method="Nelder-Mead",
                      options={"initial_simplex": simplex, "maxfev": max_evals, "xatol": 0.05, "fatol": 1e-5})
    return float(result.fun), unpack(result.x)


def search(scorer: FootprintScorer, P: np.ndarray, mu_P: np.ndarray, *, thetas, scales, flips,
           scale_bounds, h=0.5, blur_mm=1.0, refine_top=3, max_evals=300) -> list[tuple[float, Similarity2D]]:
    """FFT candidates, then exact refinement of the best distinct ones; best first."""

    candidates = fft_candidates(scorer, P, mu_P, thetas, scales, flips, h=h, blur_mm=blur_mm)
    refined = [refine(scorer, P, mu_P, t, scale_bounds, max_evals) for _, t in
               distinct_poses(candidates, refine_top, min_shift_mm=3.0, min_angle_deg=20.0)]
    return sorted(refined, key=lambda item: item[0])


# --------------------------------------------------------------------------------------
# Pose algebra (after a deformable fit)
# --------------------------------------------------------------------------------------

def similarity_lsq(P0: np.ndarray, P1: np.ndarray, w: np.ndarray | None = None) -> dict[str, Any]:
    """Weighted least-squares P1 ~ c R P0 + b (rotation only: the flow cannot reflect)."""

    w = np.ones(len(P0)) if w is None else np.asarray(w, float).ravel()
    w = w / w.sum()
    m0, m1 = w @ P0, w @ P1
    A0, A1 = P0 - m0, P1 - m1
    H = (A0 * w[:, None]).T @ A1
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, d])
    R = Vt.T @ D @ U.T
    var0 = float((w * (A0 ** 2).sum(1)).sum())
    c = float((S * np.diag(D)).sum() / var0)
    b = m1 - c * R @ m0
    residual = np.sqrt(((P1 - (c * P0 @ R.T + b)) ** 2).sum(1))
    return {"scale": c, "rotation_deg": float(np.degrees(np.arctan2(R[1, 0], R[0, 0]))), "R": R, "b": b,
            "residual_mm": {"p50": float(np.percentile(residual, 50)), "max": float(residual.max())}}


def compose(t_init: Similarity2D, fit: Mapping[str, Any]) -> Similarity2D:
    """The fitted pose as one ``Similarity2D`` (section -> atlas plane).

    The start places the section at p = A0 (q - tau0); the fit moves the atlas x -> c R x + b onto
    it.  So q = A0^-1 (c R x + b) + tau0, i.e. x = A (q - tau) with A = (c R)^-1 A0 and
    tau = tau0 + A0^-1 b.
    """

    A0 = t_init.matrix
    tau = np.asarray(t_init.tau, float) + np.linalg.solve(A0, fit["b"])
    return Similarity2D((t_init.theta_deg - fit["rotation_deg"]) % 360.0, t_init.scale / fit["scale"], t_init.flip,
                        (float(tau[0]), float(tau[1])))


def pose_distance(a: Similarity2D, b: Similarity2D, X: np.ndarray) -> float:
    """Median displacement (mm, section frame) between two poses applied to section points X."""

    return float(np.median(np.linalg.norm(a.section_to_moving(X) - b.section_to_moving(X), axis=1)))


def section_frame_atlas(pose: Similarity2D, P_plane: np.ndarray, masses: Sequence[np.ndarray]
                        ) -> tuple[np.ndarray, list[np.ndarray]]:
    """Atlas plane particles carried to the section frame by ``pose``, areas pushed forward (/ s^2)."""

    return pose.moving_to_section(P_plane), [np.asarray(m, float) / pose.scale ** 2 for m in masses]


# --------------------------------------------------------------------------------------
# Common post-fit score (fixed section frame)
# --------------------------------------------------------------------------------------

@dataclass
class EvaluationSettings:
    sigmas_mm: tuple[float, ...] = (8.0, 4.0, 2.0, 1.0)
    footprint_sigma_mm: float = 0.5
    lambda_struct: float = 1.0
    lambda_anat: float = 1.0

    @classmethod
    def from_config(cls, block: Mapping[str, Any]) -> "EvaluationSettings":
        if str(block.get("sigma_weights", "inverse")) != "inverse":
            raise ValueError("evaluation.sigma_weights: only 'inverse' (1/sigma) is implemented")
        return cls(tuple(float(s) for s in block.get("sigmas_mm", (8.0, 4.0, 2.0, 1.0))),
                   float(block.get("footprint_sigma_mm", 0.5)), float(block.get("lambda_struct", 1.0)),
                   float(block.get("lambda_anat", 1.0)))

    @property
    def weights(self) -> np.ndarray:
        w = 1.0 / np.asarray(self.sigmas_mm)
        return w / w.sum()


class CommonScore:
    """Post-fit score of atlas poses against one section, in the section's fixed frame::

        R_term = sum_s w_s |mu_A - mu_T|^2_s / |mu_T|^2_s,   w_s proportional to 1 / sigma_s
        R_struct: gray/white, atlas weighted by the section's fixed footprint (0.5 mm)
        R_anat:   atlas anatomical channel vs the section's annotation, uncensored
        R_combined = lambda_struct R_struct + lambda_anat R_anat

    ``section`` needs ``X``, ``nu`` (gray/white), ``annotation`` ((N, 1) or None) and ``annotated``.
    """

    def __init__(self, section, settings: EvaluationSettings):
        self.section, self.settings = section, settings
        self.footprint = FootprintScorer(section.X, section.nu, [1.0], footprint_sigma_mm=settings.footprint_sigma_mm)
        s = settings.sigmas_mm
        self.target_self_struct = gauss_inners(section.X, section.nu, section.X, section.nu, s)
        self.target_self_anat = (gauss_inners(section.X, section.annotation, section.X, section.annotation, s)
                                 if section.annotated else None)

    def _relative(self, aa, ab, bb) -> dict[str, Any]:
        w = self.settings.weights
        per_sigma = (aa + bb - 2.0 * ab) / bb
        return {"R": float(w @ per_sigma), "per_sigma": per_sigma.tolist(),
                "atlas_self": float(w @ (aa / bb)), "target_self": 1.0, "cross": float(w @ (2.0 * ab / bb))}

    def evaluate(self, Y: np.ndarray, nu_gmwm: np.ndarray, nu_anat: np.ndarray) -> dict[str, Any]:
        """``Y``: atlas particles in the section frame (mm); masses already pushed forward (mm^2)."""

        s, sec = self.settings.sigmas_mm, self.section
        weight = self.footprint.footprint(Y)[:, None]
        cut = weight[:, 0] > FOOTPRINT_CUTOFF
        Ys, mus = Y[cut], (nu_gmwm * weight)[cut]
        struct = self._relative(gauss_inners(Ys, mus, Ys, mus, s), gauss_inners(Ys, mus, sec.X, sec.nu, s),
                                self.target_self_struct)
        out: dict[str, Any] = {
            "R_struct": struct["R"], "struct": struct,
            "atlas_area_in_footprint_mm2": float((nu_gmwm * weight).sum()),
            "atlas_mtg_mass_mm2": float(nu_anat.sum()),
            "atlas_mtg_mass_in_footprint_mm2": float((nu_anat * weight).sum()),
            "section_area_mm2": float(sec.nu.sum()),
        }
        if sec.annotated:
            anat = self._relative(gauss_inners(Y, nu_anat, Y, nu_anat, s), gauss_inners(Y, nu_anat, sec.X, sec.annotation, s),
                                  self.target_self_anat)
            out.update(R_anat=anat["R"], anat=anat, annotated_area_mm2=float(sec.annotation.sum()),
                       R_combined=self.settings.lambda_struct * struct["R"] + self.settings.lambda_anat * anat["R"])
        else:
            out.update(R_anat=None, anat=None, annotated_area_mm2=None, R_combined=None)
        return out
