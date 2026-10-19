"""2-D similarity search between section measures, and atlas reslicing at oblique planes.

Model
=====
A section measure ``mu_Q`` (QNP particles, fixed) is compared with a moving measure
``mu_P`` (another stain, or an atlas plane) under a 2-D similarity

    p = A (x - tau),    A = s R(theta) F,    F = diag(1, -1) if flip else I,

which maps section coordinates x (mm) to moving-frame coordinates p (mm).  ``s`` is
moving-frame mm per section mm: for atlas planes it absorbs paraffin shrinkage,
template size and atrophy together, so it is a nuisance parameter, not a measurement.

The score is ``particle_utils.varifold_score_mm`` with the section as the reference
measure, after the moving particles are pulled into the section frame (masses divided
by s^2, so area is conserved) and multiplied by a *footprint weight*: the section's own
kernel density divided by a quarter of its median over the section particles and
clipped to 1 (so every section particle has weight 1).
Moving tissue far from the section therefore does not count, as in the density support
of ``LossVarifoldNormBoundary_Partial``; moving tissue missing where the section has
tissue does count.  0 is a perfect match; 1 is "no overlapping tissue".

Search = FFT cross-correlation of Gaussian-blurred rasters over translation for a grid
of (theta, s, flip), then Nelder-Mead refinement of the best candidates with the exact
particle score.  Everything here is deterministic numpy/scipy.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import minimize
from scipy.signal import correlate


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

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "Similarity2D":
        return cls(float(values["theta_deg"]), float(values["scale"]), bool(values["flip"]),
                   tuple(float(v) for v in values["tau"]))


# --------------------------------------------------------------------------------------
# Footprint-restricted score
# --------------------------------------------------------------------------------------

def _gauss_sum(X: np.ndarray, Y: np.ndarray, weights: np.ndarray, sigma: float, tree=None) -> np.ndarray:
    """sum_j weights_j exp(-|X_i - Y_j|^2 / 2 sigma^2), truncated at 5 sigma (KD-tree)."""

    from scipy.spatial import cKDTree

    tree_y = tree if tree is not None else cKDTree(Y)
    # "ndarray" output keeps zero distances (coincident points), unlike the sparse formats.
    pairs = cKDTree(X).sparse_distance_matrix(tree_y, 5.0 * sigma, output_type="ndarray")
    values = np.exp(-pairs["v"] ** 2 / (2 * sigma * sigma)) * weights[pairs["j"]]
    return np.bincount(pairs["i"], values, minlength=len(X))


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

    def restricted(self, Y: np.ndarray, mu: np.ndarray, cutoff: float = 1e-3):
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


def _distinct(candidates, k: int, min_shift_mm: float, min_angle_deg: float):
    chosen: list[tuple[float, Similarity2D]] = []
    for value, t in candidates:
        if all(t.flip != c.flip or np.hypot(*np.subtract(t.tau, c.tau)) > min_shift_mm
               or abs((t.theta_deg - c.theta_deg + 180) % 360 - 180) > min_angle_deg for _, c in chosen):
            chosen.append((value, t))
        if len(chosen) == k:
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
               _distinct(candidates, refine_top, min_shift_mm=3.0, min_angle_deg=20.0)]
    return sorted(refined, key=lambda item: item[0])


# --------------------------------------------------------------------------------------
# Atlas planes
# --------------------------------------------------------------------------------------

def plane_basis(pitch_deg: float, yaw_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Orthonormal (u, v, n) of a coronal plane tilted by pitch (about x) then yaw (about z).

    pitch = yaw = 0 gives u = +x, v = +z, n = u x v = -y, the convention of the
    MERFISH pilot's coronal plane.
    """

    p, y = np.deg2rad(pitch_deg), np.deg2rad(yaw_deg)
    Rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
    Rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    R = Rz @ Rx
    u, v = R @ np.array([1.0, 0, 0]), R @ np.array([0, 0, 1.0])
    return u, v, np.cross(u, v)


def reslice_labels(labels: np.ndarray, affine: np.ndarray, origin: Sequence[float], u: np.ndarray,
                   v: np.ndarray, half_width_mm: float, spacing_mm: float) -> tuple[np.ndarray, np.ndarray]:
    """Nearest-neighbour label grid on ``origin + a u + b v``; returns (grid[b, a], axis)."""

    axis = np.arange(-half_width_mm, half_width_mm + 1e-9, spacing_mm)
    aa, bb = np.meshgrid(axis, axis, indexing="xy")
    world = np.asarray(origin, float) + aa[..., None] * u + bb[..., None] * v
    ijk = (world.reshape(-1, 3) - affine[:3, 3]) @ np.linalg.inv(affine[:3, :3]).T
    grid = ndi.map_coordinates(labels, ijk.T, order=0, mode="constant", cval=0)
    return grid.reshape(aa.shape), axis


def label_class_masses(grid: np.ndarray, axis: np.ndarray, spacing_mm: float, classes: Mapping[str, np.ndarray],
                       aggregate: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Particles (a, b, 0) with per-class area masses (mm^2), summed over aggregate^2 pixels.

    ``classes`` maps a feature name to a boolean lookup table over label values.
    """

    n = (len(axis) // aggregate) * aggregate
    grid = grid[:n, :n]
    masses = []
    for table in classes.values():
        member = table[np.clip(grid, 0, len(table) - 1)] & (grid < len(table))
        masses.append(member.reshape(n // aggregate, aggregate, n // aggregate, aggregate).sum((1, 3)))
    nu = np.stack(masses, -1).reshape(-1, len(classes)).astype(np.float64) * spacing_mm ** 2
    centers = axis[:n].reshape(-1, aggregate).mean(1)
    aa, bb = np.meshgrid(centers, centers, indexing="xy")
    X = np.column_stack([aa.ravel(), bb.ravel(), np.zeros(aa.size)])
    keep = nu.sum(1) > 0
    return X[keep], nu[keep]


def lookup_tables(max_label: int, white_matter: Sequence[int], drop: Sequence[int],
                  fine_regions: Mapping[str, Sequence[int]]) -> tuple[dict, dict]:
    """Boolean tables for coarse (gray, white) and fine (named regions, other gray, white).

    Gray matter is every non-zero label that is neither white matter nor dropped
    (ventricles, choroid plexus): an anatomical correspondence fixed in advance, not
    learned from the data.
    """

    size = max_label + 1
    white = np.zeros(size, bool); white[list(white_matter)] = True
    dropped = np.zeros(size, bool); dropped[list(drop)] = True; dropped[0] = True
    gray = ~white & ~dropped
    coarse = {"gray_matter": gray, "white_matter": white}
    fine: dict[str, np.ndarray] = {}
    named = np.zeros(size, bool)
    for name, codes in fine_regions.items():
        table = np.zeros(size, bool); table[list(codes)] = True
        fine[name] = table & gray
        named |= table
    fine["other_gray_matter"] = gray & ~named
    fine["white_matter"] = white
    return coarse, fine
