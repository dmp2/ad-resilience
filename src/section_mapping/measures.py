"""Representation: feature-valued particle measures, their kernel inner products and compression.

A section or atlas plane is a discrete measure ``mu = sum_i nu_i delta_{x_i}``: positions
x_i (mm, 2-D) and nonnegative feature masses nu_i in R^C (mm^2 of each class; gray/white for
the structural features).  Two measures are compared through the Gaussian kernel inner product

    <mu_X, mu_Y>_sigma = sum_ij exp(-|x_i - y_j|^2 / 2 sigma^2) <a_i, b_j>.

Compression replaces a measure by fewer particles at spacing ~h whose positions and masses are
optimised jointly to minimise the varifold distance to the original (kernel exp(-d^2/sigma^2),
sigma = max(1, h) mm; ``particle_approximation.optimize_particles``).  A section is compressed
once per level and reused by every search.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from scipy import ndimage as ndi


def gauss_inners(X: np.ndarray, a: np.ndarray, Y: np.ndarray, b: np.ndarray, sigmas: Sequence[float]) -> np.ndarray:
    """<mu_X, mu_Y>_sigma for every sigma (float64, chunked)."""

    if len(X) == 0 or len(Y) == 0:
        return np.zeros(len(sigmas))
    out = np.zeros(len(sigmas))
    feature = np.asarray(a, float) @ np.asarray(b, float).T
    keep_rows = np.abs(feature).sum(1) > 0
    X, Y_, F = np.asarray(X, float)[keep_rows], np.asarray(Y, float), feature[keep_rows]
    for start in range(0, len(X), 2048):
        d2 = ((X[start:start + 2048, None, :] - Y_[None, :, :]) ** 2).sum(-1)
        f = F[start:start + 2048]
        for k, s in enumerate(sigmas):
            out[k] += float((np.exp(-d2 / (2.0 * s * s)) * f).sum())
    return out


# --------------------------------------------------------------------------------------
# Compression (joint positions + masses)
# --------------------------------------------------------------------------------------


def grid_initialization(X: np.ndarray, nu: np.ndarray, h_mm: float) -> tuple[np.ndarray, np.ndarray]:
    """Particles summed into h-mm blocks: mass-weighted centroid, summed per-feature masses."""

    X = np.asarray(X, float)[:, :2]
    nu = np.asarray(nu, float)
    keys = np.floor((X - X.min(0)) / float(h_mm)).astype(np.int64)
    _, index = np.unique(keys, axis=0, return_inverse=True)
    index = index.ravel()
    groups = index.max() + 1
    mass = nu.sum(1)
    weight = np.bincount(index, mass, groups)
    centre = np.column_stack([np.bincount(index, mass * X[:, k], groups) for k in range(2)])
    plain = np.column_stack([np.bincount(index, X[:, k], groups) for k in range(2)]) / np.bincount(index, None, groups)[:, None]
    centre = np.where(weight[:, None] > 0, centre / np.maximum(weight, 1e-300)[:, None], plain)
    masses = np.column_stack([np.bincount(index, nu[:, c], groups) for c in range(nu.shape[1])])
    keep = masses.sum(1) > 0
    return centre[keep], masses[keep]


def compression_sigma(h_mm: float) -> float:
    """Kernel width of the joint compression: sigma = max(1, h) mm (kernel exp(-d^2/sigma^2))."""

    return max(1.0, float(h_mm))


def compress(X: np.ndarray, nu: np.ndarray, h_mm: float, *, outer_steps: int = 30,
             max_pairs: int = 16_000_000) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Jointly optimised positions and masses at spacing ~h (``optimize_particles``), planar XY.

    Returns (Z[:, :2], nu_Z, info).  CPU float64, as validated in the particle-approximation
    pilot; the pair guard is ``max_pairs`` (one float64 N x N matrix of 4000 particles is 128 MB).
    """

    from preprocess.particle_approximation import optimize_particles, varifold_squared
    import torch

    X2 = np.asarray(X, float)[:, :2]
    nu = np.asarray(nu, float)
    Z0, nu0 = grid_initialization(X2, nu, h_mm)
    sigma = compression_sigma(h_mm)
    pad = lambda P: np.column_stack([P, np.zeros(len(P))])
    result = optimize_particles(pad(X2), nu, pad(Z0), nu0, sigma_mm=sigma, optimize_positions=True,
                                outer_steps=int(outer_steps), max_pairs=int(max_pairs))
    with torch.no_grad():
        err = lambda P, m: float(varifold_squared(*[torch.tensor(v, dtype=torch.float64)
                                                     for v in (pad(X2), nu, pad(P), m)], sigma))
        initial, final = err(Z0, nu0), err(result.X[:, :2], result.nu)
    info = {"h_mm": float(h_mm), "sigma_mm": sigma, "kernel": "exp(-d^2/sigma^2)", "count": int(len(Z0)),
            "reference_count": int(len(X2)), "initialization": "h-mm block sums (mass-weighted centroids)",
            "outer_steps": int(outer_steps), "seconds": result.seconds,
            "initial_squared_error": initial, "final_squared_error": final,
            "relative_error_final": final / max(result.reference_self_norm, 1e-300),
            "total_mass": float(result.nu.sum()), "reference_total_mass": float(nu.sum()),
            "total_mass_deviation_percent": float(100 * (result.nu.sum() - nu.sum()) / nu.sum()),
            "compression_method": "image_varifold", "backend": "particle_approximation.optimize_particles (CPU float64)"}
    return result.X[:, :2], result.nu, info


# --------------------------------------------------------------------------------------
# Rasterisation (figures)
# --------------------------------------------------------------------------------------


def splat(points: np.ndarray, weights: np.ndarray, xs: np.ndarray, zs: np.ndarray, res: float,
          sigma_px: float = 1.0) -> np.ndarray:
    """Particle masses summed onto the grid (zs, xs) at nearest pixels, Gaussian-blurred."""

    H = np.zeros((len(zs), len(xs)))
    ix = np.rint((points[:, 0] - xs[0]) / res).astype(int)
    iz = np.rint((points[:, 1] - zs[0]) / res).astype(int)
    ok = (ix >= 0) & (ix < len(xs)) & (iz >= 0) & (iz < len(zs))
    np.add.at(H, (iz[ok], ix[ok]), weights[ok])
    return ndi.gaussian_filter(H, sigma_px)
