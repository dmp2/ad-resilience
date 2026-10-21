"""Bounded planar IV-Particle approximation, in native millimeters.

Reference: approxCode/approximateIntermediate.py, make_loss/make_loss2/optimize.
The spatial kernel is exp(-distance**2 / sigma_mm**2), WITHOUT a factor 2.
Feature masses are a**2, initialized with sqrt(nu_Z0); zero entries therefore
remain locked at zero, as in the reference. No mass penalty or normalization.

Uses dense CPU float64 PyTorch operations and the reference's torch L-BFGS
settings, rather than its GPU float32 KeOps block truncation. This is deliberately
limited to small measures; no registration, coordinate rescaling, or I/O here.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time

import numpy as np
import torch


def _kernel(x: torch.Tensor, y: torch.Tensor, sigma_mm: float) -> torch.Tensor:
    return torch.exp(-torch.cdist(x, y, compute_mode="donot_use_mm_for_euclid_dist").square()
                     / sigma_mm**2)


def varifold_squared(x, nu_x, z, nu_z, sigma_mm: float):
    """Three-term squared distance for tensors; supports autograd, no clipping.

    Inputs must share dtype/device and feature dimensions. Sigma uses the
    IV-Particle convention. Use the optimizer below for validated array inputs.
    """
    if not math.isfinite(sigma_mm) or sigma_mm <= 0:
        raise ValueError("sigma_mm must be finite and positive")
    return ((_kernel(x, x, sigma_mm) @ nu_x) * nu_x).sum() + (
        (_kernel(z, z, sigma_mm) @ nu_z) * nu_z).sum() - 2 * (
        (_kernel(z, x, sigma_mm) @ nu_x) * nu_z).sum()


@dataclass
class ApproximationResult:
    X: np.ndarray
    nu: np.ndarray
    reference_self_norm: float
    objective_history: list[float]  # initial + accepted outer-step endpoints
    closure_history: list[float]  # includes line-search trial points
    seconds: float
    iterations: int
    gradient_max_abs: float


def optimize_particles(
    X, nu_X, Z0, nu_Z0, *, sigma_mm: float,
    optimize_positions: bool = False, outer_steps: int = 30,
    max_pairs: int = 4_000_000,
) -> ApproximationResult:
    """Optimize a fixed number of planar particles without changing inputs.

    Positions retain their order and native XY convention; z must be exactly 0
    and is excluded from the parameters. All computations use CPU float64.
    Each outer step allows 10 L-BFGS iterations/evaluations, history size 3,
    strong-Wolfe line search, and PyTorch default tolerances (as in reference).
    Pair-count guard is a pilot bound, not a general memory guarantee.
    """
    start = time.perf_counter()
    arrays = [np.asarray(v, dtype=np.float64) for v in (X, nu_X, Z0, nu_Z0)]
    x, nx, z, nz = arrays
    if not math.isfinite(sigma_mm) or sigma_mm <= 0:
        raise ValueError("sigma_mm must be finite and positive")
    if not isinstance(outer_steps, int) or outer_steps < 0:
        raise ValueError("outer_steps must be a nonnegative integer")
    for points, masses in ((x, nx), (z, nz)):
        if (points.ndim != 2 or points.shape[1] != 3 or len(points) == 0
                or masses.ndim != 2 or len(masses) != len(points) or masses.shape[1] == 0):
            raise ValueError("Expected nonempty [N,3] points and [N,F] masses")
        if not np.isfinite(points).all() or not np.isfinite(masses).all():
            raise ValueError("Coordinates and masses must be finite")
        if np.any(masses < 0) or np.any(points[:, 2] != 0):
            raise ValueError("Masses must be nonnegative and positions planar z=0")
    if nx.shape[1] != nz.shape[1]:
        raise ValueError("Measures must share feature dimensions")
    if max(len(x)**2, len(z)**2, len(x)*len(z)) > max_pairs:
        raise ValueError("Dense pilot pair-count bound exceeded; use a supported backend")
    tx, tnx, tz, tnz = [torch.tensor(v, dtype=torch.float64, device="cpu") for v in arrays]
    with torch.no_grad():
        c = ((_kernel(tx, tx, sigma_mm) @ tnx) * tnx).sum()
        fixed_k = None if optimize_positions else _kernel(tz, tz, sigma_mm)
        fixed_cross = None if optimize_positions else _kernel(tz, tx, sigma_mm) @ tnx
    n_xy = len(z) * 2 if optimize_positions else 0
    p = torch.cat([tz[:, :2].flatten(), tnz.sqrt().flatten()] if optimize_positions
                  else [tnz.sqrt().flatten()]).requires_grad_(True)

    def unpack():
        positions = (torch.cat([p[:n_xy].reshape(-1, 2), torch.zeros_like(tz[:, 2:])], dim=1)
                     if optimize_positions else tz)
        return positions, p[n_xy:].reshape(tnz.shape).square()

    def objective():
        positions, masses = unpack()
        k = _kernel(positions, positions, sigma_mm) if optimize_positions else fixed_k
        cross = _kernel(positions, tx, sigma_mm) @ tnx if optimize_positions else fixed_cross
        return ((k @ masses) * masses).sum() - 2 * (cross * masses).sum() + c

    optimizer = torch.optim.LBFGS([p], max_eval=10, max_iter=10,
                                 line_search_fn="strong_wolfe", history_size=3)
    closure_history = []

    def closure():
        optimizer.zero_grad(set_to_none=True)
        loss = objective()
        loss.backward()
        if not torch.isfinite(loss) or not torch.isfinite(p.grad).all():
            raise FloatingPointError("Nonfinite objective or gradient")
        closure_history.append(float(loss.detach()))
        return loss

    with torch.no_grad():
        history = [float(objective())]
    for _ in range(outer_steps):
        optimizer.step(closure)
        with torch.no_grad():
            value = float(objective())
        if not math.isfinite(value):
            raise FloatingPointError("Nonfinite accepted objective")
        history.append(value)
    closure()  # evaluate gradient at the returned state, not a line-search trial
    positions, masses = unpack()
    return ApproximationResult(
        positions.detach().numpy().copy(), masses.detach().numpy().copy(), float(c),
        history, closure_history, time.perf_counter() - start,
        int(optimizer.state[p].get("n_iter", 0)), float(p.grad.abs().max()),
    )
