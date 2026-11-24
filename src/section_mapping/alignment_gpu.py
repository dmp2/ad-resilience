"""Geometric alignment on the GPU: batched twin of ``alignment`` (score, FFT candidates, refinement).

The score is exactly ``alignment.FootprintScorer.score_transform`` (same footprint weight,
cutoff, kernels and normalisation; float32 like the CPU inner products), evaluated for a batch
of transforms -- and a batch of moving measures, e.g. many atlas planes padded to a common
size -- in one call.  Constants (footprint density reference, section self terms) are taken
from a CPU ``FootprintScorer``, so both paths agree by construction and can be cross-checked
(tests: <= 2e-6).

Search mirrors ``alignment.search``: FFT cross-correlation of blurred rasters over translation
for a (theta, s, flip) grid, then refinement of the best distinct candidates.  Refinement here
is batched projected Adam on (theta, tau, log s) at fixed flip, keeping the best iterate,
instead of one Nelder-Mead per start; with many starts refined at once it is cheaper to refine
more candidates than the CPU path does.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as tF

from section_mapping.alignment import FOOTPRINT_CUTOFF, TRUNCATE_SIGMAS, FootprintScorer, Similarity2D, distinct_poses


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pad_measures(measures: Sequence[tuple[np.ndarray, np.ndarray]], far: float = 1e4):
    """Stack variable-size moving measures into [B, M, 2] / [B, M, C]; padding is massless and far away."""

    size = max(len(P) for P, _ in measures)
    channels = measures[0][1].shape[1]
    P = np.full((len(measures), size, 2), far, np.float32)
    mu = np.zeros((len(measures), size, channels), np.float32)
    for b, (points, masses) in enumerate(measures):
        P[b, : len(points)] = np.asarray(points, np.float32)[:, :2]
        mu[b, : len(points)] = masses
    return P, mu


@dataclass
class TorchFootprintScorer:
    """GPU twin of a ``FootprintScorer`` (fixed section measure X, nu)."""

    cpu: FootprintScorer
    device: torch.device | None = None
    memory_budget_bytes: float = 1.5e9

    def __post_init__(self):
        self.device = self.device or default_device()
        to = dict(device=self.device, dtype=torch.float32)
        self.X = torch.as_tensor(self.cpu.X, **to)
        self.nu = torch.as_tensor(self.cpu.nu, **to)
        self.area = self.nu.sum(1)
        self.sigmas = list(self.cpu.sigmas)
        self.footprint_sigma = float(self.cpu.footprint_sigma)
        self.density_ref = float(self.cpu.density_ref)
        self.self_terms = torch.as_tensor(self.cpu.self_terms, **to)

    # ---- core -----------------------------------------------------------------------
    @staticmethod
    def _d2(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """Exact squared distances [B, M, N] (explicit differences: cdist's exact mode is ~50x slower,
        and its matmul mode loses float32 precision near zero distance)."""

        return (A[:, :, None, 0] - B[:, None, :, 0]) ** 2 + (A[:, :, None, 1] - B[:, None, :, 1]) ** 2

    def footprint(self, Y: torch.Tensor) -> torch.Tensor:
        """Section support weight in [0, 1] at section-frame points Y [B, M, 2]."""

        d2 = self._d2(Y, self.X.expand(Y.shape[0], -1, -1))
        s2 = self.footprint_sigma ** 2
        kernel = torch.exp(-d2 / (2 * s2)) * (d2 <= (TRUNCATE_SIGMAS ** 2) * s2)
        return torch.clamp(kernel @ self.area / self.density_ref, max=1.0)

    def score_points(self, Y: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        """Score of moving measures already pulled into the section frame: Y [B, M, 2], mu [B, M, C]."""

        weight = self.footprint(Y)
        weight = torch.where((weight > FOOTPRINT_CUTOFF) & (mu.sum(-1) > 0), weight, torch.zeros_like(weight))
        k = int((weight > 0).sum(1).max())
        if k == 0:
            return torch.ones(Y.shape[0], device=Y.device)
        index = torch.topk(weight, k, dim=1).indices
        Yk = torch.gather(Y, 1, index[..., None].expand(-1, -1, 2))
        muk = torch.gather(mu, 1, index[..., None].expand(-1, -1, mu.shape[-1])) * torch.gather(weight, 1, index)[..., None]
        d2_aa, d2_ab = self._d2(Yk, Yk), self._d2(Yk, self.X.expand(Y.shape[0], -1, -1))
        feature_aa = muk @ muk.transpose(1, 2)
        feature_ab = muk @ self.nu.T
        terms = []
        for s, bb in zip(self.sigmas, self.self_terms):
            aa = (torch.exp(d2_aa * (-0.5 / (s * s))) * feature_aa).sum((1, 2))
            ab = (torch.exp(d2_ab * (-0.5 / (s * s))) * feature_ab).sum((1, 2))
            terms.append(torch.clamp(aa + bb - 2.0 * ab, min=0.0) / bb)
        empty = (weight > 0).sum(1) == 0
        return torch.where(empty, torch.ones_like(terms[0]), torch.stack(terms).mean(0))

    def support_size(self, Y: torch.Tensor, mu: torch.Tensor) -> int:
        """Largest number of moving particles inside the footprint over a batch (sizes the aa terms)."""

        size, k = self.chunk_size(Y.shape[1], Y.shape[1]), 0
        with torch.no_grad():
            for a in range(0, Y.shape[0], size):
                w = self.footprint(Y[a:a + size])
                k = max(k, int(((w > FOOTPRINT_CUTOFF) & (mu[a:a + size].sum(-1) > 0)).sum(1).max()))
        return k

    def chunk_size(self, M: int, K: int) -> int:
        """Batch elements per call so the pairwise tensors (x2 for autograd) stay within the memory budget."""

        N = len(self.X)
        per_element = 4.0 * 2 * (3 * M * N + 4 * K * K + 4 * K * N)
        return max(1, int(self.memory_budget_bytes // per_element))

    # ---- transforms -----------------------------------------------------------------
    @staticmethod
    def pull_back(P: torch.Tensor, mu: torch.Tensor, theta_deg, tau, log_scale, flip):
        """Y = A^-1 p + tau with A = s R(theta) F; masses / s^2 (Similarity2D.moving_to_section)."""

        angle = torch.deg2rad(theta_deg)
        c, s = torch.cos(angle), torch.sin(angle)
        f = torch.where(flip, -torch.ones_like(c), torch.ones_like(c))
        scale = torch.exp(log_scale)
        # A^-1 = F R(theta)^T / s; rows of A^-1: [c, s] / s and f * [-s, c] / s
        inv = torch.stack([torch.stack([c, s], -1), torch.stack([-s * f, c * f], -1)], -2) / scale[:, None, None]
        Y = P @ inv.transpose(1, 2) + tau[:, None, :]
        return Y, mu / (scale ** 2)[:, None, None]

    def score_params(self, P, mu, theta_deg, tau, log_scale, flip) -> torch.Tensor:
        Y, m = self.pull_back(P, mu, theta_deg, tau, log_scale, flip)
        return self.score_points(Y, m)

    def score_transforms(self, transforms: Sequence[Similarity2D], P: np.ndarray, mu: np.ndarray) -> np.ndarray:
        """Convenience: many transforms of one moving measure (numpy in, numpy out)."""

        out = []
        Pt, mut = self._expand(P, mu, len(transforms))
        th, tau, ls, fl = self._params(transforms)
        with torch.no_grad():
            Y, m = self.pull_back(Pt, mut, th, tau, ls, fl)
        size = self.chunk_size(len(P), self.support_size(Y, m))
        for start in range(0, len(transforms), size):
            part = transforms[start:start + size]
            Pt, mut = self._expand(P, mu, len(part))
            th, tau, ls, fl = self._params(part)
            with torch.no_grad():
                out.append(self.score_params(Pt, mut, th, tau, ls, fl).cpu().numpy())
        return np.concatenate(out).astype(float)

    def _expand(self, P, mu, n):
        to = dict(device=self.device, dtype=torch.float32)
        return (torch.as_tensor(np.asarray(P)[:, :2], **to).expand(n, -1, -1),
                torch.as_tensor(np.asarray(mu), **to).expand(n, -1, -1))

    def _params(self, transforms):
        to = dict(device=self.device, dtype=torch.float32)
        return (torch.tensor([t.theta_deg for t in transforms], **to),
                torch.tensor([t.tau for t in transforms], **to).reshape(-1, 2),
                torch.tensor([np.log(t.scale) for t in transforms], **to),
                torch.tensor([t.flip for t in transforms], device=self.device))


# --------------------------------------------------------------------------------------
# Refinement
# --------------------------------------------------------------------------------------

def refine_batch(scorer: TorchFootprintScorer, P: np.ndarray, mu: np.ndarray, starts: Sequence[Similarity2D],
                 scale_bounds: tuple[float, float] = (1.0, 1.0), steps: int = 150,
                 lr: tuple[float, float, float] = (1.0, 0.1, 0.01)) -> list[tuple[float, Similarity2D]]:
    """Projected Adam on (theta [deg], tau [mm], log s) per start, flip fixed; best iterate kept.

    P, mu: one moving measure [M, 2] / [M, C] shared by every start, or a batch [B, M, 2] /
    [B, M, C] aligned with ``starts`` (e.g. one padded atlas plane per start).
    """

    dev = scorer.device
    to = dict(device=dev, dtype=torch.float32)
    B = len(starts)
    if np.ndim(P) == 2:
        Pt, mut = scorer._expand(P, mu, B)
    else:
        Pt, mut = torch.as_tensor(np.asarray(P)[..., :2], **to), torch.as_tensor(np.asarray(mu), **to)
    theta, tau, log_s, flip = scorer._params(starts)
    fixed_scale = scale_bounds[0] == scale_bounds[1]
    lo, hi = np.log(scale_bounds[0]), np.log(scale_bounds[1])
    if fixed_scale:
        log_s = torch.full_like(log_s, lo)
    variables = [theta.clone().requires_grad_(), tau.clone().requires_grad_()]
    groups = [dict(params=[variables[0]], lr=lr[0]), dict(params=[variables[1]], lr=lr[1])]
    if not fixed_scale:
        variables.append(log_s.clone().requires_grad_())
        groups.append(dict(params=[variables[2]], lr=lr[2]))
    optimiser = torch.optim.Adam(groups)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, steps, eta_min=0.0)
    best = torch.full((B,), np.inf, **to)
    best_theta, best_tau, best_log_s = theta.clone(), tau.clone(), log_s.clone()
    with torch.no_grad():
        Y0, m0 = scorer.pull_back(Pt, mut, theta, tau, log_s, flip)
    # Margin on the support size: the footprint can admit more particles as the pose moves.
    size = scorer.chunk_size(Pt.shape[1], int(1.3 * scorer.support_size(Y0, m0)) + 16)
    for _ in range(steps + 1):
        optimiser.zero_grad()
        current_log_s = log_s if fixed_scale else variables[2]
        scores = torch.empty(B, **to)
        for a in range(0, B, size):
            b = slice(a, a + size)
            value = scorer.score_params(Pt[b], mut[b], variables[0][b], variables[1][b], current_log_s[b], flip[b])
            value.sum().backward()
            scores[b] = value.detach()
        with torch.no_grad():
            improved = scores < best
            best = torch.where(improved, scores, best)
            best_theta = torch.where(improved, variables[0], best_theta)
            best_tau = torch.where(improved[:, None], variables[1], best_tau)
            best_log_s = torch.where(improved, current_log_s, best_log_s)
        optimiser.step()
        schedule.step()
        if not fixed_scale:
            with torch.no_grad():
                variables[2].clamp_(lo, hi)
    out = []
    for i in range(B):
        t = Similarity2D(float(best_theta[i]), float(np.exp(float(best_log_s[i]))), bool(flip[i]),
                         (float(best_tau[i, 0]), float(best_tau[i, 1])))
        out.append((float(best[i]), t))
    return out


# --------------------------------------------------------------------------------------
# FFT candidates
# --------------------------------------------------------------------------------------

def _raster(points: torch.Tensor, masses: torch.Tensor, origin: torch.Tensor, shape: tuple[int, int], h: float):
    """Sum masses into [B, C, rows, cols]; points [B, M, 2] (x, y), origins [B, 2]."""

    B, _, C = masses.shape
    index = torch.floor((points - origin[:, None, :]) / h).long()
    keep = (index[..., 0] >= 0) & (index[..., 1] >= 0) & (index[..., 0] < shape[1]) & (index[..., 1] < shape[0])
    flat = (index[..., 1] * shape[1] + index[..., 0]).clamp(0, shape[0] * shape[1] - 1)
    grid = torch.zeros(B, C, shape[0] * shape[1], device=points.device)
    values = (masses * keep[..., None]).transpose(1, 2)
    grid.scatter_add_(2, flat[:, None, :].expand(-1, C, -1), values)
    return grid.reshape(B, C, *shape)


def _blur(grid: torch.Tensor, sigma_px: float) -> torch.Tensor:
    radius = int(4.0 * sigma_px + 0.5)  # scipy gaussian_filter truncate=4
    x = torch.arange(-radius, radius + 1, device=grid.device, dtype=grid.dtype)
    k = torch.exp(-0.5 * (x / sigma_px) ** 2); k = k / k.sum()
    B, C, H, W = grid.shape
    g = grid.reshape(B * C, 1, H, W)
    g = tF.conv2d(tF.pad(g, (radius, radius, 0, 0), mode="reflect"), k.view(1, 1, 1, -1))
    g = tF.conv2d(tF.pad(g, (0, 0, radius, radius), mode="reflect"), k.view(1, 1, -1, 1))
    return g.reshape(B, C, H, W)


def _correlate_valid(R: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """scipy correlate(R, K, 'valid') for R [B, H, W] and a smaller K [h, w]."""

    H, W = R.shape[-2:]
    h, w = K.shape
    spectrum = torch.fft.rfft2(R) * torch.conj(torch.fft.rfft2(K, s=(H, W)))
    return torch.fft.irfft2(spectrum, s=(H, W))[..., : H - h + 1, : W - w + 1]


def fft_candidates_gpu(scorer: TorchFootprintScorer, P: np.ndarray, mu: np.ndarray, thetas: Iterable[float],
                       scales: Iterable[float], flips: Iterable[bool], h: float = 0.5, blur_mm: float = 1.0,
                       per_combo: int = 2) -> list[tuple[float, Similarity2D]]:
    """Same ranking values as ``alignment.fft_candidates``, all (theta, s, flip) combos batched."""

    dev = scorer.device
    X, nu = scorer.cpu.X, scorer.cpu.nu
    margin = 3.0 * blur_mm
    q_origin = X.min(0) - margin
    q_shape = tuple(int(v) for v in np.ceil((X.max(0) + margin - q_origin) / h)[::-1] + 1)
    sigma_px = blur_mm / h
    to = dict(device=dev, dtype=torch.float32)
    Q = _blur(_raster(torch.as_tensor(X[None], **to), torch.as_tensor(nu[None], **to),
                      torch.as_tensor(q_origin[None], **to), q_shape, h), sigma_px)[0]
    centers = q_origin + (np.indices(q_shape)[::-1].reshape(2, -1).T + 0.5) * h
    F = torch.as_tensor(scorer.cpu.footprint(centers).reshape(q_shape), **to)
    FQ = F * Q
    norm = float((Q ** 2).sum())

    combos = [(flip, scale, theta) for flip in flips for scale in scales for theta in thetas]
    transforms = [Similarity2D(theta, scale, flip, (0.0, 0.0)) for flip, scale, theta in combos]
    Pt, mut = scorer._expand(P, mu, len(combos))
    th, tau0, ls, fl = scorer._params(transforms)
    with torch.no_grad():
        Y, m = TorchFootprintScorer.pull_back(Pt, mut, th, tau0, ls, fl)
    pad = np.array(q_shape[::-1]) * h
    lo = Y.amin(1).cpu().numpy() - pad - margin
    hi = Y.amax(1).cpu().numpy() + pad + margin
    p_shape = tuple(int(v) for v in np.ceil((hi - lo).max(0) / h)[::-1] + 1)
    out = []
    chunk = max(1, int(2e8 // (np.prod(p_shape) * (m.shape[-1] + 2) * 8)))
    for a in range(0, len(combos), chunk):
        b = slice(a, a + chunk)
        origin = torch.as_tensor(lo[b], **to)
        R = _blur(_raster(Y[b], m[b], origin, p_shape, h), sigma_px)
        cross = sum(_correlate_valid(R[:, c], FQ[c]) for c in range(R.shape[1]))
        energy = _correlate_valid((R ** 2).sum(1), F ** 2)
        approx = ((norm + energy - 2.0 * cross) / norm).reshape(R.shape[0], -1)
        values, flat = torch.topk(-approx, per_combo, dim=1)
        width = p_shape[1] - q_shape[1] + 1
        for i in range(R.shape[0]):
            flip, scale, theta = combos[a + i]
            for v, f in zip((-values[i]).tolist(), flat[i].tolist()):
                d_row, d_col = divmod(f, width)
                t = q_origin - (lo[a + i] + np.array([d_col, d_row]) * h)
                out.append((float(v), Similarity2D(theta, scale, flip, (float(t[0]), float(t[1])))))
    out.sort(key=lambda item: item[0])
    return out


def search_gpu(scorer: TorchFootprintScorer, P: np.ndarray, mu: np.ndarray, *, thetas, scales, flips,
               scale_bounds, h=0.5, blur_mm=1.0, refine_top=4, steps=150) -> list[tuple[float, Similarity2D]]:
    """FFT candidates, then batched refinement of the best distinct ones; best first."""

    candidates = fft_candidates_gpu(scorer, P, mu, thetas, scales, flips, h=h, blur_mm=blur_mm)
    starts = [t for _, t in distinct_poses(candidates, refine_top, min_shift_mm=3.0, min_angle_deg=20.0)]
    return sorted(refine_batch(scorer, P, mu, starts, scale_bounds, steps), key=lambda item: item[0])


def search_planes_gpu(scorer: TorchFootprintScorer, measures: Sequence[tuple[np.ndarray, np.ndarray]], *,
                      thetas, scales, flips, scale_bounds, h=0.5, blur_mm=1.0, refine_top=4, steps=150,
                      extra_starts: Sequence[Similarity2D] = ()) -> list[list[tuple[float, Similarity2D]]]:
    """``search_gpu`` for many moving measures (e.g. atlas planes at several depths), refined in one batch.

    ``extra_starts`` (e.g. the pose found at a coarser level) are refined on every measure too.
    Returns, per measure, its refined candidates best first.
    """

    owners, starts = [], []
    for k, (P, mu) in enumerate(measures):
        candidates = fft_candidates_gpu(scorer, P, mu, thetas, scales, flips, h=h, blur_mm=blur_mm)
        chosen = [t for _, t in distinct_poses(candidates, refine_top, min_shift_mm=3.0, min_angle_deg=20.0)]
        for t in [*chosen, *extra_starts]:
            owners.append(k); starts.append(t)
    padded_P, padded_mu = pad_measures(measures)
    refined = refine_batch(scorer, padded_P[owners], padded_mu[owners], starts, scale_bounds, steps)
    out: list[list[tuple[float, Similarity2D]]] = [[] for _ in measures]
    for k, result in zip(owners, refined):
        out[k].append(result)
    return [sorted(r, key=lambda item: item[0]) for r in out]
