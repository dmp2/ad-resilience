"""Stacking: where along the atlas's AP axis a set of sections sits, with a noise-scaled window.

Procedure (methodology: docs/section_mapping.md)
================================================
1. *Roles.*  Feature QC of each section's gray/white map: anchors (clean maps) are placed on
   the atlas; passengers inherit an anchor's placement through the section<->section alignment.
2. *Depth profile.*  For each anchor and every coronal depth y, a fresh global 2-D search
   (``search_depths``) gives the best score f_a(y) (lower is better).
3. *Noise scale.*  delta = median, over the planes with the lowest mean relative score, of the
   across-anchor SD of f_a(y) - min_y f_a (``noise_delta``).
4. *Window.*  All depths within delta of an anchor's best (``window``); a profile whose
   median - best <= 2 delta on the first (wide) level is *flat* (the anchor is demoted).
5. *Escalation.*  The next level (more particles) searches only the previous windows (padded
   by one step), adds the previous best pose as an extra start, and extends the range while a
   window touches it (``escalation_ys``, ``range_extension``); stop when windows stop shrinking.
6. *Priors* (opt-in) restrict the admissible set; ``placement_status`` / ``placement_verdict``
   compare the admissible and unconstrained placements on the data-only objective.

Scores are the footprint-restricted multi-scale varifold score of ``alignment`` (atlas plane =
template, section = target).  Nothing here reads files or knows about a particular dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy import ndimage as ndi

from section_mapping.alignment import FootprintScorer, Similarity2D


# --------------------------------------------------------------------------------------
# Feature QC -> roles
# --------------------------------------------------------------------------------------


def otsu_separability(values: np.ndarray, bins: int = 256) -> float:
    """Otsu's eta: between-class / total variance at the optimal threshold (0..1)."""

    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if values.size < 2 or values.std() == 0:
        return 0.0
    hist, edges = np.histogram(values, bins=bins)
    p = hist / hist.sum()
    centres = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(p)
    mu0 = np.cumsum(p * centres)
    mu_t = mu0[-1]
    w1 = 1 - w0
    with np.errstate(divide="ignore", invalid="ignore"):
        between = (mu_t * w0 - mu0) ** 2 / (w0 * w1)
    between = np.nan_to_num(between[:-1])
    total = float((p * (centres - mu_t) ** 2).sum())
    return float(between.max() / total) if total > 0 else 0.0


def stripe_index(p_wm: np.ndarray, mask: np.ndarray, px_mm: float, detrend_mm: float = 1.0,
                 period_mm: tuple[float, float] = (0.4, 1.5), min_line_fraction: float = 0.2) -> dict[str, Any]:
    """Periodic scanner-stripe energy in the gray/white map, along image columns and rows.

    Scanner stripes are periodic bands aligned with the image axes (1.0 mm apart on the
    SEA-AD Aperio scans).  After removing structure larger than ``detrend_mm`` (normalised
    Gaussian smoothing inside tissue), the residual is averaged along each column (row);
    the index is the share of that profile's spectral power in the strongest peak (+-1 bin)
    at periods in ``period_mm``.  Gray/white boundaries give broad spectra and long-period
    peaks (excluded by the band), stripes a sharp short-period peak.
    """

    mask = np.asarray(mask, bool)
    p = np.where(mask, np.asarray(p_wm, float), 0.0)
    sigma = detrend_mm / px_mm
    smooth = ndi.gaussian_filter(p, sigma) / np.maximum(ndi.gaussian_filter(mask.astype(float), sigma), 1e-6)
    residual = np.where(mask, p - smooth, 0.0)
    out: dict[str, Any] = {}
    for name, axis in (("columns", 0), ("rows", 1)):
        count = mask.sum(axis)
        lines = count >= min_line_fraction * max(count.max(), 1)
        if lines.sum() < 8:
            out[name], out[f"{name}_period_mm"] = 0.0, None
            continue
        profile = np.where(lines, residual.sum(axis) / np.maximum(count, 1), 0.0)
        profile = np.where(lines, profile - profile[lines].mean(), 0.0)
        power = np.abs(np.fft.rfft(profile * np.hanning(len(profile)))) ** 2
        freq = np.fft.rfftfreq(len(profile), d=px_mm)
        band = (freq >= 1 / period_mm[1]) & (freq <= 1 / period_mm[0])
        total = power[freq > 0].sum()
        if not band.any() or total <= 0:
            out[name], out[f"{name}_period_mm"] = 0.0, None
            continue
        k = int(np.argmax(np.where(band, power, 0.0)))
        out[name] = float(power[max(k - 1, 0):k + 2].sum() / total)
        out[f"{name}_period_mm"] = float(1 / freq[k])
    out["max"] = max(out["columns"], out["rows"])
    return out


@dataclass
class RoleThresholds:
    otsu_separability_min: float = 0.75
    stripe_index_max: float = 0.15
    layer1_p_wm_max: float = 0.5
    glial_stains: tuple[str, ...] = ("GFAP",)

    @classmethod
    def from_config(cls, roles: Mapping[str, Any]) -> "RoleThresholds":
        return cls(float(roles.get("otsu_separability_min", 0.75)), float(roles.get("stripe_index_max", 0.15)),
                   float(roles.get("layer1_p_wm_max", 0.5)), tuple(roles.get("glial_stains", ("GFAP",))))


def assign_role(stain: str, metrics: Mapping[str, Any], thresholds: RoleThresholds) -> tuple[str, list[str]]:
    """Anchor when every check passes; otherwise passenger, with the failed checks named."""

    reasons = []
    if metrics["otsu_separability"] < thresholds.otsu_separability_min:
        reasons.append(f"Otsu separability {metrics['otsu_separability']:.2f} < {thresholds.otsu_separability_min}")
    if metrics["stripe_index"] > thresholds.stripe_index_max:
        reasons.append(f"stripe index {metrics['stripe_index']:.2f} > {thresholds.stripe_index_max}")
    l1 = metrics.get("layer1_mean_p_wm")
    if l1 is not None and l1 > thresholds.layer1_p_wm_max:
        reasons.append(f"layer 1 read as white matter (mean p(WM) {l1:.2f} > {thresholds.layer1_p_wm_max})")
    if stain in thresholds.glial_stains:
        reasons.append("glial stain")
    return ("passenger" if reasons else "anchor"), reasons


# --------------------------------------------------------------------------------------
# Search over many depths (GPU)
# --------------------------------------------------------------------------------------


def make_scorer(X: np.ndarray, nu: np.ndarray, sigmas: Sequence[float]):
    """GPU scorer of one fixed section measure (constants from the CPU reference)."""

    from section_mapping.alignment_gpu import TorchFootprintScorer

    return TorchFootprintScorer(FootprintScorer(np.asarray(X, float)[:, :2], np.asarray(nu, float), list(sigmas)))


def search_depths(scorer, measures: Sequence[tuple[np.ndarray, np.ndarray]], search: Mapping[str, Any], *,
                  flips: Sequence[bool] | None = None, extra_starts: Sequence = (), chunk: int = 8,
                  min_particles: int = 10):
    """Fresh global 2-D search (FFT candidates over rotations x scales x flips, then batched
    refinement) at every measure; returns per measure (score, Similarity2D) best first."""

    from section_mapping.alignment_gpu import search_planes_gpu

    kwargs = dict(thetas=np.arange(0, 360, float(search.get("theta_step_deg", 15))),
                  scales=[float(s) for s in search.get("scale_grid", (0.9, 1.05, 1.2, 1.35))],
                  flips=list(search.get("flips", (False, True)) if flips is None else flips),
                  scale_bounds=tuple(float(s) for s in search.get("scale_bounds", (0.8, 1.5))),
                  h=float(search.get("fft_grid_mm", 1.0)), blur_mm=float(search.get("fft_blur_mm", 1.5)),
                  refine_top=int(search.get("refine_top", 4)), steps=int(search.get("steps", 150)))
    extra = [t for t in extra_starts if t.flip in kwargs["flips"]]
    # Planes with (almost) no atlas tissue, e.g. outside a region prior, score 1 (no overlap).
    usable = [k for k, (P, _) in enumerate(measures) if len(P) >= min_particles]
    found: dict[int, list] = {}
    for start in range(0, len(usable), chunk):
        batch = usable[start:start + chunk]
        for k, ranked in zip(batch, search_planes_gpu(scorer, [measures[k] for k in batch], extra_starts=extra, **kwargs)):
            found[k] = ranked
    empty = [(1.0, Similarity2D(0.0, 1.0, False, (0.0, 0.0)))]
    return [found.get(k, empty) for k in range(len(measures))]


# --------------------------------------------------------------------------------------
# Profiles, noise scale, windows, escalation
# --------------------------------------------------------------------------------------


def profile_of(rows: Iterable[Mapping[str, Any]], anchor: str, family: str | None = None) -> dict[float, float]:
    """{y: best score} of one anchor (and reflection family) from search rows."""

    best: dict[float, float] = {}
    for r in rows:
        if r["representation"] == anchor and (family is None or r["family"] == family):
            best[r["y0_mm"]] = min(best.get(r["y0_mm"], np.inf), r["score"])
    return best


def best_row(rows: Iterable[Mapping[str, Any]], anchor: str) -> Mapping[str, Any]:
    return min((r for r in rows if r["representation"] == anchor), key=lambda r: r["score"])


def distinct_minima(ys: Sequence[float], scores: Sequence[float], keep: int, min_separation_mm: float) -> list[float]:
    """The ``keep`` best ys, greedily, at least ``min_separation_mm`` apart."""

    chosen: list[float] = []
    for k in np.argsort(scores):
        if all(abs(ys[k] - c) >= min_separation_mm for c in chosen):
            chosen.append(float(ys[k]))
        if len(chosen) == keep:
            break
    return chosen


def noise_delta(profiles: Mapping[str, Mapping[float, float]], top: int = 20) -> dict[str, Any]:
    """delta: median over the best planes of the across-anchor SD of each anchor's score
    relative to its own best (planes ranked by the mean relative score; population SD).

    ``profiles``: anchor -> {y: score}, on a shared set of ys.  Needs >= 2 anchors.
    """

    anchors = list(profiles)
    if len(anchors) < 2:
        return {"delta": None, "reason": "fewer than 2 anchors: no across-anchor noise scale"}
    ys = sorted(set.intersection(*[set(p) for p in profiles.values()]))
    rel = {a: np.array([profiles[a][y] - min(profiles[a].values()) for y in ys]) for a in anchors}
    stack = np.vstack([rel[a] for a in anchors])
    order = np.argsort(stack.mean(0))[: max(1, min(int(top), len(ys)))]
    sds = stack[:, order].std(0)
    return {"delta": float(np.median(sds)), "planes_used": [float(ys[k]) for k in order],
            "definition": "median over the planes with the lowest mean relative score of the population SD "
                          "across anchors of (anchor score - that anchor's best score)"}


def window(profile: Mapping[float, float], delta: float, step_mm: float) -> dict[str, Any]:
    """All ys within delta of the best; the window, its contiguity and the best y."""

    ys = np.array(sorted(profile))
    scores = np.array([profile[y] for y in ys])
    best = int(np.argmin(scores))
    inside = ys[scores <= scores[best] + delta]
    gaps = np.diff(np.sort(inside))
    return {"best_y": float(ys[best]), "best_score": float(scores[best]),
            "window_y": [float(inside.max()), float(inside.min())],
            "width_mm": float(inside.max() - inside.min()),
            "contiguous": bool(np.all(gaps <= step_mm + 1e-6)), "planes_within_delta": int(len(inside)),
            "contrast_median_minus_best": float(np.median(scores) - scores[best]),
            "flat": bool(np.median(scores) - scores[best] <= 2 * delta)}


def summarize_level(level: str, rows: Sequence[Mapping[str, Any]], previous: Mapping[str, Any] | None, *,
                    step_mm: float, delta_top: int, inherit_flat: tuple[set, str] | None = None) -> dict[str, Any]:
    """delta, per-anchor windows, flat anchors and window shrinkage of one profile level.

    Flatness is judged once, on the wide first-level profile (``previous`` None); an escalation
    level spans only the previous windows, where median - best is small by construction, so it
    inherits the first level's flat anchors.  ``inherit_flat`` = (flat anchors, rule) overrides
    the first-level judgement (a prior-limited range cannot show that a profile is flat).
    """

    families = sorted({r["family"] for r in rows})
    anchors = list(dict.fromkeys(r["representation"] for r in rows))
    pooled = {a: profile_of(rows, a) for a in anchors}
    noise = noise_delta(pooled, delta_top)
    delta = noise["delta"]
    out: dict[str, Any] = {"level": level, "families": families, "delta": noise, "anchors": {}}
    if delta is None:
        out["warning"] = noise["reason"]
        return out
    first_level = previous is None and inherit_flat is None
    if inherit_flat is not None and previous is None:
        inherited_flat, flat_rule = set(inherit_flat[0]), inherit_flat[1]
    else:
        inherited_flat = set(previous.get("flat_anchors", [])) if previous else set()
        flat_rule = "median - best <= 2 delta on the first level" if first_level else "inherited from the first level"
    flat = []
    for a in anchors:
        w = window(pooled[a], delta, step_mm)
        w["flat"] = bool(w["flat"]) if first_level else a in inherited_flat
        w["flat_rule"] = flat_rule
        best = best_row(rows, a)
        w["best_transform"] = {k: best[k] for k in ("theta_deg", "scale", "flip", "tau_x", "tau_y")}
        if len(families) > 1:
            w["families"] = {f: window(profile_of(rows, a, f), delta, step_mm) for f in families}
            fb = {f: w["families"][f]["best_score"] for f in families}
            w["family_gap_in_delta"] = (max(fb.values()) - min(fb.values())) / delta
        if w["flat"]:
            flat.append(a)
        out["anchors"][a] = w
    out["flat_anchors"] = flat
    if flat and len(anchors) - len(flat) >= 2:
        kept = {a: pooled[a] for a in anchors if a not in flat}
        out["delta_without_flat"] = noise_delta(kept, delta_top)
    live = [w for w in out["anchors"].values() if not w["flat"]]
    out["union_window_y"] = ([max(w["window_y"][0] for w in live), min(w["window_y"][1] for w in live)]
                             if live else None)
    if previous is not None and previous.get("anchors"):
        shrink = {a: previous["anchors"][a]["width_mm"] - out["anchors"][a]["width_mm"]
                  for a in out["anchors"] if a in previous["anchors"]}
        out["window_shrink_vs_previous_mm"] = shrink
        out["stop_escalating"] = bool(all(v <= 1e-6 for v in shrink.values()))
    return out


def escalation_ys(windows: Iterable[Mapping[str, Any]], step_mm: float, pad_steps: int = 1) -> list[float]:
    """Depths for the next level: the union of the previous windows, padded by one step."""

    ys = set()
    for w in windows:
        hi, lo = w["window_y"]
        for y in np.arange(lo - pad_steps * step_mm, hi + pad_steps * step_mm + 1e-9, step_mm):
            ys.add(round(float(y), 3))
    return sorted(ys, reverse=True)


def range_extension(summary: Mapping[str, Any], ys: Sequence[float], step_mm: float) -> list[float]:
    """Two more planes beyond each end of the scanned range that a live window touches."""

    top, bottom = max(ys), min(ys)
    live = [w for w in summary["anchors"].values() if not w["flat"]]
    return (([round(top + k * step_mm, 3) for k in (1, 2)] if any(w["window_y"][0] >= top - 1e-6 for w in live) else [])
            + ([round(bottom - k * step_mm, 3) for k in (1, 2)] if any(w["window_y"][1] <= bottom + 1e-6 for w in live)
               else []))


def windows_at_bound(summary: Mapping[str, Any], bound_y: Sequence[float], step_mm: float) -> list[str]:
    """Anchors whose window comes within one step of a hard AP bound [anterior, posterior]."""

    return [a for a, w in summary["anchors"].items()
            if not w["flat"] and (w["window_y"][0] >= bound_y[0] - step_mm + 1e-6
                                  or w["window_y"][1] <= bound_y[1] + step_mm - 1e-6)]


# --------------------------------------------------------------------------------------
# Decision rules for placements
# --------------------------------------------------------------------------------------


def search_boundary_flags(summary: Mapping[str, Any], anchor: str, scale: float,
                          scale_bounds: Sequence[float], step_mm: float) -> list[str]:
    """Flags of one placement: window on the searched AP range's edge / the prior bound, scale at a bound."""

    flags = []
    w = summary.get("anchors", {}).get(anchor)
    ys = summary.get("ys_mm") or []
    if w and ys and (w["window_y"][0] >= max(ys) - 1e-6 or w["window_y"][1] <= min(ys) + 1e-6):
        flags.append("window on the edge of the searched AP range")
    if anchor in (summary.get("windows_at_prior_bound") or []):
        flags.append(f"window within {step_mm:g} mm of the slab-prior AP bound")
    lo, hi = (float(s) for s in scale_bounds)
    if scale <= lo * 1.005 or scale >= hi * 0.995:
        flags.append(f"scale {scale:.3f} at its search bound [{lo:g}, {hi:g}]")
    return flags


def same_placement(y_a: float, y_b: float, ribbon_shift_mm: float, step_mm: float, max_shift_mm: float = 1.0) -> bool:
    """Same depth (within one step) and section points within ``max_shift_mm`` (2 atlas voxels)."""

    return abs(y_a - y_b) <= step_mm + 1e-6 and ribbon_shift_mm <= max_shift_mm


def placement_status(same_placement: bool, gap_in_delta: float | None) -> str:
    """How the admissible placement relates to the data-only optimum (lower score = better).

    ``gap_in_delta`` = (data-only score of the admissible placement - data-only score of the
    unconstrained best) / delta.
    """

    if same_placement:
        return "prior inactive: the admissible and unconstrained best placements coincide"
    if gap_in_delta is None:
        return "unresolved: no noise scale"
    if gap_in_delta <= 0:
        return "image-supported: the admissible placement fits at least as well (data only)"
    if gap_in_delta <= 1:
        return "tie within delta: the image does not discriminate; the prior selects"
    return f"prior-driven: the data prefer the excluded placement by {gap_in_delta:.1f} delta"


def placement_verdict(windows: Sequence[Sequence[float]], statuses: Sequence[str],
                      flags: Mapping[str, Sequence[str]]) -> dict[str, Any]:
    """Job verdict over anchors: admissible windows must intersect, nothing flagged, no prior-driven anchor.

    ``windows``: admissible [anterior, posterior] per anchor; ``statuses``: ``placement_status``
    labels (text before ':'); ``flags``: anchor -> search-boundary flags (non-empty only).
    """

    top, bottom = min(w[0] for w in windows), max(w[1] for w in windows)
    reasons = []
    if top < bottom:
        reasons.append("anchors' admissible depth windows do not intersect")
    if flags:
        reasons.append("search-boundary optima: " + "; ".join(f"{a}: {', '.join(f)}" for a, f in flags.items()))
    if any(s == "prior-driven" for s in statuses):
        reasons.append(f"prior-driven for {sum(s == 'prior-driven' for s in statuses)}/{len(statuses)} anchors")
    if all(s in ("prior inactive", "image-supported") for s in statuses) and not reasons:
        verdict = "supported by image geometry"
    elif any(s == "prior-driven" for s in statuses):
        verdict = "primarily prior-driven"
    elif all(s in ("prior inactive", "image-supported", "tie within delta") for s in statuses) and not reasons:
        verdict = "image ties; the prior selects"
    else:
        verdict = "unresolved"
    return {"summary": verdict, "reasons": reasons,
            "admissible_window_intersection_y_mm": [top, bottom] if top >= bottom else None,
            "anchor_statuses": list(statuses)}


ESTABLISHED_VERDICT = "supported by image geometry"
CANDIDATE_POSES = {"unconstrained": ("unconstrained", "search_pose"),
                   "admissible": ("admissible", "search_pose"),
                   "admissible-refined": ("admissible", "refined_on_full_atlas")}


def placement_concerns(report: Mapping[str, Any], anchor: str, verdicts: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """Recorded reasons against exporting an anchor's placement automatically (empty list: eligible).

    ``report``: the run's ``report.json``; ``verdicts``: prior-run folder -> ``candidates.json``
    verdict of the same job.  Concerns: the anchor's search-boundary flags, a non-contiguous window,
    anchors whose windows do not intersect, and any data-only comparison that is not
    "supported by image geometry" (prior-driven, ties, unresolved).
    """

    entry = report["sections"][anchor]
    concerns = [f"{anchor}: {f}" for f in entry.get("flags") or []]
    if not entry.get("window_contiguous", True):
        concerns.append(f"{anchor}: the within-delta window is not contiguous (several depths tie)")
    windows = [e["window_y_mm"] for e in report["sections"].values() if e.get("role") == "anchor"]
    if windows and min(w[0] for w in windows) < max(w[1] for w in windows):
        concerns.append("the anchors' depth windows do not intersect")
    for run, verdict in sorted(verdicts.items()):
        if verdict.get("summary") != ESTABLISHED_VERDICT:
            concerns.append(f"{run}/candidates.json: {verdict.get('summary')}"
                            + (f" ({'; '.join(verdict.get('reasons', []))})" if verdict.get("reasons") else ""))
    return concerns


def candidate_pose(candidates: Mapping[str, Any], anchor: str, name: str) -> tuple[float, Similarity2D, float]:
    """(y, section -> plane pose, data-only score) of a named placement in ``candidates.json``."""

    if name not in CANDIDATE_POSES:
        raise ValueError(f"unknown candidate {name!r}; choose one of {list(CANDIDATE_POSES)}")
    if anchor not in candidates["anchors"]:
        raise KeyError(f"{anchor} was not compared in candidates.json (anchors: {list(candidates['anchors'])})")
    run, which = CANDIDATE_POSES[name]
    ev = candidates["anchors"][anchor][run][which]
    p = ev["pose"]
    return float(ev["y0_mm"]), Similarity2D(float(p["theta_deg"]), float(p["scale"]), bool(p["flip"]),
                                            tuple(float(v) for v in p["tau_mm"])), float(ev["data_only_score"])


# --------------------------------------------------------------------------------------
# Placement diagnostics (boundary correspondence and support; no fitting, no thresholds)
# --------------------------------------------------------------------------------------


def grid_lookup(points: np.ndarray, values: np.ndarray, axis: np.ndarray, res: float) -> np.ndarray:
    """Nearest-pixel values of a square plane grid (grid[b, a], ``axis`` for both) at points (a, b);
    NaN outside the grid."""

    ia = np.rint((points[:, 0] - axis[0]) / res).astype(int)
    ib = np.rint((points[:, 1] - axis[0]) / res).astype(int)
    ok = (ia >= 0) & (ia < values.shape[1]) & (ib >= 0) & (ib < values.shape[0])
    out = np.full(len(points), np.nan)
    out[ok] = values[ib[ok], ia[ok]]
    return out


def boundary_distance_map(mask: np.ndarray, res: float) -> np.ndarray:
    """Distance (mm) from every pixel to the nearest boundary pixel of ``mask`` (inner edge)."""

    mask = np.asarray(mask, bool)
    edge = mask & ~ndi.binary_erosion(mask)
    if not edge.any():
        return np.full(mask.shape, np.inf)
    return ndi.distance_transform_edt(~edge) * float(res)


def region_distance_map(mask: np.ndarray, res: float) -> np.ndarray:
    """Distance (mm) to the nearest pixel of ``mask`` (0 inside)."""

    mask = np.asarray(mask, bool)
    if not mask.any():
        return np.full(mask.shape, np.inf)
    return ndi.distance_transform_edt(~mask) * float(res)


def gray_white_boundary_points(p_wm: np.ndarray, tissue: np.ndarray, origin_mm, px_mm, smooth_px: float = 2.0,
                               max_points: int = 4000) -> np.ndarray:
    """Local-frame points on a section's gray/white boundary (p(WM) = 0.5 inside the tissue).

    Pixels on the white side that touch gray tissue, after smoothing p(WM) within the tissue
    (normalised convolution); the tissue edge itself is excluded.  x = origin_x + col * px_x,
    y = origin_y + row * px_y (the multistain-map convention).
    """

    tissue = np.asarray(tissue, bool)
    p = np.where(tissue, np.asarray(p_wm, float), 0.0)
    w = ndi.gaussian_filter(tissue.astype(float), smooth_px)
    p = np.where(tissue, ndi.gaussian_filter(p, smooth_px) / np.maximum(w, 1e-6), 0.0)
    white = tissue & (p > 0.5)
    gray = tissue & (p <= 0.5)
    rows, cols = np.nonzero(white & ndi.binary_dilation(gray))
    if len(rows) > max_points:
        keep = np.linspace(0, len(rows) - 1, max_points).astype(int)
        rows, cols = rows[keep], cols[keep]
    origin, px = np.asarray(origin_mm, float), np.asarray(px_mm, float)
    return np.column_stack([origin[0] + cols * px[1], origin[1] + rows * px[0]])


def depth_inside_hull(points: np.ndarray, outline_points: np.ndarray) -> np.ndarray:
    """Distance (mm) of each point to the boundary of the convex hull of ``outline_points``.

    For pia points (which lie on the tissue): far from the hull = lining a sulcus; near it =
    the exposed surface."""

    from scipy.spatial import ConvexHull

    hull = outline_points[ConvexHull(outline_points).vertices]
    a, b = hull, np.roll(hull, -1, axis=0)
    ab = b - a
    t = np.clip(((points[:, None, :] - a[None]) * ab[None]).sum(-1) / np.maximum((ab ** 2).sum(-1), 1e-12), 0, 1)
    nearest = a[None] + t[..., None] * ab[None]
    return np.linalg.norm(points[:, None, :] - nearest, axis=-1).min(1)


def distance_summary(d: np.ndarray) -> dict[str, float | int | None]:
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    if not len(d):
        return {"p50_mm": None, "p90_mm": None, "points": 0}
    return {"p50_mm": float(np.median(d)), "p90_mm": float(np.percentile(d, 90)), "points": int(len(d))}
