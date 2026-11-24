#!/usr/bin/env python3
"""Map one donor's QNP sections onto the AHRAv2 atlas: roles, stacking coordinate, priors, xIV hand-off.

The workflow entry point (methodology, workflow and implementation map: docs/section_mapping.md).
Computations live in ``src/section_mapping``; this script reads the job's files, calls them and
writes results.  Roles: atlas plane = template (moving), QNP section = target (fixed).  A coronal
plane's stacking coordinate is its MNI y.  Prerequisites: ``prepare_qnp_section_particles.py``
(``--stage inventory``, ``particles``) and ``align_qnp_multistain.py`` for the job.

Stacking stages (``atlas_scan.stacking``; outputs in ``stacking.output_dir``)
------------------------------------------------------------------------------
``roles``      feature QC of each section's gray/white map -> anchor or passenger (``roles.json``)
``compress``   joint XY + mass compression of every section at each compressed level
``slabs``      fast pass: coronal search slabs over the target region's y-extent, one global 2-D
               search per slab and anchor (with ``--landmark-slab``: the prior's AP interval)
``profile``    ``--level L``: per anchor, a fresh global 2-D search at every coronal depth; delta and
               within-delta windows.  Later levels search only the previous windows, with the previous
               best poses as extra starts, and extend past any window touching the searched range.
``passengers`` passengers inherit an anchor's placement via the QNP<->QNP alignment, re-checked by a
               direct outline fit
``report``     depth-profile and best-pose figures, ``report.json`` and ``REPORT.md``
``stack``      roles -> compress -> slabs -> every profile level -> passengers -> report
``landmark``   (implies ``--landmark-slab``) landmark, candidate slabs, ``landmark_slab.json``, figures
``candidates`` (with a prior) per anchor, the unconstrained and admissible best placements on one
               data-only objective (full atlas), with boundary, support, pose and tilt diagnostics

Priors (off by default; each run goes to its own sub-folder beside the unconstrained run)
``--region-prior``  admissible atlas support near ``stacking.region_prior.regions`` -> ``region_prior/``
``--landmark-slab`` AP interval of the anatomical slab prior (``stacking.landmark_slab``) -> ``landmark_slab/``
                    (both: ``region_prior__landmark_slab/``)

Plane stages (``qnp_atlas_planes.py``; outputs in ``atlas_scan.output_dir``)
----------------------------------------------------------------------------
``scan``, ``summarize``  tilted-plane scan of the consensus representations and its noise summary
``export``               xIV inputs for one plane of the scan table (``--plane``, default the best scanned plane)
``evaluate``             after the xIV fit (``--plane``, ``--fit-dir``): ribbon landing before / after

Stacking -> xIV hand-off (``--section S``, with the run's prior flags)
----------------------------------------------------------------------
``export --section S``   xIV inputs at S's stacking coordinate with its stacking pose (a passenger takes its
                         host anchor's pose).  Unresolved, prior-driven or flagged placements are refused unless
                         ``--candidate`` names the placement: ``report`` (this run's report pose),
                         ``unconstrained``, ``admissible`` or ``admissible-refined`` (from the run's
                         ``candidates.json``).  Outputs: ``export.stacking_dir/<run>/<id>/``, record
                         ``<run folder>/export_<id>.json``.
``evaluate --section S --plane P --fit-dir D``  the same evaluation for that export
``graph``                transform graph of slides, QNP frame, plane hypotheses and MNI
``atlas-features``       anatomical atlas channels: checks and figures

Usage::

    PYTHONPATH=src:src/config_setup:src/preprocess:src/scripts \\
    python src/scripts/map_qnp_sections.py --config configs/project_config.yaml --job H20.33.034_MTG --stage stack
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from config_setup.project_config import load_config, resolve_path
from config_setup.qnp_jobs import resolve_qnp_job
from section_mapping.alignment import Similarity2D
from section_mapping.atlas import FINE_ORDER_TAIL, code_tables

STACK_STAGES = ("roles", "compress", "slabs", "profile", "passengers", "report", "stack", "landmark", "candidates")
PLANE_STAGES = ("scan", "summarize", "export", "evaluate", "graph", "atlas-features")
STAGE_ALIASES = {"stack-report": "report"}


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--job", required=True, help="Name under qnp_section_inputs.")
    parser.add_argument("--stage", required=True, choices=[*STACK_STAGES, *PLANE_STAGES, *STAGE_ALIASES])
    parser.add_argument("--level", help="--stage profile: stacking level (stacking.levels); default: the first "
                                        "level not yet profiled")
    parser.add_argument("--families", action="store_true",
                        help="--stage profile: per-reflection-family depth profiles instead of one global search "
                             "over both reflections")
    parser.add_argument("--overwrite", action="store_true", help="--stage compress: recompute existing archives")
    parser.add_argument("--region-prior", action="store_true", help="stacking stages: dissection-region prior")
    parser.add_argument("--landmark-slab", action="store_true", help="stacking stages: anatomical slab prior")
    parser.add_argument("--representation", action="append", help="--stage scan: only these representations")
    parser.add_argument("--plane", help="--stage export/evaluate: plane id (pitch_yaw_y0); export default: the best")
    parser.add_argument("--fit-dir", help="--stage evaluate: the xIV init_* directory")
    parser.add_argument("--section", help="--stage export/evaluate: take the pose from the stacking report of this "
                                          "section (anchor, or passenger via its host anchor)")
    parser.add_argument("--candidate", choices=["report", "unconstrained", "admissible", "admissible-refined"],
                        help="--stage export/evaluate --section: explicitly selected placement (required when the "
                             "placement is unresolved, prior-driven or flagged)")
    parser.add_argument("--dense", action="store_true",
                        help="--stage scan/summarize: re-search atlas_scan.dense_search (search-robustness check)")
    args = parser.parse_args(argv)
    args.stage = STAGE_ALIASES.get(args.stage, args.stage)
    return args


# --------------------------------------------------------------------------------------
# Job context
# --------------------------------------------------------------------------------------

class StackContext:
    """Paths, sections and settings of one job's stacking run (no donor-specific values)."""

    def __init__(self, config: dict, job: dict, region_prior: bool = False, landmark_slab: bool = False):
        self.config, self.job = config, job
        self.scan = job["atlas_scan"]
        self.cfg = self.scan["stacking"]
        self.prior = dict(self.cfg["region_prior"]) if region_prior else None
        self.slab_cfg = None
        if landmark_slab:
            cfg = self.cfg.get("landmark_slab")
            if not cfg:
                raise SystemExit("--landmark-slab: no atlas_scan.stacking.landmark_slab block for this job")
            regions = cfg.get("applies_to_regions")
            if regions and job.get("region") not in regions:
                raise SystemExit(f"--landmark-slab: the configured rule applies to {regions}, not to region "
                                 f"{job.get('region')!r}; give this job its own landmark_slab block")
            self.slab_cfg = dict(cfg)
        self.base_out = resolve_path(config, self.cfg["output_dir"])
        tags = [t for t, on in (("region_prior", self.prior), ("landmark_slab", self.slab_cfg)) if on]
        self.out = self.base_out / "__".join(tags) if tags else self.base_out
        self.out.mkdir(parents=True, exist_ok=True)
        self._slab_prior = None
        self.aligned = resolve_path(config, job["alignment"]["output_dir"])
        self.compressed = resolve_path(config, self.cfg["compressed_dir"])
        self.particle_dir = resolve_path(config, job["particle_dir"])
        self.qc_dir = resolve_path(config, job["qc_dir"])
        self.record = json.loads((self.aligned / "transforms.json").read_text())
        self.reference = job["reference_stain"]
        self.stems = {self.reference: self.record["reference_slide"],
                      **{s: e["slide"] for s, e in self.record["edges"].items()}}
        self.stains = [s for s in job["stains"] if s in self.stems]
        self.sigmas = [float(s) for s in self.scan.get("score_sigmas_mm", [2.0, 1.0, 0.5])]
        self.levels = list(self.cfg["levels"])
        self.compression_levels = dict(self.cfg["compression"]["levels"])
        self.step = float(self.cfg["profile"].get("step_mm", 0.5))
        self.scale_bounds = tuple(float(s) for s in self.cfg["search"].get("scale_bounds", (0.8, 1.5)))
        self.regions = list(self.scan["fine_regions"])
        self._atlas = None

    @property
    def atlas(self) -> dict:
        if self._atlas is None:
            from qnp_atlas_planes import atlas_setup

            self._atlas = atlas_setup(self.config, self.scan)
        return self._atlas

    @property
    def qc_tables(self) -> dict[str, np.ndarray]:
        """Extra named QC structures (e.g. insula) from ``landmark_slab.figure_label_codes``."""

        return code_tables(int(self.atlas["labels"].max()) + 1,
                           self.cfg.get("landmark_slab", {}).get("figure_label_codes") or {})

    @property
    def slab_prior(self) -> dict | None:
        """Landmark + candidate slabs + search interval (``landmark_slab.json``), or None."""

        if self.slab_cfg is None:
            return None
        if self._slab_prior is None:
            from section_mapping.atlas import landmark_slab_prior

            try:
                record = landmark_slab_prior(self.atlas, self.slab_cfg, self.scan.get("hemisphere", "left"))
            except ValueError as error:
                raise SystemExit(str(error))
            record["atlas"] = {"annotation_nifti": self.scan["annotation_nifti"],
                               "space": "AHRAv2 on MNI ICBM 2009b symmetric, world mm"}
            self._slab_prior = record
            (self.out / "landmark_slab.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        return self._slab_prior

    def search_y_range(self) -> tuple[float, float, str]:
        """(posterior y, anterior y, description) of the AP range the stacking search covers."""

        if self.slab_prior is not None:
            hi, lo = self.slab_prior["search_interval"]["y_mm"]
            lm = self.slab_prior["landmark"]
            return lo, hi, (f"landmark slab prior: {lm['name']} at y {lm['y_mm']:+.1f} mm ({lm['method'].split(':')[0]}), "
                            f"{self.slab_prior['slab_thickness_mm']:g} mm slabs j in {self.slab_prior['slab_offsets']} "
                            f"over every cutting phase, +- {self.slab_prior['extension_mm']:g} mm: y {hi:+.1f} ... {lo:+.1f}")
        lo, hi = self.atlas["region_extent_y"]
        pad = float(self.cfg["slabs"].get("y_pad_mm", 5.0))
        return lo - pad, hi + pad, (f"slab range: atlas {next(iter(self.scan['fine_regions']))} y-extent +- {pad} mm "
                                    "(dissection region)")

    def h_mm(self, level: str) -> float | None:
        return float(self.compression_levels[level]) if level in self.compression_levels else None

    def particle_path(self, stain: str, level: str, frame: str = "local") -> Path:
        stem = self.stems[stain]
        suffix = "_local" if frame == "local" else ""
        if level in self.compression_levels:
            return self.compressed / f"{stem}_tissue_classes_{level}{suffix}.npz"
        return (self.aligned / f"{stem}_tissue_classes_{level}_local.npz" if frame == "local"
                else self.particle_dir / f"{stem}_tissue_classes_{level}.npz")

    def particles(self, stain: str, level: str, frame: str = "local"):
        from preprocess.particle_utils import load_particle_archive

        X, nu, meta = load_particle_archive(self.particle_path(stain, level, frame), "nu_Sub")
        return X[:, :2].astype(np.float64), nu.astype(np.float64), meta

    def scorer(self, stain: str, level: str, tissue_only: bool = False):
        """GPU scorer of one section at one level (score SDs < h/2 dropped)."""

        from section_mapping.alignment import level_sigmas
        from section_mapping.stacking import make_scorer

        X, nu, _ = self.particles(stain, level)
        nu = nu.sum(1, keepdims=True) if tissue_only else nu
        return make_scorer(X, nu, level_sigmas(self.sigmas, self.h_mm(level))), len(X)

    def slide_to_local(self, stain: str) -> np.ndarray:
        if stain == self.reference:
            return np.eye(3)
        return np.asarray(self.record["edges"][stain]["slide_mm_to_local_mm"], float)

    def roles(self) -> dict:
        path = self.base_out / "roles.json"
        if not path.exists():
            raise SystemExit("Run --stage roles first.")
        return json.loads(path.read_text())

    def anchors(self) -> list[str]:
        return [s for s, r in self.roles()["sections"].items() if r["role"] == "anchor"]

    def ribbon_local(self):
        with np.load(resolve_path(self.config, self.scan["ribbon_points"])) as archive:
            return archive["X"][:, :2].astype(np.float64), archive["depth_from_pia_um"].astype(np.float64)

    def multistain_maps(self) -> tuple[dict, list[str]]:
        with np.load(self.aligned / "multistain_maps_32um_local.npz") as archive:
            ms = {k: archive[k] for k in archive.files}
        return ms, [str(s) for s in ms["stains"]]

    def profiled_levels(self, folder: Path | None = None) -> list[str]:
        folder = folder or self.out
        return [lv for lv in self.levels if (folder / f"profile_{lv}.json").exists()]

    def pose_panel(self, stain: str, y: float, transform: Similarity2D, x_range, z_range, title: str, flags,
                   extra: dict | None = None) -> dict:
        """One ``figures_mapping.draw_pose`` panel of the section at its full level."""

        X, nu, _ = self.particles(stain, self.levels[-1])
        rib, depth = self.ribbon_local()
        return {"pose": dict(atlas=self.atlas, regions=self.regions, extra=self.qc_tables if extra is None else extra,
                             y=y, transform=transform, X=X, nu=nu, pia=rib[depth <= 100], x_range=x_range,
                             z_range=z_range, prior=self.prior),
                "title": title + "".join(f"\n⚠ {f}" for f in flags), "flagged": bool(flags)}

    def figure_ranges(self):
        return (tuple(self.cfg.get("figure_x_range_mm", (-80.0, -10.0))),
                tuple(self.cfg.get("figure_z_range_mm", (-50.0, 30.0))))


def write_rows(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path: Path) -> list[dict]:
    with open(path, newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key in ("y0_mm", "score", "theta_deg", "scale", "tau_x", "tau_y"):
            if key in row:
                row[key] = float(row[key])
    return rows


def search_row(level: str, anchor: str, y: float, score: float, t: Similarity2D, particles: int,
               family: str | None = None) -> dict:
    row = {"level": level, "representation": anchor}
    if family is not None:
        row["family"] = family
    return {**row, "y0_mm": float(y), "score": score, **t.as_row(), "particles": particles}


# --------------------------------------------------------------------------------------
# Stage: roles
# --------------------------------------------------------------------------------------

def run_roles(ctx: StackContext) -> dict:
    from section_mapping.stacking import RoleThresholds, assign_role, stripe_index
    from visualization.figures_mapping import roles_figure

    summary = json.loads((ctx.qc_dir / "particles_summary.json").read_text())
    ms, ms_stains = ctx.multistain_maps()
    pad = int(ctx.job["alignment"].get("grid_pad_px", 60))
    ref_maps = np.load(ctx.particle_dir / f"{ctx.stems[ctx.reference]}_maps_32um.npz")
    layer1 = None
    if "layers" in ref_maps:
        layers = np.pad(ref_maps["layers"], pad)
        if layers.shape != ms["p_wm"].shape[1:]:
            raise RuntimeError(f"reference layer raster {layers.shape} does not match the local grid {ms['p_wm'].shape[1:]}")
        layer1 = layers == 1  # LAYER_ORDER[0] = L1
    roles_cfg = ctx.cfg.get("roles", {})
    thresholds = RoleThresholds.from_config(roles_cfg)
    override = roles_cfg.get("override") or {}
    sections = {}
    for stain in ctx.stains:
        maps = np.load(ctx.particle_dir / f"{ctx.stems[stain]}_maps_32um.npz")
        stripes = stripe_index(maps["p_wm"].astype(float), maps["mask"].astype(bool), float(np.mean(maps["px_mm"])))
        k = ms_stains.index(stain)
        tissue, p = ms["tissue"][k].astype(bool), ms["p_wm"][k].astype(float)
        l1 = None
        if layer1 is not None and (layer1 & tissue).sum() > 50:
            l1 = float(p[layer1 & tissue].mean())
        metrics = {"otsu_separability": float(summary[stain]["membership"]["otsu_separability"]),
                   "stripe_index": stripes["max"], "stripes": stripes, "layer1_mean_p_wm": l1,
                   "layer1_source": f"{ctx.reference} HALO layer 1 carried into the local frame" if l1 is not None else None}
        role, reasons = assign_role(stain, metrics, thresholds)
        entry = {"role_from_qc": role, "failed_checks": reasons, "metrics": metrics, "role": role,
                 "role_source": "feature QC"}
        if stain in override:
            entry.update(role=str(override[stain]), role_source="user override (stacking.roles.override)")
        sections[stain] = entry
        print(f"{stain}: {entry['role']} ({entry['role_source']}) sep {metrics['otsu_separability']:.3f} "
              f"stripes {metrics['stripe_index']:.3f} L1 {l1 if l1 is None else round(l1, 2)} {reasons}", flush=True)
    record = {"job": ctx.job["_name"], "thresholds": thresholds.__dict__, "sections": sections,
              "note": "flat depth profiles demote an anchor at --stage profile (reported there)"}
    (ctx.base_out / "roles.json").write_text(json.dumps(record, indent=2, default=list) + "\n", encoding="utf-8")
    roles_figure(ms, ms_stains, ctx.stains, sections, layer1, ctx.base_out / "roles_gray_white_maps.png")
    return record


# --------------------------------------------------------------------------------------
# Stage: compress
# --------------------------------------------------------------------------------------

def _save_compressed(path: Path, X2: np.ndarray, nu: np.ndarray, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, X=np.column_stack([X2, np.zeros(len(X2))]).astype(np.float64), nu_Sub=nu,
                        metadata_json=json.dumps(meta))


def run_compress(ctx: StackContext, overwrite: bool = False) -> dict:
    """Compress each section's native 500 um particles once per level; also write the local copy.

    The compression objective is invariant under rigid motions, so compressing in the slide
    frame and then applying the slide -> local rigid transform equals compressing in the local
    frame (up to roundoff); the native archive is what the figures draw.
    """

    from section_mapping.measures import compress

    comp = ctx.cfg["compression"]
    reference_tag = ctx.cfg.get("reference_bin_tag", "500um")
    path_json = ctx.base_out / "compression.json"
    record = json.loads(path_json.read_text()) if path_json.exists() else {}
    for stain in ctx.stains:
        X, nu, meta = ctx.particles(stain, reference_tag, frame="native")
        M = ctx.slide_to_local(stain)
        for level, h in ctx.compression_levels.items():
            native, local = ctx.particle_path(stain, level, "native"), ctx.particle_path(stain, level, "local")
            if native.exists() and local.exists() and not overwrite and f"{stain}:{level}" in record:
                print(f"exists, skipped: {stain} {level}")
                continue
            Z, nuZ, info = compress(X, nu, float(h), outer_steps=int(comp.get("outer_steps", 30)),
                                    max_pairs=int(comp.get("max_pairs", 16_000_000)))
            base = {k: meta[k] for k in ("slide", "structural_feature") if k in meta}
            common = {**base, **info, "level": level, "features": ["gray_matter", "white_matter"],
                      "source_archive": str(ctx.particle_path(stain, reference_tag, "native")), "mass_units": "mm^2"}
            _save_compressed(native, Z, nuZ, {**common, "coordinate_space": "qnp_slide_mm"})
            Zl = Z @ M[:2, :2].T + M[:2, 2]
            _save_compressed(local, Zl, nuZ, {**common, "coordinate_space": "qnp_local_mm",
                                              "slide_to_local": M.tolist(),
                                              "alignment_transforms": str(ctx.aligned / "transforms.json")})
            record[f"{stain}:{level}"] = {k: info[k] for k in ("count", "reference_count", "h_mm", "sigma_mm", "seconds",
                                                               "relative_error_final", "total_mass_deviation_percent")}
            print(f"{stain} {level}: {info['reference_count']} -> {info['count']} particles in {info['seconds']:.1f} s, "
                  f"relative error {info['relative_error_final']:.4f}, mass {info['total_mass_deviation_percent']:+.3f}%",
                  flush=True)
            path_json.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


# --------------------------------------------------------------------------------------
# Stage: slabs (fast pass)
# --------------------------------------------------------------------------------------

def run_slabs(ctx: StackContext) -> dict:
    from section_mapping.atlas import region_y_extent, slab_measure
    from section_mapping.stacking import distinct_minima, search_depths

    slabs = ctx.cfg["slabs"]
    level = ctx.levels[0]
    lo, hi = region_y_extent(ctx.atlas)
    search_lo, search_hi, search_note = ctx.search_y_range()
    stride, half = float(slabs.get("stride_mm", 3.0)), float(slabs.get("half_thickness_mm", 3.0))
    # Search slabs (blurred atlas for the fast pass), not tissue slabs: centres span the search range.
    centres = np.round(np.arange(search_hi, search_lo - 1e-9, -stride), 3)
    measures = [slab_measure(ctx.atlas, ctx.scan, y, half, float(slabs.get("plane_step_mm", 0.5)), ctx.prior)
                for y in centres]
    rows, kept = [], {}
    for anchor in ctx.anchors():
        scorer, count = ctx.scorer(anchor, level)
        t0 = time.time()
        results = search_depths(scorer, measures, ctx.cfg["search"])
        scores = []
        for y, ranked in zip(centres, results):
            score, t = ranked[0]
            scores.append(score)
            rows.append(search_row(level, anchor, y, score, t, count))
        top = distinct_minima(list(centres), scores, int(slabs.get("keep", 3)), 2 * stride)
        inside = [(s, y) for y, s in zip(centres, scores) if lo <= y <= hi]
        prior = float(min(inside)[1]) if inside else None
        kept[anchor] = {"top_distinct": top, "best_prior_consistent": prior}
        print(f"slabs {anchor}: {len(centres)} slabs in {time.time() - t0:.1f} s; top {top}, prior-consistent {prior}", flush=True)
    write_rows(ctx.out / f"slabs_{level}.csv", rows)
    union = sorted({y for k in kept.values() for y in k["top_distinct"] + ([k["best_prior_consistent"]] if k["best_prior_consistent"] is not None else [])},
                   reverse=True)
    record = {"level": level, "slab_half_thickness_mm": half, "stride_mm": stride, "centres_mm": centres.tolist(),
              "target_region_y_extent_mm": [lo, hi], "per_anchor": kept, "kept_centres_mm": union,
              "prior": f"dissection region: slabs whose centre lies in the atlas {ctx.regions[0]} y-extent",
              "region_prior": ctx.prior, "search_y_range_mm": [search_hi, search_lo], "search_range": search_note}
    if ctx.slab_prior is not None:
        # The slab interval is already narrow: profile level 1 scans every plane in it (no shortlist).
        record["landmark_slab"] = "landmark_slab.json"
        record["profile_first_level"] = "every plane of search_y_range_mm (kept_centres_mm not used)"
    (ctx.out / "slabs.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


# --------------------------------------------------------------------------------------
# Stage: profile
# --------------------------------------------------------------------------------------

def _first_level_flatness(ctx: StackContext, level: str) -> tuple[set, str] | None:
    """Under the slab prior the range is too narrow to judge flatness (and its top-plane delta is
    inflated): inherit it from the unconstrained first level, or do not judge it."""

    if ctx.slab_prior is None:
        return None
    wide = ctx.base_out / f"profile_{level}.json"
    if wide.exists():
        return (set(json.loads(wide.read_text()).get("flat_anchors", [])),
                f"inherited from the unconstrained first level ({wide.name})")
    return set(), "not judged: the landmark-slab range is too narrow (run the unconstrained profile to judge it)"


def run_profile(ctx: StackContext, level: str, families: bool) -> dict:
    from section_mapping.atlas import clamp_ys, plane_measure
    from section_mapping.stacking import (escalation_ys, range_extension, search_depths, summarize_level,
                                          windows_at_bound)

    prof = ctx.cfg["profile"]
    step = ctx.step
    index = ctx.levels.index(level)
    previous = None
    extra: dict[str, list] = {}
    bound = ctx.slab_prior["search_interval"]["y_mm"] if ctx.slab_prior is not None else None  # [anterior, posterior]
    if index == 0 and bound is not None:
        # Every plane of the slab interval, on the step lattice (no slab shortlist needed).
        ys = sorted({round(float(y), 3) for y in np.arange(np.ceil(bound[1] / step) * step, bound[0] + 1e-9, step)},
                    reverse=True)
    elif index == 0:
        slabs = json.loads((ctx.out / "slabs.json").read_text())
        pad = float(prof.get("pad_mm", 3.0))
        ys = sorted({round(float(y), 3) for c in slabs["kept_centres_mm"]
                     for y in np.arange(c - pad, c + pad + 1e-9, step)}, reverse=True)
    else:
        prev_level = ctx.levels[index - 1]
        prev_file = ctx.out / f"profile_{prev_level}.json"
        if not prev_file.exists():
            raise SystemExit(f"Run --stage profile --level {prev_level} first (escalation is conditioned on it).")
        previous = json.loads(prev_file.read_text())
        ys = escalation_ys([w for w in previous["anchors"].values() if not w["flat"]], step)
        for a, w in previous["anchors"].items():
            extra[a] = [Similarity2D.from_row(w["best_transform"])]
    if bound is not None:
        ys = clamp_ys(ys, bound)
    family_flips = {"current": [True], "flipped": [False]} if families else {"both": None}
    anchors = ctx.anchors() if previous is None else list(previous["anchors"])
    scorers = {anchor: ctx.scorer(anchor, level) for anchor in anchors}
    inherit = _first_level_flatness(ctx, level) if previous is None else None
    delta_top = int(ctx.cfg.get("delta_top_planes", 20))
    rows: list[dict] = []

    def profile_planes(planes):
        measures = [plane_measure(ctx.atlas, ctx.scan, y, ctx.prior) for y in planes]
        for anchor, (scorer, count) in scorers.items():
            for family, flips in family_flips.items():
                t0 = time.time()
                results = search_depths(scorer, measures, ctx.cfg["search"], flips=flips,
                                        extra_starts=extra.get(anchor, ()))
                for y, ranked in zip(planes, results):
                    score, t = ranked[0]
                    rows.append(search_row(level, anchor, y, score, t, count, family))
                best = min(r["score"] for r in rows if r["representation"] == anchor and r["family"] == family)
                print(f"{level} {anchor} {family}: {len(planes)} planes in {time.time() - t0:.1f} s; best so far {best:.4f}",
                      flush=True)

    def summarize():
        return summarize_level(level, rows, previous, step_mm=step, delta_top=delta_top, inherit_flat=inherit)

    profile_planes(ys)
    extensions = []
    for _ in range(int(prof.get("max_extensions", 8))):
        summary = summarize()
        if summary["delta"]["delta"] is None:
            break
        grow = range_extension(summary, ys, step)
        if bound is not None:
            grow = clamp_ys(grow, bound)  # the slab prior is a hard AP bound; touching it is reported
        if not grow:
            break
        extensions += grow
        ys = sorted(set(ys) | set(grow), reverse=True)
        print(f"a window touches the scanned range; extending by {grow}", flush=True)
        profile_planes(grow)
    write_rows(ctx.out / f"profile_{level}.csv", rows)
    summary = summarize()
    summary["range_extensions_mm"] = extensions
    if bound is not None and summary.get("anchors"):
        summary["landmark_slab_bound_y_mm"] = bound
        summary["windows_at_prior_bound"] = windows_at_bound(summary, bound, step)
    summary.update(ys_mm=ys, particles={a: int(next(r["particles"] for r in rows if r["representation"] == a))
                                        for a in anchors},
                   conditioned_on=None if previous is None else previous["level"],
                   note=("the global search includes both reflections unless --families" if not families else
                         "per-reflection-family profiles: flip fixed per family"))
    (ctx.out / f"profile_{level}.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"delta": summary["delta"]["delta"], "windows": {a: (w["best_y"], w["window_y"], w["flat"])
                                                                     for a, w in summary["anchors"].items()},
                      "stop_escalating": summary.get("stop_escalating")}, indent=1))
    return summary


def latest_level(ctx: StackContext) -> str:
    done = ctx.profiled_levels()
    if not done:
        raise SystemExit("Run --stage profile first.")
    return done[-1]


# --------------------------------------------------------------------------------------
# Stage: passengers
# --------------------------------------------------------------------------------------

def run_passengers(ctx: StackContext) -> dict:
    """Each passenger is placed through its QNP<->QNP tissue-outline fit to an anchor.

    The alignment stage already fitted every slide to the reference frame (tissue outline for
    non-structural stains, both reflections, checked by HALO ribbon / white-matter Dice), so a
    passenger inherits the chosen anchor's atlas pose and depth window.  Here the passenger is
    also re-fitted directly to that anchor (global GPU search, all rotations, both flips, coarse
    then full level) as a consistency check of the composed placement.
    """

    from section_mapping.alignment_gpu import refine_batch, search_gpu

    level = latest_level(ctx)
    summary = json.loads((ctx.out / f"profile_{level}.json").read_text())
    anchors = [a for a, w in summary["anchors"].items() if not w["flat"]]
    roles = ctx.roles()["sections"]
    passengers = [s for s in ctx.stains if s not in anchors]
    host = ctx.reference if ctx.reference in anchors else anchors[0]
    pcfg = ctx.cfg.get("passengers", {})
    coarse, fine = pcfg.get("coarse_level", ctx.levels[1] if len(ctx.levels) > 1 else ctx.levels[0]), pcfg.get("fine_level", ctx.levels[-1])
    sc_fine, _ = ctx.scorer(host, fine, tissue_only=True)
    sc_coarse, _ = ctx.scorer(host, coarse, tissue_only=True)
    out = {"host_anchor": host, "host_level": level, "host_window_y": summary["anchors"][host]["window_y"],
           "host_best_y": summary["anchors"][host]["best_y"], "passengers": {}}
    for stain in passengers:
        P, mu, _ = ctx.particles(stain, coarse)
        Pf, muf, _ = ctx.particles(stain, fine)
        ranked = search_gpu(sc_coarse, P, mu.sum(1, keepdims=True),
                            thetas=np.arange(0, 360, float(pcfg.get("theta_step_deg", 10))), scales=[1.0],
                            flips=[False, True], scale_bounds=(1.0, 1.0), h=0.5, blur_mm=1.0, refine_top=4)
        score, t = refine_batch(sc_fine, Pf, muf.sum(1, keepdims=True), [ranked[0][1]], (1.0, 1.0), 200)[0]
        # Both are in the local frame, so the composed placement is the identity: report how far
        # the direct fit moves the passenger's particles from it.
        displacement = np.linalg.norm(t.moving_to_section(Pf) - Pf, axis=1)
        out["passengers"][stain] = {
            "role_reason": roles.get(stain, {}).get("failed_checks") or ["flat depth profile"],
            "placement": f"inherits {host}'s atlas pose via the local-frame alignment edge",
            "inherited_window_y": summary["anchors"][host]["window_y"],
            "direct_outline_fit_to_host": {"score": score, "transform": t.as_dict(),
                                           "displacement_from_composed_mm": dict(zip(
                                               ["p50", "p90", "max"], np.percentile(displacement, [50, 90, 100]).tolist()))},
            "alignment_edge": {k: ctx.record["edges"][stain].get(k) for k in ("align_features", "score_fine", "reflection_check")}
            if stain in ctx.record["edges"] else "reference frame",
        }
        print(f"{stain}: outline fit to {host} score {score:.4f}, flip {t.flip}, moves particles "
              f"p50 {np.percentile(displacement, 50):.2f} mm / p90 {np.percentile(displacement, 90):.2f} mm from the composed placement",
              flush=True)
    (ctx.out / "passengers.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return out


# --------------------------------------------------------------------------------------
# Stage: report
# --------------------------------------------------------------------------------------

def _slab_report(ctx: StackContext, final: dict, rows: list[dict], anchors: list[str]) -> dict:
    """Candidate slabs S_j(delta) consistent with each anchor's final profile."""

    from section_mapping.atlas import slab_consistency
    from section_mapping.stacking import profile_of

    sp = ctx.slab_prior
    per_anchor = {}
    for a in anchors:
        res = slab_consistency(profile_of(rows, a), sp["candidates"], final["delta"]["delta"])
        per_j = {}
        for s in res["slabs"]:
            cur = per_j.get(str(s["j"]))
            if cur is None or s["best_score"] < cur["best_score"]:
                per_j[str(s["j"])] = {k: s[k] for k in ("best_score", "best_y_mm", "delta_mm")}
        per_anchor[a] = {"consistent_slabs_j_delta_range_mm": res["consistent"], "best_per_offset_j": per_j}
    return {"record": "landmark_slab.json", "landmark": {k: sp["landmark"][k] for k in ("name", "y_mm", "s_mm", "method")},
            "search_interval_y_mm": sp["search_interval"]["y_mm"],
            "windows_at_prior_bound": final.get("windows_at_prior_bound", []),
            "slab_consistency": per_anchor,
            "note": "consistent = candidate slabs S_j(delta) whose best plane is within delta of the best "
                    "slab, at the final level; a slab prior is not image evidence"}


def _report_markdown(ctx: StackContext, report: dict, level: str) -> list[str]:
    lines = [f"# Stacking coordinate: {ctx.job['_name']}", "",
             f"Final level: {level}. δ per level: " + ", ".join(f"{lv} {d:.4f}" for lv, d in report["delta_per_level"].items() if d),
             "", "| Section | Role | y (mm) | Window (mm) | Deciding level | θ (°) | scale | flip | score | MTG landing (QC) |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    target = ctx.regions[0]
    for s, e in report["sections"].items():
        t = e["transform"] if isinstance(e["transform"], dict) else {}
        lines.append(f"| {s} | {e['role']} | {e['stacking_coordinate_y_mm']:+.1f} | {e['window_y_mm'][0]:+.1f} … {e['window_y_mm'][1]:+.1f} | "
                     f"{e['deciding_level']} | {t.get('theta_deg', float('nan')):.1f} | {t.get('scale', float('nan')):.3f} | "
                     f"{t.get('flip', '')} | {e.get('score', float('nan')):.4f} | "
                     f"{e.get('ribbon_landing_qc_only', {}).get(target, float('nan')):.2f} |")
    flagged = {s: e["flags"] for s, e in report["sections"].items() if e.get("flags")}
    if flagged:
        lines += ["", "Search-boundary flags: " + "; ".join(f"{s}: {', '.join(f)}" for s, f in flagged.items()) + "."]
    lines += ["", "Priors: " + "; ".join(report["priors"]) + ".", "", report["per_slice_z"] + "."]
    if "landmark_slab" in report:
        ls_ = report["landmark_slab"]
        lines += ["", f"Landmark slab ({ls_['landmark']['name']} y {ls_['landmark']['y_mm']:+.1f} mm; "
                      f"search y {ls_['search_interval_y_mm'][0]:+.1f} … {ls_['search_interval_y_mm'][1]:+.1f}). "
                      "Candidate slabs S_j(δ) consistent with each anchor's final profile "
                      "(j: −1 anterior, 0 landmark slab, +1 posterior; δ = cutting phase, mm):", ""]
        for a, c in ls_["slab_consistency"].items():
            lines.append(f"- {a}: " + ", ".join(f"j {j} (δ {d[0]:g}–{d[1]:g})" for j, d in
                                                 sorted(c["consistent_slabs_j_delta_range_mm"].items(), key=lambda kv: int(kv[0]))))
        if ls_["windows_at_prior_bound"]:
            lines += ["", f"Windows reaching the prior's AP bound (evidence pushes against the prior): "
                          f"{', '.join(ls_['windows_at_prior_bound'])}."]
    return lines


def run_report(ctx: StackContext) -> dict:
    from section_mapping.alignment import FootprintScorer
    from section_mapping.atlas import landing_fractions, plane_geometry
    from section_mapping.stacking import best_row, search_boundary_flags
    from visualization.figures_mapping import depth_profiles_figure, poses_figure

    done = ctx.profiled_levels()
    if not done:
        raise SystemExit("Run --stage profile first.")
    level = done[-1]
    summaries = {lv: json.loads((ctx.out / f"profile_{lv}.json").read_text()) for lv in done}
    rows = {lv: read_rows(ctx.out / f"profile_{lv}.csv") for lv in done}
    roles = ctx.roles()["sections"]
    final = summaries[level]
    anchors = list(final["anchors"])
    regions = ctx.regions + list(FINE_ORDER_TAIL)
    rib, _ = ctx.ribbon_local()
    rib = rib[:: int(ctx.scan.get("ribbon_subsample", 10))]
    report = {"job": ctx.job["_name"], "final_level": level, "levels_run": done,
              "delta_per_level": {lv: s["delta"]["delta"] for lv, s in summaries.items()},
              "union_window_per_level": {lv: s.get("union_window_y") for lv, s in summaries.items()},
              "priors": ["coronal planes only (tilt not freed)",
                         *([f"dissection-region prior: atlas restricted to within {ctx.prior.get('margin_mm')} mm of "
                            f"{', '.join(ctx.prior['regions'])}"] if ctx.prior else []),
                         ctx.search_y_range()[2]],
              "sections": {}}
    if ctx.slab_prior is not None:
        report["landmark_slab"] = _slab_report(ctx, final, rows[level], anchors)
    for a in anchors:
        w = final["anchors"][a]
        r = best_row(rows[level], a)
        origin, u, v, _ = plane_geometry(ctx.atlas, 0.0, 0.0, r["y0_mm"])
        landing = landing_fractions(ctx.atlas, Similarity2D.from_row(r), origin, u, v, rib, regions)
        deciding = next((lv for lv in done if summaries[lv]["anchors"].get(a, {}).get("width_mm") == w["width_mm"]), level)
        report["sections"][a] = {
            "role": "anchor" if not w["flat"] else "passenger (flat depth profile)",
            "role_source": roles[a]["role_source"], "stacking_coordinate_y_mm": w["best_y"],
            "window_y_mm": w["window_y"], "window_contiguous": w["contiguous"],
            "deciding_level": deciding, "transform": {k: r[k] for k in ("theta_deg", "scale", "flip", "tau_x", "tau_y")},
            "score": r["score"], "contrast_in_delta": w["contrast_median_minus_best"] / final["delta"]["delta"],
            "ribbon_landing_qc_only": landing, "particles": final["particles"][a]}
    passengers_file = ctx.out / "passengers.json"
    if passengers_file.exists():
        pas = json.loads(passengers_file.read_text())
        for s, p in pas["passengers"].items():
            report["sections"][s] = {"role": "passenger", "role_source": roles.get(s, {}).get("role_source"),
                                     "failed_checks": p["role_reason"], "stacking_coordinate_y_mm": pas["host_best_y"],
                                     "window_y_mm": p["inherited_window_y"], "deciding_level": f"inherited from {pas['host_anchor']}",
                                     "transform": f"{pas['host_anchor']}'s atlas pose composed with the local-frame edge",
                                     "outline_fit_to_host": p["direct_outline_fit_to_host"]["score"],
                                     "outline_fit_displacement_mm": p["direct_outline_fit_to_host"]["displacement_from_composed_mm"]}
    # Weak ordering: pairwise tissue-outline scores in the local frame.
    parts = {s: ctx.particles(s, ctx.levels[-1]) for s in ctx.stains}
    pair = {}
    for s in ctx.stains:
        scorer = FootprintScorer(parts[s][0], parts[s][1].sum(1, keepdims=True), ctx.sigmas)
        pair[s] = {o: scorer.score(parts[o][0], parts[o][1].sum(1, keepdims=True)) for o in ctx.stains if o != s}
    report["pairwise_outline_scores"] = pair
    report["per_slice_z"] = ("not estimated: section order and spacing are below the atlas window; "
                             "pairwise outline scores give at most a weak order")

    level_names = {lv: (f"~{summaries[lv]['particles'][anchors[0]]} particles" if lv in ctx.compression_levels
                        else f"full {lv} particles") for lv in done}
    depth_profiles_figure(done, anchors, summaries, rows, level_names, ctx.step, ctx.slab_prior,
                          ctx.out / "depth_profiles.png")
    x_range, z_range = ctx.figure_ranges()
    panels = []
    for a in anchors:
        r = best_row(rows[level], a)
        flags = search_boundary_flags(final, a, float(r["scale"]), ctx.scale_bounds, ctx.step)
        report["sections"][a]["flags"] = flags
        panels.append(ctx.pose_panel(a, r["y0_mm"], Similarity2D.from_row(r), x_range, z_range,
                                     f"{a} best pose, coronal y {r['y0_mm']:+.1f} mm", flags))
    (ctx.out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    poses_figure(panels, len(anchors), regions + list(ctx.qc_tables), bool(ctx.prior), ctx.out / "best_poses.png")
    lines = _report_markdown(ctx, report, level)
    (ctx.out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return report


# --------------------------------------------------------------------------------------
# Stage: candidates (unconstrained vs admissible placement, data only)
# --------------------------------------------------------------------------------------

def _mm(value) -> str:
    return "n/a" if value is None else f"{value:.1f} mm"


def run_candidates(ctx: StackContext) -> dict:
    """Compare, per anchor, the unconstrained best and the admissible best placement.

    Both are read from the existing profiles at the deepest level run in both folders (the
    admissible run = this run's priors).  The *data-only* score is the stacking score against
    the full, uncropped coronal atlas plane, so the two placements share one objective; the
    admissible pose is also refined locally on that objective (does the data keep it there?).
    The prior enters only as admissibility (AP slab interval, atlas support); its effect on the
    score is the admissible search score minus the data-only score, reported separately.
    Diagnostics (no thresholds decide anything): ribbon landing, distance of the QNP pia to the
    atlas outer surface, of the QNP gray/white boundary to the atlas one, of the ribbon to the
    admissible gray matter, the share of section tissue inside the admissible support, pose,
    and a local pitch x yaw check (the search is coronal).
    """

    from section_mapping.alignment_gpu import pad_measures, refine_batch
    from section_mapping.atlas import landing_fractions, plane_geometry, plane_measure, plane_particles, reslice_labels
    from section_mapping.stacking import (best_row, boundary_distance_map, depth_inside_hull, distance_summary,
                                          gray_white_boundary_points, grid_lookup, placement_status,
                                          placement_verdict, region_distance_map, same_placement,
                                          search_boundary_flags)
    from visualization.figures_mapping import poses_figure

    if ctx.out == ctx.base_out:
        raise SystemExit("--stage candidates compares an admissible run (--region-prior and/or --landmark-slab) "
                         "with the unconstrained one")
    common = [lv for lv in ctx.profiled_levels() if lv in ctx.profiled_levels(ctx.base_out)]
    if not common:
        raise SystemExit("profile both the unconstrained and the admissible run first")
    level = common[-1]
    runs = {"unconstrained": ctx.base_out, "admissible": ctx.out}
    summaries = {k: json.loads((p / f"profile_{level}.json").read_text()) for k, p in runs.items()}
    rows = {k: read_rows(p / f"profile_{level}.csv") for k, p in runs.items()}
    delta = summaries["unconstrained"]["delta"]["delta"]  # data-objective noise (uncropped atlas)
    ccfg = ctx.cfg.get("candidates", {})
    steps = int(ccfg.get("refine_steps", 200))
    tilts = [float(t) for t in ccfg.get("tilt_deg", (-10, -5, 0, 5, 10))]
    step = ctx.step
    atlas, res = ctx.atlas, 0.5
    extra = ctx.qc_tables
    atlas_ext = {**atlas, "fine": {**atlas["fine"], **extra}}
    land_names = ctx.regions + list(FINE_ORDER_TAIL) + list(extra)  # later names win
    adm_regions = list(ctx.prior["regions"]) if ctx.prior else list((ctx.slab_cfg or {}).get("sampling_regions", []))
    margin = float(ctx.prior.get("margin_mm", 6.0)) if ctx.prior else None
    interval = ctx.slab_prior["search_interval"]["y_mm"] if ctx.slab_prior else None
    ms, ms_stains = ctx.multistain_maps()
    rib, depth = ctx.ribbon_local()
    pia_pts, rib_pts = rib[depth <= 100], rib[:: max(1, len(rib) // 4000)]
    maps: dict[float, dict] = {}

    def plane_maps(y: float) -> dict:
        if y not in maps:
            origin, u, v, _ = plane_geometry(atlas, 0.0, 0.0, y)
            grid, axis = reslice_labels(atlas["labels"], atlas["affine"], origin, u, v, float(ctx.scan["half_width_mm"]), res)
            g = np.clip(grid, 0, len(atlas["coarse"]["white_matter"]) - 1)
            gray, white = atlas["coarse"]["gray_matter"][g], atlas["coarse"]["white_matter"][g]
            adm = np.zeros(grid.shape, bool)
            for r in adm_regions:
                adm |= atlas["fine"][r][g]
            maps[y] = {"origin": origin, "u": u, "v": v, "axis": axis,
                       "pial": boundary_distance_map(grid > 0, res),
                       "gray_white": region_distance_map(white & ndi.binary_dilation(gray), res),
                       "admissible_gray": region_distance_map(adm, res),
                       "measure": plane_measure(atlas, ctx.scan, y, None)}
        return maps[y]

    def evaluate(scorer, y, t, X, nu, gw_pts, sulcal) -> dict:
        m = plane_maps(y)
        P, mu = m["measure"]
        place = t.section_to_moving
        out = {"y0_mm": y, "pose": {"theta_deg": t.theta_deg % 360, "scale": t.scale, "flip": t.flip,
                                    "tau_mm": list(t.tau), "pitch_deg": 0.0, "yaw_deg": 0.0},
               "data_only_score": float(scorer.score_transforms([t], P, mu)[0]),
               "ribbon_landing": landing_fractions(atlas_ext, t, m["origin"], m["u"], m["v"], rib_pts, land_names),
               "qnp_pia_to_atlas_outer_surface": distance_summary(grid_lookup(place(pia_pts), m["pial"], m["axis"], res)),
               # Split: exposed pia (on the section's convex hull) vs pia lining a sulcus (> 2.5 mm inside).
               "qnp_exposed_pia_to_atlas_outer_surface": distance_summary(
                   grid_lookup(place(pia_pts[~sulcal]), m["pial"], m["axis"], res)),
               "qnp_sulcal_pia_to_atlas_outer_surface": distance_summary(
                   grid_lookup(place(pia_pts[sulcal]), m["pial"], m["axis"], res)),
               "qnp_gray_white_to_atlas_gray_white": distance_summary(
                   grid_lookup(place(gw_pts), m["gray_white"], m["axis"], res)),
               "ribbon_to_admissible_gray": {"regions": adm_regions, **distance_summary(
                   grid_lookup(place(rib_pts), m["admissible_gray"], m["axis"], res))}}
        if margin is not None:
            d = grid_lookup(place(X), m["admissible_gray"], m["axis"], res)
            out["tissue_in_admissible_support"] = float(nu.sum(1)[np.nan_to_num(d, nan=np.inf) <= margin].sum() / nu.sum())
        if interval is not None:
            out["in_slab_interval"] = bool(min(interval) - 1e-6 <= y <= max(interval) + 1e-6)
        return out

    def tilt_check(scorer, y, t) -> dict:
        spacing, aggregate = float(ctx.scan["reslice_mm"]), int(round(float(ctx.scan["scan_bin_mm"]) / float(ctx.scan["reslice_mm"])))
        grid, measures = [], []
        for pitch in tilts:
            for yaw in tilts:
                origin, u, v, _ = plane_geometry(atlas, pitch, yaw, y)
                P, nuP = plane_particles(atlas, ctx.scan, origin, u, v, spacing, aggregate, "coarse")
                keep = nuP.sum(1) > 0
                grid.append((pitch, yaw))
                measures.append((P[keep, :2], nuP[keep]))
        Pp, mup = pad_measures(measures)
        refined = refine_batch(scorer, Pp, mup, [t] * len(grid), ctx.scale_bounds, steps)
        scores = np.array([s for s, _ in refined])
        k0 = grid.index((0.0, 0.0))
        k = int(np.argmin(scores))
        return {"grid_deg": tilts, "refined_from": "the candidate pose, per tilted plane through the same centre",
                "coronal_refined_score": float(scores[k0]), "best_pitch_yaw_deg": list(grid[k]),
                "best_score": float(scores[k]),
                "gain_in_delta": float((scores[k0] - scores[k]) / delta) if delta else None}

    record = {"job": ctx.job["_name"], "level": level, "delta_data_only": delta,
              "admissible_run": str(ctx.out.relative_to(ctx.base_out.parent)),
              "admissibility": {"ap_interval_y_mm": interval, "support_regions": adm_regions if ctx.prior else None,
                                "support_margin_mm": margin,
                                "how": "the admissible search sees only atlas tissue in the support and the AP interval; "
                                       "no overlap threshold"},
              "data_only": "stacking score vs the full uncropped coronal atlas plane (same scorer, same level)",
              "anchors": {}}
    figure_rows = []
    for anchor in [a for a, w in summaries["admissible"]["anchors"].items() if not w["flat"]]:
        X, nu, _ = ctx.particles(anchor, level)
        scorer, _ = ctx.scorer(anchor, level)
        k = ms_stains.index(anchor)
        gw_pts = gray_white_boundary_points(ms["p_wm"][k], ms["tissue"][k], ms["origin_mm"], ms["px_mm"])
        sulcal = depth_inside_hull(pia_pts, X) > 2.5
        entry, poses = {}, {}
        for name in runs:
            best = best_row(rows[name], anchor)
            t = Similarity2D.from_row(best)
            P, mu = plane_maps(best["y0_mm"])["measure"]
            (ref_score, t_ref), = refine_batch(scorer, P, mu, [t], ctx.scale_bounds, steps)
            if ref_score > float(scorer.score_transforms([t], P, mu)[0]):
                t_ref = t  # refine keeps its best iterate; never worse than the start
            e = {"search_score": best["score"],
                 "search_objective": "uncropped atlas" if name == "unconstrained" else "admissible (cropped) atlas",
                 "search_pose": evaluate(scorer, best["y0_mm"], t, X, nu, gw_pts, sulcal),
                 "refined_on_full_atlas": evaluate(scorer, best["y0_mm"], t_ref, X, nu, gw_pts, sulcal),
                 "flags": search_boundary_flags(summaries[name], anchor, float(best["scale"]), ctx.scale_bounds, step)}
            e["refined_on_full_atlas"]["tilt_check"] = tilt_check(scorer, best["y0_mm"], t_ref)
            # Prior's share of the admissible score: search score (cropped) minus data-only score.
            e["prior_effect_on_score"] = best["score"] - e["search_pose"]["data_only_score"]
            entry[name] = e
            poses[name] = (best, t, t_ref)
        bu, tu, _ = poses["unconstrained"]
        ba, ta, _ = poses["admissible"]
        wu, wa = plane_maps(bu["y0_mm"]), plane_maps(ba["y0_mm"])
        world = lambda m, t: m["origin"] + t.section_to_moving(rib_pts)[:, :1] * m["u"] + t.section_to_moving(rib_pts)[:, 1:2] * m["v"]
        shift = float(np.median(np.linalg.norm(world(wu, tu) - world(wa, ta), axis=1)))
        same = same_placement(bu["y0_mm"], ba["y0_mm"], shift, step)
        gap = (entry["admissible"]["refined_on_full_atlas"]["data_only_score"]
               - entry["unconstrained"]["refined_on_full_atlas"]["data_only_score"])
        entry["comparison"] = {"ribbon_shift_between_placements_mm": shift, "same_placement": bool(same),
                               "data_only_gap": gap, "data_only_gap_in_delta": gap / delta if delta else None,
                               "status": placement_status(same, gap / delta if delta else None)}
        record["anchors"][anchor] = entry
        figure_rows.append((anchor, poses, entry))
        print(f"{anchor}: unconstrained y {bu['y0_mm']:+.1f} / admissible y {ba['y0_mm']:+.1f}; data-only gap "
              f"{gap:+.4f} ({gap / delta:+.1f} delta); {entry['comparison']['status']}", flush=True)

    record["verdict"] = placement_verdict(
        [summaries["admissible"]["anchors"][a]["window_y"] for a in record["anchors"]],
        [e["comparison"]["status"].split(":")[0] for e in record["anchors"].values()],
        {a: e["admissible"]["flags"] for a, e in record["anchors"].items() if e["admissible"]["flags"]})
    verdict, reasons = record["verdict"]["summary"], record["verdict"]["reasons"]
    (ctx.out / "candidates.json").write_text(json.dumps(record, indent=2, default=float) + "\n", encoding="utf-8")

    # Figure: per anchor, unconstrained | admissible | admissible refined on the full atlas.
    x_range, z_range = ctx.figure_ranges()
    target = ctx.regions[0]
    panels = []
    for anchor, poses, entry in figure_rows:
        for j, (label, run, t, which) in enumerate(
                [("unconstrained best", "unconstrained", poses["unconstrained"][1], "search_pose"),
                 ("admissible best (search)", "admissible", poses["admissible"][1], "search_pose"),
                 ("admissible, refined on full atlas", "admissible", poses["admissible"][2], "refined_on_full_atlas")]):
            row = poses[run][0]
            ev = entry[run][which]
            flags = (entry[run]["flags"] if which == "search_pose"
                     else search_boundary_flags(summaries[run], anchor, t.scale, ctx.scale_bounds, step))
            land = ev["ribbon_landing"]
            shown = [n for n in (target, "STG", "ITG", "temporal_plane", "insula") if n in land]
            lines = [f"{anchor} · {label} · y {row['y0_mm']:+.1f} mm",
                     f"data-only {ev['data_only_score']:.4f} · θ {ev['pose']['theta_deg']:.0f}° · s {ev['pose']['scale']:.3f}"
                     f" · flip {ev['pose']['flip']}",
                     "ribbon: " + ", ".join(f"{n.replace('temporal_plane', 'temp. plane')} {land[n]:.2f}" for n in shown),
                     f"pia→atlas surface p50: exposed {_mm(ev['qnp_exposed_pia_to_atlas_outer_surface']['p50_mm'])}, "
                     f"sulcal {_mm(ev['qnp_sulcal_pia_to_atlas_outer_surface']['p50_mm'])} · "
                     f"gray/white p50 {_mm(ev['qnp_gray_white_to_atlas_gray_white']['p50_mm'])}"]
            if j == 2:
                lines.append(entry["comparison"]["status"].split(":")[0]
                             + f" (data-only gap {entry['comparison']['data_only_gap_in_delta']:+.1f} δ)")
            panels.append(ctx.pose_panel(anchor, row["y0_mm"], t, x_range, z_range, "\n".join(lines), flags, extra))
    poses_figure(panels, 3, ctx.regions + list(extra) + list(FINE_ORDER_TAIL), bool(ctx.prior),
                 ctx.out / "candidates.png", panel_size=(6.0, 6.0), title_size=8, title_loc="left", dpi=120,
                 suptitle=f"{ctx.job['_name']}, level {level}: data-only placements vs admissible placement. "
                          f"Verdict: {verdict}" + (f" ({'; '.join(reasons)})" if reasons else ""))
    print(json.dumps(record["verdict"], indent=1))
    return record


# --------------------------------------------------------------------------------------
# Stage: landmark
# --------------------------------------------------------------------------------------

def run_landmark(ctx: StackContext) -> dict:
    """Landmark, candidate slabs (``landmark_slab.json``) and the anatomical figures.

    Stacking windows are drawn on the sagittal view when a stacking report exists (this prior's
    folder first, else the unconstrained run).
    """

    import nibabel as nib
    from visualization.figures_mapping import render_slab_figures

    record = ctx.slab_prior
    lm = record["landmark"]
    print(f"{lm['name']}: y {lm['y_mm']:+.2f} mm (s {lm['s_mm']:.2f}); {lm['method']}", flush=True)
    print(f"search interval y {record['search_interval']['y_mm']}; offsets {record['slab_offsets']}; "
          f"{len(record['candidates'])} candidate slabs", flush=True)
    windows, source = None, None
    for folder in (ctx.out, ctx.base_out):
        if (folder / "report.json").exists():
            report = json.loads((folder / "report.json").read_text())
            windows = {s: e["window_y_mm"] for s, e in report["sections"].items() if "anchor" == e.get("role")}
            source = str(folder.relative_to(ctx.base_out.parent))
            break
    template = nib.load(str(resolve_path(ctx.config, ctx.scan["template_nifti"])))
    if not np.allclose(template.affine, ctx.atlas["affine"]) or template.shape != ctx.atlas["labels"].shape:
        raise SystemExit("template_nifti and annotation_nifti are not on the same grid")
    paths = render_slab_figures(ctx.atlas, np.asanyarray(template.dataobj), ctx.scan, ctx.slab_cfg, record, ctx.out,
                                windows=windows, windows_source=source)
    for p in paths:
        print(f"wrote {p}")
    return record


# --------------------------------------------------------------------------------------
# Stacking -> xIV hand-off
# --------------------------------------------------------------------------------------

def export_id(section: str, plane: str, candidate: str | None) -> str:
    return f"{section}_{plane}" + (f"_{candidate}" if candidate else "")


def _host_anchor(ctx: StackContext, report: dict, section: str) -> str:
    """The anchor whose pose ``section`` takes: itself, or a passenger's host (``passengers.json``)."""

    if report["sections"].get(section, {}).get("role") == "anchor":
        return section
    path = ctx.out / "passengers.json"
    passengers = json.loads(path.read_text())["passengers"] if path.exists() else {}
    if section not in passengers:
        raise SystemExit(f"{section} is not an anchor of {ctx.out.name}/report.json and has no host in "
                         f"{path.name}; run --stage passengers (with the same prior flags) first")
    return json.loads(path.read_text())["host_anchor"]


def stacking_score(ctx: StackContext, stain: str, level: str, y: float, t: Similarity2D, prior) -> float:
    """CPU stacking score of ``t`` (section -> coronal plane at y), as the profile search defines it."""

    from section_mapping.alignment import FootprintScorer, level_sigmas
    from section_mapping.atlas import plane_measure

    X, nu, _ = ctx.particles(stain, level)
    P, mu = plane_measure(ctx.atlas, ctx.scan, y, prior)
    return FootprintScorer(X, nu, level_sigmas(ctx.sigmas, ctx.h_mm(level))).score_transform(t, P, mu)


def select_placement(ctx: StackContext, section: str, candidate: str | None) -> dict:
    """The placement to hand to xIV, and why it may be used.

    Without ``candidate`` the placement must be *eligible for automatic export*: no recorded concern
    (``placement_concerns``) and, for a run with anatomical priors, that run's own ``candidates.json``.
    Eligibility is not an established anatomical correspondence: it only means nothing recorded
    contests the placement; ``anatomical_status`` reports what the data-only comparison says.
    The pose is checked by recomputing the score it was selected on (direction and composition).
    """

    from section_mapping.stacking import placement_concerns, candidate_pose

    path = ctx.out / "report.json"
    if not path.exists():
        raise SystemExit(f"no {ctx.out.name}/report.json: run --stage report (with the same prior flags) first")
    report = json.loads(path.read_text())
    host = _host_anchor(ctx, report, section)
    verdicts = {f.parent.name: json.loads(f.read_text())["verdict"] for f in sorted(ctx.base_out.glob("*/candidates.json"))}
    concerns = placement_concerns(report, host, verdicts)
    prior_run = ctx.out != ctx.base_out
    run = ctx.out.name if prior_run else "unconstrained run"
    own = ctx.out / "candidates.json"
    if prior_run and not own.exists():
        concerns.append(f"{run} uses anatomical priors but has no candidates.json (run --stage candidates with the "
                        "same prior flags): the prior's effect on the placement is unassessed")
    if prior_run and own.exists():
        anatomical_status = f"data-only comparison ({run}/candidates.json): {json.loads(own.read_text())['verdict']['summary']}"
    elif verdicts:
        anatomical_status = "unconstrained placement; prior runs: " + "; ".join(
            f"{k}: {v['summary']}" for k, v in sorted(verdicts.items()))
    else:
        anatomical_status = "not assessed: no data-only comparison with an anatomical prior"
    if candidate is None and concerns:
        raise SystemExit(f"{section} (pose of {host}, {run}) is not eligible for automatic export:\n  - "
                         + "\n  - ".join(concerns)
                         + "\nChoose it explicitly with --candidate report (this run's report pose) or, with the prior "
                           "flags of a run that has candidates.json, --candidate unconstrained|admissible|admissible-refined.")
    if candidate in (None, "report"):
        entry = report["sections"][host]
        y, t = float(entry["stacking_coordinate_y_mm"]), Similarity2D.from_row(entry["transform"])
        level, prior, expected, window = report["final_level"], ctx.prior, float(entry["score"]), entry["window_y_mm"]
        source = f"{ctx.out.name}/report.json: {host}'s best pose at level {level}"
    else:
        cfile = ctx.out / "candidates.json"
        if not cfile.exists():
            raise SystemExit(f"--candidate {candidate} needs {ctx.out.name}/candidates.json: pass the prior flags of "
                             "the admissible run and run --stage candidates there")
        cand = json.loads(cfile.read_text())
        y, t, expected = candidate_pose(cand, host, candidate)
        level, prior = cand["level"], None  # candidates.json scores are data-only (uncropped atlas)
        run = ctx.base_out if candidate == "unconstrained" else ctx.out
        window = json.loads((run / "report.json").read_text())["sections"][host]["window_y_mm"]
        source = f"{ctx.out.name}/candidates.json: {host}, {candidate} placement at level {level}"
    score = stacking_score(ctx, host, level, y, t, prior)
    if abs(score - expected) > 1e-4:
        raise RuntimeError(f"{host}: the selected pose scores {score:.6f}, not its recorded {expected:.6f}; "
                           "the pose or plane does not match the stacking run")
    return {"host": host, "y_mm": y, "transform": t, "window_y_mm": window, "source": source,
            "candidate": candidate, "concerns": concerns, "explicit": candidate is not None,
            "anatomical_status": anatomical_status,
            "score_check": {"level": level, "recorded": expected, "recomputed": score,
                            "objective": "admissible search (run priors)" if prior else "data only (uncropped atlas)"}}


def run_export_stacking(ctx: StackContext, section: str, candidate: str | None) -> dict:
    """xIV inputs for ``section`` at its stacking coordinate, written by the shared export writer.

    Target: the section's own full-level particles in the QNP local frame (same representation as
    the consensus target).  Frames: slide --M--> QNP local --t--> atlas plane, with t the host
    anchor's pose (a passenger is in the same local frame through its alignment edge M).
    """

    from preprocess.qnp_particles import section_annotation_masses
    from qnp_atlas_planes import write_xiv_inputs
    from section_mapping.atlas import plane_id

    if section not in ctx.stains:
        raise SystemExit(f"--section must be one of {ctx.stains}")
    sel = select_placement(ctx, section, candidate)
    t, y, level = sel["transform"], sel["y_mm"], ctx.levels[-1]
    X, nu, meta = ctx.particles(section, level, "local")
    Xn, nun, _ = ctx.particles(section, level, "native")
    M = ctx.slide_to_local(section)
    if abs(abs(np.linalg.det(M[:2, :2])) - 1) > 1e-6:
        raise RuntimeError(f"{section}: slide -> local is not rigid; areas would change")
    if not (np.allclose(Xn @ M[:2, :2].T + M[:2, 2], X, atol=1e-4) and np.array_equal(nu, nun)):
        raise RuntimeError(f"{section}: local particles are not the rigidly moved slide particles")
    A, tau = t.matrix, np.asarray(t.tau)
    slide_to_plane = np.eye(3)
    slide_to_plane[:2, :2], slide_to_plane[:2, 2] = A @ M[:2, :2], A @ (M[:2, 2] - tau)
    if not np.allclose(Xn @ slide_to_plane[:2, :2].T + slide_to_plane[:2, 2], t.section_to_moving(X), atol=1e-4):
        raise RuntimeError(f"{section}: slide -> local -> plane composition is inconsistent")
    export = ctx.scan["export"]
    annotation = None
    if export.get("annotation_maps"):
        if section in ctx.job.get("annotation_misfit", []):
            annotation = (None, {"status": "annotation_misfit",
                                 "rule": "the slide's HALO layers do not fit its own tissue (inventory)"})
        else:
            maps = dict(np.load(ctx.particle_dir / f"{ctx.stems[section]}_maps_32um.npz"))
            annotation = section_annotation_masses(maps, Xn, nun)
    pid = plane_id(0.0, 0.0, y)
    eid = export_id(section, pid, candidate)
    run = ctx.out.name if ctx.out != ctx.base_out else "unconstrained"
    folder = resolve_path(ctx.config, export["stacking_dir"]) / run / eid
    paths = {"source_npz": folder / "ahrav2_tissue_plane.npz", "target_npz": folder / f"qnp_{section}_{level}_in_plane.npz",
             "qc_points_npz": folder / "qnp_ribbon_points_in_plane.npz"}
    role = "anchor" if sel["host"] == section else f"passenger: inherits {sel['host']}'s pose"
    handoff = {"section": section, "role": role, "host_anchor": sel["host"], "run": run,
               "stacking_coordinate_y_mm": y, "window_y_mm": sel["window_y_mm"], "pose_source": sel["source"],
               "selection": ("explicit --candidate " + sel["candidate"]) if sel["explicit"]
               else "automatic (eligible: no recorded concern)",
               "anatomical_correspondence": sel["anatomical_status"],
               "eligibility_is_not_correspondence": "automatic eligibility means nothing recorded contests the "
                                                    "placement; it does not establish anatomical correspondence "
                                                    "(population atlas, consistency checks only)",
               "concerns_at_selection": sel["concerns"], "score_check": sel["score_check"],
               "frames": "slide mm --slide_to_local--> qnp_local_mm --similarity_qnp_local_to_plane--> atlas plane mm",
               "slide_to_local": M.tolist(), "slide_to_plane": slide_to_plane.tolist(),
               "mass_check": {"section_area_mm2_qnp": float(nu.sum()), "scale": t.scale,
                              "target_area_mm2_plane_units": float(nu.sum() * t.scale ** 2)}}
    if sel["host"] != section:
        handoff["note"] = ("a passenger is not warped on its own evidence: fit the host anchor and carry this section "
                           "through that map, or fit it with transform: rigid")
    return write_xiv_inputs(ctx.config, ctx.scan, ctx.atlas, (0.0, 0.0, y), t, (X, nu, meta, ctx.particle_path(section, level)),
                            annotation, paths, ctx.out / f"export_{eid}.json",
                            plane_status=f"coronal plane at the stacking coordinate ({sel['source']})",
                            uncertainty=f"stacking window y {sel['window_y_mm']} (within delta); {handoff['selection']}",
                            record_extra={"stacking_handoff": handoff})


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

def run_stack(ctx: StackContext, args) -> None:
    """Every stacking stage in order; existing compressed archives are kept."""

    run_roles(ctx)
    run_compress(ctx, args.overwrite)
    run_slabs(ctx)
    for level in ctx.levels:
        run_profile(ctx, level, args.families or bool(ctx.cfg.get("families", False)))
    run_passengers(ctx)
    run_report(ctx)


def main(argv=None) -> None:
    args = parse_args(argv)
    config = load_config(args.config)
    job = resolve_qnp_job(config, args.job)
    if args.stage in ("export", "evaluate") and args.section:
        ctx = StackContext(config, job, region_prior=args.region_prior, landmark_slab=args.landmark_slab)
        if args.stage == "export":
            run_export_stacking(ctx, args.section, args.candidate)
            return
        import qnp_atlas_planes

        if not (args.plane and args.fit_dir):
            raise SystemExit("--stage evaluate --section needs --plane (the exported plane id) and --fit-dir")
        qnp_atlas_planes.run_evaluate(args, config, job, job["atlas_scan"], ctx.out,
                                      record_file=ctx.out / f"export_{export_id(args.section, args.plane, args.candidate)}.json")
        return
    if args.stage in PLANE_STAGES:
        import qnp_atlas_planes

        qnp_atlas_planes.run_stage(args, config, job)
        return
    ctx = StackContext(config, job, region_prior=args.region_prior,
                       landmark_slab=args.landmark_slab or args.stage == "landmark")
    if args.stage == "profile":
        level = args.level or next((lv for lv in ctx.levels if not (ctx.out / f"profile_{lv}.json").exists()), None)
        if level not in ctx.levels:
            raise SystemExit(f"--level must be one of {ctx.levels} (all profiled already?)")
        run_profile(ctx, level, args.families or bool(ctx.cfg.get("families", False)))
    elif args.stage == "compress":
        run_compress(ctx, args.overwrite)
    elif args.stage == "stack":
        run_stack(ctx, args)
    else:
        {"roles": run_roles, "slabs": run_slabs, "passengers": run_passengers, "report": run_report,
         "landmark": run_landmark, "candidates": run_candidates}[args.stage](ctx)


if __name__ == "__main__":
    main()
