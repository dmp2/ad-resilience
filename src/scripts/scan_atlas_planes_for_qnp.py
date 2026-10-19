#!/usr/bin/env python3
"""Profile where a QNP section could sit in the AHRAv2 atlas (gate 4), and export one plane.

The atlas is *resliced* at candidate planes (coronal, tilted by pitch about x and yaw
about z, through the left-MTG centroid shifted to y0) as gray/white tissue classes:
AHRAv2 gyral labels -> gray matter, forebrain white-matter labels -> white matter,
ventricles dropped.  The correspondence QNP gray/white <-> atlas gray/white is fixed in
advance, so scores share a feature space and no cross-modal law is learned here.

For each plane and each representation (the structural consensus as gray/white or as
tissue only, and each structural stain alone), a 2-D similarity (rotation, reflection,
isotropic scale, translation) is fitted by ``section_alignment.search`` and scored with
the fixed-scale footprint-restricted varifold score (0 = identical, 1 = no overlap).
Scores share the QNP reference measure, so they are comparable across planes.

Stages
------
``scan``       fit every (representation, plane); appends to ``plane_scores.csv`` and
               resumes.  Each row also records where the NeuN HALO cortical ribbon (not
               used by the fit) lands among atlas labels.
``summarize``  profiles, the stain-replicate noise scale, confidence sets, and the
               gray/white-vs-tissue-only comparison -> ``summary.json`` and figures.
``export``     write xIV inputs for the best plane of the primary representation: atlas
               plane particles (0.5 mm, MNI plane coordinates), the QNP consensus mapped
               into that plane by the fitted similarity, and ribbon QC points.
``graph``      write ``transform_graph.json``: slides -> QNP local frame -> each exported
               atlas-plane hypothesis (rigid scan + xIV fit, if run) -> MNI.
``evaluate``   after the xIV fit of an exported ``--plane``, map the pulled-back ribbon points to MNI and compare
               label landing with the rigid placement; atlas-correspondence figure.

Low scores are not anatomical validation; see ``summary.json`` for what is and is not
discriminated.

Usage::

    PYTHONPATH=src:src/config_setup:src/preprocess python src/scripts/scan_atlas_planes_for_qnp.py \
        --config configs/project_config.yaml --job H20.33.034_MTG --stage scan
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np

from config_setup.project_config import load_config, resolve_path

FINE_ORDER_TAIL = ("other_gray_matter", "white_matter")
LANDING_CLASSES = ("outside", "dropped")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--job", required=True, help="Name under qnp_section_inputs (uses its atlas_scan block).")
    parser.add_argument("--stage", required=True, choices=["scan", "summarize", "export", "evaluate", "graph"])
    parser.add_argument("--representation", action="append", help="Restrict --stage scan to these representations.")
    parser.add_argument("--plane", help="Export this plane id (pitch_yaw_y0) instead of the best.")
    parser.add_argument("--fit-dir", help="--stage evaluate: the xIV init_* directory.")
    parser.add_argument("--dense", action="store_true",
                        help="--stage scan/summarize: re-search the planes in atlas_scan.dense_search with its "
                             "settings, writing plane_scores_dense.csv (search-robustness check).")
    return parser.parse_args()


# --------------------------------------------------------------------------------------
# Shared setup
# --------------------------------------------------------------------------------------

def atlas_setup(config: dict, scan: dict) -> dict[str, Any]:
    import csv as _csv

    import nibabel as nib
    from preprocess.section_alignment import lookup_tables

    image = nib.load(str(resolve_path(config, scan["annotation_nifti"])))
    labels = np.asanyarray(image.dataobj).astype(np.int32)
    affine = np.asarray(image.affine, float)
    with open(resolve_path(config, scan["terminology_csv"]), newline="", encoding="utf-8") as stream:
        names = {int(r["annotation_value"]): r["name"] for r in _csv.DictReader(stream) if r["annotation_value"].strip()}
    failures = []
    for region, spec in scan["fine_regions"].items():
        for code in spec["codes"]:
            if names.get(int(code)) != spec["expected_name"]:
                failures.append(f"{region}: code {code} decodes to {names.get(int(code))!r}, expected {spec['expected_name']!r}")
    if failures:
        raise SystemExit("Terminology check failed:\n  " + "\n  ".join(failures))
    coarse, fine = lookup_tables(int(labels.max()), scan["white_matter_codes"], scan["drop_codes"],
                                 {k: v["codes"] for k, v in scan["fine_regions"].items()})
    hemisphere_sign = -1.0 if scan.get("hemisphere", "left") == "left" else 1.0
    ijk = np.argwhere(fine[next(iter(scan["fine_regions"]))][labels])
    world = ijk @ affine[:3, :3].T + affine[:3, 3]
    world = world[np.sign(world[:, 0]) == hemisphere_sign]
    centroid = world.mean(0)
    return {"labels": labels, "affine": affine, "coarse": coarse, "fine": fine, "names": names,
            "centroid": centroid, "region_extent_y": [float(world[:, 1].min()), float(world[:, 1].max())]}


def plane_geometry(atlas: dict, pitch: float, yaw: float, y0: float):
    from preprocess.section_alignment import plane_basis

    u, v, n = plane_basis(pitch, yaw)
    origin = np.array([atlas["centroid"][0], y0, atlas["centroid"][2]])
    return origin, u, v, n


def plane_id(pitch: float, yaw: float, y0: float) -> str:
    return f"p{pitch:+04.0f}_w{yaw:+04.0f}_y{y0:+06.1f}"


def plane_particles(atlas: dict, scan: dict, origin, u, v, spacing: float, aggregate: int, table: str):
    from preprocess.section_alignment import label_class_masses, reslice_labels

    grid, axis = reslice_labels(atlas["labels"], atlas["affine"], origin, u, v, float(scan["half_width_mm"]), spacing)
    return label_class_masses(grid, axis, spacing, atlas[table], aggregate)


def classify_points(atlas: dict, world: np.ndarray, region_names: list[str]) -> np.ndarray:
    """Index into region_names + ('outside', 'dropped') for each world point (nearest voxel)."""

    ijk = np.rint((world - atlas["affine"][:3, 3]) @ np.linalg.inv(atlas["affine"][:3, :3]).T).astype(int)
    shape = np.array(atlas["labels"].shape)
    inside = np.all((ijk >= 0) & (ijk < shape), axis=1)
    codes = np.zeros(len(world), int)
    codes[inside] = atlas["labels"][tuple(ijk[inside].T)]
    out = np.full(len(world), len(region_names) + 1)  # dropped by default
    out[codes == 0] = len(region_names)
    for k, name in enumerate(region_names):
        table = atlas["fine"][name]
        member = np.zeros(len(world), bool)
        valid = codes < len(table)
        member[valid] = table[codes[valid]]
        out[member] = k
    return out


def landing_fractions(atlas: dict, transform, origin, u, v, points_local: np.ndarray, region_names) -> dict[str, float]:
    plane = transform.section_to_moving(points_local)
    world = origin + plane[:, :1] * u + plane[:, 1:2] * v
    index = classify_points(atlas, world, region_names)
    names = list(region_names) + list(LANDING_CLASSES)
    counts = np.bincount(index, minlength=len(names)) / max(len(index), 1)
    return dict(zip(names, counts.tolist()))


def representation_inputs(config: dict, job: dict, scan: dict) -> dict[str, dict]:
    from preprocess.particle_utils import load_particle_archive

    out = {}
    for name, spec in scan["representations"].items():
        X, nu, meta = load_particle_archive(resolve_path(config, spec["particles"]), "nu_Sub")
        if meta.get("coordinate_space") != "qnp_local_mm":
            raise ValueError(f"{name}: expected qnp_local_mm particles, got {meta.get('coordinate_space')!r}")
        out[name] = {"X": X[:, :2].astype(np.float64), "nu": nu.astype(np.float64), "mode": spec["features"]}
    return out


# --------------------------------------------------------------------------------------
# Stage: scan
# --------------------------------------------------------------------------------------

def effective_scan(scan: dict, dense: bool) -> dict:
    """The scan settings, with atlas_scan.dense_search overrides when ``dense``."""

    if not dense:
        return scan
    overrides = {k: v for k, v in scan["dense_search"].items() if k not in ("planes", "representations")}
    return {**scan, **overrides}


def scores_table(out_dir: Path, dense: bool) -> Path:
    return out_dir / ("plane_scores_dense.csv" if dense else "plane_scores.csv")


_WORKER: dict[str, Any] = {}


def _init_worker(config_path: str, job_name: str, names: list[str], dense: bool = False) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from preprocess.section_alignment import FootprintScorer

    config = load_config(config_path)
    job = config["qnp_section_inputs"][job_name]
    scan = effective_scan(job["atlas_scan"], dense)
    reps = {k: v for k, v in representation_inputs(config, job, scan).items() if k in names}
    sigmas = [float(s) for s in scan.get("score_sigmas_mm", [2.0, 1.0, 0.5])]
    scorers = {}
    for name, rep in reps.items():
        nu = rep["nu"].sum(1, keepdims=True) if rep["mode"] == "tissue" else rep["nu"]
        scorers[name] = FootprintScorer(rep["X"], nu, sigmas)
    with np.load(resolve_path(config, scan["ribbon_points"])) as archive:
        ribbon = archive["X"][:, :2].astype(np.float64)
    _WORKER.update(config=config, scan=scan, atlas=atlas_setup(config, scan), reps=reps, scorers=scorers,
                   ribbon=ribbon[:: int(scan.get("ribbon_subsample", 10))])


def _fit_plane(task: tuple[float, float, float]) -> list[dict[str, Any]]:
    from preprocess.section_alignment import search

    pitch, yaw, y0 = task
    scan, atlas = _WORKER["scan"], _WORKER["atlas"]
    origin, u, v, n = plane_geometry(atlas, pitch, yaw, y0)
    spacing, aggregate = float(scan["reslice_mm"]), int(round(float(scan["scan_bin_mm"]) / float(scan["reslice_mm"])))
    P, nu_P = plane_particles(atlas, scan, origin, u, v, spacing, aggregate, "coarse")
    regions = list(scan["fine_regions"]) + list(FINE_ORDER_TAIL)
    rows = []
    for name, rep in _WORKER["reps"].items():
        base = {"representation": name, "plane_id": plane_id(pitch, yaw, y0), "pitch_deg": pitch, "yaw_deg": yaw,
                "y0_mm": y0, "atlas_particles": int(len(P)), "atlas_area_mm2": float(nu_P.sum())}
        if len(P) < 10:
            rows.append({**base, "score": 1.0, "status": "empty_plane"})
            continue
        mu = nu_P.sum(1, keepdims=True) if rep["mode"] == "tissue" else nu_P
        ranked = search(_WORKER["scorers"][name], P[:, :2], mu,
                        thetas=np.arange(0, 360, float(scan.get("theta_step_deg", 15))),
                        scales=[float(s) for s in scan["scale_grid"]], flips=[False, True],
                        scale_bounds=tuple(float(s) for s in scan["scale_bounds"]),
                        h=float(scan.get("fft_grid_mm", 1.0)), blur_mm=float(scan.get("fft_blur_mm", 1.5)),
                        refine_top=int(scan.get("refine_top", 2)), max_evals=int(scan.get("max_evals", 150)))
        score, t = ranked[0]
        other = [s for s, c in ranked if c.flip != t.flip]
        landing = landing_fractions(atlas, t, origin, u, v, _WORKER["ribbon"], regions)
        rows.append({**base, "score": score, "status": "ok", "theta_deg": t.theta_deg, "scale": t.scale,
                     "flip": t.flip, "tau_x": t.tau[0], "tau_y": t.tau[1],
                     "best_other_flip_score": min(other) if other else None,
                     **{f"ribbon_on_{k}": val for k, val in landing.items()}})
    return rows


def run_scan(args, config, job, scan, out_dir: Path) -> None:
    if args.dense:
        dense = scan["dense_search"]
        names = args.representation or list(dense["representations"])
        planes = [(float(p), float(w), float(y)) for p, w, ys in dense["planes"] for y in ys]
    else:
        names = args.representation or list(scan["representations"])
        planes = list(product([float(v) for v in scan["pitch_deg"]], [float(v) for v in scan["yaw_deg"]],
                              np.arange(float(scan["y_range_mm"][0]), float(scan["y_range_mm"][1]) + 1e-9,
                                        float(scan["y_step_mm"])).round(2).tolist()))
    table = scores_table(out_dir, args.dense)
    done = set()
    if table.exists():
        with open(table, newline="") as stream:
            done = {(r["representation"], r["plane_id"]) for r in csv.DictReader(stream)}
    todo = [p for p in planes if any((n, plane_id(*p)) not in done for n in names)]
    print(f"{len(planes)} planes x {len(names)} representations; {len(todo)} planes to fit", flush=True)
    regions = list(scan["fine_regions"]) + list(FINE_ORDER_TAIL)
    fieldnames = (["representation", "plane_id", "pitch_deg", "yaw_deg", "y0_mm", "atlas_particles", "atlas_area_mm2",
                   "score", "status", "theta_deg", "scale", "flip", "tau_x", "tau_y", "best_other_flip_score"]
                  + [f"ribbon_on_{k}" for k in regions + list(LANDING_CLASSES)])
    with ProcessPoolExecutor(int(scan.get("workers", 8)), initializer=_init_worker,
                             initargs=(args.config, args.job, names, args.dense)) as pool:
        for count, rows in enumerate(pool.map(_fit_plane, todo, chunksize=1), start=1):
            rows = [r for r in rows if (r["representation"], r["plane_id"]) not in done]
            with open(table, "a", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
                if stream.tell() == 0:
                    writer.writeheader()
                writer.writerows(rows)
            if count % 10 == 0 or count == len(todo):
                best = min(rows, key=lambda r: r["score"])
                print(f"{count}/{len(todo)} planes; last {best['plane_id']} best {best['representation']} "
                      f"{best['score']:.3f}", flush=True)


# --------------------------------------------------------------------------------------
# Stage: summarize
# --------------------------------------------------------------------------------------

def load_scores(out_dir: Path) -> dict[str, list[dict]]:
    by_rep: dict[str, list[dict]] = {}
    with open(out_dir / "plane_scores.csv", newline="") as stream:
        for row in csv.DictReader(stream):
            for key, value in list(row.items()):
                try:
                    row[key] = float(value) if key not in ("representation", "plane_id", "status", "flip") else value
                except ValueError:
                    pass
            by_rep.setdefault(row["representation"], []).append(row)
    return by_rep


def run_summarize(config, job, scan, out_dir: Path) -> dict[str, Any]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_rep = load_scores(out_dir)
    primary = scan["primary_representation"]
    replicates = [r for r in scan.get("replicate_representations", []) if r in by_rep]
    regions = list(scan["fine_regions"]) + list(FINE_ORDER_TAIL)
    target_region = next(iter(scan["fine_regions"]))

    # Stain-replicate noise scale: at each plane, each stain's score relative to its own
    # best; the SD across stains measures how much a score difference depends on which
    # stain represented the section.  delta = its median over the primary top planes.
    index = {r: {row["plane_id"]: row for row in by_rep[r]} for r in by_rep}
    planes = sorted(index[primary], key=lambda p: index[primary][p]["score"])
    delta = None
    if len(replicates) >= 2:
        rel = {r: {p: row["score"] - min(x["score"] for x in by_rep[r]) for p, row in index[r].items()} for r in replicates}
        top = planes[: int(scan.get("noise_top_planes", 20))]
        sds = [float(np.std([rel[r][p] for r in replicates if p in rel[r]], ddof=1)) for p in top]
        delta = float(np.median(sds))

    summary: dict[str, Any] = {"primary_representation": primary, "replicates": replicates,
                               "stain_replicate_noise_delta": delta,
                               "delta_definition": "median over the primary's top planes of the across-stain SD of "
                                                   "(stain score - that stain's best score)", "representations": {}}
    for rep, rows in by_rep.items():
        ok = [r for r in rows if r.get("status") == "ok"]
        ok.sort(key=lambda r: r["score"])
        best = ok[0]
        scores = np.array([r["score"] for r in ok])
        entry = {"best": {k: best[k] for k in best if not k.startswith("ribbon_on_")},
                 "best_ribbon_landing": {k[10:]: best[k] for k in best if k.startswith("ribbon_on_")},
                 "score_quantiles": dict(zip(["min", "p10", "median", "max"],
                                             np.percentile(scores, [0, 10, 50, 100]).tolist())),
                 "contrast_median_minus_best": float(np.median(scores) - scores[0]),
                 "top5": [{k: r[k] for k in ("plane_id", "score", "scale", "theta_deg", "flip",
                                             f"ribbon_on_{target_region}")} for r in ok[:5]]}
        if delta is not None:
            members = [r for r in ok if r["score"] <= scores[0] + delta]
            entry["confidence_set"] = {
                "rule": "score <= best + delta (stain-replicate noise)", "size": len(members),
                "fraction_of_planes": len(members) / len(ok),
                "y0_range_mm": [min(r["y0_mm"] for r in members), max(r["y0_mm"] for r in members)],
                "pitch_values": sorted({r["pitch_deg"] for r in members}),
                "yaw_values": sorted({r["yaw_deg"] for r in members}),
                "flip_values": sorted({r["flip"] for r in members}),
                "scale_range": [min(r["scale"] for r in members), max(r["scale"] for r in members)],
                f"ribbon_on_{target_region}_range": [min(r[f"ribbon_on_{target_region}"] for r in members),
                                                     max(r[f"ribbon_on_{target_region}"] for r in members)],
                "contrast_in_delta_units": entry["contrast_median_minus_best"] / delta if delta else None,
            }
        summary["representations"][rep] = entry
    if replicates:
        bests = {r: summary["representations"][r]["best"] for r in replicates}
        summary["replicate_best_planes"] = {r: {k: b[k] for k in ("plane_id", "y0_mm", "pitch_deg", "yaw_deg", "score", "flip")}
                                            for r, b in bests.items()}
    if scores_table(out_dir, True).exists():
        summary["dense_basins"] = summarize_dense(out_dir, scan, delta, target_region, plt)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    # ---- figures ----------------------------------------------------------------------
    reps = list(by_rep)
    orientations = sorted({(r["pitch_deg"], r["yaw_deg"]) for r in by_rep[primary]})
    fig, axes = plt.subplots(len(reps), 1, figsize=(11, 3.2 * len(reps)), sharex=True, constrained_layout=True,
                             squeeze=False)
    cmap = plt.get_cmap("viridis", len(orientations))
    for ax, rep in zip(axes[:, 0], reps):
        for k, (pitch, yaw) in enumerate(orientations):
            rows = sorted((r for r in by_rep[rep] if r["pitch_deg"] == pitch and r["yaw_deg"] == yaw and r.get("status") == "ok"),
                          key=lambda r: r["y0_mm"])
            ax.plot([r["y0_mm"] for r in rows], [r["score"] for r in rows], color=cmap(k), lw=1.2,
                    label=f"pitch {pitch:+.0f}°, yaw {yaw:+.0f}°")
        best = summary["representations"][rep]["best"]
        ax.axhline(best["score"], color="#1a1a19", lw=0.6)
        if delta is not None:
            ax.axhspan(best["score"], best["score"] + delta, color="#eda100", alpha=0.25, lw=0,
                       label="best + stain-replicate noise δ")
        ax.set_ylabel("score (0 = identical)")
        ax.set_title(f"{rep}: best {best['plane_id']} score {best['score']:.3f}", fontsize=10)
    axes[0, 0].legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8, frameon=False)
    axes[-1, 0].set_xlabel("plane origin y0 (MNI mm, through the left-MTG centroid in x, z)")
    fig.savefig(out_dir / "score_profiles.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5.5), constrained_layout=True)
    for rep in reps:
        ok = [r for r in by_rep[rep] if r.get("status") == "ok"]
        ax.scatter([r["score"] for r in ok], [r[f"ribbon_on_{target_region}"] for r in ok], s=8, label=rep, alpha=0.6)
    ax.set_xlabel("plane score (lower = better)")
    ax.set_ylabel(f"fraction of NeuN HALO ribbon on atlas {target_region}")
    ax.set_title("Does a better score put the annotated cortex on the atlas MTG?\n(ribbon not used by the fit)", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    fig.savefig(out_dir / "score_vs_ribbon_landing.png", dpi=130)
    plt.close(fig)
    print(json.dumps({k: summary[k] for k in ("stain_replicate_noise_delta",)}, indent=2))
    for rep, entry in summary["representations"].items():
        print(rep, json.dumps({"best": entry["best"]["plane_id"], "score": round(entry["best"]["score"], 4),
                               "contrast": round(entry["contrast_median_minus_best"], 4),
                               "confidence_set": {k: entry.get("confidence_set", {}).get(k) for k in
                                                  ("size", "y0_range_mm", "pitch_values", "yaw_values", "flip_values",
                                                   f"ribbon_on_{target_region}_range")},
                               "ribbon_landing": {k: round(v, 3) for k, v in entry["best_ribbon_landing"].items()}}))
    return summary


def summarize_dense(out_dir: Path, scan: dict, delta: float | None, target_region: str, plt) -> dict[str, Any]:
    """Per orientation of the dense re-search: each representation's minimum, the
    y-width of its basin (score <= min + delta), and pose consistency inside it."""

    rows = []
    with open(scores_table(out_dir, True), newline="") as stream:
        for row in csv.DictReader(stream):
            rows.append({k: (v if k in ("representation", "plane_id", "status", "flip") else float(v))
                         for k, v in row.items() if v != ""})
    orientations = sorted({(r["pitch_deg"], r["yaw_deg"]) for r in rows})
    reps = list(dict.fromkeys(r["representation"] for r in rows))
    basins: dict[str, Any] = {}
    fig, axes = plt.subplots(1, len(orientations), figsize=(7 * len(orientations), 4.5), constrained_layout=True,
                             squeeze=False)
    for ax, (pitch, yaw) in zip(axes[0], orientations):
        key = f"pitch{pitch:+.0f}_yaw{yaw:+.0f}"
        basins[key] = {}
        for rep in reps:
            sel = sorted((r for r in rows if r["representation"] == rep and r["pitch_deg"] == pitch
                          and r["yaw_deg"] == yaw), key=lambda r: r["y0_mm"])
            best = min(sel, key=lambda r: r["score"])
            inside = [r for r in sel if delta is not None and r["score"] <= best["score"] + delta]
            basins[key][rep] = {
                "best_y0_mm": best["y0_mm"], "best_score": best["score"], "scale": best["scale"],
                "theta_deg": best["theta_deg"] % 360, "flip": best["flip"],
                f"ribbon_on_{target_region}": best[f"ribbon_on_{target_region}"],
                "ribbon_on_FuG": best.get("ribbon_on_FuG"),
                "basin_y0_range_mm": [min(r["y0_mm"] for r in inside), max(r["y0_mm"] for r in inside)] if inside else None,
            }
            ax.plot([r["y0_mm"] for r in sel], [r["score"] for r in sel], marker="o", ms=3, lw=1.2, label=rep)
        ax.set_title(f"dense re-search, pitch {pitch:+.0f}°, yaw {yaw:+.0f}°", fontsize=10)
        ax.set_xlabel("plane origin y0 (MNI mm)")
        ax.set_ylabel("score (0 = identical)")
    axes[0, 0].legend(fontsize=8, frameon=False)
    fig.savefig(out_dir / "dense_basin_profiles.png", dpi=130)
    plt.close(fig)
    return basins


# --------------------------------------------------------------------------------------
# Stage: export
# --------------------------------------------------------------------------------------

def run_export(args, config, job, scan, out_dir: Path) -> None:
    from preprocess.particle_utils import FeatureDefinition, load_particle_archive, save_particle_archive
    from preprocess.section_alignment import FootprintScorer, Similarity2D
    from preprocess.xmodmap_compat import entropy_numpy, load_xmodmap, write_vtk_xyz

    by_rep = load_scores(out_dir)
    primary = scan["primary_representation"]
    rows = [r for r in by_rep[primary] if r.get("status") == "ok"]
    row = (next(r for r in rows if r["plane_id"] == args.plane) if args.plane else min(rows, key=lambda r: r["score"]))
    t = Similarity2D(row["theta_deg"], row["scale"], row["flip"] == "True", (row["tau_x"], row["tau_y"]))
    atlas = atlas_setup(config, scan)
    origin, u, v, n = plane_geometry(atlas, row["pitch_deg"], row["yaw_deg"], row["y0_mm"])
    export = scan["export"]
    spacing = float(export["bin_mm"])
    P, nu_coarse = plane_particles(atlas, scan, origin, u, v, spacing, 1, "coarse")
    _, nu_fine = plane_particles(atlas, scan, origin, u, v, spacing, 1, "fine")
    if len(P) != len(nu_fine):
        raise RuntimeError("coarse and fine plane particles differ; check the label tables")

    target_file = resolve_path(config, export["target_particles"])
    X_q, nu_q, meta_q = load_particle_archive(target_file, "nu_Sub")
    support = FootprintScorer(X_q[:, :2], nu_q, [1.0], footprint_sigma_mm=float(export["support_sigma_mm"]))
    keep = support.footprint(t.moving_to_section(P[:, :2])) > float(export["support_min_weight"])
    regions = list(scan["fine_regions"]) + list(FINE_ORDER_TAIL)
    features = FeatureDefinition(fine_ids=tuple(range(len(regions))), fine_names=tuple(regions),
                                 coarse_names=("gray_matter", "white_matter"),
                                 fine_to_coarse=tuple(1 if r == "white_matter" else 0 for r in regions))
    plane_meta = {"origin_mm": origin.tolist(), "u_axis": u.tolist(), "v_axis": v.tolist(), "normal": n.tolist(),
                  "pitch_deg": row["pitch_deg"], "yaw_deg": row["yaw_deg"], "y0_mm": row["y0_mm"],
                  "plane_id": row["plane_id"], "reslice": "nearest-neighbour labels on a regular in-plane grid",
                  "inverse": "world = origin + plane_X[:, 0] * u_axis + plane_X[:, 1] * v_axis",
                  "status": "best plane of a discrete scan; a hypothesis, not an estimate of the cutting plane"}
    api = load_xmodmap(config)
    writer = lambda points, arrays, names, filename: write_vtk_xyz(api, points, arrays, names, filename)
    entropy = lambda nu: entropy_numpy(api, nu)
    paths = {k: resolve_path(config, export[k].format(plane_id=row["plane_id"]))
             for k in ("source_npz", "target_npz", "qc_points_npz")}
    save_particle_archive(paths["source_npz"], P[keep], nu_coarse[keep], nu_fine[keep], features, {
        "particle_role": "source_atlas_plane", "coordinate_space": "mni_icbm2009b_symmetric_plane",
        "plane": plane_meta, "atlas_annotation": str(resolve_path(config, scan["annotation_nifti"])),
        "label_grouping": {"white_matter_codes": scan["white_matter_codes"], "drop_codes": scan["drop_codes"],
                           "fine_regions": scan["fine_regions"], "gray_matter": "every other non-zero code"},
        "support_restriction": {"rule": "QNP footprint weight at the rigidly placed position > support_min_weight",
                                **{k: export[k] for k in ("support_sigma_mm", "support_min_weight")}},
        "mass_units": "mm^2 (in-plane pixel area)"}, vtk_writer=writer, entropy_function=entropy)

    Y = t.section_to_moving(X_q[:, :2])
    target_X = np.column_stack([Y, np.zeros(len(Y))])
    similarity = {**t.as_dict(), "from": "qnp_local_mm", "to": "atlas plane mm", "plane_id": row["plane_id"],
                  "uncertainty": "one member of the scan's confidence set; see summary.json"}
    save_particle_archive(paths["target_npz"], target_X, nu_q * t.scale ** 2, nu_q * t.scale ** 2,
                          FeatureDefinition((0, 1), ("gray_matter", "white_matter"), ("gray_matter", "white_matter"), (0, 1)),
                          {**meta_q, "particle_role": "target_qnp_in_atlas_plane",
                           "coordinate_space": "qnp_local_in_atlas_plane_mm", "source_particles": str(target_file),
                           "similarity_qnp_local_to_plane": similarity, "plane": plane_meta,
                           "mass_units": "mm^2 in atlas-plane units (QNP area * scale^2)"},
                          vtk_writer=writer, entropy_function=entropy)
    with np.load(resolve_path(config, scan["ribbon_points"])) as archive:
        ribbon_X, depth = archive["X"][:, :2].astype(np.float64), archive["depth_from_pia_um"]
    ribbon_plane = t.section_to_moving(ribbon_X)
    np.savez_compressed(paths["qc_points_npz"], X=np.column_stack([ribbon_plane, np.zeros(len(ribbon_plane))]).astype(np.float32),
                        depth_from_pia_um=depth, depth_units=np.asarray("um of QNP (unscaled) tissue"),
                        coordinate_space=np.asarray("qnp_local_in_atlas_plane_mm"))
    record = {"plane": plane_meta, "similarity_qnp_local_to_plane": similarity, "source_particles": int(keep.sum()),
              "source_area_mm2": float(nu_coarse[keep].sum()), "target_particles": int(len(Y)),
              "target_area_mm2_plane_units": float((nu_q * t.scale ** 2).sum()),
              "rigid_ribbon_landing": landing_fractions(atlas, t, origin, u, v, ribbon_X, regions),
              "paths": {k: str(p) for k, p in paths.items()}}
    (out_dir / f"export_{row['plane_id']}.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: record[k] for k in ("source_particles", "source_area_mm2", "target_particles",
                                              "target_area_mm2_plane_units", "rigid_ribbon_landing")}, indent=2))


# --------------------------------------------------------------------------------------
# Stage: evaluate
# --------------------------------------------------------------------------------------

def run_evaluate(args, config, job, scan, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import nibabel as nib
    from scipy import ndimage as ndi

    from preprocess.section_alignment import reslice_labels

    fit_dir = Path(args.fit_dir)
    export = json.loads((out_dir / f"export_{args.plane}.json").read_text())
    plane = export["plane"]
    origin, u, v = (np.asarray(plane[k]) for k in ("origin_mm", "u_axis", "v_axis"))
    atlas = atlas_setup(config, scan)
    regions = list(scan["fine_regions"]) + list(FINE_ORDER_TAIL)
    with np.load(fit_dir / "qc_points_pulled_back.npz") as archive:
        pulled = archive["X_plane"][:, :2].astype(np.float64)
        depth_um = archive["depth"].astype(np.float64)
    with np.load(export["paths"]["qc_points_npz"]) as archive:
        rigid = archive["X"][:, :2].astype(np.float64)
    names = regions + list(LANDING_CLASSES)
    bins = [(0, 1000), (1000, 2000), (2000, 3000), (3000, 6000)]

    def by_depth(points):
        index = classify_points(atlas, origin + points[:, :1] * u + points[:, 1:2] * v, regions)
        out = {"all": dict(zip(names, (np.bincount(index, minlength=len(names)) / len(index)).tolist()))}
        for low, high in bins:
            sel = (depth_um >= low) & (depth_um < high)
            if sel.any():
                out[f"{low}-{high}um"] = dict(zip(names, (np.bincount(index[sel], minlength=len(names)) / sel.sum()).tolist()))
        return out

    displacement = np.linalg.norm(pulled - rigid, axis=1)
    evaluation = {
        "fit_dir": str(fit_dir), "plane_id": plane["plane_id"],
        "ribbon_landing_rigid": by_depth(rigid), "ribbon_landing_after_xiv": by_depth(pulled),
        "ribbon_displacement_by_xiv_mm": dict(zip(["p50", "p90", "max"], np.percentile(displacement, [50, 90, 100]).tolist())),
        "note": "the ribbon is the NeuN HALO cortex, never seen by any fit; atlas MTG is a population-template "
                "label, so landing fractions test consistency, not ground truth",
    }
    for name in ("plausibility.json", "scores.json", "COMPLETED.json"):
        if (fit_dir / name).exists():
            evaluation[name.replace(".json", "")] = json.loads((fit_dir / name).read_text())
    (fit_dir / "anatomical_evaluation.json").write_text(json.dumps(evaluation, indent=2) + "\n", encoding="utf-8")

    # ---- atlas-correspondence figure: template + labels on the plane, QNP on top -------
    half, step = float(scan["half_width_mm"]), 0.25
    grid, axis = reslice_labels(atlas["labels"], atlas["affine"], origin, u, v, half, step)
    template = nib.load(str(resolve_path(config, scan["template_nifti"])))
    aa, bb = np.meshgrid(axis, axis, indexing="xy")
    world = origin + aa[..., None] * u + bb[..., None] * v
    ijk = (world.reshape(-1, 3) - template.affine[:3, 3]) @ np.linalg.inv(template.affine[:3, :3]).T
    t1 = ndi.map_coordinates(np.asanyarray(template.dataobj).astype(np.float32), ijk.T, order=1).reshape(aa.shape)
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.2), constrained_layout=True)
    colors = ["#eb6834", "#2a78d6", "#1baf7a", "#eda100", "#c3c2b7", "#1a1a19"]
    for ax, points, title in ((axes[0], rigid, "rigid scan placement"), (axes[1], pulled, "after xIV (pulled back)")):
        ax.imshow(t1, extent=[axis[0], axis[-1], axis[0], axis[-1]], origin="lower", cmap="gray")
        for color, name in zip(colors, regions):
            ax.contour(axis, axis, atlas["fine"][name][np.clip(grid, 0, len(atlas["fine"][name]) - 1)].astype(float),
                       [0.5], colors=[color], linewidths=0.9)
        shown = ax.scatter(points[::5, 0], points[::5, 1], c=depth_um[::5] / 1000.0, s=0.6, cmap="viridis",
                           vmin=0, vmax=3.5, linewidths=0)
        ax.set_title(f"{title}: NeuN HALO ribbon (colour = depth from pia, mm)", fontsize=10)
        ax.set_xlabel("plane u (mm)"); ax.set_ylabel("plane v (mm)"); ax.set_aspect("equal")
    fig.colorbar(shown, ax=axes, shrink=0.6, label="depth from pia (mm, QNP)")
    handles = [plt.Line2D([], [], color=c) for c in colors]
    fig.legend(handles, regions, loc="lower center", ncol=len(regions), frameon=False, fontsize=9)
    fig.suptitle(f"AHRAv2 plane {plane['plane_id']} (MNI ICBM 2009b T1, label outlines) with the QNP section", fontsize=11)
    fig.savefig(fit_dir / "atlas_correspondence.png", dpi=130)
    plt.close(fig)
    print(json.dumps({k: evaluation[k] for k in ("ribbon_displacement_by_xiv_mm",)}, indent=2))
    print("rigid:", json.dumps({k: round(v, 3) for k, v in evaluation["ribbon_landing_rigid"]["all"].items()}))
    print("xIV:  ", json.dumps({k: round(v, 3) for k, v in evaluation["ribbon_landing_after_xiv"]["all"].items()}))


def run_graph(config, job, scan, out_dir: Path) -> None:
    """Edges of the Allen <-> QNP transformation graph, with their uncertainty."""

    within = json.loads((resolve_path(config, job["alignment"]["output_dir"]) / "transforms.json").read_text())
    summary = json.loads((out_dir / "summary.json").read_text()) if (out_dir / "summary.json").exists() else {}
    nodes = {"mni_icbm2009b_symmetric": "AHRAv2 / MNI ICBM 2009b symmetric, mm",
             "qnp_local_mm": within["local_frame"]}
    edges = []
    for stain, edge in within["edges"].items():
        node = f"qnp_slide_mm:{edge['slide']}"
        nodes[node] = "slide mm of one QNP slide (x right, y down)"
        reliable = edge["structural"] or "halo_ribbon_dice_vs_reference" in edge["independent_checks"]
        edges.append({"from": node, "to": "qnp_local_mm", "type": "2-D rigid (+ reflection), scale 1",
                      "matrix_3x3": edge["slide_mm_to_local_mm"], "score": edge["score_fine"],
                      "checks": edge["independent_checks"],
                      "status": "accepted" if reliable and edge["score_fine"] < 0.25 else "unreliable (not used)"})
    hypotheses = []
    for export_file in sorted(out_dir.glob("export_*.json")):
        export = json.loads(export_file.read_text())
        plane = export["plane"]
        node = f"atlas_plane:{plane['plane_id']}"
        nodes[node] = "in-plane mm of a resliced AHRAv2 plane; world = origin + a u + b v"
        sim = export["similarity_qnp_local_to_plane"]
        fit = None
        for name, record in config.get("registrations", {}).items():
            if plane["plane_id"] in str(record.get("source_particles", "")):
                done = resolve_path(config, record["output_dir"]) / f"init_{record['reference_initialization']}"
                if (done / "COMPLETED.json").exists():
                    fit = {"registration": name, "directory": str(done),
                           "variables": str(done / "optimized_variables.pt"),
                           "transform_chain": str(done / "transform_chain.json"),
                           "evaluation": str(done / "anatomical_evaluation.json")}
        edges.append({"from": "qnp_local_mm", "to": node, "type": "2-D similarity from the plane scan, then "
                      "(optionally) xIV LDDMM on the plane", "similarity": sim, "xiv_fit": fit,
                      "status": "competing hypothesis"})
        edges.append({"from": node, "to": "mni_icbm2009b_symmetric", "type": "plane embedding (exact by definition)",
                      "origin_mm": plane["origin_mm"], "u_axis": plane["u_axis"], "v_axis": plane["v_axis"],
                      "status": "exact given the hypothesis"})
        hypotheses.append({"plane_id": plane["plane_id"], "rigid_ribbon_landing": export["rigid_ribbon_landing"]})
    graph = {
        "nodes": nodes, "edges": edges, "hypotheses": hypotheses,
        "placement_uncertainty": {
            "stain_replicate_noise_delta": summary.get("stain_replicate_noise_delta"),
            "dense_basins": summary.get("dense_basins"),
            "interpretation": "competing planes are alternative constraints, not a posterior; no hypothesis is "
                              "certified by the image data alone",
        },
        "attachable_later": {"merfish": "SEA-AD MERFISH sections of the same donor/region -> qnp_local_mm "
                                        "(no shared physical section; would be another uncertain edge)",
                             "donor_mri": "donor MRI -> mni_icbm2009b_symmetric or directly -> atlas planes"},
    }
    (out_dir / "transform_graph.json").write_text(json.dumps(graph, indent=2) + "\n", encoding="utf-8")
    print(f"{len(nodes)} nodes, {len(edges)} edges -> {out_dir / 'transform_graph.json'}")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    job = config["qnp_section_inputs"][args.job]
    scan = job["atlas_scan"]
    out_dir = resolve_path(config, scan["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.stage == "scan":
        run_scan(args, config, job, scan, out_dir)
    elif args.stage == "summarize":
        run_summarize(config, job, scan, out_dir)
    elif args.stage == "export":
        run_export(args, config, job, scan, out_dir)
    elif args.stage == "graph":
        run_graph(config, job, scan, out_dir)
    else:
        if not args.fit_dir:
            raise SystemExit("--stage evaluate needs --fit-dir")
        if not args.plane:
            raise SystemExit("--stage evaluate needs --plane (the exported plane id)")
        run_evaluate(args, config, job, scan, out_dir)


if __name__ == "__main__":
    main()
