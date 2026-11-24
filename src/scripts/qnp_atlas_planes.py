"""Atlas-plane hypotheses for a QNP block: tilted-plane scan, xIV export, post-xIV evaluation.

Stages of ``map_qnp_sections.py`` (the workflow entry point) that work on single atlas planes
rather than on the stacking coordinate:

``scan``       fit every (representation, plane) of a pitch x yaw x y grid with the CPU search
               (``section_mapping.alignment.search``); appends to ``plane_scores.csv`` and resumes.
               Each row also records where the NeuN HALO cortical ribbon (not used by the fit) lands.
``summarize``  profiles, the stain-replicate noise scale, confidence sets, and the gray/white vs
               tissue-only comparison -> ``summary.json`` and figures.
``export``     xIV inputs for one plane (best scanned plane, or ``--plane``): atlas plane particles
               (``export.bin_mm``, MNI plane coordinates, ``nu_Anat`` one-hot channels), the QNP
               consensus mapped into that plane by the fitted similarity (optionally ``nu_Annot``:
               positive-only HALO-annotated area), and ribbon QC points.
``evaluate``   after the xIV fit of an exported ``--plane``: ribbon landing before / after xIV and the
               atlas-correspondence figure.
``graph``      ``transform_graph.json``: slides -> QNP local frame -> exported plane hypotheses -> MNI.
``atlas-features``  the one-hot anatomical channels checked against the segmentation, their fixed
               projections, and figures on representative coronal planes.

Scores share the QNP reference measure, so they are comparable across planes; low scores are not
anatomical validation.
"""

from __future__ import annotations

import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage as ndi

from config_setup.project_config import load_config, resolve_path
from config_setup.qnp_jobs import resolve_qnp_job
from section_mapping.atlas import (FINE_ORDER_TAIL, LANDING_CLASSES, classify_points, landing_fractions, load_atlas,
                                   plane_geometry, plane_id, plane_particles)

INK = "#1a1a19"


def atlas_setup(config: dict, scan: dict) -> dict[str, Any]:
    """The job's atlas (``atlas_scan`` block): labels, tables, decoded names, target centroid."""

    return load_atlas(resolve_path(config, scan["annotation_nifti"]), resolve_path(config, scan["terminology_csv"]),
                      white_matter_codes=scan["white_matter_codes"], drop_codes=scan["drop_codes"],
                      fine_regions=scan["fine_regions"],
                      gray_regions=scan.get("anatomical_features", {}).get("gray_regions", ("STG", "MTG", "ITG")),
                      hemisphere=scan.get("hemisphere", "left"))


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
    from section_mapping.alignment import FootprintScorer

    config = load_config(config_path)
    job = resolve_qnp_job(config, job_name)
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
    from section_mapping.alignment import search

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
    """xIV inputs for one plane of the tilted-plane scan table (``--plane``, default the best)."""

    from preprocess.particle_utils import load_particle_archive
    from section_mapping.alignment import Similarity2D

    by_rep = load_scores(out_dir)
    primary = scan["primary_representation"]
    rows = [r for r in by_rep[primary] if r.get("status") == "ok"]
    row = (next(r for r in rows if r["plane_id"] == args.plane) if args.plane else min(rows, key=lambda r: r["score"]))
    t = Similarity2D(row["theta_deg"], row["scale"], row["flip"] == "True", (row["tau_x"], row["tau_y"]))
    export = scan["export"]
    target_file = resolve_path(config, export["target_particles"])
    X_q, nu_q, meta_q = load_particle_archive(target_file, "nu_Sub")
    annotation = None  # (masses in QNP local mm^2 or None, record); None without export.annotation_maps
    if export.get("annotation_maps"):
        from preprocess.qnp_particles import annotated_tissue_masses
        annotation = annotated_tissue_masses(resolve_path(config, export["annotation_maps"]), X_q, nu_q,
                                             meta_q["particles"]["downsample_factor"])
    paths = {k: resolve_path(config, export[k].format(plane_id=row["plane_id"]))
             for k in ("source_npz", "target_npz", "qc_points_npz")}
    # export.record_subdir (default: the scan directory itself) keeps a second export set's records apart from the
    # original ones, which --stage graph globs in the scan directory.
    record_dir = out_dir / str(export.get("record_subdir", ""))
    write_xiv_inputs(config, scan, atlas_setup(config, scan), (row["pitch_deg"], row["yaw_deg"], row["y0_mm"]), t,
                     (X_q, nu_q, meta_q, target_file), annotation, paths, record_dir / f"export_{row['plane_id']}.json",
                     plane_status="best plane of a discrete scan; a hypothesis, not an estimate of the cutting plane",
                     uncertainty="one member of the scan's confidence set; see summary.json")


def write_xiv_inputs(config, scan, atlas, plane, t, target, annotation, paths, record_file: Path, *, plane_status: str,
                     uncertainty: str, record_extra: dict | None = None) -> dict:
    """Write the xIV source (atlas plane), target (section in the plane) and ribbon QC points, plus a record.

    ``plane`` = (pitch, yaw, y0); ``t`` the section -> plane ``Similarity2D`` (x = QNP local mm,
    p = A (x - tau)); ``target`` = (X, nu, metadata, file) of the section in the QNP local frame;
    ``annotation`` = (masses or None, record) in QNP mm^2, or None.  Target positions are
    ``t.section_to_moving(X)`` and every target mass is multiplied by scale^2 (atlas-plane area).
    """

    from preprocess.particle_utils import FeatureDefinition, save_particle_archive
    from preprocess.qnp_particles import ANNOTATION_FEATURE
    from preprocess.xmodmap_compat import entropy_numpy, load_xmodmap, write_vtk_xyz
    from section_mapping.alignment import FootprintScorer
    from section_mapping.atlas import projection_matrix, projection_metadata, project

    pitch, yaw, y0 = plane
    pid = plane_id(pitch, yaw, y0)
    X_q, nu_q, meta_q, target_file = target
    origin, u, v, n = plane_geometry(atlas, pitch, yaw, y0)
    export = scan["export"]
    spacing = float(export["bin_mm"])
    P, nu_coarse = plane_particles(atlas, scan, origin, u, v, spacing, 1, "coarse")
    _, nu_fine = plane_particles(atlas, scan, origin, u, v, spacing, 1, "fine")
    _, nu_anat = plane_particles(atlas, scan, origin, u, v, spacing, 1, "anatomical")
    if len(P) != len(nu_fine) or len(P) != len(nu_anat):
        raise RuntimeError("coarse, fine and anatomical plane particles differ; check the label tables")
    anat_names = list(atlas["anatomical"])
    if not np.array_equal(project(nu_anat, projection_matrix("gmwm", anat_names)[1]), nu_coarse):
        raise RuntimeError("gmwm projection of the anatomical features differs from the gray/white features")
    # Default: keep only atlas particles near the rigidly placed section.  With
    # restrict_source_to_footprint: false the whole in-plane window (half_width_mm) is kept, so
    # neighbouring gyri (STG, insula, FuG, ...) stay in the source; xIV's support censors them.
    restrict = bool(export.get("restrict_source_to_footprint", True))
    if restrict:
        support = FootprintScorer(X_q[:, :2], nu_q, [1.0], footprint_sigma_mm=float(export["support_sigma_mm"]))
        keep = support.footprint(t.moving_to_section(P[:, :2])) > float(export["support_min_weight"])
    else:
        keep = np.ones(len(P), dtype=bool)
    regions = list(scan["fine_regions"]) + list(FINE_ORDER_TAIL)
    features = FeatureDefinition(fine_ids=tuple(range(len(regions))), fine_names=tuple(regions),
                                 coarse_names=("gray_matter", "white_matter"),
                                 fine_to_coarse=tuple(1 if r == "white_matter" else 0 for r in regions))
    plane_meta = {"origin_mm": origin.tolist(), "u_axis": u.tolist(), "v_axis": v.tolist(), "normal": n.tolist(),
                  "pitch_deg": pitch, "yaw_deg": yaw, "y0_mm": y0,
                  "plane_id": pid, "reslice": "nearest-neighbour labels on a regular in-plane grid",
                  "inverse": "world = origin + plane_X[:, 0] * u_axis + plane_X[:, 1] * v_axis",
                  "status": plane_status}
    api = load_xmodmap(config)
    writer = lambda points, arrays, names, filename: write_vtk_xyz(api, points, arrays, names, filename)
    entropy = lambda nu: entropy_numpy(api, nu)
    save_particle_archive(paths["source_npz"], P[keep], nu_coarse[keep], nu_fine[keep], features, {
        "particle_role": "source_atlas_plane", "coordinate_space": "mni_icbm2009b_symmetric_plane",
        "plane": plane_meta, "atlas_annotation": str(resolve_path(config, scan["annotation_nifti"])),
        "label_grouping": {"white_matter_codes": scan["white_matter_codes"], "drop_codes": scan["drop_codes"],
                           "fine_regions": scan["fine_regions"], "gray_matter": "every other non-zero code"},
        "support_restriction": ({"rule": "QNP footprint weight at the rigidly placed position > support_min_weight",
                                 **{k: export[k] for k in ("support_sigma_mm", "support_min_weight")}} if restrict else
                                {"rule": "none: the whole in-plane window (restrict_source_to_footprint false)",
                                 "half_width_mm": scan["half_width_mm"]}),
        "anatomical_features": {"key": "nu_Anat", "names": anat_names,
                                "rule": "one-hot label channels (section_mapping/atlas.py); nu_Sub = gmwm projection",
                                "projections": projection_metadata(anat_names)},
        "mass_units": "mm^2 (in-plane pixel area)"}, vtk_writer=writer, entropy_function=entropy,
        extra_arrays={"nu_Anat": nu_anat[keep].astype(np.float32),
                      "anatomical_feature_names": np.asarray(anat_names, dtype=str)})

    Y = t.section_to_moving(X_q[:, :2])
    target_X = np.column_stack([Y, np.zeros(len(Y))])
    similarity = {**t.as_dict(), "from": "qnp_local_mm", "to": "atlas plane mm", "plane_id": pid,
                  "uncertainty": uncertainty}
    target_meta = {**meta_q, "particle_role": "target_qnp_in_atlas_plane",
                   "coordinate_space": "qnp_local_in_atlas_plane_mm", "source_particles": str(target_file),
                   "similarity_qnp_local_to_plane": similarity, "plane": plane_meta,
                   "mass_units": "mm^2 in atlas-plane units (QNP area * scale^2)"}
    target_extra = None
    if annotation is not None:
        masses, annotation_record = annotation
        target_meta["annotation_supervision"] = {**annotation_record, "key": "nu_Annot",
                                                 "mass_units": "mm^2 in atlas-plane units (annotated area * scale^2)"}
        if masses is not None:
            target_extra = {"nu_Annot": (masses * t.scale ** 2).astype(np.float32),
                            "annotation_feature_names": np.asarray([ANNOTATION_FEATURE], dtype=str)}
    save_particle_archive(paths["target_npz"], target_X, nu_q * t.scale ** 2, nu_q * t.scale ** 2,
                          FeatureDefinition((0, 1), ("gray_matter", "white_matter"), ("gray_matter", "white_matter"), (0, 1)),
                          target_meta, vtk_writer=writer, entropy_function=entropy, extra_arrays=target_extra)
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
    if annotation is not None:
        record["annotation_supervision"] = annotation[1]
    record.update(record_extra or {})
    record_file.parent.mkdir(parents=True, exist_ok=True)
    record_file.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: record[k] for k in ("source_particles", "source_area_mm2", "target_particles",
                                              "target_area_mm2_plane_units", "rigid_ribbon_landing")}, indent=2))
    print(f"wrote {record_file}")
    return record


# --------------------------------------------------------------------------------------
# Stage: atlas-features (canonical one-hot anatomical channels: checks and figures)
# --------------------------------------------------------------------------------------

ANATOMY_COLORS = {"STG_gray_matter": "#eb6834", "MTG_gray_matter": "#2a78d6", "ITG_gray_matter": "#1baf7a",
                  "other_gray_matter": "#e6e4dc", "white_matter": "#c4c2b9"}
GMWM_COLORS = {"gray_matter": "#8c8a82", "white_matter": "#dcdad2"}


def _anatomy_checks(atlas: dict, scan: dict, config: dict) -> dict[str, Any]:
    """Channel voxel counts by table lookup vs an independent np.isin path (and Allen volumes)."""

    from section_mapping.atlas import label_channel_index, projected_tables

    labels, tables = atlas["labels"], atlas["anatomical"]
    index = label_channel_index(tables)[labels]
    wm, drop = np.isin(labels, scan["white_matter_codes"]), np.isin(labels, scan["drop_codes"]) | (labels == 0)
    regions = scan.get("anatomical_features", {}).get("gray_regions", ["STG", "MTG", "ITG"])
    region_codes = {r: scan["fine_regions"][r]["codes"] for r in regions}
    named = np.isin(labels, sum(region_codes.values(), []))
    expected = {f"{r}_gray_matter": np.isin(labels, c) & ~wm & ~drop for r, c in region_codes.items()}
    expected["other_gray_matter"] = ~wm & ~drop & ~named
    expected["white_matter"] = wm
    channels = {}
    for k, name in enumerate(tables):
        got = index == k
        channels[name] = {"voxels": int(got.sum()), "expected_voxels_isin": int(expected[name].sum()),
                          "voxels_differing": int((got != expected[name]).sum())}
    volumes_csv = scan.get("anatomical_features", {}).get("parcellation_volumes_csv")
    if volumes_csv:
        with open(resolve_path(config, scan["terminology_csv"]), newline="", encoding="utf-8") as stream:
            ident = {int(r["annotation_value"]): r["identifier"] for r in csv.DictReader(stream) if r["annotation_value"].strip()}
        with open(resolve_path(config, volumes_csv), newline="", encoding="utf-8") as stream:
            allen = {r["identifier"]: int(r["voxel_count"]) for r in csv.DictReader(stream)}
        for r, codes in region_codes.items():
            channels[f"{r}_gray_matter"]["allen_parcellation_voxels"] = int(sum(allen[ident[c]] for c in codes))
    coarse_index = np.full(len(atlas["coarse"]["gray_matter"]), -1, np.int16)
    for k, table in enumerate(atlas["coarse"].values()):
        coarse_index[table] = k
    gm_from_anat = projected_tables(tables, "gmwm")
    gm_index = np.full(len(coarse_index), -1, np.int16)
    for k, table in enumerate(gm_from_anat.values()):
        gm_index[table] = k
    return {"channels": channels,
            "unassigned_nonzero_voxels": int(((index < 0) & ~drop).sum()),
            "gmwm_projection_voxels_differing": int((gm_index[labels] != coarse_index[labels]).sum())}


def _other_gray_codes(atlas: dict, ys, half: float, hemisphere_sign: float) -> list[dict[str, Any]]:
    """Codes that make up other_gray_matter on the figure planes (one hemisphere), by area."""

    from section_mapping.atlas import reslice_labels

    counts: dict[int, int] = {}
    for y in ys:
        origin, u, v, _ = plane_geometry(atlas, 0.0, 0.0, y)
        grid, axis = reslice_labels(atlas["labels"], atlas["affine"], origin, u, v, half, 0.5)
        same_side = np.sign(atlas["centroid"][0] + axis)[None, :] == hemisphere_sign
        codes = grid[atlas["anatomical"]["other_gray_matter"][np.clip(grid, 0, None)] & same_side]
        for c, n in zip(*np.unique(codes, return_counts=True)):
            counts[int(c)] = counts.get(int(c), 0) + int(n)
    return [{"code": c, "name": atlas["names"].get(c), "area_mm2": n * 0.25}
            for c, n in sorted(counts.items(), key=lambda item: -item[1])]


def _channel_image(atlas: dict, y: float, half: float, res: float = 0.25):
    from section_mapping.atlas import label_channel_index, reslice_labels

    origin, u, v, _ = plane_geometry(atlas, 0.0, 0.0, y)
    grid, axis = reslice_labels(atlas["labels"], atlas["affine"], origin, u, v, half, res)
    index = label_channel_index(atlas["anatomical"])
    return index[np.clip(grid, 0, len(index) - 1)], atlas["centroid"][0] + axis, atlas["centroid"][2] + axis


def _draw_channels(ax, img, xs, zs, names, colors, label_names=()):
    from matplotlib.colors import ListedColormap

    step = xs[1] - xs[0]
    ax.imshow(img + 1, origin="lower", extent=(xs[0] - step / 2, xs[-1] + step / 2, zs[0] - step / 2, zs[-1] + step / 2),
              cmap=ListedColormap(["#ffffff"] + [colors[n] for n in names]), vmin=-0.5, vmax=len(names) + 0.5,
              interpolation="nearest")
    ax.contour(xs, zs, (img >= 0).astype(float), [0.5], colors="#52514e", linewidths=0.5)
    for k, name in enumerate(names):
        if name not in label_names:
            continue
        mask = img == k
        if mask.sum() < 40:
            continue
        lab, n = ndi.label(mask)
        biggest = 1 + int(np.argmax(ndi.sum(mask, lab, range(1, n + 1))))
        iz, ix = np.unravel_index(np.argmax(ndi.distance_transform_edt(lab == biggest)), mask.shape)
        ax.text(xs[ix], zs[iz], name.split("_")[0], fontsize=7, ha="center", va="center", color=INK,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))
    ax.set_aspect("equal")
    ax.tick_params(labelsize=7)


def run_atlas_features(config, job, scan, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    from section_mapping.atlas import PROJECTIONS, label_channel_index, project, projection_matrix

    atlas = atlas_setup(config, scan)
    feats = scan.get("anatomical_features", {})
    ys = [float(y) for y in feats.get("figure_y_mm", np.linspace(*atlas["region_extent_y"], 7)[1:-1].round(1))]
    reference_y = float(feats.get("reference_y_mm", ys[len(ys) // 2]))
    half = float(feats.get("figure_half_width_mm", scan["half_width_mm"]))
    names = list(atlas["anatomical"])
    target = f"{next(iter(scan['fine_regions']))}_gray_matter"
    sign = -1.0 if scan.get("hemisphere", "left") == "left" else 1.0
    fig_dir = out_dir / "atlas_features"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Plane-level checks at the particle level the scan and export use (reslice_mm pixels).
    spacing = float(scan["reslice_mm"])
    planes = []
    for y in sorted(set(ys + [reference_y])):
        origin, u, v, _ = plane_geometry(atlas, 0.0, 0.0, y)
        P, nu_coarse = plane_particles(atlas, scan, origin, u, v, spacing, 1, "coarse")
        P2, nu_fine = plane_particles(atlas, scan, origin, u, v, spacing, 1, "fine")
        P3, nu_anat = plane_particles(atlas, scan, origin, u, v, spacing, 1, "anatomical")
        if not (np.array_equal(P, P3) and np.array_equal(P2, P3)):
            raise RuntimeError(f"plane y {y}: particle sets differ between feature tables")
        row = {"y_mm": y, "particles": int(len(P)),
               "one_hot": bool(np.all((nu_anat > 0).sum(1) == 1)),
               "area_mm2": dict(zip(names, nu_anat.sum(0).round(3).tolist()))}
        for name in PROJECTIONS:
            row[f"max_abs_diff_{name}"] = float(np.abs(project(nu_anat, projection_matrix(name, names)[1]) - {
                "gmwm": nu_coarse, "mtg_gm": nu_fine[:, [list(atlas["fine"]).index(target.split("_")[0])]],
                "anatomical": nu_anat}[name]).max())
        planes.append(row)

    volume = _anatomy_checks(atlas, scan, config)
    other = _other_gray_codes(atlas, ys, half, sign)
    unnamed = sorted(int(c) for c in np.unique(atlas["labels"]) if c and c not in atlas["names"])
    index = label_channel_index(atlas["anatomical"])
    record = {
        "channels": names, "gray_regions": feats.get("gray_regions", ["STG", "MTG", "ITG"]),
        "volume_checks": volume, "plane_checks": planes,
        "other_gray_matter_codes_on_figure_planes": {
            "hemisphere": scan.get("hemisphere", "left"), "half_width_mm": half, "codes": other[:25]},
        "unnamed_codes": [{"code": c, "voxels": int((atlas["labels"] == c).sum()),
                           "channel": names[index[c]] if index[c] >= 0 else None} for c in unnamed],
        "passed": bool(all(ch["voxels_differing"] == 0 and ch.get("allen_parcellation_voxels", ch["voxels"]) == ch["voxels"]
                           for ch in volume["channels"].values())
                       and volume["unassigned_nonzero_voxels"] == 0 and volume["gmwm_projection_voxels_differing"] == 0
                       and all(p["one_hot"] and max(p[f"max_abs_diff_{n}"] for n in PROJECTIONS) == 0 for p in planes)),
    }
    (fig_dir / "atlas_features.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    # Figure 1: the five channels on representative coronal planes.
    fig, axes = plt.subplots(1, len(ys), figsize=(3.1 * len(ys), 3.9), constrained_layout=True, squeeze=False)
    for ax, y in zip(axes[0], ys):
        img, xs, zs = _channel_image(atlas, y, half)
        _draw_channels(ax, img, xs, zs, names, ANATOMY_COLORS, label_names=names[:-2])
        ax.set_title(f"coronal y = {y:g} mm", fontsize=9)
        ax.set_xlabel("MNI x (mm)", fontsize=8)
    axes[0][0].set_ylabel("MNI z (mm)", fontsize=8)
    fig.legend([Patch(fc=ANATOMY_COLORS[n], ec="#52514e", lw=0.4) for n in names],
               [n.replace("_", " ") for n in names], loc="outside lower center", ncol=len(names), frameon=False, fontsize=8)
    fig.suptitle("AHRAv2 anatomical atlas channels (one-hot; dropped codes and background white)", fontsize=10)
    fig.savefig(fig_dir / "anatomical_channels.png", dpi=150)
    plt.close(fig)

    # Figure 2: the three fixed projections of one plane.
    img, xs, zs = _channel_image(atlas, reference_y, half)
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 4.2), constrained_layout=True)
    for ax, name in zip(axes, ("gmwm", "mtg_gm", "anatomical")):
        out_names, M = projection_matrix(name, names)
        # Pixel-level projection: a one-hot pixel takes the projected channel that receives it.
        proj = np.full(img.shape, -1, int)
        for j, row in enumerate(M):
            proj[np.isin(img, np.flatnonzero(row))] = j
        colors = {**GMWM_COLORS, **ANATOMY_COLORS}
        gyri = [n for n in out_names if n not in ("gray_matter", "other_gray_matter", "white_matter")]
        _draw_channels(ax, proj, xs, zs, list(out_names), colors, label_names=gyri)
        if name == "mtg_gm":
            ax.contour(xs, zs, (img >= 0).astype(float), [0.5], colors="#52514e", linewidths=0.5)
        ax.set_title(f"{name}: {len(out_names)} channel{'s' if len(out_names) > 1 else ''}", fontsize=9)
        ax.set_xlabel("MNI x (mm)", fontsize=8)
        ax.legend([Patch(fc=colors[n], ec="#52514e", lw=0.4) for n in out_names],
                  [n.replace("_", " ") for n in out_names], loc="lower left", fontsize=6, frameon=True)
    axes[0].set_ylabel("MNI z (mm)", fontsize=8)
    fig.suptitle(f"Fixed projections of the anatomical channels, coronal y = {reference_y:g} mm "
                 "(gmwm = the original gray/white atlas features)", fontsize=10)
    fig.savefig(fig_dir / "feature_projections.png", dpi=150)
    plt.close(fig)
    print(json.dumps({"passed": record["passed"], "volume_checks": volume,
                      "figures": [str(fig_dir / "anatomical_channels.png"), str(fig_dir / "feature_projections.png")]},
                     indent=2))


# --------------------------------------------------------------------------------------
# Stage: evaluate
# --------------------------------------------------------------------------------------

def run_evaluate(args, config, job, scan, out_dir: Path, record_file: Path | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import nibabel as nib
    from scipy import ndimage as ndi

    from section_mapping.atlas import reslice_labels

    fit_dir = Path(args.fit_dir)
    record_file = record_file or out_dir / str(scan["export"].get("record_subdir", "")) / f"export_{args.plane}.json"
    export = json.loads(Path(record_file).read_text())
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




def run_stage(args, config: dict, job: dict) -> None:
    """Dispatch one plane stage (called by ``map_qnp_sections.main``)."""

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
    elif args.stage == "atlas-features":
        run_atlas_features(config, job, scan, out_dir)
    else:
        if not args.fit_dir:
            raise SystemExit("--stage evaluate needs --fit-dir")
        if not args.plane:
            raise SystemExit("--stage evaluate needs --plane (the exported plane id)")
        run_evaluate(args, config, job, scan, out_dir)
