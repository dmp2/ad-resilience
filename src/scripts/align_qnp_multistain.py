#!/usr/bin/env python3
"""Rigidly align one donor's QNP stains into a donor-local 2-D frame (gate 3).

Inputs: the particles and 32 um maps from ``prepare_qnp_section_particles.py``.

The reference stain's slide frame (``qnp_section_inputs.<job>.reference_stain``, NeuN)
is the donor-local frame ``qnp_local_mm``.  Every other slide is aligned to it with a
2-D rigid transform plus an optional reflection (sections can be mounted face down),
using gray/white tissue-class particles and ``section_alignment.search``: a 1 mm
search, then refinement on 0.5 mm particles.  Scale is held at 1 (same block, same
scanner); the best free scale is reported as a diagnostic only.

Independent checks, never used by the fit:
  * HALO ribbon Dice: HALO propagated its NeuN layer polygons to the other stains with
    its own serial-section registration, so a slide's ribbon carried into the NeuN frame
    by our transform should overlap the NeuN ribbon.  Only slides whose annotations lie
    on their own tissue (inventory) are checked.
  * white-matter Dice between each aligned slide and the reference (p_wm > 0.5).

Use ``--export-od-only`` to export existing OD archives with ``transforms.json``
without fitting, feature regeneration or figures (``--overwrite`` replaces local OD exports).
Native OD archives remain in particle_dir; local archives end in ``_od_bins_<tag>_local.npz``.

Outputs (``alignment`` block of the job): ``transforms.json`` (graph edges with
scores, runner-up gaps and checks), per-stain particles in ``qnp_local_mm``, a gray/
white consensus over the structural stains, a multistain feature table on the
consensus grid, and overlay figures.  No z position is assigned to any slide.

Usage::

    PYTHONPATH=src:src/config_setup:src/preprocess python src/scripts/align_qnp_multistain.py \
        --config configs/project_config.yaml --job H20.33.034_MTG
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi

from config_setup.project_config import load_config, resolve_path
from preprocess.dense_image_utils import save_dense_particle_archive
from preprocess.particle_utils import FeatureDefinition, load_particle_archive, save_particle_archive
from preprocess.qnp_particles import tissue_class_particles
from preprocess.section_alignment import FootprintScorer, Similarity2D, refine, search
from preprocess.xmodmap_compat import entropy_numpy, load_xmodmap, write_vtk_xyz

TISSUE_FEATURES = FeatureDefinition(fine_ids=(0, 1), fine_names=("gray_matter", "white_matter"),
                                    coarse_names=("gray_matter", "white_matter"), fine_to_coarse=(0, 1))
MAP_CENTER = 0.375  # see resample_map
STAIN_COLORS = {"NEUN": "#1a1a19", "LFB": "#2a78d6", "AT": "#eb6834", "ASYN": "#1baf7a",
                "GFAP": "#eda100", "I6": "#e87ba4"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--export-od-only", action="store_true",
                        help="Export native OD archives to qnp_local_mm using saved transforms; no fitting or figures.")
    return parser.parse_args()


def homogeneous(t: Similarity2D, direction: str) -> list[list[float]]:
    """3x3 matrix for stain mm -> local mm ("to_local") or the inverse ("to_slide")."""

    A = t.matrix
    tau = np.asarray(t.tau)
    if direction == "to_slide":  # slide = A (local - tau)
        M, b = A, -A @ tau
    else:  # local = A^-1 slide + tau
        M, b = np.linalg.inv(A), tau
    return np.vstack([np.column_stack([M, b]), [0, 0, 1]]).tolist()


def resample_map(maps: dict, transform: Similarity2D | None, grid_xy: np.ndarray, key: str, order: int) -> np.ndarray:
    """Sample a slide's 32 um map at local-frame points (identity for the reference)."""

    points = grid_xy if transform is None else transform.section_to_moving(grid_xy)
    px = maps["px_mm"]
    # Map pixel k averages level pixels 4k..4k+3, whose centre is at (k + 0.375) * px.
    coords = np.vstack([points[:, 1] / px[0] - MAP_CENTER, points[:, 0] / px[1] - MAP_CENTER])
    values = maps[key].astype(np.float64)
    return ndi.map_coordinates(values, coords, order=order, mode="constant", cval=0.0)


def dice(a: np.ndarray, b: np.ndarray) -> float:
    return float(2 * np.sum(a & b) / max(a.sum() + b.sum(), 1))


def export_od_local(job: dict, particle_dir: Path, out_dir: Path, overwrite: bool = False) -> None:
    """Move existing OD particles with saved rigid transforms, retaining all feature arrays."""

    transform_path = out_dir / "transforms.json"
    record = json.loads(transform_path.read_text())
    reference = job["reference_stain"]
    if record["job"] != job["_name"] or record["reference_stain"] != reference:
        raise ValueError("Saved alignment does not match the requested job/reference")
    # Resolve and check every input before writing any output. No inventory, maps or fits needed.
    exports = []
    for stain in job["stains"]:
        edge = None if stain == reference else record["edges"][stain]
        stem = record["reference_slide"] if edge is None else edge["slide"]
        transform = None if edge is None else Similarity2D.from_dict(edge["transform"])
        matrix = np.eye(3).tolist() if transform is None else homogeneous(transform, "to_local")
        if transform is not None:
            if transform.scale != 1.0:
                raise ValueError(f"{stain}: OD export requires a saved rigid transform (scale = 1)")
            if not np.allclose(matrix, edge["slide_mm_to_local_mm"], rtol=0, atol=1e-9):
                raise ValueError(f"{stain}: saved transform directions disagree")
        for bin_mm in job["bins_mm"]:
            tag = f"{int(round(bin_mm * 1000))}um"
            source = particle_dir / f"{stem}_od_bins_{tag}.npz"
            target = out_dir / f"{stem}_od_bins_{tag}_local.npz"
            if source.resolve() == target.resolve():
                raise ValueError("Native and local OD archives must be separate")
            if not source.is_file():
                raise FileNotFoundError(source)
            if target.exists() and not overwrite:
                raise FileExistsError(f"{target}; use --overwrite to replace local OD exports")
            exports.append((source, target, transform, matrix))
    for source, target, transform, matrix in exports:
        X, nu_intensity, meta = load_particle_archive(source, "nu_intensity")
        if meta["coordinate_space"] != "qnp_slide_mm":
            raise ValueError(f"{source}: expected native qnp_slide_mm coordinates")
        with np.load(source, allow_pickle=False) as archive:
            nu_bin = archive["nu_bin"]
            # Includes feature names and bin definitions; the dense writer retains these verbatim.
            bin_arrays = {k: archive[k] for k in archive.files
                          if k not in {"X", "nu_intensity", "nu_bin", "metadata_json"}}
        if transform is not None:
            X = X.copy()
            X[:, :2] = transform.moving_to_section(X)
        save_dense_particle_archive(target, X, nu_intensity, nu_bin, bin_arrays, {
            **meta, "particle_role": "qnp_section_in_local_frame", "coordinate_space": "qnp_local_mm",
            "coordinate_definition": f"slide mm of reference {reference} ({record['reference_slide']}); "
                                     "x right, y down, z = 0",
            "source_archive": str(source.resolve()),
            "source_coordinate_definition": meta["coordinate_definition"],
            "alignment_transforms": str(transform_path.resolve()), "slide_to_local": matrix,
        }, write_vtk=False)
        print(f"OD local: {target}", flush=True)


def registration_particles(particle_dir: Path, stem: str, tag: str, representation: str = "tissue_classes"):
    """Route existing native archives to the unchanged registration objective."""
    feature_key = {"tissue_classes": "nu_Sub", "od_bins": "nu_bin"}[representation]
    X, nu, meta = load_particle_archive(particle_dir / f"{stem}_{representation}_{tag}.npz", feature_key)
    return X[:, :2].astype(np.float64), nu.astype(np.float64), meta


def fit_registration_stages(scorer_coarse, scorer_fine, P, mu, Pf, muf, settings):
    """Existing rigid search/refinement, retaining the coarse starts for provenance.

    Fine inputs are supplied separately so a coarse representation can change without
    changing the refinement measure. Each fine fit starts from its own ranked winner.
    """
    ranked = search(scorer_coarse, P, mu, thetas=np.arange(0, 360, float(settings.get("theta_step_deg", 10))),
                    scales=[1.0], flips=[False, True], scale_bounds=(1.0, 1.0), h=0.5, blur_mm=1.0,
                    refine_top=int(settings.get("refine_top", 4)))
    best_score, best = refine(scorer_fine, Pf, muf, ranked[0][1], (1.0, 1.0), max_evals=200)
    runner = next(((v, t) for v, t in ranked[1:] if t.flip != best.flip or
                   abs((t.theta_deg - best.theta_deg + 180) % 360 - 180) > 20 or
                   np.hypot(*np.subtract(t.tau, best.tau)) > 3), None)
    free_score, free = refine(scorer_fine, Pf, muf, best, (0.85, 1.15), max_evals=200)
    return ranked, best_score, best, runner, free_score, free


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    job = dict(config["qnp_section_inputs"][args.job], _name=args.job)
    settings = dict(job.get("alignment", {}))
    particle_dir = resolve_path(config, job["particle_dir"])
    out_dir = resolve_path(config, settings.get("output_dir", job["particle_dir"] + "/aligned"))
    if args.export_od_only:
        export_od_local(job, particle_dir, out_dir, args.overwrite)
        return
    qc_dir = resolve_path(config, job["qc_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    inventory = json.loads((qc_dir / "inventory.json").read_text())
    stems = {Path(s["file"]).stem.split("-")[-1].upper(): Path(s["file"]).stem for s in inventory["slides"]}
    reference = job["reference_stain"]
    structural = [s for s, c in job["stains"].items() if c.get("structural")]
    annotated_ok = [s for s in inventory["readiness"]["slides_with_annotations_on_own_tissue"]
                    if s not in job.get("annotation_misfit", [])]
    sigmas = [float(v) for v in settings.get("score_sigmas_mm", [2.0, 1.0, 0.5])]
    coarse_tag, fine_tag = settings.get("search_bin_tag", "1000um"), settings.get("refine_bin_tag", "500um")

    def particles(stain, tag):
        return registration_particles(particle_dir, stems[stain], tag)

    ref_coarse, ref_fine = particles(reference, coarse_tag), particles(reference, fine_tag)
    scorer_coarse = FootprintScorer(ref_coarse[0], ref_coarse[1], sigmas)
    scorer_fine = FootprintScorer(ref_fine[0], ref_fine[1], sigmas)

    transforms: dict[str, Any] = {}
    fitted: dict[str, Similarity2D | None] = {reference: None}
    for stain in job["stains"]:
        if stain == reference:
            continue
        P, mu, _ = particles(stain, coarse_tag)
        Pf, muf, _ = particles(stain, fine_tag)
        ranked, best_score, best, runner, free_score, free = fit_registration_stages(
            scorer_coarse, scorer_fine, P, mu, Pf, muf, settings)
        fitted[stain] = best
        transforms[stain] = {
            "slide": stems[stain], "structural": stain in structural,
            "transform": best.as_dict(),
            "slide_mm_to_local_mm": homogeneous(best, "to_local"),
            "local_mm_to_slide_mm": homogeneous(best, "to_slide"),
            "score_fine": best_score, "score_coarse_best": ranked[0][0],
            "coarse_candidates": [{"score": v, "transform": t.as_dict()} for v, t in ranked],
            "fine_initial_transform": ranked[0][1].as_dict(),
            "runner_up_distinct_coarse": None if runner is None else {"score": runner[0], "transform": runner[1].as_dict()},
            "runner_up_gap_coarse": None if runner is None else runner[0] - ranked[0][0],
            "free_scale_diagnostic": {"scale": free.scale, "score": free_score},
            "reflection": "slide mirrored relative to the reference (mounted face down, or the reverse)"
                          if best.flip else "same handedness as the reference",
        }
        print(f"{stain}: theta {best.theta_deg:.1f} flip {best.flip} score {best_score:.3f} "
              f"(runner-up gap {transforms[stain]['runner_up_gap_coarse']}) free scale {free.scale:.3f}", flush=True)

    # ---- common grid in the local frame (reference map pixels, 32 um) ----------------
    maps = {s: dict(np.load(particle_dir / f"{stems[s]}_maps_32um.npz")) for s in job["stains"]}
    ref_maps = maps[reference]
    px = tuple(float(v) for v in ref_maps["px_mm"])
    pad = int(settings.get("grid_pad_px", 60))
    rows, cols = ref_maps["mask"].shape[0] + 2 * pad, ref_maps["mask"].shape[1] + 2 * pad
    rr, cc = np.indices((rows, cols))
    origin_mm = np.array([(MAP_CENTER - pad) * px[1], (MAP_CENTER - pad) * px[0]])
    grid_xy = origin_mm + np.column_stack([cc.ravel() * px[1], rr.ravel() * px[0]])
    sampled = {}
    for stain in job["stains"]:
        sampled[stain] = {
            "mask": resample_map(maps[stain], fitted[stain], grid_xy, "mask", 0).reshape(rows, cols) > 0.5,
            "p_wm": resample_map(maps[stain], fitted[stain], grid_xy, "p_wm", 1).reshape(rows, cols),
        }
        if "layers" in maps[stain]:
            sampled[stain]["ribbon"] = resample_map(maps[stain], fitted[stain], grid_xy, "layers", 0).reshape(rows, cols) > 0

    # ---- independent checks -------------------------------------------------------
    ref_wm = sampled[reference]["mask"] & (sampled[reference]["p_wm"] > 0.5)
    for stain, entry in transforms.items():
        s = sampled[stain]
        checks = {"tissue_dice_vs_reference": dice(s["mask"], sampled[reference]["mask"]),
                  "white_matter_dice_vs_reference": dice(s["mask"] & (s["p_wm"] > 0.5), ref_wm)}
        if stain in annotated_ok and "ribbon" in s and reference in annotated_ok:
            checks["halo_ribbon_dice_vs_reference"] = dice(s["ribbon"], sampled[reference]["ribbon"])
            # The same check without our transform: slide pixels read as if already aligned.
            identity = resample_map(maps[stain], None, grid_xy, "layers", 0).reshape(rows, cols) > 0
            checks["halo_ribbon_dice_untransformed"] = dice(identity, sampled[reference]["ribbon"])
        entry["independent_checks"] = checks
        print(f"{stain}: {json.dumps({k: round(v, 3) for k, v in checks.items()})}", flush=True)

    # ---- consensus and per-stain particles in the local frame -------------------------
    api = load_xmodmap(config)
    writer = lambda points, arrays, names, filename: write_vtk_xyz(api, points, arrays, names, filename)
    entropy = lambda nu: entropy_numpy(api, nu)
    present = np.stack([sampled[s]["mask"] for s in structural])
    p_stack = np.stack([sampled[s]["p_wm"] for s in structural])
    votes = present.sum(0)
    consensus_mask = votes >= int(settings.get("consensus_min_votes", 2))
    consensus_p = np.where(votes > 0, (p_stack * present).sum(0) / np.maximum(votes, 1), 0.0)
    disagreement = np.where(votes > 1, np.sqrt(((p_stack - consensus_p) ** 2 * present).sum(0) / np.maximum(votes, 1)), 0.0)
    shift = np.array([origin_mm[0], origin_mm[1], 0.0], dtype=np.float32)
    outputs = {}
    for bin_mm in job["bins_mm"]:
        tag = f"{int(round(bin_mm * 1000))}um"
        X, nu, diagnostics = tissue_class_particles(px, consensus_mask, consensus_p, bin_mm)
        X = X + shift
        name = out_dir / f"consensus_{'_'.join(structural)}_tissue_classes_{tag}.npz"
        save_particle_archive(name, X, nu, nu, TISSUE_FEATURES, {
            "particle_role": "qnp_local_consensus", "coordinate_space": "qnp_local_mm",
            "coordinate_definition": f"slide mm of the reference stain {reference} ({stems[reference]})",
            "structural_stains": structural, "consensus_rule": "tissue where >= consensus_min_votes stains have tissue; "
            "p_wm = mean over stains with tissue", "particles": diagnostics, "mass_units": "mm^2 (reference slide area)"},
            vtk_writer=writer, entropy_function=entropy)
        outputs[f"consensus_{tag}"] = str(name)
        for stain in job["stains"]:
            Xs, nus, meta = load_particle_archive(particle_dir / f"{stems[stain]}_tissue_classes_{tag}.npz", "nu_Sub")
            if fitted[stain] is not None:
                Xs = np.column_stack([fitted[stain].moving_to_section(Xs), np.zeros(len(Xs))])
            name = out_dir / f"{stems[stain]}_tissue_classes_{tag}_local.npz"
            save_particle_archive(name, Xs, nus, nus, TISSUE_FEATURES, {
                **meta, "particle_role": "qnp_section_in_local_frame", "coordinate_space": "qnp_local_mm",
                "slide_to_local": None if fitted[stain] is None else transforms[stain]["slide_mm_to_local_mm"]},
                vtk_writer=writer, entropy_function=entropy)
            outputs[f"{stain}_{tag}"] = str(name)
    np.savez_compressed(out_dir / "multistain_maps_32um_local.npz", origin_mm=origin_mm, px_mm=np.asarray(px),
                        stains=np.asarray(list(job["stains"]), dtype=str),
                        tissue=np.stack([sampled[s]["mask"] for s in job["stains"]]),
                        p_wm=np.stack([sampled[s]["p_wm"] for s in job["stains"]]).astype(np.float16),
                        consensus_mask=consensus_mask, consensus_p_wm=consensus_p.astype(np.float16),
                        structural_disagreement_sd=disagreement.astype(np.float16),
                        reference_ribbon=sampled[reference].get("ribbon", np.zeros_like(consensus_mask)))

    consensus_stats = {
        "structural_stains": structural, "consensus_area_mm2": float(consensus_mask.sum() * px[0] * px[1]),
        "p_wm_disagreement_sd_median_in_consensus": float(np.median(disagreement[consensus_mask & (votes > 1)])),
        "p_wm_disagreement_sd_p90_in_consensus": float(np.percentile(disagreement[consensus_mask & (votes > 1)], 90)),
        "fraction_tissue_seen_by_all_structural": float(np.mean(votes[consensus_mask] == len(structural))),
    }
    record = {"job": args.job, "reference_stain": reference, "reference_slide": stems[reference],
              "local_frame": "qnp_local_mm = slide mm of the reference stain (x right, y down, z none)",
              "z_positions": "not assigned: section order and spacing are undocumented",
              "edges": transforms, "consensus": consensus_stats, "outputs": outputs}
    (out_dir / "transforms.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    overlay_figure(sampled, structural, reference, origin_mm, px, transforms, qc_dir / "alignment_overlays.png")
    print(json.dumps(consensus_stats, indent=2))


def overlay_figure(sampled, structural, reference, origin_mm, px, transforms, path) -> None:
    stains = list(sampled)
    rows_, cols_ = sampled[reference]["mask"].shape
    xs = origin_mm[0] + np.arange(cols_) * px[1]
    ys = origin_mm[1] + np.arange(rows_) * px[0]
    ref = sampled[reference]
    fig, axes = plt.subplots(2, (len(stains) + 2) // 2, figsize=(6.2 * ((len(stains) + 2) // 2), 11),
                             constrained_layout=True, squeeze=False)
    panels = list(axes.flat)
    for ax, stain in zip(panels, [s for s in stains if s != reference]):
        s = sampled[stain]
        ax.imshow(np.where(s["mask"], s["p_wm"], np.nan), extent=[xs[0], xs[-1], ys[-1], ys[0]], cmap="Greys",
                  vmin=0, vmax=1.6)
        ax.contour(xs, ys, ref["mask"].astype(float), [0.5], colors=[STAIN_COLORS[reference]], linewidths=0.8)
        ax.contour(xs, ys, (ref["mask"] & (ref["p_wm"] > 0.5)).astype(float), [0.5], colors=[STAIN_COLORS[reference]],
                   linewidths=0.8, linestyles="--")
        ax.contour(xs, ys, s["mask"].astype(float), [0.5], colors=[STAIN_COLORS[stain]], linewidths=0.8)
        ax.contour(xs, ys, (s["mask"] & (s["p_wm"] > 0.5)).astype(float), [0.5], colors=[STAIN_COLORS[stain]],
                   linewidths=0.8, linestyles="--")
        if "ribbon" in s and "ribbon" in ref and "halo_ribbon_dice_vs_reference" in transforms[stain]["independent_checks"]:
            ax.contour(xs, ys, s["ribbon"].astype(float), [0.5], colors=["#6250d6"], linewidths=0.6)
            ax.contour(xs, ys, ref["ribbon"].astype(float), [0.5], colors=["#6250d6"], linewidths=0.6, linestyles=":")
        checks = transforms[stain]["independent_checks"]
        t = transforms[stain]["transform"]
        ax.set_title(f"{stain} -> {reference}: θ={t['theta_deg']:.0f}°, flip={t['flip']}, score {transforms[stain]['score_fine']:.3f}\n"
                     f"WM Dice {checks['white_matter_dice_vs_reference']:.2f}"
                     + (f", HALO ribbon Dice {checks['halo_ribbon_dice_vs_reference']:.2f} "
                        f"(untransformed {checks['halo_ribbon_dice_untransformed']:.2f})"
                        if "halo_ribbon_dice_vs_reference" in checks else ""), fontsize=9)
    ax = panels[len(stains) - 1]
    for stain in structural:
        s = sampled[stain]
        ax.contour(xs, ys, s["mask"].astype(float), [0.5], colors=[STAIN_COLORS[stain]], linewidths=0.8)
        ax.contour(xs, ys, (s["mask"] & (s["p_wm"] > 0.5)).astype(float), [0.5], colors=[STAIN_COLORS[stain]],
                   linewidths=0.8, linestyles="--")
    ax.set_title("structural stains in the local frame: tissue (solid), gray/white interface (dashed)\n"
                 + ", ".join(f"{s}={STAIN_COLORS[s]}" for s in structural), fontsize=9)
    for ax in panels[len(stains):]:
        ax.axis("off")
    for ax in panels[:len(stains)]:
        ax.set_aspect("equal"); ax.set_xlim(xs[0], xs[-1]); ax.set_ylim(ys[-1], ys[0])
        ax.set_xlabel("local x (mm)"); ax.set_ylabel("local y (mm, down)")
    fig.suptitle(f"Within-QNP rigid alignment to {reference}; reference solid/dashed in {STAIN_COLORS[reference]}, "
                 "HALO ribbons purple (moving solid, reference dotted)", fontsize=11)
    fig.savefig(path, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
