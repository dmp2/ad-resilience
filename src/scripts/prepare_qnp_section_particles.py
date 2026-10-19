#!/usr/bin/env python3
"""Inventory one donor's QNP slides and turn each into 2-D feature particles.

Inputs and outputs are named by ``qnp_section_inputs.<job>`` in the project YAML.

Stage ``inventory`` (gate 1)
    Reads every .svs header and HALO annotation file, checks units (Aperio MPP, level
    downsample), annotation names and whether each slide's annotations lie on its own
    tissue, and writes ``inventory.json`` plus a short readiness report.  Nothing is
    assumed about section order or spacing (see SECTION_RELATIONSHIP).

Stage ``particles`` (gate 2)
    For every slide: tissue mask, soft gray/white membership, and particles at each
    configured bin size, in two representations (tissue classes and optical-density
    bins; see ``preprocess.qnp_particles``).  Also writes 32 um maps for overlays,
    ribbon QC points with approximate depth from pia, and QC figures.

Usage::

    PYTHONPATH=src:src/config_setup:src/preprocess python src/scripts/prepare_qnp_section_particles.py \
        --config configs/project_config.yaml --job H20.33.034_MTG --stage inventory
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

from config_setup.project_config import load_config, resolve_path
from preprocess.particle_utils import FeatureDefinition, save_particle_archive
from preprocess.qnp_particles import (
    LAYER_ORDER,
    canonical_layer,
    gm_wm_membership,
    optical_density,
    optical_density_particles,
    rasterize_layers,
    read_halo_layers,
    read_slide,
    ribbon_depth_points,
    structural_feature,
    tissue_class_particles,
    tissue_mask,
)
from preprocess.xmodmap_compat import entropy_numpy, load_xmodmap, write_vtk_xyz
from dense_image_utils import save_dense_particle_archive

# What the SEA-AD QNP documentation (data/raw/sea-ad/documentation/*.pdf) establishes,
# and what it does not.  Kept verbatim in inventory.json.
SECTION_RELATIONSHIP = {
    "documented": [
        "paraffin blocks; 5 um sections cut on a microtome (section 2.A.a)",
        "Aperio AT2 scans at 20x (section 3.A.b)",
        "per donor, the different stain WSIs were registered with HALO's Serial Section "
        "registration tool and the NeuN-trained layer annotations transferred to them (section 3)",
    ],
    "not_documented": [
        "the order of the stained sections within the block",
        "the spacing between them (they need not be consecutive 5 um sections)",
        "the block's orientation in the brain or its slab of origin",
    ],
    "consequence": "slides are treated as independently located 2-D sections of one block with "
                   "unknown relative z; within-QNP alignment is 2-D only and no stack is built",
}
TISSUE_FEATURES = FeatureDefinition(fine_ids=(0, 1), fine_names=("gray_matter", "white_matter"),
                                    coarse_names=("gray_matter", "white_matter"), fine_to_coarse=(0, 1))
MAP_REDUCTION = 4  # level-2 pixels (~8 um) per overlay-map pixel (~32 um)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--job", required=True, help="Name under qnp_section_inputs.")
    parser.add_argument("--stage", required=True, choices=["inventory", "particles"])
    parser.add_argument("--stain", action="append", help="Restrict --stage particles to these stains.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def slide_paths(config: dict, job: dict) -> dict[str, Path]:
    directory = resolve_path(config, job["slide_dir"])
    paths = {path.stem.split("-")[-1].upper(): path for path in sorted(directory.glob("*.svs"))}
    missing = sorted(set(job["stains"]) - set(paths))
    if missing:
        raise FileNotFoundError(f"No .svs for stains {missing} in {directory}")
    return {stain: paths[stain] for stain in job["stains"]}


# --------------------------------------------------------------------------------------
# Stage: inventory
# --------------------------------------------------------------------------------------

def annotation_fit(slide, layers) -> dict[str, Any]:
    """How much of the HALO ribbon lies on this slide's own tissue."""

    od, _ = optical_density(slide.rgb)
    mask, _ = tissue_mask(od, slide.pixel_area_mm2)
    layer_image = rasterize_layers(layers, slide)
    vertices = np.vstack([xy for polys in layers.values() for xy, _ in polys])
    height, width = slide.baseline_shape
    outside = (vertices[:, 0] < 0) | (vertices[:, 1] < 0) | (vertices[:, 0] >= width) | (vertices[:, 1] >= height)
    ribbon = layer_image > 0
    return {
        "vertices_outside_image_fraction": float(outside.mean()),
        "ribbon_area_mm2": float(ribbon.sum() * slide.pixel_area_mm2),
        "ribbon_on_tissue_fraction": float((ribbon & mask).sum() / max(ribbon.sum(), 1)),
    }


def run_inventory(config: dict, job: dict, qc_dir: Path) -> dict[str, Any]:
    slides = []
    for stain, path in slide_paths(config, job).items():
        slide = read_slide(path, int(job.get("pyramid_level", -1)))
        entry = slide.describe()
        annotation_file = path.with_suffix(".annotations")
        masks = sorted((path.parent / "analysis_masks").glob(f"{path.stem}_*"))
        entry["halo_analysis_mask_files"] = len(masks)
        if annotation_file.exists():
            layers = read_halo_layers(annotation_file)
            import xml.etree.ElementTree as ET
            raw_names = [a.get("Name") for a in ET.parse(annotation_file).getroot().iter("Annotation")]
            entry["annotation_file"] = str(annotation_file)
            entry["annotation_raw_names"] = raw_names
            entry["annotation_canonical"] = {name: canonical_layer(name) for name in raw_names}
            entry["annotation_fit"] = annotation_fit(slide, layers)
        else:
            entry["annotation_file"] = None
        slides.append(entry)
        print(f"{stain}: {entry['level_shape']} px at {entry['px_mm_row_col']} mm, scanned {entry['scan_date']}", flush=True)

    blocks = sorted({entry["block"] for entry in slides})
    threshold = float(job.get("annotation_on_tissue_min", 0.9))
    fitting = [e["stain"] for e in slides if e.get("annotation_file") and
               e["annotation_fit"]["ribbon_on_tissue_fraction"] >= threshold]
    misfit = [e["stain"] for e in slides if e.get("annotation_file") and e["stain"] not in fitting]
    readiness = {
        "donor": slides[0]["donor"], "region": job["region"], "number_slides": len(slides),
        "block_ids_as_named": blocks,
        "block_id_note": ("block ids differ only in zero padding (e.g. A07 vs A7); treated as one block, "
                          "which the shared HALO serial-section registration supports but does not prove")
        if len({b.lstrip("A").lstrip("0") for b in blocks}) == 1 and len(blocks) > 1 else None,
        "mpp_baseline_um": sorted({e["mpp_baseline_um"] for e in slides}),
        "slides_with_annotations_on_own_tissue": fitting,
        "slides_with_misfit_annotations": misfit,
        "slides_without_annotations": [e["stain"] for e in slides if not e.get("annotation_file")],
        "structural_stains": [s for s, c in job["stains"].items() if c.get("structural")],
        "section_relationship": SECTION_RELATIONSHIP,
        "smallest_usable_set": {
            "single_stain_start": job["reference_stain"],
            "structural_set": [s for s, c in job["stains"].items() if c.get("structural")],
            "reason": "LFB and NeuN give the strongest gray/white contrast; AT8 and alpha-syn slides show it "
                      "through hematoxylin/myelin background; GFAP and Iba1/6E10 are confounded by glial signal",
        },
    }
    inventory = {"job": job["_name"], "readiness": readiness, "slides": slides}
    (qc_dir / "inventory.json").write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")
    return inventory


# --------------------------------------------------------------------------------------
# Stage: particles
# --------------------------------------------------------------------------------------

def _reduce(image: np.ndarray, factor: int, mode: str) -> np.ndarray:
    rows, cols = (image.shape[0] // factor) * factor, (image.shape[1] // factor) * factor
    blocks = image[:rows, :cols].reshape(rows // factor, factor, cols // factor, factor)
    return blocks.max(axis=(1, 3)) if mode == "max" else blocks.mean(axis=(1, 3))


def process_slide(config, job, stain, path, out_dir, writers) -> dict[str, Any]:
    settings = job["stains"][stain]
    slide = read_slide(path, int(job.get("pyramid_level", -1)))
    smooth_px = float(job.get("smoothing_um", 100.0)) / 1000.0 / float(np.mean(slide.px_mm))
    od, background = optical_density(slide.rgb)
    mask, mask_info = tissue_mask(od, slide.pixel_area_mm2)
    feature = structural_feature(od, slide.rgb, settings.get("feature", "red_od_share"), smooth_px)
    edge_px = float(job.get("membership_edge_exclusion_um", 300.0)) / 1000.0 / float(np.mean(slide.px_mm))
    p_wm, membership = gm_wm_membership(feature, mask, bool(settings.get("wm_high", True)),
                                        float(job.get("membership_softness", 0.05)), edge_px)

    layer_image = None
    annotation = path.with_suffix(".annotations")
    if annotation.exists() and stain not in job.get("annotation_misfit", []):
        layer_image = rasterize_layers(read_halo_layers(annotation), slide)
        ribbon = layer_image > 0
        membership["annotation_check"] = {
            "mean_p_wm_in_ribbon": float(p_wm[ribbon & mask].mean()),
            "mean_p_wm_by_layer": {name: float(p_wm[(layer_image == k) & mask].mean())
                                   for k, name in enumerate(LAYER_ORDER, start=1) if np.any(layer_image == k)},
            "note": "the ribbon is cortex by construction, so p_wm there should be low; deep layers are "
                    "expected to be higher than L1-3 because of myelinated radial fibres",
        }
        depth = ribbon_depth_points(layer_image, slide, step_px=int(job.get("ribbon_point_step_px", 4)))
        np.savez_compressed(out_dir / f"{path.stem}_ribbon_points.npz", **depth,
                            coordinate_space=np.asarray("qnp_slide_mm"))

    base = {
        "particle_role": "qnp_section", "coordinate_space": "qnp_slide_mm",
        "coordinate_definition": "x = column * px_x_mm, y = row * px_y_mm (image rows, y down), z = 0; "
                                 "origin = top-left baseline pixel",
        "slide": slide.describe(), "background_rgb": background, "tissue_mask": mask_info,
        "smoothing_um": float(job.get("smoothing_um", 100.0)), "structural_feature": settings,
        "membership": membership,
    }
    summary = {"stain": stain, "slide": path.name, "tissue": mask_info, "membership": membership, "particles": {}}
    for bin_mm in job["bins_mm"]:
        tag = f"{int(round(bin_mm * 1000))}um"
        X, nu, diagnostics = tissue_class_particles(slide.px_mm, mask, p_wm, bin_mm)
        save_particle_archive(
            out_dir / f"{path.stem}_tissue_classes_{tag}.npz", X, nu, nu, TISSUE_FEATURES,
            {**base, "representation": "tissue_classes", "particles": diagnostics,
             "mass_units": "mm^2 (slide area)"},
            vtk_writer=writers[0], entropy_function=writers[1],
        )
        Xo, nu_int, nu_od, transform, od_diag, bin_arrays = optical_density_particles(
            slide, mask, od, bin_mm, int(job.get("od_bins", 8)))
        save_dense_particle_archive(
            out_dir / f"{path.stem}_od_bins_{tag}.npz", Xo, nu_int, nu_od, bin_arrays,
            {**base, "representation": "optical_density_bins", "intensity_transform": transform.as_dict(),
             "particles": od_diag, "comparability": "stain-specific; not comparable across stains or with MRI"},
            vtk_writer=writers[0], entropy_function=writers[1],
        )
        summary["particles"][tag] = {"number": int(len(X)), "bin_mm_actual": diagnostics["bin_mm_actual"],
                                     "tissue_area_mm2": float(nu.sum()), "wm_area_mm2": float(nu[:, 1].sum()),
                                     "mass_relative_error": diagnostics["mass_relative_error"]}

    maps = {"mask": _reduce(mask, MAP_REDUCTION, "max"), "p_wm": _reduce(p_wm, MAP_REDUCTION, "mean").astype(np.float16),
            "rgb": np.stack([_reduce(slide.rgb[..., c].astype(np.float32), MAP_REDUCTION, "mean")
                             for c in range(3)], -1).astype(np.uint8),
            "px_mm": np.asarray(slide.px_mm) * MAP_REDUCTION}
    if layer_image is not None:
        maps["layers"] = _reduce(layer_image, MAP_REDUCTION, "max")
    np.savez_compressed(out_dir / f"{path.stem}_maps_32um.npz", **maps)
    return summary


def particle_qc_figure(out_dir: Path, slides: dict[str, Path], bin_tag: str, path: Path) -> None:
    fig, axes = plt.subplots(len(slides), 4, figsize=(18, 4.2 * len(slides)), constrained_layout=True, squeeze=False)
    for row, (stain, svs) in zip(axes, slides.items()):
        maps = np.load(out_dir / f"{svs.stem}_maps_32um.npz")
        px = maps["px_mm"]
        extent = [0, maps["mask"].shape[1] * px[1], maps["mask"].shape[0] * px[0], 0]
        row[0].imshow(maps["rgb"], extent=extent)
        row[0].contour(np.arange(maps["mask"].shape[1]) * px[1], np.arange(maps["mask"].shape[0]) * px[0],
                       maps["mask"].astype(float), levels=[0.5], colors=["#2a78d6"], linewidths=0.6)
        if "layers" in maps:
            row[0].contour(np.arange(maps["mask"].shape[1]) * px[1], np.arange(maps["mask"].shape[0]) * px[0],
                           (maps["layers"] > 0).astype(float), levels=[0.5], colors=["#eb6834"], linewidths=0.6)
        row[0].set_title(f"{stain}: RGB, tissue mask (blue), HALO ribbon (orange)", fontsize=9)
        shown = row[1].imshow(np.where(maps["mask"], maps["p_wm"].astype(float), np.nan), extent=extent,
                              cmap="viridis", vmin=0, vmax=1)
        fig.colorbar(shown, ax=row[1], shrink=0.7, label="p(white matter)")
        row[1].set_title("soft white-matter membership", fontsize=9)
        with np.load(out_dir / f"{svs.stem}_tissue_classes_{bin_tag}.npz") as archive:
            X, nu = archive["X"], archive["nu_Sub"]
        frac = nu[:, 1] / np.maximum(nu.sum(1), 1e-12)
        shown = row[2].scatter(X[:, 0], X[:, 1], c=frac, s=12 * nu.sum(1) / nu.sum(1).max(), marker="s",
                               cmap="viridis", vmin=0, vmax=1, linewidths=0)
        row[2].set_title(f"tissue-class particles ({bin_tag}, n={len(X)}); size = tissue area", fontsize=9)
        with np.load(out_dir / f"{svs.stem}_od_bins_{bin_tag}.npz") as archive:
            Xo, nu_int = archive["X"], archive["nu_intensity"]
        mean_od = nu_int[:, 1] / np.maximum(nu_int[:, 0], 1e-12)
        shown = row[3].scatter(Xo[:, 0], Xo[:, 1], c=mean_od, s=12 * nu_int[:, 0] / nu_int[:, 0].max(),
                               marker="s", cmap="magma", linewidths=0)
        fig.colorbar(shown, ax=row[3], shrink=0.7, label="scaled total OD")
        row[3].set_title("optical-density baseline (stain-specific)", fontsize=9)
        for ax in row:
            ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3]); ax.set_aspect("equal")
            ax.set_xlabel("slide x (mm)"); ax.set_ylabel("slide y (mm, down)")
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    jobs = config.get("qnp_section_inputs", {})
    if args.job not in jobs:
        raise KeyError(f"Unknown job '{args.job}'. Available: {', '.join(sorted(jobs)) or '<none>'}")
    job = dict(jobs[args.job], _name=args.job)
    qc_dir = resolve_path(config, job["qc_dir"])
    out_dir = resolve_path(config, job["particle_dir"])
    qc_dir.mkdir(parents=True, exist_ok=True)

    if args.stage == "inventory":
        inventory = run_inventory(config, job, qc_dir)
        print(json.dumps(inventory["readiness"], indent=2))
        return

    if not (qc_dir / "inventory.json").exists():
        raise SystemExit("Run --stage inventory first.")
    slides = slide_paths(config, job)
    selected = {s: p for s, p in slides.items() if not args.stain or s in [x.upper() for x in args.stain]}
    out_dir.mkdir(parents=True, exist_ok=True)
    api = load_xmodmap(config)
    writers = (lambda points, arrays, names, filename: write_vtk_xyz(api, points, arrays, names, filename),
               lambda nu: entropy_numpy(api, nu))
    summary_file = qc_dir / "particles_summary.json"
    summaries = json.loads(summary_file.read_text()) if summary_file.exists() else {}
    for stain, path in selected.items():
        if (out_dir / f"{path.stem}_maps_32um.npz").exists() and not args.overwrite and stain in summaries:
            print(f"exists, skipped: {stain}")
            continue
        summaries[stain] = process_slide(config, job, stain, path, out_dir, writers)
        print(json.dumps({k: summaries[stain][k] for k in ("stain", "particles")}), flush=True)
    summary_file.write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
    finished = {s: p for s, p in slides.items() if (out_dir / f"{p.stem}_maps_32um.npz").exists()}
    for bin_mm in job["bins_mm"]:
        tag = f"{int(round(bin_mm * 1000))}um"
        particle_qc_figure(out_dir, finished, tag, qc_dir / f"particles_{tag}.png")
    print(f"Wrote particles to {out_dir} and QC to {qc_dir}")


if __name__ == "__main__":
    main()
