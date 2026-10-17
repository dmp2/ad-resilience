#!/usr/bin/env python3
"""Prepare one atlas plane and the matching compressed MERFISH targets for xIV.

Inputs and outputs are named by ``atlas_section_inputs.<job>`` in the project YAML.

Atlas side
----------
The 3-D atlas label particles (``makeHighFieldTemplateParticles.py``) are cut with
``particle_utils.cut_plane`` at a *provisional* plane.  Each voxel particle carries
0.125 mm^3; dividing by the 0.5 mm slab thickness gives 0.25 mm^2, an area mass that
can be compared with a 2-D section.

Section side
------------
MERFISH ``X_spatial_raw`` coordinates are micrometres (median nearest-neighbour
spacing ~15).  They are scaled to mm, subclasses are summed into a few classes,
and cells are summed into 0.5 mm bins.  Cell counts become area with one constant
per target, ``a_cell = occupied bin area / number of cells``, so both measures are
in mm^2 while relative cell density is kept.

Two targets are written: the full section (stress test) and the depth-annotated
pia-to-white-matter strip (primary pilot).  The strip export stores depth from pia
(um) as row mass, so its class is the non-zero column and its depth is the row sum.

A preflight report decodes the atlas label with the Allen terminology, checks the
voxel count against Allen's volume table, records coordinate units and spacing, and
fails before writing anything if a check fails.

Usage::

    PYTHONPATH=src:src/config_setup python src/scripts/prepare_atlas_plane_and_section.py \
        --config configs/project_config.yaml --job ahrav2_mtg_coronal_L_to_1194111462
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy.spatial import cKDTree

from config_setup.project_config import load_config, resolve_path
from preprocess.particle_utils import (
    FeatureDefinition,
    bin_particles_2d,
    cut_plane,
    group_feature_columns,
    load_particle_archive,
    save_particle_archive,
)
from preprocess.xmodmap_compat import entropy_numpy, load_xmodmap, write_vtk_xyz

# Reference categorical slots 1-7 (dataviz palette), assigned in class order.
CLASS_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="Project YAML configuration.")
    parser.add_argument("--job", required=True, help="Name under atlas_section_inputs.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing outputs.")
    return parser.parse_args()


def _check(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def _terminology_check(atlas: dict[str, Any], config: dict, failures: list[str]) -> dict[str, Any]:
    """Decode the label with Allen's terminology and compare voxel counts."""

    label_id = int(atlas["label_id"])
    with open(resolve_path(config, atlas["terminology_csv"]), newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream) if row["annotation_value"].strip() == str(label_id)]
    _check(len(rows) == 1, f"terminology has {len(rows)} rows for annotation_value {label_id}", failures)
    term = rows[0] if rows else {"identifier": None, "name": None, "abbreviation": None}

    with open(resolve_path(config, atlas["parcellation_volumes_csv"]), newline="", encoding="utf-8") as stream:
        volumes = {row["identifier"]: row for row in csv.DictReader(stream)}
    allen_count = int(volumes[term["identifier"]]["voxel_count"]) if term["identifier"] in volumes else None

    annotation = np.asanyarray(nib.load(str(resolve_path(config, atlas["annotation_nifti"]))).dataobj)
    local_count = int(np.count_nonzero(annotation == label_id))

    _check(term["name"] == atlas["expected_name"],
           f"code {label_id} decodes to {term['name']!r}, expected {atlas['expected_name']!r}", failures)
    _check(allen_count == local_count,
           f"Allen voxel_count {allen_count} != local voxel count {local_count}", failures)
    return {
        "annotation_value": label_id,
        "identifier": term["identifier"],
        "name": term["name"],
        "abbreviation": term["abbreviation"],
        "allen_voxel_count": allen_count,
        "local_voxel_count": local_count,
    }


def _section_labels(section: dict[str, Any], config: dict) -> tuple[list[str], list[str]]:
    """Feature labels of the full-section NPZ and of the strip NPZ, in column order."""

    catalog = json.loads(resolve_path(config, section["full_labels_json"]).read_text(encoding="utf-8"))
    full_labels = list(catalog["subclass_colors"])  # dict order == NPZ column order
    strip_json = resolve_path(config, section["strip_npz"]).with_suffix(".json")
    strip_labels = list(json.loads(strip_json.read_text(encoding="utf-8"))["features"]["feature_labels"])
    return full_labels, strip_labels


def _compress_section(Z_mm, nu, labels, grouping, bin_mm):
    """Group subclasses, bin, and convert cell counts to area (mm^2)."""

    groups, drop = grouping["groups"], grouping.get("drop", [])
    nu_classes, class_names = group_feature_columns(nu, labels, groups, drop)
    kept_labels = [label for members in groups.values() for label in members]
    nu_subclasses = nu[:, [labels.index(label) for label in kept_labels]]
    keep = nu_classes.sum(axis=1) > 0
    X_bins, (classes_bins, subclasses_bins), cells_per_bin = bin_particles_2d(
        Z_mm[keep], [nu_classes[keep], nu_subclasses[keep]], bin_mm
    )
    occupied_area = len(X_bins) * bin_mm**2
    a_cell = occupied_area / float(keep.sum())
    fine_to_coarse = [class_names.index(name) for name, members in groups.items() for _ in members]
    features = FeatureDefinition(
        fine_ids=tuple(range(len(kept_labels))),
        fine_names=tuple(kept_labels),
        coarse_names=tuple(class_names),
        fine_to_coarse=tuple(fine_to_coarse),
    )
    summary = {
        "number_cells": int(keep.sum()),
        "number_bins": int(len(X_bins)),
        "bin_mm": float(bin_mm),
        "occupied_area_mm2": float(occupied_area),
        "a_cell_mm2": float(a_cell),
        "cells_per_bin_median": float(np.median(cells_per_bin)),
        "class_cell_counts": dict(zip(class_names, classes_bins.sum(axis=0).round().astype(int).tolist())),
        "mass_units": "mm^2 (cell count x a_cell)",
    }
    return X_bins, classes_bins * a_cell, subclasses_bins * a_cell, features, summary


def _qc_figure(path, template, annotation, affine, plane_meta, label_id, plane_X, full_X, full_nu,
               class_names, strip_mm, title):
    """Template slice with label outline, atlas plane, and section bins on equal mm axes."""

    inverse = np.linalg.inv(affine)
    j = int(round((inverse @ np.r_[plane_meta["origin_mm"], 1.0])[1]))
    x_mm = affine[0, 3] + affine[0, 0] * np.arange(template.shape[0])
    z_mm = affine[2, 3] + affine[2, 2] * np.arange(template.shape[2])
    extent = [x_mm[0] - 0.25, x_mm[-1] + 0.25, z_mm[0] - 0.25, z_mm[-1] + 0.25]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.6), constrained_layout=True)
    ax = axes[0]
    ax.imshow(template[:, j, :].T, origin="lower", extent=extent, cmap="gray")
    ax.contour(x_mm, z_mm, (annotation[:, j, :] == label_id).T.astype(float), levels=[0.5],
               colors=[CLASS_COLORS[1]], linewidths=1.0)
    ax.set_title(f"Template, coronal y = {plane_meta['origin_mm'][1]:g} mm\nlabel {label_id} outline; "
                 f"kept hemisphere: {plane_meta['hemisphere']}", fontsize=10)
    ax.set_xlabel("MNI x (mm)")
    ax.set_ylabel("MNI z (mm)")

    # Panels b and c share one span so that 1 mm looks the same in both.
    span = max(np.ptp(plane_X[:, 0]), np.ptp(plane_X[:, 1]), np.ptp(full_X[:, 0]), np.ptp(full_X[:, 1])) + 2.0

    def _frame(axis, X):
        center = 0.5 * (X[:, :2].min(axis=0) + X[:, :2].max(axis=0))
        axis.set_xlim(center[0] - span / 2, center[0] + span / 2)
        axis.set_ylim(center[1] - span / 2, center[1] + span / 2)
        axis.set_aspect("equal")

    ax = axes[1]
    ax.scatter(plane_X[:, 0], plane_X[:, 1], s=6, marker="s", color=CLASS_COLORS[0], linewidths=0)
    _frame(ax, plane_X)
    ax.set_title(f"Atlas plane particles (n = {len(plane_X)}, 0.25 mm² each)", fontsize=10)
    ax.set_xlabel("plane u = MNI x (mm)")
    ax.set_ylabel("plane v = MNI z (mm)")

    ax = axes[2]
    dominant = np.argmax(full_nu, axis=1)
    for index, name in enumerate(class_names):
        sel = dominant == index
        if np.any(sel):
            ax.scatter(full_X[sel, 0], full_X[sel, 1], s=10, marker="s", color=CLASS_COLORS[index],
                       linewidths=0, label=name)
    ax.scatter(strip_mm[:, 0], strip_mm[:, 1], s=0.3, color="#1a1a19", alpha=0.35, linewidths=0,
               label="depth-annotated strip (cells)")
    _frame(ax, full_X)
    ax.set_title("Section, 0.5 mm bins by dominant class", fontsize=10)
    ax.set_xlabel("section x (mm)")
    ax.set_ylabel("section y (mm)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8, frameon=False, markerscale=1.5)
    fig.suptitle(title, fontsize=11)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    jobs = config.get("atlas_section_inputs", {})
    if args.job not in jobs:
        raise KeyError(f"Unknown job '{args.job}'. Available: {', '.join(sorted(jobs)) or '<none>'}")
    job = jobs[args.job]
    atlas, section = job["atlas"], job["section"]
    grouping = config["feature_groupings"][section["feature_grouping"]]
    qc_dir = resolve_path(config, job["qc_dir"])

    outputs = [resolve_path(config, atlas["output_npz"])] + [
        resolve_path(config, section[key]) for key in ("full_output_npz", "strip_output_npz", "strip_cells_npz")
    ]
    existing = [str(path) for path in outputs if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"Outputs exist (use --overwrite): {existing}")

    failures: list[str] = []
    preflight: dict[str, Any] = {"job": args.job}

    # ---- atlas label identity and plane --------------------------------------------
    preflight["terminology"] = _terminology_check(atlas, config, failures)
    atlas_file = resolve_path(config, atlas["particles"])
    X3, nu_sub3, atlas_meta = load_particle_archive(atlas_file, "nu_Sub")
    _, nu_all3, _ = load_particle_archive(atlas_file, "nu_All")
    with np.load(atlas_file, allow_pickle=False) as archive:
        atlas_features = FeatureDefinition(
            fine_ids=tuple(int(v) for v in archive["fine_feature_ids"]),
            fine_names=tuple(str(v) for v in archive["fine_feature_names"]),
            coarse_names=tuple(str(v) for v in archive["coarse_feature_names"]),
            fine_to_coarse=tuple(int(v) for v in archive["fine_to_coarse"]),
        )
    _check(atlas_features.fine_ids == (int(atlas["label_id"]),),
           f"atlas archive features {atlas_features.fine_ids} != label {atlas['label_id']}", failures)
    plane = atlas["plane"]
    plane_X, (plane_sub, plane_all), plane_meta = cut_plane(
        X3, [nu_sub3, nu_all3], plane["origin_mm"], plane["u_axis"], plane["v_axis"],
        plane["half_thickness_mm"], plane.get("hemisphere"),
    )
    plane_meta["status"] = "provisional initialization; not an estimate of the cutting plane"
    plane_area = float(plane_sub.sum())
    preflight["atlas_plane"] = {**plane_meta, "area_mm2": plane_area,
                                "particle_mass_mm2": sorted(set(np.round(plane_sub.sum(axis=1), 6).tolist()))}
    _check(len(plane_X) > 0, "atlas plane is empty", failures)

    # ---- section coordinates and units ----------------------------------------------
    scale = float(section["coordinate_scale"])
    full_labels, strip_labels = _section_labels(section, config)
    with np.load(resolve_path(config, section["full_npz"])) as archive:
        Z_raw, nu_full = archive["Z"].astype(np.float64), archive["nu_Z"].astype(np.float64)
    _check(nu_full.shape[1] == len(full_labels),
           f"full section has {nu_full.shape[1]} columns but {len(full_labels)} labels", failures)
    _check(np.all(Z_raw[:, 2] == 0), "full-section z is not uniformly 0 padding", failures)
    nn_raw = cKDTree(Z_raw[:, :2]).query(Z_raw[:, :2], k=2)[0][:, 1]
    Z_full = Z_raw * scale
    preflight["section_units"] = {
        "raw_min": Z_raw[:, :2].min(axis=0).tolist(),
        "raw_max": Z_raw[:, :2].max(axis=0).tolist(),
        "coordinate_scale": scale,
        "extent_mm": np.ptp(Z_full[:, :2], axis=0).tolist(),
        "median_nearest_neighbour_raw": float(np.median(nn_raw)),
        "median_nearest_neighbour_um_if_scale_is_um_to_mm": float(np.median(nn_raw) * scale * 1000.0),
    }
    # A cortical cell spacing of ~5-50 um is what makes the um->mm reading consistent.
    _check(5.0 <= np.median(nn_raw) * scale * 1000.0 <= 50.0,
           "median nearest-neighbour spacing is not cell-like under the configured scale", failures)

    with np.load(resolve_path(config, section["strip_npz"])) as archive:
        Z_strip_raw, nu_strip_depth = archive["Z"].astype(np.float64), archive["nu_Z"].astype(np.float64)
    nonzero = (nu_strip_depth > 0).sum(axis=1)
    _check(np.all(nonzero == 1), "strip rows are not one-hot times depth", failures)
    depth_um = nu_strip_depth.sum(axis=1)
    nu_strip = (nu_strip_depth > 0).astype(np.float64)
    full_keys = {tuple(row) for row in np.round(Z_raw[:, :2], 3).tolist()}
    in_full = sum(tuple(row) in full_keys for row in np.round(Z_strip_raw[:, :2], 3).tolist())
    _check(in_full == len(Z_strip_raw), f"{len(Z_strip_raw) - in_full} strip cells are not in the full section", failures)
    Z_strip = Z_strip_raw * scale
    preflight["strip"] = {"number_cells": int(len(Z_strip)), "cells_found_in_full_section": int(in_full),
                          "depth_um_min": float(depth_um.min()), "depth_um_max": float(depth_um.max()),
                          "extent_mm": np.ptp(Z_strip[:, :2], axis=0).tolist()}

    # ---- compress both targets (raises on unknown subclass labels) ------------------
    bin_mm = float(section["bin_mm"])
    try:
        full = _compress_section(Z_full, nu_full, full_labels, grouping, bin_mm)
        strip = _compress_section(Z_strip, nu_strip, strip_labels, grouping, bin_mm)
    except (KeyError, ValueError) as error:
        failures.append(f"feature grouping: {error}")
        full = strip = None
    if full is not None:
        preflight["full_section_target"] = full[4]
        preflight["strip_target"] = strip[4]
        preflight["area_ratio_atlas_plane_to_full_section"] = plane_area / full[4]["occupied_area_mm2"]
        preflight["area_ratio_atlas_plane_to_strip"] = plane_area / strip[4]["occupied_area_mm2"]

    qc_dir.mkdir(parents=True, exist_ok=True)
    preflight["failures"] = failures
    (qc_dir / "preflight.json").write_text(json.dumps(preflight, indent=2) + "\n", encoding="utf-8")
    if failures:
        raise SystemExit("Preflight failed:\n  " + "\n  ".join(failures))

    # ---- write archives -------------------------------------------------------------
    api = load_xmodmap(config)
    vtk_writer = lambda points, arrays, names, filename: write_vtk_xyz(api, points, arrays, names, filename)
    entropy_function = lambda nu: entropy_numpy(api, nu)

    save_particle_archive(
        outputs[0], plane_X, plane_sub, plane_all, atlas_features,
        {
            "particle_role": "source_atlas_plane",
            "job_name": args.job,
            "source_particles": str(atlas_file),
            "source_metadata": atlas_meta,
            "coordinate_space": "mni_icbm2009b_symmetric_plane",
            "plane": plane_meta,
            "terminology": preflight["terminology"],
            "mass_units": "mm^2 (voxel volume / slab thickness)",
        },
        vtk_writer=vtk_writer, entropy_function=entropy_function,
    )
    for (X_bins, nu_classes, nu_subclasses, features, summary), output, role, source in (
        (full, outputs[1], "target_section_full", section["full_npz"]),
        (strip, outputs[2], "target_section_depth_strip", section["strip_npz"]),
    ):
        save_particle_archive(
            output, X_bins, nu_classes, nu_subclasses, features,
            {
                "particle_role": role,
                "job_name": args.job,
                "source_npz": str(resolve_path(config, source)),
                "coordinate_space": "seaad_section_mm",
                "coordinate_scale_from_raw": scale,
                "raw_units": "um (X_spatial_raw)",
                "feature_grouping": section["feature_grouping"],
                **summary,
            },
            vtk_writer=vtk_writer, entropy_function=entropy_function,
        )
    np.savez_compressed(outputs[3], X=Z_strip.astype(np.float32), depth_from_pia_um=depth_um.astype(np.float32),
                        subclass=np.argmax(nu_strip, axis=1), subclass_names=np.asarray(strip_labels, dtype=str))

    # ---- QC figure ------------------------------------------------------------------
    template_image = nib.load(str(resolve_path(config, atlas["template_nifti"])))
    annotation_image = nib.load(str(resolve_path(config, atlas["annotation_nifti"])))
    if not np.allclose(template_image.affine, annotation_image.affine):
        raise ValueError("Template and annotation grids differ; the QC overlay would be misplaced.")
    _qc_figure(
        qc_dir / "atlas_plane_and_section.png",
        np.asanyarray(template_image.dataobj), np.asanyarray(annotation_image.dataobj), annotation_image.affine,
        plane_meta, int(atlas["label_id"]), plane_X, full[0], full[1], list(full[3].coarse_names), Z_strip,
        f"{args.job}: {preflight['terminology']['name']} ({preflight['terminology']['identifier']})",
    )
    print(json.dumps({key: preflight[key] for key in (
        "terminology", "section_units", "area_ratio_atlas_plane_to_full_section", "area_ratio_atlas_plane_to_strip")},
        indent=2))
    print(f"Wrote {[str(p) for p in outputs]} and QC in {qc_dir}")


if __name__ == "__main__":
    main()
