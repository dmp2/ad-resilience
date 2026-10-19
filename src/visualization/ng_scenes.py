"""Neuroglancer bundles for the two pilots, built from ``scene`` layers.

Every layer keeps its original coordinates; placement into the scene frame is a
recorded per-layer source transform (see ``ng_export.source_transform``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from . import ng_export as nx
from . import scene as S
from .style import CAT3, GRID, MUTED, ORDINAL5, TEXT_2, BLUE_RAMP

QNP_STAIN_ORDER = ("NEUN", "LFB", "AT", "ASYN", "I6", "GFAP")
STAIN_LABEL = {"NEUN": "NeuN", "LFB": "LFB", "AT": "AT8+pTDP43", "ASYN": "a-syn", "I6": "Iba1+6E10", "GFAP": "GFAP"}
ATLAS_COLORS = {"MTG": CAT3[0], "FuG": CAT3[1], "ITG": CAT3[2], "STG": "#d6d5ce", "other_gray_matter": "#ecebe6",
                "white_matter": TEXT_2}
HYPOTHESES = {"A": ("p+000_w+000_y-027.5", "results/registrations/ahrav2_tissue_plane_to_qnp_H20.33.034_MTG_A/init_r000_f0",
                    "MTG basin (provisional)"),
              "B": ("p+015_w+015_y-062.5", "results/registrations/ahrav2_tissue_plane_to_qnp_H20.33.034_MTG_B/init_r000_f0",
                    "fusiform basin (alternative)")}


def _closed(ring: np.ndarray) -> np.ndarray:
    return ring if np.allclose(ring[0], ring[-1]) else np.vstack([ring, ring[:1]])


def _polyline_layer_asset(polygons: S.Polygons, out_dir: Path, scale_to_um: float) -> dict:
    rings = [_closed(r) * scale_to_um for r in polygons.rings]
    return nx.write_annotations(out_dir, "polyline", rings,
                                {"layer": ("uint8", polygons.labels.astype(np.uint8)),
                                 "hole": ("uint8", polygons.holes.astype(np.uint8))},
                                enums={"layer": polygons.label_names})


# --------------------------------------------------------------------------------------
# QNP
# --------------------------------------------------------------------------------------

def build_qnp(config: Mapping, job_name: str = "H20.33.034_MTG", level: int = 1, overwrite: bool = False) -> dict:
    job = S.qnp_job(config, job_name)
    bundle = nx.SERVE_ROOT / "qnp" / "neuroglancer"
    data = bundle / "data" / job_name
    transforms = S.load_qnp_transforms(job)
    reference = job["reference_stain"]
    misfit = set(job.get("annotation_misfit", []))
    images, halos, manifest_layers, frames = [], [], {}, {}
    for stain in QNP_STAIN_ORDER:
        svs = job["_slides"][stain]
        record = nx.write_slide_zarr(svs, data / "images" / f"{svs.stem}.ome.zarr", level=level, overwrite=overwrite)
        T = transforms[stain]
        unreliable = T.status.startswith("unreliable")
        label = STAIN_LABEL[stain]
        name = f"{label} [{svs.stem}]" + (" - alignment unreliable, native frame" if unreliable else "")
        # An unreliable within-QNP edge is not applied: the slide stays in its own frame.
        T_um = None if (stain == reference or unreliable) else T.in_units(1000.0, "um")
        images.append(nx.image_layer(name, data / "images" / f"{svs.stem}.ome.zarr", T_um, visible=stain == reference))
        manifest_layers[name] = {"status": "measured" if T_um is None else "transformed", "source": record,
                                 "transform_to_scene": (T_um.describe() if T_um is not None else
                                                        ("identity (reference)" if stain == reference else
                                                         f"not applied: {T.status}"))}
        frames[f"qnp_slide_mm:{svs.stem}"] = record["frame"]
        annotation = svs.with_suffix(".annotations")
        if annotation.exists():
            polygons = S.load_halo_polygons(annotation, record["mpp_baseline_um"])
            asset = _polyline_layer_asset(polygons, data / "annotations" / f"{svs.stem}_halo_layers", 1000.0)
            flag = " - published annotations misfit this slide" if stain in misfit else ""
            hname = f"{label} HALO layers{flag}"
            halos.append(nx.annotation_layer(hname, data / "annotations" / f"{svs.stem}_halo_layers",
                                             nx.categorical_shader("layer", ORDINAL5, size=0, line_width=1.5),
                                             T_um, visible=stain == reference))
            manifest_layers[hname] = {**asset, "status": "measured" if T_um is None else "transformed",
                                      "source": polygons.provenance, "layer_names": list(polygons.label_names)}

    # Atlas pose hypotheses: resliced AHRAv2 classes on each plane, placed by the scan's
    # rigid similarity (inverse: plane -> qnp_local).  Neither is independently verified.
    atlas_layers = {}
    for key, (plane_id, fit_dir, description) in HYPOTHESES.items():
        hyp = S.load_plane_hypothesis(job, plane_id, S.PROJECT / fit_dir)
        atlas = S.atlas_plane_classes(config, job, hyp["plane"])
        classes = atlas["classes"]
        names = atlas["class_names"]
        vol_dir = data / "atlas" / f"ahrav2_classes_{plane_id}"
        asset = nx.write_label_volume(vol_dir, classes.array.astype(np.int32) + 1, (classes.px[0] * 1e3, classes.px[1] * 1e3))
        # voxel i covers [i, i+1) px in source um; pixel i centre = origin + i * px in plane mm
        vox_to_plane = np.eye(3)
        vox_to_plane[0, 2] = classes.origin[0] - 0.5 * classes.px[0]
        vox_to_plane[1, 2] = classes.origin[1] - 0.5 * classes.px[1]
        src = S.Affine2D(vox_to_plane, f"atlas_volume_mm:{plane_id}", hyp["frame"], "translation", "exact",
                         "voxel origin of the resliced grid")
        to_local = src.then(hyp["local_to_plane"].inverse())
        lname = f"AHRAv2 pose {key}: {description} - rigid scan pose, not verified"
        colors = {i + 1: ATLAS_COLORS[n] for i, n in enumerate(names)}
        shown = [i + 1 for i, n in enumerate(names) if n != "other_gray_matter"]
        atlas_layers[key] = nx.segmentation_layer(lname, vol_dir, colors, to_local.in_units(1000.0, "um"),
                                                  visible=True, opacity=0.35, shown=shown)
        manifest_layers[lname] = {**asset, "status": "transformed (hypothesis)", "class_names": names,
                                  "colors": {n: ATLAS_COLORS[n] for n in names}, "plane": hyp["plane"],
                                  "transform_to_scene": to_local.in_units(1000.0, "um").describe(),
                                  "landing_after_xiv": hyp["evaluation"]["ribbon_landing_after_xiv"]["all"],
                                  "note": "xIV's nonrigid deformation is not an affine map and is not shown here"}

    neun = S.svs_geometry(job["_slides"][reference])
    h, w = neun["level_shapes"][0]
    centre = (w * neun["mpp_um"] / 2, h * neun["mpp_um"] / 2)
    common = {"job": job_name, "donor": job_name.split("_")[0], "scene_frame": "qnp_local_mm in um "
              "(= NeuN slide; x right, y down; block orientation undocumented)", "frames": frames,
              "wsi_pipeline": nx.wsi_provenance(), "neuroglancer": nx.neuroglancer_version(),
              "aperio_level": level, "layers": manifest_layers}
    scenes = {}
    multistain = images + halos
    scenes["qnp_multistain"] = nx.write_scene(bundle, f"{job_name}_multistain", nx.build_state(multistain, centre, 30000),
                                              {**common, "description": "Six QNP stains of one block in the NeuN "
                                               "frame via the within-QNP rigid transforms; HALO layers per stain."})
    for key in HYPOTHESES:
        layers = [images[0], atlas_layers[key], halos[0]]
        scenes[f"qnp_atlas_{key}"] = nx.write_scene(
            bundle, f"{job_name}_atlas_pose_{key}", nx.build_state(layers, centre, 30000, extra_dims={"z": [1e-6, "m"]}),
            {**common, "description": f"NeuN + HALO ribbon with AHRAv2 classes under pose hypothesis {key} "
             f"({HYPOTHESES[key][2]}). Both poses tie within stain noise (REPORT.md); neither is validated."})
    return scenes


# --------------------------------------------------------------------------------------
# MERFISH
# --------------------------------------------------------------------------------------

def build_omics(config: Mapping, cells: S.Points, polygons: S.Polygons, poly_to_section: S.Affine2D,
                xiv: Mapping[str, Any], genes: Sequence[str], bins_mm: Sequence[float] = (0.1, 0.25, 0.5, 1.0)) -> dict:
    barcode = cells.meta["barcode"]
    bundle = nx.SERVE_ROOT / "omics" / "neuroglancer"
    data = bundle / f"data/{cells.meta['donor']}_{barcode}"
    manifest_layers = {}
    xy_um = cells.xy * 1000.0  # = X_spatial_raw (original um)
    props = {"class3": ("uint8", np.where(cells.categories["class3"].codes < 0, 255, cells.categories["class3"].codes)),
             "class": ("uint8", np.where(cells.categories["class"].codes < 0, 255, cells.categories["class"].codes)),
             "layer": ("uint8", np.where(cells.categories["layer_annotation"].codes < 0, 255,
                                         cells.categories["layer_annotation"].codes)),
             "depth_um": ("float32", np.nan_to_num(cells.scalars["depth_from_pia_um"].values, nan=-1.0))}
    for g in genes:
        props[nx.property_id(g, "g_")] = ("float32", cells.scalars[f"lognorm:{g}"].values)
    asset = nx.write_annotations(data / "cells", "point", xy_um, props,
                                 enums={"class3": cells.categories["class3"].names, "class": cells.categories["class"].names,
                                        "layer": cells.categories["layer_annotation"].names})
    cells_dir = data / "cells"
    layers = [nx.annotation_layer("cells - class (3 groups)", cells_dir, nx.categorical_shader("class3", CAT3, size=3))]
    manifest_layers["cells - class (3 groups)"] = {**asset, "status": "measured", "source": cells.provenance,
                                                   "colors": dict(zip(cells.categories["class3"].names, CAT3))}
    for g in genes:
        values = cells.scalars[f"lognorm:{g}"].values
        hi = float(np.percentile(values, 99))
        name = f"cells - {g} ln(1+spots/10K)"
        layers.append(nx.annotation_layer(name, cells_dir, nx.sequential_shader(nx.property_id(g, "g_"), 0.0,
                                                                                hi, BLUE_RAMP), visible=False))
        manifest_layers[name] = {**asset, "status": "measured", "display_range": [0.0, hi],
                                 "units": cells.scalars[f"lognorm:{g}"].description}
    layers.append(nx.annotation_layer("cells - provider layer annotation", cells_dir,
                                      nx.categorical_shader("layer", ORDINAL5, missing="#e4e3dd", size=3), visible=False))
    manifest_layers["cells - provider layer annotation"] = {**asset, "status": "measured"}

    poly_asset = _polyline_layer_asset(polygons, data / "provider_layer_polygons", 1.0)
    # The source frame is already um; only the target (section mm) is rescaled to um.
    T_um = S.Affine2D(np.diag([1000.0, 1000.0, 1.0]) @ poly_to_section.matrix, "merfish_layer_polygon_um",
                      "merfish_section_um", poly_to_section.kind, poly_to_section.status, poly_to_section.provenance)
    pname = "provider layer polygons (uns/layer_polygons)"
    layers.append(nx.annotation_layer(pname, data / "provider_layer_polygons",
                                      nx.categorical_shader("layer", ORDINAL5, size=0, line_width=2.0), T_um))
    manifest_layers[pname] = {**poly_asset, "status": "transformed (exact axis permutation)",
                              "transform_to_scene": T_um.describe(), "source": polygons.provenance}

    # Scale space: dominant 3-group class per bin, one label volume per bin size.
    for bin_mm in bins_mm:
        binned = S.bin_merfish(cells, bin_mm)
        grid = binned.meta["grid_cells"]
        nu3 = np.stack([binned.meta["nu_class"][:, [binned.meta["class_names"].index(c) for c in members]].sum(1)
                        for members in S.CLASS3.values()], 1)
        g0 = grid.min(0)
        shape = grid.max(0) - g0 + 1
        labels = np.zeros((shape[1], shape[0]), np.uint32)
        labels[grid[:, 1] - g0[1], grid[:, 0] - g0[0]] = nu3.argmax(1) + 1
        vol = data / f"bins_{int(round(bin_mm * 1000))}um"
        vasset = nx.write_label_volume(vol, labels, (bin_mm * 1000.0, bin_mm * 1000.0))
        to_section = S.Affine2D(np.array([[1, 0, g0[0] * bin_mm * 1000], [0, 1, g0[1] * bin_mm * 1000], [0, 0, 1.0]]),
                                "bin_volume_um", "merfish_section_um", "translation", "exact", "bin grid origin")
        vname = f"bins {bin_mm * 1000:g} um - dominant class (derived)"
        layers.append(nx.segmentation_layer(vname, vol, {1: CAT3[0], 2: CAT3[1], 3: CAT3[2]}, to_section,
                                            visible=False, opacity=0.6))
        manifest_layers[vname] = {**vasset, "status": "derived", "source": binned.provenance,
                                  "transform_to_scene": to_section.describe(), "n_bins": int(len(grid))}

    # Atlas context: estimated correspondence only (MTG-only xIV fit, under-determined).
    for key, label, colour in (("rigid", "initial (barycentre) placement", MUTED), ("deformed", "after xIV (estimate)", "#e34948")):
        pts = xiv[key]
        aname = f"AHRAv2 MTG plane particles - {label}"
        a_asset = nx.write_annotations(data / f"atlas_mtg_{key}", "point", pts.xy * 1000.0,
                                       {"kind": ("uint8", np.zeros(len(pts.xy), np.uint8))})
        layers.append(nx.annotation_layer(aname, data / f"atlas_mtg_{key}", nx.categorical_shader("kind", [colour], size=3, alpha=0.55),
                                          visible=key == "deformed"))
        manifest_layers[aname] = {**a_asset, "status": pts.status, "source": pts.provenance,
                                  "caveat": "MTG-only fit targeting the 50-bin depth strip; placement under-determined"}

    centre = np.percentile(xy_um, [50], axis=0)[0]
    span = float(np.ptp(xy_um, axis=0).max()) * 1.1
    common = {"donor": cells.meta["donor"], "section": cells.meta["section"], "barcode": barcode,
              "scene_frame": "merfish_section_um = obsm/X_spatial_raw (um, as stored; anatomical orientation not "
                             "recorded by the provider)", "wsi_pipeline": nx.wsi_provenance(),
              "neuroglancer": nx.neuroglancer_version(), "x_check": cells.meta["x_check"], "layers": manifest_layers}
    state = nx.build_state(layers, centre, span, extra_dims={"z": [1e-6, "m"]})
    return {"omics_cells": nx.write_scene(bundle, f"{cells.meta['donor']}_{barcode}_cells", state,
                                          {**common, "description": "MERFISH cells (class / genes / layer), provider "
                                           "layer polygons, binned scale space, and xIV atlas context (estimate)."})}
