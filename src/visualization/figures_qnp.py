"""QNP figure families for H20.33.034 MTG (existing derivatives only).

A  scale space        same NeuN section: histology -> 32 um p(WM) map -> 500 / 1000 um particles
B  feature maps       provider per-layer measurements drawn on the NeuN HALO layer raster, and
                      the structural gray/white feature across the six stains in the NeuN frame
C  correspondence     within-QNP HALO-ribbon check, and the two tied AHRAv2 pose hypotheses
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from . import scene as S
from . import style as st
from .ng_scenes import ATLAS_COLORS, HYPOTHESES, STAIN_LABEL

LAYER_MEASURES = (  # (table column, short label)
    ("percent NeuN positive area", "NeuN+"),
    ("percent AT8 positive area", "AT8+"),
    ("percent GFAP positive area", "GFAP+"),
    ("percent Iba1 positive area", "Iba1+"),
    ("percent 6e10 positive area", "6E10+"),
)
MEASURE_STAIN = {"AT8+": "AT", "GFAP+": "GFAP"}


def _summary(job: Mapping) -> dict:
    return json.loads((S.PROJECT / job["qc_dir"] / "particles_summary.json").read_text())


def _mask_extent(raster: S.Raster, pad: float = 0.6) -> list[float]:
    rows, cols = np.nonzero(raster.array)
    xs = raster.origin[0] + cols * raster.px[0]
    ys = raster.origin[1] + rows * raster.px[1]
    return [xs.min() - pad, xs.max() + pad, ys.min() - pad, ys.max() + pad]


def _particle_squares(ax, points: S.Points, bin_mm: float, values: np.ndarray, cmap, norm, mass_ref: float):
    """Each particle as a square at its position, area proportional to its mass (capped at the bin)."""

    mass = points.scalars["area_mm2"].values
    if points.meta.get("compression_method") == "image_varifold":
        # Marker area is a mass encoding in points², never a physical tissue parcel.
        return ax.scatter(*points.xy.T, s=12 * mass / mass_ref, c=values, cmap=cmap,
                          norm=norm, linewidths=0, rasterized=True)
    side = bin_mm * np.sqrt(np.clip(mass / mass_ref, 0, 1))
    patches = [Rectangle((x - s / 2, y - s / 2), s, s) for (x, y), s in zip(points.xy, side)]
    coll = PatchCollection(patches, cmap=cmap, norm=norm, linewidths=0, rasterized=True)
    coll.set_array(values)
    ax.add_collection(coll)
    return coll


def _hbar(fig, mappable, rect, label, ticks=None):
    cax = fig.add_axes(rect)
    cb = fig.colorbar(mappable, cax=cax, orientation="horizontal")
    cb.set_label(label, fontsize=5.5, labelpad=2)
    cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(labelsize=5, width=0.4, length=2, pad=1)
    if ticks is not None:
        cb.set_ticks(list(ticks))
    return cb


# --------------------------------------------------------------------------------------
# A - scale space
# --------------------------------------------------------------------------------------

def figure_a(config: Mapping, job_name: str, out_dir: Path, representation: dict | None = None) -> list[Path]:
    job = S.qnp_job(config, job_name)
    stain = job["reference_stain"]
    slide = job["_slides"][stain]
    maps = S.load_qnp_map(job, stain)
    summary = _summary(job)[stain]["particles"]
    frame = S.slide_frame(slide)
    extent = _mask_extent(maps["mask"])
    norm = Normalize(0, 1)

    fig = st.figure(st.FULL_WIDTH_MM, 72)
    axes = [fig.add_axes([0.01 + i * 0.247, 0.27, 0.235, 0.58]) for i in range(4)]
    axes[0].imshow(maps["rgb"].array, extent=maps["rgb"].extent, interpolation="antialiased", rasterized=True)
    st.panel_title(axes[0], "a", "NeuN histology, 32 µm block mean", "measured")
    p = maps["p_wm"].array.astype(float)
    p[~maps["mask"].array.astype(bool)] = np.nan
    im = axes[1].imshow(p, extent=maps["p_wm"].extent, cmap=st.SEQ_BLUE, norm=norm, rasterized=True)
    st.panel_title(axes[1], "b", "p(white matter), 32 µm pixels",
                   "fixed surrogate reference" if representation else "derived")
    for ax, letter, tag in ((axes[2], "c", "500um"), (axes[3], "d", "1000um")):
        points = (representation["points"][tag] if representation else
                  S.load_qnp_particles(job["_particle_dir"] / f"{slide.stem}_tissue_classes_{tag}.npz", frame))
        bin_mm = summary[tag]["bin_mm_actual"]
        _particle_squares(ax, points, bin_mm, points.scalars[representation.get("scalar", "white_fraction") if representation else "white_fraction"].values, st.SEQ_BLUE, norm, bin_mm**2)
        title = f"{len(points.xy):,} particles, {bin_mm * 1000:.0f} µm bins"
        status = "derived"
        if representation:
            title = f"{len(points.xy):,} particles: " + ("500 µm reference" if tag == "500um" else representation["variant"])
            status = "deterministic reference" if tag == "500um" else "optimized XY and nonnegative masses"
            title = representation.get("titles", {}).get(tag, title)
            status = representation.get("statuses", {}).get(tag, status)
        st.panel_title(ax, letter, title, status)
    for ax in axes:
        st.map_axes(ax, extent)
        st.scale_bar(ax, 5, "5 mm")
    st.axes_glyph(axes[0], "+x", "+y", y_down=True)
    label = (representation["color_label"] if representation else
             "white-matter fraction: p(WM) per pixel (b); white area / tissue area per particle (c, d)")
    _hbar(fig, im, [0.30, representation.get("colorbar_y", 0.20) if representation else 0.17, 0.40, 0.022], label, ticks=[0, 0.25, 0.5, 0.75, 1])
    st.caption(fig, representation["caption"] if representation else [
        f"{slide.stem} in its own frame {frame}: x right, y down, origin at the top-left baseline pixel; the block's "
        "anatomical orientation is undocumented.",
        "c, d: binned soft gray/white area masses (prepare_qnp_section_particles); each square sits at the particle "
        "position with area proportional to its tissue area (capped at the bin).",
        "These are deterministic binnings of the 32 µm membership, not varifold-optimised representations. Total "
        "tissue area is identical at both scales (390.90 mm²)."])
    return st.save(fig, out_dir / f"figA_scale_space_{job_name}")


# --------------------------------------------------------------------------------------
# B - feature maps
# --------------------------------------------------------------------------------------

def _spread_labels(y: np.ndarray, gap: float) -> np.ndarray:
    order = np.argsort(y)
    out = y.astype(float).copy()
    for i in range(1, len(order)):
        out[order[i]] = max(out[order[i]], out[order[i - 1]] + gap)
    return out


def figure_b(config: Mapping, job_name: str, out_dir: Path, representation: dict | None = None, consensus_title: str | None = None) -> list[Path]:
    job = S.qnp_job(config, job_name)
    donor = job_name.split("_")[0]
    stain = job["reference_stain"]
    maps = S.load_qnp_map(job, stain)
    layers = maps["layers"]
    measures = S.load_qnp_layer_measures(config, donor, [m for m, _ in LAYER_MEASURES])
    misfit = set(job.get("annotation_misfit", []))
    extent = _mask_extent(layers, pad=0.4)

    fig = st.figure(st.FULL_WIDTH_MM, 150)
    w = 0.18
    for i, (column, label) in enumerate(LAYER_MEASURES):
        ax = fig.add_axes([0.012 + i * (w + 0.018), 0.62, w, 0.30])
        vals = np.array([measures[column][l] for l in S.QNP_LAYERS])
        norm = Normalize(0, vals.max())
        image = np.full(layers.array.shape, np.nan)
        inside = layers.array > 0
        image[inside] = vals[layers.array[inside].astype(int) - 1]
        ax.imshow(image, extent=layers.extent, cmap=st.SEQ_BLUE, norm=norm, rasterized=True)
        st.map_axes(ax, extent)
        st.panel_title(ax, "abcde"[i], f"{label} area, % of layer", "measured")
        if i == 0:
            st.scale_bar(ax, 5, "5 mm")
            st.axes_glyph(ax, "+x", "+y", y_down=True, loc=(0.72, 0.80))
        box = ax.get_position()
        _hbar(fig, ScalarMappable(norm=norm, cmap=st.SEQ_BLUE), [box.x0 + 0.02, 0.585, box.width - 0.04, 0.012],
              "provider layer mean (%)", ticks=[0, round(vals.max(), 3)])
        if MEASURE_STAIN.get(label) in misfit:
            ax.text(0.5, -0.02, "published annotations misfit this\nstain's own slide (inventory)",
                    transform=ax.transAxes, ha="center", va="top", fontsize=5, color=st.TEXT_2)

    # f: between-layer profiles, each measure relative to its own maximum (units differ)
    ax = fig.add_axes([0.065, 0.17, 0.22, 0.30])
    colors = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")  # categorical slots 1-5 (adjacent-safe)
    xs = np.arange(len(S.QNP_LAYERS))
    ends = []
    for (column, label), colour in zip(LAYER_MEASURES, colors):
        y = np.array([measures[column][l] for l in S.QNP_LAYERS])
        y = y / y.max()
        ax.plot(xs, y, "-o", color=colour, lw=1.2, ms=3, mec="white", mew=0.5)
        ends.append(y[-1])
    for (column, label), y_lab in zip(LAYER_MEASURES, _spread_labels(np.array(ends), 0.075)):
        ax.text(xs[-1] + 0.15, y_lab, label, fontsize=5.5, va="center", color=st.TEXT)
    ax.set_xticks(xs, S.QNP_LAYERS)
    ax.set_xlim(-0.3, len(xs) - 0.1)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("layer mean / largest layer mean")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=st.GRID, lw=0.3)
    st.panel_title(ax, "f", "Between-layer profiles", "measured (layer means)")

    # g-l: structural gray/white feature across stains, all resampled into the NeuN frame
    ms = dict(np.load(job["_aligned_dir"] / "multistain_maps_32um_local.npz"))
    transforms = S.load_qnp_transforms(job)
    stains = [str(s) for s in ms["stains"]]
    origin, px = ms["origin_mm"], ms["px_mm"]
    h, wd = ms["consensus_mask"].shape
    ext = [origin[0] - 0.5 * px[1], origin[0] + (wd - 0.5) * px[1], origin[1] + (h - 0.5) * px[0], origin[1] - 0.5 * px[0]]
    crop = _mask_extent(S.Raster("", "", ms["consensus_mask"], (origin[0], origin[1]), (px[1], px[0]), "derived", ""), 0.3)
    sep = {s: _summary(job)[s]["membership"]["otsu_separability"] for s in stains}
    norm = Normalize(0, 1)
    gw, gh = 0.125, 0.155
    for i, s in enumerate(stains):
        ax = fig.add_axes([0.355 + (i % 3) * (gw + 0.012), 0.345 - (i // 3) * (gh + 0.06), gw, gh])
        p = ms["p_wm"][i].astype(float)
        p[~ms["tissue"][i]] = np.nan
        if representation:
            points = representation["aligned_points"][s]
            im = _particle_squares(ax, points, 1.0, points.scalars[representation["scalar"]].values,
                                   st.SEQ_BLUE, norm, 1.0)
        else:
            im = ax.imshow(p, extent=ext, cmap=st.SEQ_BLUE, norm=norm, rasterized=True)
        st.map_axes(ax, crop)
        role = "structural" if job["stains"][s].get("structural", False) else "feature only"
        if transforms[s].status.startswith("unreliable"):
            role = "alignment unreliable"
        title = f"{STAIN_LABEL[s]}, sep. {sep[s]:.2f}"
        if representation:
            title = STAIN_LABEL[s] + (": OD" if representation["variant"] == "iv_modality" else ": surrogate")
            role = "OD; fixed rigid" if representation["variant"] == "iv_modality" else "IV surrogate; fixed rigid"
            if transforms[s].status.startswith("unreliable"):
                role = "native; unreliable"
        st.panel_title(ax, "ghijkl"[i], title, role)
        if i == 0:
            st.scale_bar(ax, 5, "5 mm")
    _hbar(fig, im, [0.37, 0.115, 0.38, 0.012],
          representation["color_label_b"] if representation else
          "p(white matter) in the NeuN frame (32 µm); sep. = Otsu separability", ticks=[0, 0.5, 1])
    ax = fig.add_axes([0.79, 0.19, 0.19, 0.28])
    sd = ms["structural_disagreement_sd"].astype(float)
    sd[~ms["consensus_mask"]] = np.nan
    im2 = ax.imshow(sd, extent=ext, cmap=st.SEQ_ORANGE, vmin=0, vmax=0.5, rasterized=True)
    st.map_axes(ax, crop)
    st.panel_title(ax, "m", consensus_title or "SD of p(WM): NeuN, LFB, AT8",
                   "fixed structural reference" if representation else "derived")
    _hbar(fig, im2, [0.80, 0.155, 0.17, 0.012], "across-stain SD", ticks=[0, 0.25, 0.5])
    st.caption(fig, representation["caption_b"] if representation else [
        f"a-e: {S.QNP_TABLE_SOURCE}.",
        "HALO layer means measured on each stain's own slide, painted on the NeuN HALO layer raster (32 µm); each "
        "panel has its own scale. The release gives one mean per layer,",
        "so within-layer variation cannot be shown. g-m: frame qnp_local_mm via the within-QNP rigid transforms "
        "(transforms.json; GFAP's is unreliable).",
        "No mutual-information or other informativeness ranking is computed; f is descriptive."])
    return st.save(fig, out_dir / f"figB_feature_maps_{job_name}")


# --------------------------------------------------------------------------------------
# C - correspondence
# --------------------------------------------------------------------------------------

def ribbon_mask(polygons: S.Polygons, transform: S.Affine2D | None, grid_origin, px, shape) -> np.ndarray:
    """Rasterize the polygon union (holes removed) on a local-frame grid with PIL (as qnp_particles)."""

    from PIL import Image, ImageDraw

    canvas = Image.new("1", (shape[1], shape[0]), 0)
    draw = ImageDraw.Draw(canvas)
    for i in np.argsort(polygons.holes):
        ring = polygons.rings[i] if transform is None else transform.apply(polygons.rings[i])
        cols = (ring[:, 0] - grid_origin[0]) / px[1]
        rows = (ring[:, 1] - grid_origin[1]) / px[0]
        draw.polygon(list(zip(cols.tolist(), rows.tolist())), fill=0 if polygons.holes[i] else 1)
    return np.asarray(canvas, dtype=bool)


def ribbon_dice_checks(job: Mapping) -> dict:
    transforms = S.load_qnp_transforms(job)
    record = json.loads((job["_aligned_dir"] / "transforms.json").read_text())
    ref_slide = job["_slides"][job["reference_stain"]]
    ref_poly = S.load_halo_polygons(ref_slide.with_suffix(".annotations"), S.svs_geometry(ref_slide)["mpp_um"])
    ms = dict(np.load(job["_aligned_dir"] / "multistain_maps_32um_local.npz"))
    origin, px, shape = ms["origin_mm"], ms["px_mm"], ms["consensus_mask"].shape
    ref_mask = ribbon_mask(ref_poly, None, origin, px, shape)
    dice = lambda a, b: float(2 * (a & b).sum() / max(a.sum() + b.sum(), 1))  # noqa: E731
    out = {"_reference": ref_poly}
    for stain in ("ASYN", "I6"):
        slide = job["_slides"][stain]
        poly = S.load_halo_polygons(slide.with_suffix(".annotations"), S.svs_geometry(slide)["mpp_um"])
        out[stain] = {"polygons": poly, "transform": transforms[stain],
                      "dice_transformed": dice(ribbon_mask(poly, transforms[stain], origin, px, shape), ref_mask),
                      "dice_untransformed": dice(ribbon_mask(poly, None, origin, px, shape), ref_mask),
                      "existing_qc": record["edges"][stain]["independent_checks"]}
    return out


def figure_c(config: Mapping, job_name: str, out_dir: Path, representation: dict | None = None) -> list[Path]:
    job = S.qnp_job(config, job_name)
    checks = ribbon_dice_checks(job)
    ref_poly = checks.pop("_reference")
    fig = st.figure(st.FULL_WIDTH_MM, 145)
    allx = np.vstack(ref_poly.rings)
    extent = [allx[:, 0].min() - 0.8, allx[:, 0].max() + 0.8, allx[:, 1].min() - 0.8, allx[:, 1].max() + 0.8]
    for i, stain in enumerate(("ASYN", "I6")):
        c = checks[stain]
        ax = fig.add_axes([0.01 + i * 0.27, 0.55, 0.25, 0.37])
        ax.add_collection(st.polygon_patches([r for r, h in zip(ref_poly.rings, ref_poly.holes) if not h],
                                             facecolor="#d6d5ce", edgecolor="none", rasterized=True))
        for ring, hole in zip(c["polygons"].rings, c["polygons"].holes):
            if not hole:
                ax.plot(*ring.T, color=st.MUTED, lw=0.4, ls=(0, (2, 2)))
                ax.plot(*c["transform"].apply(ring).T, color=st.CAT3[1], lw=0.6)
        st.map_axes(ax, extent)
        st.scale_bar(ax, 5, "5 mm")
        st.panel_title(ax, "ab"[i], f"{STAIN_LABEL[stain]} HALO layers in the NeuN frame",
                       f"Dice vs NeuN ribbon {c['dice_transformed']:.3f} (untransformed {c['dice_untransformed']:.3f})")
        if i == 0:
            st.axes_glyph(ax, "+x", "+y", y_down=True, loc=(0.80, 0.82))
    handles = [Patch(facecolor="#d6d5ce", label="NeuN HALO ribbon (reference)"),
               Line2D([], [], color=st.MUTED, lw=0.6, ls=(0, (2, 2)), label="stain's HALO layers, own slide frame"),
               Line2D([], [], color=st.CAT3[1], lw=0.8, label="same layers after the within-QNP rigid transform")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.56, 0.90), frameon=False, fontsize=5.5)
    q = {s: checks[s]["existing_qc"] for s in checks}
    fig.text(0.565, 0.79, "Registration QC (independent labels). The HALO ribbons are never\n"
             "seen by the within-QNP fits, so their overlap with the NeuN ribbon\n"
             "checks the rigid transforms. Recomputed here at 32 µm; existing QC\n"
             f"(transforms.json): a-syn {q['ASYN']['halo_ribbon_dice_vs_reference']:.3f} vs "
             f"{q['ASYN']['halo_ribbon_dice_untransformed']:.3f}, Iba1+6E10 "
             f"{q['I6']['halo_ribbon_dice_vs_reference']:.3f} vs {q['I6']['halo_ribbon_dice_untransformed']:.3f}.",
             fontsize=5.5, color=st.TEXT_2, va="top", linespacing=1.3)

    # c, d: the two pose hypotheses, same window size and scale
    half = 17.0
    sc = None
    planes = {}
    for i, key in enumerate(("A", "B")):
        plane_id, fit_dir, description = HYPOTHESES[key]
        hyp = S.load_plane_hypothesis(job, plane_id, S.PROJECT / fit_dir)
        atlas = S.atlas_plane_classes(config, job, hyp["plane"])
        planes[key] = hyp["plane"]
        ax = fig.add_axes([0.02 + i * 0.30, 0.08, 0.28, 0.38])
        t1 = atlas["t1"]
        ext = t1.extent
        ax.imshow(t1.array, extent=[ext[0], ext[1], ext[3], ext[2]], origin="lower", cmap="gray", rasterized=True,
                  alpha=0.8)
        classes, axis = atlas["classes"].array, atlas["axis"]
        for k, name in enumerate(atlas["class_names"]):
            if name != "other_gray_matter":
                ax.contour(axis, axis, (classes == k).astype(float), [0.5], colors=[ATLAS_COLORS[name]],
                           linewidths=1.0 if name in ("MTG", "FuG", "ITG") else 0.6)
        rigid, pulled, depth = hyp["ribbon_rigid_plane"], hyp["ribbon_xiv_plane"], hyp["ribbon_depth_um"]
        sub = slice(None, None, 7)
        ax.scatter(*rigid[sub].T, s=0.12, c=st.MUTED, linewidths=0, rasterized=True, alpha=0.6)
        sc = ax.scatter(*pulled[sub].T, s=0.2, c=depth[sub] / 1000, cmap=st.SEQ_ORANGE, vmin=0, vmax=3.5,
                        linewidths=0, rasterized=True)
        centre = np.vstack([rigid, pulled]).mean(0)
        ax.set_xlim(centre[0] - half, centre[0] + half), ax.set_ylim(centre[1] - half, centre[1] + half)
        ax.set_aspect("equal"), ax.set_xticks([]), ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        st.scale_bar(ax, 10, "10 mm (atlas)")
        rl = hyp["evaluation"]["ribbon_landing_rigid"]["all"]
        xl = hyp["evaluation"]["ribbon_landing_after_xiv"]["all"]
        target = "MTG" if key == "A" else "FuG"
        st.panel_title(ax, "cd"[i], f"Pose {key}: {description}", "inferred")
        ax.text(0.02, 0.98, f"ribbon on {target}: {rl[target]:.0%} rigid, {xl[target]:.0%} after xIV\n"
                f"on white matter: {rl['white_matter']:.0%} rigid, {xl['white_matter']:.0%} after xIV",
                transform=ax.transAxes, fontsize=5.5, va="top",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1.5))
        st.axes_glyph(ax, "+u", "+v", y_down=False, loc=(0.04, 0.06))
    names = ["MTG", "ITG", "FuG", "STG", "white_matter"]
    handles = [Line2D([], [], color=ATLAS_COLORS[n], lw=1.0, label=n.replace("_", " ")) for n in names]
    handles += [Line2D([], [], marker="o", ls="", color=st.MUTED, ms=2, label="NeuN ribbon, rigid scan pose")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.62, 0.47), frameon=False, fontsize=5.5,
               title="AHRAv2 labels resliced on each plane", title_fontsize=5.5, alignment="left")
    _hbar(fig, sc, [0.635, 0.26, 0.15, 0.012], "ribbon after xIV: depth from pia (mm)", ticks=[0, 1, 2, 3])
    fig.text(0.635, 0.21, "Estimated correspondence, not accuracy.\nPoses A and B tie within stain-replicate\n"
             "noise (delta = 0.018, REPORT.md). Landing\nfractions test consistency with a population\n"
             "template; neither pose is independently\nverified. A is provisional (dissection prior).",
             fontsize=5.5, color=st.TEXT_2, va="top", linespacing=1.3)
    fmt = lambda v: "[" + ", ".join(f"{x:.2f}" for x in v) + "]"  # noqa: E731
    st.caption(fig, representation["caption_c"] if representation else [
        "a, b: frame qnp_local_mm (= NeuN slide mm; x right, y down). c, d: in-plane atlas mm; MNI T1 with resliced "
        "AHRAv2 label outlines; ribbon points subsampled 1:7.",
        f"Plane axes (MNI): A u = {fmt(planes['A']['u_axis'])}, v = {fmt(planes['A']['v_axis'])}; "
        f"B u = {fmt(planes['B']['u_axis'])}, v = {fmt(planes['B']['v_axis'])}. +u is toward MNI +x (right), "
        "+v toward +z (superior)."])
    out = {s: {k: v for k, v in c.items() if k not in ("polygons", "transform")} for s, c in checks.items()}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"figC_ribbon_dice_check_{job_name}.json").write_text(json.dumps(out, indent=2) + "\n")
    return st.save(fig, out_dir / f"figC_correspondence_{job_name}")
