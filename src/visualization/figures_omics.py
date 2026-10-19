"""MERFISH figure families for H20.33.001 section 1194111462.

A  scale space        cells -> 100 / 250 / 500 / 1000 um bins (bin_particles_2d), class and gene
B  feature maps       a-priori marker genes + blank-probe control, per-layer summaries, class fractions
C  correspondence     provider layer polygons vs cell labels (frame check) and the MTG-only xIV estimate

All panels use ``merfish_section_mm`` (X_spatial_raw x 1e-3) drawn x right, y down, the
same convention as the Neuroglancer scene; the provider records no anatomical orientation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize, to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from scipy.spatial import cKDTree

from . import scene as S
from . import style as st

BINS_MM = (0.1, 0.25, 0.5, 1.0)
GENE_A = "CUX2"
GENES_B = ("CUX2", "RORB", "FEZF2", "THEMIS", "MOG", "Blank (mean of 40)")
CELL_DISC_MM = 0.03  # drawn cell diameter; legible at print size (cells are ~10-20 um)
GENE_NOTE = {"CUX2": "L2/3 IT", "RORB": "L4-5 IT", "FEZF2": "deep excitatory", "THEMIS": "L6 IT Car3",
             "MOG": "oligodendrocyte", "Blank (mean of 40)": "negative control"}
FRACTIONS_B = ("excitatory", "inhibitory", "astrocyte", "oligodendrocyte", "opc", "microglia_pvm")


def _extent(cells: S.Points, pad: float = 0.3) -> list[float]:
    lo, hi = cells.xy.min(0) - pad, cells.xy.max(0) + pad
    return [lo[0], hi[0], lo[1], hi[1]]


def _cell_spacing_mm(cells: S.Points) -> float:
    distance, _ = cKDTree(cells.xy).query(cells.xy, k=2)
    return float(np.median(distance[:, 1]))


def _bin_squares(ax, binned: S.Points, facecolors, mass_ref: float):
    """Bins drawn on their grid squares, area proportional to cell count (capped at the bin)."""

    bin_mm = binned.meta["bin_mm"]
    counts = binned.scalars["cells_per_bin"].values
    side = bin_mm * np.sqrt(np.clip(counts / mass_ref, 0, 1))
    centres = (binned.meta["grid_cells"] + 0.5) * bin_mm
    patches = [Rectangle((x - s / 2, y - s / 2), s, s) for (x, y), s in zip(centres, side)]
    ax.add_collection(PatchCollection(patches, facecolors=facecolors, linewidths=0, rasterized=True))


def _class3_rgba(nu_class: np.ndarray, class_names: Sequence[str]) -> np.ndarray:
    nu3 = np.stack([nu_class[:, [list(class_names).index(c) for c in members]].sum(1)
                    for members in S.CLASS3.values()], 1)
    frac = nu3 / nu3.sum(1, keepdims=True)
    dominant = frac.argmax(1)
    purity = frac.max(1)
    rgba = np.array([to_rgba(st.CAT3[k]) for k in dominant])
    rgba[:, 3] = 0.25 + 0.75 * (purity - 1 / 3) / (2 / 3)
    return rgba


def _section_axes(ax, extent, letter, title, status, bar=True, glyph=False):
    st.map_axes(ax, extent, y_down=True)
    st.panel_title(ax, letter, title, status)
    if bar:
        st.scale_bar(ax, 2, "2 mm")
    if glyph:
        st.axes_glyph(ax, "+x", "+y", y_down=True, loc=(0.03, 0.03))


# --------------------------------------------------------------------------------------
# A - scale space
# --------------------------------------------------------------------------------------

def figure_a(cells: S.Points, out_dir: Path) -> list[Path]:
    extent = _extent(cells)
    spacing = _cell_spacing_mm(cells)
    klass3 = cells.categories["class3"]
    order = np.argsort(cells.scalars[f"lognorm:{GENE_A}"].values)
    gene = cells.scalars[f"lognorm:{GENE_A}"].values
    gnorm = Normalize(0, float(np.percentile(gene, 99)))
    binned = [S.bin_merfish(cells, b, [f"lognorm:{GENE_A}"]) for b in BINS_MM]

    fig = st.figure(st.FULL_WIDTH_MM, 118)
    width, height = 0.185, 0.33
    letters = iter("abcdefghij")
    for row, kind in enumerate(("class", "gene")):
        for col in range(1 + len(BINS_MM)):
            ax = fig.add_axes([0.012 + col * (width + 0.012), 0.585 - row * (height + 0.085), width, height])
            st.map_axes(ax, extent)
            if col == 0:
                s = st.marker_size(ax, CELL_DISC_MM)
                idx = order[klass3.codes[order] >= 0]  # gene row: high values drawn last
                colours = (np.array(st.CAT3)[klass3.codes[idx]] if kind == "class"
                           else st.SEQ_BLUE(gnorm(gene[idx])))
                ax.scatter(*cells.xy[idx].T, s=s, c=colours, linewidths=0, rasterized=True)
                title, status = f"{len(idx):,} cells", "measured"
            else:
                b = binned[col - 1]
                ref = float(np.percentile(b.scalars["cells_per_bin"].values, 95))
                faces = (_class3_rgba(b.meta["nu_class"], b.meta["class_names"]) if kind == "class"
                         else st.SEQ_BLUE(gnorm(b.scalars[f"lognorm:{GENE_A}"].values)))
                _bin_squares(ax, b, faces, ref)
                title, status = f"{len(b.xy):,} bins, {BINS_MM[col - 1] * 1000:g} µm", "derived"
            _section_axes(ax, extent, next(letters), title, status, bar=col == 0, glyph=(col == 0 and row == 0))
    handles = [Line2D([], [], marker="s", ls="", color=c, ms=5, label=n) for n, c in zip(S.CLASS3, st.CAT3)]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.01, 0.105), ncol=3, frameon=False, fontsize=6,
               title="a-e: cell class; bins show the dominant group, opacity = its share of cells (1/3 to 1)",
               title_fontsize=5.5, alignment="left")
    cax = fig.add_axes([0.52, 0.15, 0.44, 0.013])
    from matplotlib.cm import ScalarMappable
    cb = fig.colorbar(ScalarMappable(norm=gnorm, cmap=st.SEQ_BLUE), cax=cax, orientation="horizontal")
    cb.set_label(f"f-j: {GENE_A}, ln(1 + spots per 10K spots in the cell); bins = mean over cells;\none scale "
                 "for all columns (0 to the cell 99th percentile)", fontsize=5.5)
    cb.ax.tick_params(labelsize=5, width=0.4, length=2)
    st.caption(fig, [
        f"{cells.meta['donor']} section {cells.meta['barcode']} ({cells.meta['section']}); frame merfish_section_mm "
        "= X_spatial_raw / 1000, drawn x right, y down (orientation as stored).",
        f"Cells drawn as {CELL_DISC_MM * 1000:.0f} µm discs (median nearest-neighbour spacing {spacing * 1000:.1f} µm). Bins: "
        "preprocess.particle_utils.bin_particles_2d on the cell points (grid from the origin), drawn on their "
        "grid squares with area",
        "proportional to cell count (capped at the 95th percentile). These are deterministic binnings, not "
        "varifold-optimised representations; the 500 µm column reproduces binned_500um_classes.npz."])
    return st.save(fig, out_dir / f"figA_scale_space_{cells.meta['donor']}_{cells.meta['barcode']}")


# --------------------------------------------------------------------------------------
# B - feature maps
# --------------------------------------------------------------------------------------

def figure_b(cells: S.Points, out_dir: Path, polygons=None) -> list[Path]:
    from matplotlib.cm import ScalarMappable

    if polygons is None:
        from .cli import merfish_polygons

        polygons = merfish_polygons()
    poly, to_section, check = polygons
    rings = [to_section.apply(r) for r in poly.rings]
    extent = _extent(cells)
    spacing = _cell_spacing_mm(cells)
    values = {g: cells.scalars[f"lognorm:{g}"].values for g in GENES_B}
    gnorm = Normalize(0, float(np.percentile(np.concatenate(list(values.values())), 99.5)))
    layer = cells.categories["layer_annotation"]

    fig = st.figure(st.FULL_WIDTH_MM, 185)
    w, h = 0.152, 0.205
    for i, g in enumerate(GENES_B):
        ax = fig.add_axes([0.012 + i * (w + 0.012), 0.715, w, h])
        st.map_axes(ax, extent)
        order = np.argsort(values[g])
        ax.scatter(*cells.xy[order].T, s=st.marker_size(ax, CELL_DISC_MM), c=st.SEQ_BLUE(gnorm(values[g][order])),
                   linewidths=0, rasterized=True)
        for r in rings:
            ax.plot(*np.vstack([r, r[:1]]).T, color=st.TEXT_2, lw=0.35)
        _section_axes(ax, extent, "abcdef"[i], g, f"measured; {GENE_NOTE[g]}", bar=i == 0, glyph=i == 0)
    cax = fig.add_axes([0.30, 0.685, 0.40, 0.009])
    cb = fig.colorbar(ScalarMappable(norm=gnorm, cmap=st.SEQ_BLUE), cax=cax, orientation="horizontal")
    cb.set_label("a-f: ln(1 + spots per 10K spots in the cell), one scale for all genes; outlines = provider layer "
                 "polygons", fontsize=5.5)
    cb.ax.tick_params(labelsize=5, width=0.4, length=2)

    # g-l: per-layer summaries over provider-labelled cells (descriptive)
    labelled = layer.codes >= 0
    xs = np.arange(len(layer.names))
    ymax = gnorm.vmax
    for i, g in enumerate(GENES_B):
        ax = fig.add_axes([0.04 + i * (w + 0.012), 0.43, w - 0.03, 0.17])
        v = values[g][labelled]
        k = layer.codes[labelled]
        for j in xs:
            sel = v[k == j]
            q1, med, q3 = np.percentile(sel, [25, 50, 75])
            ax.plot([j, j], [q1, q3], color=st.ORDINAL5[j], lw=2.2, solid_capstyle="butt")
            ax.plot(j, med, "o", color="white", mec=st.ORDINAL5[j], ms=3.2, mew=1)
            ax.text(j, ymax * 1.02, f"{np.mean(sel > 0):.0%}", fontsize=4.5, ha="center", va="bottom",
                    color=st.TEXT_2)
        ax.set_xticks(xs, [n[1:] for n in layer.names], fontsize=5)
        ax.set_xlabel("layer", fontsize=5.5, labelpad=1)
        ax.set_ylim(0, ymax * 1.12)
        ax.set_xlim(-0.6, len(xs) - 0.4)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color=st.GRID, lw=0.3)
        if i == 0:
            ax.set_ylabel("ln(1 + spots/10K)")
        else:
            ax.set_yticklabels([])
        st.panel_title(ax, "ghijkl"[i], g, None)
    fig.text(0.04, 0.378, f"g-l: provider 'Layer annotation' cells only (n = {labelled.sum():,}); bar = IQR, dot = "
             "median, numbers = share of cells with at least one spot. Descriptive; no mutual information or other "
             "ranking is computed.", fontsize=5.5, color=st.TEXT_2)

    # m-r: class fractions per 250 um bin
    b = S.bin_merfish(cells, 0.25)
    fnorm = Normalize(0, 1)
    ref = float(np.percentile(b.scalars["cells_per_bin"].values, 95))
    for i, cls in enumerate(FRACTIONS_B):
        ax = fig.add_axes([0.012 + i * (w + 0.012), 0.11, w, h])
        st.map_axes(ax, extent)
        _bin_squares(ax, b, st.SEQ_BLUE(fnorm(b.scalars[f"fraction:{cls}"].values)), ref)
        _section_axes(ax, extent, "mnopqr"[i], cls.replace("_", "-"), "derived; per 250 µm bin",
                      bar=i == 0)
    cax = fig.add_axes([0.30, 0.075, 0.40, 0.009])
    cb = fig.colorbar(ScalarMappable(norm=fnorm, cmap=st.SEQ_BLUE), cax=cax, orientation="horizontal")
    cb.set_label("m-r: fraction of the bin's cells in the class (provider Subclass grouped as in "
                 "feature_groupings); square area = cell count", fontsize=5.5)
    cb.ax.tick_params(labelsize=5, width=0.4, length=2)
    st.caption(fig, [
        "Genes were chosen a priori as known markers (labels under each title); the control is the per-cell mean of "
        "the 40 blank probes (one probe-equivalent). Nothing is ranked by the data.",
        "X in this h5ad holds raw spot counts (rows sum to 'Number of spots'), despite uns/X_normalization; the "
        "log-normalisation shown is computed here."])
    return st.save(fig, out_dir / f"figB_feature_maps_{cells.meta['donor']}_{cells.meta['barcode']}")


# --------------------------------------------------------------------------------------
# C - correspondence
# --------------------------------------------------------------------------------------

def figure_c(cells: S.Points, out_dir: Path, polygons=None, xiv=None) -> list[Path]:
    from .cli import MERFISH, merfish_polygons

    poly, to_section, check = polygons or merfish_polygons()
    xiv = xiv or S.load_merfish_xiv(S.PROJECT / MERFISH["xiv_fit"])
    rings = [to_section.apply(r) for r in poly.rings]
    layer = cells.categories["layer_annotation"]
    spacing = _cell_spacing_mm(cells)

    fig = st.figure(st.FULL_WIDTH_MM, 92)
    # a: frame identification on the depth strip
    strip = np.vstack(rings)
    ext_a = [strip[:, 0].min() - 0.3, strip[:, 0].max() + 0.3, strip[:, 1].min() - 0.3, strip[:, 1].max() + 0.3]
    ax = fig.add_axes([0.02, 0.22, 0.27, 0.62])
    st.map_axes(ax, ext_a)
    s = st.marker_size(ax, CELL_DISC_MM)
    other = layer.codes < 0
    ax.scatter(*cells.xy[other].T, s=s, c="#e4e3dd", linewidths=0, rasterized=True)
    ax.scatter(*cells.xy[~other].T, s=s, c=np.array(st.ORDINAL5)[layer.codes[~other]], linewidths=0, rasterized=True)
    for ring, lab in zip(rings, poly.labels):
        ax.plot(*np.vstack([ring, ring[:1]]).T, color=st.TEXT, lw=0.7)
    st.scale_bar(ax, 1, "1 mm")
    st.axes_glyph(ax, "+x", "+y", y_down=True, loc=(0.03, 0.03))
    st.panel_title(ax, "a", "Provider layer polygons vs cell labels", "measured; polygons transformed (exact)")
    ax.text(0.02, 0.98, f"label agreement {check['label_agreement']:.1%}\n(best other axis frame "
            f"{check['runner_up_agreement']:.1%})", transform=ax.transAxes, fontsize=5.5, va="top",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1.5))
    handles = [Line2D([], [], marker="o", ls="", color=c, ms=3, label=n) for n, c in zip(layer.names, st.ORDINAL5)]
    handles += [Line2D([], [], marker="o", ls="", color="#e4e3dd", ms=3, label="no layer label"),
                Line2D([], [], color=st.TEXT, lw=0.7, label="uns/layer_polygons")]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=5, ncol=2, handletextpad=0.3,
              columnspacing=0.8, bbox_to_anchor=(1.0, -0.22))

    # b: atlas-plane particles, initial vs after xIV, on the section
    rigid, deformed = xiv["rigid"].xy, xiv["deformed"].xy
    disp = deformed - rigid
    shift = disp.mean(0)
    residual_um = np.linalg.norm(disp - shift, axis=1) * 1000
    ax = fig.add_axes([0.33, 0.12, 0.36, 0.72])
    allp = np.vstack([cells.xy, deformed, rigid])
    ext_b = [allp[:, 0].min() - 0.5, allp[:, 0].max() + 0.5, allp[:, 1].min() - 0.5, allp[:, 1].max() + 0.5]
    st.map_axes(ax, ext_b)
    disc = st.marker_size(ax, CELL_DISC_MM)
    ax.scatter(*cells.xy.T, s=disc, c="#d6d5ce", linewidths=0, rasterized=True)
    labelled = layer.codes >= 0
    ax.scatter(*cells.xy[labelled].T, s=disc, c=st.TEXT_2, linewidths=0, rasterized=True)
    ax.scatter(*rigid.T, s=st.marker_size(ax, 0.18), facecolors="none", edgecolors=st.MUTED, linewidths=0.3,
               rasterized=True)
    ax.scatter(*deformed.T, s=st.marker_size(ax, 0.12), c=st.CAT3[1], linewidths=0, alpha=0.7, rasterized=True)
    anchor = rigid.mean(0)
    ax.annotate("", anchor + shift, anchor, arrowprops=dict(arrowstyle="-|>", color=st.TEXT, lw=0.8,
                                                            mutation_scale=7, shrinkA=0, shrinkB=0))
    st.scale_bar(ax, 5, "5 mm")
    st.panel_title(ax, "b", "AHRAv2 MTG plane particles on the section", "inferred (MTG-only xIV estimate)")
    handles = [Line2D([], [], marker="o", ls="", color="#d6d5ce", ms=3, label="MERFISH cells"),
               Line2D([], [], marker="o", ls="", color=st.TEXT_2, ms=3, label="layer-labelled strip (xIV target)"),
               Line2D([], [], marker="o", ls="", mfc="none", mec=st.MUTED, ms=3, label="atlas particles, initial placement"),
               Line2D([], [], marker="o", ls="", color=st.CAT3[1], ms=3, label="atlas particles after xIV"),
               Line2D([], [], color=st.TEXT, lw=0.8, label="mean displacement (arrow)")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.70, 0.86), frameon=False, fontsize=5.5)
    score = xiv["scores"].get("score_total_mass")
    fig.text(0.705, 0.585, "Registration QC, not accuracy: one MTG label\nfitted to the 50-bin depth strip; placement is\n"
             f"under-determined (relative varifold distance {score:.3f}).", fontsize=5.5, color=st.TEXT_2,
             va="top", linespacing=1.3)

    # c: displacement = one translation + a tiny residual
    ax = fig.add_axes([0.79, 0.14, 0.18, 0.22])
    ax.hist(residual_um, bins=30, color=st.CAT3[0], edgecolor="white", linewidth=0.3)
    ax.set_xlabel("|displacement - mean| (µm)", labelpad=1)
    ax.set_ylabel("particles", labelpad=1)
    ax.spines[["top", "right"]].set_visible(False)
    st.panel_title(ax, "c", f"Mean shift ({shift[0]:.2f}, {shift[1]:.2f}) mm", f"residual median {np.median(residual_um):.1f} µm")

    st.caption(fig, [
        "Frame merfish_section_mm (x right, y down). a: polygons are stored in (-y, x) um; the frame is identified "
        "by label agreement, not assumed. b: initial placement = barycentre",
        "alignment of the plane (M = I); xIV deformation from atlas_deformationSummary (D + target centre). The "
        "atlas plane is a provisional coronal cut at y = -30 mm; donor H20.33.001 only."], y=0.005)
    return st.save(fig, out_dir / f"figC_correspondence_{cells.meta['donor']}_{cells.meta['barcode']}")
