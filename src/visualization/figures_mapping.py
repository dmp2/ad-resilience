"""Figures of the section-to-atlas mapping workflow (``map_qnp_sections.py``).  Nothing is fitted here.

A. Anatomical slab prior: 3-D slab in the brain, labelled coronal montage around the landmark,
   sagittal localisation (``render_slab_figures``).
B. Stacking: gray/white maps by role, depth profiles with delta bands and windows, best poses
   over the labelled coronal plane, and the unconstrained vs admissible candidate panels.

Display conventions: atlas world mm (MNI RAS).  Coronal panels: x to the right = subject's
right (neurological; "L"/"R" marked on slab figures).  Sagittal panels: anterior on the left.
Profiles: y axis inverted (anterior left).  The nominal slab drawn for each offset j uses phase
delta = h/2 (landmark mid-slab); the true cutting phase is unknown, so the full cover of each j
over all phases is drawn as a bracket.  Pose overlays: atlas fills muted with thin grey
boundaries (outer surface solid, gray/white dotted); the transformed section in heavy black
(outline solid, gray/white dashed) with the pia in red; the admissible support dash-dot green.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import ndimage as ndi

from section_mapping.atlas import FINE_ORDER_TAIL, plane_geometry, reslice_labels
from section_mapping.measures import splat
from visualization import style as st

LANDMARK_COLOR = "#d6246e"
STRUCTURE_COLORS = {"MTG": "#2a78d6", "STG": "#eb6834", "ITG": "#1baf7a", "FuG": "#eda100",
                    "LGN": LANDMARK_COLOR, "MGN": "#8a5cc2", "hippocampus": "#8b8a85", "insula": "#b59a1a"}
J_COLORS = {-1: "#86b6ef", 0: "#eb6834", 1: "#1c5cab"}
FALLBACK = ("#5e53a5", "#0d366b", "#94391a", "#52514e")


def _structures(atlas: Mapping, scan: Mapping, slab_cfg: Mapping) -> dict[str, np.ndarray]:
    """name -> boolean lookup over label values: the scan's fine regions + figure_label_codes.

    The catch-all channels (other gray, white matter) are not overlaid: the template shows them.
    """

    n = int(atlas["labels"].max()) + 1
    out = {name: np.asarray(atlas["fine"][name][:n], bool) for name in scan["fine_regions"]}
    for name, codes in (slab_cfg.get("figure_label_codes") or {}).items():
        table = np.zeros(n, bool)
        table[[int(c) for c in codes if int(c) < n]] = True
        out[name] = table
    return out


def _color(name: str, k: int) -> str:
    return STRUCTURE_COLORS.get(name, FALLBACK[k % len(FALLBACK)])


def _nominal(record: Mapping, j: int) -> tuple[float, float]:
    """World-y [anterior, posterior] of S_j at phase h/2."""

    from section_mapping.atlas import slab_bounds, world_y_mm

    s0, s1 = slab_bounds(record["landmark"]["s_mm"], record["slab_thickness_mm"], j, record["slab_thickness_mm"] / 2)
    return float(world_y_mm(s0)), float(world_y_mm(s1))


def _reslice(volume, affine, origin, u, v, half, res=0.5):
    from section_mapping.atlas import reslice_labels

    return reslice_labels(volume, affine, origin, u, v, half, res)


def _brain_box(atlas: Mapping) -> np.ndarray:
    ijk = np.argwhere(atlas["labels"] > 0)
    lo, hi = ijk.min(0), ijk.max(0)
    corners = np.array([[a, b, c] for a in (lo[0], hi[0]) for b in (lo[1], hi[1]) for c in (lo[2], hi[2])], float)
    world = corners @ atlas["affine"][:3, :3].T + atlas["affine"][:3, 3]
    return np.array([world.min(0), world.max(0)])  # [[xmin, ymin, zmin], [xmax, ymax, zmax]]


def _overlay(ax, labels2d, template2d, extent, structures, alpha=0.45):
    from matplotlib.colors import ListedColormap

    t = template2d.astype(float)
    vmax = np.percentile(t[t > 0], 99.5) if (t > 0).any() else 1.0
    ax.imshow(np.where(labels2d > 0, t, np.nan), cmap="gray", vmin=0, vmax=vmax, extent=extent, origin="lower",
              interpolation="nearest")
    for k, (name, table) in enumerate(structures.items()):
        mask = table[np.clip(labels2d, 0, len(table) - 1)] & (labels2d < len(table))
        if not mask.any():
            continue
        color = _color(name, k)
        ax.imshow(np.where(mask, 1.0, np.nan), cmap=ListedColormap([color]), alpha=alpha, extent=extent,
                  origin="lower", interpolation="nearest", vmin=0, vmax=1)
        ax.contour(np.linspace(extent[0], extent[1], mask.shape[1]), np.linspace(extent[2], extent[3], mask.shape[0]),
                   mask.astype(float), [0.5], colors=[color], linewidths=0.6)


# --------------------------------------------------------------------------------------
# B. Coronal montage
# --------------------------------------------------------------------------------------


def coronal_montage(atlas, template, scan, slab_cfg, record, out_dir: Path) -> Path:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    st.setup()
    structures = _structures(atlas, scan, slab_cfg)
    y_L = float(record["landmark"]["y_mm"])
    hi, lo = record["search_interval"]["y_mm"]
    h = float(record["slab_thickness_mm"])
    planes = [y_L + k * h / 2 for k in range(4, -5, -1)]  # anterior -> posterior, h/2 apart, landmark centred
    box = _brain_box(atlas)
    half = float(max(box[1, 0] - box[0, 0], box[1, 2] - box[0, 2]) / 2 + 2)
    cx, cz = float(box[:, 0].mean()), float(box[:, 2].mean())
    u, v = np.array([1.0, 0, 0]), np.array([0, 0, 1.0])
    fig, axes = plt.subplots(3, 3, figsize=(st.FULL_WIDTH_MM * st.MM, 175 * st.MM), constrained_layout=True)
    for ax, y in zip(axes.flat, planes):
        origin = np.array([cx, y, cz])
        lab, axis = _reslice(atlas["labels"], atlas["affine"], origin, u, v, half)
        tem, _ = _reslice(template, atlas["affine"], origin, u, v, half)
        extent = (cx + axis[0], cx + axis[-1], cz + axis[0], cz + axis[-1])
        _overlay(ax, lab, tem, extent, structures)
        ax.set_xlim(box[0, 0] - 2, box[1, 0] + 2)
        ax.set_ylim(box[0, 2] - 2, box[1, 2] + 2)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        nominal = [j for j in record["slab_offsets"] if min(_nominal(record, j)) - 1e-6 <= y <= max(_nominal(record, j)) + 1e-6]
        inside = lo - 1e-6 <= y <= hi + 1e-6
        tag = ("nominal slab j " + "/".join(f"{j:+d}" for j in nominal)) if nominal else (
            "in search interval" if inside else "outside search interval")
        first = "  ·  LGN first appearance" if abs(y - y_L) < 1e-6 else ""
        ax.set_title(f"y {y:+.1f} mm (s {-y:.1f}){first}\n{tag}", fontsize=6.5)
        edge = J_COLORS.get(nominal[-1], st.MUTED) if nominal else st.GRID
        for spine in ax.spines.values():
            spine.set_edgecolor(edge)
            spine.set_linewidth(2.0 if nominal else 0.6)
        ax.text(0.02, 0.5, "L", transform=ax.transAxes, fontsize=8, fontweight="bold", color=st.TEXT_2, va="center")
        ax.text(0.98, 0.5, "R", transform=ax.transAxes, fontsize=8, fontweight="bold", color=st.TEXT_2, va="center",
                ha="right")
    handles = [Patch(fc=_color(n, k), alpha=0.6, label=n) for k, n in enumerate(structures)]
    handles += [Patch(fc="none", ec=J_COLORS[j], lw=2, label=f"nominal slab j {j:+d} (δ = h/2)")
                for j in record["slab_offsets"] if j in J_COLORS]
    fig.legend(handles=handles, loc="outside lower center", ncol=6, frameon=False)
    fig.suptitle(f"Coronal AHRAv2 planes around the {record['landmark']['name']} landmark "
                 f"(y {y_L:+.1f} mm), {h / 2:g} mm apart; search interval y {hi:+.1f} … {lo:+.1f} mm. "
                 "Neurological view (subject's right on the right).", fontsize=7)
    path = out_dir / "slab_coronal_montage.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------------------
# C. Sagittal localisation
# --------------------------------------------------------------------------------------


def sagittal_view(atlas, template, scan, slab_cfg, record, out_dir: Path, windows=None, windows_source=None) -> Path:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    st.setup()
    structures = _structures(atlas, scan, slab_cfg)
    lm = record["landmark"]
    y_L = float(lm["y_mm"])
    hi, lo = record["search_interval"]["y_mm"]
    box = _brain_box(atlas)
    lm_x = lm["centroid_mm"][0] if lm.get("centroid_mm") else -22.5
    target = next(iter(scan["fine_regions"]))
    sections = [(f"x {lm_x:+.1f} mm (through the {lm['name']})", lm_x),
                (f"x {atlas['centroid'][0]:+.1f} mm (through the {target} centroid)", float(atlas["centroid"][0]))]
    half = float(max(box[1, 1] - box[0, 1], box[1, 2] - box[0, 2]) / 2 + 2)
    cy, cz = float(box[:, 1].mean()), float(box[:, 2].mean())
    u, v = np.array([0, 1.0, 0]), np.array([0, 0, 1.0])
    fig, axes = plt.subplots(1, 2, figsize=(st.FULL_WIDTH_MM * st.MM, 82 * st.MM), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1.0, 1.0]})
    offsets = record["slab_offsets"]
    for panel, (ax, (title, x)) in enumerate(zip(axes, sections)):
        origin = np.array([x, cy, cz])
        lab, axis = _reslice(atlas["labels"], atlas["affine"], origin, u, v, half)
        tem, _ = _reslice(template, atlas["affine"], origin, u, v, half)
        extent = (cy + axis[0], cy + axis[-1], cz + axis[0], cz + axis[-1])
        _overlay(ax, lab, tem, extent, structures, alpha=0.4)
        for j in offsets:
            a, p = _nominal(record, j)
            ax.axvspan(p, a, color=J_COLORS.get(j, st.MUTED), alpha=0.18 if j else 0.28, lw=0)
        for edge in (hi, lo):
            ax.axvline(edge, color=st.TEXT_2, lw=0.7, ls=":")
        ax.axvline(y_L, color=LANDMARK_COLOR, lw=0.9, ls="--")
        if lm.get("centroid_mm") and abs(x - lm_x) < 1e-6:
            ax.plot(lm["centroid_mm"][1], lm["centroid_mm"][2], marker="o", ms=4, mfc="none", mec=LANDMARK_COLOR, mew=1.0)
        # Cover of each offset j over all cutting phases: brackets under the brain.
        # Panel 1: whole hemisphere (context).  Panel 2: zoom on the temporal lobe around the interval.
        if panel == 0:
            ylim, zlim, gap = (box[1, 1] + 3, box[0, 1] - 3), (box[0, 2], box[1, 2]), 6.0
        else:
            ylim, zlim, gap = (hi + 25, lo - 25), (-50.0, 25.0), 3.0
        zb = zlim[0] - gap
        for k, j in enumerate(offsets):
            cov = record["offset_cover_y_mm"][str(j)]
            a, p = _nominal(record, j)
            ax.plot(cov, [zb - gap * k] * 2, color=J_COLORS.get(j, st.MUTED), lw=0.8)
            ax.plot([a, p], [zb - gap * k] * 2, color=J_COLORS.get(j, st.MUTED), lw=3.0, solid_capstyle="butt")
            ax.text(hi + 2, zb - gap * k, f"j {j:+d}", fontsize=5.5, va="center", ha="right", color=st.TEXT_2)
        ztop = zlim[1] + gap
        if windows:
            for k, (name, w) in enumerate(windows.items()):
                ax.plot([w[1], w[0]], [ztop + gap * k] * 2, color=st.TEXT, lw=2.5, solid_capstyle="butt")
                ax.text(min(w) - 2, ztop + gap * k, name, fontsize=5.5, va="center", ha="left")
        ax.set_xlim(*ylim)  # anterior on the left
        ax.set_ylim(zb - gap * len(offsets), ztop + gap * len(windows or {}) + 1)
        ax.set_aspect("equal")
        ax.set_xlabel("MNI y (mm)   ← anterior · posterior →")
        ax.set_ylabel("MNI z (mm)")
        ax.set_title(title)
    handles = [Patch(fc=J_COLORS.get(j, st.MUTED), alpha=0.3, label=f"nominal slab j {j:+d} (δ = h/2)") for j in offsets]
    handles += [Line2D([], [], color=LANDMARK_COLOR, ls="--", lw=0.9, label=f"{lm['name']} first appearance y {y_L:+.1f}"),
                Line2D([], [], color=st.TEXT_2, ls=":", lw=0.7, label=f"search bound (all phases ± {record['extension_mm']:g} mm)"),
                Line2D([], [], color=st.MUTED, lw=0.8, label="cover of j over all phases (bracket)")]
    if windows:
        handles.append(Line2D([], [], color=st.TEXT, lw=2.5, label=f"stacking windows ({windows_source})"))
    handles += [Patch(fc=_color(n, k), alpha=0.6, label=n) for k, n in enumerate(structures)]
    fig.legend(handles=handles, loc="outside lower center", ncol=5, frameon=False)
    path = out_dir / "slab_sagittal.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------------------
# A. 3-D slab
# --------------------------------------------------------------------------------------


def _surface(mask: np.ndarray, affine: np.ndarray, step: int = 1):
    import pyvista as pv
    from skimage.measure import marching_cubes

    if not mask.any():
        return pv.PolyData()
    padded = np.pad(mask.astype(np.uint8), 1)
    verts, faces, _, _ = marching_cubes(padded, 0.5, step_size=step)
    verts = (verts - 1) @ affine[:3, :3].T + affine[:3, 3]
    mesh = pv.PolyData(verts, np.column_stack([np.full(len(faces), 3), faces]).ravel())
    return mesh.smooth(n_iter=30, relaxation_factor=0.1)


def slab_3d(atlas, scan, slab_cfg, record, out_dir: Path) -> Path:
    os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
    import pyvista as pv

    labels, A = atlas["labels"], atlas["affine"]
    ijk = np.indices(labels.shape, sparse=True)
    xw = A[0, 0] * ijk[0] + A[0, 3]  # axis-aligned check below
    if not np.allclose(A[:3, :3], np.diag(np.diag(A[:3, :3]))):
        raise ValueError("3-D slab figure expects an axis-aligned atlas affine")
    yw = A[1, 1] * ijk[1] + A[1, 3]
    side = -1.0 if scan.get("hemisphere", "left") == "left" else 1.0
    hemi = np.sign(xw) == side
    brain = _surface(labels > 0, A, step=2)
    hi, lo = record["search_interval"]["y_mm"]
    in_interval = (yw >= lo) & (yw <= hi)
    regions = list(record.get("sampling_regions") or [])
    lm = record["landmark"]
    box = _brain_box(atlas)
    structures = _structures(atlas, scan, slab_cfg)
    meshes = {}
    for name in regions:
        member = structures[name][np.clip(labels, 0, len(structures[name]) - 1)] & hemi
        meshes[name] = (_surface(member & in_interval, A), _surface(member, A))
    lgn = structures.get(lm["name"])
    lgn_mesh = _surface(lgn[np.clip(labels, 0, len(lgn) - 1)] & hemi, A) if lgn is not None else None

    views = [("A  Oblique (left, anterior, superior)", np.array([-1.0, 0.45, 0.45])),
             ("B  Left lateral", np.array([-1.0, 0.0, 0.0])),
             ("C  Superior", np.array([0.0, 0.0, 1.0]))]
    plot = pv.Plotter(off_screen=True, shape=(1, 3), window_size=(2700, 1000), border=False)
    centre = box.mean(0)
    for k, (title, direction) in enumerate(views):
        plot.subplot(0, k)
        plot.set_background("#ffffff")
        plot.add_mesh(brain, color="#b8b6ad", opacity=0.10, smooth_shading=True)
        for j in record["slab_offsets"]:
            a, p = _nominal(record, j)
            x0, x1 = (box[0, 0], 0.0) if side < 0 else (0.0, box[1, 0])  # sampled hemisphere only
            slab = pv.Box(bounds=(x0, x1, p, a, box[0, 2], box[1, 2]))
            plot.add_mesh(slab, color=J_COLORS.get(j, "#8b8a85"), opacity=0.10 if j == 0 else 0.04)
            plot.add_mesh(slab.extract_feature_edges(), color=J_COLORS.get(j, "#8b8a85"), line_width=1.5)
        for name, (inside, whole) in meshes.items():
            plot.add_mesh(whole, color=_color(name, 0), opacity=0.12, smooth_shading=True)
            if inside.n_points:
                plot.add_mesh(inside, color=_color(name, 0), opacity=0.9, smooth_shading=True)
        if lgn_mesh is not None and lgn_mesh.n_points:
            plot.add_mesh(lgn_mesh, color=LANDMARK_COLOR, opacity=1.0, smooth_shading=True)
        if lm.get("centroid_mm"):
            plot.add_mesh(pv.Sphere(radius=2.0, center=lm["centroid_mm"]), color=LANDMARK_COLOR)
        plot.add_text(title, position="upper_left", font_size=12, color="#0b0b0b")
        plot.camera_position = [centre + direction * 400, centre, (0, 0, 1) if direction[2] < 0.9 else (0, 1, 0)]
        plot.enable_parallel_projection()
        plot.reset_camera()
        plot.camera.zoom(1.25)
    plot.subplot(0, 0)
    plot.add_text("\n\nanterior = +y (MNI); slabs drawn at cutting phase delta = h/2 (true phase unknown)",
                  position="upper_left", font_size=9, color="#52514e")
    legend = [[f"nominal slab j {j:+d}", J_COLORS.get(j, "#8b8a85")] for j in record["slab_offsets"]]
    legend += [[f"{n}, search interval", _color(n, 0)] for n in regions]
    legend += [[f"{lm['name']}, y {lm['y_mm']:+.1f}", LANDMARK_COLOR]]
    plot.subplot(0, 2)
    plot.add_legend(legend, bcolor="white", face="rectangle", size=(0.34, 0.30), loc="upper right")
    path = out_dir / "slab_3d.png"
    plot.screenshot(str(path))
    plot.close()
    return path


def render_slab_figures(atlas, template, scan, slab_cfg, record, out_dir: Path,
                        windows: Mapping[str, Sequence[float]] | None = None, windows_source: str | None = None) -> list[Path]:
    out_dir = Path(out_dir)
    return [slab_3d(atlas, scan, slab_cfg, record, out_dir),
            coronal_montage(atlas, template, scan, slab_cfg, record, out_dir),
            sagittal_view(atlas, template, scan, slab_cfg, record, out_dir, windows, windows_source)]


# --------------------------------------------------------------------------------------
# B. Stacking figures
# --------------------------------------------------------------------------------------

REGION_COLORS = {"MTG": "#2a78d6", "STG": "#eb6834", "ITG": "#1baf7a", "FuG": "#eda100",
                 "white_matter": "#c4c2b9", "other_gray_matter": "#e6e4dc",
                 "insula": "#b59a1a", "hippocampus": "#8b8a85", "LGN": "#d6246e", "MGN": "#8a5cc2",
                 "temporal_plane": "#c98bb9"}
PIA_COLOR = "#e34948"
INK = "#1a1a19"
ATLAS_LINE = "#6b6a66"
SUPPORT_COLOR = "#1baf7a"
FLAG_COLOR = "#b3261e"


def pose_legend_handles(regions, support: bool = False):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    handles = [Patch(fc=REGION_COLORS[n], alpha=0.55, ec="none", label=f"atlas {n.replace('_', ' ')}")
               for n in regions if n in REGION_COLORS]
    handles += [Line2D([], [], color=ATLAS_LINE, lw=0.7, label="atlas outer surface (and ventricle walls)"),
                Line2D([], [], color=ATLAS_LINE, lw=0.6, ls=":", label="atlas gray/white boundary"),
                Line2D([], [], color=INK, lw=1.6, label="QNP tissue outline (transformed)"),
                Line2D([], [], color=INK, lw=1.1, ls="--", label="QNP gray/white boundary (transformed)"),
                Line2D([], [], color=PIA_COLOR, lw=2.5, label="QNP pia, ribbon depth ≤ 100 µm (transformed)")]
    if support:
        handles.append(Line2D([], [], color=SUPPORT_COLOR, lw=0.9, ls="-.", label="admissible atlas support (region prior)"))
    return handles


def class_image(atlas: Mapping, regions: Sequence[str], extra: Mapping[str, np.ndarray], y: float, x_range, z_range,
                res: float = 0.5):
    """Coronal label image at y over (x_range, z_range): 0 = background, k = names[k - 1]."""

    names = list(regions) + list(extra) + list(FINE_ORDER_TAIL)
    tables = {**atlas["fine"], **extra}
    table = np.zeros(int(atlas["labels"].max()) + 1, np.int16)
    # The catch-all tail first, then named regions over it (QC structures such as the insula
    # are part of other gray matter).
    for name in list(FINE_ORDER_TAIL) + list(regions) + list(extra):
        table[tables[name][:len(table)]] = names.index(name) + 1
    _, u, v, _ = plane_geometry(atlas, 0.0, 0.0, y)
    cx, cz = np.mean(x_range), np.mean(z_range)
    half = max(np.ptp(x_range), np.ptp(z_range)) / 2
    grid, axis = reslice_labels(atlas["labels"], atlas["affine"], np.array([cx, y, cz]), u, v, half, res)
    img = table[np.clip(grid, 0, len(table) - 1)]
    xs, zs = cx + axis, cz + axis
    ix = (xs >= x_range[0]) & (xs <= x_range[1])
    iz = (zs >= z_range[0]) & (zs <= z_range[1])
    return img[np.ix_(iz, ix)], xs[ix], zs[iz], names


def draw_pose(ax, atlas: Mapping, regions: Sequence[str], extra: Mapping[str, np.ndarray], y: float, transform,
              X: np.ndarray, nu: np.ndarray, pia: np.ndarray, x_range, z_range, prior: Mapping | None = None):
    """Labelled coronal plane at y under the section (X, nu gray/white; ``pia`` points) placed by ``transform``.

    ``prior`` ({regions, margin_mm}) draws the admissible support of the region prior.
    """

    from matplotlib.colors import ListedColormap

    img, xs, zs, names = class_image(atlas, regions, extra, y, x_range, z_range)
    colors = ["#ffffff"] + [REGION_COLORS.get(n, "#bdbbb3") for n in names]
    ax.imshow(img, origin="lower", extent=(xs[0] - 0.25, xs[-1] + 0.25, zs[0] - 0.25, zs[-1] + 0.25),
              cmap=ListedColormap(colors), vmin=-0.5, vmax=len(colors) - 0.5, interpolation="nearest", alpha=0.55)
    # Atlas boundaries: thin grey.  Outer surface = tissue / no-label edge (pia, and ventricle walls).
    ax.contour(xs, zs, (img > 0).astype(float), [0.5], colors=ATLAS_LINE, linewidths=0.7)
    white = names.index("white_matter") + 1
    ax.contour(xs, zs, (img == white).astype(float), [0.5], colors=ATLAS_LINE, linewidths=0.6, linestyles=":")
    if prior:
        mask = np.zeros(img.shape, bool)
        for r in prior["regions"]:
            mask |= img == names.index(r) + 1
        dist = ndi.distance_transform_edt(~mask) * 0.5
        ax.contour(xs, zs, (dist <= float(prior.get("margin_mm", 6.0))).astype(float), [0.5],
                   colors=SUPPORT_COLOR, linewidths=0.9, linestyles="-.")
    for k, name in enumerate(names[:-2], start=1):
        mask = img == k
        if mask.sum() < 40:
            continue
        lab, n = ndi.label(mask)
        biggest = 1 + int(np.argmax(ndi.sum(mask, lab, range(1, n + 1))))
        iz, ix = np.unravel_index(np.argmax(ndi.distance_transform_edt(lab == biggest)), mask.shape)
        ax.text(xs[ix], zs[iz], name, fontsize=7, ha="center", va="center", color=ATLAS_LINE,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))
    # Transformed section: heavy black outline, dashed black gray/white, red pia.
    origin, _, _, _ = plane_geometry(atlas, 0.0, 0.0, y)
    P = transform.section_to_moving(X) + origin[[0, 2]]
    tissue = splat(P, nu.sum(1), xs, zs, 0.5)
    wm = splat(P, nu[:, 1], xs, zs, 0.5)
    ref = np.percentile(tissue[tissue > 0], 90)
    frac = np.where(tissue > 1e-9, wm / np.maximum(tissue, 1e-9), 0.0)
    ax.contour(xs, zs, tissue / ref, [0.35], colors=INK, linewidths=1.6)
    ax.contour(xs, zs, np.where(tissue / ref > 0.35, frac, 0.0), [0.5], colors=INK, linewidths=1.1, linestyles="--")
    pia_plane = transform.section_to_moving(pia) + origin[[0, 2]]
    ax.scatter(pia_plane[::3, 0], pia_plane[::3, 1], s=2.0, c=PIA_COLOR, linewidths=0, rasterized=True)
    ax.set_xlim(x_range)
    ax.set_ylim(z_range)
    ax.set_aspect("equal")
    ax.set_xlabel("MNI x (mm)")
    ax.set_ylabel("MNI z (mm)")


def roles_figure(ms: Mapping[str, np.ndarray], ms_stains: Sequence[str], stains: Sequence[str],
                 sections: Mapping[str, Mapping], layer1: np.ndarray | None, path: Path) -> None:
    """Gray/white map per section in the local frame, titled by its role (multistain-map grid)."""

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    origin, px = ms["origin_mm"], ms["px_mm"]
    rows_, cols_ = ms["p_wm"].shape[1:]
    extent = [origin[0], origin[0] + cols_ * px[1], origin[1] + rows_ * px[0], origin[1]]
    ncol = 3
    nrow = int(np.ceil(len(stains) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 4.6 * nrow), constrained_layout=True, squeeze=False)
    for ax, stain in zip(axes.flat, stains):
        k = list(ms_stains).index(stain)
        shown = ax.imshow(np.where(ms["tissue"][k], ms["p_wm"][k].astype(float), np.nan), extent=extent,
                          cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
        if layer1 is not None:
            ax.contour(origin[0] + np.arange(cols_) * px[1], origin[1] + np.arange(rows_) * px[0],
                       layer1.astype(float), [0.5], colors=[PIA_COLOR], linewidths=0.5)
        m = sections[stain]["metrics"]
        ax.set_title(f"{stain}: {sections[stain]['role']}\nseparability {m['otsu_separability']:.2f}, "
                     f"stripes {m['stripe_index']:.2f}"
                     + (f", layer 1 p(WM) {m['layer1_mean_p_wm']:.2f}" if m["layer1_mean_p_wm"] is not None else ""),
                     fontsize=10)
        ax.set_xlabel("local x (mm)")
        ax.set_ylabel("local y (mm, down)")
        ax.set_aspect("equal")
    for ax in list(axes.flat)[len(stains):]:
        ax.axis("off")
    fig.colorbar(shown, ax=axes, shrink=0.5, label="p(white matter)")
    fig.savefig(path, dpi=110)
    plt.close(fig)


def depth_profiles_figure(levels: Sequence[str], anchors: Sequence[str], summaries: Mapping[str, Mapping],
                          rows: Mapping[str, Sequence[Mapping]], level_names: Mapping[str, str], step_mm: float,
                          slab_prior: Mapping | None, path: Path) -> None:
    """Depth profiles per anchor (columns) and level (rows), with the delta band and the window."""

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    fig, axes = plt.subplots(len(levels), len(anchors), figsize=(4.6 * len(anchors), 3.4 * len(levels)),
                             squeeze=False, constrained_layout=True)
    for i, lv in enumerate(levels):
        d = summaries[lv]["delta"]["delta"]
        for j, a in enumerate(anchors):
            ax = axes[i, j]
            for fam in sorted({r["family"] for r in rows[lv]}):
                sel = sorted((r for r in rows[lv] if r["representation"] == a and r["family"] == fam), key=lambda r: r["y0_mm"])
                yy = np.array([r["y0_mm"] for r in sel])
                ss = np.array([r["score"] for r in sel])
                gap = np.flatnonzero(np.diff(yy) > step_mm + 1e-6) + 1
                yy, ss = np.insert(yy, gap, np.nan), np.insert(ss, gap, np.nan)  # no line across unsearched depths
                ax.plot(yy, ss, color="#2a78d6" if fam != "flipped" else "#eb6834", lw=1.6, marker="o", ms=2.5)
            w = summaries[lv]["anchors"][a]
            ax.axhspan(w["best_score"], w["best_score"] + d, color="#eda100", alpha=0.25, lw=0)
            ax.axvspan(w["window_y"][1], w["window_y"][0], color="#c4c2b9", alpha=0.35, lw=0)
            if slab_prior is not None:
                ax.axvline(slab_prior["landmark"]["y_mm"], color="#1baf7a", lw=1.0, ls="--")
                for edge in slab_prior["search_interval"]["y_mm"]:
                    ax.axvline(edge, color="#52514e", lw=0.8, ls=":")
            ax.set_title(f"{a}, {level_names[lv]}", fontsize=10)
            ax.set_xlabel("coronal plane y (MNI mm)")
            ax.set_ylabel("varifold score (0 = identical)")
            ax.invert_xaxis()
    handles = [Patch(fc="#eda100", alpha=0.25, label="best score + δ"), Patch(fc="#c4c2b9", alpha=0.35, label="window (within δ)")]
    if slab_prior is not None:
        handles += [Line2D([], [], color="#1baf7a", lw=1.0, ls="--",
                           label=f"{slab_prior['landmark']['name']} first appearance"),
                    Line2D([], [], color="#52514e", lw=0.8, ls=":", label="slab-prior search bound")]
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles), frameon=False)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def poses_figure(panels: Sequence[Mapping[str, Any]], ncols: int, legend_regions: Sequence[str], support: bool,
                 path: Path, *, panel_size=(6.0, 6.4), title_size: float = 10, title_loc: str = "center",
                 suptitle: str | None = None, dpi: int = 130) -> None:
    """Grid of ``draw_pose`` panels; each panel: {kwargs for draw_pose, title, flagged}."""

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(panel_size[0] * ncols, panel_size[1] * nrows), squeeze=False,
                             constrained_layout=True)
    for ax, panel in zip(axes.flat, panels):
        draw_pose(ax, **panel["pose"])
        ax.set_title(panel["title"], fontsize=title_size, loc=title_loc, color=FLAG_COLOR if panel["flagged"] else INK)
    fig.legend(handles=pose_legend_handles(legend_regions, support=support), loc="outside lower center", ncol=4,
               frameon=False, fontsize=9)
    if suptitle:
        fig.suptitle(suptitle, fontsize=10)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
