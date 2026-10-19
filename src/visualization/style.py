"""Matplotlib conventions shared by every static figure.

Palettes follow the project's dataviz reference instance (the same hex values as
``prepare_atlas_plane_and_section.CLASS_COLORS``), validated with the dataviz
validator: the first three categorical slots pass the all-pairs colour-vision checks
(the cap for spatial maps), and the five-step blue ordinal ramp passes ``--ordinal``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

CAT3 = ("#2a78d6", "#eb6834", "#1baf7a")            # categorical slots 1-3 (all-pairs safe)
ORDINAL5 = ("#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b")  # pia -> white matter
BLUE_RAMP = ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b")
ORANGE_RAMP = ("#fde4d6", "#f9b994", "#f28c5a", "#eb6834", "#c4501f", "#94391a", "#652713")
TEXT, TEXT_2, MUTED, GRID = "#0b0b0b", "#52514e", "#8b8a85", "#c3c2b7"
SURFACE = "#ffffff"
STATUS_LABEL = {"measured": "measured", "transformed": "measured, transformed", "derived": "derived",
                "inferred": "inferred (registration estimate)"}

SEQ_BLUE = LinearSegmentedColormap.from_list("seq_blue", BLUE_RAMP)
SEQ_ORANGE = LinearSegmentedColormap.from_list("seq_orange", ORANGE_RAMP)
LAYER_CMAP = ListedColormap(ORDINAL5, name="layers")

MM = 1 / 25.4
FULL_WIDTH_MM = 180.0


def setup() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6,
        "legend.fontsize": 6, "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "axes.edgecolor": TEXT_2, "axes.labelcolor": TEXT, "text.color": TEXT, "xtick.color": TEXT_2,
        "ytick.color": TEXT_2, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "pdf.fonttype": 42, "svg.fonttype": "none", "savefig.dpi": 300,
        "image.interpolation": "nearest", "axes.titlelocation": "left", "axes.titlepad": 3,
    })


def figure(width_mm: float, height_mm: float):
    setup()
    return plt.figure(figsize=(width_mm * MM, height_mm * MM))


def map_axes(ax, extent: Sequence[float] | None = None, y_down: bool = True) -> None:
    """Equal-aspect spatial panel without ticks (a scale bar carries the length)."""

    if extent is not None:
        x0, x1, y0, y1 = extent
        ax.set_xlim(x0, x1)
        ax.set_ylim((y1, y0) if y_down else (y0, y1))
    ax.set_aspect("equal")
    ax.set_xticks([]), ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def scale_bar(ax, length: float, label: str, loc: str = "lower right") -> None:
    bar = AnchoredSizeBar(ax.transData, length, label, loc, pad=0.3, sep=2, borderpad=0.4, frameon=False,
                          size_vertical=0, color=TEXT, fontproperties={"size": 6})
    bar.size_bar.get_children()[0].set_linewidth(1.2)
    ax.add_artist(bar)


def axes_glyph(ax, x_label: str, y_label: str, y_down: bool, loc: tuple[float, float] = (0.04, 0.04),
               length: float = 0.09) -> None:
    """Small arrows naming the frame's +x / +y directions as drawn (orientation as stored)."""

    x0, y0 = loc
    kw = dict(xycoords="axes fraction", textcoords="axes fraction",
              arrowprops=dict(arrowstyle="-|>", lw=0.6, color=TEXT_2, shrinkA=0, shrinkB=0, mutation_scale=5))
    # +y down in the frame is drawn downward, so place the origin at the top of the glyph.
    oy = y0 + length if y_down else y0
    ax.annotate("", (x0 + length, oy), (x0, oy), **kw)
    ax.annotate("", (x0, oy - length if y_down else oy + length), (x0, oy), **kw)
    ax.text(x0 + length + 0.01, oy, x_label, transform=ax.transAxes, fontsize=5, va="center", color=TEXT_2)
    ax.text(x0, (oy - length - 0.012) if y_down else (oy + length + 0.012), y_label, transform=ax.transAxes,
            fontsize=5, ha="center", va="top" if y_down else "bottom", color=TEXT_2)


def panel_letter(ax, letter: str, dx: float = -0.02, dy: float = 1.02) -> None:
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="right")


def status_tag(ax, status: str, loc: str = "upper right") -> None:
    """Marks a panel as measured / transformed / derived / inferred (a consistent vocabulary)."""

    x, ha = (0.99, "right") if "right" in loc else (0.01, "left")
    y, va = (0.99, "top") if "upper" in loc else (0.01, "bottom")
    ax.text(x, y, STATUS_LABEL.get(status, status), transform=ax.transAxes, fontsize=5, ha=ha, va=va,
            color=TEXT_2, style="italic")


def marker_size(ax, diameter: float) -> float:
    """Scatter ``s`` (pt^2) for markers of ``diameter`` data units in an equal-aspect axis."""

    fig = ax.figure
    fig.canvas.draw_idle()
    bbox = ax.get_window_extent()
    x0, x1 = ax.get_xlim()
    points_per_unit = bbox.width / abs(x1 - x0) * 72.0 / fig.dpi
    return float((diameter * points_per_unit) ** 2)


def colorbar(fig, mappable, ax, label: str, ticks: Iterable[float] | None = None, horizontal: bool = True):
    cb = fig.colorbar(mappable, ax=ax, orientation="horizontal" if horizontal else "vertical",
                      fraction=0.05, pad=0.02, aspect=25)
    cb.set_label(label, fontsize=6)
    cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(labelsize=5, width=0.4, length=2)
    if ticks is not None:
        cb.set_ticks(list(ticks))
    return cb


def save(fig, stem: str | Path, formats: Sequence[str] = ("png", "pdf")) -> list[Path]:
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    out = []
    for fmt in formats:
        path = stem.parent / f"{stem.name}.{fmt}"  # donor ids contain dots; never use with_suffix
        # Fixed metadata so reruns are byte-stable where the backend allows it.
        meta = {"png": {"Software": None}, "pdf": {"CreationDate": None, "Producer": None},
                "svg": {"Date": None}}.get(fmt, {})
        fig.savefig(path, dpi=300, metadata=meta)
        out.append(path)
    plt.close(fig)
    return out


def polygon_patches(rings: Sequence[np.ndarray], **kw):
    from matplotlib.collections import PolyCollection

    return PolyCollection([np.asarray(r)[:, :2] for r in rings], **kw)


def panel_title(ax, letter: str, title: str, status: str | None = None) -> None:
    """Bold panel letter, title and (optionally) the data status, stacked above the axes.

    Offsets are in points from the axes' top-left corner, so they hold for any panel size.
    """

    from matplotlib.transforms import offset_copy

    up = 11 if status else 3
    at = lambda dx, dy: offset_copy(ax.transAxes, fig=ax.figure, x=dx, y=dy, units="points")  # noqa: E731
    ax.text(0.0, 1.0, letter, transform=at(0, up), fontsize=9, fontweight="bold", va="bottom", ha="left")
    ax.text(0.0, 1.0, title, transform=at(9, up), fontsize=7, va="bottom", ha="left")
    if status:
        ax.text(0.0, 1.0, STATUS_LABEL.get(status, status), transform=at(9, 2), fontsize=5.5, va="bottom",
                ha="left", color=TEXT_2, style="italic")


def caption(fig, lines: Sequence[str], y: float = 0.01) -> None:
    fig.text(0.01, y, "\n".join(lines), fontsize=5.5, color=TEXT_2, va="bottom", ha="left", linespacing=1.3)
