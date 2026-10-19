"""SEA-AD quantitative-neuropathology (QNP) slides as feature-valued 2-D particles.

Coordinate frame
================
A QNP slide is read at one Aperio pyramid level (the smallest, ~8 um/pixel, by
default).  Its ``qnp_slide_mm`` frame is

    x = column * px_x_mm,   y = row * px_y_mm,   z = 0,

with the origin at the top-left baseline pixel and y pointing *down* the image.  Pixel
sizes come from the Aperio ``MPP`` tag times the level downsample, per slide (the LFB
scan is 0.5049 um at baseline, the others 0.5016 um).  Because y points down, this
frame is a mirror image of a right-handed view of the glass; registrations must allow a
reflection.  HALO ``.annotations`` vertices are baseline pixels and convert with the
baseline MPP only.

Feature models
==============
``tissue classes``  soft gray-matter / white-matter membership.  A per-stain scalar
                    (default: the red-channel share of optical density, high in
                    myelinated white matter for LFB, AT8, alpha-syn and NeuN slides) is
                    smoothed and split at its Otsu threshold inside tissue; a logistic
                    of the distance to that threshold gives ``p_wm``.  Feature masses
                    are ``area * (1 - p_wm)`` and ``area * p_wm`` (mm^2), so their sum
                    is tissue area whatever the stain.  This is the representation that
                    shares meaning with atlas gray/white-matter labels.
``optical density`` the total optical density, robustly scaled to [0, 1] and spread
                    over a few soft bins.  Stain-specific and not comparable across
                    stains or with MRI; kept as a baseline and as multistain features.

Both reuse dense_image_utils' block and bin aggregation with a singleton z axis, so
particle placement and mass conservation are the project's existing code.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage as ndi

from dense_image_utils import _aggregate_by_block, fit_intensity_transform

# HALO layer names vary between slides (``LAyerIII``, ``Layer V-VI``); map them to one
# vocabulary.  Order is outer (pia) to inner (white matter).
LAYER_ORDER = ("L1", "L2", "L3", "L4", "L5-6")
_ROMAN = {"I": "1", "II": "2", "III": "3", "IV": "4", "V-VI": "5-6", "VVI": "5-6"}


def pixel_affine(px_mm: tuple[float, float]) -> np.ndarray:
    """Index (row, column, 0) -> mm (x = column * px_x, y = row * px_y, z = 0)."""

    return np.array([[0.0, px_mm[1], 0.0, 0.0],
                     [px_mm[0], 0.0, 0.0, 0.0],
                     [0.0, 0.0, 1.0, 0.0],
                     [0.0, 0.0, 0.0, 1.0]])


@dataclass
class Slide:
    """One pyramid level of an Aperio slide and the metadata needed to place it."""

    path: Path
    donor: str
    block: str
    stain: str
    rgb: np.ndarray
    level: int
    mpp_baseline_um: float
    baseline_shape: tuple[int, int]
    px_mm: tuple[float, float]  # (y = row, x = column)
    header: dict[str, str] = field(default_factory=dict)

    @property
    def affine(self) -> np.ndarray:
        return pixel_affine(self.px_mm)

    @property
    def pixel_area_mm2(self) -> float:
        return float(self.px_mm[0] * self.px_mm[1])

    def describe(self) -> dict[str, Any]:
        return {
            "file": str(self.path), "donor": self.donor, "block": self.block, "stain": self.stain,
            "level": self.level, "level_shape": list(self.rgb.shape[:2]),
            "baseline_shape": list(self.baseline_shape), "mpp_baseline_um": self.mpp_baseline_um,
            "px_mm_row_col": list(self.px_mm), "scan_date": self.header.get("Date"),
            "scanner_id": self.header.get("ScanScope ID"), "app_mag": self.header.get("AppMag"),
            "extent_mm_xy": [self.rgb.shape[1] * self.px_mm[1], self.rgb.shape[0] * self.px_mm[0]],
        }


def _aperio_header(description: str) -> dict[str, str]:
    fields = {}
    for part in description.split("|")[1:]:
        if "=" in part:
            key, value = part.split("=", 1)
            fields[key.strip()] = value.strip()
    return fields


def parse_slide_name(path: Path) -> tuple[str, str, str]:
    """``H20.33.034-A7-LFB.svs`` -> (donor, block, stain)."""

    match = re.fullmatch(r"(H\d+\.\d+\.\d+)-([A-Za-z0-9]+)-([A-Za-z0-9]+)", path.stem)
    if not match:
        raise ValueError(f"Unexpected QNP slide name: {path.name}")
    return match.group(1), match.group(2), match.group(3).upper()


def read_slide(path: str | Path, level: int = -1) -> Slide:
    """Read one level of an Aperio .svs (never the Label/Macro series)."""

    import tifffile

    path = Path(path)
    donor, block, stain = parse_slide_name(path)
    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        header = _aperio_header(tif.pages[0].description)
        baseline = series.levels[0].shape[:2]
        rgb = series.levels[level].asarray()
    if "MPP" not in header:
        raise ValueError(f"{path.name} has no Aperio MPP tag; physical size unknown.")
    mpp = float(header["MPP"])
    px_mm = (mpp * baseline[0] / rgb.shape[0] / 1000.0, mpp * baseline[1] / rgb.shape[1] / 1000.0)
    resolved = level if level >= 0 else len(series.levels) + level
    return Slide(path, donor, block, stain, rgb[..., :3], resolved, mpp, tuple(baseline), px_mm, header)


def optical_density(rgb: np.ndarray, background_percentile: float = 99.5) -> tuple[np.ndarray, list[float]]:
    """Per-channel optical density ``-log(I / I0)`` with I0 from the bright background."""

    rgb = np.asarray(rgb, dtype=np.float64)
    background = np.percentile(rgb.reshape(-1, 3), background_percentile, axis=0)
    od = -np.log(np.clip(rgb / background, 1e-3, 1.0))
    return od, background.tolist()


def _otsu(values: np.ndarray, bins: int = 256) -> float:
    from skimage.filters import threshold_otsu

    return float(threshold_otsu(values, nbins=bins))


def tissue_mask(od: np.ndarray, pixel_area_mm2: float, smooth_px: float = 3.0,
                min_object_mm2: float = 1.0, max_hole_mm2: float = 0.05,
                opening_radius_mm: float = 0.075) -> tuple[np.ndarray, dict]:
    """Tissue = smoothed total OD above a log-domain Otsu threshold, minus specks, small holes filled.

    The threshold is Otsu on log(OD + 0.01): in linear OD, slides with a dark rim and a
    pale interior (GFAP) split rim from interior instead of tissue from glass.  Large
    holes (tears, sulcal gaps) are kept as background: they are real absence of tissue
    on the glass.  Thin slide-edge lines are removed by the size filter.
    """

    from skimage.morphology import disk, remove_small_holes, remove_small_objects

    total = ndi.gaussian_filter(od.sum(axis=-1), smooth_px)
    threshold = float(np.exp(_otsu(np.log(total + 0.01))) - 0.01)
    mask = total > threshold
    # Opening removes structures thinner than ~2 * opening_radius (coverslip/slide-edge
    # lines); the size filter then removes specks.
    radius = max(1, int(round(opening_radius_mm / np.sqrt(pixel_area_mm2))))
    mask = ndi.binary_opening(mask, structure=disk(radius))
    mask = remove_small_objects(mask, max(1, int(min_object_mm2 / pixel_area_mm2)))
    mask = remove_small_holes(mask, max(1, int(max_hole_mm2 / pixel_area_mm2)))
    labels, number = ndi.label(mask)
    sizes = np.bincount(labels.ravel())[1:] * pixel_area_mm2
    return mask, {"threshold_total_od": threshold, "threshold_rule": "exp(otsu(log(od + 0.01))) - 0.01",
                  "opening_radius_mm": opening_radius_mm, "smooth_px": smooth_px, "tissue_area_mm2": float(mask.sum() * pixel_area_mm2),
                  "components": int(number), "component_areas_mm2": sorted(np.round(sizes, 2).tolist(), reverse=True)[:5]}


def structural_feature(od: np.ndarray, rgb: np.ndarray, mode: str, smooth_px: float) -> np.ndarray:
    """Smoothed per-pixel scalar used to separate gray from white matter."""

    if mode == "red_od_share":
        # Ratio of smoothed sums, not smoothed ratio: background-adjacent pixels with tiny
        # OD otherwise dominate.
        return ndi.gaussian_filter(od[..., 0], smooth_px) / (ndi.gaussian_filter(od.sum(-1), smooth_px) + 1e-6)
    if mode in {"dab", "hematoxylin"}:
        from skimage.color import rgb2hed

        channel = rgb2hed(rgb)[..., 2 if mode == "dab" else 0]
        return ndi.gaussian_filter(channel, smooth_px)
    raise ValueError(f"Unknown structural feature '{mode}'. Use red_od_share, dab or hematoxylin.")


def gm_wm_membership(feature: np.ndarray, mask: np.ndarray, wm_high: bool,
                     softness_fraction: float = 0.05, edge_exclusion_px: float = 0.0) -> tuple[np.ndarray, dict]:
    """Soft white-matter probability from a two-class Otsu split inside tissue.

    ``p_wm = logistic((f - t) / w)`` with ``t`` the Otsu threshold and ``w`` a fixed
    fraction of the 5-95 % range.  ``t`` and ``w`` are fitted on tissue eroded by
    ``edge_exclusion_px`` so that the stain-darkened rim of a section does not become
    one of the two classes.  Returned diagnostics include the Otsu separability
    (between-class / total variance), a crude check that the feature is bimodal.
    """

    fit_mask = mask
    if edge_exclusion_px > 0:
        fit_mask = ndi.binary_erosion(mask, iterations=int(round(edge_exclusion_px)))
        if fit_mask.sum() < 0.2 * mask.sum():
            fit_mask = mask
    values = feature[fit_mask]
    threshold = _otsu(values)
    low, high = np.percentile(values, [5, 95])
    width = max(softness_fraction * (high - low), 1e-9)
    z = (feature - threshold) / width
    p_wm = 1.0 / (1.0 + np.exp(-np.clip(z if wm_high else -z, -50, 50)))
    p_wm = np.where(mask, p_wm, 0.0)
    above = values > threshold if wm_high else values < threshold
    w1 = above.mean()
    between = w1 * (1 - w1) * (values[above].mean() - values[~above].mean()) ** 2 if 0 < w1 < 1 else 0.0
    return p_wm, {
        "otsu_threshold": threshold, "logistic_width": width, "wm_high": wm_high,
        "edge_exclusion_px": float(edge_exclusion_px), "fit_fraction_of_tissue": float(fit_mask.sum() / mask.sum()),
        "otsu_separability": float(between / max(values.var(), 1e-30)),
        "wm_fraction_of_tissue": float(p_wm[mask].mean()),
        "ambiguous_fraction_0.2_0.8": float(np.mean((p_wm[mask] > 0.2) & (p_wm[mask] < 0.8))),
    }


# --------------------------------------------------------------------------------------
# HALO layer annotations
# --------------------------------------------------------------------------------------

def canonical_layer(name: str) -> str | None:
    """Map HALO layer names (``Layer1``, ``LAyerIII``, ``Layer V-VI``) to LAYER_ORDER."""

    text = re.sub(r"(?i)^\s*layer\s*", "", name.strip()).replace(" ", "").upper()
    text = _ROMAN.get(text, text)
    candidate = f"L{text}"
    return candidate if candidate in LAYER_ORDER else None


def read_halo_layers(path: str | Path) -> dict[str, list[tuple[np.ndarray, bool]]]:
    """{canonical layer: [(baseline-pixel polygon (x, y), is_hole), ...]}."""

    layers: dict[str, list[tuple[np.ndarray, bool]]] = {}
    for annotation in ET.parse(path).getroot().iter("Annotation"):
        name = canonical_layer(annotation.get("Name", ""))
        if name is None:
            raise ValueError(f"Unrecognized HALO layer name {annotation.get('Name')!r} in {path}")
        for region in annotation.iter("Region"):
            xy = np.array([(float(v.get("X")), float(v.get("Y"))) for v in region.iter("V")])
            if len(xy) >= 3:
                layers.setdefault(name, []).append((xy, region.get("NegativeROA") == "1"))
    return layers


def rasterize_layers(layers: dict[str, list[tuple[np.ndarray, bool]]], slide: Slide) -> np.ndarray:
    """Layer index image on the slide's level grid: 0 = none, k = LAYER_ORDER[k - 1].

    Polygons are drawn outer layers first so that, where HALO polygons overlap, the
    deeper layer wins; holes are cleared within their own layer only.
    """

    # PIL's scanline fill; skimage.draw.polygon is O(bbox pixels x vertices), and HALO
    # polygons carry thousands of vertices.
    from PIL import Image, ImageDraw

    scale = np.array([slide.rgb.shape[1] / slide.baseline_shape[1], slide.rgb.shape[0] / slide.baseline_shape[0]])
    image = np.zeros(slide.rgb.shape[:2], dtype=np.uint8)
    for index, name in enumerate(LAYER_ORDER, start=1):
        canvas = Image.new("1", (image.shape[1], image.shape[0]), 0)
        draw = ImageDraw.Draw(canvas)
        for xy, hole in sorted(layers.get(name, []), key=lambda item: item[1]):
            draw.polygon([tuple(p) for p in (xy * scale).tolist()], fill=0 if hole else 1)
        image[np.asarray(canvas, dtype=bool)] = index
    return image


def ribbon_depth_points(layer_image: np.ndarray, slide: Slide, step_px: int) -> dict[str, np.ndarray]:
    """Points of the annotated cortical ribbon with an approximate depth from pia.

    Depth is the Euclidean distance (um) to the outer edge of layer 1, i.e. layer-1
    pixels that touch unannotated pixels.  It is a QC coordinate (as in the MERFISH
    strip), not a measured laminar depth: it is wrong where the ribbon meets a cut
    tissue edge and where sulcal banks touch.
    """

    ribbon = layer_image > 0
    pial_edge = (layer_image == 1) & ndi.binary_dilation(~ribbon)
    if not pial_edge.any():
        raise ValueError("Layer 1 has no outer edge; cannot define depth from pia.")
    distance = ndi.distance_transform_edt(~pial_edge, sampling=(slide.px_mm[0] * 1000.0, slide.px_mm[1] * 1000.0))
    rows, cols = np.nonzero(ribbon[::step_px, ::step_px])
    rows, cols = rows * step_px, cols * step_px
    return {
        "X": np.column_stack([cols * slide.px_mm[1], rows * slide.px_mm[0], np.zeros(len(rows))]).astype(np.float32),
        "depth_from_pia_um": distance[rows, cols].astype(np.float32),
        "layer_index": layer_image[rows, cols].astype(np.int16),
        "layer_names": np.asarray(LAYER_ORDER, dtype=str),
    }


# --------------------------------------------------------------------------------------
# Particles
# --------------------------------------------------------------------------------------

def downsample_factor_for(px_mm: tuple[float, float], bin_mm: float) -> int:
    return max(1, int(round(bin_mm / float(np.mean(px_mm)))))


def _particles_2d(px_mm: tuple[float, float], mask: np.ndarray, values: np.ndarray, bin_mm: float, bins: int):
    """Block-aggregate masked pixels into particles with soft-binned ``values`` in [0, 1].

    Uses dense_image_utils' block aggregation directly (its public wrapper insists on a
    3-D volume); the singleton z axis keeps the affine convention of that module.
    """

    factor = downsample_factor_for(px_mm, bin_mm)
    ijk = np.column_stack([*np.nonzero(mask), np.zeros(int(mask.sum()), dtype=np.int64)])
    area = float(px_mm[0] * px_mm[1])
    selected = np.clip(values[mask], 0.0, 1.0)
    X, inverse, mass, mass_value = _aggregate_by_block(
        ijk, selected, (*mask.shape, 1), pixel_affine(px_mm), factor, "affine_world", "block_center", area)
    # Linear (soft) binning as in dense_image_utils._aggregate_binned_features, but in
    # float64: float32 bins cannot sum millions of 6e-5 mm^2 pixel areas exactly.
    scaled = selected * (bins - 1)
    lower = np.floor(scaled).astype(np.int64)
    upper = np.minimum(lower + 1, bins - 1)
    weight_up = scaled - lower
    size = len(X) * bins
    nu_bin = (np.bincount(inverse * bins + lower, (1.0 - weight_up) * area, minlength=size)
              + np.bincount(inverse * bins + upper, weight_up * area, minlength=size)).reshape(len(X), bins)
    expected = float(mask.sum()) * area
    error = abs(float(nu_bin.sum()) - expected) / expected
    if error > 1e-9:
        raise RuntimeError(f"Particle mass not conserved: relative error {error}.")
    edges, centers = np.linspace(0.0, 1.0, bins + 1), np.linspace(0.0, 1.0, bins)
    diagnostics = {"downsample_factor": factor, "bin_mm_actual": factor * float(np.mean(px_mm)),
                   "number_particles": int(len(X)), "selected_pixels": int(mask.sum()),
                   "pixel_area_mm2": area, "tissue_area_mm2": expected, "mass_relative_error": error,
                   "position_mode": "block_center"}
    nu_intensity = np.column_stack([mass, mass_value]).astype(np.float32)
    return X.astype(np.float32), nu_intensity, nu_bin, diagnostics, {"bin_edges": edges, "bin_centers": centers}


def tissue_class_particles(px_mm: tuple[float, float], mask: np.ndarray, p_wm: np.ndarray, bin_mm: float):
    """[N,3] positions (mm) and [N,2] gray/white masses (mm^2) on a ``bin_mm`` grid.

    Two linear bins split each pixel's area as (1 - p_wm, p_wm), so the columns are
    gray and white area and their sum is tissue area.
    """

    X, _, nu, diagnostics, _ = _particles_2d(px_mm, mask, p_wm, bin_mm, bins=2)
    diagnostics["feature_mass_definition"] = "nu[:,0] = sum area*(1-p_wm) (gray), nu[:,1] = sum area*p_wm (white), mm^2"
    return X, nu.astype(np.float32), diagnostics


def optical_density_particles(slide: Slide, mask: np.ndarray, od: np.ndarray, bin_mm: float, bins: int):
    """Stain-specific baseline: total OD robustly scaled, soft-binned per particle."""

    total = od.sum(axis=-1)
    transform = fit_intensity_transform(total, mask, {"normalization": "robust_minmax"})
    X, nu_intensity, nu_bin, diagnostics, bin_arrays = _particles_2d(
        slide.px_mm, mask, transform.forward(total), bin_mm, bins)
    return X, nu_intensity, nu_bin, transform, diagnostics, bin_arrays
