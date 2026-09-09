#!/usr/bin/env python3
"""Render one Allen section OME-Zarr package as a compact PNG montage."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
try:
    import zarr
except ModuleNotFoundError:  # Pure metadata/render helpers do not require Zarr.
    zarr = None
from PIL import Image, ImageDraw, ImageFont


BACKGROUND = (18, 18, 18)
COMBINED_BACKGROUND = (255, 255, 255)
BOUNDARY_COLOR = (0, 0, 0)
OVERLAY_BOUNDARY_COLOR = (255, 0, 255)
SHEET_BACKGROUND = (245, 245, 245)
TEXT_COLOR = (24, 24, 24)
MIN_PANEL_WIDTH = 64
DEFAULT_ANNOTATIONS_ZARR = Path(
    "data/derivatives/allen/specimen_708424/annotations_ome_zarr"
)
GROUP_PANEL_FOOTNOTE = (
    "Graphic-group panels use visibility-preserving label projection; "
    "not exact categorical resampling."
)
COMBINED_PREVIEW_TITLE = "Combined modified-Brodmann atlas preview"
BOUNDARY_OVERLAY_TITLE = "Combined boundaries on Nissl"


@dataclass(frozen=True)
class Panel:
    title: str
    image: Image.Image


def _mapping(value: Any, description: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"Missing or invalid {description}")
    return value


def _list(value: Any, description: str) -> list[Any]:
    if not isinstance(value, list) or not value:
        raise RuntimeError(f"Missing or invalid {description}")
    return value


def _multiscale(group: Any, description: str) -> tuple[str, tuple[str, ...]]:
    attrs = _mapping(dict(group.attrs), f"{description} attributes")
    ome = _mapping(attrs.get("ome"), f"{description} OME metadata")
    multiscales = _list(ome.get("multiscales"), f"{description} multiscales")
    multiscale = _mapping(multiscales[0], f"{description} multiscale")
    datasets = _list(multiscale.get("datasets"), f"{description} datasets")
    dataset = _mapping(datasets[0], f"{description} dataset")
    path = dataset.get("path")
    if not isinstance(path, str) or not path:
        raise RuntimeError(f"Missing or invalid {description} dataset path")
    axes = _list(multiscale.get("axes"), f"{description} axes")
    names: list[str] = []
    for axis in axes:
        name = _mapping(axis, f"{description} axis").get("name")
        if not isinstance(name, str) or not name:
            raise RuntimeError(f"Missing or invalid {description} axis name")
        names.append(name)
    return path, tuple(names)


def _read_array(group: Any, path: str, description: str) -> np.ndarray:
    try:
        return np.asarray(group[path][:])
    except Exception as exc:
        raise RuntimeError(f"Cannot read {description} array at {path!r}") from exc


def _rgb_image(root: Any) -> tuple[np.ndarray, int, int]:
    path, axes = _multiscale(root, "section image")
    if len(axes) != 3 or set(axes) != {"c", "y", "x"}:
        raise RuntimeError(
            "Unsupported section image axes; expected exactly c, y, and x"
        )
    data = _read_array(root, path, "section image")
    if data.ndim != 3:
        raise RuntimeError(f"Section image must be 3-D, found shape {data.shape}")
    data = np.transpose(data, tuple(axes.index(name) for name in ("y", "x", "c")))
    if data.shape[2] != 3 or data.dtype != np.uint8:
        raise RuntimeError(
            "Section image must contain three uint8 RGB channels; "
            f"found shape {data.shape} and dtype {data.dtype}"
        )
    height, width = data.shape[:2]
    return data, width, height


def _label_colors(group: Any, description: str) -> list[tuple[int, tuple[int, ...]]]:
    attrs = _mapping(dict(group.attrs), f"{description} attributes")
    ome = _mapping(attrs.get("ome"), f"{description} OME metadata")
    image_label = _mapping(ome.get("image-label"), f"{description} image-label")
    colors = _list(image_label.get("colors"), f"{description} colors")
    result: list[tuple[int, tuple[int, ...]]] = []
    seen: set[int] = set()
    for item in colors:
        color = _mapping(item, f"{description} color")
        try:
            value = int(color["label-value"])
            rgba = tuple(int(channel) for channel in color["rgba"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"Invalid color entry in {description}") from exc
        if value <= 0 or value in seen:
            raise RuntimeError(f"Invalid or duplicate label value {value} in {description}")
        if len(rgba) != 4 or any(channel < 0 or channel > 255 for channel in rgba):
            raise RuntimeError(f"Invalid RGBA color for label value {value} in {description}")
        seen.add(value)
        result.append((value, rgba))
    return result


def _label_title(group: Any, name: str) -> str:
    attrs = _mapping(dict(group.attrs), f"label group {name} attributes")
    allen = _mapping(attrs.get("allen"), f"label group {name} Allen metadata")
    title = allen.get("graphic_group_name")
    if not isinstance(title, str) or not title.strip():
        raise RuntimeError(f"Missing graphic_group_name for label group {name}")
    return title.strip()


def _composite_color(rgba: tuple[int, ...]) -> np.ndarray:
    alpha = rgba[3] / 255.0
    return np.rint(
        np.asarray(rgba[:3], dtype=np.float64) * alpha
        + np.asarray(BACKGROUND, dtype=np.float64) * (1.0 - alpha)
    ).astype(np.uint8)


def _label_thumbnail(
    labels: np.ndarray,
    colors: list[tuple[int, tuple[int, ...]]],
    size: tuple[int, int],
    description: str,
) -> Image.Image:
    if labels.ndim != 2:
        raise RuntimeError(f"{description} must be 2-D, found shape {labels.shape}")
    if not np.issubdtype(labels.dtype, np.integer):
        raise RuntimeError(f"{description} must contain integer labels")

    observed = {int(value) for value in np.unique(labels) if int(value) != 0}
    defined = {value for value, _ in colors}
    missing = sorted(observed - defined)
    if missing:
        raise RuntimeError(
            f"{description} has nonzero values without color metadata: {missing}"
        )

    target_width, target_height = size
    source_height, source_width = labels.shape
    thumbnail = np.empty((target_height, target_width, 3), dtype=np.uint8)
    thumbnail[:] = BACKGROUND

    # Project every occupied source pixel into its destination cell. Painting
    # larger regions first leaves sparse structures visible at thumbnail scale.
    masks: list[tuple[int, int, tuple[int, ...], np.ndarray, np.ndarray]] = []
    for order, (value, rgba) in enumerate(colors):
        if value not in observed:
            continue
        ys, xs = np.nonzero(labels == value)
        masks.append((ys.size, order, rgba, ys, xs))
    for _, _, rgba, ys, xs in sorted(masks, key=lambda item: (-item[0], item[1])):
        target_y = np.minimum(ys * target_height // source_height, target_height - 1)
        target_x = np.minimum(xs * target_width // source_width, target_width - 1)
        thumbnail[target_y, target_x] = _composite_color(rgba)
    return Image.fromarray(thumbnail)


def _combined_display_map(label_layers: Sequence[np.ndarray]) -> np.ndarray:
    if not label_layers:
        raise RuntimeError("Cannot compose an empty label-layer list")
    combined_display_labels = np.zeros(label_layers[0].shape, dtype=np.uint32)
    for labels in label_layers:
        if labels.shape != combined_display_labels.shape:
            raise RuntimeError("Graphic-group label shapes do not match")
        if not np.issubdtype(labels.dtype, np.integer):
            raise RuntimeError("Graphic-group labels must contain integer values")
        occupied = labels != 0
        combined_display_labels[occupied] = labels[occupied]
    return combined_display_labels


def _merged_colors(
    color_layers: Sequence[list[tuple[int, tuple[int, ...]]]],
) -> list[tuple[int, tuple[int, ...]]]:
    result: list[tuple[int, tuple[int, ...]]] = []
    observed: dict[int, tuple[int, ...]] = {}
    for colors in color_layers:
        for value, rgba in colors:
            prior = observed.get(value)
            if prior is not None and prior != rgba:
                raise RuntimeError(
                    f"Label value {value} has conflicting colors across graphic groups"
                )
            if prior is None:
                observed[value] = rgba
                result.append((value, rgba))
    return result


def _resize_labels_nearest(
    labels: np.ndarray, size: tuple[int, int]
) -> np.ndarray:
    if labels.ndim != 2 or not np.issubdtype(labels.dtype, np.integer):
        raise RuntimeError("Combined display labels must be a 2-D integer array")
    return np.asarray(
        Image.fromarray(labels.astype(np.int32, copy=False)).resize(
            size, Image.Resampling.NEAREST
        ),
        dtype=np.uint32,
    )


def _internal_boundary_mask(labels: np.ndarray) -> np.ndarray:
    boundary = np.zeros(labels.shape, dtype=bool)
    horizontal = (
        (labels[:, 1:] != labels[:, :-1])
        & (labels[:, 1:] != 0)
        & (labels[:, :-1] != 0)
    )
    boundary[:, 1:] |= horizontal
    boundary[:, :-1] |= horizontal
    vertical = (
        (labels[1:, :] != labels[:-1, :])
        & (labels[1:, :] != 0)
        & (labels[:-1, :] != 0)
    )
    boundary[1:, :] |= vertical
    boundary[:-1, :] |= vertical
    return boundary


def _combined_display_thumbnail(
    combined_display_labels: np.ndarray,
    colors: list[tuple[int, tuple[int, ...]]],
    size: tuple[int, int],
) -> Image.Image:
    resized = _resize_labels_nearest(combined_display_labels, size)
    observed = {int(value) for value in np.unique(resized) if int(value) != 0}
    defined = {value for value, _ in colors}
    missing = sorted(observed - defined)
    if missing:
        raise RuntimeError(
            f"Combined display has nonzero values without color metadata: {missing}"
        )

    rgb = np.empty((*resized.shape, 3), dtype=np.uint8)
    rgb[:] = COMBINED_BACKGROUND
    background = np.asarray(COMBINED_BACKGROUND, dtype=np.float64)
    for value, rgba in colors:
        if value not in observed:
            continue
        alpha = rgba[3] / 255.0
        color = np.rint(
            np.asarray(rgba[:3], dtype=np.float64) * alpha
            + background * (1.0 - alpha)
        ).astype(np.uint8)
        rgb[resized == value] = color
    rgb[_internal_boundary_mask(resized)] = BOUNDARY_COLOR
    return Image.fromarray(rgb)


def _boundary_overlay(
    rgb: np.ndarray,
    combined_display_labels: np.ndarray,
    size: tuple[int, int],
) -> Image.Image:
    if rgb.shape[:2] != combined_display_labels.shape:
        raise RuntimeError("Nissl and combined-display shapes do not match")
    edge = np.zeros(combined_display_labels.shape, dtype=bool)
    edge[1:, :] |= (
        combined_display_labels[1:, :] != combined_display_labels[:-1, :]
    )
    edge[:, 1:] |= (
        combined_display_labels[:, 1:] != combined_display_labels[:, :-1]
    )
    overlay = rgb.copy()
    overlay[edge & (combined_display_labels != 0)] = OVERLAY_BOUNDARY_COLOR
    return Image.fromarray(overlay).resize(size, Image.Resampling.LANCZOS)


def _label_names(labels_group: Any) -> list[str]:
    attrs = _mapping(dict(labels_group.attrs), "labels-container attributes")
    ome = _mapping(attrs.get("ome"), "labels-container OME metadata")
    values = _list(ome.get("labels"), "labels-container label list")
    if not all(isinstance(value, str) and value for value in values):
        raise RuntimeError("Labels-container label names must be nonempty strings")
    if len(values) != len(set(values)):
        raise RuntimeError("Labels-container label names must be unique")
    return values


def _panels_from_root(
    root: Any,
    rgb: np.ndarray,
    *,
    heading: str,
    panel_width: int,
    label_arrays: Mapping[str, np.ndarray] | None = None,
) -> tuple[str, list[Panel]]:
    """Render panels from OME metadata and optional external label arrays."""
    if panel_width < MIN_PANEL_WIDTH:
        raise RuntimeError(f"--panel-width must be at least {MIN_PANEL_WIDTH}")
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise RuntimeError(
            "Section image must contain three uint8 RGB channels; "
            f"found shape {rgb.shape} and dtype {rgb.dtype}"
        )
    height, width = rgb.shape[:2]
    panel_height = max(1, round(height * panel_width / width))
    nissl = Image.fromarray(rgb).resize(
        (panel_width, panel_height), Image.Resampling.LANCZOS
    )
    panels = [Panel("Nissl reference", nissl)]

    try:
        labels_group = root["labels"]
    except Exception as exc:
        raise RuntimeError("OME-Zarr package has no labels group") from exc
    names = _label_names(labels_group)
    label_layers: list[np.ndarray] = []
    color_layers: list[list[tuple[int, tuple[int, ...]]]] = []
    for name in names:
        try:
            group = labels_group[name]
        except Exception as exc:
            raise RuntimeError(f"Missing declared label group {name!r}") from exc
        title = _label_title(group, name)
        if label_arrays is None:
            dataset_path, axes = _multiscale(group, f"label group {name}")
            if axes != ("y", "x"):
                raise RuntimeError(
                    f"Unsupported axes for label group {name}; expected y, x"
                )
            labels = _read_array(group, dataset_path, f"label group {name}")
        else:
            try:
                labels = label_arrays[name]
            except KeyError as exc:
                raise RuntimeError(
                    f"Bilateral derivative has no declared label group {name!r}"
                ) from exc
        if labels.shape != (height, width):
            raise RuntimeError(
                f"Label group {name} shape {labels.shape} does not match "
                f"section image shape {(height, width)}"
            )
        colors = _label_colors(group, f"label group {name}")
        label_layers.append(labels)
        color_layers.append(colors)
        panels.append(
            Panel(
                title,
                _label_thumbnail(
                    labels,
                    colors,
                    (panel_width, panel_height),
                    f"Label group {name}",
                ),
            )
        )
    combined_display_labels = _combined_display_map(label_layers)
    combined_colors = _merged_colors(color_layers)
    panel_size = (panel_width, panel_height)
    panels.extend(
        [
            Panel(
                COMBINED_PREVIEW_TITLE,
                _combined_display_thumbnail(
                    combined_display_labels, combined_colors, panel_size
                ),
            ),
            Panel(
                BOUNDARY_OVERLAY_TITLE,
                _boundary_overlay(rgb, combined_display_labels, panel_size),
            ),
        ]
    )
    return heading, panels


def load_panels(package: Path, panel_width: int = 480) -> tuple[str, list[Panel]]:
    """Read an OME-Zarr package and return its heading and rendered panels."""
    package = package.expanduser().resolve()
    if not package.is_dir() or package.is_symlink():
        raise RuntimeError(f"OME-Zarr package is not a regular directory: {package}")
    try:
        root = zarr.open_group(str(package), mode="r")
    except Exception as exc:
        raise RuntimeError(f"Cannot open OME-Zarr package {package}") from exc
    rgb, _, _ = _rgb_image(root)
    root_attrs = _mapping(dict(root.attrs), "root attributes")
    allen = root_attrs.get("allen")
    section_number = allen.get("section_number") if isinstance(allen, Mapping) else None
    heading = (
        f"Allen section {section_number:04d} annotations"
        if isinstance(section_number, int)
        else f"{package.name.removesuffix('.ome.zarr')} annotations"
    )
    return _panels_from_root(root, rgb, heading=heading, panel_width=panel_width)


def _tsv_rows(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(encoding="utf-8", newline="") as stream:
            return list(csv.DictReader(stream, delimiter="\t"))
    except OSError as exc:
        raise RuntimeError(f"Cannot read bilateral derivative manifest {path}") from exc


def _derivative_path(dataset: Path, relative: str, description: str) -> Path:
    if not relative:
        raise RuntimeError(f"Missing {description} path in bilateral derivative")
    path = (dataset / relative).resolve()
    try:
        path.relative_to(dataset)
    except ValueError as exc:
        raise RuntimeError(
            f"Unsafe {description} path outside bilateral derivative"
        ) from exc
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"Missing or invalid {description}: {path}")
    return path


def load_bilateral_panels(
    dataset: Path,
    section_number: int,
    *,
    annotations_zarr: Path = DEFAULT_ANNOTATIONS_ZARR,
    panel_width: int = 480,
) -> tuple[str, list[Panel]]:
    """Render a symmetric TIFF derivative using its source OME-Zarr metadata."""
    dataset = dataset.expanduser().resolve()
    annotations_zarr = annotations_zarr.expanduser().resolve()
    if not dataset.is_dir() or dataset.is_symlink():
        raise RuntimeError(
            f"Bilateral derivative is not a regular directory: {dataset}"
        )
    physical = _tsv_rows(dataset / "metadata/physical_sections.tsv")
    matches = [
        row
        for row in physical
        if row.get("allen_section_number") == str(section_number)
    ]
    if len(matches) != 1 or matches[0].get("image_present") != "true":
        raise RuntimeError(
            f"Bilateral derivative must contain one present section {section_number}"
        )
    image_path = _derivative_path(
        dataset, matches[0].get("prepared_relative_path", ""), "section image"
    )
    try:
        with Image.open(image_path) as image:
            rgb = np.asarray(image.convert("RGB"))
    except OSError as exc:
        raise RuntimeError(f"Cannot read bilateral section image {image_path}") from exc

    inventory = _tsv_rows(dataset / "metadata/annotations.tsv")
    section_rows = [
        row for row in inventory if row.get("section_number") == str(section_number)
    ]
    if not section_rows:
        raise RuntimeError(
            f"Bilateral derivative has no annotations for section {section_number}"
        )
    label_arrays: dict[str, np.ndarray] = {}
    for row in section_rows:
        group_id = row.get("graphic_group_id", "")
        name = f"group-{group_id}"
        if not group_id or name in label_arrays:
            raise RuntimeError(
                f"Invalid or duplicate annotation group for section {section_number}"
            )
        label_path = _derivative_path(
            dataset, row.get("path", ""), f"annotation group {group_id}"
        )
        try:
            with Image.open(label_path) as image:
                label_arrays[name] = np.asarray(image)
        except OSError as exc:
            raise RuntimeError(
                f"Cannot read bilateral annotation {label_path}"
            ) from exc

    package = annotations_zarr / f"section-{section_number:04d}.ome.zarr"
    if not package.is_dir() or package.is_symlink():
        raise RuntimeError(
            f"Source OME-Zarr package is not a regular directory: {package}"
        )
    try:
        root = zarr.open_group(str(package), mode="r")
    except Exception as exc:
        raise RuntimeError(f"Cannot open source OME-Zarr package {package}") from exc
    source_attrs = _mapping(dict(root.attrs), "source root attributes")
    source_allen = _mapping(source_attrs.get("allen"), "source Allen metadata")
    if source_allen.get("section_number") != section_number:
        raise RuntimeError("Source OME-Zarr section metadata does not match request")
    declared = set(_label_names(root["labels"]))
    if set(label_arrays) != declared:
        raise RuntimeError(
            "Bilateral annotation groups do not match the source OME-Zarr package"
        )
    heading, panels = _panels_from_root(
        root,
        rgb,
        heading=f"Allen section {section_number:04d} bilateral annotations",
        panel_width=panel_width,
        label_arrays=label_arrays,
    )
    axis = rgb.shape[1] // 2
    seam_half_width = max(4, rgb.shape[1] // 24)
    seam = Image.fromarray(
        rgb[:, axis - seam_half_width : axis + seam_half_width]
    ).resize(panels[0].image.size, Image.Resampling.NEAREST)
    seam_draw = ImageDraw.Draw(seam)
    seam_draw.line(
        [(seam.width // 2 - 1, 0), (seam.width // 2 - 1, seam.height)],
        fill=(255, 0, 255),
        width=2,
    )
    panels.append(Panel("Medial seam close-up", seam))
    return heading, panels


def _font(size: int) -> ImageFont.ImageFont:
    return ImageFont.load_default(size=size)


def _fitted_font(
    draw: ImageDraw.ImageDraw,
    text: str,
    maximum_width: int,
    preferred_size: int,
    minimum_size: int = 10,
) -> ImageFont.ImageFont:
    for size in range(preferred_size, minimum_size - 1, -1):
        font = _font(size)
        box = draw.textbbox((0, 0), text, font=font)
        if box[2] - box[0] <= maximum_width:
            return font
    return _font(minimum_size)


def _render_montage(
    heading: str, panels: list[Panel], panel_width: int
) -> tuple[Image.Image, tuple[str, ...]]:
    columns = math.ceil(math.sqrt(len(panels)))
    rows = math.ceil(len(panels) / columns)
    panel_height = panels[0].image.height
    margin = 24
    gap = 20
    heading_height = 54
    title_height = 44
    footnote_height = 36
    sheet_width = 2 * margin + columns * panel_width + (columns - 1) * gap
    sheet_height = (
        2 * margin
        + heading_height
        + rows * (title_height + panel_height)
        + (rows - 1) * gap
        + footnote_height
    )
    sheet = Image.new("RGB", (sheet_width, sheet_height), SHEET_BACKGROUND)
    draw = ImageDraw.Draw(sheet)

    heading_font = _fitted_font(
        draw, heading, sheet_width - 2 * margin, preferred_size=28, minimum_size=14
    )
    heading_box = draw.textbbox((0, 0), heading, font=heading_font)
    heading_x = (sheet_width - (heading_box[2] - heading_box[0])) // 2
    draw.text((heading_x, margin), heading, font=heading_font, fill=TEXT_COLOR)

    for index, panel in enumerate(panels):
        row, column = divmod(index, columns)
        x = margin + column * (panel_width + gap)
        y = margin + heading_height + row * (title_height + panel_height + gap)
        title_font = _fitted_font(
            draw, panel.title, panel_width, preferred_size=21, minimum_size=10
        )
        title_box = draw.textbbox((0, 0), panel.title, font=title_font)
        title_x = x + (panel_width - (title_box[2] - title_box[0])) // 2
        draw.text((title_x, y), panel.title, font=title_font, fill=TEXT_COLOR)
        sheet.paste(panel.image, (x, y + title_height))

    footnote_font = _fitted_font(
        draw,
        GROUP_PANEL_FOOTNOTE,
        sheet_width - 2 * margin,
        preferred_size=16,
        minimum_size=10,
    )
    footnote_box = draw.textbbox((0, 0), GROUP_PANEL_FOOTNOTE, font=footnote_font)
    footnote_x = (sheet_width - (footnote_box[2] - footnote_box[0])) // 2
    draw.text(
        (footnote_x, sheet_height - margin - footnote_height // 2),
        GROUP_PANEL_FOOTNOTE,
        font=footnote_font,
        fill=TEXT_COLOR,
    )

    return sheet, tuple(panel.title for panel in panels[1:])


def render_montage(
    package: Path, panel_width: int = 480
) -> tuple[Image.Image, tuple[str, ...]]:
    """Render an OME-Zarr montage and return its non-Nissl panel titles."""
    heading, panels = load_panels(package, panel_width)
    return _render_montage(heading, panels, panel_width)


def render_bilateral_montage(
    dataset: Path,
    section_number: int,
    *,
    annotations_zarr: Path = DEFAULT_ANNOTATIONS_ZARR,
    panel_width: int = 480,
) -> tuple[Image.Image, tuple[str, ...]]:
    """Render a bilateral TIFF montage using source OME-Zarr display metadata."""
    heading, panels = load_bilateral_panels(
        dataset,
        section_number,
        annotations_zarr=annotations_zarr,
        panel_width=panel_width,
    )
    return _render_montage(heading, panels, panel_width)


def default_output_path(package: Path) -> Path:
    package = package.expanduser().resolve()
    stem = package.name.removesuffix(".ome.zarr")
    return package.parent / "thumbnails" / f"{stem}.png"


def write_montage(
    package: Path,
    output: Path | None = None,
    panel_width: int = 480,
    overwrite: bool = False,
) -> tuple[Path, tuple[str, ...]]:
    package = package.expanduser().resolve()
    destination = (
        output.expanduser().resolve() if output is not None else default_output_path(package)
    )
    if destination.suffix.lower() != ".png":
        raise RuntimeError(f"Output must be a PNG path: {destination}")
    try:
        destination.relative_to(package)
    except ValueError:
        pass
    else:
        raise RuntimeError("Output cannot be written inside the OME-Zarr package")
    if destination.exists() and not overwrite:
        raise RuntimeError(
            f"Output already exists: {destination}; use --overwrite to replace it"
        )

    montage, titles = render_montage(package, panel_width)
    destination.parent.mkdir(parents=True, exist_ok=True)
    mode = "wb" if overwrite else "xb"
    try:
        with destination.open(mode) as stream:
            montage.save(stream, format="PNG", optimize=True)
    except FileExistsError as exc:
        raise RuntimeError(
            f"Output already exists: {destination}; use --overwrite to replace it"
        ) from exc
    return destination, titles


def write_bilateral_montage(
    dataset: Path,
    section_number: int,
    output: Path | None = None,
    *,
    annotations_zarr: Path = DEFAULT_ANNOTATIONS_ZARR,
    panel_width: int = 480,
    overwrite: bool = False,
) -> tuple[Path, tuple[str, ...]]:
    dataset = dataset.expanduser().resolve()
    destination = (
        output.expanduser().resolve()
        if output is not None
        else dataset / "thumbnails" / f"section-{section_number:04d}.png"
    )
    if destination.suffix.lower() != ".png":
        raise RuntimeError(f"Output must be a PNG path: {destination}")
    annotation_dir = (dataset / "annotations").resolve()
    try:
        destination.relative_to(annotation_dir)
    except ValueError:
        pass
    else:
        raise RuntimeError(
            "Output cannot be written inside bilateral annotation data"
        )
    if destination.exists() and not overwrite:
        raise RuntimeError(
            f"Output already exists: {destination}; use --overwrite to replace it"
        )
    montage, titles = render_bilateral_montage(
        dataset,
        section_number,
        annotations_zarr=annotations_zarr,
        panel_width=panel_width,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    mode = "wb" if overwrite else "xb"
    try:
        with destination.open(mode) as stream:
            montage.save(stream, format="PNG", optimize=True)
    except FileExistsError as exc:
        raise RuntimeError(
            f"Output already exists: {destination}; use --overwrite to replace it"
        ) from exc
    return destination, titles


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "package", nargs="?", type=Path, help="section-NNNN.ome.zarr package"
    )
    result.add_argument(
        "--symmetric-dataset",
        type=Path,
        help="bilateral histology_symmetric derivative",
    )
    result.add_argument("--section-number", type=int)
    result.add_argument(
        "--annotations-zarr",
        type=Path,
        default=DEFAULT_ANNOTATIONS_ZARR,
        help="source OME-Zarr derivative supplying display metadata",
    )
    result.add_argument("--output", type=Path, help="destination PNG path")
    result.add_argument("--panel-width", type=int, default=480)
    result.add_argument("--overwrite", action="store_true")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.symmetric_dataset is None:
        if args.package is None or args.section_number is not None:
            parser().error(
                "provide PACKAGE, or --symmetric-dataset with --section-number"
            )
    elif args.package is not None or args.section_number is None:
        parser().error(
            "--symmetric-dataset requires --section-number and cannot be "
            "combined with PACKAGE"
        )
    try:
        if args.symmetric_dataset is None:
            destination, titles = write_montage(
                args.package,
                args.output,
                args.panel_width,
                args.overwrite,
            )
        else:
            destination, titles = write_bilateral_montage(
                args.symmetric_dataset,
                args.section_number,
                args.output,
                annotations_zarr=args.annotations_zarr,
                panel_width=args.panel_width,
                overwrite=args.overwrite,
            )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"Saved {destination} with {len(titles)} annotation panel(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
