#!/usr/bin/env python3
"""Rasterize Allen atlas SVG paths into independent categorical label images.

Skia is deliberately isolated in this module.  The public rasterization command
and the derivative validator do not expose or import a rendering engine.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping


BOUNDARY_POLICY = "Skia non-antialiased target-grid fill"
_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_TRANSFORM = re.compile(rf"([A-Za-z]+)\s*\(([^)]*)\)")
_NUMBER_RE = re.compile(_NUMBER)
_UNSUPPORTED_VISIBLE = {"clip-path", "mask", "filter"}


@dataclass(frozen=True)
class LayerResult:
    graphic_group_id: int
    graphic_group_name: str
    labels: Any
    path_count: int
    structure_ids: tuple[int, ...]
    conflicting_overlap_pixel_count: int
    repeated_same_id_pixel_count: int
    svg_order_min: int | None
    svg_order_max: int | None
    svg_order_nondecreasing: bool


@dataclass(frozen=True)
class _PaintPath:
    group_id: int
    group_name: str
    structure_id: int
    data: str
    transform: Any
    fill_rule: str
    order: int | None


def _dependencies() -> tuple[Any, Any, Any, Any]:
    try:
        import numpy as np
        import skia
        from defusedxml import ElementTree as safe_et
        from svg.path import parse_path
    except ImportError as exc:  # pragma: no cover - exercised by CLI smoke tests
        raise RuntimeError(
            "Rasterization requires numpy, skia-python, svg.path, and defusedxml; "
            "install configs/environment-rasterize-allen.yml"
        ) from exc
    return np, skia, safe_et, parse_path


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _style(element: Any, inherited: Mapping[str, str]) -> dict[str, str]:
    result = dict(inherited)
    inline = element.attrib.get("style", "")
    for part in inline.split(";"):
        if ":" in part:
            key, value = part.split(":", 1)
            result[key.strip().lower()] = value.strip()
    for key in (
        "display",
        "visibility",
        "fill",
        "fill-rule",
        "opacity",
        "fill-opacity",
        "stroke",
    ):
        if key in element.attrib:
            result[key] = element.attrib[key].strip()
    return result


def _matrix(values: list[list[float]], np: Any) -> Any:
    return np.asarray(values, dtype=np.float64)


def parse_transform(text: str | None, np: Any | None = None) -> Any:
    """Parse the standard SVG affine transform-list grammar."""
    if np is None:
        np, _, _, _ = _dependencies()
    result = np.eye(3, dtype=np.float64)
    if not text or not text.strip():
        return result
    position = 0
    for match in _TRANSFORM.finditer(text):
        if text[position : match.start()].strip(" ,\t\r\n"):
            raise RuntimeError(f"Unsupported SVG transform syntax: {text!r}")
        position = match.end()
        name = match.group(1)
        raw = match.group(2)
        numbers = [float(item) for item in _NUMBER_RE.findall(raw)]
        residue = _NUMBER_RE.sub("", raw).strip(" ,\t\r\n")
        if residue:
            raise RuntimeError(f"Unsupported SVG transform arguments: {text!r}")
        if name == "matrix" and len(numbers) == 6:
            a, b, c, d, e, f = numbers
            current = _matrix([[a, c, e], [b, d, f], [0, 0, 1]], np)
        elif name == "translate" and len(numbers) in (1, 2):
            tx, ty = numbers[0], numbers[1] if len(numbers) == 2 else 0.0
            current = _matrix([[1, 0, tx], [0, 1, ty], [0, 0, 1]], np)
        elif name == "scale" and len(numbers) in (1, 2):
            sx, sy = numbers[0], numbers[1] if len(numbers) == 2 else numbers[0]
            current = _matrix([[sx, 0, 0], [0, sy, 0], [0, 0, 1]], np)
        elif name == "rotate" and len(numbers) in (1, 3):
            angle = math.radians(numbers[0])
            cosine, sine = math.cos(angle), math.sin(angle)
            rotation = _matrix(
                [[cosine, -sine, 0], [sine, cosine, 0], [0, 0, 1]], np
            )
            if len(numbers) == 3:
                cx, cy = numbers[1:]
                to_center = _matrix([[1, 0, cx], [0, 1, cy], [0, 0, 1]], np)
                from_center = _matrix([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], np)
                current = to_center @ rotation @ from_center
            else:
                current = rotation
        elif name in {"skewX", "skewY"} and len(numbers) == 1:
            tangent = math.tan(math.radians(numbers[0]))
            current = (
                _matrix([[1, tangent, 0], [0, 1, 0], [0, 0, 1]], np)
                if name == "skewX"
                else _matrix([[1, 0, 0], [tangent, 1, 0], [0, 0, 1]], np)
            )
        else:
            raise RuntimeError(f"Unsupported SVG transform: {match.group(0)!r}")
        result = result @ current
    if text[position:].strip(" ,\t\r\n"):
        raise RuntimeError(f"Unsupported SVG transform syntax: {text!r}")
    return result


def _svg_extent(root: Any) -> tuple[float, float, float, float]:
    view_box = root.attrib.get("viewBox")
    if view_box:
        values = [float(item) for item in _NUMBER_RE.findall(view_box)]
        if len(values) != 4 or values[2] <= 0 or values[3] <= 0:
            raise RuntimeError(f"Invalid SVG viewBox: {view_box!r}")
        return values[0], values[1], values[2], values[3]

    def length(name: str) -> float:
        value = root.attrib.get(name, "")
        match = re.fullmatch(rf"\s*({_NUMBER})(?:px)?\s*", value)
        if not match or float(match.group(1)) <= 0:
            raise RuntimeError(f"SVG requires a positive {name} or viewBox")
        return float(match.group(1))

    return 0.0, 0.0, length("width"), length("height")


def _paths(root: Any, np: Any) -> Iterator[_PaintPath]:
    defaults = {
        "display": "inline",
        "visibility": "visible",
        "fill": "black",
        "fill-rule": "nonzero",
        "opacity": "1",
        "fill-opacity": "1",
        "stroke": "none",
    }

    def visit(
        element: Any,
        inherited_transform: Any,
        inherited_style: Mapping[str, str],
        group: tuple[int, str] | None,
    ) -> Iterator[_PaintPath]:
        tag = _local_name(element.tag)
        if tag not in {"svg", "g", "path"}:
            raise RuntimeError(f"Unsupported SVG element <{tag}>")
        style = _style(element, inherited_style)
        transform = inherited_transform @ parse_transform(
            element.attrib.get("transform"), np
        )
        for attribute in _UNSUPPORTED_VISIBLE:
            value = element.attrib.get(attribute) or style.get(attribute)
            if value and value.lower() != "none":
                raise RuntimeError(f"Unsupported visible SVG {attribute}: {value}")
        hidden = (
            style.get("display", "inline").lower() == "none"
            or style.get("visibility", "visible").lower() in {"hidden", "collapse"}
            or float(style.get("opacity", "1")) <= 0
            or float(style.get("fill-opacity", "1")) <= 0
        )
        gid = element.attrib.get("graphic_group_label_id")
        if gid is not None:
            try:
                group = (int(gid), element.attrib["graphic_group_label"])
            except (KeyError, ValueError) as exc:
                raise RuntimeError("Incomplete SVG graphic-group metadata") from exc
        if tag == "path" and not hidden and style.get("fill", "black").lower() != "none":
            if group is None:
                raise RuntimeError("Visible annotation path is outside a graphic group")
            try:
                structure_id = int(element.attrib["structure_id"])
            except (KeyError, ValueError) as exc:
                raise RuntimeError("Annotation path has no integer structure_id") from exc
            data = element.attrib.get("d", "").strip()
            if not data:
                raise RuntimeError("Annotation path has empty path data")
            rule = style.get("fill-rule", "nonzero").lower()
            if rule not in {"nonzero", "evenodd"}:
                raise RuntimeError(f"Unsupported SVG fill rule {rule!r}")
            raw_order = element.attrib.get("order")
            try:
                order = None if raw_order is None else int(raw_order)
            except ValueError as exc:
                raise RuntimeError(f"Non-integer SVG order {raw_order!r}") from exc
            yield _PaintPath(
                group[0],
                group[1],
                structure_id,
                data,
                transform,
                rule,
                order,
            )
        for child in element:
            yield from visit(child, transform, style, group)

    yield from visit(root, np.eye(3, dtype=np.float64), defaults, None)


def inspect_graphic_groups(svg_path: Path) -> list[tuple[int, str]]:
    """Return graphic groups in first-path document order without loading Skia."""
    np, _, safe_et, _ = _dependencies()
    try:
        root = safe_et.parse(svg_path).getroot()
    except Exception as exc:
        raise RuntimeError(f"Cannot safely parse SVG {svg_path}") from exc
    groups: list[tuple[int, str]] = []
    for path in _paths(root, np):
        value = (path.group_id, path.group_name)
        if value not in groups:
            groups.append(value)
    return groups


def _skia_matrix(matrix: Any, skia: Any) -> Any:
    return skia.Matrix.MakeAll(
        float(matrix[0, 0]),
        float(matrix[0, 1]),
        float(matrix[0, 2]),
        float(matrix[1, 0]),
        float(matrix[1, 1]),
        float(matrix[1, 2]),
        float(matrix[2, 0]),
        float(matrix[2, 1]),
        float(matrix[2, 2]),
    )


def _skia_path(path_data: str, skia: Any, parse_path: Any) -> Any:
    """Convert standards-parsed SVG segments into a Skia path."""
    from svg.path.path import (
        Arc,
        Close,
        CubicBezier,
        Line,
        Move,
        QuadraticBezier,
    )

    path = skia.Path()
    try:
        segments = parse_path(path_data)
    except (IndexError, TypeError, ValueError) as exc:
        raise RuntimeError("Established SVG parser rejected path data") from exc
    for segment in segments:
        if isinstance(segment, Move):
            path.moveTo(float(segment.end.real), float(segment.end.imag))
        elif isinstance(segment, Line):
            path.lineTo(float(segment.end.real), float(segment.end.imag))
        elif isinstance(segment, CubicBezier):
            path.cubicTo(
                float(segment.control1.real),
                float(segment.control1.imag),
                float(segment.control2.real),
                float(segment.control2.imag),
                float(segment.end.real),
                float(segment.end.imag),
            )
        elif isinstance(segment, QuadraticBezier):
            path.quadTo(
                float(segment.control.real),
                float(segment.control.imag),
                float(segment.end.real),
                float(segment.end.imag),
            )
        elif isinstance(segment, Close):
            path.close()
        elif isinstance(segment, Arc):
            raise RuntimeError(
                "SVG arc commands are not present in the validated Allen corpus "
                "and are not enabled by this renderer"
            )
        else:  # pragma: no cover - guards future svg.path segment additions
            raise RuntimeError(f"Unsupported parsed SVG segment {type(segment).__name__}")
    return path


def _path_mask(
    path_data: str,
    matrix: Any,
    fill_rule: str,
    shape: tuple[int, int],
    np: Any,
    skia: Any,
    parse_path: Any,
) -> Any:
    path = _skia_path(path_data, skia, parse_path)
    path.setFillType(
        skia.PathFillType.kEvenOdd
        if fill_rule == "evenodd"
        else skia.PathFillType.kWinding
    )
    bounds = path.getBounds()
    corners = np.asarray(
        [
            [bounds.left(), bounds.top(), 1],
            [bounds.right(), bounds.top(), 1],
            [bounds.left(), bounds.bottom(), 1],
            [bounds.right(), bounds.bottom(), 1],
        ],
        dtype=np.float64,
    )
    mapped = (matrix @ corners.T).T
    mapped = mapped[:, :2] / mapped[:, 2:3]
    height, width = shape
    x0 = max(0, int(math.floor(mapped[:, 0].min())) - 1)
    y0 = max(0, int(math.floor(mapped[:, 1].min())) - 1)
    x1 = min(width, int(math.ceil(mapped[:, 0].max())) + 1)
    y1 = min(height, int(math.ceil(mapped[:, 1].max())) + 1)
    if x0 >= x1 or y0 >= y1:
        return x0, y0, np.zeros((0, 0), dtype=bool)
    surface = skia.Surface.MakeRasterN32Premul(x1 - x0, y1 - y0)
    if surface is None:
        raise RuntimeError("Skia could not allocate raster surface")
    canvas = surface.getCanvas()
    canvas.clear(skia.ColorTRANSPARENT)
    canvas.translate(-x0, -y0)
    canvas.concat(_skia_matrix(matrix, skia))
    paint = skia.Paint(
        Color=skia.ColorWHITE,
        Style=skia.Paint.kFill_Style,
        AntiAlias=False,
    )
    canvas.drawPath(path, paint)
    rgba = surface.makeImageSnapshot().toarray(
        colorType=skia.ColorType.kRGBA_8888_ColorType
    )
    return x0, y0, rgba[..., 3] != 0


def rasterize_svg(
    svg_path: Path,
    target_width: int,
    target_height: int,
    expected_groups: Mapping[int, str],
) -> list[LayerResult]:
    """Rasterize fills in SVG document order onto the requested target grid."""
    if target_width <= 0 or target_height <= 0:
        raise ValueError("Target dimensions must be positive")
    np, skia, safe_et, parse_path = _dependencies()
    try:
        root = safe_et.parse(svg_path).getroot()
    except Exception as exc:
        raise RuntimeError(f"Cannot safely parse SVG {svg_path}") from exc
    if _local_name(root.tag) != "svg":
        raise RuntimeError("Missing SVG root")
    min_x, min_y, source_width, source_height = _svg_extent(root)
    target = _matrix(
        [
            [target_width / source_width, 0, -min_x * target_width / source_width],
            [0, target_height / source_height, -min_y * target_height / source_height],
            [0, 0, 1],
        ],
        np,
    )
    paths = list(_paths(root, np))
    observed: list[int] = []
    for item in paths:
        if item.group_id not in expected_groups:
            raise RuntimeError(f"Unexpected graphic group {item.group_id}")
        if expected_groups[item.group_id] != item.group_name:
            raise RuntimeError(
                f"Graphic-group name mismatch for {item.group_id}: {item.group_name!r}"
            )
        if item.group_id not in observed:
            observed.append(item.group_id)
    results: list[LayerResult] = []
    for group_id in observed:
        layer = np.zeros((target_height, target_width), dtype=np.uint32)
        conflict = np.zeros(layer.shape, dtype=bool)
        repeated = np.zeros(layer.shape, dtype=bool)
        group_paths = [item for item in paths if item.group_id == group_id]
        orders = [item.order for item in group_paths if item.order is not None]
        for item in group_paths:
            x0, y0, mask = _path_mask(
                item.data,
                target @ item.transform,
                item.fill_rule,
                layer.shape,
                np,
                skia,
                parse_path,
            )
            if not mask.any():
                continue
            view = layer[y0 : y0 + mask.shape[0], x0 : x0 + mask.shape[1]]
            prior = view[mask]
            conflict_view = conflict[
                y0 : y0 + mask.shape[0], x0 : x0 + mask.shape[1]
            ]
            repeat_view = repeated[
                y0 : y0 + mask.shape[0], x0 : x0 + mask.shape[1]
            ]
            conflict_pixels = mask.copy()
            conflict_pixels[mask] = (prior != 0) & (prior != item.structure_id)
            repeated_pixels = mask.copy()
            repeated_pixels[mask] = prior == item.structure_id
            conflict_view |= conflict_pixels
            repeat_view |= repeated_pixels
            view[mask] = item.structure_id
        results.append(
            LayerResult(
                group_id,
                expected_groups[group_id],
                layer,
                len(group_paths),
                tuple(int(value) for value in np.unique(layer) if value),
                int(conflict.sum()),
                int(repeated.sum()),
                min(orders) if orders else None,
                max(orders) if orders else None,
                all(a <= b for a, b in zip(orders, orders[1:])),
            )
        )
    return results
