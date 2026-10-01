#!/usr/bin/env python3
"""Render a color montage from an existing dense Allen annotation derivative.

This is intentionally a read-only consumer of densification outputs.  It does
not fit a pair or write into the derivative being visualized.
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
import numpy as np
import tifffile

from preprocess import visualize_allen_annotations as annotation_visual


DEFAULT_ANNOTATIONS_ZARR = annotation_visual.DEFAULT_ANNOTATIONS_ZARR


def parse_pair(text: str) -> tuple[int, int]:
    """Parse a canonical physical-index pair written as LEFT-RIGHT."""
    try:
        left, right = (int(value) for value in text.split("-", 1))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("pair must be LEFT-RIGHT") from exc
    if left < 0 or right <= left:
        raise argparse.ArgumentTypeError("pair must satisfy 0 <= LEFT < RIGHT")
    return left, right


def preview_indices(left: int, right: int, count: int) -> list[int]:
    """Choose deterministic, approximately even canonical preview planes."""
    if count < 2:
        raise ValueError("preview count must be at least 2")
    count = min(count, right - left + 1)
    return sorted(set(map(int, np.rint(np.linspace(left, right, count)))))


def read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Cannot read JSON metadata {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected a JSON object in {path}")
    return payload


def physical_rows(path: Path) -> dict[int, dict[str, str]]:
    try:
        with path.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream, delimiter="\t"))
    except OSError as exc:
        raise RuntimeError(f"Cannot read physical-section metadata {path}") from exc
    result = {int(row["physical_index"]): row for row in rows}
    if len(result) != len(rows):
        raise RuntimeError(f"Duplicate physical indices in {path}")
    return result


def load_group_colors(
    annotations_zarr: Path,
    endpoint_sections: tuple[int, int],
    group: int,
) -> tuple[str, list[tuple[int, tuple[int, ...]]], list[str]]:
    """Merge the endpoint Allen OME label colors for one graphic group."""
    title: str | None = None
    colors_by_id: dict[int, tuple[int, ...]] = {}
    sources: list[str] = []
    for section in endpoint_sections:
        package = annotations_zarr / f"section-{section:04d}.ome.zarr"
        source = package / "labels" / f"group-{group}" / "zarr.json"
        if not source.is_file():
            raise RuntimeError(f"Missing endpoint color metadata: {source}")
        root = annotation_visual._StoredMetadataGroup(package)
        label_group = root["labels"][f"group-{group}"]
        current_title = annotation_visual._label_title(
            label_group, f"group-{group}"
        )
        if title is not None and current_title != title:
            raise RuntimeError(
                f"Graphic-group title changed between endpoints: {title!r} and "
                f"{current_title!r}"
            )
        title = current_title
        for label_id, rgba in annotation_visual._label_colors(
            label_group, f"section {section}, group {group}"
        ):
            prior = colors_by_id.get(label_id)
            if prior is not None and prior != rgba:
                raise RuntimeError(
                    f"Allen color for label {label_id} differs between endpoints"
                )
            colors_by_id[label_id] = rgba
        sources.append(str(source.resolve()))
    if title is None or not colors_by_id:
        raise RuntimeError(f"No color metadata found for group {group}")
    return title, sorted(colors_by_id.items()), sources


def load_plane(path: Path, expected_shape: tuple[int, int] | None) -> np.ndarray:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"Missing or invalid dense categorical plane: {path}")
    plane = np.asarray(tifffile.imread(path))
    if plane.ndim != 2 or not np.issubdtype(plane.dtype, np.integer):
        raise RuntimeError(
            f"Dense plane must be a 2-D integer raster: {path} "
            f"(shape={plane.shape}, dtype={plane.dtype})"
        )
    if expected_shape is not None and plane.shape != expected_shape:
        raise RuntimeError(
            f"Dense plane geometry changed at {path}: "
            f"expected {expected_shape}, found {plane.shape}"
        )
    return plane.astype(np.uint32, copy=False)


def common_crop(
    planes_by_group: dict[int, list[np.ndarray]], margin_fraction: float = 0.025
) -> tuple[slice, slice]:
    foreground = np.zeros(next(iter(planes_by_group.values()))[0].shape, dtype=bool)
    for planes in planes_by_group.values():
        for plane in planes:
            foreground |= plane != 0
    ys, xs = np.nonzero(foreground)
    if not ys.size:
        raise RuntimeError("Selected dense annotation planes contain no foreground ROIs")
    height, width = foreground.shape
    margin = max(4, int(round(max(height, width) * margin_fraction)))
    y0, y1 = max(0, int(ys.min()) - margin), min(height, int(ys.max()) + margin + 1)
    x0, x1 = max(0, int(xs.min()) - margin), min(width, int(xs.max()) + margin + 1)
    return slice(y0, y1), slice(x0, x1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dense-output",
        type=Path,
        required=True,
        help="existing dense annotation derivative root",
    )
    parser.add_argument(
        "--pair", type=parse_pair, required=True, help="physical indices LEFT-RIGHT"
    )
    parser.add_argument(
        "--groups",
        type=int,
        nargs="+",
        default=[31, 265297118],
        help="graphic groups to render as montage rows",
    )
    parser.add_argument(
        "--annotations-zarr",
        type=Path,
        default=DEFAULT_ANNOTATIONS_ZARR,
        help="source OME-Zarr root containing Allen label colors",
    )
    parser.add_argument(
        "--preview-count",
        type=int,
        default=9,
        help="number of approximately even planes, including both endpoints",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dense_output = args.dense_output.expanduser().resolve()
    annotations_zarr = args.annotations_zarr.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.suffix.lower() != ".png":
        raise RuntimeError("--output must name a PNG file")
    audit_output = output.with_suffix(".json")
    for destination in (output, audit_output):
        if destination.exists() and not args.overwrite:
            raise RuntimeError(
                f"Refusing to overwrite {destination}; pass --overwrite to replace it"
            )

    left, right = args.pair
    pair_id = f"{left:04d}-{right:04d}"
    pair_manifest_path = dense_output / "metadata/pairs/tiff" / f"{pair_id}.json"
    pair_manifest = read_json(pair_manifest_path)
    if pair_manifest.get("pair") != [left, right]:
        raise RuntimeError(f"Pair manifest does not describe requested pair {pair_id}")
    if pair_manifest.get("status") != "complete":
        raise RuntimeError(f"Pair {pair_id} is not complete")
    if pair_manifest.get("driver") != "annotation":
        raise RuntimeError(f"Pair {pair_id} was not driven by annotations")
    if pair_manifest.get("output_format") != "tiff":
        raise RuntimeError(f"Pair {pair_id} does not use dense TIFF output")
    available_groups = set(map(int, pair_manifest.get("selected_graphic_groups", [])))
    requested_groups = list(dict.fromkeys(args.groups))
    missing_groups = sorted(set(requested_groups) - available_groups)
    if missing_groups:
        raise RuntimeError(
            f"Pair {pair_id} has no completed output for groups {missing_groups}"
        )

    rows = physical_rows(dense_output / "metadata/physical_sections.tsv")
    try:
        endpoint_sections = (
            int(rows[left]["allen_section_number"]),
            int(rows[right]["allen_section_number"]),
        )
    except KeyError as exc:
        raise RuntimeError(f"Missing endpoint physical metadata for pair {pair_id}") from exc

    indices = preview_indices(left, right, args.preview_count)
    planes_by_group: dict[int, list[np.ndarray]] = {}
    expected_shape: tuple[int, int] | None = None
    for group in requested_groups:
        planes: list[np.ndarray] = []
        for physical in indices:
            plane_path = (
                dense_output
                / "dense_tiff/groups"
                / str(group)
                / f"{physical:06d}.tif"
            )
            plane = load_plane(plane_path, expected_shape)
            expected_shape = plane.shape
            planes.append(plane)
        planes_by_group[group] = planes

    group_titles: dict[int, str] = {}
    group_colors: dict[int, list[tuple[int, tuple[int, ...]]]] = {}
    color_sources: dict[int, list[str]] = {}
    for group in requested_groups:
        title, colors, sources = load_group_colors(
            annotations_zarr, endpoint_sections, group
        )
        defined = {label_id for label_id, _ in colors}
        observed = {
            int(label_id)
            for plane in planes_by_group[group]
            for label_id in np.unique(plane)
            if int(label_id) != 0
        }
        missing_colors = sorted(observed - defined)
        if missing_colors:
            raise RuntimeError(
                f"Group {group} has dense labels without Allen color metadata: "
                f"{missing_colors}"
            )
        group_titles[group] = title
        group_colors[group] = colors
        color_sources[group] = sources

    y_crop, x_crop = common_crop(planes_by_group)
    column_count = len(indices)
    row_count = len(requested_groups)
    figure_width = max(12.0, 2.55 * column_count)
    figure_height = 1.15 + 3.0 * row_count
    fig, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(figure_width, figure_height),
        squeeze=False,
        facecolor="white",
    )
    fig.subplots_adjust(
        left=0.075,
        right=0.995,
        bottom=0.075,
        top=0.79,
        wspace=0.035,
        hspace=0.09,
    )

    endpoint_z_um = [float(value) for value in pair_manifest["endpoint_z_um"]]
    preview_records: list[dict[str, Any]] = []
    for column, physical in enumerate(indices):
        try:
            z_um = float(rows[physical]["serial_z_center_mm"]) * 1000.0
        except KeyError as exc:
            raise RuntimeError(f"Missing physical metadata for index {physical}") from exc
        t = (z_um - endpoint_z_um[0]) / (endpoint_z_um[1] - endpoint_z_um[0])
        state = "observed endpoint" if physical in (left, right) else "inferred"
        preview_records.append(
            {
                "physical_index": physical,
                "allen_section_number": int(rows[physical]["allen_section_number"]),
                "z_um": z_um,
                "t": t,
                "state": state,
            }
        )
        axes[0, column].set_title(
            f"{state}\nidx {physical}  |  t={t:.2f}\nz={z_um / 1000.0:.3f} mm",
            fontsize=8.6,
            linespacing=1.18,
            pad=7,
        )

    for row_index, group in enumerate(requested_groups):
        colors = group_colors[group]
        for column, plane in enumerate(planes_by_group[group]):
            rgb = np.asarray(
                annotation_visual._combined_display_thumbnail(
                    plane, colors, (plane.shape[1], plane.shape[0])
                )
            )
            ax = axes[row_index, column]
            ax.imshow(rgb[y_crop, x_crop], interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        axes[row_index, 0].text(
            -0.12,
            0.5,
            f"Group {group}\n{group_titles[group].removeprefix('Atlas - ')}",
            transform=axes[row_index, 0].transAxes,
            ha="right",
            va="center",
            rotation=90,
            fontsize=9.2,
            linespacing=1.25,
        )

    fig.suptitle(
        f"Corrected annotation-driven densification · physical pair {left}–{right}",
        fontsize=16,
        fontweight="semibold",
        y=0.965,
    )
    fig.text(
        0.5,
        0.905,
        f"Allen endpoint sections {endpoint_sections[0]} → {endpoint_sections[1]}  ·  "
        f"{pair_manifest['canonical_output_planes_between_endpoints']} inferred "
        f"canonical planes  ·  graphic groups "
        + ", ".join(map(str, requested_groups)),
        ha="center",
        va="center",
        fontsize=10.5,
        color="#333333",
    )
    fig.text(
        0.5,
        0.025,
        "Categorical ROI fills use Allen OME image-label colors; black lines mark "
        "boundaries between adjacent nonzero labels.",
        ha="center",
        va="bottom",
        fontsize=8.8,
        color="#444444",
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=args.dpi, facecolor="white", metadata={"Software": "matplotlib"})
    plt.close(fig)

    audit = {
        "schema": "allen-densification-montage-v1",
        "montage": str(output),
        "dense_output": str(dense_output),
        "pair_manifest": str(pair_manifest_path.resolve()),
        "pair": [left, right],
        "pair_status": pair_manifest["status"],
        "driver": pair_manifest["driver"],
        "endpoint_sections": list(endpoint_sections),
        "groups": requested_groups,
        "preview_planes": preview_records,
        "color_mapping": "Allen OME image-label RGBA metadata merged across endpoints",
        "color_metadata_sources": {str(k): v for k, v in color_sources.items()},
        "colors_by_group": {
            str(group): {
                str(label_id): list(rgba)
                for label_id, rgba in group_colors[group]
            }
            for group in requested_groups
        },
    }
    audit_output.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"montage": str(output), "audit": str(audit_output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
