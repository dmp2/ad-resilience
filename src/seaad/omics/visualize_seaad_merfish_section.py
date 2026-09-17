#!/usr/bin/env python3
"""Visualize one SEA-AD MTG MERFISH spatial section from the merged H5AD.

Design goals
------------
* Read-only: never modifies the source H5AD.
* Section-first and memory-conscious: never materializes the full cell x gene matrix.
* Provenance-aware: section identity and coordinate representation are explicit CLI choices.
* Useful first output: a headless PNG suitable for QC.
* Optional interactive/3D bridges: napari viewing and VTK/VTP export when installed.

The current SEA-AD merged MTG H5AD (2024-12-11) is known to contain:
  obs: Section, Specimen Barcode, Class, Subclass, Supertype, Layer annotation,
       Depth from pia, Normalized depth from pia, Merscope, Cell ID, ...
  obsm: X_spatial_raw, X_spatial_tiled, spatial, X_umap, umap

For cross-resource lineage, ``Specimen Barcode`` is the conservative default section
selector because it is the established section-level anchor in this project. ``Section``
can be selected explicitly with --section-field.

Examples
--------
List section identifiers without plotting::

    python -m preprocess.visualize_seaad_merfish_section \
      --h5ad SEAAD_MTG_MERFISH.2024-12-11.h5ad --list-sections

Plot one section by subclass::

    python -m preprocess.visualize_seaad_merfish_section \
      --h5ad SEAAD_MTG_MERFISH.2024-12-11.h5ad \
      --section <SPECIMEN_BARCODE> \
      --color Subclass \
      --output results/qc/seaad_merfish_<SPECIMEN_BARCODE>_subclass.png

Use a different coordinate representation::

    ... --coordinates X_spatial_raw

Plot a gene (touches only the requested gene values, never the full X matrix)::

    ... --gene GFAP

Open the extracted points in napari (optional dependency)::

    ... --napari

Export the extracted points to ParaView-compatible VTP (optional pyvista)::

    ... --vtk-output results/qc/section.vtp
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

import h5py
import numpy as np


DEFAULT_SECTION_FIELD = "Specimen Barcode"
DEFAULT_COORDINATES = "X_spatial_raw"
DEFAULT_COLOR = "Subclass"


def _decode_scalar(x):
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="replace")
    if isinstance(x, np.bytes_):
        return bytes(x).decode("utf-8", errors="replace")
    return x.item() if isinstance(x, np.generic) else x


def _decode_array(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    if a.dtype.kind in {"S", "O", "U"}:
        return np.asarray([str(_decode_scalar(x)) for x in a], dtype=object)
    return a


def _group_index_name(group: h5py.Group) -> str:
    raw = group.attrs.get("_index", "_index")
    raw = _decode_scalar(raw)
    return str(raw)


def _read_categorical(group: h5py.Group, rows: np.ndarray | None = None) -> np.ndarray:
    codes_ds = group["codes"]
    categories = _decode_array(group["categories"][...])
    if rows is None:
        codes = codes_ds[...]
    else:
        # Fancy indexing can become pathologically slow in h5py for large row sets.
        # The caller normally uses chunked extraction; this path is mainly for small sets.
        codes = codes_ds[rows]
    out = np.empty(len(codes), dtype=object)
    missing = codes < 0
    safe = codes.copy()
    safe[missing] = 0
    out[:] = categories[safe]
    out[missing] = None
    return out


def _read_obs_column_all(obs: h5py.Group, name: str) -> np.ndarray:
    if name not in obs:
        raise KeyError(f"obs column not found: {name!r}")
    node = obs[name]
    if isinstance(node, h5py.Group):
        enc = str(_decode_scalar(node.attrs.get("encoding-type", "")))
        if enc == "categorical" or {"codes", "categories"}.issubset(node.keys()):
            return _read_categorical(node)
        raise NotImplementedError(
            f"Unsupported grouped obs encoding for {name!r}: {enc or list(node.keys())}"
        )
    return _decode_array(node[...])


def _obs_column_length(obs: h5py.Group, name: str) -> int:
    node = obs[name]
    if isinstance(node, h5py.Group) and "codes" in node:
        return int(node["codes"].shape[0])
    return int(node.shape[0])


def _read_obs_chunk(obs: h5py.Group, name: str, start: int, stop: int) -> np.ndarray:
    node = obs[name]
    if isinstance(node, h5py.Group):
        if "codes" not in node or "categories" not in node:
            raise NotImplementedError(f"Unsupported grouped obs column: {name}")
        codes = node["codes"][start:stop]
        cats = _decode_array(node["categories"][...])
        out = np.empty(len(codes), dtype=object)
        miss = codes < 0
        safe = codes.copy()
        safe[miss] = 0
        out[:] = cats[safe]
        out[miss] = None
        return out
    return _decode_array(node[start:stop])


def _section_counts(obs: h5py.Group, section_field: str, chunk_rows: int) -> Counter:
    n = _obs_column_length(obs, section_field)
    counts: Counter = Counter()
    for start in range(0, n, chunk_rows):
        stop = min(n, start + chunk_rows)
        vals = _read_obs_chunk(obs, section_field, start, stop)
        counts.update(str(v) for v in vals if v is not None and str(v) not in {"", "nan"})
    return counts


def _selection_mask_chunk(
    obs: h5py.Group,
    section_field: str,
    section: str,
    start: int,
    stop: int,
    donor: str | None = None,
) -> np.ndarray:
    vals = _read_obs_chunk(obs, section_field, start, stop)
    keep = np.asarray([str(v) == section for v in vals], dtype=bool)
    if donor is not None:
        donor_vals = _read_obs_chunk(obs, "Donor ID", start, stop)
        keep &= np.asarray([str(v) == donor for v in donor_vals], dtype=bool)
    return keep


def _selected_indices(
    obs: h5py.Group,
    section_field: str,
    section: str,
    chunk_rows: int,
    donor: str | None = None,
) -> np.ndarray:
    n = _obs_column_length(obs, section_field)
    pieces = []
    for start in range(0, n, chunk_rows):
        stop = min(n, start + chunk_rows)
        keep = _selection_mask_chunk(obs, section_field, section, start, stop, donor)
        if np.any(keep):
            pieces.append(np.flatnonzero(keep).astype(np.int64) + start)
    if not pieces:
        return np.empty(0, dtype=np.int64)
    return np.concatenate(pieces)


def _extract_obsm_rows(
    obsm: h5py.Group,
    key: str,
    selected: np.ndarray,
    n_obs: int,
    chunk_rows: int,
) -> np.ndarray:
    if key not in obsm:
        raise KeyError(f"obsm key not found: {key!r}. Available: {list(obsm.keys())}")
    node = obsm[key]
    if not isinstance(node, h5py.Dataset):
        raise NotImplementedError(f"obsm/{key} is not a plain dataset")
    if node.ndim != 2 or node.shape[0] != n_obs or node.shape[1] < 2:
        raise ValueError(f"obsm/{key} has unexpected shape {node.shape}; expected (n_obs, >=2)")

    out = np.empty((len(selected), node.shape[1]), dtype=node.dtype)
    if not len(selected):
        return out
    pos = 0
    sel_ptr = 0
    for start in range(0, n_obs, chunk_rows):
        stop = min(n_obs, start + chunk_rows)
        left = np.searchsorted(selected, start, side="left", sorter=None)
        right = np.searchsorted(selected, stop, side="left", sorter=None)
        if right <= left:
            continue
        local = selected[left:right] - start
        block = node[start:stop]
        vals = block[local]
        out[pos : pos + len(vals)] = vals
        pos += len(vals)
        sel_ptr = right
    if pos != len(selected):
        raise RuntimeError(f"Coordinate extraction mismatch: extracted {pos} / {len(selected)} rows")
    return out


def _extract_obs_rows(
    obs: h5py.Group,
    field: str,
    selected: np.ndarray,
    n_obs: int,
    chunk_rows: int,
) -> np.ndarray:
    if field not in obs:
        raise KeyError(f"obs field not found: {field!r}")
    parts = []
    for start in range(0, n_obs, chunk_rows):
        stop = min(n_obs, start + chunk_rows)
        left = np.searchsorted(selected, start)
        right = np.searchsorted(selected, stop)
        if right <= left:
            continue
        block = _read_obs_chunk(obs, field, start, stop)
        parts.append(block[selected[left:right] - start])
    if not parts:
        return np.empty(0, dtype=object)
    return np.concatenate(parts)


def _var_names(f: h5py.File) -> np.ndarray:
    var = f["var"]
    idx_name = _group_index_name(var)
    if idx_name not in var:
        # Most H5ADs use _index; handle either representation conservatively.
        if "_index" in var:
            idx_name = "_index"
        else:
            raise KeyError(f"Could not locate var index. var keys: {list(var.keys())}")
    node = var[idx_name]
    if isinstance(node, h5py.Group):
        return _read_categorical(node)
    return _decode_array(node[...])


def _extract_gene_values(
    f: h5py.File,
    gene_index: int,
    selected: np.ndarray,
    n_obs: int,
    chunk_rows: int,
) -> np.ndarray:
    """Extract one gene for selected rows from dense/CSR/CSC H5AD X.

    Dense arrays are read chunkwise by rows. CSR is reconstructed only for row chunks
    that contain selected cells. CSC is ideal for this operation and reads the requested
    column directly before selecting rows.
    """
    X = f["X"]
    if isinstance(X, h5py.Dataset):
        parts = []
        for start in range(0, n_obs, chunk_rows):
            stop = min(n_obs, start + chunk_rows)
            left = np.searchsorted(selected, start)
            right = np.searchsorted(selected, stop)
            if right <= left:
                continue
            block = X[start:stop, gene_index]
            parts.append(np.asarray(block)[selected[left:right] - start])
        return np.concatenate(parts) if parts else np.empty(0, dtype=float)

    enc = str(_decode_scalar(X.attrs.get("encoding-type", ""))).lower()
    shape = tuple(int(x) for x in X.attrs.get("shape", (n_obs, len(_var_names(f)))))

    if "csc" in enc:
        indptr = X["indptr"][gene_index : gene_index + 2]
        lo, hi = int(indptr[0]), int(indptr[1])
        rows = X["indices"][lo:hi]
        data = X["data"][lo:hi]
        # selected is sorted; intersect without creating an n_obs-sized dense vector.
        loc = np.searchsorted(selected, rows)
        valid = (loc < len(selected)) & (selected[np.minimum(loc, len(selected)-1)] == rows) if len(selected) else np.zeros(len(rows), bool)
        out = np.zeros(len(selected), dtype=data.dtype)
        if np.any(valid):
            out[loc[valid]] = data[valid]
        return out

    if "csr" in enc or {"data", "indices", "indptr"}.issubset(X.keys()):
        try:
            from scipy.sparse import csr_matrix
        except ImportError as e:
            raise RuntimeError("Gene extraction from CSR H5AD requires scipy") from e
        parts = []
        for start in range(0, n_obs, chunk_rows):
            stop = min(n_obs, start + chunk_rows)
            left = np.searchsorted(selected, start)
            right = np.searchsorted(selected, stop)
            if right <= left:
                continue
            ip = X["indptr"][start : stop + 1]
            lo, hi = int(ip[0]), int(ip[-1])
            data = X["data"][lo:hi]
            indices = X["indices"][lo:hi]
            local_ip = np.asarray(ip, dtype=np.int64) - lo
            m = csr_matrix((data, indices, local_ip), shape=(stop - start, shape[1]))
            local_rows = selected[left:right] - start
            vals = m[local_rows, gene_index].toarray().ravel()
            parts.append(vals)
        return np.concatenate(parts) if parts else np.empty(0, dtype=float)

    raise NotImplementedError(f"Unsupported X encoding: {enc or list(X.keys())}")


def _finite_xy(coords: np.ndarray, values: np.ndarray | None = None):
    xy = np.asarray(coords[:, :2], dtype=float)
    keep = np.isfinite(xy).all(axis=1)
    if values is not None and np.issubdtype(np.asarray(values).dtype, np.number):
        keep &= np.isfinite(np.asarray(values, dtype=float))
    return xy[keep], (None if values is None else np.asarray(values)[keep]), keep


def _categorical_palette(values: np.ndarray, f: h5py.File, field: str):
    values = np.asarray(values, dtype=object)
    categories = sorted({str(x) for x in values if x is not None})
    # Prefer Allen/AnnData stored colors when a matching palette exists.
    color_key_candidates = [f"{field}_colors", f"{field.lower()}_colors"]
    stored = None
    if "uns" in f:
        for key in color_key_candidates:
            if key in f["uns"] and isinstance(f["uns"][key], h5py.Dataset):
                stored = [str(x) for x in _decode_array(f["uns"][key][...])]
                break
    # We cannot safely infer category order from an unrelated taxonomy table here.
    # Use matplotlib's categorical mapping unless palette length exactly matches the
    # categories present in the plotted data.
    if stored is not None and len(stored) == len(categories):
        mapping = dict(zip(categories, stored))
        return categories, mapping
    return categories, None


def _plot(
    coords: np.ndarray,
    values: np.ndarray,
    output: Path,
    title: str,
    label: str,
    f: h5py.File,
    point_size: float,
    alpha: float,
    rasterized: bool,
    dpi: int,
    invert_y: bool,
):
    import matplotlib.pyplot as plt

    xy, vals, _ = _finite_xy(coords, values)
    if len(xy) == 0:
        raise ValueError("No finite coordinates remain to plot")

    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    arr = np.asarray(vals)
    numeric = np.issubdtype(arr.dtype, np.number)

    if numeric:
        sc = ax.scatter(
            xy[:, 0], xy[:, 1], c=arr.astype(float), s=point_size,
            alpha=alpha, linewidths=0, rasterized=rasterized,
        )
        cb = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.02)
        cb.set_label(label)
    else:
        cats, stored_map = _categorical_palette(arr, f, label)
        import matplotlib.pyplot as _plt
        cmap = _plt.get_cmap("tab20", max(1, len(cats)))
        cat_to_int = {c: i for i, c in enumerate(cats)}
        idx = np.asarray([cat_to_int[str(v)] for v in arr], dtype=int)
        if stored_map:
            colors = np.asarray([stored_map[str(v)] for v in arr], dtype=object)
            ax.scatter(xy[:, 0], xy[:, 1], c=colors, s=point_size, alpha=alpha,
                       linewidths=0, rasterized=rasterized)
        else:
            ax.scatter(xy[:, 0], xy[:, 1], c=idx, cmap=cmap, s=point_size,
                       alpha=alpha, linewidths=0, rasterized=rasterized,
                       vmin=-0.5, vmax=max(0.5, len(cats)-0.5))
        # Legend is useful up to a reasonable number of categories; beyond that it
        # dominates the tissue geometry, so emit a sidecar category table instead.
        if len(cats) <= 30:
            from matplotlib.lines import Line2D
            handles = []
            for i, c in enumerate(cats):
                color = stored_map[c] if stored_map else cmap(i)
                handles.append(Line2D([0], [0], marker="o", linestyle="", markersize=5,
                                      markerfacecolor=color, markeredgewidth=0, label=c))
            ax.legend(handles=handles, title=label, loc="center left",
                      bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=7,
                      title_fontsize=8)
        else:
            sidecar = output.with_suffix(output.suffix + ".categories.tsv")
            sidecar.write_text("category\tindex\n" + "\n".join(
                f"{c}\t{i}" for i, c in enumerate(cats)
            ) + "\n", encoding="utf-8")
            print(f"Wrote category sidecar: {sidecar}")

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    if invert_y:
        ax.invert_yaxis()
    ax.grid(False)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _napari_view(coords: np.ndarray, values: np.ndarray, label: str, title: str):
    try:
        import napari
    except ImportError as e:
        raise RuntimeError("--napari requested but napari is not installed") from e

    xy, vals, _ = _finite_xy(coords, values)
    # napari points coordinates are (row=y, col=x) for image-style viewing.
    pts = np.column_stack([xy[:, 1], xy[:, 0]])
    viewer = napari.Viewer(title=title)
    arr = np.asarray(vals)
    if np.issubdtype(arr.dtype, np.number):
        viewer.add_points(pts, features={label: arr.astype(float)}, face_color=label,
                          size=2, name="cells")
    else:
        viewer.add_points(pts, features={label: arr.astype(str)}, face_color=label,
                          size=2, name="cells")
    napari.run()


def _vtk_export(coords: np.ndarray, values: np.ndarray, label: str, output: Path):
    try:
        import pyvista as pv
    except ImportError as e:
        raise RuntimeError("--vtk-output requested but pyvista is not installed") from e
    xyz = np.column_stack([
        coords[:, 0], coords[:, 1],
        coords[:, 2] if coords.shape[1] >= 3 else np.zeros(len(coords), dtype=coords.dtype),
    ]).astype(np.float32, copy=False)
    cloud = pv.PolyData(xyz)
    arr = np.asarray(values)
    if np.issubdtype(arr.dtype, np.number):
        cloud[label] = arr
    else:
        cats = sorted({str(x) for x in arr})
        mapping = {c: i for i, c in enumerate(cats)}
        cloud[label + "_id"] = np.asarray([mapping[str(x)] for x in arr], dtype=np.int32)
        output.with_suffix(output.suffix + ".categories.tsv").write_text(
            "category\tid\n" + "\n".join(f"{c}\t{i}" for c, i in mapping.items()) + "\n",
            encoding="utf-8",
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    cloud.save(output)


def _write_metadata(
    path: Path,
    *,
    source: Path,
    section_field: str,
    section: str,
    donor: str | None,
    coordinate_key: str,
    n_cells: int,
    color_field: str | None,
    gene: str | None,
):
    payload = {
        "source_h5ad": str(source.resolve()),
        "section_field": section_field,
        "section_value": section,
        "donor_filter": donor,
        "coordinate_key": coordinate_key,
        "coordinate_space_semantics": "not inferred by this script",
        "coordinate_units": "not inferred by this script",
        "n_cells": int(n_cells),
        "color_obs_field": color_field,
        "gene": gene,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--h5ad", required=True, type=Path, help="SEA-AD merged MERFISH H5AD")
    p.add_argument("--section", help="Exact section identifier value")
    p.add_argument("--section-field", default=DEFAULT_SECTION_FIELD,
                   help=f"obs field defining a section (default: {DEFAULT_SECTION_FIELD!r})")
    p.add_argument("--donor", help="Optional exact Donor ID filter; useful as a consistency guard")
    p.add_argument("--coordinates", default=DEFAULT_COORDINATES,
                   help=f"obsm coordinate key (default: {DEFAULT_COORDINATES!r})")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--color", default=DEFAULT_COLOR,
                       help=f"obs field used to color cells (default: {DEFAULT_COLOR!r})")
    group.add_argument("--gene", help="Gene symbol/name to color by expression")
    p.add_argument("--list-sections", action="store_true",
                   help="List values/counts for --section-field and exit")
    p.add_argument("--list-coordinates", action="store_true",
                   help="List available obsm arrays and exit")
    p.add_argument("--output", type=Path, help="PNG output. Required unless --napari only")
    p.add_argument("--metadata-output", type=Path,
                   help="Optional JSON provenance sidecar (default: <output>.json)")
    p.add_argument("--vtk-output", type=Path,
                   help="Optional VTP/VTK point-cloud export for PyVista/ParaView")
    p.add_argument("--napari", action="store_true", help="Open extracted points interactively in napari")
    p.add_argument("--chunk-rows", type=int, default=250_000,
                   help="Sequential HDF5 read chunk size (default: 250000)")
    p.add_argument("--point-size", type=float, default=1.0, help="Matplotlib point size")
    p.add_argument("--alpha", type=float, default=0.8, help="Point alpha")
    p.add_argument("--dpi", type=int, default=200, help="PNG DPI")
    p.add_argument("--no-rasterize", action="store_true",
                   help="Do not rasterize scatter artists (rasterization is safer for large point sets)")
    p.add_argument("--invert-y", action="store_true",
                   help="Invert plotted y-axis; never done implicitly")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.h5ad.is_file():
        raise FileNotFoundError(args.h5ad)
    if args.chunk_rows <= 0:
        raise ValueError("--chunk-rows must be positive")

    with h5py.File(args.h5ad, "r") as f:
        for required in ("obs", "var", "obsm"):
            if required not in f:
                raise ValueError(f"Not an AnnData-like H5AD: missing /{required}")
        obs = f["obs"]
        obsm = f["obsm"]
        if args.section_field not in obs:
            raise KeyError(
                f"Section field {args.section_field!r} not present. "
                f"Available obs fields include: {list(obs.keys())}"
            )
        n_obs = _obs_column_length(obs, args.section_field)

        if args.list_coordinates:
            print("Available obsm arrays:")
            for key in obsm.keys():
                node = obsm[key]
                shape = getattr(node, "shape", None)
                dtype = getattr(node, "dtype", None)
                print(f"  {key}: shape={shape}, dtype={dtype}")
            if not args.list_sections:
                return 0

        if args.list_sections:
            counts = _section_counts(obs, args.section_field, args.chunk_rows)
            print(f"Sections by obs[{args.section_field!r}] ({len(counts)} unique):")
            for value, count in counts.most_common():
                print(f"  {count:>9,d}  {value}")
            return 0

        if not args.section:
            raise ValueError("--section is required unless --list-sections/--list-coordinates is used")
        if not args.output and not args.napari and not args.vtk_output:
            raise ValueError("Specify at least one of --output, --napari, or --vtk-output")

        selected = _selected_indices(
            obs, args.section_field, args.section, args.chunk_rows, donor=args.donor
        )
        if len(selected) == 0:
            raise ValueError(
                f"No cells matched {args.section_field}={args.section!r}" +
                (f" and Donor ID={args.donor!r}" if args.donor else "")
            )
        print(f"Selected {len(selected):,} / {n_obs:,} cells")

        coords = _extract_obsm_rows(obsm, args.coordinates, selected, n_obs, args.chunk_rows)
        if coords.shape[1] < 2:
            raise ValueError(f"Coordinate array {args.coordinates!r} has <2 dimensions")
        finite = np.isfinite(np.asarray(coords[:, :2], dtype=float)).all(axis=1)
        print(
            f"Coordinates {args.coordinates}: shape={coords.shape}, "
            f"finite_xy={int(finite.sum()):,}/{len(coords):,}, "
            f"x=[{np.nanmin(coords[:,0]):.6g}, {np.nanmax(coords[:,0]):.6g}], "
            f"y=[{np.nanmin(coords[:,1]):.6g}, {np.nanmax(coords[:,1]):.6g}]"
        )

        if args.gene:
            genes = _var_names(f)
            matches = np.flatnonzero(np.asarray([str(g) == args.gene for g in genes]))
            if len(matches) == 0:
                # Case-insensitive fallback only when unique.
                low = args.gene.lower()
                ci = np.flatnonzero(np.asarray([str(g).lower() == low for g in genes]))
                if len(ci) != 1:
                    raise KeyError(f"Gene {args.gene!r} not found uniquely in var_names")
                matches = ci
            if len(matches) != 1:
                raise ValueError(f"Gene {args.gene!r} matched {len(matches)} var entries")
            values = _extract_gene_values(f, int(matches[0]), selected, n_obs, args.chunk_rows)
            label = args.gene
            source_kind = "gene"
            print(f"Extracted gene {args.gene!r}; finite={np.isfinite(values).sum():,}/{len(values):,}")
        else:
            values = _extract_obs_rows(obs, args.color, selected, n_obs, args.chunk_rows)
            label = args.color
            source_kind = "obs"
            unique = len({str(v) for v in values if v is not None})
            print(f"Extracted obs[{args.color!r}]; unique={unique}")

        title = (
            f"SEA-AD MERFISH | {args.section_field}: {args.section} | "
            f"{args.coordinates} | {label} | n={len(selected):,}"
        )

        if args.output:
            _plot(
                coords, values, args.output, title, label, f,
                point_size=args.point_size, alpha=args.alpha,
                rasterized=not args.no_rasterize, dpi=args.dpi,
                invert_y=args.invert_y,
            )
            print(f"Wrote PNG: {args.output}")

        if args.vtk_output:
            _vtk_export(coords, values, label, args.vtk_output)
            print(f"Wrote VTK point cloud: {args.vtk_output}")

        metadata_out = args.metadata_output
        if metadata_out is None and args.output:
            metadata_out = args.output.with_suffix(args.output.suffix + ".json")
        if metadata_out:
            metadata_out.parent.mkdir(parents=True, exist_ok=True)
            _write_metadata(
                metadata_out,
                source=args.h5ad,
                section_field=args.section_field,
                section=args.section,
                donor=args.donor,
                coordinate_key=args.coordinates,
                n_cells=len(selected),
                color_field=None if args.gene else args.color,
                gene=args.gene,
            )
            print(f"Wrote metadata: {metadata_out}")

    if args.napari:
        # Launch only after the H5AD is closed: napari receives the small extracted arrays.
        _napari_view(coords, values, label, title)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
