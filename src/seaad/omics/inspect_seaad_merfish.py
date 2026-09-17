#!/usr/bin/env python3
"""Memory-conservative structural inspection of a SEA-AD MERFISH .h5ad file.

This module intentionally uses h5py for the *inspection* pass so that it does not
materialize the cell-by-gene expression matrix.  It inventories the H5AD layout,
reports matrix encoding and dimensions, lists observation/variable metadata,
inspects spatial-array candidates, and summarizes section-like observation fields
in chunks.

It is read-only: the input file is never modified.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np


_SECTION_HINT = re.compile(r"(section|specimen|barcode|sample)", re.IGNORECASE)
_COORD_HINT = re.compile(
    r"(^|[_\-\s])(x|y|z)($|[_\-\s])|coord|spatial|position|centroid|center|centre",
    re.IGNORECASE,
)


def _decode_attr(value: Any) -> Any:
    """Convert HDF5 attribute values into JSON-/print-friendly Python objects."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_decode_attr(v) for v in value.tolist()]
    return value


def _attrs(obj: h5py.Group | h5py.Dataset) -> dict[str, Any]:
    return {str(k): _decode_attr(v) for k, v in obj.attrs.items()}


def _format_bytes(n: int | None) -> str:
    if n is None:
        return "unknown"
    value = float(n)
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    for unit in units:
        if abs(value) < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{n} B"


def _shape_from_attrs(obj: h5py.Group | h5py.Dataset) -> tuple[int, ...] | None:
    raw = obj.attrs.get("shape")
    if raw is None:
        return None
    arr = np.asarray(raw).astype(np.int64, copy=False).ravel()
    return tuple(int(v) for v in arr)


def _dataset_storage_bytes(ds: h5py.Dataset) -> int | None:
    try:
        return int(ds.id.get_storage_size())
    except Exception:
        return None


def _matrix_info(obj: h5py.Group | h5py.Dataset | None) -> dict[str, Any] | None:
    """Summarize a dense or AnnData sparse matrix without reading its values."""
    if obj is None:
        return None

    attrs = _attrs(obj)
    encoding = str(attrs.get("encoding-type", "unknown"))

    if isinstance(obj, h5py.Dataset):
        shape = tuple(int(v) for v in obj.shape)
        n = int(math.prod(shape)) if shape else 0
        return {
            "kind": "dataset",
            "encoding_type": encoding,
            "shape": list(shape),
            "dtype": str(obj.dtype),
            "chunks": list(obj.chunks) if obj.chunks else None,
            "compression": obj.compression,
            "storage_bytes": _dataset_storage_bytes(obj),
            "n_elements": n,
        }

    shape = _shape_from_attrs(obj)
    keys = sorted(obj.keys())
    info: dict[str, Any] = {
        "kind": "group",
        "encoding_type": encoding,
        "shape": list(shape) if shape else None,
        "members": keys,
    }

    if encoding in {"csr_matrix", "csc_matrix"} or {
        "data",
        "indices",
        "indptr",
    }.issubset(keys):
        data = obj.get("data")
        indices = obj.get("indices")
        indptr = obj.get("indptr")
        if isinstance(data, h5py.Dataset):
            nnz = int(data.shape[0])
            info.update(
                {
                    "nnz": nnz,
                    "data_dtype": str(data.dtype),
                    "data_storage_bytes": _dataset_storage_bytes(data),
                }
            )
            if shape and len(shape) == 2 and shape[0] and shape[1]:
                info["density"] = nnz / float(shape[0] * shape[1])
        if isinstance(indices, h5py.Dataset):
            info["indices_dtype"] = str(indices.dtype)
            info["indices_storage_bytes"] = _dataset_storage_bytes(indices)
        if isinstance(indptr, h5py.Dataset):
            info["indptr_dtype"] = str(indptr.dtype)
            info["indptr_storage_bytes"] = _dataset_storage_bytes(indptr)
    return info


def _table_columns(group: h5py.Group | None) -> list[str]:
    if group is None:
        return []
    return [str(k) for k in group.keys() if k != "_index"]


def _object_info(obj: h5py.Group | h5py.Dataset) -> dict[str, Any]:
    attrs = _attrs(obj)
    if isinstance(obj, h5py.Dataset):
        return {
            "kind": "dataset",
            "shape": list(obj.shape),
            "dtype": str(obj.dtype),
            "chunks": list(obj.chunks) if obj.chunks else None,
            "compression": obj.compression,
            "storage_bytes": _dataset_storage_bytes(obj),
            "attrs": attrs,
        }

    result: dict[str, Any] = {
        "kind": "group",
        "attrs": attrs,
        "members": sorted(obj.keys()),
    }
    shape = _shape_from_attrs(obj)
    if shape is not None:
        result["shape"] = list(shape)
    return result


def _decode_strings(values: np.ndarray) -> list[str]:
    flat = np.asarray(values).ravel()
    out: list[str] = []
    for value in flat:
        if isinstance(value, bytes):
            out.append(value.decode("utf-8", errors="replace"))
        elif isinstance(value, np.bytes_):
            out.append(bytes(value).decode("utf-8", errors="replace"))
        else:
            out.append(str(value))
    return out


def _counter_from_dataset(
    ds: h5py.Dataset,
    *,
    chunk_rows: int,
) -> Counter[str]:
    """Count a 1-D dataset in chunks without retaining the whole column."""
    if ds.ndim != 1:
        raise ValueError(f"Expected a 1-D dataset, got shape {ds.shape}")

    counts: Counter[str] = Counter()
    n = int(ds.shape[0])
    for start in range(0, n, chunk_rows):
        block = ds[start : min(start + chunk_rows, n)]
        values, block_counts = np.unique(block, return_counts=True)
        for value, count in zip(values, block_counts, strict=True):
            if isinstance(value, (bytes, np.bytes_)):
                key = bytes(value).decode("utf-8", errors="replace")
            else:
                key = str(value)
            counts[key] += int(count)
    return counts


def _counter_from_categorical(
    group: h5py.Group,
    *,
    chunk_rows: int,
) -> Counter[str]:
    """Count an AnnData categorical encoding by streaming integer codes."""
    codes = group.get("codes")
    categories = group.get("categories")
    if not isinstance(codes, h5py.Dataset) or not isinstance(categories, h5py.Dataset):
        raise ValueError("Categorical group does not contain dataset codes/categories")

    labels = _decode_strings(categories[...])
    counts: Counter[str] = Counter()
    n = int(codes.shape[0])

    # Count codes chunkwise.  Negative codes are missing values in pandas categoricals.
    code_counts: Counter[int] = Counter()
    for start in range(0, n, chunk_rows):
        block = np.asarray(codes[start : min(start + chunk_rows, n)]).ravel()
        values, block_counts = np.unique(block, return_counts=True)
        for value, count in zip(values, block_counts, strict=True):
            code_counts[int(value)] += int(count)

    for code, count in code_counts.items():
        if code < 0:
            counts["<NA>"] += count
        elif code < len(labels):
            counts[labels[code]] += count
        else:
            counts[f"<INVALID_CODE:{code}>"] += count
    return counts


def _value_counts(
    obj: h5py.Group | h5py.Dataset,
    *,
    chunk_rows: int,
) -> Counter[str] | None:
    """Return counts for common H5AD series encodings, or None if unsupported."""
    if isinstance(obj, h5py.Dataset):
        if obj.ndim != 1:
            return None
        # Avoid trying to stringify complex compound records as metadata values.
        if obj.dtype.names:
            return None
        return _counter_from_dataset(obj, chunk_rows=chunk_rows)

    encoding = str(_decode_attr(obj.attrs.get("encoding-type", "")))
    if encoding == "categorical" or {"codes", "categories"}.issubset(obj.keys()):
        return _counter_from_categorical(obj, chunk_rows=chunk_rows)
    return None


def _top_counts(counts: Counter[str], top_n: int) -> list[dict[str, Any]]:
    return [
        {"value": value, "count": int(count)}
        for value, count in counts.most_common(top_n)
    ]


def _index_object(group: h5py.Group | None) -> h5py.Group | h5py.Dataset | None:
    if group is None:
        return None
    index_name = _decode_attr(group.attrs.get("_index", "_index"))
    if isinstance(index_name, str) and index_name in group:
        return group[index_name]
    if "_index" in group:
        return group["_index"]
    return None


def _peek_index(group: h5py.Group | None, n: int = 8) -> list[str]:
    obj = _index_object(group)
    if isinstance(obj, h5py.Dataset) and obj.ndim >= 1:
        take = min(n, int(obj.shape[0]))
        return _decode_strings(obj[:take])
    if isinstance(obj, h5py.Group):
        encoding = str(_decode_attr(obj.attrs.get("encoding-type", "")))
        if encoding == "categorical" or {"codes", "categories"}.issubset(obj.keys()):
            codes = obj.get("codes")
            categories = obj.get("categories")
            if isinstance(codes, h5py.Dataset) and isinstance(categories, h5py.Dataset):
                labels = _decode_strings(categories[...])
                raw_codes = np.asarray(codes[: min(n, int(codes.shape[0]))]).ravel()
                out = []
                for code in raw_codes:
                    code = int(code)
                    out.append("<NA>" if code < 0 else labels[code])
                return out
    return []


def _index_length(group: h5py.Group | None) -> int | None:
    obj = _index_object(group)
    if obj is None:
        return None

    if isinstance(obj, h5py.Dataset) and obj.ndim >= 1:
        return int(obj.shape[0])
    if isinstance(obj, h5py.Group):
        # Rare, but accommodate encoded categorical index.
        codes = obj.get("codes")
        if isinstance(codes, h5py.Dataset):
            return int(codes.shape[0])
    return None


def inspect_seaad_merfish(
    h5ad_path: str | os.PathLike[str],
    *,
    section_column: str | None = None,
    top_values: int = 12,
    chunk_rows: int = 250_000,
) -> dict[str, Any]:
    """Inspect a SEA-AD MERFISH H5AD without loading the expression matrix.

    Parameters
    ----------
    h5ad_path
        Path to the source ``.h5ad`` file.
    section_column
        Optional exact ``obs`` column to summarize as the section identifier.
        If omitted, section-like columns are *reported as candidates* but are not
        declared to be the true section identifier.
    top_values
        Maximum number of values shown for each summarized section-like column.
    chunk_rows
        Number of rows read at a time when counting metadata values.

    Returns
    -------
    dict
        JSON-serializable structural report.

    Notes
    -----
    This is an inspection function, not an AnnData analysis loader.  It uses h5py
    intentionally so the cell-by-gene matrix is never materialized in memory.
    """
    path = Path(h5ad_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    if not path.is_file():
        raise ValueError(f"Not a file: {path}")
    if path.suffix.lower() != ".h5ad":
        raise ValueError(f"Expected an .h5ad file, got: {path.name}")
    if top_values < 1:
        raise ValueError("top_values must be >= 1")
    if chunk_rows < 1:
        raise ValueError("chunk_rows must be >= 1")

    report: dict[str, Any] = {
        "path": str(path),
        "file_size_bytes": int(path.stat().st_size),
        "read_only_inspection": True,
        "expression_values_loaded": False,
    }

    with h5py.File(path, "r") as f:
        report["root_attrs"] = _attrs(f)
        report["top_level_keys"] = sorted(f.keys())

        x_obj = f.get("X")
        report["X"] = _matrix_info(x_obj)

        obs = f.get("obs")
        var = f.get("var")
        obs_group = obs if isinstance(obs, h5py.Group) else None
        var_group = var if isinstance(var, h5py.Group) else None

        obs_columns = _table_columns(obs_group)
        var_columns = _table_columns(var_group)
        n_obs = _index_length(obs_group)
        n_vars = _index_length(var_group)

        # Fall back to X shape when index length is not directly readable.
        x_shape = (report.get("X") or {}).get("shape")
        if n_obs is None and x_shape and len(x_shape) >= 1:
            n_obs = int(x_shape[0])
        if n_vars is None and x_shape and len(x_shape) >= 2:
            n_vars = int(x_shape[1])

        report["n_obs"] = n_obs
        report["n_vars"] = n_vars
        report["obs_columns"] = obs_columns
        report["var_columns"] = var_columns
        report["obs_index_examples"] = _peek_index(obs_group)
        report["var_index_examples"] = _peek_index(var_group)

        # Inventory slots important for future spatial extraction.
        for slot in ("obsm", "obsp", "varm", "varp", "layers", "uns", "raw"):
            obj = f.get(slot)
            if isinstance(obj, h5py.Group):
                report[slot] = {
                    str(k): _object_info(obj[k]) for k in sorted(obj.keys())
                }
            elif isinstance(obj, h5py.Dataset):
                report[slot] = {"<dataset>": _object_info(obj)}
            else:
                report[slot] = {}

        section_candidates = [c for c in obs_columns if _SECTION_HINT.search(c)]
        coordinate_obs_candidates = [c for c in obs_columns if _COORD_HINT.search(c)]

        obsm_candidates: list[str] = []
        obsm = f.get("obsm")
        if isinstance(obsm, h5py.Group):
            for key in sorted(obsm.keys()):
                obj = obsm[key]
                name_match = bool(_COORD_HINT.search(key))
                shape: tuple[int, ...] | None = None
                if isinstance(obj, h5py.Dataset):
                    shape = tuple(int(v) for v in obj.shape)
                elif isinstance(obj, h5py.Group):
                    shape = _shape_from_attrs(obj)
                shape_match = bool(shape and len(shape) == 2 and shape[1] in {2, 3})
                if name_match or shape_match:
                    obsm_candidates.append(key)

        report["section_candidates"] = section_candidates
        report["coordinate_candidates"] = {
            "obs_columns": coordinate_obs_candidates,
            "obsm_keys": obsm_candidates,
        }

        # Summarize candidate section columns, but never silently choose one.
        columns_to_summarize: list[str]
        if section_column is not None:
            if obs_group is None or section_column not in obs_group:
                raise KeyError(
                    f"Requested section column {section_column!r} is not present in obs"
                )
            columns_to_summarize = [section_column]
        else:
            columns_to_summarize = section_candidates

        summaries: dict[str, Any] = {}
        if obs_group is not None:
            for column in columns_to_summarize:
                obj = obs_group[column]
                try:
                    counts = _value_counts(obj, chunk_rows=chunk_rows)
                except Exception as exc:  # inspection should continue for other columns
                    summaries[column] = {
                        "supported": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    continue

                if counts is None:
                    summaries[column] = {
                        "supported": False,
                        "reason": "unsupported series encoding for safe value counting",
                    }
                else:
                    summaries[column] = {
                        "supported": True,
                        "n_unique_including_missing": len(counts),
                        "top_values": _top_counts(counts, top_values),
                    }
        report["section_summaries"] = summaries

        report["warnings"] = []
        if not section_candidates and section_column is None:
            report["warnings"].append(
                "No section-like obs column was identified by name. Specify --section-column "
                "after reviewing obs_columns; do not infer a section identifier from this report."
            )
        if len(section_candidates) > 1 and section_column is None:
            report["warnings"].append(
                "Multiple section-like obs columns were found. They are candidates only; the "
                "inspector does not choose among them."
            )
        if not obsm_candidates and not coordinate_obs_candidates:
            report["warnings"].append(
                "No obvious spatial-coordinate candidate was found in obs/obsm by name or 2D/3D "
                "shape. Coordinates may live in another source object or use unexpected naming."
            )

    return report


def _print_report(report: dict[str, Any]) -> None:
    print("SEA-AD MERFISH H5AD inspection")
    print("=" * 34)
    print(f"File: {report['path']}")
    print(
        f"Size: {_format_bytes(report['file_size_bytes'])} "
        f"({report['file_size_bytes']:,} bytes)"
    )
    print(f"Cells / observations: {report.get('n_obs')}")
    print(f"Genes / variables:    {report.get('n_vars')}")
    if report.get("obs_index_examples"):
        print(f"Cell/index examples:   {report['obs_index_examples']}")
    if report.get("var_index_examples"):
        print(f"Gene/index examples:   {report['var_index_examples']}")

    x = report.get("X")
    print("\nExpression matrix X")
    print("-------------------")
    if x is None:
        print("Missing")
    else:
        print(f"Encoding: {x.get('encoding_type')}")
        print(f"Shape:    {x.get('shape')}")
        if x.get("nnz") is not None:
            print(f"nnz:      {x['nnz']:,}")
            density = x.get("density")
            if density is not None:
                print(f"Density:  {density:.6g}")
        dtype = x.get("dtype") or x.get("data_dtype")
        if dtype:
            print(f"dtype:    {dtype}")

    obs_cols = report.get("obs_columns", [])
    var_cols = report.get("var_columns", [])
    print(f"\nobs columns ({len(obs_cols)})")
    print("----------------")
    for name in obs_cols:
        print(f"  {name}")

    print(f"\nvar columns ({len(var_cols)})")
    print("----------------")
    for name in var_cols:
        print(f"  {name}")

    print("\nobsm arrays")
    print("-----------")
    obsm = report.get("obsm", {})
    if not obsm:
        print("  <none>")
    for key, info in obsm.items():
        shape = info.get("shape")
        dtype = info.get("dtype")
        encoding = info.get("attrs", {}).get("encoding-type")
        details = ", ".join(
            part
            for part in (
                f"shape={shape}" if shape is not None else None,
                f"dtype={dtype}" if dtype is not None else None,
                f"encoding={encoding}" if encoding is not None else None,
            )
            if part
        )
        print(f"  {key}: {details or info.get('kind')}")

    print("\nSection-like obs candidates")
    print("---------------------------")
    candidates = report.get("section_candidates", [])
    if not candidates:
        print("  <none identified by name>")
    else:
        for name in candidates:
            print(f"  {name}")

    print("\nSection candidate summaries")
    print("---------------------------")
    summaries = report.get("section_summaries", {})
    if not summaries:
        print("  <none>")
    for name, summary in summaries.items():
        print(f"  {name}:")
        if not summary.get("supported"):
            reason = summary.get("error") or summary.get("reason")
            print(f"    not summarized: {reason}")
            continue
        print(
            "    unique values (including missing): "
            f"{summary['n_unique_including_missing']}"
        )
        for row in summary["top_values"]:
            print(f"    {row['count']:>10,}  {row['value']}")

    coords = report.get("coordinate_candidates", {})
    print("\nCoordinate candidates")
    print("---------------------")
    print("  obs columns:")
    for name in coords.get("obs_columns", []):
        print(f"    {name}")
    if not coords.get("obs_columns"):
        print("    <none>")
    print("  obsm keys:")
    for name in coords.get("obsm_keys", []):
        print(f"    {name}")
    if not coords.get("obsm_keys"):
        print("    <none>")

    layers = report.get("layers", {})
    print("\nOther slots")
    print("-----------")
    print(f"  layers: {', '.join(layers.keys()) if layers else '<none>'}")
    print(
        "  uns:    "
        + (", ".join(report.get("uns", {}).keys()) if report.get("uns") else "<none>")
    )

    warnings = report.get("warnings", [])
    if warnings:
        print("\nWarnings")
        print("--------")
        for warning in warnings:
            print(f"  - {warning}")

    print("\nSafety")
    print("------")
    print("  Read-only inspection: yes")
    print("  Expression values loaded: no")
    print("  Candidate metadata counts are streamed in chunks.")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect a SEA-AD MERFISH .h5ad structurally without loading the "
            "cell-by-gene expression matrix."
        )
    )
    parser.add_argument("--h5ad", required=True, type=Path, help="Path to the source .h5ad file")
    parser.add_argument(
        "--section-column",
        help=(
            "Exact obs column to summarize as the section identifier. If omitted, "
            "section-like columns are reported as candidates only."
        ),
    )
    parser.add_argument(
        "--top-values",
        type=int,
        default=12,
        help="Number of most common values shown per summarized metadata column (default: 12)",
    )
    parser.add_argument(
        "--chunk-rows",
        type=int,
        default=250_000,
        help="Rows per metadata-counting chunk (default: 250000)",
    )
    parser.add_argument(
        "--json",
        type=Path,
        dest="json_path",
        help="Optional path for the complete machine-readable JSON report",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = inspect_seaad_merfish(
        args.h5ad,
        section_column=args.section_column,
        top_values=args.top_values,
        chunk_rows=args.chunk_rows,
    )
    _print_report(report)

    if args.json_path is not None:
        out = args.json_path.expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, sort_keys=True)
            f.write("\n")
        print(f"\nWrote JSON report: {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
