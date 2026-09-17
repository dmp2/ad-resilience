#!/usr/bin/env python3
"""Export one SEA-AD MTG MERFISH section as an xIV-LDDMM particle approximation.

The output NPZ intentionally follows xIV-LDDMM's named particle contract:

    Z     : N x D spatial coordinates
    nu_Z  : N x F non-negative feature masses

This is the format consumed by::

    from xmodmap.io.getInput import readParticleApproximation
    Z, nu_Z = readParticleApproximation("section_xiv.npz")

Design goals
------------
* Run in the modern Allen/spatial Python environment; do NOT require xIV-LDDMM.
* Read the giant H5AD conservatively with h5py and never materialize all of X.
* Keep the xIV environment isolated behind a simple NumPy NPZ + JSON contract.
* Preserve a stable feature space across sections:
  categorical features use the H5AD's global category order, not section-local order.
* Make every geometry-changing operation explicit and recorded:
  coordinate key, selection rule, scaling, centering, dimensionality, and z padding.
* Reject invalid xIV measure inputs (negative/non-finite features or zero feature mass).

Default biological representation
---------------------------------
One selected MERFISH cell becomes one particle. ``Subclass`` becomes a one-hot
feature vector and each cell has unit feature mass. By default, cells are restricted
to the rectangular pia-to-white-matter analysis region using non-missing
``Depth from pia`` values, consistent with the practical selection used for the
2024-12-11 SEA-AD MTG H5AD.

The script can alternatively export a user-specified gene panel as feature channels.
Gene values are exported exactly as stored unless --row-normalize-gene-features is
requested. Rows with non-positive total feature mass are dropped because many xIV
routines normalize nu_Z by its row sum.

Examples
--------
Subclass particles for one section::

    PYTHONPATH=src python -m preprocess.export_seaad_merfish_section_for_xiv \
      --h5ad SEAAD_MTG_MERFISH.2024-12-11.h5ad \
      --section 1194111462 \
      --output results/xiv/1194111462_subclass.npz

Export all cells rather than the rectangular analysis subset::

    ... --selection all

Explicitly scale coordinates from micrometers to millimeters and mean-center::

    ... --coordinate-scale 1e-3 --source-units um --output-units mm --center mean

Export selected gene channels::

    ... --genes GFAP APOE TREM2 --row-normalize-gene-features

Optionally weight each particle by a non-negative numeric obs field::

    ... --feature-field Subclass --weight-field "Cell volume"

Notes for xIV-LDDMM
-------------------
The current xIV-LDDMM package provides ``readParticleApproximation`` for this exact
NPZ schema. Its ``readSpaceFeatureCSV`` helper also scales, centers, and pads 2-D
coordinates to 3-D, but this exporter deliberately avoids the extra CSV round-trip.
Equivalent operations are available here as explicit, provenance-recorded options.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Sequence

import h5py
import numpy as np

DEFAULT_SECTION_FIELD = "Specimen Barcode"
DEFAULT_DONOR_FIELD = "Donor ID"
DEFAULT_COORDINATE_KEY = "X_spatial_raw"
DEFAULT_SELECTION_FIELD = "Depth from pia"
DEFAULT_FEATURE_FIELD = "Subclass"
EXPORT_SCHEMA_VERSION = 1


def _decode_scalar(x: Any) -> Any:
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="replace")
    if isinstance(x, np.generic):
        return x.item()
    return x


def _decode_array(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.dtype.kind in {"S", "O"}:
        return np.array([_decode_scalar(v) for v in arr], dtype=object)
    return arr


def _safe_name(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("._") or "section"


def _table_columns(group: h5py.Group | None) -> list[str]:
    if group is None:
        return []
    idx = group.attrs.get("_index")
    idx = _decode_scalar(idx) if idx is not None else None
    return sorted([str(k) for k in group.keys() if str(k) != idx])


def _index_length(group: h5py.Group) -> int:
    idx = group.attrs.get("_index")
    idx = _decode_scalar(idx) if idx is not None else None
    if idx is not None and idx in group:
        obj = group[idx]
        if isinstance(obj, h5py.Dataset):
            return int(obj.shape[0])
        if isinstance(obj, h5py.Group) and "codes" in obj:
            return int(obj["codes"].shape[0])
    for key in group.keys():
        obj = group[key]
        if isinstance(obj, h5py.Dataset) and obj.ndim >= 1:
            return int(obj.shape[0])
        if isinstance(obj, h5py.Group) and "codes" in obj:
            return int(obj["codes"].shape[0])
    raise ValueError("Could not determine obs table length")


def _read_categorical_rows(group: h5py.Group, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    codes = np.asarray(group["codes"][rows], dtype=np.int64)
    categories = _decode_array(group["categories"][...])
    return codes, categories


def _read_obs_chunk(obs: h5py.Group, name: str, start: int, stop: int) -> np.ndarray:
    obj = obs[name]
    if isinstance(obj, h5py.Dataset):
        return _decode_array(obj[start:stop])
    if isinstance(obj, h5py.Group) and {"codes", "categories"}.issubset(obj.keys()):
        codes = np.asarray(obj["codes"][start:stop], dtype=np.int64)
        cats = _decode_array(obj["categories"][...])
        out = np.empty(codes.shape[0], dtype=object)
        out[:] = None
        valid = (codes >= 0) & (codes < len(cats))
        if np.any(valid):
            out[valid] = cats[codes[valid]]
        return out
    raise NotImplementedError(f"Unsupported obs column encoding: {name}")


def _read_obs_rows(obs: h5py.Group, name: str, rows: np.ndarray) -> np.ndarray:
    obj = obs[name]
    if isinstance(obj, h5py.Dataset):
        return _decode_array(obj[rows])
    if isinstance(obj, h5py.Group) and {"codes", "categories"}.issubset(obj.keys()):
        codes, cats = _read_categorical_rows(obj, rows)
        out = np.empty(codes.shape[0], dtype=object)
        out[:] = None
        valid = (codes >= 0) & (codes < len(cats))
        if np.any(valid):
            out[valid] = cats[codes[valid]]
        return out
    raise NotImplementedError(f"Unsupported obs column encoding: {name}")


def _nonmissing(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if np.issubdtype(values.dtype, np.number):
        return np.isfinite(values.astype(float, copy=False))
    out = np.ones(len(values), dtype=bool)
    for i, value in enumerate(values):
        if value is None:
            out[i] = False
            continue
        if isinstance(value, float) and np.isnan(value):
            out[i] = False
            continue
        text = str(value).strip()
        if text == "" or text.lower() in {"nan", "none", "null"}:
            out[i] = False
    return out


def _string_equal(values: np.ndarray, target: str) -> np.ndarray:
    target = str(target)
    return np.array([str(v) == target for v in values], dtype=bool)


def _select_rows(
    obs: h5py.Group,
    *,
    section_field: str,
    section_value: str,
    donor_field: str,
    donor_value: str | None,
    selection: str,
    selection_field: str,
    chunk_rows: int,
) -> tuple[np.ndarray, dict[str, int]]:
    n_obs = _index_length(obs)
    section_parts: list[np.ndarray] = []
    selected_parts: list[np.ndarray] = []

    for start in range(0, n_obs, chunk_rows):
        stop = min(n_obs, start + chunk_rows)
        section_values = _read_obs_chunk(obs, section_field, start, stop)
        mask = _string_equal(section_values, section_value)
        if donor_value is not None:
            donor_values = _read_obs_chunk(obs, donor_field, start, stop)
            mask &= _string_equal(donor_values, donor_value)
        if not np.any(mask):
            continue

        local = np.nonzero(mask)[0] + start
        section_parts.append(local)

        if selection == "all":
            selected_parts.append(local)
        else:
            selection_values = _read_obs_chunk(obs, selection_field, start, stop)[mask]
            keep = _nonmissing(selection_values)
            selected_parts.append(local[keep])

    section_rows = (
        np.concatenate(section_parts).astype(np.int64, copy=False)
        if section_parts
        else np.empty(0, dtype=np.int64)
    )
    selected_rows = (
        np.concatenate(selected_parts).astype(np.int64, copy=False)
        if selected_parts
        else np.empty(0, dtype=np.int64)
    )
    return selected_rows, {
        "n_cells_matching_section": int(len(section_rows)),
        "n_cells_after_selection": int(len(selected_rows)),
    }


def _extract_coordinates(obsm: h5py.Group, key: str, rows: np.ndarray) -> np.ndarray:
    obj = obsm[key]
    if not isinstance(obj, h5py.Dataset) or obj.ndim != 2 or obj.shape[1] < 2:
        raise ValueError(f"obsm[{key!r}] is not a supported N x >=2 coordinate dataset")
    return np.asarray(obj[rows], dtype=float)


def _global_categories(obs: h5py.Group, field: str, chunk_rows: int) -> tuple[np.ndarray, str]:
    """Return stable global category labels and how they were obtained."""
    obj = obs[field]
    if isinstance(obj, h5py.Group) and {"codes", "categories"}.issubset(obj.keys()):
        cats = _decode_array(obj["categories"][...])
        return np.array([str(v) for v in cats], dtype=object), "h5ad_categorical_order"

    # Fallback for plain string-like datasets: scan once and sort deterministically.
    if isinstance(obj, h5py.Dataset):
        n = obj.shape[0]
        vals: set[str] = set()
        for start in range(0, n, chunk_rows):
            stop = min(n, start + chunk_rows)
            block = _decode_array(obj[start:stop])
            for v in block:
                if not _nonmissing(np.array([v], dtype=object))[0]:
                    continue
                vals.add(str(v))
        return np.array(sorted(vals), dtype=object), "sorted_global_values"

    raise NotImplementedError(f"Cannot derive categories for obs field {field!r}")


def _categorical_feature_matrix(
    obs: h5py.Group,
    field: str,
    rows: np.ndarray,
    *,
    missing_policy: str,
    chunk_rows: int,
    dtype: np.dtype,
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, Any]]:
    labels, order_source = _global_categories(obs, field, chunk_rows)
    label_list = [str(x) for x in labels.tolist()]
    label_to_index = {label: i for i, label in enumerate(label_list)}

    values = _read_obs_rows(obs, field, rows)
    valid = _nonmissing(values)
    missing_n = int((~valid).sum())

    if missing_policy == "error" and missing_n:
        raise ValueError(f"{missing_n} selected cells have missing {field!r}")

    if missing_policy == "drop":
        rows_kept = rows[valid]
        values_kept = values[valid]
        labels_out = label_list
    elif missing_policy == "keep":
        rows_kept = rows
        values_kept = values.copy()
        missing_label = "__missing__"
        if missing_label in label_to_index:
            raise ValueError(f"Reserved missing label already exists in {field!r}: {missing_label}")
        labels_out = label_list + [missing_label]
        values_kept[~valid] = missing_label
        label_to_index = {label: i for i, label in enumerate(labels_out)}
    else:
        raise ValueError(missing_policy)

    nu = np.zeros((len(rows_kept), len(labels_out)), dtype=dtype)
    for i, value in enumerate(values_kept):
        label = str(value)
        if label not in label_to_index:
            raise ValueError(f"Unexpected category {label!r} in {field!r}")
        nu[i, label_to_index[label]] = 1.0

    meta = {
        "feature_mode": "categorical_obs",
        "feature_field": field,
        "feature_order_source": order_source,
        "n_feature_channels": int(len(labels_out)),
        "n_missing_feature_cells": missing_n,
        "missing_feature_policy": missing_policy,
    }
    return rows_kept, nu, labels_out, meta


def _var_names(f: h5py.File) -> np.ndarray:
    var = f["var"]
    idx = var.attrs.get("_index")
    idx = _decode_scalar(idx) if idx is not None else None
    if idx is None or idx not in var:
        raise ValueError("Could not determine H5AD var index")
    obj = var[idx]
    if isinstance(obj, h5py.Dataset):
        return _decode_array(obj[...])
    if isinstance(obj, h5py.Group) and {"codes", "categories"}.issubset(obj.keys()):
        codes = np.asarray(obj["codes"][:], dtype=np.int64)
        cats = _decode_array(obj["categories"][...])
        out = np.empty(len(codes), dtype=object)
        valid = (codes >= 0) & (codes < len(cats))
        out[:] = None
        out[valid] = cats[codes[valid]]
        return out
    raise NotImplementedError("Unsupported var index encoding")


def _extract_gene_matrix(
    f: h5py.File,
    rows: np.ndarray,
    genes: Sequence[str],
    *,
    chunk_rows: int,
    dtype: np.dtype,
) -> tuple[np.ndarray, list[str]]:
    names = np.array([str(v) for v in _var_names(f)], dtype=object)
    indices: list[int] = []
    labels: list[str] = []
    for gene in genes:
        match = np.where(names == str(gene))[0]
        if len(match) == 0:
            raise KeyError(f"Gene not found in H5AD var names: {gene}")
        indices.append(int(match[0]))
        labels.append(str(gene))

    X = f["X"]
    out = np.zeros((len(rows), len(indices)), dtype=dtype)

    if isinstance(X, h5py.Dataset):
        # h5py supports sorted row fancy indexing; read one requested column at a time.
        for j, gene_index in enumerate(indices):
            out[:, j] = np.asarray(X[rows, gene_index], dtype=dtype).ravel()
        return out, labels

    shape = X.attrs.get("shape")
    if shape is None:
        raise ValueError("Sparse X lacks a shape attribute")
    n_obs, n_vars = [int(v) for v in np.asarray(shape).ravel().tolist()]
    encoding = str(_decode_scalar(X.attrs.get("encoding-type", ""))).lower()

    if "csc" in encoding:
        indptr = np.asarray(X["indptr"][:], dtype=np.int64)
        for j, gene_index in enumerate(indices):
            lo, hi = int(indptr[gene_index]), int(indptr[gene_index + 1])
            col_rows = np.asarray(X["indices"][lo:hi], dtype=np.int64)
            col_data = np.asarray(X["data"][lo:hi], dtype=dtype)
            pos = np.searchsorted(rows, col_rows)
            valid = (pos < len(rows))
            valid &= rows[np.minimum(pos, max(len(rows) - 1, 0))] == col_rows if len(rows) else False
            if np.any(valid):
                out[pos[valid], j] = col_data[valid]
        return out, labels

    if "csr" in encoding or {"data", "indices", "indptr"}.issubset(X.keys()):
        try:
            from scipy.sparse import csr_matrix
        except ImportError as e:
            raise RuntimeError("CSR gene export requires scipy in the modern spatial environment") from e

        for start in range(0, n_obs, chunk_rows):
            stop = min(n_obs, start + chunk_rows)
            left = np.searchsorted(rows, start)
            right = np.searchsorted(rows, stop)
            if right <= left:
                continue
            indptr = np.asarray(X["indptr"][start : stop + 1], dtype=np.int64)
            lo, hi = int(indptr[0]), int(indptr[-1])
            data = np.asarray(X["data"][lo:hi])
            cols = np.asarray(X["indices"][lo:hi], dtype=np.int64)
            local_indptr = indptr - lo
            block = csr_matrix((data, cols, local_indptr), shape=(stop - start, n_vars))
            local_rows = rows[left:right] - start
            out[left:right, :] = np.asarray(block[local_rows, :][:, indices].toarray(), dtype=dtype)
        return out, labels

    raise NotImplementedError(f"Unsupported X encoding for gene export: {encoding or list(X.keys())}")


def _apply_weight_field(
    obs: h5py.Group,
    rows: np.ndarray,
    nu: np.ndarray,
    weight_field: str | None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if weight_field is None:
        return rows, nu, {"weight_field": None, "weight_definition": "unit_cell_weight"}

    values = _read_obs_rows(obs, weight_field, rows)
    try:
        weights = np.asarray(values, dtype=float)
    except Exception as e:
        raise ValueError(f"Weight field {weight_field!r} is not numeric") from e

    valid = np.isfinite(weights) & (weights > 0)
    dropped = int((~valid).sum())
    if np.any(weights[np.isfinite(weights)] < 0):
        raise ValueError(f"Weight field {weight_field!r} contains negative values")

    rows = rows[valid]
    nu = nu[valid] * weights[valid, None].astype(nu.dtype, copy=False)
    return rows, nu, {
        "weight_field": weight_field,
        "weight_definition": "feature_mass_multiplier",
        "n_cells_dropped_for_nonpositive_or_nonfinite_weight": dropped,
        "weight_min": float(np.min(weights[valid])) if np.any(valid) else None,
        "weight_max": float(np.max(weights[valid])) if np.any(valid) else None,
    }


def _drop_invalid_feature_rows(rows: np.ndarray, nu: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    finite = np.isfinite(nu).all(axis=1)
    nonnegative = (nu >= 0).all(axis=1)
    positive_mass = np.sum(nu, axis=1) > 0

    if np.any(finite & ~nonnegative):
        bad = int(np.sum(finite & ~nonnegative))
        raise ValueError(
            f"Feature matrix contains negative values in {bad} rows. xIV particle feature masses must be non-negative."
        )

    keep = finite & nonnegative & positive_mass
    meta = {
        "n_cells_dropped_nonfinite_features": int((~finite).sum()),
        "n_cells_dropped_zero_feature_mass": int((finite & nonnegative & ~positive_mass).sum()),
    }
    return rows[keep], nu[keep], meta


def _transform_coordinates(
    coords: np.ndarray,
    *,
    scale: float,
    center: str,
    dimensions: int,
    z_value: float,
    dtype: np.dtype,
) -> tuple[np.ndarray, dict[str, Any], np.ndarray]:
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError("Coordinates must be N x >=2")
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("coordinate_scale must be finite and > 0")

    use_dims = 2 if dimensions == 2 else min(3, coords.shape[1])
    base = np.asarray(coords[:, :use_dims], dtype=float)
    finite = np.isfinite(base).all(axis=1)
    base = base[finite]

    scaled = base * float(scale)
    if center == "none":
        center_vector = np.zeros(scaled.shape[1], dtype=float)
    elif center == "mean":
        center_vector = np.mean(scaled, axis=0)
    elif center == "bbox":
        center_vector = 0.5 * (np.min(scaled, axis=0) + np.max(scaled, axis=0))
    else:
        raise ValueError(center)
    transformed = scaled - center_vector[None, :]

    if dimensions == 3 and transformed.shape[1] == 2:
        transformed = np.column_stack(
            [transformed, np.full(len(transformed), float(z_value), dtype=float)]
        )
    elif dimensions == 2:
        transformed = transformed[:, :2]
    elif dimensions == 3 and transformed.shape[1] > 3:
        transformed = transformed[:, :3]

    meta = {
        "coordinate_scale": float(scale),
        "centering": center,
        "center_vector_in_scaled_coordinates": [float(v) for v in center_vector.tolist()],
        "output_dimensions": int(dimensions),
        "z_padding_value": float(z_value) if dimensions == 3 and base.shape[1] == 2 else None,
    }
    return transformed.astype(dtype, copy=False), meta, finite


def _memory_estimate(shape: tuple[int, int], dtype: np.dtype) -> int:
    return int(np.prod(shape, dtype=np.int64)) * int(np.dtype(dtype).itemsize)


def _human_bytes(n: int) -> str:
    x = float(n)
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if x < 1024 or unit == "TiB":
            return f"{x:.2f} {unit}"
        x /= 1024
    return f"{n} B"


def export_section_for_xiv(
    *,
    h5ad_path: str | os.PathLike[str],
    section: str,
    output: str | os.PathLike[str] | None,
    donor: str | None,
    section_field: str,
    donor_field: str,
    coordinate_key: str,
    selection: str,
    selection_field: str,
    feature_field: str | None,
    genes: Sequence[str] | None,
    missing_feature: str,
    row_normalize_gene_features: bool,
    weight_field: str | None,
    coordinate_scale: float,
    source_units: str,
    output_units: str,
    center: str,
    dimensions: int,
    z_value: float,
    dtype_name: str,
    chunk_rows: int,
    compress: bool,
    overwrite: bool,
) -> tuple[Path, Path, dict[str, Any]]:
    source = Path(h5ad_path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    if source.suffix.lower() != ".h5ad":
        raise ValueError(f"Expected .h5ad input, got {source.name}")
    if chunk_rows < 1:
        raise ValueError("chunk_rows must be >= 1")
    if dimensions not in {2, 3}:
        raise ValueError("dimensions must be 2 or 3")

    dtype = np.dtype(dtype_name)
    if dtype not in {np.dtype("float32"), np.dtype("float64")}:
        raise ValueError("dtype must be float32 or float64")

    if output is None:
        stem = f"seaad_merfish_{_safe_name(section)}_xiv"
        out_npz = Path.cwd() / f"{stem}.npz"
    else:
        out_npz = Path(output).expanduser().resolve()
        if out_npz.suffix.lower() != ".npz":
            out_npz = out_npz.with_suffix(".npz")
    out_json = out_npz.with_suffix(".json")
    out_npz.parent.mkdir(parents=True, exist_ok=True)

    if not overwrite:
        for p in [out_npz, out_json]:
            if p.exists():
                raise FileExistsError(f"Refusing to overwrite existing output: {p}")

    with h5py.File(source, "r") as f:
        if "obs" not in f or "obsm" not in f:
            raise ValueError("H5AD is missing obs and/or obsm")
        obs = f["obs"]
        obsm = f["obsm"]
        obs_cols = _table_columns(obs)

        for required in [section_field]:
            if required not in obs_cols:
                raise KeyError(f"Required obs field not found: {required}")
        if donor is not None and donor_field not in obs_cols:
            raise KeyError(f"Donor field not found: {donor_field}")
        if selection != "all" and selection_field not in obs_cols:
            raise KeyError(f"Selection field not found: {selection_field}")
        if coordinate_key not in obsm:
            raise KeyError(f"Coordinate key not found in obsm: {coordinate_key}")
        if genes and feature_field:
            raise ValueError("Choose either categorical --feature-field or --genes, not both")
        if not genes and feature_field is None:
            raise ValueError("A categorical --feature-field or --genes list is required")
        if feature_field is not None and feature_field not in obs_cols:
            raise KeyError(f"Feature field not found in obs: {feature_field}")
        if weight_field is not None and weight_field not in obs_cols:
            raise KeyError(f"Weight field not found in obs: {weight_field}")

        rows, selection_meta = _select_rows(
            obs,
            section_field=section_field,
            section_value=section,
            donor_field=donor_field,
            donor_value=donor,
            selection=selection,
            selection_field=selection_field,
            chunk_rows=chunk_rows,
        )
        if len(rows) == 0:
            raise ValueError("No cells remain after section/donor/selection filtering")

        if genes:
            nu, feature_labels = _extract_gene_matrix(
                f, rows, genes, chunk_rows=chunk_rows, dtype=dtype
            )
            feature_meta: dict[str, Any] = {
                "feature_mode": "genes",
                "feature_labels": feature_labels,
                "n_feature_channels": int(len(feature_labels)),
                "gene_values": "as_stored_in_H5AD_X",
                "row_normalized": bool(row_normalize_gene_features),
            }
            if np.any(nu < 0):
                raise ValueError(
                    "Selected gene matrix contains negative values; xIV feature masses must be non-negative. "
                    "Use a non-negative expression representation."
                )
            if row_normalize_gene_features:
                sums = np.sum(nu, axis=1)
                positive = np.isfinite(sums) & (sums > 0)
                dropped_before_norm = int((~positive).sum())
                rows = rows[positive]
                nu = nu[positive]
                sums = sums[positive]
                nu = nu / sums[:, None]
                feature_meta["n_cells_dropped_before_gene_row_normalization"] = dropped_before_norm
        else:
            rows, nu, feature_labels, feature_meta = _categorical_feature_matrix(
                obs,
                str(feature_field),
                rows,
                missing_policy=missing_feature,
                chunk_rows=chunk_rows,
                dtype=dtype,
            )
            feature_meta["feature_labels"] = feature_labels

        rows, nu, weight_meta = _apply_weight_field(obs, rows, nu, weight_field)
        rows, nu, invalid_feature_meta = _drop_invalid_feature_rows(rows, nu)
        if len(rows) == 0:
            raise ValueError("No particles remain after feature/weight validation")

        source_coords = _extract_coordinates(obsm, coordinate_key, rows)
        Z, coordinate_transform_meta, finite_coord_mask = _transform_coordinates(
            source_coords,
            scale=coordinate_scale,
            center=center,
            dimensions=dimensions,
            z_value=z_value,
            dtype=dtype,
        )
        if not np.all(finite_coord_mask):
            dropped = int((~finite_coord_mask).sum())
            rows = rows[finite_coord_mask]
            nu = nu[finite_coord_mask]
        else:
            dropped = 0

        if len(Z) != len(nu):
            raise AssertionError("Coordinate and feature row counts diverged")
        if len(Z) == 0:
            raise ValueError("No finite-coordinate particles remain")
        if not np.isfinite(Z).all():
            raise ValueError("Non-finite coordinates remain after filtering")
        if not np.isfinite(nu).all():
            raise ValueError("Non-finite features remain after filtering")
        if np.any(nu < 0):
            raise ValueError("Negative xIV feature masses remain after filtering")
        row_mass = np.sum(nu, axis=1)
        if np.any(row_mass <= 0):
            raise ValueError("Zero/non-positive xIV feature mass remains after filtering")

        coord_min = np.min(Z, axis=0)
        coord_max = np.max(Z, axis=0)
        feat_sum = np.sum(nu, axis=0)

    # Keep the NPZ deliberately strict: xIV's readParticleApproximation needs only
    # these named numeric arrays. Feature labels/provenance live in JSON.
    if compress:
        np.savez_compressed(out_npz, Z=Z, nu_Z=nu)
    else:
        np.savez(out_npz, Z=Z, nu_Z=nu)

    meta: dict[str, Any] = {
        "schema": "seaad_merfish_xiv_particle_export",
        "schema_version": EXPORT_SCHEMA_VERSION,
        "source": {
            "h5ad": str(source),
            "file_size_bytes": int(source.stat().st_size),
            "file_mtime_ns": int(source.stat().st_mtime_ns),
        },
        "selection": {
            "section_field": section_field,
            "section_value": str(section),
            "donor_field": donor_field,
            "donor_value": donor,
            "selection_mode": selection,
            "selection_field": None if selection == "all" else selection_field,
            **selection_meta,
            "n_particles_exported": int(len(Z)),
        },
        "coordinates": {
            "source_obsm_key": coordinate_key,
            "source_units": source_units,
            "output_units": output_units,
            "source_coordinate_semantics": "not inferred by exporter",
            **coordinate_transform_meta,
            "n_cells_dropped_nonfinite_coordinates": dropped,
            "coordinate_min": [float(v) for v in coord_min.tolist()],
            "coordinate_max": [float(v) for v in coord_max.tolist()],
        },
        "features": {
            **feature_meta,
            **weight_meta,
            **invalid_feature_meta,
            "nu_Z_dtype": str(nu.dtype),
            "particle_mass_min": float(np.min(row_mass)),
            "particle_mass_max": float(np.max(row_mass)),
            "particle_mass_mean": float(np.mean(row_mass)),
            "total_feature_mass_by_channel": [float(v) for v in feat_sum.tolist()],
        },
        "output": {
            "npz": str(out_npz),
            "json": str(out_json),
            "compressed": bool(compress),
            "Z_shape": [int(v) for v in Z.shape],
            "nu_Z_shape": [int(v) for v in nu.shape],
            "Z_dtype": str(Z.dtype),
            "nu_Z_dtype": str(nu.dtype),
            "estimated_uncompressed_Z_bytes": _memory_estimate(Z.shape, Z.dtype),
            "estimated_uncompressed_nu_Z_bytes": _memory_estimate(nu.shape, nu.dtype),
        },
        "xiv_lddmm_contract": {
            "reader": "xmodmap.io.getInput.readParticleApproximation",
            "required_npz_keys": ["Z", "nu_Z"],
            "preferred_reader_over": "xmodmap.io.getInput.getFromFile",
            "reason": "readParticleApproximation accesses named Z and nu_Z arrays rather than positional NPZ arrays",
            "optional_qc_writer": "xmodmap.io.getOutput.writeParticleVTK",
        },
    }
    out_json.write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    return out_npz, out_json, meta


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export one SEA-AD MTG MERFISH section as an xIV-LDDMM particle NPZ (Z, nu_Z)."
    )
    p.add_argument("--h5ad", required=True, help="Path to SEAAD_MTG_MERFISH.2024-12-11.h5ad")
    p.add_argument("--section", required=True, help="Section identifier, normally a Specimen Barcode")
    p.add_argument("--output", default=None, help="Output NPZ path; default is generated from section")
    p.add_argument("--donor", default=None, help="Optional donor restriction")
    p.add_argument("--section-field", default=DEFAULT_SECTION_FIELD)
    p.add_argument("--donor-field", default=DEFAULT_DONOR_FIELD)
    p.add_argument("--coordinate-key", default=DEFAULT_COORDINATE_KEY, help="H5AD obsm coordinate key")

    p.add_argument(
        "--selection",
        choices=["depth_nonmissing", "all", "custom_nonmissing"],
        default="depth_nonmissing",
        help="Cell selection. Default reproduces the rectangular cortical analysis subset.",
    )
    p.add_argument(
        "--selection-field",
        default=DEFAULT_SELECTION_FIELD,
        help="Used only with custom_nonmissing; depth_nonmissing always uses 'Depth from pia'",
    )

    feature = p.add_mutually_exclusive_group()
    feature.add_argument(
        "--feature-field",
        default=DEFAULT_FEATURE_FIELD,
        help="Categorical obs field exported as globally ordered one-hot channels (default: Subclass)",
    )
    feature.add_argument(
        "--genes",
        nargs="+",
        default=None,
        help="Gene symbols exported as feature channels instead of a categorical obs field",
    )
    p.add_argument(
        "--missing-feature",
        choices=["drop", "error", "keep"],
        default="drop",
        help="Policy for missing categorical features",
    )
    p.add_argument(
        "--row-normalize-gene-features",
        action="store_true",
        help="Normalize each gene-feature row to sum to one before optional weighting",
    )
    p.add_argument(
        "--weight-field",
        default=None,
        help="Optional non-negative numeric obs field multiplying each particle's feature vector",
    )

    p.add_argument(
        "--coordinate-scale",
        type=float,
        default=1.0,
        help="Explicit multiplicative coordinate scale; no unit conversion is inferred",
    )
    p.add_argument("--source-units", default="unknown", help="Metadata only")
    p.add_argument("--output-units", default="unknown", help="Metadata only")
    p.add_argument(
        "--center",
        choices=["none", "mean", "bbox"],
        default="none",
        help="Center coordinates after applying coordinate scale",
    )
    p.add_argument(
        "--dimensions",
        type=int,
        choices=[2, 3],
        default=3,
        help="Output dimension. 2-D input is zero-padded to 3-D by default for xIV compatibility.",
    )
    p.add_argument("--z-value", type=float, default=0.0, help="z value used when padding 2-D coordinates")
    p.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    p.add_argument("--chunk-rows", type=int, default=250_000)
    p.add_argument("--no-compress", action="store_true", help="Use np.savez instead of np.savez_compressed")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    selection_field = args.selection_field
    if args.selection == "depth_nonmissing":
        selection_field = "Depth from pia"
    selection_mode = "all" if args.selection == "all" else "custom_nonmissing"

    # argparse gives --feature-field its default even when --genes is supplied only
    # if both are in a mutually-exclusive group with a default on one member. Resolve
    # that explicitly here.
    feature_field = None if args.genes else args.feature_field

    out_npz, out_json, meta = export_section_for_xiv(
        h5ad_path=args.h5ad,
        section=args.section,
        output=args.output,
        donor=args.donor,
        section_field=args.section_field,
        donor_field=args.donor_field,
        coordinate_key=args.coordinate_key,
        selection=selection_mode,
        selection_field=selection_field,
        feature_field=feature_field,
        genes=args.genes,
        missing_feature=args.missing_feature,
        row_normalize_gene_features=args.row_normalize_gene_features,
        weight_field=args.weight_field,
        coordinate_scale=args.coordinate_scale,
        source_units=args.source_units,
        output_units=args.output_units,
        center=args.center,
        dimensions=args.dimensions,
        z_value=args.z_value,
        dtype_name=args.dtype,
        chunk_rows=args.chunk_rows,
        compress=not args.no_compress,
        overwrite=args.overwrite,
    )

    print(f"Wrote xIV particle NPZ: {out_npz}")
    print(f"Wrote provenance JSON: {out_json}")
    print(f"Z:    {tuple(meta['output']['Z_shape'])} {meta['output']['Z_dtype']}")
    print(f"nu_Z: {tuple(meta['output']['nu_Z_shape'])} {meta['output']['nu_Z_dtype']}")
    print(
        "Approx. uncompressed arrays: "
        + _human_bytes(meta['output']['estimated_uncompressed_Z_bytes'] + meta['output']['estimated_uncompressed_nu_Z_bytes'])
    )
    print("xIV reader: from xmodmap.io.getInput import readParticleApproximation")


if __name__ == "__main__":
    main()
