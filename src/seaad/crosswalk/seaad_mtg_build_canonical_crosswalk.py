#!/usr/bin/env python3
"""
Build canonical SEA-AD MTG crosswalk tables.

Outputs:
1) seaad_mtg_section_canonical_crosswalk.csv
   One row per spatial section / specimen barcode observed in the merged h5ad.

2) seaad_mtg_celllevel_working_crosswalk.parquet
   One row per cell/obs in the merged h5ad with section-level and donor-level linkage.

3) seaad_mtg_unmatched_section_audit.csv
   Audit of h5ad-only and manifest-only section identifiers.

Design assumptions from prior ranking step:
- `Specimen Barcode` is the best section-level key on the spatial side.
- `Donor ID` is the donor-level bridge to neuropathology and related donor tables.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import anndata as ad
import numpy as np
import pandas as pd


DONOR_RE = re.compile(r"^H\d{2}\.\d{2}\.\d{3}$")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().replace("\n", " ").replace("\r", " ") for c in df.columns]
    return df


def normalize_string(x: object) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.lower() in {"nan", "none", "null"}:
        return ""
    return s


def simplify_string(x: object) -> str:
    s = normalize_string(x).lower()
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


def read_csv_if_exists(path: Optional[Path]) -> Optional[pd.DataFrame]:
    if path is None or not path.exists():
        return None
    return normalize_columns(pd.read_csv(path))


def find_first_present(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    colset = {str(c) for c in columns}
    for c in candidates:
        if c in colset:
            return c
    return None


def find_donor_column(df: pd.DataFrame) -> Optional[str]:
    preferred = ["Donor ID", "donor_id", "donor", "AIBS ID", "aibs_id", "specimen_label"]
    for c in df.columns:
        if str(c) in preferred:
            return c
    for c in df.columns:
        s = df[c].astype(str).str.strip()
        if s.str.fullmatch(DONOR_RE.pattern, na=False).any():
            return c
    return None


def choose_barcode_column(df: pd.DataFrame) -> Optional[str]:
    candidates = [
        "Specimen Barcode",
        "specimen_barcode",
        "Section Barcode",
        "section_barcode",
        "section_id",
    ]
    return find_first_present(df.columns, candidates)


def choose_cell_id_column(df: pd.DataFrame) -> Optional[str]:
    candidates = ["Cell ID", "cell_id", "cellid", "CellID"]
    return find_first_present(df.columns, candidates)


def choose_section_name_column(df: pd.DataFrame) -> Optional[str]:
    candidates = ["Section", "section", "Section ID", "section_name"]
    return find_first_present(df.columns, candidates)


def infer_planar_obs_columns(obs: pd.DataFrame) -> List[str]:
    cols: List[str] = []
    preferred_pairs = [
        ("x", "y"),
        ("X", "Y"),
        ("x_ccf", "y_ccf"),
        ("x_section", "y_section"),
        ("center_x", "center_y"),
        ("centroid_x", "centroid_y"),
        ("x_centroid", "y_centroid"),
        ("x_coordinate", "y_coordinate"),
        ("xcoord", "ycoord"),
        ("row", "col"),
        ("pxl_row_in_fullres", "pxl_col_in_fullres"),
    ]
    colset = set(map(str, obs.columns))
    for a, b in preferred_pairs:
        if a in colset and b in colset:
            cols.extend([a, b])
            break

    patterns = ["coord", "centroid", "spatial", "row", "col", "x", "y", "depth"]
    for c in obs.columns:
        cl = str(c).lower()
        if any(p in cl for p in patterns):
            if c not in cols and pd.api.types.is_numeric_dtype(obs[c]):
                cols.append(str(c))

    # Keep a compact set.
    seen = set()
    ordered = []
    for c in cols:
        if c not in seen:
            ordered.append(c)
            seen.add(c)
    return ordered[:12]



def extract_obsm_coords(adata: ad.AnnData) -> pd.DataFrame:
    """
    Try to extract a 2-column spatial coordinate matrix from .obsm.
    Returns an empty DataFrame when nothing suitable is found.
    """
    candidate_keys: List[str] = []
    try:
        obsm_keys = list(adata.obsm.keys())
    except Exception:
        obsm_keys = []

    for k in obsm_keys:
        kl = str(k).lower()
        if any(tok in kl for tok in ["spatial", "coord", "xy", "position", "centroid"]):
            candidate_keys.append(k)
    for k in obsm_keys:
        if k not in candidate_keys:
            candidate_keys.append(k)

    for k in candidate_keys:
        try:
            arr = np.asarray(adata.obsm[k])
        except Exception:
            continue
        if arr.ndim == 2 and arr.shape[0] == adata.n_obs and arr.shape[1] >= 2:
            cols = {
                f"obsm_{k}_0": arr[:, 0],
                f"obsm_{k}_1": arr[:, 1],
            }
            if arr.shape[1] >= 3:
                cols[f"obsm_{k}_2"] = arr[:, 2]
            return pd.DataFrame(cols, index=adata.obs_names.astype(str))
    return pd.DataFrame(index=adata.obs_names.astype(str))



def build_spatial_section_lookup(
    section_manifest: pd.DataFrame,
    spatial_file_manifest: Optional[pd.DataFrame],
) -> pd.DataFrame:
    sec = section_manifest.copy()
    sec = normalize_columns(sec)

    donor_col = find_donor_column(sec)
    if donor_col is None:
        raise ValueError("Could not identify donor column in spatial section manifest")
    if donor_col != "Donor ID":
        sec = sec.rename(columns={donor_col: "Donor ID"})

    if "section_id" not in sec.columns:
        raise ValueError("Spatial section manifest is missing 'section_id'")

    sec["matched_spatial_section_id"] = sec["section_id"].map(normalize_string)

    rename_map = {
        "has_dapi_image": "has_spatial_raw_images",
        "has_detected_transcripts": "has_detected_transcripts_csv",
        "has_cellpose_detected_transcripts": "has_cellpose_detected_transcripts_csv",
    }
    sec = sec.rename(columns={k: v for k, v in rename_map.items() if k in sec.columns})

    keep_cols = [
        "Donor ID",
        "matched_spatial_section_id",
        "n_files",
        "total_size_gb",
        "first_seen",
        "last_seen",
        "has_spatial_raw_images",
        "has_detected_transcripts_csv",
        "has_cellpose_detected_transcripts_csv",
        "has_polyt_image",
    ]
    keep_cols = [c for c in keep_cols if c in sec.columns]
    sec = sec[keep_cols].copy()

    if spatial_file_manifest is not None and not spatial_file_manifest.empty:
        files = spatial_file_manifest.copy()
        files = normalize_columns(files)
        donor_col_f = find_donor_column(files)
        if donor_col_f is None and "donor_id" in files.columns:
            donor_col_f = "donor_id"
        if donor_col_f and donor_col_f != "Donor ID":
            files = files.rename(columns={donor_col_f: "Donor ID"})
        if "section_id" in files.columns:
            files["matched_spatial_section_id"] = files["section_id"].map(normalize_string)
            if "key" in files.columns:
                files["spatial_section_path"] = files["key"].map(
                    lambda x: str(Path(str(x)).parent) if normalize_string(x) else ""
                )
                path_df = (
                    files.loc[files["matched_spatial_section_id"] != "", ["Donor ID", "matched_spatial_section_id", "spatial_section_path"]]
                    .drop_duplicates()
                    .groupby(["Donor ID", "matched_spatial_section_id"], dropna=False)["spatial_section_path"]
                    .agg(lambda s: " | ".join(sorted(set(v for v in s if normalize_string(v)))))
                    .reset_index()
                )
                sec = sec.merge(path_df, on=["Donor ID", "matched_spatial_section_id"], how="left")

    if "spatial_section_path" not in sec.columns:
        sec["spatial_section_path"] = pd.NA

    return sec



def build_donor_flag_lookup(candidate_linkage_joined: Optional[pd.DataFrame]) -> pd.DataFrame:
    if candidate_linkage_joined is None or candidate_linkage_joined.empty:
        return pd.DataFrame(columns=[
            "Donor ID",
            "has_mtg_neuropath_bucket",
            "has_cognition",
            "has_mri_values",
            "has_luminex",
            "has_mtg_quant_csv",
        ])

    df = candidate_linkage_joined.copy()
    df = normalize_columns(df)
    donor_col = find_donor_column(df)
    if donor_col is None:
        raise ValueError("Could not identify donor column in candidate linkage table")
    if donor_col != "Donor ID":
        df = df.rename(columns={donor_col: "Donor ID"})

    out = pd.DataFrame({"Donor ID": df["Donor ID"].map(normalize_string)})
    mapping = {
        "has_neuropath_mtg_bucket": "has_mtg_neuropath_bucket",
        "in_cognition": "has_cognition",
        "has_any_mri_values": "has_mri_values",
        "in_luminex": "has_luminex",
        "in_mtg_quant_csv": "has_mtg_quant_csv",
        "in_mtg_neuropath": "has_mtg_quant_csv",  # fallback naming from other script family
    }
    for src, dst in mapping.items():
        if src in df.columns and dst not in out.columns:
            out[dst] = df[src].fillna(False).astype(bool)
        elif src in df.columns:
            out[dst] = out[dst].fillna(False).astype(bool) | df[src].fillna(False).astype(bool)

    for dst in [
        "has_mtg_neuropath_bucket",
        "has_cognition",
        "has_mri_values",
        "has_luminex",
        "has_mtg_quant_csv",
    ]:
        if dst not in out.columns:
            out[dst] = False

    out = out.drop_duplicates(subset=["Donor ID"]).reset_index(drop=True)
    return out



def infer_boolean(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(int).astype(bool)
    s = series.astype(str).str.strip().str.lower()
    return s.isin(["true", "1", "yes", "y"])



def build_h5ad_obs_tables(h5ad_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    print(f"Opening H5AD in backed mode: {h5ad_path}")
    adata = ad.read_h5ad(h5ad_path, backed="r")
    obs = adata.obs.copy()
    obs = normalize_columns(obs)

    donor_col = find_donor_column(obs)
    if donor_col is None:
        raise ValueError("Could not identify donor column in h5ad .obs")
    if donor_col != "Donor ID":
        obs = obs.rename(columns={donor_col: "Donor ID"})

    barcode_col = choose_barcode_column(obs)
    if barcode_col is None:
        raise ValueError("Could not identify section/specimen barcode column in h5ad .obs")
    if barcode_col != "Specimen Barcode":
        obs = obs.rename(columns={barcode_col: "Specimen Barcode"})

    cell_id_col = choose_cell_id_column(obs)
    if cell_id_col and cell_id_col != "Cell ID":
        obs = obs.rename(columns={cell_id_col: "Cell ID"})

    section_name_col = choose_section_name_column(obs)
    if section_name_col and section_name_col != "Section":
        obs = obs.rename(columns={section_name_col: "Section"})

    planar_cols = infer_planar_obs_columns(obs)
    obsm_df = extract_obsm_coords(adata)

    keep_cols = ["Donor ID", "Specimen Barcode"]
    for c in ["Cell ID", "Section"]:
        if c in obs.columns:
            keep_cols.append(c)
    keep_cols.extend([c for c in planar_cols if c not in keep_cols])

    cell_df = obs[keep_cols].copy()
    cell_df.insert(0, "obs_name", obs.index.astype(str))
    cell_df["Donor ID"] = cell_df["Donor ID"].map(normalize_string)
    cell_df["Specimen Barcode"] = cell_df["Specimen Barcode"].map(normalize_string)

    if not obsm_df.empty:
        obsm_df = obsm_df.copy()
        obsm_df.index = obsm_df.index.astype(str)
        cell_df = cell_df.merge(obsm_df.reset_index().rename(columns={"index": "obs_name"}), on="obs_name", how="left")

    section_df = (
        cell_df.groupby(["Donor ID", "Specimen Barcode"], dropna=False)
        .agg(
            n_cells_or_obs=("obs_name", "size"),
            example_section_name=("Section", lambda s: next((v for v in s.astype(str) if normalize_string(v)), "")) if "Section" in cell_df.columns else ("obs_name", "size"),
        )
        .reset_index()
    )
    if "example_section_name" in section_df.columns and pd.api.types.is_numeric_dtype(section_df["example_section_name"]):
        section_df = section_df.drop(columns=["example_section_name"])

    meta = {
        "n_obs": int(adata.n_obs),
        "n_vars": int(adata.n_vars),
        "obs_columns_used": keep_cols,
        "obsm_columns_added": [c for c in cell_df.columns if c.startswith("obsm_")],
    }
    return section_df, cell_df, meta



def build_unmatched_audit(
    section_canonical: pd.DataFrame,
    spatial_lookup: pd.DataFrame,
    h5ad_section_df: pd.DataFrame,
) -> pd.DataFrame:
    spatial_lookup = spatial_lookup.copy()
    h5ad_section_df = h5ad_section_df.copy()

    spatial_lookup["matched_spatial_section_id"] = spatial_lookup["matched_spatial_section_id"].map(normalize_string)
    h5ad_section_df["Specimen Barcode"] = h5ad_section_df["Specimen Barcode"].map(normalize_string)

    h5ad_keys = set(h5ad_section_df.loc[h5ad_section_df["Specimen Barcode"] != "", "Specimen Barcode"])
    spatial_keys = set(spatial_lookup.loc[spatial_lookup["matched_spatial_section_id"] != "", "matched_spatial_section_id"])

    unmatched_h5ad = sorted(h5ad_keys - spatial_keys)
    unmatched_spatial = sorted(spatial_keys - h5ad_keys)

    spatial_simple = {simplify_string(x): x for x in spatial_keys if simplify_string(x)}
    h5ad_simple = {simplify_string(x): x for x in h5ad_keys if simplify_string(x)}

    rows: List[Dict[str, object]] = []
    h5ad_donor_map = (
        h5ad_section_df[["Donor ID", "Specimen Barcode", "n_cells_or_obs"]]
        .drop_duplicates()
        .rename(columns={"Specimen Barcode": "section_key"})
    )
    spatial_donor_map = (
        spatial_lookup[["Donor ID", "matched_spatial_section_id", "spatial_section_path"]]
        .drop_duplicates()
        .rename(columns={"matched_spatial_section_id": "section_key"})
    )

    for key in unmatched_h5ad:
        simple = simplify_string(key)
        rows.append({
            "unmatched_type": "h5ad_only",
            "section_key": key,
            "Donor ID": next((v for v in h5ad_donor_map.loc[h5ad_donor_map["section_key"] == key, "Donor ID"].tolist() if normalize_string(v)), ""),
            "n_cells_or_obs": int(h5ad_donor_map.loc[h5ad_donor_map["section_key"] == key, "n_cells_or_obs"].sum()),
            "exists_in_h5ad": True,
            "exists_in_spatial_manifest": False,
            "possible_formatting_issue": simple in spatial_simple,
            "possible_matching_key_after_simplify": spatial_simple.get(simple, ""),
            "spatial_section_path": "",
            "note": "Present in merged h5ad but absent from spatial section manifest",
        })

    for key in unmatched_spatial:
        simple = simplify_string(key)
        rows.append({
            "unmatched_type": "spatial_manifest_only",
            "section_key": key,
            "Donor ID": next((v for v in spatial_donor_map.loc[spatial_donor_map["section_key"] == key, "Donor ID"].tolist() if normalize_string(v)), ""),
            "n_cells_or_obs": pd.NA,
            "exists_in_h5ad": False,
            "exists_in_spatial_manifest": True,
            "possible_formatting_issue": simple in h5ad_simple,
            "possible_matching_key_after_simplify": h5ad_simple.get(simple, ""),
            "spatial_section_path": next((v for v in spatial_donor_map.loc[spatial_donor_map["section_key"] == key, "spatial_section_path"].tolist() if normalize_string(v)), ""),
            "note": "Present in spatial section manifest but absent from merged h5ad",
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["unmatched_type", "Donor ID", "section_key"]).reset_index(drop=True)
    return out



def build_canonical_crosswalk(
    h5ad_path: Path,
    spatial_section_manifest_path: Path,
    outdir: Path,
    spatial_file_manifest_path: Optional[Path] = None,
    candidate_linkage_joined_path: Optional[Path] = None,
) -> Dict[str, object]:
    outdir.mkdir(parents=True, exist_ok=True)

    section_manifest = read_csv_if_exists(spatial_section_manifest_path)
    if section_manifest is None:
        raise FileNotFoundError(f"Missing spatial section manifest: {spatial_section_manifest_path}")
    spatial_file_manifest = read_csv_if_exists(spatial_file_manifest_path)
    candidate_linkage_joined = read_csv_if_exists(candidate_linkage_joined_path)

    spatial_lookup = build_spatial_section_lookup(section_manifest, spatial_file_manifest)
    donor_flags = build_donor_flag_lookup(candidate_linkage_joined)
    h5ad_section_df, h5ad_cell_df, h5ad_meta = build_h5ad_obs_tables(h5ad_path)

    # Section-level canonical table.
    section_canonical = h5ad_section_df.merge(
        spatial_lookup,
        left_on=["Donor ID", "Specimen Barcode"],
        right_on=["Donor ID", "matched_spatial_section_id"],
        how="left",
    )
    section_canonical = section_canonical.merge(donor_flags, on="Donor ID", how="left")
    for c in [
        "has_spatial_raw_images",
        "has_detected_transcripts_csv",
        "has_cellpose_detected_transcripts_csv",
        "has_polyt_image",
        "has_mtg_neuropath_bucket",
        "has_cognition",
        "has_mri_values",
        "has_luminex",
        "has_mtg_quant_csv",
    ]:
        if c in section_canonical.columns:
            section_canonical[c] = infer_boolean(section_canonical[c])

    if "matched_spatial_section_id" not in section_canonical.columns:
        section_canonical["matched_spatial_section_id"] = pd.NA
    section_canonical["section_match_status"] = np.where(
        section_canonical["matched_spatial_section_id"].fillna("").astype(str).str.strip() != "",
        "matched",
        "unmatched",
    )

    section_output_cols = [
        "Donor ID",
        "Specimen Barcode",
        "matched_spatial_section_id",
        "n_cells_or_obs",
        "spatial_section_path",
        "has_spatial_raw_images",
        "has_detected_transcripts_csv",
        "has_cellpose_detected_transcripts_csv",
        "has_polyt_image",
        "has_mtg_neuropath_bucket",
        "has_cognition",
        "has_mri_values",
        "has_luminex",
        "has_mtg_quant_csv",
        "n_files",
        "total_size_gb",
        "first_seen",
        "last_seen",
        "example_section_name",
        "section_match_status",
    ]
    section_output_cols = [c for c in section_output_cols if c in section_canonical.columns]
    section_canonical = section_canonical[section_output_cols].sort_values(["Donor ID", "Specimen Barcode"]).reset_index(drop=True)

    # Cell-level working table.
    cell_working = h5ad_cell_df.merge(
        section_canonical[[c for c in section_canonical.columns if c != "n_cells_or_obs"]],
        on=["Donor ID", "Specimen Barcode"],
        how="left",
    )

    # Audit table.
    unmatched_audit = build_unmatched_audit(section_canonical, spatial_lookup, h5ad_section_df)

    # Write outputs.
    section_csv = outdir / "seaad_mtg_section_canonical_crosswalk.csv"
    cell_parquet = outdir / "seaad_mtg_celllevel_working_crosswalk.parquet"
    unmatched_csv = outdir / "seaad_mtg_unmatched_section_audit.csv"
    summary_json = outdir / "seaad_mtg_canonical_crosswalk_summary.json"

    section_canonical.to_csv(section_csv, index=False)
    try:
        cell_working.to_parquet(cell_parquet, index=False)
        parquet_status = "ok"
    except Exception as e:
        parquet_status = f"failed: {e}"
        fallback_csv = outdir / "seaad_mtg_celllevel_working_crosswalk.csv"
        cell_working.to_csv(fallback_csv, index=False)
        cell_parquet = fallback_csv
    unmatched_audit.to_csv(unmatched_csv, index=False)

    summary = {
        "h5ad_path": str(h5ad_path),
        "n_section_rows": int(len(section_canonical)),
        "n_cell_rows": int(len(cell_working)),
        "n_matched_sections": int((section_canonical["section_match_status"] == "matched").sum()),
        "n_unmatched_sections": int((section_canonical["section_match_status"] == "unmatched").sum()),
        "n_unmatched_audit_rows": int(len(unmatched_audit)),
        "celllevel_output": str(cell_parquet),
        "celllevel_output_status": parquet_status,
        "h5ad_meta": h5ad_meta,
    }
    summary_json.write_text(json.dumps(summary, indent=2))

    print(f"Wrote section-level table to: {section_csv}")
    print(f"Wrote cell-level table to: {cell_parquet}")
    print(f"Wrote unmatched audit table to: {unmatched_csv}")
    print(f"Wrote summary to: {summary_json}")
    return summary



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build canonical section-level and cell-level SEA-AD crosswalk tables.")
    p.add_argument("--h5ad", required=True, help="Path to merged SEAAD_MTG_MERFISH h5ad")
    p.add_argument("--spatial-section-manifest", required=True, help="Path to spatial_section_manifest.csv")
    p.add_argument("--spatial-file-manifest", default=None, help="Optional path to spatial_file_manifest.csv")
    p.add_argument("--candidate-linkage-joined", default=None, help="Optional path to candidate_spatial_neuropath_donor_linkage_joined.csv")
    p.add_argument("--outdir", default="seaad_mtg_canonical_crosswalk", help="Output directory")
    return p.parse_args()



def main() -> None:
    args = parse_args()
    build_canonical_crosswalk(
        h5ad_path=Path(args.h5ad),
        spatial_section_manifest_path=Path(args.spatial_section_manifest),
        spatial_file_manifest_path=Path(args.spatial_file_manifest) if args.spatial_file_manifest else None,
        candidate_linkage_joined_path=Path(args.candidate_linkage_joined) if args.candidate_linkage_joined else None,
        outdir=Path(args.outdir),
    )


if __name__ == "__main__":
    main()
