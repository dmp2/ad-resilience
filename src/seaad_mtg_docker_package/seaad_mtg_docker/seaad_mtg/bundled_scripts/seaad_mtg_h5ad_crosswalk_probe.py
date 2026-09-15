#!/usr/bin/env python3
"""
Mine a merged SEA-AD MTG MERFISH .h5ad for hidden donor/sample/section crosswalk
signals and compare them against:
  1) the raw spatial bucket manifest
  2) the raw neuropathology bucket manifest

Best-practice design:
- anndata backed='r' for large .h5ad inspection without loading X into memory
- conservative metadata mining from obs / uns / obsm
- explicit donor-ID parsing
- normalized string matching against section/sample identifiers
- exports frequency tables so you can inspect candidate crosswalk columns manually

Typical use:
    python seaad_mtg_h5ad_crosswalk_probe.py \
      --h5ad SEAAD_MTG_MERFISH.2024-12-11.h5ad \
      --spatial-manifest seaad_mtg_linkage/spatial_section_manifest.csv \
      --neuropath-manifest seaad_mtg_linkage/neuropath_stain_manifest.csv \
      --outdir seaad_mtg_h5ad_crosswalk_probe
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import anndata as ad
import pandas as pd


# ----------------------------
# Logging
# ----------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
LOG = logging.getLogger("seaad_h5ad_crosswalk_probe")


# ----------------------------
# Constants / regex
# ----------------------------

DONOR_RE = re.compile(r"^H\d{2}\.\d{2}\.\d{3}$")
LONG_NUMERIC_RE = re.compile(r"^\d{6,}$")

ID_KEYWORDS = [
    "donor",
    "sample",
    "section",
    "specimen",
    "slice",
    "experiment",
    "dataset",
    "fov",
    "field",
    "image",
    "run",
    "library",
    "batch",
    "region",
    "brain",
    "replicate",
    "barcode",
]

VALUE_KEYWORDS = [
    "middle temporal gyrus",
    "mtg",
    "merfish",
    "spatial",
]


# ----------------------------
# Helpers
# ----------------------------

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
    """
    Aggressive normalization for matching:
    - lowercase
    - remove separators and non-alnum
    """
    s = normalize_string(x).lower()
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


def find_candidate_columns(df: pd.DataFrame) -> List[str]:
    cols = []
    for c in df.columns:
        cl = str(c).lower()
        if any(k in cl for k in ID_KEYWORDS):
            cols.append(c)
    return cols


def detect_donor_column(df: pd.DataFrame) -> Optional[str]:
    preferred = ["Donor ID", "donor_id", "donor", "AIBS ID", "aibs_id", "specimen_label"]
    for c in df.columns:
        if str(c) in preferred:
            return c

    for c in df.columns:
        s = df[c].astype(str).str.strip()
        if s.str.fullmatch(DONOR_RE.pattern, na=False).any():
            return c
    return None


def parse_donor_from_series(series: pd.Series) -> pd.Series:
    def f(x: object) -> Optional[str]:
        s = normalize_string(x)
        if DONOR_RE.fullmatch(s):
            return s
        m = re.search(r"(H\d{2}\.\d{2}\.\d{3})", s)
        return m.group(1) if m else None

    return series.map(f)


def unique_nonempty(series: pd.Series, limit: int = 20) -> List[str]:
    vals = [normalize_string(x) for x in series.dropna().tolist()]
    vals = [v for v in vals if v]
    uniq = list(dict.fromkeys(vals))
    return uniq[:limit]


def infer_id_like_values(series: pd.Series, min_frac: float = 0.05) -> bool:
    """
    Heuristic: is this column plausibly an identifier field?
    """
    s = series.dropna().map(normalize_string)
    s = s[s != ""]
    if len(s) == 0:
        return False

    frac_unique = s.nunique() / len(s)
    frac_long_numeric = s.str.fullmatch(LONG_NUMERIC_RE.pattern, na=False).mean()
    frac_donor = s.str.fullmatch(DONOR_RE.pattern, na=False).mean()

    return (frac_unique >= min_frac) or (frac_long_numeric > 0.01) or (frac_donor > 0.01)


def safe_json_dump(obj: object, path: Path) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


# ----------------------------
# Manifest readers
# ----------------------------

def read_spatial_manifest(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = normalize_columns(df)
    required = {"Donor ID", "section_id"} if "Donor ID" in df.columns else {"donor_id", "section_id"}
    if "donor_id" in df.columns and "Donor ID" not in df.columns:
        df = df.rename(columns={"donor_id": "Donor ID"})
    return df


def read_neuropath_manifest(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = normalize_columns(df)
    if "donor_id" in df.columns and "Donor ID" not in df.columns:
        df = df.rename(columns={"donor_id": "Donor ID"})
    return df


# ----------------------------
# H5AD inspection
# ----------------------------

def inspect_h5ad(h5ad_path: Path) -> Tuple[pd.DataFrame, Dict, List[str]]:
    LOG.info("Opening .h5ad in backed mode: %s", h5ad_path)
    adata = ad.read_h5ad(h5ad_path, backed="r")

    obs = adata.obs.copy()
    obs = normalize_columns(obs)

    basic = {
        "shape": tuple(adata.shape),
        "n_obs": int(adata.n_obs),
        "n_vars": int(adata.n_vars),
        "obs_columns": list(map(str, obs.columns)),
        "var_columns": list(map(str, adata.var.columns)),
        "uns_keys": list(getattr(adata, "uns_keys", lambda: list(adata.uns.keys()))()),
        "obsm_keys": list(getattr(adata.obsm, "keys", lambda: [])()),
    }

    try:
        obs_names = list(map(str, adata.obs_names[:10]))
    except Exception:
        obs_names = []

    basic["obs_names_head"] = obs_names

    return obs, basic, basic["uns_keys"]


def build_obs_column_profile(obs: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for c in obs.columns:
        s = obs[c]
        s_nonempty = s.map(normalize_string)
        s_nonempty = s_nonempty[s_nonempty != ""]

        rows.append(
            {
                "column": c,
                "dtype": str(s.dtype),
                "n_nonmissing": int(s.notna().sum()),
                "n_unique_nonempty": int(s_nonempty.nunique()),
                "is_candidate_by_name": any(k in str(c).lower() for k in ID_KEYWORDS),
                "is_candidate_by_values": infer_id_like_values(s),
                "example_values": " | ".join(unique_nonempty(s, limit=10)),
                "has_any_donor_like": bool(s_nonempty.str.contains(r"H\d{2}\.\d{2}\.\d{3}", regex=True).any()),
                "has_any_long_numeric_like": bool(s_nonempty.str.fullmatch(LONG_NUMERIC_RE.pattern, na=False).any()),
            }
        )

    out = pd.DataFrame(rows)
    out["candidate_score"] = (
        out["is_candidate_by_name"].astype(int)
        + out["is_candidate_by_values"].astype(int)
        + out["has_any_donor_like"].astype(int)
        + out["has_any_long_numeric_like"].astype(int)
    )
    return out.sort_values(["candidate_score", "n_unique_nonempty"], ascending=[False, False]).reset_index(drop=True)


def build_obs_candidate_value_tables(
    obs: pd.DataFrame,
    candidate_columns: List[str],
    outdir: Path,
) -> None:
    for c in candidate_columns:
        s = obs[c].map(normalize_string)
        s = s[s != ""]
        vc = s.value_counts(dropna=False).reset_index()
        vc.columns = [c, "count"]
        vc.to_csv(outdir / f"obs_value_counts__{sanitize_filename(c)}.csv", index=False)


def sanitize_filename(x: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", x)


def extract_obs_metadata_subset(obs: pd.DataFrame, profile_df: pd.DataFrame) -> pd.DataFrame:
    keep_cols = profile_df.loc[profile_df["candidate_score"] >= 2, "column"].tolist()

    donor_col = detect_donor_column(obs)
    if donor_col and donor_col not in keep_cols:
        keep_cols = [donor_col] + keep_cols

    # If no donor column exists, infer one from all candidate columns
    out = obs[keep_cols].copy() if keep_cols else pd.DataFrame(index=obs.index)

    if donor_col is None:
        inferred = None
        for c in out.columns:
            parsed = parse_donor_from_series(out[c])
            if parsed.notna().any():
                inferred = parsed
                break
        if inferred is not None:
            out.insert(0, "inferred_donor_id", inferred)
    else:
        out = out.rename(columns={donor_col: "Donor ID"})

    out.insert(0, "obs_name", obs.index.astype(str))
    return out


# ----------------------------
# Crosswalk probing
# ----------------------------

def build_spatial_reference_sets(spatial_manifest: pd.DataFrame) -> Dict[str, Set[str]]:
    if "Donor ID" not in spatial_manifest.columns and "donor_id" in spatial_manifest.columns:
        spatial_manifest = spatial_manifest.rename(columns={"donor_id": "Donor ID"})

    out = {
        "donor_ids": set(),
        "section_ids": set(),
        "all_tokens": set(),
    }

    if "Donor ID" in spatial_manifest.columns:
        out["donor_ids"] = {normalize_string(x) for x in spatial_manifest["Donor ID"].dropna() if normalize_string(x)}

    if "section_id" in spatial_manifest.columns:
        out["section_ids"] = {normalize_string(x) for x in spatial_manifest["section_id"].dropna() if normalize_string(x)}

    token_sources = []
    for c in ["Donor ID", "section_id"]:
        if c in spatial_manifest.columns:
            token_sources.extend(spatial_manifest[c].dropna().astype(str).tolist())

    out["all_tokens"] = {simplify_string(x) for x in token_sources if simplify_string(x)}
    return out


def build_neuropath_reference_sets(neuropath_manifest: pd.DataFrame) -> Dict[str, Set[str]]:
    if "Donor ID" not in neuropath_manifest.columns and "donor_id" in neuropath_manifest.columns:
        neuropath_manifest = neuropath_manifest.rename(columns={"donor_id": "Donor ID"})

    out = {
        "donor_ids": set(),
        "stain_folders": set(),
        "block_or_slide_codes": set(),
        "stain_codes": set(),
        "all_tokens": set(),
    }

    for c, k in [
        ("Donor ID", "donor_ids"),
        ("stain_folder", "stain_folders"),
        ("block_or_slide_code", "block_or_slide_codes"),
        ("stain_code", "stain_codes"),
    ]:
        if c in neuropath_manifest.columns:
            out[k] = {normalize_string(x) for x in neuropath_manifest[c].dropna() if normalize_string(x)}

    token_sources = []
    for c in ["Donor ID", "stain_folder", "block_or_slide_code", "stain_code"]:
        if c in neuropath_manifest.columns:
            token_sources.extend(neuropath_manifest[c].dropna().astype(str).tolist())

    out["all_tokens"] = {simplify_string(x) for x in token_sources if simplify_string(x)}
    return out


def probe_column_against_reference_sets(
    series: pd.Series,
    column_name: str,
    spatial_refs: Dict[str, Set[str]],
    neuropath_refs: Dict[str, Set[str]],
) -> Dict:
    s = series.map(normalize_string)
    s = s[s != ""]
    uniq = pd.Index(s.unique())

    donor_matches = sum(val in spatial_refs["donor_ids"] or val in neuropath_refs["donor_ids"] for val in uniq)
    section_matches = sum(val in spatial_refs["section_ids"] for val in uniq)
    stain_folder_matches = sum(val in neuropath_refs["stain_folders"] for val in uniq)
    block_matches = sum(val in neuropath_refs["block_or_slide_codes"] for val in uniq)
    stain_code_matches = sum(val in neuropath_refs["stain_codes"] for val in uniq)

    uniq_simple = [simplify_string(v) for v in uniq if simplify_string(v)]
    spatial_token_matches = sum(v in spatial_refs["all_tokens"] for v in uniq_simple)
    neuropath_token_matches = sum(v in neuropath_refs["all_tokens"] for v in uniq_simple)

    examples = []
    for v in uniq[:20]:
        sv = simplify_string(v)
        flags = []
        if normalize_string(v) in spatial_refs["section_ids"]:
            flags.append("spatial_section_exact")
        if normalize_string(v) in neuropath_refs["stain_folders"]:
            flags.append("neuropath_stain_folder_exact")
        if normalize_string(v) in neuropath_refs["block_or_slide_codes"]:
            flags.append("neuropath_block_or_slide_exact")
        if sv in spatial_refs["all_tokens"]:
            flags.append("spatial_token_match")
        if sv in neuropath_refs["all_tokens"]:
            flags.append("neuropath_token_match")
        if flags:
            examples.append({"value": v, "flags": "|".join(flags)})

    return {
        "column": column_name,
        "n_unique_nonempty": int(len(uniq)),
        "n_donor_exact_matches": int(donor_matches),
        "n_spatial_section_exact_matches": int(section_matches),
        "n_neuropath_stain_folder_exact_matches": int(stain_folder_matches),
        "n_neuropath_block_or_slide_exact_matches": int(block_matches),
        "n_neuropath_stain_code_exact_matches": int(stain_code_matches),
        "n_spatial_token_matches": int(spatial_token_matches),
        "n_neuropath_token_matches": int(neuropath_token_matches),
        "matched_examples": examples[:10],
    }


def build_candidate_section_tables(
    meta_subset: pd.DataFrame,
    profile_df: pd.DataFrame,
    outdir: Path,
) -> None:
    donor_col = "Donor ID" if "Donor ID" in meta_subset.columns else "inferred_donor_id" if "inferred_donor_id" in meta_subset.columns else None

    if donor_col is None:
        return

    candidate_cols = [
        c for c in profile_df.loc[profile_df["candidate_score"] >= 2, "column"].tolist()
        if c in meta_subset.columns and c != donor_col
    ]

    for c in candidate_cols:
        tmp = meta_subset[[donor_col, c]].copy()
        tmp[c] = tmp[c].map(normalize_string)
        tmp = tmp[(tmp[donor_col].notna()) & (tmp[c] != "")]
        if tmp.empty:
            continue

        summary = (
            tmp.groupby([donor_col, c], dropna=False)
            .size()
            .reset_index(name="n_obs")
            .sort_values([donor_col, "n_obs"], ascending=[True, False])
        )
        summary.to_csv(outdir / f"donor_by_candidate__{sanitize_filename(c)}.csv", index=False)


def build_obsname_probe_table(
    meta_subset: pd.DataFrame,
    spatial_refs: Dict[str, Set[str]],
    neuropath_refs: Dict[str, Set[str]],
) -> pd.DataFrame:
    if "obs_name" not in meta_subset.columns:
        return pd.DataFrame()

    s = meta_subset["obs_name"].astype(str)
    parsed_donor = parse_donor_from_series(s)

    rows = []
    for val in s.head(5000).unique()[:5000]:
        sval = normalize_string(val)
        sim = simplify_string(sval)
        rows.append(
            {
                "obs_name": sval,
                "parsed_donor_from_obs_name": re.search(r"(H\d{2}\.\d{2}\.\d{3})", sval).group(1)
                if re.search(r"(H\d{2}\.\d{2}\.\d{3})", sval) else None,
                "spatial_token_match": sim in spatial_refs["all_tokens"],
                "neuropath_token_match": sim in neuropath_refs["all_tokens"],
            }
        )
    return pd.DataFrame(rows)


# ----------------------------
# Main
# ----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--h5ad", required=True)
    p.add_argument("--spatial-manifest", required=True, help="CSV from prior script, e.g. spatial_section_manifest.csv")
    p.add_argument("--neuropath-manifest", required=True, help="CSV from prior script, e.g. neuropath_stain_manifest.csv")
    p.add_argument("--outdir", default="seaad_mtg_h5ad_crosswalk_probe")
    return p.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    h5ad_path = Path(args.h5ad)
    spatial_manifest_path = Path(args.spatial_manifest)
    neuropath_manifest_path = Path(args.neuropath_manifest)

    # Load manifests
    spatial_manifest = read_spatial_manifest(spatial_manifest_path)
    neuropath_manifest = read_neuropath_manifest(neuropath_manifest_path)

    # Build reference sets
    spatial_refs = build_spatial_reference_sets(spatial_manifest)
    neuropath_refs = build_neuropath_reference_sets(neuropath_manifest)

    safe_json_dump(
        {
            "spatial_ref_counts": {k: len(v) for k, v in spatial_refs.items()},
            "neuropath_ref_counts": {k: len(v) for k, v in neuropath_refs.items()},
        },
        outdir / "reference_set_counts.json",
    )

    # Inspect h5ad
    obs, basic, uns_keys = inspect_h5ad(h5ad_path)
    safe_json_dump(basic, outdir / "h5ad_basic_info.json")

    # Profile obs columns
    profile_df = build_obs_column_profile(obs)
    profile_df.to_csv(outdir / "obs_column_profile.csv", index=False)

    candidate_columns = profile_df.loc[profile_df["candidate_score"] >= 2, "column"].tolist()
    build_obs_candidate_value_tables(obs, candidate_columns, outdir)

    # Extract metadata subset
    meta_subset = extract_obs_metadata_subset(obs, profile_df)
    meta_subset.to_csv(outdir / "obs_metadata_subset.csv", index=False)

    # Per-donor candidate tables
    build_candidate_section_tables(meta_subset, profile_df, outdir)

    # Probe obs columns against manifests
    probe_rows = []
    for c in candidate_columns:
        if c in obs.columns:
            probe = probe_column_against_reference_sets(
                series=obs[c],
                column_name=c,
                spatial_refs=spatial_refs,
                neuropath_refs=neuropath_refs,
            )
            probe_rows.append(probe)

    probe_df = pd.DataFrame(
        [{k: v for k, v in row.items() if k != "matched_examples"} for row in probe_rows]
    ).sort_values(
        [
            "n_spatial_section_exact_matches",
            "n_neuropath_block_or_slide_exact_matches",
            "n_spatial_token_matches",
            "n_neuropath_token_matches",
        ],
        ascending=[False, False, False, False],
    )
    probe_df.to_csv(outdir / "obs_column_crosswalk_probe.csv", index=False)
    safe_json_dump(probe_rows, outdir / "obs_column_crosswalk_probe_examples.json")

    # Obs-name probe
    obsname_probe = build_obsname_probe_table(meta_subset, spatial_refs, neuropath_refs)
    if not obsname_probe.empty:
        obsname_probe.to_csv(outdir / "obs_name_probe.csv", index=False)

    # Donor-level summary from h5ad metadata subset
    donor_col = "Donor ID" if "Donor ID" in meta_subset.columns else "inferred_donor_id" if "inferred_donor_id" in meta_subset.columns else None
    if donor_col is not None:
        donor_summary = (
            meta_subset.groupby(donor_col, dropna=False)
            .size()
            .reset_index(name="n_obs")
            .sort_values("n_obs", ascending=False)
        )
        donor_summary.to_csv(outdir / "h5ad_donor_summary.csv", index=False)

        # Compare donor sets with raw manifests
        h5ad_donors = set(donor_summary[donor_col].dropna().astype(str))
        spatial_donors = set(spatial_refs["donor_ids"])
        neuropath_donors = set(neuropath_refs["donor_ids"])

        donor_set_report = {
            "n_h5ad_donors": len(h5ad_donors),
            "n_spatial_manifest_donors": len(spatial_donors),
            "n_neuropath_manifest_donors": len(neuropath_donors),
            "n_h5ad_intersect_spatial": len(h5ad_donors & spatial_donors),
            "n_h5ad_intersect_neuropath": len(h5ad_donors & neuropath_donors),
            "h5ad_not_in_spatial_manifest": sorted(h5ad_donors - spatial_donors),
            "h5ad_not_in_neuropath_manifest": sorted(h5ad_donors - neuropath_donors),
        }
        safe_json_dump(donor_set_report, outdir / "donor_set_comparison.json")

    # UNS key snapshot
    uns_df = pd.DataFrame({"uns_key": uns_keys})
    uns_df.to_csv(outdir / "uns_keys.csv", index=False)

    LOG.info("Done. Outputs written to %s", outdir.resolve())


if __name__ == "__main__":
    main()