#!/usr/bin/env python3
"""
Build a section-level manifest and donor crosswalk from the public
SEA-AD spatial transcriptomics S3 bucket.

Best-practice choices:
- boto3 + unsigned S3 access for public buckets
- paginated listing for scalability
- tidy pandas outputs for downstream joins
- explicit donor/section parsing with regex guards

Typical use:
    python seaad_spatial_section_manifest.py \
        --region-prefix middle-temporal-gyrus/ \
        --outdir seaad_spatial_manifest \
        --donor-metadata 68debdfdd1b8e9f8fd64dab0_sea-ad_cohort_donor_metadata_072524.xlsx \
        --cognition 68debdfd4748b7546943a7b4_sea-ad_cohort_harmonized_cognitive_scores_20241213.xlsx \
        --mri 68debdfdae5f82b97af2fb0f_sea-ad_cohort_mri_volumetrics.xlsx \
        --mtg-neuropath 68debdfd24606956df13f2dd_sea-ad_all_mtg_quant_neuropath_bydonorid_081122.csv \
        --luminex 68debdff5b8003454786ea29_sea-ad_cohort_mtg-tissue_extractions-luminex_data.xlsx
"""

from __future__ import annotations

import argparse
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import boto3
import pandas as pd
from botocore import UNSIGNED
from botocore.config import Config


# ----------------------------
# Constants
# ----------------------------

AWS_REGION = "us-west-2"
SPATIAL_BUCKET = "sea-ad-spatial-transcriptomics"
DEFAULT_REGION_PREFIX = "middle-temporal-gyrus/"

DONOR_RE = re.compile(r"^H\d{2}\.\d{2}\.\d{3}$")
NUMERIC_TOKEN_RE = re.compile(r"^\d{6,}$")


# ----------------------------
# Logging
# ----------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
LOG = logging.getLogger("seaad_spatial_manifest")


# ----------------------------
# S3 helpers
# ----------------------------

def make_s3_client(region_name: str = AWS_REGION):
    return boto3.client(
        "s3",
        region_name=region_name,
        config=Config(signature_version=UNSIGNED),
    )


def list_common_prefixes(
    s3,
    bucket: str,
    prefix: str,
    delimiter: str = "/",
) -> List[str]:
    """
    List "folders" immediately under a prefix using CommonPrefixes.
    """
    paginator = s3.get_paginator("list_objects_v2")
    out: List[str] = []

    for page in paginator.paginate(
        Bucket=bucket,
        Prefix=prefix,
        Delimiter=delimiter,
    ):
        for cp in page.get("CommonPrefixes", []):
            out.append(cp["Prefix"])

    return sorted(out)


def iter_objects(
    s3,
    bucket: str,
    prefix: str,
) -> Iterator[Dict]:
    """
    Recursively iterate all objects under a prefix.
    """
    paginator = s3.get_paginator("list_objects_v2")

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            # skip directory marker objects if present
            if key.endswith("/") and obj.get("Size", 0) == 0:
                continue
            yield obj


# ----------------------------
# Metadata parsing helpers
# ----------------------------

def split_key_tokens(key: str) -> List[str]:
    return [tok for tok in key.strip("/").split("/") if tok]


def parse_donor_from_tokens(tokens: List[str]) -> Optional[str]:
    for tok in tokens:
        if DONOR_RE.fullmatch(tok):
            return tok
    return None


def parse_section_id_from_tokens(tokens: List[str], donor_id: str) -> Optional[str]:
    """
    For keys like:
      middle-temporal-gyrus/H20.33.001/1194111462/DAPI_Max.tif
    return 1194111462

    Falls back to the token immediately after donor_id if present.
    """
    try:
        donor_idx = tokens.index(donor_id)
    except ValueError:
        return None

    if donor_idx + 1 >= len(tokens):
        return None

    candidate = tokens[donor_idx + 1]
    return candidate


def infer_file_role(filename: str) -> str:
    name = filename.lower()

    if name.endswith(".h5ad"):
        return "merged_or_annotated_h5ad"
    if "cellpose" in name and "transcript" in name and name.endswith(".csv"):
        return "cellpose_detected_transcripts"
    if "detected_transcripts" in name and name.endswith(".csv"):
        return "detected_transcripts"
    if "dapi" in name and name.endswith((".tif", ".tiff")):
        return "dapi_image"
    if "polyt" in name and name.endswith((".tif", ".tiff")):
        return "polyt_image"
    if "manifest" in name:
        return "manifest"
    if name.endswith(".csv"):
        return "csv_other"
    if name.endswith(".json"):
        return "json_other"
    if name.endswith((".tif", ".tiff")):
        return "image_other"
    if name.endswith(".html"):
        return "html"
    return "other"


def bytes_to_gb(x: int) -> float:
    return round(x / (1024 ** 3), 3)


# ----------------------------
# Table helpers for optional joins
# ----------------------------

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().replace("\n", " ").replace("\r", " ") for c in df.columns]
    return df


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


def read_table(path: Optional[str]) -> Optional[pd.DataFrame]:
    if path is None:
        return None

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p}")

    if p.suffix.lower() == ".csv":
        df = pd.read_csv(p)
    elif p.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(p)
    else:
        raise ValueError(f"Unsupported file type: {p}")

    df = normalize_columns(df)
    donor_col = find_donor_column(df)
    if donor_col is None:
        raise ValueError(f"Could not identify donor column in {p}")
    return df.rename(columns={donor_col: "Donor ID"})


def has_any_nonmissing_values(df: pd.DataFrame, id_col: str = "Donor ID") -> pd.Series:
    value_cols = [c for c in df.columns if c != id_col]
    return df[value_cols].notna().any(axis=1)


# ----------------------------
# Manifest builders
# ----------------------------

def build_spatial_file_manifest(s3, bucket: str, region_prefix: str) -> pd.DataFrame:
    rows: List[Dict] = []

    LOG.info("Crawling bucket=%s prefix=%s", bucket, region_prefix)

    for obj in iter_objects(s3, bucket, region_prefix):
        key = obj["Key"]
        tokens = split_key_tokens(key)
        filename = tokens[-1]
        donor_id = parse_donor_from_tokens(tokens)

        section_id = None
        if donor_id is not None:
            section_id = parse_section_id_from_tokens(tokens, donor_id)

        row = {
            "bucket": bucket,
            "key": key,
            "filename": filename,
            "region_prefix": region_prefix,
            "donor_id": donor_id,
            "section_id": section_id,
            "size_bytes": int(obj.get("Size", 0)),
            "size_gb": bytes_to_gb(int(obj.get("Size", 0))),
            "last_modified_utc": pd.Timestamp(obj["LastModified"]).tz_convert("UTC").isoformat()
            if hasattr(obj["LastModified"], "tzinfo")
            else str(obj["LastModified"]),
            "extension": Path(filename).suffix.lower(),
            "file_role": infer_file_role(filename),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No objects found under s3://{bucket}/{region_prefix}")
    return df


def build_donor_manifest(files_df: pd.DataFrame) -> pd.DataFrame:
    donor_df = files_df[files_df["donor_id"].notna()].copy()

    summary = (
        donor_df.groupby("donor_id", dropna=False)
        .agg(
            n_files=("key", "size"),
            n_sections=("section_id", lambda s: s.dropna().nunique()),
            total_size_gb=("size_gb", "sum"),
            first_seen=("last_modified_utc", "min"),
            last_seen=("last_modified_utc", "max"),
        )
        .reset_index()
        .rename(columns={"donor_id": "Donor ID"})
    )

    for role in [
        "cellpose_detected_transcripts",
        "detected_transcripts",
        "dapi_image",
        "polyt_image",
    ]:
        role_presence = (
            donor_df.assign(flag=donor_df["file_role"].eq(role))
            .groupby("donor_id")["flag"]
            .any()
            .reset_index()
            .rename(columns={"donor_id": "Donor ID", "flag": f"has_{role}"})
        )
        summary = summary.merge(role_presence, on="Donor ID", how="left")

    summary["total_size_gb"] = summary["total_size_gb"].round(3)
    return summary.sort_values("Donor ID").reset_index(drop=True)


def build_section_manifest(files_df: pd.DataFrame) -> pd.DataFrame:
    section_df = files_df[
        files_df["donor_id"].notna() & files_df["section_id"].notna()
    ].copy()

    summary = (
        section_df.groupby(["donor_id", "section_id"], dropna=False)
        .agg(
            n_files=("key", "size"),
            total_size_gb=("size_gb", "sum"),
            first_seen=("last_modified_utc", "min"),
            last_seen=("last_modified_utc", "max"),
        )
        .reset_index()
        .rename(columns={"donor_id": "Donor ID"})
    )

    for role in [
        "cellpose_detected_transcripts",
        "detected_transcripts",
        "dapi_image",
        "polyt_image",
    ]:
        role_presence = (
            section_df.assign(flag=section_df["file_role"].eq(role))
            .groupby(["donor_id", "section_id"])["flag"]
            .any()
            .reset_index()
            .rename(
                columns={
                    "donor_id": "Donor ID",
                    "flag": f"has_{role}",
                }
            )
        )
        summary = summary.merge(role_presence, on=["Donor ID", "section_id"], how="left")

    summary["total_size_gb"] = summary["total_size_gb"].round(3)
    return summary.sort_values(["Donor ID", "section_id"]).reset_index(drop=True)


def build_all_donors_h5ad_manifest(files_df: pd.DataFrame) -> pd.DataFrame:
    mask = files_df["key"].str.contains(r"/all_donors-h5ad/", regex=True, na=False)
    out = files_df.loc[mask].copy()
    return out.sort_values("key").reset_index(drop=True)


def build_top_level_prefix_manifest(s3, bucket: str, region_prefix: str) -> pd.DataFrame:
    prefixes = list_common_prefixes(s3, bucket, region_prefix)
    rows = []

    for pref in prefixes:
        tokens = split_key_tokens(pref)
        tail = tokens[-1] if tokens else pref
        donor_id = tail if DONOR_RE.fullmatch(tail) else None
        rows.append(
            {
                "bucket": bucket,
                "region_prefix": region_prefix,
                "prefix": pref,
                "tail_token": tail,
                "is_donor_prefix": donor_id is not None,
                "donor_id": donor_id,
            }
        )

    return pd.DataFrame(rows).sort_values("prefix").reset_index(drop=True)


# ----------------------------
# Optional donor-table joining
# ----------------------------

def join_local_tables(
    donor_manifest: pd.DataFrame,
    donor_metadata_path: Optional[str],
    cognition_path: Optional[str],
    mri_path: Optional[str],
    mtg_neuropath_path: Optional[str],
    luminex_path: Optional[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    matched = donor_manifest.copy()

    tables = {
        "donor_metadata": read_table(donor_metadata_path) if donor_metadata_path else None,
        "cognition": read_table(cognition_path) if cognition_path else None,
        "mri": read_table(mri_path) if mri_path else None,
        "mtg_neuropath": read_table(mtg_neuropath_path) if mtg_neuropath_path else None,
        "luminex": read_table(luminex_path) if luminex_path else None,
    }

    for label, df in tables.items():
        if df is None:
            continue

        present = set(df["Donor ID"].astype(str).str.strip().dropna())
        matched[f"in_{label}"] = matched["Donor ID"].astype(str).isin(present)

        if label == "mri":
            mri_has = (
                pd.DataFrame(
                    {
                        "Donor ID": df["Donor ID"].astype(str).str.strip(),
                        "has_any_mri_values": has_any_nonmissing_values(df),
                    }
                )
                .drop_duplicates(subset=["Donor ID"])
            )
            matched = matched.merge(mri_has, on="Donor ID", how="left")

    summary_rows = []
    for c in matched.columns:
        if c.startswith("in_") or c == "has_any_mri_values":
            summary_rows.append(
                {"field": c, "n_true": int(matched[c].fillna(False).astype(bool).sum())}
            )

    summary = pd.DataFrame(summary_rows).sort_values("field").reset_index(drop=True)
    return matched, summary


# ----------------------------
# Main
# ----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--bucket", default=SPATIAL_BUCKET)
    p.add_argument("--region-prefix", default=DEFAULT_REGION_PREFIX)
    p.add_argument("--outdir", default="seaad_spatial_section_manifest")

    # optional donor table joins
    p.add_argument("--donor-metadata", default=None)
    p.add_argument("--cognition", default=None)
    p.add_argument("--mri", default=None)
    p.add_argument("--mtg-neuropath", default=None)
    p.add_argument("--luminex", default=None)

    return p.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    s3 = make_s3_client()

    # 1) Top-level prefixes under region
    top_prefix_df = build_top_level_prefix_manifest(s3, args.bucket, args.region_prefix)
    top_prefix_df.to_csv(outdir / "top_level_prefixes.csv", index=False)

    # 2) Full file manifest
    files_df = build_spatial_file_manifest(s3, args.bucket, args.region_prefix)
    files_df.to_csv(outdir / "spatial_file_manifest.csv", index=False)
    files_df.to_parquet(outdir / "spatial_file_manifest.parquet", index=False)

    # 3) Section and donor manifests
    donor_manifest = build_donor_manifest(files_df)
    donor_manifest.to_csv(outdir / "spatial_donor_manifest.csv", index=False)

    section_manifest = build_section_manifest(files_df)
    section_manifest.to_csv(outdir / "spatial_section_manifest.csv", index=False)

    all_h5ad_manifest = build_all_donors_h5ad_manifest(files_df)
    all_h5ad_manifest.to_csv(outdir / "spatial_all_donors_h5ad_manifest.csv", index=False)

    # 4) Optional joins to your downloaded donor tables
    if any([
        args.donor_metadata,
        args.cognition,
        args.mri,
        args.mtg_neuropath,
        args.luminex,
    ]):
        matched, matched_summary = join_local_tables(
            donor_manifest=donor_manifest,
            donor_metadata_path=args.donor_metadata,
            cognition_path=args.cognition,
            mri_path=args.mri,
            mtg_neuropath_path=args.mtg_neuropath,
            luminex_path=args.luminex,
        )
        matched.to_csv(outdir / "spatial_donor_manifest_joined.csv", index=False)
        matched_summary.to_csv(outdir / "spatial_donor_manifest_joined_summary.csv", index=False)

        core_cols = [c for c in matched.columns if c in {
            "in_donor_metadata", "in_cognition", "in_mtg_neuropath", "in_luminex"
        }]
        if core_cols:
            core = matched[matched[core_cols].fillna(False).all(axis=1)].copy()
            if "has_any_mri_values" in core.columns:
                core = core[core["has_any_mri_values"].fillna(False)]
            core.to_csv(outdir / "core_overlap_from_raw_spatial_bucket.csv", index=False)

    LOG.info("Wrote outputs to %s", outdir.resolve())


if __name__ == "__main__":
    main()