#!/usr/bin/env python3
"""
Build candidate MTG spatial-to-neuropathology linkage tables for SEA-AD.

What this script does:
1. Crawls the public MTG neuropathology bucket
2. Parses donor IDs, stain folders, and file types
3. Summarizes donor-level and stain-level manifests
4. Crawls the public MTG spatial transcriptomics bucket
5. Intersects donors across the two buckets
6. Builds a candidate linkage table at the donor level
7. Optionally joins in your local donor metadata / cognition / MRI / MTG quant neuropath / Luminex tables

Best-practice choices:
- boto3 with unsigned S3 access for public SEA-AD buckets
- paginated recursive listing
- tidy pandas outputs for downstream joins
- explicit parsing with regex and conservative file-role inference

Typical use:
    python seaad_mtg_spatial_neuropath_linkage.py \
      --outdir seaad_mtg_linkage \
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
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import boto3
import pandas as pd
from botocore import UNSIGNED
from botocore.config import Config


# ----------------------------
# Constants
# ----------------------------

AWS_REGION = "us-west-2"

SPATIAL_BUCKET = "sea-ad-spatial-transcriptomics"
SPATIAL_PREFIX = "middle-temporal-gyrus/"

NEUROPATH_BUCKET = "sea-ad-quantitative-neuropathology"
NEUROPATH_PREFIX = "middle-temporal-gyrus/"

DONOR_RE = re.compile(r"^H\d{2}\.\d{2}\.\d{3}$")


# ----------------------------
# Logging
# ----------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
LOG = logging.getLogger("seaad_mtg_linkage")


# ----------------------------
# S3 helpers
# ----------------------------

def make_s3_client(region_name: str = AWS_REGION):
    return boto3.client(
        "s3",
        region_name=region_name,
        config=Config(signature_version=UNSIGNED),
    )


def iter_objects(s3, bucket: str, prefix: str) -> Iterator[Dict]:
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/") and obj.get("Size", 0) == 0:
                continue
            yield obj


def list_common_prefixes(s3, bucket: str, prefix: str, delimiter: str = "/") -> List[str]:
    paginator = s3.get_paginator("list_objects_v2")
    out: List[str] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter=delimiter):
        for cp in page.get("CommonPrefixes", []):
            out.append(cp["Prefix"])
    return sorted(out)


# ----------------------------
# General parsing helpers
# ----------------------------

def split_tokens(key: str) -> List[str]:
    return [tok for tok in key.strip("/").split("/") if tok]


def parse_donor_from_tokens(tokens: List[str]) -> Optional[str]:
    for tok in tokens:
        if DONOR_RE.fullmatch(tok):
            return tok
    return None


def bytes_to_gb(n: int) -> float:
    return round(n / (1024 ** 3), 3)


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
# Spatial parsing
# ----------------------------

def parse_spatial_section_id(tokens: List[str], donor_id: str) -> Optional[str]:
    try:
        i = tokens.index(donor_id)
    except ValueError:
        return None
    if i + 1 < len(tokens):
        return tokens[i + 1]
    return None


def infer_spatial_file_role(filename: str) -> str:
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
    if name.endswith(".csv"):
        return "csv_other"
    if name.endswith(".json"):
        return "json_other"
    if name.endswith(".html"):
        return "html"
    if name.endswith((".tif", ".tiff")):
        return "image_other"
    return "other"


def build_spatial_manifest(s3, bucket: str, prefix: str) -> pd.DataFrame:
    rows = []
    LOG.info("Crawling spatial bucket=%s prefix=%s", bucket, prefix)

    for obj in iter_objects(s3, bucket, prefix):
        key = obj["Key"]
        tokens = split_tokens(key)
        filename = tokens[-1]
        donor_id = parse_donor_from_tokens(tokens)
        section_id = parse_spatial_section_id(tokens, donor_id) if donor_id else None

        rows.append(
            {
                "bucket": bucket,
                "key": key,
                "filename": filename,
                "donor_id": donor_id,
                "spatial_section_id": section_id,
                "size_bytes": int(obj.get("Size", 0)),
                "size_gb": bytes_to_gb(int(obj.get("Size", 0))),
                "last_modified_utc": str(obj["LastModified"]),
                "file_role": infer_spatial_file_role(filename),
                "extension": Path(filename).suffix.lower(),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No spatial objects found under s3://{bucket}/{prefix}")
    return df


# ----------------------------
# Neuropath parsing
# ----------------------------

def parse_neuropath_stain_folder(tokens: List[str], donor_id: str) -> Optional[str]:
    """
    Example:
      middle-temporal-gyrus/H19.33.004/H19.33.004-A06-NEUN/H19.33.004-A06-NEUN.svs
    returns:
      H19.33.004-A06-NEUN
    """
    try:
        i = tokens.index(donor_id)
    except ValueError:
        return None
    if i + 1 < len(tokens):
        return tokens[i + 1]
    return None


def parse_stain_code_from_folder(stain_folder: Optional[str], donor_id: Optional[str]) -> Optional[str]:
    if stain_folder is None or donor_id is None:
        return None
    x = stain_folder
    if x.startswith(donor_id + "-"):
        x = x[len(donor_id) + 1:]
    # expected residual like A06-NEUN or A6-ASYN or A6-I6
    parts = x.split("-")
    if len(parts) >= 2:
        return parts[-1]
    return None


def parse_block_or_slide_code_from_folder(stain_folder: Optional[str], donor_id: Optional[str]) -> Optional[str]:
    if stain_folder is None or donor_id is None:
        return None
    x = stain_folder
    if x.startswith(donor_id + "-"):
        x = x[len(donor_id) + 1:]
    parts = x.split("-")
    if len(parts) >= 2:
        return "-".join(parts[:-1])
    return None


def infer_neuropath_file_role(filename: str, key: str) -> str:
    name = filename.lower()
    key_l = key.lower()

    if "/analysis_masks/" in key_l:
        return "analysis_mask"
    if "/downsampled/" in key_l:
        return "downsampled_image"
    if name.endswith(".annotations"):
        return "annotations"
    if name.endswith(".svs"):
        return "whole_slide_image_svs"
    if name.endswith((".tif", ".tiff")):
        return "image_other"
    if name.endswith(".csv"):
        return "csv_other"
    if name.endswith(".json"):
        return "json_other"
    if name.endswith(".xml"):
        return "xml_other"
    if name.endswith(".html"):
        return "html"
    return "other"


def build_neuropath_manifest(s3, bucket: str, prefix: str) -> pd.DataFrame:
    rows = []
    LOG.info("Crawling neuropath bucket=%s prefix=%s", bucket, prefix)

    for obj in iter_objects(s3, bucket, prefix):
        key = obj["Key"]
        tokens = split_tokens(key)
        filename = tokens[-1]
        donor_id = parse_donor_from_tokens(tokens)
        stain_folder = parse_neuropath_stain_folder(tokens, donor_id) if donor_id else None
        stain_code = parse_stain_code_from_folder(stain_folder, donor_id)
        block_or_slide_code = parse_block_or_slide_code_from_folder(stain_folder, donor_id)

        rows.append(
            {
                "bucket": bucket,
                "key": key,
                "filename": filename,
                "donor_id": donor_id,
                "stain_folder": stain_folder,
                "block_or_slide_code": block_or_slide_code,
                "stain_code": stain_code,
                "size_bytes": int(obj.get("Size", 0)),
                "size_gb": bytes_to_gb(int(obj.get("Size", 0))),
                "last_modified_utc": str(obj["LastModified"]),
                "file_role": infer_neuropath_file_role(filename, key),
                "extension": Path(filename).suffix.lower(),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No neuropathology objects found under s3://{bucket}/{prefix}")
    return df


# ----------------------------
# Summaries
# ----------------------------

def summarize_spatial_by_donor(files_df: pd.DataFrame) -> pd.DataFrame:
    df = files_df[files_df["donor_id"].notna()].copy()

    out = (
        df.groupby("donor_id")
        .agg(
            n_spatial_files=("key", "size"),
            n_spatial_sections=("spatial_section_id", lambda s: s.dropna().nunique()),
            spatial_total_size_gb=("size_gb", "sum"),
            spatial_first_seen=("last_modified_utc", "min"),
            spatial_last_seen=("last_modified_utc", "max"),
        )
        .reset_index()
        .rename(columns={"donor_id": "Donor ID"})
    )

    for role in ["cellpose_detected_transcripts", "dapi_image", "polyt_image"]:
        role_df = (
            df.assign(flag=df["file_role"].eq(role))
            .groupby("donor_id")["flag"]
            .any()
            .reset_index()
            .rename(columns={"donor_id": "Donor ID", "flag": f"has_{role}"})
        )
        out = out.merge(role_df, on="Donor ID", how="left")

    out["spatial_total_size_gb"] = out["spatial_total_size_gb"].round(3)
    return out.sort_values("Donor ID").reset_index(drop=True)


def summarize_neuropath_by_donor(files_df: pd.DataFrame) -> pd.DataFrame:
    df = files_df[files_df["donor_id"].notna()].copy()

    out = (
        df.groupby("donor_id")
        .agg(
            n_neuropath_files=("key", "size"),
            n_stain_folders=("stain_folder", lambda s: s.dropna().nunique()),
            n_stain_codes=("stain_code", lambda s: s.dropna().nunique()),
            neuropath_total_size_gb=("size_gb", "sum"),
            neuropath_first_seen=("last_modified_utc", "min"),
            neuropath_last_seen=("last_modified_utc", "max"),
        )
        .reset_index()
        .rename(columns={"donor_id": "Donor ID"})
    )

    for role in ["whole_slide_image_svs", "annotations", "analysis_mask", "downsampled_image"]:
        role_df = (
            df.assign(flag=df["file_role"].eq(role))
            .groupby("donor_id")["flag"]
            .any()
            .reset_index()
            .rename(columns={"donor_id": "Donor ID", "flag": f"has_{role}"})
        )
        out = out.merge(role_df, on="Donor ID", how="left")

    out["neuropath_total_size_gb"] = out["neuropath_total_size_gb"].round(3)
    return out.sort_values("Donor ID").reset_index(drop=True)


def summarize_neuropath_by_stain(files_df: pd.DataFrame) -> pd.DataFrame:
    df = files_df[
        files_df["donor_id"].notna() & files_df["stain_folder"].notna()
    ].copy()

    out = (
        df.groupby(["donor_id", "stain_folder", "block_or_slide_code", "stain_code"])
        .agg(
            n_files=("key", "size"),
            total_size_gb=("size_gb", "sum"),
            first_seen=("last_modified_utc", "min"),
            last_seen=("last_modified_utc", "max"),
        )
        .reset_index()
        .rename(columns={"donor_id": "Donor ID"})
    )

    for role in ["whole_slide_image_svs", "annotations", "analysis_mask", "downsampled_image"]:
        role_df = (
            df.assign(flag=df["file_role"].eq(role))
            .groupby(["donor_id", "stain_folder"])["flag"]
            .any()
            .reset_index()
            .rename(columns={"donor_id": "Donor ID", "flag": f"has_{role}"})
        )
        out = out.merge(role_df, on=["Donor ID", "stain_folder"], how="left")

    out["total_size_gb"] = out["total_size_gb"].round(3)
    return out.sort_values(["Donor ID", "stain_folder"]).reset_index(drop=True)


# ----------------------------
# Linkage builder
# ----------------------------

def build_candidate_linkage_table(
    spatial_donor_df: pd.DataFrame,
    neuropath_donor_df: pd.DataFrame,
    neuropath_stain_df: pd.DataFrame,
) -> pd.DataFrame:
    spatial = spatial_donor_df.copy()
    neuro = neuropath_donor_df.copy()
    stains = neuropath_stain_df.copy()

    shared = sorted(set(spatial["Donor ID"]).intersection(set(neuro["Donor ID"])))

    base = pd.DataFrame({"Donor ID": shared})
    base["has_spatial_mtg"] = True
    base["has_neuropath_mtg_bucket"] = True

    base = base.merge(spatial, on="Donor ID", how="left", suffixes=("", "_spatial"))
    base = base.merge(neuro, on="Donor ID", how="left", suffixes=("", "_neuro"))

    # Add stain-code inventory per donor
    stain_inventory = (
        stains.groupby("Donor ID")["stain_code"]
        .agg(lambda s: "|".join(sorted(set(x for x in s.dropna().astype(str)))))
        .reset_index()
        .rename(columns={"stain_code": "neuropath_stain_codes_present"})
    )
    base = base.merge(stain_inventory, on="Donor ID", how="left")

    # Add counts for common pathology artifacts
    stain_counts = (
        stains.groupby("Donor ID")
        .agg(
            n_stain_folders=("stain_folder", "nunique"),
            n_annotations=("has_annotations", lambda s: int(pd.Series(s).fillna(False).sum())),
            n_whole_slide_svs=("has_whole_slide_image_svs", lambda s: int(pd.Series(s).fillna(False).sum())),
        )
        .reset_index()
    )
    base = base.merge(stain_counts, on="Donor ID", how="left")

    return base.sort_values("Donor ID").reset_index(drop=True)


def add_local_table_overlap(
    linkage_df: pd.DataFrame,
    donor_metadata_path: Optional[str],
    cognition_path: Optional[str],
    mri_path: Optional[str],
    mtg_neuropath_path: Optional[str],
    luminex_path: Optional[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    matched = linkage_df.copy()

    tables = {
        "donor_metadata": read_table(donor_metadata_path) if donor_metadata_path else None,
        "cognition": read_table(cognition_path) if cognition_path else None,
        "mri": read_table(mri_path) if mri_path else None,
        "mtg_quant_csv": read_table(mtg_neuropath_path) if mtg_neuropath_path else None,
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
    p.add_argument("--spatial-bucket", default=SPATIAL_BUCKET)
    p.add_argument("--spatial-prefix", default=SPATIAL_PREFIX)
    p.add_argument("--neuropath-bucket", default=NEUROPATH_BUCKET)
    p.add_argument("--neuropath-prefix", default=NEUROPATH_PREFIX)
    p.add_argument("--outdir", default="seaad_mtg_linkage")

    # optional local joins
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

    # Spatial manifests
    spatial_files = build_spatial_manifest(s3, args.spatial_bucket, args.spatial_prefix)
    spatial_files.to_csv(outdir / "spatial_file_manifest.csv", index=False)
    spatial_files.to_parquet(outdir / "spatial_file_manifest.parquet", index=False)

    spatial_donor = summarize_spatial_by_donor(spatial_files)
    spatial_donor.to_csv(outdir / "spatial_donor_manifest.csv", index=False)

    # Neuropath manifests
    neuropath_files = build_neuropath_manifest(s3, args.neuropath_bucket, args.neuropath_prefix)
    neuropath_files.to_csv(outdir / "neuropath_file_manifest.csv", index=False)
    neuropath_files.to_parquet(outdir / "neuropath_file_manifest.parquet", index=False)

    neuropath_donor = summarize_neuropath_by_donor(neuropath_files)
    neuropath_donor.to_csv(outdir / "neuropath_donor_manifest.csv", index=False)

    neuropath_stain = summarize_neuropath_by_stain(neuropath_files)
    neuropath_stain.to_csv(outdir / "neuropath_stain_manifest.csv", index=False)

    # Candidate donor-level linkage
    linkage = build_candidate_linkage_table(
        spatial_donor_df=spatial_donor,
        neuropath_donor_df=neuropath_donor,
        neuropath_stain_df=neuropath_stain,
    )
    linkage.to_csv(outdir / "candidate_spatial_neuropath_donor_linkage.csv", index=False)

    # Optional local joins
    if any([
        args.donor_metadata,
        args.cognition,
        args.mri,
        args.mtg_neuropath,
        args.luminex,
    ]):
        linkage_joined, linkage_summary = add_local_table_overlap(
            linkage_df=linkage,
            donor_metadata_path=args.donor_metadata,
            cognition_path=args.cognition,
            mri_path=args.mri,
            mtg_neuropath_path=args.mtg_neuropath,
            luminex_path=args.luminex,
        )
        linkage_joined.to_csv(outdir / "candidate_spatial_neuropath_donor_linkage_joined.csv", index=False)
        linkage_summary.to_csv(outdir / "candidate_spatial_neuropath_donor_linkage_summary.csv", index=False)

        core_cols = [c for c in linkage_joined.columns if c in {
            "in_donor_metadata", "in_cognition", "in_mtg_quant_csv", "in_luminex"
        }]
        if core_cols:
            core = linkage_joined[linkage_joined[core_cols].fillna(False).all(axis=1)].copy()
            if "has_any_mri_values" in core.columns:
                core = core[core["has_any_mri_values"].fillna(False)]
            core.to_csv(outdir / "core_overlap_spatial_neuropath_local_tables.csv", index=False)

    LOG.info("Wrote outputs to %s", outdir.resolve())


if __name__ == "__main__":
    main()