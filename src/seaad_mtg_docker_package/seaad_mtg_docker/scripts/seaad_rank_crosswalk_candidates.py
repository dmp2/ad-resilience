#!/usr/bin/env python3
"""
Aggregate and rank candidate crosswalk signals across the outputs of:

1) seaad_spatial_section_manifest.py
2) seaad_mtg_spatial_neuropath_linkage.py
3) seaad_mtg_h5ad_crosswalk_probe.py

Goal:
Produce one ranked report of the strongest candidate .obs columns and
identifier-like fields that could support a donor/sample/section crosswalk.

Inputs expected:
- spatial section manifest CSV
- neuropath stain manifest CSV
- h5ad obs column profile CSV
- h5ad obs column crosswalk probe CSV
- optional donor_by_candidate__*.csv files from script 3
- optional candidate linkage CSV from script 2

Outputs:
- ranked_crosswalk_candidates.csv
- ranked_crosswalk_candidates_detailed.csv
- best_candidate_summary.json
- donor_candidate_coverage.csv
- crosswalk_recommendations.txt

Typical use:
    python seaad_rank_crosswalk_candidates.py \
      --probe-dir seaad_mtg_h5ad_crosswalk_probe \
      --linkage-dir seaad_mtg_linkage \
      --outdir seaad_crosswalk_ranked
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
LOG = logging.getLogger("seaad_crosswalk_ranker")


# ----------------------------
# Utilities
# ----------------------------

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().replace("\n", " ").replace("\r", " ") for c in df.columns]
    return df


def read_csv_if_exists(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    return normalize_columns(pd.read_csv(path))


def safe_int(x) -> int:
    if pd.isna(x):
        return 0
    try:
        return int(x)
    except Exception:
        try:
            return int(float(x))
        except Exception:
            return 0


def score01(x: float, cap: float) -> float:
    if cap <= 0:
        return 0.0
    return max(0.0, min(float(x) / float(cap), 1.0))


def sanitize_name(x: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", x)


# ----------------------------
# Core readers
# ----------------------------

def load_probe_outputs(probe_dir: Path) -> Dict[str, Optional[pd.DataFrame]]:
    files = {
        "obs_profile": probe_dir / "obs_column_profile.csv",
        "obs_probe": probe_dir / "obs_column_crosswalk_probe.csv",
        "obs_meta_subset": probe_dir / "obs_metadata_subset.csv",
        "h5ad_donor_summary": probe_dir / "h5ad_donor_summary.csv",
    }
    out = {k: read_csv_if_exists(v) for k, v in files.items()}
    return out


def load_linkage_outputs(linkage_dir: Path) -> Dict[str, Optional[pd.DataFrame]]:
    files = {
        "spatial_section_manifest": linkage_dir / "spatial_section_manifest.csv",
        "spatial_donor_manifest": linkage_dir / "spatial_donor_manifest.csv",
        "neuropath_stain_manifest": linkage_dir / "neuropath_stain_manifest.csv",
        "candidate_linkage": linkage_dir / "candidate_spatial_neuropath_donor_linkage.csv",
        "candidate_linkage_joined": linkage_dir / "candidate_spatial_neuropath_donor_linkage_joined.csv",
    }
    out = {k: read_csv_if_exists(v) for k, v in files.items()}
    return out


def find_candidate_tables(probe_dir: Path) -> List[Path]:
    return sorted(probe_dir.glob("donor_by_candidate__*.csv"))


# ----------------------------
# Feature engineering
# ----------------------------

def derive_reference_caps(linkage: Dict[str, Optional[pd.DataFrame]]) -> Dict[str, int]:
    caps = {
        "n_spatial_sections": 1,
        "n_neuropath_stain_folders": 1,
        "n_shared_donors": 1,
    }

    spatial_section = linkage.get("spatial_section_manifest")
    if spatial_section is not None:
        if "section_id" in spatial_section.columns:
            caps["n_spatial_sections"] = max(1, spatial_section["section_id"].dropna().nunique())
        elif "spatial_section_id" in spatial_section.columns:
            caps["n_spatial_sections"] = max(1, spatial_section["spatial_section_id"].dropna().nunique())

        donor_col = "Donor ID" if "Donor ID" in spatial_section.columns else "donor_id" if "donor_id" in spatial_section.columns else None
        if donor_col:
            caps["n_shared_donors"] = max(1, spatial_section[donor_col].dropna().nunique())

    neuropath_stain = linkage.get("neuropath_stain_manifest")
    if neuropath_stain is not None and "stain_folder" in neuropath_stain.columns:
        caps["n_neuropath_stain_folders"] = max(1, neuropath_stain["stain_folder"].dropna().nunique())

    candidate_linkage = linkage.get("candidate_linkage")
    if candidate_linkage is not None and "Donor ID" in candidate_linkage.columns:
        caps["n_shared_donors"] = max(1, candidate_linkage["Donor ID"].dropna().nunique())

    return caps


def merge_profile_and_probe(
    obs_profile: pd.DataFrame,
    obs_probe: pd.DataFrame,
) -> pd.DataFrame:
    df = obs_profile.merge(obs_probe, on="column", how="left")
    numeric_cols = [
        "candidate_score",
        "n_unique_nonempty",
        "n_donor_exact_matches",
        "n_spatial_section_exact_matches",
        "n_neuropath_stain_folder_exact_matches",
        "n_neuropath_block_or_slide_exact_matches",
        "n_neuropath_stain_code_exact_matches",
        "n_spatial_token_matches",
        "n_neuropath_token_matches",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    bool_cols = ["is_candidate_by_name", "is_candidate_by_values", "has_any_donor_like", "has_any_long_numeric_like"]
    for c in bool_cols:
        if c in df.columns:
            df[c] = df[c].fillna(False).astype(bool)

    return df


def compute_candidate_scores(
    df: pd.DataFrame,
    caps: Dict[str, int],
) -> pd.DataFrame:
    out = df.copy()

    # Normalized signal components
    out["score_name_signal"] = out["is_candidate_by_name"].astype(int)
    out["score_value_signal"] = out["is_candidate_by_values"].astype(int)
    out["score_donor_like"] = out["has_any_donor_like"].astype(int)
    out["score_numeric_like"] = out["has_any_long_numeric_like"].astype(int)

    out["score_donor_exact"] = out["n_donor_exact_matches"].map(lambda x: score01(x, caps["n_shared_donors"]))
    out["score_spatial_section_exact"] = out["n_spatial_section_exact_matches"].map(lambda x: score01(x, caps["n_spatial_sections"]))
    out["score_neuropath_folder_exact"] = out["n_neuropath_stain_folder_exact_matches"].map(lambda x: score01(x, caps["n_neuropath_stain_folders"]))
    out["score_neuropath_block_exact"] = out["n_neuropath_block_or_slide_exact_matches"].map(lambda x: score01(x, caps["n_neuropath_stain_folders"]))
    out["score_spatial_token"] = out["n_spatial_token_matches"].map(lambda x: score01(x, caps["n_spatial_sections"]))
    out["score_neuropath_token"] = out["n_neuropath_token_matches"].map(lambda x: score01(x, caps["n_neuropath_stain_folders"]))

    # Uniqueness / information content
    if "n_unique_nonempty" in out.columns:
        max_unique = max(1, int(out["n_unique_nonempty"].max()))
        out["score_information_density"] = out["n_unique_nonempty"].map(lambda x: score01(x, max_unique))
    else:
        out["score_information_density"] = 0.0

    # Weighted aggregate:
    # prioritize direct exact matches first, then token matches, then heuristic metadata cues
    out["crosswalk_score"] = (
        0.24 * out["score_spatial_section_exact"]
        + 0.18 * out["score_neuropath_block_exact"]
        + 0.10 * out["score_neuropath_folder_exact"]
        + 0.12 * out["score_donor_exact"]
        + 0.12 * out["score_spatial_token"]
        + 0.08 * out["score_neuropath_token"]
        + 0.05 * out["score_name_signal"]
        + 0.04 * out["score_value_signal"]
        + 0.03 * out["score_donor_like"]
        + 0.02 * out["score_numeric_like"]
        + 0.02 * out["score_information_density"]
    )

    out["crosswalk_score"] = out["crosswalk_score"].round(6)
    out["crosswalk_rank"] = (
        out["crosswalk_score"].rank(method="dense", ascending=False).astype(int)
    )

    # Human-readable class
    def classify(row) -> str:
        if row["n_spatial_section_exact_matches"] > 0 and row["n_neuropath_block_or_slide_exact_matches"] > 0:
            return "strong_multisource_candidate"
        if row["n_spatial_section_exact_matches"] > 0:
            return "strong_spatial_candidate"
        if row["n_neuropath_block_or_slide_exact_matches"] > 0 or row["n_neuropath_stain_folder_exact_matches"] > 0:
            return "strong_neuropath_candidate"
        if row["n_spatial_token_matches"] > 0 or row["n_neuropath_token_matches"] > 0:
            return "token_level_candidate"
        return "weak_candidate"

    out["candidate_class"] = out.apply(classify, axis=1)
    return out.sort_values(
        [
            "crosswalk_score",
            "n_spatial_section_exact_matches",
            "n_neuropath_block_or_slide_exact_matches",
            "n_spatial_token_matches",
            "n_neuropath_token_matches",
            "n_unique_nonempty",
        ],
        ascending=[False, False, False, False, False, False],
    ).reset_index(drop=True)


def summarize_candidate_tables(candidate_paths: List[Path]) -> pd.DataFrame:
    rows = []

    for p in candidate_paths:
        df = pd.read_csv(p)
        df = normalize_columns(df)
        colname = p.stem.replace("donor_by_candidate__", "")

        donor_col = None
        for c in df.columns:
            if c in {"Donor ID", "inferred_donor_id"}:
                donor_col = c
                break
        if donor_col is None:
            continue

        candidate_col = [c for c in df.columns if c not in {donor_col, "n_obs"}]
        if len(candidate_col) != 1:
            continue
        candidate_col = candidate_col[0]

        rows.append(
            {
                "column": candidate_col,
                "candidate_table_path": str(p),
                "n_rows": len(df),
                "n_donors": df[donor_col].dropna().nunique(),
                "n_unique_values": df[candidate_col].dropna().nunique(),
                "max_obs_for_any_pair": pd.to_numeric(df.get("n_obs", 0), errors="coerce").fillna(0).max(),
            }
        )

    if not rows:
        return pd.DataFrame(columns=["column", "candidate_table_path", "n_rows", "n_donors", "n_unique_values", "max_obs_for_any_pair"])
    return pd.DataFrame(rows)


def join_candidate_table_stats(
    ranked: pd.DataFrame,
    candidate_stats: pd.DataFrame,
) -> pd.DataFrame:
    if candidate_stats.empty:
        ranked["candidate_table_available"] = False
        ranked["n_candidate_table_donors"] = pd.NA
        ranked["n_candidate_table_unique_values"] = pd.NA
        return ranked

    out = ranked.merge(candidate_stats, on="column", how="left")
    out["candidate_table_available"] = out["candidate_table_path"].notna()
    out = out.rename(
        columns={
            "n_donors": "n_candidate_table_donors",
            "n_unique_values": "n_candidate_table_unique_values",
        }
    )
    return out


def build_donor_candidate_coverage(
    candidate_paths: List[Path],
    top_columns: List[str],
) -> pd.DataFrame:
    rows = []

    for p in candidate_paths:
        derived_col = p.stem.replace("donor_by_candidate__", "")
        if derived_col not in top_columns:
            continue

        df = pd.read_csv(p)
        df = normalize_columns(df)

        donor_col = None
        for c in df.columns:
            if c in {"Donor ID", "inferred_donor_id"}:
                donor_col = c
                break
        if donor_col is None:
            continue

        candidate_col = [c for c in df.columns if c not in {donor_col, "n_obs"}]
        if len(candidate_col) != 1:
            continue
        candidate_col = candidate_col[0]

        tmp = df[[donor_col, candidate_col]].copy()
        tmp = tmp.rename(columns={donor_col: "Donor ID", candidate_col: "candidate_value"})
        tmp["column"] = derived_col
        rows.append(tmp)

    if not rows:
        return pd.DataFrame(columns=["Donor ID", "column", "candidate_value"])

    out = pd.concat(rows, ignore_index=True)
    out["candidate_value"] = out["candidate_value"].astype(str)
    return out.sort_values(["column", "Donor ID"]).reset_index(drop=True)


def build_recommendation_text(ranked: pd.DataFrame, outpath: Path) -> None:
    top = ranked.head(10).copy()

    lines = []
    lines.append("Best candidate crosswalk columns")
    lines.append("=" * 32)
    lines.append("")

    for i, row in top.iterrows():
        lines.append(
            f"{int(i)+1}. {row['column']} "
            f"(score={row['crosswalk_score']:.3f}, class={row['candidate_class']})"
        )
        lines.append(
            f"   exact: donor={safe_int(row.get('n_donor_exact_matches'))}, "
            f"spatial_section={safe_int(row.get('n_spatial_section_exact_matches'))}, "
            f"neuropath_block={safe_int(row.get('n_neuropath_block_or_slide_exact_matches'))}, "
            f"neuropath_folder={safe_int(row.get('n_neuropath_stain_folder_exact_matches'))}"
        )
        lines.append(
            f"   token: spatial={safe_int(row.get('n_spatial_token_matches'))}, "
            f"neuropath={safe_int(row.get('n_neuropath_token_matches'))}"
        )
        ex = row.get("example_values", "")
        if isinstance(ex, str) and ex:
            lines.append(f"   examples: {ex[:250]}")
        lines.append("")

    lines.append("Recommended inspection order")
    lines.append("- Open ranked_crosswalk_candidates.csv")
    lines.append("- Start with class=strong_multisource_candidate or strong_spatial_candidate")
    lines.append("- Then inspect donor_candidate_coverage.csv for the top 3-5 columns")
    lines.append("- If no strong_multisource candidate exists, use the best spatial candidate to anchor donor/sample IDs and keep neuropathology linkage at donor level")

    outpath.write_text("\n".join(lines))


# ----------------------------
# Main
# ----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--probe-dir", required=True, help="Output dir from script 3")
    p.add_argument("--linkage-dir", required=True, help="Output dir from script 2")
    p.add_argument("--outdir", default="seaad_crosswalk_ranked")
    return p.parse_args()


def main():
    args = parse_args()
    probe_dir = Path(args.probe_dir)
    linkage_dir = Path(args.linkage_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    probe = load_probe_outputs(probe_dir)
    linkage = load_linkage_outputs(linkage_dir)

    if probe["obs_profile"] is None:
        raise FileNotFoundError(f"Missing {probe_dir / 'obs_column_profile.csv'}")
    if probe["obs_probe"] is None:
        raise FileNotFoundError(f"Missing {probe_dir / 'obs_column_crosswalk_probe.csv'}")

    merged = merge_profile_and_probe(
        obs_profile=probe["obs_profile"],
        obs_probe=probe["obs_probe"],
    )

    caps = derive_reference_caps(linkage)
    ranked = compute_candidate_scores(merged, caps)

    candidate_paths = find_candidate_tables(probe_dir)
    candidate_stats = summarize_candidate_tables(candidate_paths)
    ranked = join_candidate_table_stats(ranked, candidate_stats)

    # Save compact and full versions
    compact_cols = [
        "crosswalk_rank",
        "column",
        "candidate_class",
        "crosswalk_score",
        "n_donor_exact_matches",
        "n_spatial_section_exact_matches",
        "n_neuropath_stain_folder_exact_matches",
        "n_neuropath_block_or_slide_exact_matches",
        "n_spatial_token_matches",
        "n_neuropath_token_matches",
        "candidate_score",
        "n_unique_nonempty",
        "example_values",
        "candidate_table_available",
        "n_candidate_table_donors",
        "n_candidate_table_unique_values",
    ]
    compact_cols = [c for c in compact_cols if c in ranked.columns]

    ranked[compact_cols].to_csv(outdir / "ranked_crosswalk_candidates.csv", index=False)
    ranked.to_csv(outdir / "ranked_crosswalk_candidates_detailed.csv", index=False)

    # Donor coverage for top candidates
    top_columns = ranked.head(5)["column"].tolist()
    coverage = build_donor_candidate_coverage(candidate_paths, top_columns)
    coverage.to_csv(outdir / "donor_candidate_coverage.csv", index=False)

    # JSON summary
    summary = {
        "n_total_columns_scored": int(len(ranked)),
        "n_strong_multisource_candidates": int((ranked["candidate_class"] == "strong_multisource_candidate").sum()),
        "n_strong_spatial_candidates": int((ranked["candidate_class"] == "strong_spatial_candidate").sum()),
        "n_strong_neuropath_candidates": int((ranked["candidate_class"] == "strong_neuropath_candidate").sum()),
        "top_10_columns": ranked.head(10)[["column", "crosswalk_score", "candidate_class"]].to_dict(orient="records"),
        "reference_caps": caps,
    }
    with open(outdir / "best_candidate_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    build_recommendation_text(ranked, outdir / "crosswalk_recommendations.txt")

    LOG.info("Wrote ranked outputs to %s", outdir.resolve())


if __name__ == "__main__":
    main()