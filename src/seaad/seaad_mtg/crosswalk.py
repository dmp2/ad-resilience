from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Optional

import anndata as ad
import pandas as pd

DONOR_REGEX = re.compile(r"^H\d{2}\.\d{2}\.\d{3}$")


def normalize_colnames(df: pd.DataFrame) -> pd.DataFrame:
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
        if s.str.fullmatch(DONOR_REGEX.pattern, na=False).any():
            return c
    return None


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    elif path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported file type: {path}")
    return normalize_colnames(df)


def extract_candidate_columns(obs: pd.DataFrame) -> List[str]:
    patterns = [
        "donor", "sample", "section", "specimen", "slice",
        "experiment", "dataset", "region", "brain", "fov",
        "cell", "barcode", "library", "batch",
    ]
    out: List[str] = []
    for c in obs.columns:
        cl = str(c).lower()
        if any(p in cl for p in patterns):
            out.append(c)
    return out


def choose_obs_columns(obs: pd.DataFrame) -> List[str]:
    cols = extract_candidate_columns(obs)
    seen = set()
    ordered: List[str] = []
    for c in cols:
        if c not in seen:
            ordered.append(c)
            seen.add(c)
    return ordered


def has_any_mri_values(df: pd.DataFrame, donor_col: str) -> pd.Series:
    value_cols = [c for c in df.columns if c != donor_col]
    return df[value_cols].notna().any(axis=1)


def build_crosswalk(
    h5ad_path: Path,
    donor_metadata_path: Path,
    cognition_path: Path,
    mri_path: Path,
    mtg_neuropath_path: Path,
    luminex_path: Path,
    specimen_metadata_path: Optional[Path],
    outdir: Path,
    obs_donor_column: Optional[str] = None,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Opening H5AD in backed mode: {h5ad_path}")
    adata = ad.read_h5ad(h5ad_path, backed="r")

    print("\n=== BASIC INFO ===")
    print(f"shape: {adata.shape}")
    print(f"n_obs: {adata.n_obs}")
    print(f"n_vars: {adata.n_vars}")

    obs = adata.obs.copy()
    obs = normalize_colnames(obs)

    pd.DataFrame({"obs_column": list(obs.columns)}).to_csv(outdir / "obs_columns.csv", index=False)

    candidate_cols = choose_obs_columns(obs)
    pd.DataFrame({"candidate_obs_column": candidate_cols}).to_csv(
        outdir / "candidate_obs_columns.csv", index=False
    )

    print("\n=== CANDIDATE OBS COLUMNS ===")
    for c in candidate_cols:
        print(c)

    spatial_donor_col = obs_donor_column or find_donor_column(obs)
    print(f"\nDetected donor column in spatial .obs: {spatial_donor_col}")

    if spatial_donor_col is None:
        raise RuntimeError(
            "Could not detect a donor column in .obs automatically. "
            "Re-run with --obs-donor-column after inspecting obs_columns.csv."
        )

    keep_cols = list(dict.fromkeys([spatial_donor_col] + candidate_cols))
    spatial_meta = obs[keep_cols].copy()
    spatial_meta.insert(0, "obs_name", obs.index.astype(str))
    spatial_meta.to_csv(outdir / "spatial_obs_metadata_subset.csv", index=False)

    spatial_donors = (
        spatial_meta[[spatial_donor_col]]
        .drop_duplicates()
        .rename(columns={spatial_donor_col: "Donor ID"})
        .sort_values("Donor ID")
        .reset_index(drop=True)
    )

    cell_counts = (
        spatial_meta.groupby(spatial_donor_col, dropna=False)
        .size()
        .reset_index(name="n_obs")
        .rename(columns={spatial_donor_col: "Donor ID"})
        .sort_values(["n_obs", "Donor ID"], ascending=[False, True])
    )

    spatial_donors.to_csv(outdir / "spatial_unique_donors.csv", index=False)
    cell_counts.to_csv(outdir / "spatial_donor_obs_counts.csv", index=False)

    print(f"\nUnique spatial donors: {len(spatial_donors)}")
    print(cell_counts.head(10))

    tables = {}
    paths = [
        ("donor_metadata", donor_metadata_path),
        ("cognition", cognition_path),
        ("mri", mri_path),
        ("mtg_neuropath", mtg_neuropath_path),
        ("luminex", luminex_path),
    ]
    if specimen_metadata_path is not None:
        paths.append(("specimen_metadata", specimen_metadata_path))

    for label, path in paths:
        try:
            df = read_table(path)
            donor_col = find_donor_column(df)
            if donor_col is None:
                print(f"[WARN] Could not detect donor column in {label}: {path.name}")
                continue
            df = df.rename(columns={donor_col: "Donor ID"})
            tables[label] = df
            print(f"[OK] {label}: {df.shape[0]} rows, donor column = 'Donor ID'")
        except FileNotFoundError:
            print(f"[WARN] File not found, skipping: {path}")
        except Exception as e:
            print(f"[WARN] Failed to read {path}: {e}")

    matched = spatial_donors.copy()
    matched["in_spatial_mtg_merfish"] = True

    for label, df in tables.items():
        donors_present = set(df["Donor ID"].dropna().astype(str).str.strip())
        matched[f"in_{label}"] = matched["Donor ID"].astype(str).isin(donors_present)

    if "mri" in tables:
        mri_df = tables["mri"].copy()
        mri_has_values = has_any_mri_values(mri_df, "Donor ID")
        mri_presence = pd.DataFrame(
            {"Donor ID": mri_df["Donor ID"].astype(str), "has_any_mri_values": mri_has_values}
        ).drop_duplicates(subset=["Donor ID"])
        matched = matched.merge(mri_presence, on="Donor ID", how="left")
    else:
        matched["has_any_mri_values"] = pd.NA

    matched = matched.merge(cell_counts, on="Donor ID", how="left")
    matched = matched.sort_values("Donor ID").reset_index(drop=True)
    matched.to_csv(outdir / "spatial_matched_donor_table.csv", index=False)

    print("\n=== MATCHED TABLE SUMMARY ===")
    summary_rows = []
    for col in matched.columns:
        if col.startswith("in_") or col == "has_any_mri_values":
            summary_rows.append({"field": col, "n_true": int(matched[col].fillna(False).astype(bool).sum())})

    summary_df = pd.DataFrame(summary_rows).sort_values("field")
    summary_df.to_csv(outdir / "spatial_matched_summary.csv", index=False)
    print(summary_df)

    core_cols = [
        "in_spatial_mtg_merfish",
        "in_donor_metadata",
        "in_cognition",
        "in_mtg_neuropath",
        "in_luminex",
    ]
    if "has_any_mri_values" in matched.columns:
        core_overlap = matched[
            matched[core_cols].all(axis=1) & matched["has_any_mri_values"].fillna(False).astype(bool)
        ].copy()
    else:
        core_overlap = matched[matched[core_cols].all(axis=1)].copy()

    core_overlap.to_csv(outdir / "core_overlap_spatial_cognition_pathology_mri.csv", index=False)

    print(
        "\nCore overlap donors (spatial + donor metadata + cognition + MTG neuropath + Luminex + MRI values): "
        f"{len(core_overlap)}"
    )
    print(core_overlap[["Donor ID", "n_obs"]].head(20))

    possible_extra_cols = [c for c in keep_cols if c != spatial_donor_col]
    if possible_extra_cols:
        donor_examples = (
            spatial_meta.groupby(spatial_donor_col, dropna=False)[possible_extra_cols]
            .agg(lambda s: s.dropna().astype(str).iloc[0] if len(s.dropna()) else pd.NA)
            .reset_index()
            .rename(columns={spatial_donor_col: "Donor ID"})
        )
        donor_examples.to_csv(outdir / "spatial_donor_example_metadata.csv", index=False)

    print(f"\nDone. Outputs written to: {outdir.resolve()}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build donor-level MTG crosswalk tables from a local SEA-AD MERFISH h5ad and companion tables.")
    p.add_argument("--h5ad", required=True)
    p.add_argument("--donor-metadata", required=True)
    p.add_argument("--cognition", required=True)
    p.add_argument("--mri", required=True)
    p.add_argument("--mtg-neuropath", required=True)
    p.add_argument("--luminex", required=True)
    p.add_argument("--specimen-metadata", default=None)
    p.add_argument("--outdir", default="seaad_mtg_spatial_crosswalk_outputs")
    p.add_argument("--obs-donor-column", default=None, help="Optional manual override for the donor column in adata.obs")
    return p


def main() -> None:
    args = build_parser().parse_args()
    build_crosswalk(
        h5ad_path=Path(args.h5ad),
        donor_metadata_path=Path(args.donor_metadata),
        cognition_path=Path(args.cognition),
        mri_path=Path(args.mri),
        mtg_neuropath_path=Path(args.mtg_neuropath),
        luminex_path=Path(args.luminex),
        specimen_metadata_path=Path(args.specimen_metadata) if args.specimen_metadata else None,
        outdir=Path(args.outdir),
        obs_donor_column=args.obs_donor_column,
    )


if __name__ == "__main__":
    main()
