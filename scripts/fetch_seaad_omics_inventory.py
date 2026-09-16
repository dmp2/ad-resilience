#!/usr/bin/env python3
"""Inventory the public SEA-AD single-cell S3 bucket and fetch only its small
summary files.

Scope: this is the read-only acquisition step behind
results/analysis/01_seaad_omics_exploration.qmd. It lists object keys (free) and
downloads only objects under --max-bytes, so no multi-GB AnnData object is ever
pulled by accident. One pseudobulk object is fetched deliberately, by name, as a
structure probe: it is the smallest of the 29 and exists only to establish the
row grain and expression scale of that family.

The bucket is the AWS Open Data mirror of the SEA-AD release
(https://registry.opendata.aws/allen-sea-ad-atlas/); no credentials are used.

Run with an interpreter that has h5py + numpy, e.g.
    ~/miniconda3/envs/wsi-pipeline/bin/python scripts/fetch_seaad_omics_inventory.py
"""

from __future__ import annotations

import argparse
import csv
import json
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

BUCKET = "https://sea-ad-single-cell-profiling.s3.us-west-2.amazonaws.com/"
NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"

# Region folders in the bucket. `PFC/` holds the DFC (A9) data under
# `SEAAD_DFC_*` filenames -- the folder name is the older DLPFC/PFC label, the
# file name is the region label the rest of the release uses. `DFC/` and
# `DLPFC/` are stubs that carry only a changelog.
REGION_FOLDERS = ["AnG", "Caudate_Nucleus", "DFC", "DLPFC", "FI", "HIP", "ITG",
                  "LEC", "MEC", "MTG", "PFC", "STG", "V1C"]

# Enumerated in full: these are the object-level listings the notebook reads.
# `donors_objects/` is the per-donor split of each region and is what gives the
# donor x region grid without opening a single AnnData file.
FULL_PREFIXES = ([f"{r}/RNAseq/donors_objects/" for r in REGION_FOLDERS]
                 + ["Multiregion_2026/"])

# Enumerated one level deep only. `MTG/RNAseq/` alone holds >12,000 keys, almost
# all of them per-donor pseudoprogression plots, which would swamp the manifest.
SHALLOW_PREFIXES = ([f"{r}/RNAseq/" for r in REGION_FOLDERS]
                    + [f"{r}/ATACseq/" for r in REGION_FOLDERS])

# Excluded from the manifest, and counted rather than listed:
#   scANVI/scVI model checkpoints and per-cell probability tables -- tens of GB
#     of model internals this analysis is explicitly not going to use;
#   previous_objects/ -- the superseded MTG/PFC and AAIC microglia pre-releases,
#     ~726,000 keys, deprecated by the 2026-06-22 multiregion release itself.
EXCLUDE_SUBSTRINGS = ("/scANVI_models/", "/scVI_models/", "/previous_objects/")
EXCLUDED_COUNTS: dict[str, int] = {}

# Small published files fetched verbatim. Every one is < 1 MB.
SMALL_FILES = {
    "README.md": "README_bucket.md",
    "Multiregion_2026/README.md": "README_Multiregion_2026.md",
    "Multiregion_2026/cluster_colors_new.2026-06-22.csv":
        "cluster_colors_new.2026-06-22.csv",
    "Multiregion_2026/model_outputs/continuous_pseudo-progression_score/"
    "Global_and_Local_CPS.20260105.csv":
        "Global_and_Local_CPS.20260105.csv",
    "Multiregion_2026/model_outputs/pertpy_compositional_modeling/"
    "pertpy_summary_CPS_Local.20260622.csv":
        "pertpy_summary_CPS_Local.20260622.csv",
}

# scCODA/pertpy abundance objects: library x supertype nuclei counts, ~2 MB total.
ABUNDANCE_FILES = {
    "neuronal": "Multiregion_2026/model_outputs/pertpy_compositional_modeling/"
                "CPS_Local/objects/Neuronal: Glutamatergic Neuronal: "
                "GABAergic_Supertype_abundances.h5ad",
    "non_neuronal": "Multiregion_2026/model_outputs/pertpy_compositional_modeling/"
                    "CPS_Local/objects/Non-neuronal and Non-neural_"
                    "Supertype_abundances.h5ad",
}

# Smallest of the 29 lineage pseudobulk objects, fetched only to read its .obs.
PSEUDOBULK_PROBE = ("Multiregion_2026/pseudobulk_objects/SEAAD_Ependymal_HIP_MEC_"
                    "LEC_ITG_MTG_FI_STG_DFC_AnG_V1C_RNAseq_final-nuclei_"
                    "pseudobulked.2026-06-22.h5ad")

# .obs columns of the probe that describe grain and provenance. The released
# object carries ~120 columns; the rest are sequencing QC and donor metadata
# already available from the public donor workbook.
PROBE_OBS_COLUMNS = [
    "Donor ID", "Brain Region", "Class", "Subclass", "Supertype",
    "library_prep", "Number of nuclei", "Method", "Severely Affected Donor",
    "Neurotypical reference",
]


def find_project_root(start: Path) -> Path:
    p = start.resolve()
    while not (p / ".git").is_dir():
        if p.parent == p:
            raise SystemExit(f"Could not locate project root above {start}")
        p = p.parent
    return p


def list_prefix(prefix: str, delimiter: str | None = None) -> list[dict]:
    """Objects under `prefix`, following continuation tokens.

    With `delimiter="/"` only the objects directly in `prefix` are returned and
    subdirectories are skipped.
    """
    token, out = None, []
    while True:
        q = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if delimiter:
            q["delimiter"] = delimiter
        if token:
            q["continuation-token"] = token
        try:
            with urllib.request.urlopen(BUCKET + "?" + urllib.parse.urlencode(q),
                                        timeout=120) as r:
                root = ET.fromstring(r.read())
        except urllib.error.HTTPError as e:  # a prefix that does not exist
            if e.code == 404:
                return out
            raise
        for c in root.findall(NS + "Contents"):
            key = c.findtext(NS + "Key")
            if key.endswith("/"):
                continue
            hit = next((x for x in EXCLUDE_SUBSTRINGS if x in key), None)
            if hit:
                EXCLUDED_COUNTS[hit] = EXCLUDED_COUNTS.get(hit, 0) + 1
                continue
            out.append({
                "key": key,
                "size_bytes": int(c.findtext(NS + "Size")),
                "last_modified": c.findtext(NS + "LastModified"),
            })
        if root.findtext(NS + "IsTruncated") != "true":
            return out
        token = root.findtext(NS + "NextContinuationToken")


def download(key: str, dest: Path, max_bytes: int | None) -> int:
    """Fetch one object, refusing anything over `max_bytes`."""
    url = BUCKET + urllib.parse.quote(key)
    with urllib.request.urlopen(url, timeout=600) as r:
        size = int(r.headers.get("Content-Length", -1))
        if max_bytes is not None and size > max_bytes:
            raise SystemExit(
                f"Refusing to download {key}: {size / 1e6:.1f} MB exceeds the "
                f"{max_bytes / 1e6:.1f} MB limit. Raise --max-bytes deliberately."
            )
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(r.read())
    return dest.stat().st_size


# --- h5ad readers ----------------------------------------------------------
# Only the two AnnData shapes this script actually touches are handled; both are
# read with plain h5py so that no anndata/scanpy install is required.

def _decode(values):
    return [v.decode() if isinstance(v, bytes) else str(v) for v in values]


def read_obs_column(h5, name: str):
    """One .obs column, resolving AnnData's categorical encoding."""
    import h5py
    import numpy as np

    node = h5["obs"][name]
    if isinstance(node, h5py.Group):
        cats = np.array(_decode(node["categories"][:]))
        codes = node["codes"][:]
        out = np.where(codes >= 0, cats[np.clip(codes, 0, None)], "")
        return out
    values = node[:]
    if values.dtype.kind in "SO":
        return np.array(_decode(values))
    return values


def write_abundances_long(paths: dict[str, Path], out_csv: Path) -> int:
    """Long library x supertype nuclei counts from the scCODA abundance objects.

    Zero cells are dropped: a supertype absent from a library is not an
    observation of zero abundance so much as an absence of that supertype in the
    release's taxonomy for that region, and keeping them inflates the file
    twentyfold.
    """
    import h5py

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    n_rows = 0
    with out_csv.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["library_prep", "brain_region", "supertype", "n_nuclei",
                    "cps_local", "source_object"])
        for label, path in paths.items():
            with h5py.File(path, "r") as h5:
                x = h5["X"][:]
                lib = read_obs_column(h5, "library_prep")
                region = read_obs_column(h5, "Brain Region")
                cps = read_obs_column(h5, "CPS_Local")
                supertypes = _decode(h5["var"]["Supertype"][:])
            for i in range(x.shape[0]):
                for j in range(x.shape[1]):
                    if x[i, j] == 0:
                        continue
                    w.writerow([lib[i], region[i], supertypes[j], int(x[i, j]),
                                f"{cps[i]:.6f}", label])
                    n_rows += 1
    return n_rows


def write_probe_obs(path: Path, out_csv: Path) -> int:
    import h5py

    with h5py.File(path, "r") as h5:
        present = [c for c in PROBE_OBS_COLUMNS if c in h5["obs"]]
        cols = {c: read_obs_column(h5, c) for c in present}
        n = h5["X"].shape[0]
        n_genes = h5["X"].shape[1]
        total_umi = h5["X"][:].sum(axis=1)
        has_layers = len(h5["layers"].keys()) > 0 if "layers" in h5 else False
        x_is_integral = bool((h5["X"][:] % 1 == 0).all())

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(present + ["total_umi"])
        for i in range(n):
            w.writerow([cols[c][i] for c in present] + [int(total_umi[i])])
    return {"n_obs": n, "n_vars": n_genes, "layers_present": has_layers,
            "x_all_integral": x_is_integral}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-bytes", type=int, default=8_000_000,
                    help="Refuse any single download above this size (default 8 MB)")
    ap.add_argument("--skip-listing", action="store_true",
                    help="Reuse an existing manifest instead of re-listing the bucket")
    args = ap.parse_args()

    root = find_project_root(Path(__file__).parent)
    raw_dir = root / "data" / "raw" / "sea-ad" / "multiregion_2026"
    der_dir = root / "data" / "derivatives" / "sea-ad" / "omics_inventory"
    manifest = der_dir / "s3_object_manifest.csv"

    provenance = {
        "bucket": BUCKET,
        "retrieved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "prefixes_listed_recursively": FULL_PREFIXES,
        "prefixes_listed_one_level": SHALLOW_PREFIXES,
        "excluded_substrings": EXCLUDE_SUBSTRINGS,
        "n_keys_excluded": EXCLUDED_COUNTS,
        "max_bytes": args.max_bytes,
    }

    if not args.skip_listing:
        der_dir.mkdir(parents=True, exist_ok=True)
        seen: set[str] = set()
        with manifest.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["key", "size_bytes", "last_modified"])
            w.writeheader()
            jobs = ([(p, None) for p in FULL_PREFIXES]
                    + [(p, "/") for p in SHALLOW_PREFIXES])
            for prefix, delim in jobs:
                rows = [r for r in list_prefix(prefix, delim) if r["key"] not in seen]
                seen.update(r["key"] for r in rows)
                w.writerows(rows)
                fh.flush()
                print(f"listed {prefix}: {len(rows)} new keys", flush=True)
        provenance["n_keys"] = len(seen)
        print(f"manifest -> {manifest} ({len(seen)} keys)")
    else:
        # Carry the listing counts forward so --skip-listing does not silently
        # write a provenance file that disagrees with the manifest beside it.
        previous = json.loads((der_dir / "provenance.json").read_text())
        provenance["n_keys"] = previous.get("n_keys")
        provenance["n_keys_excluded"] = previous.get("n_keys_excluded", {})
        provenance["manifest_reused_from"] = previous.get("retrieved_utc")
        print(f"reusing manifest -> {manifest} ({provenance['n_keys']} keys)")

    for key, name in SMALL_FILES.items():
        n = download(key, raw_dir / name, args.max_bytes)
        print(f"fetched {name} ({n / 1e3:.1f} kB)")

    abundance_paths = {}
    for label, key in ABUNDANCE_FILES.items():
        dest = raw_dir / f"supertype_abundances_{label}.2026-06-22.h5ad"
        n = download(key, dest, args.max_bytes)
        abundance_paths[label] = dest
        print(f"fetched {dest.name} ({n / 1e6:.2f} MB)")

    n_rows = write_abundances_long(
        abundance_paths, der_dir / "supertype_abundances_by_library.csv")
    print(f"abundances -> supertype_abundances_by_library.csv ({n_rows} rows)")

    probe_dest = raw_dir / "pseudobulk_probe_Ependymal.2026-06-22.h5ad"
    download(PSEUDOBULK_PROBE, probe_dest, max_bytes=None)
    probe_shape = write_probe_obs(probe_dest, der_dir / "pseudobulk_probe_obs.csv")
    print(f"pseudobulk probe -> pseudobulk_probe_obs.csv {probe_shape}")

    provenance["small_files"] = list(SMALL_FILES)
    provenance["abundance_files"] = list(ABUNDANCE_FILES.values())
    provenance["pseudobulk_probe"] = PSEUDOBULK_PROBE
    provenance["pseudobulk_probe_shape"] = probe_shape
    (der_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"provenance -> {der_dir / 'provenance.json'}")


if __name__ == "__main__":
    main()
