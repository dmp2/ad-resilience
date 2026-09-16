"""Inspect selected SEA-AD pseudobulk objects without loading all of X into memory."""

from __future__ import annotations

import argparse
from pathlib import Path

from acquisition.fetch_seaad_omics_inventory import find_project_root
from omics.pseudobulk_inspection import inspect_selection, print_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Default: data/derivatives/sea-ad/omics_inspection",
    )
    args = parser.parse_args(argv)
    root = find_project_root(Path.cwd())
    output = args.output_dir or root / "data/derivatives/sea-ad/omics_inspection"
    output = output.resolve()
    summaries = inspect_selection(args.selection.resolve(), root, output)
    print_report(
        summaries,
        output / "pseudobulk_roi_coverage.csv",
        output / "pseudobulk_taxonomy.csv",
        output / "pseudobulk_roi_supertype_donor_counts.csv",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
