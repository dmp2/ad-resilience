"""Extract exact reviewed genes as raw counts from one prepared SEA-AD lineage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from acquisition.fetch_seaad_omics_inventory import find_project_root
from omics.gene_extraction import LINEAGES, extract_exact_genes, prepared_path_for_lineage, read_genes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--genes", required=True, type=Path, help="One exact symbol or gene ID per line")
    parser.add_argument("--lineage", required=True, choices=LINEAGES)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--prepared-dir", type=Path)
    parser.add_argument("--omit-row-total-umi", action="store_true")
    args = parser.parse_args(argv)
    root = find_project_root(Path.cwd())
    prepared = prepared_path_for_lineage(
        root,
        args.lineage,
        args.prepared_dir.resolve() if args.prepared_dir else None,
    )
    result = extract_exact_genes(
        prepared,
        read_genes(args.genes),
        args.output.resolve(),
        include_row_total_umi=not args.omit_row_total_umi,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
