"""Normalize prepared SEA-AD pseudobulk rows and score predefined state signatures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from acquisition.fetch_seaad_omics_inventory import find_project_root
from omics.signature_registry import OUTPUT_RELATIVE
from omics.state_scoring import CPM_PRIOR_COUNT, DETECTION_FRACTION, LINEAGES, score_selection


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--prepared-dir", type=Path)
    parser.add_argument("--lineage", action="append", choices=LINEAGES)
    parser.add_argument("--detection-fraction", type=float, default=DETECTION_FRACTION)
    parser.add_argument("--prior-count", type=float, default=CPM_PRIOR_COUNT)
    args = parser.parse_args(argv)

    root = find_project_root(Path.cwd())
    output_dir = args.output_dir.resolve() if args.output_dir else root / OUTPUT_RELATIVE
    provenance = score_selection(
        root,
        output_dir / "signature_registry.json",
        output_dir / "signature_genes.csv",
        output_dir,
        lineages=tuple(args.lineage) if args.lineage else LINEAGES,
        prepared_dir=args.prepared_dir.resolve() if args.prepared_dir else None,
        detection_fraction=args.detection_fraction,
        prior_count=args.prior_count,
    )
    print(json.dumps({"output_dir": str(output_dir), "counts": provenance["counts"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
