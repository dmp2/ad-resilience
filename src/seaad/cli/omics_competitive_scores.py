"""Add the matched-background (competitive) sensitivity to the state scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from acquisition.fetch_seaad_omics_inventory import find_project_root
from omics.competitive_scoring import COMPETITIVE_DOMAINS, competitive_selection
from omics.signature_registry import OUTPUT_RELATIVE
from omics.state_scoring import BACKGROUND_DRAWS, BACKGROUND_SEED, LINEAGES


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--prepared-dir", type=Path)
    parser.add_argument("--lineage", action="append", choices=LINEAGES)
    parser.add_argument("--domain", action="append", choices=list(COMPETITIVE_DOMAINS))
    parser.add_argument("--draws", type=int, default=BACKGROUND_DRAWS)
    parser.add_argument("--seed", type=int, default=BACKGROUND_SEED)
    parser.add_argument("--no-null-controls", action="store_true",
                        help="Skip the size-matched null controls")
    args = parser.parse_args(argv)

    root = find_project_root(Path.cwd())
    output_dir = args.output_dir.resolve() if args.output_dir else root / OUTPUT_RELATIVE
    provenance = competitive_selection(
        root,
        output_dir / "signature_registry.json",
        output_dir / "signature_genes.csv",
        output_dir,
        lineages=tuple(args.lineage) if args.lineage else LINEAGES,
        domains=tuple(args.domain) if args.domain else COMPETITIVE_DOMAINS,
        prepared_dir=args.prepared_dir.resolve() if args.prepared_dir else None,
        n_draws=args.draws,
        seed=args.seed,
        include_null_controls=not args.no_null_controls,
    )
    print(json.dumps({"output_dir": str(output_dir), "counts": provenance["counts"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
