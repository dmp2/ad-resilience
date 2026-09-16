"""Resolve the predefined glial state signatures from MSigDB and write the registry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from acquisition.fetch_seaad_omics_inventory import find_project_root
from omics.signature_registry import OUTPUT_RELATIVE, build_registry


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, help="Defaults to the omics_state derivative directory")
    parser.add_argument("--cache-dir", type=Path, help="Where the MSigDB GMT files are kept")
    args = parser.parse_args(argv)
    root = find_project_root(Path.cwd())
    output_dir = args.output_dir.resolve() if args.output_dir else root / OUTPUT_RELATIVE
    cache_dir = args.cache_dir.resolve() if args.cache_dir else output_dir / "msigdb"
    registry = build_registry(output_dir, cache_dir)
    resolved = sum(1 for s in registry["signatures"] if s["resolution_status"] == "resolved")
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "n_signatures": len(registry["signatures"]),
                "n_resolved": resolved,
                "n_unresolved": sum(
                    1 for s in registry["signatures"] if s["resolution_status"] == "unresolved"
                ),
                "msigdb_release": registry["provenance"]["msigdb_release"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
