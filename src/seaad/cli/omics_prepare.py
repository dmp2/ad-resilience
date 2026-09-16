"""Prepare reviewed SEA-AD pseudobulks at donor-region-supertype grain."""

from __future__ import annotations

import argparse
from pathlib import Path

from acquisition.fetch_seaad_omics_inventory import find_project_root
from omics.pseudobulk_preparation import prepare_selection


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--force", action="store_true", help="Replace existing derived outputs")
    args = parser.parse_args(argv)
    root = find_project_root(Path.cwd())
    prepare_selection(
        args.selection.resolve(),
        root,
        output_root=args.output_root.resolve() if args.output_root else None,
        force=args.force,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
