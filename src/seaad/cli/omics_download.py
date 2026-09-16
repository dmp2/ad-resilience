"""Download a reviewed set of SEA-AD Multiregion 2026 pseudobulk objects."""

from __future__ import annotations

import argparse
from pathlib import Path

from acquisition.fetch_seaad_omics_inventory import find_project_root
from acquisition.s3_download import run_download


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--inventory", type=Path, help="Override the inventory CSV in the selection")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    root = find_project_root(Path.cwd())
    run_download(
        args.selection.resolve(),
        root,
        inventory_path=args.inventory.resolve() if args.inventory else None,
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
