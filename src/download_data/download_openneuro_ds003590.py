#!/usr/bin/env python3
"""Clone and retrieve OpenNeuro dataset ds003590 at an exact snapshot.

The default retrieves the files most useful for working with the corrected
3D reconstruction:

* the source 7T MRI;
* derivatives/historecon-revision-1;
* derivatives/mni-mapping.

DataLad/git-annex provide resumable, content-addressed downloads. Rerunning the
script retrieves only missing content and verifies downloaded annex objects.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

DATASET_ID = "ds003590"
DEFAULT_VERSION = "1.0.2"
REPOSITORY = "https://github.com/OpenNeuroDatasets/ds003590.git"

CONTENT_PATHS: dict[str, tuple[str, ...]] = {
    "metadata": (),
    "revised": (
        "sub-01/ses-7T/anat",
        "derivatives/historecon-revision-1",
        "derivatives/mni-mapping",
    ),
    "all-recon": (
        "sub-01/ses-7T/anat",
        "derivatives/historecon",
        "derivatives/historecon-revision-1",
        "derivatives/mni-mapping",
    ),
    "all": (".",),
}


class DownloadError(RuntimeError):
    """Raised for a failed or unsafe dataset operation."""


def run(
    command: Sequence[str],
    *,
    cwd: Path | None = None,
    capture: bool = False,
) -> str:
    """Run a command, raising a readable error on failure."""

    print("+", " ".join(str(part) for part in command), flush=True)
    try:
        result = subprocess.run(
            list(command),
            cwd=cwd,
            check=True,
            text=True,
            stdout=subprocess.PIPE if capture else None,
            stderr=subprocess.PIPE if capture else None,
        )
    except subprocess.CalledProcessError as exc:
        details = (exc.stderr or exc.stdout or "").strip()
        message = f"Command failed with exit code {exc.returncode}: {' '.join(command)}"
        if details:
            message += f"\n{details}"
        raise DownloadError(message) from exc
    return result.stdout.strip() if capture and result.stdout else ""


def require_program(name: str, install_hint: str) -> None:
    if shutil.which(name) is None:
        raise DownloadError(f"Required program not found: {name}\n{install_hint}")


def validate_relative_path(value: str) -> str:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise argparse.ArgumentTypeError(
            f"Dataset paths must be relative and may not contain '..': {value!r}"
        )
    return value


def ensure_clone(destination: Path) -> None:
    """Clone the dataset if absent; otherwise validate the existing clone."""

    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        run(["datalad", "clone", REPOSITORY, str(destination)])
        return

    if not (destination / ".git").exists():
        raise DownloadError(
            f"Destination exists but is not a Git/DataLad dataset: {destination}"
        )

    origin = run(
        ["git", "remote", "get-url", "origin"], cwd=destination, capture=True
    )
    if DATASET_ID not in origin:
        raise DownloadError(
            f"Destination points to an unexpected Git remote: {origin}\n"
            f"Expected a {DATASET_ID} clone."
        )


def ensure_clean_worktree(destination: Path) -> None:
    dirty = run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=destination,
        capture=True,
    )
    if dirty:
        raise DownloadError(
            "The dataset working tree contains local changes. Keep analysis outputs "
            "outside the downloaded dataset, or save/remove these changes before "
            f"rerunning.\n\n{dirty}"
        )


def checkout_snapshot(destination: Path, version: str) -> str:
    """Fetch tags, detach at the requested snapshot, and return its commit hash."""

    run(["git", "fetch", "--tags", "--force", "origin"], cwd=destination)

    tag_ref = f"refs/tags/{version}^{{commit}}"
    try:
        expected_commit = run(
            ["git", "rev-parse", "--verify", tag_ref],
            cwd=destination,
            capture=True,
        )
    except DownloadError as exc:
        raise DownloadError(
            f"Snapshot tag {version!r} was not found for {DATASET_ID}."
        ) from exc

    current_commit = run(["git", "rev-parse", "HEAD"], cwd=destination, capture=True)
    if current_commit != expected_commit:
        run(["git", "checkout", "--detach", version], cwd=destination)

    actual_commit = run(["git", "rev-parse", "HEAD"], cwd=destination, capture=True)
    if actual_commit != expected_commit:
        raise DownloadError(
            f"Snapshot verification failed: HEAD is {actual_commit}, but tag "
            f"{version} resolves to {expected_commit}."
        )
    return actual_commit


def retrieve_content(destination: Path, paths: Sequence[str]) -> None:
    if not paths:
        print("Metadata-only clone requested; no annexed file content retrieved.")
        return
    run(["datalad", "get", "-r", *paths], cwd=destination)


def verify_content(destination: Path, paths: Sequence[str]) -> None:
    """Recompute git-annex checksums for the selected, locally present content."""

    if not paths:
        print("No annexed content selected for verification.")
        return
    run(["git", "annex", "fsck", "--", *paths], cwd=destination)


def tool_version(command: Sequence[str]) -> str:
    try:
        output = run(command, capture=True)
    except DownloadError:
        return "unknown"
    return output.splitlines()[0] if output else "unknown"


def write_record(
    destination: Path,
    *,
    version: str,
    commit: str,
    content: str,
    paths: Sequence[str],
    verified: bool,
) -> Path:
    """Write a small provenance record beside, not inside, the dataset."""

    record = {
        "dataset_id": DATASET_ID,
        "snapshot": version,
        "git_commit": commit,
        "repository": REPOSITORY,
        "destination": str(destination),
        "content_selection": content,
        "retrieved_paths": list(paths),
        "git_annex_fsck_run": verified,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "tools": {
            "git": tool_version(["git", "--version"]),
            "git_annex": tool_version(["git-annex", "version"]),
            "datalad": tool_version(["datalad", "--version"]),
        },
    }
    record_path = destination.parent / f"{destination.name}.download.json"
    temporary = record_path.with_suffix(record_path.suffix + ".tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    temporary.replace(record_path)
    return record_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Retrieve OpenNeuro ds003590 at a version-pinned DataLad snapshot."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("ds003590"),
        help="Destination dataset directory (default: ./ds003590).",
    )
    parser.add_argument(
        "--version",
        default=DEFAULT_VERSION,
        help=f"OpenNeuro snapshot tag (default: {DEFAULT_VERSION}).",
    )
    parser.add_argument(
        "--content",
        choices=tuple(CONTENT_PATHS),
        default="revised",
        help=(
            "Content to retrieve: metadata only; revised reconstruction plus MRI/MNI "
            "mapping; both reconstruction versions; or the entire dataset "
            "(default: revised)."
        ),
    )
    parser.add_argument(
        "--path",
        action="append",
        type=validate_relative_path,
        default=[],
        help=(
            "Retrieve a specific dataset-relative path. May be repeated. When used, "
            "it overrides --content."
        ),
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run git-annex fsck on selected content after retrieval.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    require_program(
        "git",
        "Install Git with your operating system package manager.",
    )
    require_program(
        "git-annex",
        "Install DataLad and git-annex, for example with: "
        "mamba install -c conda-forge datalad git-annex",
    )
    require_program(
        "datalad",
        "Install DataLad and git-annex, for example with: "
        "mamba install -c conda-forge datalad git-annex",
    )

    destination = args.data_dir.expanduser().resolve()
    paths = tuple(args.path) if args.path else CONTENT_PATHS[args.content]
    selection_name = "custom" if args.path else args.content

    ensure_clone(destination)
    ensure_clean_worktree(destination)
    commit = checkout_snapshot(destination, args.version)
    retrieve_content(destination, paths)

    if args.verify:
        verify_content(destination, paths)

    record_path = write_record(
        destination,
        version=args.version,
        commit=commit,
        content=selection_name,
        paths=paths,
        verified=args.verify,
    )

    print(f"\nDataset ready: {destination}")
    print(f"Snapshot: {args.version} ({commit})")
    print(f"Download record: {record_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except DownloadError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
