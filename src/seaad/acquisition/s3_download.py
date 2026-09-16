"""Safe, exact-key downloads from a reviewed SEA-AD selection manifest."""

from __future__ import annotations

import csv
import json
import os
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO, Callable, Iterable


DEFAULT_BUCKET = "https://sea-ad-single-cell-profiling.s3.us-west-2.amazonaws.com/"
PSEUDOBULK_PREFIX = "Multiregion_2026/pseudobulk_objects/"
PSEUDOBULK_SUFFIX = "_RNAseq_final-nuclei_pseudobulked.2026-06-22.h5ad"
RAW_ROOT = Path("data/raw/sea-ad/multiregion_2026")
MAX_SELECTION_ITEMS = 10
MAX_PSEUDOBULK_BYTES = 1_000_000_000


class SelectionError(ValueError):
    """A selection is absent, ambiguous, or outside the allowed data family."""


class DownloadError(RuntimeError):
    """A planned transfer could not be completed or verified."""


@dataclass(frozen=True)
class InventoryObject:
    key: str
    size_bytes: int
    last_modified: str = ""


@dataclass(frozen=True)
class SelectionItem:
    release: str
    object_family: str
    lineage: str
    s3_key: str
    size_bytes: int
    local_path: str


@dataclass(frozen=True)
class SelectionManifest:
    source_bucket: str
    inventory_manifest: str
    provenance_path: str
    items: tuple[SelectionItem, ...]


@dataclass(frozen=True)
class PlannedDownload:
    item: SelectionItem
    destination: Path
    action: str


def load_inventory(path: Path) -> list[InventoryObject]:
    """Read the existing exact-key CSV inventory."""
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"key", "size_bytes"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise SelectionError(
                f"Inventory {path} must contain columns: {sorted(required)}"
            )
        rows = []
        for row_number, row in enumerate(reader, start=2):
            try:
                size = int(row["size_bytes"])
            except (TypeError, ValueError) as exc:
                raise SelectionError(
                    f"Invalid size_bytes at {path}:{row_number}"
                ) from exc
            rows.append(
                InventoryObject(
                    key=row["key"],
                    size_bytes=size,
                    last_modified=row.get("last_modified", ""),
                )
            )
    return rows


def load_selection(path: Path) -> SelectionManifest:
    """Parse the deliberately small JSON selection format."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SelectionError(f"Cannot parse selection {path}: {exc}") from exc
    required_top = {"source_bucket", "inventory_manifest", "provenance_path", "items"}
    missing = required_top - raw.keys()
    if missing:
        raise SelectionError(f"Selection {path} is missing: {sorted(missing)}")
    if not isinstance(raw["items"], list) or not raw["items"]:
        raise SelectionError("Selection items must be a non-empty list")
    fields = set(SelectionItem.__dataclass_fields__)
    items = []
    for number, value in enumerate(raw["items"], start=1):
        missing_item = fields - value.keys() if isinstance(value, dict) else fields
        if missing_item:
            raise SelectionError(f"Selection item {number} is missing: {sorted(missing_item)}")
        try:
            items.append(
                SelectionItem(
                    release=str(value["release"]),
                    object_family=str(value["object_family"]),
                    lineage=str(value["lineage"]),
                    s3_key=str(value["s3_key"]),
                    size_bytes=int(value["size_bytes"]),
                    local_path=str(value["local_path"]),
                )
            )
        except (TypeError, ValueError) as exc:
            raise SelectionError(f"Invalid selection item {number}: {exc}") from exc
    return SelectionManifest(
        source_bucket=str(raw["source_bucket"]),
        inventory_manifest=str(raw["inventory_manifest"]),
        provenance_path=str(raw["provenance_path"]),
        items=tuple(items),
    )


def _resolve_local_path(project_root: Path, relative_path: str) -> Path:
    rel = Path(relative_path)
    if rel.is_absolute() or ".." in rel.parts:
        raise SelectionError(f"local_path must be project-relative: {relative_path}")
    raw_root = (project_root / RAW_ROOT).resolve()
    destination = (project_root / rel).resolve()
    if destination != raw_root and raw_root not in destination.parents:
        raise SelectionError(f"local_path is outside {RAW_ROOT}: {relative_path}")
    return destination


def resolve_selection(
    selection: SelectionManifest,
    inventory: Iterable[InventoryObject],
    project_root: Path,
) -> list[PlannedDownload]:
    """Resolve every reviewed key to exactly one inventory record."""
    if selection.source_bucket != DEFAULT_BUCKET:
        raise SelectionError(f"Unexpected source bucket: {selection.source_bucket}")
    if len(selection.items) > MAX_SELECTION_ITEMS:
        raise SelectionError(
            f"Selection has {len(selection.items)} items; limit is {MAX_SELECTION_ITEMS}"
        )
    inventory_rows = list(inventory)
    planned = []
    seen_keys: set[str] = set()
    for item in selection.items:
        if item.s3_key in seen_keys:
            raise SelectionError(f"Duplicate selected key: {item.s3_key}")
        seen_keys.add(item.s3_key)
        if item.release != "Multiregion 2026 (2026-06-22)":
            raise SelectionError(f"Unexpected release for {item.s3_key}: {item.release}")
        if not item.s3_key.startswith(PSEUDOBULK_PREFIX) or not item.s3_key.endswith(
            PSEUDOBULK_SUFFIX
        ):
            raise SelectionError(
                f"Selected key is not a Multiregion 2026 pseudobulk object: {item.s3_key}"
            )
        if item.object_family != "subclass pseudobulk":
            raise SelectionError(f"Unexpected object_family for {item.s3_key}")
        if item.size_bytes <= 0 or item.size_bytes > MAX_PSEUDOBULK_BYTES:
            raise SelectionError(
                f"Selected object size is outside the pseudobulk safety bound: {item.s3_key}"
            )
        matches = [row for row in inventory_rows if row.key == item.s3_key]
        if len(matches) != 1:
            raise SelectionError(
                f"Expected one inventory match for {item.s3_key}; found {len(matches)}"
            )
        if matches[0].size_bytes != item.size_bytes:
            raise SelectionError(
                f"Size mismatch for {item.s3_key}: selection={item.size_bytes}, "
                f"inventory={matches[0].size_bytes}"
            )
        destination = _resolve_local_path(project_root, item.local_path)
        if destination.name != Path(item.s3_key).name:
            raise SelectionError(f"local_path filename does not match S3 key: {item.s3_key}")
        if destination.exists():
            action = (
                "skip_verified"
                if destination.is_file() and destination.stat().st_size == item.size_bytes
                else "conflict"
            )
        else:
            action = "download"
        planned.append(PlannedDownload(item=item, destination=destination, action=action))
    return planned


def print_plan(plan: Iterable[PlannedDownload]) -> None:
    rows = list(plan)
    print(f"Download plan: {len(rows)} object(s)")
    for row in rows:
        print(
            f"  [{row.action}] {row.item.lineage}: {row.item.size_bytes} bytes\n"
            f"    s3://sea-ad-single-cell-profiling/{row.item.s3_key}\n"
            f"    -> {row.destination}"
        )
    print(f"Total remote size: {sum(row.item.size_bytes for row in rows)} bytes")


def _copy_response(response: BinaryIO, destination: Path, expected_size: int) -> None:
    header_size = response.headers.get("Content-Length")
    if header_size is not None and int(header_size) != expected_size:
        raise DownloadError(
            f"Remote Content-Length {header_size} differs from expected {expected_size}"
        )
    written = 0
    with destination.open("wb") as handle:
        while True:
            block = response.read(1024 * 1024)
            if not block:
                break
            handle.write(block)
            written += len(block)
    if written != expected_size:
        raise DownloadError(
            f"Incomplete transfer: wrote {written} of {expected_size} bytes"
        )


def download_one(
    planned: PlannedDownload,
    bucket_url: str,
    opener: Callable[..., BinaryIO] = urllib.request.urlopen,
) -> None:
    """Stream one exact key to a temporary file and atomically install it."""
    if planned.action != "download":
        raise DownloadError(f"Cannot transfer plan action {planned.action}")
    destination = planned.destination
    if destination.exists():
        raise DownloadError(f"Refusing to overwrite existing file: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    if partial.exists():
        partial.unlink()
    url = bucket_url.rstrip("/") + "/" + urllib.parse.quote(planned.item.s3_key)
    try:
        with opener(url, timeout=600) as response:
            _copy_response(response, partial, planned.item.size_bytes)
        if partial.stat().st_size != planned.item.size_bytes:
            raise DownloadError(f"Final size verification failed for {partial}")
        if destination.exists():
            raise DownloadError(f"Refusing to overwrite file created during transfer: {destination}")
        os.replace(partial, destination)
    except Exception:
        partial.unlink(missing_ok=True)
        raise


def write_provenance(
    path: Path,
    selection: SelectionManifest,
    plan: Iterable[PlannedDownload],
) -> None:
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    records = []
    for row in plan:
        record = asdict(row.item)
        record.update(
            {
                "source_bucket": selection.source_bucket,
                "remote_size_bytes": row.item.size_bytes,
                "download_timestamp": timestamp,
                "verification_status": "size_verified",
            }
        )
        records.append(record)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"objects": records}, indent=2) + "\n", encoding="utf-8")


def run_download(
    selection_path: Path,
    project_root: Path,
    *,
    inventory_path: Path | None = None,
    dry_run: bool = False,
    opener: Callable[..., BinaryIO] = urllib.request.urlopen,
) -> list[PlannedDownload]:
    selection = load_selection(selection_path)
    inventory_path = inventory_path or project_root / selection.inventory_manifest
    plan = resolve_selection(selection, load_inventory(inventory_path), project_root)
    print_plan(plan)
    conflicts = [row for row in plan if row.action == "conflict"]
    if conflicts:
        paths = ", ".join(str(row.destination) for row in conflicts)
        raise DownloadError(f"Refusing to overwrite conflicting local file(s): {paths}")
    if dry_run:
        print("Dry run complete; no files were created or changed.")
        return plan
    downloaded = skipped = 0
    for row in plan:
        if row.action == "skip_verified":
            skipped += 1
            print(f"Verified existing file: {row.destination}", flush=True)
            continue
        download_one(row, selection.source_bucket, opener=opener)
        downloaded += 1
        print(f"Downloaded and size-verified: {row.destination}", flush=True)
    for row in plan:
        if not row.destination.is_file() or row.destination.stat().st_size != row.item.size_bytes:
            raise DownloadError(f"Post-download verification failed: {row.destination}")
    write_provenance(project_root / selection.provenance_path, selection, plan)
    print(f"Summary: {downloaded} downloaded, {skipped} verified and skipped, {len(plan)} total")
    return plan
