from __future__ import annotations

import csv
import io
import json
from dataclasses import replace
from pathlib import Path

import pytest

from acquisition.s3_download import (
    DownloadError,
    InventoryObject,
    PlannedDownload,
    SelectionError,
    SelectionItem,
    SelectionManifest,
    DEFAULT_BUCKET,
    download_one,
    load_inventory,
    load_selection,
    resolve_selection,
    run_download,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SELECTION_PATH = (
    Path(__file__).resolve().parents[1]
    / "acquisition/selections/glial_pseudobulk_multiregion_2026.json"
)


def item(key="Multiregion_2026/pseudobulk_objects/SEAAD_Test_RNAseq_final-nuclei_pseudobulked.2026-06-22.h5ad", size=4):
    return SelectionItem(
        release="Multiregion 2026 (2026-06-22)",
        object_family="subclass pseudobulk",
        lineage="Test",
        s3_key=key,
        size_bytes=size,
        local_path=f"data/raw/sea-ad/multiregion_2026/pseudobulk_objects/{Path(key).name}",
    )


def selection(selected: SelectionItem) -> SelectionManifest:
    return SelectionManifest(
        source_bucket=DEFAULT_BUCKET,
        inventory_manifest="inventory.csv",
        provenance_path="provenance.json",
        items=(selected,),
    )


def write_inventory(path: Path, rows: list[tuple[str, int]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["key", "size_bytes", "last_modified"])
        for key, size in rows:
            writer.writerow([key, size, "2026-01-01T00:00:00Z"])


def write_selection(path: Path, selected: SelectionItem) -> None:
    path.write_text(
        json.dumps(
            {
                "source_bucket": DEFAULT_BUCKET,
                "inventory_manifest": "inventory.csv",
                "provenance_path": "provenance.json",
                "items": [
                    {
                        "release": selected.release,
                        "object_family": selected.object_family,
                        "lineage": selected.lineage,
                        "s3_key": selected.s3_key,
                        "size_bytes": selected.size_bytes,
                        "local_path": selected.local_path,
                    }
                ],
            }
        )
    )


def test_parse_synthetic_inventory_and_selection(tmp_path):
    selected = item()
    inventory_path = tmp_path / "inventory.csv"
    selection_path = tmp_path / "selection.json"
    write_inventory(inventory_path, [(selected.s3_key, selected.size_bytes)])
    write_selection(selection_path, selected)
    inventory = load_inventory(inventory_path)
    parsed = load_selection(selection_path)
    assert inventory == [InventoryObject(selected.s3_key, 4, "2026-01-01T00:00:00Z")]
    assert parsed.items == (selected,)


def test_resolve_exact_selected_object(tmp_path):
    selected = item()
    plan = resolve_selection(selection(selected), [InventoryObject(selected.s3_key, 4)], tmp_path)
    assert len(plan) == 1
    assert plan[0].action == "download"
    assert plan[0].destination == tmp_path / selected.local_path


def test_resolve_fails_on_zero_matches(tmp_path):
    selected = item()
    with pytest.raises(SelectionError, match="found 0"):
        resolve_selection(selection(selected), [], tmp_path)


def test_resolve_fails_on_multiple_matches(tmp_path):
    selected = item()
    duplicate = InventoryObject(selected.s3_key, selected.size_bytes)
    with pytest.raises(SelectionError, match="found 2"):
        resolve_selection(selection(selected), [duplicate, duplicate], tmp_path)


def test_dry_run_does_not_create_destination(tmp_path):
    selected = item()
    inventory_path = tmp_path / "inventory.csv"
    selection_path = tmp_path / "selection.json"
    write_inventory(inventory_path, [(selected.s3_key, selected.size_bytes)])
    write_selection(selection_path, selected)
    plan = run_download(selection_path, tmp_path, dry_run=True)
    assert plan[0].action == "download"
    assert not (tmp_path / selected.local_path).exists()
    assert not (tmp_path / "provenance.json").exists()


def test_matching_existing_file_is_size_verified_and_skipped(tmp_path):
    selected = item(size=4)
    destination = tmp_path / selected.local_path
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"data")
    plan = resolve_selection(
        selection(selected), [InventoryObject(selected.s3_key, 4)], tmp_path
    )
    assert plan[0].action == "skip_verified"


def test_refuses_to_overwrite_conflicting_file(tmp_path):
    selected = item(size=4)
    destination = tmp_path / selected.local_path
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"wrong")
    inventory_path = tmp_path / "inventory.csv"
    selection_path = tmp_path / "selection.json"
    write_inventory(inventory_path, [(selected.s3_key, selected.size_bytes)])
    write_selection(selection_path, selected)
    with pytest.raises(DownloadError, match="Refusing to overwrite"):
        run_download(selection_path, tmp_path)
    assert destination.read_bytes() == b"wrong"


class FakeResponse(io.BytesIO):
    def __init__(self, data: bytes, header_size: int):
        super().__init__(data)
        self.headers = {"Content-Length": str(header_size)}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def test_download_verifies_final_size_and_removes_partial(tmp_path):
    selected = item(size=4)
    planned = PlannedDownload(selected, tmp_path / selected.local_path, "download")
    download_one(planned, "https://example.invalid", opener=lambda *a, **k: FakeResponse(b"data", 4))
    assert planned.destination.read_bytes() == b"data"

    bad = replace(planned, destination=planned.destination.with_name("bad.h5ad"))
    with pytest.raises(DownloadError, match="Incomplete transfer"):
        download_one(bad, "https://example.invalid", opener=lambda *a, **k: FakeResponse(b"bad", 4))
    assert not bad.destination.exists()
    assert not bad.destination.with_name("bad.h5ad.part").exists()


def test_reviewed_glial_selection_has_exactly_four_expected_items():
    parsed = load_selection(SELECTION_PATH)
    assert [value.lineage for value in parsed.items] == [
        "Immune",
        "Astrocyte",
        "Oligodendrocyte",
        "OPC",
    ]
    assert sum(value.size_bytes for value in parsed.items) == 523_665_711
    inventory = load_inventory(
        PROJECT_ROOT / "data/derivatives/sea-ad/omics_inventory/s3_object_manifest.csv"
    )
    plan = resolve_selection(parsed, inventory, PROJECT_ROOT)
    assert len(plan) == 4


def test_rejects_large_region_level_anndata(tmp_path):
    key = "MTG/RNAseq/SEAAD_MTG_RNAseq_final-nuclei.2026-06-22.h5ad"
    selected = item(key=key, size=398_000_000_000)
    with pytest.raises(SelectionError, match="not a Multiregion 2026 pseudobulk"):
        resolve_selection(selection(selected), [InventoryObject(key, selected.size_bytes)], tmp_path)
