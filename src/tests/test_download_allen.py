from __future__ import annotations

import argparse
import fcntl
import hashlib
import io
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from scipy.io import savemat

import download_allen as allen


class FakeResponse:
    def __init__(self, content=b"", payload=None):
        self.content = content
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class ImageSession:
    def __init__(self, content):
        self.content = content
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return FakeResponse(self.content)


def jpeg_bytes(color=(10, 20, 30), size=(4, 3)):
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, "JPEG", quality=100, subsampling=0)
    return buffer.getvalue()


def section(series="nissl", section_id=101, number=7):
    treatment = {"nissl": (3, "NISSL"), "pv": (16, "IHC:Parvalbumin")}[series]
    return allen.Section(
        series,
        section_id,
        number,
        *treatment,
        900 + treatment[0],
        allen.SPECIMEN_ID,
        allen.DONOR_ID,
    )


def test_treatment_inventory_classification_and_smi32_vocabulary(monkeypatch):
    replies = iter(
        [
            {
                "msg": [
                    {"id": 3, "name": "NISSL"},
                    {"id": 16, "name": "IHC:Parvalbumin"},
                    {"id": 5, "name": "IHC:SMI-32"},
                ]
            },
            {
                "msg": [
                    {
                        "id": 11,
                        "specimen": {"donor": {"id": allen.DONOR_ID}},
                        "treatments": [{"id": 3, "name": "NISSL"}],
                        "section_images": [{"id": 101, "section_number": 7}],
                    },
                    {
                        "id": 12,
                        "specimen": {"donor": {"id": allen.DONOR_ID}},
                        "treatments": [{"id": 16, "name": "IHC:Parvalbumin"}],
                        "section_images": [{"id": 201, "section_number": 8}],
                    },
                ]
            },
        ]
    )
    monkeypatch.setattr(allen, "api_json", lambda session, criteria: next(replies))
    records, inventory = allen.AllenSectionDataSetProvider(object()).discover(
        allen.SPECIMEN_ID
    )
    assert allen.classify_treatment("NISSL") == "nissl"
    assert allen.classify_treatment("IHC:Parvalbumin") == "pv"
    assert allen.classify_treatment("IHC:SMI-32") == "smi32"
    assert [len(records[name]) for name in allen.SERIES_LABELS] == [1, 1, 0]
    assert inventory["series"]["smi32"] == {
        "treatment_id": 5,
        "treatment_name": "IHC:SMI-32",
        "data_set_ids": [],
        "image_count": 0,
    }


def changed_inventory():
    series = {key: dict(value) for key, value in allen.BASELINE_SERIES.items()}
    series["nissl"]["image_count"] = 642
    return allen.inventory_payload(
        allen.SPECIMEN_ID, allen.DONOR_ID, series, "2026-07-29T00:00:00Z"
    )


def test_inventory_change_report_and_digest_acceptance(tmp_path):
    observed = changed_inventory()
    report = tmp_path / "observed.json"
    with pytest.raises(allen.APIInventoryChanged):
        allen.gate_inventory(allen.baseline_inventory(), observed, report, None)
    assert (
        json.loads(report.read_text())["inventory_sha256"]
        == observed["inventory_sha256"]
    )
    allen.gate_inventory(
        allen.baseline_inventory(), observed, None, observed["inventory_sha256"]
    )
    with pytest.raises(RuntimeError, match="does not match"):
        allen.gate_inventory(allen.baseline_inventory(), observed, None, "0" * 64)


def test_inventory_change_happens_before_dataset_mutation(tmp_path, monkeypatch):
    data = tmp_path / "not-created"
    records = {name: [] for name in allen.SERIES_LABELS}
    monkeypatch.setattr(allen, "build_session", lambda **kwargs: object())
    monkeypatch.setattr(
        allen.AllenSectionDataSetProvider,
        "discover",
        lambda self, specimen_id: (records, changed_inventory()),
    )
    assert allen.main(["--data-dir", str(data)]) == 3
    assert not data.exists()


def test_series_defaults_and_dated_smi32_unavailability():
    records = {"nissl": [section()], "pv": [section("pv")], "smi32": []}
    args = argparse.Namespace(series=None, stains=None)
    assert allen.selected_series(args, records) == ["nissl", "pv"]
    args.series = ["smi32"]
    with pytest.raises(RuntimeError, match="2026-07-28"):
        allen.selected_series(args, records)


def test_mapping_resolution_repair_fallback_and_conflicts():
    nissl = [section(section_id=101, number=7), section(section_id=102, number=9)]
    plates = [allen.AtlasPlate(101, 7), allen.AtlasPlate(500, 9)]
    mapped = allen.resolve_mappings(plates, nissl)
    assert [(item.nissl_id, item.mapping_status) for item in mapped] == [
        (101, "exact_id"),
        (102, "repaired_unique_section"),
    ]
    fallback = allen.resolve_mappings(
        [allen.AtlasPlate(600, 12)], nissl, filenames={12}
    )
    assert fallback[0].mapping_status == "filename_only"
    with pytest.raises(RuntimeError, match="Conflicting"):
        allen.resolve_mappings([allen.AtlasPlate(101, 7)], nissl, cached={101: 102})
    duplicate = [section(section_id=103, number=11), section(section_id=104, number=11)]
    with pytest.raises(RuntimeError, match="Multiple"):
        allen.resolve_mappings([allen.AtlasPlate(600, 11)], duplicate)


def ontology_payload(structure_id=1):
    return {
        "success": True,
        "msg": [
            {
                "id": structure_id,
                "acronym": "ROOT",
                "name": "root",
                "parent_structure_id": None,
                "color_hex_triplet": "ABCDEF",
                "structure_id_path": f"/{structure_id}/",
                "children": [
                    {
                        "id": structure_id + 1,
                        "acronym": "CH",
                        "name": "child",
                        "parent_structure_id": structure_id,
                        "color_hex_triplet": "010203",
                        "structure_id_path": f"/{structure_id}/{structure_id + 1}/",
                        "children": [],
                    }
                ],
            }
        ],
    }


def svg_bytes(structure_id=1, groups=(31,)):
    body = "".join(
        f'<g graphic_group_label_id="{gid}" graphic_group_label="{allen.GROUP_LABELS[gid]}"><path structure_id="{structure_id}"/></g>'
        for gid in groups
    )
    return f'<svg xmlns="http://www.w3.org/2000/svg">{body}</svg>'.encode()


def test_ontology_flattening_and_svg_group_inventory():
    structures = allen.flatten_ontology(ontology_payload())
    assert [item.structure_id for item in structures] == [1, 2]
    assert all(item.structure_graph_id == 16 for item in structures)
    assert structures[1].parent_structure_id == 1
    present, counts, ids = allen.inspect_svg(svg_bytes(2, (31, 141667008)))
    assert present == [31, 141667008]
    assert counts == {31: 1, 141667008: 1}
    assert ids[31] == {2}


def test_manifest_semicolon_groups_and_excludes_metadata(tmp_path):
    path = tmp_path / "manifest.tsv"
    artifact = allen.Artifact(
        "annotation_svg",
        "atlas",
        7,
        None,
        101,
        "nissl/labels_orig/seg_0007.svg",
        None,
        None,
        None,
        "a" * 64,
        None,
        "downloaded",
        "test",
        graphic_groups_present="31;141667008",
        matching_nissl_section_image_id=101,
        mapping_status="exact_id",
    )
    allen.save_manifest(path, {artifact.path: artifact})
    text = path.read_text()
    assert "31;141667008" in text
    assert "{" not in text
    assert "metadata/dataset.json" not in text
    assert allen.load_manifest(path)[artifact.path] == artifact


def test_resume_adopts_identical_and_quarantines_mismatch(tmp_path):
    paths = allen.Paths.make(tmp_path)
    paths.create(["nissl"])
    item = section()
    target = allen.image_path(paths, item)
    content = jpeg_bytes()
    target.write_bytes(content)
    adopted = allen.acquire_image(
        ImageSession(content), paths, item, {}, 5, "allen-direct", False, "run"
    )
    assert adopted.status == "verified-existing"
    old = jpeg_bytes((200, 1, 1))
    new = jpeg_bytes((1, 200, 1))
    target.write_bytes(old)
    replaced = allen.acquire_image(
        ImageSession(new), paths, item, {}, 5, "allen-direct", False, "run2"
    )
    assert replaced.status == "redownloaded-after-quarantine"
    assert target.read_bytes() == new
    assert (
        tmp_path / "quarantine/run2/nissl/images_orig/image_0007.jpg"
    ).read_bytes() == old


def tree_state(root):
    return {
        path.relative_to(root).as_posix(): (
            path.stat().st_mtime_ns,
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        for path in root.rglob("*")
        if path.is_file()
    }


def make_valid_fixture(root, monkeypatch):
    monkeypatch.setattr(allen, "EXPECTED_SVGS", 1)
    paths = allen.Paths.make(root)
    paths.create(["nissl"])
    image_path = paths.image_dir("nissl") / "image_0007.jpg"
    image_path.write_bytes(jpeg_bytes())
    svg_path = paths.svg_dir / "seg_0007.svg"
    svg_path.write_bytes(svg_bytes(1))
    ontology = json.dumps(ontology_payload()).encode()
    paths.ontology.write_bytes(ontology)
    structures = allen.flatten_ontology(ontology_payload())
    allen.save_structures(paths.structures, structures)
    inventory_series = {
        "nissl": {
            "treatment_id": 3,
            "treatment_name": "NISSL",
            "data_set_ids": [11],
            "image_count": 1,
        },
        "pv": {
            "treatment_id": 16,
            "treatment_name": "IHC:Parvalbumin",
            "data_set_ids": [],
            "image_count": 0,
        },
        "smi32": {
            "treatment_id": 5,
            "treatment_name": "IHC:SMI-32",
            "data_set_ids": [],
            "image_count": 0,
        },
    }
    inventory = allen.inventory_payload(
        allen.SPECIMEN_ID, allen.DONOR_ID, inventory_series, "test"
    )
    allen.atomic_json(
        paths.dataset,
        {
            "allen_specimen_id": allen.SPECIMEN_ID,
            "allen_atlas_id": allen.ATLAS_ID,
            "allen_structure_graph_id": allen.GRAPH_ID,
            "graphic_groups": [
                {"id": gid, "name": allen.GROUP_LABELS[gid]} for gid in allen.GROUPS
            ],
            "accepted_api_inventory": inventory,
        },
    )
    artifacts = {
        "nissl/images_orig/image_0007.jpg": allen.Artifact(
            "histology_jpeg",
            "nissl",
            7,
            101,
            None,
            "nissl/images_orig/image_0007.jpg",
            4,
            3,
            32.0,
            allen.sha_file(image_path),
            "source",
            "downloaded",
            "test",
            11,
            3,
        ),
        "nissl/labels_orig/seg_0007.svg": allen.Artifact(
            "annotation_svg",
            "atlas",
            7,
            None,
            101,
            "nissl/labels_orig/seg_0007.svg",
            None,
            None,
            None,
            allen.sha_file(svg_path),
            "source",
            "downloaded",
            "test",
            graphic_groups_present="31",
            matching_nissl_section_image_id=101,
            mapping_status="exact_id",
        ),
        "ontology/structure_graph_16.json": allen.Artifact(
            "ontology_json",
            "structure_graph_16",
            None,
            None,
            None,
            "ontology/structure_graph_16.json",
            None,
            None,
            None,
            allen.sha_file(paths.ontology),
            "source",
            "downloaded",
            "test",
        ),
    }
    allen.save_manifest(paths.manifest, artifacts)
    allen.atomic_json(
        paths.secjson,
        {"nissl": [{"section_id": 101}], "ihc": [], "atlas_annotations": [{}]},
    )
    savemat(
        paths.secmat,
        {
            "nissl_section_id": np.array([101]),
            "ihc_section_id": np.array([], dtype=int),
            "atlas_image_id": np.array([101]),
        },
    )
    return paths


def test_read_only_validation_and_unresolved_id_failure(tmp_path, monkeypatch):
    paths = make_valid_fixture(tmp_path, monkeypatch)
    before = tree_state(tmp_path)
    report = allen.validate_dataset(paths)
    after = tree_state(tmp_path)
    assert report["status"] == "PASS"
    assert before == after
    svg_path = paths.svg_dir / "seg_0007.svg"
    svg_path.write_bytes(svg_bytes(999))
    manifest = allen.load_manifest(paths.manifest)
    item = manifest["nissl/labels_orig/seg_0007.svg"]
    manifest[item.path] = allen.Artifact(
        **{**allen.asdict(item), "sha256": allen.sha_file(svg_path)}
    )
    allen.save_manifest(paths.manifest, manifest)
    report = allen.validate_dataset(paths)
    assert report["status"] == "FAIL"
    assert any(
        item["category"] == "unresolved_structure_id" for item in report["failures"]
    )
    assert {item.structure_id for item in allen.load_structures(paths.structures)} == {
        1,
        2,
    }


def test_metadata_migration_and_limit_inventory_stability(tmp_path):
    paths = allen.Paths.make(tmp_path)
    paths.metadata.mkdir(parents=True)
    for name in ("sections.tsv", "atlas_annotations.tsv", "image_files.tsv"):
        (paths.metadata / name).write_text(name)
    for name in ("dataset.json", "manifest.tsv", "structures.tsv"):
        (paths.metadata / name).write_text(name)
    allen.migrate_superseded_metadata(paths, "run")
    assert {item.name for item in paths.metadata.iterdir()} == {
        "dataset.json",
        "manifest.tsv",
        "structures.tsv",
    }
    assert (tmp_path / "quarantine/run/superseded_metadata/sections.tsv").is_file()
    payload = allen.dataset_payload(
        allen.baseline_inventory(),
        {"structure_graph_name": "Developing Human Brain Atlas", "graphic_groups": []},
        "allen-direct",
        5,
        {},
    )
    assert payload["accepted_api_inventory"]["series"]["nissl"]["image_count"] == 641
    assert payload["published_corpus"]["counts"]["smi32"] == 338


def test_shell_wrapper_syntax_and_safety_markers():
    root = Path(__file__).resolve().parents[2]
    wrapper = root / "scripts/download_allen_sections.sh"
    subprocess.run(["bash", "-n", str(wrapper)], check=True)
    text = wrapper.read_text()
    assert "--skip-masks" in text
    assert "--stains" not in text
    assert "\n    --accept-api-inventory-sha256" not in text
    assert "flock -n" in text
    assert ".tmp.$$" in text


def wrapper_environment(tmp_path, downloader):
    return {
        **os.environ,
        "PROJECT_ROOT": str(Path(__file__).resolve().parents[2]),
        "DATA_DIR": str(tmp_path / "data"),
        "DOWNLOADER": str(downloader),
        "PYTHON_BIN": "/usr/bin/python3",
        "MIN_FREE_GB": "0",
        "LOG_DIR": str(tmp_path / "logs"),
        "STATUS_DIR": str(tmp_path / "status"),
        "LOCK_FILE": str(tmp_path / "status/writer.lock"),
    }


def test_shell_wrapper_preflight_and_atomic_status(tmp_path):
    root = Path(__file__).resolve().parents[2]
    wrapper = root / "scripts/download_allen_sections.sh"
    environment = wrapper_environment(
        tmp_path, root / "src/download_data/download_allen.py"
    )
    result = subprocess.run(
        ["bash", str(wrapper), "--preflight-only"],
        env=environment,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    statuses = list((tmp_path / "status").glob("*.status"))
    assert len(statuses) == 1
    assert "exit_code=0" in statuses[0].read_text()
    assert not list((tmp_path / "status").glob("*.tmp.*"))
    assert not (tmp_path / "data").exists()


def test_shell_wrapper_lock_and_inventory_change_propagation(tmp_path):
    root = Path(__file__).resolve().parents[2]
    wrapper = root / "scripts/download_allen_sections.sh"
    status_dir = tmp_path / "status"
    status_dir.mkdir()
    lock_path = status_dir / "writer.lock"
    environment = wrapper_environment(
        tmp_path, root / "src/download_data/download_allen.py"
    )
    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        locked = subprocess.run(
            ["bash", str(wrapper), "--preflight-only"],
            env=environment,
            capture_output=True,
        )
    assert locked.returncode == 75

    fake = tmp_path / "inventory_changed.py"
    fake.write_text("raise SystemExit(3)\n")
    environment = wrapper_environment(tmp_path, fake)
    changed = subprocess.run(
        ["bash", str(wrapper)], env=environment, text=True, capture_output=True
    )
    assert changed.returncode == 3
    assert "API_INVENTORY_CHANGED" in changed.stdout + changed.stderr
    statuses = list(status_dir.glob("*.status"))
    assert len(statuses) == 1
    assert "exit_code=3" in statuses[0].read_text()
    assert not list(status_dir.glob("*.tmp.*"))
