from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

import audit_allen_annotations as audit


RAW = Path(__file__).resolve().parents[2] / "data/raw/allen/specimen_708424"
MANIFEST_FIELDS = (
    "kind",
    "series_or_layer",
    "section_number",
    "allen_section_image_id",
    "path",
    "sha256",
    "graphic_groups_present",
    "matching_nissl_section_image_id",
    "mapping_status",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_tsv(path: Path, fields, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def fixture_dataset(tmp_path: Path) -> Path:
    root = tmp_path / "specimen"
    metadata = root / "metadata"
    labels = root / "nissl/labels_orig"
    images = root / "nissl/images_orig"
    labels.mkdir(parents=True)
    images.mkdir(parents=True)
    dataset = {
        "allen_specimen_id": 1,
        "allen_donor_id": 2,
        "allen_atlas_id": 3,
        "allen_structure_graph_id": 16,
        "graphic_groups": [
            {"id": 31, "name": "Anatomy"},
            {"id": 32, "name": "Landmarks"},
        ],
    }
    metadata.mkdir()
    (metadata / "dataset.json").write_text(json.dumps(dataset))
    write_tsv(
        metadata / "structures.tsv",
        ("structure_id", "acronym", "name", "structure_graph_id"),
        [
            {"structure_id": 1, "acronym": "A", "name": "Area", "structure_graph_id": 16},
            {"structure_id": 2, "acronym": "B", "name": "Border", "structure_graph_id": 16},
            {"structure_id": 3, "acronym": "C", "name": "Core", "structure_graph_id": 16},
        ],
    )
    manifest = []
    for section, nissl_id, groups, paths in (
        (7, 107, "31", [(31, "Anatomy", 1), (31, "Anatomy", 2)]),
        (9, 109, "31;32", [(31, "Anatomy", 2), (32, "Landmarks", 3)]),
    ):
        image = images / f"image_{section:04d}.jpg"
        image.write_bytes(f"jpeg-{section}".encode())
        svg = labels / f"seg_{section:04d}.svg"
        body = "".join(
            f'<g graphic_group_label_id="{group_id}" graphic_group_label="{name}">'
            f'<path structure_id="{structure_id}" d="M0 0H1V1Z"/></g>'
            for group_id, name, structure_id in paths
        )
        svg.write_text(
            f'<svg xmlns="http://www.w3.org/2000/svg" width="1" height="1">{body}</svg>'
        )
        manifest.extend(
            [
                {
                    "kind": "histology_jpeg",
                    "series_or_layer": "nissl",
                    "section_number": section,
                    "allen_section_image_id": nissl_id,
                    "path": image.relative_to(root).as_posix(),
                    "sha256": sha(image),
                    "graphic_groups_present": "n/a",
                    "matching_nissl_section_image_id": "n/a",
                    "mapping_status": "n/a",
                },
                {
                    "kind": "annotation_svg",
                    "series_or_layer": "atlas",
                    "section_number": section,
                    "allen_section_image_id": "n/a",
                    "path": svg.relative_to(root).as_posix(),
                    "sha256": sha(svg),
                    "graphic_groups_present": groups,
                    "matching_nissl_section_image_id": nissl_id,
                    "mapping_status": "exact_id",
                },
            ]
        )
    write_tsv(metadata / "manifest.tsv", MANIFEST_FIELDS, manifest)
    return root


def test_audit_counts_groups_sections_and_separates_counting_units(tmp_path):
    root = fixture_dataset(tmp_path)
    result = audit.build_audit(root)
    assert len(result.sections) == 2
    assert result.polygon_count == 4
    assert result.unique_drawn_structure_count == 3
    assert result.mapped_plate_count == 2
    assert [(item.group_id, item.plate_count, item.polygon_count) for item in result.groups] == [
        (31, 2, 3),
        (32, 1, 1),
    ]
    assert [item.unique_structure_count for item in result.sections] == [2, 2]
    report = audit.format_audit(result)
    assert "Validation status: PASS" in report
    assert "Official modified-Brodmann SVG plate-series acquisition" in report
    assert "Local SVG inventory" in report
    assert "Ding et al. publication" in report
    assert "Comparability to local SVG counts" in report
    assert "Not established" in report
    assert "Publication comparison" not in report
    assert "Annotated structures / drawn IDs" not in report
    assert "Polygons / current SVG paths" not in report
    assert "Delta" not in report
    assert "Annotated sections (anterior-to-posterior manifest order)" in report
    assert "seg_0007.svg" in report
    assert "11,398" in report


def test_summary_only_omits_section_table(tmp_path, capsys):
    root = fixture_dataset(tmp_path)
    assert audit.main(["--data-dir", str(root), "--summary-only"]) == 0
    output = capsys.readouterr().out
    assert "Graphic-group breakdown" in output
    assert "Annotated sections" not in output


def test_checksum_mapping_and_unresolved_id_fail_cleanly(tmp_path):
    root = fixture_dataset(tmp_path)
    svg = root / "nissl/labels_orig/seg_0007.svg"
    svg.write_text(svg.read_text().replace('structure_id="1"', 'structure_id="99"'))
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        audit.build_audit(root)

    fields, rows = audit._read_tsv(root / "metadata/manifest.tsv")
    for row in rows:
        if row["path"].endswith("seg_0007.svg"):
            row["sha256"] = sha(svg)
    write_tsv(root / "metadata/manifest.tsv", fields, rows)
    with pytest.raises(RuntimeError, match="Unresolved structure_id 99"):
        audit.build_audit(root)


@pytest.mark.skipif(not RAW.is_dir(), reason="frozen Allen raw dataset not available")
def test_current_frozen_snapshot_audit_counts():
    result = audit.build_audit(RAW)
    assert len(result.sections) == 106
    assert result.mapped_plate_count == 106
    assert result.unique_drawn_structure_count == 801
    assert result.polygon_count == 16_491
    assert [
        (item.group_id, item.plate_count, item.polygon_count, item.unique_structure_count)
        for item in result.groups
    ] == [
        (31, 106, 6_427, 529),
        (113753816, 106, 1_235, 54),
        (141667008, 102, 1_557, 78),
        (265297118, 106, 7_272, 156),
    ]
    report = audit.format_audit(result, include_sections=False)
    assert "Annotated plates                               106 / 106" in report
    assert "Distinct drawn structure IDs                         801" in report
    assert (
        "801 distinct structure IDs were observed across the complete locally "
        "acquired official modified-Brodmann SVG plate series."
    ) in report
    assert "SVG path elements                                 16,491" in report
    assert "Reported annotated structures                        862" in report
    assert "Reported polygons                                 11,398" in report
