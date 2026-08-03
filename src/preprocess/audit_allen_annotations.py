#!/usr/bin/env python3
"""Print a read-only audit of the Allen/Ding 2016 annotated atlas sections."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


PAPER_CITATION = "Ding et al., J Comp Neurol. 2016;524:3127-3481"
PAPER_DOI = "10.1002/cne.24080"
PAPER_URL = "https://pmc.ncbi.nlm.nih.gov/articles/PMC5054943/"
PAPER_ANNOTATED_PLATES = 106
PAPER_ANNOTATED_STRUCTURES = 862
PAPER_POLYGONS = 11_398
ALLOWED_MAPPING_STATUS = {"exact_id", "repaired_unique_section"}
REQUIRED_MANIFEST_FIELDS = {
    "kind",
    "series_or_layer",
    "section_number",
    "allen_section_image_id",
    "path",
    "sha256",
    "graphic_groups_present",
    "matching_nissl_section_image_id",
    "mapping_status",
}
REQUIRED_STRUCTURE_FIELDS = {
    "structure_id",
    "acronym",
    "name",
    "structure_graph_id",
}


@dataclass(frozen=True)
class SectionAudit:
    plate_index: int
    section_number: int
    svg_path: str
    nissl_path: str
    nissl_section_image_id: int
    mapping_status: str
    graphic_groups: tuple[int, ...]
    polygon_count: int
    unique_structure_count: int


@dataclass(frozen=True)
class GroupAudit:
    group_id: int
    name: str
    plate_count: int
    polygon_count: int
    unique_structure_count: int


@dataclass(frozen=True)
class AnnotationAudit:
    data_dir: Path
    specimen_id: int
    donor_id: int
    atlas_id: int
    structure_graph_id: int
    manifest_sha256: str
    structures_sha256: str
    ontology_structure_count: int
    sections: tuple[SectionAudit, ...]
    groups: tuple[GroupAudit, ...]
    unique_drawn_structure_count: int
    polygon_count: int
    resolved_structure_references: int
    mapped_plate_count: int


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_tsv(path: Path) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    try:
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream, delimiter="\t")
            return tuple(reader.fieldnames or ()), list(reader)
    except OSError as exc:
        raise RuntimeError(f"Cannot read canonical metadata {path}") from exc


def _integer(row: Mapping[str, str], key: str, context: str) -> int:
    try:
        return int(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"{context} has no integer {key}") from exc


def _safe_source(data_dir: Path, relative: str) -> Path:
    path = (data_dir / relative).resolve()
    try:
        path.relative_to(data_dir)
    except ValueError as exc:
        raise RuntimeError(f"Manifest path escapes the raw dataset: {relative}") from exc
    return path


def _verify_artifact(data_dir: Path, row: Mapping[str, str]) -> Path:
    relative = row["path"]
    path = _safe_source(data_dir, relative)
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"Missing or non-regular source artifact: {relative}")
    expected = row["sha256"]
    observed = sha_file(path)
    if observed != expected:
        raise RuntimeError(
            f"Source checksum mismatch for {relative}: "
            f"expected {expected}, observed {observed}"
        )
    return path


def _graphic_group_catalog(dataset: Mapping[str, Any]) -> tuple[tuple[int, str], ...]:
    payload = dataset.get("graphic_groups")
    if not isinstance(payload, list) or not payload:
        raise RuntimeError("dataset.json has no graphic-group catalog")
    try:
        groups = tuple((int(item["id"]), str(item["name"])) for item in payload)
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("dataset.json has invalid graphic-group metadata") from exc
    if len(groups) != len({group_id for group_id, _ in groups}):
        raise RuntimeError("dataset.json has duplicate graphic-group IDs")
    return groups


def _inspect_svg(
    path: Path,
    group_catalog: Mapping[int, str],
    ontology_ids: set[int],
) -> tuple[tuple[int, ...], int, set[int], Counter[int], dict[int, set[int]]]:
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError) as exc:
        raise RuntimeError(f"Cannot parse annotation SVG {path}") from exc
    if root.tag.rsplit("}", 1)[-1] != "svg":
        raise RuntimeError(f"Annotation file has no SVG root: {path}")

    group_order: list[int] = []
    section_ids: set[int] = set()
    group_polygons: Counter[int] = Counter()
    group_ids: dict[int, set[int]] = defaultdict(set)
    polygon_count = 0

    def visit(element: ET.Element, group_id: int | None = None) -> None:
        nonlocal polygon_count
        raw_group = element.attrib.get("graphic_group_label_id")
        if raw_group is not None:
            try:
                group_id = int(raw_group)
            except ValueError as exc:
                raise RuntimeError(
                    f"Invalid graphic_group_label_id in {path}: {raw_group!r}"
                ) from exc
            if group_id not in group_catalog:
                raise RuntimeError(f"Unknown graphic group {group_id} in {path}")
            observed_name = element.attrib.get("graphic_group_label")
            if observed_name != group_catalog[group_id]:
                raise RuntimeError(
                    f"Graphic-group name mismatch for {group_id} in {path}"
                )
        if element.tag.rsplit("}", 1)[-1] == "path":
            if group_id is None:
                raise RuntimeError(f"Annotation path outside a graphic group in {path}")
            raw_id = element.attrib.get("structure_id")
            if raw_id is None:
                raise RuntimeError(f"Annotation path without structure_id in {path}")
            try:
                structure_id = int(raw_id)
            except ValueError as exc:
                raise RuntimeError(
                    f"Invalid structure_id in {path}: {raw_id!r}"
                ) from exc
            if structure_id not in ontology_ids:
                raise RuntimeError(
                    f"Unresolved structure_id {structure_id} in {path}"
                )
            if group_id not in group_order:
                group_order.append(group_id)
            polygon_count += 1
            section_ids.add(structure_id)
            group_polygons[group_id] += 1
            group_ids[group_id].add(structure_id)
        for child in element:
            visit(child, group_id)

    visit(root)
    return (
        tuple(group_order),
        polygon_count,
        section_ids,
        group_polygons,
        group_ids,
    )


def build_audit(data_dir: Path) -> AnnotationAudit:
    """Validate canonical inputs and calculate the annotated-section audit."""
    data_dir = data_dir.expanduser().resolve()
    dataset_path = data_dir / "metadata/dataset.json"
    manifest_path = data_dir / "metadata/manifest.tsv"
    structures_path = data_dir / "metadata/structures.tsv"
    try:
        dataset = json.loads(dataset_path.read_text())
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Cannot read canonical metadata {dataset_path}") from exc
    if not isinstance(dataset, Mapping):
        raise RuntimeError("dataset.json root must be an object")
    groups = _graphic_group_catalog(dataset)
    group_catalog = dict(groups)

    structure_fields, structure_rows = _read_tsv(structures_path)
    if not REQUIRED_STRUCTURE_FIELDS.issubset(structure_fields):
        raise RuntimeError("Noncanonical structures.tsv columns")
    graph_id = int(dataset["allen_structure_graph_id"])
    ontology_ids: set[int] = set()
    for row in structure_rows:
        structure_id = _integer(row, "structure_id", "structures.tsv row")
        if _integer(row, "structure_graph_id", "structures.tsv row") != graph_id:
            raise RuntimeError(f"Structure {structure_id} belongs to another graph")
        if structure_id in ontology_ids:
            raise RuntimeError(f"Duplicate structure ID {structure_id}")
        ontology_ids.add(structure_id)

    manifest_fields, manifest_rows = _read_tsv(manifest_path)
    if not REQUIRED_MANIFEST_FIELDS.issubset(manifest_fields):
        raise RuntimeError("Noncanonical manifest.tsv columns")
    nissl_by_id = {
        _integer(row, "allen_section_image_id", "Nissl manifest row"): row
        for row in manifest_rows
        if row["kind"] == "histology_jpeg" and row["series_or_layer"] == "nissl"
    }
    svg_rows = [row for row in manifest_rows if row["kind"] == "annotation_svg"]
    svg_rows.sort(key=lambda row: _integer(row, "section_number", "SVG manifest row"))
    if not svg_rows:
        raise RuntimeError("Canonical manifest contains no annotation SVG rows")
    manifest_svg_paths = {row["path"] for row in svg_rows}
    disk_svg_paths = {
        path.relative_to(data_dir).as_posix()
        for path in (data_dir / "nissl/labels_orig").glob("*.svg")
        if path.is_file()
    }
    if disk_svg_paths != manifest_svg_paths:
        raise RuntimeError(
            "Annotation SVG files on disk do not exactly match the canonical manifest"
        )

    all_ids: set[int] = set()
    total_polygons = 0
    total_references = 0
    group_plate_counts: Counter[int] = Counter()
    group_polygon_counts: Counter[int] = Counter()
    group_structure_ids: dict[int, set[int]] = defaultdict(set)
    section_payloads: list[dict[str, Any]] = []
    mapped = 0
    seen_sections: set[int] = set()
    for svg_row in svg_rows:
        section_number = _integer(svg_row, "section_number", "SVG manifest row")
        if section_number in seen_sections:
            raise RuntimeError(f"Duplicate annotated section number {section_number}")
        seen_sections.add(section_number)
        mapping_status = svg_row["mapping_status"]
        if mapping_status not in ALLOWED_MAPPING_STATUS:
            raise RuntimeError(
                f"Unresolved SVG-to-Nissl mapping for section {section_number}: "
                f"{mapping_status}"
            )
        nissl_id = _integer(
            svg_row, "matching_nissl_section_image_id", "SVG manifest row"
        )
        nissl_row = nissl_by_id.get(nissl_id)
        if nissl_row is None:
            raise RuntimeError(
                f"Section {section_number} maps to missing Nissl ID {nissl_id}"
            )
        if _integer(nissl_row, "section_number", "Nissl manifest row") != section_number:
            raise RuntimeError(f"SVG-to-Nissl section mismatch at {section_number}")
        svg_path = _verify_artifact(data_dir, svg_row)
        _verify_artifact(data_dir, nissl_row)
        (
            observed_groups,
            polygon_count,
            section_ids,
            per_group_polygons,
            per_group_ids,
        ) = _inspect_svg(svg_path, group_catalog, ontology_ids)
        manifest_groups = tuple(
            int(value)
            for value in svg_row["graphic_groups_present"].split(";")
            if value and value != "n/a"
        )
        if set(observed_groups) != set(manifest_groups):
            raise RuntimeError(
                f"Graphic-group set differs from manifest for section {section_number}"
            )
        all_ids.update(section_ids)
        total_polygons += polygon_count
        total_references += polygon_count
        for group_id in observed_groups:
            group_plate_counts[group_id] += 1
            group_polygon_counts[group_id] += per_group_polygons[group_id]
            group_structure_ids[group_id].update(per_group_ids[group_id])
        mapped += 1
        section_payloads.append(
            {
                "section_number": section_number,
                "svg_path": svg_row["path"],
                "nissl_path": nissl_row["path"],
                "nissl_section_image_id": nissl_id,
                "mapping_status": mapping_status,
                "graphic_groups": observed_groups,
                "polygon_count": polygon_count,
                "unique_structure_count": len(section_ids),
            }
        )

    sections = tuple(
        SectionAudit(plate_index=index, **payload)
        for index, payload in enumerate(section_payloads, start=1)
    )
    group_audits = tuple(
        GroupAudit(
            group_id,
            name,
            group_plate_counts[group_id],
            group_polygon_counts[group_id],
            len(group_structure_ids[group_id]),
        )
        for group_id, name in groups
    )
    return AnnotationAudit(
        data_dir=data_dir,
        specimen_id=int(dataset["allen_specimen_id"]),
        donor_id=int(dataset["allen_donor_id"]),
        atlas_id=int(dataset["allen_atlas_id"]),
        structure_graph_id=graph_id,
        manifest_sha256=sha_file(manifest_path),
        structures_sha256=sha_file(structures_path),
        ontology_structure_count=len(ontology_ids),
        sections=sections,
        groups=group_audits,
        unique_drawn_structure_count=len(all_ids),
        polygon_count=total_polygons,
        resolved_structure_references=total_references,
        mapped_plate_count=mapped,
    )


def format_audit(audit: AnnotationAudit, include_sections: bool = True) -> str:
    """Format a stable plain-text report for terminals and captured logs."""
    section_counts = [section.unique_structure_count for section in audit.sections]
    polygon_counts = [section.polygon_count for section in audit.sections]
    plate_coverage = f"{len(audit.sections):,} / {PAPER_ANNOTATED_PLATES:,}"
    if len(audit.sections) == PAPER_ANNOTATED_PLATES:
        acquisition_statement = (
            f"{audit.unique_drawn_structure_count:,} distinct structure IDs were "
            "observed across the complete locally acquired official "
            "modified-Brodmann SVG plate series."
        )
    else:
        acquisition_statement = (
            f"{audit.unique_drawn_structure_count:,} distinct structure IDs were "
            "observed across the locally acquired official modified-Brodmann "
            "SVG plate series."
        )
    lines = [
        "Allen/Ding 2016 annotated-section audit",
        "=" * 43,
        f"Raw dataset: {audit.data_dir}",
        (
            f"Allen IDs: specimen {audit.specimen_id}, donor {audit.donor_id}, "
            f"atlas {audit.atlas_id}, structure graph {audit.structure_graph_id}"
        ),
        f"Raw manifest SHA-256:   {audit.manifest_sha256}",
        f"Structures SHA-256:     {audit.structures_sha256}",
        "Validation status: PASS",
        "",
        "Official modified-Brodmann SVG plate-series acquisition",
        f"{'Atlas ID':<44}{audit.atlas_id:>12}",
        f"{'Annotated plates':<44}{plate_coverage:>12}",
        f"{'Distinct drawn structure IDs':<44}{audit.unique_drawn_structure_count:>12,}",
        f"{'Status':<44}{'PASS':>12}",
        "",
        acquisition_statement,
        "",
        "Local SVG inventory",
        f"{'SVG path elements':<44}{audit.polygon_count:>12,}",
        f"{'Counting unit':<38}{'SVG <path> elements':>18}",
        "",
        "Ding et al. publication",
        f"Source: {PAPER_CITATION}; doi:{PAPER_DOI}",
        f"Paper URL: {PAPER_URL}",
        f"{'Reported annotated structures':<44}{PAPER_ANNOTATED_STRUCTURES:>12,}",
        f"{'Reported polygons':<44}{PAPER_POLYGONS:>12,}",
        f"{'Comparability to local SVG counts':<38}{'Not established':>18}",
        "",
        "Current source summary",
        f"Annotation SVGs:              {len(audit.sections):,}",
        f"Mapped SVG-to-Nissl plates:   {audit.mapped_plate_count:,}",
        f"Ontology rows:                {audit.ontology_structure_count:,}",
        f"Resolved path references:     {audit.resolved_structure_references:,} / {audit.polygon_count:,}",
        f"Unique drawn structure IDs:   {audit.unique_drawn_structure_count:,}",
        f"SVG path elements:            {audit.polygon_count:,}",
        f"Structures per plate:         min {min(section_counts)}, median {statistics.median(section_counts):g}, mean {statistics.mean(section_counts):.2f}, max {max(section_counts)}",
        f"Polygons per plate:           min {min(polygon_counts)}, median {statistics.median(polygon_counts):g}, mean {statistics.mean(polygon_counts):.2f}, max {max(polygon_counts)}",
        "",
        "Graphic-group breakdown",
        f"{'ID':>10}  {'Name':<43}{'Plates':>8}{'Paths':>10}{'IDs':>8}",
    ]
    lines.extend(
        f"{group.group_id:>10}  {group.name:<43}{group.plate_count:>8,}{group.polygon_count:>10,}{group.unique_structure_count:>8,}"
        for group in audit.groups
    )
    if include_sections:
        lines.extend(
            [
                "",
                "Annotated sections (anterior-to-posterior manifest order)",
                f"{'Plate':>5} {'Section':>7} {'SVG':<16}{'Paths':>7}{'IDs':>6}  {'Groups':<44}{'Nissl ID':>12}  Mapping",
            ]
        )
        lines.extend(
            (
                f"{section.plate_index:>5} {section.section_number:>7} "
                f"{Path(section.svg_path).name:<16}{section.polygon_count:>7,}"
                f"{section.unique_structure_count:>6,}  "
                f"{','.join(map(str, section.graphic_groups)):<44}"
                f"{section.nissl_section_image_id:>12}  {section.mapping_status}"
            )
            for section in audit.sections
        )
    return "\n".join(lines) + "\n"


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw/allen/specimen_708424"),
        help="raw Allen specimen directory (default: %(default)s)",
    )
    result.add_argument(
        "--summary-only",
        action="store_true",
        help="omit the 106-row annotated-section table",
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        audit = build_audit(args.data_dir)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Validation status: FAIL\n{exc}", file=sys.stderr)
        return 2
    print(format_audit(audit, include_sections=not args.summary_only), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
