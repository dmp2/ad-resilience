#!/usr/bin/env python3
"""Create section-level OME-Zarr image-plus-label derivatives for Allen 708424."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import logging
import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

VERSION = "1.0.0"
NGFF_VERSION = "0.5"
ZARR_FORMAT = 3
EXPECTED_SECTIONS = 106
EXPECTED_GROUP_COVERAGE = {
    31: 106,
    113753816: 106,
    141667008: 102,
    265297118: 106,
}
BOUNDARY_POLICY = "Skia non-antialiased target-grid fill"
IMAGE_CHUNKS = (3, 512, 512)
LABEL_CHUNKS = (512, 512)
CODEC_PIPELINES = {
    # Endianness has no representation for one-byte elements in Zarr v3.
    "uint8": [
        {"name": "bytes"},
        {"name": "zstd", "configuration": {"level": 3, "checksum": True}},
    ],
    "uint32": [
        {"name": "bytes", "configuration": {"endian": "little"}},
        {"name": "zstd", "configuration": {"level": 3, "checksum": True}},
    ],
}
MANIFEST_FIELDS = (
    "section_number",
    "path",
    "source_nissl_path",
    "source_nissl_sha256",
    "source_svg_path",
    "source_svg_sha256",
    "graphic_groups_present",
    "width_px",
    "height_px",
    "pixel_size_um",
    "unique_structure_count",
    "within_layer_conflicting_pixel_count_sum",
    "tree_sha256",
    "status",
    "software_version",
)
LOG = logging.getLogger("rasterize_allen_annotations")


@dataclass(frozen=True)
class Structure:
    structure_id: int
    acronym: str
    name: str
    rgba: tuple[int, int, int, int]


@dataclass(frozen=True)
class SourceSnapshot:
    inventory_sha256: str
    manifest_sha256: str
    structures_sha256: str
    specimen_id: int
    donor_id: int
    atlas_id: int
    structure_graph_id: int
    graphic_groups: tuple[tuple[int, str], ...]


@dataclass(frozen=True)
class SectionSpec:
    section_number: int
    nissl_path: str
    nissl_sha256: str
    svg_path: str
    svg_sha256: str
    graphic_groups: tuple[int, ...]
    width_px: int
    height_px: int
    pixel_size_um: float

    @property
    def package_name(self) -> str:
        return f"section-{self.section_number:04d}.ome.zarr"


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    atomic_bytes(path, (json.dumps(payload, indent=2) + "\n").encode())


def atomic_manifest(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=MANIFEST_FIELDS,
        delimiter="\t",
        lineterminator="\n",
        extrasaction="raise",
    )
    writer.writeheader()
    for row in sorted(rows, key=lambda value: int(value["section_number"])):
        writer.writerow({key: row[key] for key in MANIFEST_FIELDS})
    atomic_bytes(path, buffer.getvalue().encode())


def tree_sha256(root: Path) -> str:
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError(f"OME-Zarr package is not a regular directory: {root}")
    files: list[Path] = []
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise RuntimeError(f"Symlink is forbidden in package: {relative}")
        if path.is_file():
            if path.name.endswith((".tmp", ".temp", ".swp")) or path.name.startswith(
                ".nfs"
            ):
                raise RuntimeError(f"Transient file is forbidden in package: {relative}")
            files.append(path)
        elif not path.is_dir():
            raise RuntimeError(f"Non-regular package entry: {relative}")
    digest = hashlib.sha256()
    for path in sorted(files, key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(sha_file(path).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _safe_relative(root: Path, value: str) -> Path:
    candidate = (root / value).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(f"Source path escapes raw dataset: {value}") from exc
    return candidate


def _read_tsv(path: Path) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        fields = tuple(reader.fieldnames or ())
        return fields, list(reader)


def _integer(row: Mapping[str, str], key: str) -> int:
    try:
        return int(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"Manifest row has no integer {key}") from exc


def _floating(row: Mapping[str, str], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"Manifest row has no numeric {key}") from exc


def load_source(
    data_dir: Path,
) -> tuple[SourceSnapshot, list[SectionSpec], dict[int, Structure], dict[str, Any]]:
    data_dir = data_dir.expanduser().resolve()
    dataset_path = data_dir / "metadata/dataset.json"
    manifest_path = data_dir / "metadata/manifest.tsv"
    structures_path = data_dir / "metadata/structures.tsv"
    try:
        raw_dataset = json.loads(dataset_path.read_text())
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Cannot read canonical raw metadata {dataset_path}") from exc
    accepted = raw_dataset.get("accepted_api_inventory")
    if not isinstance(accepted, Mapping):
        raise RuntimeError("Raw dataset has no accepted_api_inventory")
    core = {
        key: value
        for key, value in accepted.items()
        if key not in {"observed_at_utc", "inventory_sha256"}
    }
    inventory_digest = hashlib.sha256(
        json.dumps(core, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if accepted.get("inventory_sha256") != inventory_digest:
        raise RuntimeError(
            "Raw accepted inventory digest does not match its canonical content"
        )
    manifest_digest = sha_file(manifest_path)
    structures_digest = sha_file(structures_path)
    groups_payload = raw_dataset.get("graphic_groups")
    if not isinstance(groups_payload, list):
        raise RuntimeError("Raw dataset has no graphic-group catalog")
    groups = tuple((int(item["id"]), str(item["name"])) for item in groups_payload)
    if len(groups) != len({item[0] for item in groups}):
        raise RuntimeError("Raw graphic-group IDs are not unique")
    snapshot = SourceSnapshot(
        inventory_digest,
        manifest_digest,
        structures_digest,
        int(raw_dataset["allen_specimen_id"]),
        int(raw_dataset["allen_donor_id"]),
        int(raw_dataset["allen_atlas_id"]),
        int(raw_dataset["allen_structure_graph_id"]),
        groups,
    )

    structure_fields, structure_rows = _read_tsv(structures_path)
    required_structure_fields = {
        "structure_id",
        "acronym",
        "name",
        "color_hex",
        "structure_graph_id",
    }
    if not required_structure_fields.issubset(structure_fields):
        raise RuntimeError("Noncanonical structures.tsv columns")
    structures: dict[int, Structure] = {}
    for row in structure_rows:
        structure_id = _integer(row, "structure_id")
        if _integer(row, "structure_graph_id") != snapshot.structure_graph_id:
            raise RuntimeError(f"Structure {structure_id} belongs to another graph")
        color = row["color_hex"]
        if color in {"", "n/a"}:
            rgba = (255, 255, 255, 255)
        else:
            if not color.startswith("#") or len(color) != 7:
                raise RuntimeError(f"Invalid structure color {color!r}")
            rgba = tuple(int(color[index : index + 2], 16) for index in (1, 3, 5)) + (
                255,
            )
        if structure_id in structures:
            raise RuntimeError(f"Duplicate structure ID {structure_id}")
        structures[structure_id] = Structure(
            structure_id, row["acronym"], row["name"], rgba
        )

    manifest_fields, manifest_rows = _read_tsv(manifest_path)
    required_manifest_fields = {
        "kind",
        "series_or_layer",
        "section_number",
        "allen_section_image_id",
        "path",
        "width_px",
        "height_px",
        "pixel_size_um",
        "sha256",
        "graphic_groups_present",
        "matching_nissl_section_image_id",
        "mapping_status",
    }
    if not required_manifest_fields.issubset(manifest_fields):
        raise RuntimeError("Noncanonical raw manifest columns")
    nissl_by_id = {
        _integer(row, "allen_section_image_id"): row
        for row in manifest_rows
        if row["kind"] == "histology_jpeg" and row["series_or_layer"] == "nissl"
    }
    sections: list[SectionSpec] = []
    for svg in manifest_rows:
        if svg["kind"] != "annotation_svg":
            continue
        if svg["mapping_status"] not in {"exact_id", "repaired_unique_section"}:
            raise RuntimeError(
                f"Unresolved SVG→Nissl mapping for {svg['path']}: "
                f"{svg['mapping_status']}"
            )
        nissl_id = _integer(svg, "matching_nissl_section_image_id")
        if nissl_id not in nissl_by_id:
            raise RuntimeError(f"Missing mapped Nissl row {nissl_id}")
        nissl = nissl_by_id[nissl_id]
        section_number = _integer(svg, "section_number")
        if _integer(nissl, "section_number") != section_number:
            raise RuntimeError(f"Section-number mismatch for {svg['path']}")
        width, height = _integer(nissl, "width_px"), _integer(nissl, "height_px")
        manifest_group_ids = tuple(
            int(value)
            for value in svg["graphic_groups_present"].split(";")
            if value and value != "n/a"
        )
        source_groups = inspect_svg_groups(_safe_relative(data_dir, svg["path"]))
        group_ids = tuple(item[0] for item in source_groups)
        unknown = set(group_ids) - {item[0] for item in groups}
        if unknown:
            raise RuntimeError(f"Unknown graphic groups in {svg['path']}: {unknown}")
        if set(group_ids) != set(manifest_group_ids):
            raise RuntimeError(
                f"SVG graphic-group set differs from raw manifest for {svg['path']}"
            )
        for group_id, group_name in source_groups:
            if dict(groups)[group_id] != group_name:
                raise RuntimeError(
                    f"SVG graphic-group name differs from catalog for {group_id}"
                )
        sections.append(
            SectionSpec(
                section_number,
                nissl["path"],
                nissl["sha256"],
                svg["path"],
                svg["sha256"],
                group_ids,
                width,
                height,
                _floating(nissl, "pixel_size_um"),
            )
        )
    sections.sort(key=lambda item: item.section_number)
    if len(sections) != EXPECTED_SECTIONS:
        raise RuntimeError(
            f"Raw manifest has {len(sections)} annotated sections; "
            f"expected {EXPECTED_SECTIONS}"
        )
    if len({item.section_number for item in sections}) != len(sections):
        raise RuntimeError("Raw manifest has duplicate annotated section numbers")
    coverage = {
        group_id: sum(group_id in section.graphic_groups for section in sections)
        for group_id, _ in groups
    }
    if coverage != EXPECTED_GROUP_COVERAGE:
        raise RuntimeError(
            f"Raw graphic-group coverage changed: {coverage} != "
            f"{EXPECTED_GROUP_COVERAGE}"
        )
    return snapshot, sections, structures, raw_dataset


def verify_source_files(data_dir: Path, sections: Iterable[SectionSpec]) -> None:
    for section in sections:
        for relative, expected in (
            (section.nissl_path, section.nissl_sha256),
            (section.svg_path, section.svg_sha256),
        ):
            path = _safe_relative(data_dir, relative)
            if not path.is_file():
                raise RuntimeError(f"Missing raw source file {relative}")
            observed = sha_file(path)
            if observed != expected:
                raise RuntimeError(
                    f"Raw source checksum mismatch for {relative}: "
                    f"expected {expected}, observed {observed}"
                )


def default_output_dir(data_dir: Path) -> Path:
    data_dir = data_dir.expanduser().resolve()
    raw_root = data_dir.parent.parent
    if raw_root.name != "raw":
        raise RuntimeError(
            "Cannot derive output location from a noncanonical raw path; "
            "provide --output-dir"
        )
    return (
        raw_root.parent
        / "derivatives"
        / data_dir.relative_to(raw_root)
        / "annotations_ome_zarr"
    )


def git_revision() -> str:
    try:
        root = Path(__file__).resolve().parents[2]
        return subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def snapshot_identity(snapshot: SourceSnapshot) -> dict[str, str]:
    return {
        "raw_inventory_sha256": snapshot.inventory_sha256,
        "raw_manifest_sha256": snapshot.manifest_sha256,
        "raw_structures_sha256": snapshot.structures_sha256,
    }


def derivative_dataset(snapshot: SourceSnapshot) -> dict[str, Any]:
    return {
        "dataset_title": "Allen 708424 section image-and-label OME-Zarr derivative",
        "purpose": (
            "Analysis-ready local-section Nissl images with separate Allen "
            "graphic-group label layers"
        ),
        "source": {
            "allen_specimen_id": snapshot.specimen_id,
            "allen_donor_id": snapshot.donor_id,
            "allen_atlas_id": snapshot.atlas_id,
            "allen_structure_graph_id": snapshot.structure_graph_id,
            **snapshot_identity(snapshot),
            "structures_path": "metadata/structures.tsv",
        },
        "graphic_groups": [
            {"id": group_id, "name": name}
            for group_id, name in snapshot.graphic_groups
        ],
        "software": {
            "name": "rasterize_allen_annotations.py",
            "version": VERSION,
            "git_revision": git_revision(),
        },
        "format": {
            "ome_zarr_version": NGFF_VERSION,
            "zarr_format": ZARR_FORMAT,
            "resolution_levels": ["0"],
            "chunk_grid": {"name": "regular"},
            "chunk_key_encoding": {
                "name": "default",
                "configuration": {"separator": "/"},
            },
            "codec_pipelines": CODEC_PIPELINES,
        },
        "spatial_metadata": {
            "coordinate_space": "local 2-D section coordinates",
            "pixel_size": (
                "Per-section isotropic in-plane micrometer spacing from the "
                "matched canonical Nissl manifest row"
            ),
            "no_asserted_coordinates": [
                "MRI coordinates",
                "3-D section origin",
                "inter-plate spacing",
                "anterior-posterior position",
            ],
        },
        "tree_checksum": {
            "name": "sha256",
            "algorithm": "sorted POSIX relative_path + NUL + file_sha256 + LF",
        },
        "expected_section_count": EXPECTED_SECTIONS,
        "created_at_utc": now(),
    }


def ensure_snapshot(
    output_dir: Path, snapshot: SourceSnapshot, filtered: bool, verify_only: bool
) -> dict[str, Any]:
    dataset_path = output_dir / "dataset.json"
    if not dataset_path.is_file():
        if verify_only:
            raise RuntimeError(f"Derivative snapshot does not exist: {dataset_path}")
        if filtered:
            raise RuntimeError(
                "Filtered runs cannot initialize a derivative snapshot; "
                "run once without --section-number or --limit"
            )
        if output_dir.exists() and any(output_dir.iterdir()):
            raise RuntimeError(
                f"Refusing to initialize nonempty output without dataset.json: {output_dir}"
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        payload = derivative_dataset(snapshot)
        atomic_json(dataset_path, payload)
        return payload
    try:
        payload = json.loads(dataset_path.read_text())
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Unreadable derivative snapshot {dataset_path}") from exc
    source = payload.get("source", {})
    expected = snapshot_identity(snapshot)
    observed = {key: source.get(key) for key in expected}
    if observed != expected:
        raise RuntimeError(
            "SOURCE_SNAPSHOT_DRIFT: derivative source identity differs from raw "
            f"dataset (stored={observed}, current={expected}). Use a new "
            "--output-dir; --overwrite cannot accept source drift."
        )
    if payload.get("expected_section_count") != EXPECTED_SECTIONS:
        raise RuntimeError("Derivative snapshot has an incompatible section count")
    expected_format = derivative_dataset(snapshot)["format"]
    if payload.get("format") != expected_format:
        raise RuntimeError("Derivative snapshot has incompatible NGFF/Zarr storage metadata")
    expected_source_ids = {
        "allen_specimen_id": snapshot.specimen_id,
        "allen_donor_id": snapshot.donor_id,
        "allen_atlas_id": snapshot.atlas_id,
        "allen_structure_graph_id": snapshot.structure_graph_id,
        "structures_path": "metadata/structures.tsv",
    }
    if any(source.get(key) != value for key, value in expected_source_ids.items()):
        raise RuntimeError("Derivative snapshot has incompatible Allen source identity")
    return payload


def inspect_svg_groups(svg_path: Path) -> list[tuple[int, str]]:
    try:
        from defusedxml import ElementTree as safe_et
    except ImportError as exc:
        raise RuntimeError("Validation requires defusedxml") from exc
    try:
        root = safe_et.parse(svg_path).getroot()
    except Exception as exc:
        raise RuntimeError(f"Cannot safely parse SVG {svg_path}") from exc
    groups: list[tuple[int, str]] = []

    def visit(element: Any, group: tuple[int, str] | None = None) -> None:
        raw_id = element.attrib.get("graphic_group_label_id")
        if raw_id is not None:
            try:
                group = (int(raw_id), element.attrib["graphic_group_label"])
            except (KeyError, ValueError) as exc:
                raise RuntimeError("Incomplete SVG graphic-group metadata") from exc
        if element.tag.rsplit("}", 1)[-1] == "path" and element.attrib.get(
            "structure_id"
        ):
            if group is None:
                raise RuntimeError("Annotation path is outside a graphic group")
            if group not in groups:
                groups.append(group)
        for child in element:
            visit(child, group)

    visit(root)
    return groups


def _zarr_dependencies() -> tuple[Any, Any, Any, Any]:
    try:
        import numpy as np
        import zarr
        from zarr.codecs import BytesCodec, ZstdCodec
    except ImportError as exc:
        raise RuntimeError(
            "OME-Zarr access requires numpy and zarr; install "
            "configs/environment-rasterize-allen.yml"
        ) from exc
    return np, zarr, BytesCodec, ZstdCodec


def _create_array(
    group: Any,
    name: str,
    data: Any,
    chunks: tuple[int, ...],
    dimension_names: tuple[str, ...],
) -> Any:
    _, _, BytesCodec, ZstdCodec = _zarr_dependencies()
    return group.create_array(
        name,
        data=data,
        chunks=chunks,
        filters=None,
        serializer=BytesCodec(endian="little"),
        compressors=[ZstdCodec(level=3, checksum=True)],
        chunk_key_encoding={
            "name": "default",
            "configuration": {"separator": "/"},
        },
        dimension_names=dimension_names,
        fill_value=0,
    )


def _axes(label: bool = False) -> list[dict[str, str]]:
    spatial = [
        {"name": "y", "type": "space", "unit": "micrometer"},
        {"name": "x", "type": "space", "unit": "micrometer"},
    ]
    return spatial if label else [{"name": "c", "type": "channel"}] + spatial


def _multiscales(name: str, pixel_size_um: float, label: bool) -> list[dict[str, Any]]:
    scale = (
        [pixel_size_um, pixel_size_um]
        if label
        else [1.0, pixel_size_um, pixel_size_um]
    )
    return [
        {
            "name": name,
            "axes": _axes(label),
            "datasets": [
                {
                    "path": "0",
                    "coordinateTransformations": [{"type": "scale", "scale": scale}],
                }
            ],
        }
    ]


def write_package(
    package: Path,
    data_dir: Path,
    section: SectionSpec,
    snapshot: SourceSnapshot,
    structures: Mapping[int, Structure],
) -> None:
    np, zarr, _, _ = _zarr_dependencies()
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Rasterization requires Pillow") from exc
    try:
        from allen_svg_renderer import rasterize_svg
    except ImportError:
        from .allen_svg_renderer import rasterize_svg

    nissl_path = _safe_relative(data_dir, section.nissl_path)
    svg_path = _safe_relative(data_dir, section.svg_path)
    with Image.open(nissl_path) as image:
        image_data = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if image_data.shape != (section.height_px, section.width_px, 3):
        raise RuntimeError(
            f"Decoded Nissl dimensions differ from manifest for section "
            f"{section.section_number}"
        )
    expected_groups = dict(snapshot.graphic_groups)
    layers = rasterize_svg(
        svg_path, section.width_px, section.height_px, expected_groups
    )
    observed_groups = tuple(layer.graphic_group_id for layer in layers)
    if observed_groups != section.graphic_groups:
        raise RuntimeError(
            f"SVG group set/order differs from raw manifest for section "
            f"{section.section_number}: {observed_groups} != {section.graphic_groups}"
        )
    observed_ids = {value for layer in layers for value in layer.structure_ids}
    unresolved = sorted(observed_ids - set(structures))
    if unresolved:
        raise RuntimeError(
            f"SVG contains structure IDs absent from structures.tsv: {unresolved}"
        )

    root = zarr.open_group(str(package), mode="w", zarr_format=3)
    present = [f"group-{group_id}" for group_id in observed_groups]
    absent = [
        {"id": group_id, "name": name, "status": "absent-in-source"}
        for group_id, name in snapshot.graphic_groups
        if group_id not in observed_groups
    ]
    root.attrs.update(
        {
            "ome": {
                "version": NGFF_VERSION,
                "multiscales": _multiscales(
                    f"Allen Nissl section {section.section_number}",
                    section.pixel_size_um,
                    False,
                ),
            },
            "allen": {
                "section_number": section.section_number,
                "source_nissl": {
                    "path": section.nissl_path,
                    "sha256": section.nissl_sha256,
                },
                "source_svg": {
                    "path": section.svg_path,
                    "sha256": section.svg_sha256,
                },
                "target_dimensions": {
                    "width_px": section.width_px,
                    "height_px": section.height_px,
                },
                "pixel_size_um": section.pixel_size_um,
                "coordinate_space": "local 2-D section coordinates",
                "graphic_groups_present": list(observed_groups),
                "graphic_groups_absent": absent,
                "paint_order": "SVG document order",
                "svg_order_usage": "validation and provenance only",
                "boundary_policy": BOUNDARY_POLICY,
                "stroke_policy": "SVG strokes are not categorical label fills and are ignored",
                "structures": {
                    "path": "metadata/structures.tsv",
                    "sha256": snapshot.structures_sha256,
                },
                "software": {
                    "name": "rasterize_allen_annotations.py",
                    "version": VERSION,
                },
            },
        }
    )
    _create_array(
        root,
        "0",
        image_data.transpose(2, 0, 1),
        IMAGE_CHUNKS,
        ("c", "y", "x"),
    )
    labels_group = root.create_group("labels")
    labels_group.attrs.update(
        {"ome": {"version": NGFF_VERSION, "labels": present}}
    )
    for layer in layers:
        label_group = labels_group.create_group(f"group-{layer.graphic_group_id}")
        colors = []
        properties = []
        for structure_id in layer.structure_ids:
            structure = structures[structure_id]
            colors.append(
                {"label-value": structure_id, "rgba": list(structure.rgba)}
            )
            properties.append(
                {
                    "label-value": structure_id,
                    "name": structure.name,
                    "acronym": structure.acronym,
                }
            )
        label_group.attrs.update(
            {
                "ome": {
                    "version": NGFF_VERSION,
                    "multiscales": _multiscales(
                        f"Allen {layer.graphic_group_name}",
                        section.pixel_size_um,
                        True,
                    ),
                    "image-label": {
                        "version": NGFF_VERSION,
                        "colors": colors,
                        "properties": properties,
                        "source": {"image": "../../"},
                    },
                },
                "allen": {
                    "graphic_group_id": layer.graphic_group_id,
                    "graphic_group_name": layer.graphic_group_name,
                    "path_count": layer.path_count,
                    "unique_label_count": len(layer.structure_ids),
                    "conflicting_overlap_pixel_count": (
                        layer.conflicting_overlap_pixel_count
                    ),
                    "repeated_same_id_pixel_count": (
                        layer.repeated_same_id_pixel_count
                    ),
                    "paint_order": "SVG document order",
                    "svg_order": {
                        "minimum": layer.svg_order_min,
                        "maximum": layer.svg_order_max,
                        "nondecreasing": layer.svg_order_nondecreasing,
                        "used_for_painting": False,
                    },
                    "fill_rules": "inherited SVG nonzero/evenodd",
                    "boundary_policy": BOUNDARY_POLICY,
                },
            }
        )
        _create_array(
            label_group,
            "0",
            layer.labels,
            LABEL_CHUNKS,
            ("y", "x"),
        )


def _metadata(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Unreadable Zarr metadata {path}") from exc


def _assert_array_metadata(
    path: Path,
    shape: tuple[int, ...],
    chunks: tuple[int, ...],
    dtype: str,
    dimensions: tuple[str, ...],
) -> None:
    metadata = _metadata(path / "zarr.json")
    if metadata.get("zarr_format") != 3 or metadata.get("node_type") != "array":
        raise RuntimeError(f"Not a Zarr v3 array: {path}")
    expected = {
        "shape": list(shape),
        "data_type": dtype,
        "dimension_names": list(dimensions),
        "chunk_grid": {
            "name": "regular",
            "configuration": {"chunk_shape": list(chunks)},
        },
        "chunk_key_encoding": {
            "name": "default",
            "configuration": {"separator": "/"},
        },
        "codecs": CODEC_PIPELINES[dtype],
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise RuntimeError(
                f"Unexpected {key} metadata for {path}: "
                f"{metadata.get(key)!r} != {value!r}"
            )
    if metadata.get("fill_value") != 0:
        raise RuntimeError(f"Array fill value is not zero: {path}")


def validate_package(
    package: Path,
    data_dir: Path,
    section: SectionSpec,
    snapshot: SourceSnapshot,
    structures: Mapping[int, Structure],
    compare_source_image: bool = True,
) -> dict[str, int]:
    np, zarr, _, _ = _zarr_dependencies()
    if not package.is_dir() or package.is_symlink():
        raise RuntimeError(f"Missing package {package}")
    group_names = inspect_svg_groups(_safe_relative(data_dir, section.svg_path))
    expected_names = [
        (group_id, dict(snapshot.graphic_groups)[group_id])
        for group_id in section.graphic_groups
    ]
    if group_names != expected_names:
        raise RuntimeError(
            f"Source SVG group set differs from manifest for section "
            f"{section.section_number}: {group_names} != {expected_names}"
        )
    root_metadata = _metadata(package / "zarr.json")
    root_attrs = root_metadata.get("attributes", {})
    root_ome = root_attrs.get("ome", {})
    root_allen = root_attrs.get("allen", {})
    if root_ome.get("version") != NGFF_VERSION:
        raise RuntimeError(f"Wrong NGFF version in {package}")
    multiscales = root_ome.get("multiscales")
    if (
        not isinstance(multiscales, list)
        or len(multiscales) != 1
        or [axis.get("name") for axis in multiscales[0].get("axes", [])]
        != ["c", "y", "x"]
        or multiscales[0].get("datasets", [{}])[0].get("path") != "0"
    ):
        raise RuntimeError(f"Invalid root multiscales metadata in {package}")
    if root_allen.get("graphic_groups_present") != list(section.graphic_groups):
        raise RuntimeError(f"Wrong root graphic-group list in {package}")
    expected_root_provenance = {
        "source_nissl": {
            "path": section.nissl_path,
            "sha256": section.nissl_sha256,
        },
        "source_svg": {"path": section.svg_path, "sha256": section.svg_sha256},
        "target_dimensions": {
            "width_px": section.width_px,
            "height_px": section.height_px,
        },
        "pixel_size_um": section.pixel_size_um,
        "paint_order": "SVG document order",
        "boundary_policy": BOUNDARY_POLICY,
    }
    if any(
        root_allen.get(key) != value
        for key, value in expected_root_provenance.items()
    ):
        raise RuntimeError(f"Wrong Allen source/spatial provenance in {package}")
    if multiscales[0].get("datasets", [{}])[0].get(
        "coordinateTransformations"
    ) != [{
        "type": "scale",
        "scale": [1.0, section.pixel_size_um, section.pixel_size_um],
    }]:
        raise RuntimeError(f"Wrong image physical scale in {package}")
    expected_absent = [
        item[0] for item in snapshot.graphic_groups if item[0] not in section.graphic_groups
    ]
    if [item.get("id") for item in root_allen.get("graphic_groups_absent", [])] != expected_absent:
        raise RuntimeError(f"Wrong absent graphic-group list in {package}")
    if root_allen.get("structures") != {
        "path": "metadata/structures.tsv",
        "sha256": snapshot.structures_sha256,
    }:
        raise RuntimeError(f"Wrong structures provenance in {package}")
    _assert_array_metadata(
        package / "0",
        (3, section.height_px, section.width_px),
        IMAGE_CHUNKS,
        "uint8",
        ("c", "y", "x"),
    )
    root = zarr.open_group(str(package), mode="r")
    image = root["0"]
    if compare_source_image:
        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("Image comparison requires Pillow") from exc
        with Image.open(_safe_relative(data_dir, section.nissl_path)) as source:
            expected_image = np.asarray(source.convert("RGB"), dtype=np.uint8).transpose(
                2, 0, 1
            )
        if not np.array_equal(image[:], expected_image):
            raise RuntimeError(f"Stored RGB image differs from source in {package}")
    labels_meta = _metadata(package / "labels/zarr.json")
    expected_label_names = [f"group-{value}" for value in section.graphic_groups]
    if labels_meta.get("attributes", {}).get("ome") != {
        "version": NGFF_VERSION,
        "labels": expected_label_names,
    }:
        raise RuntimeError(f"Invalid labels-group metadata in {package}")
    actual_label_names = sorted(
        path.name
        for path in (package / "labels").iterdir()
        if path.is_dir() and path.name.startswith("group-")
    )
    if actual_label_names != sorted(expected_label_names):
        raise RuntimeError(f"Unexpected label groups in {package}")
    all_ids: set[int] = set()
    conflict_sum = 0
    for group_id in section.graphic_groups:
        group_name = f"group-{group_id}"
        group_path = package / "labels" / group_name
        group_meta = _metadata(group_path / "zarr.json")
        attrs = group_meta.get("attributes", {})
        ome = attrs.get("ome", {})
        allen = attrs.get("allen", {})
        multiscales = ome.get("multiscales")
        if (
            ome.get("version") != NGFF_VERSION
            or not isinstance(multiscales, list)
            or len(multiscales) != 1
            or [axis.get("name") for axis in multiscales[0].get("axes", [])]
            != ["y", "x"]
            or multiscales[0].get("datasets", [{}])[0].get("path") != "0"
        ):
            raise RuntimeError(f"Invalid label multiscales metadata in {group_path}")
        if multiscales[0].get("datasets", [{}])[0].get(
            "coordinateTransformations"
        ) != [{
            "type": "scale",
            "scale": [section.pixel_size_um, section.pixel_size_um],
        }]:
            raise RuntimeError(f"Wrong label physical scale in {group_path}")
        image_label = ome.get("image-label", {})
        if (
            image_label.get("version") != NGFF_VERSION
            or image_label.get("source") != {"image": "../../"}
        ):
            raise RuntimeError(f"Invalid image-label link in {group_path}")
        if allen.get("graphic_group_id") != group_id or allen.get(
            "graphic_group_name"
        ) != dict(snapshot.graphic_groups)[group_id]:
            raise RuntimeError(f"Invalid Allen group metadata in {group_path}")
        _assert_array_metadata(
            group_path / "0",
            (section.height_px, section.width_px),
            LABEL_CHUNKS,
            "uint32",
            ("y", "x"),
        )
        values = {
            int(value)
            for value in np.unique(root[f"labels/{group_name}/0"][:])
            if value
        }
        unresolved = values - set(structures)
        if unresolved:
            raise RuntimeError(f"Unresolved structure IDs in {group_path}: {unresolved}")
        colors = image_label.get("colors", [])
        properties = image_label.get("properties", [])
        color_ids = {item.get("label-value") for item in colors}
        property_ids = {item.get("label-value") for item in properties}
        if color_ids != values or property_ids != values:
            raise RuntimeError(f"Ontology metadata does not match labels in {group_path}")
        for item in colors:
            structure = structures[item["label-value"]]
            if item.get("rgba") != list(structure.rgba):
                raise RuntimeError(f"Wrong structure color in {group_path}")
        for item in properties:
            structure = structures[item["label-value"]]
            if item.get("name") != structure.name or item.get(
                "acronym"
            ) != structure.acronym or set(item) != {
                "label-value",
                "name",
                "acronym",
            }:
                raise RuntimeError(f"Wrong compact structure properties in {group_path}")
        if allen.get("unique_label_count") != len(values):
            raise RuntimeError(f"Wrong unique-label count in {group_path}")
        conflict_sum += int(allen.get("conflicting_overlap_pixel_count", -1))
        all_ids.update(values)
    return {
        "unique_structure_count": len(all_ids),
        "within_layer_conflicting_pixel_count_sum": conflict_sum,
    }


def manifest_record(
    section: SectionSpec, metrics: Mapping[str, int], checksum: str
) -> dict[str, Any]:
    return {
        "section_number": section.section_number,
        "path": section.package_name,
        "source_nissl_path": section.nissl_path,
        "source_nissl_sha256": section.nissl_sha256,
        "source_svg_path": section.svg_path,
        "source_svg_sha256": section.svg_sha256,
        "graphic_groups_present": json.dumps(
            list(section.graphic_groups), separators=(",", ":")
        ),
        "width_px": section.width_px,
        "height_px": section.height_px,
        "pixel_size_um": section.pixel_size_um,
        "unique_structure_count": metrics["unique_structure_count"],
        "within_layer_conflicting_pixel_count_sum": metrics[
            "within_layer_conflicting_pixel_count_sum"
        ],
        "tree_sha256": checksum,
        "status": "complete",
        "software_version": VERSION,
    }


def load_derivative_manifest(path: Path) -> dict[int, dict[str, str]]:
    if not path.is_file():
        return {}
    fields, rows = _read_tsv(path)
    if fields != MANIFEST_FIELDS:
        raise RuntimeError(f"Noncanonical derivative manifest columns in {path}")
    result: dict[int, dict[str, str]] = {}
    for row in rows:
        section = _integer(row, "section_number")
        if section in result:
            raise RuntimeError(f"Duplicate derivative manifest section {section}")
        result[section] = row
    return result


def _record_matches(
    stored: Mapping[str, str], expected: Mapping[str, Any]
) -> bool:
    return all(str(expected[key]) == stored.get(key) for key in MANIFEST_FIELDS)


def publish_package(
    output_dir: Path,
    data_dir: Path,
    section: SectionSpec,
    snapshot: SourceSnapshot,
    structures: Mapping[int, Structure],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{section.package_name}.", dir=output_dir)
    )
    target = output_dir / section.package_name
    try:
        write_package(temporary, data_dir, section, snapshot, structures)
        metrics = validate_package(
            temporary, data_dir, section, snapshot, structures, True
        )
        checksum = tree_sha256(temporary)
        if target.exists():
            run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
            quarantine = output_dir / "quarantine" / run_id / target.name
            quarantine.parent.mkdir(parents=True, exist_ok=True)
            os.replace(target, quarantine)
            try:
                os.replace(temporary, target)
            except BaseException:
                os.replace(quarantine, target)
                raise
        else:
            os.replace(temporary, target)
        return manifest_record(section, metrics, checksum)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def verify_one(
    output_dir: Path,
    data_dir: Path,
    section: SectionSpec,
    snapshot: SourceSnapshot,
    structures: Mapping[int, Structure],
    stored: Mapping[str, str] | None,
) -> dict[str, Any]:
    target = output_dir / section.package_name
    metrics = validate_package(target, data_dir, section, snapshot, structures, True)
    checksum = tree_sha256(target)
    expected = manifest_record(section, metrics, checksum)
    if stored is not None and not _record_matches(stored, expected):
        raise RuntimeError(
            f"Derivative manifest row does not match package {section.package_name}"
        )
    return expected


def write_qc(
    output_dir: Path,
    selected: Sequence[SectionSpec],
) -> Path:
    np, zarr, _, _ = _zarr_dependencies()
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise RuntimeError("QC output requires Pillow") from exc
    cards = []
    for section in selected:
        root = zarr.open_group(str(output_dir / section.package_name), mode="r")
        rgb = root["0"][:].transpose(1, 2, 0)
        composite = np.zeros((section.height_px, section.width_px), dtype=np.uint32)
        for group_id in section.graphic_groups:
            labels = root[f"labels/group-{group_id}/0"][:]
            composite[labels != 0] = labels[labels != 0]
        edge = np.zeros(composite.shape, dtype=bool)
        edge[1:, :] |= composite[1:, :] != composite[:-1, :]
        edge[:, 1:] |= composite[:, 1:] != composite[:, :-1]
        overlay = rgb.copy()
        overlay[edge & (composite != 0)] = (255, 0, 255)
        card = Image.fromarray(overlay).convert("RGB")
        card.thumbnail((384, 384))
        framed = Image.new("RGB", (400, 420), "white")
        framed.paste(card, ((400 - card.width) // 2, 24))
        ImageDraw.Draw(framed).text(
            (8, 4), f"section {section.section_number:04d}", fill="black"
        )
        cards.append(framed)
    qc_dir = output_dir / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)
    if len(cards) == 1:
        destination = qc_dir / f"section-{selected[0].section_number:04d}.png"
        cards[0].save(destination)
        return destination
    columns = 6
    rows = (len(cards) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * 400, rows * 420), "white")
    for index, card in enumerate(cards):
        sheet.paste(card, ((index % columns) * 400, (index // columns) * 420))
    destination = qc_dir / "contact-sheet.png"
    sheet.save(destination)
    return destination


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--data-dir", type=Path, required=True)
    result.add_argument("--output-dir", type=Path)
    result.add_argument("--section-number", type=int)
    result.add_argument("--limit", type=int)
    result.add_argument("--workers", type=int, default=1)
    result.add_argument("--overwrite", action="store_true")
    result.add_argument("--verify-existing", action="store_true")
    result.add_argument("--write-qc", action="store_true")
    result.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    result.add_argument("--version", action="version", version=VERSION)
    return result


def run(args: argparse.Namespace) -> int:
    if args.limit is not None and args.limit < 1:
        raise RuntimeError("--limit must be positive")
    if args.workers < 1:
        raise RuntimeError("--workers must be positive")
    if args.verify_existing and (args.overwrite or args.write_qc):
        raise RuntimeError(
            "--verify-existing cannot be combined with --overwrite or --write-qc"
        )
    data_dir = args.data_dir.expanduser().resolve()
    snapshot, sections, structures, _ = load_source(data_dir)
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else default_output_dir(data_dir)
    )
    filtered = args.section_number is not None or args.limit is not None
    if args.section_number is not None:
        selected = [
            section
            for section in sections
            if section.section_number == args.section_number
        ]
        if not selected:
            raise RuntimeError(
                f"No annotated section {args.section_number} in raw manifest"
            )
    else:
        selected = list(sections)
    if args.limit is not None:
        selected = selected[: args.limit]
    verify_source_files(data_dir, selected)
    ensure_snapshot(output_dir, snapshot, filtered, args.verify_existing)
    manifest_path = output_dir / "metadata/manifest.tsv"
    stored = load_derivative_manifest(manifest_path)

    if args.verify_existing:
        if not filtered and set(stored) != {
            section.section_number for section in sections
        }:
            raise RuntimeError(
                f"Full derivative manifest contains {len(stored)} sections, "
                f"expected {EXPECTED_SECTIONS}"
            )
        if not filtered:
            actual_packages = {
                path.name
                for path in output_dir.glob("section-*.ome.zarr")
                if path.is_dir()
            }
            expected_packages = {section.package_name for section in sections}
            if actual_packages != expected_packages:
                raise RuntimeError(
                    "Derivative package directory set does not match the 106 "
                    "expected sections"
                )
        for section in selected:
            if section.section_number not in stored:
                raise RuntimeError(
                    f"No derivative manifest row for section {section.section_number}"
                )
            verify_one(
                output_dir,
                data_dir,
                section,
                snapshot,
                structures,
                stored[section.section_number],
            )
        print(
            f"PASS: verified {len(selected)} package(s), "
            f"{sum(len(item.graphic_groups) for item in selected)} label layer(s); "
            "no files written"
        )
        return 0

    records: dict[int, Mapping[str, Any]] = dict(stored)
    pending: list[SectionSpec] = []
    skipped = 0
    for section in selected:
        target = output_dir / section.package_name
        if target.exists() and not args.overwrite:
            try:
                record = verify_one(
                    output_dir,
                    data_dir,
                    section,
                    snapshot,
                    structures,
                    stored.get(section.section_number),
                )
            except RuntimeError as exc:
                raise RuntimeError(
                    f"Existing output is invalid for section "
                    f"{section.section_number}; inspect it or use --overwrite: {exc}"
                ) from exc
            records[section.section_number] = record
            skipped += 1
        else:
            pending.append(section)
    completed = 0
    if args.workers == 1:
        for section in pending:
            LOG.info("Rasterizing section %04d", section.section_number)
            record = publish_package(
                output_dir, data_dir, section, snapshot, structures
            )
            records[section.section_number] = record
            atomic_manifest(manifest_path, records.values())
            completed += 1
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    publish_package,
                    output_dir,
                    data_dir,
                    section,
                    snapshot,
                    structures,
                ): section
                for section in pending
            }
            for future in as_completed(futures):
                section = futures[future]
                record = future.result()
                records[section.section_number] = record
                atomic_manifest(manifest_path, records.values())
                completed += 1
                LOG.info("Completed section %04d", section.section_number)
    if skipped and not pending:
        atomic_manifest(manifest_path, records.values())
    if not filtered:
        expected_sections = {section.section_number for section in sections}
        if set(records) != expected_sections:
            raise RuntimeError(
                f"Full run produced {len(records)} manifest rows, expected "
                f"{EXPECTED_SECTIONS}"
            )
        actual_packages = {
            path.name
            for path in output_dir.glob("section-*.ome.zarr")
            if path.is_dir()
        }
        if actual_packages != {section.package_name for section in sections}:
            raise RuntimeError("Full run package set is not the expected 106 sections")
    qc_path = write_qc(output_dir, selected) if args.write_qc else None
    print(
        f"Complete: {completed} written, {skipped} valid existing, "
        f"{len(selected)} selected, "
        f"{sum(len(item.graphic_groups) for item in selected)} label layers"
        + (f", QC {qc_path}" if qc_path else "")
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    try:
        return run(args)
    except (OSError, RuntimeError, ValueError) as exc:
        LOG.error("%s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
