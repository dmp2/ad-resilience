#!/usr/bin/env python3
"""Acquire and validate raw Allen/Ding atlas data for specimen 708424.

Only source JPEGs, multi-group SVGs, the raw Allen ontology, and compact
metadata are handled here. Rasterization and all 3-D products are out of scope.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import logging
import os
import tempfile
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import requests
from PIL import Image, UnidentifiedImageError
from scipy.io import loadmat, savemat

try:
    from .download_allen_legacy import (
        _atomic_save_pil,
        build_session,
        generate_ihc_mask,
        generate_nissl_mask,
    )
except ImportError:  # Direct execution places this directory on sys.path.
    from download_allen_legacy import (
        _atomic_save_pil,
        build_session,
        generate_ihc_mask,
        generate_nissl_mask,
    )

API_BASE = "https://api.brain-map.org/api/v2"
SPECIMEN_ID = 708424
DONOR_ID = 12767
ATLAS_ID = 265297126
ATLAS_IMAGE_TYPE = "Atlas - Developing Human Brodmann"
GRAPH_ID = 16
GROUPS = (31, 113753816, 141667008, 265297118)
GROUP_LABELS = {
    31: "Atlas - Developing Human",
    113753816: "Atlas - Developing Human Sulci",
    141667008: "Atlas - Developing Human Hotspots",
    265297118: "Atlas - Developing Human Brodmann",
}
SERIES_LABELS = {"nissl": "Nissl", "pv": "PV", "smi32": "SMI-32"}
SERIES_ROOTS = {"nissl": "nissl", "pv": "ihc", "smi32": "smi32"}
TREATMENT_SERIES = {
    "nissl": "nissl",
    "ihc:parvalbumin": "pv",
    "ihc:smi-32": "smi32",
}
BASELINE_SERIES = {
    "nissl": {
        "treatment_id": 3,
        "treatment_name": "NISSL",
        "data_set_ids": [100149965],
        "image_count": 641,
    },
    "pv": {
        "treatment_id": 16,
        "treatment_name": "IHC:Parvalbumin",
        "data_set_ids": [100147602],
        "image_count": 287,
    },
    "smi32": {
        "treatment_id": 5,
        "treatment_name": "IHC:SMI-32",
        "data_set_ids": [],
        "image_count": 0,
    },
}
PUBLISHED = {"nissl": 679, "pv": 339, "smi32": 338}
DISCOVERY_DATE = "2026-07-28"
EXPECTED_SVGS = 106
VERSION = "2.0.0"
VERIFIED = {
    "downloaded",
    "verified-existing",
    "manifest-verified-existing",
    "redownloaded-after-quarantine",
}
MANIFEST_FIELDS = (
    "kind",
    "series_or_layer",
    "section_number",
    "allen_section_image_id",
    "allen_atlas_image_id",
    "path",
    "width_px",
    "height_px",
    "pixel_size_um",
    "sha256",
    "source_url",
    "status",
    "source_provider",
    "allen_data_set_id",
    "treatment_id",
    "graphic_groups_present",
    "matching_nissl_section_image_id",
    "mapping_status",
)
STRUCTURE_FIELDS = (
    "structure_id",
    "acronym",
    "name",
    "parent_structure_id",
    "color_hex",
    "structure_graph_id",
    "structure_id_path",
)
LOG = logging.getLogger("download_allen")


class APIInventoryChanged(RuntimeError):
    pass


@dataclass(frozen=True)
class Section:
    series: str
    section_id: int
    section_number: int
    treatment_id: int
    treatment_name: str
    data_set_id: int
    specimen_id: int
    donor_id: int | None
    annotated: bool = False
    resolution: float | None = None
    width: int | None = None
    height: int | None = None


@dataclass(frozen=True)
class AtlasPlate:
    atlas_image_id: int
    section_number: int
    width: int | None = None
    height: int | None = None
    resolution: float | None = None
    nissl_id: int | None = None
    mapping_status: str = "unresolved"


@dataclass(frozen=True)
class Artifact:
    kind: str
    series_or_layer: str
    section_number: int | None
    allen_section_image_id: int | None
    allen_atlas_image_id: int | None
    path: str
    width_px: int | None
    height_px: int | None
    pixel_size_um: float | None
    sha256: str
    source_url: str | None
    status: str
    source_provider: str
    allen_data_set_id: int | None = None
    treatment_id: int | None = None
    graphic_groups_present: str | None = None
    matching_nissl_section_image_id: int | None = None
    mapping_status: str | None = None


@dataclass(frozen=True)
class Structure:
    structure_id: int
    acronym: str
    name: str
    parent_structure_id: int | None
    color_hex: str | None
    structure_graph_id: int
    structure_id_path: str | None


@dataclass(frozen=True)
class Paths:
    root: Path
    metadata: Path
    dataset: Path
    manifest: Path
    structures: Path
    secjson: Path
    secmat: Path
    ontology: Path

    @classmethod
    def make(cls, value: Path) -> "Paths":
        root = value.expanduser().resolve()
        return cls(
            root,
            root / "metadata",
            root / "metadata/dataset.json",
            root / "metadata/manifest.tsv",
            root / "metadata/structures.tsv",
            root / "secInfo.json",
            root / "secInfo.mat",
            root / "ontology/structure_graph_16.json",
        )

    def image_dir(self, series: str) -> Path:
        return self.root / SERIES_ROOTS[series] / "images_orig"

    @property
    def svg_dir(self) -> Path:
        return self.root / "nissl/labels_orig"

    def create(self, available: Iterable[str]) -> None:
        for path in (
            self.root,
            self.metadata,
            self.svg_dir,
            self.ontology.parent,
            self.root / "nissl/masks_orig",
            self.root / "ihc/masks_orig",
        ):
            path.mkdir(parents=True, exist_ok=True)
        for series in available:
            self.image_dir(series).mkdir(parents=True, exist_ok=True)


def now() -> str:
    return (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )


def opt_int(value: Any) -> int | None:
    try:
        return None if value in (None, "", "n/a") else int(value)
    except (TypeError, ValueError):
        return None


def opt_float(value: Any) -> float | None:
    try:
        return None if value in (None, "", "n/a") else float(value)
    except (TypeError, ValueError):
        return None


def values(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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


def atomic_text(path: Path, text: str) -> None:
    atomic_bytes(path, text.encode())


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    atomic_text(path, json.dumps(payload, indent=2) + "\n")


def atomic_tsv(
    path: Path, fields: Sequence[str], rows: Iterable[Mapping[str, Any]]
) -> None:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=fields,
        delimiter="\t",
        lineterminator="\n",
        extrasaction="raise",
    )
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {key: "n/a" if row.get(key) is None else row.get(key) for key in fields}
        )
    atomic_text(path, buffer.getvalue())


def api_json(session: requests.Session, criteria: str) -> Mapping[str, Any]:
    response = session.get(
        f"{API_BASE}/data/query.json",
        params={"criteria": criteria, "num_rows": "all", "start_row": 0},
        timeout=(20, 180),
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, Mapping) or payload.get("success") is False:
        raise RuntimeError(f"Allen API query failed: {payload!r}")
    return payload


def classify_treatment(name: str) -> str | None:
    return TREATMENT_SERIES.get(name.strip().lower())


def inventory_payload(
    specimen_id: int,
    donor_id: int | None,
    series: Mapping[str, Mapping[str, Any]],
    observed: str,
) -> dict[str, Any]:
    core = {
        "specimen_id": specimen_id,
        "donor_id": donor_id,
        "series": {
            key: {
                "treatment_id": int(series[key]["treatment_id"]),
                "treatment_name": str(series[key]["treatment_name"]),
                "data_set_ids": sorted(map(int, series[key].get("data_set_ids", []))),
                "image_count": int(series[key]["image_count"]),
            }
            for key in SERIES_LABELS
        },
    }
    digest = sha_bytes(json.dumps(core, sort_keys=True, separators=(",", ":")).encode())
    return {**core, "observed_at_utc": observed, "inventory_sha256": digest}


def baseline_inventory() -> dict[str, Any]:
    return inventory_payload(
        SPECIMEN_ID, DONOR_ID, BASELINE_SERIES, "2026-07-28T14:22:45Z"
    )


def inventory_core(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in payload.items()
        if key not in {"observed_at_utc", "inventory_sha256"}
    }


class AllenSectionDataSetProvider:
    name = "AllenSectionDataSetProvider"

    def __init__(self, session: requests.Session):
        self.session = session

    def discover(
        self, specimen_id: int
    ) -> tuple[dict[str, list[Section]], dict[str, Any]]:
        treatment_data = api_json(
            self.session, "model::Treatment,rma::options[num_rows$eqall]"
        )
        vocabulary = {}
        for item in values(treatment_data.get("msg")):
            if isinstance(item, Mapping) and (
                series := classify_treatment(str(item.get("name", "")))
            ):
                vocabulary[series] = {
                    "treatment_id": int(item["id"]),
                    "treatment_name": str(item["name"]),
                }
        criteria = (
            "model::SectionDataSet,"
            f"rma::criteria,specimen[id$eq{specimen_id}],"
            "rma::include,treatments,specimen(donor),section_images(treatments),"
            "rma::options[num_rows$eqall]"
        )
        payload = api_json(self.session, criteria)
        records = {key: [] for key in SERIES_LABELS}
        datasets = defaultdict(set)
        metadata = {}
        donor_id = None
        for dataset in values(payload.get("msg")):
            if not isinstance(dataset, Mapping):
                continue
            did = int(dataset["id"])
            specimen = dataset.get("specimen", {})
            if isinstance(specimen, Mapping):
                donor_id = opt_int(specimen.get("donor_id")) or donor_id
                if isinstance(specimen.get("donor"), Mapping):
                    donor_id = opt_int(specimen["donor"].get("id")) or donor_id
            recognized = [
                (classify_treatment(str(t.get("name", ""))), t)
                for t in values(dataset.get("treatments"))
                if isinstance(t, Mapping)
            ]
            recognized = [(s, t) for s, t in recognized if s]
            if len(recognized) != 1:
                raise RuntimeError(f"Dataset {did} has ambiguous treatment metadata")
            series, treatment = recognized[0]
            tid = int(treatment["id"])
            tname = str(treatment["name"])
            datasets[series].add(did)
            metadata[series] = {"treatment_id": tid, "treatment_name": tname}
            for image in values(dataset.get("section_images")):
                if not isinstance(image, Mapping):
                    continue
                iid, number = opt_int(image.get("id")), opt_int(
                    image.get("section_number")
                )
                if iid is None or number is None:
                    raise RuntimeError(f"Incomplete image in dataset {did}")
                records[series].append(
                    Section(
                        series,
                        iid,
                        number,
                        tid,
                        tname,
                        did,
                        specimen_id,
                        donor_id,
                        bool(image.get("annotated", False)),
                        opt_float(image.get("resolution")),
                        opt_int(image.get("width")),
                        opt_int(image.get("height")),
                    )
                )
        snapshot_series = {}
        for series in SERIES_LABELS:
            meta = metadata.get(series) or vocabulary.get(series)
            if meta is None:
                raise RuntimeError(f"Missing Allen treatment vocabulary for {series}")
            records[series].sort(
                key=lambda item: (item.section_number, item.section_id)
            )
            if len({item.section_id for item in records[series]}) != len(
                records[series]
            ):
                raise RuntimeError(f"Duplicate {series} SectionImage IDs")
            snapshot_series[series] = {
                **meta,
                "data_set_ids": sorted(datasets[series]),
                "image_count": len(records[series]),
            }
        return records, inventory_payload(specimen_id, donor_id, snapshot_series, now())


class AllenAtlasPlateProvider:
    name = "AllenAtlasPlateProvider"

    def __init__(self, session: requests.Session):
        self.session = session

    def discover(self) -> tuple[dict[str, Any], list[AtlasPlate]]:
        criteria = (
            f"model::Atlas,rma::criteria,[id$eq{ATLAS_ID}],"
            "rma::include,structure_graph,treatment,specimen,graphic_group_labels,atlas_data_sets,"
            "rma::options[num_rows$eqall]"
        )
        rows = values(api_json(self.session, criteria).get("msg"))
        if len(rows) != 1:
            raise RuntimeError("Expected exactly one selected Atlas")
        atlas = rows[0]
        graph = atlas.get("structure_graph", {})
        if int(graph.get("id", -1)) != GRAPH_ID:
            raise RuntimeError("Selected atlas is not associated with graph 16")
        specimen = atlas.get("specimen", {})
        treatment = atlas.get("treatment", {})
        if (
            opt_int(specimen.get("id")) != SPECIMEN_ID
            or opt_int(specimen.get("donor_id")) != DONOR_ID
        ):
            raise RuntimeError("Selected atlas specimen/donor identity changed")
        if opt_int(treatment.get("id")) != 3 or treatment.get("name") != "NISSL":
            raise RuntimeError("Selected atlas treatment identity changed")
        labels = {
            int(item["id"]): str(item["name"])
            for item in values(atlas.get("graphic_group_labels"))
        }
        if any(labels.get(gid) != GROUP_LABELS[gid] for gid in GROUPS):
            raise RuntimeError(f"Atlas graphic-group metadata changed: {labels}")
        info = {
            "atlas_id": ATLAS_ID,
            "name": atlas.get("name"),
            "structure_graph_id": GRAPH_ID,
            "structure_graph_name": graph.get("name"),
            "graphic_groups": [{"id": gid, "name": labels[gid]} for gid in GROUPS],
        }
        criteria = (
            "model::AtlasImage,rma::criteria,[annotated$eqtrue],"
            f"atlas_data_set(atlases[id$eq{ATLAS_ID}]),"
            f"alternate_images[image_type$eq'{ATLAS_IMAGE_TYPE}'],"
            "rma::options[order$eq'section_number'][num_rows$eqall]"
        )
        plates = [
            AtlasPlate(
                int(item["id"]),
                int(item["section_number"]),
                opt_int(item.get("width")),
                opt_int(item.get("height")),
                opt_float(item.get("resolution")),
            )
            for item in values(api_json(self.session, criteria).get("msg"))
        ]
        plates.sort(key=lambda item: item.section_number)
        if len(plates) != EXPECTED_SVGS:
            raise RuntimeError(f"Atlas returned {len(plates)} plates, expected 106")
        return info, plates


class AllenStructureGraphProvider:
    name = "AllenStructureGraphProvider"
    url = f"{API_BASE}/structure_graph_download/{GRAPH_ID}.json"

    def __init__(self, session: requests.Session):
        self.session = session

    def fetch(self) -> bytes:
        response = self.session.get(self.url, timeout=(20, 180))
        response.raise_for_status()
        parse_ontology(response.content)
        return response.content


def load_stored_inventory(paths: Paths) -> dict[str, Any]:
    if paths.dataset.is_file():
        try:
            stored = json.loads(paths.dataset.read_text()).get("accepted_api_inventory")
            if isinstance(stored, Mapping):
                return dict(stored)
        except (OSError, ValueError):
            pass
    return baseline_inventory()


def gate_inventory(
    stored: Mapping[str, Any],
    observed: Mapping[str, Any],
    report_path: Path | None,
    accepted_digest: str | None,
) -> None:
    if inventory_core(stored) == inventory_core(observed):
        return
    if accepted_digest:
        if accepted_digest != observed["inventory_sha256"]:
            raise RuntimeError(
                f"Accepted inventory digest does not match live digest {observed['inventory_sha256']}"
            )
        return
    print("Stored inventory:\n" + json.dumps(stored, indent=2))
    print("Observed inventory:\n" + json.dumps(observed, indent=2))
    if report_path:
        atomic_json(report_path.expanduser().resolve(), observed)
    print("Status: API_INVENTORY_CHANGED")
    print(
        "Corrective action: review and rerun with --accept-api-inventory-sha256 "
        + observed["inventory_sha256"]
    )
    raise APIInventoryChanged("API_INVENTORY_CHANGED")


def resolve_mappings(
    plates: Sequence[AtlasPlate],
    nissl: Sequence[Section],
    cached: Mapping[int, int] | None = None,
    filenames: set[int] | None = None,
) -> list[AtlasPlate]:
    cached, filenames = cached or {}, filenames or set()
    by_id = {item.section_id: item for item in nissl}
    by_number = defaultdict(list)
    for item in nissl:
        by_number[item.section_number].append(item)
    result = []
    for plate in plates:
        direct, prior = by_id.get(plate.atlas_image_id), cached.get(
            plate.atlas_image_id
        )
        if prior is not None and prior not in by_id:
            raise RuntimeError(f"Unknown cached Nissl ID {prior}")
        if direct and prior is not None and direct.section_id != prior:
            raise RuntimeError("Conflicting non-null plate mappings")
        if direct:
            if direct.section_number != plate.section_number:
                raise RuntimeError("Atlas/Nissl section-number conflict")
            nid, status = direct.section_id, "exact_id"
        elif prior is not None:
            if by_id[prior].section_number != plate.section_number:
                raise RuntimeError("Cached mapping section conflict")
            nid, status = prior, "repaired_unique_section"
        elif len(by_number[plate.section_number]) == 1:
            nid, status = (
                by_number[plate.section_number][0].section_id,
                "repaired_unique_section",
            )
        elif len(by_number[plate.section_number]) > 1:
            raise RuntimeError(
                f"Multiple Nissl candidates for plate {plate.atlas_image_id}"
            )
        elif plate.section_number in filenames:
            nid, status = None, "filename_only"
        else:
            nid, status = None, "unresolved"
        result.append(
            AtlasPlate(
                plate.atlas_image_id,
                plate.section_number,
                plate.width,
                plate.height,
                plate.resolution,
                nid,
                status,
            )
        )
    return result


def parse_ontology(content: bytes) -> Mapping[str, Any]:
    try:
        payload = json.loads(content)
    except (ValueError, UnicodeDecodeError) as exc:
        raise RuntimeError("Invalid ontology JSON") from exc
    roots = values(payload.get("msg")) if isinstance(payload, Mapping) else []
    if payload.get("success") is False or len(roots) != 1:
        raise RuntimeError("Expected one successful ontology root")
    return payload


def flatten_ontology(payload: Mapping[str, Any]) -> list[Structure]:
    result, seen = [], set()

    def visit(node: Mapping[str, Any], parent: int | None) -> None:
        sid = int(node["id"])
        if sid in seen:
            raise RuntimeError(f"Duplicate structure {sid}")
        seen.add(sid)
        pid = opt_int(node.get("parent_structure_id")) or parent
        color = node.get("color_hex_triplet")
        result.append(
            Structure(
                sid,
                str(node.get("acronym", "")),
                str(node.get("name", "")),
                pid,
                None if not color else "#" + str(color).lstrip("#").upper(),
                GRAPH_ID,
                (
                    None
                    if not node.get("structure_id_path")
                    else str(node["structure_id_path"])
                ),
            )
        )
        for child in values(node.get("children")):
            if not isinstance(child, Mapping):
                raise RuntimeError("Non-object ontology child")
            visit(child, sid)

    visit(values(payload["msg"])[0], None)
    return result


def inspect_svg(
    content: bytes,
) -> tuple[list[int], dict[int, int], dict[int, set[int]]]:
    try:
        root = ET.fromstring(content)
    except ET.ParseError as exc:
        raise RuntimeError("Unparseable SVG") from exc
    if root.tag.rsplit("}", 1)[-1] != "svg":
        raise RuntimeError("Missing SVG root")
    present, counts, ids = [], Counter(), defaultdict(set)
    for group in root.iter():
        gid = opt_int(group.attrib.get("graphic_group_label_id"))
        if gid is None:
            continue
        if (
            gid not in GROUPS
            or group.attrib.get("graphic_group_label") != GROUP_LABELS[gid]
        ):
            raise RuntimeError(f"Unexpected SVG graphic group {gid}")
        if gid not in present:
            present.append(gid)
        for element in group.iter():
            sid = opt_int(element.attrib.get("structure_id"))
            if sid is not None:
                counts[gid] += 1
                ids[gid].add(sid)
    present.sort(key=GROUPS.index)
    return present, dict(counts), ids


def load_manifest(path: Path) -> dict[str, Artifact]:
    if not path.is_file():
        return {}
    result = {}
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if tuple(reader.fieldnames or ()) != MANIFEST_FIELDS:
            raise RuntimeError("Noncanonical manifest columns")
        for row in reader:
            artifact = Artifact(
                row["kind"],
                row["series_or_layer"],
                opt_int(row["section_number"]),
                opt_int(row["allen_section_image_id"]),
                opt_int(row["allen_atlas_image_id"]),
                row["path"],
                opt_int(row["width_px"]),
                opt_int(row["height_px"]),
                opt_float(row["pixel_size_um"]),
                row["sha256"],
                None if row["source_url"] == "n/a" else row["source_url"],
                row["status"],
                row["source_provider"],
                opt_int(row["allen_data_set_id"]),
                opt_int(row["treatment_id"]),
                (
                    None
                    if row["graphic_groups_present"] == "n/a"
                    else row["graphic_groups_present"]
                ),
                opt_int(row["matching_nissl_section_image_id"]),
                None if row["mapping_status"] == "n/a" else row["mapping_status"],
            )
            if artifact.path in result:
                raise RuntimeError(f"Duplicate manifest path {artifact.path}")
            result[artifact.path] = artifact
    return result


def save_manifest(path: Path, items: Mapping[str, Artifact]) -> None:
    ordered = sorted(
        items.values(),
        key=lambda item: (
            item.kind,
            item.series_or_layer,
            item.section_number or -1,
            item.path,
        ),
    )
    atomic_tsv(path, MANIFEST_FIELDS, (asdict(item) for item in ordered))


def image_dimensions(path: Path) -> tuple[int, int]:
    try:
        with Image.open(path) as image:
            image.load()
            return image.size
    except (OSError, UnidentifiedImageError) as exc:
        raise RuntimeError(f"Unreadable image {path}") from exc


def image_path(paths: Paths, section: Section) -> Path:
    return paths.image_dir(section.series) / f"image_{section.section_number:04d}.jpg"


def source_url(section: Section, level: int) -> str:
    return (
        f"{API_BASE}/image_download/{section.section_id}?downsample={level}&quality=100"
    )


def artifact_image(
    paths: Paths, section: Section, path: Path, status: str, level: int
) -> Artifact:
    width, height = image_dimensions(path)
    pixel = (
        None
        if section.resolution is None
        else section.resolution * (2 ** (level if level >= 0 else 0))
    )
    return Artifact(
        "histology_jpeg",
        section.series,
        section.section_number,
        section.section_id,
        None,
        path.relative_to(paths.root).as_posix(),
        width,
        height,
        pixel,
        sha_file(path),
        source_url(section, level),
        status,
        AllenSectionDataSetProvider.name,
        section.data_set_id,
        section.treatment_id,
    )


def quarantine(path: Path, paths: Paths, run_id: str) -> Path:
    destination = paths.root / "quarantine" / run_id / path.relative_to(paths.root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(path, destination)
    return destination


def acquire_image(
    session: requests.Session,
    paths: Paths,
    section: Section,
    manifest: Mapping[str, Artifact],
    downsample: int,
    mode: str,
    overwrite: bool,
    run_id: str,
) -> Artifact:
    path = image_path(paths, section)
    relative = path.relative_to(paths.root).as_posix()
    prior = manifest.get(relative)
    if (
        path.is_file()
        and not overwrite
        and prior
        and prior.status in VERIFIED
        and sha_file(path) == prior.sha256
    ):
        return artifact_image(
            paths, section, path, "manifest-verified-existing", downsample
        )
    level = downsample if mode == "allen-direct" else downsample - 1
    response = session.get(
        f"{API_BASE}/image_download/{section.section_id}",
        params={"downsample": level, "quality": 100},
        timeout=(20, 300),
    )
    response.raise_for_status()
    try:
        with Image.open(io.BytesIO(response.content)) as image:
            image.load()
            if mode == "allen-direct":
                output = response.content
            else:
                resized = image.convert("RGB").resize(
                    (max(1, image.width // 2), max(1, image.height // 2)),
                    Image.Resampling.BICUBIC,
                )
                buffer = io.BytesIO()
                resized.save(buffer, format="JPEG", quality=100, subsampling=0)
                output = buffer.getvalue()
    except (OSError, UnidentifiedImageError) as exc:
        raise RuntimeError("Allen returned unreadable JPEG") from exc
    digest, status = sha_bytes(output), "downloaded"
    if path.is_file() and not overwrite:
        if sha_file(path) == digest:
            status = "verified-existing"
        else:
            quarantine(path, paths, run_id)
            atomic_bytes(path, output)
            status = "redownloaded-after-quarantine"
    else:
        atomic_bytes(path, output)
    return artifact_image(paths, section, path, status, downsample)


def acquire_svg(
    session: requests.Session,
    paths: Paths,
    plate: AtlasPlate,
    manifest: Mapping[str, Artifact],
    overwrite: bool,
    run_id: str,
) -> Artifact:
    path = paths.svg_dir / f"seg_{plate.section_number:04d}.svg"
    relative = path.relative_to(paths.root).as_posix()
    prior = manifest.get(relative)
    group_query = ",".join(map(str, GROUPS))
    if (
        path.is_file()
        and not overwrite
        and prior
        and prior.status in VERIFIED
        and sha_file(path) == prior.sha256
    ):
        present, _, _ = inspect_svg(path.read_bytes())
        status = "manifest-verified-existing"
    else:
        response = session.get(
            f"{API_BASE}/svg_download/{plate.atlas_image_id}",
            params={"groups": group_query},
            timeout=(20, 300),
        )
        response.raise_for_status()
        content = response.content
        present, _, _ = inspect_svg(content)
        digest = sha_bytes(content)
        status = "downloaded"
        if path.is_file() and not overwrite:
            if sha_file(path) == digest:
                status = "verified-existing"
            else:
                quarantine(path, paths, run_id)
                atomic_bytes(path, content)
                status = "redownloaded-after-quarantine"
        else:
            atomic_bytes(path, content)
    return Artifact(
        "annotation_svg",
        "atlas",
        plate.section_number,
        None,
        plate.atlas_image_id,
        relative,
        plate.width,
        plate.height,
        plate.resolution,
        sha_file(path),
        f"{API_BASE}/svg_download/{plate.atlas_image_id}?groups={group_query}",
        status,
        AllenAtlasPlateProvider.name,
        graphic_groups_present=";".join(map(str, present)) or None,
        matching_nissl_section_image_id=plate.nissl_id,
        mapping_status=plate.mapping_status,
    )


def acquire_ontology(
    provider: AllenStructureGraphProvider,
    paths: Paths,
    manifest: Mapping[str, Artifact],
    overwrite: bool,
    run_id: str,
) -> Artifact:
    path = paths.ontology
    relative = path.relative_to(paths.root).as_posix()
    prior = manifest.get(relative)
    if (
        path.is_file()
        and not overwrite
        and prior
        and prior.status in VERIFIED
        and sha_file(path) == prior.sha256
    ):
        parse_ontology(path.read_bytes())
        status = "manifest-verified-existing"
    else:
        content = provider.fetch()
        digest = sha_bytes(content)
        status = "downloaded"
        if path.is_file() and not overwrite:
            if sha_file(path) == digest:
                status = "verified-existing"
            else:
                quarantine(path, paths, run_id)
                atomic_bytes(path, content)
                status = "redownloaded-after-quarantine"
        else:
            atomic_bytes(path, content)
    return Artifact(
        "ontology_json",
        "structure_graph_16",
        None,
        None,
        None,
        relative,
        None,
        None,
        None,
        sha_file(path),
        provider.url,
        status,
        provider.name,
    )


def legacy_provenance(paths: Paths) -> dict[str, dict[str, str]]:
    """Load the old image provenance only as migration evidence."""
    source = paths.metadata / "image_files.tsv"
    if not source.is_file():
        return {}
    with source.open(newline="") as stream:
        return {
            row["relative_output_path"]: row
            for row in csv.DictReader(stream, delimiter="\t")
        }


def legacy_plate_mappings(paths: Paths) -> dict[int, int]:
    result: dict[int, int] = {}
    source = paths.metadata / "atlas_annotations.tsv"
    if source.is_file():
        with source.open(newline="") as stream:
            for row in csv.DictReader(stream, delimiter="\t"):
                atlas_id = opt_int(row.get("atlas_image_id"))
                nissl_id = opt_int(row.get("matching_nissl_section_image_id"))
                if atlas_id is not None and nissl_id is not None:
                    result[atlas_id] = nissl_id
    if paths.secjson.is_file():
        try:
            payload = json.loads(paths.secjson.read_text())
            for row in values(payload.get("atlas_annotations")):
                if not isinstance(row, Mapping):
                    continue
                atlas_id = opt_int(row.get("atlas_image_id"))
                nissl_id = opt_int(row.get("matching_nissl_section_image_id"))
                if atlas_id is not None and nissl_id is not None:
                    prior = result.get(atlas_id)
                    if prior is not None and prior != nissl_id:
                        raise RuntimeError(
                            f"Conflicting cached mappings for AtlasImage {atlas_id}"
                        )
                    result[atlas_id] = nissl_id
        except (OSError, ValueError) as exc:
            raise RuntimeError(
                f"Cannot read legacy mapping evidence: {paths.secjson}"
            ) from exc
    return result


def scan_existing(
    paths: Paths,
    sections: Mapping[str, Sequence[Section]],
    plates: Sequence[AtlasPlate],
    downsample: int,
) -> dict[str, Artifact]:
    """Build a complete manifest view without letting limited runs truncate it."""
    try:
        result = load_manifest(paths.manifest)
    except RuntimeError:
        result = {}
    provenance = legacy_provenance(paths)
    for series, records in sections.items():
        for section in records:
            path = image_path(paths, section)
            if not path.is_file():
                continue
            relative = path.relative_to(paths.root).as_posix()
            prior = result.get(relative)
            legacy = provenance.get(relative, {})
            digest = sha_file(path)
            verified = (
                prior is not None
                and prior.sha256 == digest
                and prior.status in VERIFIED
            ) or (
                legacy.get("sha256") == digest
                and legacy.get("verified_mode") == "allen-direct"
                and legacy.get("status")
                in {
                    "downloaded",
                    "verified-existing",
                    "manifest-verified-existing",
                }
            )
            status = "manifest-verified-existing" if verified else "existing-unverified"
            result[relative] = artifact_image(paths, section, path, status, downsample)
    by_section = {plate.section_number: plate for plate in plates}
    for path in sorted(paths.svg_dir.glob("*.svg")) if paths.svg_dir.is_dir() else []:
        number = opt_int(path.stem.rsplit("_", 1)[-1])
        plate = by_section.get(number) if number is not None else None
        if plate is None:
            continue
        present, _, _ = inspect_svg(path.read_bytes())
        relative = path.relative_to(paths.root).as_posix()
        prior = result.get(relative)
        status = (
            "manifest-verified-existing"
            if prior and prior.sha256 == sha_file(path) and prior.status in VERIFIED
            else "existing-unverified"
        )
        result[relative] = Artifact(
            "annotation_svg",
            "atlas",
            plate.section_number,
            None,
            plate.atlas_image_id,
            relative,
            plate.width,
            plate.height,
            plate.resolution,
            sha_file(path),
            f"{API_BASE}/svg_download/{plate.atlas_image_id}?groups={','.join(map(str, GROUPS))}",
            status,
            AllenAtlasPlateProvider.name,
            graphic_groups_present=";".join(map(str, present)) or None,
            matching_nissl_section_image_id=plate.nissl_id,
            mapping_status=plate.mapping_status,
        )
    if paths.ontology.is_file():
        parse_ontology(paths.ontology.read_bytes())
        relative = paths.ontology.relative_to(paths.root).as_posix()
        prior = result.get(relative)
        status = (
            "manifest-verified-existing"
            if prior
            and prior.sha256 == sha_file(paths.ontology)
            and prior.status in VERIFIED
            else "existing-unverified"
        )
        result[relative] = Artifact(
            "ontology_json",
            "structure_graph_16",
            None,
            None,
            None,
            relative,
            None,
            None,
            None,
            sha_file(paths.ontology),
            AllenStructureGraphProvider.url,
            status,
            AllenStructureGraphProvider.name,
        )
    for series in ("nissl", "pv"):
        mask_dir = paths.root / SERIES_ROOTS[series] / "masks_orig"
        if not mask_dir.is_dir():
            continue
        for path in sorted(mask_dir.glob("*.png")):
            relative = path.relative_to(paths.root).as_posix()
            width, height = image_dimensions(path)
            result[relative] = Artifact(
                "tissue_mask",
                series,
                opt_int(path.stem.rsplit("_", 1)[-1]),
                None,
                None,
                relative,
                width,
                height,
                None,
                sha_file(path),
                None,
                "existing",
                "download_allen mask helper",
            )
    allowed = {
        "histology_jpeg",
        "annotation_svg",
        "ontology_json",
        "tissue_mask",
        "derivative",
    }
    return {key: value for key, value in result.items() if value.kind in allowed}


def save_structures(path: Path, structures: Sequence[Structure]) -> None:
    atomic_tsv(path, STRUCTURE_FIELDS, (asdict(item) for item in structures))


def load_structures(path: Path) -> list[Structure]:
    if not path.is_file():
        raise RuntimeError(f"Missing canonical structure table: {path}")
    result: list[Structure] = []
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if tuple(reader.fieldnames or ()) != STRUCTURE_FIELDS:
            raise RuntimeError("Noncanonical structures.tsv columns")
        for row in reader:
            result.append(
                Structure(
                    int(row["structure_id"]),
                    row["acronym"],
                    row["name"],
                    opt_int(row["parent_structure_id"]),
                    None if row["color_hex"] == "n/a" else row["color_hex"],
                    int(row["structure_graph_id"]),
                    (
                        None
                        if row["structure_id_path"] == "n/a"
                        else row["structure_id_path"]
                    ),
                )
            )
    return result


def write_legacy_exports(
    paths: Paths,
    sections: Mapping[str, Sequence[Section]],
    plates: Sequence[AtlasPlate],
    downsample: int,
    mode: str,
) -> None:
    def section_row(item: Section, stain: str) -> dict[str, Any]:
        return {
            "stain": stain,
            "section_id": item.section_id,
            "section_number": item.section_number,
            "annotated": item.annotated,
            "resolution_um_per_pixel": item.resolution,
            "width_px": item.width,
            "height_px": item.height,
        }

    annotation_rows = [
        {
            "atlas_image_id": item.atlas_image_id,
            "section_number": item.section_number,
            "matching_nissl_section_image_id": item.nissl_id,
            "mapping_status": item.mapping_status,
        }
        for item in plates
    ]
    payload = {
        "specimen_id": SPECIMEN_ID,
        "atlas_id": ATLAS_ID,
        "atlas_image_type": ATLAS_IMAGE_TYPE,
        "groups": list(GROUPS),
        "downsample": downsample,
        "image_download_mode": mode,
        "histology_sampling": {
            "paper_nominal_scan_resolution_um_per_pixel": 1.0,
            "physical_section_thickness_um": 50.0,
            "nissl_nominal_section_spacing_um": 200.0,
            "pv_nominal_section_spacing_um": 400.0,
            "server_downsample_level": downsample,
            "server_linear_downsample_factor": 2**downsample,
            "nominal_effective_resolution_um_per_pixel": float(2**downsample),
        },
        "nissl": [section_row(item, "nissl") for item in sections["nissl"]],
        # Legacy IHC means parvalbumin, never SMI-32.
        "ihc": [section_row(item, "ihc") for item in sections["pv"]],
        "atlas_annotations": annotation_rows,
    }
    atomic_json(paths.secjson, payload)
    mat_payload = {
        "specimen_id": SPECIMEN_ID,
        "atlas_id": ATLAS_ID,
        "nissl_section_id": np.asarray(
            [item.section_id for item in sections["nissl"]], dtype=np.int64
        ),
        "nissl_section_number": np.asarray(
            [item.section_number for item in sections["nissl"]], dtype=np.int64
        ),
        "ihc_section_id": np.asarray(
            [item.section_id for item in sections["pv"]], dtype=np.int64
        ),
        "ihc_section_number": np.asarray(
            [item.section_number for item in sections["pv"]], dtype=np.int64
        ),
        "atlas_image_id": np.asarray(
            [item.atlas_image_id for item in plates], dtype=np.int64
        ),
        "atlas_section_number": np.asarray(
            [item.section_number for item in plates], dtype=np.int64
        ),
        "matching_nissl_section_image_id": np.asarray(
            [item.nissl_id or -1 for item in plates], dtype=np.int64
        ),
    }
    paths.secmat.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{paths.secmat.name}.", dir=paths.secmat.parent
    )
    os.close(descriptor)
    try:
        savemat(temporary, mat_payload, appendmat=False)
        os.replace(temporary, paths.secmat)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def dataset_payload(
    accepted_inventory: Mapping[str, Any],
    atlas_info: Mapping[str, Any],
    mode: str,
    downsample: int,
    artifacts: Mapping[str, Artifact],
) -> dict[str, Any]:
    acquired = Counter(
        item.series_or_layer
        for item in artifacts.values()
        if item.kind == "histology_jpeg" and item.status in VERIFIED
    )
    coverage = {
        series: {
            "acquired": acquired[series],
            "published": PUBLISHED[series],
            "unavailable": max(0, PUBLISHED[series] - acquired[series]),
        }
        for series in SERIES_LABELS
    }
    return {
        "dataset_title": "Allen/Ding et al. 2016 single-donor human brain atlas raw acquisition",
        "citation": "Ding et al., A comprehensive human brain reference atlas, 2016",
        "doi": "10.1002/cne.24080",
        "allen_specimen_id": SPECIMEN_ID,
        "allen_donor_id": DONOR_ID,
        "allen_atlas_id": ATLAS_ID,
        "allen_structure_graph_id": GRAPH_ID,
        "allen_structure_graph_name": atlas_info.get("structure_graph_name"),
        "allen_api_base": API_BASE,
        "accepted_api_inventory": dict(accepted_inventory),
        "published_corpus": {
            "counts": {**PUBLISHED, "total": sum(PUBLISHED.values())},
            "coverage": {
                **coverage,
                "total": {
                    "acquired": sum(acquired.values()),
                    "published": 1356,
                    "unavailable": max(0, 1356 - sum(acquired.values())),
                },
            },
        },
        "providers": {
            "nissl": {
                "provider": AllenSectionDataSetProvider.name,
                "available": bool(accepted_inventory["series"]["nissl"]["image_count"]),
            },
            "pv": {
                "provider": AllenSectionDataSetProvider.name,
                "available": bool(accepted_inventory["series"]["pv"]["image_count"]),
                "legacy_directory": "ihc/",
            },
            "smi32": {
                "provider": (
                    AllenSectionDataSetProvider.name
                    if accepted_inventory["series"]["smi32"]["image_count"]
                    else None
                ),
                "available": bool(accepted_inventory["series"]["smi32"]["image_count"]),
                "published_denominator": 338,
                "finding": f"No qualifying official provider found in bounded search on {DISCOVERY_DATE}; reassess when sources change.",
            },
        },
        "source_discovery": {
            "searched_on": DISCOVERY_DATE,
            "scope": "bounded official-source search",
            "smi32_result": "no qualifying provider found",
            "permanent_unavailability_claim": False,
        },
        "graphic_groups": atlas_info["graphic_groups"],
        "nominal_native_pixel_size_um": 1.0,
        "downloaded_nominal_pixel_size_um": float(2**downsample),
        "physical_section_thickness_um": 50.0,
        "nominal_series_spacing_um": {"nissl": 200.0, "pv": 400.0, "smi32": 400.0},
        "annotated_atlas_plate_spacing": "variable, approximately 0.4–3.4 mm; distinct from section thickness and series spacing",
        "image_download_mode": mode,
        "server_downsample_level": downsample,
        "coordinate_space": "Raw histology JPEGs and SVG annotations use local 2-D section-image coordinates; no origin, 3-D orientation, AP coordinate, or MRI affine is asserted.",
        "reuse_terms_url": "https://alleninstitute.org/legal/terms-use/",
        "software": {"name": "download_allen.py", "version": VERSION},
        "updated_at_utc": now(),
    }


def create_masks(
    paths: Paths, artifacts: dict[str, Artifact], series: Sequence[str]
) -> None:
    for name in series:
        if name == "smi32":
            continue
        generator = generate_nissl_mask if name == "nissl" else generate_ihc_mask
        for item in [
            x
            for x in artifacts.values()
            if x.kind == "histology_jpeg" and x.series_or_layer == name
        ]:
            source = paths.root / item.path
            destination = (
                paths.root
                / SERIES_ROOTS[name]
                / "masks_orig"
                / f"mask_{item.section_number:04d}.png"
            )
            if not destination.is_file():
                with Image.open(source) as image:
                    mask = generator(np.asarray(image.convert("RGB")))
                _atomic_save_pil(Image.fromarray(mask), destination, format="PNG")
            width, height = image_dimensions(destination)
            relative = destination.relative_to(paths.root).as_posix()
            artifacts[relative] = Artifact(
                "tissue_mask",
                name,
                item.section_number,
                item.allen_section_image_id,
                None,
                relative,
                width,
                height,
                item.pixel_size_um,
                sha_file(destination),
                item.path,
                "generated",
                "download_allen mask helper",
                item.allen_data_set_id,
                item.treatment_id,
            )


def failure(
    category: str, affected: str, expected: Any, observed: Any, action: str
) -> dict[str, Any]:
    return {
        "category": category,
        "affected": affected,
        "expected": expected,
        "observed": observed,
        "corrective_action": action,
    }


def validate_dataset(paths: Paths, allow_superseded: bool = False) -> dict[str, Any]:
    """Validate entirely from disk. The caller controls whether the JSON report is written."""
    failures: list[dict[str, Any]] = []
    try:
        dataset = json.loads(paths.dataset.read_text())
    except (OSError, ValueError) as exc:
        return {
            "status": "FAIL",
            "failures": [
                failure(
                    "metadata",
                    str(paths.dataset),
                    "readable JSON",
                    str(exc),
                    "run acquisition to create dataset.json",
                )
            ],
        }
    try:
        manifest = load_manifest(paths.manifest)
    except (OSError, RuntimeError) as exc:
        return {
            "status": "FAIL",
            "failures": [
                failure(
                    "metadata",
                    str(paths.manifest),
                    "canonical manifest",
                    str(exc),
                    "regenerate manifest.tsv",
                )
            ],
        }
    try:
        structures = load_structures(paths.structures)
    except (OSError, RuntimeError, ValueError) as exc:
        return {
            "status": "FAIL",
            "failures": [
                failure(
                    "ontology",
                    str(paths.structures),
                    "graph-16 table",
                    str(exc),
                    "reacquire ontology",
                )
            ],
        }
    canonical = {"dataset.json", "manifest.tsv", "structures.tsv"}
    actual_metadata = {item.name for item in paths.metadata.iterdir() if item.is_file()}
    extras = sorted(actual_metadata - canonical)
    if extras and not allow_superseded:
        failures.append(
            failure(
                "metadata",
                "metadata/",
                sorted(canonical),
                sorted(actual_metadata),
                "complete metadata migration",
            )
        )
    for forbidden in (
        "metadata/dataset.json",
        "metadata/manifest.tsv",
        "metadata/structures.tsv",
        "secInfo.json",
        "secInfo.mat",
    ):
        if forbidden in manifest:
            failures.append(
                failure(
                    "manifest",
                    forbidden,
                    "excluded",
                    "present",
                    "remove metadata/compatibility rows",
                )
            )
    accepted = dataset.get("accepted_api_inventory", {})
    accepted_series = (
        accepted.get("series", {}) if isinstance(accepted, Mapping) else {}
    )
    if (
        dataset.get("allen_specimen_id") != SPECIMEN_ID
        or dataset.get("allen_atlas_id") != ATLAS_ID
        or dataset.get("allen_structure_graph_id") != GRAPH_ID
    ):
        failures.append(
            failure(
                "metadata_identity",
                "dataset.json",
                {"specimen": SPECIMEN_ID, "atlas": ATLAS_ID, "graph": GRAPH_ID},
                {
                    "specimen": dataset.get("allen_specimen_id"),
                    "atlas": dataset.get("allen_atlas_id"),
                    "graph": dataset.get("allen_structure_graph_id"),
                },
                "review and regenerate canonical metadata",
            )
        )
    expected_groups = [{"id": gid, "name": GROUP_LABELS[gid]} for gid in GROUPS]
    if dataset.get("graphic_groups") != expected_groups:
        failures.append(
            failure(
                "graphic_group_catalog",
                "dataset.json",
                expected_groups,
                dataset.get("graphic_groups"),
                "re-query the selected atlas and regenerate dataset.json",
            )
        )
    try:
        digest = sha_bytes(
            json.dumps(
                inventory_core(accepted), sort_keys=True, separators=(",", ":")
            ).encode()
        )
        if accepted.get("inventory_sha256") != digest:
            failures.append(
                failure(
                    "inventory_digest",
                    "dataset.json accepted_api_inventory",
                    digest,
                    accepted.get("inventory_sha256"),
                    "review the accepted inventory and regenerate dataset.json",
                )
            )
    except (AttributeError, TypeError, ValueError) as exc:
        failures.append(
            failure(
                "inventory_schema",
                "dataset.json accepted_api_inventory",
                "canonical inventory object",
                str(exc),
                "regenerate dataset.json from reviewed inventory",
            )
        )
    count = Counter()
    checksum_failures = 0
    unreadable = 0
    svg_rows: list[tuple[Artifact, bytes]] = []
    section_ids: dict[str, set[int]] = {series: set() for series in SERIES_LABELS}
    section_numbers: dict[str, set[int]] = {series: set() for series in SERIES_LABELS}
    nissl_by_id: dict[int, Artifact] = {}
    atlas_ids: set[int] = set()
    for item in manifest.values():
        path = (paths.root / item.path).resolve()
        try:
            path.relative_to(paths.root)
        except ValueError:
            failures.append(
                failure(
                    "manifest_path",
                    item.path,
                    "path within dataset root",
                    str(path),
                    "remove unsafe manifest row",
                )
            )
            continue
        if not path.is_file():
            failures.append(
                failure(
                    "artifact",
                    item.path,
                    "file exists",
                    "missing",
                    "resume acquisition",
                )
            )
            continue
        observed_hash = sha_file(path)
        if observed_hash != item.sha256:
            checksum_failures += 1
            failures.append(
                failure(
                    "checksum",
                    item.path,
                    item.sha256,
                    observed_hash,
                    "review, quarantine, and redownload",
                )
            )
        if item.kind in {"histology_jpeg", "tissue_mask"}:
            try:
                dimensions = image_dimensions(path)
                if item.width_px is not None and dimensions != (
                    item.width_px,
                    item.height_px,
                ):
                    failures.append(
                        failure(
                            "dimensions",
                            item.path,
                            (item.width_px, item.height_px),
                            dimensions,
                            "regenerate manifest or reacquire",
                        )
                    )
            except RuntimeError as exc:
                unreadable += 1
                failures.append(
                    failure(
                        "readability",
                        item.path,
                        "decodable image",
                        str(exc),
                        "redownload or regenerate",
                    )
                )
        if item.kind == "histology_jpeg" and item.status in VERIFIED:
            count[item.series_or_layer] += 1
            series = item.series_or_layer
            identity = accepted_series.get(series, {})
            if series not in SERIES_LABELS:
                failures.append(
                    failure(
                        "series",
                        item.path,
                        sorted(SERIES_LABELS),
                        series,
                        "repair manifest series",
                    )
                )
            else:
                if item.allen_section_image_id in section_ids[series]:
                    failures.append(
                        failure(
                            "duplicate_section_id",
                            item.path,
                            "unique ID",
                            item.allen_section_image_id,
                            "repair manifest",
                        )
                    )
                if item.section_number in section_numbers[series]:
                    failures.append(
                        failure(
                            "duplicate_section_number",
                            item.path,
                            "unique number",
                            item.section_number,
                            "review API inventory",
                        )
                    )
                section_ids[series].add(item.allen_section_image_id)
                section_numbers[series].add(item.section_number)
                if item.treatment_id != opt_int(
                    identity.get("treatment_id")
                ) or item.allen_data_set_id not in set(
                    identity.get("data_set_ids", [])
                ):
                    failures.append(
                        failure(
                            "histology_identity",
                            item.path,
                            {
                                "treatment_id": identity.get("treatment_id"),
                                "data_set_ids": identity.get("data_set_ids"),
                            },
                            {
                                "treatment_id": item.treatment_id,
                                "data_set_id": item.allen_data_set_id,
                            },
                            "reconcile manifest with the accepted API inventory",
                        )
                    )
                if series == "nissl" and item.allen_section_image_id is not None:
                    nissl_by_id[item.allen_section_image_id] = item
        if item.kind == "annotation_svg":
            if item.allen_atlas_image_id in atlas_ids:
                failures.append(
                    failure(
                        "duplicate_atlas_id",
                        item.path,
                        "unique AtlasImage ID",
                        item.allen_atlas_image_id,
                        "repair manifest",
                    )
                )
            atlas_ids.add(item.allen_atlas_image_id)
            try:
                svg_rows.append((item, path.read_bytes()))
            except OSError as exc:
                unreadable += 1
                failures.append(
                    failure(
                        "readability", item.path, "readable SVG", str(exc), "redownload"
                    )
                )
    inventoried = set(manifest)
    discovered: set[str] = set()
    for directory, pattern in (
        (paths.image_dir("nissl"), "*.jpg"),
        (paths.image_dir("pv"), "*.jpg"),
        (paths.image_dir("smi32"), "*.jpg"),
        (paths.svg_dir, "*.svg"),
        (paths.root / "nissl/masks_orig", "*.png"),
        (paths.root / "ihc/masks_orig", "*.png"),
        (paths.root / "smi32/masks_orig", "*.png"),
    ):
        if directory.is_dir():
            discovered.update(
                path.relative_to(paths.root).as_posix()
                for path in directory.glob(pattern)
            )
    if paths.ontology.is_file():
        discovered.add(paths.ontology.relative_to(paths.root).as_posix())
    for relative in sorted(discovered - inventoried):
        failures.append(
            failure(
                "unmanifested_artifact",
                relative,
                "manifest row",
                "absent",
                "verify and add artifact to manifest",
            )
        )
    expected = {
        series: int(accepted_series.get(series, {}).get("image_count", -1))
        for series in SERIES_LABELS
    }
    for series in SERIES_LABELS:
        if count[series] != expected[series]:
            failures.append(
                failure(
                    "api_count",
                    series,
                    expected[series],
                    count[series],
                    "resume acquisition and verify files",
                )
            )
    ontology_rows = [item for item in manifest.values() if item.kind == "ontology_json"]
    if (
        len(ontology_rows) != 1
        or ontology_rows[0].path != "ontology/structure_graph_16.json"
    ):
        failures.append(
            failure(
                "ontology",
                "manifest",
                "one graph-16 raw ontology",
                [x.path for x in ontology_rows],
                "reacquire ontology",
            )
        )
    if any(item.structure_graph_id != GRAPH_ID for item in structures):
        failures.append(
            failure(
                "ontology",
                "structures.tsv",
                GRAPH_ID,
                "other graph ID",
                "regenerate strictly from graph 16",
            )
        )
    structure_ids = {item.structure_id for item in structures}
    if len(structure_ids) != len(structures):
        failures.append(
            failure(
                "ontology",
                "structures.tsv",
                "unique structure IDs",
                "duplicates",
                "regenerate structures.tsv",
            )
        )
    if paths.ontology.is_file():
        try:
            raw_structures = flatten_ontology(
                parse_ontology(paths.ontology.read_bytes())
            )
            if raw_structures != structures:
                failures.append(
                    failure(
                        "ontology",
                        "structures.tsv",
                        "exact flattening of raw graph 16",
                        "different rows",
                        "regenerate structures.tsv",
                    )
                )
        except RuntimeError as exc:
            failures.append(
                failure(
                    "ontology",
                    str(paths.ontology),
                    "valid graph-16 JSON",
                    str(exc),
                    "redownload ontology",
                )
            )
    group_stats = {
        str(gid): {"plates_present": 0, "structure_paths": 0, "unique_structure_ids": 0}
        for gid in GROUPS
    }
    group_ids: dict[int, set[int]] = {gid: set() for gid in GROUPS}
    unresolved_ids: set[int] = set()
    mappings = 0
    for item, content in svg_rows:
        try:
            present, path_counts, ids = inspect_svg(content)
        except RuntimeError as exc:
            unreadable += 1
            failures.append(
                failure(
                    "svg_parse", item.path, "parseable SVG", str(exc), "redownload SVG"
                )
            )
            continue
        expected_present = ";".join(map(str, present)) or None
        if item.graphic_groups_present != expected_present:
            failures.append(
                failure(
                    "svg_groups",
                    item.path,
                    expected_present,
                    item.graphic_groups_present,
                    "regenerate manifest",
                )
            )
        for gid in GROUPS:
            if gid in present:
                group_stats[str(gid)]["plates_present"] += 1
            group_stats[str(gid)]["structure_paths"] += path_counts.get(gid, 0)
            group_ids[gid].update(ids.get(gid, set()))
            unresolved = sorted(ids.get(gid, set()) - structure_ids)
            unresolved_ids.update(unresolved)
            for sid in unresolved:
                failures.append(
                    failure(
                        "unresolved_structure_id",
                        f"{item.path}:{sid}",
                        "ID in graph 16",
                        sid,
                        "review ontology or source SVG",
                    )
                )
        if (
            item.matching_nissl_section_image_id is not None
            and item.mapping_status in {"exact_id", "repaired_unique_section"}
        ):
            matched = nissl_by_id.get(item.matching_nissl_section_image_id)
            if matched is None or matched.section_number != item.section_number:
                failures.append(
                    failure(
                        "plate_mapping",
                        item.path,
                        "Nissl manifest row with the same section number",
                        item.matching_nissl_section_image_id,
                        "repair the atlas-to-Nissl mapping",
                    )
                )
            else:
                mappings += 1
        elif item.mapping_status == "filename_only":
            failures.append(
                failure(
                    "plate_mapping",
                    item.path,
                    "metadata-derived Nissl ID",
                    "filename_only",
                    "review section metadata",
                )
            )
        else:
            failures.append(
                failure(
                    "plate_mapping",
                    item.path,
                    "resolved Nissl ID",
                    item.mapping_status,
                    "repair mapping from Allen metadata",
                )
            )
    for gid in GROUPS:
        group_stats[str(gid)]["unique_structure_ids"] = len(group_ids[gid])
    if len(svg_rows) != EXPECTED_SVGS:
        failures.append(
            failure(
                "svg_count",
                "annotation SVGs",
                EXPECTED_SVGS,
                len(svg_rows),
                "resume SVG acquisition",
            )
        )
    if mappings != EXPECTED_SVGS:
        failures.append(
            failure(
                "plate_mapping",
                "all atlas plates",
                EXPECTED_SVGS,
                mappings,
                "repair canonical mappings",
            )
        )
    if not paths.secjson.is_file() or not paths.secmat.is_file():
        failures.append(
            failure(
                "legacy_exports",
                "secInfo.json/secInfo.mat",
                "both present",
                "missing",
                "regenerate compatibility exports",
            )
        )
    else:
        try:
            legacy = json.loads(paths.secjson.read_text())
            mat = loadmat(paths.secmat)
            legacy_counts = (
                len(values(legacy.get("nissl"))),
                len(values(legacy.get("ihc"))),
                len(values(legacy.get("atlas_annotations"))),
            )
            mat_counts = (
                np.asarray(mat["nissl_section_id"]).size,
                np.asarray(mat["ihc_section_id"]).size,
                np.asarray(mat["atlas_image_id"]).size,
            )
            target_counts = (expected["nissl"], expected["pv"], EXPECTED_SVGS)
            if legacy_counts != target_counts or mat_counts != target_counts:
                failures.append(
                    failure(
                        "legacy_exports",
                        "secInfo exports",
                        target_counts,
                        {"json": legacy_counts, "mat": mat_counts},
                        "regenerate compatibility exports",
                    )
                )
            expected_nissl_ids = sorted(section_ids["nissl"])
            expected_pv_ids = sorted(section_ids["pv"])
            json_nissl_ids = sorted(
                opt_int(row.get("section_id")) for row in values(legacy.get("nissl"))
            )
            json_pv_ids = sorted(
                opt_int(row.get("section_id")) for row in values(legacy.get("ihc"))
            )
            mat_nissl_ids = sorted(np.asarray(mat["nissl_section_id"]).ravel().tolist())
            mat_pv_ids = sorted(np.asarray(mat["ihc_section_id"]).ravel().tolist())
            if (
                json_nissl_ids != expected_nissl_ids
                or json_pv_ids != expected_pv_ids
                or mat_nissl_ids != expected_nissl_ids
                or mat_pv_ids != expected_pv_ids
            ):
                failures.append(
                    failure(
                        "legacy_exports",
                        "secInfo section identities",
                        {"nissl": expected_nissl_ids, "pv": expected_pv_ids},
                        "JSON or MAT identity arrays differ",
                        "regenerate compatibility exports from canonical metadata",
                    )
                )
        except (OSError, ValueError, KeyError) as exc:
            failures.append(
                failure(
                    "legacy_exports",
                    "secInfo exports",
                    "readable and consistent",
                    str(exc),
                    "regenerate compatibility exports",
                )
            )
    acquired_total = sum(count.values())
    published = {
        series: {
            "acquired": count[series],
            "published": PUBLISHED[series],
            "unavailable": max(0, PUBLISHED[series] - count[series]),
        }
        for series in SERIES_LABELS
    }
    return {
        "status": "PASS" if not failures else "FAIL",
        "api_acquisition": {
            **{
                series: {"observed": count[series], "expected": expected[series]}
                for series in SERIES_LABELS
            },
            "total": {"observed": acquired_total, "expected": sum(expected.values())},
        },
        "published_coverage": {
            **published,
            "total": {
                "acquired": acquired_total,
                "published": 1356,
                "unavailable": max(0, 1356 - acquired_total),
            },
            "status": "INCOMPLETE" if acquired_total < 1356 else "COMPLETE",
        },
        "atlas_svgs": {"observed": len(svg_rows), "expected": EXPECTED_SVGS},
        "graphic_groups": group_stats,
        "ontology": {
            "structures": len(structures),
            "svg_unique_structure_ids": len(set().union(*group_ids.values())),
            "unresolved_references": len(unresolved_ids),
        },
        "plate_mappings": {"observed": mappings, "expected": EXPECTED_SVGS},
        "checksum_failures": checksum_failures,
        "unreadable_files": unreadable,
        "failures": failures,
    }


def print_validation(report: Mapping[str, Any]) -> None:
    api = report.get("api_acquisition", {})
    if api:
        print("API acquisition")
        for series in SERIES_LABELS:
            row = api[series]
            print(
                f"{SERIES_LABELS[series]:<24}{row['observed']:>5} / {row['expected']}"
            )
        row = api["total"]
        print(f"{'API histology total':<24}{row['observed']:>5} / {row['expected']}")
        print(
            f"{'Status':<24}{'PASS' if all(api[s]['observed'] == api[s]['expected'] for s in SERIES_LABELS) else 'FAIL'}"
        )
        print("\nPublished coverage")
        coverage = report["published_coverage"]
        for series in SERIES_LABELS:
            row = coverage[series]
            print(
                f"{SERIES_LABELS[series]:<24}{row['acquired']:>5} / {row['published']:<4} unavailable {row['unavailable']}"
            )
        row = coverage["total"]
        print(
            f"{'Published total':<24}{row['acquired']:>5} / {row['published']:<4} unavailable {row['unavailable']}"
        )
        print(f"{'Status':<24}{coverage['status']}")
        print(
            f"\n{'Atlas SVGs':<24}{report['atlas_svgs']['observed']:>5} / {report['atlas_svgs']['expected']}"
        )
        for gid in GROUPS:
            stats = report["graphic_groups"][str(gid)]
            print(
                f"Group {gid:<17}{stats['plates_present']:>5} plates, {stats['structure_paths']} paths, {stats['unique_structure_ids']} IDs"
            )
        ontology = report["ontology"]
        print(
            f"{'Ontology IDs resolved':<24}{ontology['svg_unique_structure_ids'] - ontology['unresolved_references']:>5} / {ontology['svg_unique_structure_ids']}"
        )
        print(
            f"{'Plate mappings':<24}{report['plate_mappings']['observed']:>5} / {report['plate_mappings']['expected']}"
        )
        print(f"{'Checksum failures':<24}{report['checksum_failures']:>5}")
        print(f"{'Unreadable files':<24}{report['unreadable_files']:>5}")
    for item in report.get("failures", []):
        print(f"FAIL [{item['category']}] {item['affected']}")
        print(f"  expected: {item['expected']}")
        print(f"  observed: {item['observed']}")
        print(f"  corrective action: {item['corrective_action']}")
    print(f"{'Status':<24}{report['status']}")


def migrate_superseded_metadata(paths: Paths, run_id: str) -> None:
    sources = [
        paths.metadata / name
        for name in ("sections.tsv", "atlas_annotations.tsv", "image_files.tsv")
    ]
    sources = [path for path in sources if path.exists()]
    if not sources:
        return
    destination = paths.root / "quarantine" / run_id / "superseded_metadata"
    destination.mkdir(parents=True, exist_ok=True)
    for source in sources:
        os.replace(source, destination / source.name)


def selected_series(
    args: argparse.Namespace, records: Mapping[str, Sequence[Section]]
) -> list[str]:
    requested = args.series
    if args.stains:
        if requested:
            raise RuntimeError(
                "Use either --series or the deprecated --stains alias, not both"
            )
        requested = ["pv" if value == "ihc" else value for value in args.stains]
        LOG.warning("--stains is deprecated; use --series nissl pv")
    available = [series for series in SERIES_LABELS if records[series]]
    chosen = available if not requested else list(dict.fromkeys(requested))
    unavailable = [series for series in chosen if not records[series]]
    if unavailable:
        if unavailable == ["smi32"]:
            raise RuntimeError(
                f"SMI-32 is unavailable from accepted providers: no qualifying official source was found in the bounded search on {DISCOVERY_DATE}. "
                "Provide and review an authoritative donor/treatment/section-identified source before acquisition."
            )
        raise RuntimeError(f"Unavailable requested series: {', '.join(unavailable)}")
    return chosen


def run_acquisition(args: argparse.Namespace) -> int:
    paths = Paths.make(args.data_dir)
    if args.validate_only:
        report = validate_dataset(paths)
        print_validation(report)
        if args.validation_json:
            text = json.dumps(report, indent=2) + "\n"
            if str(args.validation_json) == "-":
                print(text, end="")
            else:
                atomic_text(Path(args.validation_json).expanduser().resolve(), text)
        return 0 if report["status"] == "PASS" else 1

    session = build_session(retries=args.retries, backoff_factor=args.retry_backoff)
    records, observed = AllenSectionDataSetProvider(session).discover(SPECIMEN_ID)
    stored = load_stored_inventory(paths)
    gate_inventory(
        stored, observed, args.inventory_json, args.accept_api_inventory_sha256
    )
    accepted = observed if args.accept_api_inventory_sha256 else stored
    chosen = selected_series(args, records)
    atlas_info, raw_plates = AllenAtlasPlateProvider(session).discover()
    cached = legacy_plate_mappings(paths)
    filenames = (
        {
            opt_int(path.stem.rsplit("_", 1)[-1])
            for path in paths.image_dir("nissl").glob("*.jpg")
        }
        if paths.image_dir("nissl").is_dir()
        else set()
    )
    plates = resolve_mappings(
        raw_plates, records["nissl"], cached, {x for x in filenames if x is not None}
    )
    if any(item.mapping_status in {"filename_only", "unresolved"} for item in plates):
        raise RuntimeError(
            "One or more atlas plates lack a metadata-derived Nissl mapping"
        )

    paths.create(series for series in SERIES_LABELS if records[series])
    artifacts = scan_existing(paths, records, plates, args.downsample)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    provider = AllenStructureGraphProvider(session)
    if not args.skip_downloads:
        ontology = acquire_ontology(provider, paths, artifacts, args.overwrite, run_id)
        artifacts[ontology.path] = ontology
    elif not paths.ontology.is_file():
        raise RuntimeError("--skip-downloads requires an existing raw ontology")
    structures = flatten_ontology(parse_ontology(paths.ontology.read_bytes()))

    if not args.skip_downloads and not args.metadata_only and not args.annotations_only:
        work = [item for series in chosen for item in records[series]]
        if args.limit is not None:
            work = work[: args.limit]
        for index, section in enumerate(work, 1):
            artifact = acquire_image(
                session,
                paths,
                section,
                artifacts,
                args.downsample,
                args.image_download_mode,
                args.overwrite,
                run_id,
            )
            artifacts[artifact.path] = artifact
            if index % 25 == 0:
                save_manifest(paths.manifest, artifacts)
        svg_work = list(plates)
        if args.limit is not None:
            svg_work = svg_work[: args.limit]
        for index, plate in enumerate(svg_work, 1):
            artifact = acquire_svg(
                session, paths, plate, artifacts, args.overwrite, run_id
            )
            artifacts[artifact.path] = artifact
            if index % 25 == 0:
                save_manifest(paths.manifest, artifacts)
    elif args.annotations_only and not args.skip_downloads:
        svg_work = (
            list(plates)[: args.limit] if args.limit is not None else list(plates)
        )
        for plate in svg_work:
            artifact = acquire_svg(
                session, paths, plate, artifacts, args.overwrite, run_id
            )
            artifacts[artifact.path] = artifact

    save_structures(paths.structures, structures)
    write_legacy_exports(
        paths, records, plates, args.downsample, args.image_download_mode
    )
    if not args.skip_masks:
        create_masks(paths, artifacts, chosen)
    save_manifest(paths.manifest, artifacts)
    atomic_json(
        paths.dataset,
        dataset_payload(
            accepted, atlas_info, args.image_download_mode, args.downsample, artifacts
        ),
    )
    if args.limit is None:
        preliminary = validate_dataset(paths, allow_superseded=True)
        if preliminary["status"] != "PASS":
            print_validation(preliminary)
            return 1
        migrate_superseded_metadata(paths, run_id)
        report = validate_dataset(paths)
        print_validation(report)
        return 0 if report["status"] == "PASS" else 1
    print(
        "Limited run complete; canonical expected counts were not truncated. Run --validate-only after completing acquisition."
    )
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--data-dir", type=Path, required=True)
    result.add_argument("--series", nargs="+", choices=tuple(SERIES_LABELS))
    result.add_argument(
        "--stains", nargs="+", choices=("nissl", "ihc"), help=argparse.SUPPRESS
    )
    result.add_argument("--downsample", type=int, default=5)
    result.add_argument(
        "--image-download-mode",
        choices=("allen-direct", "matlab-compatible"),
        default="allen-direct",
    )
    result.add_argument("--metadata-only", action="store_true")
    result.add_argument("--annotations-only", action="store_true")
    result.add_argument("--skip-downloads", action="store_true")
    result.add_argument("--skip-masks", action="store_true")
    result.add_argument("--overwrite", action="store_true")
    result.add_argument(
        "--verify-existing", action="store_true", help=argparse.SUPPRESS
    )
    result.add_argument("--limit", type=int)
    result.add_argument("--inventory-json", type=Path)
    result.add_argument("--accept-api-inventory-sha256")
    result.add_argument("--validate-only", action="store_true")
    result.add_argument("--validation-json")
    result.add_argument("--retries", type=int, default=5)
    result.add_argument("--retry-backoff", type=float, default=0.75)
    result.add_argument(
        "--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO"
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be positive")
    if args.validate_only and any(
        (
            args.series,
            args.stains,
            args.accept_api_inventory_sha256,
            args.metadata_only,
            args.annotations_only,
            args.skip_downloads,
            args.overwrite,
        )
    ):
        raise SystemExit("--validate-only cannot be combined with acquisition options")
    try:
        return run_acquisition(args)
    except APIInventoryChanged:
        return 3
    except (OSError, RuntimeError, requests.RequestException) as exc:
        LOG.error("%s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
