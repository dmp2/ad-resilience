#!/usr/bin/env python3
"""Legacy Nissl/PV Allen downloader retained for reproducibility and mask helpers.

Python translation of:
    acasamitjana/3dhirest/database/preprocessing/download_allen.m

The workflow:
1. Query section metadata for Allen specimen 708424.
2. Separate Nissl (treatment ID 3) and IHC/parvalbumin (treatment ID 16).
3. Download JPEG section images at the requested effective downsample level.
4. Query the selected Allen 2-D atlas for its AtlasImage records.
5. Download the atlas SVG annotations from those AtlasImage IDs.
6. Generate tissue masks using the thresholds and morphology in the MATLAB code.
7. Save legacy ``secInfo.json``/``secInfo.mat`` plus standards-aligned JSON/TSV metadata.

Notes on fidelity and fixes
---------------------------
* Allen image downsample is logarithmic: level n reduces each image axis by 2**n.
* The default ``allen-direct`` mode requests the final pyramid level directly and
  preserves the returned JPEG bytes. ``matlab-compatible`` remains available to
  reproduce the original request-at-one-level-higher plus local bicubic resize.
* The original MATLAB script references undefined variables ``NISSL_DIR`` and
  ``it_slice``. Here, the Nissl directory is derived from ``data_dir`` and the
  intended one-based loop ordinal is used for the slice-specific thresholds.
* The original MATLAB script inferred annotated sections from the specimen image
  metadata. This implementation instead queries ``AtlasImage`` records for the
  requested atlas ID, as required by Allen's atlas/SVG workflow, then downloads
  each annotation by AtlasImage ID.
* SVGs are requested by graphic-group IDs only. They remain vector graphics and
  retain the coordinate system encoded by their SVG ``viewBox``.
* Allen ``SectionImage.resolution`` and image dimensions are retained per section.
* Standards-aligned metadata are written to ``metadata/dataset.json`` and TSV tables;
  legacy ``secInfo.json``/``secInfo.mat`` remain available for 3DHiResT compatibility.
* Image provenance is recorded per file in ``metadata/image_files.tsv``. Existing
  files are not assumed to match the requested mode; use ``--verify-existing`` or
  ``--overwrite`` before treating them as verified direct downloads.
* All writes are atomic, HTTP requests use retries, and existing files are
  skipped unless ``--overwrite`` is supplied.

Install dependencies:
    python -m pip install numpy scipy pillow requests

Example:
    python download_allen.py --data-dir /path/to/Allen/downloads

Metadata-only smoke test:
    python download_allen.py --data-dir ./allen_downloads --metadata-only
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import logging
import os
import sys
import tempfile
from datetime import datetime, timezone
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import requests
from PIL import Image, UnidentifiedImageError
from requests.adapters import HTTPAdapter
from scipy import ndimage
from scipy.io import loadmat, savemat
from urllib3.util.retry import Retry

API_BASE = "https://api.brain-map.org/api/v2"
DEFAULT_ATLAS_ID = 265297126  # Human, 34 years, Cortex - Modified Brodmann.
DEFAULT_ATLAS_IMAGE_TYPE = "Atlas - Developing Human Brodmann"
DEFAULT_SPECIMEN_ID = 708424
DEFAULT_GROUPS = (31, 113753816, 141667008, 265297118)  # Brodmann groups.
DEFAULT_DOWNSAMPLE = 5
DEFAULT_IMAGE_DOWNLOAD_MODE = "allen-direct"
METADATA_SCHEMA_VERSION = "1.1.0"
BIDS_MICROSCOPY_VERSION = "1.11.1"
DING_PAPER_DOI = "10.1002/cne.24080"
DING_SCAN_RESOLUTION_UM_PER_PIXEL = 1.0
DING_SECTION_THICKNESS_UM = 50.0
NISSL_NOMINAL_SPACING_UM = 200.0
PV_NOMINAL_SPACING_UM = 400.0
NISSL_TREATMENT_ID = 3
IHC_TREATMENT_ID = 16

LOGGER = logging.getLogger("download_allen")


@dataclass(frozen=True, slots=True)
class SectionRecord:
    """Metadata required for one downloaded section."""

    stain: str
    section_id: int
    section_number: int
    annotated: bool = False
    resolution_um_per_pixel: float | None = None
    width_px: int | None = None
    height_px: int | None = None


@dataclass(frozen=True, slots=True)
class AtlasAnnotationRecord:
    """One atlas plate whose anatomical drawings are downloadable as SVG."""

    atlas_image_id: int
    section_number: int


@dataclass(frozen=True, slots=True)
class ImageFileRecord:
    """Observed provenance for one local section JPEG."""

    stain: str
    allen_section_image_id: int
    section_number: int
    relative_output_path: str
    requested_mode: str
    status: str
    verified_mode: str | None
    source_url: str
    sha256: str
    size_bytes: int
    width_px: int
    height_px: int
    recorded_at_utc: str


@dataclass(frozen=True, slots=True)
class OutputPaths:
    """Filesystem layout matching the original 3dhirest downloader."""

    data_dir: Path
    nissl_images: Path
    nissl_labels: Path
    nissl_masks: Path
    ihc_images: Path
    ihc_masks: Path
    metadata_json: Path
    metadata_mat: Path
    metadata_dir: Path
    dataset_metadata_json: Path
    sections_tsv: Path
    atlas_annotations_tsv: Path
    image_files_tsv: Path

    @classmethod
    def from_data_dir(cls, data_dir: Path) -> "OutputPaths":
        data_dir = data_dir.expanduser().resolve()
        return cls(
            data_dir=data_dir,
            nissl_images=data_dir / "nissl" / "images_orig",
            nissl_labels=data_dir / "nissl" / "labels_orig",
            nissl_masks=data_dir / "nissl" / "masks_orig",
            ihc_images=data_dir / "ihc" / "images_orig",
            ihc_masks=data_dir / "ihc" / "masks_orig",
            metadata_json=data_dir / "secInfo.json",
            metadata_mat=data_dir / "secInfo.mat",
            metadata_dir=data_dir / "metadata",
            dataset_metadata_json=data_dir / "metadata" / "dataset.json",
            sections_tsv=data_dir / "metadata" / "sections.tsv",
            atlas_annotations_tsv=data_dir / "metadata" / "atlas_annotations.tsv",
            image_files_tsv=data_dir / "metadata" / "image_files.tsv",
        )

    def create(self) -> None:
        for directory in (
            self.data_dir,
            self.nissl_images,
            self.nissl_labels,
            self.nissl_masks,
            self.ihc_images,
            self.ihc_masks,
            self.metadata_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)


def build_session(retries: int = 5, backoff_factor: float = 1.0) -> requests.Session:
    """Create an HTTP session with bounded retries for transient failures."""

    retry = Retry(
        total=retries,
        connect=retries,
        read=retries,
        status=retries,
        backoff_factor=backoff_factor,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        respect_retry_after_header=True,
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update(
        {
            "User-Agent": (
                "download_allen.py/1.0 "
                "(Python translation of acasamitjana/3dhirest download_allen.m)"
            )
        }
    )
    return session


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return False


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _section_record_from_json(item: Mapping[str, Any]) -> SectionRecord:
    """Load current or older JSON records while ignoring derived fields."""

    allowed = {field.name for field in fields(SectionRecord)}
    return SectionRecord(**{key: value for key, value in item.items() if key in allowed})


def query_section_metadata(
    session: requests.Session,
    specimen_id: int = DEFAULT_SPECIMEN_ID,
    timeout: tuple[float, float] = (20.0, 180.0),
) -> tuple[list[SectionRecord], list[SectionRecord]]:
    """Query Allen RMA metadata and return sorted Nissl and IHC records."""

    criteria = (
        "model::SectionDataSet,"
        f"rma::criteria,specimen[id$eq{specimen_id}],"
        "rma::include,section_images(associates,alternate_images,treatments)"
    )
    url = f"{API_BASE}/data/query.json"
    LOGGER.info("Querying section metadata for specimen %d", specimen_id)
    response = session.get(
        url,
        params={"criteria": criteria, "num_rows": "all", "start_row": 0},
        timeout=timeout,
    )
    response.raise_for_status()
    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError("Allen metadata endpoint did not return valid JSON") from exc

    if payload.get("success") is False:
        raise RuntimeError(f"Allen API query failed: {payload.get('msg')!r}")

    datasets = _as_list(payload.get("msg"))
    if not datasets:
        raise RuntimeError(f"No SectionDataSet found for specimen {specimen_id}")

    records: dict[tuple[str, int], SectionRecord] = {}
    for dataset in datasets:
        if not isinstance(dataset, Mapping):
            continue
        for image in _as_list(dataset.get("section_images")):
            if not isinstance(image, Mapping):
                continue
            section_id = image.get("id")
            section_number = image.get("section_number")
            if section_id is None or section_number is None:
                LOGGER.warning("Skipping section image with incomplete metadata: %r", image)
                continue

            treatment_ids = {
                int(treatment["id"])
                for treatment in _as_list(image.get("treatments"))
                if isinstance(treatment, Mapping) and treatment.get("id") is not None
            }
            annotated = _as_bool(image.get("annotated"))

            if NISSL_TREATMENT_ID in treatment_ids:
                record = SectionRecord(
                    stain="nissl",
                    section_id=int(section_id),
                    section_number=int(section_number),
                    annotated=annotated,
                    resolution_um_per_pixel=_optional_float(image.get("resolution")),
                    width_px=_optional_int(image.get("width")),
                    height_px=_optional_int(image.get("height")),
                )
                records[(record.stain, record.section_id)] = record

            if IHC_TREATMENT_ID in treatment_ids:
                record = SectionRecord(
                    stain="ihc",
                    section_id=int(section_id),
                    section_number=int(section_number),
                    annotated=False,
                    resolution_um_per_pixel=_optional_float(image.get("resolution")),
                    width_px=_optional_int(image.get("width")),
                    height_px=_optional_int(image.get("height")),
                )
                records[(record.stain, record.section_id)] = record

    nissl = sorted(
        (record for record in records.values() if record.stain == "nissl"),
        key=lambda record: (record.section_number, record.section_id),
    )
    ihc = sorted(
        (record for record in records.values() if record.stain == "ihc"),
        key=lambda record: (record.section_number, record.section_id),
    )

    if not nissl and not ihc:
        raise RuntimeError(
            "The API response contained no sections with treatment IDs 3 (Nissl) "
            "or 16 (IHC). Inspect the returned metadata or verify the specimen ID."
        )

    LOGGER.info("Found %d Nissl and %d IHC sections", len(nissl), len(ihc))
    resolutions = sorted(
        {
            record.resolution_um_per_pixel
            for record in [*nissl, *ihc]
            if record.resolution_um_per_pixel is not None
        }
    )
    if resolutions:
        LOGGER.info(
            "Allen SectionImage source resolution(s): %s µm/pixel",
            ", ".join(f"{value:g}" for value in resolutions),
        )
    else:
        LOGGER.warning(
            "Allen SectionImage records did not include a usable resolution field; "
            "paper-level 1 µm/pixel remains nominal only"
        )
    return nissl, ihc


def query_atlas_annotation_metadata(
    session: requests.Session,
    atlas_id: int = DEFAULT_ATLAS_ID,
    atlas_image_type: str = DEFAULT_ATLAS_IMAGE_TYPE,
    timeout: tuple[float, float] = (20.0, 180.0),
) -> list[AtlasAnnotationRecord]:
    """Return the ordered AtlasImage records belonging to one 2-D atlas.

    Allen's SVG service accepts a SectionImage ID, and anatomical atlas drawings
    are attached specifically to ``AtlasImage`` records. Querying by ``atlas_id``
    avoids relying on the separate specimen metadata's ``annotated`` flag and
    makes the selected annotation scheme explicit.
    """

    if "'" in atlas_image_type:
        raise ValueError("atlas_image_type may not contain a single quote")

    criteria = (
        "model::AtlasImage,"
        "rma::criteria,"
        "[annotated$eqtrue],"
        f"atlas_data_set(atlases[id$eq{atlas_id}]),"
        f"alternate_images[image_type$eq'{atlas_image_type}'],"
        "rma::options[order$eq'sub_images.section_number'][num_rows$eqall]"
    )
    url = f"{API_BASE}/data/query.json"
    LOGGER.info("Querying AtlasImage metadata for atlas %d", atlas_id)
    response = session.get(
        url,
        params={"criteria": criteria, "num_rows": "all", "start_row": 0},
        timeout=timeout,
    )
    response.raise_for_status()
    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError("Allen atlas metadata endpoint did not return valid JSON") from exc

    if payload.get("success") is False:
        raise RuntimeError(f"Allen atlas API query failed: {payload.get('msg')!r}")

    images = _as_list(payload.get("msg"))
    if not images:
        raise RuntimeError(f"No AtlasImage records found for atlas {atlas_id}")

    records: dict[int, AtlasAnnotationRecord] = {}
    section_to_image: dict[int, int] = {}
    for image in images:
        if not isinstance(image, Mapping):
            continue
        atlas_image_id = image.get("id")
        section_number = image.get("section_number")
        if atlas_image_id is None or section_number is None:
            LOGGER.warning("Skipping AtlasImage with incomplete metadata: %r", image)
            continue

        record = AtlasAnnotationRecord(
            atlas_image_id=int(atlas_image_id),
            section_number=int(section_number),
        )
        prior_image_id = section_to_image.get(record.section_number)
        if prior_image_id is not None and prior_image_id != record.atlas_image_id:
            raise RuntimeError(
                "Atlas contains multiple AtlasImage IDs for section number "
                f"{record.section_number}: {prior_image_id}, {record.atlas_image_id}. "
                "The current filename convention would be ambiguous."
            )
        records[record.atlas_image_id] = record
        section_to_image[record.section_number] = record.atlas_image_id

    annotations = sorted(
        records.values(),
        key=lambda record: (record.section_number, record.atlas_image_id),
    )
    if not annotations:
        raise RuntimeError(
            f"Atlas {atlas_id} returned no usable AtlasImage IDs and section numbers"
        )

    LOGGER.info(
        "Found %d atlas plates with SVG annotation records for atlas %d",
        len(annotations),
        atlas_id,
    )
    return annotations


def _atomic_save_mat(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".mat", dir=path.parent
    )
    os.close(fd)
    try:
        savemat(temporary_name, payload, do_compression=True, appendmat=False)
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise



def _tsv_value(value: Any) -> Any:
    """Return a stable TSV representation; ``n/a`` follows BIDS convention."""

    return "n/a" if value is None else value


def _atomic_write_tsv(
    path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]
) -> None:
    """Atomically write a tab-separated metadata table."""

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=list(fieldnames),
        delimiter="\t",
        lineterminator="\n",
        extrasaction="raise",
    )
    writer.writeheader()
    for row in rows:
        writer.writerow({key: _tsv_value(row.get(key)) for key in fieldnames})
    _atomic_write_text(path, buffer.getvalue())


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _read_image_dimensions(path: Path) -> tuple[int, int]:
    try:
        with Image.open(path) as image:
            image.load()
            return image.size
    except UnidentifiedImageError as exc:
        raise RuntimeError(f"Local image is unreadable: {path}") from exc


def load_image_file_manifest(paths: OutputPaths) -> dict[str, ImageFileRecord]:
    """Load observed per-file provenance, keyed by relative output path."""

    if not paths.image_files_tsv.exists():
        return {}

    records: dict[str, ImageFileRecord] = {}
    with paths.image_files_tsv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            relative_path = row.get("relative_output_path")
            if not relative_path:
                continue
            verified_mode = row.get("verified_mode")
            if verified_mode in {None, "", "n/a"}:
                verified_mode = None
            record = ImageFileRecord(
                stain=str(row["stain"]),
                allen_section_image_id=int(row["allen_section_image_id"]),
                section_number=int(row["section_number"]),
                relative_output_path=relative_path,
                requested_mode=str(row["requested_mode"]),
                status=str(row["status"]),
                verified_mode=verified_mode,
                source_url=str(row["source_url"]),
                sha256=str(row["sha256"]),
                size_bytes=int(row["size_bytes"]),
                width_px=int(row["width_px"]),
                height_px=int(row["height_px"]),
                recorded_at_utc=str(row["recorded_at_utc"]),
            )
            records[relative_path] = record
    LOGGER.info("Loaded %d image provenance records from %s", len(records), paths.image_files_tsv)
    return records


def save_image_file_manifest(
    paths: OutputPaths, records: Mapping[str, ImageFileRecord]
) -> None:
    """Atomically save observed per-file provenance."""

    rows = [asdict(record) for record in sorted(
        records.values(), key=lambda item: (item.stain, item.section_number)
    )]
    _atomic_write_tsv(
        paths.image_files_tsv,
        (
            "stain",
            "allen_section_image_id",
            "section_number",
            "relative_output_path",
            "requested_mode",
            "status",
            "verified_mode",
            "source_url",
            "sha256",
            "size_bytes",
            "width_px",
            "height_px",
            "recorded_at_utc",
        ),
        rows,
    )


def _image_file_record(
    *,
    record: SectionRecord,
    destination: Path,
    data_dir: Path,
    requested_mode: str,
    status: str,
    verified_mode: str | None,
    source_url: str,
) -> ImageFileRecord:
    width_px, height_px = _read_image_dimensions(destination)
    return ImageFileRecord(
        stain=record.stain,
        allen_section_image_id=record.section_id,
        section_number=record.section_number,
        relative_output_path=destination.relative_to(data_dir).as_posix(),
        requested_mode=requested_mode,
        status=status,
        verified_mode=verified_mode,
        source_url=source_url,
        sha256=_sha256_file(destination),
        size_bytes=destination.stat().st_size,
        width_px=width_px,
        height_px=height_px,
        recorded_at_utc=_utc_now(),
    )


def _image_processing_parameters(
    *, downsample: int, image_download_mode: str
) -> dict[str, Any]:
    """Return explicit server/local sampling provenance for one download mode."""

    if image_download_mode == "allen-direct":
        return {
            "server_downsample_level": downsample,
            "server_linear_downsample_factor": 2**downsample,
            "local_resize_scale": 1.0,
            "local_interpolation": None,
            "effective_linear_downsample_factor": 2**downsample,
            "jpeg_transcoding": False,
        }
    if image_download_mode == "matlab-compatible":
        if downsample < 1:
            raise ValueError(
                "matlab-compatible mode requires downsample >= 1"
            )
        return {
            "server_downsample_level": downsample - 1,
            "server_linear_downsample_factor": 2 ** (downsample - 1),
            "local_resize_scale": 0.5,
            "local_interpolation": "Pillow Image.Resampling.BICUBIC",
            "effective_linear_downsample_factor": 2**downsample,
            "jpeg_transcoding": True,
        }
    raise ValueError(f"Unsupported image download mode: {image_download_mode}")


def save_standard_metadata(
    paths: OutputPaths,
    nissl: Sequence[SectionRecord],
    ihc: Sequence[SectionRecord],
    annotations: Sequence[AtlasAnnotationRecord],
    *,
    specimen_id: int,
    atlas_id: int,
    atlas_image_type: str,
    groups: Sequence[int],
    downsample: int,
    image_download_mode: str,
    image_files: Mapping[str, ImageFileRecord] | None = None,
) -> None:
    """Write concise standards-aligned JSON/TSV metadata.

    The directory deliberately retains the original 3DHiResT filenames and JPEG/SVG
    products, so it is not claimed to be a fully BIDS-compliant microscopy dataset.
    BIDS Microscopy field names and units are used where they fit; detailed per-file
    provenance is kept in TSV tables. OME-TIFF/OME-Zarr is the intended later image
    container when the sections and rasterized label maps are converted.
    """

    image_files = {} if image_files is None else image_files
    processing = _image_processing_parameters(
        downsample=downsample, image_download_mode=image_download_mode
    )
    server_level = int(processing["server_downsample_level"])
    server_factor = int(processing["server_linear_downsample_factor"])
    local_resize_scale = float(processing["local_resize_scale"])
    effective_factor = int(processing["effective_linear_downsample_factor"])

    dataset_payload = {
        "MetadataSchema": {
            "Name": "ad-resilience Allen histology metadata",
            "Version": METADATA_SCHEMA_VERSION,
            "StandardsAlignment": [
                {
                    "Name": "BIDS Microscopy",
                    "Version": BIDS_MICROSCOPY_VERSION,
                    "Use": (
                        "Field vocabulary, explicit units, sample/acquisition concepts, "
                        "and separation of lower-resolution derivatives"
                    ),
                    "Compliance": "aligned, not fully BIDS-formatted",
                    "Reason": (
                        "The legacy 3DHiResT-compatible directory names and Allen JPEG/SVG "
                        "products are intentionally retained."
                    ),
                },
                {
                    "Name": "REMBI",
                    "Use": (
                        "Coverage of study, biosample, specimen, acquisition, image data, "
                        "correlation, and analysed-data provenance"
                    ),
                },
                {
                    "Name": "OME",
                    "Use": (
                        "Target metadata/container model for later OME-TIFF or OME-Zarr "
                        "conversion; source JPEG and SVG files are not rewritten here"
                    ),
                },
            ],
        },
        "Dataset": {
            "Name": "Allen Human Brain Atlas histology, specimen 708424",
            "Description": (
                "Nissl and parvalbumin histology section images plus selected Ding 2016 "
                "two-dimensional atlas SVG annotations."
            ),
            "SpecimenID": specimen_id,
            "BodyPart": "BRAIN",
            "SampleEnvironment": "ex vivo",
            "Modality": "bright-field microscopy",
            "CitationDOI": DING_PAPER_DOI,
            "SourceAPI": API_BASE,
        },
        "Specimen": {
            "SliceThickness": DING_SECTION_THICKNESS_UM,
            "SliceThicknessUnits": "um",
            "NominalSeriesSpacing": {
                "nissl": NISSL_NOMINAL_SPACING_UM,
                "parvalbumin": PV_NOMINAL_SPACING_UM,
                "Units": "um",
                "Note": (
                    "Nominal retained-series spacing; missing sections and slab gaps make "
                    "the physical stack irregular."
                ),
            },
            "SampleStaining": ["Nissl", "parvalbumin immunohistochemistry"],
        },
        "SourceImages": {
            "PixelSize": [
                DING_SCAN_RESOLUTION_UM_PER_PIXEL,
                DING_SCAN_RESOLUTION_UM_PER_PIXEL,
            ],
            "PixelSizeUnits": "um",
            "PixelSizeStatus": "paper-level nominal value",
            "AuthoritativePerImageField": "Allen SectionImage.resolution",
            "Note": (
                "Per-image values in sections.tsv supersede the paper-level nominal value "
                "when Allen reports them."
            ),
        },
        "DerivedImages": {
            "RequestedImageDownloadMode": image_download_mode,
            "ServerDownsampleLevel": server_level,
            "ServerLinearDownsampleFactor": server_factor,
            "LocalResizeScale": [local_resize_scale, local_resize_scale],
            "LocalInterpolation": processing["local_interpolation"],
            "JPEGTranscoding": processing["jpeg_transcoding"],
            "EffectiveLinearDownsampleFactor": effective_factor,
            "PixelSizeFormula": "source PixelSize multiplied by 2**downsample",
            "NominalPixelSize": [
                DING_SCAN_RESOLUTION_UM_PER_PIXEL * effective_factor,
                DING_SCAN_RESOLUTION_UM_PER_PIXEL * effective_factor,
            ],
            "PixelSizeUnits": "um",
            "Encoding": "JPEG",
            "ServerJPEGQualityRequest": 100,
            "OutputJPEGQuality": (
                None if image_download_mode == "allen-direct" else 100
            ),
            "OutputJPEGSubsampling": (
                None if image_download_mode == "allen-direct" else 0
            ),
            "EncodingPreservedFromServerForNewDownloads": (
                image_download_mode == "allen-direct"
            ),
            "PerFileProvenance": (
                "The requested mode applies to new downloads only. Existing files are "
                "never assumed to match it; consult metadata/image_files.tsv and the "
                "local-file columns in metadata/sections.tsv."
            ),
            "DerivativeStatus": (
                "Lower-resolution derivative generated from the Allen image-download "
                "service; despite the legacy images_orig directory name, it is not the "
                "native scanner image."
            ),
        },
        "AtlasAnnotations": {
            "AtlasID": atlas_id,
            "AtlasImageType": atlas_image_type,
            "GraphicGroupIDs": list(groups),
            "Format": "SVG",
            "CoordinateSystem": (
                "Allen full-resolution section-image pixel canvas with SVG transforms; "
                "all nested transforms must be honored during rasterization."
            ),
        },
        "Tables": {
            "Sections": paths.sections_tsv.relative_to(paths.data_dir).as_posix(),
            "AtlasAnnotations": paths.atlas_annotations_tsv.relative_to(
                paths.data_dir
            ).as_posix(),
            "ImageFiles": paths.image_files_tsv.relative_to(paths.data_dir).as_posix(),
            "LegacyJSON": paths.metadata_json.relative_to(paths.data_dir).as_posix(),
            "LegacyMATLAB": paths.metadata_mat.relative_to(paths.data_dir).as_posix(),
        },
        "GeneratedBy": {
            "Name": "download_allen.py",
            "Description": (
                "Python translation of acasamitjana/3dhirest download_allen.m with "
                "atlas-specific SVG acquisition and explicit resolution provenance."
            ),
        },
    }
    _atomic_write_text(
        paths.dataset_metadata_json,
        json.dumps(dataset_payload, indent=2) + "\n",
    )

    section_rows: list[dict[str, Any]] = []
    for record in [*nissl, *ihc]:
        source_pixel_size = record.resolution_um_per_pixel
        output_pixel_size = (
            None if source_pixel_size is None else source_pixel_size * effective_factor
        )
        nominal_spacing = (
            NISSL_NOMINAL_SPACING_UM
            if record.stain == "nissl"
            else PV_NOMINAL_SPACING_UM
        )
        output_root = "nissl" if record.stain == "nissl" else "ihc"
        relative_output_path = (
            f"{output_root}/images_orig/image_{record.section_number:04d}.jpg"
        )
        local_path = paths.data_dir / relative_output_path
        observed = image_files.get(relative_output_path)
        if local_path.exists() and observed is not None:
            local_status = observed.status
        elif local_path.exists():
            local_status = "existing-unverified"
        elif observed is not None:
            local_status = "missing-despite-manifest"
        else:
            local_status = "missing"

        section_rows.append(
            {
                "stain": record.stain,
                "allen_section_image_id": record.section_id,
                "section_number": record.section_number,
                "annotated_in_section_metadata": int(record.annotated),
                "source_pixel_size_x_um": source_pixel_size,
                "source_pixel_size_y_um": source_pixel_size,
                "source_width_px": record.width_px,
                "source_height_px": record.height_px,
                "physical_section_thickness_um": DING_SECTION_THICKNESS_UM,
                "nominal_series_spacing_um": nominal_spacing,
                "server_downsample_level": server_level,
                "server_linear_downsample_factor": server_factor,
                "requested_image_download_mode": image_download_mode,
                "local_resize_scale_x": local_resize_scale,
                "local_resize_scale_y": local_resize_scale,
                "effective_linear_downsample_factor": effective_factor,
                "output_pixel_size_x_um": output_pixel_size,
                "output_pixel_size_y_um": output_pixel_size,
                "source_url": (
                    f"{API_BASE}/image_download/{record.section_id}"
                    f"?downsample={server_level}&quality=100"
                ),
                "relative_output_path": relative_output_path,
                "local_file_status": local_status,
                "local_file_verified_mode": (
                    None if observed is None else observed.verified_mode
                ),
                "local_file_sha256": None if observed is None else observed.sha256,
                "local_file_size_bytes": (
                    None if observed is None else observed.size_bytes
                ),
                "local_width_px": None if observed is None else observed.width_px,
                "local_height_px": None if observed is None else observed.height_px,
                "provenance_recorded_at_utc": (
                    None if observed is None else observed.recorded_at_utc
                ),
            }
        )

    _atomic_write_tsv(
        paths.sections_tsv,
        (
            "stain",
            "allen_section_image_id",
            "section_number",
            "annotated_in_section_metadata",
            "source_pixel_size_x_um",
            "source_pixel_size_y_um",
            "source_width_px",
            "source_height_px",
            "physical_section_thickness_um",
            "nominal_series_spacing_um",
            "server_downsample_level",
            "server_linear_downsample_factor",
            "requested_image_download_mode",
            "local_resize_scale_x",
            "local_resize_scale_y",
            "effective_linear_downsample_factor",
            "output_pixel_size_x_um",
            "output_pixel_size_y_um",
            "source_url",
            "relative_output_path",
            "local_file_status",
            "local_file_verified_mode",
            "local_file_sha256",
            "local_file_size_bytes",
            "local_width_px",
            "local_height_px",
            "provenance_recorded_at_utc",
        ),
        section_rows,
    )

    nissl_by_section = {record.section_number: record for record in nissl}
    group_text = ",".join(str(group) for group in groups)
    annotation_rows: list[dict[str, Any]] = []
    for record in annotations:
        matching_nissl = nissl_by_section.get(record.section_number)
        annotation_rows.append(
            {
                "atlas_id": atlas_id,
                "atlas_image_type": atlas_image_type,
                "atlas_image_id": record.atlas_image_id,
                "section_number": record.section_number,
                "matching_nissl_section_image_id": (
                    None if matching_nissl is None else matching_nissl.section_id
                ),
                "graphic_group_ids": group_text,
                "coordinate_system": (
                    "full-resolution Allen section-image pixels; honor SVG transforms"
                ),
                "source_url": (
                    f"{API_BASE}/svg_download/{record.atlas_image_id}?groups={group_text}"
                ),
                "relative_output_path": (
                    f"nissl/labels_orig/seg_{record.section_number:04d}.svg"
                ),
            }
        )

    _atomic_write_tsv(
        paths.atlas_annotations_tsv,
        (
            "atlas_id",
            "atlas_image_type",
            "atlas_image_id",
            "section_number",
            "matching_nissl_section_image_id",
            "graphic_group_ids",
            "coordinate_system",
            "source_url",
            "relative_output_path",
        ),
        annotation_rows,
    )

    save_image_file_manifest(paths, image_files)
    LOGGER.info(
        "Saved standards-aligned metadata to %s, %s, %s, and %s",
        paths.dataset_metadata_json,
        paths.sections_tsv,
        paths.atlas_annotations_tsv,
        paths.image_files_tsv,
    )

def save_metadata(
    paths: OutputPaths,
    nissl: Sequence[SectionRecord],
    ihc: Sequence[SectionRecord],
    annotations: Sequence[AtlasAnnotationRecord],
    *,
    specimen_id: int,
    atlas_id: int,
    atlas_image_type: str,
    groups: Sequence[int],
    downsample: int,
    image_download_mode: str,
    image_files: Mapping[str, ImageFileRecord] | None = None,
) -> None:
    """Save transparent JSON metadata and a MATLAB-compatible secInfo.mat."""

    processing = _image_processing_parameters(
        downsample=downsample, image_download_mode=image_download_mode
    )
    payload = {
        "specimen_id": specimen_id,
        "atlas_id": atlas_id,
        "atlas_image_type": atlas_image_type,
        "groups": list(groups),
        "downsample": downsample,
        "image_download_mode": image_download_mode,
        "histology_sampling": {
            "paper_nominal_scan_resolution_um_per_pixel": DING_SCAN_RESOLUTION_UM_PER_PIXEL,
            "physical_section_thickness_um": DING_SECTION_THICKNESS_UM,
            "nissl_nominal_section_spacing_um": NISSL_NOMINAL_SPACING_UM,
            "pv_nominal_section_spacing_um": PV_NOMINAL_SPACING_UM,
            "source_resolution_field": "Allen SectionImage.resolution",
            "server_downsample_level": processing["server_downsample_level"],
            "server_linear_downsample_factor": processing[
                "server_linear_downsample_factor"
            ],
            "local_resize_linear_factor": processing["local_resize_scale"],
            "local_interpolation": processing["local_interpolation"],
            "jpeg_transcoding": processing["jpeg_transcoding"],
            "effective_linear_downsample_factor": processing[
                "effective_linear_downsample_factor"
            ],
            "nominal_effective_resolution_um_per_pixel": (
                DING_SCAN_RESOLUTION_UM_PER_PIXEL * (2**downsample)
            ),
        },
        "nissl": [asdict(record) for record in nissl],
        "ihc": [asdict(record) for record in ihc],
        "atlas_annotations": [asdict(record) for record in annotations],
    }
    _atomic_write_text(paths.metadata_json, json.dumps(payload, indent=2) + "\n")

    _atomic_save_mat(
        paths.metadata_mat,
        {
            "annotatedNissl": np.asarray(
                [int(record.annotated) for record in nissl], dtype=np.uint8
            ),
            "secIDNissl": np.asarray(
                [record.section_id for record in nissl], dtype=np.int64
            ),
            "secNumberNissl": np.asarray(
                [record.section_number for record in nissl], dtype=np.int64
            ),
            "secIDIHC": np.asarray(
                [record.section_id for record in ihc], dtype=np.int64
            ),
            "secNumberIHC": np.asarray(
                [record.section_number for record in ihc], dtype=np.int64
            ),
            "atlasImageID": np.asarray(
                [record.atlas_image_id for record in annotations], dtype=np.int64
            ),
            "atlasSectionNumber": np.asarray(
                [record.section_number for record in annotations], dtype=np.int64
            ),
            "resolutionNisslUmPerPixel": np.asarray(
                [
                    np.nan if record.resolution_um_per_pixel is None
                    else record.resolution_um_per_pixel
                    for record in nissl
                ],
                dtype=np.float64,
            ),
            "resolutionIHCUmPerPixel": np.asarray(
                [
                    np.nan if record.resolution_um_per_pixel is None
                    else record.resolution_um_per_pixel
                    for record in ihc
                ],
                dtype=np.float64,
            ),
            "widthNisslPx": np.asarray(
                [-1 if record.width_px is None else record.width_px for record in nissl],
                dtype=np.int64,
            ),
            "heightNisslPx": np.asarray(
                [-1 if record.height_px is None else record.height_px for record in nissl],
                dtype=np.int64,
            ),
            "widthIHCPx": np.asarray(
                [-1 if record.width_px is None else record.width_px for record in ihc],
                dtype=np.int64,
            ),
            "heightIHCPx": np.asarray(
                [-1 if record.height_px is None else record.height_px for record in ihc],
                dtype=np.int64,
            ),
            "imageDownloadMode": np.asarray([image_download_mode], dtype=object),
            "serverDownsampleLevel": np.asarray(
                [[processing["server_downsample_level"]]], dtype=np.int64
            ),
            "localResizeLinearFactor": np.asarray(
                [[processing["local_resize_scale"]]], dtype=np.float64
            ),
            "effectiveLinearDownsampleFactor": np.asarray(
                [[processing["effective_linear_downsample_factor"]]], dtype=np.int64
            ),
            "paperNominalScanResolutionUmPerPixel": np.asarray(
                [[DING_SCAN_RESOLUTION_UM_PER_PIXEL]], dtype=np.float64
            ),
            "physicalSectionThicknessUm": np.asarray(
                [[DING_SECTION_THICKNESS_UM]], dtype=np.float64
            ),
            "nisslNominalSectionSpacingUm": np.asarray(
                [[NISSL_NOMINAL_SPACING_UM]], dtype=np.float64
            ),
            "pvNominalSectionSpacingUm": np.asarray(
                [[PV_NOMINAL_SPACING_UM]], dtype=np.float64
            ),
            "DS_FACTOR": np.asarray([[downsample]], dtype=np.int64),
        },
    )
    save_standard_metadata(
        paths,
        nissl,
        ihc,
        annotations,
        specimen_id=specimen_id,
        atlas_id=atlas_id,
        atlas_image_type=atlas_image_type,
        groups=groups,
        downsample=downsample,
        image_download_mode=image_download_mode,
        image_files=image_files,
    )
    LOGGER.info("Saved metadata to %s and %s", paths.metadata_json, paths.metadata_mat)


def load_cached_metadata(
    paths: OutputPaths,
) -> tuple[list[SectionRecord], list[SectionRecord], list[AtlasAnnotationRecord]]:
    """Load metadata from JSON, falling back to the original MATLAB cache format."""

    if paths.metadata_json.exists():
        payload = json.loads(paths.metadata_json.read_text(encoding="utf-8"))
        nissl = [
            _section_record_from_json(item) for item in payload.get("nissl", [])
        ]
        ihc = [
            _section_record_from_json(item) for item in payload.get("ihc", [])
        ]
        annotations = [
            AtlasAnnotationRecord(**item)
            for item in payload.get("atlas_annotations", [])
        ]
        if nissl or ihc:
            LOGGER.info("Loaded cached metadata from %s", paths.metadata_json)
            return nissl, ihc, annotations

    if paths.metadata_mat.exists():
        mat = loadmat(paths.metadata_mat, squeeze_me=True)
        nissl_ids = np.atleast_1d(mat.get("secIDNissl", [])).astype(int)
        nissl_numbers = np.atleast_1d(mat.get("secNumberNissl", [])).astype(int)
        nissl_annotated = np.atleast_1d(mat.get("annotatedNissl", [])).astype(bool)
        ihc_ids = np.atleast_1d(mat.get("secIDIHC", [])).astype(int)
        ihc_numbers = np.atleast_1d(mat.get("secNumberIHC", [])).astype(int)
        nissl_resolutions = np.atleast_1d(
            mat.get("resolutionNisslUmPerPixel", np.full(len(nissl_ids), np.nan))
        ).astype(float)
        ihc_resolutions = np.atleast_1d(
            mat.get("resolutionIHCUmPerPixel", np.full(len(ihc_ids), np.nan))
        ).astype(float)
        nissl_widths = np.atleast_1d(
            mat.get("widthNisslPx", np.full(len(nissl_ids), -1))
        ).astype(int)
        nissl_heights = np.atleast_1d(
            mat.get("heightNisslPx", np.full(len(nissl_ids), -1))
        ).astype(int)
        ihc_widths = np.atleast_1d(
            mat.get("widthIHCPx", np.full(len(ihc_ids), -1))
        ).astype(int)
        ihc_heights = np.atleast_1d(
            mat.get("heightIHCPx", np.full(len(ihc_ids), -1))
        ).astype(int)
        atlas_image_ids = np.atleast_1d(mat.get("atlasImageID", [])).astype(int)
        atlas_section_numbers = np.atleast_1d(
            mat.get("atlasSectionNumber", [])
        ).astype(int)

        if not (
            len(nissl_ids)
            == len(nissl_numbers)
            == len(nissl_annotated)
            == len(nissl_resolutions)
            == len(nissl_widths)
            == len(nissl_heights)
        ):
            raise RuntimeError(f"Inconsistent Nissl arrays in {paths.metadata_mat}")
        if not (
            len(ihc_ids)
            == len(ihc_numbers)
            == len(ihc_resolutions)
            == len(ihc_widths)
            == len(ihc_heights)
        ):
            raise RuntimeError(f"Inconsistent IHC arrays in {paths.metadata_mat}")
        if len(atlas_image_ids) != len(atlas_section_numbers):
            raise RuntimeError(
                f"Inconsistent atlas annotation arrays in {paths.metadata_mat}"
            )

        nissl = [
            SectionRecord(
                "nissl",
                int(section_id),
                int(section_number),
                bool(annotated),
                None if np.isnan(resolution) else float(resolution),
                None if width < 0 else int(width),
                None if height < 0 else int(height),
            )
            for section_id, section_number, annotated, resolution, width, height in zip(
                nissl_ids,
                nissl_numbers,
                nissl_annotated,
                nissl_resolutions,
                nissl_widths,
                nissl_heights,
                strict=True,
            )
        ]
        ihc = [
            SectionRecord(
                "ihc",
                int(section_id),
                int(section_number),
                False,
                None if np.isnan(resolution) else float(resolution),
                None if width < 0 else int(width),
                None if height < 0 else int(height),
            )
            for section_id, section_number, resolution, width, height in zip(
                ihc_ids,
                ihc_numbers,
                ihc_resolutions,
                ihc_widths,
                ihc_heights,
                strict=True,
            )
        ]
        annotations = [
            AtlasAnnotationRecord(int(image_id), int(section_number))
            for image_id, section_number in zip(
                atlas_image_ids, atlas_section_numbers, strict=True
            )
        ]
        LOGGER.info("Loaded cached metadata from %s", paths.metadata_mat)
        return nissl, ihc, annotations

    raise FileNotFoundError("No cached secInfo.json or secInfo.mat exists")


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _atomic_write_text(path: Path, text: str) -> None:
    _atomic_write_bytes(path, text.encode("utf-8"))


def _atomic_save_pil(image: Image.Image, path: Path, **save_kwargs: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix or ".tmp"
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=suffix, dir=path.parent
    )
    os.close(fd)
    try:
        image.save(temporary_name, **save_kwargs)
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def download_image(
    session: requests.Session,
    record: SectionRecord,
    destination: Path,
    *,
    data_dir: Path,
    downsample: int,
    image_download_mode: str,
    overwrite: bool,
    verify_existing: bool,
    prior_record: ImageFileRecord | None,
    timeout: tuple[float, float] = (20.0, 300.0),
) -> ImageFileRecord:
    """Download or inspect one image and return observed per-file provenance."""

    processing = _image_processing_parameters(
        downsample=downsample, image_download_mode=image_download_mode
    )
    server_level = int(processing["server_downsample_level"])
    url = f"{API_BASE}/image_download/{record.section_id}"
    source_url = f"{url}?downsample={server_level}&quality=100"

    if destination.exists() and not overwrite:
        current_sha256 = _sha256_file(destination)
        if (
            prior_record is not None
            and prior_record.sha256 == current_sha256
            and prior_record.verified_mode is not None
        ):
            LOGGER.info(
                "Image exists with verified manifest provenance (%s); skipping %s",
                prior_record.verified_mode,
                destination,
            )
            return _image_file_record(
                record=record,
                destination=destination,
                data_dir=data_dir,
                requested_mode=image_download_mode,
                status="manifest-verified-existing",
                verified_mode=prior_record.verified_mode,
                source_url=prior_record.source_url,
            )

        if prior_record is not None and prior_record.sha256 != current_sha256:
            LOGGER.warning(
                "Existing image no longer matches its manifest checksum: %s", destination
            )

        if verify_existing:
            if image_download_mode != "allen-direct":
                raise ValueError(
                    "--verify-existing currently supports only --image-download-mode "
                    "allen-direct"
                )
            LOGGER.info(
                "Verifying existing %s section %d against Allen direct bytes",
                record.stain.upper(),
                record.section_number,
            )
            response = session.get(
                url,
                params={"downsample": server_level, "quality": 100},
                timeout=timeout,
            )
            response.raise_for_status()
            try:
                with Image.open(io.BytesIO(response.content)) as source:
                    source.load()
            except UnidentifiedImageError as exc:
                content_type = response.headers.get("Content-Type", "unknown")
                raise RuntimeError(
                    f"Image endpoint returned unreadable data for section "
                    f"{record.section_id} (Content-Type: {content_type})"
                ) from exc

            if current_sha256 == _sha256_bytes(response.content):
                LOGGER.info("Existing image exactly matches Allen direct response: %s", destination)
                return _image_file_record(
                    record=record,
                    destination=destination,
                    data_dir=data_dir,
                    requested_mode=image_download_mode,
                    status="verified-existing",
                    verified_mode="allen-direct",
                    source_url=source_url,
                )

            LOGGER.warning(
                "Existing image differs from Allen direct response; leaving it unchanged: %s",
                destination,
            )
            return _image_file_record(
                record=record,
                destination=destination,
                data_dir=data_dir,
                requested_mode=image_download_mode,
                status="existing-mismatch",
                verified_mode=None,
                source_url=source_url,
            )

        LOGGER.warning(
            "Image exists without verified provenance; skipping %s. Use "
            "--verify-existing or --overwrite before treating it as %s.",
            destination,
            image_download_mode,
        )
        return _image_file_record(
            record=record,
            destination=destination,
            data_dir=data_dir,
            requested_mode=image_download_mode,
            status="existing-unverified",
            verified_mode=None,
            source_url=source_url,
        )

    LOGGER.info(
        "Downloading %s section %d (image ID %d; mode=%s, server downsample=%d)",
        record.stain.upper(),
        record.section_number,
        record.section_id,
        image_download_mode,
        server_level,
    )
    response = session.get(
        url,
        params={"downsample": server_level, "quality": 100},
        timeout=timeout,
    )
    response.raise_for_status()

    try:
        with Image.open(io.BytesIO(response.content)) as source:
            source.load()
            server_size = source.size
            if image_download_mode == "matlab-compatible":
                image = source.convert("RGB")
            else:
                image = None
    except UnidentifiedImageError as exc:
        content_type = response.headers.get("Content-Type", "unknown")
        raise RuntimeError(
            f"Image endpoint returned unreadable data for section {record.section_id} "
            f"(Content-Type: {content_type})"
        ) from exc

    if image_download_mode == "allen-direct":
        _atomic_write_bytes(destination, response.content)
        output_size = server_size
    else:
        assert image is not None
        output_size = (max(1, image.width // 2), max(1, image.height // 2))
        image = image.resize(output_size, resample=Image.Resampling.BICUBIC)
        _atomic_save_pil(
            image, destination, format="JPEG", quality=100, subsampling=0
        )

    if record.resolution_um_per_pixel is not None:
        LOGGER.debug(
            "Saved %s at %s pixels; effective pixel size %.6g µm",
            destination,
            output_size,
            record.resolution_um_per_pixel * (2**downsample),
        )

    return _image_file_record(
        record=record,
        destination=destination,
        data_dir=data_dir,
        requested_mode=image_download_mode,
        status="downloaded",
        verified_mode=image_download_mode,
        source_url=source_url,
    )


def download_atlas_svg(
    session: requests.Session,
    record: AtlasAnnotationRecord,
    destination: Path,
    *,
    groups: Sequence[int],
    overwrite: bool,
    timeout: tuple[float, float] = (20.0, 180.0),
) -> None:
    """Download one atlas plate's anatomical drawings as SVG."""

    if destination.exists() and not overwrite:
        LOGGER.info("SVG exists; skipping %s", destination)
        return

    LOGGER.info(
        "Downloading atlas SVG for section %d (AtlasImage ID %d)",
        record.section_number,
        record.atlas_image_id,
    )
    url = f"{API_BASE}/svg_download/{record.atlas_image_id}"
    response = session.get(
        url,
        params={"groups": ",".join(str(group) for group in groups)},
        timeout=timeout,
    )
    response.raise_for_status()
    content = response.content
    if b"<svg" not in content[:4096].lower():
        content_type = response.headers.get("Content-Type", "unknown")
        raise RuntimeError(
            "SVG endpoint returned unexpected data for AtlasImage "
            f"{record.atlas_image_id} (section {record.section_number}) "
            f"(Content-Type: {content_type})"
        )
    _atomic_write_bytes(destination, content)


def _zero_border(mask: np.ndarray, width: int = 10) -> np.ndarray:
    result = mask.copy()
    if result.ndim != 2:
        raise ValueError(f"Expected a 2-D mask, got shape {result.shape}")
    width = min(width, result.shape[0] // 2, result.shape[1] // 2)
    if width <= 0:
        return result
    result[:width, :] = False
    result[-width:, :] = False
    result[:, :width] = False
    result[:, -width:] = False
    return result


def _retain_large_components(mask: np.ndarray, min_size: int) -> np.ndarray:
    """Keep 8-connected components containing more than ``min_size`` pixels."""

    labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=bool))
    if count == 0:
        return np.zeros_like(mask, dtype=bool)
    sizes = np.bincount(labels.ravel())
    keep = sizes > min_size
    keep[0] = False
    return keep[labels]


def _mean_filter(image: np.ndarray, size: int) -> np.ndarray:
    kernel = np.full((size, size), 1.0 / (size * size), dtype=np.float32)
    return ndimage.correlate(image, kernel, mode="constant", cval=0.0)


def generate_nissl_mask(image: Image.Image, ordinal: int) -> np.ndarray:
    """Reproduce the Nissl mask logic; ``ordinal`` is one-based."""

    gray = np.asarray(image.convert("L"), dtype=np.float32)
    median = ndimage.median_filter(gray, size=3, mode="constant", cval=0.0)
    average = _mean_filter(median, size=3)

    threshold = 240.0 if ordinal in {166, 222} else 248.0
    coarse = _zero_border(average < threshold, width=10)
    coarse = _retain_large_components(coarse, min_size=30_000)

    gx = ndimage.correlate(
        average, np.asarray([[-1.0], [0.0], [1.0]], dtype=np.float32),
        mode="constant", cval=0.0,
    )
    gy = ndimage.correlate(
        average, np.asarray([[-1.0, 0.0, 1.0]], dtype=np.float32),
        mode="constant", cval=0.0,
    )
    gradient_mask = np.hypot(gx, gy) > 1.0

    combined = coarse & gradient_mask
    filled = ndimage.binary_fill_holes(combined)
    opened = ndimage.binary_opening(
        filled, structure=np.ones((10, 10), dtype=bool)
    )
    return _retain_large_components(opened, min_size=30_000)


def generate_ihc_mask(image: Image.Image, ordinal: int) -> np.ndarray:
    """Reproduce the IHC mask logic; ``ordinal`` is one-based."""

    if ordinal < 10:
        min_size = 10_000
    elif ordinal < 150:
        min_size = 60_000
    else:
        min_size = 30_000

    gray = np.asarray(image.convert("L"), dtype=np.float32)
    median = ndimage.median_filter(gray, size=3, mode="constant", cval=0.0)
    if ordinal < 148:
        average = _mean_filter(median, size=3)
        threshold = 250.0
    else:
        average = _mean_filter(median, size=10)
        threshold = 245.0

    mask = _zero_border(average < threshold, width=10)
    filled = ndimage.binary_fill_holes(mask)
    opened = ndimage.binary_opening(
        filled, structure=np.ones((5, 5), dtype=bool)
    )
    retained = _retain_large_components(opened, min_size=min_size)
    return ndimage.binary_opening(
        retained, structure=np.ones((5, 5), dtype=bool)
    )


def save_mask(mask: np.ndarray, destination: Path) -> None:
    image = Image.fromarray(mask.astype(np.uint8) * 255)
    _atomic_save_pil(image, destination, format="PNG", optimize=True)


def generate_masks(
    records: Sequence[SectionRecord],
    *,
    image_dir: Path,
    mask_dir: Path,
    stain: str,
    overwrite: bool,
) -> None:
    """Generate masks for all downloaded sections of one stain."""

    total = len(records)
    for ordinal, record in enumerate(records, start=1):
        image_path = image_dir / f"image_{record.section_number:04d}.jpg"
        if stain == "nissl":
            mask_path = mask_dir / f"image_{record.section_number:04d}.png"
        elif stain == "ihc":
            mask_path = mask_dir / f"slice_{record.section_number:03d}.png"
        else:
            raise ValueError(f"Unsupported stain: {stain}")

        LOGGER.info("Generating %s mask %d/%d", stain.upper(), ordinal, total)
        if mask_path.exists() and not overwrite:
            LOGGER.info("Mask exists; skipping %s", mask_path)
            continue
        if not image_path.exists():
            raise FileNotFoundError(
                f"Cannot generate mask because image is missing: {image_path}"
            )

        with Image.open(image_path) as image:
            image.load()
            if stain == "nissl":
                mask = generate_nissl_mask(image, ordinal)
            else:
                mask = generate_ihc_mask(image, ordinal)
        save_mask(mask, mask_path)


def _limited(records: Sequence[SectionRecord], limit: int | None) -> Sequence[SectionRecord]:
    return records if limit is None else records[:limit]


def run(args: argparse.Namespace) -> None:
    paths = OutputPaths.from_data_dir(args.data_dir)
    paths.create()
    session = build_session(retries=args.retries, backoff_factor=args.backoff)
    image_files = load_image_file_manifest(paths)

    if args.refresh_metadata:
        nissl, ihc = query_section_metadata(session, specimen_id=args.specimen_id)
        annotations = query_atlas_annotation_metadata(
            session,
            atlas_id=args.atlas_id,
            atlas_image_type=args.atlas_image_type,
        )
        save_metadata(
            paths,
            nissl,
            ihc,
            annotations,
            specimen_id=args.specimen_id,
            atlas_id=args.atlas_id,
            atlas_image_type=args.atlas_image_type,
            groups=args.groups,
            downsample=args.downsample,
            image_download_mode=args.image_download_mode,
            image_files=image_files,
        )
    else:
        try:
            nissl, ihc, annotations = load_cached_metadata(paths)
        except FileNotFoundError:
            nissl, ihc = query_section_metadata(session, specimen_id=args.specimen_id)
            annotations = query_atlas_annotation_metadata(
                session,
                atlas_id=args.atlas_id,
                atlas_image_type=args.atlas_image_type,
            )
            save_metadata(
                paths,
                nissl,
                ihc,
                annotations,
                specimen_id=args.specimen_id,
                atlas_id=args.atlas_id,
                atlas_image_type=args.atlas_image_type,
                groups=args.groups,
                downsample=args.downsample,
                image_download_mode=args.image_download_mode,
                image_files=image_files,
            )
        else:
            if not annotations:
                LOGGER.info(
                    "Cached metadata predates atlas-specific SVG metadata; "
                    "querying atlas %d",
                    args.atlas_id,
                )
                annotations = query_atlas_annotation_metadata(
                    session,
                    atlas_id=args.atlas_id,
                    atlas_image_type=args.atlas_image_type,
                )
                save_metadata(
                    paths,
                    nissl,
                    ihc,
                    annotations,
                    specimen_id=args.specimen_id,
                    atlas_id=args.atlas_id,
                    atlas_image_type=args.atlas_image_type,
                    groups=args.groups,
                    downsample=args.downsample,
                    image_download_mode=args.image_download_mode,
                    image_files=image_files,
                )

    save_standard_metadata(
        paths,
        nissl,
        ihc,
        annotations,
        specimen_id=args.specimen_id,
        atlas_id=args.atlas_id,
        atlas_image_type=args.atlas_image_type,
        groups=args.groups,
        downsample=args.downsample,
        image_download_mode=args.image_download_mode,
        image_files=image_files,
    )

    nissl = list(_limited(nissl, args.limit))
    ihc = list(_limited(ihc, args.limit))
    annotations = list(_limited(annotations, args.limit))

    if args.metadata_only:
        return

    if not args.skip_downloads:
        if "nissl" in args.stains:
            if not args.annotations_only:
                for ordinal, record in enumerate(nissl, start=1):
                    LOGGER.info("NISSL section %d/%d", ordinal, len(nissl))
                    destination = (
                        paths.nissl_images / f"image_{record.section_number:04d}.jpg"
                    )
                    relative_path = destination.relative_to(paths.data_dir).as_posix()
                    observed = download_image(
                        session,
                        record,
                        destination,
                        data_dir=paths.data_dir,
                        downsample=args.downsample,
                        image_download_mode=args.image_download_mode,
                        overwrite=args.overwrite,
                        verify_existing=args.verify_existing,
                        prior_record=image_files.get(relative_path),
                    )
                    image_files[relative_path] = observed
                    save_image_file_manifest(paths, image_files)

            for ordinal, record in enumerate(annotations, start=1):
                LOGGER.info("ATLAS SVG %d/%d", ordinal, len(annotations))
                download_atlas_svg(
                    session,
                    record,
                    paths.nissl_labels / f"seg_{record.section_number:04d}.svg",
                    groups=args.groups,
                    overwrite=args.overwrite,
                )

        if "ihc" in args.stains and not args.annotations_only:
            for ordinal, record in enumerate(ihc, start=1):
                LOGGER.info("IHC section %d/%d", ordinal, len(ihc))
                destination = (
                    paths.ihc_images / f"image_{record.section_number:04d}.jpg"
                )
                relative_path = destination.relative_to(paths.data_dir).as_posix()
                observed = download_image(
                    session,
                    record,
                    destination,
                    data_dir=paths.data_dir,
                    downsample=args.downsample,
                    image_download_mode=args.image_download_mode,
                    overwrite=args.overwrite,
                    verify_existing=args.verify_existing,
                    prior_record=image_files.get(relative_path),
                )
                image_files[relative_path] = observed
                save_image_file_manifest(paths, image_files)

    save_standard_metadata(
        paths,
        nissl,
        ihc,
        annotations,
        specimen_id=args.specimen_id,
        atlas_id=args.atlas_id,
        atlas_image_type=args.atlas_image_type,
        groups=args.groups,
        downsample=args.downsample,
        image_download_mode=args.image_download_mode,
        image_files=image_files,
    )

    if args.annotations_only:
        return

    if not args.skip_masks:
        if "nissl" in args.stains:
            generate_masks(
                nissl,
                image_dir=paths.nissl_images,
                mask_dir=paths.nissl_masks,
                stain="nissl",
                overwrite=args.overwrite,
            )
        if "ihc" in args.stains:
            generate_masks(
                ihc,
                image_dir=paths.ihc_images,
                mask_dir=paths.ihc_masks,
                stain="ihc",
                overwrite=args.overwrite,
            )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download Allen specimen 708424 Nissl/IHC sections, annotations, "
            "and MATLAB-equivalent tissue masks."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("./allen_downloads"),
        help="Output root (default: ./allen_downloads).",
    )
    parser.add_argument(
        "--specimen-id",
        type=int,
        default=DEFAULT_SPECIMEN_ID,
        help=f"Allen specimen ID (default: {DEFAULT_SPECIMEN_ID}).",
    )
    parser.add_argument(
        "--atlas-id",
        type=int,
        default=DEFAULT_ATLAS_ID,
        help=(
            "2-D Allen atlas whose AtlasImage SVG annotations are downloaded "
            f"(default: {DEFAULT_ATLAS_ID}, Modified Brodmann)."
        ),
    )
    parser.add_argument(
        "--atlas-image-type",
        default=DEFAULT_ATLAS_IMAGE_TYPE,
        help=(
            "Allen AlternateImage.image_type used to select the atlas's annotated "
            f"plates (default: {DEFAULT_ATLAS_IMAGE_TYPE!r}). Change this together "
            "with --atlas-id and --groups when selecting another 2-D atlas."
        ),
    )
    parser.add_argument(
        "--groups",
        type=lambda text: tuple(int(item) for item in text.split(",") if item),
        default=DEFAULT_GROUPS,
        help="Comma-separated SVG graphic-group IDs.",
    )
    parser.add_argument(
        "--downsample",
        type=int,
        default=DEFAULT_DOWNSAMPLE,
        help=(
            "Effective -log2 image downsample (default: 5, nominally 32 µm/pixel "
            "for 1 µm/pixel source images)."
        ),
    )
    parser.add_argument(
        "--image-download-mode",
        choices=("allen-direct", "matlab-compatible"),
        default=DEFAULT_IMAGE_DOWNLOAD_MODE,
        help=(
            "allen-direct requests the final Allen pyramid level and preserves the "
            "returned JPEG bytes; matlab-compatible reproduces the original "
            "3DHiResT request-one-level-higher plus local bicubic resize "
            f"(default: {DEFAULT_IMAGE_DOWNLOAD_MODE})."
        ),
    )
    parser.add_argument(
        "--stains",
        nargs="+",
        choices=("nissl", "ihc"),
        default=("nissl", "ihc"),
        help="Stains to process (default: nissl ihc).",
    )
    parser.add_argument(
        "--refresh-metadata",
        action="store_true",
        help="Ignore cached secInfo.json/secInfo.mat and query the API again.",
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Query/load and save metadata, but do not download images or make masks.",
    )
    parser.add_argument(
        "--annotations-only",
        action="store_true",
        help=(
            "Download only the selected atlas's SVG annotations. Section JPEGs "
            "and tissue masks are not processed."
        ),
    )
    parser.add_argument(
        "--skip-downloads",
        action="store_true",
        help="Do not download images/SVGs; useful when files already exist.",
    )
    parser.add_argument(
        "--skip-masks",
        action="store_true",
        help="Do not generate masks.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing images, SVGs, and masks.",
    )
    parser.add_argument(
        "--verify-existing",
        action="store_true",
        help=(
            "For existing JPEGs in allen-direct mode, redownload the corresponding "
            "Allen response and verify an exact byte match without replacing the file. "
            "This is intended for adopting a small number of pre-existing files into "
            "the provenance manifest."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Process only the first N Nissl sections, IHC sections, and atlas "
            "annotation plates (testing aid)."
        ),
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=5,
        help="HTTP retries for transient errors (default: 5).",
    )
    parser.add_argument(
        "--backoff",
        type=float,
        default=1.0,
        help="HTTP retry backoff factor in seconds (default: 1.0).",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
        help="Logging verbosity (default: INFO).",
    )
    args = parser.parse_args(argv)
    if args.downsample < 0:
        parser.error("--downsample must be non-negative")
    if args.verify_existing and args.overwrite:
        parser.error("--verify-existing and --overwrite are mutually exclusive")
    if args.verify_existing and args.image_download_mode != "allen-direct":
        parser.error("--verify-existing currently requires --image-download-mode allen-direct")
    if args.image_download_mode == "matlab-compatible" and args.downsample < 1:
        parser.error(
            "--image-download-mode matlab-compatible requires --downsample >= 1"
        )
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if not args.groups:
        parser.error("--groups must contain at least one integer")
    if args.metadata_only and args.annotations_only:
        parser.error("--metadata-only and --annotations-only cannot be combined")
    if args.annotations_only and args.skip_downloads:
        parser.error("--annotations-only and --skip-downloads cannot be combined")
    if args.annotations_only and "nissl" not in args.stains:
        parser.error("--annotations-only requires nissl in --stains")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    try:
        run(args)
    except KeyboardInterrupt:
        LOGGER.error("Interrupted")
        return 130
    except Exception:
        LOGGER.exception("Failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
