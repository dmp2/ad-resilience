#!/usr/bin/env python3
"""Download and preprocess Allen human brain atlas histology sections.

Python translation of:
    acasamitjana/3dhirest/database/preprocessing/download_allen.m

The workflow:
1. Query section metadata for Allen specimen 708424.
2. Separate Nissl (treatment ID 3) and IHC/parvalbumin (treatment ID 16).
3. Download JPEG section images at the requested effective downsample level.
4. Download available Nissl SVG annotations.
5. Generate tissue masks using the thresholds and morphology in the MATLAB code.
6. Save metadata as both JSON and MATLAB-compatible ``secInfo.mat``.

Notes on fidelity and fixes
---------------------------
* Allen image downsample is logarithmic: level n reduces each image axis by 2**n.
  To match the MATLAB code, images are requested at ``downsample - 1`` and then
  resized by 1/2 locally with bicubic interpolation.
* The original MATLAB script references undefined variables ``NISSL_DIR`` and
  ``it_slice``. Here, the Nissl directory is derived from ``data_dir`` and the
  intended one-based loop ordinal is used for the slice-specific thresholds.
* The original SVG URL contains two ``?`` characters and requests a documented-
  unsupported SVG downsample argument. This implementation requests the SVG by
  graphic-group IDs only; SVGs remain vector graphics and retain their viewBox.
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
import io
import json
import logging
import os
import sys
import tempfile
from dataclasses import asdict, dataclass
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
DEFAULT_ATLAS_ID = 265297126  # Retained for provenance; unused by these endpoints.
DEFAULT_SPECIMEN_ID = 708424
DEFAULT_GROUPS = (31, 113753816, 141667008, 265297118)  # Brodmann groups.
DEFAULT_DOWNSAMPLE = 5
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
        )

    def create(self) -> None:
        for directory in (
            self.data_dir,
            self.nissl_images,
            self.nissl_labels,
            self.nissl_masks,
            self.ihc_images,
            self.ihc_masks,
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
                )
                records[(record.stain, record.section_id)] = record

            if IHC_TREATMENT_ID in treatment_ids:
                record = SectionRecord(
                    stain="ihc",
                    section_id=int(section_id),
                    section_number=int(section_number),
                    annotated=False,
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
    return nissl, ihc


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


def save_metadata(
    paths: OutputPaths,
    nissl: Sequence[SectionRecord],
    ihc: Sequence[SectionRecord],
    *,
    specimen_id: int,
    atlas_id: int,
    groups: Sequence[int],
    downsample: int,
) -> None:
    """Save transparent JSON metadata and a MATLAB-compatible secInfo.mat."""

    payload = {
        "specimen_id": specimen_id,
        "atlas_id": atlas_id,
        "groups": list(groups),
        "downsample": downsample,
        "nissl": [asdict(record) for record in nissl],
        "ihc": [asdict(record) for record in ihc],
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
            "DS_FACTOR": np.asarray([[downsample]], dtype=np.int64),
        },
    )
    LOGGER.info("Saved metadata to %s and %s", paths.metadata_json, paths.metadata_mat)


def load_cached_metadata(paths: OutputPaths) -> tuple[list[SectionRecord], list[SectionRecord]]:
    """Load metadata from JSON, falling back to the original MATLAB cache format."""

    if paths.metadata_json.exists():
        payload = json.loads(paths.metadata_json.read_text(encoding="utf-8"))
        nissl = [SectionRecord(**item) for item in payload.get("nissl", [])]
        ihc = [SectionRecord(**item) for item in payload.get("ihc", [])]
        if nissl or ihc:
            LOGGER.info("Loaded cached metadata from %s", paths.metadata_json)
            return nissl, ihc

    if paths.metadata_mat.exists():
        mat = loadmat(paths.metadata_mat, squeeze_me=True)
        nissl_ids = np.atleast_1d(mat.get("secIDNissl", [])).astype(int)
        nissl_numbers = np.atleast_1d(mat.get("secNumberNissl", [])).astype(int)
        nissl_annotated = np.atleast_1d(mat.get("annotatedNissl", [])).astype(bool)
        ihc_ids = np.atleast_1d(mat.get("secIDIHC", [])).astype(int)
        ihc_numbers = np.atleast_1d(mat.get("secNumberIHC", [])).astype(int)

        if not (len(nissl_ids) == len(nissl_numbers) == len(nissl_annotated)):
            raise RuntimeError(f"Inconsistent Nissl arrays in {paths.metadata_mat}")
        if len(ihc_ids) != len(ihc_numbers):
            raise RuntimeError(f"Inconsistent IHC arrays in {paths.metadata_mat}")

        nissl = [
            SectionRecord("nissl", int(section_id), int(section_number), bool(annotated))
            for section_id, section_number, annotated in zip(
                nissl_ids, nissl_numbers, nissl_annotated, strict=True
            )
        ]
        ihc = [
            SectionRecord("ihc", int(section_id), int(section_number), False)
            for section_id, section_number in zip(ihc_ids, ihc_numbers, strict=True)
        ]
        LOGGER.info("Loaded cached metadata from %s", paths.metadata_mat)
        return nissl, ihc

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
    downsample: int,
    overwrite: bool,
    timeout: tuple[float, float] = (20.0, 300.0),
) -> None:
    """Download one section image and reproduce the MATLAB two-stage resize."""

    if destination.exists() and not overwrite:
        LOGGER.info("Image exists; skipping %s", destination)
        return
    if downsample < 1:
        raise ValueError(
            "The MATLAB-compatible two-stage download requires downsample >= 1."
        )

    url = f"{API_BASE}/image_download/{record.section_id}"
    LOGGER.info(
        "Downloading %s section %d (image ID %d)",
        record.stain.upper(),
        record.section_number,
        record.section_id,
    )
    response = session.get(
        url,
        params={"downsample": downsample - 1, "quality": 100},
        timeout=timeout,
    )
    response.raise_for_status()

    try:
        with Image.open(io.BytesIO(response.content)) as source:
            source.load()
            image = source.convert("RGB")
    except UnidentifiedImageError as exc:
        content_type = response.headers.get("Content-Type", "unknown")
        raise RuntimeError(
            f"Image endpoint returned unreadable data for section {record.section_id} "
            f"(Content-Type: {content_type})"
        ) from exc

    output_size = (max(1, image.width // 2), max(1, image.height // 2))
    image = image.resize(output_size, resample=Image.Resampling.BICUBIC)
    _atomic_save_pil(image, destination, format="JPEG", quality=100, subsampling=0)


def download_svg(
    session: requests.Session,
    record: SectionRecord,
    destination: Path,
    *,
    groups: Sequence[int],
    overwrite: bool,
    timeout: tuple[float, float] = (20.0, 180.0),
) -> None:
    """Download an annotated Nissl section as SVG."""

    if not record.annotated:
        LOGGER.info(
            "Annotations unavailable for Nissl section %d", record.section_number
        )
        return
    if destination.exists() and not overwrite:
        LOGGER.info("SVG exists; skipping %s", destination)
        return

    url = f"{API_BASE}/svg_download/{record.section_id}"
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
            f"SVG endpoint returned unexpected data for section {record.section_id} "
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

    if args.refresh_metadata:
        nissl, ihc = query_section_metadata(session, specimen_id=args.specimen_id)
        save_metadata(
            paths,
            nissl,
            ihc,
            specimen_id=args.specimen_id,
            atlas_id=args.atlas_id,
            groups=args.groups,
            downsample=args.downsample,
        )
    else:
        try:
            nissl, ihc = load_cached_metadata(paths)
        except FileNotFoundError:
            nissl, ihc = query_section_metadata(session, specimen_id=args.specimen_id)
            save_metadata(
                paths,
                nissl,
                ihc,
                specimen_id=args.specimen_id,
                atlas_id=args.atlas_id,
                groups=args.groups,
                downsample=args.downsample,
            )

    nissl = list(_limited(nissl, args.limit))
    ihc = list(_limited(ihc, args.limit))

    if args.metadata_only:
        return

    if not args.skip_downloads:
        if "nissl" in args.stains:
            for ordinal, record in enumerate(nissl, start=1):
                LOGGER.info("NISSL section %d/%d", ordinal, len(nissl))
                download_image(
                    session,
                    record,
                    paths.nissl_images / f"image_{record.section_number:04d}.jpg",
                    downsample=args.downsample,
                    overwrite=args.overwrite,
                )
                download_svg(
                    session,
                    record,
                    paths.nissl_labels / f"seg_{record.section_number:04d}.svg",
                    groups=args.groups,
                    overwrite=args.overwrite,
                )

        if "ihc" in args.stains:
            for ordinal, record in enumerate(ihc, start=1):
                LOGGER.info("IHC section %d/%d", ordinal, len(ihc))
                download_image(
                    session,
                    record,
                    paths.ihc_images / f"image_{record.section_number:04d}.jpg",
                    downsample=args.downsample,
                    overwrite=args.overwrite,
                )

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
            f"Atlas provenance ID saved in metadata (default: {DEFAULT_ATLAS_ID}); "
            "the original script also does not pass it to the download endpoints."
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
            "Effective -log2 image downsample. The script requests one level higher "
            "and halves locally, as in MATLAB (default: 5)."
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
        "--limit",
        type=int,
        default=None,
        help="Process only the first N sections of each selected stain (testing aid).",
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
    if args.downsample < 1:
        parser.error("--downsample must be at least 1 for MATLAB-compatible resizing")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if not args.groups:
        parser.error("--groups must contain at least one integer")
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
