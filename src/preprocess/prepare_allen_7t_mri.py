"""Resolve the archived Allen 7T T1 as a geometry-corrected MRI template.

The immutable archive stores ``T1_rot.mgz`` with placeholder 1-mm voxel sizes.
This command changes only the three voxel-size values in the MGH header to the
documented 0.2-mm acquisition resolution. It does not resample, reorient, or
modify the voxel array.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import os
import shutil
import struct
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Iterable

import nibabel as nib
import numpy as np


LOG = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARCHIVE = Path("data/raw/allen/brainspan_34yr/allen_34yr_7T_structural_mri.tgz")
DEFAULT_OUTPUT_DIR = Path("data/derivatives/allen/specimen_708424/mri_7t_whole")
DEFAULT_PROVENANCE_NAME = "mri_provenance.json"
SELECTED_MEMBER = "T1_rot.mgz"
TEMPLATE_NAME = "T1_rot_space-MRI_7T_WHOLE_desc-header-corrected.nii"
LEGACY_MGZ_TEMPLATE_NAME = "T1_rot_space-MRI_7T_WHOLE_desc-header-corrected.mgz"
SOURCE_VOXEL_SIZE_MM = (1.0, 1.0, 1.0)
CORRECTED_VOXEL_SIZE_MM = (0.2, 0.2, 0.2)
MGH_HEADER_SIZE = 284
MGH_DELTA_OFFSET = 30
MGH_DELTA_END = 42
SUPPORTED_SUFFIXES = (".mgz", ".mgh")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _project_path(path: Path) -> str:
    absolute = path.resolve()
    try:
        return absolute.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(absolute)


def _normalized_member_path(name: str) -> str:
    path = PurePosixPath(name)
    if path.is_absolute() or name.startswith(("/", "\\")):
        raise ValueError(f"Unsafe absolute archive member path: {name!r}")
    if any(part in ("", ".", "..") for part in path.parts):
        raise ValueError(f"Unsafe archive member path: {name!r}")
    return path.as_posix()


def validate_archive_members(
    members: Iterable[tarfile.TarInfo],
) -> dict[str, tarfile.TarInfo]:
    """Reject unsafe tar structure and return normalized member names."""

    validated: dict[str, tarfile.TarInfo] = {}
    for member in members:
        normalized = _normalized_member_path(member.name)
        if normalized in validated:
            raise ValueError(f"Duplicate normalized archive member path: {normalized}")
        if member.issym() or member.islnk():
            raise ValueError(f"Unsafe archive link: {member.name}")
        if member.isdev() or member.isfifo():
            raise ValueError(f"Unsafe archive special file: {member.name}")
        if not (member.isfile() or member.isdir()):
            raise ValueError(f"Unsupported archive member type: {member.name}")
        validated[normalized] = member
    return validated


def _open_mgh_payload(stream: BinaryIO, member_name: str) -> BinaryIO:
    if member_name.lower().endswith(".mgz"):
        return gzip.GzipFile(fileobj=stream, mode="rb")
    return stream


def _parse_mgh_header(raw_header: bytes, member_name: str) -> dict[str, Any]:
    if len(raw_header) != MGH_HEADER_SIZE:
        raise ValueError(f"Truncated MGH header: {member_name}")
    version, width, height, depth, frames, type_code, _dof = struct.unpack(
        ">7i", raw_header[:28]
    )
    ras_good = struct.unpack(">h", raw_header[28:30])[0]
    spacing = struct.unpack(">3f", raw_header[MGH_DELTA_OFFSET:MGH_DELTA_END])
    directions = (
        np.frombuffer(raw_header[42:78], dtype=">f4")
        .astype(np.float64)
        .reshape((3, 3), order="F")
    )
    center = np.frombuffer(raw_header[78:90], dtype=">f4").astype(np.float64)
    dimensions = (width, height, depth)
    if version != 1:
        raise ValueError(f"Unsupported MGH version {version}: {member_name}")
    if ras_good != 1:
        raise ValueError(f"MGH RAS geometry is unavailable: {member_name}")
    if any(value <= 0 for value in (*dimensions, frames)):
        raise ValueError(f"Non-positive MGH dimensions: {member_name}")
    if frames != 1:
        raise ValueError(
            f"Unsupported extra frame/component dimension ({frames}): " f"{member_name}"
        )
    if not np.all(np.isfinite(spacing)) or any(value <= 0 for value in spacing):
        raise ValueError(f"Invalid MGH voxel spacing: {member_name}")
    if not np.all(np.isfinite(directions)) or not np.all(np.isfinite(center)):
        raise ValueError(f"Non-finite MGH physical geometry: {member_name}")
    determinant = float(np.linalg.det(directions))
    if np.isclose(determinant, 0.0):
        raise ValueError(f"Singular MGH direction cosines: {member_name}")
    return {
        "dimensions": dimensions,
        "frames": frames,
        "type_code": type_code,
        "spacing": tuple(float(value) for value in spacing),
        "directions": directions,
        "center": center,
        "direction_determinant": determinant,
    }


def _member_header(bundle: tarfile.TarFile, member: tarfile.TarInfo) -> dict[str, Any]:
    extracted = bundle.extractfile(member)
    if extracted is None:
        raise ValueError(f"Cannot read archive member: {member.name}")
    with extracted:
        payload = _open_mgh_payload(extracted, member.name)
        try:
            raw_header = payload.read(MGH_HEADER_SIZE)
        finally:
            if payload is not extracted:
                payload.close()
    return _parse_mgh_header(raw_header, member.name)


def inspect_archive(archive: Path) -> list[dict[str, Any]]:
    """Return readable scalar MGH/MGZ candidates without extracting them."""

    candidates: list[dict[str, Any]] = []
    try:
        with tarfile.open(archive, mode="r:gz") as bundle:
            members = validate_archive_members(bundle.getmembers())
            for name in sorted(members):
                member = members[name]
                if not member.isfile() or not name.lower().endswith(SUPPORTED_SUFFIXES):
                    continue
                try:
                    header = _member_header(bundle, member)
                except (OSError, EOFError, ValueError):
                    continue
                candidates.append(
                    {
                        "member_path": name,
                        "format": Path(name).suffix.lower().lstrip("."),
                        "dimensions": list(header["dimensions"]),
                        "voxel_size_mm": list(header["spacing"]),
                        "datatype_code": header["type_code"],
                        "frames": header["frames"],
                    }
                )
    except (tarfile.TarError, OSError) as error:
        raise ValueError(f"Unreadable or corrupt MRI archive: {archive}") from error
    return candidates


def select_candidate(
    candidates: list[dict[str, Any]],
    selected_basename: str | None = SELECTED_MEMBER,
) -> dict[str, Any]:
    """Select a unique reviewed basename, or require an unambiguous archive."""

    if selected_basename is None:
        if len(candidates) != 1:
            raise RuntimeError(
                f"ambiguous: found {len(candidates)} plausible MRI candidates"
            )
        return candidates[0]
    matches = [
        candidate
        for candidate in candidates
        if PurePosixPath(candidate["member_path"]).name == selected_basename
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Selected MRI basename {selected_basename!r} matched "
            f"{len(matches)} archive members"
        )
    return matches[0]


def _copy_member_and_hash(
    bundle: tarfile.TarFile,
    member: tarfile.TarInfo,
    output: Path,
) -> str:
    source = bundle.extractfile(member)
    if source is None:
        raise ValueError(f"Cannot read archive member: {member.name}")
    digest = hashlib.sha256()
    with source, output.open("wb") as target:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
            target.write(chunk)
    return digest.hexdigest()


def _correct_mgz_header(source: Path, output: Path) -> None:
    with source.open("rb") as source_raw, gzip.GzipFile(
        fileobj=source_raw, mode="rb"
    ) as decoded:
        header = decoded.read(MGH_HEADER_SIZE)
        parsed = _parse_mgh_header(header, source.name)
        if not np.allclose(
            parsed["spacing"], SOURCE_VOXEL_SIZE_MM, rtol=0.0, atol=1e-6
        ):
            raise ValueError(
                "T1_rot.mgz does not contain the reviewed placeholder "
                f"1-mm spacing: {parsed['spacing']}"
            )
        corrected = (
            header[:MGH_DELTA_OFFSET]
            + struct.pack(">3f", *CORRECTED_VOXEL_SIZE_MM)
            + header[MGH_DELTA_END:]
        )
        with output.open("wb") as output_raw, gzip.GzipFile(
            filename="",
            fileobj=output_raw,
            mode="wb",
            compresslevel=6,
            mtime=0,
        ) as encoded:
            encoded.write(corrected)
            shutil.copyfileobj(decoded, encoded, length=1024 * 1024)


def _direction_cosines(image: nib.spatialimages.SpatialImage) -> np.ndarray:
    spacing = np.asarray(image.header.get_zooms()[:3], dtype=np.float64)
    return np.asarray(image.affine[:3, :3], dtype=np.float64) @ np.diag(1.0 / spacing)


def _physical_center(image: nib.spatialimages.SpatialImage) -> np.ndarray:
    center_voxel = np.asarray(image.shape[:3], dtype=np.float64) / 2.0
    return nib.affines.apply_affine(image.affine, center_voxel)


def _mgh_dtype(type_code: int) -> np.dtype[Any]:
    try:
        return {
            0: np.dtype("u1"),
            1: np.dtype(">i4"),
            3: np.dtype(">f4"),
            4: np.dtype(">i2"),
        }[type_code]
    except KeyError as error:
        raise ValueError(f"Unsupported MGH datatype code: {type_code}") from error


def _decoded_voxel_audit(
    path: Path,
    *,
    lr_axis: int | None = None,
) -> dict[str, Any]:
    """Hash exact decoded voxels and optionally audit bilateral signal.

    MGH stores the voxel array in Fortran order after its 284-byte header.
    Streaming avoids nibabel's 32-bit stride overflow on this 1.1-billion-voxel
    volume while still validating the exact array bytes that the loader reads.
    """

    with path.open("rb") as raw, gzip.GzipFile(fileobj=raw, mode="rb") as stream:
        header = stream.read(MGH_HEADER_SIZE)
        parsed = _parse_mgh_header(header, path.name)
        dimensions = tuple(int(value) for value in parsed["dimensions"])
        dtype = _mgh_dtype(int(parsed["type_code"]))
        voxel_count = int(np.prod(dimensions, dtype=np.int64)) * int(parsed["frames"])
        remaining = voxel_count * dtype.itemsize
        digest = hashlib.sha256()
        offset = 0
        support_counts = [0, 0]
        half_counts = [0, 0]
        while remaining:
            requested = min(8 * 1024 * 1024, remaining)
            chunk = stream.read(requested)
            if len(chunk) != requested:
                raise ValueError(f"Truncated MGH voxel payload: {path}")
            digest.update(chunk)
            if lr_axis is not None:
                values = np.frombuffer(chunk, dtype=dtype)
                indices = np.arange(offset, offset + values.size, dtype=np.int64)
                if lr_axis == 0:
                    coordinates = indices % dimensions[0]
                elif lr_axis == 1:
                    coordinates = (indices // dimensions[0]) % dimensions[1]
                elif lr_axis == 2:
                    coordinates = indices // (dimensions[0] * dimensions[1])
                else:
                    raise ValueError("Anatomical L/R axis must be spatial")
                midpoint = dimensions[lr_axis] // 2
                first = coordinates < midpoint
                finite = np.isfinite(values)
                support = finite & (np.abs(values) > np.finfo(np.float32).eps)
                support_counts[0] += int(np.count_nonzero(support & first))
                support_counts[1] += int(np.count_nonzero(support & ~first))
                half_counts[0] += int(np.count_nonzero(first))
                half_counts[1] += int(np.count_nonzero(~first))
                offset += values.size
            remaining -= requested
    result: dict[str, Any] = {
        "voxel_array_sha256": digest.hexdigest(),
        "voxel_count": voxel_count,
    }
    if lr_axis is not None:
        fractions = [
            support / total if total else 0.0
            for support, total in zip(support_counts, half_counts, strict=True)
        ]
        result["bilateral_signal_fraction"] = fractions
        result["whole_brain_support"] = all(value > 0.001 for value in fractions)
    return result


def _write_lossless_nifti(corrected_mgz: Path, output: Path) -> None:
    """Stream unchanged x-fastest voxels into a NIfTI-1 container."""

    image = nib.load(str(corrected_mgz))
    shape = tuple(int(value) for value in image.shape[:3])
    dtype = image.get_data_dtype()
    if dtype != np.dtype(">f4"):
        raise ValueError(f"Reviewed T1 datatype is not big-endian float32: {dtype}")
    header = nib.Nifti1Header(endianness=">")
    header.set_data_shape(shape)
    header.set_data_dtype(dtype)
    header.set_zooms(tuple(float(value) for value in CORRECTED_VOXEL_SIZE_MM))
    header.set_xyzt_units("mm")
    header.set_sform(image.affine, code="scanner")
    header.set_qform(image.affine, code="scanner")
    header["vox_offset"] = 352.0
    voxel_bytes = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
    with output.open("wb") as target:
        # Nifti1Header.write_to pads through vox_offset, including the extension
        # marker. Adding another four bytes here would shift every voxel.
        header.write_to(target)
        with corrected_mgz.open("rb") as raw, gzip.GzipFile(
            fileobj=raw, mode="rb"
        ) as source:
            source.read(MGH_HEADER_SIZE)
            remaining = voxel_bytes
            while remaining:
                chunk = source.read(min(8 * 1024 * 1024, remaining))
                if not chunk:
                    raise ValueError("Corrected MGZ voxel payload is truncated")
                target.write(chunk)
                remaining -= len(chunk)


def _nifti_voxel_sha256(path: Path) -> str:
    """Hash the exact x-fastest voxel payload, excluding the NIfTI header."""

    image = nib.load(str(path))
    shape = tuple(int(value) for value in image.shape[:3])
    dtype = image.get_data_dtype()
    offset = int(image.dataobj.offset)
    remaining = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        stream.seek(offset)
        while remaining:
            chunk = stream.read(min(8 * 1024 * 1024, remaining))
            if not chunk:
                raise ValueError("Normalized NIfTI voxel payload is truncated")
            digest.update(chunk)
            remaining -= len(chunk)
    return digest.hexdigest()


def _validate_lossless_nifti(
    corrected_mgz: Path,
    normalized_nifti: Path,
    voxel_sha256: str,
) -> None:
    source = nib.load(str(corrected_mgz))
    normalized = nib.load(str(normalized_nifti))
    if source.shape != normalized.shape:
        raise ValueError("Normalized NIfTI shape differs from corrected MGZ")
    if source.get_data_dtype() != normalized.get_data_dtype():
        raise ValueError("Normalized NIfTI datatype differs from corrected MGZ")
    if not np.allclose(source.affine, normalized.affine, rtol=0.0, atol=1e-5):
        raise ValueError("Normalized NIfTI affine differs from corrected MGZ")
    if normalized.header.get_xyzt_units()[0] != "mm":
        raise ValueError("Normalized NIfTI physical units are not millimetres")
    if _nifti_voxel_sha256(normalized_nifti) != voxel_sha256:
        raise ValueError("Normalized NIfTI voxel bytes differ from corrected MGZ")


def _validate_corrected_pair(source_path: Path, corrected_path: Path) -> dict[str, Any]:
    source = nib.load(str(source_path))
    corrected = nib.load(str(corrected_path))
    source_shape = tuple(int(value) for value in source.shape)
    corrected_shape = tuple(int(value) for value in corrected.shape)
    if source_shape != corrected_shape or len(source_shape) not in (3, 4):
        raise ValueError("Corrected MRI shape differs from the source")
    if len(source_shape) == 4 and source_shape[3] != 1:
        raise ValueError("Unsupported MRI frame/component dimension")
    if source.get_data_dtype() != corrected.get_data_dtype():
        raise ValueError("Corrected MRI datatype differs from the source")
    source_spacing = np.asarray(source.header.get_zooms()[:3], dtype=np.float64)
    corrected_spacing = np.asarray(corrected.header.get_zooms()[:3], dtype=np.float64)
    if not np.allclose(source_spacing, SOURCE_VOXEL_SIZE_MM, rtol=0.0, atol=1e-6):
        raise ValueError("Source MRI header does not report reviewed 1-mm spacing")
    if not np.allclose(corrected_spacing, CORRECTED_VOXEL_SIZE_MM, rtol=0.0, atol=1e-6):
        raise ValueError("Corrected MRI does not report 0.2-mm spacing")
    source_directions = _direction_cosines(source)
    corrected_directions = _direction_cosines(corrected)
    if not np.allclose(source_directions, corrected_directions, rtol=0.0, atol=1e-6):
        raise ValueError("Corrected MRI direction cosines changed")
    source_center = _physical_center(source)
    corrected_center = _physical_center(corrected)
    if not np.allclose(source_center, corrected_center, rtol=0.0, atol=1e-4):
        raise ValueError("Corrected MRI physical center changed")
    if nib.aff2axcodes(source.affine) != nib.aff2axcodes(corrected.affine):
        raise ValueError("Corrected MRI anatomical orientation changed")
    determinant = float(np.linalg.det(corrected.affine[:3, :3]))
    if not np.isfinite(determinant) or np.isclose(determinant, 0.0):
        raise ValueError("Corrected MRI affine is singular")
    source_axis_codes = nib.aff2axcodes(source.affine)
    lr_axes = [
        index for index, code in enumerate(source_axis_codes) if code in ("L", "R")
    ]
    if len(lr_axes) != 1:
        raise ValueError("MRI orientation does not identify one anatomical L/R axis")
    source_voxels = _decoded_voxel_audit(source_path)
    corrected_voxels = _decoded_voxel_audit(corrected_path, lr_axis=lr_axes[0])
    if source_voxels["voxel_array_sha256"] != corrected_voxels["voxel_array_sha256"]:
        raise ValueError("Corrected MRI voxel array differs from the source")

    dimensions = np.asarray(corrected.shape[:3], dtype=np.int64)
    corrected_fov = dimensions * np.asarray(CORRECTED_VOXEL_SIZE_MM)
    source_fov = dimensions * np.asarray(SOURCE_VOXEL_SIZE_MM)
    if np.any(corrected_fov < 120.0) or np.any(corrected_fov > 300.0):
        raise ValueError(f"Corrected MRI field of view is implausible: {corrected_fov}")
    if not np.allclose(source_fov, corrected_fov * 5.0):
        raise ValueError("The reviewed 1-mm interpretation is not fivefold")
    if not corrected_voxels["whole_brain_support"]:
        raise ValueError("Corrected MRI lacks plausible bilateral whole-brain support")

    affine = np.asarray(corrected.affine, dtype=np.float64)
    return {
        "dimensions": [int(value) for value in dimensions],
        "frame_count": 1 if len(corrected.shape) == 3 else corrected.shape[3],
        "datatype": str(corrected.get_data_dtype()),
        "source_header_voxel_size_mm": source_spacing.tolist(),
        "corrected_voxel_size_mm": list(CORRECTED_VOXEL_SIZE_MM),
        "direction_cosines": corrected_directions.tolist(),
        "physical_center_mm": corrected_center.tolist(),
        "orientation": "".join(nib.aff2axcodes(corrected.affine)),
        "handedness": "right" if determinant > 0 else "left",
        "voxel_to_physical_affine_mm": affine.tolist(),
        "affine_determinant_mm3": determinant,
        "field_of_view_mm": corrected_fov.tolist(),
        "placeholder_1mm_field_of_view_mm": source_fov.tolist(),
        "voxel_array_sha256": corrected_voxels["voxel_array_sha256"],
        "bilateral_signal_fraction": corrected_voxels["bilateral_signal_fraction"],
        "whole_brain_verified": True,
    }


def _candidate_table(candidates: list[dict[str, Any]]) -> str:
    lines = ["member path\tformat\tdimensions\tvoxel size\tdatatype\tframes"]
    for candidate in candidates:
        lines.append(
            "\t".join(
                (
                    candidate["member_path"],
                    candidate["format"],
                    "x".join(str(value) for value in candidate["dimensions"]),
                    "x".join(str(value) for value in candidate["voxel_size_mm"]),
                    str(candidate["datatype_code"]),
                    str(candidate["frames"]),
                )
            )
        )
    return "\n".join(lines)


def prepare_mri(
    *,
    archive: Path,
    output_dir: Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Select, correct, validate, and record the canonical whole-brain T1."""

    archive = archive.resolve()
    output_dir = output_dir.resolve()
    provenance_path = output_dir / DEFAULT_PROVENANCE_NAME
    template_path = output_dir / TEMPLATE_NAME
    if provenance_path.exists() and template_path.exists() and not overwrite:
        return verify_existing(provenance_path)

    candidates = inspect_archive(archive)
    selected = select_candidate(candidates)
    selected_path = selected["member_path"]
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="allen-7t-mri-", dir=output_dir
    ) as temporary:
        temporary_dir = Path(temporary)
        source_path = temporary_dir / SELECTED_MEMBER
        corrected_path = temporary_dir / LEGACY_MGZ_TEMPLATE_NAME
        normalized_path = temporary_dir / TEMPLATE_NAME
        try:
            with tarfile.open(archive, mode="r:gz") as bundle:
                members = validate_archive_members(bundle.getmembers())
                member = members[selected_path]
                source_member_sha256 = _copy_member_and_hash(
                    bundle, member, source_path
                )
            _correct_mgz_header(source_path, corrected_path)
            geometry = _validate_corrected_pair(source_path, corrected_path)
            _write_lossless_nifti(corrected_path, normalized_path)
            _validate_lossless_nifti(
                corrected_path,
                normalized_path,
                geometry["voxel_array_sha256"],
            )
            template_sha256 = sha256_file(normalized_path)
            os.replace(normalized_path, template_path)
        except Exception:
            if template_path.exists() and overwrite:
                template_path.unlink()
            raise

    provenance: dict[str, Any] = {
        "schema_version": 1,
        "status": "ready",
        "specimen_id": "708424",
        "space_name": "MRI_7T_WHOLE",
        "contrast_name": "7T_T1",
        "source_archive": _project_path(archive),
        "source_archive_sha256": sha256_file(archive),
        "selected_member": SELECTED_MEMBER,
        "source_member_sha256": source_member_sha256,
        "selection_basis": "primary T1 structural registration contrast",
        "template_path": _project_path(template_path),
        "template_sha256": template_sha256,
        "materialization": "lossless_format_normalization",
        "geometry_correction_basis": ("documented Allen 7T acquisition resolution"),
        "geometry_operation": ("header_geometry_correction_without_resampling"),
        "units": "millimeter",
        "voxel_array_changed": False,
        "intensities_changed": False,
        "orientation_changed": False,
        **geometry,
    }
    temporary_provenance = output_dir / f".{DEFAULT_PROVENANCE_NAME}.tmp"
    temporary_provenance.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_provenance, provenance_path)
    legacy_template = output_dir / LEGACY_MGZ_TEMPLATE_NAME
    if legacy_template.is_file():
        legacy_template.unlink()
    return provenance


def verify_existing(provenance_path: Path) -> dict[str, Any]:
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if provenance.get("status") != "ready":
        raise RuntimeError(
            f"MRI provenance status is {provenance.get('status')!r}, not 'ready'"
        )
    required_identity = {
        "specimen_id": "708424",
        "space_name": "MRI_7T_WHOLE",
        "contrast_name": "7T_T1",
        "selected_member": SELECTED_MEMBER,
        "geometry_operation": ("header_geometry_correction_without_resampling"),
    }
    for key, expected in required_identity.items():
        if provenance.get(key) != expected:
            raise ValueError(f"MRI provenance {key} differs from {expected!r}")
    archive = PROJECT_ROOT / provenance["source_archive"]
    template = PROJECT_ROOT / provenance["template_path"]
    if not archive.is_file() or not template.is_file():
        raise FileNotFoundError("MRI archive or materialized template is absent")
    if sha256_file(archive) != provenance["source_archive_sha256"]:
        raise ValueError("MRI source archive checksum mismatch")
    if sha256_file(template) != provenance["template_sha256"]:
        raise ValueError("MRI template checksum mismatch")
    if not np.allclose(
        provenance["corrected_voxel_size_mm"],
        CORRECTED_VOXEL_SIZE_MM,
        rtol=0.0,
        atol=1e-6,
    ):
        raise ValueError("MRI provenance lacks corrected 0.2-mm geometry")
    image = nib.load(str(template))
    if [int(value) for value in image.shape[:3]] != provenance["dimensions"]:
        raise ValueError("MRI template dimensions differ from provenance")
    if not np.allclose(
        image.header.get_zooms()[:3],
        provenance["corrected_voxel_size_mm"],
        rtol=0.0,
        atol=1e-6,
    ):
        raise ValueError("MRI template spacing differs from provenance")
    if "".join(nib.aff2axcodes(image.affine)) != provenance["orientation"]:
        raise ValueError("MRI template orientation differs from provenance")
    return provenance


def _write_failure(output_dir: Path, status: str, reason: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "status": status,
        "specimen_id": "708424",
        "space_name": "MRI_7T_WHOLE",
        "contrast_name": "7T_T1",
        "source_archive": _project_path(DEFAULT_ARCHIVE),
        "selected_member": SELECTED_MEMBER,
        "failure_reason": reason,
    }
    (output_dir / DEFAULT_PROVENANCE_NAME).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--inspect-only", "--inventory-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verify-existing", action="store_true")
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    logging.basicConfig(level=args.log_level, format="%(levelname)s %(message)s")
    if args.inspect_only:
        candidates = inspect_archive(args.archive)
        print(_candidate_table(candidates))
        return 0
    provenance_path = args.output_dir / DEFAULT_PROVENANCE_NAME
    if args.verify_existing:
        provenance = verify_existing(provenance_path)
    else:
        try:
            provenance = prepare_mri(
                archive=args.archive,
                output_dir=args.output_dir,
                overwrite=args.overwrite,
            )
        except RuntimeError as error:
            status = "ambiguous" if str(error).startswith("ambiguous:") else "invalid"
            _write_failure(args.output_dir, status, str(error))
            raise
        except Exception as error:
            _write_failure(args.output_dir, "invalid", str(error))
            raise
    LOG.info("Validated whole-brain 7T T1: %s", provenance["template_path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
