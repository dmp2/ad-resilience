"""Preflight and initialize PV-to-aligned-Nissl rigid section registration.

The fixed Nissl image is loaded in memory from the observed half of the
existing native symmetric derivative. No unilateral Nissl derivative is made.
The saved atlas-free Nissl A2d supplies the initial output-to-input pullback;
only a subsequent PV-specific residual may be fitted.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import tifffile
import nibabel as nib


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "data/derivatives/allen/specimen_708424"
NATIVE = BASE / "emlddmm_7t"
ALIGNED_SYMMETRIC = BASE / "histology_symmetric_nissl_native_200um_section_aligned"
MRI_NISSL = ROOT / ("results/allen/specimen_708424/emlddmm/native-200um-clean/"
                    "HIST_NISSL_SYMMETRIC_SECTION_ALIGNED_to_MRI_7T_WHOLE/"
                    "postprocessed_qc/nissl_reconstruction_on_native_mri_grid.nii")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def load_fixed_nissl(section_number: int, dataset: Path = ALIGNED_SYMMETRIC) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return aligned unilateral RGB and its physical row/column axes."""
    transforms = dataset / "metadata/transforms"
    row_axis = np.load(transforms / "row_axis_um.npy")
    left_axis = np.load(transforms / "left_lr_axis_um.npy")
    image = tifffile.imread(
        dataset / "inputs/views/HIST_NISSL" /
        f"allen_708424_nissl_{section_number:04d}.tif"
    )
    width = len(left_axis)
    if image.shape != (len(row_axis), width * 2, 3):
        raise ValueError("Aligned Nissl source has unexpected shape")
    observed = image[:, width:]
    if not np.array_equal(image[:, :width], observed[:, ::-1]):
        raise ValueError("Symmetric source did not preserve the observed half")
    return observed, row_axis, left_axis


def pairing_and_initializers(
    native: Path = NATIVE, aligned: Path = ALIGNED_SYMMETRIC,
) -> tuple[list[dict], np.ndarray]:
    source_rows = _rows(native / "metadata/physical_sections.tsv")
    target_rows = _rows(aligned / "metadata/physical_sections.tsv")
    if len(source_rows) != len(target_rows) or len(source_rows) != 2846:
        raise ValueError("Physical section lattices differ")
    for source, target in zip(source_rows, target_rows):
        for key in ("physical_index", "allen_section_number", "serial_z_center_mm",
                    "stain", "allen_section_image_id"):
            if source[key] != target[key]:
                raise ValueError(f"Nissl source identity changed: {key}")
    transforms = aligned / "metadata/transforms"
    provenance = json.loads((transforms / "provenance.json").read_text())
    symmetry = json.loads((aligned / "metadata/symmetry.json").read_text())
    if (provenance["schema"] != "allen-native-symmetric-section-aligned-v1"
            or symmetry["operation"] != "exact_reflection_no_medial_duplication"):
        raise ValueError("Fixed Nissl source lacks expected section alignment")
    matrices = np.load(transforms / "atlas_free_expanded_A2d.npy")
    observed = np.load(transforms / "observed_physical_indices.npy")
    baseline = np.loadtxt(transforms / "atlas_free_common_bookkeeping_frame.txt")
    if matrices.shape != (len(source_rows), 3, 3):
        raise ValueError("Saved Nissl A2d has unexpected shape")
    unsupported = np.ones(len(source_rows), dtype=bool)
    unsupported[observed] = False
    if not np.array_equal(matrices[unsupported],
                          np.broadcast_to(baseline, matrices[unsupported].shape)):
        raise ValueError("Atlas-free common frame no longer matches")
    residuals = np.linalg.inv(baseline)[None] @ matrices
    nissl = [r for r in source_rows if r["stain"] == "nissl" and r["image_present"] == "true"]
    pv = [r for r in source_rows if r["stain"] == "pv" and r["image_present"] == "true"]
    if len(nissl) != 641 or len(pv) != 287:
        raise ValueError("Observed Nissl/PV counts changed")
    pairs: list[dict] = []
    initial = np.empty((len(pv), 3, 3), dtype=np.float64)
    for order, moving in enumerate(pv):
        block = moving["block_id"]
        candidates = [r for r in nissl if r["block_id"] == block] if block else nissl
        if not candidates:
            raise ValueError(f"No Nissl candidates for PV {moving['allen_section_number']}")
        source_id = int(moving["allen_section_number"])
        separations = [abs(int(r["allen_section_number"]) - source_id) for r in candidates]
        nearest = min(separations)
        choices = [r for r, gap in zip(candidates, separations) if gap == nearest]
        if len(choices) != 1:
            raise ValueError(f"Ambiguous nearest Nissl for PV {source_id}")
        fixed = choices[0]
        fixed_index = int(fixed["physical_index"])
        matrix = residuals[fixed_index]
        if (not np.all(np.isfinite(matrix))
                or not np.allclose(matrix[2], [0, 0, 1], atol=1e-6)
                or not np.allclose(matrix[:2, :2].T @ matrix[:2, :2],
                                   np.eye(2), atol=1e-3)
                or np.linalg.det(matrix[:2, :2]) <= 0):
            raise ValueError(f"Nissl initializer is not rigid: {source_id}")
        initial[order] = matrix
        status = "ready" if block and nearest <= 3 else "review_pairing"
        pairs.append({
            "pv_source_section_id": source_id,
            "pv_source_image_id": moving["allen_section_image_id"],
            "pv_source_physical_index": int(moving["physical_index"]),
            "pv_source_z_mm": float(moving["serial_z_center_mm"]),
            "pv_prepared_relative_path": moving["prepared_relative_path"],
            "nissl_source_section_id": int(fixed["allen_section_number"]),
            "nissl_source_image_id": fixed["allen_section_image_id"],
            "nissl_physical_index": fixed_index,
            "canonical_nissl_z_mm": float(fixed["serial_z_center_mm"]),
            "serial_section_offset": int(fixed["allen_section_number"]) - source_id,
            "estimated_separation_um": nearest * 50.0,
            "pairing_rule": "nearest observed Nissl section in matching block"
                            if block else "nearest observed Nissl; PV block unresolved",
            "pairing_qc_status": status,
            "initial_matrix_index": order,
            "initial_matrix_direction": "aligned-Nissl grid to raw-PV grid pullback",
        })
    return pairs, initial


def preflight() -> dict:
    pairs, initial = pairing_and_initializers()
    source = json.loads((NATIVE / "dataset.json").read_text())
    symmetric = json.loads((ALIGNED_SYMMETRIC / "dataset.json").read_text())
    symmetry = json.loads((ALIGNED_SYMMETRIC / "metadata/symmetry.json").read_text())
    transforms = ALIGNED_SYMMETRIC / "metadata/transforms"
    axis = np.load(transforms / "left_lr_axis_um.npy")
    row = np.load(transforms / "row_axis_um.npy")
    serial = np.load(transforms / "serial_axis_um.npy")
    mri_image = nib.load(str(MRI_NISSL))
    return {
        "fixed_target": {
            "representation": "observed high-column half of aligned symmetric Nissl, loaded in memory",
            "source_path": str(ALIGNED_SYMMETRIC),
            "space_name": "HIST_NISSL_HEMISPHERE_SECTION_ALIGNED_NATIVE_200UM",
            "shape_yx": [len(row), len(axis)],
            "spacing_um": [float(serial[1]-serial[0]), float(row[1]-row[0]), float(axis[1]-axis[0])],
            "origin_xy_um": [float(axis[0]), float(row[0])],
            "persisted_separately": False,
            "hemisphere": "observed unilateral; section aligned",
            "mri_geometry_applied": False,
        },
        "original_nissl": {
            "path": str(NATIVE / "inputs/sections/nissl"),
            "allen_raw_jpeg_path": str(ROOT / "data/raw/allen/specimen_708424/nissl/images_orig"),
            "space_name": "HIST_NISSL_LEFT",
            "shape_yx": source["prepared_canvas_shape_yx"],
            "spacing_um": [50.0, 200.0, 200.0],
            "hemisphere": "observed unilateral",
            "mri_geometry_applied": False,
        },
        "symmetric_nissl": {
            "path": str(ALIGNED_SYMMETRIC),
            "space_name": symmetric["space_name"],
            "shape_yx": symmetry["bilateral_shape_yx"],
            "spacing_um": [symmetry["serial_spacing_um"], symmetry["pixel_size_um"], symmetry["pixel_size_um"]],
            "hemisphere": "observed plus exact reflected copy",
            "mri_geometry_applied": False,
        },
        "mri_grid_nissl": {
            "path": str(MRI_NISSL),
            "space_name": "MRI_7T_WHOLE",
            "shape": list(mri_image.shape),
            "spacing_mm": list(map(float, mri_image.header.get_zooms()[:3])),
            "hemisphere": "bilateral; derived from reflected Nissl",
            "mri_geometry_applied": True,
        },
        "pair_count": len(pairs),
        "pairing_qc_counts": {
            status: sum(p["pairing_qc_status"] == status for p in pairs)
            for status in ("ready", "review_pairing")
        },
        "pairing_review_section_ids": [p["pv_source_section_id"] for p in pairs
                                       if p["pairing_qc_status"] != "ready"],
        "initial_A2d_source_sha256": _sha256(transforms / "atlas_free_expanded_A2d.npy"),
        "initial_shape": list(initial.shape),
        "symmetry_method": "exact reflection; no estimated Nissl symmetry transform",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    report = preflight()
    if args.output_dir:
        output = args.output_dir
        output.mkdir(parents=True, exist_ok=True)
        pairs, initial = pairing_and_initializers()
        with (output / "pv_nissl_pairs.tsv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(pairs[0]), delimiter="\t")
            writer.writeheader()
            writer.writerows(pairs)
        np.save(output / "pv_nissl_rigid_initial_A2d.npy", initial)
        (output / "preflight.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
