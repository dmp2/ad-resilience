#!/usr/bin/env python3
"""Inspect whether NIfTI files plausibly share a physical coordinate system.

"Same subject" and "same voxel grid" are different claims.  The Ding source material
and OpenNeuro ds003590 reconstruction refer to the same Allen donor and MRI reference
lineage, but files may still differ in resolution, crop, orientation, or transform.
This script reports geometry instead of assuming equivalence.

Usage::

    python src/inspect_spaces.py --output geometry.json file1.nii.gz file2.nii.gz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np

from particle_utils import summarize_nifti_geometry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("niftis", nargs="+", help="NIfTI files to compare.")
    parser.add_argument("--output", help="Optional JSON report path.")
    parser.add_argument(
        "--affine-atol", type=float, default=1e-4, help="Absolute affine tolerance."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reports = []
    images = []
    for filename in args.niftis:
        path = Path(filename).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        image = nib.load(str(path))
        images.append(image)
        reports.append(summarize_nifti_geometry(image, path))

    reference = images[0]
    comparisons = []
    for index, image in enumerate(images[1:], start=1):
        comparisons.append(
            {
                "reference": reports[0]["filename"],
                "other": reports[index]["filename"],
                "same_shape": tuple(reference.shape[:3]) == tuple(image.shape[:3]),
                "affines_close": bool(
                    np.allclose(reference.affine, image.affine, atol=args.affine_atol, rtol=0)
                ),
                "maximum_affine_absolute_difference": float(
                    np.max(np.abs(reference.affine - image.affine))
                ),
                "interpretation": (
                    "Identical shape and close affines support a shared voxel grid. "
                    "Different grids can still share a physical space if a documented "
                    "transform or consistent world coordinate convention exists."
                ),
            }
        )

    report = {"images": reports, "comparisons_to_first": comparisons}
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
