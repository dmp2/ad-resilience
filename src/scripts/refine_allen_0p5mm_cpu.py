"""Short CPU continuation on genuine 0.5 mm images, with fixed physical bounds."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt
import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np
import torch

import tune_allen_1mm_registration as tuning
from export_allen_native_labels import NATIVE_TARGET

INPUTS = tuning.TUNING / "cpu_0p5mm_inputs"
ORIGINAL_TEMPLATE = tuning.TEMPLATE.with_name(
    "ahrav2_icbm2009b_symmetric_template_500um.nii.gz"
)


def fine_reference(reference):
    """Twice the sample density, identical endpoint coordinates to the 1 mm fit."""
    shape = tuple(2 * n - 1 for n in reference.shape)
    affine = reference.affine.copy()
    affine[:3, :3] *= 0.5
    return shape, affine


def prepare_inputs():
    INPUTS.mkdir(parents=True, exist_ok=True)
    paths = dict(
        template=INPUTS / "atlas_0p5mm.nii.gz",
        target=INPUTS / "target_0p5mm.nii.gz",
        mask=INPUTS / "synthstrip_nocsf_mask_0p5mm.nii.gz",
        labels=INPUTS / "labels_0p5mm.nii.gz",
    )
    if not all(p.exists() for p in paths.values()):
        for source, fitted, destination in (
            (ORIGINAL_TEMPLATE, tuning.TEMPLATE, paths["template"]),
            (NATIVE_TARGET, tuning.TARGET, paths["target"]),
        ):
            print("Resampling original image:", source.name, flush=True)
            # Float32 avoids loading the 1.1-billion-voxel native MRI as float64.
            original = nib.load(source)
            image = nib.Nifti1Image(
                np.asarray(original.dataobj, dtype=np.float32), original.affine
            )
            output = resample_from_to(image, fine_reference(nib.load(fitted)), order=1)
            nib.save(output, destination)
            del image, output, original
        atlas_reference = fine_reference(nib.load(tuning.TEMPLATE))
        for source, destination in (
            (tuning.TUNING / "masks/synthstrip_nocsf_mask.nii.gz", paths["mask"]),
            (tuning.LABELS, paths["labels"]),
        ):
            nib.save(
                resample_from_to(nib.load(source), atlas_reference, order=0),
                destination,
            )
    arrays = {}
    thresholds = {}
    axes = {}
    for key, fitted in (("template", tuning.TEMPLATE), ("target", tuning.TARGET)):
        old = nib.load(fitted)
        old_array = np.asarray(old.dataobj, dtype=np.float32)
        lo, hi = np.percentile(old_array[np.isfinite(old_array)], [0.5, 99.5])
        thresholds[key] = [float(lo), float(hi)]
        fine = nib.load(paths[key])
        image = np.asarray(fine.dataobj, dtype=np.float32)
        arrays[key] = ((np.clip(image, lo, hi) - lo) / (hi - lo)).astype(np.float32)[
            None
        ]
        axes[key] = [
            (np.arange(2 * n - 1, dtype=np.float32) * 0.5 - (n - 1) / 2)
            for n in old.shape
        ]
    mask = (np.asarray(nib.load(paths["mask"]).dataobj) > 0)[None].astype(np.float32)
    labels = np.asarray(nib.load(paths["labels"]).dataobj, dtype=np.int32)[None]
    data = dict(
        xI=axes["template"],
        I=arrays["template"],
        I_brain=arrays["template"] * mask,
        xJ=axes["target"],
        J=arrays["target"],
        S=labels,
        brain_mask=mask,
        label_mask=(labels != 0).astype(np.float32),
        atlas_mask="synthstrip_nocsf",
        atlas_mask_sha256=hashlib.sha256(mask.astype(np.uint8).tobytes()).hexdigest(),
    )
    provenance = dict(
        original_template=str(ORIGINAL_TEMPLATE),
        original_target=str(NATIVE_TARGET),
        fitted_template_reference=str(tuning.TEMPLATE),
        fitted_target_reference=str(tuning.TARGET),
        image_interpolation="linear from original data; equal 0.5 mm sampling on both images",
        mask_interpolation="nearest from existing 1 mm no-CSF mask",
        normalization_thresholds_from_1mm=thresholds,
        coordinate_convention="Same centered coordinates and endpoint bounds as selected 1 mm fit; unchanged 2 mm velocity domain.",
        files={k: str(v) for k, v in paths.items()},
    )
    (INPUTS / "provenance.json").write_text(json.dumps(provenance, indent=2))
    return data, paths


def run(parent, name, iterations=50, threads=6):
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    data, paths = prepare_inputs()
    previous = tuning.load_result(parent)
    # The input bounds and dv are identical, so no implicit sinc velocity resize occurs.
    expected_axes = [
        np.arange(
            x[0] - (x[-1] - x[0]) * 0.2,
            x[-1] + (x[-1] - x[0]) * 0.2,
            2.0,
            dtype=np.float32,
        )
        for x in data["xI"]
    ]
    for actual, expected in zip(previous["xv"], expected_axes):
        if actual.shape != expected.shape or not np.allclose(
            actual, expected, atol=0.002
        ):
            raise ValueError(
                "Parent velocity domain does not match the retained physical bounds"
            )
    config = tuning.resume_config(
        data, parent, scales=[1], iterations=[iterations], half_steps=True
    )
    config.update(device="cpu", dtype="float32", n_draw=10, local_contrast=[[128] * 3])
    directory = tuning.TUNING / name
    directory.mkdir(parents=True, exist_ok=True)
    print(
        "CPU pilot:",
        name,
        "threads:",
        threads,
        "image shapes:",
        data["I"].shape,
        data["J"].shape,
        flush=True,
    )
    (directory / "pilot.json").write_text(
        json.dumps(
            dict(
                parent=parent,
                threads=threads,
                image_spacing_mm=0.5,
                velocity_spacing_mm=2.0,
                iterations=iterations,
                eA=config["eA"],
                ev=config["ev"],
                selection="Trial only; current selected result retained.",
            ),
            indent=2,
        )
    )
    old_template, old_target = tuning.TEMPLATE, tuning.TARGET
    tuning.TEMPLATE, tuning.TARGET = paths["template"], paths["target"]
    try:
        result = tuning.run_candidate(name, data, config, parent=parent)
        tuning.export_selected(data, name, select=False, mask_zero_background=True)
    finally:
        tuning.TEMPLATE, tuning.TARGET = old_template, old_target
    # Identical 1 mm display samples make comparison independent of plot resolution.
    display_data = dict(data)
    for key in ("I", "I_brain", "J", "S", "brain_mask", "label_mask"):
        display_data[key] = data[key][:, ::2, ::2, ::2]
    for key in ("xI", "xJ"):
        display_data[key] = [x[::2] for x in data[key]]
    for label, transform in (
        ("parent_same_slices", previous),
        ("pilot_same_slices", result),
    ):
        review = directory / label
        review.mkdir(exist_ok=True)
        tuning.visual_review(display_data, transform, review)
    print(
        "CPU pilot finished. Previous selection retained for visual comparison.",
        flush=True,
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", default="fresh_nocsf_boundary_refine")
    parser.add_argument("--name", default="fresh_nocsf_0p5mm_cpu_pilot")
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()
    run(args.parent, args.name, args.iterations, args.threads)
