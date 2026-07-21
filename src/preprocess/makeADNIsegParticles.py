#!/usr/bin/env python3
"""Create target particles from one or more labeled NIfTI volumes.

Historical name
===============
The colleague's script was named ``makeADNIsegParticles.py`` because its original
targets were ADNI-like low-field segmentations.  The implementation below is generic:
a "target" can be an individual MRI segmentation, an OpenNeuro reconstruction label
volume, a SEA-AD-derived label image, or any other 3-D integer segmentation.

It does *not* convert an MRI intensity image, diffusion image, Nissl image, or gene
expression volume directly.  Those data first need an explicit feature construction
or segmentation.  This distinction prevents accidentally treating intensity values as
anatomical labels.

Usage
-----
Create all files for a named target job::

    python src/makeADNIsegParticles.py \
        --config config/project_config.yaml \
        --job example_target

A target job may use either ``input_nifti`` for one file or ``input_glob`` for a cohort.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np

from particle_utils import (
    construct_particles,
    feature_definition_from_config,
    save_particle_archive,
    summarize_nifti_geometry,
)
from project_config import get_named, load_config, resolve_path
from xmodmap_compat import (
    categorical_particles_xmodmap_native,
    entropy_numpy,
    load_xmodmap,
    write_vtk_xyz,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Project YAML configuration.")
    parser.add_argument("--job", required=True, help="Name under particle_jobs.targets.")
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace existing NPZ/VTK outputs."
    )
    return parser.parse_args()


def _target_job(config: dict, name: str) -> dict:
    jobs = config.get("particle_jobs", {}).get("targets", {})
    if name not in jobs:
        available = ", ".join(sorted(jobs)) or "<none>"
        raise KeyError(f"Unknown target job '{name}'. Available: {available}")
    return dict(jobs[name])


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    job = _target_job(config, args.job)

    label_map_name = job.get("label_map")
    label_map = None
    if label_map_name:
        label_map = get_named(config, "label_maps", label_map_name)

    # Resolve one input file or a glob.  Globs are useful for subject cohorts, whereas
    # a named Allen/OpenNeuro prototype will usually use one explicit file.
    if "input_nifti" in job:
        input_files = [resolve_path(config, job["input_nifti"])]
    elif "input_glob" in job:
        pattern = resolve_path(config, job["input_glob"])
        input_files = sorted(pattern.parent.glob(pattern.name))
    else:
        raise ValueError("Target job needs input_nifti or input_glob.")
    if not input_files:
        raise FileNotFoundError("No target NIfTI files matched the configured job.")

    output_dir = resolve_path(config, job["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    background = job.get("background_labels", [0])
    factor = int(job.get("downsample_factor", 1))
    coordinate_mode = str(job.get("coordinate_mode", "affine_world"))
    coordinate_space = str(job.get("coordinate_space", "unspecified"))
    backend = str(job.get("construction_backend", "project_affine_block"))
    api = load_xmodmap(config)
    vtk_convention = str(job.get("vtk_coordinate_convention", "xyz"))
    vtk_writer = lambda points, arrays, names, filename: write_vtk_xyz(
        api, points, arrays, names, filename, coordinate_convention=vtk_convention
    )
    entropy_function = lambda nu: entropy_numpy(api, nu)

    for input_file in input_files:
        if not input_file.is_file():
            raise FileNotFoundError(f"Target segmentation is missing: {input_file}")
        image = nib.load(str(input_file))
        segmentation = np.asanyarray(image.dataobj)
        features = feature_definition_from_config(segmentation, label_map, background)

        if backend == "xmodmap_native":
            if factor != 1 or coordinate_mode != "legacy_centered":
                raise ValueError(
                    "xmodmap_native categorical construction is limited to native "
                    "sampling with coordinate_mode=legacy_centered. Use "
                    "project_affine_block for Allen/OpenNeuro affine-world data or "
                    "for mass-conserving 3-D block aggregation."
                )
            voxel_sizes = np.sqrt(np.sum(np.asarray(image.affine)[:3, :3] ** 2, axis=0))
            voxel_volume = float(abs(np.linalg.det(np.asarray(image.affine)[:3, :3])))
            X, nu_all = categorical_particles_xmodmap_native(
                api,
                input_file,
                voxel_sizes_mm=voxel_sizes,
                voxel_volume_mm3=voxel_volume,
                fine_label_ids=features.fine_ids,
                background_labels=background,
            )
            grouping = np.zeros((features.number_fine, features.number_coarse), dtype=np.float32)
            grouping[np.arange(features.number_fine), np.asarray(features.fine_to_coarse)] = 1.0
            nu_sub = nu_all @ grouping
            diagnostics = {
                "construction_backend": "xmodmap_native",
                "downsample_factor": 1,
                "number_particles": int(X.shape[0]),
                "voxel_volume_mm3": voxel_volume,
                "coordinate_mode": coordinate_mode,
                "note": "Created by xmodmap.io.getInput.makeFromSingleChannelImage.",
            }
        elif backend == "project_affine_block":
            X, nu_sub, nu_all, diagnostics = construct_particles(
                segmentation=segmentation,
                affine=image.affine,
                features=features,
                downsample_factor=factor,
                coordinate_mode=coordinate_mode,
            )
            diagnostics["construction_backend"] = backend
        else:
            raise ValueError(
                "construction_backend must be xmodmap_native or project_affine_block."
            )

        # Correctly remove both .nii and .nii.gz from the output stem.
        name = input_file.name[:-7] if input_file.name.endswith(".nii.gz") else input_file.stem
        output_npz = output_dir / f"{name}_ds{factor}.npz"
        if output_npz.exists() and not args.overwrite:
            print(f"Skipping existing output: {output_npz}")
            continue

        geometry = summarize_nifti_geometry(image, input_file)
        metadata = {
            "particle_role": "target",
            "job_name": args.job,
            "source_nifti": str(input_file),
            "coordinate_space": coordinate_space,
            "geometry": geometry,
            "diagnostics": diagnostics,
            "label_map": label_map_name or "inferred_all_nonbackground",
            "construction_backend": backend,
            "notes": job.get("notes", ""),
        }
        save_particle_archive(
            output_npz,
            X,
            nu_sub,
            nu_all,
            features,
            metadata,
            write_vtk=bool(job.get("write_vtk", True)),
            vtk_writer=vtk_writer,
            entropy_function=entropy_function,
        )
        print(json.dumps({"created": str(output_npz), **diagnostics}, indent=2))


if __name__ == "__main__":
    main()
