#!/usr/bin/env python3
"""Create full- and reduced-resolution source/atlas particles.

This modernizes the colleague's high-field template converter while preserving its
central mathematical idea: a coarse particle stores the complete label composition of
a voxel block, not only its majority label.

The source can be:

* a Ding/OpenNeuro same-donor segmentation in subject space;
* the Allen Human Reference Atlas annotation volume in MNI ICBM 2009b symmetric space;
* another anatomical reference label volume.

Those sources are not automatically interchangeable.  The NPZ metadata records the
coordinate space, and the registration script checks it.  The Allen HRA is an MNI
population-template parcellation; it is not the native Ding donor volume merely because
its ontology is adapted from Ding's work.

Usage::

    python src/makeHighFieldTemplateParticles.py \
        --config config/project_config.yaml \
        --job ding_subject_source
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
from config_setup.project_config import get_named, load_config, resolve_path
from xmodmap_compat import (
    categorical_particles_xmodmap_native,
    entropy_numpy,
    load_xmodmap,
    write_vtk_xyz,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Project YAML configuration.")
    parser.add_argument("--job", required=True, help="Name under particle_jobs.sources.")
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace existing NPZ/VTK outputs."
    )
    return parser.parse_args()


def _source_job(config: dict, name: str) -> dict:
    jobs = config.get("particle_jobs", {}).get("sources", {})
    if name not in jobs:
        available = ", ".join(sorted(jobs)) or "<none>"
        raise KeyError(f"Unknown source job '{name}'. Available: {available}")
    return dict(jobs[name])


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    job = _source_job(config, args.job)

    input_file = resolve_path(config, job["input_nifti"])
    if not input_file.is_file():
        raise FileNotFoundError(f"Source segmentation is missing: {input_file}")

    label_map_name = job.get("label_map")
    label_map = None
    if label_map_name:
        label_map = get_named(config, "label_maps", label_map_name)

    image = nib.load(str(input_file))
    segmentation = np.asanyarray(image.dataobj)
    features = feature_definition_from_config(
        segmentation,
        label_map,
        job.get("background_labels", [0]),
    )

    output_dir = resolve_path(config, job["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    factors = [int(value) for value in job.get("downsample_factors", [1, 5])]
    if len(set(factors)) != len(factors):
        raise ValueError("downsample_factors contains duplicate values.")

    name = input_file.name[:-7] if input_file.name.endswith(".nii.gz") else input_file.stem
    coordinate_mode = str(job.get("coordinate_mode", "affine_world"))
    coordinate_space = str(job.get("coordinate_space", "unspecified"))
    geometry = summarize_nifti_geometry(image, input_file)
    backend = str(job.get("construction_backend", "project_affine_block"))
    api = load_xmodmap(config)
    vtk_convention = str(job.get("vtk_coordinate_convention", "xyz"))
    vtk_writer = lambda points, arrays, names, filename: write_vtk_xyz(
        api, points, arrays, names, filename, coordinate_convention=vtk_convention
    )
    entropy_function = lambda nu: entropy_numpy(api, nu)

    for factor in factors:
        output_npz = output_dir / f"{name}_ds{factor}.npz"
        if output_npz.exists() and not args.overwrite:
            print(f"Skipping existing output: {output_npz}")
            continue

        if backend == "xmodmap_native":
            if factor != 1 or coordinate_mode != "legacy_centered":
                raise ValueError(
                    "xmodmap_native categorical construction is limited to factor 1 "
                    "and coordinate_mode=legacy_centered. The public xmodmap "
                    "downsampleParticles helper is fixed-factor and 2-D; use "
                    "project_affine_block for the colleague's 3-D factor-five idea or "
                    "for Allen affine-world data."
                )
            voxel_sizes = np.sqrt(np.sum(np.asarray(image.affine)[:3, :3] ** 2, axis=0))
            voxel_volume = float(abs(np.linalg.det(np.asarray(image.affine)[:3, :3])))
            X, nu_all = categorical_particles_xmodmap_native(
                api,
                input_file,
                voxel_sizes_mm=voxel_sizes,
                voxel_volume_mm3=voxel_volume,
                fine_label_ids=features.fine_ids,
                background_labels=job.get("background_labels", [0]),
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
        metadata = {
            "particle_role": "source_atlas",
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
