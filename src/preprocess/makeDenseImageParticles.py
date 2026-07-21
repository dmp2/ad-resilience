#!/usr/bin/env python3
"""Create particle representations from a dense scalar NIfTI image.

This is the configurable, modality-generic successor to ``makeT2Particles_255bin.py``.
It supports T1-weighted MRI, T2-weighted MRI, scalar quantitative maps, reconstructed
histology intensities, and other 3-D scalar images, provided that their preprocessing
and intensity interpretation are scientifically appropriate.

It does not automatically make unrelated contrasts comparable.  T1 and T2 intensities,
for example, should not be registered as though equal bins had equal biological meaning.
Raw 4-D diffusion MRI also is not a scalar image; use a derived scalar map (b0, FA, MD,
etc.) or build an explicit vector/tensor feature representation.

Two outputs are saved:

* ``nu_bin``: volume-preserving intensity-bin features for same-contrast registration;
* ``nu_intensity``: [mass, mass*transformed_intensity] for later image reconstruction.

Usage::

    python src/makeDenseImageParticles.py \
        --config config/project_config.yaml \
        --job ding_7t_t1_example
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np

from dense_image_utils import (
    build_analysis_mask,
    construct_dense_image_particles,
    fit_intensity_transform,
    save_dense_particle_archive,
)
from particle_utils import summarize_nifti_geometry
from project_config import load_config, resolve_path
from xmodmap_compat import (
    dense_bins_xmodmap_native,
    entropy_numpy,
    load_xmodmap,
    write_vtk_xyz,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Project YAML configuration.")
    parser.add_argument(
        "--job", required=True, help="Name under dense_image_particle_jobs."
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace existing NPZ/VTK outputs."
    )
    return parser.parse_args()


def _get_job(config: dict, name: str) -> dict:
    jobs = config.get("dense_image_particle_jobs", {})
    if name not in jobs:
        available = ", ".join(sorted(jobs)) or "<none>"
        raise KeyError(f"Unknown dense-image job '{name}'. Available: {available}")
    return dict(jobs[name])


def _load_optional_mask(config: dict, job: dict, image: nib.spatialimages.SpatialImage):
    mask_path_text = job.get("mask_nifti")
    if not mask_path_text:
        return None, None
    mask_path = resolve_path(config, mask_path_text)
    if not mask_path.is_file():
        raise FileNotFoundError(f"Configured mask NIfTI is missing: {mask_path}")
    mask_image = nib.load(str(mask_path))
    if tuple(mask_image.shape[:3]) != tuple(image.shape[:3]):
        raise ValueError(
            f"Mask shape {mask_image.shape[:3]} differs from image shape {image.shape[:3]}."
        )
    if not np.allclose(mask_image.affine, image.affine, atol=1e-5, rtol=1e-5):
        raise ValueError(
            "Mask and image affines differ. Resample the mask explicitly; the particle "
            "builder will not silently align them."
        )
    return np.asanyarray(mask_image.dataobj), mask_path


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    job = _get_job(config, args.job)

    input_file = resolve_path(config, job["input_nifti"])
    if not input_file.is_file():
        raise FileNotFoundError(f"Dense input image is missing: {input_file}")
    image = nib.load(str(input_file))
    if len(image.shape) > 3 and any(size != 1 for size in image.shape[3:]):
        raise ValueError(
            f"Input has non-singleton dimensions beyond 3-D: {image.shape}. "
            "Select or derive one scalar volume first."
        )

    # np.asanyarray(dataobj) applies NIfTI slope/intercept.  That is normally what we
    # want.  ``load_scaled: false`` is retained only for exact reproduction of files
    # whose on-disk integer values are intentionally the scientific intensities.
    if bool(job.get("load_scaled", True)):
        data = np.asanyarray(image.dataobj)
    else:
        data = image.dataobj.get_unscaled()

    external_mask, mask_path = _load_optional_mask(config, job, image)
    mask = build_analysis_mask(data, job.get("mask"), external_mask)
    transform = fit_intensity_transform(data, mask, job.get("intensity"))

    output_dir = resolve_path(config, job["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    factors = [int(value) for value in job.get("downsample_factors", [1])]
    if len(factors) != len(set(factors)):
        raise ValueError("downsample_factors contains duplicate values.")

    name = input_file.name[:-7] if input_file.name.endswith(".nii.gz") else input_file.stem
    coordinate_mode = str(job.get("coordinate_mode", "affine_world"))
    coordinate_space = str(job.get("coordinate_space", "unspecified"))
    position_mode = str(job.get("position_mode", "block_center"))
    geometry = summarize_nifti_geometry(image, input_file)
    backend = str(job.get("construction_backend", "project_affine_block"))
    api = load_xmodmap(config)
    vtk_convention = str(job.get("vtk_coordinate_convention", "xyz"))
    vtk_writer = lambda points, arrays, names, filename: write_vtk_xyz(
        api, points, arrays, names, filename, coordinate_convention=vtk_convention
    )
    entropy_function = lambda nu: entropy_numpy(api, nu)

    for factor in factors:
        output_npz = output_dir / f"{name}_intensity_ds{factor}.npz"
        if output_npz.exists() and not args.overwrite:
            print(f"Skipping existing output: {output_npz}")
            continue

        if backend == "xmodmap_native":
            if factor != 1 or coordinate_mode != "legacy_centered":
                raise ValueError(
                    "xmodmap_native dense binning is limited to factor 1 and "
                    "coordinate_mode=legacy_centered. Use project_affine_block for "
                    "Allen affine-world coordinates or coarse mass-conserving particles."
                )
            histogram = dict(job.get("histogram", {}))
            bins = int(histogram.get("bins", 256))
            voxel_sizes = np.sqrt(np.sum(np.asarray(image.affine)[:3, :3] ** 2, axis=0))
            X, nu_bin = dense_bins_xmodmap_native(
                api,
                input_file,
                voxel_sizes_mm=voxel_sizes,
                bins=bins,
                mask_flat=mask.reshape(-1),
                reverse=bool(histogram.get("reverse", False)),
            )
            transformed_selected = transform.forward(np.squeeze(np.asarray(data)))[mask]
            native_mass = float(np.prod(voxel_sizes))
            nu_intensity = np.column_stack(
                [
                    np.full(transformed_selected.shape[0], native_mass),
                    native_mass * transformed_selected,
                ]
            ).astype(np.float32)
            bin_arrays = {
                "bin_feature_indices": np.arange(nu_bin.shape[1], dtype=np.int32),
            }
            diagnostics = {
                "construction_backend": "xmodmap_native",
                "downsample_factor": 1,
                "number_particles": int(X.shape[0]),
                "voxel_volume_mm3": native_mass,
                "coordinate_mode": coordinate_mode,
                "histogram_bins": int(nu_bin.shape[1]),
                "note": "nu_bin created by xmodmap.io.getInput.makeBinsFromMultiChannelImage.",
            }
        elif backend == "project_affine_block":
            X, nu_intensity, nu_bin, diagnostics, bin_arrays = construct_dense_image_particles(
                image_data=data,
                affine=image.affine,
                mask=mask,
                transform=transform,
                downsample_factor=factor,
                coordinate_mode=coordinate_mode,
                position_mode=position_mode,
                histogram_config=job.get("histogram"),
            )
            diagnostics["construction_backend"] = backend
        else:
            raise ValueError(
                "construction_backend must be xmodmap_native or project_affine_block."
            )
        metadata = {
            "particle_role": str(job.get("particle_role", "dense_scalar_image")),
            "job_name": args.job,
            "source_nifti": str(input_file),
            "mask_nifti": str(mask_path) if mask_path else None,
            "coordinate_space": coordinate_space,
            "geometry": geometry,
            "diagnostics": diagnostics,
            "mask_config": dict(job.get("mask", {})),
            "intensity_transform": transform.as_dict(),
            "histogram_config": dict(job.get("histogram", {})),
            "construction_backend": backend,
            "recommended_registration_feature_key": (
                "nu_bin" if nu_bin is not None else None
            ),
            "reconstruction_feature_key": "nu_intensity",
            "notes": job.get("notes", ""),
        }
        save_dense_particle_archive(
            output_npz=output_npz,
            X=X,
            nu_intensity=nu_intensity,
            nu_bin=nu_bin,
            bin_arrays=bin_arrays,
            metadata=metadata,
            write_vtk=bool(job.get("write_vtk", True)),
            write_legacy_t2_alias=bool(job.get("write_legacy_t2_alias", False)),
            vtk_writer=vtk_writer,
            entropy_function=entropy_function,
        )
        print(json.dumps({"created": str(output_npz), **diagnostics}, indent=2))


if __name__ == "__main__":
    main()
