#!/usr/bin/env python3
"""Reconstruct scalar or categorical NIfTI volumes from particle features.

This replaces the workstation-specific reconstruction portions of
``makeParticlesToT2seg_yxie.py``.  It accepts either:

* NPZ particles with explicit ``X`` and feature keys; or
* PyTorch deformation summaries with configured coordinate/feature keys.

A reference NIfTI grid is strongly preferred.  Its shape and affine define exactly where
particle values are sampled.  The fallback generated grid is axis-aligned in world
coordinates and should be used mainly for visualization or synthetic tests.

Scalar reconstruction expects two feature channels:

    channel 0 = particle mass
    channel 1 = particle mass * transformed intensity

Categorical reconstruction expects one non-negative mass column per label.  Nearest
or Gaussian k-nearest interpolation is implemented with SciPy's cKDTree, avoiding the
unshared ``EmpiricalDistributions_yxie`` dependency in the colleague script.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from project_config import load_config, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Project YAML configuration.")
    parser.add_argument(
        "--job", required=True, help="Name under particle_reconstruction_jobs."
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace outputs.")
    return parser.parse_args()


def _get_job(config: dict, name: str) -> dict:
    jobs = config.get("particle_reconstruction_jobs", {})
    if name not in jobs:
        available = ", ".join(sorted(jobs)) or "<none>"
        raise KeyError(f"Unknown reconstruction job '{name}'. Available: {available}")
    return dict(jobs[name])


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def load_points_and_features(
    filename: Path,
    coordinate_key: str | None,
    feature_key: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Load particles by explicit keys, with conservative coordinate-key autodetection."""

    metadata: dict[str, Any] = {}
    if filename.suffix == ".pt":
        import torch

        info = torch.load(filename, map_location="cpu")
        if not isinstance(info, dict):
            raise ValueError(f"PyTorch file must contain a dictionary: {filename}")
        if coordinate_key is None:
            candidates = [key for key in ("D", "Td", "X", "qx") if key in info]
            if len(candidates) != 1:
                raise KeyError(
                    "Set coordinate_key explicitly. Autodetection found: "
                    f"{candidates}; available keys: {list(info)}"
                )
            coordinate_key = candidates[0]
        if coordinate_key not in info or feature_key not in info:
            raise KeyError(
                f"Need '{coordinate_key}' and '{feature_key}' in {filename}; "
                f"available keys: {list(info)}"
            )
        X = _to_numpy(info[coordinate_key])
        nu = _to_numpy(info[feature_key])
    elif filename.suffix == ".npz":
        with np.load(filename, allow_pickle=False) as archive:
            coordinate_key = coordinate_key or "X"
            if coordinate_key not in archive or feature_key not in archive:
                raise KeyError(
                    f"Need '{coordinate_key}' and '{feature_key}' in {filename}; "
                    f"available keys: {archive.files}"
                )
            X = np.asarray(archive[coordinate_key])
            nu = np.asarray(archive[feature_key])
            if "metadata_json" in archive:
                metadata = json.loads(str(np.asarray(archive["metadata_json"]).item()))
    else:
        raise ValueError("Particle input must end in .npz or .pt.")

    X = np.asarray(X, dtype=np.float64)
    nu = np.asarray(nu, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] != 3:
        raise ValueError(f"Particle coordinates must have shape [N,3], got {X.shape}.")
    if nu.ndim != 2 or nu.shape[0] != X.shape[0]:
        raise ValueError(
            f"Features must have shape [N,F] with N={X.shape[0]}, got {nu.shape}."
        )
    if not np.isfinite(X).all() or not np.isfinite(nu).all():
        raise ValueError("Particles contain NaN or infinite values.")
    if np.any(nu < 0):
        raise ValueError("Particle feature masses must be non-negative.")
    return X, nu, metadata


def _generated_grid(points: np.ndarray, grid_config: dict) -> tuple[tuple[int, int, int], np.ndarray]:
    resolution = float(grid_config.get("resolution_mm", 0.2))
    padding = float(grid_config.get("padding_mm", 2.0 * resolution))
    if resolution <= 0 or padding < 0:
        raise ValueError("Generated-grid resolution must be >0 and padding >=0.")
    lower = points.min(axis=0) - padding
    upper = points.max(axis=0) + padding
    shape = tuple((np.ceil((upper - lower) / resolution).astype(int) + 1).tolist())
    affine = np.eye(4, dtype=np.float64)
    affine[0, 0] = resolution
    affine[1, 1] = resolution
    affine[2, 2] = resolution
    affine[:3, 3] = lower
    return shape, affine


def _grid_from_job(config: dict, job: dict, points: np.ndarray):
    import nibabel as nib

    reference_text = job.get("reference_nifti")
    if reference_text:
        reference_path = resolve_path(config, reference_text)
        if not reference_path.is_file():
            raise FileNotFoundError(f"Reference NIfTI is missing: {reference_path}")
        reference = nib.load(str(reference_path))
        if len(reference.shape) < 3:
            raise ValueError("Reference NIfTI must have at least three dimensions.")
        return tuple(int(v) for v in reference.shape[:3]), np.asarray(reference.affine), reference
    shape, affine = _generated_grid(points, dict(job.get("generated_grid", {})))
    return shape, affine, None


def _indices_to_world(indices: np.ndarray, affine: np.ndarray) -> np.ndarray:
    return indices @ affine[:3, :3].T + affine[:3, 3]


def _query_neighbors(
    tree: cKDTree,
    query_points: np.ndarray,
    method: str,
    k: int,
    support_radius: float,
):
    if method == "nearest":
        distances, indices = tree.query(
            query_points, k=1, distance_upper_bound=support_radius
        )
        return distances[:, None], indices[:, None]
    if method == "gaussian_knn":
        return tree.query(query_points, k=k, distance_upper_bound=support_radius)
    raise ValueError("Interpolation method must be nearest or gaussian_knn.")


def reconstruct(
    points: np.ndarray,
    features: np.ndarray,
    shape: tuple[int, int, int],
    affine: np.ndarray,
    mode: str,
    interpolation: dict,
    label_values: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate particles onto a NIfTI grid in bounded-memory chunks."""

    method = str(interpolation.get("method", "gaussian_knn"))
    k = int(interpolation.get("k", 8))
    sigma = float(interpolation.get("sigma_mm", 0.25))
    support_radius = float(interpolation.get("support_radius_mm", 4.0 * sigma))
    chunk_size = int(interpolation.get("chunk_size", 200_000))
    minimum_weight = float(interpolation.get("minimum_weight", 1e-8))
    if k < 1 or sigma <= 0 or support_radius <= 0 or chunk_size < 1:
        raise ValueError("Invalid interpolation k/sigma/support_radius/chunk_size.")

    if mode == "scalar" and features.shape[1] < 2:
        raise ValueError("Scalar reconstruction needs [mass, mass*intensity] channels.")
    if mode not in {"scalar", "categorical"}:
        raise ValueError("Reconstruction mode must be scalar or categorical.")

    n_voxels = int(np.prod(shape))
    output = np.zeros(n_voxels, dtype=np.float32)
    assigned_mass = np.zeros(n_voxels, dtype=np.float32)
    support = np.zeros(n_voxels, dtype=np.uint8)
    tree = cKDTree(points)
    particle_count = points.shape[0]

    if mode == "scalar":
        mass = features[:, 0]
        numerator_feature = features[:, 1]
    else:
        if label_values is None:
            label_values = np.arange(1, features.shape[1] + 1, dtype=np.int32)
        if label_values.shape != (features.shape[1],):
            raise ValueError("label_values length must equal the number of feature columns.")

    yz = shape[1] * shape[2]
    for start in range(0, n_voxels, chunk_size):
        stop = min(start + chunk_size, n_voxels)
        linear = np.arange(start, stop, dtype=np.int64)
        i = linear // yz
        remainder = linear % yz
        j = remainder // shape[2]
        k_index = remainder % shape[2]
        ijk = np.column_stack([i, j, k_index]).astype(np.float64)
        world = _indices_to_world(ijk, affine)

        distances, neighbors = _query_neighbors(
            tree, world, method, k, support_radius
        )
        distances = np.asarray(distances)
        neighbors = np.asarray(neighbors)
        if distances.ndim == 1:
            distances = distances[:, None]
            neighbors = neighbors[:, None]
        valid = np.isfinite(distances) & (neighbors < particle_count)
        safe_neighbors = np.where(valid, neighbors, 0)

        if method == "nearest":
            kernel = valid.astype(np.float64)
        else:
            kernel = np.exp(-0.5 * (distances / sigma) ** 2)
            kernel[~valid] = 0.0

        if mode == "scalar":
            denominator = np.sum(kernel * mass[safe_neighbors], axis=1)
            numerator = np.sum(kernel * numerator_feature[safe_neighbors], axis=1)
            values = np.divide(
                numerator,
                denominator,
                out=np.zeros_like(numerator),
                where=denominator > minimum_weight,
            )
            output[start:stop] = values.astype(np.float32)
            assigned_mass[start:stop] = denominator.astype(np.float32)
            support[start:stop] = (denominator > minimum_weight).astype(np.uint8)
        else:
            # [chunk, neighbors, features] is acceptable for small anatomical feature
            # sets.  Large molecular panels should be reconstructed selectively.
            weighted = np.sum(
                kernel[:, :, None] * features[safe_neighbors, :], axis=1
            )
            totals = weighted.sum(axis=1)
            selected = totals > minimum_weight
            labels = np.zeros(stop - start, dtype=np.int32)
            labels[selected] = label_values[np.argmax(weighted[selected], axis=1)]
            output[start:stop] = labels.astype(np.float32)
            assigned_mass[start:stop] = totals.astype(np.float32)
            support[start:stop] = selected.astype(np.uint8)

    return output.reshape(shape), assigned_mass.reshape(shape), support.reshape(shape)


def _inverse_transform(
    values: np.ndarray,
    metadata: dict,
    output_units: str,
    transform_override: dict | None = None,
) -> np.ndarray:
    if output_units == "transformed":
        return values
    if output_units != "original":
        raise ValueError("output_units must be transformed or original.")
    transform = dict(transform_override or metadata.get("intensity_transform", {}))
    if "lower" not in transform or "upper" not in transform:
        raise ValueError(
            "Cannot restore original units: particle metadata lacks intensity_transform."
        )
    lower, upper = float(transform["lower"]), float(transform["upper"])
    return values * (upper - lower) + lower


def main() -> None:
    import nibabel as nib

    args = parse_args()
    config = load_config(args.config)
    job = _get_job(config, args.job)
    input_file = resolve_path(config, job["input_particles"])
    if not input_file.is_file():
        raise FileNotFoundError(f"Particle file is missing: {input_file}")
    output_file = resolve_path(config, job["output_nifti"])
    if output_file.exists() and not args.overwrite:
        print(f"Skipping existing output: {output_file}")
        return

    mode = str(job.get("mode", "scalar"))
    feature_key = str(job.get("feature_key", "nu_intensity"))
    X, nu, metadata = load_points_and_features(
        input_file,
        coordinate_key=job.get("coordinate_key"),
        feature_key=feature_key,
    )
    shape, affine, reference = _grid_from_job(config, job, X)
    label_values = None
    if mode == "categorical" and job.get("label_values") is not None:
        label_values = np.asarray(job["label_values"], dtype=np.int32)

    dense, mass, support = reconstruct(
        points=X,
        features=nu,
        shape=shape,
        affine=affine,
        mode=mode,
        interpolation=dict(job.get("interpolation", {})),
        label_values=label_values,
    )
    if mode == "scalar":
        dense = _inverse_transform(
            dense,
            metadata,
            str(job.get("output_units", "original")),
            transform_override=job.get("intensity_transform"),
        )
        output_dtype = np.dtype(job.get("output_dtype", "float32"))
    else:
        output_dtype = np.dtype(job.get("output_dtype", "uint16"))

    output_file.parent.mkdir(parents=True, exist_ok=True)
    header = reference.header.copy() if reference is not None else None
    image = nib.Nifti1Image(dense.astype(output_dtype), affine, header=header)
    image.set_data_dtype(output_dtype)
    nib.save(image, str(output_file))

    def related_output(tag: str, extension: str = ".nii.gz") -> Path:
        name = output_file.name
        if name.endswith(".nii.gz"):
            stem = name[:-7]
        elif name.endswith(".nii"):
            stem = name[:-4]
        else:
            stem = output_file.stem
        return output_file.with_name(f"{stem}_{tag}{extension}")

    if bool(job.get("write_mass_image", True)):
        mass_path = related_output("mass")
        nib.save(nib.Nifti1Image(mass.astype(np.float32), affine), str(mass_path))
    if bool(job.get("write_mask_image", True)):
        mask_path = related_output("mask")
        nib.save(nib.Nifti1Image(support.astype(np.uint8), affine), str(mask_path))

    sidecar = related_output("reconstruction", extension=".json")
    with sidecar.open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "job_name": args.job,
                "input_particles": str(input_file),
                "output_nifti": str(output_file),
                "mode": mode,
                "feature_key": feature_key,
                "shape": list(shape),
                "affine": affine.tolist(),
                "reference_nifti": job.get("reference_nifti"),
                "interpolation": dict(job.get("interpolation", {})),
                "particle_metadata": metadata,
            },
            stream,
            indent=2,
            sort_keys=True,
        )
    print(json.dumps({"created": str(output_file), "shape": shape, "mode": mode}, indent=2))


if __name__ == "__main__":
    main()
