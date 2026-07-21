"""Utilities for representing a dense scalar image as feature-valued particles.

Why a separate dense-image representation?
==========================================
A segmentation voxel has a discrete anatomical label, so a one-hot label vector is a
natural particle feature.  A T1-weighted, T2-weighted, quantitative map, or scalar
histology volume instead has a continuous intensity.  Treating every raw intensity as
an anatomical label is usually wrong, particularly when MRI values are arbitrary and
scanner/protocol dependent.

This module creates two complementary representations for each particle:

``nu_intensity[:, 0]``
    Physical tissue mass (voxel volume summed within a block).

``nu_intensity[:, 1]``
    Tissue mass times transformed scalar intensity.  Their ratio recovers the block's
    mass-weighted mean transformed intensity.  This compact two-channel object is useful
    for transporting and reconstructing an image after particle deformation.  It should
    not automatically be used as the xIV-LDDMM data feature because summing its two
    channels makes total feature weight intensity-dependent.

``nu_bin``
    A non-negative distribution over intensity bins whose row sum equals physical
    tissue mass.  This is generally the safer feature matrix for same-contrast
    registration because intensity changes feature composition without changing total
    mass.  Hard binning reproduces the colleague's 256-bin idea; linear binning splits
    mass between adjacent bins and reduces quantization artifacts.

The default intensity transform is robust min-max scaling to [0, 1].  The transform and
its inverse parameters are saved in metadata.  This is a computational normalization,
not biological or scanner harmonization.  Cross-donor MRI and cross-stain histology may
still require bias correction, stain normalization, or other modality-specific steps.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from particle_utils import _coordinates_for_indices, voxel_volume_mm3


@dataclass(frozen=True)
class IntensityTransform:
    """Affine scalar transform ``y=(x-lower)/(upper-lower)`` plus clipping policy."""

    mode: str
    lower: float
    upper: float
    clip: bool

    def forward(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        scale = self.upper - self.lower
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError(
                f"Intensity transform needs upper > lower; got {self.lower}, {self.upper}."
            )
        transformed = (values - self.lower) / scale
        if self.clip:
            transformed = np.clip(transformed, 0.0, 1.0)
        return transformed

    def inverse(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        return values * (self.upper - self.lower) + self.lower

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "lower": self.lower,
            "upper": self.upper,
            "clip": self.clip,
            "formula": "transformed=(original-lower)/(upper-lower)",
        }


def _validate_scalar_image(data: np.ndarray) -> np.ndarray:
    """Return a squeezed finite-capable 3-D floating image."""

    array = np.squeeze(np.asarray(data))
    if array.ndim != 3:
        raise ValueError(
            f"Expected a scalar 3-D image after squeeze; got shape {array.shape}. "
            "Raw 4-D diffusion or multichannel data need an explicit feature model."
        )
    return array.astype(np.float64, copy=False)


def build_analysis_mask(
    data: np.ndarray,
    mask_config: Mapping[str, Any] | None = None,
    external_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Construct the voxel mask used to create particles.

    Supported modes are ``finite``, ``nonzero``, ``positive``, and ``threshold``.  An
    external mask is intersected with the selected mode.  NaN and infinite voxels are
    always excluded.
    """

    data = _validate_scalar_image(data)
    config = dict(mask_config or {})
    mode = str(config.get("mode", "positive"))
    finite = np.isfinite(data)

    if mode == "finite":
        selected = finite
    elif mode == "nonzero":
        selected = finite & (data != 0)
    elif mode == "positive":
        selected = finite & (data > 0)
    elif mode == "threshold":
        selected = finite.copy()
        lower = config.get("lower")
        upper = config.get("upper")
        if lower is not None:
            selected &= data >= float(lower)
        if upper is not None:
            selected &= data <= float(upper)
    else:
        raise ValueError(
            f"Unknown mask mode '{mode}'. Use finite, nonzero, positive, or threshold."
        )

    if external_mask is not None:
        mask = np.squeeze(np.asarray(external_mask))
        if mask.shape != data.shape:
            raise ValueError(
                f"External mask shape {mask.shape} does not match image shape {data.shape}."
            )
        selected &= np.isfinite(mask) & (mask > 0)

    if not np.any(selected):
        raise ValueError("The configured mask selected no voxels.")
    return selected


def fit_intensity_transform(
    data: np.ndarray,
    mask: np.ndarray,
    config: Mapping[str, Any] | None = None,
) -> IntensityTransform:
    """Fit a non-negative [0, 1]-oriented intensity transform on selected voxels."""

    settings = dict(config or {})
    mode = str(settings.get("normalization", "robust_minmax"))
    clip = bool(settings.get("clip", True))
    values = np.asarray(data, dtype=np.float64)[mask]

    if mode == "robust_minmax":
        lower_q = float(settings.get("lower_quantile", 0.01))
        upper_q = float(settings.get("upper_quantile", 0.99))
        if not 0 <= lower_q < upper_q <= 1:
            raise ValueError("Require 0 <= lower_quantile < upper_quantile <= 1.")
        lower, upper = np.quantile(values, [lower_q, upper_q]).astype(float)
    elif mode == "minmax":
        lower, upper = float(values.min()), float(values.max())
    elif mode == "fixed_range":
        value_range = settings.get("value_range")
        if not isinstance(value_range, Sequence) or len(value_range) != 2:
            raise ValueError("fixed_range needs intensity.value_range: [lower, upper].")
        lower, upper = float(value_range[0]), float(value_range[1])
    elif mode == "uint8_legacy":
        # This reproduces the original interpretation of integer values 0..255 while
        # still saving normalized [0,1] moments for generic reconstruction.
        lower, upper = 0.0, 255.0
        clip = True
    else:
        raise ValueError(
            f"Unknown intensity normalization '{mode}'. Use robust_minmax, minmax, "
            "fixed_range, or uint8_legacy."
        )

    if not np.isfinite([lower, upper]).all() or upper <= lower:
        raise ValueError(
            f"Cannot normalize a constant or invalid image range: lower={lower}, upper={upper}."
        )
    return IntensityTransform(mode=mode, lower=lower, upper=upper, clip=clip)


def _aggregate_by_block(
    ijk: np.ndarray,
    values: np.ndarray,
    image_shape: Sequence[int],
    affine: np.ndarray,
    factor: int,
    coordinate_mode: str,
    position_mode: str,
    voxel_volume: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate selected voxels into spatial blocks without materializing dense 4-D arrays.

    Returns particle coordinates, inverse voxel-to-particle indices, particle mass, and
    particle mass-weighted intensity.
    """

    factor = int(factor)
    if factor < 1:
        raise ValueError("downsample_factor must be at least 1.")
    shape = np.asarray(image_shape, dtype=np.int64)
    block_shape = np.ceil(shape / factor).astype(np.int64)
    block_ijk = (ijk // factor).astype(np.int64)
    linear = np.ravel_multi_index(block_ijk.T, tuple(block_shape))
    unique_linear, inverse = np.unique(linear, return_inverse=True)
    unique_block_ijk = np.column_stack(
        np.unravel_index(unique_linear, tuple(block_shape))
    ).astype(np.float64)
    n_particles = unique_linear.size

    counts = np.bincount(inverse, minlength=n_particles).astype(np.float64)
    mass = counts * voxel_volume
    mass_intensity = np.bincount(
        inverse, weights=values * voxel_volume, minlength=n_particles
    ).astype(np.float64)

    if position_mode == "block_center":
        starts = unique_block_ijk * factor
        ends = np.minimum(starts + factor, shape)
        particle_ijk = (starts + ends - 1.0) / 2.0
    elif position_mode == "sample_centroid":
        particle_ijk = np.zeros((n_particles, 3), dtype=np.float64)
        for axis in range(3):
            particle_ijk[:, axis] = np.bincount(
                inverse, weights=ijk[:, axis], minlength=n_particles
            ) / counts
    else:
        raise ValueError(
            f"Unknown position_mode '{position_mode}'. Use block_center or sample_centroid."
        )

    X = _coordinates_for_indices(
        particle_ijk, shape=shape, affine=np.asarray(affine), coordinate_mode=coordinate_mode
    )
    return X, inverse, mass, mass_intensity


def _aggregate_binned_features(
    transformed_values: np.ndarray,
    inverse: np.ndarray,
    n_particles: int,
    voxel_volume: float,
    number_bins: int,
    assignment: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate voxel mass into hard or linearly interpolated intensity bins."""

    bins = int(number_bins)
    if bins < 2:
        raise ValueError("Histogram representation requires at least two bins.")
    values = np.clip(np.asarray(transformed_values, dtype=np.float64), 0.0, 1.0)
    nu_bin = np.zeros((n_particles, bins), dtype=np.float32)
    scaled = values * (bins - 1)

    if assignment == "hard":
        columns = np.rint(scaled).astype(np.int64)
        np.add.at(nu_bin, (inverse, columns), np.float32(voxel_volume))
    elif assignment == "linear":
        lower = np.floor(scaled).astype(np.int64)
        upper = np.minimum(lower + 1, bins - 1)
        upper_weight = scaled - lower
        lower_weight = 1.0 - upper_weight
        np.add.at(
            nu_bin,
            (inverse, lower),
            (lower_weight * voxel_volume).astype(np.float32),
        )
        np.add.at(
            nu_bin,
            (inverse, upper),
            (upper_weight * voxel_volume).astype(np.float32),
        )
    else:
        raise ValueError("Histogram assignment must be 'hard' or 'linear'.")

    edges = np.linspace(0.0, 1.0, bins + 1, dtype=np.float32)
    centers = np.linspace(0.0, 1.0, bins, dtype=np.float32)
    return nu_bin, edges, centers


def construct_dense_image_particles(
    image_data: np.ndarray,
    affine: np.ndarray,
    mask: np.ndarray,
    transform: IntensityTransform,
    downsample_factor: int = 1,
    coordinate_mode: str = "affine_world",
    position_mode: str = "block_center",
    histogram_config: Mapping[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, dict[str, Any], dict[str, np.ndarray]]:
    """Convert a scalar 3-D image to compact moments and optional binned features."""

    data = _validate_scalar_image(image_data)
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != data.shape:
        raise ValueError(f"Mask shape {mask.shape} does not match image shape {data.shape}.")

    ijk = np.argwhere(mask).astype(np.int64)
    original_values = data[mask]
    transformed = transform.forward(original_values)
    if np.any(~np.isfinite(transformed)):
        raise ValueError("Transformed intensity contains NaN or infinite values.")
    if np.any(transformed < 0):
        raise ValueError(
            "Transformed intensities must be non-negative for particle feature masses."
        )

    volume = voxel_volume_mm3(affine)
    X, inverse, mass, mass_intensity = _aggregate_by_block(
        ijk=ijk,
        values=transformed,
        image_shape=data.shape,
        affine=affine,
        factor=downsample_factor,
        coordinate_mode=coordinate_mode,
        position_mode=position_mode,
        voxel_volume=volume,
    )
    nu_intensity = np.column_stack([mass, mass_intensity]).astype(np.float32)

    histogram = dict(histogram_config or {})
    histogram_enabled = bool(histogram.get("enabled", True))
    nu_bin: np.ndarray | None = None
    bin_arrays: dict[str, np.ndarray] = {}
    if histogram_enabled:
        nu_bin, edges, centers = _aggregate_binned_features(
            transformed_values=transformed,
            inverse=inverse,
            n_particles=X.shape[0],
            voxel_volume=volume,
            number_bins=int(histogram.get("bins", 64)),
            assignment=str(histogram.get("assignment", "linear")),
        )
        bin_arrays = {"bin_edges": edges, "bin_centers": centers}

        # Binned feature mass should equal selected tissue volume, independent of
        # intensity and binning policy.
        expected = float(mask.sum()) * volume
        observed = float(np.sum(nu_bin, dtype=np.float64))
        rel_error = abs(observed - expected) / max(expected, 1e-12)
        if rel_error > 2e-6:
            raise RuntimeError(
                f"Binned feature mass was not conserved: expected {expected}, got {observed}."
            )
    else:
        rel_error = None

    recovered_mean = np.divide(
        nu_intensity[:, 1],
        nu_intensity[:, 0],
        out=np.zeros(X.shape[0], dtype=np.float32),
        where=nu_intensity[:, 0] > 0,
    )
    diagnostics = {
        "image_shape": [int(v) for v in data.shape],
        "downsample_factor": int(downsample_factor),
        "position_mode": position_mode,
        "coordinate_mode": coordinate_mode,
        "selected_voxels": int(mask.sum()),
        "number_particles": int(X.shape[0]),
        "voxel_volume_mm3": volume,
        "selected_volume_mm3": float(mask.sum()) * volume,
        "transformed_voxel_min": float(transformed.min()),
        "transformed_voxel_max": float(transformed.max()),
        "transformed_particle_mean_min": float(recovered_mean.min()),
        "transformed_particle_mean_max": float(recovered_mean.max()),
        "histogram_enabled": histogram_enabled,
        "histogram_mass_relative_error": rel_error,
        "nu_intensity_uncompressed_mb": float(nu_intensity.nbytes / 1024**2),
        "nu_bin_uncompressed_mb": (
            float(nu_bin.nbytes / 1024**2) if nu_bin is not None else 0.0
        ),
        "coordinate_min_mm": X.min(axis=0).tolist(),
        "coordinate_max_mm": X.max(axis=0).tolist(),
    }
    return X.astype(np.float32), nu_intensity, nu_bin, diagnostics, bin_arrays


def save_dense_particle_archive(
    output_npz: str | Path,
    X: np.ndarray,
    nu_intensity: np.ndarray,
    nu_bin: np.ndarray | None,
    bin_arrays: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
    write_vtk: bool = True,
    write_legacy_t2_alias: bool = False,
    *,
    vtk_writer: Any | None = None,
    entropy_function: Any | None = None,
) -> None:
    """Save dense-image particles with explicit keys and optional legacy alias."""

    output = Path(output_npz)
    output.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, Any] = {
        "X": np.asarray(X, dtype=np.float32),
        "nu_intensity": np.asarray(nu_intensity, dtype=np.float32),
        "intensity_feature_names": np.asarray(
            ["mass_mm3", "mass_times_transformed_intensity"], dtype=str
        ),
        "metadata_json": np.asarray(json.dumps(dict(metadata), sort_keys=True)),
    }
    if nu_bin is not None:
        arrays["nu_bin"] = np.asarray(nu_bin, dtype=np.float32)
        arrays.update({key: np.asarray(value) for key, value in bin_arrays.items()})
    if write_legacy_t2_alias:
        arrays["nu_T2"] = np.asarray(nu_intensity, dtype=np.float32)
    np.savez_compressed(output, **arrays)

    if write_vtk:
        if vtk_writer is None or entropy_function is None:
            raise ValueError(
                "write_vtk=True requires xmodmap-backed vtk_writer and entropy_function."
            )
        mass = nu_intensity[:, 0]
        mean_transformed = np.divide(
            nu_intensity[:, 1],
            mass,
            out=np.zeros_like(mass),
            where=mass > 0,
        )
        transform_info = dict(metadata.get("intensity_transform", {}))
        lower = float(transform_info.get("lower", 0.0))
        upper = float(transform_info.get("upper", 1.0))
        mean_original = mean_transformed * (upper - lower) + lower
        values = [mass, mean_transformed, mean_original]
        names = [
            "Mass_mm3",
            "MeanTransformedIntensity",
            "ApproxMeanOriginalIntensity",
        ]
        if nu_bin is not None:
            values.extend([np.argmax(nu_bin, axis=1), entropy_function(nu_bin)])
            names.extend(["DominantBinIndex", "BinEntropy"])
        vtk_writer(X, values, names, output.with_suffix(".vtk"))
