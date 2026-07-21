"""Small synthetic checks for dense-image particle creation and reconstruction."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dense_image_utils import (  # noqa: E402
    build_analysis_mask,
    construct_dense_image_particles,
    fit_intensity_transform,
)
from reconstructParticlesToNifti import reconstruct  # noqa: E402


def run() -> None:
    shape = (9, 8, 7)
    affine = np.array(
        [[-0.2, 0.0, 0.0, 12.0], [0.0, 0.25, 0.0, -3.0], [0.0, 0.0, 0.3, 5.0], [0, 0, 0, 1]],
        dtype=float,
    )
    i, j, k = np.meshgrid(
        np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing="ij"
    )
    data = (10.0 + i + 2.0 * j + 0.5 * k).astype(np.float32)
    data[:2] = 0.0
    mask = build_analysis_mask(data, {"mode": "positive"})
    transform = fit_intensity_transform(
        data, mask, {"normalization": "minmax", "clip": True}
    )

    for factor in (1, 2, 3):
        X, moments, bins, diagnostics, _ = construct_dense_image_particles(
            image_data=data,
            affine=affine,
            mask=mask,
            transform=transform,
            downsample_factor=factor,
            coordinate_mode="affine_world",
            position_mode="block_center",
            histogram_config={"enabled": True, "bins": 16, "assignment": "linear"},
        )
        assert X.shape[0] == moments.shape[0] == bins.shape[0]
        assert np.allclose(bins.sum(), diagnostics["selected_volume_mm3"], rtol=2e-6)
        assert np.allclose(moments[:, 0].sum(), diagnostics["selected_volume_mm3"])

    # Native-resolution nearest reconstruction should exactly recover transformed
    # intensity on selected voxels because particle coordinates coincide with voxel centers.
    X, moments, _, _, _ = construct_dense_image_particles(
        image_data=data,
        affine=affine,
        mask=mask,
        transform=transform,
        downsample_factor=1,
        coordinate_mode="affine_world",
        histogram_config={"enabled": False},
    )
    dense, _, support = reconstruct(
        points=X,
        features=moments,
        shape=shape,
        affine=affine,
        mode="scalar",
        interpolation={
            "method": "nearest",
            "k": 1,
            "sigma_mm": 0.1,
            "support_radius_mm": 0.05,
            "chunk_size": 100,
            "minimum_weight": 1e-12,
        },
    )
    expected = np.zeros(shape, dtype=float)
    expected[mask] = transform.forward(data[mask])
    assert np.array_equal(support.astype(bool), mask)
    assert np.allclose(dense, expected, atol=1e-6)
    print("dense-image particle tests passed")


if __name__ == "__main__":
    run()
