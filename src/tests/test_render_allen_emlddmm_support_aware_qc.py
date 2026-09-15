from __future__ import annotations

import numpy as np

from preprocess import render_allen_emlddmm_support_aware_qc as qc
from preprocess.run_allen_emlddmm import pinned_emlddmm


def _sparse_rgb() -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    axes = [
        np.arange(5, dtype=np.float32),
        np.arange(2, dtype=np.float32),
        np.arange(2, dtype=np.float32),
    ]
    image = np.zeros((3, 5, 2, 2), dtype=np.float32)
    first = np.asarray([0.2, 0.4, 0.6], dtype=np.float32)
    last = np.asarray([0.8, 0.6, 0.4], dtype=np.float32)
    image[:, 0] = first[:, None, None]
    image[:, 4] = last[:, None, None]
    support = np.zeros((5, 2, 2), dtype=np.float32)
    support[[0, 4]] = 1.0
    return axes, image, support


def _query(z: np.ndarray) -> np.ndarray:
    return np.stack(
        np.meshgrid(
            np.asarray(z, dtype=np.float32),
            np.asarray([0.0], dtype=np.float32),
            np.asarray([0.0], dtype=np.float32),
            indexing="ij",
        )
    )


def test_sparse_zero_planes_are_not_zero_valued_anatomy() -> None:
    em = pinned_emlddmm()
    axes, image, support = _sparse_rgb()
    z = np.asarray([-1.0, 0.0, 0.5, 1.0, 2.0, 3.0, 3.5, 4.0, 5.0])

    rendered, coverage, supported = qc.support_aware_interp(
        em, axes, image, support, _query(z), threshold=0.05
    )
    ordinary = np.asarray(em.interp(axes, image, _query(z)))

    first = image[:, 0, 0, 0]
    last = image[:, 4, 0, 0]
    np.testing.assert_allclose(rendered[:, 1, 0, 0], first, atol=2.0e-6)
    np.testing.assert_allclose(rendered[:, 2, 0, 0], first, atol=2.0e-6)
    np.testing.assert_allclose(rendered[:, 6, 0, 0], last, atol=2.0e-6)
    np.testing.assert_allclose(rendered[:, 7, 0, 0], last, atol=2.0e-6)
    np.testing.assert_allclose(ordinary[:, 2, 0, 0], first * 0.5, atol=2.0e-6)
    np.testing.assert_allclose(ordinary[:, 6, 0, 0], last * 0.5, atol=2.0e-6)

    unsupported = np.asarray([0, 3, 4, 5, 8])
    assert not np.any(supported[unsupported, 0, 0])
    np.testing.assert_array_equal(coverage[unsupported, 0, 0], 0.0)
    np.testing.assert_array_equal(rendered[:, unsupported, 0, 0], 0.0)


def test_rgb_channels_share_one_spatial_denominator() -> None:
    em = pinned_emlddmm()
    axes, image, support = _sparse_rgb()
    z = np.asarray([0.25, 0.5, 0.75, 3.25, 3.5, 3.75])

    rendered, coverage, supported = qc.support_aware_interp(
        em, axes, image, support, _query(z), threshold=0.05
    )

    assert np.all(supported)
    np.testing.assert_allclose(coverage[:3, 0, 0], [0.75, 0.5, 0.25])
    np.testing.assert_allclose(coverage[3:, 0, 0], [0.25, 0.5, 0.75])
    np.testing.assert_allclose(
        rendered[:, :3, 0, 0],
        np.repeat(image[:, 0, 0, 0, None], 3, axis=1),
        atol=2.0e-6,
    )
    np.testing.assert_allclose(
        rendered[:, 3:, 0, 0],
        np.repeat(image[:, 4, 0, 0, None], 3, axis=1),
        atol=2.0e-6,
    )
    ratios = rendered[:, 1, 0, 0] / rendered[0, 1, 0, 0]
    np.testing.assert_allclose(ratios, [1.0, 2.0, 3.0], atol=2.0e-6)


def test_small_or_zero_denominators_stay_unsupported_and_finite() -> None:
    em = pinned_emlddmm()
    axes, image, support = _sparse_rgb()
    z = np.asarray([0.99, 1.0, 2.0, 3.0, 3.01])

    rendered, coverage, supported = qc.support_aware_interp(
        em, axes, image, support, _query(z), threshold=0.05
    )
    raw_numerator, raw_coverage = qc.interpolate_numerator_and_support(
        em, axes, image * support[None], support, _query(z)
    )

    assert 0.0 < coverage[0, 0, 0] < 0.05
    np.testing.assert_allclose(raw_coverage, coverage)
    assert np.all(raw_numerator[:, [0, 4], 0, 0] > 0.0)
    assert 0.0 < coverage[-1, 0, 0] < 0.05
    np.testing.assert_array_equal(coverage[1:4, 0, 0], 0.0)
    assert not np.any(supported)
    np.testing.assert_array_equal(rendered, 0.0)
    assert np.all(np.isfinite(rendered))
    assert np.all(np.isfinite(coverage))


def test_support_normalize_zero_denominators_remain_finite() -> None:
    numerator = np.ones((3, 2, 2), dtype=np.float32)
    support = np.asarray([[1.0, 0.0], [0.5, 0.0]], dtype=np.float32)
    rendered, mask = qc.support_normalize(numerator, support, threshold=0.05)

    assert np.all(np.isfinite(rendered))
    np.testing.assert_array_equal(mask, support >= 0.05)
    np.testing.assert_array_equal(rendered[:, support == 0.0], 0.0)
