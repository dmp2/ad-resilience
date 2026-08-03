from __future__ import annotations

import numpy as np

from preprocess.emlddmm_support import (
    UNSUPPORTED_SECTION_INDEX,
    assign_observed_section_support,
    resolve_label_collisions,
)


def test_sparse_label_support_does_not_fill_neighboring_z_positions() -> None:
    query_z = np.arange(-100.0, 301.0, 25.0)
    section_z = np.array([0.0, 200.0])
    group_present = np.array([True, False])

    assignment = assign_observed_section_support(
        query_z,
        section_z,
        section_present=group_present,
        tissue_thickness_um=50.0,
    )

    # Generic nearest-slice reconstruction assigns every query to a section.
    assert np.all(assignment.nearest_indices >= 0)
    # The sparse exporter retains only the physical +/-25 um support of the
    # single annotated section and does not fill neighboring retained levels.
    expected_support = np.abs(query_z) <= 25.0
    np.testing.assert_array_equal(assignment.support_mask, expected_support)
    np.testing.assert_array_equal(
        assignment.source_indices[~expected_support],
        UNSUPPORTED_SECTION_INDEX,
    )


def test_absent_section_has_no_observed_support() -> None:
    assignment = assign_observed_section_support(
        np.array([0.0, 200.0, 400.0]),
        np.array([0.0, 200.0, 400.0]),
        section_present=np.array([True, False, True]),
        tissue_thickness_um=50.0,
    )
    np.testing.assert_array_equal(
        assignment.source_indices, np.array([0, -1, 2])
    )


def test_collision_policy_uses_distance_then_lower_section_index() -> None:
    labels = np.array(
        [
            [17, 99, 23],
            [17, 42, 24],
        ],
        dtype=np.uint32,
    )
    distances = np.array(
        [
            [10.0, 12.0, 5.0],
            [20.0, 3.0, 5.0],
        ]
    )
    sections = np.array(
        [
            [12, 12, 20],
            [16, 16, 18],
        ]
    )

    result = resolve_label_collisions(labels, distances, sections)

    np.testing.assert_array_equal(
        result.labels, np.array([17, 42, 24], dtype=np.uint32)
    )
    np.testing.assert_array_equal(result.source_indices, np.array([12, 16, 18]))
    assert result.multiply_sampled_voxels == 3
    assert result.same_id_collision_voxels == 1
    assert result.conflicting_id_collision_voxels == 2
    assert result.labels.dtype == np.uint32
