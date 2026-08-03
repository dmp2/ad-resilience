"""Support-aware helpers for sparse serial-section reconstruction.

EM-LDDMM's generic 2-D-to-3-D reconstruction assigns every query location to
the closest section.  That behavior is useful for dense intensity
reconstruction, but it must not be used to imply that sparse labels were
observed between physical tissue sections.

The functions here deliberately do not reimplement EM-LDDMM transformations.
They operate on registered-space z coordinates after the upstream point chain
has mapped target points into the registered histology space and before the
upstream per-section affine is selected.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


UNSUPPORTED_SECTION_INDEX = -1


@dataclass(frozen=True)
class SectionAssignment:
    """Nearest-section assignment plus independently defined observed support."""

    nearest_indices: np.ndarray
    distances_um: np.ndarray
    support_mask: np.ndarray
    source_indices: np.ndarray


@dataclass(frozen=True)
class CollisionResolution:
    """Deterministic result and audit counters for competing label samples."""

    labels: np.ndarray
    source_indices: np.ndarray
    candidate_count: np.ndarray
    multiply_sampled_voxels: int
    same_id_collision_voxels: int
    conflicting_id_collision_voxels: int


def assign_observed_section_support(
    query_z_um: np.ndarray,
    section_z_um: np.ndarray,
    *,
    section_present: np.ndarray | None = None,
    tissue_thickness_um: float,
    unsupported_index: int = UNSUPPORTED_SECTION_INDEX,
    atol_um: float = 1e-6,
) -> SectionAssignment:
    """Assign nearest serial sections while preserving finite tissue support.

    Ties are resolved toward the lower section index.  ``source_indices`` is
    set to ``unsupported_index`` outside the half-thickness support of a
    present section; ``nearest_indices`` retains the generic closest-section
    assignment for diagnostics.
    """

    query = np.asarray(query_z_um, dtype=np.float64)
    sections = np.asarray(section_z_um, dtype=np.float64)
    if sections.ndim != 1 or sections.size == 0:
        raise ValueError("section_z_um must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(sections)):
        raise ValueError("section_z_um must contain only finite values")
    if sections.size > 1 and not np.all(np.diff(sections) > 0):
        raise ValueError("section_z_um must be strictly increasing")
    if not np.isfinite(tissue_thickness_um) or tissue_thickness_um <= 0:
        raise ValueError("tissue_thickness_um must be finite and positive")

    if section_present is None:
        present = np.ones(sections.shape, dtype=bool)
    else:
        present = np.asarray(section_present, dtype=bool)
        if present.shape != sections.shape:
            raise ValueError("section_present must match section_z_um")

    flat_query = query.reshape(-1)
    distances = np.abs(flat_query[:, None] - sections[None, :])
    nearest = np.argmin(distances, axis=1)
    nearest_distance = distances[np.arange(flat_query.size), nearest]
    supported = present[nearest] & (
        nearest_distance <= (float(tissue_thickness_um) / 2.0 + float(atol_um))
    )
    source = np.where(supported, nearest, int(unsupported_index))

    return SectionAssignment(
        nearest_indices=nearest.reshape(query.shape),
        distances_um=nearest_distance.reshape(query.shape),
        support_mask=supported.reshape(query.shape),
        source_indices=source.reshape(query.shape),
    )


def resolve_label_collisions(
    candidate_labels: np.ndarray,
    candidate_distances_um: np.ndarray,
    candidate_section_indices: np.ndarray,
    *,
    candidate_valid: np.ndarray | None = None,
    unsupported_index: int = UNSUPPORTED_SECTION_INDEX,
) -> CollisionResolution:
    """Resolve multiple categorical samples by distance, then section index.

    Arrays have shape ``(candidate, voxel...)``.  Background label 0 remains a
    valid observed categorical value.  Invalid candidates are excluded through
    ``candidate_valid`` rather than by overloading the label value.
    """

    labels = np.asarray(candidate_labels)
    distances = np.asarray(candidate_distances_um, dtype=np.float64)
    sections = np.asarray(candidate_section_indices, dtype=np.int64)
    if labels.shape != distances.shape or labels.shape != sections.shape:
        raise ValueError("candidate arrays must have identical shapes")
    if labels.ndim < 2 or labels.shape[0] == 0:
        raise ValueError("candidate arrays must have shape (candidate, voxel...)")
    if labels.dtype.kind not in {"u", "i"}:
        raise TypeError("candidate_labels must contain integer categorical IDs")

    if candidate_valid is None:
        valid = np.isfinite(distances)
    else:
        valid = np.asarray(candidate_valid, dtype=bool)
        if valid.shape != labels.shape:
            raise ValueError("candidate_valid must match candidate arrays")
        valid &= np.isfinite(distances)

    voxel_shape = labels.shape[1:]
    labels_2d = labels.reshape(labels.shape[0], -1)
    distances_2d = distances.reshape(distances.shape[0], -1)
    sections_2d = sections.reshape(sections.shape[0], -1)
    valid_2d = valid.reshape(valid.shape[0], -1)

    out_labels = np.zeros(labels_2d.shape[1], dtype=labels.dtype)
    out_sections = np.full(
        labels_2d.shape[1], int(unsupported_index), dtype=np.int64
    )
    counts = np.sum(valid_2d, axis=0, dtype=np.int64)
    same_id = 0
    conflicting_id = 0

    for voxel in range(labels_2d.shape[1]):
        candidates = np.flatnonzero(valid_2d[:, voxel])
        if candidates.size == 0:
            continue
        order = np.lexsort(
            (sections_2d[candidates, voxel], distances_2d[candidates, voxel])
        )
        chosen = candidates[order[0]]
        out_labels[voxel] = labels_2d[chosen, voxel]
        out_sections[voxel] = sections_2d[chosen, voxel]
        if candidates.size > 1:
            unique_ids = np.unique(labels_2d[candidates, voxel])
            if unique_ids.size == 1:
                same_id += 1
            else:
                conflicting_id += 1

    return CollisionResolution(
        labels=out_labels.reshape(voxel_shape),
        source_indices=out_sections.reshape(voxel_shape),
        candidate_count=counts.reshape(voxel_shape),
        multiply_sampled_voxels=int(np.count_nonzero(counts > 1)),
        same_id_collision_voxels=same_id,
        conflicting_id_collision_voxels=conflicting_id,
    )
