"""Shared particle-construction and validation utilities.

Conceptual model
================
xIV-LDDMM does not operate directly on a labeled NIfTI array.  It operates on a
feature-valued particle measure

    mu = sum_i delta_(x_i) tensor nu_i,

where ``x_i`` is a 3-D position and ``nu_i`` is a non-negative vector of feature
*masses*.  For a hard segmentation at native voxel resolution, one voxel becomes one
particle and its feature vector is one-hot.  Multiplying that vector by voxel volume
means that total feature mass has units of mm^3 and approximates regional volume.

For a block-reduced representation with factor ``f``, a coarse particle summarizes an
``f x f x f`` block.  Its feature mass is

    nu[j, k] = voxel_volume * count(voxels in block j with label k).

Thus a boundary block can carry a mixture of labels instead of being assigned only the
majority label.  This is the main reason the legacy high-field script downsampled by
aggregation rather than nearest-neighbor resampling.

Coordinate handling
===================
The modern default is ``affine_world``: voxel indices are transformed through the full
NIfTI affine.  This preserves axis directions, rotations, translations, and physical
spacing.  The legacy scripts used only diagonal voxel sizes and then centered the grid
at zero; that behavior remains available as ``legacy_centered`` for reproduction, but
it should not be used casually for the Allen/OpenNeuro/MNI coordinate hierarchy.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class FeatureDefinition:
    """A flattened description of fine labels and coarser anatomical groups."""

    fine_ids: tuple[int, ...]
    fine_names: tuple[str, ...]
    coarse_names: tuple[str, ...]
    fine_to_coarse: tuple[int, ...]

    @property
    def number_fine(self) -> int:
        return len(self.fine_ids)

    @property
    def number_coarse(self) -> int:
        return len(self.coarse_names)


def _as_int_label_array(image_data: np.ndarray) -> np.ndarray:
    """Validate that a NIfTI contains a 3-D integer-like segmentation."""

    data = np.asarray(image_data)
    data = np.squeeze(data)
    if data.ndim != 3:
        raise ValueError(f"Expected a 3-D segmentation after squeeze; got {data.shape}.")
    if not np.all(np.isfinite(data)):
        raise ValueError("Segmentation contains NaN or infinite values.")
    rounded = np.rint(data)
    if not np.allclose(data, rounded, atol=1e-5):
        raise ValueError(
            "Segmentation contains non-integer values. Use a label image, not an MRI "
            "intensity or probabilistic image."
        )
    return rounded.astype(np.int64, copy=False)


def feature_definition_from_config(
    segmentation: np.ndarray,
    label_map: Mapping[str, Any] | None,
    background_labels: Sequence[int] = (0,),
) -> FeatureDefinition:
    """Create fine and coarse feature definitions.

    Two modes are supported.

    1. Explicit groups::

           groups:
             - name: amygdala
               labels:
                 - {id: 1, name: amygdala}

       Fine features are the listed labels.  Coarse features are group sums.

    2. Inferred labels::

           infer_all_nonbackground: true

       Every non-background integer found in the image becomes both a fine feature and
       its own coarse group.  This is useful for an initial Allen HRA conversion, but a
       scientifically meaningful crosswalk is still required before single-modality
       registration to another label system.
    """

    background = {int(value) for value in background_labels}
    if label_map is None or label_map.get("infer_all_nonbackground", False):
        ids = [int(value) for value in np.unique(segmentation) if int(value) not in background]
        if not ids:
            raise ValueError("No non-background labels were found in the segmentation.")
        names = [f"label_{value}" for value in ids]
        return FeatureDefinition(
            fine_ids=tuple(ids),
            fine_names=tuple(names),
            coarse_names=tuple(names),
            fine_to_coarse=tuple(range(len(ids))),
        )

    groups = label_map.get("groups")
    if not isinstance(groups, list) or not groups:
        raise ValueError("A label map must provide non-empty 'groups'.")

    fine_ids: list[int] = []
    fine_names: list[str] = []
    coarse_names: list[str] = []
    fine_to_coarse: list[int] = []

    for coarse_index, group in enumerate(groups):
        coarse_name = str(group["name"])
        coarse_names.append(coarse_name)
        labels = group.get("labels", [])
        if not labels:
            raise ValueError(f"Label group '{coarse_name}' has no labels.")
        for label in labels:
            if isinstance(label, Mapping):
                label_id = int(label["id"])
                label_name = str(label.get("name", f"label_{label_id}"))
            else:
                label_id = int(label)
                label_name = f"label_{label_id}"
            if label_id in background:
                raise ValueError(f"Background label {label_id} cannot be a feature.")
            if label_id in fine_ids:
                raise ValueError(f"Label {label_id} occurs more than once in the label map.")
            fine_ids.append(label_id)
            fine_names.append(label_name)
            fine_to_coarse.append(coarse_index)

    return FeatureDefinition(
        fine_ids=tuple(fine_ids),
        fine_names=tuple(fine_names),
        coarse_names=tuple(coarse_names),
        fine_to_coarse=tuple(fine_to_coarse),
    )


def voxel_volume_mm3(affine: np.ndarray) -> float:
    """Return physical voxel volume from the affine's 3x3 linear component."""

    volume = abs(float(np.linalg.det(np.asarray(affine, dtype=float)[:3, :3])))
    if not math.isfinite(volume) or volume <= 0:
        raise ValueError(f"Invalid voxel volume derived from affine: {volume}")
    return volume


def _apply_affine(affine: np.ndarray, ijk: np.ndarray) -> np.ndarray:
    """Apply a homogeneous 4x4 affine without requiring nibabel in this module."""

    ijk = np.asarray(ijk, dtype=np.float64)
    return ijk @ affine[:3, :3].T + affine[:3, 3]


def _coordinates_for_indices(
    indices: np.ndarray,
    shape: Sequence[int],
    affine: np.ndarray,
    coordinate_mode: str,
) -> np.ndarray:
    """Transform voxel index coordinates according to the selected policy."""

    if coordinate_mode == "affine_world":
        return _apply_affine(affine, indices)

    if coordinate_mode == "legacy_centered":
        # Reproduce the conceptual behavior of the colleague scripts: keep only voxel
        # sizes, discard NIfTI orientation/origin, and center each axis at zero.
        spacing = np.sqrt(np.sum(np.asarray(affine)[:3, :3] ** 2, axis=0))
        center = (np.asarray(shape, dtype=np.float64) - 1.0) / 2.0
        return (indices - center) * spacing

    raise ValueError(
        f"Unknown coordinate_mode '{coordinate_mode}'. Use 'affine_world' or "
        "'legacy_centered'."
    )


def construct_particles(
    segmentation: np.ndarray,
    affine: np.ndarray,
    features: FeatureDefinition,
    downsample_factor: int = 1,
    coordinate_mode: str = "affine_world",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Convert a labeled 3-D image to fine and coarse particle feature masses.

    Parameters
    ----------
    segmentation:
        3-D integer label image.
    affine:
        NIfTI voxel-index-to-world 4x4 affine.
    features:
        Fine labels and their grouping into coarse features.
    downsample_factor:
        Edge length of an aggregation block.  ``1`` yields one particle per retained
        voxel.  ``5`` summarizes up to 125 native voxels per particle.
    coordinate_mode:
        ``affine_world`` (recommended) or ``legacy_centered`` (reproduction only).

    Returns
    -------
    X:
        ``[N, 3]`` particle positions in millimeters.
    nu_sub:
        ``[N, C]`` coarse feature masses.
    nu_all:
        ``[N, F]`` fine feature masses.
    diagnostics:
        Counts and mass-conservation checks.
    """

    labels = _as_int_label_array(segmentation)
    affine = np.asarray(affine, dtype=np.float64)
    factor = int(downsample_factor)
    if factor < 1:
        raise ValueError("downsample_factor must be at least 1.")

    shape = np.asarray(labels.shape, dtype=int)
    volume = voxel_volume_mm3(affine)
    fine_ids = np.asarray(features.fine_ids, dtype=np.int64)

    if factor == 1:
        # Keep only voxels that belong to one of the configured feature labels.
        retained = np.isin(labels, fine_ids)
        ijk = np.argwhere(retained).astype(np.float64)
        retained_labels = labels[retained]

        nu_all = np.zeros((ijk.shape[0], features.number_fine), dtype=np.float32)
        id_to_column = {label_id: column for column, label_id in enumerate(features.fine_ids)}
        columns = np.fromiter(
            (id_to_column[int(label)] for label in retained_labels),
            dtype=np.int64,
            count=retained_labels.size,
        )
        nu_all[np.arange(ijk.shape[0]), columns] = np.float32(volume)
        X = _coordinates_for_indices(ijk, shape, affine, coordinate_mode)

    else:
        # Pad only for block reshaping.  Padded values are guaranteed not to contribute
        # because feature counting tests explicit configured label IDs.
        output_shape = np.ceil(shape / factor).astype(int)
        padded_shape = output_shape * factor
        padding = [(0, int(padded_shape[d] - shape[d])) for d in range(3)]
        padded = np.pad(labels, padding, mode="constant", constant_values=0)

        # Arrange as [block_x, within_x, block_y, within_y, block_z, within_z].
        blocks = padded.reshape(
            output_shape[0], factor,
            output_shape[1], factor,
            output_shape[2], factor,
        ).transpose(0, 2, 4, 1, 3, 5)
        flattened_blocks = blocks.reshape(-1, factor**3)

        nu_all = np.zeros(
            (flattened_blocks.shape[0], features.number_fine), dtype=np.float32
        )
        for column, label_id in enumerate(features.fine_ids):
            counts = np.count_nonzero(flattened_blocks == label_id, axis=1)
            nu_all[:, column] = counts.astype(np.float32) * np.float32(volume)

        retained = np.sum(nu_all, axis=1) > 0
        nu_all = nu_all[retained]

        # Represent each block at the physical center of its *actual* unpadded extent.
        starts = [np.arange(n) * factor for n in output_shape]
        centers = []
        for dimension, start_values in enumerate(starts):
            ends = np.minimum(start_values + factor, shape[dimension])
            centers.append((start_values + ends - 1.0) / 2.0)
        grid = np.meshgrid(*centers, indexing="ij")
        ijk = np.stack([axis.ravel() for axis in grid], axis=1)[retained]
        X = _coordinates_for_indices(ijk, shape, affine, coordinate_mode)

    # Coarse masses are sums of configured fine features.  Matrix multiplication makes
    # the relation explicit: nu_sub = nu_all @ G, where G[f, c] is 1 when fine feature
    # f belongs to coarse group c.
    grouping = np.zeros(
        (features.number_fine, features.number_coarse), dtype=np.float32
    )
    grouping[np.arange(features.number_fine), np.asarray(features.fine_to_coarse)] = 1.0
    nu_sub = nu_all @ grouping

    # The total mass should equal selected voxel count times native voxel volume,
    # irrespective of aggregation factor.  This is an important regression check.
    selected_voxels = int(np.count_nonzero(np.isin(labels, fine_ids)))
    expected_mass = selected_voxels * volume
    observed_mass = float(np.sum(nu_all, dtype=np.float64))
    relative_error = abs(observed_mass - expected_mass) / max(expected_mass, 1e-12)
    if relative_error > 1e-6:
        raise RuntimeError(
            "Particle aggregation failed mass conservation: "
            f"expected {expected_mass}, observed {observed_mass}."
        )

    diagnostics = {
        "image_shape": [int(value) for value in shape],
        "downsample_factor": factor,
        "number_particles": int(X.shape[0]),
        "selected_voxels": selected_voxels,
        "voxel_volume_mm3": volume,
        "expected_total_mass_mm3": expected_mass,
        "observed_total_mass_mm3": observed_mass,
        "relative_mass_error": relative_error,
        "coordinate_mode": coordinate_mode,
        "coordinate_min_mm": X.min(axis=0).tolist() if len(X) else [],
        "coordinate_max_mm": X.max(axis=0).tolist() if len(X) else [],
    }
    return X.astype(np.float32), nu_sub, nu_all, diagnostics


def save_particle_archive(
    output_npz: str | Path,
    X: np.ndarray,
    nu_sub: np.ndarray,
    nu_all: np.ndarray,
    features: FeatureDefinition,
    metadata: Mapping[str, Any],
    write_vtk: bool = True,
    *,
    vtk_writer: Any | None = None,
    entropy_function: Any | None = None,
) -> None:
    """Save a self-describing NPZ and an optional xmodmap-compatible VTK.

    The first three arrays deliberately remain ``X``, ``nu_Sub``, and ``nu_All`` in
    that order for compatibility with the colleague's scripts.  New code should still
    access them by key rather than relying on archive order.

    VTK output is delegated to :mod:`xmodmap.io.getOutput` through the adapter supplied
    by the calling script.  This avoids maintaining a second VTK implementation while
    still allowing the adapter to correct xmodmap's historical YXZ convention.
    """

    output_npz = Path(output_npz)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    metadata_json = json.dumps(dict(metadata), sort_keys=True)
    np.savez_compressed(
        output_npz,
        X=np.asarray(X, dtype=np.float32),
        nu_Sub=np.asarray(nu_sub, dtype=np.float32),
        nu_All=np.asarray(nu_all, dtype=np.float32),
        fine_feature_ids=np.asarray(features.fine_ids, dtype=np.int64),
        fine_feature_names=np.asarray(features.fine_names, dtype=str),
        coarse_feature_names=np.asarray(features.coarse_names, dtype=str),
        fine_to_coarse=np.asarray(features.fine_to_coarse, dtype=np.int64),
        metadata_json=np.asarray(metadata_json),
    )

    if write_vtk:
        if vtk_writer is None or entropy_function is None:
            raise ValueError(
                "write_vtk=True requires xmodmap-backed vtk_writer and entropy_function."
            )
        fine_weight = nu_all.sum(axis=1)
        coarse_weight = nu_sub.sum(axis=1)
        vtk_writer(
            X,
            [
                np.argmax(nu_all, axis=1) + 1,
                np.argmax(nu_sub, axis=1) + 1,
                fine_weight,
                coarse_weight,
                entropy_function(nu_all),
                entropy_function(nu_sub),
            ],
            [
                "FineFeatureIndex",
                "CoarseFeatureIndex",
                "FineWeight_mm3",
                "CoarseWeight_mm3",
                "FineEntropy",
                "CoarseEntropy",
            ],
            output_npz.with_suffix(".vtk"),
        )

def load_particle_archive(
    filename: str | Path,
    feature_key: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Load positions, one feature matrix, and metadata explicitly by key."""

    filename = Path(filename)
    with np.load(filename, allow_pickle=False) as archive:
        if "X" not in archive or feature_key not in archive:
            raise KeyError(
                f"{filename} must contain 'X' and '{feature_key}'. Keys: {archive.files}"
            )
        X = np.asarray(archive["X"])
        nu = np.asarray(archive[feature_key])
        metadata: dict[str, Any] = {}
        if "metadata_json" in archive:
            metadata = json.loads(str(np.asarray(archive["metadata_json"]).item()))
    return X, nu, metadata


def summarize_nifti_geometry(image: Any, filename: str | Path) -> dict[str, Any]:
    """Return geometry metadata from a loaded nibabel image."""

    affine = np.asarray(image.affine, dtype=np.float64)
    shape = tuple(int(v) for v in image.shape[:3])
    corners = np.array(
        [
            [i, j, k]
            for i in (0, shape[0] - 1)
            for j in (0, shape[1] - 1)
            for k in (0, shape[2] - 1)
        ],
        dtype=float,
    )
    world_corners = _apply_affine(affine, corners)
    return {
        "filename": str(Path(filename).resolve()),
        "shape": list(shape),
        "affine": affine.tolist(),
        "voxel_sizes_mm": np.sqrt(np.sum(affine[:3, :3] ** 2, axis=0)).tolist(),
        "voxel_volume_mm3": voxel_volume_mm3(affine),
        "world_bounds_min_mm": world_corners.min(axis=0).tolist(),
        "world_bounds_max_mm": world_corners.max(axis=0).tolist(),
    }


def cut_plane(
    X: np.ndarray,
    feature_arrays: Sequence[np.ndarray],
    origin_mm: Sequence[float],
    u_axis: Sequence[float],
    v_axis: Sequence[float],
    half_thickness_mm: float,
    hemisphere: str | None = None,
) -> tuple[np.ndarray, list[np.ndarray], dict[str, Any]]:
    """Keep 3-D particles in a slab around a plane and express them in plane coordinates.

    The plane is ``origin + a*u + b*v`` with orthonormal ``u`` and ``v``; its normal is
    ``n = u x v``.  A particle is kept when ``|(x - origin) . n| <= half_thickness``.
    Kept particles are returned as ``(a, b, 0)``, so ``x ~= origin + a*u + b*v``.

    Feature masses are divided by the slab thickness ``2 * half_thickness``.  A volume
    mass in mm^3 therefore becomes an area mass in mm^2, which is what a 2-D section
    measure carries.  ``hemisphere`` keeps ``x < 0`` (``"left"``) or ``x > 0``
    (``"right"``) in the input frame; ``None`` keeps both.
    """

    X = np.asarray(X, dtype=np.float64)
    origin = np.asarray(origin_mm, dtype=np.float64)
    u = np.asarray(u_axis, dtype=np.float64)
    v = np.asarray(v_axis, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] != 3 or origin.shape != (3,) or u.shape != (3,) or v.shape != (3,):
        raise ValueError("cut_plane needs [N, 3] positions and 3-vectors for origin, u, v.")
    if not np.allclose([u @ u, v @ v, u @ v], [1.0, 1.0, 0.0], atol=1e-9):
        raise ValueError("u_axis and v_axis must be orthonormal.")
    thickness = 2.0 * float(half_thickness_mm)
    if not thickness > 0:
        raise ValueError("half_thickness_mm must be positive.")
    normal = np.cross(u, v)

    offset = X - origin
    keep = np.abs(offset @ normal) <= float(half_thickness_mm)
    if hemisphere == "left":
        keep &= X[:, 0] < 0
    elif hemisphere == "right":
        keep &= X[:, 0] > 0
    elif hemisphere is not None:
        raise ValueError("hemisphere must be 'left', 'right', or None.")

    plane_X = np.zeros((int(keep.sum()), 3), dtype=np.float64)
    plane_X[:, 0] = offset[keep] @ u
    plane_X[:, 1] = offset[keep] @ v
    kept = [np.asarray(array)[keep] / thickness for array in feature_arrays]
    metadata = {
        "origin_mm": origin.tolist(),
        "u_axis": u.tolist(),
        "v_axis": v.tolist(),
        "normal": normal.tolist(),
        "half_thickness_mm": float(half_thickness_mm),
        "hemisphere": hemisphere,
        "mass_divisor_mm": thickness,
        "mass_units": "input mass / mm (mm^3 -> mm^2 for voxel volumes)",
        "inverse": "world = origin + plane_X[:, 0] * u_axis + plane_X[:, 1] * v_axis",
        "number_input": int(X.shape[0]),
        "number_kept": int(keep.sum()),
    }
    return plane_X, kept, metadata


def group_feature_columns(
    nu: np.ndarray,
    labels: Sequence[str],
    groups: Mapping[str, Sequence[str]],
    drop: Sequence[str] = (),
) -> tuple[np.ndarray, list[str]]:
    """Sum labelled feature columns into named groups.

    Every label must belong to exactly one group or to ``drop``, and every grouped
    label must exist, so a typo or an unexpected category fails instead of silently
    losing mass.  Returns ``(nu_grouped, group_names)``.
    """

    nu = np.asarray(nu)
    labels = [str(label) for label in labels]
    if nu.ndim != 2 or nu.shape[1] != len(labels):
        raise ValueError(f"nu has shape {nu.shape} but {len(labels)} labels were given.")
    column = {label: index for index, label in enumerate(labels)}
    assigned: dict[str, str] = {}
    for name, members in groups.items():
        for label in members:
            if label not in column:
                raise KeyError(f"Group '{name}' names unknown label '{label}'.")
            if label in assigned:
                raise ValueError(f"Label '{label}' is in groups '{assigned[label]}' and '{name}'.")
            assigned[label] = name
    unassigned = [label for label in labels if label not in assigned and label not in set(drop)]
    if unassigned:
        raise ValueError(f"Labels in neither a group nor drop: {unassigned}")

    names = list(groups)
    grouped = np.zeros((nu.shape[0], len(names)), dtype=nu.dtype)
    for index, name in enumerate(names):
        grouped[:, index] = nu[:, [column[label] for label in groups[name]]].sum(axis=1)
    return grouped, names


def bin_particles_2d(
    X: np.ndarray,
    feature_arrays: Sequence[np.ndarray],
    bin_mm: float,
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    """Aggregate particles into square ``bin_mm`` bins on their first two coordinates.

    Feature masses are summed per bin, so total mass is conserved exactly.  Each bin is
    placed at the centroid of its particles weighted by the total mass of the first
    feature array; the third coordinate is 0.  Bins start at the coordinate origin.
    Returns ``(X_bins, binned_arrays, particles_per_bin)``.
    """

    X = np.asarray(X, dtype=np.float64)
    if not bin_mm > 0:
        raise ValueError("bin_mm must be positive.")
    keys = np.floor(X[:, :2] / float(bin_mm)).astype(np.int64)
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.reshape(-1)
    number_bins = counts.shape[0]

    binned = []
    for array in feature_arrays:
        array = np.asarray(array, dtype=np.float64)
        out = np.zeros((number_bins, array.shape[1]), dtype=np.float64)
        np.add.at(out, inverse, array)
        binned.append(out)

    weight = np.asarray(feature_arrays[0], dtype=np.float64).sum(axis=1)
    weight_sum = np.zeros(number_bins)
    np.add.at(weight_sum, inverse, weight)
    centroid = np.zeros((number_bins, 2))
    np.add.at(centroid, inverse, X[:, :2] * weight[:, None])
    if np.any(weight_sum <= 0):
        raise ValueError("A bin has no positive mass; drop zero-mass particles first.")
    X_bins = np.zeros((number_bins, 3))
    X_bins[:, :2] = centroid / weight_sum[:, None]
    return X_bins, binned, counts


def varifold_score_mm(
    X_a: np.ndarray,
    nu_a: np.ndarray,
    X_b: np.ndarray,
    nu_b: np.ndarray,
    sigmas_mm: Sequence[float] = (2.0, 1.0, 0.5),
) -> float:
    """Relative Gaussian-kernel distance between two feature-valued measures.

    For each physical scale ``sigma`` (mm), with ``k(x, y) = exp(-|x-y|^2 / 2 sigma^2)``
    and a linear feature kernel,

        d_sigma = |mu_a - mu_b|^2_sigma / |mu_b|^2_sigma,

    and the score is the mean of ``d_sigma`` over scales.  Unlike xmodmap's training
    loss, the scales and weights are fixed, so the score can be compared across runs
    that share the same reference measure ``b``.  0 means identical measures.
    """

    def inner(X, nu, Y, mu, sigma):
        total = 0.0
        for start in range(0, X.shape[0], 2048):
            stop = start + 2048
            d2 = ((X[start:stop, None, :] - Y[None, :, :]) ** 2).sum(axis=-1)
            total += float(np.sum(np.exp(-d2 / (2.0 * sigma * sigma)) * (nu[start:stop] @ mu.T)))
        return total

    X_a, X_b = np.asarray(X_a, np.float64), np.asarray(X_b, np.float64)
    nu_a, nu_b = np.asarray(nu_a, np.float64), np.asarray(nu_b, np.float64)
    if X_a.shape[1] != X_b.shape[1] or nu_a.shape[1] != nu_b.shape[1]:
        raise ValueError("Measures must share spatial and feature dimensions.")
    scores = []
    for sigma in sigmas_mm:
        bb = inner(X_b, nu_b, X_b, nu_b, sigma)
        aa = inner(X_a, nu_a, X_a, nu_a, sigma)
        ab = inner(X_a, nu_a, X_b, nu_b, sigma)
        scores.append(max(aa + bb - 2.0 * ab, 0.0) / bb)
    return float(np.mean(scores))
