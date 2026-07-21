"""Thin compatibility layer around the project-pinned :mod:`xmodmap` repository.

This module intentionally does *not* reimplement xIV-LDDMM algorithms.  It centralizes
three practical adaptations required by this project:

1. Load the repository selected in ``project_config.yaml`` instead of relying on a
   machine-specific ``PYTHONPATH``.
2. Call repository-native preprocessing and output functions whenever they already
   implement the desired operation.
3. Make the historical VTK coordinate convention explicit.  ``xmodmap.io.getOutput``
   names its input ``YXZ`` and swaps the first two columns while writing.  Our modern
   particle archives store ordinary world ``XYZ`` coordinates, so the adapter swaps
   before calling xmodmap; the two swaps cancel and the VTK file remains XYZ.

The adapters are deliberately small.  Project-specific extensions such as complete
NIfTI-affine handling, self-describing NPZ metadata, arbitrary 3-D block aggregation,
and reference-grid NIfTI reconstruction remain outside xmodmap because the public
``unpack`` branch does not provide those facilities.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from project_config import resolve_path


@dataclass(frozen=True)
class XmodmapAPI:
    """Repository functions used by the configured workflow."""

    xmodmap: Any
    get_from_file: Callable[..., Any]
    make_from_single_channel_image: Callable[..., Any]
    make_bins_from_multichannel_image: Callable[..., Any]
    get_entropy: Callable[..., Any]
    get_mass_ratio: Callable[..., Any]
    write_particle_vtk: Callable[..., Any]
    write_vtk: Callable[..., Any]
    make_pq: Callable[..., Any]
    resize_data: Callable[..., Any]


def load_xmodmap(config: dict[str, Any]) -> XmodmapAPI:
    """Import the exact xmodmap tree selected by the project configuration."""

    repository = resolve_path(config, config["software"]["xiv_lddmm_repository"])
    if not repository.is_dir():
        raise FileNotFoundError(
            "Configured xIV-LDDMM repository is missing: "
            f"{repository}. Clone the public 'unpack' branch or, preferably for "
            "reproduction, point the YAML to the colleague's exact copy."
        )
    repository_text = str(repository)
    if repository_text not in sys.path:
        sys.path.insert(0, repository_text)

    try:
        import xmodmap  # type: ignore
        from xmodmap.io.getInput import (  # type: ignore
            getFromFile,
            makeBinsFromMultiChannelImage,
            makeFromSingleChannelImage,
        )
        from xmodmap.io.getOutput import (  # type: ignore
            getEntropy,
            getJacobian,
            writeParticleVTK,
            writeVTK,
        )
        from xmodmap.preprocess.makePQ_legacy import makePQ  # type: ignore
        from xmodmap.preprocess.preprocess import resizeData  # type: ignore
    except Exception as error:  # pragma: no cover - depends on external repo/environment
        raise RuntimeError(
            f"Could not import xmodmap from {repository}. Verify the repository commit, "
            "PyTorch/PyKeOps environment, and package dependencies."
        ) from error

    return XmodmapAPI(
        xmodmap=xmodmap,
        get_from_file=getFromFile,
        make_from_single_channel_image=makeFromSingleChannelImage,
        make_bins_from_multichannel_image=makeBinsFromMultiChannelImage,
        get_entropy=getEntropy,
        get_mass_ratio=getJacobian,
        write_particle_vtk=writeParticleVTK,
        write_vtk=writeVTK,
        make_pq=makePQ,
        resize_data=resizeData,
    )


def _points_for_xmodmap_vtk(points: Any, coordinate_convention: str) -> Any:
    """Convert ordinary XYZ arrays to xmodmap's historical YXZ writer convention."""

    if coordinate_convention not in {"xyz", "xmodmap_yxz"}:
        raise ValueError(
            "VTK coordinate_convention must be 'xyz' or 'xmodmap_yxz'."
        )
    if coordinate_convention == "xmodmap_yxz":
        return points

    # Preserve torch tensors when supplied, avoiding a device/dtype round trip.
    if hasattr(points, "index_select"):
        import torch

        order = torch.tensor([1, 0, 2], device=points.device)
        return points.index_select(-1, order)
    array = np.asarray(points)
    return array[:, [1, 0, 2]]


def write_vtk_xyz(
    api: XmodmapAPI,
    points: Any,
    features: Sequence[Any],
    feature_names: Sequence[str],
    filename: str | Path,
    *,
    coordinate_convention: str = "xyz",
) -> None:
    """Write VTK through xmodmap while preserving ordinary XYZ coordinates."""

    api.write_vtk(
        _points_for_xmodmap_vtk(points, coordinate_convention),
        list(features),
        list(feature_names),
        str(filename),
    )


def write_particle_vtk_xyz(
    api: XmodmapAPI,
    points: Any,
    feature_mass: Any,
    filename: str | Path,
    *,
    coordinate_convention: str = "xyz",
    norm: bool = True,
    condense: bool = False,
    feature_names: Sequence[str] | None = None,
    support_weights: Any | None = None,
) -> None:
    """Call xmodmap's particle VTK writer with an explicit coordinate convention."""

    api.write_particle_vtk(
        _points_for_xmodmap_vtk(points, coordinate_convention),
        feature_mass,
        str(filename),
        norm=norm,
        condense=condense,
        featNames=None if feature_names is None else list(feature_names),
        sW=support_weights,
    )


def entropy_numpy(api: XmodmapAPI, feature_mass: np.ndarray) -> np.ndarray:
    """Return xmodmap's per-particle Shannon entropy as a NumPy vector."""

    value = api.get_entropy(np.asarray(feature_mass))
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def categorical_particles_xmodmap_native(
    api: XmodmapAPI,
    image_file: str | Path,
    *,
    voxel_sizes_mm: Sequence[float],
    voxel_volume_mm3: float,
    fine_label_ids: Sequence[int],
    background_labels: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Use xmodmap's native single-channel image converter in its valid regime.

    The public function accepts only one scalar spatial resolution and constructs
    centered index coordinates.  We therefore require an approximately isotropic image
    and use this backend only for legacy/reproduction jobs at native sampling.
    """

    voxel_sizes = np.asarray(voxel_sizes_mm, dtype=float)
    if voxel_sizes.shape != (3,) or not np.allclose(
        voxel_sizes, voxel_sizes[0], rtol=1e-4, atol=1e-6
    ):
        raise ValueError(
            "xmodmap.makeFromSingleChannelImage accepts one isotropic resolution. "
            f"Observed voxel sizes: {voxel_sizes.tolist()}. Use the project_affine_block "
            "backend for anisotropic or affine-world data."
        )

    points, features = api.make_from_single_channel_image(
        str(image_file),
        float(voxel_sizes[0]),
        bg=list(background_labels),
        ordering=np.asarray(fine_label_ids),
        ds=1,
        weights=float(voxel_volume_mm3),
    )
    if hasattr(points, "detach"):
        points = points.detach().cpu().numpy()
    if hasattr(features, "detach"):
        features = features.detach().cpu().numpy()
    return np.asarray(points, dtype=np.float32), np.asarray(features, dtype=np.float32)


def dense_bins_xmodmap_native(
    api: XmodmapAPI,
    image_file: str | Path,
    *,
    voxel_sizes_mm: Sequence[float],
    bins: int,
    mask_flat: np.ndarray,
    reverse: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Use xmodmap's native scalar-image binning and then apply a project mask.

    This deliberately uses ``ds=1``.  In the public implementation, stride
    downsampling does not multiply the coordinate spacing by ``ds`` and therefore
    changes the physical field of view.  Coarse Allen particles should instead use the
    project's mass-conserving 3-D block aggregation.
    """

    voxel_sizes = np.asarray(voxel_sizes_mm, dtype=float)
    if voxel_sizes.shape != (3,):
        raise ValueError("voxel_sizes_mm must contain three values.")
    points, features = api.make_bins_from_multichannel_image(
        str(image_file),
        voxel_sizes.tolist(),
        dimEff=3,
        dimFeats=1,
        ds=1,
        threshold=0,
        bins=int(bins),
        reverse=bool(reverse),
    )
    if hasattr(points, "detach"):
        points = points.detach().cpu().numpy()
    if hasattr(features, "detach"):
        features = features.detach().cpu().numpy()
    points = np.asarray(points, dtype=np.float32)
    features = np.asarray(features, dtype=np.float32)
    mask_flat = np.asarray(mask_flat, dtype=bool).reshape(-1)
    if points.shape[0] != mask_flat.size:
        raise RuntimeError(
            "xmodmap dense-bin output no longer matches the flattened image size; "
            "check the selected repository commit and image dimensionality."
        )
    return points[mask_flat], features[mask_flat]
