"""Render support-aware QC from the saved Allen 400-um EM-LDDMM state.

This is a rendering-only companion to the pinned upstream QC writer.  It reads
the same immutable registration state and native histology, propagates the
loader's W0 support with the image numerator, and never writes a registration
input, transform, checkpoint, or production reconstruction.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
import numpy as np
import torch

from preprocess import render_allen_emlddmm_native_qc as canonical


SUPPORT_THRESHOLD = 0.05
N_SLICES = 5
OUTPUT_NAME = "emlddmm_support_aware_qc_400um"
CANONICAL_REVERSE = Path(
    "MRI_7T_WHOLE/HIST_NISSL_registered_to_MRI_7T_WHOLE/qc/"
    "HIST_NISSL_HIST_NISSL_to_MRI_7T_WHOLE.jpg"
)
OUTPUTS = {
    "canonical_histology_to_mri": "canonical_upstream_histology_to_mri.jpg",
    "support_aware_histology_to_mri": (
        "support_aware_histology_to_mri_orthogonal.png"
    ),
    "interpolated_support": "interpolated_support_coverage_orthogonal.png",
    "mri_to_histology_observed": "mri_to_histology_observed_sections.png",
    "observed_section_overlays": "observed_section_overlays.png",
}


def _as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def support_normalize(
    numerator: np.ndarray,
    denominator: np.ndarray,
    *,
    threshold: float = SUPPORT_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray]:
    """Divide all channels by one spatial support map above ``threshold``.

    Unsupported output is numerically zero but must be displayed and described
    using the returned mask; zero is not assigned anatomical meaning.
    """
    values = np.asarray(numerator)
    support = np.asarray(denominator)
    if support.ndim == values.ndim and support.shape[0] == 1:
        support = support[0]
    if values.ndim < 2 or values.shape[1:] != support.shape:
        raise ValueError("Numerator channels and spatial support shapes differ")
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("Support threshold must be finite and positive")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(support)):
        raise ValueError("Numerator and support must be finite")
    if np.any(support < -1.0e-6):
        raise ValueError("Support must be nonnegative")

    support = np.clip(support.astype(np.float32, copy=False), 0.0, 1.0)
    supported = support >= np.float32(threshold)
    output = np.zeros(values.shape, dtype=np.float32)
    np.divide(
        values.astype(np.float32, copy=False),
        support[None],
        out=output,
        where=supported[None],
    )
    if not np.all(np.isfinite(output)):
        raise RuntimeError("Support normalization emitted NaN or Inf")
    return output, supported


def interpolate_numerator_and_support(
    em: Any,
    axes: list[np.ndarray],
    numerator: np.ndarray,
    support: np.ndarray,
    points: np.ndarray | torch.Tensor,
    *,
    interp2d: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample numerator and support identically without applying a threshold."""
    kwargs = {"padding_mode": "zeros"}
    if interp2d:
        kwargs["interp2d"] = True
    sampled_numerator = _as_numpy(em.interp(axes, numerator, points, **kwargs))
    sampled_support = _as_numpy(
        em.interp(axes, support[None], points, **kwargs)
    )[0]
    sampled_numerator = np.asarray(sampled_numerator, dtype=np.float32)
    sampled_support = np.clip(sampled_support, 0.0, 1.0).astype(
        np.float32, copy=False
    )
    if (
        not np.all(np.isfinite(sampled_numerator))
        or not np.all(np.isfinite(sampled_support))
    ):
        raise RuntimeError("Interpolation emitted NaN or Inf")
    return sampled_numerator, sampled_support



def interpolate_supported_numerator(
    em: Any,
    axes: list[np.ndarray],
    numerator: np.ndarray,
    support: np.ndarray,
    points: np.ndarray | torch.Tensor,
    *,
    threshold: float = SUPPORT_THRESHOLD,
    interp2d: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample a preweighted numerator and support using pinned conventions."""
    sampled_numerator, sampled_support = interpolate_numerator_and_support(
        em,
        axes,
        numerator,
        support,
        points,
        interp2d=interp2d,
    )
    image, supported = support_normalize(
        sampled_numerator, sampled_support, threshold=threshold
    )
    return image, sampled_support.astype(np.float32, copy=False), supported


def support_aware_interp(
    em: Any,
    axes: list[np.ndarray],
    image: np.ndarray,
    support: np.ndarray,
    points: np.ndarray | torch.Tensor,
    *,
    threshold: float = SUPPORT_THRESHOLD,
    interp2d: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate ``support * image`` and support, then safely normalize."""
    data = np.asarray(image, dtype=np.float32)
    weights = np.asarray(support, dtype=np.float32)
    if data.shape[1:] != weights.shape:
        raise ValueError("Image channels and support shapes differ")
    return interpolate_supported_numerator(
        em,
        axes,
        data * weights[None],
        weights,
        points,
        threshold=threshold,
        interp2d=interp2d,
    )


def _representative_observed(observed: np.ndarray) -> np.ndarray:
    positions = np.rint(
        np.linspace(0, len(observed) - 1, N_SLICES + 2)[1:-1]
    ).astype(np.int64)
    return np.asarray(observed, dtype=np.int64)[positions]


def _registered_sparse_histology(
    em: Any,
    state: canonical.ResolvedState,
    selected: set[int],
) -> tuple[
    np.ndarray,
    np.ndarray,
    list[np.ndarray],
    np.ndarray,
    dict[int, tuple[np.ndarray, np.ndarray]],
]:
    """Apply saved A2d section transforms to numerator and W0 independently."""
    spatial_shape = state.target_shape[1:]
    numerator = np.zeros((3, *spatial_shape), dtype=np.float32)
    support = np.zeros(spatial_shape, dtype=np.float32)
    originals: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    expected_axes: list[np.ndarray] | None = None
    mean_translation = np.mean(state.arrays["A2d"][:, :2, -1], axis=0)

    for count, index_value in enumerate(state.histology.observed, 1):
        index = int(index_value)
        image, native_support = canonical._load_native_section(
            state.histology, index, image=True
        )
        assert image is not None
        section_axes = [
            state.histology.axes[0][index : index + 1],
            *state.histology.axes[1:],
        ]
        down_axes, down_image, down_support = canonical._weighted_section_downsample(
            em, section_axes, image, native_support, state.down_j
        )
        down_image = np.asarray(down_image[:, 0], dtype=np.float32)
        down_support = np.asarray(down_support[0], dtype=np.float32)
        inplane = [np.asarray(axis) for axis in down_axes[1:]]
        if expected_axes is None:
            expected_axes = [state.histology.axes[0], *inplane]
        elif any(
            not np.array_equal(actual, expected)
            for actual, expected in zip(inplane, expected_axes[1:], strict=True)
        ):
            raise RuntimeError("Sectionwise downsampling produced varying axes")

        output_grid = torch.stack(
            torch.meshgrid(
                torch.as_tensor(inplane[0] - mean_translation[0], dtype=torch.float32),
                torch.as_tensor(inplane[1] - mean_translation[1], dtype=torch.float32),
                indexing="ij",
            )
        )
        matrix = torch.as_tensor(state.arrays["A2d"][index], dtype=torch.float32)
        sample_points = (
            matrix[:2, :2]
            @ output_grid.permute(1, 2, 0)[..., None]
        )[..., 0] + matrix[:2, -1]
        sample_points = sample_points.permute(2, 0, 1)
        registered_numerator, registered_support = interpolate_numerator_and_support(
            em,
            inplane,
            down_image * down_support[None],
            down_support,
            sample_points,
            interp2d=True,
        )
        numerator[:, index] = registered_numerator
        support[index] = registered_support
        if index in selected:
            originals[index] = (down_image.copy(), down_support.copy())
        if count % 50 == 0:
            print(
                f"support-warped {count}/{len(state.histology.observed)} sections",
                flush=True,
            )

    if expected_axes is None or set(originals) != selected:
        raise RuntimeError("Observed histology reconstruction is incomplete")
    observed_planes = np.any(support >= SUPPORT_THRESHOLD, axis=(1, 2))
    allowed = np.zeros(len(observed_planes), dtype=bool)
    allowed[state.histology.observed] = True
    if np.any(observed_planes & ~allowed):
        raise RuntimeError("Unsupported physical planes acquired support")
    if not np.all(np.isfinite(numerator)) or not np.all(np.isfinite(support)):
        raise RuntimeError("Registered sparse histology contains NaN or Inf")
    return numerator, support, expected_axes, mean_translation, originals


def _forward_field(
    em: Any, state: canonical.ResolvedState
) -> tuple[list[torch.Tensor], torch.Tensor]:
    xv = [torch.as_tensor(state.arrays[f"xv{i}"], dtype=torch.float32) for i in range(3)]
    velocity = torch.as_tensor(state.arrays["v"], dtype=torch.float32)
    affine = torch.as_tensor(state.arrays["A"], dtype=torch.float32)
    phi = em.v_to_phii(xv, -velocity.flip(0))
    mapped = (
        affine[:3, :3] @ phi.permute(1, 2, 3, 0)[..., None]
    )[..., 0] + affine[:3, -1]
    return xv, mapped.permute(3, 0, 1, 2)


def _plane_grid(
    axes: list[np.ndarray], axis: int, index: int
) -> torch.Tensor:
    selected = [np.asarray(value) for value in axes]
    selected[axis] = selected[axis][index : index + 1]
    return torch.stack(
        torch.meshgrid(
            *(torch.as_tensor(value, dtype=torch.float32) for value in selected),
            indexing="ij",
        )
    )


def _orthogonal_supported_planes(
    em: Any,
    state: canonical.ResolvedState,
    source_axes: list[np.ndarray],
    histology_axes: list[np.ndarray],
    numerator: np.ndarray,
    support: np.ndarray,
    mean_translation: np.ndarray,
    *,
    threshold: float,
) -> dict[int, list[tuple[int, np.ndarray, np.ndarray]]]:
    xv, affine_phi = _forward_field(em, state)
    result: dict[int, list[tuple[int, np.ndarray, np.ndarray]]] = {}
    for axis in range(3):
        indices = np.rint(
            np.linspace(0, len(source_axes[axis]) - 1, N_SLICES + 2)[1:-1]
        ).astype(int)
        result[axis] = []
        for index in indices:
            output_grid = _plane_grid(source_axes, axis, int(index))
            hist_points = em.interp(xv, affine_phi, output_grid)
            hist_points = hist_points.clone()
            hist_points[1:] += torch.as_tensor(
                mean_translation, dtype=hist_points.dtype
            )[:, None, None, None]
            image, coverage, _ = interpolate_supported_numerator(
                em,
                histology_axes,
                numerator,
                support,
                hist_points,
                threshold=threshold,
            )
            image = np.squeeze(image, axis=axis + 1)
            coverage = np.squeeze(coverage, axis=axis)
            result[axis].append((int(index), image, coverage))
    return result


def _mri_on_observed_sections(
    em: Any,
    state: canonical.ResolvedState,
    source: canonical.ImageFacade,
    observed: np.ndarray,
    histology_axes: list[np.ndarray],
) -> dict[int, np.ndarray]:
    xv = [torch.as_tensor(state.arrays[f"xv{i}"], dtype=torch.float32) for i in range(3)]
    velocity = torch.as_tensor(state.arrays["v"], dtype=torch.float32)
    affine = torch.as_tensor(state.arrays["A"], dtype=torch.float32)
    inverse_affine = torch.linalg.inv(affine)
    identity = torch.stack(torch.meshgrid(*xv, indexing="ij"))
    inverse_phi = em.v_to_phii(xv, velocity)
    displacement = inverse_phi - identity
    a2d_inverse = torch.linalg.inv(
        torch.as_tensor(state.arrays["A2d"], dtype=torch.float32)
    )
    output: dict[int, np.ndarray] = {}
    inplane_axes = histology_axes[1:]
    for index_value in observed:
        index = int(index_value)
        grid = torch.stack(
            torch.meshgrid(
                torch.as_tensor(
                    state.histology.axes[0][index : index + 1], dtype=torch.float32
                ),
                *(torch.as_tensor(axis, dtype=torch.float32) for axis in inplane_axes),
                indexing="ij",
            )
        )
        points = grid.clone()
        points[1:] = (
            a2d_inverse[index, :2, :2]
            @ grid[1:].permute(1, 2, 3, 0)[..., None]
        )[..., 0].permute(3, 0, 1, 2) + a2d_inverse[index, :2, -1][
            :, None, None, None
        ]
        atlas_points = (
            inverse_affine[:3, :3]
            @ points.permute(1, 2, 3, 0)[..., None]
        )[..., 0] + inverse_affine[:3, -1]
        atlas_points = atlas_points.permute(3, 0, 1, 2)
        atlas_points = em.interp(xv, displacement, atlas_points) + atlas_points
        sampled = _as_numpy(em.interp(source.x, source.data, atlas_points))
        output[index] = sampled[:, 0]
    return output


def _extent(axes: list[np.ndarray], axis: int) -> tuple[float, float, float, float]:
    if axis == 0:
        horizontal, vertical = axes[2], axes[1]
    elif axis == 1:
        horizontal, vertical = axes[2], axes[0]
    else:
        horizontal, vertical = axes[1], axes[0]
    return (horizontal[0], horizontal[-1], vertical[-1], vertical[0])


def _save_support_aware_orthogonal(
    path: Path,
    planes: dict[int, list[tuple[int, np.ndarray, np.ndarray]]],
    axes: list[np.ndarray],
    threshold: float,
    *,
    title: str | None = None,
    supported_label: str = "interpolated with support",
    draw_support_boundary: bool = True,
    image_interpolation: str | None = None,
) -> None:
    figure, panels = plt.subplots(3, N_SLICES, figsize=(18, 11), squeeze=False)
    for axis in range(3):
        for column, (index, image, coverage) in enumerate(planes[axis]):
            shown = np.moveaxis(np.clip(image, 0.0, 1.0), 0, -1)
            supported = coverage >= threshold
            shown[~supported] = 0.16
            panel = panels[axis, column]
            panel.imshow(
                shown, extent=_extent(axes, axis), aspect="equal",
                interpolation=image_interpolation,
            )
            if (
                draw_support_boundary
                and np.any(supported)
                and np.any(~supported)
            ):
                panel.contour(
                    supported.astype(np.uint8),
                    levels=[0.5],
                    colors=["#00e5ff"],
                    linewidths=0.35,
                    extent=_extent(axes, axis),
                )
            panel.set_title(f"axis {axis}: {axes[axis][index] / 1000.0:.1f} mm")
            panel.set_xticks([])
            panel.set_yticks([])
    figure.legend(
        handles=[
            Patch(
                facecolor="white",
                edgecolor=("#00e5ff" if draw_support_boundary else "white"),
                label=supported_label,
            ),
            Patch(facecolor="0.16", edgecolor="0.16", label="unsupported (not zero anatomy)"),
        ],
        loc="lower center",
        ncol=2,
    )
    figure.suptitle(title or (
        "Support-aware histology → MRI (400 µm QC)\n"
        "RGB = interp(W0 × J) / interp(W0); cyan boundary = coverage ≥ "
        f"{threshold:g}. These samples are interpolated, not observed tissue."
    ))
    figure.tight_layout(rect=(0, 0.05, 1, 0.94))
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _save_coverage_orthogonal(
    path: Path,
    planes: dict[int, list[tuple[int, np.ndarray, np.ndarray]]],
    axes: list[np.ndarray],
    threshold: float,
    *,
    title: str | None = None,
    colorbar_label: str = "interpolated W0 coverage",
    image_interpolation: str | None = None,
) -> None:
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_under("0.16")
    figure, panels = plt.subplots(3, N_SLICES, figsize=(18, 11), squeeze=False)
    for axis in range(3):
        for column, (index, _, coverage) in enumerate(planes[axis]):
            panel = panels[axis, column]
            panel.imshow(
                coverage,
                cmap=cmap,
                vmin=threshold,
                vmax=1.0,
                extent=_extent(axes, axis),
                aspect="equal",
                interpolation=image_interpolation,
            )
            panel.set_title(f"axis {axis}: {axes[axis][index] / 1000.0:.1f} mm")
            panel.set_xticks([])
            panel.set_yticks([])
    colorbar = figure.colorbar(
        plt.cm.ScalarMappable(norm=Normalize(threshold, 1.0), cmap=cmap),
        ax=panels.ravel().tolist(),
        fraction=0.018,
        pad=0.015,
    )
    colorbar.set_label(colorbar_label)
    figure.suptitle(title or (
        "Histology support transported to MRI grid (400 µm QC)\n"
        f"gray = unsupported (coverage < {threshold:g}); colored = interpolation support, not observed tissue"
    ))
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _save_observed_panels(
    paths: tuple[Path, Path],
    observed: np.ndarray,
    originals: dict[int, tuple[np.ndarray, np.ndarray]],
    sampled_mri: dict[int, np.ndarray],
    threshold: float,
) -> None:
    all_mri = np.concatenate([sampled_mri[int(index)].ravel() for index in observed])
    low, high = np.quantile(all_mri, [0.01, 0.99])
    high = max(float(high), float(low) + 1.0e-6)
    overview, axes = plt.subplots(3, N_SLICES, figsize=(18, 9), squeeze=False)
    overlays, overlay_axes = plt.subplots(2, N_SLICES, figsize=(18, 6), squeeze=False)
    for column, index_value in enumerate(observed):
        index = int(index_value)
        histology, support = originals[index]
        mri = np.clip((sampled_mri[index][0] - low) / (high - low), 0.0, 1.0)
        mri_rgb = np.repeat(mri[..., None], 3, axis=2)
        hist_rgb = np.moveaxis(np.clip(histology, 0.0, 1.0), 0, -1)
        direct = support >= threshold
        hist_shown = hist_rgb.copy()
        hist_shown[~direct] = 0.16
        overlay = mri_rgb.copy()
        overlay[direct] = 0.5 * mri_rgb[direct] + 0.5 * hist_rgb[direct]
        title = f"physical {index} (observed)"
        for row, shown in enumerate((hist_shown, mri, support)):
            axes[row, column].imshow(
                shown,
                cmap=("gray" if row == 1 else "viridis" if row == 2 else None),
                vmin=(0.0 if row else None),
                vmax=(1.0 if row else None),
            )
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
        axes[0, column].set_title(title)
        for row, shown in enumerate((overlay, direct)):
            overlay_axes[row, column].imshow(
                shown,
                cmap=("gray" if row else None),
                vmin=(0.0 if row else None),
                vmax=(1.0 if row else None),
            )
            overlay_axes[row, column].set_xticks([])
            overlay_axes[row, column].set_yticks([])
        overlay_axes[0, column].set_title(title)
    axes[0, 0].set_ylabel("observed histology")
    axes[1, 0].set_ylabel("dense MRI sampled\nat section")
    axes[2, 0].set_ylabel("observed W0")
    overview.suptitle(
        "MRI → histology QC at five actual observed sections (400 µm in-plane)\n"
        "Observed refers to the native histology section and W0; "
        "MRI values are interpolated from dense MRI."
    )
    overview.tight_layout(rect=(0, 0, 1, 0.93))
    overview.savefig(paths[0], dpi=180, bbox_inches="tight")
    plt.close(overview)

    overlay_axes[0, 0].set_ylabel("MRI + observed\nhistology")
    overlay_axes[1, 0].set_ylabel("observed /\nunsupported")
    overlays.suptitle(
        "Selected observed-section overlays\n"
        f"top: 50% MRI + 50% histology only where observed W0 ≥ {threshold:g}; "
        "bottom: white observed, black unsupported"
    )
    overlays.tight_layout(rect=(0, 0, 1, 0.91))
    overlays.savefig(paths[1], dpi=180, bbox_inches="tight")
    plt.close(overlays)


def _output_paths() -> tuple[Path, Path]:
    final = canonical.POSTPROCESSED_QC / OUTPUT_NAME
    return canonical.POSTPROCESSED_QC / f".{OUTPUT_NAME}.tmp", final


def run(*, dry_run: bool = False, threshold: float = SUPPORT_THRESHOLD) -> Path | None:
    if not np.isfinite(threshold) or threshold <= 0.0 or threshold > 1.0:
        raise ValueError("Support threshold must be finite and in (0, 1]")
    em = canonical.pinned_emlddmm()
    state = canonical.resolve_state(em, require_upstream_output_available=False)
    stage, final = _output_paths()
    canonical._ensure_output_paths_available(stage, final)
    upstream = state.final_dir / CANONICAL_REVERSE
    if not upstream.is_file():
        raise RuntimeError(f"Canonical upstream reverse QC is missing: {upstream}")
    print(
        json.dumps(
            {
                "source_registration": str(canonical.REGISTRATION_ROOT),
                "commit": canonical.PIN,
                "support_threshold": threshold,
                "canonical_reverse_qc": str(upstream),
                "output_directory": str(final),
            },
            indent=2,
        ),
        flush=True,
    )
    if dry_run:
        return None

    state_hashes = {
        key: canonical._array_sha256(value) for key, value in state.arrays.items()
    }
    numerical_hash = canonical.sha256_file(state.numerical_path)
    stage.mkdir()
    shutil.copy2(upstream, stage / OUTPUTS["canonical_histology_to_mri"])
    source = canonical._build_source(em, state)
    selected = _representative_observed(state.histology.observed)
    numerator, support, histology_axes, mean_translation, originals = (
        _registered_sparse_histology(em, state, set(map(int, selected)))
    )
    planes = _orthogonal_supported_planes(
        em,
        state,
        source.x,
        histology_axes,
        numerator,
        support,
        mean_translation,
        threshold=threshold,
    )
    _save_support_aware_orthogonal(
        stage / OUTPUTS["support_aware_histology_to_mri"],
        planes,
        source.x,
        threshold,
    )
    _save_coverage_orthogonal(
        stage / OUTPUTS["interpolated_support"], planes, source.x, threshold
    )
    sampled_mri = _mri_on_observed_sections(
        em, state, source, selected, histology_axes
    )
    _save_observed_panels(
        (
            stage / OUTPUTS["mri_to_histology_observed"],
            stage / OUTPUTS["observed_section_overlays"],
        ),
        selected,
        originals,
        sampled_mri,
        threshold,
    )
    del numerator, support, planes, sampled_mri, source
    gc.collect()

    if numerical_hash != canonical.sha256_file(state.numerical_path):
        raise RuntimeError("Saved numerical registration product changed during QC")
    after_hashes = {
        key: canonical._array_sha256(value) for key, value in state.arrays.items()
    }
    if state_hashes != after_hashes:
        raise RuntimeError("In-memory saved registration state changed during QC")
    report = {
        "schema": "allen-emlddmm-support-aware-qc-v1",
        "status": "complete",
        "source_registration": str(canonical.REGISTRATION_ROOT.resolve()),
        "registration_checkpoint": str(canonical.CHECKPOINT.resolve()),
        "commit": canonical.PIN,
        "completed_scale_um": canonical.EXPECTED_EFFECTIVE_INPLANE_UM,
        "rendering_method": (
            "pinned linear/trilinear interpolation (torch grid_sample, "
            "align_corners=True) of W0*J and W0 separately; zero padding; "
            "RGB division only where interpolated W0 meets threshold"
        ),
        "support_threshold": threshold,
        "input_W0_source": {
            "directory": str(canonical.SUPPORT.resolve()),
            "meaning": (
                "native per-section loader support used to construct pinned "
                "EM-LDDMM W0"
            ),
            "downsampling": (
                "pinned support-weighted downsample_image_domain at [1,2,2]"
            ),
        },
        "gap_policy": (
            "all 2,846 physical planes are retained; absent planes remain zero "
            "support; only the local trilinear kernel can contribute and no gap "
            "filling is run"
        ),
        "canonical_upstream_qc": {
            "source": str(upstream.resolve()),
            "source_sha256": canonical.sha256_file(upstream),
            "comparison_copy": OUTPUTS["canonical_histology_to_mri"],
        },
        "selected_observed_physical_indices": selected.tolist(),
        "outputs": OUTPUTS,
        "unsupported_representation": (
            "neutral gray in RGB QC and under-range gray in coverage QC; numeric "
            "zeros outside support are display sentinels, not anatomical intensity"
        ),
        "intended_use": (
            "QC display only; must not be used as a registration target or "
            "interpreted as a production histology reconstruction"
        ),
        "registration_state_sha256": state_hashes,
        "numerical_product": str(state.numerical_path),
        "numerical_product_sha256": numerical_hash,
        "registration_inputs_or_transforms_written": False,
        "production_histology_written": False,
        "interpretation_limit": (
            "Improved sparse-data rendering does not itself demonstrate improved "
            "registration."
        ),
    }
    (stage / "support_aware_qc_provenance.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    expected = {Path(value) for value in OUTPUTS.values()} | {
        Path("support_aware_qc_provenance.json")
    }
    actual = {path.relative_to(stage) for path in stage.iterdir() if path.is_file()}
    if actual != expected or any(
        not (stage / path).stat().st_size for path in expected
    ):
        raise RuntimeError(
            f"Incomplete support-aware QC inventory: {sorted(map(str, actual))}"
        )
    os.rename(stage, final)
    print(f"Published support-aware QC: {final}", flush=True)
    return final


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render support-aware QC from the saved Allen 400-um state."
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--support-threshold", type=float, default=SUPPORT_THRESHOLD
    )
    args = parser.parse_args(argv)
    run(dry_run=args.dry_run, threshold=args.support_threshold)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
