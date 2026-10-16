#!/usr/bin/env python3
"""Experimental WSI image-mode pseudotime test on one Allen Nissl endpoint pair.

Runs the public WSI upsampler directly on the accepted final-A2d Nissl anchors
and their jointly placed weights, at the same canonical physical pseudotimes as
annotation-driven densification. It does not change the Allen ``--driver nissl``
densifier, which still transports annotations with its own source flows.
Everything is written to one new experimental directory.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import zarr

from preprocess import densify_allen_annotations as dense

DEFAULT_MONTAGE_PSEUDOTIMES = (0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0)


def nissl_pair_endpoints(
    context: dense.SourceContext, anchor_root: Any, pair: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read placed Nissl endpoints and weights, requiring one registered grid."""
    ordinal_by_physical = {
        context.physical_by_section[section]: ordinal
        for ordinal, section in enumerate(context.annotation_sections)
    }
    missing = [physical for physical in pair if physical not in ordinal_by_physical]
    if missing:
        raise RuntimeError(f"Physical indices {missing} have no placed Nissl anchor")
    registered_shape = tuple(len(axis) for axis in context.registered_axes)
    images, weights = [], []
    for physical in pair:
        ordinal = ordinal_by_physical[physical]
        image = np.asarray(anchor_root["nissl"][ordinal], dtype=np.float32)
        weight = np.asarray(anchor_root["nissl_weight"][ordinal], dtype=np.float32)
        # Both were placed by the same final-A2d pullback onto the registered axes.
        if image.shape != (3, *registered_shape) or weight.shape != registered_shape:
            raise RuntimeError(
                f"Nissl anchor {physical} is not on the registered axes: image "
                f"{image.shape}, weight {weight.shape}, axes {registered_shape}"
            )
        if not (np.all(np.isfinite(image)) and np.all(np.isfinite(weight))):
            raise RuntimeError(f"Nissl anchor {physical} has non-finite values")
        if np.any(weight < 0) or not np.any(weight > 0):
            raise RuntimeError(f"Nissl anchor {physical} weight is negative or empty")
        images.append(image)
        weights.append(weight)
    return images[0], images[1], weights[0], weights[1]


def nissl_wsi_call(
    context: dense.SourceContext,
    pair: tuple[int, int],
    left_image: np.ndarray,
    right_image: np.ndarray,
    left_weight: np.ndarray,
    right_weight: np.ndarray,
    config: Mapping[str, Any],
    *,
    device: str,
    path_mode: str,
) -> tuple[dict[str, Any], np.ndarray]:
    """Return upsample_between_slices keyword arguments and the canonical indices."""
    if path_mode not in dense.WSI_PATH_MODES:
        raise ValueError(f"Unsupported WSI path_mode {path_mode!r}")
    xJ, J, W, present_mask = dense.wsi_pair_stack(
        context, pair, left_image, right_image, left_weight, right_weight
    )
    interior, pseudotimes = dense.canonical_pair_pseudotimes(context, pair)
    if not interior.size:
        raise RuntimeError(f"Pair {dense._pair_id(pair)} has no canonical interior plane")
    return {
        "xJ": xJ,
        "J": J,
        "W": W,
        "present_mask": present_mask,
        "mode": "img",
        "config": {**config, "device": device},
        "parallel": False,
        "pseudotimes": pseudotimes,
        "path_mode": path_mode,
        "return_velocities": True,
    }, interior


def validate_samples(
    samples: Mapping[str, Any],
    call: Mapping[str, Any],
    canonical_z: np.ndarray,
    path_mode: str,
) -> None:
    """Require WSI samples at exactly the requested pair, times, and z."""
    pseudotimes = call["pseudotimes"]
    channels, z_count, height, width = call["J"].shape
    if (
        str(samples["path_mode"]) != path_mode
        or not np.array_equal(np.asarray(samples["pseudotimes"], np.float64), pseudotimes)
        or not np.array_equal(
            np.asarray(samples["pair_indices"]),
            np.tile([[0, z_count - 1]], (len(pseudotimes), 1)),
        )
        or not np.allclose(np.asarray(samples["z"], np.float64), canonical_z, rtol=0, atol=1e-6)
    ):
        raise RuntimeError("WSI pseudotime samples do not match the requested times and z")
    if np.shape(samples["J"]) != (channels, len(pseudotimes), height, width):
        raise RuntimeError(f"Unexpected WSI sample shape {np.shape(samples['J'])}")
    if not np.all(np.isfinite(samples["J"])):
        raise RuntimeError("WSI pseudotime samples are not finite")


def montage_columns(
    pseudotimes: np.ndarray, targets: Sequence[float]
) -> list[tuple[str, int | None, float]]:
    """Use the observed endpoints at s=0/1 and the nearest canonical sample otherwise."""
    columns = []
    for target in targets:
        if target <= 0.0:
            columns.append(("left", None, float(target)))
        elif target >= 1.0:
            columns.append(("right", None, float(target)))
        else:
            index = int(np.argmin(np.abs(pseudotimes - target)))
            columns.append(("inferred", index, float(target)))
    return columns


def render_montage(
    path: Path,
    *,
    title: str,
    subtitle: str,
    columns: Sequence[tuple[str, int | None, float]],
    left_image: np.ndarray,
    right_image: np.ndarray,
    support: np.ndarray,
    samples: Mapping[str, Any],
    endpoint_z_um: tuple[float, float],
    pair: tuple[int, int],
    canonical_indices: np.ndarray,
    sections: Mapping[int, int],
) -> list[dict[str, Any]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows, cols = np.nonzero(support)
    pad = 10
    y_crop = slice(max(rows.min() - pad, 0), rows.max() + pad + 1)
    x_crop = slice(max(cols.min() - pad, 0), cols.max() + pad + 1)
    fig, axes = plt.subplots(
        1, len(columns), figsize=(2.4 * len(columns), 3.6), squeeze=False,
        facecolor="white",
    )
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.04, top=0.70, wspace=0.04)
    records = []
    for ax, (kind, index, target) in zip(axes[0], columns):
        if kind == "inferred":
            assert index is not None
            image = np.asarray(samples["J"])[:, index]
            s = float(samples["pseudotimes"][index])
            z_um = float(samples["z"][index])
            physical = int(canonical_indices[index])
            label = f"inferred (WSI)\nidx {physical}  |  s={s:.4f}"
        else:
            image = left_image if kind == "left" else right_image
            s = 0.0 if kind == "left" else 1.0
            z_um = endpoint_z_um[0 if kind == "left" else 1]
            physical = pair[0 if kind == "left" else 1]
            label = f"observed endpoint\nsection {sections[physical]}  |  idx {physical}"
        ax.imshow(
            np.clip(image.transpose(1, 2, 0), 0.0, 1.0)[y_crop, x_crop],
            interpolation="nearest",
        )
        ax.set_title(
            f"{label}\ntarget s={target:.3f}  |  z={z_um / 1000.0:.4f} mm",
            fontsize=8.0, linespacing=1.2,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#1b7837" if kind != "inferred" else "#2166ac")
            spine.set_linewidth(2.0)
        records.append({
            "state": "observed" if kind != "inferred" else "inferred",
            "physical_index": physical,
            "target_pseudotime": target,
            "pseudotime": s,
            "z_um": z_um,
        })
    fig.suptitle(title, fontsize=13, fontweight="semibold", y=0.97)
    fig.text(0.5, 0.86, subtitle, ha="center", va="center", fontsize=9, color="#333333")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)
    return records


def _save_samples(path: Path, samples: Mapping[str, Any], canonical_indices: np.ndarray) -> None:
    np.savez_compressed(
        path,
        J=np.asarray(samples["J"], dtype=np.float32),
        z_um=np.asarray(samples["z"], dtype=np.float64),
        pseudotimes=np.asarray(samples["pseudotimes"], dtype=np.float64),
        pair_indices=np.asarray(samples["pair_indices"]),
        canonical_indices=canonical_indices,
        path_mode=np.array(str(samples["path_mode"])),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", type=Path, default=dense.DEFAULT_DATASET)
    parser.add_argument("--annotations", type=Path, default=None)
    parser.add_argument("--registration-run", type=Path, default=dense.DEFAULT_REGISTRATION)
    parser.add_argument("--output", type=Path, required=True, help="new experimental directory")
    parser.add_argument("--pair", type=dense._parse_pair, required=True)
    parser.add_argument(
        "--expect-sections", type=int, nargs=2, required=True,
        metavar=("LEFT_SECTION", "RIGHT_SECTION"),
        help="Allen section numbers that the pair endpoints must be",
    )
    parser.add_argument(
        "--graphic-groups", type=int, nargs="+",
        default=list(dense.DEFAULT_ANNOTATION_GROUPS),
        help="groups used only to select the same anchor context as annotation tests",
    )
    parser.add_argument("--pair-config", type=Path, default=dense.DEFAULT_PAIR_CONFIG)
    parser.add_argument("--path-mode", choices=dense.WSI_PATH_MODES, required=True)
    parser.add_argument(
        "--compare-path-mode", choices=dense.WSI_PATH_MODES, default=None,
        help="also evaluate this path mode from the same persisted velocities",
    )
    parser.add_argument(
        "--montage-pseudotimes", type=float, nargs="+",
        default=list(DEFAULT_MONTAGE_PSEUDOTIMES),
    )
    parser.add_argument("--wsi-repository", type=Path, default=dense.DEFAULT_WSI_REPOSITORY)
    parser.add_argument("--device", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Experimental output must be new or empty: {output}")
    if args.compare_path_mode == args.path_mode:
        raise ValueError("--compare-path-mode must differ from --path-mode")
    pair = args.pair
    pair_id = dense._pair_id(pair)

    # Verifies the WSI and EM-LDDMM pins and the imported module path first.
    upsample_between_slices, torch, wsi_commit, module_path = dense._load_wsi_upsampler(
        args.wsi_repository
    )
    from wsi_pipeline.registration.upsample import evaluate_pseudotimes, load_pair_velocities

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {args.device}, but CUDA is unavailable")

    torch_device = torch.device(args.device)

    if torch_device.type == "cuda":
        cuda_index = (
            torch_device.index
            if torch_device.index is not None
            else torch.cuda.current_device()
        )
        torch.cuda.set_device(cuda_index)
    else:
        cuda_index = None

    context = dense.select_graphic_groups(
        dense.discover_inputs(args.dataset, args.registration_run, args.annotations),
        args.graphic_groups,
    )
    for physical, section in zip(pair, args.expect_sections):
        if context.physical_by_section.get(section) != physical:
            raise RuntimeError(
                f"Allen section {section} is physical index "
                f"{context.physical_by_section.get(section)}, not {physical}"
            )
    sections = dict(zip(pair, args.expect_sections))
    anchor_report = dense.materialize_anchors(context, output)
    anchor_root = zarr.open_group(str(output / "anchors.zarr"), mode="r")
    left_image, right_image, left_weight, right_weight = nissl_pair_endpoints(
        context, anchor_root, pair
    )
    # The Allen Nissl driver's solver configuration: the pair JSON with no profile.
    config = dense.effective_pair_config(args.pair_config, None)
    call, interior = nissl_wsi_call(
        context, pair, left_image, right_image, left_weight, right_weight, config,
        device=args.device, path_mode=args.path_mode,
    )
    canonical_z = np.asarray(context.axes[0], dtype=np.float64)[interior]
    endpoint_z_um = (float(context.axes[0][pair[0]]), float(context.axes[0][pair[1]]))

    if cuda_index is not None:
        torch.cuda.reset_peak_memory_stats(cuda_index)
    started = time.monotonic()
    result = upsample_between_slices(**call)
    if cuda_index is not None:
        torch.cuda.synchronize(cuda_index)
    elapsed = time.monotonic() - started
    samples = result["pseudotime_samples"]
    validate_samples(samples, call, canonical_z, args.path_mode)

    velocity_path = output / "pair_velocities.npz"
    archive = dense._persist_wsi_pair_velocities(result["pair_velocities"], velocity_path)
    stored = load_pair_velocities(velocity_path)
    # Post-hoc evaluation from the archive must reproduce the in-run samples.
    replay = evaluate_pseudotimes(
        call["xJ"], call["J"], call["W"], pair_velocities=stored,
        pseudotimes=call["pseudotimes"], path_mode=args.path_mode, device=args.device,
    )
    replay_error = float(np.max(np.abs(np.asarray(replay["J"]) - np.asarray(samples["J"]))))

    mode_slug = args.path_mode.replace("-", "_")
    _save_samples(output / f"pseudotime_samples_{mode_slug}.npz", samples, interior)
    columns = montage_columns(call["pseudotimes"], args.montage_pseudotimes)
    support = (left_weight > 0) | (right_weight > 0)
    subtitle = (
        f"Allen sections {sections[pair[0]]} → {sections[pair[1]]}  ·  WSI mode=img, "
        f"path_mode={{}}  ·  {len(interior)} canonical interior pseudotimes  ·  "
        f"WSI {wsi_commit[:10]}"
    )
    montage_kwargs = {
        "columns": columns, "left_image": left_image, "right_image": right_image,
        "support": support, "endpoint_z_um": endpoint_z_um, "pair": pair,
        "canonical_indices": interior, "sections": sections,
    }
    montages = {}
    panels = {}
    montages[args.path_mode] = output / f"nissl_wsi_montage_{mode_slug}.png"
    panels[args.path_mode] = render_montage(
        montages[args.path_mode],
        title=f"Nissl-driven WSI pseudotime interpolation · physical pair {pair[0]}–{pair[1]}",
        subtitle=subtitle.format(args.path_mode), samples=samples, **montage_kwargs,
    )
    comparison = None
    if args.compare_path_mode is not None:
        other = evaluate_pseudotimes(
            call["xJ"], call["J"], call["W"], pair_velocities=stored,
            pseudotimes=call["pseudotimes"], path_mode=args.compare_path_mode,
            device=args.device,
        )
        validate_samples(other, call, canonical_z, args.compare_path_mode)
        other_slug = args.compare_path_mode.replace("-", "_")
        _save_samples(output / f"pseudotime_samples_{other_slug}.npz", other, interior)
        montages[args.compare_path_mode] = output / f"nissl_wsi_montage_{other_slug}.png"
        panels[args.compare_path_mode] = render_montage(
            montages[args.compare_path_mode],
            title=(
                f"Nissl-driven WSI pseudotime interpolation · physical pair "
                f"{pair[0]}–{pair[1]} (same velocities, no re-registration)"
            ),
            subtitle=subtitle.format(args.compare_path_mode), samples=other,
            **montage_kwargs,
        )
        difference = np.abs(np.asarray(other["J"]) - np.asarray(samples["J"]))
        comparison = {
            "path_mode": args.compare_path_mode,
            "source": "evaluate_pseudotimes on the persisted pair_velocities.npz",
            "mean_abs_rgb_difference_by_plane": difference.mean(axis=(0, 2, 3)).tolist(),
            "max_abs_rgb_difference": float(difference.max()),
        }

    metadata = {
        "schema": "allen-nissl-wsi-pseudotime-test-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "experimental WSI img-mode test on Allen Nissl anchors; does not change "
            "the Allen --driver nissl annotation transport"
        ),
        "pair": list(pair),
        "pair_id": pair_id,
        "endpoint_allen_sections": list(args.expect_sections),
        "endpoint_z_um": list(endpoint_z_um),
        "canonical_interior_indices": interior.tolist(),
        "requested_pseudotimes": call["pseudotimes"].tolist(),
        "returned_pseudotime_z_um": np.asarray(samples["z"], np.float64).tolist(),
        "pseudotime_rule": "s=(z-z0)/(z1-z0) at each canonical interior z (context.axes[0])",
        "wsi_call": {
            "function": "wsi_pipeline.registration.upsample.upsample_between_slices",
            "mode": call["mode"],
            "path_mode": call["path_mode"],
            "return_velocities": call["return_velocities"],
            "parallel": call["parallel"],
            "device": args.device,
            "J_shape": list(call["J"].shape),
            "W_shape": list(call["W"].shape),
        },
        "weights": {
            "source": "anchors.zarr nissl_weight, placed with nissl by the same final A2d",
            "alignment_check": "image (3,Y,X) and weight (Y,X) both match registered axes",
            "ownership": "passed as W; WSI routes target W0, source WI and blend weights",
        },
        "wsi_commit": wsi_commit,
        "wsi_module_path": module_path,
        "emlddmm_pinned_commit": (
            dense.PROJECT / "configs/emlddmm-annotation-commit.txt"
        ).read_text().strip(),
        "solver_configuration": config,
        "solver_configuration_sha256": dense._stable_json_sha256(config),
        "solver_configuration_source": f"{args.pair_config} (Allen Nissl driver; no profile)",
        "pair_velocity_archive": archive,
        "posthoc_replay_max_abs_difference": replay_error,
        "path_mode_comparison": comparison,
        "montages": {mode: str(path) for mode, path in montages.items()},
        "montage_panels": panels,
        "wall_time_seconds": elapsed,
        "peak_cuda_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(cuda_index))
            if cuda_index is not None else None
        ),
        "anchors": anchor_report["zarr"],
        "reproducibility_scope": (
            "WSI input fingerprints identify registration inputs only, not the "
            "software environment"
        ),
    }
    dense._json(output / "nissl_wsi_test.json", metadata)
    print(json.dumps({
        "metadata": str(output / "nissl_wsi_test.json"),
        "montages": metadata["montages"],
        "posthoc_replay_max_abs_difference": replay_error,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
