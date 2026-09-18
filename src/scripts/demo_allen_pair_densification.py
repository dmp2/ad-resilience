#!/usr/bin/env python3
"""Quick lab-meeting demo of Allen through-plane diffeomorphic densification.

Uses the production densification module's accepted final-A2d placement, WSI pair fit,
and arbitrary-pseudotime source-flow evaluators. Only the two demo endpoints are loaded. Produces support-aware RGB Nissl
interpolants at the *actual canonical z positions* between two observed endpoints.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tifffile

from preprocess import densify_allen_annotations as dens


def parse_pair(text: str) -> tuple[int, int]:
    a, b = text.split("-", 1)
    a, b = int(a), int(b)
    if not 0 <= a < b:
        raise argparse.ArgumentTypeError("pair must be LEFT-RIGHT with 0 <= LEFT < RIGHT")
    return a, b


def pick_demo_pair(context: dens.SourceContext) -> tuple[int, int]:
    physical = sorted(context.physical_by_section[s] for s in context.annotation_sections)
    candidates = [(a, b) for a, b in zip(physical[:-1], physical[1:]) if b - a > 1]
    if not candidates:
        raise RuntimeError("No consecutive observed annotation/Nissl anchors bracket a missing canonical plane")
    # Prefer ~4 missing planes: enough to show a trajectory, but not a huge anatomical jump.
    return min(candidates, key=lambda p: (abs((p[1] - p[0] - 1) - 4), p[1] - p[0]))


def warp(channels, phi, axes, em, torch):
    return dens._warp_channels(channels, phi, axes, em, torch)


def to_u8_rgb(chw: np.ndarray) -> np.ndarray:
    return np.clip(np.moveaxis(chw, 0, -1), 0.0, 1.0).astype(np.float32) * 255.0

def render_labels_to_rgb(labels: np.ndarray) -> np.ndarray:
    """Deterministically render categorical IDs as 3-channel float image."""
    labels = np.asarray(labels, dtype=np.uint32)
    rgb = np.zeros((3, *labels.shape), dtype=np.float32)

    ids = np.unique(labels)
    ids = ids[ids != 0]

    for label_id in ids:
        # Deterministic pseudo-color from the integer label.
        x = int(label_id)
        r = ((x * 37) % 251 + 4) / 255.0
        g = ((x * 73) % 251 + 4) / 255.0
        b = ((x * 109) % 251 + 4) / 255.0

        mask = labels == label_id
        rgb[0, mask] = r
        rgb[1, mask] = g
        rgb[2, mask] = b

    return rgb

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, default=dens.DEFAULT_DATASET)
    p.add_argument("--annotations", type=Path, default=None)
    p.add_argument(
        "--driver",
        choices=("nissl", "annotation"),
        default="nissl",
        help="representation used to estimate the diffeomorphic trajectory",
    )

    p.add_argument(
        "--section-pair",
        type=parse_pair,
        default=None,
        help="Allen section numbers LEFT-RIGHT; resolved to canonical physical indices",
    )

    p.add_argument(
        "--annotation-groups",
        type=int,
        nargs="+",
        default=[31, 265297118],
        help="graphic groups jointly used as binary ROI registration channels",
    )
    p.add_argument("--registration-run", type=Path, default=dens.DEFAULT_REGISTRATION)
    p.add_argument("--pair", type=parse_pair, default=None, help="physical indices LEFT-RIGHT")
    p.add_argument("--pair-config", type=Path, default=dens.DEFAULT_PAIR_CONFIG)
    p.add_argument("--wsi-repository", type=Path, default=dens.DEFAULT_WSI_REPOSITORY)
    p.add_argument("--device", default="auto")
    p.add_argument("--nt", type=int, default=None, help="optional temporal integration override")
    p.add_argument("--output", type=Path, default=Path("results/allen_densification_demo"))
    p.add_argument("--max-preview-width", type=int, default=1100)
    args = p.parse_args()

    context = dens.discover_inputs(
        args.dataset,
        args.registration_run,
        args.annotations,
    )

    if args.section_pair is not None:
        left_section, right_section = args.section_pair
        try:
            pair = (
                context.physical_by_section[left_section],
                context.physical_by_section[right_section],
            )
        except KeyError as exc:
            raise RuntimeError(
                f"Requested Allen section is not on the canonical lattice: {exc}"
            ) from exc
    else:
        pair = args.pair or pick_demo_pair(context)
        left_section = int(context.rows[pair[0]]["allen_section_number"])
        right_section = int(context.rows[pair[1]]["allen_section_number"])

    left_index, right_index = pair

    print(
        f"Endpoint sections {left_section}->{right_section}; "
        f"physical indices {left_index}->{right_index}; "
        f"{right_index-left_index-1} missing canonical planes"
    )
    def load_placed_nissl(physical_index: int):
        row = context.rows[physical_index]
        rel = row.get("prepared_relative_path", "")
        if not rel:
            raise RuntimeError(f"Physical index {physical_index} has no prepared Nissl image")
        image_path = (context.dataset / rel).resolve()
        stain = row.get("stain", "")
        weight_path = context.dataset / "support" / stain / image_path.name
        image = tifffile.imread(image_path)
        weight = tifffile.imread(weight_path).astype(np.float32)
        if image.shape != (*context.shape, 3) or weight.shape != context.shape:
            raise RuntimeError(f"Unexpected Nissl/support geometry at physical index {physical_index}")
        placed_image, placed_weight = dens.coarse._warp_saved_section(
            image.transpose(2, 0, 1).astype(np.float32) / 255.0,
            weight,
            context.final_a2d[physical_index],
            context.registered_axes[0],
            context.registered_axes[1],
            source_row_um=context.axes[1],
            source_column_um=context.axes[2],
        )
        return np.asarray(placed_image, dtype=np.float32), np.asarray(placed_weight, dtype=np.float32)

    def load_placed_annotation(physical_index: int, group: int) -> np.ndarray:
        section = int(context.rows[physical_index]["allen_section_number"])

        record = context.annotation_inventory.get((section, group))
        if record is None:
            raise RuntimeError(
                f"No annotation for section {section}, group {group}"
            )

        label_path = (context.annotations / record["path"]).resolve()
        labels = tifffile.imread(label_path)

        if labels.shape != context.shape or not np.issubdtype(labels.dtype, np.integer):
            raise RuntimeError(
                f"Invalid annotation raster at {label_path}: "
                f"dtype={labels.dtype}, shape={labels.shape}"
            )

        placed = dens.warp_categorical_section(
            labels,
            context.final_a2d[physical_index],
            context.registered_axes[0],
            context.registered_axes[1],
            source_row_um=context.axes[1],
            source_column_um=context.axes[2],
        )

        return np.asarray(placed, dtype=np.uint32)

    print("Loading and placing only the two selected endpoint sections...")

    left_maps = None
    right_maps = None
    driver_keys = None

    if args.driver == "nissl":
        left, wleft = load_placed_nissl(left_index)
        right, wright = load_placed_nissl(right_index)

    else:
        left_maps = {
            group: load_placed_annotation(left_index, group)
            for group in args.annotation_groups
        }
        right_maps = {
            group: load_placed_annotation(right_index, group)
            for group in args.annotation_groups
        }

        # One pair-level channel vocabulary so left/right channel semantics match.
        driver_keys = []

        for group in args.annotation_groups:
            ids = (
                set(map(int, np.unique(left_maps[group])))
                | set(map(int, np.unique(right_maps[group])))
            )
            ids.discard(0)

            driver_keys.extend(
                (group, label_id)
                for label_id in sorted(ids)
            )

        left = np.stack(
            [
                (left_maps[group] == label_id).astype(np.float32)
                for group, label_id in driver_keys
            ],
            axis=0,
        )

        right = np.stack(
            [
                (right_maps[group] == label_id).astype(np.float32)
                for group, label_id in driver_keys
            ],
            axis=0,
        )

        wleft = np.any(left > 0.0, axis=0).astype(np.float32)
        wright = np.any(right > 0.0, axis=0).astype(np.float32)

        print(
            f"Annotation registration driver: {len(driver_keys)} binary ROI channels "
            f"from groups {args.annotation_groups}"
        )

    config = dens._load_pair_config(args.pair_config.resolve())
    if args.nt is not None:
        config = dens.pair_solver_config(config, args.nt)

    print(
        f"Fitting demo pair physical {left_index}->{right_index} "
        f"({right_index-left_index-1} canonical missing planes; nt={config['nt']})"
    )
    left_flow, right_flow, flow_report, em, torch = dens.fit_pair_trajectories(
        left, right, wleft, wright, context.registered_axes, config,
        wsi_repository=args.wsi_repository, device=args.device,
    )

    out = args.output.resolve() / f"pair_{left_index:06d}_{right_index:06d}"
    planes_dir = out / "planes"
    planes_dir.mkdir(parents=True, exist_ok=True)

    z0 = float(context.axes[0][left_index])
    z1 = float(context.axes[0][right_index])
    frames = []
    rows = []
    eps = 1e-6

    for args.driver in "nissl":
        print(f"Rendering RGB Nissl interpolants at canonical physical z positions...")
        for physical in range(left_index, right_index + 1):
            z = float(context.axes[0][physical])
            t = (z - z0) / (z1 - z0)
            if physical == left_index:
                blended = left
            elif physical == right_index:
                blended = right
            else:
                phi_l = left_flow.evaluate(t)
                phi_r = right_flow.evaluate(1.0 - t)
                il = warp(left, phi_l, context.registered_axes, em, torch)
                ir = warp(right, phi_r, context.registered_axes, em, torch)
                wl = warp(wleft[None], phi_l, context.registered_axes, em, torch)[0]
                wr = warp(wright[None], phi_r, context.registered_axes, em, torch)[0]
                a = (1.0 - t) * wl
                b = t * wr
                den = a + b
                blended = np.zeros_like(il, dtype=np.float32)
                valid = den > eps
                blended[:, valid] = (
                    il[:, valid] * a[valid][None] + ir[:, valid] * b[valid][None]
                ) / den[valid][None]

        rgb = np.rint(to_u8_rgb(blended)).astype(np.uint8)
        kind = "observed" if physical in (left_index, right_index) else "inferred"
        path = planes_dir / f"{physical:06d}_t-{t:.3f}_{kind}.tif"
        tifffile.imwrite(path, rgb, photometric="rgb", metadata=None)
        frames.append((physical, t, kind, rgb))
        rows.append({"physical_index": physical, "z_um": z, "t": t, "state": kind, "file": str(path)})

        # Make a lightweight horizontal montage with only matplotlib, downsampling for display.
        import matplotlib.pyplot as plt

        n = len(frames)
        fig, axes = plt.subplots(1, n, figsize=(min(3.1*n, 18), 4.2), squeeze=False)
        for ax, (physical, t, kind, rgb) in zip(axes[0], frames):
            stride = max(1, int(np.ceil(rgb.shape[1] / args.max_preview_width)))
            ax.imshow(rgb[::stride, ::stride])
            ax.set_title(f"{kind}\nidx {physical} | t={t:.2f}", fontsize=9)
            ax.axis("off")
        fig.suptitle(
            f"Allen through-plane diffeomorphic densification: physical {left_index} → {right_index}",
            fontsize=13,
        )
        fig.tight_layout()
        montage = out / "densification_montage.png"
        fig.savefig(montage, dpi=180, bbox_inches="tight")
        plt.close(fig)
         
    else:
        print("Rendering categorical annotation interpolants at canonical physical z positions...")

        # Only show a sparse subset in the montage, but write every canonical plane.
        preview_indices = set(
            map(int, np.rint(np.linspace(left_index, right_index, 9)))
        )

        rows_by_group: dict[str, list[dict[str, object]]] = {
            str(group): [] for group in args.annotation_groups
        }
        preview_frames: dict[int, list[tuple[int, float, str, np.ndarray]]] = {
            group: [] for group in args.annotation_groups
        }
        montage_paths: dict[str, str] = {}

        # Ensure output directories exist.
        for group in args.annotation_groups:
            (out / f"group-{group}" / "planes").mkdir(parents=True, exist_ok=True)

        for physical in range(left_index, right_index + 1):
            z = float(context.axes[0][physical])
            t = (z - z0) / (z1 - z0)

            if physical not in (left_index, right_index):
                phi_l = left_flow.evaluate(t)
                phi_r = right_flow.evaluate(1.0 - t)

            for group in args.annotation_groups:
                endpoint_left = left_maps[group]
                endpoint_right = right_maps[group]
                group_dir = out / f"group-{group}" / "planes"

                if physical == left_index:
                    plane = endpoint_left
                    kind = "observed"

                elif physical == right_index:
                    plane = endpoint_right
                    kind = "observed"

                else:
                    vocabulary = sorted(
                        set(map(int, np.unique(endpoint_left)))
                        | set(map(int, np.unique(endpoint_right)))
                        | {0}
                    )

                    plane = dens.categorical_pair_plane(
                        endpoint_left,
                        endpoint_right,
                        vocabulary,
                        t,
                        phi_l,
                        phi_r,
                        context.registered_axes,
                        em,
                        torch,
                    )
                    kind = "inferred"

                plane = np.asarray(plane, dtype=np.uint32)

                path = group_dir / f"{physical:06d}_t-{t:.3f}_{kind}.tif"
                tifffile.imwrite(
                    path,
                    plane,
                    photometric="minisblack",
                    metadata=None,
                )

                rows_by_group[str(group)].append(
                    {
                        "group": int(group),
                        "physical_index": int(physical),
                        "allen_section_number": int(context.rows[physical]["allen_section_number"]),
                        "z_um": z,
                        "t": t,
                        "state": kind,
                        "file": str(path),
                    }
                )

                if physical in preview_indices:
                    preview_frames[group].append((physical, t, kind, plane.copy()))

        # Build one montage per group using categorical display indices.
        for group in args.annotation_groups:
            frames_group = preview_frames[group]
            if not frames_group:
                continue

            # Stable display LUT for this montage.
            display_ids = sorted(
                set().union(
                    *(
                        set(map(int, np.unique(plane)))
                        for _, _, _, plane in frames_group
                    )
                )
            )
            display_lut = {
                label_id: display_index
                for display_index, label_id in enumerate(display_ids)
            }

            fig, axes = plt.subplots(
                1,
                len(frames_group),
                figsize=(min(3.1 * len(frames_group), 18), 4.2),
                squeeze=False,
            )

            for ax, (physical, t, kind, plane) in zip(axes[0], frames_group):
                display = np.zeros_like(plane, dtype=np.int32)
                for label_id, display_index in display_lut.items():
                    display[plane == label_id] = display_index

                ax.imshow(display, interpolation="nearest", cmap="nipy_spectral")
                ax.set_title(f"{kind}\nidx {physical} | t={t:.2f}", fontsize=9)
                ax.axis("off")

            fig.suptitle(
                f"Annotation-driven diffeomorphic densification: "
                f"Allen sections {left_section} \u2192 {right_section} "
                f"(group {group})",
                fontsize=13,
            )
            fig.tight_layout()

            montage = out / f"group-{group}" / "densification_montage.png"
            fig.savefig(montage, dpi=180, bbox_inches="tight")
            plt.close(fig)

            montage_paths[str(group)] = str(montage)

        # Expose a generic rows/montage object so the downstream metadata block can use it.
        rows = rows_by_group
        montage = montage_paths

    if args.driver == "nissl":
        meta = {
            "pair": [left_index, right_index],
            "endpoint_sections": [left_section, right_section],
            "endpoint_z_um": [z0, z1],
            "canonical_missing_planes": right_index - left_index - 1,
            "nt": int(config["nt"]),
            "sampling_rule": "t=(z-z0)/(z1-z0) at canonical physical z positions",
            "continuous_demo_estimator": (
                "support-aware blend of left and right Nissl images "
                "transported by endpoint-conditioned inverse flows"
            ),
            "production_note": (
                "categorical Allen annotations use transported one-hot memberships "
                "and hardening; this RGB Nissl montage is for visual demonstration "
                "of the same fitted pair trajectories"
            ),
            "flow_report": flow_report,
            "frames": rows,
            "montage": str(montage),
        }
    else:
        meta = {
            "pair": [left_index, right_index],
            "endpoint_sections": [left_section, right_section],
            "endpoint_z_um": [z0, z1],
            "canonical_missing_planes": right_index - left_index - 1,
            "nt": int(config["nt"]),
            "sampling_rule": "t=(z-z0)/(z1-z0) at canonical physical z positions",
            "driver": "annotation",
            "driver_groups": list(map(int, args.annotation_groups)),
            "driver_representation": "binary ROI membership channels",
            "driver_channel_count": int(len(driver_keys)),
            "categorical_estimator": (
                "two-sided transported one-hot membership fusion with deterministic hardening"
            ),
            "jacobian_weighting": False,
            "flow_report": flow_report,
            "frames_by_group": rows,
            "montage_by_group": montage,
        }

        (out / "demo.json").write_text(json.dumps(meta, indent=2) + "\n")
        print(json.dumps({
            "status": "complete",
            **{k: meta[k] for k in ("pair", "canonical_missing_planes", "nt")}
        }, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
