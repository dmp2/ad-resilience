#!/usr/bin/env python3
"""Read-only CPU/CUDA registration pilot for Allen densification pairs."""

from __future__ import annotations

import argparse
import gc
import json
import resource
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import zarr

from preprocess import densify_allen_annotations as dense


DEFAULT_SOURCE_OUTPUT = dense.PROJECT / (
    "data/derivatives/allen/specimen_708424/"
    "annotations_dense_annotation_driven_200um_groups_31_265297118"
)
DEFAULT_ANNOTATIONS = dense.PROJECT / (
    "data/derivatives/allen/specimen_708424/"
    "annotations_symmetric_nissl_native_200um_section_aligned"
)
DIAGNOSTIC_ROOT = (dense.PROJECT / "results/diagnostics").resolve()
RESOURCE_PREFLIGHT = (
    DIAGNOSTIC_ROOT
    / "allen_cpu_gpu_scheduling/preflight_resources_105_pairs.json"
)


def _checkpoint_configuration(source_output: Path) -> dict[str, Any]:
    configurations: dict[str, dict[str, Any]] = {}
    for path in sorted((source_output / "metadata/pairs/tiff").glob("*.json")):
        status = json.loads(path.read_text())
        identity = status.get("checkpoint_identity", {})
        config = identity.get("solver_configuration")
        if not isinstance(config, dict):
            continue
        digest = dense._stable_json_sha256(config)
        configurations[digest] = config
    if len(configurations) != 1:
        raise RuntimeError(
            "Expected exactly one effective configuration in completed checkpoints; "
            f"found {sorted(configurations)}"
        )
    return next(iter(configurations.values()))


def _pilot_configuration(
    source_output: Path, configuration_kind: str
) -> dict[str, Any]:
    if configuration_kind == "checkpoint":
        return _checkpoint_configuration(source_output)
    preflight = json.loads(RESOURCE_PREFLIGHT.read_text())
    key = {
        "corrected-short": "corrected_short_validation_configuration",
        "corrected-production": "intended_corrected_profile_configuration",
    }[configuration_kind]
    config = preflight.get(key)
    if not isinstance(config, dict):
        raise RuntimeError(f"Resource preflight lacks {key}")
    return config


def _pair_groups(source_output: Path, pair: tuple[int, int]) -> tuple[int, ...]:
    for row in dense._read_tsv(source_output / "metadata/endpoint_pairs.tsv"):
        if row["pair_id"] == dense._pair_id(pair):
            return tuple(map(int, json.loads(row["graphic_groups"])))
    raise RuntimeError(f"Pair {dense._pair_id(pair)} is absent from endpoint_pairs.tsv")


def _diagnostic_path(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_relative_to(DIAGNOSTIC_ROOT):
        raise argparse.ArgumentTypeError(
            f"pilot reports must be inside {DIAGNOSTIC_ROOT}"
        )
    return path


def run_pilot(
    pair: tuple[int, int],
    device: str,
    source_output: Path,
    report_path: Path,
    *,
    dataset: Path,
    annotations: Path,
    registration: Path,
    wsi_repository: Path,
    configuration_kind: str,
) -> dict[str, Any]:
    source_output = source_output.expanduser().resolve()
    context = dense.select_graphic_groups(
        dense.discover_inputs(dataset, registration, annotations),
        dense.DEFAULT_ANNOTATION_GROUPS,
    )
    groups = _pair_groups(source_output, pair)
    config = _pilot_configuration(source_output, configuration_kind)
    anchor_root = zarr.open_group(str(source_output / "anchors.zarr"), mode="r")
    ordinal_by_physical = {
        context.physical_by_section[section]: ordinal
        for ordinal, section in enumerate(context.annotation_sections)
    }
    left_ordinal = ordinal_by_physical[pair[0]]
    right_ordinal = ordinal_by_physical[pair[1]]
    (
        left_image,
        right_image,
        left_weight,
        right_weight,
        driver_keys,
        driver_report,
    ) = dense.build_annotation_pair_driver(
        anchor_root, left_ordinal, right_ordinal, groups
    )
    estimate = dense.estimate_pair_registration_memory(
        len(driver_keys),
        context.shape,
        context.registered_axes,
        config,
        inferred_group_planes=(pair[1] - pair[0] - 1) * len(groups),
    )
    pilot: dict[str, Any] = {
        "schema": "allen-registration-pilot-v1",
        "pair": list(pair),
        "pair_id": dense._pair_id(pair),
        "device": device,
        "configuration_kind": configuration_kind,
        "driver_channel_count": len(driver_keys),
        "image_shape_yx": list(context.shape),
        "pair_driver_groups": list(groups),
        "driver_report": driver_report,
        "effective_solver_configuration": config,
        "effective_solver_configuration_sha256": dense._stable_json_sha256(config),
        "wsi_commit": dense._verify_wsi_repository(wsi_repository),
        "memory_estimate": estimate,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "numerically_valid": False,
    }
    started = time.monotonic()
    start_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    torch = None
    try:
        if device.startswith("cuda"):
            _, _, torch, _, _, _ = dense._load_wsi(wsi_repository)
            torch.cuda.set_device(device)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            free_bytes, total_bytes = torch.cuda.mem_get_info(device)
            pilot["cuda_total_bytes"] = int(total_bytes)
            pilot["cuda_free_before_bytes"] = int(free_bytes)
        left_flow, right_flow, map_report, _, torch = dense.fit_pair_trajectories(
            left_image,
            right_image,
            left_weight,
            right_weight,
            context.registered_axes,
            config,
            wsi_repository=wsi_repository,
            device=device,
        )
        midpoint_left = left_flow.evaluate(0.5)
        midpoint_right = right_flow.evaluate(0.5)
        if not (
            np.all(np.isfinite(midpoint_left))
            and np.all(np.isfinite(midpoint_right))
        ):
            raise RuntimeError("Pilot midpoint maps contain non-finite values")
        pilot["map_convention"] = map_report
        pilot["numerically_valid"] = True
    except Exception as exc:
        pilot["error_type"] = type(exc).__name__
        pilot["error"] = str(exc)
        raise
    finally:
        pilot["registration_wall_time_seconds"] = time.monotonic() - started
        pilot["peak_host_rss_kib"] = max(
            start_rss, int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        )
        if torch is not None and device.startswith("cuda"):
            pilot["peak_cuda_allocated_bytes"] = int(
                torch.cuda.max_memory_allocated(device)
            )
            pilot["peak_cuda_reserved_bytes"] = int(
                torch.cuda.max_memory_reserved(device)
            )
            free_bytes, _ = torch.cuda.mem_get_info(device)
            pilot["cuda_free_after_bytes"] = int(free_bytes)
        pilot["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        dense._json(report_path, pilot)
        gc.collect()
    return pilot


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", required=True, type=dense._parse_pair)
    parser.add_argument("--device", required=True)
    parser.add_argument("--source-output", type=Path, default=DEFAULT_SOURCE_OUTPUT)
    parser.add_argument("--report", required=True, type=_diagnostic_path)
    parser.add_argument(
        "--configuration",
        choices=("corrected-short", "corrected-production", "checkpoint"),
        default="corrected-short",
    )
    parser.add_argument("--dataset", type=Path, default=dense.DEFAULT_DATASET)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--registration-run", type=Path, default=dense.DEFAULT_REGISTRATION)
    parser.add_argument("--wsi-repository", type=Path, default=dense.DEFAULT_WSI_REPOSITORY)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_pilot(
        args.pair,
        args.device,
        args.source_output,
        args.report,
        dataset=args.dataset,
        annotations=args.annotations,
        registration=args.registration_run,
        wsi_repository=args.wsi_repository,
        configuration_kind=args.configuration,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
