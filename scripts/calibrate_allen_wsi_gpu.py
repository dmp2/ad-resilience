#!/usr/bin/env python3
"""Prepare cached WSI resources, run serial isolated pilots, and rebuild worklists."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

from preprocess import densify_allen_annotations as dense
from scripts import calibrate_allen_gpu_assignments as assignments


def prepare_resources(
    source_output: Path, directory: Path, wsi_repository: Path, device: str
) -> dict[str, Any]:
    """Rescore an existing preflight; only geometry metadata is read anew."""
    preflight_path = source_output / "metadata/preflight.json"
    resources = json.loads(preflight_path.read_text())
    context = dense.select_graphic_groups(
        dense.discover_inputs(
            dense.DEFAULT_DATASET,
            dense.DEFAULT_REGISTRATION,
            dense.PROJECT
            / "data/derivatives/allen/specimen_708424/annotations_symmetric_nissl_native_200um_section_aligned",
        ),
        dense.DEFAULT_ANNOTATION_GROUPS,
    )
    if resources["registered_image_shape_yx"] != list(context.shape) or resources[
        "authoritative_source_identity"
    ] != dense._authoritative_source_identity(context):
        raise RuntimeError(
            "Cached preflight geometry/source identity is stale; regenerate preflight first"
        )
    config = dense.effective_pair_config(
        dense.PROJECT / "configs/allen_dense_pairwise_wsi_seg.json", None
    )
    digest = dense._stable_json_sha256(config)
    if any(
        row["effective_solver_configuration_sha256"] != digest
        for row in resources["pairs"]
    ):
        raise RuntimeError(
            "Cached preflight solver SHA is stale; regenerate preflight first"
        )
    identity = dense.gpu_calibration_identity(
        context.shape,
        context.registered_axes,
        config,
        wsi_repository=wsi_repository,
        device=device,
    )
    old_resources = json.loads(
        (
            dense.PROJECT
            / "results/diagnostics/allen_cpu_gpu_scheduling/preflight_resources_105_pairs.json"
        ).read_text()
    )
    history = {
        row["pair_id"]: row.get("checkpoint_state", "unknown")
        for row in old_resources["pairs"]
    }
    resources.update(
        {
            "schema": "allen-wsi-calibration-resources-v1",
            "source_preflight": str(preflight_path.resolve()),
            "source_preflight_sha256": dense._sha256(preflight_path),
            "registered_axes_yx": [axis.tolist() for axis in context.registered_axes],
            "intended_corrected_profile_configuration": config,
            "intended_corrected_profile_configuration_sha256": digest,
            "calibration_identity": identity,
            "calibration_identity_sha256": dense.gpu_memory.identity_hash(identity),
        }
    )
    for row in resources["pairs"]:
        row["checkpoint_state"] = history.get(row["pair_id"], "unknown")
        row["memory_estimate"] = dense.estimate_pair_registration_memory(
            int(row["driver_channel_count"]),
            context.shape,
            context.registered_axes,
            config,
            inferred_group_planes=int(row["estimated_inferred_planes"]),
        )
        row["structural_memory_estimate_bytes"] = row["memory_estimate"][
            "structural_estimated_bytes"
        ]
        row["calibration_identity"] = identity
        row["calibration_identity_sha256"] = resources["calibration_identity_sha256"]
    dense._json(directory / "preflight_resources_105_pairs.json", resources)
    return resources


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-output", required=True, type=Path)
    parser.add_argument(
        "--directory",
        type=assignments._diagnostic_path,
        default=assignments.DEFAULT_DIRECTORY,
    )
    parser.add_argument(
        "--wsi-repository", type=Path, default=dense.DEFAULT_WSI_REPOSITORY
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--select-channels", type=float, nargs="+")
    parser.add_argument("--select-quantiles", type=float, nargs="+")
    args = parser.parse_args(argv)
    args.source_output = args.source_output.resolve()
    args.directory.mkdir(parents=True, exist_ok=True)
    resources = prepare_resources(
        args.source_output, args.directory, args.wsi_repository, args.device
    )
    reports = [
        json.loads(path.read_text())
        for path in sorted(args.directory.glob("pilot_cuda_production_*.json"))
    ]
    for report in reports:
        dense.gpu_memory.require_identity(report, resources["calibration_identity"])
        if report.get("outcome") not in {"success", "oom"}:
            raise RuntimeError(
                "Existing non-OOM pilot failure requires investigation before calibration can continue"
            )
    if args.select_channels or args.select_quantiles:
        candidates = []
        for target in args.select_channels or []:
            candidates.append(
                dense.gpu_memory.select_candidate(
                    resources["pairs"], reports, channel_count=target
                )
            )
        for target in args.select_quantiles or []:
            candidates.append(
                dense.gpu_memory.select_candidate(
                    resources["pairs"], reports, quantile=target
                )
            )
        print(
            json.dumps(
                [
                    (
                        {key: row[key] for key in ("pair_id", "driver_channel_count")}
                        if row
                        else None
                    )
                    for row in candidates
                ]
            )
        )
        return 0
    if args.prepare_only:
        return 0

    def pilot(target: float, candidates=None):
        nonlocal reports
        available = resources["pairs"] if candidates is None else candidates
        # Completed count targets need no duplicate pilot at the same score.
        if candidates is None and any(
            r["driver_channel_count"] == target for r in reports
        ):
            return
        row = dense.gpu_memory.select_candidate(
            available, reports, channel_count=target
        )
        if row is None:
            return
        pair_id = row["pair_id"]
        report_path = args.directory / f"pilot_cuda_production_{pair_id}.json"
        log_path = report_path.with_suffix(".log")
        command = [
            sys.executable,
            str(dense.PROJECT / "scripts/benchmark_allen_pair_registration.py"),
            "--pair",
            pair_id,
            "--device",
            args.device,
            "--configuration",
            "corrected-production",
            "--source-output",
            str(args.source_output),
            "--resources",
            str(args.directory / "preflight_resources_105_pairs.json"),
            "--report",
            str(report_path),
            "--wsi-repository",
            str(args.wsi_repository),
        ]
        print(
            f"Pilot {pair_id}: {row['driver_channel_count']} channels; log={log_path}",
            flush=True,
        )
        env = dict(os.environ)
        env["PYTHONPATH"] = str(dense.PROJECT / "src") + os.pathsep + str(dense.PROJECT)
        with log_path.open("w") as log:
            result = subprocess.run(
                command, env=env, stdout=log, stderr=subprocess.STDOUT
            )
        if not report_path.is_file():
            raise RuntimeError(
                f"Pilot failed before writing measurements (exit={result.returncode}); inspect {log_path}"
            )
        report = json.loads(report_path.read_text())
        dense.gpu_memory.require_identity(report, resources["calibration_identity"])
        reports.append(report)
        print(
            f"  {report['outcome']}; reserved={report.get('peak_cuda_reserved_bytes')}; elapsed={report['registration_wall_time_seconds']:.1f}s",
            flush=True,
        )
        if report["outcome"] not in {"success", "oom"} or result.returncode not in {
            0,
            2,
        }:
            raise RuntimeError(f"Non-OOM pilot failure; inspect {log_path}")

    # Legacy 94-channel failure lacks the complete identity and remains historical.
    pilot(4)
    if not any(r["outcome"] == "oom" for r in reports):
        pilot(94)
    for target in (32, 49, 65):
        pilot(target)
    if any(
        r["outcome"] == "success" and r["driver_channel_count"] == 65 for r in reports
    ):
        pilot(83)
    while True:
        successes = [r for r in reports if r["outcome"] == "success"]
        ooms = [r for r in reports if r["outcome"] == "oom"]
        if not successes or not ooms:
            break
        low = max(r["driver_channel_count"] for r in successes)
        high = min(r["driver_channel_count"] for r in ooms)
        if high - low <= 5:
            break
        between = [
            r for r in resources["pairs"] if low < r["driver_channel_count"] < high
        ]
        before = len(reports)
        pilot((low + high) / 2, between)
        if len(reports) == before:
            break
    # Use the assignment CLI so TSV/JSON artifacts follow one serialization path.
    options = [
        "--resources",
        str(args.directory / "preflight_resources_105_pairs.json"),
        "--pilot-directory",
        str(args.directory),
    ]
    for flag, filename in {
        "calibration-report": "gpu_memory_calibration.json",
        "inventory-tsv": "final_cpu_gpu_eligibility_105_pairs.tsv",
        "inventory-json": "final_cpu_gpu_eligibility_105_pairs.json",
        "remaining-tsv": "historically_unfinished_cpu_gpu_assignments.tsv",
        "remaining-json": "historically_unfinished_cpu_gpu_assignments.json",
        "gpu-worklist-tsv": "corrected_production_gpu_worklist.tsv",
        "gpu-worklist-json": "corrected_production_gpu_worklist.json",
        "cpu-worklist-tsv": "corrected_production_cpu_worklist.tsv",
        "cpu-worklist-json": "corrected_production_cpu_worklist.json",
    }.items():
        options += ["--" + flag, str(args.directory / filename)]
    assignments.main(options)
    schedule = dense.validate_corrected_production_schedule(
        args.directory / "corrected_production_gpu_worklist.tsv",
        args.directory / "corrected_production_cpu_worklist.tsv",
        args.directory / "final_cpu_gpu_eligibility_105_pairs.tsv",
        expected_identity=resources["calibration_identity"],
    )
    print(
        f"Validated {len(schedule['inventory_pair_ids'])} pairs: {len(schedule['gpu_pair_ids'])} GPU, {len(schedule['cpu_pair_ids'])} CPU, {len(schedule['known_unresolved_pair_ids'])} unresolved"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"Calibration blocked: {exc}", file=sys.stderr)
        raise SystemExit(1)
