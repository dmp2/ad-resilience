#!/usr/bin/env python3
"""Apply calibrated CUDA admission and classify all Allen endpoint pairs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence

from preprocess import densify_allen_annotations as dense

DIAGNOSTIC_ROOT = (dense.PROJECT / "results/diagnostics").resolve()
DEFAULT_DIRECTORY = DIAGNOSTIC_ROOT / "allen_wsi_gpu_calibration"


def _diagnostic_path(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_relative_to(DIAGNOSTIC_ROOT):
        raise argparse.ArgumentTypeError(
            f"calibration artifacts must be inside {DIAGNOSTIC_ROOT}"
        )
    return path


def build_assignments(
    resources: dict[str, Any],
    gpu_pilot_reports: Sequence[dict[str, Any]],
    cpu_pilot_reports: Sequence[dict[str, Any]] = (),
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    identity = dense.gpu_memory.require_identity(resources)
    if not gpu_pilot_reports:
        raise RuntimeError(
            "No CUDA memory calibration exists for "
            + json.dumps(identity, sort_keys=True)
            + ". Run GPU calibration pilots before generating the production worklist."
        )
    for report in [*gpu_pilot_reports, *cpu_pilot_reports]:
        dense.gpu_memory.require_identity(report, identity)
    calibration = dense.fit_gpu_memory_calibration(gpu_pilot_reports)
    dense.gpu_memory.require_identity(calibration, identity)
    gpu_reports = {str(report["pair_id"]): report for report in gpu_pilot_reports}
    cpu_reports = {str(report["pair_id"]): report for report in cpu_pilot_reports}
    rows = []
    seen = set()
    for resource in resources["pairs"]:
        dense.gpu_memory.require_identity(resource, identity)
        pair_id = str(resource["pair_id"])
        if pair_id in seen:
            raise RuntimeError("Duplicate resource pair")
        seen.add(pair_id)
        estimate = resource["memory_estimate"]
        if estimate["model"] != dense.gpu_memory.MODEL:
            raise RuntimeError("Resource structural model is stale")
        structural = int(estimate["structural_estimated_bytes"])
        reason = dense.gpu_memory.exclusion_reason(structural, calibration)
        gpu_report, cpu_report = gpu_reports.get(pair_id), cpu_reports.get(pair_id)
        if gpu_report and gpu_report.get("outcome") == "other-failure":
            if (
                cpu_report
                and cpu_report.get("outcome") == "success"
                and cpu_report.get("numerically_valid")
            ):
                device, category = "cpu", "CPU assigned — demonstrated GPU failure"
                reason = "GPU numerical failure; matching CPU pilot succeeded"
            else:
                device, category = "unresolved", "Unresolved"
                reason = "GPU non-OOM failure; no matching successful CPU evidence"
        elif gpu_report and gpu_report.get("outcome") == "oom":
            device, category = "cpu", "CPU assigned — memory"
            reason = "matching production CUDA pilot OOM"
        elif reason:
            device, category = "cpu", "CPU assigned — memory"
        elif int(resource["driver_channel_count"]) == 0:
            device, category, reason = (
                "cpu",
                "CPU assigned — no channels",
                "zero usable driver channels",
            )
        else:
            device, category, reason = (
                "gpu",
                "GPU eligible",
                "within budget, measured coverage, and matching OOM bound",
            )
        predicted = dense.gpu_memory.predicted_bytes(
            structural, calibration, safety_adjusted=False
        )
        admission = dense.calibrated_gpu_memory_bytes(structural, calibration)
        low, high = (
            calibration["smallest_successful_calibration_score"],
            calibration["largest_successful_calibration_score"],
        )
        status = (
            "below measured range"
            if structural < low
            else (
                "above measured range" if structural > high else "within measured range"
            )
        )
        historical = resource.get("checkpoint_state", "unknown")
        rows.append(
            {
                "pair_id": pair_id,
                "historical_checkpoint_status": historical,
                "historically_completed": historical == "complete",
                "corrected_run_requires_regeneration": True,
                "driver_channel_count": int(resource["driver_channel_count"]),
                "left_support_pixels": resource.get("left_support_pixels"),
                "right_support_pixels": resource.get("right_support_pixels"),
                "structural_memory_estimate_bytes": structural,
                "structural_score": structural,
                "predicted_reserved_vram_bytes": predicted,
                "predicted_reserved_vram_gib": predicted / 1024**3,
                "admission_safety_factor": calibration["prediction_safety_factor"],
                "safety_adjusted_admission_bytes": admission,
                "safety_adjusted_admission_gib": admission / 1024**3,
                "gpu_admission_budget_bytes": calibration["gpu_admission_budget_bytes"],
                "gpu_admission_budget_gib": calibration["gpu_admission_budget_bytes"]
                / 1024**3,
                "exceeds_gpu_admission_budget": admission
                > calibration["gpu_admission_budget_bytes"],
                "calibration_range_status": status,
                "measured_coverage": structural <= high,
                "largest_successful_calibration_score": high,
                "lowest_observed_oom_score": calibration["lowest_observed_oom_score"],
                "observed_peak_gpu_reserved_bytes": (
                    gpu_report.get("peak_cuda_reserved_bytes") if gpu_report else None
                ),
                "observed_gpu_numerically_valid": (
                    gpu_report.get("numerically_valid") if gpu_report else None
                ),
                "observed_gpu_error_type": (
                    gpu_report.get("error_type") if gpu_report else None
                ),
                "observed_cpu_numerically_valid": (
                    cpu_report.get("numerically_valid") if cpu_report else None
                ),
                "eligibility_category": category,
                "final_assigned_device": device,
                "assignment_reason": reason,
                "calibration_identity": identity,
                "calibration_identity_sha256": dense.gpu_memory.identity_hash(identity),
            }
        )
    calibration.update(
        {
            "inventory_pair_count": len(rows),
            "total_corrected_run_regeneration_pair_count": len(rows),
            "historically_complete_pair_count": sum(
                row["historically_completed"] for row in rows
            ),
            "gpu_assigned_pair_count": sum(
                row["final_assigned_device"] == "gpu" for row in rows
            ),
            "cpu_assigned_pair_count": sum(
                row["final_assigned_device"] == "cpu" for row in rows
            ),
            "unresolved_pair_count": sum(
                row["final_assigned_device"] == "unresolved" for row in rows
            ),
            "calibration_extrapolated_pair_ids": [
                row["pair_id"]
                for row in rows
                if row["calibration_range_status"] != "within measured range"
            ],
            "known_invalid_cuda_pairs": {
                key: report.get("error_type")
                for key, report in gpu_reports.items()
                if report.get("outcome") == "other-failure"
            },
        }
    )
    for row in rows:
        row["calibration_sha256"] = dense._stable_json_sha256(calibration)
    return calibration, rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resources",
        type=_diagnostic_path,
        default=DEFAULT_DIRECTORY / "preflight_resources_105_pairs.json",
    )
    parser.add_argument(
        "--pilot-directory", type=_diagnostic_path, default=DEFAULT_DIRECTORY
    )
    parser.add_argument(
        "--calibration-report",
        type=_diagnostic_path,
        default=DEFAULT_DIRECTORY / "gpu_memory_calibration.json",
    )
    parser.add_argument(
        "--inventory-tsv",
        type=_diagnostic_path,
        default=DEFAULT_DIRECTORY / "final_cpu_gpu_eligibility_105_pairs.tsv",
    )
    parser.add_argument(
        "--inventory-json",
        type=_diagnostic_path,
        default=DEFAULT_DIRECTORY / "final_cpu_gpu_eligibility_105_pairs.json",
    )
    parser.add_argument(
        "--remaining-tsv",
        type=_diagnostic_path,
        default=DEFAULT_DIRECTORY / "historically_unfinished_cpu_gpu_assignments.tsv",
    )
    parser.add_argument(
        "--remaining-json",
        type=_diagnostic_path,
        default=DEFAULT_DIRECTORY / "historically_unfinished_cpu_gpu_assignments.json",
    )
    parser.add_argument(
        "--gpu-worklist-tsv",
        type=_diagnostic_path,
        default=DEFAULT_DIRECTORY / "corrected_production_gpu_worklist.tsv",
    )
    parser.add_argument(
        "--gpu-worklist-json",
        type=_diagnostic_path,
        default=DEFAULT_DIRECTORY / "corrected_production_gpu_worklist.json",
    )
    parser.add_argument(
        "--cpu-worklist-tsv",
        type=_diagnostic_path,
        default=DEFAULT_DIRECTORY / "corrected_production_cpu_worklist.tsv",
    )
    parser.add_argument(
        "--cpu-worklist-json",
        type=_diagnostic_path,
        default=DEFAULT_DIRECTORY / "corrected_production_cpu_worklist.json",
    )
    args = parser.parse_args(argv)
    resources = json.loads(args.resources.read_text())
    gpu_pilots = [
        json.loads(path.read_text())
        for path in sorted(args.pilot_directory.glob("pilot_cuda_production_*.json"))
    ]
    cpu_pilots = [
        json.loads(path.read_text())
        for path in sorted(args.pilot_directory.glob("pilot_cpu_production_*.json"))
    ]
    calibration, rows = build_assignments(resources, gpu_pilots, cpu_pilots)
    dense._json(args.calibration_report, calibration)
    args.inventory_tsv.parent.mkdir(parents=True, exist_ok=True)

    def write_tsv(path: Path, values: Sequence[dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t")
            writer.writeheader()
            writer.writerows(
                {
                    key: (
                        json.dumps(value, sort_keys=True, separators=(",", ":"))
                        if isinstance(value, (dict, list))
                        else value
                    )
                    for key, value in row.items()
                }
                for row in values
            )

    remaining = [row for row in rows if not row["historically_completed"]]
    gpu_worklist = [row for row in rows if row["final_assigned_device"] == "gpu"]
    cpu_worklist = [row for row in rows if row["final_assigned_device"] == "cpu"]
    write_tsv(args.inventory_tsv, rows)
    write_tsv(args.remaining_tsv, remaining)
    write_tsv(args.gpu_worklist_tsv, gpu_worklist)
    write_tsv(args.cpu_worklist_tsv, cpu_worklist)
    dense._json(
        args.inventory_json,
        {
            "schema": "allen-cpu-gpu-eligibility-inventory-v2",
            "calibration": calibration,
            "pairs": rows,
        },
    )
    dense._json(
        args.remaining_json,
        {
            "schema": "allen-cpu-gpu-remaining-assignments-v1",
            "calibration": calibration,
            "pairs": remaining,
        },
    )
    dense._json(
        args.gpu_worklist_json,
        {
            "schema": "allen-wsi-production-gpu-worklist-v2",
            "calibration_identity": calibration["calibration_identity"],
            "calibration_identity_sha256": calibration["calibration_identity_sha256"],
            "pair_count": len(gpu_worklist),
            "pairs": gpu_worklist,
        },
    )
    dense._json(
        args.cpu_worklist_json,
        {
            "schema": "allen-wsi-production-cpu-worklist-v2",
            "calibration_identity": calibration["calibration_identity"],
            "calibration_identity_sha256": calibration["calibration_identity_sha256"],
            "pair_count": len(cpu_worklist),
            "pairs": cpu_worklist,
        },
    )
    print(
        json.dumps(
            {
                key: value
                for key, value in calibration.items()
                if key != "calibration_identity"
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
