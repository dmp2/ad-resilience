#!/usr/bin/env python3
"""Apply calibrated CUDA admission and classify all Allen endpoint pairs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Sequence

from preprocess import densify_allen_annotations as dense


DIAGNOSTIC_ROOT = (dense.PROJECT / "results/diagnostics").resolve()
DEFAULT_DIRECTORY = DIAGNOSTIC_ROOT / "allen_cpu_gpu_scheduling"


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
    calibration = dense.fit_gpu_memory_calibration(gpu_pilot_reports)
    expected_configuration = resources[
        "intended_corrected_profile_configuration_sha256"
    ]
    if calibration["effective_solver_configuration_sha256"] != expected_configuration:
        raise RuntimeError("Pilot calibration and resource table configurations differ")
    gpu_reports = {
        str(report["pair_id"]): report
        for report in gpu_pilot_reports
        if report.get("configuration_kind") == "corrected-production"
    }
    cpu_reports = {
        str(report["pair_id"]): report
        for report in cpu_pilot_reports
        if report.get("configuration_kind") == "corrected-production"
    }
    budget = int(calibration["gpu_admission_budget_bytes"])
    safety_factor = float(calibration["prediction_safety_factor"])
    calibration_reports = [
        report
        for report in gpu_pilot_reports
        if report.get("configuration_kind") == "corrected-production"
        and report.get("numerically_valid")
    ]
    calibration_scores = [
        int(report["memory_estimate"]["structural_estimated_bytes"])
        for report in calibration_reports
    ]
    measured_min = min(calibration_scores)
    measured_max = max(calibration_scores)

    def as_gib(value: int | None) -> float | None:
        return None if value is None else value / 1024**3

    rows = []
    for resource in resources["pairs"]:
        structural = int(resource["structural_memory_estimate_bytes"])
        predicted = int(
            math.ceil(
                float(calibration["intercept_bytes"])
                + float(calibration["structural_score_coefficient"])
                * structural
                + float(calibration["positive_residual_margin_bytes"])
            )
        )
        admission = dense.calibrated_gpu_memory_bytes(structural, calibration)
        pair_id = str(resource["pair_id"])
        exceeds_budget = admission > budget
        if structural < measured_min:
            calibration_range = "below measured range"
        elif structural > measured_max:
            calibration_range = "above measured range"
        else:
            calibration_range = "within measured range"
        historical_status = str(resource["checkpoint_state"])
        gpu_report = gpu_reports.get(pair_id)
        cpu_report = cpu_reports.get(pair_id)
        observed_gpu_peak = (
            int(gpu_report["peak_cuda_reserved_bytes"])
            if gpu_report is not None
            and gpu_report.get("peak_cuda_reserved_bytes") is not None
            else None
        )
        if gpu_report is not None and not gpu_report.get("numerically_valid"):
            if cpu_report is not None and cpu_report.get("numerically_valid"):
                category = "CPU assigned — demonstrated GPU failure"
                device = "cpu"
                reason = (
                    "corrected-production GPU pilot failed numerically and the "
                    "equivalent CPU pilot succeeded"
                )
            else:
                category = "Unresolved"
                device = "unresolved"
                cpu_detail = (
                    str(cpu_report.get("error_type", "unknown CPU failure"))
                    if cpu_report is not None
                    else "no equivalent CPU pilot"
                )
                reason = (
                    "corrected-production GPU pilot failed numerically; CPU path "
                    f"not demonstrated suitable ({cpu_detail})"
                )
        elif not exceeds_budget:
            category = "GPU eligible"
            device = "gpu"
            reason = (
                "safety-adjusted admission estimate is within budget and no GPU "
                "numerical failure was observed"
            )
        else:
            category = "CPU assigned — memory"
            device = "cpu"
            reason = (
                "GPU registration succeeded in isolation"
                if gpu_report is not None and gpu_report.get("numerically_valid")
                else "no GPU numerical failure was observed"
            )
            reason += "; safety-adjusted admission estimate exceeds budget"
        rows.append(
            {
                "pair_id": pair_id,
                "historical_checkpoint_status": historical_status,
                "historically_completed": historical_status == "complete",
                "corrected_run_requires_regeneration": True,
                "driver_channel_count": int(resource["driver_channel_count"]),
                "left_support_pixels": int(resource["left_support_pixels"]),
                "right_support_pixels": int(resource["right_support_pixels"]),
                "structural_memory_estimate_bytes": structural,
                "predicted_reserved_vram_bytes": predicted,
                "predicted_reserved_vram_gib": as_gib(predicted),
                "admission_safety_factor": safety_factor,
                "safety_adjusted_admission_bytes": admission,
                "safety_adjusted_admission_gib": as_gib(admission),
                "gpu_admission_budget_bytes": budget,
                "gpu_admission_budget_gib": as_gib(budget),
                "exceeds_gpu_admission_budget": exceeds_budget,
                "calibration_range_status": calibration_range,
                "observed_peak_gpu_reserved_bytes": observed_gpu_peak,
                "observed_peak_gpu_reserved_gib": as_gib(observed_gpu_peak),
                "observed_gpu_numerically_valid": (
                    gpu_report.get("numerically_valid") if gpu_report else None
                ),
                "observed_gpu_error_type": (
                    gpu_report.get("error_type") if gpu_report else None
                ),
                "observed_cpu_numerically_valid": (
                    cpu_report.get("numerically_valid") if cpu_report else None
                ),
                "observed_cpu_error_type": (
                    cpu_report.get("error_type") if cpu_report else None
                ),
                "eligibility_category": category,
                "final_assigned_device": device,
                "assignment_reason": reason,
            }
        )
    calibration.update(
        {
            "schema": "allen-gpu-memory-calibration-v3",
            "inventory_pair_count": len(rows),
            "total_corrected_run_regeneration_pair_count": len(rows),
            "historically_complete_pair_count": sum(
                row["historically_completed"] for row in rows
            ),
            "historically_unfinished_pair_count": sum(
                not row["historically_completed"] for row in rows
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
            "gpu_budget_exceeding_pair_count": sum(
                row["exceeds_gpu_admission_budget"] for row in rows
            ),
            "gpu_budget_exceeding_pair_ids": [
                row["pair_id"]
                for row in rows
                if row["exceeds_gpu_admission_budget"]
            ],
            "calibration_extrapolated_pair_count": sum(
                row["calibration_range_status"] != "within measured range"
                for row in rows
            ),
            "calibration_extrapolated_pair_ids": [
                row["pair_id"]
                for row in rows
                if row["calibration_range_status"] != "within measured range"
            ],
            "measured_calibration_structural_min_bytes": measured_min,
            "measured_calibration_structural_max_bytes": measured_max,
            "known_invalid_cuda_pairs": {
                pair_id: str(report.get("error_type", "invalid pilot"))
                for pair_id, report in gpu_reports.items()
                if not report.get("numerically_valid")
            },
            "known_invalid_cpu_pairs": {
                pair_id: str(report.get("error_type", "invalid pilot"))
                for pair_id, report in cpu_reports.items()
                if not report.get("numerically_valid")
            },
        }
    )
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
        default=DEFAULT_DIRECTORY / "remaining_46_cpu_gpu_assignments.tsv",
    )
    parser.add_argument(
        "--remaining-json",
        type=_diagnostic_path,
        default=DEFAULT_DIRECTORY / "remaining_46_cpu_gpu_assignments.json",
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
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0]), delimiter="\t")
            writer.writeheader()
            writer.writerows(values)

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
            "schema": "allen-corrected-production-gpu-worklist-v1",
            "pair_count": len(gpu_worklist),
            "pairs": gpu_worklist,
        },
    )
    dense._json(
        args.cpu_worklist_json,
        {
            "schema": "allen-corrected-production-cpu-worklist-v1",
            "pair_count": len(cpu_worklist),
            "pairs": cpu_worklist,
        },
    )
    print(json.dumps(calibration, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
