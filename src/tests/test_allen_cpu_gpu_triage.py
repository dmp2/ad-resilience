from __future__ import annotations

from scripts import calibrate_allen_gpu_assignments as triage


def _identity():
    memory = triage.dense.gpu_memory
    identity = {key: "corrected" for key in memory.IDENTITY_FIELDS}
    identity.update(
        mapping_backend=memory.BACKEND,
        structural_model=memory.MODEL,
        upsample_mode="seg",
        image_shape_yx=[10, 20],
        cuda_total_bytes=1000,
    )
    return {
        "calibration_identity": identity,
        "calibration_identity_sha256": memory.identity_hash(identity),
    }


def _resource(pair_id: str, status: str, structural: int) -> dict:
    return {
        **_identity(),
        "pair_id": pair_id,
        "checkpoint_state": status,
        "driver_channel_count": structural // 10,
        "left_support_pixels": 100,
        "right_support_pixels": 200,
        "structural_memory_estimate_bytes": structural,
        "memory_estimate": {
            "model": triage.dense.gpu_memory.MODEL,
            "structural_estimated_bytes": structural,
        },
    }


def _pilot(pair_id: str, device: str, valid: bool, structural: int = 900) -> dict:
    report = {
        **_identity(),
        "pair_id": pair_id,
        "device": device,
        "configuration_kind": "corrected-production",
        "numerically_valid": valid,
        "outcome": "success" if valid else "other-failure",
    }
    if device == "cuda:0":
        report["peak_cuda_reserved_bytes"] = 700
        report["memory_estimate"] = {"structural_estimated_bytes": structural}
    if not valid:
        report["error_type"] = "_LinAlgError"
    return report


def test_triage_keeps_failures_unresolved_and_history_separate(monkeypatch):
    calibration = {
        **_identity(),
        "effective_solver_configuration_sha256": "corrected",
        "intercept_bytes": 0.0,
        "structural_score_coefficient": 1.0,
        "positive_residual_margin_bytes": 0.0,
        "prediction_safety_factor": 1.15,
        "gpu_admission_budget_bytes": 800,
        "smallest_successful_calibration_score": 600,
        "largest_successful_calibration_score": 800,
        "lowest_observed_oom_score": None,
    }
    monkeypatch.setattr(
        triage.dense,
        "fit_gpu_memory_calibration",
        lambda reports: dict(calibration),
    )
    resources = {
        **_identity(),
        "intended_corrected_profile_configuration_sha256": "corrected",
        "pairs": [
            _resource("complete", "complete", 100),
            _resource("gpu", "remaining", 600),
            _resource("memory", "remaining", 800),
            _resource("failed", "remaining", 900),
        ],
    }
    gpu_pilots = [
        _pilot("gpu", "cuda:0", True, 600),
        _pilot("memory", "cuda:0", True, 800),
        _pilot("failed", "cuda:0", False, 900),
    ]
    cpu_pilots = [_pilot("failed", "cpu", False)]

    summary, rows = triage.build_assignments(resources, gpu_pilots, cpu_pilots)
    by_pair = {row["pair_id"]: row for row in rows}

    assert by_pair["complete"]["historically_completed"] is True
    assert by_pair["complete"]["corrected_run_requires_regeneration"] is True
    assert by_pair["complete"]["eligibility_category"] == "GPU eligible"
    assert by_pair["complete"]["final_assigned_device"] == "gpu"
    assert by_pair["gpu"]["eligibility_category"] == "GPU eligible"
    assert by_pair["gpu"]["final_assigned_device"] == "gpu"
    assert by_pair["memory"]["eligibility_category"] == "CPU assigned — memory"
    assert by_pair["memory"]["final_assigned_device"] == "cpu"
    assert by_pair["failed"]["eligibility_category"] == "Unresolved"
    assert by_pair["failed"]["final_assigned_device"] == "unresolved"
    assert summary["historically_complete_pair_count"] == 1
    assert summary["total_corrected_run_regeneration_pair_count"] == 4
    assert summary["gpu_assigned_pair_count"] == 2
    assert summary["cpu_assigned_pair_count"] == 1
    assert summary["unresolved_pair_count"] == 1
    assert summary["calibration_extrapolated_pair_ids"] == ["complete", "failed"]
