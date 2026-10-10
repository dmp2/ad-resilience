"""Backend-specific structural scores and measured CUDA admission.

Scores use byte-scaled allocation proxies; they are not peak-memory claims.
Only matching, instrumented production pilots establish GPU eligibility.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np

MODEL = "wsi-upsample-seg-v1"
BACKEND = "wsi_pipeline.registration.upsample.upsample_between_slices"
IDENTITY_FIELDS = {
    "wsi_commit",
    "emlddmm_commit",
    "backend_source_sha256",
    "mapping_backend",
    "upsample_mode",
    "effective_solver_configuration_sha256",
    "image_shape_yx",
    "registered_geometry",
    "dtype",
    "structural_model",
    "gpu_model",
    "cuda_total_bytes",
}


def identity_hash(identity: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def validate_identity(identity: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(identity, Mapping) or not IDENTITY_FIELDS <= identity.keys():
        raise RuntimeError(
            "No current CUDA memory calibration identity exists. Run GPU calibration pilots before generating the production worklist."
        )
    if any(identity[key] is None for key in IDENTITY_FIELDS):
        raise RuntimeError(
            "Incomplete CUDA memory calibration identity. Run instrumented GPU calibration pilots."
        )
    if (
        identity["structural_model"] != MODEL
        or identity["mapping_backend"] != BACKEND
        or identity["upsample_mode"] != "seg"
    ):
        raise RuntimeError(
            "Stale CUDA backend/model identity. Run GPU calibration pilots for wsi-upsample-seg-v1."
        )
    return dict(identity)


def require_identity(
    record: Mapping[str, Any], expected: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    identity = validate_identity(record.get("calibration_identity"))
    if record.get("calibration_identity_sha256") != identity_hash(identity):
        raise RuntimeError("Missing or corrupt CUDA calibration identity hash")
    if expected is not None and identity != dict(expected):
        raise RuntimeError(
            "Stale CUDA memory calibration; expected "
            + json.dumps(expected, sort_keys=True)
            + ". Run GPU calibration pilots before generating the production worklist."
        )
    return identity


def estimate(
    channel_count: int,
    image_shape: Sequence[int],
    registered_axes: Sequence[np.ndarray],
    config: Mapping[str, Any],
    *,
    inferred_group_planes: int = 0,
) -> dict[str, Any]:
    if channel_count < 0:
        raise ValueError("channel_count must be nonnegative")
    h, w = map(int, image_shape)
    if (h, w) != tuple(len(axis) for axis in registered_axes):
        raise ValueError("Registered axes and image shape differ")
    dtype = np.dtype(config.get("dtype", "float32"))
    if dtype.name not in {"float32", "float64"}:
        raise ValueError("Structural model requires float32 or float64")
    b, nt, pixels = dtype.itemsize, int(config["nt"]), h * w
    steps = [float(np.mean(np.abs(np.diff(axis)))) for axis in registered_axes]
    spacing_z = float(np.mean(steps))
    # Pinned symmetric adapter: dv_z = mean image spacing, >=4 intervals,
    # odd synthetic depth => five planes. This is independent of physical gap.
    depth = max(5, math.ceil(4 * spacing_z / spacing_z) + 1)
    if depth % 2 == 0:
        depth += 1
    dv = config.get("dv", [spacing_z])
    while isinstance(dv, (list, tuple)) and len(dv) == 1:
        dv = dv[0]
    if isinstance(dv, (list, tuple)) and isinstance(dv[-1], (list, tuple)):
        dv = dv[-1]
    if np.isscalar(dv):
        dv = [float(dv), float(dv)]
    # The WSI adapter discards a supplied synthetic-z spacing.
    dv = [spacing_z, *list(map(float, dv))[-2:]]
    extents = [
        (depth - 1) * spacing_z,
        *[float(np.ptp(axis)) for axis in registered_axes],
    ]
    expansion = 1 + 2 * float(config.get("v_expand_factor", 0.2))
    grid = [
        max(1, math.ceil(expansion * extent / step))
        for extent, step in zip(extents, dv, strict=True)
    ]
    image = channel_count * pixels * b
    slab = depth * image
    velocity = nt * 3 * math.prod(grid) * b
    components = {
        "endpoint_driver_score_bytes": 2 * image,
        "synthetic_endpoint_score_bytes": 2 * slab,
        "active_inner_optimization_allowance_score_bytes": 12 * slab,
        "channel_trajectory_score_bytes": 2 * (nt + 1) * image,
        "transform_jacobian_score_bytes": 8 * (nt + 1) * pixels * b,
        "velocity_fft_gradient_score_bytes": 6 * velocity,
        # emlddmm returns v/A/weights on CPU but leaves xv on its device.
        # First outer result retains two inner coordinate sets while the second
        # outer call retains its first inner result during the backward solve.
        "nested_retained_cuda_axes_score_bytes": 3 * sum(grid) * b,
        "fixed_workspace_score_bytes": 64 * 1024**2,
    }
    iterations = config.get("n_iter", [1])
    if np.isscalar(iterations):
        iterations = [iterations]
    return {
        "model": MODEL,
        "interpretation": "byte-scaled structural score; calibrate against peak reserved VRAM",
        "scalar_bytes": b,
        "image_shape_yx": [h, w],
        "synthetic_backend_z_planes": depth,
        "velocity_grid_shape_zyx": grid,
        "components": components,
        "structural_estimated_bytes": sum(components.values()),
        "structural_score": sum(components.values()),
        "retention": {
            "outer_symmetric_calls": 2,
            "inner_optimizations_per_outer_call": 2,
            "first_outer_result_retained_during_second": True,
            "returned_trajectories_device": "cpu",
            "returned_inner_velocity_weights_device": "cpu",
            "returned_inner_axes_device": "solver device",
        },
        "host_output_score_bytes": 2 * (inferred_group_planes + 2) * image,
        "host_retained_outer_trajectory_score_bytes": 4 * (nt + 1) * image,
        "relative_runtime_work_units": int(
            (4 * sum(map(int, iterations)) + 2 * inferred_group_planes)
            * channel_count
            * pixels
        ),
        "calibration_required_for_gpu_eligibility": True,
    }


def fit(
    pilot_reports: Sequence[Mapping[str, Any]],
    *,
    safety_factor: float = 1.15,
    free_memory_reserve_fraction: float = 0.15,
    minimum_free_memory_reserve_bytes: int = 1024**3,
) -> dict[str, Any]:
    if safety_factor < 1 or not 0 <= free_memory_reserve_fraction < 1:
        raise ValueError("Invalid safety factor or reserve fraction")
    reports = list(pilot_reports)
    if not reports:
        raise RuntimeError(
            "No CUDA memory calibration exists. Run GPU calibration pilots before generating the production worklist."
        )
    expected = require_identity(reports[0])
    for report in reports:
        require_identity(report, expected)
        if (
            not str(report.get("device", "")).startswith("cuda")
            or report.get("configuration_kind") != "corrected-production"
        ):
            raise RuntimeError("Calibration requires production CUDA pilots")
        if report.get("memory_estimate", {}).get("model") != MODEL:
            raise RuntimeError("Pilot structural model is stale")
        if (
            report.get("effective_solver_configuration_sha256")
            != expected["effective_solver_configuration_sha256"]
            or report.get("image_shape_yx") != expected["image_shape_yx"]
            or report.get("cuda_total_bytes") != expected["cuda_total_bytes"]
        ):
            raise RuntimeError("Pilot measurements disagree with calibration identity")
    valid = [
        r
        for r in reports
        if r.get("outcome") == "success" and r.get("numerically_valid")
    ]
    x = np.asarray(
        [float(r["memory_estimate"]["structural_estimated_bytes"]) for r in valid]
    )
    y = np.asarray([float(r["peak_cuda_reserved_bytes"]) for r in valid])
    if len(set(x)) < 3:
        raise RuntimeError(
            "At least three successful production CUDA pilots with distinct structural scores are required"
        )
    slope = float(np.sum((x - x.mean()) * (y - y.mean())) / np.sum((x - x.mean()) ** 2))
    intercept = float(y.mean() - slope * x.mean())
    if slope <= 0 or intercept < 0:
        raise RuntimeError("Pilot memory fit is not physically usable")
    residual = max(0.0, float(np.max(y - (intercept + slope * x))))
    total = int(expected["cuda_total_bytes"])
    reserve = max(
        minimum_free_memory_reserve_bytes,
        math.ceil(total * free_memory_reserve_fraction),
    )
    free = min(int(r["cuda_free_before_bytes"]) for r in valid)
    if free <= reserve:
        raise RuntimeError("CUDA availability reserve consumes observed free memory")
    ooms = [r for r in reports if r.get("outcome") == "oom"]
    return {
        "model": "linear-reserved-vram-v1",
        "schema": "allen-gpu-memory-calibration-v4",
        "calibration_identity": expected,
        "calibration_identity_sha256": identity_hash(expected),
        "structural_score_coefficient": slope,
        "intercept_bytes": intercept,
        "positive_residual_margin_bytes": residual,
        "prediction_safety_factor": safety_factor,
        "cuda_total_bytes": total,
        "minimum_observed_free_bytes": free,
        "availability_reserve_bytes": reserve,
        "gpu_admission_budget_bytes": free - reserve,
        "effective_solver_configuration_sha256": expected[
            "effective_solver_configuration_sha256"
        ],
        "image_shape_yx": expected["image_shape_yx"],
        "calibration_pair_ids": [r["pair_id"] for r in valid],
        "largest_successful_calibration_score": int(max(x)),
        "smallest_successful_calibration_score": int(min(x)),
        "lowest_observed_oom_score": min(
            (int(r["memory_estimate"]["structural_estimated_bytes"]) for r in ooms),
            default=None,
        ),
        "oom_evidence": [
            {
                "pair_id": r["pair_id"],
                "driver_channel_count": r["driver_channel_count"],
                "structural_score": r["memory_estimate"]["structural_estimated_bytes"],
                "calibration_identity_sha256": r["calibration_identity_sha256"],
            }
            for r in ooms
        ],
    }


def predicted_bytes(
    score: int, calibration: Mapping[str, Any], *, safety_adjusted: bool = True
) -> int:
    base = (
        float(calibration["intercept_bytes"])
        + float(calibration["structural_score_coefficient"]) * score
        + float(calibration["positive_residual_margin_bytes"])
    )
    return math.ceil(
        base
        * (float(calibration["prediction_safety_factor"]) if safety_adjusted else 1)
    )


def exclusion_reason(score: int, calibration: Mapping[str, Any]) -> str | None:
    oom = calibration.get("lowest_observed_oom_score")
    if oom is not None and score >= int(oom):
        return "at or above matching observed OOM bound"
    if score > int(calibration["largest_successful_calibration_score"]):
        return "above largest successful calibration score"
    if predicted_bytes(score, calibration) > int(
        calibration["gpu_admission_budget_bytes"]
    ):
        return "safety-adjusted prediction exceeds admission budget"
    return None


def select_candidate(
    pairs: Sequence[Mapping[str, Any]],
    reports: Sequence[Mapping[str, Any]],
    *,
    channel_count: float | None = None,
    quantile: float | None = None,
) -> Mapping[str, Any] | None:
    if (channel_count is None) == (quantile is None):
        raise ValueError("Specify one channel count or quantile")
    if quantile is not None:
        if not 0 <= quantile <= 1:
            raise ValueError("Quantile must be in [0, 1]")
        channel_count = float(
            np.quantile([p["driver_channel_count"] for p in pairs], quantile)
        )
    tested = {r["pair_id"] for r in reports}
    oom = min(
        (
            int(r["memory_estimate"]["structural_estimated_bytes"])
            for r in reports
            if r.get("outcome") == "oom"
        ),
        default=None,
    )
    candidates = [
        p
        for p in pairs
        if p["pair_id"] not in tested
        and p["driver_channel_count"] > 0
        and (oom is None or p["memory_estimate"]["structural_estimated_bytes"] < oom)
    ]
    return min(
        candidates,
        key=lambda p: (abs(p["driver_channel_count"] - channel_count), p["pair_id"]),
        default=None,
    )
