"""Regression coverage for current-backend CUDA admission and pilot isolation."""

from __future__ import annotations

import csv
import json
from types import SimpleNamespace

import numpy as np
import pytest

from preprocess import allen_gpu_memory as memory
from preprocess import densify_allen_annotations as dense
from scripts import calibrate_allen_gpu_assignments as triage
from scripts import benchmark_allen_pair_registration as benchmark


def identity():
    return {
        "wsi_commit": "wsi",
        "emlddmm_commit": "em",
        "backend_source_sha256": {"wsi": "hash"},
        "mapping_backend": memory.BACKEND,
        "upsample_mode": "seg",
        "effective_solver_configuration_sha256": "solver",
        "image_shape_yx": [10, 20],
        "registered_geometry": [[0, 1], [0, 1]],
        "dtype": "float32",
        "structural_model": memory.MODEL,
        "gpu_model": "2080 SUPER",
        "cuda_total_bytes": 10 * 1024**3,
    }


def pilot(pair_id, score, outcome="success", device="cuda:0"):
    ident = identity()
    return {
        "pair_id": pair_id,
        "device": device,
        "outcome": outcome,
        "numerically_valid": outcome == "success",
        "configuration_kind": "corrected-production",
        "calibration_identity": ident,
        "calibration_identity_sha256": memory.identity_hash(ident),
        "effective_solver_configuration_sha256": "solver",
        "image_shape_yx": [10, 20],
        "cuda_total_bytes": ident["cuda_total_bytes"],
        "cuda_free_before_bytes": 9 * 1024**3,
        "peak_cuda_reserved_bytes": score + 50,
        "driver_channel_count": score // 10,
        "memory_estimate": {"model": memory.MODEL, "structural_estimated_bytes": score},
    }


def resource(pair_id, score, status="remaining"):
    ident = identity()
    return {
        "pair_id": pair_id,
        "checkpoint_state": status,
        "driver_channel_count": score // 10,
        "memory_estimate": {"model": memory.MODEL, "structural_estimated_bytes": score},
        "calibration_identity": ident,
        "calibration_identity_sha256": memory.identity_hash(ident),
    }


def resources(rows):
    ident = identity()
    return {
        "pairs": rows,
        "calibration_identity": ident,
        "calibration_identity_sha256": memory.identity_hash(ident),
    }


def pilots():
    return [pilot("0000-0001", 100), pilot("0001-0002", 200), pilot("0002-0003", 300)]


@pytest.mark.parametrize(
    "field,value",
    [
        ("wsi_commit", "changed"),
        ("mapping_backend", "allen.direct"),
        ("dtype", "float64"),
        ("cuda_total_bytes", 8 * 1024**3),
        ("registered_geometry", [[0, 2], [0, 1]]),
        ("backend_source_sha256", {"wsi": "dirty"}),
        ("gpu_model", "different"),
    ],
)
def test_identity_change_invalidates_same_solver(field, value):
    reports = pilots()
    reports[1]["calibration_identity"][field] = value
    reports[1]["calibration_identity_sha256"] = memory.identity_hash(
        reports[1]["calibration_identity"]
    )
    with pytest.raises(RuntimeError):
        memory.fit(reports)


def test_missing_provenance_and_cpu_pilots_rejected():
    reports = pilots()
    del reports[1]["calibration_identity"]["registered_geometry"]
    with pytest.raises(RuntimeError):
        memory.fit(reports)
    reports = pilots()
    reports[1]["device"] = "cpu"
    with pytest.raises(RuntimeError, match="production CUDA"):
        memory.fit(reports)
    with pytest.raises(RuntimeError, match="No CUDA"):
        memory.fit([])


def test_distinct_success_scores_required_oom_not_regressed():
    reports = pilots()
    fit = memory.fit(reports)
    oom = pilot("0003-0004", 400, "oom")
    oom["peak_cuda_reserved_bytes"] = 10**20
    with_oom = memory.fit([*reports, oom])
    assert (
        with_oom["structural_score_coefficient"] == fit["structural_score_coefficient"]
    )
    assert with_oom["lowest_observed_oom_score"] == 400
    with pytest.raises(RuntimeError, match="three successful"):
        memory.fit([reports[0], reports[1], oom])


def test_structural_score_monotone_gap_independent_and_nested_retention():
    axes = [np.arange(10) * 200.0, np.arange(20) * 200.0]
    config = {"nt": 10, "n_iter": [100], "dv": [[1000.0, 1000.0]], "dtype": "float32"}
    low = memory.estimate(4, [10, 20], axes, config, inferred_group_planes=1)
    long = memory.estimate(4, [10, 20], axes, config, inferred_group_planes=68)
    high = memory.estimate(94, [10, 20], axes, config)
    assert (
        low["structural_score"] == long["structural_score"] < high["structural_score"]
    )
    assert low["relative_runtime_work_units"] < long["relative_runtime_work_units"]
    assert low["synthetic_backend_z_planes"] == 5
    assert low["velocity_grid_shape_zyx"][0] == 6
    assert low["retention"]["outer_symmetric_calls"] == 2
    assert low["retention"]["inner_optimizations_per_outer_call"] == 2
    assert low["retention"]["returned_trajectories_device"] == "cpu"
    assert low["components"]["nested_retained_cuda_axes_score_bytes"] > 0


def test_three_admission_boundaries():
    fit = memory.fit(pilots())
    fit["prediction_safety_factor"] = 1
    fit["gpu_admission_budget_bytes"] = 350
    assert memory.exclusion_reason(300, fit) is None
    assert "largest successful" in memory.exclusion_reason(301, fit)
    fit["gpu_admission_budget_bytes"] = 349
    assert "budget" in memory.exclusion_reason(300, fit)
    fit["gpu_admission_budget_bytes"] = 350
    fit["lowest_observed_oom_score"] = 300
    assert "OOM" in memory.exclusion_reason(300, fit)


def test_triage_preserves_history_and_distinguishes_oom_and_numerics(monkeypatch):
    monkeypatch.setattr(
        dense, "discover_inputs", lambda *a: pytest.fail("scheduler loaded images")
    )
    reports = [
        *pilots(),
        pilot("0003-0004", 400, "oom"),
        pilot("0004-0005", 200, "other-failure"),
    ]
    rows = [
        resource("0000-0001", 100, "complete"),
        resource("0002-0003", 300),
        resource("0005-0006", 350),
        resource("0003-0004", 400),
        resource("0004-0005", 200),
    ]
    fit, output = triage.build_assignments(resources(rows), reports)
    by_id = {r["pair_id"]: r for r in output}
    assert by_id["0000-0001"]["historically_completed"] is True
    assert [row["final_assigned_device"] for row in output] == [
        "gpu",
        "gpu",
        "cpu",
        "cpu",
        "unresolved",
    ]
    _, output = triage.build_assignments(
        resources(rows), reports, [pilot("0004-0005", 200, device="cpu")]
    )
    assert output[-1]["final_assigned_device"] == "cpu"
    assert fit["historically_complete_pair_count"] == 1


def test_deterministic_candidate_selection_and_oom_exclusion():
    rows = [
        resource("0003-0004", 400),
        resource("0002-0003", 300),
        resource("0000-0001", 100),
        resource("0001-0002", 100),
    ]
    assert memory.select_candidate(rows, [], channel_count=10)["pair_id"] == "0000-0001"
    assert memory.select_candidate(rows, [], quantile=1)["pair_id"] == "0003-0004"
    reports = [pilot("0003-0004", 300, "oom")]
    assert (
        memory.select_candidate(rows, reports, channel_count=40)["driver_channel_count"]
        == 10
    )
    with pytest.raises(ValueError):
        memory.select_candidate(rows, [], quantile=1.1)


def write_schedule(tmp_path, rows, fit):
    dense._json(tmp_path / "gpu_memory_calibration.json", fit)
    for filename, subset in [
        ("inventory.tsv", rows),
        ("gpu.tsv", [r for r in rows if r["final_assigned_device"] == "gpu"]),
        ("cpu.tsv", [r for r in rows if r["final_assigned_device"] == "cpu"]),
    ]:
        with (tmp_path / filename).open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t")
            writer.writeheader()
            for row in subset:
                writer.writerow(
                    {
                        key: (
                            json.dumps(value, sort_keys=True)
                            if isinstance(value, dict)
                            else value
                        )
                        for key, value in row.items()
                    }
                )


def test_dynamic_schedule_empty_worklist_and_stale_identity(tmp_path):
    fit, rows = triage.build_assignments(
        resources([resource("0000-0001", 100), resource("0002-0003", 300)]), pilots()
    )
    write_schedule(tmp_path, rows, fit)
    schedule = dense.validate_corrected_production_schedule(
        tmp_path / "gpu.tsv",
        tmp_path / "cpu.tsv",
        tmp_path / "inventory.tsv",
        expected_identity=identity(),
    )
    assert len(schedule["gpu_pair_ids"]) == 2
    assert schedule["cpu_pair_ids"] == []
    stale = identity()
    stale["wsi_commit"] = "new"
    with pytest.raises(dense.GlobalRunError, match="Run GPU calibration pilots"):
        dense.validate_corrected_production_schedule(
            tmp_path / "gpu.tsv",
            tmp_path / "cpu.tsv",
            tmp_path / "inventory.tsv",
            expected_identity=stale,
        )
    rows[0]["structural_memory_estimate_bytes"] = 400
    write_schedule(tmp_path, rows, fit)
    with pytest.raises(dense.GlobalRunError, match="prediction|admission"):
        dense.validate_corrected_production_schedule(
            tmp_path / "gpu.tsv", tmp_path / "cpu.tsv", tmp_path / "inventory.tsv"
        )


@pytest.mark.parametrize("outcome", ["success", "oom"])
def test_pilot_uses_production_adapter_and_captures_oom(monkeypatch, tmp_path, outcome):
    class OOM(RuntimeError):
        pass

    cuda = SimpleNamespace(
        OutOfMemoryError=OOM,
        set_device=lambda d: None,
        synchronize=lambda d: None,
        empty_cache=lambda: None,
        reset_peak_memory_stats=lambda d: None,
        mem_get_info=lambda d: (900, 1000),
        max_memory_allocated=lambda d: 500,
        max_memory_reserved=lambda d: 600,
    )
    torch = SimpleNamespace(cuda=cuda)
    context = SimpleNamespace(
        shape=(2, 2),
        registered_axes=[np.arange(2.0), np.arange(2.0)],
        annotation_sections=[1, 2],
        physical_by_section={1: 0, 2: 2},
    )
    monkeypatch.setattr(dense, "discover_inputs", lambda *a: context)
    monkeypatch.setattr(dense, "select_graphic_groups", lambda context, groups: context)
    monkeypatch.setattr(benchmark, "_pair_groups", lambda *a: (31,))
    config = {"nt": 2, "n_iter": [1], "dv": [[1.0, 1.0]], "dtype": "float32"}
    monkeypatch.setattr(benchmark, "_pilot_configuration", lambda *a: config)
    monkeypatch.setattr(benchmark.zarr, "open_group", lambda *a, **k: object())
    image = np.zeros((1, 2, 2), dtype=np.float32)
    monkeypatch.setattr(
        dense,
        "build_annotation_pair_driver",
        lambda *a: (image, image, image[0], image[0], [(31, 1)], {}),
    )
    monkeypatch.setattr(dense, "gpu_calibration_identity", lambda *a, **k: identity())
    monkeypatch.setattr(dense, "_verify_wsi_repository", lambda *a: "wsi")
    monkeypatch.setattr(
        dense, "_load_wsi_upsampler", lambda *a: (object(), torch, "wsi", "module")
    )
    monkeypatch.setattr(
        dense, "fit_pair_trajectories", lambda *a, **k: pytest.fail("legacy pilot path")
    )

    def adapter(*a, **k):
        if outcome == "oom":
            raise OOM("test CUDA allocation")
        return np.zeros((1, 3, 2, 2)), {"mapping_backend": memory.BACKEND}, torch

    monkeypatch.setattr(dense, "_upsample_annotation_pair_with_wsi", adapter)
    cache = resources([resource("0000-0002", 10)])
    path = tmp_path / "resources.json"
    path.write_text(json.dumps(cache))
    report_path = tmp_path / "pilot.json"
    report = benchmark.run_pilot(
        (0, 2),
        "cuda:0",
        tmp_path,
        report_path,
        dataset=tmp_path,
        annotations=tmp_path,
        registration=tmp_path,
        wsi_repository=tmp_path,
        configuration_kind="corrected-production",
        resources=path,
    )
    assert report["outcome"] == outcome
    assert report["peak_cuda_reserved_bytes"] == 600
    assert json.loads(report_path.read_text())["outcome"] == outcome


def test_worker_rejects_stale_initialized_schedule_and_empty_worklist(tmp_path):
    fit = memory.fit(pilots())
    schedule = {
        "calibration_identity": identity(),
        "calibration_identity_sha256": memory.identity_hash(identity()),
        "calibration": fit,
        "calibration_sha256": dense._stable_json_sha256(fit),
    }
    dense.validate_initialized_schedule(schedule, identity())
    changed = identity()
    changed["wsi_commit"] = "next-commit"
    with pytest.raises(dense.GlobalRunError, match="Stale initialized schedule"):
        dense.validate_initialized_schedule(schedule, changed)
    schedule["calibration_sha256"] = "stale"
    with pytest.raises(dense.GlobalRunError, match="calibration is stale"):
        dense.validate_initialized_schedule(schedule, identity())
    worklist = tmp_path / "empty.tsv"
    worklist.write_text("pair_id\tfinal_assigned_device\n")
    assert dense._worklist_pairs(worklist) == ()
