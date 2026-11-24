"""Regression references of the section mapping, checked against the saved two-donor runs.

Software references, not anatomical truth (docs/section_mapping_findings.md §1).  Skipped when the
AHRAv2 atlas or the saved runs (``results/``, ``data/``: not under version control) are absent.
Nothing here searches or registers; the slowest test re-runs ``--stage report`` on a copy (~30 s).

    PYTHONPATH=src:src/config_setup:src/preprocess:src/scripts pytest -q src/tests/test_mapping_regression.py
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1]
REPO = SRC.parent
sys.path[:0] = [str(SRC), str(SRC / "preprocess"), str(SRC / "config_setup"), str(SRC / "scripts")]

RESULTS = REPO / "results/registrations"
ATLAS = REPO / "data/raw/allen/ahrav2/2025/ahrav2_gyral_annotations_500um.nii.gz"
JOBS = ("H20.33.034_MTG", "H20.33.043_MTG")
PRIOR_RUNS = ("", "landmark_slab", "region_prior", "region_prior__landmark_slab")
pytestmark = pytest.mark.skipif(not (ATLAS.is_file() and all((RESULTS / f"qnp_{j}_stacking/report.json").is_file()
                                                             for j in JOBS)),
                                reason="AHRAv2 atlas or the saved stacking runs are not present")


def _saved(job: str, run: str = "") -> Path:
    return RESULTS / f"qnp_{job}_stacking" / run


@pytest.fixture(scope="module")
def config():
    from config_setup.project_config import load_config

    return load_config(REPO / "configs/project_config.yaml")


def _ctx(config, job, region_prior=False, landmark_slab=False, output_dir=None):
    import copy

    from config_setup.qnp_jobs import resolve_qnp_job
    from map_qnp_sections import StackContext

    job_cfg = resolve_qnp_job(config, job)
    if output_dir is not None:
        job_cfg = copy.deepcopy(job_cfg)
        job_cfg["atlas_scan"]["stacking"]["output_dir"] = str(output_dir)
    return StackContext(config, job_cfg, region_prior=region_prior, landmark_slab=landmark_slab)


def test_lgn_landmark_and_slab_interval(config):
    from section_mapping.atlas import landmark_slab_prior

    ctx = _ctx(config, "H20.33.034_MTG")
    record = landmark_slab_prior(ctx.atlas, ctx.cfg["landmark_slab"], "left")
    assert record["landmark"]["y_mm"] == -22.0
    assert record["search_interval"]["y_mm"] == [-13.0, -31.0]
    assert record["slab_thickness_mm"] == 4.0 and record["slab_offsets"] == [-1, 0, 1]
    assert len(record["candidates"]) == 24


def test_saved_profiles_are_reproduced_by_summarize_level():
    from section_mapping.stacking import summarize_level

    checked = 0
    for job in JOBS:
        for run in PRIOR_RUNS:
            previous = None
            for level in ("h1.5", "h1.0", "500um"):
                folder = _saved(job, run)
                if not (folder / f"profile_{level}.json").is_file():
                    continue
                saved = json.loads((folder / f"profile_{level}.json").read_text())
                from map_qnp_sections import read_rows

                rows = read_rows(folder / f"profile_{level}.csv")
                inherit = None
                if previous is None and run:
                    wide = _saved(job) / f"profile_{level}.json"
                    inherit = ((set(json.loads(wide.read_text())["flat_anchors"]),
                                f"inherited from the unconstrained first level ({wide.name})") if wide.is_file()
                               else (set(), "not judged"))
                got = json.loads(json.dumps(summarize_level(level, rows, previous, step_mm=0.5, delta_top=20,
                                                            inherit_flat=inherit)))
                for key in ("delta", "flat_anchors", "union_window_y", "window_shrink_vs_previous_mm", "stop_escalating"):
                    if key in saved:
                        assert got[key] == saved[key], (job, run, level, key)
                for anchor, w in saved["anchors"].items():
                    for key in ("best_y", "best_score", "window_y", "width_mm", "flat", "contiguous"):
                        assert got["anchors"][anchor][key] == w[key], (job, run, level, anchor, key)
                previous = saved
                checked += 1
    assert checked >= 14


def test_reference_placements_and_verdicts():
    final = json.loads((_saved("H20.33.034_MTG") / "report.json").read_text())
    assert {a: final["sections"][a]["window_y_mm"] for a in ("NEUN", "LFB", "AT")} == \
        {"NEUN": [-27.5, -28.0], "LFB": [-27.5, -28.0], "AT": [-27.5, -28.0]}
    c034 = json.loads((_saved("H20.33.034_MTG", "region_prior__landmark_slab") / "candidates.json").read_text())
    assert c034["verdict"]["summary"] == "supported by image geometry"
    c043 = json.loads((_saved("H20.33.043_MTG", "region_prior__landmark_slab") / "candidates.json").read_text())
    assert c043["verdict"]["summary"] == "primarily prior-driven"
    for anchor, entry in c043["anchors"].items():
        assert entry["admissible"]["search_pose"]["y0_mm"] == -24.5, anchor
        assert entry["comparison"]["status"].startswith("prior-driven"), anchor
        assert entry["unconstrained"]["search_pose"]["y0_mm"] in (-15.0, -14.0, -13.5), anchor


def test_report_stage_reruns_identically(config, tmp_path):
    from map_qnp_sections import run_report

    shutil.copytree(_saved("H20.33.043_MTG"), tmp_path / "run")
    ctx = _ctx(config, "H20.33.043_MTG", region_prior=True, landmark_slab=True, output_dir=tmp_path / "run")
    saved = json.loads((_saved("H20.33.043_MTG", "region_prior__landmark_slab") / "report.json").read_text())
    got = json.loads(json.dumps(run_report(ctx)))

    def same(a, b):
        if isinstance(a, dict):
            return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
        if isinstance(a, list):
            return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
        if isinstance(a, float) or isinstance(b, float):
            return abs(a - b) <= 1e-6 * max(1.0, abs(a))
        return a == b

    assert same(saved, got)


def test_stacking_handoff_poses_reproduce_their_scores(config):
    """select_placement re-scores each pose on the plane it was chosen on (direction / composition check)."""

    from map_qnp_sections import select_placement

    established = select_placement(_ctx(config, "H20.33.034_MTG"), "NEUN", None)
    assert established["y_mm"] == -27.5 and not established["explicit"] and established["concerns"] == []
    with pytest.raises(SystemExit, match="not eligible for automatic export"):
        select_placement(_ctx(config, "H20.33.043_MTG"), "NEUN", None)
    # A prior run is eligible only with its own data-only comparison.
    with pytest.raises(SystemExit, match="no candidates.json"):
        select_placement(_ctx(config, "H20.33.034_MTG", landmark_slab=True), "NEUN", None)
    compared = select_placement(_ctx(config, "H20.33.034_MTG", True, True), "NEUN", None)
    assert compared["y_mm"] == -27.5 and "supported by image geometry" in compared["anatomical_status"]
    admissible = select_placement(_ctx(config, "H20.33.043_MTG", True, True), "NEUN", "admissible")
    assert admissible["y_mm"] == -24.5 and admissible["explicit"]
    assert any("prior-driven" in c for c in admissible["concerns"])
    passenger = select_placement(_ctx(config, "H20.33.043_MTG"), "GFAP", "report")
    assert passenger["host"] == "NEUN"
    for sel in (established, compared, admissible, passenger):
        assert abs(sel["score_check"]["recomputed"] - sel["score_check"]["recorded"]) <= 1e-4
