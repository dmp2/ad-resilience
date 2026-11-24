"""A QNP job resolved from shared defaults plus a thin per-donor block (config_setup.qnp_jobs)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

SRC = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(SRC), str(SRC / "config_setup")]

from config_setup.project_config import ConfigError, load_config  # noqa: E402
from config_setup.qnp_jobs import deep_merge, resolve_qnp_job  # noqa: E402

DEFAULTS = {
    "slide_dir": "raw/{region}/{donor}",
    "particle_dir": "particles/{region}/{donor}",
    "qc_dir": "qc/{job}",
    "reference_stain": "NEUN",
    "stains": {"NEUN": {"structural": True}, "LFB": {"structural": True}, "AT": {"structural": True},
               "GFAP": {"structural": False, "align_features": "tissue"}},
    "alignment": {"output_dir": "{particle_dir}/aligned", "theta_step_deg": 10},
    "atlas_scan": {
        "ribbon_points": "{particle_dir}/{reference_stem}_ribbon_points.npz",
        "representations": {"NEUN_gmwm": {"particles": "{aligned_dir}/{stem[NEUN]}_1000um_local.npz"},
                            "consensus": {"particles": "{aligned_dir}/consensus_{structural}_1000um.npz"}},
        "export": {"source_npz": "inputs/{job}/{plane_id}/plane.npz"},
        "output_dir": "results/qnp_{job}_scan",
    },
}


def _project(tmp_path: Path, jobs: dict, slides: list[str] | None = None) -> dict:
    config = {"project": {"root": str(tmp_path)}, "qnp_section_defaults": DEFAULTS, "qnp_section_inputs": jobs}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    if slides is not None:
        folder = tmp_path / "raw" / "middle-temporal-gyrus" / "D99.99.999"
        folder.mkdir(parents=True)
        for name in slides:
            (folder / name).write_bytes(b"")
    return load_config(path)


def test_thin_block_fills_paths_stems_and_stains(tmp_path):
    config = _project(tmp_path, {"D99.99.999_MTG": {"donor": "D99.99.999", "region": "middle-temporal-gyrus"}},
                      ["D99.99.999-A07-NEUN.svs", "D99.99.999-A7-LFB.svs", "D99.99.999-A7-AT.svs"])
    job = resolve_qnp_job(config, "D99.99.999_MTG")
    assert job["slide_dir"] == "raw/middle-temporal-gyrus/D99.99.999"
    assert job["qc_dir"] == "qc/D99.99.999_MTG"
    # block names come from the slide files (NeuN "A07", the others "A7"), not from config
    assert job["_slide_stems"] == {"NEUN": "D99.99.999-A07-NEUN", "LFB": "D99.99.999-A7-LFB", "AT": "D99.99.999-A7-AT"}
    assert set(job["stains"]) == {"NEUN", "LFB", "AT"} and job["_stains_absent"] == ["GFAP"]
    scan = job["atlas_scan"]
    aligned = "particles/middle-temporal-gyrus/D99.99.999/aligned"
    assert scan["representations"]["NEUN_gmwm"]["particles"] == f"{aligned}/D99.99.999-A07-NEUN_1000um_local.npz"
    assert scan["representations"]["consensus"]["particles"] == f"{aligned}/consensus_NEUN_LFB_AT_1000um.npz"
    assert scan["ribbon_points"].endswith("/D99.99.999-A07-NEUN_ribbon_points.npz")
    assert scan["export"]["source_npz"] == "inputs/D99.99.999_MTG/{plane_id}/plane.npz"  # filled later
    assert scan["output_dir"] == "results/qnp_D99.99.999_MTG_scan"
    assert job["annotation_misfit"] == [] and job["_annotation_misfit_source"].startswith("none")


def test_annotation_misfit_from_inventory_unless_set(tmp_path):
    jobs = {"D99.99.999_MTG": {"donor": "D99.99.999", "region": "middle-temporal-gyrus"},
            "D99.99.999_SET": {"donor": "D99.99.999", "region": "middle-temporal-gyrus", "qc_dir": "qc/D99.99.999_MTG",
                               "annotation_misfit": ["AT"]}}
    config = _project(tmp_path, jobs, ["D99.99.999-A07-NEUN.svs", "D99.99.999-A7-GFAP.svs"])
    qc = tmp_path / "qc" / "D99.99.999_MTG"
    qc.mkdir(parents=True)
    (qc / "inventory.json").write_text(json.dumps({"readiness": {"slides_with_misfit_annotations": ["GFAP"]}}))
    derived = resolve_qnp_job(config, "D99.99.999_MTG")
    assert derived["annotation_misfit"] == ["GFAP"] and derived["_annotation_misfit_source"] == "inventory.json"
    explicit = resolve_qnp_job(config, "D99.99.999_SET")
    assert explicit["annotation_misfit"] == ["AT"] and explicit["_annotation_misfit_source"] == "config"


def test_explicit_block_wins_and_stain_list_is_replaced(tmp_path):
    block = {"region": "middle-temporal-gyrus", "slide_dir": "elsewhere", "particle_dir": "p", "qc_dir": "q",
             "stains": {"NEUN": {"structural": True}, "GFAP": {"structural": False}},
             "alignment": {"output_dir": "p/aligned"}}
    config = _project(tmp_path, {"OLD_MTG": block})
    job = resolve_qnp_job(config, "OLD_MTG")
    for key in ("slide_dir", "particle_dir", "qc_dir"):
        assert job[key] == block[key]
    assert job["stains"] == block["stains"]  # no align_features merged in from the defaults
    assert job["alignment"] == {"output_dir": "p/aligned", "theta_step_deg": 10}
    assert job["donor"] == "OLD"


def test_unknown_job_raises_and_minimal_block_is_tolerated(tmp_path):
    config = _project(tmp_path, {"X_MTG": {"donor": "X"}})
    with pytest.raises(ConfigError):
        resolve_qnp_job(config, "nope")
    job = resolve_qnp_job(config, "X_MTG")  # no region: templates needing it stay unfilled
    assert job["slide_dir"] == "raw/{region}/X" and job["_slide_stems"] == {}


def test_deep_merge_replaces_lists():
    assert deep_merge({"a": {"b": [1, 2], "c": 1}}, {"a": {"b": [3]}}) == {"a": {"b": [3], "c": 1}}


PROJECT_CONFIG = SRC.parent / "configs" / "project_config.yaml"


@pytest.mark.skipif(not PROJECT_CONFIG.exists(), reason="project config not present")
def test_reference_donor_resolves_to_its_explicit_block():
    """Every value of the explicit H20.33.034_MTG block survives resolution unchanged."""

    config = load_config(PROJECT_CONFIG)
    original = config["qnp_section_inputs"]["H20.33.034_MTG"]
    job = resolve_qnp_job(config, "H20.33.034_MTG")

    def same(a, b, path="H20.33.034_MTG"):
        if isinstance(a, dict):
            for key, value in a.items():
                same(value, b[key], f"{path}.{key}")
        else:
            assert a == b, path

    same(original, job)
