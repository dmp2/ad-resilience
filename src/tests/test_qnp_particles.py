"""QNP-specific data handling: slide particles, multistain OD export, HALO annotation masses."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(SRC), str(SRC / "preprocess")]

from preprocess.qnp_particles import canonical_layer, section_annotation_masses, tissue_class_particles  # noqa: E402
from section_mapping.alignment import Similarity2D  # noqa: E402


@pytest.mark.parametrize("name, expected", [
    ("Layer1", "L1"), ("LAyerIII", "L3"), ("LAyer IV", "L4"), ("Layer V-VI", "L5-6"),
    ("Layer5-6", "L5-6"), ("LayerII", "L2"), ("Vessel", None),
])
def test_canonical_layer_names(name, expected):
    assert canonical_layer(name) == expected


def test_tissue_class_particles_conserve_area_and_split_by_membership():
    mask = np.zeros((40, 60), bool)
    mask[5:35, 10:50] = True
    p_wm = np.where(np.arange(60)[None, :] < 30, 0.0, 1.0) * mask
    px = (0.01, 0.01)
    X, nu, diagnostics = tissue_class_particles(px, mask, p_wm, bin_mm=0.1)
    assert nu.sum() == pytest.approx(mask.sum() * 1e-4, rel=1e-6)  # float32 archive precision
    assert nu[:, 0].sum() == pytest.approx(30 * 20 * 1e-4)  # columns 10..29 are gray
    assert np.all(X[:, 2] == 0)
    # Block centres of 10-pixel blocks (x = column, y = row): mask columns 10-49 -> blocks 1-4.
    np.testing.assert_allclose(np.unique(X[:, 0]), [0.145, 0.245, 0.345, 0.445], atol=1e-6)
    np.testing.assert_allclose(np.unique(X[:, 1]), [0.045, 0.145, 0.245, 0.345], atol=1e-6)
    assert diagnostics["downsample_factor"] == 10


@pytest.fixture
def od_export_case(tmp_path):
    import json
    from preprocess.dense_image_utils import save_dense_particle_archive
    from scripts import align_qnp_multistain as align

    native, aligned = tmp_path / "native", tmp_path / "aligned"
    aligned.mkdir()
    job = {"_name": "synthetic", "reference_stain": "NEUN", "stains": {"NEUN": {}, "AT": {}},
           "bins_mm": [0.5, 1.0], "particle_dir": str(native), "alignment": {"output_dir": str(aligned)}}
    truth = Similarity2D(37.0, 1.0, True, (3.0, -2.0))
    record = {"job": "synthetic", "reference_stain": "NEUN", "reference_slide": "slide-NEUN",
              "edges": {"AT": {"slide": "slide-AT", "transform": truth.as_dict(),
                                "slide_mm_to_local_mm": align.homogeneous(truth, "to_local")}}}
    (aligned / "transforms.json").write_text(json.dumps(record))
    X = np.array([[1, 2, 0], [4, -1, 0], [2, 5, 0]], dtype=np.float32)
    for stain in job["stains"]:
        for tag in ("500um", "1000um"):
            bins = np.array([[0.1, 0.2, 0.3], [0.4, 0, 0.2], [0.2, 0.3, 0.1]], dtype=np.float32)
            moments = np.column_stack([bins.sum(1), bins @ [0.0, 0.5, 1.0]])
            save_dense_particle_archive(native / f"slide-{stain}_od_bins_{tag}.npz", X, moments, bins,
                                        {"bin_edges": np.linspace(0, 1, 4), "bin_centers": np.linspace(0, 1, 3),
                                         "intensity_feature_names": np.array(["area_mm2", "area_times_scaled_od"])},
                                        {"coordinate_space": "qnp_slide_mm", "coordinate_definition": "native x right, y down",
                                         "representation": "optical_density_bins", "slide": {"path": f"{stain}.svs"},
                                         "intensity_transform": {"mode": "robust_minmax", "lower": 0.2, "upper": 3.7},
                                         "comparability": "stain-specific", "background_rgb": [250, 251, 252]},
                                        write_vtk=False)
    return align, job, native, aligned, truth


def test_od_export_only_preserves_features_and_provenance(od_export_case, monkeypatch):
    import json
    from preprocess import dense_image_utils, qnp_particles

    align, job, native, aligned, truth = od_export_case
    sources = {p: p.read_bytes() for p in native.glob("*.npz")}
    transforms_before = (aligned / "transforms.json").read_bytes()

    def forbidden(*args, **kwargs):
        pytest.fail("export-only must not fit, rebuild features, load maps or render")

    for name in ("FootprintScorer", "search", "refine", "tissue_class_particles", "resample_map",
                 "load_xmodmap", "overlay_figure"):
        monkeypatch.setattr(align, name, forbidden)
    monkeypatch.setattr(dense_image_utils, "fit_intensity_transform", forbidden)
    monkeypatch.setattr(qnp_particles, "optical_density_particles", forbidden)
    monkeypatch.setattr(align, "load_config", lambda _: {"qnp_section_inputs": {"synthetic": job}})
    monkeypatch.setattr(sys, "argv", ["align_qnp_multistain.py", "--config", "unused", "--job", "synthetic",
                                     "--export-od-only"])
    align.main()  # No inventory, SVS, maps or tissue-class archives exist in this fixture.
    for source, original in sources.items():
        assert source.read_bytes() == original
        target = aligned / f"{source.stem}_local.npz"
        with np.load(source) as before, np.load(target) as after:
            assert set(after.files) == set(before.files)
            for key in before.files:
                if key not in {"X", "metadata_json"}:
                    np.testing.assert_array_equal(after[key], before[key])
                    assert after[key].dtype == before[key].dtype
            for key in ("nu_intensity", "nu_bin"):
                np.testing.assert_array_equal(after[key].sum(0), before[key].sum(0))
                assert after[key].sum() == before[key].sum()
            if "NEUN" in source.stem:
                np.testing.assert_array_equal(after["X"], before["X"])
            else:
                # Independent affine expression for slide -> local, including reflection/translation.
                expected = np.linalg.solve(truth.matrix, before["X"][:, :2].T).T + truth.tau
                np.testing.assert_allclose(after["X"][:, :2], expected, atol=1e-6)
            np.testing.assert_array_equal(after["X"][:, 2], before["X"][:, 2])
            old, new = (json.loads(str(a["metadata_json"])) for a in (before, after))
            for key in old.keys() - {"coordinate_space", "coordinate_definition", "particle_role"}:
                assert new[key] == old[key]
            assert new["coordinate_space"] == "qnp_local_mm"
            assert new["source_coordinate_definition"] == old["coordinate_definition"]
            assert new["source_archive"] == str(source.resolve())
            assert new["alignment_transforms"] == str((aligned / "transforms.json").resolve())
            matrix = np.asarray(new["slide_to_local"])
            np.testing.assert_allclose(after["X"][:, :2], before["X"][:, :2] @ matrix[:2, :2].T + matrix[:2, 2], atol=1e-6)
    assert (aligned / "transforms.json").read_bytes() == transforms_before
    with pytest.raises(FileExistsError):
        align.main()
    monkeypatch.setattr(sys, "argv", sys.argv + ["--overwrite"])
    align.main()
    assert len(list(aligned.glob("*.npz"))) == 4


@pytest.mark.parametrize("invalid", ["reference", "scale", "direction", "missing"])
def test_od_export_rejects_incompatible_inputs(od_export_case, invalid):
    import json

    align, job, native, aligned, _ = od_export_case
    path = aligned / "transforms.json"
    record = json.loads(path.read_text())
    edge = record["edges"]["AT"]
    if invalid == "reference":
        record["reference_stain"] = "LFB"
    elif invalid == "scale":
        edge["transform"]["scale"] = 1.1
    elif invalid == "direction":
        edge["slide_mm_to_local_mm"][0][2] += 1
    else:
        next(native.glob("*AT*1000um.npz")).unlink()
    path.write_text(json.dumps(record))
    with pytest.raises((ValueError, FileNotFoundError)):
        align.export_od_local(job, native, aligned)
    assert not list(aligned.glob("*.npz"))


@pytest.mark.parametrize("fails", [False, True])
def test_validation_determinism_scratch_under_home(tmp_path, monkeypatch, fails):
    from visualization.validate import determinism_checks

    monkeypatch.setenv("HOME", str(tmp_path))
    folders = []

    def render(folder):
        assert folder.is_relative_to(tmp_path)
        folders.append(folder)
        if fails:
            raise RuntimeError("synthetic render failure")
        path = folder / "figure.png"
        path.write_bytes(b"deterministic synthetic artifact")
        return [path]

    if fails:
        with pytest.raises(RuntimeError, match="synthetic render failure"):
            determinism_checks(render, "synthetic")
    else:
        assert determinism_checks(render, "synthetic")["passed"]
        assert len(folders) == 2
    assert folders and all(not folder.exists() for folder in folders)


def test_annotation_masses_lie_between_zero_and_tissue():
    rng = np.random.default_rng(5)
    px = (0.032, 0.032)
    mask = np.zeros((200, 200), bool)
    mask[20:180, 30:170] = True
    layers = np.zeros((200, 200), np.uint8)
    layers[20:100, 30:170] = 3
    centres = [(x, y) for y in np.arange(0.9, 5.8, 0.5) for x in np.arange(1.2, 5.5, 0.5)]
    X = np.column_stack([np.asarray(centres), np.zeros(len(centres))])
    nu = rng.uniform(0.1, 0.25, (len(X), 2))
    masses, record = section_annotation_masses({"mask": mask, "layers": layers, "px_mm": np.array(px)}, X, nu)
    tissue = nu.sum(1)
    assert np.all(masses[:, 0] >= 0) and np.all(masses[:, 0] <= tissue + 1e-12)
    upper = X[:, 1] < 100 * 0.032 - 0.3                     # well inside the annotated half
    lower = X[:, 1] > 100 * 0.032 + 0.3
    assert np.allclose(masses[upper, 0], tissue[upper]) and np.allclose(masses[lower, 0], 0.0)
    none, rec = section_annotation_masses({"mask": mask, "px_mm": np.array(px)}, X, nu)
    assert none is None and rec["status"] == "no_annotation"
