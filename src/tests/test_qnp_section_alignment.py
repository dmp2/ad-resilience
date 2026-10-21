"""Synthetic checks for QNP particles, 2-D similarity search and atlas reslicing."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(SRC), str(SRC / "preprocess")]

from preprocess.particle_utils import cut_plane  # noqa: E402
from preprocess.qnp_particles import canonical_layer, tissue_class_particles  # noqa: E402
from preprocess.section_alignment import (  # noqa: E402
    FootprintScorer,
    Similarity2D,
    label_class_masses,
    lookup_tables,
    plane_basis,
    reslice_labels,
    search,
)


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


def test_similarity_round_trip():
    t = Similarity2D(33.0, 1.2, True, (3.0, -2.0))
    X = np.random.default_rng(0).normal(size=(20, 2))
    np.testing.assert_allclose(t.moving_to_section(t.section_to_moving(X)), X, atol=1e-12)
    assert Similarity2D.from_dict(t.as_dict()) == t


def _two_class_shape():
    """An asymmetric 'L' of gray tissue with a white-matter bar (1 mm particles)."""

    xs, ys = np.meshgrid(np.arange(0, 20.0), np.arange(0, 14.0))
    X = np.column_stack([xs.ravel(), ys.ravel()])
    keep = (X[:, 1] < 4) | (X[:, 0] < 5)
    X = X[keep]
    white = (X[:, 1] < 2) & (X[:, 0] > 3)
    nu = np.column_stack([~white, white]).astype(float)
    return X, nu


def test_search_recovers_rotation_and_reflection():
    X, nu = _two_class_shape()
    truth = Similarity2D(70.0, 1.0, True, (8.0, 5.0))
    P = truth.section_to_moving(X)  # the moving measure is the section seen through `truth`
    scorer = FootprintScorer(X, nu, sigmas_mm=(2.0, 1.0))
    score, found = search(scorer, P, nu, thetas=np.arange(0, 360, 15), scales=[1.0], flips=[False, True],
                          scale_bounds=(1.0, 1.0), h=0.5, blur_mm=1.0, refine_top=3, max_evals=300)[0]
    assert found.flip is True
    assert abs((found.theta_deg - 70.0 + 180) % 360 - 180) < 2.0
    np.testing.assert_allclose(found.moving_to_section(P), X, atol=0.3)
    assert score < 1e-2


def test_footprint_score_is_one_without_overlap():
    X, nu = _two_class_shape()
    scorer = FootprintScorer(X, nu)
    assert scorer.score(X + 100.0, nu) == 1.0
    assert scorer.score(X, nu) == pytest.approx(0.0, abs=1e-9)


def test_reslice_matches_cut_plane_for_a_coronal_plane():
    rng = np.random.default_rng(1)
    labels = rng.integers(0, 3, size=(12, 10, 8))
    affine = np.diag([0.5, 0.5, 0.5, 1.0])
    affine[:3, 3] = [-3.0, -2.5, -2.0]
    u, v, n = plane_basis(0.0, 0.0)
    np.testing.assert_allclose(n, [0, -1, 0])
    origin = np.array([0.0, -0.5, 0.0])  # a voxel-centre plane
    grid, axis = reslice_labels(labels, affine, origin, u, v, half_width_mm=2.0, spacing_mm=0.5)
    ijk = np.argwhere(labels == 2)
    world = ijk * 0.5 + affine[:3, 3]
    plane_X, (kept,), _ = cut_plane(world, [np.ones((len(world), 1))], origin, u, v, 0.25)
    inside = (np.abs(plane_X[:, 0]) <= 2.0) & (np.abs(plane_X[:, 1]) <= 2.0)
    from_cut = {tuple(np.round(p, 6)) for p in plane_X[inside, :2]}
    bb, aa = np.nonzero(grid == 2)
    from_reslice = {(round(axis[a], 6), round(axis[b], 6)) for a, b in zip(aa, bb)}
    assert from_cut == from_reslice


def test_lookup_tables_partition_labels():
    coarse, fine = lookup_tables(10, white_matter=[5], drop=[7], fine_regions={"MTG": [3]})
    assert not coarse["gray_matter"][0] and not coarse["white_matter"][0]
    assert coarse["white_matter"][5] and not coarse["gray_matter"][5]
    assert not coarse["gray_matter"][7] and not coarse["white_matter"][7]
    assert fine["MTG"][3] and not fine["other_gray_matter"][3] and fine["other_gray_matter"][4]
    grid = np.array([[3, 5], [7, 4]])
    X, nu = label_class_masses(grid, np.array([0.0, 0.5]), 0.5, coarse)
    assert nu.sum() == pytest.approx(3 * 0.25)  # the dropped pixel carries no mass


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
