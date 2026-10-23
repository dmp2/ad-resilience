"""Numerical checks of scale, orientation, transform direction and determinism.

Every check compares something this package computes with an existing, trusted
project output (or with an independent label) and records the tolerance it used.
``run`` writes ``results/visualization_prototypes/validation.json``.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from . import scene as S


def _check(name: str, value: float, tol: float, detail: str, compare: str = "<=") -> dict[str, Any]:
    ok = value <= tol if compare == "<=" else value >= tol
    return {"check": name, "value": float(value), "tolerance": f"{compare} {tol}", "passed": bool(ok), "detail": detail}


def _vtk_points(path: Path) -> np.ndarray:
    return read_particle_vtk(path)[0][:, :2]


def merfish_checks(cells: S.Points, merfish: Mapping) -> list[dict]:
    out = []
    base = S.PROJECT / "data/derivatives/sea-ad/merfish" / cells.meta["donor"] / "MTG" / cells.meta["barcode"]
    Z = np.load(base / "full_section_particles.npz")["Z"][:, :2] * 1e-3
    out.append(_check("merfish coordinates vs full_section_particles.npz (mm)", np.abs(Z - cells.xy).max(), 1e-5,
                      "X_spatial_raw x 1e-3 read here vs the existing particle export (float32)"))
    old = np.load(base / "binned_500um_classes.npz")
    a_cell = json.loads(str(old["metadata_json"]))["a_cell_mm2"]
    b = S.bin_merfish(cells, 0.5)
    o, n = np.lexsort(old["X"][:, :2].T[::-1]), np.lexsort(b.xy.T[::-1])
    out.append(_check("500 um bins: count equals binned_500um_classes.npz", abs(len(b.xy) - len(old["X"])), 0,
                      f"{len(b.xy)} bins"))
    out.append(_check("500 um bins: positions vs existing (mm)", np.abs(old["X"][o, :2] - b.xy[n]).max(), 1e-5,
                      "same bin_particles_2d, same class grouping"))
    out.append(_check("500 um bins: class masses vs existing (mm^2)",
                      np.abs(old["nu_Sub"][o] - b.meta["nu_class"][n] * a_cell).max(), 1e-6, "count x a_cell"))
    out.append(_check("X is raw counts (rows sum to Number of spots)",
                      0 if cells.meta["x_check"]["row_sum_equals_number_of_spots"] else 1, 0,
                      json.dumps(cells.meta["x_check"])))
    from .cli import merfish_polygons

    _, _, check = merfish_polygons()
    out.append(_check("provider layer polygons: label agreement in identified frame", check["label_agreement"], 0.95,
                      f"frame polygon = {check['frame_matrix_polygon_from_raw']} @ raw", ">="))
    out.append(_check("provider layer polygons: best alternative frame agreement", check["runner_up_agreement"], 0.05,
                      "orientation is identified, not ambiguous"))
    fit = S.PROJECT / merfish["xiv_fit"]
    xiv = S.load_merfish_xiv(fit)
    chain = xiv["chain"]
    out.append(_check("xIV initial placement vs originalAtlas.vtk (mm)",
                      np.abs(_vtk_points(fit / "originalAtlas.vtk") - (xiv["plane_xy"] - chain["source_center_plane_mm"])).max(),
                      1e-5, "plane - source_center, the frame of D"))
    out.append(_check("xIV deformed points vs atlas_deformationSummary.vtk (mm)",
                      np.abs(_vtk_points(fit / "atlas_deformationSummary.vtk")
                             - (xiv["deformed"].xy - chain["target_center_section_mm"])).max(), 1e-5,
                      "D + target_center places the estimate in merfish_section_mm"))
    strip = np.load(base / "binned_500um_classes_depth_strip.npz")["X"][:, :2]
    out.append(_check("xIV target frame: originalTarget.vtk vs strip - target_center (mm)",
                      np.abs(_vtk_points(fit / "originalTarget.vtk") - (strip - chain["target_center_section_mm"])).max(),
                      1e-5, "confirms the section-frame direction of the chain"))
    disp = xiv["deformed"].xy - xiv["rigid"].xy
    out.append({"check": "xIV displacement decomposition (descriptive)", "passed": True,
                "value": float(np.median(np.linalg.norm(disp - disp.mean(0), axis=1)) * 1000),
                "tolerance": "n/a", "detail": f"mean translation {disp.mean(0).round(3).tolist()} mm; value = median "
                "non-rigid residual in um"})
    return out


def qnp_checks(config: Mapping, job_name: str) -> list[dict]:
    from section_alignment import Similarity2D

    from .figures_qnp import ribbon_dice_checks

    job = S.qnp_job(config, job_name)
    out = []
    record = json.loads((job["_aligned_dir"] / "transforms.json").read_text())
    worst = 0.0
    for stain, edge in record["edges"].items():
        sim = S.Affine2D.from_similarity(Similarity2D.from_dict(edge["transform"]), "local", "slide", "", "")
        worst = max(worst, np.abs(np.asarray(edge["slide_mm_to_local_mm"]) @ sim.matrix - np.eye(3)).max())
    out.append(_check("within-QNP: slide_mm_to_local_mm is the inverse of the fitted similarity", worst, 1e-9,
                      "moving = A (section - tau) with section = qnp_local; so the stored matrix maps slide -> local"))
    ribbon = S.load_ribbon_points(job)
    for key, (plane_id, fit_dir, _) in __import__("visualization.ng_scenes", fromlist=["HYPOTHESES"]).HYPOTHESES.items():
        hyp = S.load_plane_hypothesis(job, plane_id, S.PROJECT / fit_dir)
        moved = hyp["local_to_plane"].apply(ribbon.xy)
        out.append(_check(f"pose {key}: local -> plane transform reproduces qnp_ribbon_points_in_plane (mm)",
                          np.abs(moved - hyp["ribbon_rigid_plane"]).max(), 1e-4,
                          "direction and reflection of the scan similarity"))
    summary = json.loads((S.PROJECT / job["qc_dir"] / "particles_summary.json").read_text())
    for stain in ("NEUN", "LFB", "AT"):
        areas = []
        for tag in ("500um", "1000um"):
            slide = job["_slides"][stain]
            p = S.load_qnp_particles(job["_particle_dir"] / f"{slide.stem}_tissue_classes_{tag}.npz", "x")
            areas.append(p.scalars["area_mm2"].values.sum())
        out.append(_check(f"{stain}: particle area identical at 500 / 1000 um (mm^2)", abs(areas[0] - areas[1]), 1e-3,
                          f"{areas[0]:.3f} mm^2; summary tissue {summary[stain]['tissue']['tissue_area_mm2']:.3f}"))
    checks = ribbon_dice_checks(job)
    for stain in ("ASYN", "I6"):
        c = checks[stain]
        out.append(_check(f"{stain}: HALO ribbon Dice (transformed) vs transforms.json",
                          abs(c["dice_transformed"] - c["existing_qc"]["halo_ribbon_dice_vs_reference"]), 0.02,
                          f"here {c['dice_transformed']:.3f} at 32 um, existing {c['existing_qc']['halo_ribbon_dice_vs_reference']:.3f}"))
        out.append(_check(f"{stain}: transform improves ribbon overlap",
                          c["dice_transformed"] - c["dice_untransformed"], 0.0, "independent HALO labels", ">="))
    return out


def qnp_pyramid_checks(out_dir: Path) -> list[dict]:
    """Each OME-Zarr level is either a native Aperio level (copied) or the 2x2 mean of the level above."""

    import tifffile
    import zarr
    from wsi_pipeline.omezarr.pyramid import _average_pool_2x2_yxc

    out = []
    for record_path in sorted(out_dir.glob("qnp/neuroglancer/data/*/images/*.ome.zarr/visualization_source.json")):
        record = json.loads(record_path.read_text())
        group = zarr.open_group(str(record_path.parent), mode="r")
        keys = sorted(group.array_keys(), key=lambda k: int(k[1:]))
        level = lambda k: np.moveaxis(np.asarray(group[k][:]), 0, -1)  # noqa: E731
        name = record_path.parent.name
        if record.get("pyramid", "").startswith("native"):
            with tifffile.TiffFile(S.PROJECT / record["source_svs"]) as tif:
                same = all(np.array_equal(level(k), tif.series[0].levels[L["aperio_level"]].asarray()[..., :3])
                           for k, L in zip(keys, record["levels"]))
            out.append(_check(f"{name}: every level is a byte-identical native Aperio level", 0 if same else 1, 0,
                              f"{len(keys)} levels, factors {[L['downsample'] for L in record['levels']]}"))
            continue
        with tifffile.TiffFile(S.PROJECT / record["source_svs"]) as tif:
            s0_native = np.array_equal(level(keys[0]), tif.series[0].levels[record["aperio_level"]].asarray()[..., :3])
        worst, previous = 0, level(keys[0])
        for k in keys[1:]:
            current = level(k)
            worst = max(worst, int(np.abs(current.astype(int) - _average_pool_2x2_yxc(previous).astype(int)).max()))
            previous = current
        out.append(_check(f"{name}: s0 is native level {record['aperio_level']} ({s0_native}); generated levels vs "
                          "exact 2x2 mean (grey levels)", worst if s0_native else 99, 1,
                          "built before the native-copy path; tinybrain truncation vs rounding = 1"))
    return out


def neuroglancer_checks(out_dir: Path) -> list[dict]:
    """State transforms decode back to the manifest's physical affines; verify reports pass."""

    out = []
    for state_path in sorted(out_dir.glob("*/neuroglancer/*.state.json")):
        state = json.loads(state_path.read_text())
        manifest = json.loads(state_path.with_name(state_path.name.replace(".state.json", ".manifest.json")).read_text())
        worst, n = 0.0, 0
        for layer in state["layers"]:
            spec = manifest["layers"].get(layer["name"], {}).get("transform_to_scene")
            source = layer["source"]
            if not isinstance(spec, dict) or "transform" not in source:
                continue
            t = source["transform"]
            dims = list(t["outputDimensions"])
            scale = {d: t["outputDimensions"][d][0] * 1e6 for d in dims}
            M = np.asarray(t["matrix"])
            A = np.eye(3)
            for k, d in enumerate(("x", "y")):
                r = dims.index(d)
                A[k, 0] = M[r, dims.index("x")] * scale[d] / scale["x"]
                A[k, 1] = M[r, dims.index("y")] * scale[d] / scale["y"]
                A[k, 2] = M[r, -1] * scale[d]
            worst = max(worst, np.abs(A - np.asarray(spec["matrix_3x3"])).max())
            n += 1
        out.append(_check(f"{state_path.name}: Neuroglancer transforms decode to manifest affines (um)", worst, 1e-6,
                          f"{n} transformed layers"))
        verify = state_path.with_name(state_path.name.replace(".state.json", ".verify.json"))
        if verify.exists():
            report = json.loads(verify.read_text())
            ok = report["all_sources_ok"] and report.get("screenshot", {}).get("ok", False)
            out.append(_check(f"{state_path.name}: assets read back over HTTP + screenshot rendered", 0 if ok else 1, 0,
                              f"{len(report['sources'])} sources; screenshot {report.get('screenshot', {}).get('path')}"))
        else:
            out.append(_check(f"{state_path.name}: verify report present", 1, 0, "run ng-verify"))
    return out


def determinism_checks(render: Callable[[Path], list[Path]], label: str) -> dict:
    """Render twice into temporary folders and compare bytes."""

    scratch = Path.home() / ".cache" / "ad-resilience" / "validation"
    scratch.mkdir(parents=True, exist_ok=True)
    digests = []
    for _ in range(2):
        with tempfile.TemporaryDirectory(dir=scratch) as tmp:
            paths = render(Path(tmp))
            digests.append({p.suffix: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    same = {k: digests[0][k] == digests[1][k] for k in digests[0]}
    return _check(f"{label}: byte-identical rerun ({', '.join(f'{k} {v}' for k, v in same.items())})",
                  0 if all(same.values()) else 1, 0, "PNG/PDF written with fixed metadata")


def run(config: Mapping, cells_fn: Callable[[], S.Points], qnp_job_name: str, merfish: Mapping, out_dir: Path) -> dict:
    from . import figures_omics, figures_qnp

    checks = (merfish_checks(cells_fn(), merfish) + qnp_checks(config, qnp_job_name) + qnp_pyramid_checks(out_dir)
              + neuroglancer_checks(out_dir))
    checks.append(determinism_checks(lambda d: figures_qnp.figure_a(config, qnp_job_name, d), "QNP figure A"))
    checks.append(determinism_checks(lambda d: figures_omics.figure_a(cells_fn(), d), "MERFISH figure A"))
    report = {"checks": checks, "summary": {"n": len(checks), "passed": sum(c["passed"] for c in checks),
                                            "all_passed": all(c["passed"] for c in checks),
                                            "failed": [c["check"] for c in checks if not c["passed"]]}}
    (out_dir / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def read_particle_vtk(path):
    lines = Path(path).read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("POINTS "))
    count = int(lines[start].split()[1])
    points = np.loadtxt(lines[start + 1:start + 1 + count]).reshape(count, 3)
    arrays = {}
    for i, line in enumerate(lines):
        if line.startswith("SCALARS "):
            arrays[line.split()[1]] = np.loadtxt(lines[i + 2:i + 2 + count]).reshape(count)
    if not np.isfinite(points).all() or not all(np.isfinite(v).all() for v in arrays.values()):
        raise ValueError(f"Non-finite VTK data in {path}")
    return points, arrays


def validate_xiv_vtks(out_dir, atlas, target, convention, get_entropy, files):
    checks = {}
    for filename in (name for name in files if name.endswith(".vtk")):
        points, arrays = read_particle_vtk(out_dir / filename)
        checks[filename] = {"particles": len(points), "fields": list(arrays), "finite": True}
        if filename.startswith("original"):
            continue
        state = target if filename.startswith("target") else atlas
        key = "Td" if filename.startswith("target") else "D"
        expected = state[key].numpy()
        if expected.shape[1] == 2:
            expected = np.column_stack([expected, np.zeros(len(expected))])
        if convention == "xmodmap_yxz":
            expected = expected[:, [1, 0, 2]]
        np.testing.assert_allclose(points, expected, rtol=0, atol=1e-6)
        if "deformationSummary" in filename:
            mass = state["nu_Td" if key == "Td" else "nu_Dpi"].numpy()
            np.testing.assert_allclose(arrays["Entropy"], get_entropy(mass), rtol=1e-6, atol=1e-8)
            np.testing.assert_allclose(arrays["Jacobian"], state["mass_ratio_proxy"], rtol=1e-6, atol=1e-8)
            np.testing.assert_array_equal(arrays["Jacobian"], arrays["MassRatioProxy"])
            np.testing.assert_allclose(arrays["Weight_TargetFeatures"], mass.sum(-1), rtol=1e-6, atol=1e-8)
            np.testing.assert_array_equal(arrays["MaxVal_TargetFeatures"], mass.argmax(-1) + 1)
            if key == "D":
                np.testing.assert_allclose(arrays["Weight_AtlasFeatures"], state["nu_D"].numpy().sum(-1),
                                           rtol=1e-6, atol=1e-8)
                np.testing.assert_array_equal(arrays["MaxVal_AtlasFeatures"], state["nu_D"].numpy().argmax(-1) + 1)
    return checks
