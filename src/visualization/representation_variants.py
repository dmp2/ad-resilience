"""Render supported variants from saved products with --stage render-complete.

QNP anatomy is fixed to production for visualization; OMICS uses its two saved
fits. Legacy experimental stages remain separate and are never called by rendering.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from . import scene as S
from . import figures_omics as O
from . import figures_qnp as Q
from .cli import OUT, QNP_JOB, MERFISH, config, merfish_cells
from .ng_scenes import HYPOTHESES

LOG = OUT / "logs/representation_variants"
LIMIT = 4_000_000


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def read(path, feature):
    with np.load(path, allow_pickle=False) as a:
        return a["X"].astype(float), a[feature].astype(float)


def inventory(path):
    with np.load(path, allow_pickle=False) as a:
        x = a["X"] if "X" in a else a["Z"]
        features = {k: {"shape": list(a[k].shape), "mass": a[k].sum(0).tolist(),
                        "finite": bool(np.isfinite(a[k]).all()), "minimum": float(a[k].min())}
                    for k in ("nu_Sub", "nu_All", "nu_bin", "nu_Z") if k in a}
        return {"path": str(path), "sha256": digest(path), "count": len(x),
                "dimension": x.shape[1], "planar_z0": bool(np.all(x[:, 2] == 0)),
                "xy_bounds": [x[:, :2].min(0).tolist(), x[:, :2].max(0).tolist()],
                "finite_coordinates": bool(np.isfinite(x).all()), "features": features,
                "self_pairs": len(x)**2, "one_float64_matrix_mib": len(x)**2 * 8 / 2**20,
                "metadata": json.loads(str(a["metadata_json"])) if "metadata_json" in a else None}


def preflight():
    job = S.qnp_job(config(), QNP_JOB)
    paths = list(job["_particle_dir"].glob("*classes*um.npz")) + list(job["_particle_dir"].glob("*od_bins*um.npz"))
    for _, fit, _ in HYPOTHESES.values():
        m = json.loads((S.PROJECT / fit / "input_manifest.json").read_text())
        paths += [Path(m[k]) for k in ("source_particles", "target_particles")]
    m = json.loads((S.PROJECT / MERFISH["xiv_fit"] / "input_manifest.json").read_text())
    paths += [Path(m[k]) for k in ("source_particles", "target_particles")]
    paths += [Path(m["source_metadata"]["source_particles"])]
    records = [inventory(p) for p in sorted(set(paths))]
    cells = merfish_cells()
    bins = {str(b): len(S.bin_merfish(cells, b).xy) for b in O.BINS_MM}
    report = {"archives": records, "omics_cells": len(cells.xy), "omics_bins": bins,
              "omics_classes": list(cells.categories["class"].names), "pair_limit": LIMIT,
              "kernel": "exp(-d2/sigma_mm**2)", "sigma_mm": 1., "outer_steps": 30,
              "bounded_runs": "QNP 500->1000 um (all six stains, 2 or 8 features); OMICS 250->500/1000 um (7 classes)",
              "largest_launched_matrix_pairs": 1844**2,
              "memory_note": "One matrix <=25.95 MiB; multiple matrices/autograd/runtime required. Prior NeuN pilot peak RSS 699.91 MiB. Four CPU threads; sequential fits.",
              "registrations_launched": False,
              "existing_figures": [str(p.relative_to(OUT)) for p in sorted(OUT.glob("*/fig*.png"))],
              "existing_scenes": [str(p.relative_to(OUT)) for p in sorted(OUT.glob("*/neuroglancer/*.state.json"))]}
    dump(LOG / "preflight.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "archives"}, indent=2), flush=True)
    return report


def approximate(x, nx, z, nz, dest, metadata, pilot=None):
    from preprocess.particle_approximation import optimize_particles, varifold_squared

    assert max(len(x)**2, len(z)**2, len(x)*len(z)) <= LIMIT
    assert x.shape[1] == z.shape[1] == 3 and np.all(x[:, 2] == 0) and np.all(z[:, 2] == 0)
    start = time.perf_counter()
    def error(a, na):
        with torch.no_grad():
            return float(varifold_squared(*[torch.tensor(v, dtype=torch.float64) for v in (x, nx, a, na)], 1.))
    initial = error(z, nz)
    if pilot:
        metrics = json.loads((pilot.parent / "metrics.json").read_text())
        assert metadata["input_sha256"] == metrics["input_sha256"], "Pilot inputs changed"
        with np.load(pilot) as a:
            zz, nn = a["X"], a["nu_bin"]
        seconds = metrics["representations"]["joint"]["optimization_seconds_including_shared_warmup"]
        history = metrics["representations"]["joint"]["objective_history"]
        provenance = str(pilot.relative_to(S.PROJECT))
    else:
        r = optimize_particles(x, nx, z, nz, sigma_mm=1., optimize_positions=True, outer_steps=30)
        zz, nn, seconds, history = r.X, r.nu, r.seconds, r.objective_history
        provenance = "preprocess.particle_approximation.optimize_particles; 30 outer steps, joint XY/mass"
    final = error(zz, nn)
    assert zz.shape == z.shape and nn.shape == nz.shape
    assert np.isfinite(zz).all() and np.isfinite(nn).all() and np.all(nn >= 0) and np.all(zz[:, 2] == 0)
    assert final < initial, (initial, final)
    assert max(np.diff(history), default=0) < 1e-8
    ref = nx.sum(0); delta = nn.sum(0) - ref
    info = {**metadata, "count": len(zz), "reference_count": len(x), "sigma_mm": 1.,
            "compression_method": "image_varifold", "kernel": "exp(-d2/sigma_mm**2)",
            "optimization_provenance": provenance, "registration_provenance": "none; native-frame compression only",
            "initial_squared_error": initial, "final_squared_error": final,
            "error_reduction_percent": 100*(1-final/initial),
            "reference_self_norm": error(np.zeros((1, 3)), np.zeros((1, nx.shape[1]))),
            "total_mass": float(nn.sum()), "reference_total_mass": float(nx.sum()),
            "total_mass_deviation": float(delta.sum()), "per_feature_mass_deviation": delta.tolist(),
            "total_mass_deviation_percent": float(100*delta.sum()/nx.sum()),
            "optimization_seconds": seconds, "current_evaluation_and_run_seconds": time.perf_counter()-start,
            "objective_history": history, "finite": True, "nonnegative": True, "z_zero": True,
            "mass_conservation_imposed": False, "converged_optimum_claimed": False}
    dest.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest / (metadata["name"] + "_reference.npz"), X=x, nu=nx, Z0=z, nu_Z0=nz)
    np.savez_compressed(dest / (metadata["name"] + ".npz"), X=zz, nu=nn, metadata_json=json.dumps(info))
    dump(dest / (metadata["name"] + ".json"), info)
    print(metadata["name"], f"{initial:.6g} -> {final:.6g}", flush=True)
    return zz, nn, info


def build():
    if not (LOG / "preflight.json").exists():
        raise SystemExit("Run --stage preflight first")
    torch.set_num_threads(4)
    cfg, cells = config(), merfish_cells()
    job = S.qnp_job(cfg, QNP_JOB)
    summary = {"baseline": [], "approximations": []}
    # Complete baseline first, one dataset at a time; no original is overwritten.
    for modality, module, args in (("qnp", Q, (cfg, QNP_JOB)), ("omics", O, (cells,))):
        for family in "abc":
            start = time.perf_counter()
            paths = getattr(module, "figure_" + family)(*args, OUT / modality / "baseline")
            summary["baseline"].append({"modality": modality, "family": family,
                "seconds": time.perf_counter()-start,
                "files": [{"path": str(p.relative_to(S.PROJECT)), "sha256": digest(p),
                           "equals_original": digest(p) == digest(OUT / modality / p.name)} for p in paths]})
            print("baseline", modality, family, flush=True)
    # All six stain pairs share native image XY and original column definitions.
    neun = {}
    for stain, slide in job["_slides"].items():
        for variant, suffix, key in (("iv_structural", "tissue_classes", "nu_Sub"),
                                     ("iv_modality", "od_bins", "nu_bin")):
            fine = job["_particle_dir"] / f"{slide.stem}_{suffix}_500um.npz"
            coarse = job["_particle_dir"] / f"{slide.stem}_{suffix}_1000um.npz"
            x, nx = read(fine, key); z, nz = read(coarse, key)
            if key == "nu_bin":
                with np.load(fine) as a, np.load(coarse) as b:
                    assert np.array_equal(a["bin_centers"], b["bin_centers"])
                    assert np.array_equal(a["bin_edges"], b["bin_edges"])
            meta = {"name": f"{stain}_1000um", "variant": variant,
                    "compression_features": "gray/white soft area" if key == "nu_Sub" else "8 normalized OD interpolation-node area masses",
                    "retained_features": key, "downstream_registration_features": None,
                    "feature_labels": ["gray_matter", "white_matter"] if key == "nu_Sub" else list(np.linspace(0, 1, 8)),
                    "mass_units": "mm^2", "frame": S.slide_frame(slide),
                    "reference": str(fine), "initialization": str(coarse), "input_sha256": [digest(fine), digest(coarse)]}
            pilot = S.PROJECT / "results/analysis/neun_particle_approximation/joint_experimental.npz" if stain == "NEUN" and key == "nu_bin" else None
            zz, nn, info = approximate(x, nx, z, nz, OUT / "qnp" / variant / "numerical", meta, pilot)
            summary["approximations"].append(info)
            if stain == "NEUN":
                area = nn.sum(1)
                reference = S.load_qnp_particles(job["_particle_dir"] / f"{slide.stem}_tissue_classes_500um.npz", S.slide_frame(slide))
                scalar = "white_fraction" if key == "nu_Sub" else "normalized_od_mean"
                values = nn[:, 1]/np.maximum(area, 1e-12) if key == "nu_Sub" else nn @ np.linspace(0,1,8)/np.maximum(area,1e-12)
                points = S.Points("optimized NeuN", S.slide_frame(slide), zz, "derived", info["optimization_provenance"],
                     scalars={"area_mm2": S.Scalar(area,"mm^2"), scalar: S.Scalar(values,"fraction" if key == "nu_Sub" else "normalized OD")}, meta=info)
                if key == "nu_bin":
                    reference.scalars[scalar] = S.Scalar(nx @ np.linspace(0,1,8)/np.maximum(nx.sum(1),1e-12), "normalized OD")
                neun[variant] = {"variant": variant, "points": {"500um": reference, "1000um": points}, "scalar": scalar,
                    "color_label": "b: p(WM); c,d: white area / total area" if key == "nu_Sub" else "b: p(WM); c,d: mean normalized OD (appearance only)",
                    "caption": [f"PARTIAL Fig A, {variant}; native NeuN slide mm, x right / y down. a,b: unchanged measured/derived inputs.",
                       "c: deterministic 500 µm reference, not optimized (no compatible finer archive within the dense guard).",
                       f"d: optimized XY and {meta['compression_features']}; 450 points; sigma = 1 mm. Circle area = mass, not parcel area.",
                       f"Total mass deviation {info['total_mass_deviation_percent']:+.4f}%; no registration performed. OD is normalized appearance, not antigen abundance."]}
    for variant, rep in neun.items():
        Q.figure_a(cfg, QNP_JOB, OUT / "qnp" / variant, representation=rep)
    # Seven classes, count units: scalar conversion to area is not part of this visualization.
    fine = S.bin_merfish(cells, .25)
    x = np.column_stack([fine.xy, np.zeros(len(fine.xy))]); nx = fine.meta["nu_class"]
    rep = {"points": {}, "mass_refs": {}, "caption": [
        "PARTIAL Fig A, IV-modality: seven existing molecular cell classes; merfish_section_mm, x right / y down.",
        "500/1000 µm counts matched to baseline; reference = deterministic 250 µm class-count masses, not individual cells.",
        "Optimized XY and nonnegative class masses; sigma = 1 mm. Circle area = class-count mass, opacity = dominant group fraction.",
        "Fine levels exceed the dense guard; genes at optimized positions are unavailable. No registration or anatomical improvement is claimed."]}
    for b in (.5, 1.):
        original = S.bin_merfish(cells, b)
        z = np.column_stack([original.xy, np.zeros(len(original.xy))]); nz = original.meta["nu_class"]
        zz, nn, info = approximate(x,nx,z,nz,OUT / "omics/iv_modality/numerical",
            {"name": f"classes_{int(b*1000)}um", "variant": "iv_modality", "compression_features": list(fine.meta["class_names"]),
             "retained_features": "7 class-count masses", "downstream_registration_features": None,
             "mass_units": "cells (fractional optimized class-count masses)", "frame": cells.frame,
             "reference": "scene.bin_merfish(cells, 0.25)", "initialization": f"scene.bin_merfish(cells, {b})",
             "input_sha256": [digest(S.PROJECT / MERFISH["h5ad"])], "reference_is_cells": False})
        total = nn.sum(1)
        rep["points"][b] = S.Points("optimized molecular classes", cells.frame, zz, "derived", info["optimization_provenance"],
            scalars={"cells_per_bin": S.Scalar(total,"fractional cell-count mass")},
            meta={**info,"nu_class":nn,"class_names":fine.meta["class_names"]})
        rep["mass_refs"][b] = float(np.percentile(original.scalars["cells_per_bin"].values,95))
        summary["approximations"].append(info)
    O.figure_a(cells, OUT / "omics/iv_modality", representation=rep)
    # Recompute the existing independent endpoints only for their baseline fits.
    from .validate import merfish_checks, qnp_checks
    checks = merfish_checks(cells, MERFISH) + qnp_checks(cfg, QNP_JOB)
    dump(LOG / "baseline_validation.json", {"checks":checks, "all_passed":all(c["passed"] for c in checks)})
    dump(LOG / "numerical_summary.json", summary)
    print("Built bounded cases; unavailable full-suite dependencies are documented in REPORT.md", flush=True)


def baseline_scenes():
    """Re-emit existing baseline states and render them against compatible original assets."""
    from . import ng_export as nx
    checks = []
    for old in sorted(OUT.glob("*/neuroglancer/*.state.json")):
        modality = old.relative_to(OUT).parts[0]
        scene = old.name.removesuffix(".state.json")
        state = json.loads(old.read_text())
        manifest = json.loads(old.with_name(scene + ".manifest.json").read_text())
        manifest["representation_variant"] = "baseline"
        manifest["asset_provenance"] = "Original compatible baseline assets; numerical inputs and transforms unchanged"
        bundle = OUT / modality / "baseline/neuroglancer"
        nx.write_scene(bundle, scene, state, manifest)
        report = nx.verify_scene(bundle, scene, port=8017, shot=True)
        checks.append(report)
        print(scene, report["all_sources_ok"], report.get("screenshot", {}).get("ok"), flush=True)
    dump(LOG / "baseline_scene_validation.json", checks)


def omics_inputs():
    """Prepare a bounded strip case with the baseline area calibration held fixed."""
    import copy
    import yaml
    from prepare_atlas_plane_and_section import _section_labels, _compress_section
    from preprocess.particle_approximation import optimize_particles, varifold_squared

    torch.set_num_threads(4)
    cfg = config()
    spec = cfg["atlas_section_inputs"]["ahrav2_mtg_coronal_L_to_1194111462"]["section"]
    _, labels = _section_labels(spec, cfg)
    raw = S.resolve(cfg, spec["strip_npz"])
    with np.load(raw) as a:
        xraw, presence = a["Z"].astype(float)*.001, (a["nu_Z"] > 0).astype(float)
    x, nx, _, features, fine_meta = _compress_section(xraw, presence, labels,
          cfg["feature_groupings"][spec["feature_grouping"]], .25)
    target = S.resolve(cfg, spec["strip_output_npz"])
    z, nz = read(target, "nu_Sub")
    with np.load(target) as a:
        old_meta = json.loads(str(a["metadata_json"]))
    # Occupancy-based a_cell is scale-dependent: preserve the baseline calibration.
    nx *= old_meta["a_cell_mm2"] / fine_meta["a_cell_mm2"]
    dest = OUT / "omics/iv_modality/numerical"
    zz, nn, info = approximate(x,nx,z,nz,dest,
        {"name":"depth_strip_500um", "variant":"iv_modality", "frame":"seaad_section_mm",
         "compression_features":list(features.coarse_names), "retained_features":"nu_Sub (7 class area masses)",
         "downstream_registration_features":"nu_Sub with learned 1x7 pi_ST",
         "mass_units":"mm^2", "a_cell_mm2":old_meta["a_cell_mm2"],
         "reference":"existing strip cells binned at 250 um; fixed baseline a_cell calibration",
         "initialization":str(target), "input_sha256":[digest(raw),digest(target)]})
    target_out = dest / "registration_target.npz"
    np.savez_compressed(target_out, X=zz, nu_Sub=nn,
        coarse_feature_names=np.array(features.coarse_names),
        metadata_json=json.dumps({**old_meta, **info, "coordinate_space":"seaad_section_mm"}))
    original_source = S.resolve(cfg,cfg["registrations"]["ahrav2_mtg_plane_to_strip_1194111462"]["source_particles"])
    ax, an = read(original_source,"nu_Sub")
    # At the original 1333-particle count, the voxel-plane input is already exact.
    ar = optimize_particles(ax,an,ax,an,sigma_mm=1.,optimize_positions=True,outer_steps=1)
    with torch.no_grad():
        err = float(varifold_squared(*[torch.tensor(v,dtype=torch.float64) for v in (ax,an,ar.X,ar.nu)],1.))
    assert abs(err)<1e-8 and np.array_equal(ar.X,ax) and np.array_equal(ar.nu,an)
    source_out = dest / "registration_atlas_exact.npz"
    with np.load(original_source) as a:
        arrays = {k:a[k] for k in a.files}
    ameta = json.loads(str(arrays["metadata_json"]))
    ameta["variant_compression"] = {"variant":"iv_modality", "reference":str(original_source),
        "features":"one anatomical MTG area channel; no molecular atlas data",
        "count":len(ax), "squared_error":err, "positions_changed":False,
        "reason":"already exact voxel-plane measure at baseline count; joint optimizer verified no-op",
        "runtime_seconds":ar.seconds, "source_sha256":digest(original_source)}
    arrays["metadata_json"] = json.dumps(ameta)
    np.savez_compressed(source_out,**arrays)
    dump(dest / "registration_atlas_exact.json",ameta["variant_compression"])
    outcfg = copy.deepcopy(cfg)
    outcfg["project"]["root"] = str(S.PROJECT)
    rec = outcfg["registrations"]["ahrav2_mtg_plane_to_strip_1194111462"]
    rec["source_particles"] = str(source_out)
    rec["target_particles"] = str(target_out)
    rec["output_dir"] = str(OUT / "omics/iv_modality/registration")
    for k in ("_project_root","_config_path"):
        outcfg.pop(k,None)
    (LOG / "omics_registration.yaml").write_text(yaml.safe_dump(outcfg,sort_keys=False))
    dump(LOG / "omics_registration_preflight.json",{
        "source_count":len(ax),"target_count":len(zz),"target_reference_count":len(x),
        "source_features":1,"target_features":7,"mapping":"existing learned nonnegative pi_ST, 1x7",
        "source_self_pairs":len(ax)**2,"source_target_pairs":len(ax)*len(zz),
        "model":"unchanged cross-modal boundary model, float32 cuda:0; 10 shooting steps",
        "first_run":"support-check then 2 optimizer steps; assess before 150-step fit",
        "qc":"same provider strip cells, original depth and frame; same scores/plausibility endpoints",
        "a_cell_mm2":old_meta["a_cell_mm2"],"3d_volume_optimized":False})
    print("Prepared OMICS 1333->50 fit with independent strip calibration",flush=True)


def omics_correspondence():
    fit = OUT / "omics/iv_modality/registration/init_r000_f0"
    if not (fit / "COMPLETED.json").exists():
        raise SystemExit("Complete the supported registration first")
    xiv = S.load_merfish_xiv(fit)
    from .validate import _vtk_points
    from preprocess.particle_utils import varifold_score_mm
    completed = json.loads((fit / "COMPLETED.json").read_text())
    assert completed["level1_passed"] and completed["steps"] == 150
    chain = xiv["chain"]
    manifest = xiv["manifest"]
    assert Path(manifest["source_particles"]).parent == OUT / "omics/iv_modality/numerical"
    assert Path(manifest["target_particles"]).parent == OUT / "omics/iv_modality/numerical"
    src, _ = read(Path(manifest["source_particles"]), "nu_Sub")
    tgt, _ = read(Path(manifest["target_particles"]), "nu_Sub")
    errors = {
        "initial_atlas_vtk_mm": float(np.abs(_vtk_points(fit / "originalAtlas.vtk") - (src[:,:2]-chain["source_center_plane_mm"])).max()),
        "deformed_atlas_vtk_mm": float(np.abs(_vtk_points(fit / "atlas_deformationSummary.vtk") - (xiv["deformed"].xy-chain["target_center_section_mm"])).max()),
        "target_vtk_mm": float(np.abs(_vtk_points(fit / "originalTarget.vtk") - (tgt[:,:2]-chain["target_center_section_mm"])).max())}
    assert max(errors.values()) < 1e-5
    baseline = S.PROJECT / MERFISH["xiv_fit"]
    bm = json.loads((baseline / "input_manifest.json").read_text())
    bx, bn = read(Path(bm["target_particles"]), "nu_Sub")
    endpoints = {}
    for label, directory in (("baseline",baseline),("iv_modality",fit)):
        b = torch.load(directory / "atlas_deformationSummary.pt",map_location="cpu",weights_only=False)
        tc = json.loads((directory / "transform_chain.json").read_text())["target_center_section_mm"]
        d = b["D"].double().numpy()+tc
        n = b["nu_Dpi"].double().numpy()*b["relative_support_weight"].double().numpy().reshape(-1,1)
        endpoints[label] = {
            "total_mass_score":varifold_score_mm(d,n.sum(1,keepdims=True),bx[:,:2],bn.sum(1,keepdims=True)),
            "class_score":varifold_score_mm(d,n,bx[:,:2],bn),
            "depth_diagnostic":json.loads((directory / "plausibility.json").read_text())["depth_stratified_post_fit_diagnostic"]}
    dump(LOG / "omics_registration_validation.json",{
        "all_passed":True,"frame_errors":errors,"source_count":len(src),"target_count":len(tgt),
        "same_endpoint":str(bm["target_particles"]),"endpoint_sigmas_mm":[2.,1.,.5],"comparisons":endpoints,
        "note":"Common original 50-bin target for both scores; descriptive fit diagnostics, not independent anatomical accuracy. Depth QC uses same cells and is not independent of fit positions."})
    paths = O.figure_c(merfish_cells(),OUT / "omics/iv_modality",xiv=xiv,representation={"caption":[
        "IV-modality: frame merfish_section_mm, x right / y down. Provider polygons and labels are unchanged independent observations.",
        "New xIV fit: 50 optimized XY / seven class-area particles from a 250 µm binned strip reference; baseline a_cell fixed.",
        "Atlas: 1333 single-MTG voxel-plane particles, joint compression verified exact no-op; new learned 1x7 correspondence.",
        "Initial placement = barycentres (M = I); deformed = new D + target centre. Provisional coronal plane y = -30 mm; not registration accuracy."]})
    # The existing layout already gives the exact frame and cross-modal estimator.
    dump(LOG / "omics_correspondence_provenance.json",{
        "variant":"iv_modality","fit":str(fit),"outputs":[str(p) for p in paths],
        "atlas_compression":"verified exact no-op at 1333 voxel-plane particles",
        "target_compression":"50 XY/mass optimized particles, 7 class-area features; 250 um binned reference",
        "mapping":"newly fitted 1x7 pi_ST","baseline_registration_reused":False})




def write_completion_manifest(manifest, dest):
    """Write one auditable row per panel, including original measurement sources."""
    import csv
    job=S.qnp_job(config(),QNP_JOB)
    tables=list((S.PROJECT/'data/raw/sea-ad').glob('*/6a4579d1d67150b0e69ad044_sea-ad-quantitative-neuropathology-063026.xlsx'))
    assert len(tables)==1
    table=str(tables[0].resolve())
    manifest['assets'][table]=dict(path=table,sha256=digest(table),feature_key='MTG worksheet')
    refann=str(job['_slides']['NEUN'].with_suffix('.annotations'))
    manifest['assets'][refann]=dict(path=refann,sha256=digest(refann),feature_key='HALO XML layer polygons')
    for info in manifest['assets'].values():
        if info['path'].endswith('.npz') and info.get('feature_key') not in (None,'MTG worksheet'):
            with np.load(info['path'],allow_pickle=False)as a:
                meta=json.loads(str(a['metadata_json']))if'metadata_json'in a else {}
                info['coordinate_feature_metadata']={k:meta[k]for k in ('frame','coordinate_space','mass_units','variant',
                    'retained_features','compression_features','reference','initialization','input_sha256','a_cell_mm2')if k in meta}
                info['coordinate_array_sha256']=hashlib.sha256(a['X'].tobytes()).hexdigest()
                info['feature_array_sha256']=hashlib.sha256(a[info['feature_key']].tobytes()).hexdigest()
    state=OUT/'omics/neuroglancer/H20.33.001_1194111462_cells.manifest.json'
    ng=json.loads(state.read_text())
    provider=ng['layers']['provider layer polygons (uns/layer_polygons)']
    manifest['provider_polygon_reference']=provider
    for item in manifest['panels']:
        family,letter=item['figure'],item['panel']
        if item['dataset']=='qnp':
            if family=='A':
                item['quantity']={'a':'rgb','b':'p_wm (fixed computational surrogate)','c':'eight-node OD mean'if item['variant']=='iv_modality'else'surrogate white-area fraction',
                                  'd':'eight-node OD mean'if item['variant']=='iv_modality'else'surrogate white-area fraction'}[letter]
            elif family=='B':
                if letter in 'abcde':
                    item['inputs']=[table,*[i for i in item['inputs']if i!=S.QNP_TABLE_SOURCE]]
                    item['quantity']=dict(worksheet='MTG',donor=QNP_JOB.split('_')[0],column=Q.LAYER_MEASURES['abcde'.index(letter)][0],
                                          layers=list(S.QNP_LAYERS),painting='fixed NeuN HALO layer raster')
                elif letter=='f':
                    item['inputs']=[table]
                    item['quantity']='Original five provider layer measurements, each divided by its own maximum'
                elif letter=='m':item['quantity']='Original structural_disagreement_sd with original consensus_mask; fixed NeuN/LFB/AT8 reference'
                else:item['quantity']='Original p_wm raster'if item['variant']=='baseline'else'nu @ saved OD bin_centers / nu.sum(1); appearance only'if item['variant']=='iv_modality'else'nu[:,1] / nu.sum(1); computational structural surrogate'
            elif family=='C'and letter in 'ab':item['inputs'].append(refann)
        elif (family=='B'and letter in 'abcdef')or(family=='C'and letter=='a'):
            item['provider_polygon_geometry']=dict(manifest=str(state),source=provider['source'],
                transformation=provider['transform_to_scene'])
    assert len(manifest['figures'])==15 and len(manifest['panels'])==125
    for figure in manifest['figures']:
        figure['output_sha256']={p:digest(p)for p in figure['outputs']}
    manifest['source_sha256']={str(p):digest(p)for p in (Path(__file__),S.PROJECT/'src/visualization/figures_qnp.py',
        S.PROJECT/'src/visualization/figures_omics.py',S.PROJECT/'src/visualization/scene.py',S.PROJECT/'src/visualization/style.py')}
    dump(dest/'manifest.json',manifest)
    with (dest/'panel_manifest.csv').open('w',newline='')as f:
        fields=['dataset','variant','figure','panel','displayed_data','inputs','transformation','note','quantity']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for item in manifest['panels']:
            writer.writerow({key:json.dumps(item.get(key,''),ensure_ascii=False)if isinstance(item.get(key),(dict,list))
                             else item.get(key,'')for key in fields})


def render_complete():
    """Complete supported suites from saved products; no numerical fitting stages."""
    import copy
    import os
    from unittest.mock import patch
    from . import style as st
    from .validate import qnp_checks, merfish_checks, _vtk_points

    dest = OUT / 'completed_representations'
    dest.mkdir(parents=True, exist_ok=True)
    scratch = Path(os.environ['TMPDIR']).resolve()
    assert scratch.is_relative_to(Path.home()) and scratch.is_dir()
    cfg = config(); job = S.qnp_job(cfg, QNP_JOB)
    # Hash existing products, never write to their directories. Numerical files remain read-only.
    protected = set(job['_particle_dir'].rglob('*'))
    for root in (OUT, S.PROJECT/'results/registrations'):
        protected.update(p for p in root.rglob('*') if p.is_file() and not p.is_relative_to(dest)
                         and p.suffix in ('.npz','.pt','.json','.png','.pdf','.vtk','.csv'))
    protected_hash = {str(p):digest(p) for p in sorted(protected) if p.is_file()}
    dump(dest/'protected_before.json', protected_hash)
    manifest = {'policy':'QNP correspondence frozen to original production; OMICS uses the two existing saved fits.',
                'assets':{}, 'panels':[], 'figures':[], 'checks':[], 'scratch':str(scratch)}

    def asset(path, key=None):
        path = Path(path).resolve(); ident = str(path) + (f'::{key}' if key else '')
        if ident not in manifest['assets']:
            info = dict(path=str(path), sha256=digest(path), feature_key=key)
            if path.suffix == '.npz' and key:
                with np.load(path, allow_pickle=False) as a:
                    x, nu = a['X'], a[key]
                    assert x.ndim==2 and x.shape[1]==3 and nu.ndim==2 and len(x)==len(nu)
                    assert np.isfinite(x).all() and np.isfinite(nu).all() and (nu>=0).all() and (x[:,2]==0).all()
                    info.update(count=len(x), feature_dimension=nu.shape[1], per_feature_mass=nu.sum(0).tolist(),
                                xy_bounds=[x[:,:2].min(0).tolist(),x[:,:2].max(0).tolist()])
            manifest['assets'][ident] = info
        return ident

    def panel(dataset, variant, family, letter, kind, sources, transform, note=''):
        manifest['panels'].append(dict(dataset=dataset,variant=variant,figure=family,panel=letter,
            displayed_data=kind,inputs=sources,transformation=transform,note=note))

    transforms = S.load_qnp_transforms(job)
    tpath = asset(job['_aligned_dir']/'transforms.json')
    graphpath = asset(job['_scan_dir']/'transform_graph.json')
    ms = asset(job['_aligned_dir']/'multistain_maps_32um_local.npz')
    mapfiles = {s:asset(job['_particle_dir']/f'{slide.stem}_maps_32um.npz') for s,slide in job['_slides'].items()}
    record = json.loads(Path(tpath).read_text())
    # Freeze production graph, including exact structural membership and reliability status.
    with np.load(job['_aligned_dir']/'multistain_maps_32um_local.npz') as a:
        fixed_consensus = {'mask_sha256':hashlib.sha256(a['consensus_mask'].tobytes()).hexdigest(),
            'p_wm_sha256':hashlib.sha256(a['consensus_p_wm'].tobytes()).hexdigest()}
    manifest['fixed_qnp_reference'] = dict(transforms=tpath,graph=graphpath,
        structural_stains=[s for s,c in job['stains'].items() if c.get('structural')],
        reliability={s:t.status for s,t in transforms.items()},**fixed_consensus)
    reference_frame = S.slide_frame(job['_slides']['NEUN'])

    def qnp_points(stain, variant, tag):
        slide = job['_slides'][stain]; frame = S.slide_frame(slide)
        suffix,key = ('od_bins','nu_bin') if variant=='iv_modality' else ('tissue_classes','nu_Sub')
        native = job['_particle_dir']/f'{slide.stem}_{suffix}_{tag}.npz'
        optimized = variant!='baseline' and tag=='1000um'
        path = OUT/'qnp'/variant/'numerical'/f'{stain}_1000um.npz' if optimized else native
        source = asset(path,'nu' if optimized else key)
        with np.load(path,allow_pickle=False) as a:
            x = a['X']; nu = a['nu' if optimized else key]
            meta = json.loads(str(a['metadata_json']))
        expected_channels = 8 if variant=='iv_modality' else 2
        assert nu.shape[1]==expected_channels
        if optimized:
            assert meta['frame']==frame and meta['retained_features']==key and meta['variant']==variant
            assert meta['input_sha256']==[digest(meta['reference']),digest(meta['initialization'])]
            assert meta['count']==len(x) and meta['compression_method']=='image_varifold'
        else:
            assert meta['coordinate_space']=='qnp_slide_mm'
        area=nu.sum(1)
        scalar='normalized_od_mean' if variant=='iv_modality' else 'white_fraction'
        if variant=='iv_modality':
            fine=job['_particle_dir']/f'{slide.stem}_od_bins_500um.npz'
            with np.load(fine) as a:nodes=a['bin_centers']
            if optimized:np.testing.assert_array_equal(nodes,meta['feature_labels'])
            values=nu@nodes/np.maximum(area,1e-12)
        else:values=nu[:,1]/np.maximum(area,1e-12)
        points=S.Points(f'{stain} {variant}',frame,x,'derived',source,
            scalars={'area_mm2':S.Scalar(area,'mm^2'),scalar:S.Scalar(values,'fraction')},
            meta={**meta,'source_archive':str(path),'source_feature_key':'nu' if optimized else key})
        np.testing.assert_array_equal(points.xy,x[:,:2])
        np.testing.assert_array_equal(points.scalars['area_mm2'].values,nu.sum(1))
        return points, source

    original_save = st.save
    layouts={}
    def render(dataset,variant,family,call):
        def capture(fig,stem,formats=('png','pdf')):
            geometry=dict(size_inches=fig.get_size_inches().tolist(),
                axes_bounds=[ax.get_position().bounds for ax in fig.axes])
            signature=(dataset,family)
            if variant=='baseline':layouts[signature]=geometry
            else:
                np.testing.assert_allclose(geometry['size_inches'],layouts[signature]['size_inches'],atol=0,rtol=0)
                np.testing.assert_allclose(geometry['axes_bounds'],layouts[signature]['axes_bounds'],atol=1e-12,rtol=0)
            # Compare spatial axes limits where fixed reference geometry is required.
            info=dict(dataset=dataset,variant=variant,family=family,geometry=geometry,
                axes_limits=[dict(x=ax.get_xlim(),y=ax.get_ylim())for ax in fig.axes],
                color_scales=[dict(cmap=a.get_cmap().name,vmin=a.norm.vmin,vmax=a.norm.vmax)
                              for ax in fig.axes for a in [*ax.images,*ax.collections] if a.get_array() is not None])
            paths=original_save(fig,stem,formats)
            info['outputs']=[str(p)for p in paths]
            if variant=='baseline':
                info['byte_identity_to_original']={str(p):digest(p)==digest(OUT/dataset/p.name) for p in paths}
                if not (dataset=='qnp' and family=='B'):
                    assert all(info['byte_identity_to_original'].values()),info
                else:
                    info['label_only_change']='Shorten panel m title to SD: NeuN/LFB/AT8 to prevent inherited right-edge text clipping.'
            manifest['figures'].append(info)
            return paths
        with patch.object(st,'save',capture):call()
        dump(dest/'manifest.json',manifest)
        print(dataset,variant,family,'complete',flush=True)

    # First complete every QNP condition using exactly the production anatomy.
    for variant in ('baseline','iv_structural','iv_modality'):
        target=dest/'qnp'/variant
        rep=None
        refs={};p1000={}
        for s in job['_slides']:
            points,source=qnp_points(s,variant,'1000um');refs[s]=source;p1000[s]=points
        if variant!='baseline':
            fine,finesource=qnp_points('NEUN',variant,'500um')
            aligned={}
            for s,points in p1000.items():
                moved=copy.deepcopy(points)
                if not transforms[s].status.startswith('unreliable'):
                    moved.xy=transforms[s].apply(points.xy);moved.frame='qnp_local_mm'
                    # Mass channels are carried unchanged by these rigid transforms.
                    np.testing.assert_allclose(transforms[s].inverse().apply(moved.xy),points.xy,atol=1e-12)
                else:moved.meta['display_frame_note']='Native: production edge unreliable; excluded from consensus.'
                np.testing.assert_array_equal(moved.scalars['area_mm2'].values,points.scalars['area_mm2'].values)
                aligned[s]=moved
            feature='normalized_od_mean' if variant=='iv_modality' else 'white_fraction'
            label='normalized OD mean (stain appearance)' if variant=='iv_modality' else 'surrogate white-area / total area'
            rep=dict(variant=variant,scalar=feature,points={'500um':fine,'1000um':p1000['NEUN']},aligned_points=aligned,
                titles={'500um':f'{len(fine.xy):,} particles, 500 µm','1000um':f'{len(p1000["NEUN"].xy):,} particles, 1000 µm'},
                statuses={'500um':'deterministic reference','1000um':'optimized OD masses' if variant=='iv_modality' else 'optimized surrogate masses'},
                colorbar_y=.17,color_label=f'b: fixed structural reference; c,d: {label}',
                color_label_b='Stain-specific normalized OD mean (appearance only)' if variant=='iv_modality' else 'Surrogate white-area / total area; fixed anatomy',
                caption=[f'{variant}: NeuN native slide mm, x right / y down. a,b: original RGB and fixed computational structural reference.',
                    'c: unchanged deterministic 500 µm reference. d: existing optimized 1000 µm XY/masses; circles show mass, not aggregation blocks.',
                    'OD is stain-specific appearance; the structural surrogate is not a validated biological probability. No correspondence was re-estimated.'],
                caption_b=[f'{variant}: a-f retain original provider layer measurements and anatomical labels; no measurements assigned to moved particles.',
                    'g-l: existing optimized 1000 µm particles, circles proportional to mass; fixed production rigid transforms. GFAP shown native (unreliable).',
                    'm: original NeuN/LFB/AT8 structural-consensus SD, a fixed reference; no OD consensus or new gray/white features are computed.',
                    'OD colours retain stain-specific meaning. Structural colours are computational surrogate fractions, not validated biological probabilities.'],
                caption_c=[f'{variant}: all panels retain fixed production anatomical correspondence; A/B show the changed representation, C does not refit it.',
                    'a,b: original HALO observations / production rigid transforms. c,d: both saved atlas-pose hypotheses and original xIV-pulled ribbon observations.',
                    'QNP local mm (x right / y down) in a,b; original in-plane atlas axes and physical scales in c,d. No new atlas or feature mapping.'])
        else:finesource=asset(job['_particle_dir']/f'{job["_slides"]["NEUN"].stem}_tissue_classes_500um.npz','nu_Sub')
        # Figure A manifest, including deterministic fine reference under optimized conditions.
        panel('qnp',variant,'A','a','measured RGB',[mapfiles['NEUN']],'identity native NeuN')
        panel('qnp',variant,'A','b','fixed computational structural reference',[mapfiles['NEUN']],'identity native NeuN')
        panel('qnp',variant,'A','c','deterministic reference',[finesource],'identity native NeuN')
        panel('qnp',variant,'A','d','deterministic' if variant=='baseline' else 'optimized',[refs['NEUN']],'identity native NeuN')
        render('qnp',variant,'A',lambda:Q.figure_a(cfg,QNP_JOB,target,representation=rep))
        for letter in 'abcdef':panel('qnp',variant,'B',letter,'original measured provider layer quantity',[mapfiles['NEUN'],S.QNP_TABLE_SOURCE],
            'original NeuN HALO layer raster', 'Provider layer means; no within-layer inference.')
        with np.load(job['_aligned_dir']/'multistain_maps_32um_local.npz')as a:stains=[str(s)for s in a['stains']]
        for letter,s in zip('ghijkl',stains):
            unreliable=transforms[s].status.startswith('unreliable')
            transform={'source':tpath,'edge':s,'matrix':transforms[s].matrix.tolist(),'status':transforms[s].status}
            if unreliable and variant!='baseline':transform={'identity_native':True,'excluded_production_edge':s,'status':transforms[s].status}
            panel('qnp',variant,'B',letter,'fixed original surrogate raster' if variant=='baseline' else 'optimized stain-specific OD' if variant=='iv_modality' else 'optimized structural surrogate',
                  [ms] if variant=='baseline' else [refs[s]],transform)
        panel('qnp',variant,'B','m','fixed reference consensus SD',[ms],{'source':tpath,'members':['NEUN','LFB','AT']})
        render('qnp',variant,'B',lambda:Q.figure_b(cfg,QNP_JOB,target,representation=rep,consensus_title="SD: NeuN/LFB/AT8"))
        for letter,s in zip('ab',('ASYN','I6')):
            ann=asset(job['_slides'][s].with_suffix('.annotations'))
            panel('qnp',variant,'C',letter,'fixed original HALO observations',[ann,tpath],transforms[s].describe(),'Production fit, including ASYN: no experimental OD transform.')
        for letter,(key,(plane,fit,_))in zip('cd',HYPOTHESES.items()):
            fit=S.PROJECT/fit
            sources=[asset(job['_scan_dir']/f'export_{plane}.json'),asset(fit/'qc_points_pulled_back.npz'),asset(fit/'anatomical_evaluation.json'),asset(fit/'input_manifest.json')]
            panel('qnp',variant,'C',letter,'fixed reference atlas pose / saved xIV-mapped observations',sources,
                  {'production_graph':graphpath,'hypothesis':key,'fit':str(fit)},'Existing baseline estimate, not a variant-specific fitted correspondence.')
        render('qnp',variant,'C',lambda:Q.figure_c(cfg,QNP_JOB,target,representation=rep))
    manifest['checks'] += qnp_checks(cfg,QNP_JOB)
    dump(dest/'manifest.json',manifest)
    print('All nine QNP figures complete; loading original OMICS measurements.',flush=True)

    # Then OMICS: original measurements in unsupported scales and every gene panel.
    cells=merfish_cells()
    cellsource=str(S.PROJECT/MERFISH['h5ad'])
    manifest['measured_omics_input']=dict(path=cellsource,size_bytes=Path(cellsource).stat().st_size,
        mtime_ns=Path(cellsource).stat().st_mtime_ns,selected_rows=len(cells.xy),raw_count_checks=cells.meta['x_check'],
        coordinate_sha256=hashlib.sha256(cells.xy.tobytes()).hexdigest(),
        gene_sha256={g:hashlib.sha256(cells.scalars[f'lognorm:{g}'].values.tobytes()).hexdigest()for g in O.GENES_B})
    references={b:S.bin_merfish(cells,b,[f'lognorm:{O.GENE_A}'])for b in O.BINS_MM}
    for variant in ('baseline','iv_modality'):
        target=dest/'omics'/variant
        rep=None
        sources={b:dict(measured_cells=cellsource,aggregation=f'scene.bin_merfish(cells, {b})',feature='nu_class')for b in O.BINS_MM}
        if variant=='iv_modality':
            points={}
            for b in (.5,1.):
                path=OUT/'omics/iv_modality/numerical'/f'classes_{int(b*1000)}um.npz';source=asset(path,'nu')
                x,nu=read(path,'nu')
                with np.load(path)as a:meta=json.loads(str(a['metadata_json']))
                assert meta['frame']==cells.frame and meta['compression_method']=='image_varifold' and nu.shape==(len(x),7)
                assert meta['compression_features']==list(references[b].meta['class_names'])
                points[b]=S.Points('optimized molecular class masses',cells.frame,x,'derived',source,
                    scalars={'cells_per_bin':S.Scalar(nu.sum(1),'fractional cell-count mass')},
                    meta={**meta,'nu_class':nu,'class_names':references[b].meta['class_names']})
                assert f'lognorm:{O.GENE_A}' not in points[b].scalars
                sources[b]=source
            rep=dict(points=points,mass_refs={},caption=[
                'IV modality: a,f measured cells. b,c deterministic 100/250 µm class reference; d,e existing optimized 500/1000 µm class masses.',
                'g-j: original deterministic CUX2 bin means at original positions; no gene values copied or interpolated onto moved particles.',
                'Circles encode optimized class-count mass; squares encode deterministic bin counts. Original section frame, palettes and physical scales.'],
                caption_b=['a-l: original measured cell-level genes, blank controls and provider-layer summaries; no gene reconstruction from compressed classes.',
                    'm-r: original deterministic 250 µm class fractions (fixed reference); no supported optimization exists at this fine scale.'])
        for row,letters in [('class','abcde'),('gene','fghij')]:
            for j,letter in enumerate(letters):
                b=O.BINS_MM[j-1]if j else None
                optimized=variant=='iv_modality' and row=='class' and b in (.5,1.)
                panel('omics',variant,'A',letter,'measured cells' if j==0 else 'optimized class masses' if optimized else 'deterministic measured gene mean' if row=='gene' else 'deterministic class reference',
                    [sources[b]]if optimized else [dict(measured_cells=cellsource,aggregation=None if j==0 else f'scene.bin_merfish(cells, {b})',
                        feature='nu_class'if row=='class'else f'lognorm:{O.GENE_A}')], 'original merfish_section_mm; no fitted spatial transform')
        render('omics',variant,'A',lambda:O.figure_a(cells,target,representation=rep))
        for letter,g in zip('abcdef',O.GENES_B):panel('omics',variant,'B',letter,'measured gene/control',[dict(measured_cells=cellsource,feature=f'lognorm:{g}')],'original section / exact provider polygon frame')
        for letter,g in zip('ghijkl',O.GENES_B):panel('omics',variant,'B',letter,'measured layer summaries',[dict(measured_cells=cellsource,feature=f'lognorm:{g}')],'provider layer labels; descriptive IQR/median')
        for letter,cls in zip('mnopqr',O.FRACTIONS_B):panel('omics',variant,'B',letter,'deterministic 250 µm class reference',[dict(measured_cells=cellsource,aggregation='scene.bin_merfish(cells, .25)',feature=f'fraction:{cls}')],'original section coordinates')
        render('omics',variant,'B',lambda:O.figure_b(cells,target,representation=rep))
        fit=S.PROJECT/MERFISH['xiv_fit']if variant=='baseline'else OUT/'omics/iv_modality/registration/init_r000_f0'
        completed=json.loads((fit/'COMPLETED.json').read_text());assert completed['level1_passed']and completed['steps']==150
        xiv=S.load_merfish_xiv(fit);fm=xiv['manifest'];chain=xiv['chain']
        for name in ('source_particles','target_particles'):asset(fm[name],'nu_Sub')
        fit_sources=[asset(fit/name)for name in ('input_manifest.json','transform_chain.json','atlas_deformationSummary.pt','pi_ST.csv','COMPLETED.json')]
        import csv
        with (fit/'pi_ST.csv').open() as f:
            rows=list(csv.reader(line for line in f if not line.startswith('#')))
        assert rows[0][1:]==list(references[.5].meta['class_names'])
        pi=np.asarray([[float(v)for v in row[1:]]for row in rows[1:]])
        assert pi.shape==(1,7)and np.isfinite(pi).all()and(pi>=0).all()
        if variant=='iv_modality':
            assert Path(fm['source_particles']).parent==OUT/'omics/iv_modality/numerical'
            assert Path(fm['target_particles']).parent==OUT/'omics/iv_modality/numerical'
            assert json.loads((LOG/'omics_registration_validation.json').read_text())['all_passed']
        # Validate existing deformation placement against existing VTK, without recomputing a flow.
        err=float(np.abs(_vtk_points(fit/'atlas_deformationSummary.vtk')-(xiv['deformed'].xy-chain['target_center_section_mm'])).max())
        assert err<1e-5
        manifest['checks'].append(dict(check=f'{variant} saved OMICS deformation frame',passed=True,value=err,tolerance=1e-5))
        if variant=='baseline':
            baseline_xiv=xiv
        reference_points=np.vstack([cells.xy,baseline_xiv['rigid'].xy,baseline_xiv['deformed'].xy])
        lo,hi=reference_points.min(0)-.5,reference_points.max(0)+.5
        assert (xiv['deformed'].xy>=lo).all()and(xiv['deformed'].xy<=hi).all()
        c_rep=None if variant=='baseline'else dict(extent_b=[lo[0],hi[0],lo[1],hi[1]],caption=[
            'IV modality correspondence: existing saved 150-step fit to 50 optimized seven-class area particles; no new fit is performed here.',
            'Source: 1333 single-MTG plane particles (verified exact no-op compression); existing learned 1×7 mapping. a: original measured labels/polygons.',
            'b,c: saved atlas D + target centre and displacement residuals; baseline uses its separate original deterministic-strip fit. Not anatomical accuracy.'])
        panel('omics',variant,'C','a','original measured labels and provider polygons',[cellsource],'exact established polygon-to-section axis frame')
        for letter in 'bc':panel('omics',variant,'C',letter,'existing saved baseline correspondence'if variant=='baseline'else'existing saved IV-modality correspondence',
            fit_sources,dict(fit=str(fit),source=fm['source_particles'],target=fm['target_particles'],mapping='learned 1×7 pi_ST',
                placement='saved D + target_center_section_mm',new_fit_this_task=False))
        render('omics',variant,'C',lambda:O.figure_c(cells,target,xiv=xiv,representation=c_rep))
    manifest['checks'] += merfish_checks(cells,MERFISH)
    assert all(c['passed']for c in manifest['checks'])
    changed=[p for p,h in protected_hash.items()if digest(p)!=h];assert not changed,changed
    dump(dest/'preservation.json',dict(all_unchanged=True,checked_files=len(protected_hash)))
    # Existing states remain compatible references, without mislabelling grid assets as moved particles.
    scenes=[str(p)for p in sorted(OUT.glob('*/neuroglancer/*.state.json'))]
    dump(dest/'scene_compatibility.json',dict(fixed_reference_states=scenes,
        optimized_points='Static weighted point rendering only; no existing optimized-point NG asset path.',
        qnp='Existing production RGB/HALO and A/B atlas states retain fixed correspondence for every variant.',
        omics='Existing measured-cell/gene states remain compatible; baseline atlas layers remain explicitly baseline, not the saved IV fit.'))
    dump(dest/'manifest.json',manifest)
    write_completion_manifest(manifest,dest)
    print('All 15 supported figures complete; preserved',len(protected_hash),'existing products.',flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("preflight", "build", "baseline-scenes", "omics-inputs", "omics-correspondence", "render-complete"), required=True)
    args = p.parse_args()
    {"preflight": preflight, "build": build, "baseline-scenes": baseline_scenes, "omics-inputs": omics_inputs, "omics-correspondence": omics_correspondence, "render-complete": render_complete}[args.stage]()


if __name__ == "__main__":
    main()
