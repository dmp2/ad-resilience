"""``python -m visualization.cli <command>`` - build, verify and serve the prototypes.

Run from the project root in the ``wsi-pipeline`` env with
``PYTHONPATH=src:src/seaad:src/preprocess:src/config_setup:src/scripts``.

Commands
--------
ng-build   write Neuroglancer assets + states   (--modality omics|qnp|all)
ng-verify  read every asset back over HTTP and screenshot each scene in headless Chrome
ng-serve   serve all scenes (CORS + Range) and print one viewer URL per scene; blocks
figures    static figure families A/B/C        (--modality, --family)
validate   numerical checks against existing trusted outputs
xiv-outputs export reference-compatible VTK/PT bundles from saved fits
"""

from __future__ import annotations

import argparse
import json
import sys
from functools import lru_cache
from pathlib import Path

from . import scene as S

MERFISH = {"h5ad": "data/raw/sea-ad/merfish/SEAAD_MTG_MERFISH.2024-12-11.h5ad", "barcode": "1194111462",
           "xiv_fit": "results/registrations/ahrav2_mtg_plane_to_strip_1194111462/init_r000_f0",
           "grouping": "seaad_merfish_subclass_classes"}
# A priori markers (not ranked from the data): L2/3 IT, L4-5 IT, deep-layer excitatory,
# L6 IT Car3, oligodendrocyte; the mean of the 40 blank probes is a negative control.
GENES = ("CUX2", "RORB", "FEZF2", "THEMIS", "MOG")
CONTROL = {"Blank (mean of 40)": ("Blank-", "mean")}
QNP_JOB = "H20.33.034_MTG"
OUT = S.PROJECT / "results" / "visualization_prototypes"


@lru_cache(maxsize=1)
def config() -> dict:
    return S.load_project_config()


@lru_cache(maxsize=1)
def merfish_cells() -> S.Points:
    groups = config()["feature_groupings"][MERFISH["grouping"]]["groups"]
    return S.load_merfish_cells(S.PROJECT / MERFISH["h5ad"], MERFISH["barcode"], groups, genes=GENES, gene_pools=CONTROL)


@lru_cache(maxsize=1)
def merfish_polygons():
    return S.merfish_layer_polygons(S.PROJECT / MERFISH["h5ad"], merfish_cells())


def cmd_ng_build(args) -> None:
    from . import ng_scenes

    out = {}
    if args.modality in ("qnp", "all"):
        out.update(ng_scenes.build_qnp(config(), QNP_JOB, level=args.level, overwrite=args.overwrite))
    if args.modality in ("omics", "all"):
        polygons, transform, _ = merfish_polygons()
        xiv = S.load_merfish_xiv(S.PROJECT / MERFISH["xiv_fit"])
        out.update(ng_scenes.build_omics(config(), merfish_cells(), polygons, transform, xiv,
                                         list(GENES) + list(CONTROL)))
    print(json.dumps({k: {kk: str(vv.relative_to(S.PROJECT)) for kk, vv in v.items()} for k, v in out.items()}, indent=2))


def _scene_names(modality: str) -> list[tuple[Path, str]]:
    out = []
    for m in (("omics", "qnp") if modality == "all" else (modality,)):
        bundle = OUT / m / "neuroglancer"
        out += [(bundle, p.name[: -len(".state.json")]) for p in sorted(bundle.glob("*.state.json"))]
    return out


def cmd_ng_verify(args) -> None:
    from . import ng_export as nx

    failures = 0
    for bundle, scene in _scene_names(args.modality):
        report = nx.verify_scene(bundle, scene, port=args.port, shot=not args.no_screenshot)
        ok = report["all_sources_ok"] and report.get("screenshot", {}).get("ok", True)
        failures += not ok
        print(f"{scene}: sources {'ok' if report['all_sources_ok'] else 'FAILED'}; screenshot "
              f"{report.get('screenshot', {}).get('ok')}; console errors {len(report.get('screenshot', {}).get('console_errors', []))}")
        for check in report["sources"]:
            if not check["ok"]:
                print("   FAILED", json.dumps(check)[:400])
    sys.exit(1 if failures else 0)


def cmd_ng_serve(args) -> None:
    from . import ng_export as nx

    nx.serve([b / f"{s}.state.json" for b, s in _scene_names(args.modality)], port=args.port, host=args.host,
             ng_port=args.viewer_port)


def cmd_figures(args) -> None:
    families = args.family or ["A", "B", "C"]
    if args.modality in ("omics", "all"):
        from . import figures_omics

        for family in families:
            paths = getattr(figures_omics, f"figure_{family.lower()}")(merfish_cells(), OUT / "omics")
            print("omics", family, [str(p.relative_to(S.PROJECT)) for p in paths])
    if args.modality in ("qnp", "all"):
        from . import figures_qnp

        for family in families:
            paths = getattr(figures_qnp, f"figure_{family.lower()}")(config(), QNP_JOB, OUT / "qnp")
            print("qnp", family, [str(p.relative_to(S.PROJECT)) for p in paths])


def cmd_validate(args) -> None:
    from . import validate

    report = validate.run(config(), merfish_cells, QNP_JOB, MERFISH, OUT)
    print(json.dumps(report["summary"], indent=2))
    sys.exit(0 if report["summary"]["all_passed"] else 1)


XIV_OUTPUT_FILES = (
    "originalAtlas.vtk", "originalTarget.vtk", "atlas_nu_D.vtk", "atlas_nu_Dpi.vtk",
    "atlas_deformationSummary.pt", "atlas_deformationSummary.vtk", "target_nu_Td.vtk",
    "target_deformationSummary.pt", "target_deformationSummary.vtk",
)


def _export_xiv_fit(api, fit_dir: Path, out_dir: Path) -> dict:
    import hashlib
    import shutil
    import numpy as np
    import torch
    from preprocess.xmodmap_compat import save_atlas_outputs, save_target_outputs
    from scripts.framework_script_AtlasToSectionCrossModalBoundary import _writers
    from .validate import validate_xiv_vtks

    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    atlas = torch.load(fit_dir / "atlas_deformationSummary.pt", map_location="cpu", weights_only=False)
    target = torch.load(fit_dir / "target_deformationSummary.pt", map_location="cpu", weights_only=False)
    inputs = json.loads((fit_dir / "input_manifest.json").read_text())
    config = json.loads((fit_dir.parent / "resolved_project_config.json").read_text())
    registration = json.loads((fit_dir / "COMPLETED.json").read_text())["registration"]
    convention = config["registrations"][registration].get("vtk_coordinate_convention", "xyz")
    write_vtk, write_particle = _writers(api, convention)
    names = []
    for kind in ("source", "target"):
        with np.load(inputs[f"{kind}_particles"], allow_pickle=False) as archive:
            names.append([str(v) for v in archive["coarse_feature_names"]])
    if inputs["target_feature_mode"] == "total_mass":
        names[1] = ["total_mass"]
    # Fail before writing on corrupt/incomplete saved states.
    for blob in (atlas, target):
        for value in blob.values():
            if torch.is_tensor(value) and not bool(torch.isfinite(value).all()):
                raise ValueError(f"Non-finite saved state in {fit_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    save_atlas_outputs(write_particle, write_vtk, api.get_entropy, atlas, out_dir, *names)
    save_target_outputs(write_particle, write_vtk, api.get_entropy, target, out_dir)
    for filename in ("originalAtlas.vtk", "originalTarget.vtk", "input_manifest.json", "transform_chain.json"):
        shutil.copy2(fit_dir / filename, out_dir / filename)
    # Reload the exported states and verify the flow and measures were preserved exactly.
    for filename, original in (("atlas_deformationSummary.pt", atlas), ("target_deformationSummary.pt", target)):
        exported = torch.load(out_dir / filename, map_location="cpu", weights_only=False)
        for key, value in original.items():
            if torch.is_tensor(value):
                torch.testing.assert_close(exported[key], value, rtol=0, atol=0)
        torch.testing.assert_close(exported["jac"], exported["mass_ratio_proxy"], rtol=0, atol=0)
    report = {
        "source_fit": str(fit_dir), "output_dir": str(out_dir), "figure_families": ["A", "B", "C"],
        "scope": "A/B show measured or compressed inputs; C uses this saved registration context",
        "optimization_rerun": False, "vtk_coordinate_convention": convention,
        "coordinates": "physical mm in the saved fit frame; see transform_chain.json for centering and initialization",
        "Jacobian_definition": "legacy alias of MassRatioProxy = sum(nu_deformed)/sum(nu_initial), not a spatial derivative determinant",
        "entropy_definition": "-sum(p * log(p)), p = nu / sum(nu); atlas entropy uses nu_Dpi",
        "source_sha256": {name: sha(fit_dir / name) for name in
                          ("atlas_deformationSummary.pt", "target_deformationSummary.pt", "input_manifest.json",
                           "transform_chain.json", "COMPLETED.json")},
        "files": {name: {"bytes": (out_dir / name).stat().st_size, "sha256": sha(out_dir / name)} for name in XIV_OUTPUT_FILES},
        "saved_tensors_preserved": True,
        "vtk_readback": validate_xiv_vtks(out_dir, atlas, target, convention, api.get_entropy, XIV_OUTPUT_FILES),
    }
    (out_dir / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def export_xiv_prototypes(config, out, modality="all"):
    from preprocess.xmodmap_compat import load_xmodmap
    from .ng_scenes import HYPOTHESES

    # Match the user-specified public reference, independently of legacy fit settings.
    output_config = {**config, "software": {**config["software"],
                     "xiv_lddmm_repository": str(S.PROJECT / "vendor/xIV-LDDMM-Particle")}}
    api = load_xmodmap(output_config)
    fits = []
    if modality in ("omics", "all"):
        fits.append(("omics", "baseline", S.PROJECT / MERFISH["xiv_fit"]))
        variant = out / "omics/iv_modality/registration/init_r000_f0"
        if (variant / "COMPLETED.json").is_file():
            fits.append(("omics", "iv_modality", variant))
    if modality in ("qnp", "all"):
        fits.extend(("qnp", f"pose_{key}", S.PROJECT / values[1]) for key, values in HYPOTHESES.items())
    reports = [_export_xiv_fit(api, fit, out / mod / "xiv_outputs" / label) for mod, label, fit in fits]
    index = out / f"xiv_outputs_{modality}.json"
    index.write_text(json.dumps({"bundles": reports}, indent=2) + "\n")
    return index


def cmd_xiv_outputs(args) -> None:
    print(export_xiv_prototypes(config(), OUT, args.modality))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="visualization.cli", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, func in (("ng-build", cmd_ng_build), ("ng-verify", cmd_ng_verify), ("ng-serve", cmd_ng_serve),
                       ("figures", cmd_figures), ("validate", cmd_validate), ("xiv-outputs", cmd_xiv_outputs)):
        p = sub.add_parser(name)
        p.set_defaults(func=func)
        p.add_argument("--modality", choices=("omics", "qnp", "all"), default="all")
        if name == "ng-build":
            p.add_argument("--level", type=int, default=1, help="Aperio level for QNP images (1 = ~2 um/px)")
            p.add_argument("--overwrite", action="store_true", help="rewrite existing OME-Zarr images")
        if name in ("ng-verify", "ng-serve"):
            p.add_argument("--port", type=int, default=8000)
        if name == "ng-verify":
            p.add_argument("--no-screenshot", action="store_true")
        if name == "ng-serve":
            p.add_argument("--host", default="127.0.0.1")
            p.add_argument("--viewer-port", type=int, default=0, help="fixed Neuroglancer viewer port (0 = any)")
        if name == "figures":
            p.add_argument("--family", nargs="*", choices=("A", "B", "C"))
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
