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


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="visualization.cli", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, func in (("ng-build", cmd_ng_build), ("ng-verify", cmd_ng_verify), ("ng-serve", cmd_ng_serve),
                       ("figures", cmd_figures), ("validate", cmd_validate)):
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
