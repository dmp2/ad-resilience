from __future__ import annotations

import argparse
import os
import subprocess
import sys
from importlib import resources
from pathlib import Path
from typing import Iterable, List


DEFAULT_FILES = {
    "h5ad": "SEAAD_MTG_MERFISH.2024-12-11.h5ad",
    "donor_metadata": "68debdfdd1b8e9f8fd64dab0_sea-ad_cohort_donor_metadata_072524.xlsx",
    "cognition": "68debdfd4748b7546943a7b4_sea-ad_cohort_harmonized_cognitive_scores_20241213.xlsx",
    "mri": "68debdfdae5f82b97af2fb0f_sea-ad_cohort_mri_volumetrics.xlsx",
    "mtg_neuropath": "68debdfd24606956df13f2dd_sea-ad_all_mtg_quant_neuropath_bydonorid_081122.csv",
    "luminex": "68debdff5b8003454786ea29_sea-ad_cohort_mtg-tissue_extractions-luminex_data.xlsx",
    "specimen_metadata": "SpecimenMetadata.csv",
}


def _script_path(name: str) -> str:
    return str(resources.files("seaad_mtg").joinpath("bundled_scripts", name))


def _run_python(script_name: str, args: Iterable[str], cwd: Path) -> int:
    cmd = [sys.executable, _script_path(script_name), *args]
    print("+", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(cwd), check=False)
    return result.returncode


def _join(base: Path, leaf: str | None) -> str | None:
    if not leaf:
        return None
    return str((base / leaf).resolve())


def _add_data_dir_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--data-dir", default="/work/data", help="Directory containing the .h5ad and companion files")
    p.add_argument("--h5ad", default=DEFAULT_FILES["h5ad"])
    p.add_argument("--donor-metadata", default=DEFAULT_FILES["donor_metadata"])
    p.add_argument("--cognition", default=DEFAULT_FILES["cognition"])
    p.add_argument("--mri", default=DEFAULT_FILES["mri"])
    p.add_argument("--mtg-neuropath", default=DEFAULT_FILES["mtg_neuropath"])
    p.add_argument("--luminex", default=DEFAULT_FILES["luminex"])
    p.add_argument("--specimen-metadata", default=DEFAULT_FILES["specimen_metadata"])
    p.add_argument("--work-dir", default=None, help="Run directory. Defaults to --data-dir so outputs land next to your data.")


def _run_crosswalk_cmd(args: argparse.Namespace) -> int:
    data_dir = Path(args.data_dir).resolve()
    work_dir = Path(args.work_dir or args.data_dir).resolve()
    os.chdir(work_dir)
    argv = [
        "--h5ad", _join(data_dir, args.h5ad),
        "--donor-metadata", _join(data_dir, args.donor_metadata),
        "--cognition", _join(data_dir, args.cognition),
        "--mri", _join(data_dir, args.mri),
        "--mtg-neuropath", _join(data_dir, args.mtg_neuropath),
        "--luminex", _join(data_dir, args.luminex),
        "--outdir", args.outdir,
    ]
    specimen_path = _join(data_dir, args.specimen_metadata) if args.specimen_metadata else None
    if specimen_path and Path(specimen_path).exists():
        argv.extend(["--specimen-metadata", specimen_path])
    if args.obs_donor_column:
        argv.extend(["--obs-donor-column", args.obs_donor_column])

    saved = sys.argv
    try:
        sys.argv = ["seaad-mtg crosswalk", *argv]
        from .crosswalk import main as crosswalk_main
        crosswalk_main()
    finally:
        sys.argv = saved
    return 0


def _run_pipeline_cmd(args: argparse.Namespace) -> int:
    data_dir = Path(args.data_dir).resolve()
    work_dir = Path(args.work_dir or args.data_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    rc = _run_crosswalk_cmd(args)
    if rc != 0:
        return rc

    common = [
        "--donor-metadata", _join(data_dir, args.donor_metadata),
        "--cognition", _join(data_dir, args.cognition),
        "--mri", _join(data_dir, args.mri),
        "--mtg-neuropath", _join(data_dir, args.mtg_neuropath),
        "--luminex", _join(data_dir, args.luminex),
    ]

    rc = _run_python(
        "seaad_spatial_section_manifest.py",
        [
            "--region-prefix", args.region_prefix,
            "--outdir", args.spatial_outdir,
            *common,
        ],
        cwd=work_dir,
    )
    if rc != 0:
        return rc

    rc = _run_python(
        "seaad_mtg_spatial_neuropath_linkage.py",
        [
            "--outdir", args.linkage_outdir,
            *common,
        ],
        cwd=work_dir,
    )
    if rc != 0:
        return rc

    rc = _run_python(
        "seaad_mtg_h5ad_crosswalk_probe.py",
        [
            "--h5ad", _join(data_dir, args.h5ad),
            "--spatial-manifest", str((work_dir / args.linkage_outdir / "spatial_section_manifest.csv").resolve()),
            "--neuropath-manifest", str((work_dir / args.linkage_outdir / "neuropath_stain_manifest.csv").resolve()),
            "--outdir", args.probe_outdir,
        ],
        cwd=work_dir,
    )
    if rc != 0:
        return rc

    return _run_python(
        "seaad_rank_crosswalk_candidates.py",
        [
            "--probe-dir", str((work_dir / args.probe_outdir).resolve()),
            "--linkage-dir", str((work_dir / args.linkage_outdir).resolve()),
            "--outdir", args.rank_outdir,
        ],
        cwd=work_dir,
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Docker-friendly CLI for the SEA-AD MTG crosswalk pipeline.")
    sub = p.add_subparsers(dest="command", required=True)

    crosswalk = sub.add_parser("crosswalk", help="Run the local .h5ad donor crosswalk step")
    _add_data_dir_args(crosswalk)
    crosswalk.add_argument("--outdir", default="seaad_mtg_spatial_crosswalk_outputs")
    crosswalk.add_argument("--obs-donor-column", default=None)
    crosswalk.set_defaults(func=_run_crosswalk_cmd)

    pipeline = sub.add_parser("pipeline", help="Run the full five-step MTG pipeline")
    _add_data_dir_args(pipeline)
    pipeline.add_argument("--region-prefix", default="middle-temporal-gyrus/")
    pipeline.add_argument("--spatial-outdir", default="seaad_spatial_section_manifest")
    pipeline.add_argument("--linkage-outdir", default="seaad_mtg_linkage")
    pipeline.add_argument("--probe-outdir", default="seaad_mtg_h5ad_crosswalk_probe")
    pipeline.add_argument("--rank-outdir", default="seaad_crosswalk_ranked")
    pipeline.add_argument("--outdir", default="seaad_mtg_spatial_crosswalk_outputs")
    pipeline.add_argument("--obs-donor-column", default=None)
    pipeline.set_defaults(func=_run_pipeline_cmd)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    rc = args.func(args)
    raise SystemExit(rc)
