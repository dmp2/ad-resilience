"""Prepare and draw atlas masks for the Allen 1 mm registration notebook.

SynthStrip runs on the CPU using an isolated dependency directory. Official
script/model URLs and hashes are saved beside the resulting masks. Neither the
registration Python environment nor the ex vivo target is modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.request

import tune_allen_1mm_registration as tuning

CACHE = tuning.PROJECT / ".cache/synthstrip"
MASKS = tuning.TUNING / "masks"
DOCUMENTATION = "https://surfer.nmr.mgh.harvard.edu/docs/synthstrip/"
SCRIPT_URL = "https://raw.githubusercontent.com/freesurfer/freesurfer/dev/mri_synthstrip/mri_synthstrip"
MODEL_BASE = DOCUMENTATION + "requirements/"


def prepare_synthstrip():
    CACHE.mkdir(parents=True, exist_ok=True)
    MASKS.mkdir(parents=True, exist_ok=True)
    sources = {
        "mri_synthstrip.py": SCRIPT_URL,
        "synthstrip.1.pt": MODEL_BASE + "synthstrip.1.pt",
        "synthstrip.nocsf.1.pt": MODEL_BASE + "synthstrip.nocsf.1.pt",
    }
    for name, url in sources.items():
        path = CACHE / name
        if not path.exists():
            temporary = path.with_suffix(path.suffix + ".download")
            urllib.request.urlretrieve(url, temporary)
            temporary.replace(path)
    packages = CACHE / "python"
    if not (
        (packages / "surfa").exists()
        and (packages / "xxhash").exists()
        and (packages / "numpy-1.26.4.dist-info").exists()
    ):
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--target",
                str(packages),
                "--no-deps",
                "surfa==0.6.3",
                "xxhash==4.0.2",
                "numpy==1.26.4",
            ],
            check=True,
        )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(packages)
    provenance = dict(
        documentation=DOCUMENTATION,
        sources=sources,
        sha256={
            name: hashlib.sha256((CACHE / name).read_bytes()).hexdigest()
            for name in sources
        },
        dependencies={"surfa": "0.6.3", "xxhash": "4.0.2", "numpy": "1.26.4"},
        input=str(tuning.TEMPLATE),
        device="cpu",
        threads=6,
        commands={},
    )
    for name, model, flags in [
        ("synthstrip", "synthstrip.1.pt", []),
        ("synthstrip_nocsf", "synthstrip.nocsf.1.pt", ["--no-csf"]),
    ]:
        command = [
            sys.executable,
            str(CACHE / "mri_synthstrip.py"),
            "-i",
            str(tuning.TEMPLATE),
            "-o",
            str(MASKS / (name + "_atlas.nii.gz")),
            "-m",
            str(MASKS / (name + "_mask.nii.gz")),
            "--model",
            str(CACHE / model),
            "-t",
            "6",
            *flags,
        ]
        provenance["commands"][name] = command
        if not (MASKS / (name + "_mask.nii.gz")).exists():
            print("Running", name, flush=True)
            with (MASKS / (name + ".log")).open("w") as log:
                subprocess.run(
                    command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True
                )
        print("Prepared", name, flush=True)
    (MASKS / "synthstrip_provenance.json").write_text(json.dumps(provenance, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthstrip-only", action="store_true")
    args = parser.parse_args()
    if not args.synthstrip_only:
        data = tuning.load_inputs()
        tuning.prepare_convex_hull(data)
        tuning.review_atlas_mask(data)
        tuning.review_atlas_mask(tuning.load_inputs("convex_hull"))
    prepare_synthstrip()
    for name in ("synthstrip", "synthstrip_nocsf"):
        tuning.review_atlas_mask(tuning.load_inputs(name))
    print("Atlas mask preparation complete:", MASKS, flush=True)


if __name__ == "__main__":
    main()
