"""Run the pinned serial loader against prepared Allen views and audit W0."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from preprocess.prepare_allen_emlddmm_inputs import accepted_loader_axes
from preprocess.run_allen_emlddmm import (
    load_physical_rows,
    pinned_emlddmm,
    validate_loaded_support,
)


LOG = logging.getLogger(__name__)


def audit_loader(
    dataset: Path,
    view_names: list[str],
) -> dict[str, Any]:
    rows = load_physical_rows(dataset)
    canvas = json.loads(
        (dataset / "metadata" / "loader_canvas_audit.json").read_text(
            encoding="utf-8"
        )
    )
    requested_axes = accepted_loader_axes(rows, canvas)
    emlddmm = pinned_emlddmm()
    report: dict[str, Any] = {
        "dataset": str(dataset),
        "row_count": len(rows),
        "z_coordinate_count": len(requested_axes[0]),
        "automatic_canvas_accepted": canvas["automatic_canvas_accepted"],
        "accepted_canvas_shape_yx": canvas["accepted_canvas_shape_yx"],
        "w0_definition": (
            "Pinned load_slices technical support, operationally "
            "first_loaded_channel > 0"
        ),
        "views": {},
    }
    for view_name in view_names:
        view_dir = dataset / "inputs" / "views" / view_name
        loaded_axes, images, support = emlddmm.load_slices(
            str(view_dir), xJ=requested_axes
        )
        for actual, expected in zip(loaded_axes, requested_axes, strict=True):
            np.testing.assert_allclose(actual, expected)
        sample_rows = []
        import csv

        with (view_dir / "samples.tsv").open(
            encoding="utf-8", newline=""
        ) as stream:
            sample_rows = list(csv.DictReader(stream, delimiter="\t"))
        counts = validate_loaded_support(support, sample_rows)
        report["views"][view_name] = {
            "image_shape_czyx": list(images.shape),
            "w0_shape_zyx": list(support.shape),
            "support_counts": counts,
            "finite_images": bool(np.all(np.isfinite(images))),
            "finite_w0": bool(np.all(np.isfinite(support))),
        }
    output = dataset / "metadata" / "loader_support_audit.json"
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--views",
        nargs="+",
        choices=("HIST_ALL", "HIST_NISSL", "HIST_PV"),
        default=["HIST_ALL", "HIST_NISSL", "HIST_PV"],
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    logging.basicConfig(level=args.log_level, format="%(levelname)s %(message)s")
    report = audit_loader(args.dataset, args.views)
    LOG.info(
        "Validated pinned loader: %d rows across %d views",
        report["row_count"],
        len(report["views"]),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
