"""Build or verify the Allen 708424 mixed-stain physical cutting lattice."""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

from preprocess.prepare_allen_emlddmm_inputs import (
    DEFAULT_DATA_DIR,
    DEFAULT_OUTPUT_DIR,
    MAXIMUM_SLOT,
    MINIMUM_SLOT,
    PHYSICAL_FIELDS,
    build_physical_rows,
)


LOG = logging.getLogger(__name__)
DEFAULT_MANIFEST = DEFAULT_DATA_DIR / "metadata" / "manifest.tsv"
DEFAULT_OUTPUT = DEFAULT_OUTPUT_DIR / "metadata" / "physical_sections.tsv"


def build_lattice_rows(
    manifest: Path,
) -> tuple[list[dict[str, str]], dict[str, object]]:
    return build_physical_rows(
        manifest,
        section_range=(MINIMUM_SLOT, MAXIMUM_SLOT),
        validate_counts=True,
    )


def verify_lattice(manifest: Path, output: Path) -> dict[str, object]:
    """Verify the preparation-owned canonical lattice against raw Allen IDs."""

    expected, summary = build_lattice_rows(manifest)
    with output.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        actual = list(reader)
        if list(reader.fieldnames or ()) != PHYSICAL_FIELDS:
            raise ValueError(f"Unexpected canonical lattice schema: {output}")
    if len(actual) != len(expected):
        raise ValueError(f"Canonical lattice row count mismatch: {output}")
    keys = (
        "physical_index",
        "specimen_id",
        "allen_section_number",
        "serial_z_center_mm",
        "section_thickness_um",
        "serial_pitch_um",
        "image_present",
        "stain",
        "allen_section_image_id",
        "allen_data_set_id",
        "source_relative_path",
        "source_sha256",
        "nominal_series_interval_um",
        "observation_class",
    )
    for position, (actual_row, expected_row) in enumerate(
        zip(actual, expected, strict=True)
    ):
        if any(actual_row[key] != expected_row[key] for key in keys):
            raise ValueError(f"Canonical lattice differs at row {position}: {output}")
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    logging.basicConfig(level=args.log_level, format="%(levelname)s %(message)s")
    summary = verify_lattice(args.manifest, args.output)
    LOG.info(
        "Validated mixed lattice: %d slots, %d present, %d absent",
        summary["row_count"],
        summary["present_count"],
        summary["absent_count"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
