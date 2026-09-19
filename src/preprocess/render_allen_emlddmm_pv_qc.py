"""Render sparse symmetric PV and paired Nissl QC in shared coordinates.

This rendering-only adapter reads the immutable products of ``pv_postprocess``
and reuses the Allen/EM-LDDMM support-aware orthogonal and section-montage
conventions. It never invokes registration, densification, or postprocessing.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
from matplotlib.patches import Patch

from preprocess import render_allen_emlddmm_support_aware_qc as support_qc
from preprocess import run_allen_emlddmm_full_coarse_nissl as coarse
from preprocess import run_allen_emlddmm_full_resolution_nissl as native
from preprocess.prepare_allen_emlddmm_inputs import accepted_loader_axes
from preprocess.run_allen_emlddmm import load_physical_rows


PV_DATASET = native.PV_SYMMETRIC_DATASET
NISSL_DATASET = native.CLEAN_SYMMETRIC_DATASET
QC_ROOT = native.PV_OUTPUT / "postprocessed_qc"
OUTPUT_NAME = "pv_nissl_support_aware_qc"
SUPPORT_THRESHOLD = support_qc.SUPPORT_THRESHOLD
INPLANE_DOWNSAMPLE = 4
EXPECTED_PHYSICAL_POSITIONS = 2846
EXPECTED_PV_SECTIONS = 285
EXPECTED_NISSL_SECTIONS = 641
OUTPUTS = {
    "symmetric_pv": "symmetric_pv_orthogonal.png",
    "pv_nissl_combined": "symmetric_pv_nissl_orthogonal.png",
    "pv_nissl_matched_sections": "symmetric_pv_nissl_matched_sections.png",
    "pv_coverage": "symmetric_pv_coverage_orthogonal.png",
}
PRIMARY_OUTPUTS = (
    "symmetric_pv", "pv_nissl_combined", "pv_nissl_matched_sections",
)
OPTIONAL_OUTPUTS = ("pv_coverage",)


@dataclass(frozen=True)
class DatasetView:
    dataset: Path
    view: Path
    support: Path
    samples: list[dict[str, str]]
    observed: np.ndarray


def _samples(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        rows = list(reader)
        fields = list(reader.fieldnames or ())
    if fields != ["sample_id", "participant_id", "species", "status"]:
        raise RuntimeError(f"Unexpected samples.tsv schema: {path}")
    return rows


def _load_inputs() -> tuple[DatasetView, DatasetView, list[np.ndarray]]:
    rows = load_physical_rows(NISSL_DATASET)
    if len(rows) != EXPECTED_PHYSICAL_POSITIONS:
        raise RuntimeError("Symmetric Nissl physical lattice changed")
    canvas = json.loads(
        (NISSL_DATASET / "metadata/loader_canvas_audit.json").read_text(
            encoding="utf-8"
        )
    )
    axes = [np.asarray(axis) for axis in accepted_loader_axes(rows, canvas)]
    transforms = NISSL_DATASET / "metadata/transforms"
    recorded_axes = [
        np.load(transforms / "serial_axis_um.npy"),
        np.load(transforms / "row_axis_um.npy"),
        np.load(transforms / "symmetric_lr_axis_um.npy"),
    ]
    if any(
        not np.array_equal(actual, expected)
        for actual, expected in zip(axes, recorded_axes, strict=True)
    ):
        raise RuntimeError("Accepted loader axes differ from postprocessed axes")

    contexts = []
    for dataset, view_name, support_name, expected_count in (
        (PV_DATASET, "HIST_PV", "pv", EXPECTED_PV_SECTIONS),
        (NISSL_DATASET, "HIST_NISSL", "nissl", EXPECTED_NISSL_SECTIONS),
    ):
        view = dataset / "inputs/views" / view_name
        samples = _samples(view / "samples.tsv")
        if len(samples) != EXPECTED_PHYSICAL_POSITIONS:
            raise RuntimeError(f"Physical sample lattice changed: {dataset}")
        observed = np.flatnonzero(
            [sample["status"] == "present" for sample in samples]
        ).astype(np.int64)
        if observed.size != expected_count:
            raise RuntimeError(
                f"Expected {expected_count} observed {support_name} sections, "
                f"found {observed.size}"
            )
        contexts.append(
            DatasetView(
                dataset, view, dataset / "support" / support_name,
                samples, observed,
            )
        )
    pv, nissl = contexts
    if not np.all(np.isin(pv.observed, nissl.observed)):
        raise RuntimeError("A paired PV plane lacks section-aligned Nissl")
    geometry = json.loads(
        (PV_DATASET / "metadata/geometry_check.json").read_text(encoding="utf-8")
    )
    if (
        geometry.get("status") != "passed"
        or geometry.get("sections_checked") != EXPECTED_PV_SECTIONS
        or geometry.get("canonical_serial_positions") != EXPECTED_PHYSICAL_POSITIONS
        or Path(geometry.get("reference", "")).resolve() != NISSL_DATASET.resolve()
    ):
        raise RuntimeError("PV/Nissl postprocess geometry check is not valid")
    return pv, nissl, axes


def _load_section(
    context: DatasetView, index: int, expected_shape: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    sample = context.samples[index]
    if sample["status"] != "present":
        raise RuntimeError(f"Requested absent physical plane {index}")
    image_path = context.view / sample["sample_id"]
    support_path = context.support / image_path.name
    image = tifffile.imread(image_path)[..., :3]
    weights = tifffile.imread(support_path).astype(np.float32)
    if image.dtype != np.uint8 or image.shape != (*expected_shape, 3):
        raise RuntimeError(f"Invalid RGB section geometry: {image_path}")
    if (
        weights.shape != expected_shape
        or not np.all(np.isfinite(weights))
        or np.min(weights) < 0.0
        or np.max(weights) > 1.0
    ):
        raise RuntimeError(f"Invalid section support: {support_path}")
    image = image.transpose(2, 0, 1).astype(np.float32) / 255.0
    return image, weights


def _plane_indices(length: int) -> np.ndarray:
    return np.rint(
        np.linspace(0, length - 1, support_qc.N_SLICES + 2)[1:-1]
    ).astype(np.int64)


def _empty_planes(
    axes: list[np.ndarray], observed: np.ndarray
) -> dict[int, list[tuple[int, np.ndarray, np.ndarray]]]:
    indices = {
        1: _plane_indices(len(axes[1])),
        2: _plane_indices(len(axes[2])),
    }
    planes: dict[int, list[tuple[int, np.ndarray, np.ndarray]]] = {
        0: [], 1: [], 2: [],
    }
    for axis in (1, 2):
        shape = (len(axes[0]), len(axes[2 if axis == 1 else 1]))
        for index in indices[axis]:
            planes[axis].append(
                (int(index), np.zeros((3, *shape), np.float32),
                 np.zeros(shape, np.float32))
            )
    if support_qc._representative_observed(observed).size != support_qc.N_SLICES:
        raise RuntimeError("Representative PV selection is incomplete")
    return planes


def _build_planes_and_pairs(
    pv_context: DatasetView,
    nissl_context: DatasetView,
    native_axes: list[np.ndarray],
    display_axes: list[np.ndarray],
) -> tuple[
    dict[int, list[tuple[int, np.ndarray, np.ndarray]]],
    dict[int, list[tuple[int, np.ndarray, np.ndarray]]],
    dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
]:
    pv_planes = _empty_planes(display_axes, pv_context.observed)
    nissl_planes = _empty_planes(display_axes, pv_context.observed)
    selected = set(map(int, support_qc._representative_observed(pv_context.observed)))
    pairs = {}
    expected_shape = tuple(map(len, native_axes[1:]))
    for count, index_value in enumerate(pv_context.observed, 1):
        index = int(index_value)
        native_pv, native_pv_support = _load_section(
            pv_context, index, expected_shape
        )
        native_nissl, native_nissl_support = _load_section(
            nissl_context, index, expected_shape
        )
        if index in selected:
            pairs[index] = (
                native_pv, native_pv_support,
                native_nissl, native_nissl_support,
            )
        pv, pv_support = coarse.downsample_section(
            native_pv, native_pv_support, factor=INPLANE_DOWNSAMPLE
        )
        nissl, nissl_support = coarse.downsample_section(
            native_nissl, native_nissl_support, factor=INPLANE_DOWNSAMPLE
        )
        if pv.shape[1:] != tuple(map(len, display_axes[1:])):
            raise RuntimeError("Downsampled section and display axes differ")
        common_support = np.minimum(pv_support, nissl_support)
        if index in selected:
            pv_planes[0].append((index, pv, pv_support))
            nissl_planes[0].append((index, nissl, common_support))
        for pv_plane, nissl_plane in zip(
            pv_planes[1], nissl_planes[1], strict=True
        ):
            row = pv_plane[0]
            pv_plane[1][:, index] = pv[:, row]
            pv_plane[2][index] = pv_support[row]
            nissl_plane[1][:, index] = nissl[:, row]
            nissl_plane[2][index] = common_support[row]
        for pv_plane, nissl_plane in zip(
            pv_planes[2], nissl_planes[2], strict=True
        ):
            column = pv_plane[0]
            pv_plane[1][:, index] = pv[:, :, column]
            pv_plane[2][index] = pv_support[:, column]
            nissl_plane[1][:, index] = nissl[:, :, column]
            nissl_plane[2][index] = common_support[:, column]
        if count % 50 == 0:
            print(
                f"loaded {count}/{len(pv_context.observed)} paired sections",
                flush=True,
            )
    pv_planes[0].sort(key=lambda item: item[0])
    nissl_planes[0].sort(key=lambda item: item[0])
    if len(pv_planes[0]) != support_qc.N_SLICES or set(pairs) != selected:
        raise RuntimeError("Representative paired-section reconstruction is incomplete")
    return pv_planes, nissl_planes, pairs


def _save_matched_sections(
    path: Path,
    pairs: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    axes: list[np.ndarray],
    threshold: float,
) -> None:
    """Use the established section-montage layout for native-color pairs."""
    selected = sorted(pairs)
    figure, panels = coarse.plt.subplots(
        2, len(selected), figsize=(3.1 * len(selected), 6.7), squeeze=False,
        sharex=True, sharey=True,
    )
    extent = tuple(value / 1000.0 for value in support_qc._extent(axes, 0))
    for column, index in enumerate(selected):
        pv, pv_support, nissl, nissl_support = pairs[index]
        for row, (image, weights) in enumerate(
            ((pv, pv_support), (nissl, nissl_support))
        ):
            shown = np.moveaxis(np.clip(image, 0.0, 1.0), 0, -1).copy()
            shown[weights < threshold] = 0.16
            panels[row, column].imshow(
                shown, extent=extent, aspect="equal", interpolation="none"
            )
            panels[row, column].tick_params(labelsize=6)
        panels[0, column].set_title(
            f"physical {index}\nz={axes[0][index] / 1000.0:.2f} mm",
            fontsize=8,
        )
        panels[-1, column].set_xlabel("left-right (mm)", fontsize=7)
    panels[0, 0].set_ylabel("Symmetric PV\nrow (mm)", fontsize=8)
    panels[1, 0].set_ylabel("Symmetric Nissl\nrow (mm)", fontsize=8)
    figure.legend(
        handles=[
            Patch(facecolor="white", edgecolor="white", label="observed support"),
            Patch(facecolor="0.16", edgecolor="0.16", label="unsupported (not zero)"),
        ],
        loc="lower center", ncol=2,
    )
    figure.suptitle(
        "Matched native-color PV and section-aligned Nissl\n"
        "Identical physical positions, geometry, extents, and framing; "
        "gray is unsupported, not zero signal."
    )
    figure.tight_layout(rect=(0, 0.07, 1, 0.91))
    coarse._atomic_figure(path, figure, dpi=180)


def _save_combined_orthogonal(
    path: Path,
    pv_planes: dict[int, list[tuple[int, np.ndarray, np.ndarray]]],
    nissl_planes: dict[int, list[tuple[int, np.ndarray, np.ndarray]]],
    axes: list[np.ndarray],
    threshold: float,
) -> None:
    """Show native-RGB PV and Nissl as matched orthogonal panel rows."""
    figure, panels = coarse.plt.subplots(
        6, support_qc.N_SLICES, figsize=(18, 20), squeeze=False
    )
    for axis in range(3):
        for column, (pv_plane, nissl_plane) in enumerate(
            zip(pv_planes[axis], nissl_planes[axis], strict=True)
        ):
            if pv_plane[0] != nissl_plane[0]:
                raise RuntimeError("PV/Nissl orthogonal positions differ")
            index = pv_plane[0]
            for modality, (_, image, coverage) in enumerate(
                (pv_plane, nissl_plane)
            ):
                shown = np.moveaxis(np.clip(image, 0.0, 1.0), 0, -1)
                shown[coverage < threshold] = 0.16
                panel = panels[2 * axis + modality, column]
                panel.imshow(
                    shown,
                    extent=support_qc._extent(axes, axis),
                    aspect="equal",
                    interpolation="none",
                )
                panel.set_xticks([])
                panel.set_yticks([])
            panels[2 * axis, column].set_title(
                f"axis {axis}: {axes[axis][index] / 1000.0:.1f} mm"
            )
        panels[2 * axis, 0].set_ylabel(f"axis {axis} — PV")
        panels[2 * axis + 1, 0].set_ylabel(f"axis {axis} — Nissl")
    figure.legend(
        handles=[
            Patch(facecolor="white", edgecolor="white", label="observed support"),
            Patch(
                facecolor="0.16", edgecolor="0.16",
                label="unobserved/unsupported (not zero)",
            ),
        ],
        loc="lower center", ncol=2,
    )
    figure.suptitle(
        "Symmetric PV + section-aligned Nissl — full physical stack\n"
        "Separate native-RGB panels at matched orthogonal positions; Nissl is "
        "shown only on common PV-observed support across all 2,846 serial planes."
    )
    figure.tight_layout(rect=(0, 0.03, 1, 0.96))
    coarse._atomic_figure(path, figure, dpi=180)


def _output_paths() -> tuple[Path, Path]:
    return QC_ROOT / f".{OUTPUT_NAME}.tmp", QC_ROOT / OUTPUT_NAME


def _provenance(
    final: Path,
    display_axes: list[np.ndarray],
    pv: DatasetView,
    pairs: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    threshold: float,
) -> dict[str, Any]:
    figures = {
        key: str((final / value).resolve()) for key, value in OUTPUTS.items()
    }
    return {
        "schema": "allen-emlddmm-pv-nissl-qc-v1",
        "status": "complete",
        "inputs": {
            "symmetric_pv_dataset": str(PV_DATASET.resolve()),
            "symmetric_section_aligned_nissl_dataset": str(
                NISSL_DATASET.resolve()
            ),
        },
        "figures": figures,
        "primary_figures": [figures[key] for key in PRIMARY_OUTPUTS],
        "optional_diagnostics": [figures[key] for key in OPTIONAL_OUTPUTS],
        "physical_axes_um": {
            "serial": [float(display_axes[0][0]), float(display_axes[0][-1])],
            "row": [float(display_axes[1][0]), float(display_axes[1][-1])],
            "left_right": [
                float(display_axes[2][0]), float(display_axes[2][-1])
            ],
        },
        "physical_serial_positions": len(display_axes[0]),
        "observed_pv_sections": len(pv.observed),
        "observed_physical_indices": pv.observed.tolist(),
        "matched_section_physical_indices": sorted(pairs),
        "support_threshold": threshold,
        "inplane_downsample_factor": INPLANE_DOWNSAMPLE,
        "matched_sections_inplane_downsample_factor": 1,
        "serial_downsample_factor": 1,
        "gap_policy": (
            "All 2,846 physical serial positions are retained; only the 285 "
            "observed PV planes receive signal/support; no interpolation, "
            "densification, or gap filling is performed."
        ),
        "comparison_display": {
            "full_stack": (
                "Separate native RGB PV and Nissl panels at matched orthogonal "
                "positions; Nissl restricted to common PV-observed support; "
                "no blending, channel remapping, or support-color encoding"
            ),
            "matched_sections": (
                "Native RGB PV and Nissl at the same five physical positions; "
                "identical geometry and framing; unsupported pixels shown gray"
            ),
        },
        "registration_invoked": False,
        "densification_invoked": False,
        "postprocessed_datasets_written": False,
    }


def refresh_combined(*, threshold: float = SUPPORT_THRESHOLD) -> Path:
    """Publish only the combined orthogonal figure and updated provenance."""
    if not np.isfinite(threshold) or threshold <= 0.0 or threshold > 1.0:
        raise ValueError("Support threshold must be finite and in (0, 1]")
    pv, nissl, native_axes = _load_inputs()
    display_axes = [
        native_axes[0],
        coarse.block_axis(native_axes[1], INPLANE_DOWNSAMPLE),
        coarse.block_axis(native_axes[2], INPLANE_DOWNSAMPLE),
    ]
    final = _output_paths()[1]
    required = {
        final / OUTPUTS[key]
        for key in ("symmetric_pv", "pv_nissl_matched_sections", "pv_coverage")
    }
    if not final.is_dir() or any(not path.is_file() for path in required):
        raise RuntimeError("Combined-only refresh requires the completed QC output")
    pv_planes, nissl_planes, pairs = _build_planes_and_pairs(
        pv, nissl, native_axes, display_axes
    )
    target = final / OUTPUTS["pv_nissl_combined"]
    temporary = target.with_name(f".{target.stem}.tmp{target.suffix}")
    if temporary.exists():
        raise FileExistsError(f"Refusing existing temporary output: {temporary}")
    try:
        _save_combined_orthogonal(
            temporary, pv_planes, nissl_planes, display_axes, threshold
        )
        os.replace(temporary, target)
        coarse.atomic_json(
            final / "provenance.json",
            _provenance(final, display_axes, pv, pairs, threshold),
        )
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Published combined PV/Nissl QC: {target}", flush=True)
    return target


def run(*, dry_run: bool = False, threshold: float = SUPPORT_THRESHOLD) -> Path | None:
    if not np.isfinite(threshold) or threshold <= 0.0 or threshold > 1.0:
        raise ValueError("Support threshold must be finite and in (0, 1]")
    pv, nissl, native_axes = _load_inputs()
    display_axes = [
        native_axes[0],
        coarse.block_axis(native_axes[1], INPLANE_DOWNSAMPLE),
        coarse.block_axis(native_axes[2], INPLANE_DOWNSAMPLE),
    ]
    stage, final = _output_paths()
    if final.exists() or stage.exists():
        raise FileExistsError(f"Refusing existing QC output: {final}")
    print(
        json.dumps(
            {
                "pv_dataset": str(PV_DATASET.resolve()),
                "nissl_dataset": str(NISSL_DATASET.resolve()),
                "output_directory": str(final.resolve()),
                "physical_serial_positions": len(display_axes[0]),
                "observed_pv_sections": len(pv.observed),
                "support_threshold": threshold,
            },
            indent=2,
        ),
        flush=True,
    )
    if dry_run:
        return None

    QC_ROOT.mkdir(parents=True, exist_ok=True)
    stage.mkdir()
    try:
        pv_planes, nissl_planes, pairs = _build_planes_and_pairs(
            pv, nissl, native_axes, display_axes
        )
        support_qc._save_support_aware_orthogonal(
            stage / OUTPUTS["symmetric_pv"],
            pv_planes,
            display_axes,
            threshold,
            title=(
                "Symmetric PV — full 2,846-position physical stack\n"
                "Five observed transverse sections; orthogonal views retain empty "
                "serial planes. Gray = unobserved/unsupported, not zero signal."
            ),
            supported_label="observed PV support",
            draw_support_boundary=False,
            image_interpolation="none",
        )
        _save_combined_orthogonal(
            stage / OUTPUTS["pv_nissl_combined"],
            pv_planes, nissl_planes, display_axes, threshold,
        )
        _save_matched_sections(
            stage / OUTPUTS["pv_nissl_matched_sections"],
            pairs, native_axes, threshold,
        )
        support_qc._save_coverage_orthogonal(
            stage / OUTPUTS["pv_coverage"],
            pv_planes,
            display_axes,
            threshold,
            title=(
                "Symmetric PV observed coverage on the full physical stack\n"
                f"Gray = support < {threshold:g}; colored planes are observed PV, "
                "with no serial interpolation or gap filling."
            ),
            colorbar_label=(
                "PV support after support-weighted in-plane downsampling"
            ),
            image_interpolation="none",
        )
        provenance = _provenance(
            final, display_axes, pv, pairs, threshold
        )
        (stage / "provenance.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        expected = {Path(value) for value in OUTPUTS.values()} | {
            Path("provenance.json")
        }
        actual = {
            path.relative_to(stage) for path in stage.iterdir() if path.is_file()
        }
        if actual != expected or any(
            not (stage / path).stat().st_size for path in expected
        ):
            raise RuntimeError("Incomplete PV/Nissl QC output inventory")
        os.rename(stage, final)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    print(f"Published PV/Nissl QC: {final}", flush=True)
    return final


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--combined-only",
        action="store_true",
        help="write only the combined orthogonal figure and provenance",
    )
    parser.add_argument(
        "--support-threshold", type=float, default=SUPPORT_THRESHOLD
    )
    args = parser.parse_args(argv)
    if args.combined_only:
        if args.dry_run:
            parser.error("--dry-run and --combined-only are mutually exclusive")
        refresh_combined(threshold=args.support_threshold)
    else:
        run(dry_run=args.dry_run, threshold=args.support_threshold)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
