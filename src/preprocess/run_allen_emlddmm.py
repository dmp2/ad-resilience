"""Run stop-gated direct-MRI to Allen mixed serial-section EM-LDDMM."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import logging
import os
import subprocess
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import torch

from preprocess.prepare_allen_emlddmm_inputs import (
    accepted_loader_axes,
    sha256_file,
)


LOG = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PIN_FILE = PROJECT_ROOT / "configs" / "emlddmm-upstream-commit.txt"
DEFAULT_DATASET = Path(
    "data/derivatives/allen/specimen_708424/emlddmm_7t_symmetric"
)
DEFAULT_MRI_PROVENANCE = Path(
    "data/derivatives/allen/specimen_708424/mri_7t_whole/mri_provenance.json"
)
DEFAULT_REGISTRATION_CONTRACT = Path(
    "configs/emlddmm/allen_708424_hist_symmetric_to_mri7t_t1.json"
)
VIEW_NAMES = {
    "all": ("HIST_ALL", "mixed"),
    "nissl": ("HIST_NISSL", "nissl"),
    "pv": ("HIST_PV", "pv"),
}
EXPECTED_PRESENT_BY_SERIES = {
    "pilot": {"all": 12, "nissl": 8, "pv": 4},
    "full": {"all": 928, "nissl": 641, "pv": 287},
}


def pinned_emlddmm() -> ModuleType:
    configured = os.environ.get("EMLDDMM_REPO")
    checkout = (
        Path(configured).expanduser().resolve()
        if configured
        else PROJECT_ROOT.parent / "emlddmm"
    )
    expected = PIN_FILE.read_text(encoding="utf-8").strip()
    actual = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if actual != expected:
        raise RuntimeError(
            f"EM-LDDMM checkout is {actual}; required pinned commit is {expected}"
        )
    spec = importlib.util.spec_from_file_location(
        "allen_runner_pinned_emlddmm", checkout / "emlddmm.py"
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import pinned EM-LDDMM from {checkout}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def mri_physical_axes_from_provenance(
    provenance: dict[str, Any],
) -> list[np.ndarray]:
    """Return the separable MRI physical axes used by the pinned loader adapter."""
    if provenance.get("status") != "ready" or provenance.get("units") != "millimeter":
        raise RuntimeError("MRI provenance must be ready with millimetre geometry")
    dimensions = tuple(int(value) for value in provenance["dimensions"])
    affine = np.asarray(provenance["voxel_to_physical_affine_mm"], dtype=np.float64)
    if affine.shape != (4, 4):
        raise ValueError("MRI provenance affine must be 4x4")
    linear = affine[:3, :3]
    if not np.allclose(linear, np.diag(np.diag(linear)), atol=1e-8):
        raise ValueError(
            "Pinned separable-coordinate adapter cannot represent oblique MRI axes"
        )
    return [
        (affine[axis, 3] + np.arange(size) * linear[axis, axis]) * 1000.0
        for axis, size in enumerate(dimensions)
    ]


def load_pinned_mri_image(
    emlddmm: ModuleType,
    *,
    mri_path: Path,
    provenance: dict[str, Any],
) -> Any:
    """Load nibabel-backed MRI through the pinned Image API and keep geometry.

    The pinned revision forwards its internal ``normalize`` flag to
    ``nibabel.load``. The scoped shim removes only that unsupported keyword;
    EM-LDDMM's own normalization still runs. The pinned fallback also centers
    nibabel coordinates, so the adapter restores separable affine axes and
    converts the provenance-declared millimetres to the histology package's
    micrometre coordinate unit.
    """

    physical_axes = mri_physical_axes_from_provenance(provenance)
    original_load = emlddmm.nibabel.load

    def compatible_nibabel_load(filename: str, **kwargs: Any) -> Any:
        kwargs.pop("normalize", None)
        return original_load(filename, **kwargs)

    emlddmm.nibabel.load = compatible_nibabel_load
    try:
        image = emlddmm.Image(
            space="MRI_7T_WHOLE", name="7T_T1", fpath=str(mri_path)
        )
    finally:
        emlddmm.nibabel.load = original_load
    dimensions = tuple(map(len, physical_axes))
    if tuple(image.data.shape[1:]) != dimensions:
        raise ValueError("Pinned MRI loader dimensions differ from provenance")
    image.x = physical_axes
    image.coordinate_units = "um"
    image.geometry_source = "mri_provenance_affine"
    return image


def load_physical_rows(dataset: Path) -> list[dict[str, str]]:
    path = dataset / "metadata" / "physical_sections.tsv"
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    sections = np.array([int(row["allen_section_number"]) for row in rows])
    if rows and not np.all(np.diff(sections) == 1):
        raise ValueError("physical_sections.tsv rows must be consecutive")
    return rows


def validated_config(config_path: Path, mode: str) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    required = {
        "slice_matching": True,
        "order": 1,
        "local_contrast": [[]],
    }
    for key, expected in required.items():
        if config.get(key) != expected:
            raise ValueError(
                f"{key} must equal {expected!r} for the mixed-stain runner"
            )
    if config.get("downJ") is None or any(
        int(level[0]) != 1 for level in config["downJ"]
    ):
        raise ValueError("Every downJ level must preserve the 50-um z lattice")
    config["full_outputs"] = mode == "pilot"
    config["n_draw"] = 0
    config["dtype"] = torch.float32
    return config


def _contract_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def validate_registration_contract(
    contract_path: Path,
    *,
    require_prepared_histology: bool = False,
) -> dict[str, Any]:
    """Validate graph direction separately from EM-LDDMM I/J conventions."""

    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    edge = contract.get("logical_edge", {})
    expected_edge = {
        "source_space": "HIST_SYMMETRIC",
        "source_view": "HIST_ALL",
        "target_space": "MRI_7T_WHOLE",
        "target_contrast": "7T_T1",
        "direction": "source_to_target",
    }
    if edge != expected_edge:
        raise ValueError(
            "Registration graph edge must be HIST_SYMMETRIC/HIST_ALL to "
            "MRI_7T_WHOLE/7T_T1"
        )
    adapter = contract.get("algorithm_adapter", {})
    if adapter != {
        "I": "MRI_7T_WHOLE/7T_T1",
        "I_role": "graph_target",
        "J": "HIST_SYMMETRIC/HIST_ALL",
        "J_role": "graph_source",
    }:
        raise ValueError("Pinned adapter must explicitly use I=MRI and J=histology")
    inputs = contract.get("inputs", {})
    forbidden_geometry = {
        "dimensions",
        "voxel_size",
        "voxel_size_mm",
        "orientation",
        "affine",
        "source_archive_sha256",
        "source_member_sha256",
    }
    if forbidden_geometry.intersection(inputs):
        raise ValueError("Registration config duplicates authoritative MRI geometry")
    provenance_path = _contract_path(inputs["mri_provenance"])
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if (
        provenance.get("status") != "ready"
        or provenance.get("space_name") != edge["target_space"]
        or provenance.get("contrast_name") != edge["target_contrast"]
    ):
        raise RuntimeError("Registration target MRI provenance is not ready")
    template_path = _contract_path(inputs["mri_template"])
    if template_path.resolve() != _contract_path(provenance["template_path"]).resolve():
        raise ValueError("Registration MRI path differs from its provenance")
    if inputs["mri_template_sha256"] != provenance["template_sha256"]:
        raise ValueError("Registration MRI checksum differs from its provenance")
    if sha256_file(template_path) != inputs["mri_template_sha256"]:
        raise ValueError("Registration MRI template checksum mismatch")
    histology_path = _contract_path(inputs["prepared_histology"])
    if require_prepared_histology:
        prepared = json.loads(
            (histology_path / "dataset.json").read_text(encoding="utf-8")
        )
        if (
            prepared.get("space_name") != "HIST_SYMMETRIC"
            or prepared.get("preparation_mode") != "preserve_source_grid"
        ):
            raise ValueError("Registration histology is not the preserved symmetric package")
    initialization = contract.get("initial_transform", {})
    if (
        initialization.get("type") not in {"rigid", "similitude"}
        or initialization.get("initial_midlines_must_coincide") is not False
    ):
        raise ValueError("Registration requires a rigid/similitude cross-space initialization")
    output = _contract_path(contract["output_directory"]).resolve()
    derivatives = (PROJECT_ROOT / "data" / "derivatives").resolve()
    if output == derivatives or derivatives in output.parents:
        raise ValueError("Registration output must be separate from all derivatives")
    reconstruction = contract.get("reconstruction_requests", [])
    if not any(
        request.get("payload") == "bilateral_annotations"
        and request.get("sampling") == "categorical_nearest_neighbor"
        and request.get("target_space") == "MRI_7T_WHOLE"
        for request in reconstruction
    ):
        raise ValueError("Contract lacks label-preserving MRI-space reconstruction")
    if contract.get("synthetic_mri_required") is not False:
        raise ValueError("Direct registration must not require a synthetic MRI")
    validated_config(
        _contract_path(contract["emlddmm_parameter_file"]), "full"
    )
    return {**contract, "resolved_mri_provenance": provenance}


def _available_host_bytes() -> int:
    values: dict[str, int] = {}
    with Path("/proc/meminfo").open(encoding="utf-8") as stream:
        for line in stream:
            key, value = line.split(":", maxsplit=1)
            values[key] = int(value.strip().split()[0]) * 1024
    return values["MemAvailable"]


def estimate_registration_bytes(
    hist_shape_zyx: tuple[int, int, int],
    mri_shape_zyx: tuple[int, int, int],
    *,
    full_outputs: bool,
) -> int:
    """Conservative float32 working-set estimate for the pinned core."""

    hist_voxels = int(np.prod(hist_shape_zyx))
    mri_voxels = int(np.prod(mri_shape_zyx))
    # Histology RGB, transformed prediction/error, W0 and three mixture
    # responsibilities dominate.  The multiplier intentionally includes
    # interpolation/gradient temporaries and one MRI-domain working copy.
    hist_arrays = 18 if full_outputs else 14
    mri_arrays = 10
    return 4 * (hist_arrays * hist_voxels + mri_arrays * mri_voxels)


def select_device(estimated_bytes: int) -> tuple[str, dict[str, Any]]:
    host_available = _available_host_bytes()
    report: dict[str, Any] = {
        "estimated_peak_bytes": estimated_bytes,
        "host_available_bytes": host_available,
        "maximum_fraction": 0.7,
    }
    if torch.cuda.is_available():
        gpu_free, gpu_total = (
            int(value) for value in torch.cuda.mem_get_info(0)
        )
        report["gpu_total_bytes"] = gpu_total
        report["gpu_available_bytes"] = gpu_free
        if estimated_bytes <= gpu_free * 0.7:
            report["selected_device"] = "cuda:0"
            return "cuda:0", report
    if estimated_bytes <= host_available * 0.7:
        report["selected_device"] = "cpu"
        return "cpu", report
    raise MemoryError(
        "Estimated registration peak exceeds 70% of both available GPU and "
        "host memory; input resolution will not be silently changed"
    )


def _finite(name: str, value: Any) -> None:
    array = (
        value.detach().cpu().numpy()
        if isinstance(value, torch.Tensor)
        else np.asarray(value)
    )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"Pilot output {name} contains NaN or Inf")


def _status_rows(view_dir: Path) -> list[dict[str, str]]:
    with (view_dir / "samples.tsv").open(
        encoding="utf-8", newline=""
    ) as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def validate_loaded_support(
    support: np.ndarray, sample_rows: list[dict[str, str]]
) -> dict[str, int]:
    if support.shape[0] != len(sample_rows):
        raise ValueError(
            f"TSV/support positional mismatch: {len(sample_rows)} rows versus "
            f"{support.shape[0]} z coordinates"
        )
    present_count = 0
    absent_count = 0
    for index, row in enumerate(sample_rows):
        has_support = bool(np.any(support[index] > 0))
        if row["status"] == "present":
            present_count += 1
            if not has_support:
                raise ValueError(
                    f"Present row {index} ({row['sample_id']}) has zero W0 support"
                )
        elif row["status"] == "absent":
            absent_count += 1
            if has_support:
                raise ValueError(
                    f"Absent row {index} ({row['sample_id']}) has nonzero W0"
                )
        else:
            raise ValueError(f"Unexpected status {row['status']!r}")
    return {"present": present_count, "absent": absent_count}


def _mri_metadata(
    mri_path: Path, provenance_path: Path = DEFAULT_MRI_PROVENANCE
) -> dict[str, Any]:
    metadata = json.loads(provenance_path.read_text(encoding="utf-8"))
    if metadata.get("status") != "ready":
        raise RuntimeError("MRI provenance status is not ready")
    resolved = _contract_path(metadata["template_path"]).resolve()
    if resolved != mri_path.resolve():
        raise ValueError("MRI input path differs from compact provenance")
    if metadata["template_sha256"] != sha256_file(mri_path):
        raise ValueError("MRI derivative checksum differs from provenance")
    if not np.allclose(
        metadata["corrected_voxel_size_mm"], [0.2, 0.2, 0.2]
    ):
        raise ValueError("MRI provenance does not verify 200-um resolution")
    return metadata


def load_reviewed_initial_affine(
    matrix_path: Path,
    *,
    source_image_identifier: str,
    target_view: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load a finite 4x4 matrix with checksum-matched review metadata."""

    review_path = matrix_path.with_suffix(".json")
    review = json.loads(review_path.read_text(encoding="utf-8"))
    if review.get("review_status") != "accepted":
        raise RuntimeError(
            "Coarse initialization review_status must be 'accepted'"
        )
    if review.get("source_image_identifier") != source_image_identifier:
        raise ValueError("Initialization review names a different MRI source")
    if review.get("target_view") != target_view:
        raise ValueError("Initialization review names a different histology view")
    if review.get("matrix_sha256") != sha256_file(matrix_path):
        raise ValueError("Initialization matrix checksum differs from its review")
    matrix = np.loadtxt(matrix_path)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError("Reviewed initialization must be a finite 4x4 matrix")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0]):
        raise ValueError("Reviewed initialization has an invalid homogeneous row")
    return matrix, review


def require_unused_output_root(output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            "Registration output root is nonempty; refusing to overwrite an "
            "existing graph edge or transform"
        )


def validate_registration_domain(
    rows: list[dict[str, str]], mode: str, series: str
) -> None:
    sections = [int(row["section_number"]) for row in rows]
    expected = (
        list(range(1448, 1478))
        if mode == "pilot"
        else list(range(36, 2882))
    )
    if sections != expected:
        raise ValueError(
            f"{mode} registration requires the exact canonical "
            f"{expected[0]}-{expected[-1]} row/z domain"
        )
    observed = sum(
        row["status"] == "present"
        and (series == "all" or row["stain"] == series)
        for row in rows
    )
    required = EXPECTED_PRESENT_BY_SERIES[mode][series]
    if observed != required:
        raise ValueError(
            f"{mode} {series} view requires {required} present sections; "
            f"found {observed}"
        )


def run_registration(
    *,
    dataset: Path,
    mri_path: Path,
    config_path: Path,
    initial_affine_path: Path,
    output_dir: Path,
    mode: str,
    series: str,
    mri_provenance_path: Path = DEFAULT_MRI_PROVENANCE,
    contract_path: Path = DEFAULT_REGISTRATION_CONTRACT,
) -> dict[str, Any]:
    contract = validate_registration_contract(
        contract_path, require_prepared_histology=True
    )
    if contract.get("registration_execution_enabled") is not True:
        raise RuntimeError(
            "Registration contract remains non-executable until initialization review"
        )
    contracted_paths = {
        "prepared histology": (
            dataset,
            _contract_path(contract["inputs"]["prepared_histology"]),
        ),
        "MRI template": (
            mri_path,
            _contract_path(contract["inputs"]["mri_template"]),
        ),
        "MRI provenance": (
            mri_provenance_path,
            _contract_path(contract["inputs"]["mri_provenance"]),
        ),
        "EM-LDDMM parameters": (
            config_path,
            _contract_path(contract["emlddmm_parameter_file"]),
        ),
        "registration output": (
            output_dir,
            _contract_path(contract["output_directory"]),
        ),
    }
    for name, (actual, expected) in contracted_paths.items():
        if actual.resolve() != expected.resolve():
            raise ValueError(f"Runner {name} differs from registration contract")
    if mode not in {"pilot", "full"}:
        raise ValueError("mode must be pilot or full")
    view_name, image_name = VIEW_NAMES[series]
    rows = load_physical_rows(dataset)
    validate_registration_domain(rows, mode, series)
    canvas_audit = json.loads(
        (dataset / "metadata" / "loader_canvas_audit.json").read_text(
            encoding="utf-8"
        )
    )
    axes = accepted_loader_axes(rows, canvas_audit)
    view_dir = dataset / "inputs" / "views" / view_name
    sample_rows = _status_rows(view_dir)
    if len(sample_rows) != len(axes[0]):
        raise ValueError("samples.tsv and selected canonical z axis differ")

    mri_metadata = _mri_metadata(mri_path, mri_provenance_path)
    initial_affine, initialization_review = load_reviewed_initial_affine(
        initial_affine_path,
        source_image_identifier=mri_metadata["contrast_name"],
        target_view=view_name,
    )
    require_unused_output_root(output_dir)
    hist_shape = (len(axes[0]), len(axes[1]), len(axes[2]))
    mri_shape = tuple(int(v) for v in mri_metadata["dimensions"])
    config = validated_config(config_path, mode)
    config["A"] = initial_affine
    estimated = estimate_registration_bytes(
        hist_shape, mri_shape, full_outputs=config["full_outputs"]
    )
    device, memory_report = select_device(estimated)
    config["device"] = device

    emlddmm = pinned_emlddmm()
    source_identifier = mri_metadata["contrast_name"]
    source = load_pinned_mri_image(
        emlddmm, mri_path=mri_path, provenance=mri_metadata
    )
    target = emlddmm.Image(
        space="HIST_SYMMETRIC",
        name=view_name,
        fpath=str(view_dir),
        x=axes,
    )
    support_counts = validate_loaded_support(target.mask, sample_rows)

    outputs = emlddmm.emlddmm_multiscale(
        I=source.data,
        xI=[source.x],
        J=target.data,
        xJ=[target.x],
        W0=target.mask,
        **config,
    )
    final = outputs[-1]
    output_dir.mkdir(parents=True, exist_ok=True)
    emlddmm.write_transform_outputs(
        str(output_dir), final, source, target
    )
    emlddmm.write_qc_outputs(str(output_dir), final, source, target)

    validation: dict[str, Any] = {
        "mode": mode,
        "series": series,
        "logical_source_space": "HIST_SYMMETRIC",
        "logical_source_image": view_name,
        "logical_target_space": "MRI_7T_WHOLE",
        "logical_target_image": source_identifier,
        "logical_transform_direction": "histology_to_mri",
        "algorithm_I": "MRI_7T_WHOLE/7T_T1",
        "algorithm_J": f"HIST_SYMMETRIC/{view_name}",
        "row_count": len(rows),
        "support_counts": support_counts,
        "memory": memory_report,
        "full_outputs": config["full_outputs"],
        "n_draw": config["n_draw"],
        "local_contrast_multiscale_argument": config["local_contrast"],
        "output_directory": str(output_dir),
        "initial_affine": str(initial_affine_path),
        "initial_affine_sha256": sha256_file(initial_affine_path),
        "initialization_review": initialization_review,
    }
    for name in ("A", "v", "A2d"):
        _finite(name, final[name])

    if mode == "pilot":
        retained: dict[str, np.ndarray] = {}
        for name in ("W0", "WM", "WA", "WB", "coeffs", "A", "A2d"):
            _finite(name, final[name])
            value = final[name]
            retained[name] = (
                value.detach().cpu().numpy()
                if isinstance(value, torch.Tensor)
                else np.asarray(value)
            )
        np.savez_compressed(output_dir / "pilot_numerical_outputs.npz", **retained)
        validation["retained_numerical_outputs"] = str(
            output_dir / "pilot_numerical_outputs.npz"
        )
        validation["coefficient_shape"] = list(retained["coeffs"].shape)
        if retained["coeffs"].shape[0] != len(rows):
            raise ValueError(
                "Pilot did not estimate independent coefficients per section"
            )
        if retained["A2d"].shape[0] != len(rows):
            raise ValueError("Pilot did not return one A2d transform per TSV row")
        present_indices = [
            index
            for index, sample in enumerate(sample_rows)
            if sample["status"] == "present"
        ]
        validation["present_rigid_transform_count"] = len(present_indices)
        stains = ("nissl", "pv") if series == "all" else (series,)
        matching_contribution: dict[str, float] = {}
        for stain in stains:
            indices = [
                index
                for index, row in enumerate(rows)
                if row["stain"] == stain and sample_rows[index]["status"] == "present"
            ]
            contribution = float(
                np.sum(retained["WM"][indices] * retained["W0"][indices])
            )
            if not np.isfinite(contribution) or contribution <= 0:
                raise ValueError(
                    f"Pilot has no finite positive matching contribution for {stain}"
                )
            matching_contribution[stain] = contribution
        validation["matching_class_weight_sum_by_stain"] = matching_contribution

    validation_path = output_dir / "allen_emlddmm_validation.json"
    validation_path.write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return validation


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("pilot", "full"))
    parser.add_argument("--series", choices=("all", "nissl", "pv"), default="all")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--mri", type=Path, required=True)
    parser.add_argument(
        "--mri-provenance", type=Path, default=DEFAULT_MRI_PROVENANCE
    )
    parser.add_argument(
        "--contract", type=Path, default=DEFAULT_REGISTRATION_CONTRACT
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--initial-affine", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    logging.basicConfig(level=args.log_level, format="%(levelname)s %(message)s")
    validation = run_registration(
        dataset=args.dataset,
        mri_path=args.mri,
        config_path=args.config,
        initial_affine_path=args.initial_affine,
        output_dir=args.output_dir,
        mode=args.mode,
        series=args.series,
        mri_provenance_path=args.mri_provenance,
        contract_path=args.contract,
    )
    LOG.info(
        "Completed %s MRI7T/%s -> %s/%s",
        args.mode,
        validation["logical_source_image"],
        validation["logical_target_space"],
        validation["logical_target_image"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
