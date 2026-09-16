"""Inspect SEA-AD pseudobulk H5AD structure, taxonomy, and ROI coverage."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import numpy as np

from acquisition.s3_download import SelectionItem, load_selection


CURRENT_ROIS = ("DFC", "MEC", "MTG", "STG", "V1C", "HIP")


@dataclass(frozen=True)
class ObjectSummary:
    lineage: str
    local_file: str
    file_size_bytes: int
    n_obs: int
    n_vars: int
    obs_columns: list[str]
    var_columns: list[str]
    layers: list[str]
    n_donors: int
    n_libraries: int
    n_unique_donor_region_supertype: int
    repeated_rows_beyond_donor_region_supertype: int
    row_grain: str
    brain_regions: list[str]
    class_names: list[str]
    subclass_names: list[str]
    n_supertypes: int
    supertypes: list[str]
    total_n_nuclei: int
    nuclei_min: int
    nuclei_median: float
    nuclei_max: int
    x_storage: str
    x_dtype: str
    x_integer_like: bool
    x_nonnegative: bool
    x_consistent_with_summed_umi: bool
    x_assessment: str


def _decode(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values]
    )


def read_dataframe_column(group: h5py.Group, name: str) -> np.ndarray:
    node = group[name]
    if isinstance(node, h5py.Group):
        categories = _decode(node["categories"][:])
        codes = node["codes"][:]
        result = np.full(codes.shape, "", dtype=object)
        valid = codes >= 0
        result[valid] = categories[codes[valid]]
        return result
    values = node[:]
    if values.dtype.kind in "SO":
        return _decode(values)
    return values


def _matrix_shape(node: h5py.Dataset | h5py.Group) -> tuple[int, int]:
    if isinstance(node, h5py.Dataset):
        return tuple(int(value) for value in node.shape)
    shape = node.attrs.get("shape")
    if shape is None:
        raise ValueError("Sparse X lacks an AnnData shape attribute")
    return tuple(int(value) for value in shape)


def inspect_count_storage(node: h5py.Dataset | h5py.Group) -> tuple[str, str, bool, bool]:
    """Check all stored values in bounded chunks without materializing X."""
    if isinstance(node, h5py.Group):
        values = node["data"]
        storage = str(node.attrs.get("encoding-type", "sparse_matrix"))
    else:
        values = node
        storage = "dense"
    dtype = str(values.dtype)
    integer_like = True
    nonnegative = True
    if isinstance(values, h5py.Dataset) and values.ndim > 1:
        step = max(1, min(values.shape[0], 64))
    else:
        step = max(1, min(values.shape[0], 1_000_000))
    for start in range(0, values.shape[0], step):
        block = np.asarray(values[start : start + step])
        if not np.isfinite(block).all():
            integer_like = False
            nonnegative = False
            break
        if np.any(block < 0):
            nonnegative = False
        if block.dtype.kind not in "iu" and not np.all(block == np.floor(block)):
            integer_like = False
    return storage, dtype, bool(integer_like), bool(nonnegative)


def classify_immune_supertype(supertype: str) -> tuple[str, str]:
    if supertype == "Monocyte":
        return "monocytes", "Released supertype label is explicit."
    if supertype == "Lymphocyte":
        return "lymphocytes", "Released supertype label is explicit."
    if supertype.startswith("Micro-PVM_"):
        return (
            "microglia_or_PVM_macrophage_like_ambiguous",
            "The released Micro-PVM label does not distinguish microglia from PVM/macrophage-like cells.",
        )
    return "other_or_ambiguous_immune", "No more specific documented immune grouping was assigned."


def inspect_object(
    item: SelectionItem, project_root: Path
) -> tuple[ObjectSummary, list[dict], list[dict], list[dict]]:
    path = project_root / item.local_path
    if not path.is_file():
        raise FileNotFoundError(f"Selected object is missing: {path}")
    actual_size = path.stat().st_size
    if actual_size != item.size_bytes:
        raise ValueError(
            f"Selected object has wrong size: {path} ({actual_size} != {item.size_bytes})"
        )
    with h5py.File(path, "r") as handle:
        obs = handle["obs"]
        var = handle["var"]
        required = {
            "Donor ID",
            "Brain Region",
            "Class",
            "Subclass",
            "Supertype",
            "Number of nuclei",
            "library_prep",
        }
        missing = required - set(obs.keys())
        if missing:
            raise ValueError(f"{path} is missing required obs fields: {sorted(missing)}")
        donors = read_dataframe_column(obs, "Donor ID").astype(str)
        regions = read_dataframe_column(obs, "Brain Region").astype(str)
        classes = read_dataframe_column(obs, "Class").astype(str)
        subclasses = read_dataframe_column(obs, "Subclass").astype(str)
        supertypes = read_dataframe_column(obs, "Supertype").astype(str)
        nuclei = np.asarray(read_dataframe_column(obs, "Number of nuclei"), dtype=np.int64)
        libraries = read_dataframe_column(obs, "library_prep").astype(str)
        n_obs, n_vars = _matrix_shape(handle["X"])
        if n_obs != len(donors):
            raise ValueError(f"X and obs row counts disagree in {path}")
        storage, dtype, integer_like, nonnegative = inspect_count_storage(handle["X"])
        obs_columns = sorted(
            value.decode() if isinstance(value, bytes) else str(value)
            for value in obs.attrs.get("column-order", [])
        )
        var_columns = sorted(
            value.decode() if isinstance(value, bytes) else str(value)
            for value in var.attrs.get("column-order", [])
        )
        layers = sorted(str(key) for key in handle.get("layers", {}).keys())

    consistent = integer_like and nonnegative
    assessment = (
        "X contains only finite, nonnegative, integer-valued entries, consistent with summed raw UMI counts."
        if consistent
        else "X failed at least one raw-count check (finite, nonnegative, integer-valued)."
    )
    unique_supertypes = sorted(set(supertypes))
    summary = ObjectSummary(
        lineage=item.lineage,
        local_file=str(path),
        file_size_bytes=actual_size,
        n_obs=n_obs,
        n_vars=n_vars,
        obs_columns=obs_columns,
        var_columns=var_columns,
        layers=layers,
        n_donors=len(set(donors)),
        n_libraries=len(set(libraries)),
        n_unique_donor_region_supertype=len(set(zip(donors, regions, supertypes))),
        repeated_rows_beyond_donor_region_supertype=(
            n_obs - len(set(zip(donors, regions, supertypes)))
        ),
        row_grain="Donor ID x Brain Region x Supertype x library_prep",
        brain_regions=sorted(set(regions)),
        class_names=sorted(set(classes)),
        subclass_names=sorted(set(subclasses)),
        n_supertypes=len(unique_supertypes),
        supertypes=unique_supertypes,
        total_n_nuclei=int(nuclei.sum()),
        nuclei_min=int(nuclei.min()),
        nuclei_median=float(np.median(nuclei)),
        nuclei_max=int(nuclei.max()),
        x_storage=storage,
        x_dtype=dtype,
        x_integer_like=integer_like,
        x_nonnegative=nonnegative,
        x_consistent_with_summed_umi=consistent,
        x_assessment=assessment,
    )
    coverage = []
    for roi in CURRENT_ROIS:
        mask = regions == roi
        coverage.append(
            {
                "lineage": item.lineage,
                "ROI": roi,
                "n_donors": len(set(donors[mask])),
                "n_donor_supertype_rows": len(set(zip(donors[mask], supertypes[mask]))),
                "n_pseudobulk_rows": int(mask.sum()),
                "n_supertypes": len(set(supertypes[mask])),
                "total_n_nuclei": int(nuclei[mask].sum()),
            }
        )
    donor_counts_by_supertype = []
    for roi in CURRENT_ROIS:
        roi_mask = regions == roi
        for supertype in unique_supertypes:
            mask = roi_mask & (supertypes == supertype)
            donor_counts_by_supertype.append(
                {
                    "lineage": item.lineage,
                    "ROI": roi,
                    "supertype": supertype,
                    "n_donors": len(set(donors[mask])),
                    "n_pseudobulk_rows": int(mask.sum()),
                    "total_n_nuclei": int(nuclei[mask].sum()),
                }
            )
    taxonomy = []
    for supertype in unique_supertypes:
        group = interpretation = ""
        if item.lineage == "Immune":
            group, interpretation = classify_immune_supertype(supertype)
        taxonomy.append(
            {
                "lineage": item.lineage,
                "subclass": "; ".join(sorted(set(subclasses[supertypes == supertype]))),
                "supertype": supertype,
                "immune_group": group,
                "interpretation": interpretation,
            }
        )
    return summary, coverage, taxonomy, donor_counts_by_supertype


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def inspect_selection(selection_path: Path, project_root: Path, output_dir: Path) -> list[ObjectSummary]:
    selection = load_selection(selection_path)
    summaries: list[ObjectSummary] = []
    coverage: list[dict] = []
    taxonomy: list[dict] = []
    donor_counts_by_supertype: list[dict] = []
    for item in selection.items:
        summary, item_coverage, item_taxonomy, item_donor_counts = inspect_object(
            item, project_root
        )
        summaries.append(summary)
        coverage.extend(item_coverage)
        taxonomy.extend(item_taxonomy)
        donor_counts_by_supertype.extend(item_donor_counts)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "pseudobulk_structural_summary.json").write_text(
        json.dumps({"objects": [asdict(summary) for summary in summaries]}, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_csv(output_dir / "pseudobulk_roi_coverage.csv", coverage)
    _write_csv(output_dir / "pseudobulk_taxonomy.csv", taxonomy)
    _write_csv(
        output_dir / "pseudobulk_roi_supertype_donor_counts.csv",
        donor_counts_by_supertype,
    )
    return summaries


def print_report(
    summaries: list[ObjectSummary],
    coverage_path: Path,
    taxonomy_path: Path,
    donor_counts_path: Path,
) -> None:
    for summary in summaries:
        print(
            f"{summary.lineage}: {summary.n_obs} x {summary.n_vars}; "
            f"{summary.n_donors} donors; {summary.n_libraries} libraries; "
            f"{len(summary.brain_regions)} regions; {summary.n_supertypes} supertypes; "
            f"nuclei min/median/max "
            f"{summary.nuclei_min}/{summary.nuclei_median:g}/{summary.nuclei_max}; "
            f"X integer-like={summary.x_integer_like}, summed-UMI-consistent="
            f"{summary.x_consistent_with_summed_umi}"
        )
        print(f"  supertypes: {', '.join(summary.supertypes)}")
    print(f"Coverage table: {coverage_path}")
    print(f"Taxonomy table: {taxonomy_path}")
    print(f"ROI-by-supertype donor counts: {donor_counts_path}")
