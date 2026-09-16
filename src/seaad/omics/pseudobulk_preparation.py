"""Prepare SEA-AD library-level pseudobulks as donor-region-supertype counts."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np

from acquisition.s3_download import SelectionItem, load_selection
from omics.pseudobulk_inspection import read_dataframe_column


RELEASE = "Multiregion 2026 (2026-06-22)"
OUTPUT_RELATIVE = Path("data/derivatives/sea-ad/omics_prepared")
H5AD_RELATIVE = OUTPUT_RELATIVE / "multiregion_2026"
KEY_COLUMNS = ("Donor ID", "Brain Region", "Supertype")
PROVENANCE_COLUMNS = (
    "library_prep",
    "sample_name",
    "ar_id",
    "load_name",
    "exp_component_vendor_name",
    "rna_amplification",
    "method",
    "alignment",
    "batch_vendor_name",
    "facs_population_plan",
)
EXCLUDED_CONSTANT_COLUMNS = {
    "Donor ID",
    "Brain Region",
    "Supertype",
    "Subclass",
    "Class",
    "Number of nuclei",
}


class PreparationError(RuntimeError):
    """The source objects cannot be prepared without violating an invariant."""


@dataclass(frozen=True)
class FileFingerprint:
    size_bytes: int
    mtime_ns: int
    sha256: str


@dataclass
class PreparedLineage:
    lineage: str
    source_path: str
    prepared_path: str
    source_rows: int
    prepared_rows: int
    unresolved_rows: int
    unresolved_groups: int
    safely_aggregated_groups: int
    n_donors: int
    n_regions: int
    n_supertypes: int
    n_genes: int
    source_n_donors: int
    source_n_regions: int
    source_n_supertypes: int
    source_total_umi: int
    prepared_total_umi: int
    source_total_nuclei: int
    prepared_total_nuclei: int
    source_fingerprint_before: dict[str, Any]
    source_fingerprint_after: dict[str, Any]
    preserved_constant_metadata: list[str]
    omitted_nonconstant_metadata: list[str]
    gene_index_field: str
    gene_id_field: str
    gene_symbols_duplicated: int
    gene_ids_duplicated: int
    workflow_identifiers_one_to_one_with_library: dict[str, bool]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(path: Path) -> FileFingerprint:
    stat = path.stat()
    return FileFingerprint(stat.st_size, stat.st_mtime_ns, sha256_file(path))


def dataframe_columns(group: h5py.Group) -> list[str]:
    values = group.attrs.get("column-order", [])
    return [value.decode() if isinstance(value, bytes) else str(value) for value in values]


def _canonical(value: Any) -> str:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, float) and np.isnan(value):
        return "<NA>"
    return str(value)


def _distinct(values: np.ndarray, indices: list[int]) -> list[str]:
    return sorted({_canonical(values[index]) for index in indices})


def _joined(values: np.ndarray, indices: list[int]) -> str:
    return "|".join(_distinct(values, indices))


def _slug(name: str) -> str:
    result = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").lower()
    return result or "field"


def _is_one_to_one(left: np.ndarray, right: np.ndarray) -> bool:
    forward: dict[str, set[str]] = defaultdict(set)
    reverse: dict[str, set[str]] = defaultdict(set)
    for lhs, rhs in zip(left, right):
        forward[_canonical(lhs)].add(_canonical(rhs))
        reverse[_canonical(rhs)].add(_canonical(lhs))
    return all(len(values) == 1 for values in forward.values()) and all(
        len(values) == 1 for values in reverse.values()
    )


def _write_array(group: h5py.Group, name: str, values: Iterable[Any]) -> None:
    array = np.asarray(list(values))
    if array.dtype.kind in "iufb":
        dataset = group.create_dataset(name, data=array)
        dataset.attrs["encoding-type"] = "array"
        dataset.attrs["encoding-version"] = "0.2.0"
        return
    text = np.asarray([_canonical(value) if _canonical(value) != "<NA>" else "" for value in array])
    dataset = group.create_dataset(name, data=text.astype(object), dtype=h5py.string_dtype("utf-8"))
    dataset.attrs["encoding-type"] = "string-array"
    dataset.attrs["encoding-version"] = "0.2.0"


def _write_dataframe(
    handle: h5py.File,
    name: str,
    columns: dict[str, Iterable[Any]],
    *,
    index_name: str = "_index",
) -> None:
    group = handle.create_group(name)
    group.attrs["encoding-type"] = "dataframe"
    group.attrs["encoding-version"] = "0.2.0"
    group.attrs["_index"] = index_name
    order = [column for column in columns if column != index_name]
    group.attrs.create("column-order", np.asarray(order, dtype=h5py.string_dtype("utf-8")))
    for column, values in columns.items():
        _write_array(group, column, values)


def _empty_anndata_group(handle: h5py.File, name: str) -> None:
    group = handle.create_group(name)
    group.attrs["encoding-type"] = "dict"
    group.attrs["encoding-version"] = "0.1.0"


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if not rows and not fieldnames:
        raise PreparationError(f"Cannot write empty table without columns: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = fieldnames or list(rows[0])
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _taxonomy_fields(lineage: str, supertype: str) -> tuple[str, str]:
    if lineage == "Immune" and supertype.startswith("Micro-PVM_"):
        return "Micro-PVM", "unresolved_microglia_vs_PVM"
    if lineage == "Immune" and supertype in {"Monocyte", "Lymphocyte"}:
        return supertype, "released_label_explicit"
    return lineage, "released_supertype_preserved"


def _source_groups(
    columns: dict[str, np.ndarray],
) -> tuple[list[tuple[str, str, str]], list[list[int]]]:
    keys_by_row = list(
        zip(
            columns["Donor ID"].astype(str),
            columns["Brain Region"].astype(str),
            columns["Supertype"].astype(str),
        )
    )
    members: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for row, key in enumerate(keys_by_row):
        members[key].append(row)
    keys = sorted(members)
    return keys, [members[key] for key in keys]


def _output_units(
    keys: list[tuple[str, str, str]], groups: list[list[int]], n_source: int
) -> tuple[
    list[tuple[str, str, str]],
    list[list[int]],
    list[str],
    list[int],
    np.ndarray,
]:
    """Retain undocumented repeated sample units instead of silently pooling."""
    output_keys: list[tuple[str, str, str]] = []
    units: list[list[int]] = []
    statuses: list[str] = []
    group_sizes: list[int] = []
    mapping = np.empty(n_source, dtype=np.int64)
    for key, indices in zip(keys, groups):
        if len(indices) == 1:
            output_keys.append(key)
            units.append(indices)
            statuses.append("single_row")
            group_sizes.append(1)
            mapping[indices[0]] = len(units) - 1
            continue
        for source_row in indices:
            output_keys.append(key)
            units.append([source_row])
            statuses.append("ambiguous")
            group_sizes.append(len(indices))
            mapping[source_row] = len(units) - 1
    return output_keys, units, statuses, group_sizes, mapping


def _constant_columns(
    obs_columns: list[str], values: dict[str, np.ndarray], units: list[list[int]]
) -> tuple[list[str], list[str]]:
    constant, varying = [], []
    for name in obs_columns:
        if name in EXCLUDED_CONSTANT_COLUMNS or name in PROVENANCE_COLUMNS:
            continue
        is_constant = all(len(_distinct(values[name], indices)) == 1 for indices in units)
        (constant if is_constant else varying).append(name)
    return constant, varying


def _audit_rows(
    lineage: str,
    keys: list[tuple[str, str, str]],
    groups: list[list[int]],
    values: dict[str, np.ndarray],
    source_indices: np.ndarray,
) -> list[dict[str, Any]]:
    rows = []
    for key, indices in zip(keys, groups):
        n_rows = len(indices)
        methods = _distinct(values["method"], indices)
        status = "single_row" if n_rows == 1 else "ambiguous"
        if n_rows == 1:
            note = "No aggregation is needed; the source row already has the requested key."
        else:
            note = (
                "Not collapsed. Rows share donor, released brain region, and released supertype, "
                "but have distinct sample_name and library_prep identifiers. The release has no "
                "specimen/tissue-block field and does not define sample_name semantics, so technical "
                "versus biological sampling cannot be resolved from the source metadata."
            )
            if len(methods) > 1:
                note += " Multiple released RNA assay methods are also present."
        rows.append(
            {
                "lineage": lineage,
                "donor_id": key[0],
                "brain_region": key[1],
                "supertype": key[2],
                "n_source_rows": n_rows,
                "n_library_preps": len(_distinct(values["library_prep"], indices)),
                "n_distinct_specimens": "",
                "n_distinct_samples": len(_distinct(values["sample_name"], indices)),
                "n_distinct_blocks_if_available": "",
                "aggregation_status": status,
                "aggregation_note": note,
                "source_obs_indices": _joined(source_indices, indices),
                "source_library_preps": _joined(values["library_prep"], indices),
                "source_sample_names": _joined(values["sample_name"], indices),
                "source_ar_ids": _joined(values["ar_id"], indices),
                "source_load_names": _joined(values["load_name"], indices),
                "source_exp_component_vendor_names": _joined(
                    values["exp_component_vendor_name"], indices
                ),
                "source_rna_amplifications": _joined(
                    values["rna_amplification"], indices
                ),
                "source_batch_vendor_names": _joined(
                    values["batch_vendor_name"], indices
                ),
                "source_facs_population_plans": _joined(
                    values["facs_population_plan"], indices
                ),
                "source_methods": "|".join(methods),
                "source_alignments": _joined(values["alignment"], indices),
            }
        )
    return rows


def _prepared_obs(
    item: SelectionItem,
    keys: list[tuple[str, str, str]],
    units: list[list[int]],
    statuses: list[str],
    source_key_group_sizes: list[int],
    values: dict[str, np.ndarray],
    source_indices: np.ndarray,
    constant_columns: list[str],
    nuclei: np.ndarray,
) -> tuple[dict[str, list[Any]], dict[str, str]]:
    obs: dict[str, list[Any]] = {
        "_index": [],
        "donor_id": [],
        "brain_region": [],
        "source_subclass_or_lineage": [],
        "released_class": [],
        "released_supertype": [],
        "supertype": [],
        "released_family": [],
        "taxonomy_resolution_status": [],
        "Number of nuclei": [],
        "n_source_rows": [],
        "n_library_preps": [],
        "n_rows_in_source_key_group": [],
        "source_release": [],
        "aggregation_status": [],
        "source_obs_indices": [],
        "source_library_preps": [],
        "source_sample_names": [],
        "source_ar_ids": [],
        "source_load_names": [],
        "source_exp_component_vendor_names": [],
        "source_rna_amplifications": [],
        "source_batch_vendor_names": [],
        "source_facs_population_plans": [],
        "source_methods": [],
        "source_alignments": [],
    }
    metadata_name_map: dict[str, str] = {}
    occupied = set(obs)
    for source_name in constant_columns:
        prepared_name = "released_meta_" + _slug(source_name)
        counter = 2
        base = prepared_name
        while prepared_name in occupied:
            prepared_name = f"{base}_{counter}"
            counter += 1
        occupied.add(prepared_name)
        metadata_name_map[source_name] = prepared_name
        obs[prepared_name] = []

    for key, indices, status, source_group_size in zip(
        keys, units, statuses, source_key_group_sizes
    ):
        donor, region, supertype = key
        family, taxonomy_status = _taxonomy_fields(item.lineage, supertype)
        library = _canonical(values["library_prep"][indices[0]])
        base_index = f"{item.lineage}__{donor}__{region}__{supertype}"
        obs["_index"].append(
            base_index if status == "single_row" else f"{base_index}__{library}"
        )
        obs["donor_id"].append(donor)
        obs["brain_region"].append(region)
        obs["source_subclass_or_lineage"].append(_canonical(values["Subclass"][indices[0]]))
        obs["released_class"].append(_canonical(values["Class"][indices[0]]))
        obs["released_supertype"].append(supertype)
        obs["supertype"].append(supertype)
        obs["released_family"].append(family)
        obs["taxonomy_resolution_status"].append(taxonomy_status)
        obs["Number of nuclei"].append(int(nuclei[indices].sum()))
        obs["n_source_rows"].append(len(indices))
        obs["n_library_preps"].append(len(_distinct(values["library_prep"], indices)))
        obs["n_rows_in_source_key_group"].append(source_group_size)
        obs["source_release"].append(item.release)
        obs["aggregation_status"].append(status)
        obs["source_obs_indices"].append(_joined(source_indices, indices))
        obs["source_library_preps"].append(_joined(values["library_prep"], indices))
        obs["source_sample_names"].append(_joined(values["sample_name"], indices))
        obs["source_ar_ids"].append(_joined(values["ar_id"], indices))
        obs["source_load_names"].append(_joined(values["load_name"], indices))
        obs["source_exp_component_vendor_names"].append(
            _joined(values["exp_component_vendor_name"], indices)
        )
        obs["source_rna_amplifications"].append(
            _joined(values["rna_amplification"], indices)
        )
        obs["source_batch_vendor_names"].append(
            _joined(values["batch_vendor_name"], indices)
        )
        obs["source_facs_population_plans"].append(
            _joined(values["facs_population_plan"], indices)
        )
        obs["source_methods"].append(_joined(values["method"], indices))
        obs["source_alignments"].append(_joined(values["alignment"], indices))
        for source_name, prepared_name in metadata_name_map.items():
            distinct = _distinct(values[source_name], indices)
            if len(distinct) != 1:
                raise PreparationError(
                    f"Metadata field {source_name} unexpectedly varies in output unit {key}"
                )
            obs[prepared_name].append(values[source_name][indices[0]])
    if len(obs["_index"]) != len(set(obs["_index"])):
        raise PreparationError(f"Prepared obs indices are not unique for {item.lineage}")
    return obs, metadata_name_map

def _prepare_one(
    item: SelectionItem,
    project_root: Path,
    output_dir: Path,
    *,
    force: bool,
    gene_block_size: int = 1024,
) -> tuple[PreparedLineage, list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    source_path = project_root / item.local_path
    if not source_path.is_file() or source_path.stat().st_size != item.size_bytes:
        raise PreparationError(f"Missing or size-invalid source: {source_path}")
    before = fingerprint(source_path)
    prepared_path = output_dir / f"{item.lineage}_donor_region_supertype_counts.h5ad"
    if prepared_path.exists() and not force:
        raise PreparationError(f"Refusing to overwrite prepared file without --force: {prepared_path}")
    temporary = prepared_path.with_suffix(".h5ad.tmp")
    temporary.unlink(missing_ok=True)

    with h5py.File(source_path, "r") as source:
        obs_group = source["obs"]
        obs_columns = dataframe_columns(obs_group)
        required = set(KEY_COLUMNS) | {
            "Subclass",
            "Class",
            "Number of nuclei",
            *PROVENANCE_COLUMNS,
        }
        missing = required - set(obs_columns)
        if missing:
            raise PreparationError(f"{source_path} lacks required obs columns: {sorted(missing)}")
        values = {name: read_dataframe_column(obs_group, name) for name in obs_columns}
        source_indices = read_dataframe_column(obs_group, str(obs_group.attrs["_index"]))
        source_keys, source_groups = _source_groups(values)
        keys, units, statuses, source_group_sizes, mapping = _output_units(
            source_keys, source_groups, len(source_indices)
        )
        constant_columns, varying_columns = _constant_columns(obs_columns, values, units)
        workflow_aliases = (
            "sample_name",
            "ar_id",
            "load_name",
            "exp_component_vendor_name",
            "rna_amplification",
        )
        one_to_one = {
            name: _is_one_to_one(values[name], values["library_prep"])
            for name in workflow_aliases
        }
        if not all(one_to_one.values()):
            raise PreparationError(
                f"A source identifier is not one-to-one with library_prep in {source_path}: "
                f"{one_to_one}"
            )
        nuclei = np.asarray(values["Number of nuclei"], dtype=np.int64)
        prepared_obs, metadata_name_map = _prepared_obs(
            item,
            keys,
            units,
            statuses,
            source_group_sizes,
            values,
            source_indices,
            constant_columns,
            nuclei,
        )
        audit = _audit_rows(
            item.lineage, source_keys, source_groups, values, source_indices
        )
        source_x = source["X"]
        if not isinstance(source_x, h5py.Dataset):
            raise PreparationError(f"Only dense source X is supported: {source_path}")
        n_source, n_genes = source_x.shape
        if n_source != len(mapping):
            raise PreparationError("Source X and obs row counts differ")
        gene_symbols = read_dataframe_column(source["var"], str(source["var"].attrs["_index"]))
        gene_ids = read_dataframe_column(source["var"], "gene_ids")
        if len(gene_symbols) != n_genes or len(gene_ids) != n_genes:
            raise PreparationError("Source X and var dimensions differ")
        if len(set(map(str, gene_ids))) != n_genes:
            raise PreparationError("Source gene_ids are duplicated; exact identity is not safe")

        prepared_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(temporary, "w") as target:
            target.attrs["encoding-type"] = "anndata"
            target.attrs["encoding-version"] = "0.1.0"
            dataset = target.create_dataset(
                "X",
                shape=(len(keys), n_genes),
                dtype=np.int64,
                chunks=(min(32, len(keys)), min(gene_block_size, n_genes)),
                compression="gzip",
                compression_opts=4,
                shuffle=True,
            )
            dataset.attrs["encoding-type"] = "array"
            dataset.attrs["encoding-version"] = "0.2.0"
            _write_dataframe(target, "obs", prepared_obs)
            source.copy("var", target)
            for group_name in ("layers", "obsm", "varm", "obsp", "varp", "uns"):
                _empty_anndata_group(target, group_name)

            prepared_total_umi = 0
            source_total_umi = 0
            row_totals = np.zeros(len(keys), dtype=np.int64)
            gene_totals = np.zeros(n_genes, dtype=np.int64)
            gene_nonzero = np.zeros(n_genes, dtype=np.int64)
            for start in range(0, n_genes, gene_block_size):
                stop = min(start + gene_block_size, n_genes)
                source_block_float = np.asarray(source_x[:, start:stop])
                if not np.isfinite(source_block_float).all() or np.any(source_block_float < 0):
                    raise PreparationError(f"Invalid count values in {source_path}")
                if not np.all(source_block_float == np.floor(source_block_float)):
                    raise PreparationError(f"Non-integer count values in {source_path}")
                source_block = source_block_float.astype(np.int64)
                aggregated = np.zeros((len(keys), stop - start), dtype=np.int64)
                np.add.at(aggregated, mapping, source_block)
                dataset[:, start:stop] = aggregated
                source_sum = int(source_block.sum(dtype=np.int64))
                prepared_sum = int(aggregated.sum(dtype=np.int64))
                if source_sum != prepared_sum:
                    raise PreparationError(f"Block count conservation failed in {source_path}")
                source_total_umi += source_sum
                prepared_total_umi += prepared_sum
                row_totals += aggregated.sum(axis=1, dtype=np.int64)
                gene_totals[start:stop] = aggregated.sum(axis=0, dtype=np.int64)
                gene_nonzero[start:stop] = np.count_nonzero(aggregated, axis=0)

            total_umi_dataset = target["obs"].create_dataset("total_umi", data=row_totals)
            total_umi_dataset.attrs["encoding-type"] = "array"
            total_umi_dataset.attrs["encoding-version"] = "0.2.0"
            column_order = list(target["obs"].attrs["column-order"])
            del target["obs"].attrs["column-order"]
            target["obs"].attrs.create(
                "column-order",
                np.asarray(column_order + ["total_umi"], dtype=h5py.string_dtype("utf-8")),
            )

        source_total_nuclei = int(nuclei.sum())
        prepared_nuclei = np.asarray(prepared_obs["Number of nuclei"], dtype=np.int64)
        prepared_total_nuclei = int(prepared_nuclei.sum())
        if source_total_umi != prepared_total_umi:
            raise PreparationError(f"Total UMI conservation failed in {source_path}")
        if source_total_nuclei != prepared_total_nuclei:
            raise PreparationError(f"Nucleus conservation failed in {source_path}")
        prepared_key_statuses: dict[tuple[str, str, str], list[str]] = defaultdict(list)
        for key, status in zip(keys, statuses):
            prepared_key_statuses[key].append(status)
        invalid_duplicates = {
            key: key_statuses
            for key, key_statuses in prepared_key_statuses.items()
            if len(key_statuses) > 1 and set(key_statuses) != {"ambiguous"}
        }
        if invalid_duplicates:
            raise PreparationError(
                f"Prepared duplicate keys are not explicitly ambiguous: {invalid_duplicates}"
            )
        source_n_donors = len(set(values["Donor ID"].astype(str)))
        source_n_regions = len(set(values["Brain Region"].astype(str)))
        source_n_supertypes = len(set(values["Supertype"].astype(str)))
        if (
            source_n_donors != len({key[0] for key in keys})
            or source_n_regions != len({key[1] for key in keys})
            or source_n_supertypes != len({key[2] for key in keys})
        ):
            raise PreparationError(f"Donor, region, or supertype was filtered: {source_path}")

        index_rows = []
        for index, key in enumerate(keys):
            index_rows.append(
                {
                    "lineage": item.lineage,
                    "donor_id": key[0],
                    "brain_region": key[1],
                    "supertype": key[2],
                    "n_nuclei": int(prepared_nuclei[index]),
                    "n_source_rows": prepared_obs["n_source_rows"][index],
                    "n_library_preps": prepared_obs["n_library_preps"][index],
                    "n_rows_in_source_key_group": prepared_obs[
                        "n_rows_in_source_key_group"
                    ][index],
                    "total_umi": int(row_totals[index]),
                    "source_release": item.release,
                    "aggregation_status": prepared_obs["aggregation_status"][index],
                    "prepared_h5ad": str(prepared_path.relative_to(project_root)),
                    "prepared_obs_index": prepared_obs["_index"][index],
                }
            )
        gene_rows = [
            {
                "lineage": item.lineage,
                "gene_symbol_or_name": str(gene_symbols[index]),
                "gene_id": str(gene_ids[index]),
                "n_prepared_rows": len(keys),
                "n_nonzero_rows": int(gene_nonzero[index]),
                "fraction_nonzero": f"{gene_nonzero[index] / len(keys):.8f}",
                "total_counts": int(gene_totals[index]),
            }
            for index in range(n_genes)
        ]

    os.replace(temporary, prepared_path)
    after = fingerprint(source_path)
    if before != after:
        prepared_path.unlink(missing_ok=True)
        raise PreparationError(f"Source file changed during preparation: {source_path}")
    with h5py.File(prepared_path, "r") as prepared, h5py.File(source_path, "r") as source:
        if prepared["X"].shape != (len(keys), n_genes):
            raise PreparationError(f"Prepared shape verification failed: {prepared_path}")
        prepared_symbols = read_dataframe_column(
            prepared["var"], str(prepared["var"].attrs["_index"])
        )
        prepared_ids = read_dataframe_column(prepared["var"], "gene_ids")
        if not np.array_equal(gene_symbols, prepared_symbols) or not np.array_equal(
            gene_ids, prepared_ids
        ):
            raise PreparationError(f"Prepared gene identity/order changed: {prepared_path}")
        if not np.array_equal(prepared["obs"]["total_umi"][:], row_totals):
            raise PreparationError(f"Prepared row totals failed verification: {prepared_path}")
        disk_total_umi = 0
        disk_row_totals = np.zeros(len(keys), dtype=np.int64)
        for start in range(0, n_genes, gene_block_size):
            stop = min(start + gene_block_size, n_genes)
            block = np.asarray(prepared["X"][:, start:stop], dtype=np.int64)
            disk_total_umi += int(block.sum(dtype=np.int64))
            disk_row_totals += block.sum(axis=1, dtype=np.int64)
        if disk_total_umi != source_total_umi or not np.array_equal(
            disk_row_totals, row_totals
        ):
            raise PreparationError(f"On-disk prepared count conservation failed: {prepared_path}")
        if int(prepared["obs"]["Number of nuclei"][:].sum()) != source_total_nuclei:
            raise PreparationError(f"On-disk nucleus conservation failed: {prepared_path}")

    summary = PreparedLineage(
        lineage=item.lineage,
        source_path=str(source_path.relative_to(project_root)),
        prepared_path=str(prepared_path.relative_to(project_root)),
        source_rows=n_source,
        prepared_rows=len(keys),
        unresolved_rows=sum(status == "ambiguous" for status in statuses),
        unresolved_groups=sum(len(indices) > 1 for indices in source_groups),
        safely_aggregated_groups=0,
        n_donors=len({key[0] for key in keys}),
        n_regions=len({key[1] for key in keys}),
        n_supertypes=len({key[2] for key in keys}),
        n_genes=n_genes,
        source_n_donors=source_n_donors,
        source_n_regions=source_n_regions,
        source_n_supertypes=source_n_supertypes,
        source_total_umi=source_total_umi,
        prepared_total_umi=prepared_total_umi,
        source_total_nuclei=source_total_nuclei,
        prepared_total_nuclei=prepared_total_nuclei,
        source_fingerprint_before=asdict(before),
        source_fingerprint_after=asdict(after),
        preserved_constant_metadata=constant_columns,
        omitted_nonconstant_metadata=varying_columns,
        gene_index_field="var/index (AnnData var index; gene symbols/names)",
        gene_id_field="var/gene_ids (Ensembl gene IDs)",
        gene_symbols_duplicated=n_genes - len(set(map(str, gene_symbols))),
        gene_ids_duplicated=n_genes - len(set(map(str, gene_ids))),
        workflow_identifiers_one_to_one_with_library=one_to_one,
    )
    return summary, audit, index_rows, gene_rows


def _coverage(index_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in index_rows:
        groups[(row["lineage"], row["brain_region"])].append(row)
    output = []
    for (lineage, region), rows in sorted(groups.items()):
        nuclei = np.asarray([int(row["n_nuclei"]) for row in rows], dtype=np.int64)
        output.append(
            {
                "lineage": lineage,
                "brain_region": region,
                "n_donors": len({row["donor_id"] for row in rows}),
                "n_supertypes": len({row["supertype"] for row in rows}),
                "n_prepared_rows": len(rows),
                "total_nuclei": int(nuclei.sum()),
                "median_nuclei": float(np.median(nuclei)),
                "min_nuclei": int(nuclei.min()),
                "max_nuclei": int(nuclei.max()),
            }
        )
    return output


def _taxonomy(index_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in index_rows:
        groups[(row["lineage"], row["supertype"])].append(row)
    output = []
    for (lineage, supertype), rows in sorted(groups.items()):
        if lineage == "Immune" and supertype.startswith("Micro-PVM_"):
            interpretation = "Released Micro-PVM family; microglia versus PVM is unresolved."
            status = "unresolved_microglia_vs_PVM"
        elif lineage == "Immune":
            interpretation = "Released immune supertype label preserved exactly."
            status = "released_label_explicit"
        else:
            interpretation = "Released supertype label preserved exactly; no regrouping applied."
            status = "released_taxonomy_preserved"
        output.append(
            {
                "lineage": lineage,
                "supertype": supertype,
                "n_donors": len({row["donor_id"] for row in rows}),
                "n_regions": len({row["brain_region"] for row in rows}),
                "total_nuclei": sum(int(row["n_nuclei"]) for row in rows),
                "taxonomy_interpretation": interpretation,
                "semantic_status": status,
            }
        )
    return output


def _support(index_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in index_rows:
        groups[(row["lineage"], row["brain_region"], row["supertype"])].append(row)
    output = []
    for (lineage, region, supertype), rows in sorted(groups.items()):
        nuclei = np.asarray([int(row["n_nuclei"]) for row in rows], dtype=np.int64)
        total_umi = np.asarray([int(row["total_umi"]) for row in rows], dtype=np.int64)
        output.append(
            {
                "lineage": lineage,
                "brain_region": region,
                "supertype": supertype,
                "n_donors": len({row["donor_id"] for row in rows}),
                "n_prepared_rows": len(rows),
                "total_nuclei": int(nuclei.sum()),
                "min_nuclei": int(nuclei.min()),
                "median_nuclei": float(np.median(nuclei)),
                "max_nuclei": int(nuclei.max()),
                "min_total_umi": int(total_umi.min()),
                "median_total_umi": float(np.median(total_umi)),
                "max_total_umi": int(total_umi.max()),
            }
        )
    return output


def prepare_selection(
    selection_path: Path,
    project_root: Path,
    *,
    output_root: Path | None = None,
    force: bool = False,
) -> list[PreparedLineage]:
    selection = load_selection(selection_path)
    if {item.lineage for item in selection.items} != {
        "Immune",
        "Astrocyte",
        "Oligodendrocyte",
        "OPC",
    }:
        raise PreparationError("Preparation requires the reviewed four-lineage selection")
    output_root = output_root or project_root / OUTPUT_RELATIVE
    h5ad_dir = output_root / "multiregion_2026"
    if not force:
        existing = [
            h5ad_dir / f"{item.lineage}_donor_region_supertype_counts.h5ad"
            for item in selection.items
            if (h5ad_dir / f"{item.lineage}_donor_region_supertype_counts.h5ad").exists()
        ]
        if existing:
            raise PreparationError(
                "Refusing to overwrite existing prepared files without --force: "
                + ", ".join(map(str, existing))
            )
    summaries: list[PreparedLineage] = []
    audit_rows: list[dict[str, Any]] = []
    index_rows: list[dict[str, Any]] = []
    gene_rows: list[dict[str, Any]] = []
    for item in selection.items:
        print(f"Preparing {item.lineage} from {item.local_path}", flush=True)
        summary, audit, index_part, gene_part = _prepare_one(
            item, project_root, h5ad_dir, force=force
        )
        summaries.append(summary)
        audit_rows.extend(audit)
        index_rows.extend(index_part)
        gene_rows.extend(gene_part)
        print(
            f"  {summary.source_rows} source rows -> {summary.prepared_rows} prepared rows; "
            f"ambiguous retained rows={summary.unresolved_rows}; "
            f"UMI={summary.prepared_total_umi}; nuclei={summary.prepared_total_nuclei}",
            flush=True,
        )

    _write_csv(output_root / "pseudobulk_prepared_index.csv", index_rows)
    _write_csv(output_root / "pseudobulk_coverage.csv", _coverage(index_rows))
    _write_csv(output_root / "pseudobulk_taxonomy.csv", _taxonomy(index_rows))
    _write_csv(output_root / "pseudobulk_aggregation_audit.csv", audit_rows)
    _write_csv(output_root / "pseudobulk_gene_catalog.csv", gene_rows)
    _write_csv(output_root / "pseudobulk_support_distribution.csv", _support(index_rows))
    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_release": RELEASE,
        "selection_manifest": str(selection_path),
        "aggregation_key": ["donor_id", "brain_region", "released_supertype"],
        "count_operation": "exact integer sum of source X before normalization",
        "nucleus_operation": "sum of source Number of nuclei",
        "normalization_applied": False,
        "filtering_applied": False,
        "minimum_nuclei_threshold": None,
        "regions_restricted": False,
        "specimen_field_available": False,
        "tissue_block_field_available": False,
        "sample_name_semantics": (
            "Not defined in the release README; observed one-to-one with library_prep and retained "
            "for provenance rather than treated as an independent biological grouping."
        ),
        "aggregation_interpretation": (
            "Repeated donor-region-supertype keys were not collapsed. They have distinct "
            "sample_name/library_prep records, while the release supplies no specimen/tissue-block "
            "field and does not define sample_name semantics. Single-row keys require no aggregation."
        ),
        "lineages": [asdict(summary) for summary in summaries],
        "totals": {
            "source_rows": sum(summary.source_rows for summary in summaries),
            "prepared_rows": sum(summary.prepared_rows for summary in summaries),
            "unresolved_rows": sum(summary.unresolved_rows for summary in summaries),
            "unresolved_groups": sum(summary.unresolved_groups for summary in summaries),
            "safely_aggregated_groups": sum(
                summary.safely_aggregated_groups for summary in summaries
            ),
            "source_total_umi": sum(summary.source_total_umi for summary in summaries),
            "prepared_total_umi": sum(summary.prepared_total_umi for summary in summaries),
            "source_total_nuclei": sum(summary.source_total_nuclei for summary in summaries),
            "prepared_total_nuclei": sum(summary.prepared_total_nuclei for summary in summaries),
        },
    }
    provenance_path = output_root / "preparation_provenance.json"
    temporary = provenance_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, provenance_path)
    return summaries
