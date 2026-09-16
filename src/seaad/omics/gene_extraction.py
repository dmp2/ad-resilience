"""Exact gene extraction from prepared SEA-AD pseudobulk H5AD files."""

from __future__ import annotations

import csv
import gzip
from pathlib import Path
from typing import TextIO

import h5py
import numpy as np

from omics.pseudobulk_inspection import read_dataframe_column
from omics.pseudobulk_preparation import H5AD_RELATIVE, PreparationError


LINEAGES = ("Immune", "Astrocyte", "Oligodendrocyte", "OPC")


def read_genes(path: Path) -> list[str]:
    genes: list[str] = []
    seen: set[str] = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        gene = raw.strip()
        if not gene or gene.startswith("#"):
            continue
        if gene not in seen:
            genes.append(gene)
            seen.add(gene)
    if not genes:
        raise PreparationError(f"Gene list is empty: {path}")
    return genes


def _open_output(path: Path, *, compressed: bool) -> TextIO:
    if compressed:
        return gzip.open(path, "wt", newline="", encoding="utf-8")
    return path.open("w", newline="", encoding="utf-8")


def extract_exact_genes(
    prepared_path: Path,
    genes: list[str],
    output_path: Path,
    *,
    include_row_total_umi: bool = True,
) -> dict[str, object]:
    if output_path.suffix not in {".csv", ".gz"}:
        raise PreparationError("Output must end in .csv or .csv.gz")
    with h5py.File(prepared_path, "r") as handle:
        symbols = read_dataframe_column(handle["var"], str(handle["var"].attrs["_index"])).astype(str)
        gene_ids = read_dataframe_column(handle["var"], "gene_ids").astype(str)
        symbol_map: dict[str, list[int]] = {}
        id_map: dict[str, list[int]] = {}
        for index, value in enumerate(symbols):
            symbol_map.setdefault(value, []).append(index)
        for index, value in enumerate(gene_ids):
            id_map.setdefault(value, []).append(index)

        requests: list[tuple[str, int, str]] = []
        missing = []
        for gene in genes:
            indices = sorted(set(symbol_map.get(gene, [])) | set(id_map.get(gene, [])))
            if not indices:
                missing.append(gene)
                continue
            for index in indices:
                by_symbol = gene == symbols[index]
                by_id = gene == gene_ids[index]
                match_type = "symbol_and_id" if by_symbol and by_id else "symbol" if by_symbol else "gene_id"
                requests.append((gene, index, match_type))

        selected_indices = sorted({index for _, index, _ in requests})
        position = {gene_index: column for column, gene_index in enumerate(selected_indices)}
        counts = np.asarray(handle["X"][:, selected_indices], dtype=np.int64) if selected_indices else np.empty((handle["X"].shape[0], 0), dtype=np.int64)
        obs = handle["obs"]
        metadata_names = [
            "_index",
            "donor_id",
            "brain_region",
            "source_subclass_or_lineage",
            "released_supertype",
            "Number of nuclei",
            "n_source_rows",
            "n_library_preps",
            "source_release",
            "aggregation_status",
            "total_umi",
        ]
        metadata = {name: read_dataframe_column(obs, name) for name in metadata_names}
        if np.any(counts < 0):
            raise PreparationError(f"Prepared object contains negative counts: {prepared_path}")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_name(output_path.name + ".tmp")
        fields = [
            "prepared_obs_index",
            "donor_id",
            "brain_region",
            "lineage",
            "released_supertype",
            "n_nuclei",
            "n_source_rows",
            "n_library_preps",
            "source_release",
            "aggregation_status",
        ]
        if include_row_total_umi:
            fields.append("row_total_umi")
        fields += ["requested_gene", "match_type", "gene_symbol_or_name", "gene_id", "raw_count"]
        with _open_output(temporary, compressed=output_path.suffix == ".gz") as output:
            writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            for row in range(counts.shape[0]):
                base = {
                    "prepared_obs_index": str(metadata["_index"][row]),
                    "donor_id": str(metadata["donor_id"][row]),
                    "brain_region": str(metadata["brain_region"][row]),
                    "lineage": str(metadata["source_subclass_or_lineage"][row]),
                    "released_supertype": str(metadata["released_supertype"][row]),
                    "n_nuclei": int(metadata["Number of nuclei"][row]),
                    "n_source_rows": int(metadata["n_source_rows"][row]),
                    "n_library_preps": int(metadata["n_library_preps"][row]),
                    "source_release": str(metadata["source_release"][row]),
                    "aggregation_status": str(metadata["aggregation_status"][row]),
                }
                if include_row_total_umi:
                    base["row_total_umi"] = int(metadata["total_umi"][row])
                for requested, gene_index, match_type in requests:
                    writer.writerow(
                        base
                        | {
                            "requested_gene": requested,
                            "match_type": match_type,
                            "gene_symbol_or_name": symbols[gene_index],
                            "gene_id": gene_ids[gene_index],
                            "raw_count": int(counts[row, position[gene_index]]),
                        }
                    )
        temporary.replace(output_path)

    missing_path = output_path.with_name(output_path.name + ".missing_genes.txt")
    missing_path.write_text("".join(f"{gene}\n" for gene in missing), encoding="utf-8")
    return {
        "prepared_h5ad": str(prepared_path),
        "output": str(output_path),
        "n_requested": len(genes),
        "n_matched_requests": len(genes) - len(missing),
        "n_matched_features": len(requests),
        "n_rows_written": len(requests) * counts.shape[0],
        "missing_genes": missing,
        "missing_genes_file": str(missing_path),
        "raw_counts": True,
        "normalization_applied": False,
    }


def prepared_path_for_lineage(project_root: Path, lineage: str, prepared_root: Path | None = None) -> Path:
    if lineage not in LINEAGES:
        raise PreparationError(f"Unknown lineage {lineage}; choose from {', '.join(LINEAGES)}")
    root = prepared_root or project_root / H5AD_RELATIVE
    return root / f"{lineage}_donor_region_supertype_counts.h5ad"
