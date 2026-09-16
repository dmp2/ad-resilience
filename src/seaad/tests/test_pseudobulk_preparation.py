from __future__ import annotations

import csv
import gzip
import hashlib
from pathlib import Path

import h5py
import numpy as np

from acquisition.s3_download import SelectionItem
from omics.gene_extraction import extract_exact_genes
from omics.pseudobulk_inspection import read_dataframe_column
from omics.pseudobulk_preparation import _empty_anndata_group, _prepare_one, _write_dataframe


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_source(path: Path) -> None:
    path.parent.mkdir(parents=True)
    with h5py.File(path, "w") as handle:
        handle.attrs["encoding-type"] = "anndata"
        handle.attrs["encoding-version"] = "0.1.0"
        x = handle.create_dataset(
            "X", data=np.asarray([[1, 2, 0], [3, 0, 4], [0, 5, 1]], dtype=float)
        )
        x.attrs["encoding-type"] = "array"
        x.attrs["encoding-version"] = "0.2.0"
        obs = {
            "_index": ["row0", "row1", "row2"],
            "Donor ID": ["H01.01.001", "H01.01.001", "H02.02.002"],
            "Brain Region": ["DFC", "DFC", "MEC"],
            "Supertype": ["Astro_1", "Astro_1", "Astro_2"],
            "Subclass": ["Astrocyte"] * 3,
            "Class": ["Non-neuronal and Non-neural"] * 3,
            "Number of nuclei": [2, 3, 1],
            "library_prep": ["lib1", "lib2", "lib3"],
            "sample_name": ["sample1", "sample2", "sample3"],
            "ar_id": ["ar1", "ar2", "ar3"],
            "load_name": ["load1", "load2", "load3"],
            "exp_component_vendor_name": ["component1", "component2", "component3"],
            "rna_amplification": ["amp1", "amp2", "amp3"],
            "method": ["10x", "10x", "10x"],
            "alignment": ["align", "align", "align"],
            "batch_vendor_name": ["batch1", "batch2", "batch3"],
            "facs_population_plan": ["plan", "plan", "plan"],
            "Sex": ["Female", "Female", "Male"],
            "GEX_Mean_raw_reads_per_cell": [10.0, 11.0, 12.0],
        }
        _write_dataframe(handle, "obs", obs)
        var = {
            "index": ["GENEA", "GENEB", "GENEC"],
            "gene_ids": ["ENSG1", "ENSG2", "ENSG3"],
            "feature_types": ["Gene Expression"] * 3,
            "genome": ["GRCh38"] * 3,
        }
        _write_dataframe(handle, "var", var, index_name="index")
        for group in ("layers", "obsm", "varm", "obsp", "varp", "uns"):
            _empty_anndata_group(handle, group)


def test_preparation_conserves_source_counts_nuclei_genes_and_keys(tmp_path):
    source = tmp_path / "data/raw/sea-ad/multiregion_2026/pseudobulk_objects/source.h5ad"
    make_source(source)
    before = digest(source)
    item = SelectionItem(
        release="Multiregion 2026 (2026-06-22)",
        object_family="subclass pseudobulk",
        lineage="Astrocyte",
        s3_key="unused",
        size_bytes=source.stat().st_size,
        local_path=str(source.relative_to(tmp_path)),
    )
    summary, audit, index_rows, genes = _prepare_one(
        item, tmp_path, tmp_path / "prepared", force=False, gene_block_size=2
    )
    prepared = tmp_path / summary.prepared_path
    assert digest(source) == before
    assert summary.source_total_umi == summary.prepared_total_umi == 16
    assert summary.source_total_nuclei == summary.prepared_total_nuclei == 6
    assert summary.source_rows == 3
    assert summary.prepared_rows == 3
    assert summary.unresolved_rows == 2
    assert summary.unresolved_groups == 1
    assert summary.safely_aggregated_groups == 0
    assert [row["aggregation_status"] for row in audit] == ["ambiguous", "single_row"]
    assert [row["aggregation_status"] for row in index_rows] == [
        "ambiguous",
        "ambiguous",
        "single_row",
    ]
    assert [row["n_source_rows"] for row in index_rows] == [1, 1, 1]
    assert [row["n_rows_in_source_key_group"] for row in index_rows] == [2, 2, 1]
    assert [row["gene_symbol_or_name"] for row in genes] == ["GENEA", "GENEB", "GENEC"]
    with h5py.File(prepared, "r") as handle:
        np.testing.assert_array_equal(
            handle["X"][:], [[1, 2, 0], [3, 0, 4], [0, 5, 1]]
        )
        np.testing.assert_array_equal(handle["obs"]["Number of nuclei"][:], [2, 3, 1])
        keys = list(
            zip(
                read_dataframe_column(handle["obs"], "donor_id"),
                read_dataframe_column(handle["obs"], "brain_region"),
                read_dataframe_column(handle["obs"], "supertype"),
            )
        )
        assert len(keys) == 3
        assert len(set(keys)) == 2
        statuses = list(read_dataframe_column(handle["obs"], "aggregation_status"))
        assert statuses == ["ambiguous", "ambiguous", "single_row"]
        assert list(read_dataframe_column(handle["var"], "index")) == ["GENEA", "GENEB", "GENEC"]
        assert list(read_dataframe_column(handle["var"], "gene_ids")) == ["ENSG1", "ENSG2", "ENSG3"]


def test_exact_gene_extraction_reports_absent_and_never_fuzzy_matches(tmp_path):
    source = tmp_path / "data/raw/sea-ad/multiregion_2026/pseudobulk_objects/source.h5ad"
    make_source(source)
    item = SelectionItem(
        release="Multiregion 2026 (2026-06-22)",
        object_family="subclass pseudobulk",
        lineage="Astrocyte",
        s3_key="unused",
        size_bytes=source.stat().st_size,
        local_path=str(source.relative_to(tmp_path)),
    )
    summary, _, _, _ = _prepare_one(item, tmp_path, tmp_path / "prepared", force=False)
    output = tmp_path / "genes.csv.gz"
    result = extract_exact_genes(
        tmp_path / summary.prepared_path,
        ["GENEA", "ENSG2", "genea"],
        output,
    )
    assert result["missing_genes"] == ["genea"]
    assert result["n_rows_written"] == 6
    with gzip.open(output, "rt", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["requested_gene"] for row in rows} == {"GENEA", "ENSG2"}
    assert {row["gene_symbol_or_name"] for row in rows} == {"GENEA", "GENEB"}
    assert all(row["raw_count"].isdigit() for row in rows)
