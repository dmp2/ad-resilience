from __future__ import annotations

import csv
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from omics.pseudobulk_preparation import PreparationError, _empty_anndata_group, _write_dataframe
from omics.signature_registry import Candidate, build_registry, read_gmt
from omics.state_scoring import (
    _average_rank,
    _upper_quartile_block,
    domain_of,
    scan_lineage,
    score_domain,
    score_selection,
    tmm_factor,
)


# --- TMM ------------------------------------------------------------------


def test_tmm_factor_is_one_for_proportional_libraries():
    rng = np.random.default_rng(0)
    reference = rng.integers(5, 500, size=400).astype(float)
    for scale in (0.5, 1.0, 3.0):
        counts = reference * scale
        factor = tmm_factor(counts, reference, float(counts.sum()), float(reference.sum()))
        assert factor == pytest.approx(1.0, abs=1e-6)


def test_tmm_factor_responds_to_composition_bias():
    rng = np.random.default_rng(1)
    reference = rng.integers(50, 500, size=600).astype(float)
    counts = reference.copy()
    # A minority of features soaks up most of the library: the remaining
    # features are under-sampled, so TMM must scale this library down.
    counts[:60] *= 40
    factor = tmm_factor(counts, reference, float(counts.sum()), float(reference.sum()))
    assert factor < 0.9


def test_tmm_factor_degenerate_inputs_are_neutral():
    zeros = np.zeros(10)
    ones = np.ones(10)
    assert tmm_factor(zeros, ones, 0.0, 10.0) == 1.0
    assert tmm_factor(ones, ones, 10.0, 10.0) == 1.0


def test_average_rank_matches_r_semantics():
    values = np.array([10.0, 20.0, 20.0, 5.0])
    assert _average_rank(values).tolist() == [2.0, 3.5, 3.5, 1.0]


def test_upper_quartile_block_handles_empty_libraries():
    block = np.array([[0.0, 0.0, 0.0], [1.0, 3.0, 8.0]])
    sizes = np.array([0.0, 12.0])
    quartiles = _upper_quartile_block(block, sizes)
    assert quartiles[0] == 0.0
    assert quartiles[1] > 0.0


# --- domains --------------------------------------------------------------


def test_immune_taxonomy_is_not_collapsed():
    assert domain_of("Immune", "Micro-PVM_2_3-SEAAD") == "Micro/PVM"
    assert domain_of("Immune", "Lymphocyte") == "Lymphocyte"
    assert domain_of("Immune", "Monocyte") == "Monocyte"
    assert domain_of("Astrocyte", "Astro_1") == "Astrocyte"
    with pytest.raises(PreparationError):
        domain_of("Immune", "Something_else")


# --- registry -------------------------------------------------------------


def test_read_gmt_preserves_order_and_drops_duplicates(tmp_path: Path):
    path = tmp_path / "x.gmt"
    path.write_text("SET_A\thttp://url\tB\tA\tB\nSET_B\thttp://url\tC\n", encoding="utf-8")
    sets = read_gmt(path)
    assert sets["SET_A"] == ["B", "A"]
    assert sets["SET_B"] == ["C"]


def test_registry_refuses_to_substitute_a_missing_set(tmp_path: Path):
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "h.all.v2025.1.Hs.symbols.gmt").write_text(
        "HALLMARK_REAL\thttp://url\tAAA\tBBB\n", encoding="utf-8"
    )
    candidate = Candidate(
        signature_name="absent",
        biological_axis="x",
        lineage="Astrocyte",
        domain="Astrocyte",
        source_publication="p",
        gene_definition_source="s",
        feature_origin="external_predefined",
        directionality="d",
        msigdb_set="HALLMARK_NOT_THERE",
        msigdb_collection="h.all",
    )
    with pytest.raises(Exception, match="refusing to substitute"):
        build_registry(tmp_path / "out", cache, [candidate])


def test_unresolved_candidate_carries_no_genes(tmp_path: Path):
    cache = tmp_path / "cache"
    cache.mkdir()
    candidate = Candidate(
        signature_name="unrecoverable",
        biological_axis="x",
        lineage="Astrocyte",
        domain="Astrocyte",
        source_publication="p",
        gene_definition_source="figure image",
        feature_origin="external_predefined",
        directionality="d",
        unresolved_reason="image only",
    )
    registry = build_registry(tmp_path / "out", cache, [candidate])
    entry = registry["signatures"][0]
    assert entry["resolution_status"] == "unresolved"
    assert entry["n_genes"] is None
    assert (tmp_path / "out" / "signature_genes.csv").read_text().strip() == "signature_name,gene_symbol"


# --- end-to-end on a synthetic prepared object ----------------------------


GENES = ["SIG1", "SIG2", "SIG3", "BG1", "BG2", "BG3", "BG4", "BG5"]


def write_prepared(path: Path, counts: np.ndarray, obs: dict[str, list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.attrs["encoding-type"] = "anndata"
        handle.attrs["encoding-version"] = "0.1.0"
        dataset = handle.create_dataset("X", data=counts.astype(np.int64))
        dataset.attrs["encoding-type"] = "array"
        dataset.attrs["encoding-version"] = "0.2.0"
        _write_dataframe(handle, "obs", obs, index_name="_index")
        _write_dataframe(
            handle,
            "var",
            {
                "index": GENES,
                "gene_ids": [f"ENSG{i:011d}" for i in range(len(GENES))],
                "feature_types": ["Gene Expression"] * len(GENES),
                "genome": ["GRCh38"] * len(GENES),
            },
            index_name="index",
        )
        for name in ("obsm", "varm", "obsp", "varp", "layers", "uns"):
            _empty_anndata_group(handle, name)


def synthetic(tmp_path: Path) -> Path:
    rng = np.random.default_rng(7)
    n = 12
    counts = rng.integers(20, 200, size=(n, len(GENES))).astype(np.int64)
    # Rows 0-5 carry a strong signature; rows 6-11 do not.
    counts[:6, :3] *= 8
    obs = {
        "_index": [f"row{i}" for i in range(n)],
        "donor_id": [f"H01.01.{i // 2:03d}" for i in range(n)],
        "brain_region": ["MTG"] * 6 + ["DFC"] * 6,
        "source_subclass_or_lineage": ["Astrocyte"] * n,
        "released_class": ["Non-neuronal"] * n,
        "released_supertype": ["Astro_1"] * n,
        "supertype": ["Astro_1"] * n,
        "released_family": ["Astrocyte"] * n,
        "taxonomy_resolution_status": ["released"] * n,
        "Number of nuclei": list(range(5, 5 + n)),
        "n_source_rows": [1] * n,
        "n_library_preps": [1] * n,
        "n_rows_in_source_key_group": [1] * n,
        "source_release": ["test"] * n,
        "aggregation_status": ["single_row"] * n,
        "source_obs_indices": [f"row{i}" for i in range(n)],
        "source_library_preps": [f"lib{i}" for i in range(n)],
        "source_sample_names": [f"sample{i}" for i in range(n)],
        "source_methods": ["10Xv3.1"] * 6 + ["10xMulti"] * 6,
        "source_alignments": ["cellranger"] * n,
        "source_batch_vendor_names": ["batch"] * n,
        "released_meta_severely_affected_donor": ["N"] * n,
        "released_meta_neurotypical_reference": ["False"] * n,
        "total_umi": counts.sum(axis=1).tolist(),
    }
    path = tmp_path / "prepared" / "Astrocyte_donor_region_supertype_counts.h5ad"
    write_prepared(path, counts, obs)
    return path


def test_scan_and_score_separate_the_planted_signature(tmp_path: Path):
    path = synthetic(tmp_path)
    scan = scan_lineage(path, {"planted": ["SIG1", "SIG2", "SIG3"]}, detection_fraction=0.1)
    assert scan["domain_names"] == ["Astrocyte"]
    assert scan["library_size"].shape == (12,)
    # Factors are rescaled to geometric mean one inside each domain.
    assert float(np.exp(np.mean(np.log(scan["tmm"])))) == pytest.approx(1.0, abs=1e-9)

    result = score_domain(scan, "Astrocyte", "planted", ["SIG1", "SIG2", "SIG3", "ABSENT"])
    assert result["n_defined"] == 4
    assert result["n_present"] == 3
    assert result["missing_genes"] == ["ABSENT"]
    assert result["score_z_mean"][:6].mean() > result["score_z_mean"][6:].mean()
    assert len(result["gene_rows"]) == 3


def test_score_selection_writes_row_level_outputs_without_collapsing(tmp_path: Path):
    path = synthetic(tmp_path)
    output = tmp_path / "out"
    output.mkdir()
    registry = {
        "provenance": {"created_utc": "2026-01-01T00:00:00+00:00", "msigdb_release": "test"},
        "signatures": [
            {
                "signature_name": "planted",
                "lineage": "Astrocyte",
                "domain": "Astrocyte",
                "feature_origin": "external_predefined",
                "resolution_status": "resolved",
                "role": "candidate_state",
            },
            {
                "signature_name": "null_control_size_matched_Astrocyte",
                "lineage": "Astrocyte",
                "domain": "Astrocyte",
                "feature_origin": "null_control_current_data",
                "resolution_status": "resolved_at_scoring_time",
                "role": "null_control",
            },
            {
                "signature_name": "unrecoverable",
                "lineage": "Astrocyte",
                "domain": "Astrocyte",
                "feature_origin": "external_predefined",
                "resolution_status": "unresolved",
                "role": "candidate_state",
            },
        ],
    }
    (output / "signature_registry.json").write_text(json.dumps(registry), encoding="utf-8")
    with (output / "signature_genes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["signature_name", "gene_symbol"])
        writer.writeheader()
        for gene in ("SIG1", "SIG2", "SIG3"):
            writer.writerow({"signature_name": "planted", "gene_symbol": gene})

    provenance = score_selection(
        tmp_path,
        output / "signature_registry.json",
        output / "signature_genes.csv",
        output,
        lineages=("Astrocyte",),
        prepared_dir=path.parent,
    )

    assert provenance["rows_collapsed_before_scoring"] is False
    assert provenance["normalization"]["single_cell_normalization_used"] is False
    assert provenance["counts"]["normalization_rows"] == 12

    with (output / "row_state_scores.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    names = {row["signature_name"] for row in rows}
    assert names == {"planted", "null_control_size_matched_Astrocyte"}
    assert len(rows) == 24
    # Every prepared row keeps its own identity and support columns.
    planted = [row for row in rows if row["signature_name"] == "planted"]
    assert len({row["prepared_obs_index"] for row in planted}) == 12
    assert all(row["sample_name"] and row["library_prep"] and row["assay_method"] for row in planted)
    assert all(row["n_nuclei"] and row["total_umi"] for row in planted)

    with (output / "signature_gene_coverage.csv").open(encoding="utf-8") as handle:
        coverage = {row["signature_name"]: row for row in csv.DictReader(handle)}
    assert coverage["planted"]["n_present"] == "3"
    assert int(coverage["null_control_size_matched_Astrocyte"]["n_used"]) == 3


def test_pooled_scan_sums_repeated_rows_without_touching_row_level(tmp_path: Path):
    from omics.state_scoring import pooled_scan_lineage

    path = synthetic(tmp_path)
    # synthetic() gives two rows per donor, both in the same region and supertype,
    # so pooling must halve the row count and sum the nuclei.
    pooled = pooled_scan_lineage(path, {"planted": ["SIG1", "SIG2", "SIG3"]})
    assert pooled["pooled"] is True
    assert pooled["n_rows"] == 6
    assert pooled["n_source_rows_pooled"] == 12
    assert pooled["obs"]["n_source_rows"].tolist() == ["2"] * 6

    row_level = scan_lineage(path, {"planted": ["SIG1", "SIG2", "SIG3"]})
    assert row_level["n_rows"] == 12
    assert float(pooled["library_size"].sum()) == pytest.approx(float(row_level["library_size"].sum()))
