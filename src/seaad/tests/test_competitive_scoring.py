"""Tests for the two things added on top of the frozen state-scoring pipeline.

1. the second signature-resolution route: a published supplementary workbook,
   read with the stdlib-only XLSX reader and pinned by SHA-256;
2. the matched-background (competitive) score.

Both are additive. A test at the bottom pins the property that matters most for
trusting the rest of notebook 02: with no background requested, the scan behaves
exactly as it did before.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import h5py
import numpy as np
import pytest

from omics.pseudobulk_preparation import _empty_anndata_group, _write_dataframe
from omics.signature_registry import (
    Candidate,
    RegistryError,
    SupplementSource,
    build_registry,
    read_supplement_groups,
)
from omics.state_scoring import (
    BackgroundSpec,
    draw_matched_background,
    scan_lineage,
    score_domain,
)
from omics.xlsx import XlsxError, read_records, read_sheet, sheet_names


# --- a minimal workbook, built by hand ------------------------------------

_WORKBOOK = (
    '<?xml version="1.0"?>'
    '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"'
    ' xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
    '<sheets><sheet name="{name}" sheetId="1" r:id="rId1"/></sheets></workbook>'
)
_RELS = (
    '<?xml version="1.0"?>'
    '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
    '<Relationship Id="rId1" Type="worksheet" Target="worksheets/sheet1.xml"/>'
    "</Relationships>"
)


def write_workbook(path: Path, sheet: str, rows: list[list[str]], *, shared: bool) -> None:
    """Write a tiny .xlsx, either with a shared-string table or with inline strings."""
    strings: list[str] = []
    body = []
    for row_number, row in enumerate(rows, start=1):
        cells = []
        for column, value in enumerate(row):
            reference = f"{chr(ord('A') + column)}{row_number}"
            if shared:
                if value not in strings:
                    strings.append(value)
                cells.append(f'<c r="{reference}" t="s"><v>{strings.index(value)}</v></c>')
            else:
                cells.append(f'<c r="{reference}" t="inlineStr"><is><t>{value}</t></is></c>')
        body.append(f'<row r="{row_number}">' + "".join(cells) + "</row>")

    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("xl/workbook.xml", _WORKBOOK.format(name=sheet))
        archive.writestr("xl/_rels/workbook.xml.rels", _RELS)
        archive.writestr(
            "xl/worksheets/sheet1.xml",
            '<?xml version="1.0"?><worksheet xmlns="http://schemas.openxmlformats.org/'
            'spreadsheetml/2006/main"><sheetData>' + "".join(body) + "</sheetData></worksheet>",
        )
        if shared:
            items = "".join(f"<si><t>{s}</t></si>" for s in strings)
            archive.writestr(
                "xl/sharedStrings.xml",
                '<?xml version="1.0"?><sst xmlns="http://schemas.openxmlformats.org/'
                f'spreadsheetml/2006/main">{items}</sst>',
            )


MARKERS = [
    ["microgliaState", "gene", "avg_log2FC"],
    ["MG4", "APOE", "1.2"],
    ["MG4", "TREM2", "0.9"],
    ["MG8", "LRRK2", "0.8"],
    ["MG4", "APOE", "1.2"],  # the sheet repeats a symbol
    ["MG4", "ABCA1", "0.4"],
]


@pytest.mark.parametrize("shared", [True, False])
def test_xlsx_reader_handles_both_string_encodings(tmp_path: Path, shared: bool):
    path = tmp_path / "book.xlsx"
    write_workbook(path, "Page 2.StateMarkers", MARKERS, shared=shared)
    assert sheet_names(path) == ["Page 2.StateMarkers"]
    table = read_sheet(path, "Page 2.StateMarkers")
    assert table[0] == ["microgliaState", "gene", "avg_log2FC"]
    assert len(table) == len(MARKERS)
    records = read_records(path, "Page 2.StateMarkers")
    assert records[0] == {"microgliaState": "MG4", "gene": "APOE", "avg_log2FC": "1.2"}


def test_xlsx_reader_refuses_an_absent_sheet(tmp_path: Path):
    path = tmp_path / "book.xlsx"
    write_workbook(path, "Sheet1", MARKERS, shared=True)
    with pytest.raises(XlsxError):
        read_sheet(path, "Not A Sheet")


def source_for(path: Path) -> SupplementSource:
    return SupplementSource(
        key="test",
        citation="test",
        url="https://example.invalid/book.xlsx",
        filename=path.name,
        sha256="unused in this test",
        sheet="Page 2.StateMarkers",
        group_column="microgliaState",
        gene_column="gene",
        released_filter="none",
    )


def test_supplement_groups_preserve_order_and_drop_repeats(tmp_path: Path):
    path = tmp_path / "book.xlsx"
    write_workbook(path, "Page 2.StateMarkers", MARKERS, shared=True)
    groups = read_supplement_groups(source_for(path), path)
    # Published order, not alphabetical, and the repeated APOE appears once.
    assert groups["MG4"] == ["APOE", "TREM2", "ABCA1"]
    assert groups["MG8"] == ["LRRK2"]


def test_supplement_groups_refuse_a_missing_column(tmp_path: Path):
    path = tmp_path / "book.xlsx"
    write_workbook(path, "Page 2.StateMarkers", [["state", "gene"], ["MG4", "APOE"]], shared=True)
    with pytest.raises(RegistryError):
        read_supplement_groups(source_for(path), path)


def test_a_candidate_may_not_name_two_resolution_routes(tmp_path: Path):
    candidate = Candidate(
        signature_name="two_routes",
        biological_axis="x",
        lineage="Astrocyte",
        domain="Astrocyte",
        source_publication="p",
        gene_definition_source="both",
        feature_origin="external_predefined",
        directionality="d",
        msigdb_set="GOBP_ASTROCYTE_ACTIVATION",
        msigdb_collection="c5.go.bp",
        supplement_key="sun_2023_table_s1_state_markers",
        supplement_group="MG4",
    )
    gmt = tmp_path / "cache" / "c5.go.bp.v2025.1.Hs.symbols.gmt"
    gmt.parent.mkdir(parents=True)
    gmt.write_text("GOBP_ASTROCYTE_ACTIVATION\thttp://x\tGFAP\tVIM\n", encoding="utf-8")
    with pytest.raises(RegistryError, match="two resolution routes"):
        build_registry(tmp_path / "out", gmt.parent, [candidate])


# --- matched-background draws ---------------------------------------------


def background_fixture(n_genes: int = 400, seed: int = 3):
    rng = np.random.default_rng(seed)
    mean_cpm = np.exp(rng.normal(3.0, 2.0, size=n_genes))
    detection = rng.uniform(0.1, 1.0, size=n_genes)
    eligible = np.arange(n_genes)
    # A signature drawn from the top of the abundance range, so a *uniform* draw
    # would be badly mismatched and a stratified one should not be.
    signature = np.argsort(mean_cpm)[-40:]
    return mean_cpm, detection, eligible, signature


def test_background_draws_are_size_matched_and_exclude_the_signature():
    mean_cpm, detection, eligible, signature = background_fixture()
    draws, summary = draw_matched_background(
        signature, eligible, mean_cpm, detection,
        n_draws=10, rng=np.random.default_rng(0),
        n_abundance_bins=10, n_detection_bins=5,
    )
    assert len(draws) == 10
    for draw in draws:
        assert len(draw) == len(signature)
        assert len(set(draw)) == len(draw)          # no gene twice inside a draw
        assert not set(draw) & set(signature.tolist())
    assert summary["n_signature_genes_matched"] == len(signature)


def test_background_draws_match_abundance_better_than_a_uniform_draw():
    mean_cpm, detection, eligible, signature = background_fixture()
    draws, summary = draw_matched_background(
        signature, eligible, mean_cpm, detection,
        n_draws=25, rng=np.random.default_rng(0),
        n_abundance_bins=10, n_detection_bins=5,
    )
    rng = np.random.default_rng(0)
    pool = np.setdiff1d(eligible, signature)
    uniform = np.mean([
        np.mean(np.log2(mean_cpm[rng.choice(pool, size=len(signature), replace=False)] + 1))
        for _ in range(25)
    ])
    target = summary["signature_mean_log2_cpm"]
    assert abs(summary["background_mean_log2_cpm"] - target) < abs(uniform - target)


def test_background_draws_are_reproducible_from_the_seed():
    mean_cpm, detection, eligible, signature = background_fixture()
    kwargs = dict(n_draws=5, n_abundance_bins=10, n_detection_bins=5)
    first, _ = draw_matched_background(
        signature, eligible, mean_cpm, detection, rng=np.random.default_rng(11), **kwargs
    )
    second, _ = draw_matched_background(
        signature, eligible, mean_cpm, detection, rng=np.random.default_rng(11), **kwargs
    )
    assert first == second


# --- end to end on a synthetic prepared object ----------------------------

GENES = [f"G{i:03d}" for i in range(60)]


def synthetic_astrocytes(tmp_path: Path) -> Path:
    rng = np.random.default_rng(5)
    n = 16
    counts = rng.integers(20, 400, size=(n, len(GENES))).astype(np.int64)
    counts[:8, :10] *= 6          # a planted signature in the first half of the rows
    obs = {
        "_index": [f"row{i}" for i in range(n)],
        "donor_id": [f"H01.01.{i // 2:03d}" for i in range(n)],
        "brain_region": ["MTG"] * 8 + ["DFC"] * 8,
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
        "source_methods": ["10Xv3.1"] * n,
        "source_alignments": ["cellranger"] * n,
        "source_batch_vendor_names": ["batch"] * n,
        "released_meta_severely_affected_donor": ["N"] * n,
        "released_meta_neurotypical_reference": ["False"] * n,
        "total_umi": counts.sum(axis=1).tolist(),
    }
    path = tmp_path / "prepared" / "Astrocyte_donor_region_supertype_counts.h5ad"
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.attrs["encoding-type"] = "anndata"
        handle.attrs["encoding-version"] = "0.1.0"
        dataset = handle.create_dataset("X", data=counts)
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
    return path


PLANTED = GENES[:10]


def test_competitive_score_is_the_signature_minus_its_background(tmp_path: Path):
    path = synthetic_astrocytes(tmp_path)
    spec = BackgroundSpec(signatures=("planted",), n_draws=8, seed=42)
    scan = scan_lineage(path, {"planted": PLANTED}, background=spec)

    draws = scan["background_genes"][("planted", "Astrocyte")]
    assert len(draws) == 8
    for draw in draws:
        assert len(draw) == len(PLANTED)
        assert not set(draw) & set(PLANTED)

    signature = score_domain(scan, "Astrocyte", "planted", PLANTED)
    background = np.vstack([
        score_domain(scan, "Astrocyte", "bg", draw, collect_genes=False)["score_z_mean"]
        for draw in draws
    ])
    competitive = signature["score_z_mean"] - background.mean(axis=0)

    # The planted rows must stay higher after the correction, and the correction
    # must not be a no-op: the background carries the shared recovery axis.
    assert competitive[:8].mean() > competitive[8:].mean()
    assert np.any(np.abs(competitive - signature["score_z_mean"]) > 1e-9)
    # collect_genes=False really does skip the per-gene diagnostics.
    assert score_domain(scan, "Astrocyte", "bg", draws[0], collect_genes=False)["gene_rows"] == []


def test_requesting_no_background_leaves_the_scan_unchanged(tmp_path: Path):
    path = synthetic_astrocytes(tmp_path)
    plain = scan_lineage(path, {"planted": PLANTED})
    with_background = scan_lineage(
        path, {"planted": PLANTED}, background=BackgroundSpec(signatures=("planted",), n_draws=4)
    )
    assert plain["background_genes"] == {}
    # Normalization and the signature's own score are identical either way: the
    # extra extracted columns must not perturb TMM, logCPM or standardization.
    assert np.array_equal(plain["tmm"], with_background["tmm"])
    assert np.allclose(
        score_domain(plain, "Astrocyte", "planted", PLANTED)["score_z_mean"],
        score_domain(with_background, "Astrocyte", "planted", PLANTED)["score_z_mean"],
    )
