"""Row-level TMM/logCPM normalization and predefined module scoring of prepared pseudobulk.

Scope and non-scope
-------------------
This module scores *already prepared* SEA-AD pseudobulk rows. It performs no
acquisition and changes nothing upstream. Every operation is at the grain the
release delivers, namely

    donor x brain region x released supertype x sample/library preparation,

and repeated sample/library rows are **never** summed, averaged or otherwise
collapsed before normalization or scoring. Combining them is a separate,
explicitly compared decision made downstream.

Normalization
-------------
Counts are summed raw UMIs, so a conventional bulk workflow applies:

1. library size ``L_i`` = sum of raw counts over all 36,601 released features;
2. TMM normalization factors ``f_i`` (Robinson & Oshlack 2010, Genome Biology
   11:R25), implemented here directly rather than through edgeR, which is not
   installed in this project's R environment. The reference row is the row whose
   upper-quartile of ``count / library size`` is closest to the mean
   upper-quartile, exactly as in ``edgeR::calcNormFactors``; trimming is 30% on
   the log-ratios and 5% on the average abundances; factors are rescaled to
   geometric mean 1;
3. effective library size ``L_i * f_i``;
4. log2 counts per million with an abundance-scaled prior, as in
   ``edgeR::cpm(log = TRUE, prior.count = 2)``:

       p_i       = prior * (L_i f_i) / mean_j(L_j f_j)
       logCPM_gi = log2( 1e6 * (y_gi + p_i) / (L_i f_i + 2 p_i) )

No single-cell normalization is used anywhere: no per-nucleus scaling, no
``scanpy`` recipe, no highly-variable-gene selection.

Domains
-------
Normalization factors, gene detectability and gene standardization are computed
within a *scoring domain* rather than within a whole released lineage, because
TMM assumes most genes are not differentially expressed between the rows being
compared. The released ``Immune`` lineage is therefore split into Micro/PVM,
Lymphocyte and Monocyte, and the primary resident-myeloid axis uses Micro/PVM
only.

Scoring rule
------------
For each signature and domain, over the genes that are both present in the
released feature set and detectably expressed in that domain:

    z_gi   = (logCPM_gi - mean_i logCPM_gi) / sd_i logCPM_gi
    score_i = mean_g z_gi                         (primary, ``score_z_mean``)
    score_i = mean_g logCPM_gi                    (secondary, ``score_logcpm_mean``)

The primary score is a plain unweighted mean of within-domain gene z-scores. It
has no tuned parameter, and nothing about it is fitted to pathology, morphology
or cognition. The secondary score omits the data-dependent standardization so
that a conclusion which depends on standardization is visible as such.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import h5py
import numpy as np

from omics.pseudobulk_inspection import read_dataframe_column
from omics.pseudobulk_preparation import H5AD_RELATIVE, PreparationError
from omics.signature_registry import DOMAINS, OUTPUT_RELATIVE


LINEAGES = ("Immune", "Astrocyte", "Oligodendrocyte", "OPC")

#: A gene counts as detectably expressed in a domain when it is non-zero in at
#: least this fraction of the domain's rows.
DETECTION_FRACTION = 0.10

#: edgeR defaults, restated here so the notebook can print them.
TMM_LOGRATIO_TRIM = 0.30
TMM_SUM_TRIM = 0.05
CPM_PRIOR_COUNT = 2.0

#: Rows are read from the H5AD in blocks of this many, so a 36,601-column
#: int64 matrix never has to be held whole.
ROW_BLOCK = 512

NULL_CONTROL_SEED = 20260916

#: Seed for the matched-background draws of the competitive score. Separate from
#: the null-control seed so that the two controls can never share a draw.
BACKGROUND_SEED = 20260917

#: How many matched-background sets are drawn per signature. Enough for the mean
#: background to be stable; deliberately not enough to be an empirical-null
#: testing framework, which is not what this correction is for.
BACKGROUND_DRAWS = 25

#: Matching strata. Genes are binned jointly by abundance and by how often they
#: are detected, because those are the two properties a size-matched random set
#: fails to control and the two that drive the generic recovery axis.
BACKGROUND_ABUNDANCE_BINS = 10
BACKGROUND_DETECTION_BINS = 5


@dataclass(frozen=True)
class BackgroundSpec:
    """Which signatures get matched-background draws, and how they are drawn."""

    signatures: tuple[str, ...]
    n_draws: int = BACKGROUND_DRAWS
    seed: int = BACKGROUND_SEED
    n_abundance_bins: int = BACKGROUND_ABUNDANCE_BINS
    n_detection_bins: int = BACKGROUND_DETECTION_BINS

OBS_METADATA = (
    "_index",
    "donor_id",
    "brain_region",
    "source_subclass_or_lineage",
    "released_supertype",
    "Number of nuclei",
    "total_umi",
    "n_source_rows",
    "n_library_preps",
    "aggregation_status",
    "source_release",
    "source_sample_names",
    "source_library_preps",
    "source_methods",
    "source_alignments",
    "source_batch_vendor_names",
    "released_meta_severely_affected_donor",
    "released_meta_neurotypical_reference",
)

ROW_KEY_FIELDS = (
    "prepared_obs_index",
    "lineage",
    "domain",
    "donor_id",
    "brain_region",
    "supertype",
    "sample_name",
    "library_prep",
    "assay_method",
    "alignment",
    "batch_vendor_name",
    "n_nuclei",
    "total_umi",
    "severely_affected_donor",
)


def prepared_path_for_lineage(root: Path, lineage: str, prepared_dir: Path | None = None) -> Path:
    directory = prepared_dir or (root / H5AD_RELATIVE)
    path = directory / f"{lineage}_donor_region_supertype_counts.h5ad"
    if not path.exists():
        raise PreparationError(f"Prepared object is missing: {path}")
    return path


def domain_of(lineage: str, supertype: str) -> str:
    """Map a released supertype onto its scoring domain.

    The released Immune taxonomy is preserved: Lymphocyte and Monocyte are their
    own domains and are never folded into the resident-myeloid axis, and the
    Micro-PVM family keeps the conservative Micro/PVM label because the release
    does not resolve microglia from perivascular macrophages.
    """
    if lineage != "Immune":
        return lineage
    if supertype.startswith("Micro-PVM"):
        return "Micro/PVM"
    if supertype in {"Lymphocyte", "Monocyte"}:
        return supertype
    raise PreparationError(f"Unmapped Immune supertype: {supertype!r}")


def _average_rank(values: np.ndarray) -> np.ndarray:
    """Average ranks, 1-based, matching R's ``rank(ties.method = 'average')``."""
    n = values.shape[0]
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    boundaries = np.flatnonzero(
        np.concatenate(([True], sorted_values[1:] != sorted_values[:-1], [True]))
    )
    starts, stops = boundaries[:-1], boundaries[1:]
    group_rank = 0.5 * (starts + stops + 1)
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.repeat(group_rank, stops - starts)
    return ranks


def _upper_quartile_block(block: np.ndarray, library_sizes: np.ndarray) -> np.ndarray:
    """edgeR's ``.calcFactorQuantile`` with p = 0.75, for a block of rows."""
    safe = np.where(library_sizes > 0, library_sizes, 1.0)
    scaled = block / safe[:, None]
    quartile = np.quantile(scaled, 0.75, axis=1)
    fallback = scaled.mean(axis=1)
    quartile = np.where(quartile > 0, quartile, fallback)
    return np.where(library_sizes > 0, quartile, 0.0)


def tmm_factor(
    counts: np.ndarray,
    reference: np.ndarray,
    library_size: float,
    reference_library_size: float,
    *,
    logratio_trim: float = TMM_LOGRATIO_TRIM,
    sum_trim: float = TMM_SUM_TRIM,
) -> float:
    """One TMM normalization factor, before the geometric-mean rescaling."""
    if library_size <= 0 or reference_library_size <= 0:
        return 1.0
    usable = (counts > 0) & (reference > 0)
    if usable.sum() < 2:
        return 1.0
    obs = counts[usable].astype(np.float64) / library_size
    ref = reference[usable].astype(np.float64) / reference_library_size
    log_ratio = np.log2(obs / ref)
    abundance = 0.5 * (np.log2(obs) + np.log2(ref))
    # Approximate asymptotic variance of the log ratio (Robinson & Oshlack eq. 7).
    variance = (
        (library_size - counts[usable]) / (library_size * counts[usable])
        + (reference_library_size - reference[usable]) / (reference_library_size * reference[usable])
    )
    finite = np.isfinite(log_ratio) & np.isfinite(abundance) & np.isfinite(variance) & (variance > 0)
    log_ratio, abundance, variance = log_ratio[finite], abundance[finite], variance[finite]
    n = log_ratio.shape[0]
    if n < 2:
        return 1.0
    if float(np.max(np.abs(log_ratio))) < 1e-6:
        return 1.0

    low_ratio = math.floor(n * logratio_trim) + 1
    high_ratio = n + 1 - low_ratio
    low_sum = math.floor(n * sum_trim) + 1
    high_sum = n + 1 - low_sum
    rank_ratio = _average_rank(log_ratio)
    rank_abundance = _average_rank(abundance)
    keep = (
        (rank_ratio >= low_ratio)
        & (rank_ratio <= high_ratio)
        & (rank_abundance >= low_sum)
        & (rank_abundance <= high_sum)
    )
    if not keep.any():
        return 1.0
    weights = 1.0 / variance[keep]
    factor = float(np.sum(log_ratio[keep] * weights) / np.sum(weights))
    if not np.isfinite(factor):
        return 1.0
    return float(2.0**factor)


def _read_obs(handle: h5py.File) -> dict[str, np.ndarray]:
    obs = handle["obs"]
    out: dict[str, np.ndarray] = {}
    for name in OBS_METADATA:
        out[name] = np.asarray(read_dataframe_column(obs, name)).astype(str)
    return out


def _numeric(values: np.ndarray) -> np.ndarray:
    return np.array([float(v) if v not in {"", "nan", "None"} else np.nan for v in values])


def draw_matched_background(
    signature_columns: Sequence[int],
    eligible: np.ndarray,
    mean_cpm: np.ndarray,
    detection_frequency: np.ndarray,
    *,
    n_draws: int,
    rng: np.random.Generator,
    n_abundance_bins: int,
    n_detection_bins: int,
) -> tuple[list[list[int]], dict[str, Any]]:
    """Draw background gene sets matched to a signature on abundance and detection.

    ``eligible`` is the domain's genome-wide pool of detectably expressed genes.
    Genes are binned jointly into quantile strata of mean CPM and of the fraction
    of rows in which they are non-zero; each draw then replaces the signature's
    genes stratum by stratum, so a background set has the same abundance and
    detection profile as the signature and differs only in identity.

    The signature's own genes are never eligible as their own background. When a
    stratum cannot supply enough partners, the deficit is taken from the nearest
    strata by abundance rank and counted in ``widened_draws``, rather than being
    filled silently or by sampling the same gene twice.
    """
    signature = np.asarray(sorted(set(int(c) for c in signature_columns)), dtype=np.int64)
    pool = np.setdiff1d(eligible, signature, assume_unique=False)
    if signature.shape[0] == 0 or pool.shape[0] == 0:
        return [], {
            "n_signature_genes_matched": int(signature.shape[0]),
            "n_eligible_background_genes": int(pool.shape[0]),
            "n_strata_used": 0,
            "widened_draws": 0,
            "n_genes_taken_from_neighbouring_strata": 0,
        }

    abundance = np.log2(mean_cpm[eligible] + 1.0)
    detection = detection_frequency[eligible]

    def quantile_bin(values: np.ndarray, n_bins: int) -> np.ndarray:
        edges = np.quantile(values, np.linspace(0.0, 1.0, n_bins + 1)[1:-1])
        return np.searchsorted(edges, values, side="right")

    stratum = (
        quantile_bin(abundance, n_abundance_bins) * n_detection_bins
        + quantile_bin(detection, n_detection_bins)
    )
    stratum_of = dict(zip(eligible.tolist(), stratum.tolist()))
    signature_set = set(signature.tolist())

    # Per stratum: the genes eligible as partners, and how many partners the
    # signature needs from it. The signature's own genes are excluded from every
    # stratum, so a gene can never be its own background.
    available: dict[int, list[int]] = {}
    for gene, code in zip(eligible.tolist(), stratum.tolist()):
        if gene not in signature_set:
            available.setdefault(code, []).append(gene)
    available = {code: np.asarray(genes, dtype=np.int64) for code, genes in available.items()}

    wanted: dict[int, int] = {}
    for column in signature.tolist():
        code = stratum_of.get(column)
        if code is not None:
            wanted[code] = wanted.get(code, 0) + 1

    # Strata ordered by mean abundance, so "nearest stratum" has a meaning when
    # one of them cannot supply enough partners. Ordered over *every* occupied
    # stratum, including one that holds only signature genes and so offers no
    # partners of its own - that is exactly the stratum that has to borrow.
    all_members: dict[int, list[int]] = {}
    for gene, code in zip(eligible.tolist(), stratum.tolist()):
        all_members.setdefault(code, []).append(gene)
    stratum_abundance = {
        code: float(np.mean(np.log2(mean_cpm[np.asarray(genes, dtype=np.int64)] + 1.0)))
        for code, genes in all_members.items()
    }
    order = sorted(stratum_abundance, key=lambda code: stratum_abundance[code])
    rank_of = {code: index for index, code in enumerate(order)}

    draws: list[list[int]] = []
    widened_draws = 0
    borrowed_total = 0
    for _ in range(n_draws):
        taken: set[int] = set()
        borrowed = 0
        for code, count in wanted.items():
            here = available.get(code, np.zeros(0, dtype=np.int64))
            here = np.asarray([g for g in here.tolist() if g not in taken], dtype=np.int64)
            if here.shape[0] >= count:
                taken.update(int(g) for g in rng.choice(here, size=count, replace=False))
                continue
            # Not enough partners in the matched stratum: take all of them and
            # make up the deficit from the nearest strata by abundance, counting
            # how much had to be borrowed rather than filling it silently.
            taken.update(int(g) for g in here)
            deficit = count - here.shape[0]
            borrowed += deficit
            for neighbour in sorted(order, key=lambda c: abs(rank_of[c] - rank_of[code])):
                if deficit <= 0:
                    break
                if neighbour == code:
                    continue
                spare = np.asarray(
                    [g for g in available.get(neighbour, np.zeros(0, dtype=np.int64)).tolist()
                     if g not in taken],
                    dtype=np.int64,
                )
                if spare.shape[0] == 0:
                    continue
                take = min(deficit, spare.shape[0])
                taken.update(int(g) for g in rng.choice(spare, size=take, replace=False))
                deficit -= take
        if borrowed:
            widened_draws += 1
            borrowed_total += borrowed
        draws.append(sorted(taken))

    summary = {
        "n_signature_genes_matched": int(signature.shape[0]),
        "n_eligible_background_genes": int(pool.shape[0]),
        "n_strata_used": len(wanted),
        "widened_draws": widened_draws,
        "n_genes_taken_from_neighbouring_strata": borrowed_total,
        "signature_mean_log2_cpm": float(np.mean(np.log2(mean_cpm[signature] + 1.0))),
        "signature_mean_detection_fraction": float(np.mean(detection_frequency[signature])),
        "background_mean_log2_cpm": float(
            np.mean([np.mean(np.log2(mean_cpm[np.asarray(d)] + 1.0)) for d in draws])
        )
        if draws
        else float("nan"),
        "background_mean_detection_fraction": float(
            np.mean([np.mean(detection_frequency[np.asarray(d)]) for d in draws])
        )
        if draws
        else float("nan"),
    }
    return draws, summary


def _scan_core(
    matrix: Any,
    symbols: np.ndarray,
    obs: dict[str, np.ndarray],
    domains: np.ndarray,
    signature_genes: dict[str, list[str]],
    *,
    null_domains: Sequence[str],
    detection_fraction: float,
    prior_count: float,
    row_block: int,
    null_seed: int,
    background: BackgroundSpec | None = None,
) -> dict[str, Any]:
    """Normalize and extract, given anything that slices rows like ``m[a:b, :]``.

    Two passes over the row blocks, so a 36,601-column dense matrix is never held
    whole. Pass A accumulates library sizes, upper quartiles and genome-wide
    non-zero counts per domain. Between the passes the TMM reference row and the
    size-matched null-control genes are chosen. Pass B computes the TMM factors
    and extracts the counts of the genes that will actually be scored.
    """
    n_rows, n_genes = matrix.shape
    domain_names = sorted(set(domains.tolist()))

    library_size = np.zeros(n_rows, dtype=np.float64)
    upper_quartile = np.zeros(n_rows, dtype=np.float64)
    genes_detected = np.zeros(n_rows, dtype=np.int64)
    nonzero_by_domain = {d: np.zeros(n_genes, dtype=np.int64) for d in domain_names}
    # Genome-wide per-domain abundance, accumulated as a plain count fraction of
    # the row's library. It is a *matching* statistic only, used to bin genes for
    # the competitive score's background draws; no score is computed from it, so
    # it deliberately does not wait for the TMM factors of pass B.
    fraction_by_domain = {d: np.zeros(n_genes, dtype=np.float64) for d in domain_names}
    rows_by_domain = {d: int(np.sum(domains == d)) for d in domain_names}

    # --- pass A -----------------------------------------------------------
    for start in range(0, n_rows, row_block):
        stop = min(start + row_block, n_rows)
        block = np.asarray(matrix[start:stop, :], dtype=np.float64)
        if np.any(block < 0):
            raise PreparationError("Prepared counts contain negative values")
        sums = block.sum(axis=1)
        library_size[start:stop] = sums
        nonzero = block > 0
        genes_detected[start:stop] = nonzero.sum(axis=1)
        upper_quartile[start:stop] = _upper_quartile_block(block, sums)
        positive = sums > 0
        fractions = np.zeros_like(block)
        if positive.any():
            fractions[positive] = block[positive] / sums[positive][:, None]
        for domain in domain_names:
            mask = domains[start:stop] == domain
            if mask.any():
                nonzero_by_domain[domain] += nonzero[mask].sum(axis=0)
                fraction_by_domain[domain] += fractions[mask].sum(axis=0)

    detectable = {
        domain: (
            nonzero_by_domain[domain] / rows_by_domain[domain] >= detection_fraction
            if rows_by_domain[domain]
            else np.zeros(n_genes, dtype=bool)
        )
        for domain in domain_names
    }
    position = {symbol: index for index, symbol in enumerate(symbols)}

    # Null controls are size matched to the median resolved signature of the same
    # domain and drawn, with a fixed seed, from that domain's genome-wide pool of
    # detectably expressed genes. They are drawn here, between the passes, so the
    # pool is the whole feature set rather than the signature genes.
    null_genes: dict[str, list[str]] = {}
    for domain in null_domains:
        if domain not in domain_names:
            continue
        sizes = [
            sum(
                1
                for gene in dict.fromkeys(genes)
                if gene in position and detectable[domain][position[gene]]
            )
            for genes in signature_genes.values()
        ]
        sizes = [s for s in sizes if s > 0]
        size = int(np.median(sizes)) if sizes else 0
        pool = symbols[detectable[domain]]
        if size <= 0 or pool.shape[0] <= size:
            null_genes[domain] = sorted(pool.tolist())
        else:
            rng = np.random.default_rng(null_seed)
            null_genes[domain] = sorted(rng.choice(pool, size=size, replace=False).tolist())

    detection_frequency = {
        domain: (
            nonzero_by_domain[domain] / rows_by_domain[domain]
            if rows_by_domain[domain]
            else np.zeros(n_genes, dtype=np.float64)
        )
        for domain in domain_names
    }
    mean_cpm = {
        domain: (
            1e6 * fraction_by_domain[domain] / rows_by_domain[domain]
            if rows_by_domain[domain]
            else np.zeros(n_genes, dtype=np.float64)
        )
        for domain in domain_names
    }

    # Matched-background draws for the competitive score. Drawn here, like the
    # null controls, so that the pool is the whole feature set rather than the
    # genes that happen to be scored.
    background_genes: dict[tuple[str, str], list[list[str]]] = {}
    background_summary: list[dict[str, Any]] = []
    if background is not None:
        rng = np.random.default_rng(background.seed)
        for name in background.signatures:
            genes = signature_genes.get(name)
            if not genes:
                continue
            for domain in domain_names:
                eligible = np.flatnonzero(detectable[domain])
                columns_wanted = [
                    position[gene]
                    for gene in dict.fromkeys(genes)
                    if gene in position and detectable[domain][position[gene]]
                ]
                if not columns_wanted or eligible.shape[0] == 0:
                    continue
                draws, summary = draw_matched_background(
                    columns_wanted,
                    eligible,
                    mean_cpm[domain],
                    detection_frequency[domain],
                    n_draws=background.n_draws,
                    rng=rng,
                    n_abundance_bins=background.n_abundance_bins,
                    n_detection_bins=background.n_detection_bins,
                )
                if not draws:
                    continue
                background_genes[(name, domain)] = [
                    [str(symbols[column]) for column in draw] for draw in draws
                ]
                background_summary.append(
                    {"signature_name": name, "domain": domain, "n_draws": len(draws), **summary}
                )

    wanted = sorted(
        {gene for genes in signature_genes.values() for gene in genes}
        | {gene for genes in null_genes.values() for gene in genes}
        | {gene for draws in background_genes.values() for draw in draws for gene in draw}
    )
    present = [gene for gene in wanted if gene in position]
    columns = np.array([position[gene] for gene in present], dtype=np.int64)

    reference_index: dict[str, str] = {}
    reference_counts: dict[str, np.ndarray] = {}
    reference_library: dict[str, float] = {}
    for domain in domain_names:
        rows = np.flatnonzero((domains == domain) & (library_size > 0))
        if rows.shape[0] < 2:
            continue
        quartiles = upper_quartile[rows]
        reference_row = int(rows[int(np.argmin(np.abs(quartiles - float(np.mean(quartiles)))))])
        reference_index[domain] = str(obs["_index"][reference_row])
        reference_counts[domain] = np.asarray(matrix[reference_row, :], dtype=np.float64)
        reference_library[domain] = float(library_size[reference_row])

    # --- pass B -----------------------------------------------------------
    tmm = np.ones(n_rows, dtype=np.float64)
    selected = np.zeros((n_rows, columns.shape[0]), dtype=np.float64)
    for start in range(0, n_rows, row_block):
        stop = min(start + row_block, n_rows)
        block = np.asarray(matrix[start:stop, :], dtype=np.float64)
        if columns.shape[0]:
            selected[start:stop, :] = block[:, columns]
        for offset in range(stop - start):
            row = start + offset
            domain = str(domains[row])
            if domain not in reference_counts or library_size[row] <= 0:
                continue
            tmm[row] = tmm_factor(
                block[offset],
                reference_counts[domain],
                float(library_size[row]),
                reference_library[domain],
            )

    for domain in reference_counts:
        rows = np.flatnonzero(domains == domain)
        positive = tmm[rows][tmm[rows] > 0]
        if positive.shape[0]:
            tmm[rows] = tmm[rows] / float(np.exp(np.mean(np.log(positive))))

    effective = library_size * tmm
    log_cpm = np.full(selected.shape, np.nan, dtype=np.float64)
    for domain in domain_names:
        rows = np.flatnonzero(domains == domain)
        if rows.shape[0] == 0:
            continue
        mean_effective = float(np.mean(effective[rows]))
        if mean_effective <= 0:
            continue
        prior = prior_count * effective[rows] / mean_effective
        denominator = effective[rows] + 2.0 * prior
        log_cpm[rows, :] = np.log2(1e6 * (selected[rows, :] + prior[:, None]) / denominator[:, None])

    detected = {
        domain: (
            np.array([detectable[domain][position[gene]] for gene in present], dtype=bool)
            if present
            else np.zeros(0, dtype=bool)
        )
        for domain in domain_names
    }

    return {
        "n_rows": n_rows,
        "n_genes": n_genes,
        "symbols": symbols,
        "obs": obs,
        "domains": domains,
        "domain_names": domain_names,
        "present_genes": present,
        "gene_columns": columns,
        "library_size": library_size,
        "upper_quartile": upper_quartile,
        "genes_detected": genes_detected,
        "tmm": tmm,
        "effective_library_size": effective,
        "log_cpm": log_cpm,
        "detected": detected,
        "n_detectable_genome_wide": {d: int(detectable[d].sum()) for d in domain_names},
        "reference_index": reference_index,
        "raw_selected": selected,
        "null_genes": null_genes,
        "detection_frequency": detection_frequency,
        "mean_cpm": mean_cpm,
        "background_genes": background_genes,
        "background_summary": background_summary,
    }


def _read_var_and_obs(handle: h5py.File) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    symbols = np.asarray(
        read_dataframe_column(handle["var"], str(handle["var"].attrs["_index"]))
    ).astype(str)
    obs = _read_obs(handle)
    domains = np.array(
        [
            domain_of(lineage, supertype)
            for lineage, supertype in zip(
                obs["source_subclass_or_lineage"], obs["released_supertype"]
            )
        ]
    )
    return symbols, obs, domains


def scan_lineage(
    prepared_path: Path,
    signature_genes: dict[str, list[str]],
    *,
    null_domains: Sequence[str] = (),
    detection_fraction: float = DETECTION_FRACTION,
    prior_count: float = CPM_PRIOR_COUNT,
    row_block: int = ROW_BLOCK,
    null_seed: int = NULL_CONTROL_SEED,
    background: BackgroundSpec | None = None,
) -> dict[str, Any]:
    """Normalize one prepared lineage at the released row grain. Nothing is collapsed."""
    with h5py.File(prepared_path, "r") as handle:
        symbols, obs, domains = _read_var_and_obs(handle)
        scan = _scan_core(
            handle["X"],
            symbols,
            obs,
            domains,
            signature_genes,
            null_domains=null_domains,
            detection_fraction=detection_fraction,
            prior_count=prior_count,
            row_block=row_block,
            null_seed=null_seed,
            background=background,
        )
    scan["prepared_path"] = str(prepared_path)
    scan["pooled"] = False
    return scan


#: Columns that identify one repeated-measurement group. Pooling sums raw counts
#: inside a group; it is a *sensitivity* arm only and never replaces the
#: row-level scores.
POOL_KEY = ("donor_id", "brain_region", "released_supertype")


def pooled_scan_lineage(
    prepared_path: Path,
    signature_genes: dict[str, list[str]],
    *,
    null_domains: Sequence[str] = (),
    detection_fraction: float = DETECTION_FRACTION,
    prior_count: float = CPM_PRIOR_COUNT,
    row_block: int = ROW_BLOCK,
    null_seed: int = NULL_CONTROL_SEED,
) -> dict[str, Any]:
    """Sum raw counts across repeated sample/library rows, then normalize and score.

    This is the pooled sensitivity arm of section 9. Summing before normalization
    weights each repeated measurement by its own library size, which is exactly
    the behaviour the row-level workflow is designed to avoid; it is computed so
    the two can be compared, not because it is preferred.
    """
    with h5py.File(prepared_path, "r") as handle:
        symbols, obs, domains = _read_var_and_obs(handle)
        n_rows, n_genes = handle["X"].shape

        keys = list(zip(*(obs[column] for column in POOL_KEY)))
        order: dict[tuple[str, ...], int] = {}
        assignment = np.empty(n_rows, dtype=np.int64)
        for row, key in enumerate(keys):
            assignment[row] = order.setdefault(key, len(order))
        n_pooled = len(order)

        pooled = np.zeros((n_pooled, n_genes), dtype=np.int64)
        for start in range(0, n_rows, row_block):
            stop = min(start + row_block, n_rows)
            block = np.asarray(handle["X"][start:stop, :], dtype=np.int64)
            np.add.at(pooled, assignment[start:stop], block)

        first = np.zeros(n_pooled, dtype=np.int64)
        seen: set[int] = set()
        for row in range(n_rows):
            group = int(assignment[row])
            if group not in seen:
                first[group] = row
                seen.add(group)

        pooled_obs: dict[str, np.ndarray] = {}
        for column, values in obs.items():
            pooled_obs[column] = values[first]
        pooled_obs["_index"] = np.array(
            ["__".join(key) for key in order.keys()], dtype=object
        ).astype(str)
        pooled_obs["Number of nuclei"] = np.array(
            [
                str(int(np.sum(_numeric(obs["Number of nuclei"])[assignment == group])))
                for group in range(n_pooled)
            ]
        )
        pooled_obs["n_source_rows"] = np.array(
            [str(int(np.sum(assignment == group))) for group in range(n_pooled)]
        )
        pooled_obs["source_sample_names"] = np.array(
            [
                ";".join(sorted(set(obs["source_sample_names"][assignment == group].tolist())))
                for group in range(n_pooled)
            ]
        )
        pooled_obs["source_methods"] = np.array(
            [
                ";".join(sorted(set(obs["source_methods"][assignment == group].tolist())))
                for group in range(n_pooled)
            ]
        )
        pooled_domains = domains[first]

    scan = _scan_core(
        pooled,
        symbols,
        pooled_obs,
        pooled_domains,
        signature_genes,
        null_domains=null_domains,
        detection_fraction=detection_fraction,
        prior_count=prior_count,
        row_block=row_block,
        null_seed=null_seed,
    )
    scan["prepared_path"] = str(prepared_path)
    scan["pooled"] = True
    scan["n_source_rows_pooled"] = n_rows
    return scan


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.shape[0] < 3:
        return float("nan")
    xs, ys = x - x.mean(), y - y.mean()
    denominator = float(np.sqrt(np.sum(xs**2) * np.sum(ys**2)))
    if denominator <= 0:
        return float("nan")
    return float(np.sum(xs * ys) / denominator)


def score_domain(
    scan: dict[str, Any],
    domain: str,
    signature_name: str,
    genes: Sequence[str],
    *,
    collect_genes: bool = True,
) -> dict[str, Any] | None:
    """Score one signature inside one domain; ``None`` when no gene is usable.

    ``collect_genes=False`` skips the per-gene contribution diagnostics. It is for
    the background draws of the competitive score, which are averaged over and
    whose individual genes are never reported.
    """
    rows = np.flatnonzero(scan["domains"] == domain)
    if rows.shape[0] == 0:
        return None
    index_of = {gene: i for i, gene in enumerate(scan["present_genes"])}
    detected = scan["detected"][domain]

    defined = list(dict.fromkeys(genes))
    present = [gene for gene in defined if gene in index_of]
    expressed = [gene for gene in present if detected[index_of[gene]]]
    if not expressed:
        return {
            "signature_name": signature_name,
            "domain": domain,
            "rows": rows,
            "n_defined": len(defined),
            "n_present": len(present),
            "n_detectably_expressed": 0,
            "n_used": 0,
            "missing_genes": [g for g in defined if g not in index_of],
            "undetected_genes": [g for g in present if g not in set(expressed)],
            "score_z_mean": np.full(rows.shape[0], np.nan),
            "score_logcpm_mean": np.full(rows.shape[0], np.nan),
            "gene_rows": [],
        }

    columns = [index_of[gene] for gene in expressed]
    values = scan["log_cpm"][np.ix_(rows, columns)]
    mean = values.mean(axis=0)
    sd = values.std(axis=0, ddof=1) if rows.shape[0] > 1 else np.zeros(values.shape[1])
    usable = sd > 0
    used_genes = [gene for gene, keep in zip(expressed, usable) if keep]
    if not used_genes:
        z = np.zeros((rows.shape[0], 0))
    else:
        z = (values[:, usable] - mean[usable]) / sd[usable]
    score_z = z.mean(axis=1) if z.shape[1] else np.full(rows.shape[0], np.nan)
    score_logcpm = values.mean(axis=1)

    gene_rows: list[dict[str, Any]] = []
    for index, gene in enumerate(used_genes if collect_genes else []):
        gene_z = z[:, index]
        if z.shape[1] > 1:
            without = (z.sum(axis=1) - gene_z) / (z.shape[1] - 1)
            loo_corr = _pearson(score_z, without)
        else:
            without = np.full(rows.shape[0], np.nan)
            loo_corr = float("nan")
        gene_rows.append(
            {
                "signature_name": signature_name,
                "domain": domain,
                "gene_symbol": gene,
                "mean_log_cpm": float(mean[usable][index]),
                "sd_log_cpm": float(sd[usable][index]),
                "fraction_rows_nonzero": float(
                    (scan["raw_selected"][np.ix_(rows, [index_of[gene]])] > 0).mean()
                ),
                "corr_gene_z_with_score": _pearson(gene_z, score_z),
                "leave_one_out_corr": loo_corr,
            }
        )

    return {
        "signature_name": signature_name,
        "domain": domain,
        "rows": rows,
        "n_defined": len(defined),
        "n_present": len(present),
        "n_detectably_expressed": len(expressed),
        "n_used": len(used_genes),
        "missing_genes": [g for g in defined if g not in index_of],
        "undetected_genes": [g for g in present if g not in set(expressed)],
        "score_z_mean": score_z,
        "score_logcpm_mean": score_logcpm,
        "gene_rows": gene_rows,
    }


def _row_metadata(scan: dict[str, Any], row: int) -> dict[str, Any]:
    obs = scan["obs"]
    return {
        "prepared_obs_index": str(obs["_index"][row]),
        "lineage": str(obs["source_subclass_or_lineage"][row]),
        "domain": str(scan["domains"][row]),
        "donor_id": str(obs["donor_id"][row]),
        "brain_region": str(obs["brain_region"][row]),
        "supertype": str(obs["released_supertype"][row]),
        "sample_name": str(obs["source_sample_names"][row]),
        "library_prep": str(obs["source_library_preps"][row]),
        "assay_method": str(obs["source_methods"][row]),
        "alignment": str(obs["source_alignments"][row]),
        "batch_vendor_name": str(obs["source_batch_vendor_names"][row]),
        "n_nuclei": str(obs["Number of nuclei"][row]),
        "total_umi": str(obs["total_umi"][row]),
        "severely_affected_donor": str(obs["released_meta_severely_affected_donor"][row]),
    }


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: Sequence[str]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    written = 0
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
            written += 1
    temporary.replace(path)
    return written


def _collect(
    scan: dict[str, Any],
    lineage: str,
    resolved: Sequence[dict[str, Any]],
    nulls: Sequence[dict[str, Any]],
    genes_by_signature: dict[str, list[str]],
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    """Score every signature that belongs to this scan and flatten the results."""
    results: list[dict[str, Any]] = []
    for signature in resolved:
        if signature["lineage"] != lineage:
            continue
        domain = signature["domain"]
        if domain not in scan["domain_names"]:
            continue
        result = score_domain(
            scan, domain, signature["signature_name"], genes_by_signature[signature["signature_name"]]
        )
        if result is not None:
            results.append(result)
    for null in nulls:
        domain = null["domain"]
        if null["lineage"] != lineage or domain not in scan["null_genes"]:
            continue
        drawn = scan["null_genes"][domain]
        genes_by_signature[null["signature_name"]] = drawn
        result = score_domain(scan, domain, null["signature_name"], drawn)
        if result is not None:
            results.append(result)

    normalization_rows: list[dict] = []
    for row in range(scan["n_rows"]):
        record = _row_metadata(scan, row)
        record.update(
            {
                "computed_library_size": f"{scan['library_size'][row]:.0f}",
                "upper_quartile": f"{scan['upper_quartile'][row]:.10g}",
                "tmm_factor": f"{scan['tmm'][row]:.10g}",
                "effective_library_size": f"{scan['effective_library_size'][row]:.2f}",
                "n_genes_detected": str(scan["genes_detected"][row]),
            }
        )
        normalization_rows.append(record)

    score_rows: list[dict] = []
    coverage_rows: list[dict] = []
    contribution_rows: list[dict] = []
    for result in results:
        coverage_rows.append(
            {
                "lineage": lineage,
                "domain": result["domain"],
                "signature_name": result["signature_name"],
                "n_defined": result["n_defined"],
                "n_present": result["n_present"],
                "n_detectably_expressed": result["n_detectably_expressed"],
                "n_used": result["n_used"],
                "fraction_present": (
                    f"{result['n_present'] / result['n_defined']:.6f}" if result["n_defined"] else ""
                ),
                "fraction_detectably_expressed": (
                    f"{result['n_detectably_expressed'] / result['n_defined']:.6f}"
                    if result["n_defined"]
                    else ""
                ),
                "n_rows_scored": int(result["rows"].shape[0]),
                "missing_genes": ";".join(result["missing_genes"]),
                "undetected_genes": ";".join(result["undetected_genes"]),
            }
        )
        for gene_row in result["gene_rows"]:
            contribution_rows.append({"lineage": lineage, **gene_row})
        for offset, row in enumerate(result["rows"].tolist()):
            record = _row_metadata(scan, row)
            record.update(
                {
                    "signature_name": result["signature_name"],
                    "n_genes_used": result["n_used"],
                    "score_z_mean": f"{result['score_z_mean'][offset]:.6f}",
                    "score_logcpm_mean": f"{result['score_logcpm_mean'][offset]:.6f}",
                    "tmm_factor": f"{scan['tmm'][row]:.10g}",
                    "effective_library_size": f"{scan['effective_library_size'][row]:.2f}",
                }
            )
            score_rows.append(record)
    return normalization_rows, score_rows, coverage_rows, contribution_rows


SCORE_FIELDS = list(ROW_KEY_FIELDS) + [
    "signature_name",
    "n_genes_used",
    "score_z_mean",
    "score_logcpm_mean",
    "tmm_factor",
    "effective_library_size",
]

COVERAGE_FIELDS = [
    "lineage",
    "domain",
    "signature_name",
    "n_defined",
    "n_present",
    "n_detectably_expressed",
    "n_used",
    "fraction_present",
    "fraction_detectably_expressed",
    "n_rows_scored",
    "missing_genes",
    "undetected_genes",
]


def score_selection(
    root: Path,
    registry_path: Path,
    genes_path: Path,
    output_dir: Path,
    *,
    lineages: Sequence[str] = LINEAGES,
    prepared_dir: Path | None = None,
    detection_fraction: float = DETECTION_FRACTION,
    prior_count: float = CPM_PRIOR_COUNT,
    include_pooled_sensitivity: bool = True,
) -> dict[str, Any]:
    """Normalize and score every prepared lineage, writing the row-level derivatives.

    The row-level arm is authoritative. The pooled arm sums raw counts across
    repeated sample/library rows before normalizing, and is written to separate
    files so that it can never overwrite the row-level scores.
    """
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    signatures = registry["signatures"]
    genes_by_signature: dict[str, list[str]] = defaultdict(list)
    with genes_path.open(encoding="utf-8") as handle:
        for record in csv.DictReader(handle):
            genes_by_signature[record["signature_name"]].append(record["gene_symbol"])

    resolved = [s for s in signatures if s["resolution_status"] == "resolved"]
    nulls = [s for s in signatures if s["feature_origin"] == "null_control_current_data"]

    normalization_rows: list[dict] = []
    score_rows: list[dict] = []
    coverage_rows: list[dict] = []
    contribution_rows: list[dict] = []
    pooled_score_rows: list[dict] = []
    pooled_coverage_rows: list[dict] = []
    per_lineage: list[dict[str, Any]] = []

    for lineage in lineages:
        prepared = prepared_path_for_lineage(root, lineage, prepared_dir)
        lineage_signatures = [s for s in resolved if s["lineage"] == lineage]
        wanted = {
            s["signature_name"]: genes_by_signature[s["signature_name"]] for s in lineage_signatures
        }
        null_domains = tuple(dict.fromkeys(n["domain"] for n in nulls if n["lineage"] == lineage))

        scan = scan_lineage(
            prepared,
            wanted,
            null_domains=null_domains,
            detection_fraction=detection_fraction,
            prior_count=prior_count,
        )
        rows, scores, coverage, contributions = _collect(
            scan, lineage, lineage_signatures, nulls, genes_by_signature
        )
        normalization_rows += rows
        score_rows += scores
        coverage_rows += coverage
        contribution_rows += contributions

        entry = {
            "lineage": lineage,
            "prepared_path": scan["prepared_path"],
            "n_rows": scan["n_rows"],
            "n_genes": scan["n_genes"],
            "domains": scan["domain_names"],
            "n_rows_by_domain": {
                d: int((scan["domains"] == d).sum()) for d in scan["domain_names"]
            },
            "n_genes_detectable_by_domain": scan["n_detectable_genome_wide"],
            "tmm_reference_row": scan["reference_index"],
            "n_signatures_scored": len(coverage),
        }

        if include_pooled_sensitivity:
            pooled = pooled_scan_lineage(
                prepared,
                wanted,
                null_domains=null_domains,
                detection_fraction=detection_fraction,
                prior_count=prior_count,
            )
            _, pooled_scores, pooled_coverage, _ = _collect(
                pooled, lineage, lineage_signatures, nulls, dict(genes_by_signature)
            )
            pooled_score_rows += pooled_scores
            pooled_coverage_rows += pooled_coverage
            entry["n_pooled_rows"] = pooled["n_rows"]

        per_lineage.append(entry)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        output_dir / "row_normalization.csv",
        normalization_rows,
        list(ROW_KEY_FIELDS)
        + [
            "computed_library_size",
            "upper_quartile",
            "tmm_factor",
            "effective_library_size",
            "n_genes_detected",
        ],
    )
    write_csv(output_dir / "row_state_scores.csv", score_rows, SCORE_FIELDS)
    write_csv(output_dir / "signature_gene_coverage.csv", coverage_rows, COVERAGE_FIELDS)
    write_csv(
        output_dir / "signature_gene_contributions.csv",
        contribution_rows,
        [
            "lineage",
            "domain",
            "signature_name",
            "gene_symbol",
            "mean_log_cpm",
            "sd_log_cpm",
            "fraction_rows_nonzero",
            "corr_gene_z_with_score",
            "leave_one_out_corr",
        ],
    )
    write_csv(
        output_dir / "null_control_genes.csv",
        [
            {"signature_name": null["signature_name"], "gene_symbol": gene}
            for null in nulls
            for gene in genes_by_signature.get(null["signature_name"], [])
        ],
        ["signature_name", "gene_symbol"],
    )
    if include_pooled_sensitivity:
        write_csv(output_dir / "pooled_state_scores.csv", pooled_score_rows, SCORE_FIELDS)
        write_csv(
            output_dir / "pooled_signature_gene_coverage.csv",
            pooled_coverage_rows,
            COVERAGE_FIELDS,
        )

    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "registry": str(registry_path),
        "registry_created_utc": registry["provenance"]["created_utc"],
        "msigdb_release": registry["provenance"]["msigdb_release"],
        "row_grain": "donor x brain region x released supertype x sample/library preparation",
        "rows_collapsed_before_scoring": False,
        "normalization": {
            "input": "summed raw UMI counts over the released features",
            "library_size": "row sum of raw counts",
            "method": "TMM (Robinson & Oshlack 2010) implemented in this module, edgeR defaults",
            "logratio_trim": TMM_LOGRATIO_TRIM,
            "sum_trim": TMM_SUM_TRIM,
            "reference_selection": "row with upper-quartile closest to the domain mean upper-quartile",
            "rescaling": "factors rescaled to geometric mean 1 within each domain",
            "log_cpm": "log2(1e6 * (y + p_i) / (L_i f_i + 2 p_i)), p_i = prior * L_i f_i / mean(L f)",
            "prior_count": prior_count,
            "single_cell_normalization_used": False,
        },
        "detectability": {
            "rule": "non-zero in at least this fraction of the domain's rows",
            "fraction": detection_fraction,
        },
        "scoring": {
            "primary": "score_z_mean = unweighted mean of within-domain gene z-scores of logCPM",
            "secondary": "score_logcpm_mean = unweighted mean logCPM of the same genes",
            "parameters_tuned_against_outcomes": False,
        },
        "pooled_sensitivity_arm": {
            "computed": include_pooled_sensitivity,
            "key": list(POOL_KEY),
            "note": "raw counts summed before normalization; written to separate files",
        },
        "null_control_seed": NULL_CONTROL_SEED,
        "lineages": per_lineage,
        "counts": {
            "normalization_rows": len(normalization_rows),
            "score_rows": len(score_rows),
            "coverage_rows": len(coverage_rows),
            "contribution_rows": len(contribution_rows),
            "pooled_score_rows": len(pooled_score_rows),
        },
    }
    (output_dir / "state_scoring_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    return provenance
