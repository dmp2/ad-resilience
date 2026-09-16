"""One sensitivity on the frozen state score: a matched-background correction.

`state_scoring` measures a signature as the unweighted mean of its genes'
within-domain z-scores. That number is not zero for an arbitrary gene set: a row
with deeper recovery, or a supertype whose whole transcriptome sits high, pushes
*every* gene up together, which is why `02_seaad_omics_state_exploration.qmd`
found that size-matched random gene sets reproduce across replicate libraries
just as well as any candidate signature.

The competitive score subtracts that shared axis:

    competitive_score = signature mean-z  -  matched-background mean-z

The background is not a random gene set. Each background gene replaces one
signature gene from the same joint stratum of *mean abundance* and *detection
frequency*, so the two sets have the same technical profile and differ only in
which genes they are. Averaging over a few dozen draws makes the background
estimate stable without turning this into an empirical-null testing framework;
the question here is only whether a simple background correction materially
changes the biological measurement, not whether any score is significant.

Nothing in this module changes the primary score. It reads the same prepared
counts through the same normalization and writes to its own files; the row-level
`score_z_mean` it reports is copied from the same code path and can be checked
against `row_state_scores.csv`.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from omics.state_scoring import (
    BACKGROUND_ABUNDANCE_BINS,
    BACKGROUND_DETECTION_BINS,
    BACKGROUND_DRAWS,
    BACKGROUND_SEED,
    CPM_PRIOR_COUNT,
    DETECTION_FRACTION,
    LINEAGES,
    ROW_KEY_FIELDS,
    BackgroundSpec,
    _row_metadata,
    prepared_path_for_lineage,
    scan_lineage,
    score_domain,
    write_csv,
)

#: The two domains a molecular-state covariate is being built for. Oligodendrocyte
#: and OPC are secondary axes (see notebook 03) and are not corrected here.
COMPETITIVE_DOMAINS: tuple[str, ...] = ("Micro/PVM", "Astrocyte")

COMPETITIVE_FIELDS = list(ROW_KEY_FIELDS) + [
    "signature_name",
    "n_genes_used",
    "score_z_mean",
    "background_mean_z",
    "background_sd_z",
    "competitive_score",
    "n_background_draws",
]

SUMMARY_FIELDS = [
    "lineage",
    "domain",
    "signature_name",
    "n_draws",
    "n_signature_genes_matched",
    "n_eligible_background_genes",
    "n_strata_used",
    "widened_draws",
    "n_genes_taken_from_neighbouring_strata",
    "signature_mean_log2_cpm",
    "background_mean_log2_cpm",
    "signature_mean_detection_fraction",
    "background_mean_detection_fraction",
    "median_background_sd_z",
    "standard_error_of_background_mean",
]


def competitive_selection(
    root: Path,
    registry_path: Path,
    genes_path: Path,
    output_dir: Path,
    *,
    lineages: Sequence[str] = LINEAGES,
    domains: Sequence[str] = COMPETITIVE_DOMAINS,
    prepared_dir: Path | None = None,
    detection_fraction: float = DETECTION_FRACTION,
    prior_count: float = CPM_PRIOR_COUNT,
    n_draws: int = BACKGROUND_DRAWS,
    seed: int = BACKGROUND_SEED,
    include_null_controls: bool = True,
) -> dict[str, Any]:
    """Score every resolved candidate state in ``domains`` against matched backgrounds."""
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    genes_by_signature: dict[str, list[str]] = defaultdict(list)
    with genes_path.open(encoding="utf-8") as handle:
        for record in csv.DictReader(handle):
            genes_by_signature[record["signature_name"]].append(record["gene_symbol"])

    # The size-matched null controls are drawn at scoring time, so their genes
    # live in their own file rather than in signature_genes.csv. Correcting them
    # too is what makes the reliability comparison fair: a competitive score can
    # only be compared against a null that has been through the same correction.
    roles = {"candidate_state"}
    if include_null_controls:
        roles.add("null_control")
        null_genes_path = genes_path.with_name("null_control_genes.csv")
        if null_genes_path.exists():
            with null_genes_path.open(encoding="utf-8") as handle:
                for record in csv.DictReader(handle):
                    genes_by_signature[record["signature_name"]].append(record["gene_symbol"])

    selected = [
        signature
        for signature in registry["signatures"]
        if signature["role"] in roles
        and genes_by_signature.get(signature["signature_name"])
        and signature["domain"] in set(domains)
    ]
    if not selected:
        raise ValueError(f"No resolved candidate state signature lives in {tuple(domains)}")

    score_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    per_lineage: list[dict[str, Any]] = []

    for lineage in lineages:
        here = [s for s in selected if s["lineage"] == lineage]
        if not here:
            continue
        wanted = {s["signature_name"]: genes_by_signature[s["signature_name"]] for s in here}
        spec = BackgroundSpec(
            signatures=tuple(wanted),
            n_draws=n_draws,
            seed=seed,
            n_abundance_bins=BACKGROUND_ABUNDANCE_BINS,
            n_detection_bins=BACKGROUND_DETECTION_BINS,
        )
        scan = scan_lineage(
            prepared_path_for_lineage(root, lineage, prepared_dir),
            wanted,
            null_domains=(),
            detection_fraction=detection_fraction,
            prior_count=prior_count,
            background=spec,
        )
        summary_by_key = {
            (row["signature_name"], row["domain"]): row for row in scan["background_summary"]
        }

        scored_here = 0
        for signature in here:
            name, domain = signature["signature_name"], signature["domain"]
            draws = scan["background_genes"].get((name, domain))
            if not draws:
                continue
            result = score_domain(scan, domain, name, wanted[name])
            if result is None or result["n_used"] == 0:
                continue
            background = np.vstack(
                [
                    score_domain(scan, domain, f"{name}__background", draw, collect_genes=False)[
                        "score_z_mean"
                    ]
                    for draw in draws
                ]
            )
            background_mean = background.mean(axis=0)
            background_sd = background.std(axis=0, ddof=1) if background.shape[0] > 1 else np.zeros(
                background.shape[1]
            )
            competitive = result["score_z_mean"] - background_mean

            for offset, row in enumerate(result["rows"].tolist()):
                record = _row_metadata(scan, row)
                record.update(
                    {
                        "signature_name": name,
                        "n_genes_used": result["n_used"],
                        "score_z_mean": f"{result['score_z_mean'][offset]:.6f}",
                        "background_mean_z": f"{background_mean[offset]:.6f}",
                        "background_sd_z": f"{background_sd[offset]:.6f}",
                        "competitive_score": f"{competitive[offset]:.6f}",
                        "n_background_draws": len(draws),
                    }
                )
                score_rows.append(record)

            summary = dict(summary_by_key.get((name, domain), {}))
            summary.update(
                {
                    "lineage": lineage,
                    "domain": domain,
                    "signature_name": name,
                    "n_draws": len(draws),
                    "median_background_sd_z": float(np.median(background_sd)),
                    # How precisely the *mean* background is pinned down, which is
                    # the quantity actually subtracted.
                    "standard_error_of_background_mean": float(
                        np.median(background_sd) / np.sqrt(len(draws))
                    ),
                }
            )
            summary_rows.append(summary)
            scored_here += 1

        per_lineage.append(
            {
                "lineage": lineage,
                "prepared_path": scan["prepared_path"],
                "n_rows": scan["n_rows"],
                "domains_corrected": sorted({s["domain"] for s in here}),
                "n_signatures_corrected": scored_here,
                "n_genes_extracted": len(scan["present_genes"]),
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "row_competitive_scores.csv", score_rows, COMPETITIVE_FIELDS)
    write_csv(output_dir / "background_matching_summary.csv", summary_rows, SUMMARY_FIELDS)

    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "registry": str(registry_path),
        "row_grain": "donor x brain region x released supertype x sample/library preparation",
        "definition": "competitive_score = signature mean-z - matched-background mean-z",
        "background": {
            "matched_on": ["mean CPM within the scoring domain", "fraction of domain rows non-zero"],
            "binning": (
                f"{BACKGROUND_ABUNDANCE_BINS} quantile bins of log2(mean CPM + 1) crossed with "
                f"{BACKGROUND_DETECTION_BINS} quantile bins of detection fraction"
            ),
            "pool": "genes detectably expressed in the domain, excluding the signature's own genes",
            "n_draws": n_draws,
            "seed": seed,
            "draw_rule": (
                "each draw replaces the signature's genes stratum by stratum, without "
                "replacement within a draw; a stratum that cannot supply enough partners "
                "borrows from the nearest strata by abundance and the borrowing is counted"
            ),
            "not_an_empirical_null": (
                "a few dozen draws stabilise the subtracted mean; no p-value is computed "
                "from them and no signature is selected by them"
            ),
        },
        "domains_corrected": list(domains),
        "signatures": [s["signature_name"] for s in selected],
        "null_controls_corrected": include_null_controls,
        "normalization": "unchanged; the same TMM / logCPM code path as row_state_scores.csv",
        "lineages": per_lineage,
        "counts": {"score_rows": len(score_rows), "summary_rows": len(summary_rows)},
    }
    (output_dir / "competitive_scoring_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    return provenance
