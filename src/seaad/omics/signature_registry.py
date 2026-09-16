"""Predefined glial molecular-state signatures, and where their genes come from.

Nothing in this module looks at the SEA-AD pseudobulk data. It is a *curation*
layer: it names candidate state axes, records their provenance, and resolves the
gene membership of the resolvable ones from an external, versioned, openly
downloadable source (MSigDB GMT files).

Two rules are enforced structurally rather than by convention:

1. A candidate whose exact published gene list cannot be recovered is kept in the
   registry with ``resolution_status = "unresolved"`` and **no genes**. Genes are
   never guessed, completed, or reconstructed from a paper's prose.
2. Membership for a resolved candidate is copied verbatim from the external
   source file, whose URL, release and SHA-256 are recorded next to it.

There are exactly two resolution routes, and both are file-level copies: an
MSigDB GMT set, and a named group inside a published supplementary workbook. The
second route exists because the only human, AD-cohort-derived microglial *state*
definitions are released as a supplementary spreadsheet rather than as a gene
set. It applies no threshold of its own: every released row of the named group is
taken, in the order the authors published it.

The registry deliberately contains no signature derived from the current data,
from morphology, from cognition, or from cognitive resilience. The single
data-dependent entry is a size-matched *null control* whose genes are drawn from
detectably expressed genes at scoring time; it is flagged ``null_control`` and is
not a candidate state variable.
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from omics.xlsx import read_records


MSIGDB_RELEASE = "2025.1.Hs"
MSIGDB_BASE_URL = "https://data.broadinstitute.org/gsea-msigdb/msigdb/release"
MSIGDB_COLLECTIONS = ("h.all", "c5.go.bp", "c2.cgp")


@dataclass(frozen=True)
class SupplementSource:
    """A published supplementary table that lists gene membership by state name.

    The file is downloaded once, pinned by SHA-256, and cached next to the MSigDB
    GMTs. ``group_column`` and ``gene_column`` are the authors' own column names;
    nothing is renamed, filtered or thresholded on the way in.
    """

    key: str
    citation: str
    url: str
    filename: str
    sha256: str
    sheet: str
    group_column: str
    gene_column: str
    released_filter: str


SUPPLEMENTS: dict[str, SupplementSource] = {
    "sun_2023_table_s1_state_markers": SupplementSource(
        key="sun_2023_table_s1_state_markers",
        citation=(
            "Sun N, Victor MB, Park YP, et al. Human microglial state dynamics in "
            "Alzheimer's disease progression. Cell 2023;186(20):4386-4403. "
            "Table S1, page 2 (microglial state marker genes)."
        ),
        url=(
            "https://ars.els-cdn.com/content/image/"
            "1-s2.0-S0092867423009716-mmc1.xlsx"
        ),
        filename="sun_2023_cell_table_s1.xlsx",
        sha256="957c47ea82f0f884d8e470cc31fd37567ac3c65b6f6cabeae043b010c657d949",
        sheet="Page 2.StateMarkers",
        group_column="microgliaState",
        gene_column="gene",
        released_filter=(
            "none applied here. The released sheet is already the authors' own "
            "FindAllMarkers output: every row has avg_log2FC >= 0.25 and is an "
            "induced marker of its state. All rows of the named state are taken "
            "verbatim, in published order, deduplicated only if the sheet repeats "
            "a symbol."
        ),
    )
}

OUTPUT_RELATIVE = Path("data/derivatives/sea-ad/omics_state")

#: Scoring domains. A domain is the set of prepared rows over which normalization
#: factors, gene detectability and gene standardization are computed. The Immune
#: lineage is split because the released `Immune` subclass mixes resident myeloid
#: cells with lymphocytes and monocytes, and those must never be pooled.
DOMAINS: dict[str, dict[str, str]] = {
    "Micro/PVM": {
        "lineage": "Immune",
        "supertype_rule": "released supertype starts with 'Micro-PVM'",
        "taxonomy_note": (
            "The release does not separate microglia from perivascular macrophages "
            "inside the Micro-PVM family, so the conservative label Micro/PVM is used."
        ),
    },
    "Lymphocyte": {
        "lineage": "Immune",
        "supertype_rule": "released supertype == 'Lymphocyte'",
        "taxonomy_note": "Held separately; never merged into the resident-myeloid domain.",
    },
    "Monocyte": {
        "lineage": "Immune",
        "supertype_rule": "released supertype == 'Monocyte'",
        "taxonomy_note": "Held separately; never merged into the resident-myeloid domain.",
    },
    "Astrocyte": {
        "lineage": "Astrocyte",
        "supertype_rule": "all released supertypes of the Astrocyte lineage",
        "taxonomy_note": "Released Astro_1..Astro_6-SEAAD preserved exactly.",
    },
    "Oligodendrocyte": {
        "lineage": "Oligodendrocyte",
        "supertype_rule": "all released supertypes of the Oligodendrocyte lineage",
        "taxonomy_note": "Released Oligo_1..Oligo_5-SEAAD preserved exactly.",
    },
    "OPC": {
        "lineage": "OPC",
        "supertype_rule": "all released supertypes of the OPC lineage",
        "taxonomy_note": "Kept apart from Oligodendrocyte; the two may move in opposite directions.",
    },
}


class RegistryError(RuntimeError):
    """The signature registry cannot be built without guessing gene membership."""


@dataclass(frozen=True)
class Candidate:
    """One candidate molecular-state signature.

    ``msigdb_set`` is the only route by which genes enter the registry. A
    candidate with ``msigdb_set = None`` is unresolved by construction.
    """

    signature_name: str
    biological_axis: str
    lineage: str
    domain: str
    source_publication: str
    gene_definition_source: str
    feature_origin: str
    directionality: str
    role: str = "candidate_state"
    msigdb_set: str | None = None
    msigdb_collection: str | None = None
    supplement_key: str | None = None
    supplement_group: str | None = None
    unresolved_reason: str | None = None
    notes: str = ""

    @property
    def resolution_status(self) -> str:
        if self.msigdb_set or self.supplement_key:
            return "resolved"
        if self.role == "null_control":
            return "resolved_at_scoring_time"
        return "unresolved"


_EXT = "external_predefined"
_SEAAD = "SEAAD_published_predefined"
_NULL = "null_control_current_data"


RESIDENT_MYELOID: tuple[Candidate, ...] = (
    Candidate(
        signature_name="microglial_glial_activation_GOBP",
        biological_axis="resident-myeloid activation",
        lineage="Immune",
        domain="Micro/PVM",
        source_publication="Gene Ontology Consortium, GO:0061900 glial cell activation",
        gene_definition_source=f"MSigDB {MSIGDB_RELEASE} GOBP_GLIAL_CELL_ACTIVATION",
        feature_origin=_EXT,
        directionality="higher score = more activated glia",
        msigdb_set="GOBP_GLIAL_CELL_ACTIVATION",
        msigdb_collection="c5.go.bp",
        notes=(
            "Ontology-defined, human, and independent of any AD dataset. It is an "
            "activation axis, not an AD-derived disease-associated-microglia state."
        ),
    ),
    Candidate(
        signature_name="neuroinflammatory_response_GOBP",
        biological_axis="neuroinflammation",
        lineage="Immune",
        domain="Micro/PVM",
        source_publication="Gene Ontology Consortium, GO:0150076 neuroinflammatory response",
        gene_definition_source=f"MSigDB {MSIGDB_RELEASE} GOBP_NEUROINFLAMMATORY_RESPONSE",
        feature_origin=_EXT,
        directionality="higher score = stronger neuroinflammatory programme",
        msigdb_set="GOBP_NEUROINFLAMMATORY_RESPONSE",
        msigdb_collection="c5.go.bp",
    ),
    Candidate(
        signature_name="inflammatory_response_HALLMARK",
        biological_axis="generic inflammatory response",
        lineage="Immune",
        domain="Micro/PVM",
        source_publication="Liberzon et al. 2015, Cell Systems, MSigDB Hallmark collection",
        gene_definition_source=f"MSigDB {MSIGDB_RELEASE} HALLMARK_INFLAMMATORY_RESPONSE",
        feature_origin=_EXT,
        directionality="higher score = stronger inflammatory response",
        msigdb_set="HALLMARK_INFLAMMATORY_RESPONSE",
        msigdb_collection="h.all",
        notes="Not brain-specific. Included as a broad comparator for the two GO axes.",
    ),
    Candidate(
        signature_name="tnfa_signaling_via_nfkb_HALLMARK",
        biological_axis="NF-kB inflammatory signalling",
        lineage="Immune",
        domain="Micro/PVM",
        source_publication="Liberzon et al. 2015, Cell Systems, MSigDB Hallmark collection",
        gene_definition_source=f"MSigDB {MSIGDB_RELEASE} HALLMARK_TNFA_SIGNALING_VIA_NFKB",
        feature_origin=_EXT,
        directionality="higher score = stronger NF-kB inflammatory signalling",
        msigdb_set="HALLMARK_TNFA_SIGNALING_VIA_NFKB",
        msigdb_collection="h.all",
    ),
    Candidate(
        signature_name="sun_2023_MG4_lipid_processing",
        biological_axis="human lipid-processing microglial state",
        lineage="Immune",
        domain="Micro/PVM",
        source_publication="Sun et al. 2023, Cell 186:4386-4403",
        gene_definition_source="Table S1 page 2 (StateMarkers), group MG4",
        feature_origin=_EXT,
        directionality="higher score = more MG4-like lipid-processing state",
        supplement_key="sun_2023_table_s1_state_markers",
        supplement_group="MG4",
        notes=(
            "The only human, AD-cohort-derived, state-level Micro/PVM definition "
            "available. Derived from 194 ROSMAP prefrontal-cortex donors by "
            "unsupervised clustering of microglial nuclei. The authors' own GO "
            "enrichment of this marker set (Table S1 page 3) is dominated by "
            "cholesterol storage and efflux, lipid storage and lipoprotein "
            "biosynthesis (ABCA1, APOE, TREM2, PPARG, MSR1, GPNMB), which is what "
            "'lipid processing' names. Scored on Micro/PVM pseudobulk it measures a "
            "shift of the whole resident-myeloid pool towards that state, not the "
            "abundance of MG4 cells; the two are not separable at this grain."
        ),
    ),
    Candidate(
        signature_name="sun_2023_MG8_inflammatory_II",
        biological_axis="human inflammatory microglial state II",
        lineage="Immune",
        domain="Micro/PVM",
        source_publication="Sun et al. 2023, Cell 186:4386-4403",
        gene_definition_source="Table S1 page 2 (StateMarkers), group MG8",
        feature_origin=_EXT,
        directionality="higher score = more MG8-like inflammatory state",
        supplement_key="sun_2023_table_s1_state_markers",
        supplement_group="MG8",
        notes=(
            "Same source and cohort as MG4, and a distinct construct: the authors' "
            "GO enrichment of this set is led by cytokine-mediated signalling and "
            "response to lipopolysaccharide (CD86, CIITA, IL15, IL10RA, TNFRSF1B, "
            "IRAK3), with LRRK2, SPON1 and FOXP1 among its top markers. Retained as "
            "the secondary Micro/PVM alternative because it shares MG4's provenance "
            "but not its biology."
        ),
    ),
)

REACTIVE_ASTROCYTE: tuple[Candidate, ...] = (
    Candidate(
        signature_name="astrocyte_activation_GOBP",
        biological_axis="reactive astrocyte state",
        lineage="Astrocyte",
        domain="Astrocyte",
        source_publication="Gene Ontology Consortium, GO:0048143 astrocyte activation",
        gene_definition_source=f"MSigDB {MSIGDB_RELEASE} GOBP_ASTROCYTE_ACTIVATION",
        feature_origin=_EXT,
        directionality="higher score = more reactive astrocytes",
        msigdb_set="GOBP_ASTROCYTE_ACTIVATION",
        msigdb_collection="c5.go.bp",
        notes=(
            "Escartin et al. 2021 recommend describing astrocyte reactivity by "
            "measured markers rather than by A1/A2 polarity; an ontology activation "
            "axis is consistent with that recommendation."
        ),
    ),
    Candidate(
        signature_name="astrocyte_activation_regulation_GOBP",
        biological_axis="regulation of astrocyte reactivity",
        lineage="Astrocyte",
        domain="Astrocyte",
        source_publication="Gene Ontology Consortium, GO:0061889 regulation of astrocyte activation",
        gene_definition_source=f"MSigDB {MSIGDB_RELEASE} GOBP_REGULATION_OF_ASTROCYTE_ACTIVATION",
        feature_origin=_EXT,
        directionality="higher score = stronger regulatory programme over astrocyte reactivity",
        msigdb_set="GOBP_REGULATION_OF_ASTROCYTE_ACTIVATION",
        msigdb_collection="c5.go.bp",
        notes="Small set; retained as a secondary comparator only.",
    ),
)

OLIGODENDROCYTE_MYELINATION: tuple[Candidate, ...] = (
    Candidate(
        signature_name="myelin_assembly_GOBP",
        biological_axis="myelination",
        lineage="Oligodendrocyte",
        domain="Oligodendrocyte",
        source_publication="Gene Ontology Consortium, GO:0032288 myelin assembly",
        gene_definition_source=f"MSigDB {MSIGDB_RELEASE} GOBP_MYELIN_ASSEMBLY",
        feature_origin=_EXT,
        directionality="higher score = more myelin-assembly programme",
        msigdb_set="GOBP_MYELIN_ASSEMBLY",
        msigdb_collection="c5.go.bp",
    ),
    Candidate(
        signature_name="myelin_maintenance_GOBP",
        biological_axis="myelin maintenance",
        lineage="Oligodendrocyte",
        domain="Oligodendrocyte",
        source_publication="Gene Ontology Consortium, GO:0043217 myelin maintenance",
        gene_definition_source=f"MSigDB {MSIGDB_RELEASE} GOBP_MYELIN_MAINTENANCE",
        feature_origin=_EXT,
        directionality="higher score = more myelin-maintenance programme",
        msigdb_set="GOBP_MYELIN_MAINTENANCE",
        msigdb_collection="c5.go.bp",
    ),
    Candidate(
        signature_name="cns_axon_ensheathment_GOBP",
        biological_axis="CNS axon ensheathment",
        lineage="Oligodendrocyte",
        domain="Oligodendrocyte",
        source_publication="Gene Ontology Consortium, GO:0032291 axon ensheathment in central nervous system",
        gene_definition_source=f"MSigDB {MSIGDB_RELEASE} GOBP_AXON_ENSHEATHMENT_IN_CENTRAL_NERVOUS_SYSTEM",
        feature_origin=_EXT,
        directionality="higher score = more axon-ensheathment programme",
        msigdb_set="GOBP_AXON_ENSHEATHMENT_IN_CENTRAL_NERVOUS_SYSTEM",
        msigdb_collection="c5.go.bp",
    ),
)

OPC_RESPONSE: tuple[Candidate, ...] = (
    Candidate(
        signature_name="oligodendrocyte_differentiation_GOBP",
        biological_axis="OPC-to-oligodendrocyte differentiation",
        lineage="OPC",
        domain="OPC",
        source_publication="Gene Ontology Consortium, GO:0048709 oligodendrocyte differentiation",
        gene_definition_source=f"MSigDB {MSIGDB_RELEASE} GOBP_OLIGODENDROCYTE_DIFFERENTIATION",
        feature_origin=_EXT,
        directionality="higher score = more differentiation programme",
        msigdb_set="GOBP_OLIGODENDROCYTE_DIFFERENTIATION",
        msigdb_collection="c5.go.bp",
        notes="Closest ontology proxy for a remyelination response; not remyelination-specific.",
    ),
    Candidate(
        signature_name="positive_regulation_oligodendrocyte_differentiation_GOBP",
        biological_axis="pro-differentiation regulation",
        lineage="OPC",
        domain="OPC",
        source_publication="Gene Ontology Consortium, GO:0048714 positive regulation of oligodendrocyte differentiation",
        gene_definition_source=(
            f"MSigDB {MSIGDB_RELEASE} GOBP_POSITIVE_REGULATION_OF_OLIGODENDROCYTE_DIFFERENTIATION"
        ),
        feature_origin=_EXT,
        directionality="higher score = stronger pro-differentiation regulation",
        msigdb_set="GOBP_POSITIVE_REGULATION_OF_OLIGODENDROCYTE_DIFFERENTIATION",
        msigdb_collection="c5.go.bp",
    ),
    Candidate(
        signature_name="opc_proliferation_GOBP",
        biological_axis="OPC proliferation",
        lineage="OPC",
        domain="OPC",
        source_publication="Gene Ontology Consortium, GO:0070445 oligodendrocyte progenitor proliferation",
        gene_definition_source=f"MSigDB {MSIGDB_RELEASE} GOBP_OLIGODENDROCYTE_PROGENITOR_PROLIFERATION",
        feature_origin=_EXT,
        directionality="higher score = more OPC proliferation programme",
        msigdb_set="GOBP_OLIGODENDROCYTE_PROGENITOR_PROLIFERATION",
        msigdb_collection="c5.go.bp",
        notes="Very small set; reported but expected to be unstable.",
    ),
    Candidate(
        signature_name="core_oligodendrocyte_differentiation_GOBERT",
        biological_axis="experimentally derived oligodendrocyte differentiation core",
        lineage="OPC",
        domain="OPC",
        source_publication="Gobert et al. 2009, Molecular and Cellular Biology",
        gene_definition_source=f"MSigDB {MSIGDB_RELEASE} GOBERT_CORE_OLIGODENDROCYTE_DIFFERENTIATION",
        feature_origin=_EXT,
        directionality="higher score = more differentiation programme",
        msigdb_set="GOBERT_CORE_OLIGODENDROCYTE_DIFFERENTIATION",
        msigdb_collection="c2.cgp",
        notes="Experimentally derived rather than ontology-derived; oligodendroglial, not AD-specific.",
    ),
)

#: Lineage-identity sets. These are *not* state axes. They are scored anyway so
#: that a state score which merely tracks lineage identity is visible as such.
IDENTITY_CONTROLS: tuple[Candidate, ...] = (
    Candidate(
        signature_name="astrocyte_identity_LEIN",
        biological_axis="astrocyte identity (control, not a state axis)",
        lineage="Astrocyte",
        domain="Astrocyte",
        source_publication="Lein et al. 2007, Nature (Allen Mouse Brain Atlas cell-type markers)",
        gene_definition_source=f"MSigDB {MSIGDB_RELEASE} LEIN_ASTROCYTE_MARKERS",
        feature_origin=_EXT,
        directionality="higher score = stronger astrocyte identity programme",
        role="identity_control",
        msigdb_set="LEIN_ASTROCYTE_MARKERS",
        msigdb_collection="c2.cgp",
    ),
    Candidate(
        signature_name="oligodendrocyte_identity_LEIN",
        biological_axis="oligodendrocyte identity (control, not a state axis)",
        lineage="Oligodendrocyte",
        domain="Oligodendrocyte",
        source_publication="Lein et al. 2007, Nature (Allen Mouse Brain Atlas cell-type markers)",
        gene_definition_source=f"MSigDB {MSIGDB_RELEASE} LEIN_OLIGODENDROCYTE_MARKERS",
        feature_origin=_EXT,
        directionality="higher score = stronger oligodendrocyte identity programme",
        role="identity_control",
        msigdb_set="LEIN_OLIGODENDROCYTE_MARKERS",
        msigdb_collection="c2.cgp",
    ),
)

#: Candidates whose exact published definition could not be recovered without
#: guessing. They carry no genes and are not scored.
UNRESOLVED: tuple[Candidate, ...] = (
    Candidate(
        signature_name="liddelow_2017_pan_reactive_astrocyte",
        biological_axis="pan-reactive astrocyte state",
        lineage="Astrocyte",
        domain="Astrocyte",
        source_publication="Liddelow et al. 2017, Nature 541:481-487",
        gene_definition_source="Extended Data Fig. 12a",
        feature_origin=_EXT,
        directionality="higher = more pan-reactive",
        unresolved_reason=(
            "The PAN/A1/A2 cassettes appear only inside a figure image; the article "
            "text and tables list the full qPCR panel, not the partition into cassettes."
        ),
    ),
    Candidate(
        signature_name="liddelow_2017_A1_neurotoxic_astrocyte",
        biological_axis="A1 (neuroinflammatory) astrocyte state",
        lineage="Astrocyte",
        domain="Astrocyte",
        source_publication="Liddelow et al. 2017, Nature 541:481-487",
        gene_definition_source="Extended Data Fig. 12a",
        feature_origin=_EXT,
        directionality="higher = more A1-like",
        unresolved_reason=(
            "Cassette membership is image-only. Escartin et al. 2021 additionally "
            "advise against treating A1/A2 as a definitive polarity."
        ),
    ),
    Candidate(
        signature_name="liddelow_2017_A2_astrocyte",
        biological_axis="A2 (ischaemic) astrocyte state",
        lineage="Astrocyte",
        domain="Astrocyte",
        source_publication="Liddelow et al. 2017, Nature 541:481-487",
        gene_definition_source="Extended Data Fig. 12a",
        feature_origin=_EXT,
        directionality="higher = more A2-like",
        unresolved_reason="Cassette membership is image-only.",
    ),
    Candidate(
        signature_name="habib_2020_disease_associated_astrocyte",
        biological_axis="disease-associated astrocyte (DAA)",
        lineage="Astrocyte",
        domain="Astrocyte",
        source_publication="Habib et al. 2020, Nature Neuroscience 23:701-706",
        gene_definition_source="Supplementary tables (spreadsheet)",
        feature_origin=_EXT,
        directionality="higher = more DAA-like",
        unresolved_reason="Mouse-derived; gene list lives in a supplementary spreadsheet not retrieved here.",
    ),
    Candidate(
        signature_name="keren_shaul_2017_DAM_up",
        biological_axis="disease-associated microglia, induced arm",
        lineage="Immune",
        domain="Micro/PVM",
        source_publication="Keren-Shaul et al. 2017, Cell 169:1276-1290",
        gene_definition_source="Table S3 (spreadsheet); GEO GSE98969",
        feature_origin=_EXT,
        directionality="higher = more DAM-like",
        unresolved_reason=(
            "Mouse-derived and defined in a supplementary spreadsheet. The article text "
            "names only exemplar genes, which is not a signature."
        ),
    ),
    Candidate(
        signature_name="homeostatic_microglia_down",
        biological_axis="homeostatic microglia, repressed arm",
        lineage="Immune",
        domain="Micro/PVM",
        source_publication="Butovsky et al. 2014, Nature Neuroscience; Keren-Shaul et al. 2017",
        gene_definition_source="Supplementary tables (spreadsheet)",
        feature_origin=_EXT,
        directionality="lower = loss of homeostatic identity",
        unresolved_reason=(
            "No exact human list recovered. The commonly quoted P2RY12/P2RY13/CX3CR1/"
            "TMEM119/SELPLG core is a prose exemplar, not a published cassette."
        ),
    ),
    Candidate(
        signature_name="opc_remyelination_response",
        biological_axis="OPC remyelination response",
        lineage="OPC",
        domain="OPC",
        source_publication="no single predefined human signature identified",
        gene_definition_source="none",
        feature_origin=_EXT,
        directionality="undefined",
        unresolved_reason=(
            "No externally predefined human OPC remyelination-response gene set was "
            "found. The ontology differentiation sets are used as declared proxies."
        ),
    ),
    Candidate(
        signature_name="seaad_gabitto_2024_cps_gene_program",
        biological_axis="SEA-AD pseudoprogression-associated expression change",
        lineage="all glial lineages",
        domain="per supertype",
        source_publication="Gabitto et al. 2024, Nature Neuroscience 27:2366-2383",
        gene_definition_source=(
            "effect_size_table.csv produced by SEA-AD_2024/Single nucleus omics/"
            "04_Differential expression analysis/02_Build gene dynamic space.ipynb"
        ),
        feature_origin=_SEAAD,
        directionality="signed effect size along the continuous pseudo-progression score",
        unresolved_reason=(
            "SEA-AD publishes a per-gene per-supertype effect-size table, not a named "
            "gene set; a signature would require choosing a threshold. The table is MTG-only "
            "and is defined against CPS, which is also the QC axis in section 12, so any "
            "score built from it would be partly circular there. Not retrieved: this task "
            "performs no further acquisition."
        ),
    ),
    Candidate(
        signature_name="seaad_multiregion_2026_published_state_score",
        biological_axis="any SEA-AD-published continuous molecular-state score",
        lineage="all glial lineages",
        domain="per supertype",
        source_publication="SEA-AD Multiregion 2026 release (2026-06-22)",
        gene_definition_source="none published",
        feature_origin=_SEAAD,
        directionality="n/a",
        unresolved_reason=(
            "The release publishes no molecular-state score and no disease-state cell "
            "label for any lineage. The '-SEAAD' supertypes are taxonomy additions, "
            "i.e. composition, not within-lineage expression state."
        ),
    ),
)

NULL_CONTROLS: tuple[Candidate, ...] = tuple(
    Candidate(
        signature_name=f"null_control_size_matched_{domain.replace('/', '_')}",
        biological_axis="size-matched random gene set (negative control)",
        lineage=spec["lineage"],
        domain=domain,
        source_publication="not a published signature",
        gene_definition_source=(
            "drawn at scoring time, with a fixed seed, from genes detectably expressed "
            "in this domain; size matched to the median resolved signature"
        ),
        feature_origin=_NULL,
        directionality="none; a null reference for reliability and effect size",
        role="null_control",
        notes="Never a candidate state variable. Present only as a floor for sections 5-9.",
    )
    for domain, spec in DOMAINS.items()
    if domain in {"Micro/PVM", "Astrocyte", "Oligodendrocyte", "OPC"}
)


def all_candidates() -> tuple[Candidate, ...]:
    return (
        RESIDENT_MYELOID
        + REACTIVE_ASTROCYTE
        + OLIGODENDROCYTE_MYELINATION
        + OPC_RESPONSE
        + IDENTITY_CONTROLS
        + UNRESOLVED
        + NULL_CONTROLS
    )


def _gmt_url(collection: str) -> str:
    return f"{MSIGDB_BASE_URL}/{MSIGDB_RELEASE}/{collection}.v{MSIGDB_RELEASE}.symbols.gmt"


def _fetch(collection: str, cache_dir: Path, *, timeout: float = 300.0) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / f"{collection}.v{MSIGDB_RELEASE}.symbols.gmt"
    if target.exists() and target.stat().st_size > 0:
        return target
    url = _gmt_url(collection)
    temporary = target.with_name(target.name + ".tmp")
    with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310 - fixed https host
        temporary.write_bytes(response.read())
    temporary.replace(target)
    return target


def _fetch_supplement(
    source: SupplementSource, cache_dir: Path, *, timeout: float = 600.0
) -> Path:
    """Cache a published supplementary workbook, pinned by SHA-256.

    The digest is checked on every call, cached or freshly downloaded, so a
    silently re-issued supplement cannot change a gene list without failing here.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / source.filename
    if not (target.exists() and target.stat().st_size > 0):
        temporary = target.with_name(target.name + ".tmp")
        request = urllib.request.Request(  # noqa: S310 - fixed https host
            source.url, headers={"User-Agent": "ad-resilience/0.1 (signature registry)"}
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            temporary.write_bytes(response.read())
        temporary.replace(target)
    digest = sha256_file(target)
    if digest != source.sha256:
        raise RegistryError(
            f"{source.filename} has SHA-256 {digest}, expected {source.sha256}; "
            "refusing to read gene membership from an unexpected file"
        )
    return target


def read_supplement_groups(source: SupplementSource, path: Path) -> dict[str, list[str]]:
    """Parse one supplementary sheet into ``{group name: [gene symbols]}``.

    Order is the order the authors published. Nothing is filtered by p-value,
    effect size or rank: applying a cut-off here would make the membership this
    project's choice rather than the publication's.
    """
    records = read_records(path, source.sheet)
    if not records:
        raise RegistryError(f"{source.sheet} in {path.name} has no rows")
    for column in (source.group_column, source.gene_column):
        if column not in records[0]:
            raise RegistryError(
                f"{source.sheet} in {path.name} has no column {column!r}; "
                f"columns are {sorted(records[0])}"
            )
    groups: dict[str, list[str]] = {}
    seen: dict[str, set[str]] = {}
    for record in records:
        group = (record.get(source.group_column) or "").strip()
        gene = (record.get(source.gene_column) or "").strip()
        if not group or not gene:
            continue
        members = groups.setdefault(group, [])
        known = seen.setdefault(group, set())
        if gene not in known:
            members.append(gene)
            known.add(gene)
    return groups


def read_gmt(path: Path) -> dict[str, list[str]]:
    """Parse a GMT file into ``{set name: [gene symbols]}``, order preserved."""
    sets: dict[str, list[str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        fields = line.split("\t")
        if len(fields) < 3:
            raise RegistryError(f"Malformed GMT line in {path}: {fields[:1]}")
        name = fields[0]
        genes: list[str] = []
        seen: set[str] = set()
        for gene in fields[2:]:
            gene = gene.strip()
            if gene and gene not in seen:
                genes.append(gene)
                seen.add(gene)
        sets[name] = genes
    return sets


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build_registry(
    output_dir: Path,
    cache_dir: Path,
    candidates: Iterable[Candidate] | None = None,
) -> dict[str, object]:
    """Resolve gene membership and write the registry plus its long gene table."""
    candidates = tuple(candidates) if candidates is not None else all_candidates()
    wanted = {c.msigdb_collection for c in candidates if c.msigdb_collection}
    gmt_paths = {collection: _fetch(collection, cache_dir) for collection in sorted(wanted)}
    gmts = {collection: read_gmt(path) for collection, path in gmt_paths.items()}

    supplement_dir = cache_dir.parent / "supplements"
    wanted_supplements = sorted({c.supplement_key for c in candidates if c.supplement_key})
    supplement_paths: dict[str, Path] = {}
    supplement_groups: dict[str, dict[str, list[str]]] = {}
    for key in wanted_supplements:
        if key not in SUPPLEMENTS:
            raise RegistryError(f"Unknown supplement source {key!r}")
        source = SUPPLEMENTS[key]
        path = _fetch_supplement(source, supplement_dir)
        supplement_paths[key] = path
        supplement_groups[key] = read_supplement_groups(source, path)

    entries: list[dict[str, object]] = []
    gene_rows: list[dict[str, str]] = []
    for candidate in candidates:
        genes: list[str] = []
        if candidate.msigdb_set and candidate.supplement_key:
            raise RegistryError(
                f"{candidate.signature_name} names two resolution routes; "
                "a signature must have exactly one source of truth"
            )
        if candidate.msigdb_set:
            collection = candidate.msigdb_collection
            if collection is None:
                raise RegistryError(f"{candidate.signature_name} names a set but no collection")
            if candidate.msigdb_set not in gmts[collection]:
                raise RegistryError(
                    f"{candidate.msigdb_set} is absent from {collection} "
                    f"v{MSIGDB_RELEASE}; refusing to substitute another set"
                )
            genes = gmts[collection][candidate.msigdb_set]
        elif candidate.supplement_key:
            group = candidate.supplement_group
            available = supplement_groups[candidate.supplement_key]
            if group is None:
                raise RegistryError(
                    f"{candidate.signature_name} names a supplement but no group"
                )
            if group not in available:
                raise RegistryError(
                    f"Group {group!r} is absent from "
                    f"{SUPPLEMENTS[candidate.supplement_key].filename}; "
                    f"present groups are {sorted(available)}"
                )
            genes = available[group]
        if genes:
            for gene in genes:
                gene_rows.append({"signature_name": candidate.signature_name, "gene_symbol": gene})
        entry = {
            "signature_name": candidate.signature_name,
            "biological_axis": candidate.biological_axis,
            "lineage": candidate.lineage,
            "domain": candidate.domain,
            "source_publication": candidate.source_publication,
            "gene_definition_source": candidate.gene_definition_source,
            "feature_origin": candidate.feature_origin,
            "directionality": candidate.directionality,
            "role": candidate.role,
            "resolution_status": candidate.resolution_status,
            "n_genes": len(genes) if (candidate.msigdb_set or candidate.supplement_key) else None,
            "unresolved_reason": candidate.unresolved_reason,
            "notes": candidate.notes,
            "msigdb_set": candidate.msigdb_set,
            "msigdb_collection": candidate.msigdb_collection,
            "supplement_key": candidate.supplement_key,
            "supplement_group": candidate.supplement_group,
        }
        entries.append(entry)

    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "msigdb_release": MSIGDB_RELEASE,
        "sources": [
            {
                "collection": collection,
                "url": _gmt_url(collection),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
                "n_sets": len(gmts[collection]),
            }
            for collection, path in sorted(gmt_paths.items())
        ],
        "supplements": [
            {
                "key": key,
                "citation": SUPPLEMENTS[key].citation,
                "url": SUPPLEMENTS[key].url,
                "sheet": SUPPLEMENTS[key].sheet,
                "group_column": SUPPLEMENTS[key].group_column,
                "gene_column": SUPPLEMENTS[key].gene_column,
                "released_filter": SUPPLEMENTS[key].released_filter,
                "sha256": sha256_file(supplement_paths[key]),
                "size_bytes": supplement_paths[key].stat().st_size,
                "n_groups": len(supplement_groups[key]),
                "group_sizes": {
                    group: len(members)
                    for group, members in sorted(supplement_groups[key].items())
                },
            }
            for key in wanted_supplements
        ],
        "rules": [
            "Gene membership is copied verbatim from the GMT; no symbol is edited or mapped.",
            "Supplement membership is every released row of the named group, in published "
            "order, with no threshold applied by this project.",
            "A candidate without an MSigDB set is recorded unresolved and carries no genes.",
            "No signature is derived from the SEA-AD pseudobulk data, morphology or cognition.",
        ],
        "domains": DOMAINS,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    registry = {"provenance": provenance, "signatures": entries}
    (output_dir / "signature_registry.json").write_text(
        json.dumps(registry, indent=2) + "\n", encoding="utf-8"
    )
    _write_csv(
        output_dir / "signature_registry.csv",
        entries,
        [
            "signature_name",
            "biological_axis",
            "lineage",
            "domain",
            "source_publication",
            "gene_definition_source",
            "feature_origin",
            "directionality",
            "role",
            "resolution_status",
            "n_genes",
            "unresolved_reason",
            "notes",
            "msigdb_set",
            "msigdb_collection",
            "supplement_key",
            "supplement_group",
        ],
    )
    _write_csv(output_dir / "signature_genes.csv", gene_rows, ["signature_name", "gene_symbol"])
    return registry


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    import csv

    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})
    temporary.replace(path)
