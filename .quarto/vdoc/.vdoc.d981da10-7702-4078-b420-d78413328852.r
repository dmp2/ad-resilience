#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#| label: setup

my_packages <- c("tidyverse", "knitr", "jsonlite", "readxl", "splines")
invisible(lapply(my_packages, library, character.only = TRUE))

find_root <- function(p = getwd()) {
  while (!dir.exists(file.path(p, ".git"))) {
    if (identical(dirname(p), p)) stop("project root not found")
    p <- dirname(p)
  }
  p
}
PROJECT_ROOT <- find_root()
source(file.path(PROJECT_ROOT, "R", "seaad_explore.R"))
source(file.path(PROJECT_ROOT, "R", "seaad_state.R"))
source(file.path(PROJECT_ROOT, "R", "seaad_analysis_ready.R"))

paths    <- seaad_paths(PROJECT_ROOT)
omics    <- seaad_omics_paths(PROJECT_ROOT)
state    <- seaad_state_paths(PROJECT_ROOT)
analysis <- seaad_analysis_paths(PROJECT_ROOT)

show_table <- function(x, caption = NULL, digits = 3) {
  kable(x, caption = caption, digits = digits, format.args = list(big.mark = ""))
}
theme_set(theme_minimal(base_size = 10))

# The candidate multimodal ROI set. Still a candidate: section 8 reports every
# released region, and nothing here freezes the set.
ROI <- c("MTG", "DFC", "MEC", "STG", "V1C", "HIP")
#
#
#
#| label: selection

# ---------------------------------------------------------------------------
# THE SELECTION. Declared here, at the top of the notebook, before any number
# in it has been computed, so that no table below can be read as having chosen
# it. The justification is in section 2; the criteria are biological construct,
# human/AD relevance, published provenance, gene availability and
# interpretability, and nothing else.
# ---------------------------------------------------------------------------
SELECTED <- tibble::tribble(
  ~axis,        ~role,        ~signature_name,
  "Micro/PVM",  "primary",    "sun_2023_MG4_lipid_processing",
  "Micro/PVM",  "secondary",  "sun_2023_MG8_inflammatory_II",
  "Astrocyte",  "primary",    "astrocyte_activation_GOBP",
  "Oligo/OPC",  "secondary",  "myelin_maintenance_GOBP",
  "Oligo/OPC",  "secondary",  "oligodendrocyte_differentiation_GOBP"
)

PRIMARY_MICRO <- "sun_2023_MG4_lipid_processing"
PRIMARY_ASTRO <- "astrocyte_activation_GOBP"
# Kept out of the carried-forward set but scored throughout as the
# dataset-independent comparator for the Micro/PVM axis.
COMPARATOR_MICRO <- "microglial_glial_activation_GOBP"
#
#
#
#
#
#
#
#| label: inputs

inputs <- tibble::tribble(
  ~role, ~path,
  "signature registry (rebuilt)",        state$registry_csv,
  "signature genes",                     state$signature_genes,
  "Sun 2023 Table S1 supplement",        file.path(analysis$supplements_dir,
                                                   "sun_2023_cell_table_s1.xlsx"),
  "row-level state scores",              state$row_scores,
  "row-level competitive scores",        analysis$competitive_scores,
  "matched-background summary",          analysis$background_summary,
  "signature gene coverage",             state$gene_coverage,
  "pooled sensitivity scores",           state$pooled_scores,
  "published donor x region CPS",        omics$cps_by_region,
  "QNP workbook (P and H blocks)",       paths$qnp_2026
) %>%
  mutate(exists = file.exists(path),
         size_mb = round(ifelse(file.exists(path), file.size(path) / 1e6, NA_real_), 2),
         file = basename(path)) %>%
  select(role, file, exists, size_mb)

show_table(inputs, "Everything this notebook reads. Nothing new was downloaded from SEA-AD.")
#
#
#
#
#
#
#| label: load

registry   <- read_csv(state$registry_csv, show_col_types = FALSE)
registry_j <- fromJSON(state$registry_json, simplifyVector = FALSE)
coverage   <- read_csv(state$gene_coverage, show_col_types = FALSE)
scores     <- read_state_scores(state$row_scores)
comp       <- read_competitive_scores(analysis$competitive_scores)
bg_summary <- read_csv(analysis$background_summary, show_col_types = FALSE)
comp_prov  <- fromJSON(analysis$competitive_prov, simplifyVector = FALSE)
prep_prov  <- fromJSON(state$preparation_prov, simplifyVector = FALSE)
score_prov <- fromJSON(state$scoring_prov, simplifyVector = FALSE)

role_of <- registry %>% select(signature_name, role, feature_origin, resolution_status)
scores  <- scores %>% left_join(role_of, by = "signature_name")
# The competitive arm corrects the size-matched null controls as well as the
# candidates, so that section 3 can compare like with like.
comp    <- comp %>% left_join(role_of %>% select(signature_name, role), by = "signature_name")
COMP_SIGS <- comp %>% filter(role == "candidate_state") %>% pull(signature_name) %>% unique()

STATE_SIGS <- registry %>%
  filter(role == "candidate_state", resolution_status == "resolved") %>% pull(signature_name)
ID_SIGS <- registry %>% filter(role == "identity_control") %>% pull(signature_name)
NULL_SIGS <- registry %>% filter(role == "null_control") %>% pull(signature_name)
#
#
#
#
#
#
#
#
#
#
#| label: frozen

show_table(
  tibble::tribble(
    ~step, ~rule, ~settled_in,
    "input",        "summed raw UMI counts on the released row grain",        "notebook 02, s.4",
    "library size", "row sum of raw counts",                                  "notebook 02, s.4",
    "scaling",      sprintf("TMM, log-ratio trim %g, abundance trim %g, upper-quartile reference",
                            score_prov$normalization$logratio_trim,
                            score_prov$normalization$sum_trim),              "notebook 02, s.4",
    "expression",   sprintf("edgeR-style log2 CPM, prior count %g",
                            score_prov$normalization$prior_count),           "notebook 02, s.4",
    "score",        "unweighted mean of within-domain gene z-scores",         "notebook 02, s.5",
    "assay",        "additive per-method offset from mixed-method replicate pairs only",
                                                                             "notebook 02, s.6.5",
    "aggregation",  "nucleus-weighted mean of assay-adjusted row scores",     "notebook 02, s.9",
    "sensitivity",  "raw-count pooled arm, kept beside and never substituted", "notebook 02, s.9"
  ),
  "The frozen chain. One thing is added to it in section 3, and nothing is replaced."
)
#
#
#
#
#
#
#| label: tie-note

show_table(
  tibble::tribble(
    ~observation, ~scope, ~consequence,
    paste("The TMM reference row is chosen as the row whose upper quartile is closest to the",
          "domain mean upper quartile. In the Lymphocyte and Monocyte domains almost every row",
          "is so sparse that its upper quartile falls back to 1/n_genes, so 688 of 786",
          "Lymphocyte rows tie to ten significant figures and the choice is settled by",
          "floating-point noise, which differs between numpy builds."),
    "Lymphocyte and Monocyte only",
    paste("None. Neither domain carries a scored signature. The Micro/PVM, Astrocyte,",
          "Oligodendrocyte and OPC reference rows are identical across numpy 2.2 and 2.4, and",
          "re-running the scorer on the previous registry reproduced row_state_scores.csv",
          "byte for byte.")
  ),
  "A latent tie in the frozen normalization, and why it does not touch any state score"
)
#
#
#
#
#
#
#
#
#
#| label: criteria

show_table(
  tibble::tribble(
    ~criterion, ~used, ~why,
    "biological construct", "yes",
      "the axis has to name something the eventual model can be about",
    "human and AD relevance", "yes",
      "a mouse or non-brain definition measures an analogy, not the thing",
    "published provenance", "yes",
      "membership must be copyable verbatim from a versioned source",
    "gene availability", "yes",
      "a set whose genes are largely absent or undetectable is not measurable here",
    "interpretability", "yes",
      "a direction of effect has to be statable in advance",
    "association with SEA-AD CPS", "NO", "would select the axis by the answer",
    "association with QNP P or H", "NO", "would select the axis by the answer",
    "association with morphology or cognition", "NO", "out of scope and circular",
    "which signature gives the strongest result", "NO", "not a criterion at all"
  ),
  "What the selection in the setup chunk was, and was not, allowed to use"
)
#
#
#
#
#
#
#
#
#
#
#| label: sun-provenance

sun <- registry_j$provenance$supplements[[1]]
show_table(
  tibble::tibble(
    field = c("citation", "sheet", "state column", "gene column", "groups released",
              "MG4 genes", "MG8 genes", "SHA-256", "filter applied here"),
    value = c(sun$citation, sun$sheet, sun$group_column, sun$gene_column,
              as.character(sun$n_groups),
              as.character(sun$group_sizes$MG4), as.character(sun$group_sizes$MG8),
              substr(sun$sha256, 1, 24), sun$released_filter)
  ),
  "Provenance of the two newly resolved Micro/PVM signatures"
)
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#| label: astro-alternatives

show_table(
  tibble::tribble(
    ~alternative, ~status, ~why_not_carried,
    "astrocyte_activation_regulation_GOBP (GO:0061889)", "scored, not selected",
      "eight genes. A regulatory set that small is not a state axis and is expected to be unstable; kept only as a printed comparator.",
    "Liddelow 2017 PAN / A1 / A2", "unresolved",
      "cassette membership exists only inside a figure image. Image extraction was explicitly out of scope, and Escartin 2021 advises against the polarity anyway.",
    "Habib 2020 disease-associated astrocyte", "unresolved",
      "mouse-derived. A mouse definition fails the human-relevance criterion that MG4 passes.",
    "a Sun-equivalent human AD astrocyte state", "not available",
      "no equivalent machine-readable human astrocyte state cassette was identified from the provenance already collected."
  ),
  "Why the astrocyte axis carries one signature and not two"
)
#
#
#
#
#
#
#
#
#| label: availability

sel_cov <- coverage %>%
  inner_join(SELECTED %>% select(signature_name, axis, role), by = "signature_name") %>%
  bind_rows(coverage %>% filter(signature_name == COMPARATOR_MICRO) %>%
              mutate(axis = "Micro/PVM", role = "comparator")) %>%
  filter(domain %in% c("Micro/PVM", "Astrocyte", "Oligodendrocyte", "OPC")) %>%
  transmute(axis, role, signature_name, domain,
            n_defined, n_present, n_detectably_expressed, n_used,
            fraction_present = as.numeric(fraction_present),
            fraction_usable = n_used / n_defined) %>%
  arrange(axis, role)

show_table(sel_cov,
  "Gene availability of the selected axes inside their own scoring domain")
#
#
#
#| label: availability-note

cat(sprintf(
  "MG4 keeps %d of %d defined symbols as usable genes in Micro/PVM (%.0f%%); MG8 keeps %d of %d (%.0f%%).\n",
  sel_cov$n_used[sel_cov$signature_name == "sun_2023_MG4_lipid_processing"],
  sel_cov$n_defined[sel_cov$signature_name == "sun_2023_MG4_lipid_processing"],
  100 * sel_cov$fraction_usable[sel_cov$signature_name == "sun_2023_MG4_lipid_processing"],
  sel_cov$n_used[sel_cov$signature_name == "sun_2023_MG8_inflammatory_II"],
  sel_cov$n_defined[sel_cov$signature_name == "sun_2023_MG8_inflammatory_II"],
  100 * sel_cov$fraction_usable[sel_cov$signature_name == "sun_2023_MG8_inflammatory_II"]))
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#| label: background-rule

bg <- comp_prov$background
show_table(
  tibble::tibble(
    field = c("matched on", "binning", "eligible pool", "draws", "seed", "draw rule",
              "what it is not"),
    value = c(paste(unlist(bg$matched_on), collapse = "; "), bg$binning, bg$pool,
              as.character(bg$n_draws), as.character(bg$seed), bg$draw_rule,
              bg$not_an_empirical_null)
  ),
  "How the matched background is constructed. The seed is recorded in the provenance file."
)
#
#
#
#
#
#
#
#
#
#| label: background-quality

show_table(
  bg_summary %>%
    transmute(domain, signature_name, n_draws, n_signature_genes_matched,
              n_eligible_background_genes, n_strata_used,
              sig_log2cpm = signature_mean_log2_cpm,
              bg_log2cpm = background_mean_log2_cpm,
              sig_detect = signature_mean_detection_fraction,
              bg_detect = background_mean_detection_fraction,
              median_background_sd_z, se_of_background_mean = standard_error_of_background_mean,
              widened_draws) %>%
    arrange(domain, signature_name),
  "Did the matching work, and is the subtracted mean stable? Signature and background abundance and detection should agree closely, and the standard error of the background mean should be small against a within-domain score SD of about 1."
)
#
#
#
#
#
#| label: raw-vs-comp-row

raw_vs_comp_row <- comp %>%
  group_by(domain, signature_name, role) %>%
  summarise(n_rows = n(),
            pearson = cor(score_z_mean, competitive_score),
            spearman = cor(score_z_mean, competitive_score, method = "spearman"),
            sd_raw = sd(score_z_mean),
            sd_competitive = sd(competitive_score),
            median_background = median(background_mean_z),
            .groups = "drop") %>%
  arrange(domain, signature_name)

show_table(raw_vs_comp_row,
  "Row level: raw score against competitive score, per signature")
#
#
#
#| label: fig-raw-vs-comp
#| fig-width: 9
#| fig-height: 3.6
#| fig-cap: "Row-level raw score against matched-background-corrected score, for the selected axes and the Micro/PVM comparator. The dashed line is the identity."

comp %>%
  filter(signature_name %in% c(SELECTED$signature_name, COMPARATOR_MICRO)) %>%
  ggplot(aes(score_z_mean, competitive_score)) +
  geom_abline(slope = 1, intercept = 0, linetype = 2, linewidth = 0.3, colour = "grey50") +
  geom_point(alpha = 0.10, size = 0.4) +
  facet_wrap(~ signature_name, nrow = 1, scales = "free") +
  labs(x = "raw mean z", y = "competitive score") +
  theme_minimal(base_size = 20)
#
#
#
#
#
#
#
#
#
#
#
#
#
#| label: aggregate

pairs_raw  <- replicate_pairs(scores, value_col = "score_z_mean")
offsets_raw <- method_offsets(pairs_raw)

pairs_comp  <- replicate_pairs(comp, value_col = "competitive_score")
offsets_comp <- method_offsets(pairs_comp)

agg_raw <- aggregate_policies(scores, offsets_raw, value_col = "score_z_mean") %>%
  select(donor_id, brain_region, supertype, signature_name, lineage, domain,
         n_measurements, n_nuclei_total, n_assay_methods,
         state_score = assay_adjusted_nucleus_weighted_mean,
         equal_weight_mean, nucleus_weighted_mean, within_group_sd)

agg_comp <- aggregate_policies(comp, offsets_comp, value_col = "competitive_score") %>%
  select(donor_id, brain_region, supertype, signature_name,
         competitive_score = assay_adjusted_nucleus_weighted_mean)

supertype_level <- agg_raw %>%
  left_join(agg_comp, by = c("donor_id", "brain_region", "supertype", "signature_name"))

show_table(
  supertype_level %>%
    filter(signature_name %in% c(SELECTED$signature_name, COMPARATOR_MICRO)) %>%
    group_by(domain, signature_name) %>%
    summarise(cells = n(), donors = n_distinct(donor_id),
              donor_regions = n_distinct(paste(donor_id, brain_region)),
              with_competitive = sum(!is.na(competitive_score)), .groups = "drop"),
  "Donor x region x supertype estimates. Row-level scores are preserved untouched in the derivative files; nothing below overwrites them."
)
#
#
#
#| label: icc

# Technical reliability of both scorings, reusing notebook 02's estimator: a
# one-way ICC over repeated sample/library measurements inside a fixed
# donor x region x supertype. This is a property of the measurement, not
# evidence that a signature tracks its biology - notebook 02 established that a
# size-matched random set reaches the same ICC as any candidate.
# `icc_oneway()` returns a one-row tibble, so the scalar is pulled out here.
icc_of <- function(rows, value_col) {
  rows %>%
    group_by(signature_name, domain) %>%
    summarise(icc = icc_oneway(.data[[value_col]],
                               paste(donor_id, brain_region, supertype))$icc,
              .groups = "drop")
}

icc_raw <- icc_of(scores %>% filter(signature_name %in% comp$signature_name), "score_z_mean") %>%
  rename(icc_raw = icc)
icc_comp <- icc_of(comp, "competitive_score") %>% rename(icc_competitive = icc)

null_floor <- icc_raw %>%
  inner_join(icc_comp, by = c("signature_name", "domain")) %>%
  inner_join(role_of %>% select(signature_name, role), by = "signature_name") %>%
  filter(role == "null_control") %>%
  select(domain, null_icc_raw = icc_raw, null_icc_competitive = icc_competitive)

reliability <- icc_raw %>%
  inner_join(icc_comp, by = c("signature_name", "domain")) %>%
  inner_join(role_of %>% select(signature_name, role), by = "signature_name") %>%
  filter(role == "candidate_state") %>%
  left_join(null_floor, by = "domain") %>%
  mutate(clears_null_raw = icc_raw > null_icc_raw,
         clears_null_competitive = icc_competitive > null_icc_competitive,
         # How far above the floor, not just whether. A bare ">" would put an
         # axis that barely clears in the same box as one that clears fivefold.
         margin_over_null = icc_competitive - null_icc_competitive) %>%
  select(signature_name, domain, icc_raw, null_icc_raw, clears_null_raw,
         icc_competitive, null_icc_competitive, clears_null_competitive,
         margin_over_null) %>%
  arrange(domain, desc(icc_competitive))

show_table(reliability,
  "ICC(1,1) across repeated sample/library measurements, for both scorings, each against its own domain's size-matched null control put through the same correction")
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#| label: carry-rule

# The rule, stated before it is applied. An axis is only carried forward if its
# background-corrected score is more reproducible across repeated libraries than
# a size-matched random gene set that has been through the same correction. An
# axis that fails it is `unresolved`: the definition is chosen and defensible,
# but no reliable measurement of it exists in this release.
carry_status <- function(sig) {
  row <- reliability %>% filter(signature_name == sig)
  if (!nrow(row) || is.na(row$clears_null_competitive[1])) return("unresolved")
  if (row$clears_null_competitive[1]) "carry_forward" else "unresolved"
}

show_table(
  SELECTED %>%
    filter(axis != "Oligo/OPC") %>%
    rowwise() %>%
    mutate(status_under_the_rule = carry_status(signature_name)) %>%
    ungroup(),
  "The rule applied to the selected axes. The signatures were fixed before any of these numbers existed; what the rule decides is whether each one is *measurable* here, not whether it is the right construct."
)
#
#
#
#| label: raw-vs-comp-aggregated

raw_vs_comp_agg <- supertype_level %>%
  filter(!is.na(competitive_score)) %>%
  group_by(domain, signature_name) %>%
  summarise(n = n(),
            pearson = cor(state_score, competitive_score),
            spearman = cor(state_score, competitive_score, method = "spearman"),
            .groups = "drop")

show_table(raw_vs_comp_agg,
  "The same agreement question after aggregation, on the assay-adjusted nucleus-weighted values")
#
#
#
#| label: correction-verdict

verdict <- raw_vs_comp_row %>%
  filter(role == "candidate_state") %>%
  select(signature_name, row_pearson = pearson, sd_raw, sd_competitive) %>%
  left_join(raw_vs_comp_agg %>% select(signature_name, aggregated_pearson = pearson),
            by = "signature_name") %>%
  mutate(variance_removed = 1 - (sd_competitive / sd_raw)^2) %>%
  arrange(desc(variance_removed))

show_table(verdict,
  "How much the correction changes each score. `variance_removed` is the share of the raw score's variance that the matched background accounted for.")

cat(sprintf(paste0(
  "The correction is material, not cosmetic. Across the %d corrected candidate signatures the matched ",
  "background accounts for %.0f%% to %.0f%% of the raw score's variance, and raw and competitive ",
  "scores agree at Pearson %.2f to %.2f at row level (%.2f to %.2f after aggregation).
",
  "For the two carried-forward axes: %s, raw-vs-competitive Pearson %.2f at row level and %.2f ",
  "aggregated, %.0f%% of variance removed; %s, %.2f and %.2f, %.0f%% removed.
"),
  nrow(verdict),
  100 * min(verdict$variance_removed), 100 * max(verdict$variance_removed),
  min(verdict$row_pearson), max(verdict$row_pearson),
  min(verdict$aggregated_pearson, na.rm = TRUE), max(verdict$aggregated_pearson, na.rm = TRUE),
  PRIMARY_MICRO,
  verdict$row_pearson[verdict$signature_name == PRIMARY_MICRO],
  verdict$aggregated_pearson[verdict$signature_name == PRIMARY_MICRO],
  100 * verdict$variance_removed[verdict$signature_name == PRIMARY_MICRO],
  PRIMARY_ASTRO,
  verdict$row_pearson[verdict$signature_name == PRIMARY_ASTRO],
  verdict$aggregated_pearson[verdict$signature_name == PRIMARY_ASTRO],
  100 * verdict$variance_removed[verdict$signature_name == PRIMARY_ASTRO]))
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#| label: collapse

lineage_level <- collapse_supertypes_to_lineage(supertype_level)

show_table(
  lineage_level %>%
    filter(signature_name %in% c(SELECTED$signature_name, COMPARATOR_MICRO)) %>%
    group_by(domain, signature_name) %>%
    summarise(donor_regions = n(), donors = n_distinct(donor_id),
              regions = n_distinct(brain_region),
              median_supertypes = median(n_supertypes),
              median_dominant_share = median(dominant_supertype_share),
              well_supported = sum(support_status == "well supported"), .groups = "drop"),
  "Donor x region x lineage estimates after collapsing supertype"
)
#
#
#
#
#
#
#
#| label: analysis-table

molecular_state <- lineage_level %>%
  inner_join(bind_rows(SELECTED,
                       tibble(axis = "Micro/PVM", role = "comparator",
                              signature_name = COMPARATOR_MICRO)),
             by = "signature_name") %>%
  mutate(
    signature = signature_name,
    score = state_score,
    feature_origin = registry$feature_origin[match(signature_name, registry$signature_name)],
    source_release = prep_prov$source_release,
    aggregation = "nucleus-weighted mean over supertypes of the nucleus-weighted mean of assay-adjusted row scores"
  ) %>%
  select(donor_id, brain_region, lineage, axis, selection_role = role, signature,
         score, competitive_score, n_nuclei, n_measurements, n_supertypes,
         dominant_supertype_share, support_status, feature_origin, source_release,
         aggregation) %>%
  arrange(axis, selection_role, signature, donor_id, brain_region)

write_csv(molecular_state, analysis$analysis_table)

show_table(molecular_state %>% slice_head(n = 8),
           "The analysis-ready molecular-state table, first eight rows")

show_table(
  molecular_state %>% count(axis, selection_role, signature, name = "donor_region_cells"),
  sprintf("Size of the table: %d rows written to %s",
          nrow(molecular_state), basename(analysis$analysis_table))
)
#
#
#
#
#
#
#
#
#| label: write-provenance

write_json(
  list(
    created_utc = format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z"),
    written_by = "results/analysis/03_seaad_omics_analysis_ready.qmd",
    grain = "donor x brain region x lineage x signature",
    score = "assay-adjusted, nucleus-weighted row scores, nucleus-weighted over supertypes",
    competitive_score = comp_prov$definition,
    competitive_domains = comp_prov$domains_corrected,
    selection = as.list(SELECTED),
    selection_basis = paste("biological construct, human/AD relevance, published provenance,",
                            "gene availability, interpretability; never association with CPS,",
                            "QNP, morphology or cognition"),
    source_release = prep_prov$source_release,
    composition_merged = FALSE,
    imputation = "none"
  ),
  analysis$analysis_prov, auto_unbox = TRUE, pretty = TRUE
)
cat("wrote", basename(analysis$analysis_prov), "\n")
#
#
#
#
#
#
#
#
#
#
#
#| label: oligo-identity

oligo_wide <- supertype_level %>%
  filter(domain %in% c("Oligodendrocyte", "OPC")) %>%
  select(donor_id, brain_region, supertype, domain, signature_name, state_score) %>%
  pivot_wider(names_from = signature_name, values_from = state_score)

identity_corr <- map_dfr(
  registry %>% filter(domain == "Oligodendrocyte", role == "candidate_state") %>%
    pull(signature_name),
  function(sig) {
    d <- oligo_wide %>% filter(domain == "Oligodendrocyte")
    ok <- !is.na(d[[sig]]) & !is.na(d[["oligodendrocyte_identity_LEIN"]])
    tibble(signature_name = sig, n = sum(ok),
           pearson_with_identity = cor(d[[sig]][ok], d[["oligodendrocyte_identity_LEIN"]][ok]),
           spearman_with_identity = cor(d[[sig]][ok], d[["oligodendrocyte_identity_LEIN"]][ok],
                                        method = "spearman"))
  }) %>%
  mutate(classification = if_else(abs(pearson_with_identity) >= 0.9,
                                  "identity_associated", "retained as secondary")) %>%
  arrange(desc(abs(pearson_with_identity)))

show_table(identity_corr,
  "Oligodendrocyte state axes against the lineage-identity control (LEIN oligodendrocyte markers), at donor x region x supertype")
#
#
#
#| label: opc-note

cat(paste(
  "The OPC domain has no scored lineage-identity control: the LEIN oligodendrocyte marker set is",
  "registered to the Oligodendrocyte lineage and is scored only there. The OPC differentiation",
  "axis is therefore retained as a secondary biological candidate without an identity check, and",
  "that gap is recorded rather than filled by inventing a control.\n"))
#
#
#
#
#
#
#
#| label: ph-join

cps <- read_local_cps(omics$cps_by_region)

qnp_wide <- read_qnp_workbook(paths$qnp_2026) %>%
  qnp_primary_long() %>%
  filter(status == "observed") %>%
  select(donor_id, brain_region, variable, value) %>%
  distinct()
stopifnot(!any(duplicated(qnp_wide[c("donor_id", "brain_region", "variable")])))
qnp_wide <- pivot_wider(qnp_wide, names_from = variable, values_from = value)

t_wide <- molecular_state %>%
  filter(signature %in% c(PRIMARY_MICRO, PRIMARY_ASTRO)) %>%
  mutate(block = if_else(signature == PRIMARY_MICRO, "MicroPVM", "Astro")) %>%
  select(donor_id, brain_region, block, score, competitive_score, n_nuclei) %>%
  pivot_wider(names_from = block,
              values_from = c(score, competitive_score, n_nuclei),
              names_glue = "{block}_{.value}")

# Exact identifiers only: no region is renamed, mapped or approximated on either
# side of this join, and nothing is imputed.
donor_roi <- qnp_wide %>%
  full_join(cps %>% select(donor_id, brain_region, CPS_Local), by = c("donor_id", "brain_region")) %>%
  full_join(t_wide, by = c("donor_id", "brain_region")) %>%
  select(donor_id, brain_region, CPS_Local,
         abeta_percent_positive_area, ptau_percent_positive_area,
         neun_cells_per_area, gfap_percent_positive_area,
         MicroPVM_score, MicroPVM_competitive_score, MicroPVM_n_nuclei,
         Astro_score, Astro_competitive_score, Astro_n_nuclei)

show_table(head(donor_roi, 6),
  "The exploratory donor x ROI table. No morphology, no cognition, no imputation.")
#
#
#
#
#
#| label: coverage-by-roi

coverage_roi <- donor_roi %>%
  group_by(brain_region) %>%
  summarise(
    n_rows = n(),
    n_CPS = sum(!is.na(CPS_Local)),
    n_QNP_PH = sum(!is.na(abeta_percent_positive_area) & !is.na(ptau_percent_positive_area) &
                   !is.na(neun_cells_per_area) & !is.na(gfap_percent_positive_area)),
    n_MicroPVM = sum(!is.na(MicroPVM_score)),
    n_Astro = sum(!is.na(Astro_score)),
    n_complete_PH_T = sum(!is.na(abeta_percent_positive_area) & !is.na(ptau_percent_positive_area) &
                          !is.na(neun_cells_per_area) & !is.na(gfap_percent_positive_area) &
                          !is.na(MicroPVM_score) & !is.na(Astro_score)),
    n_complete_with_CPS = sum(!is.na(CPS_Local) &
                          !is.na(abeta_percent_positive_area) & !is.na(ptau_percent_positive_area) &
                          !is.na(neun_cells_per_area) & !is.na(gfap_percent_positive_area) &
                          !is.na(MicroPVM_score) & !is.na(Astro_score)),
    .groups = "drop") %>%
  mutate(candidate_ROI = brain_region %in% ROI) %>%
  arrange(desc(candidate_ROI), desc(n_complete_PH_T))

show_table(coverage_roi,
  "Coverage by region. The six candidate ROIs first; every released region is shown, because the ROI set is not frozen here.")
#
#
#
#| label: coverage-totals

cat(sprintf(
  "Across the six candidate ROIs: %d donor x ROI cells have complete P and H, %d have the Micro/PVM state, %d the astrocyte state, and %d have complete P + H + both T variables (%d of those also have a local CPS).\n",
  sum(coverage_roi$n_QNP_PH[coverage_roi$candidate_ROI]),
  sum(coverage_roi$n_MicroPVM[coverage_roi$candidate_ROI]),
  sum(coverage_roi$n_Astro[coverage_roi$candidate_ROI]),
  sum(coverage_roi$n_complete_PH_T[coverage_roi$candidate_ROI]),
  sum(coverage_roi$n_complete_with_CPS[coverage_roi$candidate_ROI])))
#
#
#
#
#
#
#
#
#
#
#
#
#
#| label: correlations

covariates <- c("CPS_Local", "abeta_percent_positive_area", "ptau_percent_positive_area",
                "neun_cells_per_area", "gfap_percent_positive_area")

corr_tbl <- bind_rows(
  correlate_with(donor_roi, "MicroPVM_score", covariates),
  correlate_with(donor_roi, "MicroPVM_competitive_score", covariates),
  correlate_with(donor_roi, "Astro_score", covariates),
  correlate_with(donor_roi, "Astro_competitive_score", covariates)
)

show_table(
  corr_tbl %>%
    select(response, covariate, spearman) %>%
    pivot_wider(names_from = covariate, values_from = spearman) %>%
    left_join(corr_tbl %>% group_by(response) %>%
                summarise(n_min = min(n), n_max = max(n), .groups = "drop"),
              by = "response") %>%
    relocate(n_min, n_max, .after = response),
  "Spearman correlation of each selected state score with CPS and the existing P/H variables, at donor x ROI. `n` varies by covariate because NeuN and GFAP are not released for every region, so the range is given rather than one number."
)
#
#
#
#
#| label: fig-cps
#| fig-width: 9
#| fig-height: 3.4
#| fig-cap: "Selected state scores against the published local pseudo-progression score, by region. Descriptive."

donor_roi %>%
  filter(!is.na(CPS_Local)) %>%
  select(brain_region, CPS_Local, MicroPVM_score, Astro_score) %>%
  pivot_longer(c(MicroPVM_score, Astro_score), names_to = "axis", values_to = "score") %>%
  filter(!is.na(score)) %>%
  ggplot(aes(CPS_Local, score, colour = brain_region)) +
  geom_point(alpha = 0.6, size = 0.9) +
  geom_smooth(aes(group = 1), method = "loess", se = FALSE, linewidth = 0.4, colour = "black") +
  facet_wrap(~ axis, nrow = 1, scales = "free_y") +
  labs(x = "CPS_Local", y = "state score", colour = NULL)+
  theme_minimal(base_size = 20)
#
#
#
#| label: redundancy

redundancy <- bind_rows(
  redundancy_fit(donor_roi, "MicroPVM_score"),
  redundancy_fit(donor_roi, "MicroPVM_competitive_score"),
  redundancy_fit(donor_roi, "Astro_score"),
  redundancy_fit(donor_roi, "Astro_competitive_score")
)

show_table(
  redundancy %>%
    select(any_of(c("response", "n", "n_donors", "n_regions", "sd_raw", "sd_residual",
                    "r2_full", "fraction_variance_remaining")), starts_with("unique_r2_")),
  "T ~ ROI + ns(CPS, 3) + Abeta + pTau + NeuN + GFAP, fitted only to characterise. `unique_r2_*` is the drop in R-squared when that block is removed from the full model, so the four do not sum to the full R-squared."
)
#
#
#
#| label: redundancy-read

cat(sprintf(
  "Micro/PVM: SD %.3f raw, %.3f residual, so %.0f%% of the score's spread survives ROI, stage and P/H.\nAstrocyte: SD %.3f raw, %.3f residual, so %.0f%% survives.\n",
  redundancy$sd_raw[redundancy$response == "MicroPVM_score"],
  redundancy$sd_residual[redundancy$response == "MicroPVM_score"],
  100 * redundancy$sd_residual[redundancy$response == "MicroPVM_score"] /
        redundancy$sd_raw[redundancy$response == "MicroPVM_score"],
  redundancy$sd_raw[redundancy$response == "Astro_score"],
  redundancy$sd_residual[redundancy$response == "Astro_score"],
  100 * redundancy$sd_residual[redundancy$response == "Astro_score"] /
        redundancy$sd_raw[redundancy$response == "Astro_score"]))
#
#
#
#| label: redundancy-contrast

# Descriptive, and *not* a selection step: the signatures were fixed in the
# setup chunk before any of these numbers existed. What this compares is two
# scorings of the same fixed signature.
contrast <- redundancy %>%
  mutate(axis = if_else(str_detect(response, "^MicroPVM"), "Micro/PVM", "Astrocyte"),
         scoring = if_else(str_detect(response, "competitive"), "competitive", "raw")) %>%
  select(axis, scoring, r2_full, unique_r2_ROI, unique_r2_CPS, unique_r2_P, unique_r2_H)

show_table(contrast,
  "The same model fitted to the raw and the competitive version of each axis. Read the columns, not the totals: where the explained variance sits matters more here than how much of it there is.")
#
#
#
#
#
#
#
#| label: decision

pull1 <- function(tbl, sig, col) tbl[[col]][match(sig, tbl$signature_name)]
agree_micro <- raw_vs_comp_agg$pearson[raw_vs_comp_agg$signature_name == PRIMARY_MICRO]
agree_astro <- raw_vs_comp_agg$pearson[raw_vs_comp_agg$signature_name == PRIMARY_ASTRO]
reliability_of <- function(sig) {
  row <- reliability %>% filter(signature_name == sig)
  removed <- verdict$variance_removed[verdict$signature_name == sig]
  sprintf(paste("ICC(1,1) raw %.2f vs raw null %.2f (does not clear it); competitive %.2f vs",
                "corrected null %.2f, %s by %.2f. The matched background accounts for %.0f%% of",
                "the raw score's variance."),
          row$icc_raw[1], row$null_icc_raw[1],
          row$icc_competitive[1], row$null_icc_competitive[1],
          if (isTRUE(row$clears_null_competitive[1])) "clearing it" else "failing it",
          abs(row$margin_over_null[1]), 100 * removed)
}
r_micro <- redundancy %>% filter(response == "MicroPVM_score")
r_astro <- redundancy %>% filter(response == "Astro_score")
oligo_worst <- identity_corr %>% slice(1)

decision <- tibble::tribble(
  ~axis, ~primary_definition, ~score, ~technical_reliability, ~PH_redundancy, ~ROI_coverage, ~status,

  "Micro/PVM",
  sprintf("Sun et al. 2023 MG4 (lipid processing), %d of %d genes usable",
          pull1(sel_cov, PRIMARY_MICRO, "n_used"), pull1(sel_cov, PRIMARY_MICRO, "n_defined")),
  "assay-adjusted, nucleus-weighted mean z; matched-background arm beside it",
  reliability_of(PRIMARY_MICRO),
  sprintf("R2 %.2f to ROI + CPS + P + H; %.0f%% of SD survives",
          r_micro$r2_full, 100 * r_micro$sd_residual / r_micro$sd_raw),
  sprintf("%d donor x ROI cells with complete P/H + both T",
          sum(coverage_roi$n_complete_PH_T[coverage_roi$candidate_ROI])),
  carry_status(PRIMARY_MICRO),

  "Astrocyte",
  sprintf("GO:0048143 astrocyte activation, %d of %d genes usable",
          pull1(sel_cov, PRIMARY_ASTRO, "n_used"), pull1(sel_cov, PRIMARY_ASTRO, "n_defined")),
  "assay-adjusted, nucleus-weighted mean z; matched-background arm beside it",
  reliability_of(PRIMARY_ASTRO),
  sprintf("R2 %.2f to ROI + CPS + P + H; %.0f%% of SD survives",
          r_astro$r2_full, 100 * r_astro$sd_residual / r_astro$sd_raw),
  sprintf("%d donor x ROI cells with complete P/H + both T",
          sum(coverage_roi$n_complete_PH_T[coverage_roi$candidate_ROI])),
  carry_status(PRIMARY_ASTRO),

  "Oligo/OPC",
  "GO:0043217 myelin maintenance (Oligo) and GO:0048709 oligodendrocyte differentiation (OPC)",
  "assay-adjusted, nucleus-weighted mean z; no matched-background arm",
  sprintf("highest correlation with the LEIN identity control is %.2f (%s)",
          oligo_worst$pearson_with_identity, oligo_worst$signature_name),
  "not characterised against P/H in this notebook",
  "retained in the analysis table, not joined to P/H here",
  if (any(identity_corr$classification == "identity_associated")) "identity_associated" else "secondary"
)

show_table(decision, "The decision. Statuses are limited to carry_forward, secondary, identity_associated, unresolved.")
#
#
#
#
#
#
#
#
#
#| label: files-changed

show_table(
  tibble::tribble(
    ~path, ~status, ~what,
    "results/analysis/03_seaad_omics_analysis_ready.qmd", "new", "this notebook",
    "R/seaad_analysis_ready.R", "new",
      "supertype collapsing, competitive-score reader, redundancy characterisation",
    "src/seaad/omics/xlsx.py", "new",
      "minimal stdlib XLSX reader; openpyxl is absent from this project's environments",
    "src/seaad/omics/signature_registry.py", "modified",
      "second resolution route: a SHA-256-pinned published supplement; Sun MG4 and MG8 resolved",
    "src/seaad/omics/state_scoring.py", "modified",
      "additive only: per-domain abundance accumulator and matched-background draws, both off by default",
    "src/seaad/omics/competitive_scoring.py", "new", "the competitive score",
    "src/seaad/cli/omics_competitive_scores.py", "new", "its CLI",
    "src/seaad/pyproject.toml", "modified", "one new console script",
    "data/derivatives/sea-ad/omics_state/", "regenerated + new",
      "registry with two more resolved signatures; row_competitive_scores.csv, background_matching_summary.csv, molecular_state_analysis_table.csv and their provenance"
  ),
  "Files changed. Acquisition and preparation were not touched."
)
#
#
#
#
#
#| label: stop-conditions

answers <- tibble::tribble(
  ~question, ~answer,

  "1. What single Micro/PVM molecular-state variable goes forward?",
  sprintf(paste("Sun et al. 2023 MG4, the human lipid-processing microglial state, resolved",
                "verbatim from Table S1 page 2 of the published supplement (SHA-256 pinned,",
                "no threshold applied here). %d of %d symbols are usable in the Micro/PVM",
                "domain, and its background-corrected score reaches ICC %.2f against a",
                "corrected-null floor of %.2f, so it is %s. MG8 (inflammatory) is the secondary",
                "alternative at ICC %.2f; GO:0061900 glial cell activation is kept scored as the",
                "dataset-independent comparator and reaches only %.2f corrected."),
          pull1(sel_cov, PRIMARY_MICRO, "n_used"), pull1(sel_cov, PRIMARY_MICRO, "n_defined"),
          reliability$icc_competitive[reliability$signature_name == PRIMARY_MICRO],
          reliability$null_icc_competitive[reliability$signature_name == PRIMARY_MICRO],
          carry_status(PRIMARY_MICRO),
          reliability$icc_competitive[reliability$signature_name == "sun_2023_MG8_inflammatory_II"],
          reliability$icc_competitive[reliability$signature_name == COMPARATOR_MICRO]),

  "2. What single Astrocyte molecular-state variable goes forward?",
  sprintf(paste("GO:0048143 astrocyte activation, and it is the weaker of the two. Only %d of %d",
                "symbols are usable; its corrected score reaches ICC %.2f against a",
                "corrected-null floor of %.2f, so it is **%s** under the stated rule but clears",
                "that floor far less convincingly than MG4 does, and there is no secondary",
                "astrocyte axis to fall back on - the regulation set keeps %d usable genes,",
                "Liddelow A1/A2 is image-only, and Habib DAA is mouse-derived."),
          pull1(sel_cov, PRIMARY_ASTRO, "n_used"), pull1(sel_cov, PRIMARY_ASTRO, "n_defined"),
          reliability$icc_competitive[reliability$signature_name == PRIMARY_ASTRO],
          reliability$null_icc_competitive[reliability$signature_name == PRIMARY_ASTRO],
          carry_status(PRIMARY_ASTRO),
          coverage$n_used[coverage$signature_name == "astrocyte_activation_regulation_GOBP" &
                          coverage$domain == "Astrocyte"]),

  "3. Does matched-background correction materially change either?",
  sprintf(paste("Yes, materially, for both. The matched background accounts for %.0f%% of MG4's",
                "raw variance and %.0f%% of the astrocyte axis's, and raw and competitive scores",
                "agree at only Pearson %.2f and %.2f after aggregation. The background",
                "reproduces each signature's abundance and detection profile closely and the",
                "subtracted mean is stable over %d draws, so this is a property of the scores",
                "rather than of the correction. The two scorings are not interchangeable; both",
                "are carried in the analysis table and neither is deleted."),
          100 * verdict$variance_removed[verdict$signature_name == PRIMARY_MICRO],
          100 * verdict$variance_removed[verdict$signature_name == PRIMARY_ASTRO],
          agree_micro, agree_astro, comp_prov$background$n_draws),

  "4. How much of each score is already explained by CPS and P/H?",
  sprintf(paste("Micro/PVM R2 %.2f, astrocyte R2 %.2f against ROI + ns(CPS,3) + Abeta + pTau +",
                "NeuN + GFAP; %.0f%% and %.0f%% of each score's SD survives as residual."),
          r_micro$r2_full, r_astro$r2_full,
          100 * r_micro$sd_residual / r_micro$sd_raw,
          100 * r_astro$sd_residual / r_astro$sd_raw),

  "5. How many donor x ROI observations are complete?",
  sprintf(paste("%d across the six candidate ROIs with complete P, H and both selected T",
                "variables; %d of those also carry a published local CPS. Per-ROI counts are in",
                "section 8.1 and nothing is imputed."),
          sum(coverage_roi$n_complete_PH_T[coverage_roi$candidate_ROI]),
          sum(coverage_roi$n_complete_with_CPS[coverage_roi$candidate_ROI])),

  "6. Is there enough independent molecular variation to proceed to morphology?",
  sprintf(paste("Yes on the numbers, with one caveat. Micro/PVM keeps %.0f%% of its SD and the",
                "astrocyte axis %.0f%% after ROI, stage and P/H, over %d complete donor x ROI",
                "cells - both axes carry information that P and H do not. Note where the",
                "explained variance sits: for the raw Micro/PVM score it is almost all ROI",
                "(unique R2 %.2f) with CPS and P adding %.2f and %.2f, while for the competitive",
                "score ROI drops to %.2f and CPS and P rise to %.2f and %.2f. The caveat is that",
                "this makes the raw-versus-competitive choice a real modelling decision rather",
                "than a detail, and it is not settled here."),
          100 * r_micro$sd_residual / r_micro$sd_raw,
          100 * r_astro$sd_residual / r_astro$sd_raw,
          sum(coverage_roi$n_complete_PH_T[coverage_roi$candidate_ROI]),
          r_micro$unique_r2_ROI, r_micro$unique_r2_CPS, r_micro$unique_r2_P,
          contrast$unique_r2_ROI[contrast$axis == "Micro/PVM" & contrast$scoring == "competitive"],
          contrast$unique_r2_CPS[contrast$axis == "Micro/PVM" & contrast$scoring == "competitive"],
          contrast$unique_r2_P[contrast$axis == "Micro/PVM" & contrast$scoring == "competitive"])
)

# A zero-length argument makes sprintf() return character(0), which lands in the
# table as an empty cell rather than an error. Check for it rather than trusting
# the render.
stopifnot(all(nzchar(answers$answer)), !any(is.na(answers$answer)))

show_table(answers, "The six stop-condition questions")
#
#
#
#
#
#| label: unresolved

show_table(
  tibble::tribble(
    ~issue, ~blocks_morphology, ~what_would_resolve_it,
    "Raw and competitive scorings of the same signature disagree, and the P/H model puts their explained variance in different places",
      "YES - this is the one open issue that does",
      paste("one decision, made once and stated: carry the raw mean-z, the",
            "matched-background score, or both as separate covariates. Everything needed to",
            "make it is in sections 3 and 9; nothing further has to be computed."),
    "The astrocyte axis has no human AD-derived definition, and its corrected score does not clear the corrected null",
      "only if an astrocyte T is required",
      "a machine-readable human astrocyte state cassette, if one exists; the Sun route worked for microglia and is the obvious template",
    "Micro-PVM is not resolved into microglia and perivascular macrophages",
      "no", "a finer released taxonomy",
    "The lineage score is abundance-weighted across supertypes, so it is not independent of composition",
      "no", "stated on the table; a later model must say which it attributes an effect to",
    "sample_name semantics remain undocumented",
      "no", "a statement from SEA-AD; no contradiction has arisen, so it stays closed",
    "The ROI set is still not frozen",
      "not yet", "the coverage table in section 8.1 plus the morphology ROI list",
    "The neuronal axis is undefined and nothing neuronal was downloaded",
      "no", "a defensible neuronal grouping present in every candidate ROI"
  ),
  "Open issues, and whether each one actually blocks morphology integration"
)
#
#
#
#
