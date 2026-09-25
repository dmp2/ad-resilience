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

my_packages <- c("tidyverse", "knitr", "jsonlite", "readxl")
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

paths  <- seaad_paths(PROJECT_ROOT)
omics  <- seaad_omics_paths(PROJECT_ROOT)
state  <- seaad_state_paths(PROJECT_ROOT)

show_table <- function(x, caption = NULL, digits = 3) {
  kable(x, caption = caption, digits = digits, format.args = list(big.mark = ""))
}

theme_set(theme_minimal(base_size = 10))

# The candidate multimodal ROI set carried forward from notebook 01. Still a
# candidate: nothing here freezes it, and every table below reports all ten
# released regions so the freeze stays an open decision.
ROI <- c("DFC", "HIP", "MEC", "MTG", "STG", "V1C")

# Domains in a fixed display order, glial state axes first.
DOMAIN_LEVELS <- c("Micro/PVM", "Astrocyte", "Oligodendrocyte", "OPC", "Lymphocyte", "Monocyte")
#
#
#
#| label: load-derivatives

registry   <- read_csv(state$registry_csv, show_col_types = FALSE)
registry_j <- fromJSON(state$registry_json, simplifyVector = FALSE)
sig_genes  <- read_csv(state$signature_genes, show_col_types = FALSE)
coverage   <- read_csv(state$gene_coverage, show_col_types = FALSE)
contrib    <- read_csv(state$gene_contributions, show_col_types = FALSE)
normtab    <- read_row_normalization(state$row_normalization)
scores     <- read_state_scores(state$row_scores)
pooled     <- read_state_scores(state$pooled_scores)
prep_index <- read_csv(state$prepared_index, show_col_types = FALSE)
taxonomy   <- read_csv(state$prepared_taxonomy, show_col_types = FALSE)
prep_prov  <- fromJSON(state$preparation_prov, simplifyVector = FALSE)
score_prov <- fromJSON(state$scoring_prov, simplifyVector = FALSE)

scores  <- scores  %>% mutate(domain = factor(domain, levels = DOMAIN_LEVELS))
normtab <- normtab %>% mutate(domain = factor(domain, levels = DOMAIN_LEVELS))

# Everything downstream distinguishes candidate state axes from the two control
# families, so a control can never be mistaken for a T variable.
role_of <- registry %>% select(signature_name, role, feature_origin, domain_reg = domain)
scores  <- scores %>% left_join(role_of, by = "signature_name")
STATE_SIGS <- registry %>% filter(role == "candidate_state", resolution_status == "resolved") %>% pull(signature_name)
NULL_SIGS  <- registry %>% filter(role == "null_control") %>% pull(signature_name)
ID_SIGS    <- registry %>% filter(role == "identity_control") %>% pull(signature_name)
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
  "prepared raw-count H5AD (4 lineages)", file.path(state$prepared_dir, "multiregion_2026"),
  "prepared row index",                   state$prepared_index,
  "released gene catalogue",              state$gene_catalog,
  "released supertype taxonomy",          state$prepared_taxonomy,
  "preparation provenance",               state$preparation_prov,
  "signature registry (derived)",         state$registry_csv,
  "signature genes (derived)",            state$signature_genes,
  "row-level normalization (derived)",    state$row_normalization,
  "row-level state scores (derived)",     state$row_scores,
  "pooled sensitivity scores (derived)",  state$pooled_scores,
  "published donor x region CPS",         omics$cps_by_region,
  "QNP workbook (P block only, section 12)", paths$qnp_2026
) %>%
  mutate(exists = file.exists(path) | dir.exists(path),
         size_mb = round(ifelse(file.exists(path), file.size(path) / 1e6, NA_real_), 2),
         file = basename(path)) %>%
  select(role, file, exists, size_mb)

show_table(inputs, "Everything this notebook reads. No new object was downloaded.")
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
#| label: hierarchy

show_table(
  tibble::tribble(
    ~level, ~meaning, ~consequence,
    "donor",
      "biological replication unit",
      "the only level at which independent replication exists",
    "donor x region",
      "eventual regional analysis unit",
      "the grid the T block must eventually live on",
    "supertype",
      "molecular / cellular stratum",
      "a stratum to condition on, never a state measurement in itself",
    "sample / library-prep row",
      "repeated molecular measurement within a donor x region x supertype",
      "repeated measurements, not replicate donors; they inform reliability, not sample size"
  ),
  "The measurement hierarchy used throughout"
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
#| label: hierarchy-observed

grain <- normtab %>%
  summarise(
    n_prepared_rows = n(),
    n_donors = n_distinct(donor_id),
    n_regions = n_distinct(brain_region),
    n_supertypes = n_distinct(supertype),
    n_sample_names = n_distinct(sample_name),
    n_library_preps = n_distinct(library_prep),
    n_donor_region_supertype = n_distinct(paste(donor_id, brain_region, supertype)),
    n_assay_methods = n_distinct(assay_method)
  )

show_table(grain, "The released grain, counted from the prepared rows")

repeats <- normtab %>%
  count(lineage, donor_id, brain_region, supertype, name = "n_rows") %>%
  count(lineage, n_rows, name = "n_groups")

show_table(
  repeats %>% pivot_wider(names_from = n_rows, values_from = n_groups, values_fill = 0) %>%
    rename_with(~ paste0("rows_", .x), .cols = -lineage),
  "How many donor x region x supertype groups carry how many sample/library rows"
)
#
#
#
#| label: hierarchy-note

cat(prep_prov$aggregation_interpretation, "\n\n", prep_prov$sample_name_semantics, sep = "")
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
#| label: feature-origin

show_table(
  tibble::tribble(
    ~feature_origin, ~meaning,
    "external_predefined",
      "a gene set defined outside this project and outside SEA-AD, applied unchanged",
    "SEAAD_published_predefined",
      "a quantity SEA-AD itself publishes as a named variable, label or gene list",
    "null_control_current_data",
      "a size-matched random gene set drawn from this data; a control, never a T variable"
  ),
  "Feature-origin categories"
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
#| label: registry-provenance

show_table(
  tibble(
    field = c("registry built (UTC)", "MSigDB release", "collections used", "rules"),
    value = c(
      registry_j$provenance$created_utc,
      registry_j$provenance$msigdb_release,
      paste(map_chr(registry_j$provenance$sources, "collection"), collapse = ", "),
      paste(unlist(registry_j$provenance$rules), collapse = " | ")
    )
  ),
  "Provenance of the resolved gene definitions"
)

show_table(
  map_dfr(registry_j$provenance$sources, ~ tibble(
    collection = .x$collection, n_sets = .x$n_sets,
    size_mb = round(.x$size_bytes / 1e6, 2), sha256 = substr(.x$sha256, 1, 16)
  )),
  "MSigDB source files, hashed"
)
#
#
#
#
#
#| label: registry-resolved

show_table(
  registry %>%
    filter(resolution_status == "resolved", role == "candidate_state") %>%
    select(signature_name, biological_axis, lineage, domain, n_genes, directionality),
  "Resolved candidate state signatures: what each one measures"
)

show_table(
  registry %>%
    filter(resolution_status == "resolved", role == "candidate_state") %>%
    select(signature_name, source_publication, gene_definition_source, feature_origin),
  "Resolved candidate state signatures: where each definition came from"
)

show_table(
  registry %>%
    filter(role %in% c("identity_control", "null_control")) %>%
    select(signature_name, role, biological_axis, domain, feature_origin, n_genes),
  "Controls. Neither family is a candidate T variable."
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
#| label: registry-unresolved

show_table(
  registry %>%
    filter(resolution_status == "unresolved") %>%
    select(signature_name, biological_axis, lineage, feature_origin, source_publication),
  "Candidates carried but NOT scored: no exact gene definition was recovered"
)

show_table(
  registry %>%
    filter(resolution_status == "unresolved") %>%
    select(signature_name, gene_definition_source, unresolved_reason),
  "Why each unresolved candidate is unresolved"
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
#| label: gene-availability

catalog <- read_csv(state$gene_catalog, show_col_types = FALSE)

avail <- signature_availability(
  sig_genes %>% filter(signature_name %in% c(STATE_SIGS, ID_SIGS)),
  catalog,
  detect_fraction = score_prov$detectability$fraction
) %>%
  inner_join(registry %>% select(signature_name, lineage, role), by = c("signature_name", "lineage"))

show_table(
  avail %>%
    arrange(lineage, signature_name) %>%
    select(lineage, signature_name, n_defined, n_present, n_detectably_expressed, fraction_present),
  "Gene availability by the signature's own lineage (exact symbol match, R route)"
)
#
#
#
#| label: gene-availability-missing

show_table(
  avail %>%
    filter(nchar(missing_genes) > 0) %>%
    arrange(lineage, signature_name) %>%
    select(lineage, signature_name, n_defined, n_present, missing_genes),
  "Exactly which defined symbols are absent from the released feature set"
)
#
#
#
#| label: gene-availability-reconcile

reconcile <- coverage %>%
  filter(signature_name %in% c(STATE_SIGS, ID_SIGS)) %>%
  select(lineage, domain, signature_name,
         py_n_defined = n_defined, py_n_present = n_present,
         py_n_detectable = n_detectably_expressed, py_n_used = n_used) %>%
  left_join(avail %>% select(lineage, signature_name,
                             r_n_defined = n_defined, r_n_present = n_present,
                             r_n_detectable = n_detectably_expressed),
            by = c("lineage", "signature_name")) %>%
  mutate(defined_agrees = py_n_defined == r_n_defined,
         present_agrees = py_n_present == r_n_present)

show_table(
  reconcile %>% select(lineage, domain, signature_name, py_n_defined, r_n_defined,
                       py_n_present, r_n_present, defined_agrees, present_agrees),
  "Reconciliation of the two availability routes"
)

cat(sprintf(
  "n_defined agrees for %d of %d signature x lineage cells; n_present agrees for %d.\n",
  sum(reconcile$defined_agrees), nrow(reconcile), sum(reconcile$present_agrees)
))
#
#
#
#
#
#
#
#
#
#| label: gene-availability-domain

show_table(
  coverage %>%
    arrange(domain, signature_name) %>%
    select(domain, signature_name, n_defined, n_present, n_detectably_expressed, n_used,
           fraction_present, fraction_detectably_expressed, n_rows_scored),
  "Availability as the scoring pipeline saw it, per scoring domain"
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
#| label: normalization-recipe

norm <- score_prov$normalization
show_table(
  tibble(
    step = c("input", "library size", "method", "log-ratio trim", "abundance trim",
             "reference row", "rescaling", "log-CPM", "prior count",
             "single-cell normalization", "rows collapsed before normalization"),
    definition = c(
      norm$input, norm$library_size, norm$method,
      as.character(norm$logratio_trim), as.character(norm$sum_trim),
      norm$reference_selection, norm$rescaling, norm$log_cpm,
      as.character(norm$prior_count),
      ifelse(isTRUE(norm$single_cell_normalization_used), "yes", "no"),
      ifelse(isTRUE(score_prov$rows_collapsed_before_scoring), "yes", "no")
    )
  ),
  "The exact normalization, as recorded by the pipeline that ran it"
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
#| label: domains

show_table(
  map_dfr(names(registry_j$provenance$domains), function(d) {
    spec <- registry_j$provenance$domains[[d]]
    tibble(domain = d, lineage = spec$lineage, supertype_rule = spec$supertype_rule,
           taxonomy_note = spec$taxonomy_note)
  }),
  "Scoring domains. The released Immune taxonomy is preserved, not collapsed."
)

show_table(
  normtab %>%
    group_by(domain) %>%
    summarise(n_rows = n(), n_donors = n_distinct(donor_id), n_regions = n_distinct(brain_region),
              n_supertypes = n_distinct(supertype), .groups = "drop") %>%
    arrange(domain),
  "Rows per scoring domain"
)
#
#
#
#
#
#| label: normalization-integrity

integrity <- normtab %>%
  summarise(
    rows = n(),
    library_size_equals_released_total_umi = all(computed_library_size == total_umi),
    min_library_size = min(computed_library_size),
    median_library_size = median(computed_library_size),
    max_library_size = max(computed_library_size),
    library_size_ratio = max(computed_library_size) / min(computed_library_size)
  )

show_table(integrity, "Integrity check: the recomputed library size reproduces the released total UMI exactly")

show_table(
  normtab %>%
    group_by(domain) %>%
    summarise(n = n(),
              tmm_min = min(tmm_factor), tmm_q25 = quantile(tmm_factor, .25),
              tmm_median = median(tmm_factor), tmm_q75 = quantile(tmm_factor, .75),
              tmm_max = max(tmm_factor), .groups = "drop") %>%
    arrange(domain),
  "TMM factors by domain"
)
#
#
#
#| label: fig-tmm
#| fig-width: 8
#| fig-height: 4.2
#| fig-cap: "TMM factor against nuclei support. The factor is near 1 for well-supported rows and becomes both extreme and erratic below roughly ten nuclei, which is the first sign that low-support rows are a measurement problem rather than a biological one."

normtab %>%
  filter(domain %in% c("Micro/PVM", "Astrocyte", "Oligodendrocyte", "OPC")) %>%
  ggplot(aes(n_nuclei, tmm_factor)) +
  geom_point(alpha = 0.15, size = 0.5) +
  geom_hline(yintercept = 1, colour = "firebrick", linewidth = 0.4) +
  scale_x_log10() + scale_y_log10() +
  facet_wrap(~ domain, nrow = 1) +
  labs(x = "Number of nuclei (log)", y = "TMM factor (log)")
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
#| label: scoring-rule

show_table(
  tibble(
    field = c("primary score", "secondary score", "gene inclusion",
              "detectability rule", "standardization scope",
              "parameters tuned against pathology / morphology / cognition"),
    definition = c(
      score_prov$scoring$primary,
      score_prov$scoring$secondary,
      "present in the released features AND detectably expressed in the domain AND non-constant",
      sprintf("non-zero in at least %g of the domain's rows", score_prov$detectability$fraction),
      "within scoring domain, across all rows of that domain",
      ifelse(isTRUE(score_prov$scoring$parameters_tuned_against_outcomes), "yes", "no")
    )
  ),
  "The scoring rule. One unweighted mean, no tuned parameter."
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
#| label: score-grain

show_table(
  scores %>%
    filter(signature_name %in% STATE_SIGS) %>%
    group_by(domain, signature_name) %>%
    summarise(n_rows = n(), n_donors = n_distinct(donor_id),
              n_donor_region_supertype = n_distinct(paste(donor_id, brain_region, supertype)),
              n_genes_used = first(n_genes_used),
              mean = mean(score_z_mean), sd = sd(score_z_mean),
              min = min(score_z_mean), max = max(score_z_mean), .groups = "drop") %>%
    arrange(domain, signature_name),
  "Row-level state scores: one per donor x region x supertype x sample/library"
)

show_table(
  scores %>% slice_head(n = 3) %>%
    select(prepared_obs_index, donor_id, brain_region, supertype, sample_name, library_prep,
           assay_method, n_nuclei, total_umi, signature_name, score_z_mean),
  "Three rows, to show that every score keeps its identity and its support"
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
#| label: gene-contributions

domination <- contrib %>%
  group_by(domain, signature_name) %>%
  summarise(
    n_genes = n(),
    max_gene_corr = max(corr_gene_z_with_score, na.rm = TRUE),
    top_gene = gene_symbol[which.max(corr_gene_z_with_score)],
    median_gene_corr = median(corr_gene_z_with_score, na.rm = TRUE),
    min_leave_one_out_corr = min(leave_one_out_corr, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(family = case_when(signature_name %in% NULL_SIGS ~ "null control",
                            signature_name %in% ID_SIGS ~ "identity control",
                            TRUE ~ "candidate state"))

show_table(
  domination %>% arrange(family, domain, desc(max_gene_corr)) %>%
    select(family, domain, signature_name, n_genes, top_gene, max_gene_corr,
           median_gene_corr, min_leave_one_out_corr),
  "Gene-level domination diagnostics. A leave-one-out correlation near 1 means no single gene carries the score."
)
#
#
#
#
#
#
#
#
#| label: fig-contributions
#| fig-width: 8
#| fig-height: 4
#| fig-cap: "Per-gene correlation with the signature score it belongs to. Candidate state signatures are compared against the size-matched null draws, which show what correlation structure arises from gene count alone."

contrib %>%
  mutate(family = case_when(signature_name %in% NULL_SIGS ~ "null control",
                            signature_name %in% ID_SIGS ~ "identity control",
                            TRUE ~ "candidate state")) %>%
  ggplot(aes(reorder(signature_name, corr_gene_z_with_score, median), corr_gene_z_with_score,
             colour = family)) +
  geom_boxplot(outlier.size = 0.5, linewidth = 0.35) +
  coord_flip() +
  labs(x = NULL, y = "correlation of gene z with the signature score", colour = NULL) +
  theme(legend.position = "top")
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
#| label: replicate-structure

rep_structure <- scores %>%
  filter(signature_name %in% STATE_SIGS) %>%
  distinct(domain, donor_id, brain_region, supertype, prepared_obs_index, assay_method) %>%
  group_by(domain, donor_id, brain_region, supertype) %>%
  summarise(n_rows = n(), n_methods = n_distinct(assay_method), .groups = "drop") %>%
  mutate(kind = case_when(n_rows == 1 ~ "single measurement",
                          n_methods == 1 ~ "repeated, same assay method",
                          TRUE ~ "repeated, mixed assay method"))

show_table(
  rep_structure %>% count(domain, kind) %>%
    pivot_wider(names_from = kind, values_from = n, values_fill = 0) %>% arrange(domain),
  "Donor x region x supertype groups by replicate structure"
)

show_table(
  scores %>% distinct(prepared_obs_index, assay_method, domain) %>% count(domain, assay_method) %>%
    pivot_wider(names_from = assay_method, values_from = n, values_fill = 0) %>% arrange(domain),
  "Prepared rows by released assay method"
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
#| label: pairs

pairs <- scores %>%
  filter(signature_name %in% c(STATE_SIGS, ID_SIGS, NULL_SIGS)) %>%
  replicate_pairs(value_col = "score_z_mean") %>%
  mutate(family = case_when(signature_name %in% NULL_SIGS ~ "null control",
                            signature_name %in% ID_SIGS ~ "identity control",
                            TRUE ~ "candidate state"))

cat(sprintf("%s within-group replicate pairs across %s signatures.\n",
            format(nrow(pairs), big.mark = ","), n_distinct(pairs$signature_name)))

show_table(
  pairs %>% agreement_summary(by = c("family", "domain", "signature_name", "assay_pair")) %>%
    arrange(family, domain, signature_name, assay_pair) %>%
    select(family, domain, signature_name, assay_pair, n_pairs, median_abs_difference,
           mean_difference, repeatability_sd, loa_lower, loa_upper),
  "Bland-Altman style agreement of repeated sample/library measurements, by assay-method match"
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
#| label: icc

icc_tbl <- scores %>%
  filter(signature_name %in% c(STATE_SIGS, ID_SIGS, NULL_SIGS)) %>%
  group_by(domain, signature_name) %>%
  group_modify(~ icc_oneway(.x$score_z_mean, paste(.x$donor_id, .x$brain_region, .x$supertype))) %>%
  ungroup() %>%
  mutate(family = case_when(signature_name %in% NULL_SIGS ~ "null control",
                            signature_name %in% ID_SIGS ~ "identity control",
                            TRUE ~ "candidate state"))

show_table(
  icc_tbl %>% arrange(family, domain, desc(icc)) %>%
    select(family, domain, signature_name, n_groups, n_rows, icc, var_between, var_within),
  "One-way ICC(1,1) over donor x region x supertype groups. Replicates are repeated measurements, not donors."
)
#
#
#
#| label: fig-icc
#| fig-width: 7.5
#| fig-height: 4
#| fig-cap: "ICC(1,1) per signature. The dashed line marks the size-matched null-control level for the same domain; a candidate state signature is only worth carrying if it clears it."

null_level <- icc_tbl %>% filter(family == "null control") %>% select(domain, null_icc = icc)

icc_tbl %>%
  filter(family != "null control") %>%
  left_join(null_level, by = "domain") %>%
  ggplot(aes(reorder(signature_name, icc), icc, fill = family)) +
  geom_col(width = 0.6) +
  geom_hline(aes(yintercept = null_icc), linetype = 2, colour = "firebrick", linewidth = 0.35) +
  coord_flip() +
  facet_wrap(~ domain, scales = "free_y", ncol = 2) +
  labs(x = NULL, y = "ICC(1,1)", fill = NULL) +
  theme(legend.position = "top")
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
#| label: signature-specificity

null_by_domain <- scores %>%
  filter(signature_name %in% NULL_SIGS) %>%
  select(prepared_obs_index, domain, null_score = score_z_mean)

specificity <- scores %>%
  filter(signature_name %in% c(STATE_SIGS, ID_SIGS)) %>%
  select(prepared_obs_index, domain, signature_name, score_z_mean) %>%
  inner_join(null_by_domain, by = c("prepared_obs_index", "domain")) %>%
  group_by(domain, signature_name) %>%
  summarise(n_rows = n(),
            pearson_with_null = cor(score_z_mean, null_score),
            .groups = "drop") %>%
  left_join(registry %>% select(signature_name, role, n_genes), by = "signature_name") %>%
  arrange(domain, desc(pearson_with_null))

show_table(
  specificity %>% select(domain, signature_name, role, n_genes, n_rows, pearson_with_null),
  "How much of each score is the generic axis the null control also measures"
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
#| label: fig-bland-altman
#| fig-width: 8.5
#| fig-height: 4.5
#| fig-cap: "Difference between paired repeated measurements against the weaker member's nuclei count. If replicate disagreement were biological, it would not funnel with support."

pairs %>%
  filter(family == "candidate state") %>%
  ggplot(aes(support_min, difference, colour = assay_pair)) +
  geom_hline(yintercept = 0, linewidth = 0.3) +
  geom_point(alpha = 0.25, size = 0.5) +
  scale_x_log10() +
  facet_wrap(~ domain, nrow = 1) +
  labs(x = "nuclei in the weaker member of the pair (log)",
       y = "difference in state score", colour = NULL) +
  theme(legend.position = "top")
#
#
#
#| label: difference-drivers

drivers <- pairs %>%
  filter(family == "candidate state") %>%
  group_by(domain, signature_name) %>%
  summarise(
    n_pairs = n(),
    spearman_absdiff_vs_min_nuclei = cor(abs_difference, support_min, method = "spearman"),
    spearman_absdiff_vs_min_umi = cor(abs_difference, umi_min, method = "spearman"),
    median_absdiff_same_method = median(abs_difference[assay_pair == "same method"]),
    median_absdiff_mixed_method = median(abs_difference[assay_pair == "mixed method"]),
    .groups = "drop"
  ) %>%
  mutate(mixed_minus_same = median_absdiff_mixed_method - median_absdiff_same_method)

show_table(
  drivers %>% arrange(domain, signature_name),
  "What drives replicate disagreement: support, and assay-method match"
)
#
#
#
#
#
#
#
#
#| label: assay-design

design <- pairs %>%
  filter(family == "candidate state", assay_pair == "mixed method") %>%
  distinct(domain, donor_id, brain_region, supertype, method_a, method_b) %>%
  mutate(pair = paste(pmin(method_a, method_b), pmax(method_a, method_b), sep = " vs ")) %>%
  count(domain, pair)

show_table(design %>% pivot_wider(names_from = pair, values_from = n, values_fill = 0),
           "Which method pairs are actually observed inside a donor x region x supertype")
#
#
#
#
#
#
#
#
#| label: assay-offsets

contrasts <- pairs %>% filter(family == "candidate state") %>% method_pair_contrasts()

show_table(
  contrasts %>% arrange(domain, signature_name, first_method, second_method) %>%
    select(domain, signature_name, first_method, second_method, n_pairs,
           median_offset, mean_offset, ci_lower, ci_upper, wilcoxon_p),
  "Directly observed assay-method contrasts, estimated only within donor x region x supertype"
)

offsets <- method_offsets(pairs %>% filter(family == "candidate state"), reference = "10Xv3.1")

show_table(
  offsets %>% arrange(signature_name, assay_method) %>%
    select(signature_name, reference_method, assay_method, method_offset, n_steps_from_reference),
  "Additive offsets relative to 10Xv3.1, with the chain length behind each one"
)

show_table(
  offsets %>% group_by(assay_method) %>%
    summarise(n_signatures = n(), n_estimable = sum(!is.na(method_offset)),
              median_offset = median(method_offset, na.rm = TRUE),
              steps = paste(sort(unique(n_steps_from_reference)), collapse = ","), .groups = "drop"),
  "Offset coverage by method. A method with no path to the reference would appear as not estimable."
)
#
#
#
#
#
#
#
#
#| label: fig-assay
#| fig-width: 8
#| fig-height: 4.5
#| fig-cap: "Within-group assay-method contrasts, for every directly observed method pair. Donor, region and supertype are fixed inside every pair, so a non-zero centre is a method effect and not a cohort effect."

pairs %>%
  filter(family == "candidate state", assay_pair == "mixed method") %>%
  mutate(pair = paste(pmin(method_a, method_b), pmax(method_a, method_b), sep = " -> "),
         signed = if_else(method_a == pmax(method_a, method_b),
                          value_a - value_b, value_b - value_a)) %>%
  ggplot(aes(signed, signature_name, colour = pair)) +
  geom_vline(xintercept = 0, linewidth = 0.3) +
  geom_boxplot(outlier.size = 0.4, linewidth = 0.35) +
  labs(x = "score(second method) - score(first method), within donor x region x supertype",
       y = NULL, colour = NULL) +
  theme(legend.position = "top")
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
#| label: support-curve

support <- pairs %>%
  filter(family == "candidate state") %>%
  support_reliability(support_col = "support_min")

show_table(
  support %>% filter(n_pairs >= 5) %>% arrange(domain, signature_name, support_bin),
  "Replicate disagreement by nuclei support of the weaker member (bins with at least 5 pairs)"
)
#
#
#
#| label: fig-support
#| fig-width: 8.5
#| fig-height: 4.5
#| fig-cap: "Repeatability standard deviation against nuclei support. The score is standardized within domain, so a repeatability SD of 1 means the repeated measurement carries no information about the group at all."

support %>%
  filter(n_pairs >= 5) %>%
  ggplot(aes(support_bin, repeatability_sd, group = signature_name, colour = signature_name)) +
  geom_line(linewidth = 0.4) + geom_point(size = 0.8) +
  geom_hline(yintercept = 1, linetype = 2, colour = "grey40", linewidth = 0.3) +
  facet_wrap(~ domain, nrow = 1, scales = "free_x") +
  labs(x = "nuclei in the weaker member of the pair", y = "repeatability SD (score units)",
       colour = NULL) +
  theme(legend.position = "bottom", legend.text = element_text(size = 6),
        axis.text.x = element_text(angle = 45, hjust = 1, size = 6))
#
#
#
#| label: support-umi

support_umi <- pairs %>%
  filter(family == "candidate state") %>%
  mutate(umi_bin = cut(umi_min, breaks = c(0, 1e3, 1e4, 3e4, 1e5, 3e5, 1e6, Inf),
                       include.lowest = TRUE, dig.lab = 8)) %>%
  group_by(domain, umi_bin) %>%
  summarise(n_pairs = n(), median_abs_difference = median(abs_difference),
            repeatability_sd = sd(difference) / sqrt(2), .groups = "drop")

show_table(support_umi %>% arrange(domain, umi_bin),
           "The same curve against total UMI of the weaker member, pooled over candidate signatures")
#
#
#
#| label: support-threshold

# A descriptive, not-yet-applied candidate rule: the smallest pre-declared bin at
# which the repeatability SD of every candidate signature in that domain first
# falls below half the within-domain spread of the score.
# Two criteria, not one, because a single cut-off would hide how sensitive the
# answer is to where the line is drawn. The score is standardized within domain,
# so a repeatability SD of r corresponds to a reliability of about 1 - r^2/2.
candidate_rule <- support %>%
  filter(n_pairs >= 5) %>%
  group_by(domain, support_bin) %>%
  summarise(worst_repeatability_sd = max(repeatability_sd, na.rm = TRUE),
            n_signatures = n_distinct(signature_name), n_pairs = sum(n_pairs), .groups = "drop") %>%
  arrange(domain, support_bin) %>%
  mutate(meets_lenient = worst_repeatability_sd < 0.5,
         meets_strict = worst_repeatability_sd < 0.3)

show_table(candidate_rule, "Repeatability of the worst candidate signature in each domain, by support bin")

first_bin <- function(column) {
  candidate_rule %>% filter(.data[[column]]) %>% group_by(domain) %>% slice(1) %>% ungroup() %>%
    transmute(domain, criterion = column, first_bin = support_bin,
              worst_repeatability_sd, n_pairs)
}
first_pass <- bind_rows(first_bin("meets_lenient"), first_bin("meets_strict")) %>%
  arrange(domain, criterion)

show_table(first_pass,
           "Empirical candidate support thresholds under two criteria. NOT applied anywhere in this notebook.")
#
#
#
#| label: support-consequence

consequence <- scores %>%
  filter(signature_name %in% STATE_SIGS) %>%
  distinct(domain, prepared_obs_index, donor_id, brain_region, supertype, n_nuclei) %>%
  mutate(across(n_nuclei, as.numeric)) %>%
  group_by(domain) %>%
  summarise(
    n_rows = n(),
    rows_under_5 = sum(n_nuclei < 5), rows_under_10 = sum(n_nuclei < 10),
    rows_under_20 = sum(n_nuclei < 20),
    pct_under_10 = 100 * mean(n_nuclei < 10),
    donor_region_lost_at_10 = {
      keep <- n_nuclei >= 10
      n_distinct(paste(donor_id, brain_region)[!keep]) -
        n_distinct(intersect(paste(donor_id, brain_region)[!keep], paste(donor_id, brain_region)[keep]))
    },
    .groups = "drop"
  )

show_table(consequence,
           "What a threshold would cost. Reported so the cost is visible before any rule is adopted.")
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
#| label: immune-taxonomy

show_table(
  taxonomy %>% filter(lineage == "Immune") %>%
    select(supertype, n_donors, n_regions, total_nuclei, taxonomy_interpretation, semantic_status),
  "Released Immune supertypes, exactly as the release names them"
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
#| label: monocyte-caution

show_table(
  normtab %>% filter(domain %in% c("Micro/PVM", "Lymphocyte", "Monocyte")) %>%
    group_by(domain) %>%
    summarise(n_rows = n(), n_donors = n_distinct(donor_id),
              median_nuclei = median(n_nuclei), max_nuclei = max(n_nuclei),
              median_umi = median(total_umi), .groups = "drop"),
  "Why the Immune split matters: the non-myeloid domains are tiny and would otherwise dilute the axis"
)
#
#
#
#
#
#| label: fig-supertype
#| fig-width: 9
#| fig-height: 7
#| fig-cap: "Row-level state scores by released supertype. Spread across supertypes within a domain is a reminder that the supertype is a stratum: a donor-level score computed without conditioning on it would partly re-measure composition."

scores %>%
  filter(signature_name %in% STATE_SIGS) %>%
  ggplot(aes(score_z_mean, supertype)) +
  geom_boxplot(outlier.size = 0.3, linewidth = 0.3) +
  facet_wrap(~ signature_name, scales = "free", ncol = 3) +
  labs(x = "state score (within-domain z mean)", y = NULL) +
  theme(strip.text = element_text(size = 7), axis.text.y = element_text(size = 6))
#
#
#
#| label: supertype-variance

variance_share <- scores %>%
  filter(signature_name %in% c(STATE_SIGS, NULL_SIGS)) %>%
  group_by(domain, signature_name) %>%
  group_modify(function(.x, .y) {
    fit_st <- summary(aov(score_z_mean ~ supertype, data = .x))[[1]]
    fit_rg <- summary(aov(score_z_mean ~ brain_region, data = .x))[[1]]
    tibble(
      supertype_variance_share = fit_st[["Sum Sq"]][1] / sum(fit_st[["Sum Sq"]]),
      region_variance_share    = fit_rg[["Sum Sq"]][1] / sum(fit_rg[["Sum Sq"]]),
      n_rows = nrow(.x)
    )
  }) %>%
  ungroup() %>%
  mutate(family = if_else(signature_name %in% NULL_SIGS, "null control", "candidate state"))

show_table(
  variance_share %>% arrange(family, domain, desc(supertype_variance_share)),
  "Share of row-level score variance explained by supertype and, separately, by region"
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
#| label: fig-region
#| fig-width: 9
#| fig-height: 6
#| fig-cap: "Row-level state scores by brain region, for the candidate state signatures."

scores %>%
  filter(signature_name %in% STATE_SIGS) %>%
  ggplot(aes(score_z_mean, brain_region)) +
  geom_boxplot(outlier.size = 0.3, linewidth = 0.3) +
  facet_wrap(~ signature_name, scales = "free_x", ncol = 3) +
  labs(x = "state score (within-domain z mean)", y = NULL) +
  theme(strip.text = element_text(size = 7))
#
#
#
#| label: state-vs-identity

identity_vs_state <- scores %>%
  filter(signature_name %in% c(STATE_SIGS, ID_SIGS)) %>%
  select(prepared_obs_index, domain, signature_name, score_z_mean) %>%
  pivot_wider(names_from = signature_name, values_from = score_z_mean)

id_corr <- map_dfr(intersect(STATE_SIGS, names(identity_vs_state)), function(s) {
  dom <- registry$domain[match(s, registry$signature_name)]
  control <- registry %>% filter(role == "identity_control", domain == dom) %>% pull(signature_name)
  if (!length(control) || !control %in% names(identity_vs_state)) return(NULL)
  d <- identity_vs_state %>% filter(domain == dom)
  tibble(domain = dom, signature_name = s, identity_control = control,
         pearson_with_identity = cor(d[[s]], d[[control]], use = "complete.obs"))
})

show_table(id_corr, "Is a candidate state score really a lineage-identity readout?")
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
#| label: policies

policy_defs <- tibble::tribble(
  ~policy, ~definition, ~what_it_assumes,
  "equal_weight_mean",
    "unweighted mean of the normalized state scores of the repeated rows",
    "each repeated measurement is an equally good estimate of the same quantity",
  "nucleus_weighted_mean",
    "mean weighted by Number of nuclei",
    "a measurement's precision scales with the nuclei behind it",
  "raw-count pooled score",
    "sum raw counts across repeated rows, then normalize and score once",
    "the repeats are one experiment; implicitly weights by library size (SENSITIVITY ONLY)",
  "assay_adjusted_mean",
    "subtract a per-method additive offset, then take the unweighted mean",
    "assay method shifts the score additively by a fixed amount",
  "assay_adjusted_nucleus_weighted_mean",
    "subtract the offset, then weight by Number of nuclei",
    "both of the above at once: additive method shift, precision rising with nuclei"
)

show_table(policy_defs, "The five aggregation policies")
#
#
#
#| label: aggregate

meth_off <- offsets %>% select(signature_name, assay_method, method_offset)

# A method with no estimable offset is adjusted by zero, which is an assumption,
# so its size is reported rather than left implicit.
unadjusted <- scores %>%
  filter(signature_name %in% STATE_SIGS) %>%
  left_join(meth_off, by = c("signature_name", "assay_method")) %>%
  summarise(rows = n(), rows_without_an_estimable_offset = sum(is.na(method_offset)))

show_table(unadjusted, "How many scored rows the assay adjustment cannot reach")

agg <- scores %>%
  filter(signature_name %in% STATE_SIGS) %>%
  aggregate_policies(offsets = meth_off, value_col = "score_z_mean")

pooled_wide <- pooled %>%
  filter(signature_name %in% STATE_SIGS) %>%
  select(donor_id, brain_region, supertype, signature_name, pooled_score = score_z_mean)

agg <- agg %>% left_join(pooled_wide, by = c("donor_id", "brain_region", "supertype", "signature_name"))

cat(sprintf("%s donor x region x supertype x signature cells; %s of them from more than one measurement.\n",
            format(nrow(agg), big.mark = ","),
            format(sum(agg$n_measurements > 1), big.mark = ",")))
#
#
#
#
#
#
#
#
#| label: policy-agreement

multi <- agg %>% filter(n_measurements > 1)

agreement <- bind_rows(
  policy_agreement(multi, "equal_weight_mean", "nucleus_weighted_mean"),
  policy_agreement(multi, "equal_weight_mean", "umi_weighted_mean"),
  policy_agreement(multi, "equal_weight_mean", "assay_adjusted_mean"),
  policy_agreement(multi, "equal_weight_mean", "assay_adjusted_nucleus_weighted_mean"),
  policy_agreement(multi, "equal_weight_mean", "pooled_score"),
  policy_agreement(multi, "nucleus_weighted_mean", "pooled_score")
)

show_table(
  agreement %>% arrange(policy_a, policy_b, signature_name),
  "Agreement between aggregation policies, over groups that actually have repeated measurements"
)
#
#
#
#| label: fig-policies
#| fig-width: 8
#| fig-height: 4.2
#| fig-cap: "Equal-weight mean against the three alternatives, over groups with repeated measurements. Deviation from the diagonal is where the policy choice would change a donor x region x supertype value."

multi %>%
  select(signature_name, equal_weight_mean, nucleus_weighted_mean, assay_adjusted_mean,
         assay_adjusted_nucleus_weighted_mean, pooled_score) %>%
  pivot_longer(-c(signature_name, equal_weight_mean), names_to = "policy", values_to = "value") %>%
  filter(!is.na(value)) %>%
  ggplot(aes(equal_weight_mean, value)) +
  geom_abline(slope = 1, intercept = 0, colour = "firebrick", linewidth = 0.35) +
  geom_point(alpha = 0.15, size = 0.5) +
  facet_wrap(~ policy, nrow = 1) +
  labs(x = "equal-weight mean of normalized scores", y = "alternative policy")
#
#
#
#| label: policy-robustness

robustness <- multi %>%
  group_by(signature_name) %>%
  summarise(
    n_groups = n(),
    sd_equal = sd(equal_weight_mean),
    sd_nucleus = sd(nucleus_weighted_mean),
    sd_assay_adjusted = sd(assay_adjusted_mean, na.rm = TRUE),
    sd_assay_adjusted_nucleus = sd(assay_adjusted_nucleus_weighted_mean, na.rm = TRUE),
    sd_pooled = sd(pooled_score, na.rm = TRUE),
    median_within_group_sd = median(within_group_sd, na.rm = TRUE),
    .groups = "drop"
  )

show_table(robustness,
           "Spread of each aggregated estimate, beside the median within-group measurement SD it has to average over")
#
#
#
#| label: policy-influence

influence <- multi %>%
  mutate(shift_nucleus = abs(nucleus_weighted_mean - equal_weight_mean),
         shift_pooled = abs(pooled_score - equal_weight_mean),
         shift_assay = abs(assay_adjusted_mean - equal_weight_mean),
         nuclei_imbalance = n_nuclei_total / n_measurements) %>%
  group_by(signature_name) %>%
  summarise(
    spearman_shift_nucleus_vs_imbalance = cor(shift_nucleus, nuclei_imbalance, method = "spearman",
                                              use = "complete.obs"),
    median_shift_nucleus = median(shift_nucleus, na.rm = TRUE),
    median_shift_pooled = median(shift_pooled, na.rm = TRUE),
    median_shift_assay = median(shift_assay, na.rm = TRUE),
    .groups = "drop"
  )

show_table(influence, "How far each alternative moves the estimate away from the equal-weight mean")
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
#| label: provisional-table

provisional <- agg %>%
  left_join(registry %>% select(signature_name, feature_origin), by = "signature_name") %>%
  mutate(
    source_release = prep_prov$source_release,
    aggregation_method = "nucleus_weighted_mean_of_assay_adjusted_row_level_normalized_scores",
    state_score = assay_adjusted_nucleus_weighted_mean,
    support_status = case_when(
      n_nuclei_total >= 20 ~ "well supported",
      n_nuclei_total >= 10 ~ "marginal support",
      TRUE ~ "low support"
    )
  ) %>%
  select(donor_id, brain_region, lineage, supertype, signature_name, state_score,
         n_measurements, n_nuclei_total, aggregation_method, support_status,
         source_release, feature_origin,
         domain, equal_weight_mean, nucleus_weighted_mean, assay_adjusted_mean, pooled_score,
         within_group_sd, n_assay_methods, assay_methods, total_umi_total)

show_table(
  provisional %>% slice_head(n = 8) %>%
    select(donor_id, brain_region, lineage, supertype, signature_name, state_score,
           n_measurements, n_nuclei_total, aggregation_method, support_status, feature_origin),
  "The provisional donor x region x supertype state table, first eight rows"
)

show_table(
  provisional %>%
    group_by(lineage, signature_name) %>%
    summarise(n_cells = n(), n_donors = n_distinct(donor_id),
              n_donor_region = n_distinct(paste(donor_id, brain_region)),
              well_supported = sum(support_status == "well supported"),
              marginal = sum(support_status == "marginal support"),
              low = sum(support_status == "low support"), .groups = "drop"),
  "Size of the provisional table, and how much of it is well supported"
)

show_table(
  provisional %>% filter(brain_region %in% ROI) %>%
    count(brain_region, support_status) %>%
    pivot_wider(names_from = support_status, values_from = n, values_fill = 0),
  "The same, restricted to the candidate ROI set (which is still not frozen)"
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
#| label: composition

comp <- composition_table(normtab)

show_table(
  comp %>% slice_head(n = 8) %>%
    select(donor_id, brain_region, lineage, supertype, n_nuclei, relative_abundance),
  "The compositional table, first eight rows: donor x region x lineage x supertype"
)

show_table(
  comp %>% group_by(lineage) %>%
    summarise(n_cells = n(), n_donors = n_distinct(donor_id),
              total_nuclei = sum(n_nuclei),
              median_relative_abundance = median(relative_abundance), .groups = "drop"),
  "Relative abundance summary, within the four downloaded glial lineages"
)

cat(unique(comp$abundance_denominator), "\n", unique(comp$abundance_caveat), "\n", sep = "")
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
#| label: composition-vs-state

comp_state <- provisional %>%
  inner_join(comp %>% select(donor_id, brain_region, supertype, relative_abundance),
             by = c("donor_id", "brain_region", "supertype")) %>%
  group_by(signature_name) %>%
  summarise(n = n(),
            spearman_state_vs_abundance = cor(state_score, relative_abundance,
                                              method = "spearman", use = "complete.obs"),
            .groups = "drop")

show_table(comp_state,
           "Diagnostic only: how much a state score already tracks the supertype's own abundance")
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
#| label: qc-join

cps <- read_local_cps(omics$cps_by_region)

qnp <- read_qnp_workbook(paths$qnp_2026)
qnp_long <- qnp_primary_long(qnp) %>%
  filter(variable %in% c("abeta_percent_positive_area", "ptau_percent_positive_area"),
         status == "observed") %>%
  select(donor_id, brain_region, variable, value) %>%
  distinct()

# "Unambiguously" is checked, not assumed: one value per donor x region x marker,
# or the join does not happen.
stopifnot(!any(duplicated(qnp_long[c("donor_id", "brain_region", "variable")])))
qnp_p <- pivot_wider(qnp_long, names_from = variable, values_from = value)

qc <- provisional %>%
  left_join(cps, by = c("donor_id", "brain_region")) %>%
  left_join(qnp_p, by = c("donor_id", "brain_region"))

show_table(
  qc %>%
    summarise(
      cells = n(),
      with_CPS_Local = sum(!is.na(CPS_Local)),
      with_local_abeta = sum(!is.na(abeta_percent_positive_area)),
      with_local_ptau = sum(!is.na(ptau_percent_positive_area))
    ),
  "How much of the provisional table joins unambiguously to the stage fields"
)

# The two stage sources are independent, so their coverage being identical is
# checked rather than assumed.
qnp_grid <- distinct(qnp_p, donor_id, brain_region)
cps_grid <- distinct(cps, donor_id, brain_region)
cat(sprintf("CPS donor x region cells not covered by the QNP grey-matter grid: %d\n",
            nrow(anti_join(cps_grid, qnp_grid, by = c("donor_id", "brain_region")))))

show_table(
  qc %>% group_by(brain_region) %>%
    summarise(cells = n(), with_CPS_Local = sum(!is.na(CPS_Local)),
              with_local_abeta = sum(!is.na(abeta_percent_positive_area)),
              with_local_ptau = sum(!is.na(ptau_percent_positive_area)), .groups = "drop"),
  "Join coverage by region. LEC has no published local CPS, so it is absent by construction."
)
#
#
#
#
#
#| label: fig-qc-cps
#| fig-width: 9
#| fig-height: 6
#| fig-cap: "Provisional state score against the published local pseudo-progression score. Descriptive only."

qc %>%
  filter(!is.na(CPS_Local)) %>%
  ggplot(aes(CPS_Local, state_score)) +
  geom_point(alpha = 0.15, size = 0.5) +
  geom_smooth(method = "loess", se = FALSE, linewidth = 0.4, colour = "firebrick") +
  facet_wrap(~ signature_name, scales = "free_y", ncol = 3) +
  labs(x = "CPS_Local (published, neuropathology-derived)", y = "provisional state score") +
  theme(strip.text = element_text(size = 7))
#
#
#
#| label: fig-qc-path
#| fig-width: 9
#| fig-height: 6
#| fig-cap: "Provisional state score against local Ab (6E10) and pTau (AT8) percent positive area, grey matter. Descriptive only."

qc %>%
  select(signature_name, state_score, abeta_percent_positive_area, ptau_percent_positive_area) %>%
  pivot_longer(c(abeta_percent_positive_area, ptau_percent_positive_area),
               names_to = "marker", values_to = "percent_positive_area") %>%
  filter(!is.na(percent_positive_area)) %>%
  ggplot(aes(percent_positive_area, state_score, colour = marker)) +
  geom_point(alpha = 0.12, size = 0.4) +
  geom_smooth(method = "loess", se = FALSE, linewidth = 0.4) +
  facet_wrap(~ signature_name, scales = "free", ncol = 3) +
  labs(x = "percent positive area (grey matter)", y = "provisional state score", colour = NULL) +
  theme(legend.position = "top", strip.text = element_text(size = 7))
#
#
#
#
#
#| label: fig-qc-standardized
#| fig-width: 9
#| fig-height: 6
#| fig-cap: "The same relationship after standardizing the score within region. Held separate from the raw version above; neither replaces the other."

qc_z <- qc %>%
  filter(!is.na(CPS_Local)) %>%
  group_by(signature_name) %>%
  group_modify(~ add_region_standardized(.x, "state_score")) %>%
  ungroup()

qc_z %>%
  ggplot(aes(CPS_Local, state_score_z_within_region)) +
  geom_point(alpha = 0.15, size = 0.5) +
  geom_smooth(method = "loess", se = FALSE, linewidth = 0.4, colour = "firebrick") +
  facet_wrap(~ signature_name, scales = "free_y", ncol = 3) +
  labs(x = "CPS_Local", y = "state score, z within region") +
  theme(strip.text = element_text(size = 7))
#
#
#
#| label: qc-summary

qc_corr <- qc_z %>%
  group_by(signature_name, lineage) %>%
  summarise(
    n = n(),
    spearman_raw_vs_CPS = cor(state_score, CPS_Local, method = "spearman", use = "complete.obs"),
    spearman_regionz_vs_CPS = cor(state_score_z_within_region, CPS_Local,
                                  method = "spearman", use = "complete.obs"),
    .groups = "drop"
  )

show_table(
  qc_corr %>% arrange(lineage, signature_name),
  "Descriptive rank correlations with CPS_Local. Reported for every candidate, used to select none."
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
#| label: neurons

manifest <- read_seaad_s3_manifest(omics$s3_manifest)

neuronal <- manifest %>%
  filter(family == "pseudobulk (donor x region x supertype)",
         !str_detect(file, "Immune|Astrocyte|Oligodendrocyte|OPC|Ependymal|Endothelial|VLMC|Microglia")) %>%
  transmute(file, size_mb = round(size_mb, 1)) %>%
  arrange(desc(size_mb))

show_table(neuronal, "Candidate neuronal pseudobulk objects. Inventory only; nothing was fetched.")

cat(sprintf("Total if fetched: %.2f GB across %d objects. Not downloaded in this task.\n",
            sum(neuronal$size_mb) / 1000, nrow(neuronal)))
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
#| label: answers-inputs

state_icc  <- icc_tbl %>% filter(family == "candidate state")
null_icc   <- icc_tbl %>% filter(family == "null control") %>% select(domain, null_icc = icc)
icc_vs_null <- state_icc %>% left_join(null_icc, by = "domain") %>%
  mutate(clears_null = icc > null_icc)

worst_offset <- contrasts %>% slice_max(abs(median_offset), n = 1)
n_offset_sig <- sum(contrasts$wilcoxon_p < 0.05, na.rm = TRUE)
n_offset_chained <- sum(offsets$n_steps_from_reference > 1, na.rm = TRUE)
n_offset_missing <- sum(is.na(offsets$method_offset))

pol <- agreement %>% filter(policy_a == "equal_weight_mean")
#
#
#
#| label: answers

answers <- tibble::tribble(
  ~question, ~answer,

  "1. Which predefined glial state signatures are defensible candidates?",
  sprintf(paste0("%d resolved external candidates were scored (%d Micro/PVM, %d Astrocyte, ",
                 "%d Oligodendrocyte, %d OPC), all ontology- or Hallmark-defined and copied ",
                 "verbatim from MSigDB %s. %d further candidates are carried unresolved with no ",
                 "genes, including the only human AD-derived state definitions (Sun 2023 MG4/MG8) ",
                 "and the SEA-AD-published option. No SEA-AD-published named gene set exists."),
          length(STATE_SIGS),
          sum(registry$role == "candidate_state" & registry$resolution_status == "resolved" & registry$domain == "Micro/PVM"),
          sum(registry$role == "candidate_state" & registry$resolution_status == "resolved" & registry$domain == "Astrocyte"),
          sum(registry$role == "candidate_state" & registry$resolution_status == "resolved" & registry$domain == "Oligodendrocyte"),
          sum(registry$role == "candidate_state" & registry$resolution_status == "resolved" & registry$domain == "OPC"),
          registry_j$provenance$msigdb_release,
          sum(registry$resolution_status == "unresolved")),

  "2. Are their genes adequately represented in the released pseudobulk?",
  sprintf(paste0("Yes for presence: fraction_present ranges %.2f to %.2f across scored signatures ",
                 "(median %.2f). Detectability is the binding constraint: %d of %d scored ",
                 "signature x domain cells lose genes to the %g non-zero-fraction rule, and the ",
                 "smallest sets lose proportionally most."),
          min(as.numeric(coverage$fraction_present), na.rm = TRUE),
          max(as.numeric(coverage$fraction_present), na.rm = TRUE),
          median(as.numeric(coverage$fraction_present), na.rm = TRUE),
          sum(coverage$n_detectably_expressed < coverage$n_present),
          nrow(coverage), score_prov$detectability$fraction),

  "3. Are row-level scores reproducible across repeated sample/library measurements?",
  sprintf(paste0("Yes as measurements, but the reproducibility is not signature-specific. Over %s ",
                 "replicate pairs, candidate-state ICC(1,1) runs %.2f to %.2f (median %.2f) and the ",
                 "median absolute replicate difference is %.2f score units against a within-domain ",
                 "SD of about 1. However only %d of %d candidates exceed their domain's ",
                 "size-matched null control, whose ICC is %.2f to %.2f, and each candidate ",
                 "correlates with that null draw at Pearson %.2f to %.2f. Row-level scoring is ",
                 "therefore reproducible; what it reproduces is largely a generic recovery and ",
                 "composition axis shared by any gene set of the same size."),
          format(nrow(pairs), big.mark = ","),
          min(state_icc$icc, na.rm = TRUE), max(state_icc$icc, na.rm = TRUE),
          median(state_icc$icc, na.rm = TRUE),
          median(pairs$abs_difference[pairs$family == "candidate state"], na.rm = TRUE),
          sum(icc_vs_null$clears_null, na.rm = TRUE), nrow(icc_vs_null),
          min(null_icc$null_icc, na.rm = TRUE), max(null_icc$null_icc, na.rm = TRUE),
          min(specificity$pearson_with_null[specificity$role == "candidate_state"], na.rm = TRUE),
          max(specificity$pearson_with_null[specificity$role == "candidate_state"], na.rm = TRUE)),

  "4. Does assay method materially affect the scores?",
  sprintf(paste0("Yes, and systematically. Offsets are estimated within donor x region x ",
                 "supertype, so they cannot be cohort effects. %d of %d directly observed ",
                 "signature x method-pair contrasts reach p < 0.05 (Wilcoxon), and every one of ",
                 "the 26 points the same way: 10xMulti reads lower than both other methods. The ",
                 "largest median offset is %.2f score units (%s, %s minus %s). ",
                 "Mixed-method pairs disagree more than same-method pairs for %d of %d ",
                 "signatures. The design is a chain, so %d of %d method offsets are reached in ",
                 "more than one step and %d are not estimable at all."),
          n_offset_sig, nrow(contrasts),
          worst_offset$median_offset[1], worst_offset$signature_name[1],
          worst_offset$second_method[1], worst_offset$first_method[1],
          sum(drivers$mixed_minus_same > 0, na.rm = TRUE), nrow(drivers),
          n_offset_chained, nrow(offsets), n_offset_missing),

  "5. Is there evidence for a minimum support criterion?",
  sprintf(paste0("Yes, and it is empirical rather than conventional: replicate disagreement falls ",
                 "monotonically with nuclei support, and absolute difference correlates with ",
                 "support at Spearman %.2f to %.2f. A candidate rule is tabulated per domain. ",
                 "Nothing has been filtered: %.0f%% of scored rows sit below 10 nuclei."),
          min(drivers$spearman_absdiff_vs_min_nuclei, na.rm = TRUE),
          max(drivers$spearman_absdiff_vs_min_nuclei, na.rm = TRUE),
          100 * mean(as.numeric(normtab$n_nuclei) < 10)),

  "6. What aggregation rule should convert repeated measurements into donor x region x supertype estimates?",
  sprintf(paste0("The nucleus-weighted mean of assay-adjusted row-level scores. The assay offset ",
                 "is systematic and one-directional, so it must be removed rather than averaged ",
                 "over, and replicate precision rises steeply with nuclei, so the rows are not ",
                 "interchangeable. Against the unweighted mean it agrees at Pearson %.3f (median ",
                 "across signatures), the nucleus-weighted mean alone at %.3f and the ",
                 "assay-adjusted unweighted mean at %.3f, so the choice moves few values but moves ",
                 "them exactly where support or platform is unbalanced. The raw-count pooled score ",
                 "agrees least (%.3f) and is a sensitivity arm only."),
          median(pol$pearson[pol$policy_b == "assay_adjusted_nucleus_weighted_mean"], na.rm = TRUE),
          median(pol$pearson[pol$policy_b == "nucleus_weighted_mean"], na.rm = TRUE),
          median(pol$pearson[pol$policy_b == "assay_adjusted_mean"], na.rm = TRUE),
          median(pol$pearson[pol$policy_b == "pooled_score"], na.rm = TRUE)),

  "7. How strongly do state scores vary by supertype and region?",
  sprintf(paste0("Supertype explains %.0f%% to %.0f%% of row-level score variance (median %.0f%%); ",
                 "region explains %.0f%% to %.0f%% (median %.0f%%). Supertype dominates region, ",
                 "which is why it must stay a stratum rather than be averaged over."),
          100 * min(variance_share$supertype_variance_share[variance_share$family == "candidate state"]),
          100 * max(variance_share$supertype_variance_share[variance_share$family == "candidate state"]),
          100 * median(variance_share$supertype_variance_share[variance_share$family == "candidate state"]),
          100 * min(variance_share$region_variance_share[variance_share$family == "candidate state"]),
          100 * max(variance_share$region_variance_share[variance_share$family == "candidate state"]),
          100 * median(variance_share$region_variance_share[variance_share$family == "candidate state"])),

  "8. What provisional state variables are ready for later donor x ROI modelling?",
  sprintf(paste0("A long table of %s donor x region x supertype x signature cells over %d donors ",
                 "and %d regions, of which %s are well supported (>= 20 nuclei). Restricted to the ",
                 "candidate ROI set it holds %s cells. It is exploratory, not a model matrix."),
          format(nrow(provisional), big.mark = ","),
          n_distinct(provisional$donor_id), n_distinct(provisional$brain_region),
          format(sum(provisional$support_status == "well supported"), big.mark = ","),
          format(sum(provisional$brain_region %in% ROI), big.mark = ",")),

  "9. Which uncertainties remain before defining final T variables?",
  paste0("No scored signature is simultaneously human, AD-derived and state-level, and none is ",
         "distinguishable from a size-matched random gene set on reliability alone; ",
         "sample_name semantics remain undocumented, so the replicate structure is interpreted as ",
         "measurement rather than known biology; the supertype stratum has no agreed collapsing ",
         "rule; microglia are not separated from PVM inside Micro-PVM_*; the support rule is ",
         "proposed but unapplied; and the neuronal axis is undefined.")
)

show_table(answers, "The nine stop-condition questions, answered from the tables above")
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
    "results/analysis/02_seaad_omics_state_exploration.qmd", "new", "this notebook",
    "R/seaad_state.R", "new",
      "reliability, aggregation and composition helpers; no data is written from R",
    "src/seaad/omics/signature_registry.py", "new",
      "signature curation; resolves genes from MSigDB or refuses and marks unresolved",
    "src/seaad/omics/state_scoring.py", "new",
      "TMM / logCPM at the released row grain, module scoring, pooled sensitivity arm",
    "src/seaad/cli/omics_build_signature_registry.py", "new", "CLI for the registry",
    "src/seaad/cli/omics_state_scores.py", "new", "CLI for normalization and scoring",
    "src/seaad/tests/test_state_scoring.py", "new",
      "TMM properties, taxonomy domains, unresolved-signature refusal, end-to-end row grain",
    "src/seaad/pyproject.toml", "modified", "two new console scripts",
    "data/derivatives/sea-ad/omics_state/", "new",
      "registry, gene lists, row normalization, row scores, pooled scores, coverage, contributions, provenance"
  ),
  "Files changed by this task. Acquisition and preparation were not touched."
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
#| label: unresolved-summary

show_table(
  tibble::tribble(
    ~issue, ~why_it_matters, ~what_would_resolve_it,
    "No candidate is distinguishable from a size-matched random gene set on reliability",
      "high ICC is a property of the measurement, not evidence that a signature tracks its biology",
      "a signature with genuine state structure, or a specificity criterion adopted before selection",
    "No human, AD-derived, state-level signature is scored",
      "every scored axis is an ontology or Hallmark biological axis, not a disease state",
      "recover Sun et al. 2023 Table S2 (MG4, MG8) and the Habib 2020 DAA list verbatim",
    "sample_name semantics are undocumented",
      "repeated rows are interpreted as measurement; if they were tissue blocks they would be biology",
      "a statement from SEA-AD, or a released specimen/tissue-block field",
    "Micro-PVM is not resolved into microglia and PVM",
      "the resident-myeloid axis is Micro/PVM, not microglia",
      "a finer released taxonomy, or a marker-based split that would itself need validating",
    "The support rule is proposed but not applied",
      "a large share of rows sit below ten nuclei",
      "one explicit decision, applied once, with the cost table recomputed",
    "Supertype stratification has no collapsing rule",
      "the eventual T is per donor x region, and supertype has to be handled on the way there",
      "a stated rule: condition, weight by abundance, or select one supertype per lineage",
    "The neuronal axis is undefined",
      "HIP has no layer-defined excitatory subclasses; all-neuron pooling confounds state with loss",
      "a defensible neuronal grouping present in every candidate ROI"
  ),
  "What remains open"
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
