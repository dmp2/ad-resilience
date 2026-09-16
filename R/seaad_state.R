# ---------------------------------------------------------------------------
# Helpers for the SEA-AD molecular-state exploration
# (results/analysis/02_seaad_omics_state_exploration.qmd).
#
# Scope: read the row-level derivatives written by
# `seaad-omics-build-signature-registry` and `seaad-omics-state-scores`, and
# summarise them as measurement-reliability quantities. Nothing here writes to
# data/, opens an .h5ad (R in this project has no HDF5 reader), joins morphology
# or cognition, or selects a signature by any association.
#
# Measurement hierarchy assumed throughout, and enforced by the group keys:
#
#   donor                                  biological replication unit
#   donor x region                         eventual regional analysis unit
#   supertype                              molecular / cellular stratum
#   sample / library-prep row              repeated measurement inside a
#                                          donor x region x supertype
#
# A repeated sample/library row is never treated as an independent donor.
# ---------------------------------------------------------------------------

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(purrr)
  library(tibble)
  library(stringr)
  library(readr)
})

# --- paths -----------------------------------------------------------------

#' Resolve the prepared-pseudobulk and state-score derivative files.
seaad_state_paths <- function(root = find_project_root()) {
  prepared <- file.path(root, "data", "derivatives", "sea-ad", "omics_prepared")
  state <- file.path(root, "data", "derivatives", "sea-ad", "omics_state")
  list(
    root                 = root,
    prepared_dir         = prepared,
    prepared_index       = file.path(prepared, "pseudobulk_prepared_index.csv"),
    prepared_coverage    = file.path(prepared, "pseudobulk_coverage.csv"),
    prepared_taxonomy    = file.path(prepared, "pseudobulk_taxonomy.csv"),
    prepared_audit       = file.path(prepared, "pseudobulk_aggregation_audit.csv"),
    gene_catalog         = file.path(prepared, "pseudobulk_gene_catalog.csv"),
    support_distribution = file.path(prepared, "pseudobulk_support_distribution.csv"),
    preparation_prov     = file.path(prepared, "preparation_provenance.json"),
    state_dir            = state,
    registry_json        = file.path(state, "signature_registry.json"),
    registry_csv         = file.path(state, "signature_registry.csv"),
    signature_genes      = file.path(state, "signature_genes.csv"),
    null_control_genes   = file.path(state, "null_control_genes.csv"),
    gene_coverage        = file.path(state, "signature_gene_coverage.csv"),
    gene_contributions   = file.path(state, "signature_gene_contributions.csv"),
    row_normalization    = file.path(state, "row_normalization.csv"),
    row_scores           = file.path(state, "row_state_scores.csv"),
    pooled_scores        = file.path(state, "pooled_state_scores.csv"),
    pooled_coverage      = file.path(state, "pooled_signature_gene_coverage.csv"),
    scoring_prov         = file.path(state, "state_scoring_provenance.json")
  )
}

#' The group key for one repeated-measurement unit.
STATE_GROUP_KEYS <- c("donor_id", "brain_region", "supertype", "signature_name")

#' Read the row-level scores, typed, with the measurement grain made explicit.
read_state_scores <- function(path) {
  read_csv(path, show_col_types = FALSE, progress = FALSE) %>%
    mutate(
      n_nuclei  = as.numeric(n_nuclei),
      total_umi = as.numeric(total_umi),
      across(c(score_z_mean, score_logcpm_mean, tmm_factor, effective_library_size), as.numeric),
      measurement_grain = "donor x region x supertype x sample/library"
    )
}

#' Read the per-row normalization table.
read_row_normalization <- function(path) {
  read_csv(path, show_col_types = FALSE, progress = FALSE) %>%
    mutate(across(c(n_nuclei, total_umi, computed_library_size, upper_quartile,
                    tmm_factor, effective_library_size, n_genes_detected), as.numeric))
}

# --- gene availability -----------------------------------------------------

#' Exact-symbol availability of each signature inside each released lineage.
#'
#' Matching is exact on the released `gene_symbol_or_name`. No fuzzy matching,
#' no case folding, no alias resolution: a symbol that is not in the release is
#' reported missing rather than mapped to something similar.
signature_availability <- function(signature_genes, gene_catalog,
                                   detect_fraction = 0.10) {
  lineages <- unique(gene_catalog$lineage)
  crossing(signature_name = unique(signature_genes$signature_name), lineage = lineages) %>%
    left_join(signature_genes, by = "signature_name", relationship = "many-to-many") %>%
    left_join(
      gene_catalog %>%
        select(lineage, gene_symbol_or_name, fraction_nonzero),
      by = c("lineage", "gene_symbol" = "gene_symbol_or_name")
    ) %>%
    group_by(signature_name, lineage) %>%
    summarise(
      n_defined = n_distinct(gene_symbol),
      n_present = sum(!is.na(fraction_nonzero)),
      n_detectably_expressed = sum(!is.na(fraction_nonzero) & fraction_nonzero >= detect_fraction),
      fraction_present = n_present / n_defined,
      missing_genes = paste(sort(gene_symbol[is.na(fraction_nonzero)]), collapse = "; "),
      .groups = "drop"
    )
}

# --- repeated-measurement reliability --------------------------------------

#' One-way random-effects ICC(1,1) with unequal group sizes.
#'
#' Groups are donor x region x supertype; the replicate units inside a group are
#' sample/library rows, which are repeated *measurements*, not replicate donors.
#' Returns NA rather than a number when fewer than two groups have replicates.
icc_oneway <- function(value, group) {
  keep <- !is.na(value) & !is.na(group)
  value <- value[keep]
  group <- as.character(group[keep])
  sizes <- table(group)
  if (length(sizes) < 2L || sum(sizes > 1L) < 2L) {
    return(tibble(icc = NA_real_, n_groups = length(sizes), n_rows = length(value),
                  var_between = NA_real_, var_within = NA_real_))
  }
  k <- length(sizes)
  n <- length(value)
  grand <- mean(value)
  means <- tapply(value, group, mean)
  ss_between <- sum(as.numeric(sizes) * (means - grand)^2)
  ss_within <- sum((value - means[group])^2)
  df_within <- n - k
  ms_between <- ss_between / (k - 1)
  ms_within <- if (df_within > 0) ss_within / df_within else NA_real_
  k0 <- (n - sum(as.numeric(sizes)^2) / n) / (k - 1)
  icc <- (ms_between - ms_within) / (ms_between + (k0 - 1) * ms_within)
  tibble(
    icc = as.numeric(icc),
    n_groups = k,
    n_rows = n,
    var_between = as.numeric(max((ms_between - ms_within) / k0, 0)),
    var_within = as.numeric(ms_within)
  )
}

#' Every within-group pair of repeated sample/library measurements.
#'
#' One row per unordered pair. `assay_pair` records whether the two measurements
#' share an assay method; `support_min` is the weaker of the two measurements,
#' because a pair is only as reliable as its weaker member.
replicate_pairs <- function(scores, value_col = "score_z_mean") {
  scores %>%
    group_by(across(all_of(STATE_GROUP_KEYS))) %>%
    filter(n() > 1) %>%
    arrange(prepared_obs_index, .by_group = TRUE) %>%
    group_modify(function(.x, .y) {
      idx <- utils::combn(nrow(.x), 2)
      tibble(
        row_a = .x$prepared_obs_index[idx[1, ]],
        row_b = .x$prepared_obs_index[idx[2, ]],
        value_a = .x[[value_col]][idx[1, ]],
        value_b = .x[[value_col]][idx[2, ]],
        method_a = .x$assay_method[idx[1, ]],
        method_b = .x$assay_method[idx[2, ]],
        nuclei_a = .x$n_nuclei[idx[1, ]],
        nuclei_b = .x$n_nuclei[idx[2, ]],
        umi_a = .x$total_umi[idx[1, ]],
        umi_b = .x$total_umi[idx[2, ]],
        domain = .x$domain[idx[1, ]],
        lineage = .x$lineage[idx[1, ]]
      )
    }) %>%
    ungroup() %>%
    mutate(
      difference = value_a - value_b,
      abs_difference = abs(difference),
      mean_value = (value_a + value_b) / 2,
      assay_pair = if_else(method_a == method_b, "same method", "mixed method"),
      method_pair = if_else(method_a == method_b, method_a,
                            paste(pmin(method_a, method_b), pmax(method_a, method_b), sep = " vs ")),
      support_min = pmin(nuclei_a, nuclei_b),
      support_max = pmax(nuclei_a, nuclei_b),
      umi_min = pmin(umi_a, umi_b)
    )
}

#' Bland-Altman style agreement summary for a set of replicate pairs.
agreement_summary <- function(pairs, by = c("signature_name", "assay_pair")) {
  pairs %>%
    group_by(across(all_of(by))) %>%
    summarise(
      n_pairs = n(),
      median_abs_difference = median(abs_difference, na.rm = TRUE),
      mean_difference = mean(difference, na.rm = TRUE),
      sd_difference = sd(difference, na.rm = TRUE),
      loa_lower = mean_difference - 1.96 * sd_difference,
      loa_upper = mean_difference + 1.96 * sd_difference,
      repeatability_sd = sd_difference / sqrt(2),
      pearson_between_members = suppressWarnings(
        cor(c(value_a, value_b), c(value_b, value_a), use = "complete.obs")
      ),
      .groups = "drop"
    )
}

#' Every directly observed assay-method contrast, inside donor x region x supertype.
#'
#' Only mixed-method pairs contribute, so the donor, region and supertype are
#' held fixed by construction and the contrast cannot be confounded by them.
#' Methods are ordered lexicographically and the reported offset is always
#' `second_method - first_method`, so the sign is unambiguous.
method_pair_contrasts <- function(pairs) {
  pairs %>%
    filter(assay_pair == "mixed method") %>%
    mutate(
      first_method  = pmin(method_a, method_b),
      second_method = pmax(method_a, method_b),
      diff_second_minus_first = if_else(method_a == pmax(method_a, method_b),
                                        value_a - value_b, value_b - value_a)
    ) %>%
    group_by(signature_name, domain, first_method, second_method) %>%
    summarise(
      n_pairs = n(),
      median_offset = median(diff_second_minus_first),
      mean_offset = mean(diff_second_minus_first),
      sd_offset = sd(diff_second_minus_first),
      ci_lower = mean_offset - 1.96 * sd_offset / sqrt(n_pairs),
      ci_upper = mean_offset + 1.96 * sd_offset / sqrt(n_pairs),
      wilcoxon_p = tryCatch(stats::wilcox.test(diff_second_minus_first)$p.value,
                            error = function(e) NA_real_),
      .groups = "drop"
    )
}

#' Additive per-method offsets relative to a reference method.
#'
#' Not every method is directly paired with the reference inside a
#' donor x region x supertype: the release's replicate design is a *chain*. The
#' offsets are therefore propagated along the graph of directly observed
#' contrasts, and `n_steps` records how far each method sits from the reference,
#' so a chained estimate is never mistaken for a direct one. A method that is not
#' connected to the reference at all is returned with an NA offset rather than a
#' silent zero.
method_offsets <- function(pairs, reference = "10Xv3.1") {
  contrasts <- method_pair_contrasts(pairs)
  all_methods <- sort(unique(c(pairs$method_a, pairs$method_b)))

  map_dfr(sort(unique(contrasts$signature_name)), function(sig) {
    edges <- filter(contrasts, signature_name == sig)
    offset <- stats::setNames(0, reference)
    steps <- stats::setNames(0L, reference)
    repeat {
      added <- FALSE
      for (i in seq_len(nrow(edges))) {
        from <- edges$first_method[i]
        to <- edges$second_method[i]
        delta <- edges$median_offset[i]   # to - from
        if (from %in% names(offset) && !(to %in% names(offset))) {
          offset[to] <- offset[[from]] + delta
          steps[to] <- steps[[from]] + 1L
          added <- TRUE
        } else if (to %in% names(offset) && !(from %in% names(offset))) {
          offset[from] <- offset[[to]] - delta
          steps[from] <- steps[[to]] + 1L
          added <- TRUE
        }
      }
      if (!added) break
    }
    tibble(
      signature_name = sig,
      assay_method = all_methods,
      method_offset = unname(offset[all_methods]),
      n_steps_from_reference = unname(steps[all_methods]),
      reference_method = reference
    )
  })
}

# --- support ---------------------------------------------------------------

#' Fixed, pre-declared support bins. These are bins for *describing* the
#' reliability curve, not a filter and not a threshold.
SUPPORT_BINS <- c(0, 2, 5, 10, 20, 50, 100, 200, 500, Inf)

bin_support <- function(x, breaks = SUPPORT_BINS) {
  cut(x, breaks = breaks, right = TRUE, include.lowest = TRUE, dig.lab = 6)
}

#' Reliability of replicate agreement as a function of the weaker measurement.
support_reliability <- function(pairs, support_col = "support_min",
                                breaks = SUPPORT_BINS) {
  pairs %>%
    mutate(support_bin = bin_support(.data[[support_col]], breaks)) %>%
    group_by(signature_name, domain, support_bin) %>%
    summarise(
      n_pairs = n(),
      median_abs_difference = median(abs_difference, na.rm = TRUE),
      repeatability_sd = sd(difference, na.rm = TRUE) / sqrt(2),
      .groups = "drop"
    )
}

# --- aggregation policies --------------------------------------------------

#' Convert repeated sample/library measurements into one value per
#' donor x region x supertype, under several explicitly named policies.
#'
#' No policy is selected here. `assay_adjusted` subtracts a signature-specific,
#' additive method offset estimated *only* from mixed-method replicate pairs, so
#' the adjustment never uses between-donor variation.
aggregate_policies <- function(scores, offsets = NULL, value_col = "score_z_mean") {
  base <- scores %>%
    rename(value = all_of(value_col)) %>%
    group_by(across(all_of(STATE_GROUP_KEYS)), lineage, domain)

  equal <- base %>%
    summarise(
      n_measurements = n(),
      n_nuclei_total = sum(n_nuclei),
      total_umi_total = sum(total_umi),
      n_assay_methods = n_distinct(assay_method),
      assay_methods = paste(sort(unique(assay_method)), collapse = ";"),
      equal_weight_mean = mean(value),
      nucleus_weighted_mean = if (sum(n_nuclei) > 0) sum(value * n_nuclei) / sum(n_nuclei) else NA_real_,
      umi_weighted_mean = if (sum(total_umi) > 0) sum(value * total_umi) / sum(total_umi) else NA_real_,
      within_group_sd = if (n() > 1) sd(value) else NA_real_,
      within_group_range = if (n() > 1) diff(range(value)) else NA_real_,
      .groups = "drop"
    )

  if (is.null(offsets)) return(equal)

  adjusted <- scores %>%
    rename(value = all_of(value_col)) %>%
    left_join(offsets, by = c("signature_name", "assay_method")) %>%
    mutate(value_adjusted = value - coalesce(method_offset, 0)) %>%
    group_by(across(all_of(STATE_GROUP_KEYS))) %>%
    summarise(
      assay_adjusted_mean = mean(value_adjusted),
      assay_adjusted_nucleus_weighted_mean = dplyr::if_else(
        sum(n_nuclei) > 0,
        sum(value_adjusted * n_nuclei) / max(sum(n_nuclei), 1),
        NA_real_
      ),
      .groups = "drop"
    )

  equal %>% left_join(adjusted, by = STATE_GROUP_KEYS)
}

#' Agreement between two aggregation policies, per signature.
policy_agreement <- function(aggregated, a, b) {
  aggregated %>%
    filter(!is.na(.data[[a]]), !is.na(.data[[b]])) %>%
    group_by(signature_name) %>%
    summarise(
      n = n(),
      pearson = cor(.data[[a]], .data[[b]]),
      spearman = cor(.data[[a]], .data[[b]], method = "spearman"),
      median_abs_difference = median(abs(.data[[a]] - .data[[b]])),
      max_abs_difference = max(abs(.data[[a]] - .data[[b]])),
      .groups = "drop"
    ) %>%
    mutate(policy_a = a, policy_b = b, .before = n)
}

# --- composition, kept separate from expression state ----------------------

#' Nuclei and relative abundance per donor x region x supertype.
#'
#' The denominator is the nuclei of the four downloaded glial lineages in that
#' donor x region, *not* all cells: no neuronal object was downloaded. These are
#' naive proportions carrying dissociation and sampling effects; they are not an
#' inferential composition model, and they are never combined with a state score.
composition_table <- function(scores_or_rows) {
  scores_or_rows %>%
    distinct(prepared_obs_index, lineage, domain, donor_id, brain_region, supertype, n_nuclei) %>%
    group_by(donor_id, brain_region, lineage, domain, supertype) %>%
    summarise(n_nuclei = sum(n_nuclei), .groups = "drop") %>%
    group_by(donor_id, brain_region) %>%
    mutate(
      n_nuclei_glial_total = sum(n_nuclei),
      relative_abundance = n_nuclei / n_nuclei_glial_total
    ) %>%
    ungroup() %>%
    mutate(
      abundance_denominator = "nuclei of Immune + Astrocyte + Oligodendrocyte + OPC in this donor x region",
      abundance_caveat = "naive proportion; not a compositional model; no neuronal denominator"
    )
}

# --- disease-stage QC joins ------------------------------------------------

#' The published donor x region pseudo-progression scores, renamed to the omics keys.
read_local_cps <- function(path) {
  read_csv(path, show_col_types = FALSE, progress = FALSE) %>%
    select(donor_id = `Donor ID`, brain_region = `Brain Region`,
           CPS_Local, CPS_Local_ABeta, CPS_Local_pTau,
           CPS_Global, CPS_Global_ABeta, CPS_Global_pTau) %>%
    distinct()
}
