# ---------------------------------------------------------------------------
# Helpers for the SEA-AD molecular-state *decision* notebook
# (results/analysis/03_seaad_omics_analysis_ready.qmd).
#
# Scope: reduce the exploratory state derivatives written by
# `seaad-omics-state-scores` and `seaad-omics-competitive-scores` to a small set
# of donor x region molecular-state variables, and characterise what they still
# contain once global stage and the existing P/H measurements are accounted for.
#
# This file is deliberately thin. Normalization, scoring, replicate reliability,
# the assay offset and the aggregation policies are all settled in
# `R/seaad_state.R` and are reused unchanged; nothing here re-derives them.
#
# Two things here that `R/seaad_state.R` does not do:
#
#   1. it collapses the supertype stratum, under one stated rule, because the
#      eventual T block is per donor x region and supertype cannot survive into
#      it;
#   2. it writes one derivative file, the analysis-ready molecular-state table.
#      That is the only place in R where this project writes to data/.
#
# Nothing here joins morphology or cognition, and nothing here selects a
# signature by its association with anything.
# ---------------------------------------------------------------------------

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(purrr)
  library(tibble)
  library(readr)
  library(splines)
})

# --- paths -----------------------------------------------------------------

#' The competitive-score derivatives, and where the analysis table is written.
seaad_analysis_paths <- function(root = find_project_root()) {
  state <- file.path(root, "data", "derivatives", "sea-ad", "omics_state")
  list(
    state_dir           = state,
    competitive_scores  = file.path(state, "row_competitive_scores.csv"),
    background_summary  = file.path(state, "background_matching_summary.csv"),
    competitive_prov    = file.path(state, "competitive_scoring_provenance.json"),
    supplements_dir     = file.path(state, "supplements"),
    analysis_table      = file.path(state, "molecular_state_analysis_table.csv"),
    analysis_prov       = file.path(state, "molecular_state_analysis_provenance.json")
  )
}

#' Read the row-level competitive scores, typed, at the same grain as the
#' primary row-level scores.
read_competitive_scores <- function(path) {
  read_csv(path, show_col_types = FALSE, progress = FALSE) %>%
    mutate(
      n_nuclei  = as.numeric(n_nuclei),
      total_umi = as.numeric(total_umi),
      across(c(score_z_mean, background_mean_z, background_sd_z, competitive_score),
             as.numeric),
      measurement_grain = "donor x region x supertype x sample/library"
    )
}

# --- support ---------------------------------------------------------------

#' The support bands used throughout notebook 02, restated so the analysis table
#' carries the same labels. These are descriptive bands, not a filter.
support_status_of <- function(n_nuclei) {
  case_when(
    n_nuclei >= 20 ~ "well supported",
    n_nuclei >= 10 ~ "marginal support",
    TRUE           ~ "low support"
  )
}

# --- collapsing the supertype stratum --------------------------------------

#' Collapse donor x region x supertype estimates to donor x region x lineage.
#'
#' The rule is the nucleus-weighted mean across the lineage's supertypes, which
#' is the same weighting already justified for combining repeated sample/library
#' measurements: a supertype observed in four nuclei is not evidence on a par
#' with one observed in four hundred.
#'
#' The consequence is stated rather than hidden. A nucleus-weighted lineage score
#' is an *abundance-weighted* average of within-supertype states, so a donor whose
#' Micro/PVM pool is dominated by one supertype is represented mostly by that
#' supertype. That is the intended reading - the covariate is the state of the
#' lineage as sampled - but it means the lineage score is not independent of
#' composition, and `n_supertypes` plus `dominant_supertype_share` are carried so
#' a later model can see how concentrated each value is.
collapse_supertypes_to_lineage <- function(aggregated,
                                           value_cols = c("state_score", "competitive_score"),
                                           weight_col = "n_nuclei_total") {
  present <- intersect(value_cols, names(aggregated))
  if (!length(present)) stop("none of `value_cols` is present in `aggregated`")

  aggregated %>%
    rename(supertype_weight = all_of(weight_col)) %>%
    group_by(donor_id, brain_region, lineage, domain, signature_name) %>%
    summarise(
      across(all_of(present),
             ~ if (sum(supertype_weight[!is.na(.x)]) > 0) {
                 stats::weighted.mean(.x, supertype_weight, na.rm = TRUE)
               } else NA_real_),
      n_nuclei = sum(supertype_weight, na.rm = TRUE),
      n_measurements = sum(n_measurements, na.rm = TRUE),
      n_supertypes = n(),
      dominant_supertype_share = if (sum(supertype_weight, na.rm = TRUE) > 0) {
        max(supertype_weight, na.rm = TRUE) / sum(supertype_weight, na.rm = TRUE)
      } else NA_real_,
      .groups = "drop"
    ) %>%
    mutate(support_status = support_status_of(n_nuclei))
}

# --- redundancy against stage and the existing P/H block -------------------

#' Terms of the exploratory characterisation model, as named blocks.
#'
#' `mgcv` is absent from this project's R environment, so the smooth in
#' `T ~ ROI + s(CPS) + Ab + pTau + NeuN + GFAP` is a natural cubic spline with
#' three degrees of freedom. The purpose is identical: let stage enter
#' non-linearly so that a monotone but curved dependence on CPS is not left in
#' the residual and mistaken for independent molecular variation.
REDUNDANCY_BLOCKS <- list(
  ROI  = "brain_region",
  CPS  = "ns(CPS_Local, df = 3)",
  P    = c("abeta_percent_positive_area", "ptau_percent_positive_area"),
  H    = c("neun_cells_per_area", "gfap_percent_positive_area")
)

#' Fit the characterisation model and report how much of T it accounts for.
#'
#' Returns one row per response with the SD of the raw score, the SD of its
#' residual, the full model R^2, and the *drop* in R^2 when each block is
#' removed from the full model. Those drops are unique contributions: they do not
#' sum to the full R^2 when the blocks are correlated, which they are.
#'
#' This is a characterisation tool. No signature is selected by it, no p-value is
#' reported from it, and the residual is never promoted to the canonical
#' molecular variable.
redundancy_fit <- function(data, response, blocks = REDUNDANCY_BLOCKS) {
  terms_all <- unlist(blocks, use.names = FALSE)
  needed <- c(response, "brain_region", "CPS_Local",
              unlist(blocks[c("P", "H")], use.names = FALSE))
  complete <- data %>% filter(if_all(all_of(needed), ~ !is.na(.x)))
  if (nrow(complete) < 20 || dplyr::n_distinct(complete$brain_region) < 2) {
    return(tibble(response = response, n = nrow(complete)))
  }

  formula_of <- function(terms) {
    stats::as.formula(paste(response, "~", paste(terms, collapse = " + ")))
  }
  full <- stats::lm(formula_of(terms_all), data = complete)
  r2 <- function(model) summary(model)$r.squared

  drops <- map_dbl(names(blocks), function(block) {
    kept <- unlist(blocks[setdiff(names(blocks), block)], use.names = FALSE)
    if (!length(kept)) return(r2(full))
    r2(full) - r2(stats::lm(formula_of(kept), data = complete))
  })
  names(drops) <- paste0("unique_r2_", names(blocks))

  tibble(
    response = response,
    n = nrow(complete),
    n_donors = dplyr::n_distinct(complete$donor_id),
    n_regions = dplyr::n_distinct(complete$brain_region),
    sd_raw = stats::sd(complete[[response]]),
    sd_residual = stats::sd(stats::residuals(full)),
    r2_full = r2(full),
    fraction_variance_remaining = 1 - r2(full),
    !!!as.list(drops)
  )
}

#' Pairwise correlation of one score against each named covariate.
#'
#' Spearman, complete pairs, with the number of pairs kept beside every
#' coefficient so a correlation over thirty donors is never read as one over
#' three hundred.
correlate_with <- function(data, response, covariates) {
  map_dfr(covariates, function(covariate) {
    ok <- !is.na(data[[response]]) & !is.na(data[[covariate]])
    tibble(
      response = response,
      covariate = covariate,
      n = sum(ok),
      spearman = if (sum(ok) >= 5) {
        suppressWarnings(stats::cor(data[[response]][ok], data[[covariate]][ok],
                                    method = "spearman"))
      } else NA_real_,
      pearson = if (sum(ok) >= 5) {
        suppressWarnings(stats::cor(data[[response]][ok], data[[covariate]][ok]))
      } else NA_real_
    )
  })
}
