# ---------------------------------------------------------------------------
# Helpers for the multiregional resilience notebook
# (results/analysis/05_seaad_multiregional_resilience.qmd).
#
# Scope: the few operations notebooks 00-04 do not already provide - a
# within-region rank transform, the donor-wide / regional-deviation split, a
# balanced variance decomposition, a donor-cluster bootstrap, and per-donor
# summaries of the harmonized cognitive visits. Nothing here reads a source
# file, writes to data/, rescores a molecular state or redefines CPS.
#
# Why no mixed-model package: neither nlme nor lme4 is installed in the
# r-quarto env. The donor random intercept of
#   y ~ ROI + donor-wide x + regional deviation of x + (1 | donor)
# is handled by the Mundlak decomposition instead: the coefficient on the
# within-donor deviation is the donor fixed-effects estimate, the coefficient
# on the donor-wide term is the between-donor association, and uncertainty
# comes from resampling donors, never donor x ROI rows.
# ---------------------------------------------------------------------------

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(purrr)
  library(tibble)
})

#' Rank-based normal score (Blom), computed on whatever vector is passed.
#'
#' Call it inside `group_by(brain_region)` so that each stain keeps its own
#' region and denominator: the result says where a donor sits among all donors
#' measured in that region, on a normal scale, and it bounds the influence of
#' floor values and implausible extremes without dropping them.
normal_score <- function(x) {
  n <- sum(!is.na(x))
  r <- rank(x, na.last = "keep", ties.method = "average")
  stats::qnorm((r - 3 / 8) / (n + 1 / 4))
}

#' Split region-level variables into a donor-wide mean and a regional deviation.
#'
#' The donor mean is taken over the rows passed in, so the caller decides which
#' regions define "donor-wide" (here: the fixed MTG / DFC / MEC set, from a
#' table in which every donor has all three). Adds `<var>_donor` and `<var>_dev`.
within_between <- function(df, vars, donor = "donor_id") {
  df %>%
    group_by(.data[[donor]]) %>%
    mutate(across(all_of(vars), list(donor = ~ mean(.x), dev = ~ .x - mean(.x)),
                  .names = "{.col}_{.fn}")) %>%
    ungroup()
}

#' Variance components of one variable over a balanced donor x ROI table.
#'
#' Two-way ANOVA without replication (one value per donor x ROI). ROI is a
#' fixed set, so its component is the variance of the ROI means net of noise;
#' "donor" is between-donor variance after ROI means are removed; "donor x ROI"
#' is everything a donor's regions do not share - regional deviation plus
#' measurement error, which this design cannot separate.
variance_components <- function(df, var, donor = "donor_id", roi = "brain_region") {
  d <- df %>% select(donor = all_of(donor), roi = all_of(roi), y = all_of(var)) %>%
    filter(!is.na(y))
  k <- n_distinct(d$roi)
  per_donor <- count(d, donor)
  stopifnot("variance_components needs a balanced table" = all(per_donor$n == k))
  n <- nrow(per_donor)
  grand <- mean(d$y)
  ss_donor <- k * sum((tapply(d$y, d$donor, mean) - grand)^2)
  ss_roi <- n * sum((tapply(d$y, d$roi, mean) - grand)^2)
  ss_res <- sum((d$y - grand)^2) - ss_donor - ss_roi
  ms_donor <- ss_donor / (n - 1)
  ms_roi <- ss_roi / (k - 1)
  ms_res <- ss_res / ((n - 1) * (k - 1))
  comp <- c(ROI = max((ms_roi - ms_res) / n, 0),
            donor = max((ms_donor - ms_res) / k, 0),
            `donor x ROI` = ms_res)
  tibble(variable = var, n_donors = n, n_regions = k,
         component = names(comp), variance = unname(comp),
         share = unname(comp) / sum(comp),
         donor_share_after_roi = comp[["donor"]] / (comp[["donor"]] + comp[["donor x ROI"]]))
}

#' Donor-cluster bootstrap of any statistic.
#'
#' `stat_fun(data)` must return a named numeric vector. Donors are drawn with
#' replacement and each draw is relabelled, so a donor drawn twice contributes
#' two independent copies of all its regions - the donor, not the donor x ROI
#' row, is the resampling unit. Percentile intervals.
cluster_bootstrap <- function(data, stat_fun, B = 2000, cluster = "donor_id", seed = 1L) {
  est <- stat_fun(data)
  ids <- unique(data[[cluster]])
  rows_by_id <- split(seq_len(nrow(data)), data[[cluster]])
  set.seed(seed)
  draws <- matrix(NA_real_, nrow = B, ncol = length(est), dimnames = list(NULL, names(est)))
  for (b in seq_len(B)) {
    pick <- sample(ids, length(ids), replace = TRUE)
    idx <- unlist(rows_by_id[pick], use.names = FALSE)
    d <- data[idx, , drop = FALSE]
    d[[cluster]] <- rep(seq_along(pick), lengths(rows_by_id[pick]))
    out <- tryCatch(stat_fun(d), error = function(e) rep(NA_real_, length(est)))
    if (length(out) == length(est)) draws[b, ] <- out
  }
  tibble(term = names(est), estimate = unname(est),
         lo = apply(draws, 2, stats::quantile, 0.025, na.rm = TRUE),
         hi = apply(draws, 2, stats::quantile, 0.975, na.rm = TRUE),
         boot_sd = apply(draws, 2, stats::sd, na.rm = TRUE),
         n_boot_ok = colSums(!is.na(draws)))
}

#' Per-donor summary of harmonized visit-level domain scores.
#'
#' `visits` has one row per donor x visit with `age_vis`, `age` (at death) and
#' the domain column. Everything returned is descriptive of the visits actually
#' observed: last observed score and its timing, the mean over a common window
#' before death, and an ordinary least-squares rate that is used only to check
#' what the released slope variables are on this scale - never as a substitute
#' for them.
donor_domain_summary <- function(visits, domain, window_years = 10) {
  visits %>%
    filter(!is.na(.data[[domain]])) %>%
    mutate(ybd = age - age_vis, y = .data[[domain]]) %>%
    group_by(donor_id) %>%
    summarise(
      domain = domain,
      n_visits = n(),
      first_ybd = max(ybd),
      last_ybd = min(ybd),
      span_years = max(age_vis) - min(age_vis),
      last_score = y[which.max(age_vis)],
      n_window = sum(ybd <= window_years),
      window_mean = if (sum(ybd <= window_years) >= 2) mean(y[ybd <= window_years]) else NA_real_,
      ols_rate = if (n_distinct(age_vis) >= 2) unname(coef(lm(y ~ age_vis))[2]) else NA_real_,
      .groups = "drop"
    )
}
