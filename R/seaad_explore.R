# ---------------------------------------------------------------------------
# Small helpers for exploring the public SEA-AD tabular releases.
#
# Scope: read-only inspection of the files under data/raw/sea-ad/. Nothing here
# writes, caches or derives anything; it exists purely so that
# results/analysis/00_seaad_exploration.qmd stays readable.
# ---------------------------------------------------------------------------

suppressPackageStartupMessages({
  library(tibble)
  library(dplyr)
  library(tidyr)
  library(purrr)
  library(stringr)
  library(readxl)
})

# --- paths -----------------------------------------------------------------

#' Walk up from `start` until a directory containing `.git` is found.
find_project_root <- function(start = getwd()) {
  p <- normalizePath(start, mustWork = TRUE)
  while (!dir.exists(file.path(p, ".git"))) {
    parent <- dirname(p)
    if (identical(parent, p)) stop("Could not locate project root (no .git above ", start, ")")
    p <- parent
  }
  p
}

#' Resolve the public SEA-AD tabular files.
#'
#' The Brain Knowledge Platform download directory is named with an opaque
#' hash, so it is globbed rather than hard-coded.
seaad_paths <- function(root = find_project_root()) {
  meta_dir <- Sys.glob(file.path(root, "data", "raw", "sea-ad", "*_specimen_metadata"))
  if (length(meta_dir) != 1L) {
    stop("Expected exactly one *_specimen_metadata directory, found ", length(meta_dir))
  }
  p <- list(
    root                = root,
    specimen_metadata   = meta_dir,
    qnp_2026            = file.path(meta_dir, "6a4579d1d67150b0e69ad044_sea-ad-quantitative-neuropathology-063026.xlsx"),
    donor_metadata      = file.path(meta_dir, "68debdfdd1b8e9f8fd64dab0_sea-ad_cohort_donor_metadata_072524.xlsx"),
    cognition           = file.path(meta_dir, "68debdfd4748b7546943a7b4_sea-ad_cohort_harmonized_cognitive_scores_20241213.xlsx"),
    specimen_index      = file.path(meta_dir, "SpecimenMetadata.csv"),
    mri_volumetrics     = file.path(meta_dir, "68debdfdae5f82b97af2fb0f_sea-ad_cohort_mri_volumetrics.xlsx"),
    luminex_mtg         = file.path(meta_dir, "68debdff5b8003454786ea29_sea-ad_cohort_mtg-tissue_extractions-luminex_data.xlsx"),
    imaging_availability = file.path(meta_dir, "68debdfcf51ccbd2e5ca83d5_seaad_donor_available_imaging_information071725.xlsx"),
    caudate_pre2026     = file.path(meta_dir, "pre_june_2026",
                                    "694b138ad3882eea00667f44_SEA-AD_CaudateNucleus_Quant-Neuropath-Summary_122325.csv"),
    mtg_qnp_pre2026     = file.path(meta_dir, "pre_june_2026",
                                    "68debdfd24606956df13f2dd_sea-ad_all_mtg_quant_neuropath_bydonorid_081122.csv"),
    # "Information on neuropathology images and image analysis data" (June 2026
    # revision). The copy inside the download bundle is the earlier revision and
    # lacks the data-dictionary section, so the current one is kept separately.
    qnp_white_paper     = file.path(root, "data", "raw", "sea-ad", "documentation",
                                    "6a4584cb0e0018d98867f61c_sea-ad_quantitativeneuropathology_awsdocumentationtutorial_063026.pdf"),
    qnp_white_paper_pre2026 = file.path(meta_dir,
                                    "68deba1b30760d231939c74e_sea-ad_quantitativeneuropathology_awsdocumentationtutorial.pdf")
  )
  p
}

# --- QNP workbook ----------------------------------------------------------

QNP_ID_COLS <- c("Donor ID", "brain region", "analysis region")

#' Read every data sheet of the QNP workbook into a named list of tibbles.
#'
#' The README sheet is free text and is returned separately by
#' `read_qnp_readme()`.
read_qnp_workbook <- function(path, skip_sheets = "README") {
  sheets <- setdiff(excel_sheets(path), skip_sheets)
  out <- map(sheets, ~ suppressMessages(read_excel(path, sheet = .x, guess_max = 1e5)))
  set_names(out, sheets)
}

#' Return the README sheet as a character vector of non-empty lines.
read_qnp_readme <- function(path) {
  raw <- suppressMessages(read_excel(path, sheet = "README", col_names = FALSE,
                                     .name_repair = "minimal"))
  lines <- apply(raw, 1, function(r) paste(r[!is.na(r)], collapse = " "))
  lines[nzchar(trimws(lines))]
}

#' One-row-per-sheet inventory of a QNP sheet.
summarize_qnp_sheet <- function(df, sheet, markers = qnp_marker_patterns()) {
  metric_cols <- setdiff(names(df), QNP_ID_COLS)
  brain <- if ("brain region" %in% names(df)) sort(unique(df[["brain region"]])) else NA_character_
  analysis <- if ("analysis region" %in% names(df)) sort(unique(df[["analysis region"]])) else NA_character_
  present <- names(markers)[map_lgl(markers, ~ any(str_detect(metric_cols, regex(.x, ignore_case = TRUE))))]

  tibble(
    sheet       = sheet,
    n_rows      = nrow(df),
    n_donors    = dplyr::n_distinct(df[["Donor ID"]]),
    brain_region = paste(brain, collapse = ", "),
    resolution  = resolution_label(analysis),
    n_analysis_regions = length(analysis[!is.na(analysis)]),
    analysis_regions = paste(analysis, collapse = ", "),
    n_metric_cols = length(metric_cols),
    markers     = paste(present, collapse = ", ")
  )
}

#' Describe the anatomical resolution implied by a sheet's `analysis region`
#' levels. Deliberately descriptive: nothing is collapsed here.
resolution_label <- function(analysis_levels) {
  if (all(is.na(analysis_levels))) return("not stated in sheet")
  lv <- analysis_levels[!is.na(analysis_levels)]
  parts <- c(
    if (any(lv == "Grey matter")) "whole-region grey matter",
    if (any(str_detect(lv, "^Layer"))) "cortical layer",
    if (any(str_detect(lv, "^(CA[1-4]|DG|Sub)"))) "hippocampal subfield",
    if (any(str_detect(lv, "^(EC|TEC)"))) "entorhinal subregion",
    if (any(str_detect(lv, "-(L|R)$"))) "hemisphere-resolved"
  )
  if (!length(parts)) return(paste("other:", paste(lv, collapse = ", ")))
  paste(parts, collapse = " + ")
}

#' Long table of every (sheet, analysis region) cell, so that anatomical grain
#' stays visible rather than being averaged away.
qnp_region_grid <- function(qnp) {
  imap_dfr(qnp, function(df, sheet) {
    if (!all(c("brain region", "analysis region") %in% names(df))) {
      return(tibble(sheet = sheet, brain_region = NA_character_,
                    analysis_region = NA_character_,
                    n_rows = nrow(df), n_donors = dplyr::n_distinct(df[["Donor ID"]])))
    }
    df %>%
      group_by(brain_region = .data[["brain region"]],
               analysis_region = .data[["analysis region"]]) %>%
      summarise(n_rows = dplyr::n(),
                n_donors = dplyr::n_distinct(.data[["Donor ID"]]), .groups = "drop") %>%
      mutate(sheet = sheet, .before = 1)
  })
}

# --- marker / metric parsing ----------------------------------------------

#' Explicit regexes for the five markers currently in scope.
#'
#' These are matched against source column names and the matches are always
#' printed in the notebook before anything is selected.
qnp_marker_patterns <- function() {
  c(
    "Ab (6E10)"   = "6e10",
    "pTau (AT8)"  = "\\bAT8\\b",
    "NeuN"        = "NeuN",
    "IBA1"        = "Iba1",
    "GFAP"        = "GFAP",
    "pTDP-43"     = "pTDP43",
    "a-synuclein" = "aSyn",
    "Hematoxylin" = "[Hh]ematoxylin"
  )
}

#' Map each column name to the marker(s) whose pattern it matches.
#'
#' Returns one row per (column, marker) pair plus a `n_markers` flag so that
#' colocalization columns are obvious rather than silently attributed to one
#' marker.
match_marker_columns <- function(cols, patterns = qnp_marker_patterns()) {
  hits <- imap_dfr(patterns, function(pat, marker) {
    tibble(column = cols[str_detect(cols, regex(pat, ignore_case = TRUE))],
           marker = marker, pattern = pat)
  })
  hits %>%
    group_by(column) %>%
    mutate(n_markers = dplyr::n(),
           is_colocalization = n_markers > 1L) %>%
    ungroup() %>%
    arrange(marker, column)
}

#' Strip the `_<analysis region>` suffix used by the Gabitto 2024 sheet.
#'
#' That sheet encodes anatomical resolution in the column name
#' (`percent AT8 positive area_Layer3`) instead of an `analysis region` column.
#' Returns the base column name and the suffix separately so that neither is
#' lost.
qnp_split_region_suffix <- function(col) {
  suffix_re <- "_(Grey matter|Layer[0-9](-[0-9])?)$"
  tibble(
    column = col,
    base_column = str_remove(col, suffix_re),
    region_suffix = str_remove(str_extract(col, suffix_re), "^_")
  )
}

#' Classify a QNP column name into a metric family and a unit.
#'
#' Units come from Table 6 of the SEA-AD quantitative-neuropathology AWS
#' documentation PDF shipped alongside the workbook. Rules are ordered and
#' anything unmatched is returned as "unclassified" rather than guessed.
#'
#' Gabitto-style `_Layer3` suffixes are stripped before matching so that, e.g.,
#' `number of AT8 positive cells per area_Layer3` is classified as a density
#' rather than falling through to the generic "raw count" rule.
qnp_metric_family <- function(col) {
  split <- qnp_split_region_suffix(col)
  classify_base(split$base_column) %>%
    transmute(column = split$column,
              region_suffix = split$region_suffix,
              metric, unit)
}

classify_base <- function(col) {
  rules <- tibble::tribble(
    ~pattern,                                  ~metric,                             ~unit,
    "area analyzed$",                          "denominator: tissue area analysed", "um^2",
    "per tissue area|per_tissue_area",         "density (per analysed area)",       "count per um^2",
    "^percent of .*(colocalized|co-localized)", "colocalization fraction",          "% of objects",
    "^percent .*(plaque|positive) area$",      "positive area fraction",            "% of analysed area",
    "^percent .*co-expression",                "co-expression fraction",            "%",
    "^number of .* per area$",                 "density (per analysed area)",       "count per um^2",
    "^number of .*(colocalized|co-localized)", "colocalized object count",          "count",
    "^number of ",                             "raw count",                         "count",
    "process length per cell$",                "per-cell process length",           "um per cell",
    "process area per cell$",                  "per-cell process area",             "um^2 per cell",
    "median diameter$",                        "mean object diameter",              "um",
    "roundness$",                              "mean object roundness",             "unitless",
    "perimeter$",                              "mean object perimeter",             "um",
    "^average .* area$",                       "mean object area",                  "um^2",
    "^average .* length$",                     "mean branch/process length",        "um",
    "^total .* length$",                       "total process length",              "um",
    "^total .* area$",                         "total positive area",               "um^2",
    "^(ripa|guhcl) ",                          "Luminex biochemical measure",       NA_character_
  )
  map_dfr(col, function(x) {
    i <- which(str_detect(x, regex(rules$pattern, ignore_case = TRUE)))[1]
    if (is.na(i)) {
      tibble(column = x, metric = "unclassified - needs review", unit = NA_character_)
    } else {
      tibble(column = x, metric = rules$metric[i], unit = rules$unit[i])
    }
  })
}

# --- coverage / distribution ----------------------------------------------

#' Per-column coverage for an arbitrary donor-keyed table.
summarize_variable_coverage <- function(df, id_col = "Donor ID", columns = NULL) {
  cols <- columns %||% setdiff(names(df), id_col)
  map_dfr(cols, function(cc) {
    v <- df[[cc]]
    tibble(
      source_column = cc,
      type          = class(v)[1],
      n_rows        = length(v),
      n_donors      = dplyr::n_distinct(df[[id_col]]),
      n_nonmissing  = sum(!is.na(v)),
      n_donors_nonmissing = dplyr::n_distinct(df[[id_col]][!is.na(v)]),
      pct_missing   = round(100 * mean(is.na(v)), 1),
      example       = paste(utils::head(unique(as.character(v[!is.na(v)])), 3), collapse = "; ")
    )
  })
}

#' Compact numeric distribution summary.
numeric_distribution <- function(x) {
  v <- x[!is.na(x)]
  if (!length(v)) {
    return(tibble(min = NA_real_, q25 = NA_real_, median = NA_real_,
                  q75 = NA_real_, max = NA_real_, n_zero = NA_integer_))
  }
  q <- stats::quantile(v, c(0, .25, .5, .75, 1), names = FALSE)
  tibble(min = q[1], q25 = q[2], median = q[3], q75 = q[4], max = q[5],
         n_zero = sum(v == 0))
}

#' Candidate-feature table: one row per (sheet, analysis region, column).
#'
#' `markers` restricts the output to columns matching the given marker names;
#' the matching itself is done by `match_marker_columns()` and the caller is
#' expected to display those matches first.
qnp_candidate_features <- function(qnp, markers, patterns = qnp_marker_patterns(),
                                   include_colocalization = TRUE) {
  imap_dfr(qnp, function(df, sheet) {
    metric_cols <- setdiff(names(df), QNP_ID_COLS)
    hits <- match_marker_columns(metric_cols, patterns[markers])
    if (!include_colocalization) hits <- filter(hits, !is_colocalization)
    if (!nrow(hits)) return(NULL)

    has_region <- all(c("brain region", "analysis region") %in% names(df))
    keyed <- df %>%
      mutate(brain_region = if (has_region) .data[["brain region"]] else NA_character_,
             analysis_region = if (has_region) .data[["analysis region"]] else NA_character_)

    pmap_dfr(list(hits$column, hits$marker, hits$is_colocalization), function(cc, mk, coloc) {
      keyed %>%
        group_by(brain_region, analysis_region) %>%
        group_modify(~ bind_cols(
          tibble(n_donors = dplyr::n_distinct(.x[["Donor ID"]]),
                 n_nonmissing = sum(!is.na(.x[[cc]]))),
          numeric_distribution(.x[[cc]])
        )) %>%
        ungroup() %>%
        mutate(sheet = sheet, source_column = cc, marker = mk,
               is_colocalization = coloc, .before = 1)
    })
  }) %>%
    left_join(qnp_metric_family(unique(.$source_column)) %>% select(column, metric, unit),
              by = c("source_column" = "column")) %>%
    relocate(metric, unit, .after = marker)
}

`%||%` <- function(a, b) if (is.null(a)) b else a

# --- primary regional QNP block -------------------------------------------

# The nine current-release per-region tabs. Deliberately excludes
# "Gabitto 2024 (deprecated; MTG)" (superseded), "Travaglini 2026
# (multi-region)" (a redundant grey-matter view of these nine) and
# "Kana 2026 (CaH)" (a CaH overlay that adds only pTDP-43).
QNP_PRIMARY_SHEETS <- c("AnG", "CaH", "DFC", "FI", "Hip-MEC", "ITG", "MTG",
                        "STG", "VIC-ESOC")

#' Provisional definition of the primary whole-region QNP variables.
#'
#' `block` groups the four into the pathology pair (P: Ab, pTau) and the
#' cellular/host-response pair (H: NeuN, GFAP). IBA1 is deliberately absent:
#' it spans several biologically distinct measurement families and its
#' activation classifier is undocumented, so it is held for secondary use.
qnp_primary_spec <- function() {
  tibble::tribble(
    ~variable,                     ~block, ~marker,      ~source_column,
    "abeta_percent_positive_area", "P",    "Ab (6E10)",  "percent 6e10 positive area",
    "ptau_percent_positive_area",  "P",    "pTau (AT8)", "percent AT8 positive area",
    "neun_cells_per_area",         "H",    "NeuN",       "number of NeuN positive cells per area",
    "gfap_percent_positive_area",  "H",    "GFAP",       "percent GFAP positive area"
  )
}

#' Long donor x brain-region x primary-variable table with an explicit
#' missingness class.
#'
#' Rows come only from the sheet rows at `analysis_region`; layer and subfield
#' rows are never mixed in. A value is never filled from another sheet.
#'
#' `status` distinguishes:
#'   observed                            - a value is present
#'   structurally_unavailable_for_region - the stain has no non-colocalization
#'                                         column at all in that sheet, i.e. it
#'                                         was not provided for the region
#'   source_metric_unavailable           - the stain is present for the region
#'                                         but this particular metric column is
#'                                         not released
#'   assayed_but_missing                 - the column exists and the value is NA
qnp_primary_long <- function(qnp,
                             spec = qnp_primary_spec(),
                             sheets = QNP_PRIMARY_SHEETS,
                             analysis_region = "Grey matter",
                             patterns = qnp_marker_patterns()) {
  imap_dfr(qnp[sheets], function(df, sheet) {
    rows <- df %>% filter(.data[["analysis region"]] == analysis_region)
    if (!nrow(rows)) return(NULL)

    metric_cols <- setdiff(names(df), QNP_ID_COLS)
    markers_in_sheet <- match_marker_columns(metric_cols, patterns) %>%
      filter(!is_colocalization) %>%
      pull(marker) %>%
      unique()

    pmap_dfr(spec, function(variable, block, marker, source_column) {
      col_present <- source_column %in% metric_cols
      marker_present <- marker %in% markers_in_sheet
      value <- if (col_present) rows[[source_column]] else NA_real_
      tibble(
        donor_id        = rows[["Donor ID"]],
        brain_region    = rows[["brain region"]],
        analysis_region = analysis_region,
        source_sheet    = sheet,
        block           = block,
        variable        = variable,
        marker          = marker,
        source_column   = source_column,
        value           = value,
        status = dplyr::case_when(
          !is.na(value)   ~ "observed",
          !marker_present ~ "structurally_unavailable_for_region",
          !col_present    ~ "source_metric_unavailable",
          TRUE            ~ "assayed_but_missing"
        )
      )
    })
  })
}

#' Pivot `qnp_primary_long()` to one row per donor x brain region.
#'
#' Provenance is kept as `source_sheet` plus a `*_status` column per variable;
#' the source column names themselves are constant per variable and are carried
#' by `qnp_primary_spec()`.
qnp_primary_wide <- function(long, spec = qnp_primary_spec()) {
  values <- long %>%
    select(donor_id, brain_region, analysis_region, source_sheet, variable, value) %>%
    pivot_wider(names_from = variable, values_from = value)
  statuses <- long %>%
    select(donor_id, brain_region, variable, status) %>%
    mutate(variable = paste0(variable, "_status")) %>%
    pivot_wider(names_from = variable, values_from = status)

  values %>%
    left_join(statuses, by = c("donor_id", "brain_region")) %>%
    select(donor_id, brain_region, analysis_region, source_sheet,
           all_of(spec$variable), all_of(paste0(spec$variable, "_status")))
}

# ---------------------------------------------------------------------------
# Processed-omics inventory helpers.
#
# These read the small artefacts written by scripts/fetch_seaad_omics_inventory.py
# (an S3 key manifest plus the release's own sub-megabyte summary files). No
# AnnData object is opened here; R in this project has no HDF5 reader, and the
# multiregion AnnData objects are 5-40 GB each.
# ---------------------------------------------------------------------------

#' Resolve the processed-omics inventory files.
seaad_omics_paths <- function(root = find_project_root()) {
  raw <- file.path(root, "data", "raw", "sea-ad", "multiregion_2026")
  der <- file.path(root, "data", "derivatives", "sea-ad", "omics_inventory")
  list(
    raw_dir            = raw,
    derived_dir        = der,
    s3_manifest        = file.path(der, "s3_object_manifest.csv"),
    abundances         = file.path(der, "supertype_abundances_by_library.csv"),
    pseudobulk_probe   = file.path(der, "pseudobulk_probe_obs.csv"),
    provenance         = file.path(der, "provenance.json"),
    bucket_readme      = file.path(raw, "README_bucket.md"),
    multiregion_readme = file.path(raw, "README_Multiregion_2026.md"),
    taxonomy           = file.path(raw, "cluster_colors_new.2026-06-22.csv"),
    cps_by_region      = file.path(raw, "Global_and_Local_CPS.20260105.csv"),
    pertpy_summary     = file.path(raw, "pertpy_summary_CPS_Local.20260622.csv")
  )
}

# The bucket's region folders are not all region labels. `PFC/` carries the DFC
# (Brodmann A9) data under `SEAAD_DFC_*` file names; `DFC/` and `DLPFC/` hold
# only a changelog and a forwarding note. Everything downstream keys on the
# label parsed out of the *file name*, never the folder.
SEAAD_FOLDER_TO_REGION <- c(PFC = "DFC", DLPFC = "DFC")

#' Parse the S3 key manifest into an object-level table.
#'
#' `region` is taken from the `SEAAD_<REGION>_RNAseq` / `_ATACseq` element of the
#' file name where present, and is NA for objects that are not region-specific
#' (the multiregion subclass and pseudobulk objects, model outputs).
read_seaad_s3_manifest <- function(path) {
  readr::read_csv(path, show_col_types = FALSE) %>%
    mutate(
      folder       = str_match(key, "^([^/]+)/")[, 2],
      file         = basename(key),
      size_mb      = size_bytes / 1e6,
      assay        = case_when(str_detect(key, "/RNAseq/")  ~ "snRNAseq",
                               str_detect(key, "/ATACseq/") ~ "snATACseq",
                               TRUE                         ~ "multiregion/model output"),
      region       = str_match(file, "SEAAD_([A-Za-z0-9]+)_(?:RNAseq|ATACseq)")[, 2],
      donor_id     = str_extract(file, "^H\\d{2}\\.\\d{2}\\.\\d{3}"),
      release_date = str_extract(file, "\\d{4}-\\d{2}-\\d{2}"),
      nuclei_set   = case_when(str_detect(file, "all-nuclei")   ~ "all-nuclei",
                               str_detect(file, "final-nuclei") ~ "final-nuclei",
                               TRUE                             ~ NA_character_),
      family = case_when(
        str_detect(key, "donors_objects/")              ~ "per-donor region object",
        str_detect(key, "pseudobulk_objects/")          ~ "pseudobulk (donor x region x supertype)",
        str_detect(key, "subclass_objects/")            ~ "multiregion subclass object",
        str_detect(key, "pertpy_compositional_modeling/") ~ "compositional model output",
        str_detect(key, "gpboost_differential")         ~ "differential-expression model output",
        str_detect(key, "supertype_annotation/|subclass_annotation/") ~ "taxonomy mapping output",
        str_detect(key, "final_embeddings/")            ~ "integrated latent representation",
        str_detect(key, "continuous_pseudo-progression_score/") ~ "published pseudo-progression score",
        str_detect(file, "metadata\\.")                 ~ "per-nucleus metadata table",
        str_detect(file, "cell-annotation")             ~ "per-nucleus annotation table",
        str_detect(file, "\\.h5ad$|\\.rds$")            ~ "region-level AnnData",
        TRUE                                            ~ "other"
      )
    )
}

#' Donor x region grid implied by the per-donor release objects.
#'
#' The donor identity is read from the released file name, which SEA-AD writes as
#' `<Donor ID>_SEAAD_<REGION>_RNAseq_...`. The regex is the same exact-match
#' donor pattern used by src/seaad_mtg_docker_package (`^H\d{2}\.\d{2}\.\d{3}$`);
#' nothing is matched approximately.
seaad_donor_region_objects <- function(manifest) {
  manifest %>%
    filter(family == "per-donor region object", !is.na(donor_id), !is.na(region)) %>%
    transmute(donor_id,
              brain_region = dplyr::recode(region, !!!as.list(SEAAD_FOLDER_TO_REGION)),
              folder, size_mb, key) %>%
    distinct()
}

#' Read the taxonomy table and attach the lineage grouping used in this notebook.
#'
#' `lineage` is a deliberate, stated regrouping of the released Subclass labels:
#' the released `Immune` subclass mixes microglia/PVM with lymphocytes and
#' monocytes, so `microglia_PVM` is defined by the `Micro-PVM` supertype prefix
#' rather than by subclass. Oligodendrocyte and OPC are kept apart.
seaad_taxonomy <- function(path) {
  readr::read_csv(path, show_col_types = FALSE) %>%
    transmute(
      supertype = cluster_label, subclass = subclass_label, class = class_label,
      lineage = dplyr::case_when(
        str_starts(supertype, "Micro-PVM")             ~ "microglia_PVM",
        supertype %in% c("Lymphocyte", "Monocyte")     ~ "other_immune",
        subclass == "Astrocyte"                        ~ "astrocyte",
        subclass == "Oligodendrocyte"                  ~ "oligodendrocyte",
        subclass == "OPC"                              ~ "OPC",
        str_starts(class, "Neuronal")                  ~ "neuron",
        TRUE                                           ~ "other_non_neural"
      )
    )
}

#' Library x lineage nuclei counts and fractions from the scCODA abundance tables.
#'
#' Rows are *library preparations*, not donors: the released abundance objects
#' carry `library_prep` and no `Donor ID`, and several libraries share a donor.
#' Nothing here may be read as a donor-level measurement.
seaad_library_composition <- function(abundances, taxonomy, regions = NULL) {
  ab <- abundances %>% left_join(taxonomy, by = "supertype")
  stopifnot("every supertype must map to the released taxonomy" = !any(is.na(ab$lineage)))
  if (!is.null(regions)) ab <- filter(ab, brain_region %in% regions)

  ab %>%
    group_by(library_prep, brain_region, cps_local, lineage) %>%
    summarise(n_nuclei = sum(n_nuclei), .groups = "drop") %>%
    group_by(library_prep, brain_region, cps_local) %>%
    mutate(n_nuclei_library = sum(n_nuclei),
           fraction = n_nuclei / n_nuclei_library) %>%
    ungroup()
}

#' Within-region z-scores, kept beside the raw value rather than replacing it.
add_region_standardized <- function(df, value_col, region_col = "brain_region",
                                    suffix = "_z_within_region") {
  df %>%
    group_by(.data[[region_col]]) %>%
    mutate("{value_col}{suffix}" :=
             (.data[[value_col]] - mean(.data[[value_col]], na.rm = TRUE)) /
             stats::sd(.data[[value_col]], na.rm = TRUE)) %>%
    ungroup()
}
