#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
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

my_packages <- c("tidyverse", "readxl", "knitr", "jsonlite")
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

paths      <- seaad_paths(PROJECT_ROOT)
omics      <- seaad_omics_paths(PROJECT_ROOT)

show_table <- function(x, caption = NULL, digits = 4) {
  kable(x, caption = caption, digits = digits, format.args = list(big.mark = ""))
}

# The candidate multimodal ROI set. Still candidate, not frozen.
ROI <- c("DFC", "HIP", "MEC", "MTG", "STG", "V1C")
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#| label: provenance

prov <- fromJSON(omics$provenance)

show_table(
  tibble(
    field = c("bucket", "retrieved (UTC)", "keys listed", "keys excluded",
              "single-download size limit", "pseudobulk structure probe"),
    value = c(prov$bucket,
              prov$retrieved_utc,
              format(prov$n_keys, big.mark = ","),
              paste(sprintf("%s %s", trimws(format(unlist(prov$n_keys_excluded), big.mark = ",")),
                            names(prov$n_keys_excluded)), collapse = "; "),
              paste0(prov$max_bytes / 1e6, " MB"),
              basename(prov$pseudobulk_probe))
  ),
  "Acquisition provenance for this notebook"
)
#
#
#
#
#
#
#
#| label: local-files

local_inventory <- tibble(
  role = c("S3 key manifest (derived)", "Bucket README", "Multiregion 2026 README",
           "Cell-type taxonomy + region presence", "Published donor x region CPS",
           "Published compositional-model summary",
           "Supertype abundances by library (derived from 2 h5ad)",
           "Pseudobulk structure probe .obs (derived from 1 h5ad)"),
  path = c(omics$s3_manifest, omics$bucket_readme, omics$multiregion_readme,
           omics$taxonomy, omics$cps_by_region, omics$pertpy_summary,
           omics$abundances, omics$pseudobulk_probe)
) %>%
  mutate(file = basename(path), exists = file.exists(path),
         size_kb = round(file.size(path) / 1024)) %>%
  select(role, file, exists, size_kb)

show_table(local_inventory, "Everything this notebook reads. Total on disk is under 8 MB.")
#
#
#
#
#
#
#
#
#
#| label: readme
#| results: asis

mr <- readLines(omics$multiregion_readme, warn = FALSE)
cat(paste0("> ", mr[seq_len(12)], collapse = "\n"), "\n")
#
#
#
#
#
#| label: manifest

manifest <- read_seaad_s3_manifest(omics$s3_manifest)

show_table(
  manifest %>%
    group_by(family) %>%
    summarise(n_objects = n(),
              total_gb = round(sum(size_mb) / 1000, 1),
              largest_gb = round(max(size_mb) / 1000, 1),
              .groups = "drop") %>%
    arrange(desc(total_gb)),
  "Object families under the listed prefixes, by size"
)
#
#
#
#
#
#| label: folder-vs-region

show_table(
  manifest %>%
    filter(assay == "snRNAseq", !is.na(region)) %>%
    distinct(folder, region) %>%
    arrange(folder),
  "Bucket folder vs the region label written into the file name"
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
#| label: region-objects

region_objects <- manifest %>%
  filter(family %in% c("region-level AnnData", "per-nucleus metadata table",
                       "per-nucleus annotation table"),
         !is.na(region), assay == "snRNAseq") %>%
  transmute(region, nuclei_set, file, size_gb = round(size_mb / 1000, 1),
            release_date, local = file.exists(file.path(omics$raw_dir, file)))

show_table(
  region_objects %>% filter(region %in% ROI) %>% arrange(region, desc(size_gb)),
  "Region-level snRNA-seq objects for the candidate ROI set. None is local."
)
#
#
#
#| label: region-object-cost

show_table(
  region_objects %>%
    filter(region %in% ROI, nuclei_set == "final-nuclei",
           str_detect(file, "\\.h5ad$")) %>%
    summarise(n_objects = n(), total_gb = round(sum(size_gb), 1)),
  "Cost of the obvious-but-wrong route: the six final-nuclei region objects"
)
#
#
#
#
#
#
#
#
#| label: atac

show_table(
  manifest %>%
    filter(assay == "snATACseq", !str_detect(file, "Changelog")) %>%
    transmute(file, size_gb = round(size_mb / 1000, 2), release_date),
  "All snATAC-seq objects in the listed prefixes"
)
#
#
#
#
#
#
#
#
#| label: object-inventory

n_nuclei_by_region <- read_csv(omics$abundances, show_col_types = FALSE) %>%
  group_by(brain_region) %>%
  summarise(n_nuclei = sum(n_nuclei), .groups = "drop")

donor_objects <- seaad_donor_region_objects(manifest)

inventory <- tibble::tribble(
  ~object, ~release, ~assay, ~regions, ~row_grain, ~donor_id_field, ~region_field,
  "SEAAD_<REGION>_RNAseq_final-nuclei.2026-06-22.h5ad",
    "Multiregion 2026-06-22", "snRNAseq", "one per region (10)", "nucleus",
    "obs$`Donor ID`", "implied by object; obs$`Brain Region` in aggregates",
  "<DONOR>_SEAAD_<REGION>_RNAseq_final-nuclei.2026-06-22.h5ad",
    "Multiregion 2026-06-22", "snRNAseq", "10 regions x donors", "nucleus",
    "file name + obs$`Donor ID`", "file name",
  "SEAAD_<SUBCLASS>_..._final-nuclei.2026-06-22.h5ad (subclass_objects/)",
    "Multiregion 2026-06-22", "snRNAseq", "all 10 pooled", "nucleus",
    "obs$`Donor ID`", "obs$`Brain Region`",
  "SEAAD_<SUBCLASS>_..._pseudobulked.2026-06-22.h5ad (pseudobulk_objects/)",
    "Multiregion 2026-06-22", "snRNAseq", "all 10 pooled", "donor x region x supertype",
    "obs$`Donor ID`", "obs$`Brain Region`",
  "SEAAD_..._all-nuclei_metadata.2026-06-22.csv",
    "Multiregion 2026-06-22", "snRNAseq", "all 10 pooled", "nucleus",
    "column `Donor ID`", "column `Brain Region`",
  "Global_and_Local_CPS.20260105.csv",
    "Multiregion 2026 model outputs", "derived from neuropathology", "9 regions",
    "donor x region", "column `Donor ID`", "column `Brain Region`",
  "<CLASS>_Supertype_abundances.h5ad (pertpy objects/)",
    "Multiregion 2026 model outputs", "snRNAseq", "all 10 pooled", "library x supertype",
    "ABSENT", "obs$`Brain Region`"
)

show_table(inventory, "Processed RNA objects and their identifier fields")
#
#
#
#| label: object-inventory-2

show_table(
  tibble::tribble(
    ~field, ~value,
    "taxonomy version",
      "SEA-AD MTG/PFC taxonomy, multiregion expansion of 2026-06-22; no version string is published",
    "n supertypes", as.character(nrow(seaad_taxonomy(omics$taxonomy))),
    "cell-class / subclass fields", "obs$Class, obs$Subclass, obs$Supertype",
    "published disease-state fields",
      "obs carries Thal / Braak / CERAD / ADNC / LATE / Cognitive Status (donor-level, not cell state)",
    "expression in region and subclass objects",
      "X = log-normalised UMI per 10,000; layers = raw UMI counts (per release README)",
    "expression in pseudobulk objects",
      "X = summed raw UMI counts; no layers (verified in 3.2)",
    "pseudobulk / aggregate availability",
      "29 subclass pseudobulk objects, 4-460 MB each; 2 supertype-abundance objects, <1.5 MB each"
  ),
  "Taxonomy, annotation and expression fields"
)
#
#
#
#| label: object-inventory-counts

show_table(
  donor_objects %>%
    count(brain_region, name = "n_donors") %>%
    left_join(n_nuclei_by_region, by = "brain_region") %>%
    left_join(manifest %>%
                filter(family == "region-level AnnData", assay == "snRNAseq",
                       nuclei_set == "final-nuclei",
                       str_detect(file, "\\.h5ad$"), !is.na(region)) %>%
                transmute(brain_region = region, region_object_gb = round(size_mb / 1000, 1)),
              by = "brain_region") %>%
    arrange(desc(n_donors)),
  "n donors and n nuclei per region. Nuclei counts are the final-nuclei totals in the released compositional-model input (section 3.3)."
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
#| label: crosswalk-check

donor_pattern <- "^H\\d{2}\\.\\d{2}\\.\\d{3}$"   # as in crosswalk.py::DONOR_REGEX

qnp        <- read_qnp_workbook(paths$qnp_2026)
qnp_ph     <- qnp_primary_wide(qnp_primary_long(qnp))
probe_obs  <- read_csv(omics$pseudobulk_probe, show_col_types = FALSE)
# The published CSV carries an unnamed pandas index column; it is dropped here.
cps_region <- read_csv(omics$cps_by_region, show_col_types = FALSE,
                       name_repair = ~ ifelse(.x == "", "row_index", .x)) %>%
  select(donor_id = `Donor ID`, brain_region = `Brain Region`, starts_with("CPS_"))

show_table(
  tibble::tribble(
    ~source, ~donor_field, ~region_field, ~n_donors, ~all_ids_match_pattern,
    "QNP workbook (notebook 00)", "Donor ID", "brain region",
      n_distinct(qnp_ph$donor_id), all(str_detect(qnp_ph$donor_id, donor_pattern)),
    "per-donor omics objects (file names)", "leading token of file name", "file name",
      n_distinct(donor_objects$donor_id), all(str_detect(donor_objects$donor_id, donor_pattern)),
    "pseudobulk .obs (probe)", "Donor ID", "Brain Region",
      n_distinct(probe_obs$`Donor ID`), all(str_detect(probe_obs$`Donor ID`, donor_pattern)),
    "published donor x region CPS", "Donor ID", "Brain Region",
      n_distinct(cps_region$donor_id), all(str_detect(cps_region$donor_id, donor_pattern))
  ),
  "Donor identifiers across the omics and neuropathology sources"
)

stopifnot(
  "every omics donor must already be a QNP donor" =
    length(setdiff(donor_objects$donor_id, qnp_ph$donor_id)) == 0,
  "region labels must agree without remapping" =
    length(setdiff(intersect(donor_objects$brain_region, ROI), qnp_ph$brain_region)) == 0
)

cat("donors in omics but not QNP:",
    length(setdiff(donor_objects$donor_id, qnp_ph$donor_id)), "\n")
cat("ROI labels needing translation:",
    length(setdiff(ROI, intersect(donor_objects$brain_region, qnp_ph$brain_region))), "\n")
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#| label: donor-roi-grid

roi_objects <- donor_objects %>% filter(brain_region %in% ROI)

show_table(
  roi_objects %>%
    count(brain_region, name = "n_donors") %>%
    left_join(n_nuclei_by_region, by = "brain_region") %>%
    mutate(nuclei_per_donor = round(n_nuclei / n_donors)) %>%
    arrange(desc(n_donors)),
  "Donors with a released per-donor object, by ROI"
)
#
#
#
#| label: readme-discrepancy

readme_claim <- c("MTG" = 84L, "DFC" = 84L, "MEC" = 84L)
observed <- roi_objects %>% count(brain_region) %>% deframe()

show_table(
  tibble(brain_region = names(readme_claim),
         readme_says = as.integer(readme_claim),
         per_donor_objects = as.integer(observed[names(readme_claim)])) %>%
    mutate(difference = per_donor_objects - readme_says),
  "The release README states that all 84 donors were profiled in MTG, DFC and MEC"
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
#| label: lineage-coverage

taxonomy <- seaad_taxonomy(omics$taxonomy)

show_table(
  taxonomy %>% count(lineage, name = "n_supertypes") %>% arrange(desc(n_supertypes)),
  "Lineage regrouping of the 2026 taxonomy used throughout this notebook"
)
#
#
#
#
#
#
#
#| label: lineage-by-roi

abundances <- read_csv(omics$abundances, show_col_types = FALSE)
lib_comp   <- seaad_library_composition(abundances, taxonomy, regions = ROI)

show_table(
  lib_comp %>%
    filter(lineage %in% c("microglia_PVM", "astrocyte", "neuron",
                          "oligodendrocyte", "OPC")) %>%
    group_by(brain_region, lineage) %>%
    summarise(n_libraries_with_lineage = n_distinct(library_prep),
              n_nuclei = sum(n_nuclei), .groups = "drop") %>%
    left_join(lib_comp %>% group_by(brain_region) %>%
                summarise(n_libraries = n_distinct(library_prep), .groups = "drop"),
              by = "brain_region") %>%
    mutate(libraries_missing_lineage = n_libraries - n_libraries_with_lineage) %>%
    select(brain_region, lineage, n_libraries, n_libraries_with_lineage,
           libraries_missing_lineage, n_nuclei) %>%
    arrange(brain_region, lineage),
  "Lineage presence per ROI at LIBRARY grain, not donor grain"
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
#| label: qnp-overlap

is_obs <- function(x) !is.na(x)

qnp_roi <- qnp_ph %>%
  filter(brain_region %in% ROI) %>%
  mutate(ph_complete = is_obs(abeta_percent_positive_area) &
                       is_obs(ptau_percent_positive_area) &
                       is_obs(neun_cells_per_area) &
                       is_obs(gfap_percent_positive_area))

coverage <- qnp_roi %>%
  group_by(brain_region) %>%
  summarise(n_QNP_rows = n(), n_QNP_PH = sum(ph_complete), .groups = "drop") %>%
  left_join(roi_objects %>% count(brain_region, name = "n_omics"), by = "brain_region") %>%
  left_join(
    qnp_roi %>% filter(ph_complete) %>%
      inner_join(roi_objects, by = c("donor_id", "brain_region")) %>%
      count(brain_region, name = "n_complete_QNP_PH_omics"),
    by = "brain_region") %>%
  mutate(n_omics_without_QNP_PH = n_omics - n_complete_QNP_PH_omics) %>%
  arrange(desc(n_complete_QNP_PH_omics))

show_table(coverage, "QNP P/H and omics coverage by ROI, exact donor identities only")

cat("donor x ROI observations with both QNP P/H and omics:",
    sum(coverage$n_complete_QNP_PH_omics), "\n")
#
#
#
#| label: where-the-losses-are

show_table(
  qnp_primary_long(qnp) %>%
    filter(brain_region %in% ROI, status != "observed") %>%
    count(brain_region, variable, status, name = "n_donors"),
  "Why the QNP side is short: the only incomplete ROI is V1C, and the cause is missing values, not a missing stain"
)

show_table(
  roi_objects %>%
    anti_join(qnp_roi %>% filter(ph_complete), by = c("donor_id", "brain_region")) %>%
    select(donor_id, brain_region),
  "The donor x ROI cells with an omics object but no complete QNP P/H"
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
#| label: simultaneous

both <- qnp_roi %>% filter(ph_complete) %>%
  inner_join(roi_objects, by = c("donor_id", "brain_region"))

show_table(
  both %>% count(donor_id, name = "n_roi") %>% count(n_roi, name = "n_donors") %>%
    arrange(desc(n_roi)),
  "Donors by number of ROIs with both QNP P/H and omics"
)

complete_in <- function(regions) {
  both %>% filter(brain_region %in% regions) %>%
    count(donor_id) %>% filter(n == length(regions)) %>% nrow()
}

show_table(
  tribble(
    ~region_set, ~regions,
    "all six", ROI,
    "drop HIP", setdiff(ROI, "HIP"),
    "drop HIP, STG, V1C", setdiff(ROI, c("HIP", "STG", "V1C")),
    "the three 80+ donor regions", c("DFC", "MEC", "MTG")
  ) %>%
    mutate(n_regions = lengths(regions),
           n_donors_complete = map_int(regions, complete_in),
           regions = map_chr(regions, paste, collapse = ", ")),
  "Donors complete in EVERY region of the set"
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
#| label: pseudobulk-inventory

# The subclass token in the file name is a filesystem-safe spelling of the
# taxonomy's Subclass label ("L23-IT" for "L2/3 IT"). The three irregular cases
# are listed explicitly rather than guessed at.
subclass_from_filename <- function(token, taxonomy_subclasses) {
  irregular <- c("L23-IT" = "L2/3 IT", "L56-NP" = "L5/6 NP",
                 "VLMC-Perivascular" = "VLMC & Perivascular")
  out <- dplyr::coalesce(
    unname(irregular[token]),
    if_else(token %in% taxonomy_subclasses, token, NA_character_),
    if_else(str_replace_all(token, "-", " ") %in% taxonomy_subclasses,
            str_replace_all(token, "-", " "), NA_character_)
  )
  out
}

subclass_class <- taxonomy %>% distinct(subclass, class)

pseudobulk <- manifest %>%
  filter(family == "pseudobulk (donor x region x supertype)") %>%
  mutate(file_token = str_match(file, "^SEAAD_(.+?)_HIP_MEC")[, 2],
         subclass = subclass_from_filename(file_token, unique(taxonomy$subclass))) %>%
  left_join(subclass_class, by = "subclass") %>%
  mutate(lineage_group = case_when(
    subclass == "Astrocyte"       ~ "astrocyte",
    subclass == "Oligodendrocyte" ~ "oligodendrocyte",
    subclass == "OPC"             ~ "OPC",
    subclass == "Immune"          ~ "microglia / immune (mixed)",
    str_starts(class, "Neuronal") ~ "neuron",
    TRUE                          ~ "other non-neural")) %>%
  transmute(file_token, subclass, lineage_group,
            size_mb = round(size_mb, 1), release_date)

stopifnot(
  "every pseudobulk file token must resolve to a taxonomy subclass" =
    !any(is.na(pseudobulk$subclass)),
  "one pseudobulk object per taxonomy subclass" =
    nrow(pseudobulk) == n_distinct(taxonomy$subclass)
)

show_table(
  pseudobulk %>% arrange(desc(size_mb)),
  "All 29 published pseudobulk objects, one per taxonomy subclass. None is local."
)

show_table(
  pseudobulk %>%
    group_by(lineage_group) %>%
    summarise(n_objects = n(), total_mb = round(sum(size_mb)), .groups = "drop") %>%
    arrange(desc(total_mb)),
  "Fetch cost by lineage group"
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
#| label: pseudobulk-probe

probe_shape <- prov$pseudobulk_probe_shape

show_table(
  tibble::tribble(
    ~property, ~value,
    "row grain", paste("one row per", paste(c("Donor ID", "Brain Region", "Supertype"),
                                            collapse = " x ")),
    "n rows", as.character(probe_shape$n_obs),
    "n genes", format(probe_shape$n_vars, big.mark = ","),
    "X all integral", as.character(probe_shape$x_all_integral),
    "layers present", as.character(probe_shape$layers_present),
    "expression scale", "summed raw UMI counts (X is integral and scales with nuclei)",
    "raw counts available", "yes - X *is* the raw count sum; no normalised layer to undo",
    "donor identifier", "obs$`Donor ID`, exact H##.##.### form",
    "region identifier", "obs$`Brain Region`",
    "nuclei denominator", "obs$`Number of nuclei`, per row"
  ),
  "Verified properties of the pseudobulk family, from one probed object"
)

show_table(
  probe_obs %>%
    group_by(`Brain Region`, Subclass, Supertype) %>%
    summarise(n_donors = n_distinct(`Donor ID`),
              n_nuclei = sum(`Number of nuclei`),
              nuclei_min = min(`Number of nuclei`),
              nuclei_median = median(`Number of nuclei`),
              umi_per_nucleus_median = round(median(total_umi / `Number of nuclei`)),
              .groups = "drop"),
  "Probe object contents: Ependymal occurs in HIP only, despite the 10-region file name"
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
#| label: composition-objects

show_table(
  manifest %>%
    filter(family == "compositional model output") %>%
    transmute(file = str_remove(key, "^Multiregion_2026/model_outputs/pertpy_compositional_modeling/"),
              size_mb = round(size_mb, 2)) %>%
    arrange(size_mb),
  "pertpy / scCODA compositional-model outputs"
)

show_table(
  tibble::tribble(
    ~property, ~value,
    "row grain", "library_prep x supertype",
    "n libraries", as.character(n_distinct(abundances$library_prep)),
    "n supertypes", as.character(n_distinct(abundances$supertype)),
    "value", "nuclei count",
    "donor identifier", "ABSENT - obs carries library_prep and scCODA_sample_id only",
    "region identifier", "obs$`Brain Region`",
    "covariates carried", "CPS_Local, PMI, Sex, APOE4 status, binned age"
  ),
  "Structure of the supertype-abundance objects"
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
#| label: pertpy-summary

pertpy <- read_csv(omics$pertpy_summary, show_col_types = FALSE,
                   name_repair = ~ ifelse(.x == "", "row_index", .x)) %>%
  select(cell_type = `Cell Type`, local_model = `Local Model`, region = Region)

show_table(
  pertpy %>%
    group_by(region) %>%
    summarise(n_cell_types = n(),
              n_with_effect = sum(!is.na(local_model)),
              n_zero_effect = sum(local_model == 0, na.rm = TRUE),
              .groups = "drop") %>%
    filter(region %in% ROI),
  "Published compositional effect sizes vs local CPS, by ROI"
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
#| label: cps-grid

show_table(
  cps_region %>% count(brain_region, name = "n_donors") %>% arrange(brain_region),
  "Donor x region grid of Global_and_Local_CPS.20260105.csv"
)

cps_matches_qnp <- map_dfr(intersect(unique(cps_region$brain_region),
                                     unique(qnp_ph$brain_region)), function(r) {
  q <- qnp_ph %>% filter(brain_region == r) %>% pull(donor_id)
  c2 <- cps_region %>% filter(brain_region == r) %>% pull(donor_id)
  tibble(brain_region = r, same_donor_set_as_QNP = setequal(q, c2))
})

show_table(cps_matches_qnp, "Its grid is the QNP grid, region by region")
#
#
#
#
#
#
#
#
#| label: cps-three-way

specimen_index <- read.csv(paths$specimen_index, check.names = FALSE)

cps_compare <- cps_region %>%
  distinct(donor_id, CPS_Global) %>%
  inner_join(specimen_index %>%
               transmute(donor_id = `Donor ID`,
                         cps_specimen = `Continuous Pseudo-progression Score`),
             by = "donor_id") %>%
  mutate(difference = CPS_Global - cps_specimen)

show_table(
  tibble(n_donors = nrow(cps_compare),
         max_abs_difference = max(abs(cps_compare$difference)),
         median_abs_difference = median(abs(cps_compare$difference)),
         n_donors_differing_by_gt_0.1 = sum(abs(cps_compare$difference) > 0.1)),
  "CPS_Global (2026 omics release) vs Continuous Pseudo-progression Score (SpecimenMetadata.csv)"
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
#| label: out-of-scope

show_table(
  manifest %>%
    filter(family %in% c("differential-expression model output",
                         "integrated latent representation",
                         "taxonomy mapping output")) %>%
    group_by(family) %>%
    summarise(n_objects = n(), total_gb = round(sum(size_mb) / 1000, 1), .groups = "drop"),
  "Released model outputs excluded by the guardrails of this task"
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
#| label: feature-origin

show_table(
  tibble::tribble(
    ~feature_origin, ~meaning,
    "SEAAD_published_predefined",
      "a quantity SEA-AD itself publishes as a named variable or label",
    "external_predefined",
      "a gene set or score defined in an external publication, applied unchanged",
    "current_analysis_derived",
      "computed here from released counts by a stated rule; no gene set invented"
  ),
  "Feature-origin categories used in sections 4 and 10"
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
#| label: state-label-search

show_table(
  tibble::tribble(
    ~sought, ~found_in_release, ~where,
    "a donor x region continuous molecular-state score per lineage", "NO", "-",
    "a per-nucleus disease-associated state label (e.g. 'DAM', 'reactive astrocyte')", "NO",
      "obs carries Class / Subclass / Supertype only",
    "disease-associated SUPERTYPES named as such", "PARTIAL",
      "`-SEAAD` suffixed supertypes are new types found in the AD cohort, not state labels",
    "donor-level neuropathology and cognition in obs", "YES",
      "Thal / Braak / CERAD / ADNC / LATE / Cognitive Status",
    "a published per-region pseudo-progression score", "YES",
      "Global_and_Local_CPS.20260105.csv (neuropathology-derived)",
    "published composition-vs-stage effect sizes", "YES",
      "pertpy_summary_CPS_Local.20260622.csv (region-level, not donor-level)"
  ),
  "Search for published molecular-state representations"
)

show_table(
  taxonomy %>%
    mutate(is_seaad_new = str_detect(supertype, "-SEAAD$")) %>%
    filter(lineage %in% c("microglia_PVM", "astrocyte", "oligodendrocyte", "OPC")) %>%
    group_by(lineage) %>%
    summarise(n_supertypes = n(), n_new_in_SEAAD = sum(is_seaad_new),
              supertypes = paste(supertype, collapse = ", "), .groups = "drop"),
  "Glial supertypes, and which are new in the SEA-AD taxonomy"
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
#| label: candidate-axes

candidates <- tibble::tribble(
  ~candidate, ~axis, ~lineage, ~kind, ~interpretation, ~source_object, ~feature_origin,
  "T_microglia_state",
    "T_microglia", "Micro-PVM supertypes", "state",
    "What microglia in this ROI are expressing, at fixed lineage definition",
    "pseudobulk_objects/SEAAD_Immune_... (124 MB)", "current_analysis_derived",
  "T_microglia_composition",
    "T_microglia", "Micro-PVM supertypes", "composition",
    "Share of nuclei that are microglia/PVM, and the mix of Micro-PVM supertypes",
    "pseudobulk obs$`Number of nuclei`; abundance objects (library grain)", "SEAAD_published_predefined",
  "T_astrocyte_state",
    "T_astrocyte", "Astrocyte subclass", "state",
    "Expression state of astrocytes, e.g. a reactive-astrocyte axis over Astro_1..6",
    "pseudobulk_objects/SEAAD_Astrocyte_... (169 MB)", "current_analysis_derived",
  "T_astrocyte_composition",
    "T_astrocyte", "Astrocyte subclass", "composition",
    "Astrocyte share of nuclei; relative abundance of Astro_6-SEAAD",
    "pseudobulk obs$`Number of nuclei`", "SEAAD_published_predefined",
  "T_neuronal_state_subclass_conditioned",
    "T_neuronal", "one defensible neuronal grouping, conditioned", "state",
    "Molecular / synaptic state within a fixed neuronal subclass present in all six ROIs",
    "pseudobulk_objects/SEAAD_<neuronal subclass>_... (see 6.2)", "current_analysis_derived",
  "T_neuronal_state_all_neuron",
    "T_neuronal", "all neurons pooled", "state - CONFOUNDED",
    "All-neuron expression summary; mixes state, subtype composition and selective loss",
    "any neuronal pseudobulk pooled", "current_analysis_derived",
  "T_vulnerable_neuron_composition",
    "T_neuronal", "vulnerable neuronal supertypes", "composition",
    "Relative abundance of the neuronal types SEA-AD reports as selectively lost",
    "pseudobulk obs$`Number of nuclei`; pertpy_summary_CPS_Local", "SEAAD_published_predefined",
  "T_oligo_OPC_state",
    "T_oligodendrocyte_OPC", "Oligodendrocyte + OPC, kept separate", "state",
    "Myelination / remyelination-related expression state within each lineage",
    "pseudobulk SEAAD_Oligodendrocyte_... (147 MB) + SEAAD_OPC_... (84 MB)",
    "current_analysis_derived",
  "T_oligo_OPC_composition",
    "T_oligodendrocyte_OPC", "Oligodendrocyte + OPC, kept separate", "composition",
    "Oligodendrocyte and OPC shares of nuclei; OPC-to-oligodendrocyte ratio",
    "pseudobulk obs$`Number of nuclei`", "SEAAD_published_predefined"
)

show_table(
  candidates %>% select(candidate, axis, kind, interpretation),
  "Candidate T variables: what each one means"
)

show_table(
  candidates %>% select(candidate, lineage, source_object, feature_origin),
  "Candidate T variables: where each one would come from"
)
#
#
#
#| label: candidate-coverage

roi_donor_counts <- coverage %>% select(brain_region, n_complete_QNP_PH_omics)

show_table(
  candidates %>%
    transmute(candidate, kind,
              regions_available = c(
                rep("all six ROI", 4),
                "depends on the neuronal subclass chosen (section 6.2)",
                "all six ROI",
                "depends on which supertypes are called vulnerable (section 6.2)",
                rep("all six ROI", 2)),
              n_donors_min_roi = min(roi_donor_counts$n_complete_QNP_PH_omics),
              n_donors_max_roi = max(roi_donor_counts$n_complete_QNP_PH_omics),
              n_donor_x_roi = sum(roi_donor_counts$n_complete_QNP_PH_omics),
              remaining_ambiguity = c(
                "no published microglial state score; the scoring rule is undefined and must be chosen",
                "Immune subclass mixes microglia with lymphocytes/monocytes; supertype filter required",
                "no published reactive-astrocyte score; the scoring rule is undefined",
                "astrocyte share depends on dissociation yield as well as biology",
                "which neuronal grouping is defensible in all six ROIs is unsettled (section 6)",
                "NOT USABLE as a primary T - listed to be excluded explicitly",
                "SEA-AD's vulnerability claims are per region; a cross-ROI definition is not published",
                "oligodendrocyte and OPC may behave oppositely; must not be pooled",
                "nuclei-yield sensitivity as above, plus very low OPC counts in some libraries")),
  "Candidate T variables: coverage and what is still unresolved"
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
#| label: state-vs-composition

show_table(
  tibble::tribble(
    ~pair, ~composition_variable, ~state_variable, ~why_not_interchangeable,
    "microglia",
      "fraction of nuclei that are microglia/PVM; fraction of microglia that are Micro-PVM_3-SEAAD",
      "expression profile computed across microglial nuclei only",
      "A donor can have unchanged microglial numbers and a wholly changed microglial programme, or the reverse",
    "astrocyte",
      "astrocyte fraction of nuclei; share of Astro_6-SEAAD",
      "expression profile across astrocytic nuclei only",
      "Astrocyte share also tracks dissociation yield and neuronal loss in the denominator",
    "neuron",
      "neuronal subtype composition; relative loss of vulnerable supertypes",
      "expression profile within a fixed neuronal grouping",
      "A drop in an all-neuron summary can be pure composition change with no change in any surviving cell",
    "oligodendrocyte / OPC",
      "oligodendrocyte and OPC shares; OPC-to-oligodendrocyte ratio",
      "expression profile within each lineage separately",
      "Remyelination predicts a composition shift AND a state shift, in possibly opposite directions"
  ),
  "Composition and state are separate candidate variables for every lineage"
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
#| label: composition-illustration

show_table(
  lib_comp %>%
    filter(lineage %in% c("microglia_PVM", "astrocyte", "neuron",
                          "oligodendrocyte", "OPC")) %>%
    group_by(brain_region, lineage) %>%
    summarise(median_fraction = median(fraction), .groups = "drop") %>%
    pivot_wider(names_from = lineage, values_from = median_fraction),
  "Median lineage fraction per ROI, library grain. A composition measurement, and nothing more."
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
#| label: neuron-composition-spread

neuron_mix <- abundances %>%
  filter(brain_region %in% ROI) %>%
  left_join(taxonomy, by = "supertype") %>%
  filter(lineage == "neuron") %>%
  group_by(brain_region, subclass) %>%
  summarise(n_nuclei = sum(n_nuclei), .groups = "drop") %>%
  group_by(brain_region) %>%
  mutate(share_of_neurons = n_nuclei / sum(n_nuclei)) %>%
  ungroup()

show_table(
  neuron_mix %>%
    select(brain_region, subclass, share_of_neurons) %>%
    pivot_wider(names_from = brain_region, values_from = share_of_neurons,
                values_fill = 0) %>%
    arrange(desc(MTG)),
  "Neuronal subclass share of all neurons, by ROI. A zero is a genuine absence: no nucleus of that subclass was mapped in that ROI."
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
#| label: neuron-subclass-coverage

tax_regions <- read_csv(omics$taxonomy, show_col_types = FALSE) %>%
  select(supertype = cluster_label, subclass = subclass_label, class = class_label,
         all_of(ROI)) %>%
  pivot_longer(all_of(ROI), names_to = "brain_region", values_to = "present")

neuronal_coverage <- tax_regions %>%
  filter(str_starts(class, "Neuronal")) %>%
  group_by(subclass, brain_region) %>%
  summarise(n_supertypes_present = sum(present), .groups = "drop") %>%
  pivot_wider(names_from = brain_region, values_from = n_supertypes_present) %>%
  mutate(n_roi_present = rowSums(across(all_of(ROI)) > 0)) %>%
  arrange(desc(n_roi_present), subclass)

show_table(neuronal_coverage,
           "Neuronal supertypes present per ROI, from the release's own taxonomy table")

show_table(
  neuronal_coverage %>%
    filter(n_roi_present == length(ROI)) %>%
    left_join(neuron_mix %>% group_by(subclass) %>%
                summarise(min_share = min(share_of_neurons),
                          max_share = max(share_of_neurons), .groups = "drop"),
              by = "subclass") %>%
    left_join(subclass_class, by = "subclass") %>%
    mutate(share_ratio = round(max_share / min_share, 1),
           neuron_class = str_remove(class, "^Neuronal: ")) %>%
    select(subclass, neuron_class, all_of(ROI), min_share, max_share, share_ratio) %>%
    arrange(share_ratio),
  "Neuronal subclasses present in all six ROI, ordered by how much their share of neurons varies across ROI"
)

show_table(
  neuronal_coverage %>%
    filter(HIP == 0) %>%
    select(subclass, all_of(ROI)),
  "Neuronal subclasses with no supertype in HIP at all"
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
#| label: latent-excluded

show_table(
  manifest %>%
    filter(family == "integrated latent representation") %>%
    transmute(file, size_mb = round(size_mb)),
  "Released scVI/scANVI artefacts"
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
#| label: qnp-vs-cps
#| fig-width: 9
#| fig-height: 5.5

qnp_cps <- qnp_roi %>%
  inner_join(cps_region, by = c("donor_id", "brain_region")) %>%
  select(donor_id, brain_region, CPS_Local, CPS_Local_ABeta, CPS_Local_pTau,
         abeta = abeta_percent_positive_area, ptau = ptau_percent_positive_area,
         neun = neun_cells_per_area, gfap = gfap_percent_positive_area)

qnp_cps %>%
  pivot_longer(c(abeta, ptau, neun, gfap), names_to = "qnp", values_to = "value") %>%
  filter(!is.na(value)) %>%
  ggplot(aes(CPS_Local, value, colour = brain_region)) +
  geom_point(size = 0.8, alpha = 0.6) +
  geom_smooth(method = "loess", se = FALSE, linewidth = 0.5, span = 1) +
  facet_wrap(~ qnp, scales = "free_y") +
  labs(title = "Measured QNP P/H against the published local CPS",
       subtitle = "Rows are donor x ROI. CPS_Local from Global_and_Local_CPS.20260105.csv",
       x = "CPS_Local", y = "QNP value", colour = "ROI") +
  theme_minimal(base_size = 20)
#
#
#
#| label: qnp-vs-local-path
#| fig-width: 9
#| fig-height: 4

qnp_cps %>%
  select(brain_region, CPS_Local_ABeta, CPS_Local_pTau, abeta, ptau) %>%
  pivot_longer(c(abeta, ptau), names_to = "qnp", values_to = "measured") %>%
  mutate(model = if_else(qnp == "abeta", CPS_Local_ABeta, CPS_Local_pTau)) %>%
  filter(!is.na(measured)) %>%
  ggplot(aes(model, measured, colour = brain_region)) +
  geom_point(size = 0.8, alpha = 0.6) +
  facet_wrap(~ qnp, scales = "free") +
  labs(title = "Measured QNP pathology against the release's modelled local pathology score",
       subtitle = "These are different quantities: one is percent positive area, the other a pseudo-progression coordinate",
       x = "CPS_Local_ABeta / CPS_Local_pTau", y = "QNP percent positive area",
       colour = "ROI") +
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
#| label: composition-vs-cps
#| fig-width: 9
#| fig-height: 6

comp_focus <- lib_comp %>%
  filter(lineage %in% c("microglia_PVM", "astrocyte", "oligodendrocyte", "OPC", "neuron"))

comp_focus %>%
  ggplot(aes(cps_local, fraction, colour = brain_region)) +
  geom_point(size = 0.7, alpha = 0.5) +
  geom_smooth(method = "loess", se = FALSE, linewidth = 0.5, span = 1.2) +
  facet_wrap(~ lineage, scales = "free_y") +
  labs(title = "Lineage composition against local CPS, by ROI",
       subtitle = "LIBRARY grain, not donor grain: several libraries share a donor. Shape only; no test, no selection.",
       x = "CPS_Local (carried in the abundance object)", y = "fraction of nuclei in library",
       colour = "ROI") +
  theme_minimal(base_size = 20)
#
#
#
#| label: composition-binned

show_table(
  comp_focus %>%
    group_by(brain_region) %>%
    mutate(cps_bin = cut(cps_local, breaks = quantile(cps_local, seq(0, 1, 0.25)),
                         include.lowest = TRUE, labels = c("Q1", "Q2", "Q3", "Q4"))) %>%
    group_by(brain_region, lineage, cps_bin) %>%
    summarise(median_fraction = median(fraction), n_libraries = n(), .groups = "drop") %>%
    select(-n_libraries) %>%
    pivot_wider(names_from = cps_bin, values_from = median_fraction) %>%
    arrange(lineage, brain_region),
  "Median lineage fraction by within-ROI CPS_Local quartile (library grain)"
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
#| label: region-raw

show_table(
  comp_focus %>%
    group_by(brain_region, lineage) %>%
    summarise(median_fraction = median(fraction), .groups = "drop") %>%
    pivot_wider(names_from = lineage, values_from = median_fraction) %>%
    arrange(desc(microglia_PVM)),
  "Raw median lineage fraction by ROI. Retained, never overwritten."
)
#
#
#
#| label: region-standardized
#| fig-width: 9
#| fig-height: 4.5

microglia_std <- comp_focus %>%
  filter(lineage == "microglia_PVM") %>%
  add_region_standardized("fraction")

stopifnot(
  "the raw value must survive standardisation" =
    all(c("fraction", "fraction_z_within_region") %in% names(microglia_std)),
  "standardisation must not alter the raw column" =
    identical(microglia_std$fraction,
              comp_focus %>% filter(lineage == "microglia_PVM") %>% pull(fraction))
)

bind_rows(
  microglia_std %>% transmute(brain_region, cps_local, value = fraction, scale = "raw fraction"),
  microglia_std %>% transmute(brain_region, cps_local, value = fraction_z_within_region,
                             scale = "within-ROI z")
) %>%
  ggplot(aes(cps_local, value, colour = brain_region)) +
  geom_point(size = 0.7, alpha = 0.5) +
  geom_smooth(method = "loess", se = FALSE, linewidth = 0.5, span = 1.2) +
  facet_wrap(~ scale, scales = "free_y") +
  labs(title = "Microglial fraction, raw and standardised within ROI",
       subtitle = "The standardised view is an exploratory aid; the raw value and the ROI label are what get carried forward",
       x = "CPS_Local", y = NULL, colour = "ROI") +
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
#
#| label: decision-table

roi_cov_string <- paste0(nrow(coverage), "/", length(ROI), " ROI")
min_don <- min(coverage$n_complete_QNP_PH_omics)
max_don <- max(coverage$n_complete_QNP_PH_omics)
tot_obs <- sum(coverage$n_complete_QNP_PH_omics)

decision <- tibble::tribble(
  ~axis, ~candidate_definition, ~lineage, ~state_or_composition, ~source_release,
  ~feature_origin, ~status, ~remaining_issue,

  "microglial molecular state",
    "Expression summary over Micro-PVM supertypes only, from donor x ROI pseudobulk raw counts",
    "Micro-PVM supertypes", "state", "Multiregion 2026-06-22 pseudobulk_objects/SEAAD_Immune",
    "current_analysis_derived", "primary candidate",
    "Scoring rule undefined; no published microglial state score exists. Immune subclass must be filtered to Micro-PVM.",

  "microglial composition",
    "Micro-PVM share of ROI nuclei, and Micro-PVM supertype mix",
    "Micro-PVM supertypes", "composition", "pseudobulk obs$`Number of nuclei`",
    "SEAAD_published_predefined", "secondary",
    "Depends on the neuronal denominator and on dissociation yield.",

  "astrocyte molecular state",
    "Expression summary over Astrocyte-subclass nuclei, from donor x ROI pseudobulk raw counts",
    "Astrocyte", "state", "Multiregion 2026-06-22 pseudobulk_objects/SEAAD_Astrocyte",
    "current_analysis_derived", "primary candidate",
    "Scoring rule undefined; no published reactive-astrocyte score in the release.",

  "astrocyte composition",
    "Astrocyte share of ROI nuclei; Astro_6-SEAAD share of astrocytes",
    "Astrocyte", "composition", "pseudobulk obs$`Number of nuclei`",
    "SEAAD_published_predefined", "secondary",
    "Same denominator dependence as microglial composition.",

  "neuronal / synaptic molecular state",
    "Expression summary within ONE neuronal subclass present in all six ROI (options A and B in 6.2)",
    "a fixed neuronal subclass", "state",
    "Multiregion 2026-06-22 pseudobulk_objects/SEAAD_<subclass>",
    "current_analysis_derived", "UNRESOLVED",
    "Two defensible groupings (GABAergic-only, which spans all six ROI, vs weighted excitatory, which is undefined in HIP). Not chosen here.",

  "vulnerable-neuron composition",
    "Share of neurons in the supertypes SEA-AD reports as selectively lost",
    "vulnerable neuronal supertypes", "composition",
    "pseudobulk obs$`Number of nuclei`; pertpy_summary_CPS_Local",
    "SEAAD_published_predefined", "secondary",
    "SEA-AD's vulnerability claims are per region; no cross-ROI vulnerable set is published.",

  "oligodendrocyte / OPC state",
    "Separate expression summaries within Oligodendrocyte and within OPC, never pooled",
    "Oligodendrocyte; OPC", "state",
    "Multiregion 2026-06-22 pseudobulk_objects/SEAAD_Oligodendrocyte + SEAAD_OPC",
    "current_analysis_derived", "secondary (prespecified)",
    "Two lineages, so two variables. OPC counts are low in some libraries; a nuclei floor is required.",

  "oligodendrocyte / OPC composition",
    "Oligodendrocyte and OPC shares of ROI nuclei; OPC-to-oligodendrocyte ratio",
    "Oligodendrocyte; OPC", "composition", "pseudobulk obs$`Number of nuclei`",
    "SEAAD_published_predefined", "secondary",
    "The ratio is the interpretable form but is unstable at low OPC counts.",

  "all-neuron molecular state",
    "Expression summary over all neurons pooled",
    "all neurons", "state (confounded)", "any neuronal pseudobulk pooled",
    "current_analysis_derived", "EXCLUDED",
    "Confounds within-cell state, subtype composition and selective loss (section 6.1).",

  "integrated latent coordinates",
    "Donor x ROI mean of X_scVI",
    "any", "neither", "Multiregion 2026 final_embeddings/",
    "SEAAD_published_predefined", "EXCLUDED",
    "Not an expression measure; excluded by section 7."
) %>%
  mutate(excluded = status == "EXCLUDED",
         roi_coverage = case_when(
           excluded ~ "n/a",
           axis == "neuronal / synaptic molecular state" ~ "6/6 under option A; 5/6 under option B (no HIP)",
           TRUE ~ roi_cov_string),
         donor_coverage = if_else(excluded, "n/a",
                                  paste0(min_don, "-", max_don, " per ROI")),
         qnp_overlap_coverage = if_else(excluded, "n/a",
                                        paste0(tot_obs, " donor x ROI with QNP P/H"))) %>%
  select(-excluded)

show_table(
  decision %>% select(axis, candidate_definition, lineage, state_or_composition, status),
  "Candidate T variables: definition and status"
)

show_table(
  decision %>% select(axis, source_release, feature_origin, roi_coverage,
                     donor_coverage, qnp_overlap_coverage),
  "Candidate T variables: source and coverage"
)

show_table(
  decision %>% select(axis, status, remaining_issue),
  "Candidate T variables: what is still open"
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
#| label: unresolved

show_table(
  tibble::tribble(
    ~issue, ~kind, ~what_would_settle_it,
    "No pseudobulk object is local; 29 exist, 4-460 MB each",
      "coverage",
      "Fetch the four lineage objects for the prespecified axes: Immune 124 MB + Astrocyte 169 MB + Oligodendrocyte 147 MB + OPC 84 MB = 524 MB",
    "Donor-level lineage counts per ROI are unverified (library grain only)",
      "coverage", "The same four pseudobulk objects; obs carries Donor ID, Brain Region and Number of nuclei",
    "The release README claims 84 donors for DFC and MEC; 80 and 81 per-donor objects exist",
      "coverage", "The region metadata CSV (1.3-1.4 GB), or a note from SEA-AD",
    "No published molecular-state score or disease-state label for any lineage",
      "semantics", "A stated scoring rule, chosen before any association is inspected",
    "Three distinct published CPS columns now exist (SpecimenMetadata, caudate file, CPS_Global)",
      "semantics", "A decision on which CPS the joint P/H/T analysis uses; notebook 00 pinned SpecimenMetadata and that stands",
    "CPS_Local's grid is the QNP grid, not the omics grid",
      "semantics", "Nothing - established here; it simply must not be read as omics coverage",
    "The neuronal primary has two defensible definitions",
      "semantics", "A scientific choice between options A and B in section 6.2",
    "HIP has no layer-defined excitatory subclasses at all, so an excitatory-based neuronal T is undefined there",
      "coverage", "Either the GABAergic option A, or dropping HIP from the ROI set",
    "Nuclei-count floor per donor x ROI x supertype is unset",
      "semantics", "The pseudobulk Number of nuclei distributions, once fetched",
    "The ROI set is still candidate; HIP costs 40+ donors",
      "coverage", "The multimodal ROI decision, which is outside this notebook"
  ),
  "Open issues, separated into coverage and semantics"
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
#| label: session

sessionInfo()
#
#
#
#
