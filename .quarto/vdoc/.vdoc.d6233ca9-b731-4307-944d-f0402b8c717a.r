#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
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

suppressPackageStartupMessages({
  library(tidyverse)
  library(readxl)
  library(knitr)
  library(cowplot)
  library(scales)
})

find_root <- function(p = getwd()) {
  while (!dir.exists(file.path(p, ".git"))) {
    if (identical(dirname(p), p)) stop("project root not found")
    p <- dirname(p)
  }
  p
}
PROJECT_ROOT <- find_root()
source(file.path(PROJECT_ROOT, "R", "seaad_explore.R"))
source(file.path(PROJECT_ROOT, "R", "seaad_analysis_ready.R"))

paths    <- seaad_paths(PROJECT_ROOT)
analysis <- seaad_analysis_paths(PROJECT_ROOT)

# One output directory for this notebook's exported composites. Files are
# overwritten on re-render; there are no draft or versioned copies.
FIG_DIR <- file.path(PROJECT_ROOT, "results", "analysis", "figures", "04_phenotype")
dir.create(FIG_DIR, recursive = TRUE, showWarnings = FALSE)

show_table <- function(x, caption = NULL, digits = 2, align = NULL) {
  kable(x, caption = caption, digits = digits, align = align,
        format.args = list(big.mark = ""))
}

ROI6 <- c("MTG", "DFC", "MEC", "STG", "V1C", "HIP")
ROI3 <- c("MTG", "DFC", "MEC")
SIG_MG4   <- "sun_2023_MG4_lipid_processing"
SIG_ASTRO <- "astrocyte_activation_GOBP"
SIG_MG8   <- "sun_2023_MG8_inflammatory_II"
set.seed(20260924)
#
#
#
#| label: design-system

# --- figure design system ---------------------------------------------------
# One set of encodings for all five figures. Palettes were checked with the
# dataviz skill's validator (scripts/validate_palette.js):
#   Braak   ordinal single-hue blue ramp, --ordinal: all checks pass
#   APOE    aqua / orange, --pairs all: CVD dE 9.2, normal-vision dE 27.6
#   ROI     green / violet / magenta, --pairs all: CVD dE 17.6, normal dE 33.9
# Aqua and magenta sit below 3:1 contrast on white, so every filled point gets
# a thin dark outline and every colour is named in a legend.
FIG_FONT <- "Arial"
FIG_WIDTH_MM <- 180
INK   <- "#0b0b0b"
INK2  <- "#52514e"
MUTED <- "#8a8984"
GRID  <- "#ebeae6"
OUTLINE <- "#2b2b29"

BRAAK_LEVELS <- c("0–III", "IV", "V", "VI")
PAL_BRAAK <- setNames(c("#86b6ef", "#3987e5", "#1c5cab", "#0d366b"), BRAAK_LEVELS)
APOE_LEVELS <- c("ε4 noncarrier", "ε4 carrier")
PAL_APOE <- setNames(c("#1baf7a", "#eb6834"), APOE_LEVELS)
PAL_ROI <- c(MTG = "#008300", DFC = "#4a3aa7", MEC = "#e87ba4")

theme_fig <- function(base_size = 8) {
  theme_classic(base_size = base_size, base_family = FIG_FONT) +
    theme(
      text = element_text(colour = INK),
      axis.text = element_text(colour = INK2, size = rel(0.9)),
      axis.title = element_text(colour = INK, size = rel(0.95)),
      axis.line = element_line(colour = MUTED, linewidth = 0.3),
      axis.ticks = element_line(colour = MUTED, linewidth = 0.3),
      panel.grid.major.y = element_line(colour = GRID, linewidth = 0.3),
      plot.title = element_text(face = "bold", size = rel(1.05), margin = margin(b = 2)),
      plot.subtitle = element_text(colour = INK2, size = rel(0.85), margin = margin(b = 4),
                                   lineheight = 1.1),
      plot.caption = element_text(colour = MUTED, size = rel(0.8), hjust = 0),
      strip.background = element_blank(),
      strip.text = element_text(face = "bold", hjust = 0, size = rel(0.95)),
      legend.title = element_text(size = rel(0.9), colour = INK, margin = margin(r = 12)),
      legend.text = element_text(size = rel(0.85), colour = INK2),
      legend.key.size = unit(3.2, "mm"),
      legend.key.spacing.x = unit(2.5, "mm"),
      legend.title.position = "left",
      legend.background = element_blank(),
      plot.margin = margin(8, 8, 6, 8),
      plot.background = element_rect(fill = "white", colour = NA)
    )
}
theme_set(theme_fig())

# Filled circle with a thin dark outline: the one point glyph used throughout.
point_args <- list(shape = 21, size = 1.9, stroke = 0.2, colour = OUTLINE, alpha = 0.9)

braak_fill <- function(...) {
  scale_fill_manual(values = PAL_BRAAK, name = "Braak stage", drop = FALSE, ...)
}
apoe_fill <- function(...) scale_fill_manual(values = PAL_APOE, name = "APOE", ...)
roi_fill  <- function(...) scale_fill_manual(values = PAL_ROI, name = "Region", ...)
roi_colour <- function(...) scale_colour_manual(values = PAL_ROI, name = "Region", ...)

# A transparent pseudo-log axis for right-skewed percent-area measures that
# contain true zeros. sigma sets the width of the linear zone around zero; the
# tick labels stay in the measurement's own units.
pct_axis_trans <- function(sigma) pseudo_log_trans(sigma = sigma, base = 10)
pct_breaks <- c(0, 0.01, 0.1, 1, 10, 30)
pct_label  <- function(x) {
  vapply(x, function(v) {
    if (is.na(v)) return(NA_character_)
    if (v == 0) return("0")
    format(v, scientific = FALSE, drop0trailing = TRUE, trim = TRUE)
  }, character(1))
}

# Legends are pulled out with cowplot::get_legend() and placed once per figure.
shared_legend <- function(p, position = "top") {
  cowplot::get_legend(p + theme(legend.position = position,
                                legend.justification = "center",
                                legend.box.margin = margin(0, 0, 0, 0)))
}
no_legend <- theme(legend.position = "none")

compose <- function(..., labels = "AUTO", label_size = 10) {
  plot_grid(..., labels = labels, label_size = label_size, label_fontfamily = FIG_FONT,
            label_fontface = "bold", align = "hv", axis = "tblr")
}

# Export every composite as vector PDF and 400 dpi PNG at the same physical size.
save_figure <- function(plot, name, height_mm, width_mm = FIG_WIDTH_MM) {
  plot <- ggdraw(plot) + theme(plot.background = element_rect(fill = "white", colour = NA))
  base <- file.path(FIG_DIR, name)
  ggsave(paste0(base, ".pdf"), plot, width = width_mm, height = height_mm, units = "mm",
         device = cairo_pdf)
  ggsave(paste0(base, ".png"), plot, width = width_mm, height = height_mm, units = "mm",
         dpi = 400, device = ragg::agg_png, bg = "white")
  paste0(base, ".png")
}

fmt_n <- function(n) format(n, big.mark = "")
fmt_ci <- function(est, lo, hi, digits = 2) {
  sprintf(paste0("%.", digits, "f [%.", digits, "f, %.", digits, "f]"), est, lo, hi)
}
#
#
#
#| label: donor-table

donor_meta <- read_excel(paths$donor_metadata, guess_max = 1e5)
specimen   <- read.csv(paths$specimen_index, check.names = FALSE)

# The exact source fields. Every one is asserted to exist under this name.
SRC <- list(
  apoe       = c(file = "donor_metadata", column = "APOE Genotype"),
  braak      = c(file = "donor_metadata", column = "Braak"),
  casi       = c(file = "donor_metadata", column = "Last CASI Score"),
  casi_int   = c(file = "donor_metadata", column = "Interval from last CASI in months"),
  cog_status = c(file = "donor_metadata", column = "Cognitive Status"),
  cps        = c(file = "SpecimenMetadata.csv", column = "Continuous Pseudo-progression Score")
)
stopifnot(
  "donor-metadata fields present" =
    all(map_chr(keep(SRC, ~ .x["file"] == "donor_metadata"), "column") %in% names(donor_meta)),
  "CPS field present in SpecimenMetadata.csv" = SRC$cps["column"] %in% names(specimen),
  "one row per donor in each source" =
    !anyDuplicated(donor_meta$`Donor ID`) && !anyDuplicated(specimen$`Donor ID`),
  "the two sources carry the same 84 donors" =
    setequal(donor_meta$`Donor ID`, specimen$`Donor ID`) && nrow(donor_meta) == 84L
)

donor <- donor_meta %>%
  transmute(
    donor_id = `Donor ID`, cohort = `Primary Study Name`,
    age = `Age at Death`, sex = Sex, education = `Years of education`,
    apoe = `APOE Genotype`, braak = Braak, cog_status = `Cognitive Status`,
    casi = `Last CASI Score`, casi_interval = `Interval from last CASI in months`,
    late = LATE, lewy = `Highest Lewy Body Disease`, caa = `Overall CAA Score`,
    arteriolo = Arteriolosclerosis, athero = Atherosclerosis,
    microinfarcts = `Total Microinfarcts (not observed grossly)`
  ) %>%
  left_join(specimen %>% transmute(donor_id = `Donor ID`,
                                   cps = `Continuous Pseudo-progression Score`,
                                   apoe_si = `APOE genotype`, braak_si = `Braak stage`,
                                   cog_si = `Cognitive status`),
            by = "donor_id")

# The two files repeat APOE, Braak and cognitive status; they must agree exactly.
stopifnot(
  "APOE agrees across sources"  = identical(donor$apoe, donor$apoe_si),
  "Braak agrees across sources" = identical(donor$braak, donor$braak_si),
  "cognitive status agrees"     = identical(donor$cog_status, donor$cog_si),
  "no missing APOE, Braak, cognitive status or CPS" =
    !anyNA(donor[c("apoe", "braak", "cog_status", "cps")])
)

# --- APOE grouping ------------------------------------------------------------
# Observed genotypes: 2/2, 2/3, 3/3, 2/4, 3/4, 4/4. The primary display is
# carrier / noncarrier because the 4/4 group is small (and smaller still with
# CASI). 2/4 is counted as an e4 carrier - it carries one e4 allele - but it is
# kept visible in the genotype panel and removed in a sensitivity check, because
# its e2 allele means it is not interchangeable with 3/4.
APOE_GENOTYPES <- c("2/2", "2/3", "3/3", "2/4", "3/4", "4/4")
stopifnot("only the six expected genotypes occur" = all(donor$apoe %in% APOE_GENOTYPES))

donor <- donor %>%
  mutate(
    apoe = factor(apoe, levels = APOE_GENOTYPES),
    e4_alleles = str_count(as.character(apoe), "4"),
    apoe_group = factor(if_else(e4_alleles > 0, "ε4 carrier", "ε4 noncarrier"),
                        levels = APOE_LEVELS),
    braak_stage = factor(str_remove(braak, "Braak "), levels = c("0", "I", "II", "III", "IV", "V", "VI")),
    # Display grouping only: no donor is at stage I, and 0/II/III hold 2/4/6
    # donors, too few to summarise separately.
    braak_group = factor(case_when(braak_stage %in% c("0", "I", "II", "III") ~ "0–III",
                                   TRUE ~ as.character(braak_stage)), levels = BRAAK_LEVELS),
    cog_status = factor(cog_status, levels = c("No dementia", "Dementia")),
    has_casi = !is.na(casi)
  )

# CASI is recorded only for ACT donors. Checked, not assumed, because it makes
# every CASI analysis an ACT-only analysis and makes cohort inestimable there.
stopifnot("CASI present exactly for the ACT donors" =
            identical(donor$has_casi, donor$cohort == "ACT"))
#
#
#
#| label: casi-model

# The one baseline model. Linear in CPS on purpose: 69 donors do not support a
# flexible fit, and the residual QC below checks what the linear term leaves.
# Cohort is not included: every donor with CASI is ACT, so it is constant.
CASI_FORMULA <- casi ~ cps + age + sex + education + casi_interval

casi_set <- donor %>% filter(has_casi)
casi_fit <- lm(CASI_FORMULA, data = casi_set)
stopifnot(
  "no hidden complete-case loss in the model" = nobs(casi_fit) == nrow(casi_set),
  "the CASI set is the 69 ACT donors" = nrow(casi_set) == 69L
)

casi_set <- casi_set %>% mutate(adj_cognition = residuals(casi_fit))
donor <- donor %>% left_join(casi_set %>% select(donor_id, adj_cognition), by = "donor_id")

casi_r2 <- summary(casi_fit)$r.squared
casi_adj_r2 <- summary(casi_fit)$adj.r.squared

# Residual QC, reported in text rather than as another figure.
casi_quad <- lm(update(CASI_FORMULA, . ~ . + I(cps^2)), data = casi_set)
resid_qc <- list(
  rho_cps = cor(casi_set$adj_cognition, casi_set$cps, method = "spearman"),
  rho_interval = cor(casi_set$adj_cognition, casi_set$casi_interval, method = "spearman"),
  tertile_means = casi_set %>% mutate(t = ntile(cps, 3)) %>% group_by(t) %>%
    summarise(m = mean(adj_cognition), .groups = "drop") %>% pull(m),
  quad_delta_r2 = summary(casi_quad)$r.squared - casi_r2,
  quad_p = anova(casi_fit, casi_quad)$`Pr(>F)`[2],
  n_casi_ge95 = sum(casi_set$casi >= 95)
)
#
#
#
#| label: roi-table

qnp_wide <- read_qnp_workbook(paths$qnp_2026) %>%
  qnp_primary_long() %>%
  filter(status == "observed") %>%
  select(donor_id, brain_region, variable, value)
stopifnot("one QNP value per donor x region x variable" =
            !anyDuplicated(qnp_wide[c("donor_id", "brain_region", "variable")]))
qnp_wide <- pivot_wider(qnp_wide, names_from = variable, values_from = value)

state_tbl <- read_csv(analysis$analysis_table, show_col_types = FALSE)
stopifnot("one molecular value per donor x region x signature" =
            !anyDuplicated(state_tbl[c("donor_id", "brain_region", "signature")]))

state_wide <- state_tbl %>%
  filter(signature %in% c(SIG_MG4, SIG_ASTRO, SIG_MG8)) %>%
  mutate(key = recode(signature, !!SIG_MG4 := "mg4", !!SIG_ASTRO := "astro", !!SIG_MG8 := "mg8")) %>%
  select(donor_id, brain_region, key, competitive_score, n_nuclei, support_status) %>%
  pivot_wider(names_from = key, values_from = c(competitive_score, n_nuclei, support_status))

# Exact (donor, region) equality on both sides; nothing renamed or imputed.
donor_roi <- qnp_wide %>%
  full_join(state_wide, by = c("donor_id", "brain_region")) %>%
  left_join(donor, by = "donor_id") %>%
  mutate(neun_per_mm2 = neun_cells_per_area * 1e6)   # count per um^2 -> per mm^2

stopifnot(
  "donor x ROI rows are unique" = !anyDuplicated(donor_roi[c("donor_id", "brain_region")]),
  "every donor x ROI row maps to a known donor" = all(donor_roi$donor_id %in% donor$donor_id)
)
#
#
#
#
#
#
#
#| label: fig1-build

braak_n_casi <- casi_set %>% count(braak_group, .drop = FALSE)

# --- A. terminal CASI vs CPS ---------------------------------------------------
rho_casi_cps <- cor(casi_set$casi, casi_set$cps, method = "spearman")
f1a <- ggplot(casi_set, aes(cps, casi)) +
  geom_smooth(method = "lm", formula = y ~ x, se = FALSE, colour = INK2,
              linewidth = 0.45, linetype = "22") +
  exec(geom_point, !!!point_args, mapping = aes(fill = braak_group)) +
  braak_fill() +
  scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.25), expand = expansion(0.02)) +
  scale_y_continuous(limits = c(60, 101), breaks = seq(60, 100, 10)) +
  annotate("text", x = 0.02, y = 62, hjust = 0, vjust = 0, size = 2.4, colour = INK2,
           family = FIG_FONT,
           label = sprintf("n = %d ACT donors · Spearman ρ = %.2f", nrow(casi_set), rho_casi_cps)) +
  labs(title = "CASI falls with CPS, with wide spread",
       x = "Continuous pseudo-progression score (CPS)", y = "Last CASI score (0–100)")

# --- B. Braak x documented cognitive status ------------------------------------
f1b_counts <- donor %>%
  count(braak_stage, cog_status, .drop = FALSE) %>%
  filter(braak_stage != "I") %>%                       # no donor is at stage I
  mutate(braak_stage = droplevels(braak_stage))
f1b_casi <- donor %>% filter(braak_stage != "I") %>%
  mutate(braak_stage = droplevels(braak_stage)) %>%
  group_by(braak_stage, .drop = FALSE) %>%
  summarise(n = n(), n_casi = sum(has_casi), .groups = "drop")

f1b <- ggplot(f1b_counts, aes(cog_status, braak_stage)) +
  geom_tile(aes(fill = n), colour = "white", linewidth = 0.8) +
  geom_text(aes(label = n, colour = n > 12), size = 2.8, family = FIG_FONT) +
  geom_text(data = f1b_casi, aes(x = 2.72, y = braak_stage,
                                 label = sprintf("%d/%d", n_casi, n)),
            inherit.aes = FALSE, hjust = 0, size = 2.3, colour = INK2, family = FIG_FONT) +
  annotate("text", x = 2.72, y = 0.25, label = "with CASI", hjust = 0, size = 2.3,
           colour = INK2, family = FIG_FONT, fontface = "italic") +
  scale_fill_gradient(low = "#f1f0ec", high = "#3a3a37", guide = "none") +
  scale_colour_manual(values = c(`FALSE` = INK, `TRUE` = "white"), guide = "none") +
  scale_x_discrete(expand = expansion(add = c(0.5, 0.95))) +
  coord_cartesian(clip = "off") +
  labs(title = "Documented cognitive status by Braak stage",
       x = "Cognitive status (as documented)", y = "Braak stage") +
  theme(panel.grid.major.y = element_blank(), axis.line = element_blank(),
        axis.ticks = element_blank(), plot.margin = margin(8, 30, 6, 8))

# --- C. exploratory pathology-adjusted cognition --------------------------------
f1c <- ggplot(casi_set, aes(casi, adj_cognition)) +
  geom_hline(yintercept = 0, colour = MUTED, linewidth = 0.3) +
  exec(geom_point, !!!point_args, mapping = aes(fill = braak_group)) +
  braak_fill() +
  scale_x_continuous(limits = c(60, 101), breaks = seq(60, 100, 10)) +
  scale_y_continuous(breaks = seq(-15, 15, 5)) +
  annotate("text", x = 61, y = max(casi_set$adj_cognition) + 1.5, hjust = 0, vjust = 1,
           size = 2.3, colour = INK2, family = FIG_FONT, lineheight = 1.05,
           label = sprintf("CASI ~ CPS + age + sex + education\n       + CASI-to-death interval\nn = %d, R² = %.2f (adjusted %.2f)",
                           nobs(casi_fit), casi_r2, casi_adj_r2)) +
  labs(title = "Exploratory pathology-adjusted cognition",
       x = "Last CASI score (0–100)",
       y = "Residual CASI (points)\n> 0 better than expected")

# --- D. longitudinal memory and executive function ------------------------------
cognition <- suppressMessages(read_excel(paths$cognition, guess_max = 1e5)) %>%
  rename(donor_id = `Donor ID`)
long <- cognition %>%
  inner_join(casi_set %>% select(donor_id, age, adj_cognition), by = "donor_id") %>%
  mutate(years_before_death = age - age_vis,
         phenotype = factor(if_else(adj_cognition > 0, "better than expected", "worse than expected"),
                            levels = c("better than expected", "worse than expected"))) %>%
  select(donor_id, phenotype, years_before_death, Memory = MEM_E, `Executive function` = EXF_E) %>%
  pivot_longer(c(Memory, `Executive function`), names_to = "domain", values_to = "score") %>%
  filter(!is.na(score)) %>%
  mutate(domain = factor(domain, levels = c("Memory", "Executive function")))

# Visits are ~biennial and ages are whole years, so trajectories are summarised
# in 2-year bins of time before death, and only where a bin holds >= 5 donors
# of that phenotype.
long_bins <- long %>%
  mutate(bin = pmin(floor(years_before_death / 2) * 2 + 1, 19)) %>%
  group_by(domain, phenotype, bin) %>%
  summarise(n = n_distinct(donor_id), mean = mean(score), .groups = "drop") %>%
  filter(n >= 5)

f1d <- ggplot(long, aes(years_before_death, score)) +
  geom_line(aes(group = donor_id), colour = "#cfcec9", linewidth = 0.2) +
  geom_line(data = long_bins, aes(bin, mean, linetype = phenotype), colour = INK, linewidth = 0.6) +
  geom_point(data = long_bins, aes(bin, mean, shape = phenotype), colour = INK, fill = "white",
             size = 1.3, stroke = 0.4) +
  scale_x_reverse(breaks = seq(0, 20, 5), limits = c(21, 0)) +
  scale_linetype_manual(values = c("solid", "22"), name = "Residual CASI") +
  scale_shape_manual(values = c(16, 21), name = "Residual CASI") +
  facet_wrap(~ domain, nrow = 1) +
  labs(title = "Harmonized cognitive trajectories",
       x = "Years before death", y = "Harmonized domain score") +
  theme(legend.position = "inside", legend.position.inside = c(0.02, 0.02),
        legend.justification = c(0, 0), legend.key.width = unit(6, "mm"),
        legend.background = element_rect(fill = alpha("white", 0.85), colour = NA),
        legend.title.position = "top",
        panel.spacing.x = unit(3, "mm"))

fig1 <- plot_grid(
  shared_legend(f1a),
  compose(f1a + no_legend, f1b, f1c + no_legend, f1d, ncol = 2, rel_widths = c(1, 1)),
  ncol = 1, rel_heights = c(0.05, 1)
)
fig1_png <- save_figure(fig1, "fig1_pathology_cognition", height_mm = 160)
fig1_cap <- sprintf("**Figure 1.** Pathology–cognition landscape. **A** Last CASI against the canonical CPS for the %d ACT donors with CASI; dashed line is a descriptive least-squares fit. **B** Documented cognitive status by Braak stage for all 84 donors; right-hand column gives how many donors in each stage have CASI (CASI exists only for ACT, and Braak VI is under-represented in it). No donor is at Braak I. **C** Residual of the single baseline model against raw CASI. It is an exploratory phenotype, not a validated resilience measure. **D** Harmonized memory and executive-function scores for the same donors; grey lines are donors, black lines are 2-year-bin means shown where a bin holds at least five donors, split by the sign of the residual in C.", nrow(casi_set))
#
#
#
#| label: fig1
#| fig-cap: !expr fig1_cap
#| out-width: "100%"

include_graphics(fig1_png)
#
#
#
#| label: fig1-numbers

stage_mix <- donor %>% filter(braak_stage %in% c("V", "VI")) %>% count(cog_status)
long_first <- cognition %>%
  inner_join(casi_set %>% select(donor_id, age, adj_cognition), by = "donor_id") %>%
  filter(!is.na(MEM_E)) %>%
  group_by(donor_id) %>%
  summarise(adj = first(adj_cognition),
            mem_first = MEM_E[which.min(age_vis)], exf_first = EXF_E[which.min(age_vis)],
            first_ytd = first(age) - min(age_vis),
            slope_mem = first(slope_zmem0), slope_exf = first(slope_zexf0), .groups = "drop")
rho_long <- c(
  mem_first = cor(long_first$adj, long_first$mem_first, method = "spearman", use = "complete.obs"),
  exf_first = cor(long_first$adj, long_first$exf_first, method = "spearman", use = "complete.obs"),
  slope_mem = cor(long_first$adj, long_first$slope_mem, method = "spearman", use = "complete.obs"),
  slope_exf = cor(long_first$adj, long_first$slope_exf, method = "spearman", use = "complete.obs")
)
mem_ceiling <- max(long_first$mem_first)
n_at_ceiling <- sum(long_first$mem_first >= mem_ceiling - 0.01)
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#| label: casi-selection

pct <- function(x) sprintf("%d/%d (%.0f%%)", sum(x), length(x), 100 * mean(x))
cohort_summary <- function(d) {
  c(
    Donors = as.character(nrow(d)),
    `CPS, median [IQR]` = sprintf("%.2f [%.2f–%.2f]", median(d$cps),
                                  quantile(d$cps, .25), quantile(d$cps, .75)),
    `Braak 0–III / IV / V / VI` = paste(table(d$braak_group), collapse = " / "),
    `Documented dementia` = pct(d$cog_status == "Dementia"),
    `Dementia among Braak V–VI` = pct(d$cog_status[d$braak_group %in% c("V", "VI")] == "Dementia"),
    `ε4 carrier` = pct(d$apoe_group == "ε4 carrier"),
    `ε4/ε4` = pct(d$apoe == "4/4"),
    `Age at death, median [range]` = sprintf("%.0f [%.0f–%.0f]", median(d$age), min(d$age), max(d$age))
  )
}
act_donors  <- donor %>% filter(cohort == "ACT")
adrc_donors <- donor %>% filter(cohort != "ACT")
stopifnot("the CASI set is exactly the ACT donors" = setequal(act_donors$donor_id, casi_set$donor_id))

selection_tbl <- tibble(
  Variable = names(cohort_summary(donor)),
  `ACT (has CASI)` = cohort_summary(act_donors),
  `ADRC (no CASI)` = cohort_summary(adrc_donors),
  `All donors` = cohort_summary(donor)
)
show_table(selection_tbl, caption = "Selection into the CASI analyses. Descriptive only; no CASI value is imputed for ADRC donors, and documented dementia status is not converted into a CASI-like score.")

sel <- list(
  cps_act = median(act_donors$cps), cps_adrc = median(adrc_donors$cps),
  age_act = median(act_donors$age), age_adrc = median(adrc_donors$age),
  b6_adrc = sum(adrc_donors$braak_group == "VI"), b03_adrc = sum(adrc_donors$braak_group == "0–III"),
  carriers_adrc = sum(adrc_donors$apoe_group == "ε4 carrier"),
  e44_adrc = sum(adrc_donors$apoe == "4/4"),
  cps_carrier_act = median(act_donors$cps[act_donors$apoe_group == "ε4 carrier"]),
  cps_non_act = median(act_donors$cps[act_donors$apoe_group == "ε4 noncarrier"])
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
#| label: fig2-build

apoe_counts <- donor %>% count(apoe_group, apoe, name = "n")
apoe_casi_counts <- casi_set %>% count(apoe_group, name = "n")

# --- A. genotype x Braak stage -------------------------------------------------
f2a_counts <- donor %>%
  filter(braak_stage != "I") %>% mutate(braak_stage = droplevels(braak_stage)) %>%
  count(apoe, braak_stage, .drop = FALSE) %>%
  left_join(donor %>% distinct(apoe, apoe_group), by = "apoe") %>%
  mutate(apoe = factor(apoe, levels = rev(APOE_GENOTYPES)))
f2a_totals <- donor %>% count(apoe) %>% mutate(apoe = factor(apoe, levels = rev(APOE_GENOTYPES)))

f2a <- ggplot(f2a_counts, aes(braak_stage, apoe)) +
  geom_tile(aes(fill = n), colour = "white", linewidth = 0.8) +
  geom_text(aes(label = if_else(n > 0, as.character(n), "·"), colour = n > 10),
            size = 2.6, family = FIG_FONT) +
  geom_tile(data = distinct(f2a_counts, apoe, apoe_group), aes(x = 0.2, y = apoe),
            fill = PAL_APOE[as.character(distinct(f2a_counts, apoe, apoe_group)$apoe_group)],
            width = 0.18, colour = NA, inherit.aes = FALSE) +
  geom_text(data = f2a_totals, aes(x = 6.75, y = apoe, label = n), inherit.aes = FALSE,
            hjust = 0, size = 2.4, colour = INK2, family = FIG_FONT) +
  annotate("text", x = 6.75, y = 6.65, label = "n", hjust = 0, size = 2.4, colour = INK2,
           family = FIG_FONT, fontface = "italic") +
  scale_fill_gradient(low = "#f1f0ec", high = "#3a3a37", guide = "none") +
  scale_colour_manual(values = c(`FALSE` = INK, `TRUE` = "white"), guide = "none") +
  scale_x_discrete(expand = expansion(add = c(0.9, 0.9))) +
  coord_cartesian(clip = "off") +
  labs(title = "Genotype by Braak stage (all donors)", x = "Braak stage", y = "APOE genotype") +
  theme(panel.grid.major.y = element_blank(), axis.line = element_blank(),
        axis.ticks = element_blank())

# --- B. CPS by APOE group ------------------------------------------------------
cps_by_apoe <- donor %>% group_by(apoe_group) %>%
  summarise(n = n(), median = median(cps), q1 = quantile(cps, .25), q3 = quantile(cps, .75),
            .groups = "drop")
cps_wilcox <- wilcox.test(cps ~ apoe_group, data = donor, exact = FALSE)

f2b <- ggplot(donor, aes(apoe_group, cps)) +
  geom_boxplot(width = 0.45, outlier.shape = NA, colour = INK2, fill = NA, linewidth = 0.35) +
  exec(geom_point, !!!point_args, mapping = aes(fill = apoe_group),
       position = position_jitter(width = 0.13, height = 0, seed = 1)) +
  geom_text(data = cps_by_apoe, aes(apoe_group, 1.04, label = paste0("n = ", n)),
            size = 2.4, colour = INK2, family = FIG_FONT) +
  apoe_fill() +
  scale_y_continuous(limits = c(0, 1.06), breaks = seq(0, 1, 0.25)) +
  labs(title = "Pathological burden", x = NULL, y = "CPS")

# --- C. CASI vs CPS by APOE group, with overlap made explicit --------------------
# Comparable pathology is judged on pooled CPS tertiles of the CASI set: a group
# trend is drawn only across tertiles where both groups have >= 5 donors.
cps_cuts <- quantile(casi_set$cps, c(1/3, 2/3))
overlap <- casi_set %>%
  mutate(tertile = cut(cps, c(-Inf, cps_cuts, Inf), labels = c("low", "mid", "high"))) %>%
  count(tertile, apoe_group, .drop = FALSE) %>%
  pivot_wider(names_from = apoe_group, values_from = n)
supported <- overlap %>% filter(`ε4 noncarrier` >= 5, `ε4 carrier` >= 5) %>% pull(tertile)
support_range <- c(c(min(casi_set$cps), cps_cuts, max(casi_set$cps))[min(as.integer(supported))],
                   c(min(casi_set$cps), cps_cuts, max(casi_set$cps))[max(as.integer(supported)) + 1])
tertile_mid <- c(mean(c(min(casi_set$cps), cps_cuts[1])), mean(cps_cuts), mean(c(cps_cuts[2], max(casi_set$cps))))

f2c <- ggplot(casi_set, aes(cps, casi)) +
  annotate("rect", xmin = support_range[1], xmax = support_range[2], ymin = -Inf, ymax = Inf,
           fill = "#f3f2ee") +
  geom_vline(xintercept = cps_cuts, colour = MUTED, linewidth = 0.25, linetype = "13") +
  geom_smooth(data = casi_set %>% filter(cps >= support_range[1], cps <= support_range[2]),
              aes(colour = apoe_group), method = "lm", formula = y ~ x, se = FALSE,
              linewidth = 0.6, show.legend = FALSE) +
  exec(geom_point, !!!point_args, mapping = aes(fill = apoe_group)) +
  annotate("text", x = tertile_mid, y = 103.5, size = 2.2, colour = INK2, family = FIG_FONT,
           label = sprintf("%d vs %d", overlap$`ε4 noncarrier`, overlap$`ε4 carrier`)) +
  annotate("text", x = min(casi_set$cps), y = 107.5, hjust = 0, size = 2.2, colour = INK2,
           family = FIG_FONT, fontface = "italic",
           label = "donors per CPS tertile: noncarrier vs carrier") +
  apoe_fill() +
  scale_colour_manual(values = PAL_APOE) +
  scale_x_continuous(limits = c(0.1, 0.95), breaks = seq(0.25, 1, 0.25)) +
  scale_y_continuous(limits = c(60, 108), breaks = seq(60, 100, 10)) +
  labs(title = "Cognition conditional on pathology",
       x = "CPS", y = "Last CASI score (0–100)")

# --- D. adjusted cognition by APOE group -----------------------------------------
resid_by_apoe <- t.test(adj_cognition ~ apoe_group, data = casi_set)
resid_diff <- c(est = unname(resid_by_apoe$estimate[2] - resid_by_apoe$estimate[1]),
                lo = -resid_by_apoe$conf.int[2], hi = -resid_by_apoe$conf.int[1])
no24 <- t.test(adj_cognition ~ apoe_group, data = casi_set %>% filter(apoe != "2/4"))
resid_diff_no24 <- c(est = unname(no24$estimate[2] - no24$estimate[1]),
                     lo = -no24$conf.int[2], hi = -no24$conf.int[1])
# The same contrast restricted to the CPS range where both groups are represented.
in_support <- casi_set %>% filter(cps >= support_range[1], cps <= support_range[2])
sup_t <- t.test(adj_cognition ~ apoe_group, data = in_support)
resid_diff_support <- c(est = unname(sup_t$estimate[2] - sup_t$estimate[1]),
                        lo = -sup_t$conf.int[2], hi = -sup_t$conf.int[1])

f2d <- ggplot(casi_set, aes(apoe_group, adj_cognition)) +
  geom_hline(yintercept = 0, colour = MUTED, linewidth = 0.3) +
  geom_boxplot(width = 0.45, outlier.shape = NA, colour = INK2, fill = NA, linewidth = 0.35) +
  exec(geom_point, !!!point_args, mapping = aes(fill = apoe_group),
       position = position_jitter(width = 0.13, height = 0, seed = 2)) +
  geom_text(data = apoe_casi_counts, aes(apoe_group, 17.5, label = paste0("n = ", n)),
            size = 2.4, colour = INK2, family = FIG_FONT) +
  apoe_fill() +
  scale_y_continuous(limits = c(-17, 18.5), breaks = seq(-15, 15, 5)) +
  scale_x_discrete(labels = c("ε4\nnoncarrier", "ε4\ncarrier")) +
  labs(title = "Adjusted cognition", x = NULL,
       y = "Residual CASI (points)",
       caption = sprintf("carrier − noncarrier: %s", fmt_ci(resid_diff["est"], resid_diff["lo"], resid_diff["hi"], 1)))

fig2 <- plot_grid(
  shared_legend(f2b),
  plot_grid(
    compose(f2a, f2b + no_legend, labels = c("A", "B"), rel_widths = c(1.25, 0.75), ncol = 2),
    compose(f2c + no_legend, f2d + no_legend, labels = c("C", "D"), rel_widths = c(1.3, 0.7), ncol = 2),
    ncol = 1),
  ncol = 1, rel_heights = c(0.05, 1)
)
fig2_png <- save_figure(fig2, "fig2_apoe", height_mm = 160)
fig2_cap <- sprintf("**Figure 2.** APOE and pathology and cognition. **A** Observed genotype by Braak stage for all 84 donors (source: `APOE Genotype`, no missing values); the coloured bar marks the ε4 grouping used in B–D, and ε2/ε4 (n = %d) is counted as a carrier. **B** CPS by ε4 carrier status, all donors. **C** Last CASI against CPS in the ACT donors with CASI. Dotted lines mark pooled CPS tertiles; the counts above them are noncarrier vs carrier donors per tertile; the shaded band and group lines cover only the tertiles where both groups have at least five donors. **D** Baseline-model residual (Figure 1C) by carrier status; caption gives the difference in means with Welch 95%% CI.", sum(donor$apoe == "2/4"))
#
#
#
#| label: fig2
#| fig-cap: !expr fig2_cap
#| out-width: "100%"

include_graphics(fig2_png)
#
#
#
#| label: fig2-numbers

n44 <- sum(donor$apoe == "4/4"); n44_casi <- sum(casi_set$apoe == "4/4")
med44 <- median(donor$cps[donor$apoe == "4/4"])
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#| label: fig3-build

f3_data <- donor_roi %>%
  filter(brain_region %in% ROI6, has_casi, !is.na(ptau_percent_positive_area)) %>%
  mutate(brain_region = factor(brain_region, levels = ROI6))
stopifnot("one row per donor within each ROI" =
            !anyDuplicated(f3_data[c("donor_id", "brain_region")]))

# Comparable pathology, defined transparently: within-ROI tertiles of local AT8
# percent area. Each tertile's CASI median and IQR is drawn and tabulated.
f3_data <- f3_data %>% group_by(brain_region) %>%
  mutate(at8_tertile = ntile(ptau_percent_positive_area, 3)) %>% ungroup()
f3_bins <- f3_data %>% group_by(brain_region, at8_tertile) %>%
  summarise(n = n(), at8_lo = min(ptau_percent_positive_area),
            at8_hi = max(ptau_percent_positive_area),
            x = median(ptau_percent_positive_area),
            casi_med = median(casi), casi_q1 = quantile(casi, .25), casi_q3 = quantile(casi, .75),
            casi_min = min(casi), casi_max = max(casi), .groups = "drop")
f3_bounds <- f3_bins %>% filter(at8_tertile < 3) %>%
  left_join(f3_bins %>% filter(at8_tertile > 1) %>% mutate(at8_tertile = at8_tertile - 1L) %>%
              select(brain_region, at8_tertile, next_lo = at8_lo),
            by = c("brain_region", "at8_tertile")) %>%
  mutate(boundary = (at8_hi + next_lo) / 2)
f3_labels <- f3_data %>% count(brain_region) %>%
  mutate(label = paste0(brain_region, "   n = ", n))

f3 <- ggplot(f3_data, aes(ptau_percent_positive_area, casi)) +
  geom_vline(data = f3_bounds, aes(xintercept = boundary), colour = MUTED, linewidth = 0.25,
             linetype = "13") +
  exec(geom_point, !!!point_args, mapping = aes(fill = braak_group)) +
  geom_linerange(data = f3_bins, aes(x = x, ymin = casi_q1, ymax = casi_q3), inherit.aes = FALSE,
                 colour = INK, linewidth = 0.7, position = position_nudge(x = 0)) +
  geom_point(data = f3_bins, aes(x = x, y = casi_med), inherit.aes = FALSE, shape = 23,
             size = 2, fill = "white", colour = INK, stroke = 0.5) +
  facet_wrap(~ brain_region, nrow = 2,
             labeller = as_labeller(setNames(f3_labels$label, f3_labels$brain_region))) +
  braak_fill() +
  scale_x_continuous(transform = pct_axis_trans(0.001), breaks = pct_breaks, labels = pct_label) +
  scale_y_continuous(limits = c(60, 100), breaks = seq(60, 100, 10)) +
  labs(x = "Local AT8 (pTau) positive area, % of analysed grey matter (pseudo-log scale)",
       y = "Last CASI score (0–100)") +
  theme(panel.spacing = unit(4, "mm"), legend.position = "top",
        panel.grid.major.x = element_line(colour = GRID, linewidth = 0.25))

fig3_png <- save_figure(f3, "fig3_regional_identifiability", height_mm = 125)
fig3_cap <- sprintf("**Figure 3.** Local pTau against terminal cognition in the six candidate ROIs. Each point is one ACT donor with CASI and a Grey matter AT8 value in that ROI; facets are independent donor sets of the stated size. Dotted lines divide each ROI into tertiles of local AT8, and white diamonds with bars give the CASI median and interquartile range within each tertile. Tertiles span wide burden ranges, so this shows broad spread at similar local burden, not matched pathology; the comparable-burden check below pairs donors on CPS and local AT8 jointly. No regression is drawn. The x-axis is pseudo-log (linear within ±0.001%%) so that true zeros stay visible; tick labels are in %% area.")
#
#
#
#| label: fig3
#| fig-cap: !expr fig3_cap
#| out-width: "100%"

include_graphics(fig3_png)
#
#
#
#| label: fig3-table

braak_cov <- f3_data %>% count(brain_region, braak_group, .drop = FALSE) %>%
  group_by(brain_region) %>%
  summarise(braak = paste(n, collapse = " / "), .groups = "drop")

roi_feasibility <- f3_data %>%
  group_by(brain_region) %>%
  summarise(
    complete = n(),
    at8_p10 = quantile(ptau_percent_positive_area, .1),
    at8_med = median(ptau_percent_positive_area),
    at8_p90 = quantile(ptau_percent_positive_area, .9),
    rho = cor(ptau_percent_positive_area, casi, method = "spearman"),
    n_T = sum(!is.na(competitive_score_mg4) & !is.na(competitive_score_astro)),
    .groups = "drop") %>%
  left_join(braak_cov, by = "brain_region") %>%
  left_join(f3_bins %>% group_by(brain_region) %>%
              summarise(spread = paste(sprintf("%.0f", casi_q3 - casi_q1), collapse = " / "),
                        top = sprintf("%.0f–%.0f", casi_min[at8_tertile == 3], casi_max[at8_tertile == 3]),
                        top_at8 = at8_lo[at8_tertile == 3], .groups = "drop"),
            by = "brain_region")

# The reading column is written by hand, so the facts it rests on are asserted.
rf <- function(roi, col) roi_feasibility[[col]][roi_feasibility$brain_region == roi]
stopifnot(
  "V1C pTau sits near the floor for most donors" = rf("V1C", "at8_med") < 0.05,
  "MEC and HIP carry substantial pTau even in the lowest tertile" =
    all(f3_bins$at8_hi[f3_bins$at8_tertile == 1 & f3_bins$brain_region %in% c("MEC", "HIP")] > 1),
  "STG, V1C and HIP have molecular state for about half the donors" =
    all(c(rf("STG", "n_T"), rf("V1C", "n_T"), rf("HIP", "n_T")) < 0.55 * 69),
  "MTG, DFC and MEC have molecular state for nearly every donor" =
    all(c(rf("MTG", "n_T"), rf("DFC", "n_T"), rf("MEC", "n_T")) >= 0.9 * c(rf("MTG", "complete"), rf("DFC", "complete"), rf("MEC", "complete")))
)
reading <- c(
  MTG = "Spread in the middle and upper AT8 tertiles; complete molecular coverage.",
  DFC = "Spread mainly in the upper tertile; lower absolute pTau than MTG; near-complete molecular coverage.",
  MEC = "Early-involved: pTau already > 1% in the lowest tertile, so all spread is at high burden; strongest NeuN change (Fig. 4).",
  STG = "Spread similar to MTG; molecular state for about half the donors.",
  V1C = "pTau near detection floor for most donors; spread appears only in the top tertile. Mainly a late-progression region.",
  HIP = "Early-involved like MEC; spread at high burden; molecular state for about half the donors."
)

roi_feasibility %>%
  mutate(brain_region = as.character(brain_region),
         range = sprintf("%s–%s (median %s)", pct_label(signif(at8_p10, 2)),
                         pct_label(signif(at8_p90, 2)), pct_label(signif(at8_med, 2))),
         spread_txt = sprintf("IQR %s; top tertile (≥ %s%%) CASI %s", spread,
                              pct_label(signif(top_at8, 2)), top),
         reading = reading[brain_region]) %>%
  transmute(ROI = brain_region, `Complete donors` = complete,
            `Braak coverage (0–III / IV / V / VI)` = braak,
            `Local pTau, % area (P10–P90)` = range,
            `Spearman ρ, AT8 vs CASI` = sprintf("%.2f", rho),
            `Broad spread (CASI IQR by AT8 tertile, low / mid / high; top-tertile range)` = spread_txt,
            `MG4/Astro coverage` = sprintf("%d/%d", n_T, complete),
            Reading = reading) %>%
  show_table(caption = "Regional feasibility. Complete = ACT donors with CASI and a local AT8 value. Tertiles are within ROI, the same bins as in Figure 3; a tertile is a wide burden band, not a matched set. MG4/Astro coverage counts those donors who also have both competitive molecular-state scores.")
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#| label: fig3-comparable

CPS_TOL <- 0.05
LOG_TOL <- log10(2)
CASI_DISCORDANT <- 10
cps_elevated <- median(casi_set$cps)

pair_base <- f3_data %>%
  mutate(brain_region = as.character(brain_region),
         l_at8 = log10(ptau_percent_positive_area + 0.001),
         l_ab = log10(abeta_percent_positive_area + 0.001),
         has_T = !is.na(competitive_score_mg4) & !is.na(competitive_score_astro)) %>%
  filter(cps > cps_elevated) %>%
  select(brain_region, donor_id, cps, l_at8, l_ab, casi, has_T)
pairs <- inner_join(pair_base, pair_base, by = "brain_region", suffix = c("_a", "_b"),
                    relationship = "many-to-many") %>%
  filter(donor_id_a < donor_id_b) %>%
  mutate(d_casi = abs(casi_a - casi_b),
         matched = abs(cps_a - cps_b) <= CPS_TOL & abs(l_at8_a - l_at8_b) <= LOG_TOL,
         matched_ab = matched & !is.na(l_ab_a) & !is.na(l_ab_b) & abs(l_ab_a - l_ab_b) <= LOG_TOL,
         discordant = d_casi >= CASI_DISCORDANT)

donors_in <- function(d) n_distinct(c(d$donor_id_a, d$donor_id_b))
comparable <- pairs %>% group_by(brain_region) %>%
  group_modify(function(d, key) tibble(
    elevated_donors = n_distinct(c(d$donor_id_a, d$donor_id_b)),
    median_all = median(d$d_casi),
    matched_donors = donors_in(filter(d, matched)),
    median_matched = median(d$d_casi[d$matched]),
    disc_donors = donors_in(filter(d, matched, discordant)),
    disc_ab_donors = donors_in(filter(d, matched_ab, discordant)),
    disc_T_donors = donors_in(filter(d, matched, discordant, has_T_a, has_T_b))
  )) %>% ungroup() %>%
  mutate(brain_region = factor(brain_region, levels = ROI6)) %>% arrange(brain_region) %>%
  # Declared support rule: a narrow contrast is "observed" when at least 10 donors sit in a
  # discordant matched pair; "insufficient" when fewer than 10 donors have any matched partner.
  mutate(support = case_when(
    matched_donors < 10 ~ "insufficient support for a narrow contrast",
    disc_donors >= 10   ~ "discordance observed at comparable CPS + local AT8",
    TRUE                ~ "broad spread only"),
    # Same rule applied to pairs where both donors also carry MG4 and astrocyte scores.
    support_T = case_when(disc_T_donors >= 10 ~ "supported",
                          disc_T_donors >= 5  ~ "limited",
                          TRUE                ~ "insufficient"),
    support = paste0(support, "; with molecular state: ", support_T))

comparable %>%
  transmute(ROI = brain_region,
            `Elevated-burden donors` = elevated_donors,
            `Donors with a matched partner` = matched_donors,
            `Median |ΔCASI|, matched (all elevated pairs)` = sprintf("%.1f (%.1f)", median_matched, median_all),
            `Donors in a discordant matched pair` = disc_donors,
            `… also matched on Aβ` = disc_ab_donors,
            `… both with MG4/Astro` = disc_T_donors,
            Reading = support) %>%
  show_table(caption = sprintf("Comparable-burden check at elevated burden (CPS > %.2f, the CASI-set median). Matched = |ΔCPS| ≤ %.2f and local AT8 within a factor of 2; discordant = |ΔCASI| ≥ %d points. Counts are distinct donors.", cps_elevated, CPS_TOL, CASI_DISCORDANT))

cmp <- function(roi, col) comparable[[col]][comparable$brain_region == roi]
stopifnot(
  "most elevated-burden donors have a matched partner" =
    all(comparable$matched_donors >= 0.8 * comparable$elevated_donors),
  "DFC has the fewest Abeta-matched discordant donors, and the smallest matched spread" =
    cmp("DFC", "disc_ab_donors") == min(comparable$disc_ab_donors) &&
    cmp("DFC", "median_matched") == min(comparable$median_matched),
  "joint molecular support is lower in STG, HIP and V1C than in MTG, DFC and MEC" =
    max(comparable$disc_T_donors[comparable$brain_region %in% c("STG", "HIP", "V1C")]) <
      min(comparable$disc_T_donors[comparable$brain_region %in% ROI3])
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
#| label: fig4-build

f4_data <- donor_roi %>%
  filter(brain_region %in% ROI3) %>%
  mutate(brain_region = factor(brain_region, levels = ROI3))
braak_n_all <- donor %>% count(braak_group, .drop = FALSE)
braak_axis_labels <- setNames(sprintf("%s\nn = %d", braak_n_all$braak_group, braak_n_all$n),
                              braak_n_all$braak_group)

f4_panel <- function(var, title, ylab, trans = "identity", breaks = waiver(), labels = waiver()) {
  d <- f4_data %>% filter(!is.na(.data[[var]]))
  s <- d %>% group_by(brain_region, braak_group) %>%
    summarise(med = median(.data[[var]]), q1 = quantile(.data[[var]], .25),
              q3 = quantile(.data[[var]], .75), .groups = "drop")
  dodge <- position_dodge(width = 0.75)
  ggplot(d, aes(braak_group, .data[[var]], fill = brain_region)) +
    geom_point(shape = 21, size = 1.3, stroke = 0.15, colour = OUTLINE, alpha = 0.55,
               position = position_jitterdodge(jitter.width = 0.18, dodge.width = 0.75, seed = 3)) +
    geom_linerange(data = s, aes(x = braak_group, ymin = q1, ymax = q3, group = brain_region),
                   position = dodge, colour = INK, linewidth = 0.55, inherit.aes = FALSE) +
    geom_point(data = s, aes(braak_group, med, fill = brain_region), position = dodge,
               shape = 23, size = 2.1, stroke = 0.45, colour = INK, inherit.aes = FALSE) +
    roi_fill() +
    scale_x_discrete(labels = braak_axis_labels, drop = FALSE) +
    scale_y_continuous(transform = trans, breaks = breaks, labels = labels) +
    labs(title = title, x = "Braak stage", y = ylab)
}

f4a <- f4_panel("abeta_percent_positive_area", "Aβ (6E10)", "Positive area (% grey matter)",
                trans = pct_axis_trans(0.01), breaks = pct_breaks, labels = pct_label)
f4b <- f4_panel("ptau_percent_positive_area", "pTau (AT8)", "Positive area (% grey matter)",
                trans = pct_axis_trans(0.01), breaks = pct_breaks, labels = pct_label)
f4c <- f4_panel("neun_per_mm2", "NeuN", "Positive cells per mm² analysed")
f4d <- f4_panel("gfap_percent_positive_area", "GFAP", "Positive area (% grey matter)")

fig4 <- plot_grid(
  shared_legend(f4a + guides(fill = guide_legend(override.aes = list(size = 2.4, alpha = 1)))),
  compose(f4a + no_legend, f4b + no_legend, f4c + no_legend, f4d + no_legend, ncol = 2),
  ncol = 1, rel_heights = c(0.05, 1)
)
fig4_png <- save_figure(fig4, "fig4_tissue_response", height_mm = 150)
fig4_cap <- "**Figure 4.** Grey matter QNP measures by Braak stage in MTG, DFC and MEC, all donors with a value (MTG 84, DFC 84, MEC 83). Small points are donors; diamonds with bars are the median and interquartile range for each region × stage. Each panel keeps its own units and scale; Aβ and pTau use a pseudo-log axis (linear within ±0.01%) with tick labels in % area. Braak 0–III pools 2, 4 and 6 donors at stages 0, II and III; no donor is at stage I. Cross-sectional stage groups are different donors, not trajectories. Absolute GFAP levels are not assumed comparable across regions."
#
#
#
#| label: fig4
#| fig-cap: !expr fig4_cap
#| out-width: "100%"

include_graphics(fig4_png)
#
#
#
#| label: fig4-numbers

braak_num <- c("0" = 0, "I" = 1, "II" = 2, "III" = 3, "IV" = 4, "V" = 5, "VI" = 6)
scor <- function(a, b) {
  ok <- !is.na(a) & !is.na(b)
  if (sum(ok) < 10) NA_real_ else cor(a[ok], b[ok], method = "spearman")
}
all_roi_stage <- donor_roi %>%
  mutate(b = braak_num[as.character(braak_stage)]) %>%
  group_by(brain_region) %>%
  summarise(n = sum(!is.na(abeta_percent_positive_area)),
            `Aβ` = scor(abeta_percent_positive_area, b), pTau = scor(ptau_percent_positive_area, b),
            NeuN = scor(neun_per_mm2, b), GFAP = scor(gfap_percent_positive_area, b),
            `NeuN vs local pTau` = scor(neun_per_mm2, ptau_percent_positive_area),
            `GFAP vs local pTau` = scor(gfap_percent_positive_area, ptau_percent_positive_area),
            .groups = "drop") %>%
  filter(!is.na(`Aβ`))

m4 <- f4_data %>% group_by(brain_region, braak_group) %>%
  summarise(neun = median(neun_per_mm2, na.rm = TRUE), gfap = median(gfap_percent_positive_area, na.rm = TRUE),
            .groups = "drop")
mec_neun <- m4 %>% filter(brain_region == "MEC")
mtg_gfap <- m4 %>% filter(brain_region == "MTG")
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#| label: fig4-all-regions

all_roi_stage %>%
  mutate(across(where(is.numeric) & !n, ~ if_else(is.na(.x), "–", sprintf("%.2f", .x)))) %>%
  rename(Region = brain_region, Donors = n) %>%
  show_table(caption = "Grey matter, all donors with a value. NeuN and GFAP are not released for AnG or FI. The remaining ROIs show the same qualitative split as Figure 4 (pTau tracks stage more tightly than Aβ; NeuN barely tracks stage; GFAP tracks it only in some regions), so no second grid is drawn.")
#
#
#
#
#
#
#
#
#
#| label: fig5-build

f5_data <- donor_roi %>%
  filter(brain_region %in% ROI3, !is.na(competitive_score_mg4) | !is.na(competitive_score_astro)) %>%
  mutate(brain_region = factor(brain_region, levels = ROI3))
support_levels <- unique(c(f5_data$support_status_mg4, f5_data$support_status_astro))
support_levels <- support_levels[!is.na(support_levels)]
min_nuclei <- min(c(f5_data$n_nuclei_mg4, f5_data$n_nuclei_astro), na.rm = TRUE)

# Within an ROI each donor contributes one row, so an ordinary bootstrap over
# rows is a bootstrap over donors. The pooled estimate (mean of the three
# within-ROI rho) resamples donors, carrying all of a donor's ROIs together, so
# repeated donors are never treated as independent.
B <- 2000
boot_rho <- function(x, y) {
  ok <- !is.na(x) & !is.na(y); x <- x[ok]; y <- y[ok]
  est <- cor(x, y, method = "spearman")
  bs <- replicate(B, { i <- sample.int(length(x), replace = TRUE); cor(x[i], y[i], method = "spearman") })
  tibble(n = length(x), rho = est, lo = quantile(bs, .025, na.rm = TRUE), hi = quantile(bs, .975, na.rm = TRUE))
}
pooled_rho <- function(score, covariate, data = f5_data) {
  d <- data %>% filter(!is.na(.data[[score]]), !is.na(.data[[covariate]]))
  by_donor <- split(d[c("brain_region", score, covariate)], d$donor_id)
  ids <- names(by_donor)
  stat <- function(dd) mean(map_dbl(split(dd, dd$brain_region, drop = TRUE),
                                    ~ cor(.x[[score]], .x[[covariate]], method = "spearman")))
  bs <- replicate(B, stat(bind_rows(by_donor[sample(ids, replace = TRUE)])))
  tibble(n_donors = length(ids), rho = stat(d), lo = quantile(bs, .025), hi = quantile(bs, .975))
}

score_cols <- c(MG4 = "competitive_score_mg4", Astro = "competitive_score_astro", MG8 = "competitive_score_mg8")
cov_cols <- c(CPS = "cps", `Adjusted cognition` = "adj_cognition",
              `Local AT8` = "ptau_percent_positive_area", NeuN = "neun_per_mm2",
              GFAP = "gfap_percent_positive_area")
rho_tbl <- expand_grid(score = names(score_cols), covariate = names(cov_cols), roi = ROI3) %>%
  mutate(res = pmap(list(score, covariate, roi), function(s, cv, r) {
    d <- f5_data %>% filter(brain_region == r)
    boot_rho(d[[score_cols[[s]]]], d[[cov_cols[[cv]]]])
  })) %>% unnest(res)
pooled_tbl <- expand_grid(score = c("MG4", "Astro"), covariate = c("CPS", "Adjusted cognition")) %>%
  mutate(res = map2(score, covariate, ~ pooled_rho(score_cols[[.x]], cov_cols[[.y]]))) %>%
  unnest(res)

f5_panel <- function(score, covariate, ylab, xlab, title) {
  d <- f5_data %>% filter(!is.na(.data[[score_cols[[score]]]]), !is.na(.data[[cov_cols[[covariate]]]]))
  est <- rho_tbl %>% filter(score == !!score, covariate == !!covariate) %>%
    mutate(txt = sprintf("%s  ρ %+.2f [%+.2f, %+.2f], n = %d", roi, rho, lo, hi, n))
  ggplot(d, aes(.data[[cov_cols[[covariate]]]], .data[[score_cols[[score]]]])) +
    { if (covariate == "Adjusted cognition") geom_vline(xintercept = 0, colour = MUTED, linewidth = 0.3) } +
    exec(geom_point, !!!modifyList(point_args, list(size = 1.5, alpha = 0.8)),
         mapping = aes(fill = brain_region)) +
    geom_smooth(aes(colour = brain_region), method = "lm", formula = y ~ x, se = FALSE,
                linewidth = 0.5, show.legend = FALSE) +
    roi_fill() + roi_colour() +
    scale_y_continuous(limits = range(f5_data[[score_cols[[score]]]], na.rm = TRUE)) +
    labs(title = title, subtitle = paste(est$txt, collapse = "\n"),
         x = xlab, y = ylab) +
    theme(plot.subtitle = element_text(size = 6.2, lineheight = 1.15))
}

lab_mg4 <- "MG4 competitive score\n(Micro/PVM)"
lab_ast <- "Astrocyte activation\ncompetitive score"
lab_res <- "Residual CASI (points), > 0 better than expected"
f5a <- f5_panel("MG4", "CPS", lab_mg4, "CPS", "MG4 vs disease progression")
f5b <- f5_panel("MG4", "Adjusted cognition", lab_mg4, lab_res, "MG4 vs adjusted cognition")
f5c <- f5_panel("Astro", "CPS", lab_ast, "CPS", "Astrocyte activation vs disease progression")
f5d <- f5_panel("Astro", "Adjusted cognition", lab_ast, lab_res, "Astrocyte activation vs adjusted cognition")

fig5 <- plot_grid(
  shared_legend(f5a + guides(fill = guide_legend(override.aes = list(size = 2.4)))),
  compose(f5a + no_legend, f5b + no_legend, f5c + no_legend, f5d + no_legend, ncol = 2),
  ncol = 1, rel_heights = c(0.05, 1)
)
fig5_png <- save_figure(fig5, "fig5_molecular_state", height_mm = 165)
fig5_cap <- sprintf("**Figure 5.** Competitive molecular-state scores (notebook 03: assay-adjusted, nucleus-weighted, matched-background corrected) in MTG, DFC and MEC against the canonical CPS (left) and against the exploratory adjusted cognition of Figure 1C (right; ACT donors only). Points are donor × ROI; lines are per-ROI least-squares fits, descriptive only. Estimates are within-ROI Spearman ρ with donor-bootstrap 95%% CI (%d resamples); within an ROI each donor appears once. Support: every plotted value is `%s` (minimum %d nuclei).", B, paste(support_levels, collapse = ", "), min_nuclei)
#
#
#
#| label: fig5
#| fig-cap: !expr fig5_cap
#| out-width: "100%"

include_graphics(fig5_png)
#
#
#
#| label: fig5-numbers

pr <- function(s, cv) pooled_tbl %>% filter(score == s, covariate == cv)
fmt_pooled <- function(s, cv) { x <- pr(s, cv); fmt_ci(x$rho, x$lo, x$hi) }

# Sensitivity checks, reported in text: does the MG4 - adjusted-cognition
# association survive local pTau, and does APOE change the stage association?
sens <- map_dfr(ROI3, function(r) {
  d <- f5_data %>% filter(brain_region == r, !is.na(adj_cognition), !is.na(competitive_score_mg4),
                          !is.na(ptau_percent_positive_area))
  fit <- lm(adj_cognition ~ scale(competitive_score_mg4) + log10(ptau_percent_positive_area + 0.01), data = d)
  ci <- confint(fit)[2, ]
  a <- f5_data %>% filter(brain_region == r)
  fit_apoe <- lm(scale(competitive_score_mg4) ~ cps + apoe_group, data = a)
  fit_apoe_a <- lm(scale(competitive_score_astro) ~ cps + apoe_group, data = a)
  tibble(ROI = r, n = nobs(fit),
         mg4_given_ptau = fmt_ci(coef(fit)[2], ci[1], ci[2], 1),
         mg4_carrier = fmt_ci(coef(fit_apoe)[3], confint(fit_apoe)[3, 1], confint(fit_apoe)[3, 2]),
         astro_carrier = fmt_ci(coef(fit_apoe_a)[3], confint(fit_apoe_a)[3, 1], confint(fit_apoe_a)[3, 2]),
         given_ptau_est = coef(fit)[2], given_ptau_lo = ci[1], given_ptau_hi = ci[2],
         apoe_ci_has_zero = confint(fit_apoe)[3, 1] < 0 & confint(fit_apoe)[3, 2] > 0 &
                            confint(fit_apoe_a)[3, 1] < 0 & confint(fit_apoe_a)[3, 2] > 0)
})

# The prose below states these; assert them so a changed input cannot leave it stale.
rc <- function(s, cv, r) rho_tbl %>% filter(score == s, covariate == cv, roi == r)
stopifnot(
  "MG4 - adjusted cognition is negative in all three ROIs" =
    all(rho_tbl$rho[rho_tbl$score == "MG4" & rho_tbl$covariate == "Adjusted cognition"] < 0),
  "its MEC interval crosses zero" = rc("MG4", "Adjusted cognition", "MEC")$hi > 0,
  "MG4 given local AT8 stays negative; interval includes zero in DFC and MEC" =
    all(sens$given_ptau_est < 0) && all(sens$given_ptau_hi[sens$ROI %in% c("DFC", "MEC")] > 0),
  "no APOE carrier contrast excludes zero" = all(sens$apoe_ci_has_zero),
  "MG8 shows no cognition association" =
    all(abs(rho_tbl$rho[rho_tbl$score == "MG8" & rho_tbl$covariate == "Adjusted cognition"]) < 0.1),
  "astrocyte activation is inversely associated with NeuN in all three ROIs" =
    all(rho_tbl$hi[rho_tbl$score == "Astro" & rho_tbl$covariate == "NeuN"] < 0),
  "MG4 is not associated with NeuN" =
    all(rho_tbl$lo[rho_tbl$score == "MG4" & rho_tbl$covariate == "NeuN"] < 0 &
        rho_tbl$hi[rho_tbl$score == "MG4" & rho_tbl$covariate == "NeuN"] > 0)
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
#| label: fig5-joint

zstd <- function(x) (x - mean(x)) / sd(x)
joint_fit <- function(r, score) {
  d <- donor_roi %>%
    filter(brain_region == r, has_casi, !is.na(.data[[score_cols[[score]]]])) %>%
    mutate(state = zstd(.data[[score_cols[[score]]]]),
           late_2_3 = late %in% c("LATE Stage 2", "LATE Stage 3"),
           arteriolo_severe = arteriolo == "Severe")
  stopifnot("no missing model covariate" =
              !anyNA(d[c("casi", "cps", "age", "sex", "education", "casi_interval", "state")]),
            "no unclassifiable LATE rating in the CASI set" = all(d$late != "Unclassifiable"),
            "one row per donor" = !anyDuplicated(d$donor_id))
  base  <- lm(CASI_FORMULA, data = d)
  joint <- update(base, . ~ . + state)
  base_c  <- update(base, . ~ . + late_2_3 + arteriolo_severe)
  joint_c <- update(base_c, . ~ . + state)
  stopifnot("all four models use the identical donors" =
              length(unique(c(nobs(base), nobs(joint), nobs(base_c), nobs(joint_c), nrow(d)))) == 1)
  r2 <- function(m) summary(m)$r.squared
  ar2 <- function(m) summary(m)$adj.r.squared
  ci <- confint(joint)["state", ]; ci_c <- confint(joint_c)["state", ]
  tibble(ROI = r, score = score, n = nrow(d),
         est = coef(joint)["state"], lo = ci[1], hi = ci[2],
         r2_base = r2(base), d_r2 = r2(joint) - r2(base), d_adj = ar2(joint) - ar2(base),
         est_c = coef(joint_c)["state"], lo_c = ci_c[1], hi_c = ci_c[2],
         d_r2_c = r2(joint_c) - r2(base_c))
}
joint_tbl <- expand_grid(ROI = ROI3, score = c("MG4", "Astro")) %>%
  pmap_dfr(function(ROI, score) joint_fit(ROI, score))

joint_tbl %>%
  mutate(Score = recode(score, MG4 = "MG4", Astro = "Astrocyte activation")) %>%
  transmute(Score, ROI, Donors = n,
            `Baseline R²` = sprintf("%.2f", r2_base),
            `State, CASI points per SD [95% CI]` = fmt_ci(est, lo, hi, 1),
            `ΔR² (Δ adjusted R²)` = sprintf("%+.3f (%+.3f)", d_r2, d_adj),
            `+ LATE + arteriolosclerosis: state per SD` = fmt_ci(est_c, lo_c, hi_c, 1),
            `ΔR² over co-pathology baseline` = sprintf("%+.3f", d_r2_c)) %>%
  arrange(factor(Score, levels = c("MG4", "Astrocyte activation"))) %>%
  show_table(caption = "Joint adjusted models, one per ROI and score. Every row compares nested models on the identical donors. The state estimate is a pathology- and covariate-adjusted association, not an effect.")

jt <- function(r, s, col) joint_tbl[[col]][joint_tbl$ROI == r & joint_tbl$score == s]
jci <- function(r, s, c = "") fmt_ci(jt(r, s, paste0("est", c)), jt(r, s, paste0("lo", c)), jt(r, s, paste0("hi", c)), 1)
stopifnot(
  "MG4 joint estimate excludes zero in MTG and DFC, not in MEC" =
    jt("MTG", "MG4", "hi") < 0 && jt("DFC", "MG4", "hi") < 0 && jt("MEC", "MG4", "hi") > 0,
  "no astrocyte joint estimate excludes zero" = all(joint_tbl$hi[joint_tbl$score == "Astro"] > 0),
  "MG4 stays negative with LATE and arteriolosclerosis added" =
    all(joint_tbl$est_c[joint_tbl$score == "MG4"] < 0),
  "with co-pathology, the MG4 interval excludes zero only in MTG" =
    jt("MTG", "MG4", "hi_c") < 0 && jt("DFC", "MG4", "hi_c") > 0 && jt("MEC", "MG4", "hi_c") > 0
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
#| label: fig5-support

rho_tbl %>%
  mutate(cell = sprintf("%+.2f [%+.2f, %+.2f]", rho, lo, hi)) %>%
  select(score, roi, covariate, cell) %>%
  pivot_wider(names_from = covariate, values_from = cell) %>%
  mutate(score = factor(score, levels = c("MG4", "Astro", "MG8"))) %>%
  arrange(score, factor(roi, levels = ROI3)) %>%
  rename(Score = score, ROI = roi) %>%
  show_table(caption = "Within-ROI Spearman ρ [donor-bootstrap 95% CI] of each competitive score against disease stage, local pathology, tissue measures and adjusted cognition. MG8 is the secondary Micro/PVM sensitivity variable.")

sens %>%
  select(ROI, n, mg4_given_ptau, mg4_carrier, astro_carrier) %>%
  rename(`n (ACT)` = n,
         `Adjusted cognition per SD of MG4, given local AT8 (points)` = mg4_given_ptau,
         `MG4 carrier − noncarrier, given CPS (SD)` = mg4_carrier,
         `Astro carrier − noncarrier, given CPS (SD)` = astro_carrier) %>%
  show_table(caption = "Sensitivity: the MG4–cognition association with local pTau added, and the APOE ε4 carrier contrast in each score at fixed CPS. Estimates with 95% CI from within-ROI linear models, one row per donor.")
#
#
#
#
#
#
#
#
#
#| label: cohort-table

roi_counts <- function(filter_expr) {
  donor_roi %>% filter(brain_region %in% ROI6, {{ filter_expr }}) %>%
    count(brain_region) %>% mutate(brain_region = factor(brain_region, levels = ROI6)) %>%
    arrange(brain_region) %>% mutate(s = paste(brain_region, n)) %>% pull(s) %>% paste(collapse = " · ")
}
n_long <- n_distinct(cognition$donor_id[!is.na(cognition$MEM_E)])
apoe_txt <- donor %>% count(apoe) %>% mutate(s = paste0(apoe, " ", n)) %>% pull(s) %>% paste(collapse = " · ")

tibble::tribble(
  ~`Analysis set`, ~Donors, ~`Replication unit`, ~Used, ~`Restriction and source`,
  "All SEA-AD donors", "84 (ACT 69, ADRC 15)", "donor", "1B, 2A–B",
    sprintf("APOE, Braak, cognitive status and CPS complete; genotypes %s", apoe_txt),
  "Donors with terminal CASI", sprintf("%d (all ACT)", nrow(casi_set)), "donor", "1A, 1C, 2C–D, 3, 5 right",
    sprintf("`Last CASI Score` absent for all 15 ADRC donors; %d of 15 Braak VI donors have it", sum(casi_set$braak_stage == "VI")),
  "Harmonized longitudinal cognition", sprintf("%d with memory scores; %d ACT used", n_long, nrow(long_first)), "donor × visit", "1D",
    "ages at visit in whole years; ~biennial visits",
  "QNP Grey matter, P + H", roi_counts(!is.na(abeta_percent_positive_area) & !is.na(ptau_percent_positive_area) & !is.na(neun_per_mm2) & !is.na(gfap_percent_positive_area)),
    "donor × ROI (donor is the biological unit)", "3 (AT8, six ROIs); 4 (MTG/DFC/MEC)", "`qnp_primary_long()`, June 2026 release; Figure 3 needs only AT8, which is complete in all six ROIs",
  "Competitive MG4 + astrocyte", roi_counts(!is.na(competitive_score_mg4) & !is.na(competitive_score_astro)),
    "donor × ROI (donor is the biological unit)", "5 (MTG/DFC/MEC)", "`molecular_state_analysis_table.csv`; every value well supported"
) %>%
  show_table(caption = "Cohort and coverage. Figures that pool ROIs never treat a donor's ROIs as independent donors: per-ROI estimates use one row per donor, and the pooled Figure 5 estimate resamples donors.")
#
#
#
#
#
#| label: copathology

# Each variable is a donor-level neuropathology rating. "Not identified" is a
# rating, not a missing value; LATE "Unclassifiable" is excluded from its own
# denominator rather than counted as absent. For Lewy body disease, "Not
# Identified (olfactory bulb not assessed)" is kept as not-identified in the
# brain: limbic and neocortical staging does not depend on the olfactory bulb.
copath <- donor %>%
  mutate(
    `LATE-NC stage 2–3` = if_else(late == "Unclassifiable", NA, late %in% c("LATE Stage 2", "LATE Stage 3")),
    `Lewy body disease, limbic or neocortical` = lewy %in% c("Limbic (Transitional)", "Neocortical (Diffuse)"),
    `CAA moderate–severe` = caa %in% c("Moderate", "Severe"),
    `Arteriolosclerosis severe` = arteriolo == "Severe",
    `≥ 1 microinfarct` = microinfarcts >= 1,
    `Atherosclerosis moderate–severe` = athero %in% c("Moderate", "Severe")
  )
copath_vars <- c("LATE-NC stage 2–3", "Lewy body disease, limbic or neocortical", "CAA moderate–severe",
                 "Arteriolosclerosis severe", "≥ 1 microinfarct", "Atherosclerosis moderate–severe")
stopifnot("no co-pathology rating is missing apart from LATE unclassifiable" =
            !anyNA(donor[c("late", "lewy", "caa", "arteriolo", "athero", "microinfarcts")]))

groups <- list(
  `Residual > 0` = copath$adj_cognition > 0 & !is.na(copath$adj_cognition),
  `Residual ≤ 0` = copath$adj_cognition <= 0 & !is.na(copath$adj_cognition),
  `Braak V–VI, no dementia` = copath$braak_stage %in% c("V", "VI") & copath$cog_status == "No dementia",
  `Braak V–VI, dementia` = copath$braak_stage %in% c("V", "VI") & copath$cog_status == "Dementia",
  `ε4 noncarrier` = copath$apoe_group == "ε4 noncarrier",
  `ε4 carrier` = copath$apoe_group == "ε4 carrier"
)
cell <- function(v, g) {
  x <- copath[[v]][g]; x <- x[!is.na(x)]
  sprintf("%d/%d (%.0f%%)", sum(x), length(x), 100 * mean(x))
}
copath_tbl <- map_dfr(copath_vars, function(v) {
  as_tibble(c(list(`Co-pathology` = v), map(groups, ~ cell(v, .x))))
})
header_n <- map_int(groups, sum)
names(copath_tbl)[-1] <- sprintf("%s (n = %d)", names(groups), header_n)
show_table(copath_tbl, caption = "Co-pathology by phenotype. Residual groups are the 69 ACT donors with CASI; the Braak V–VI contrast uses documented cognitive status in all donors; APOE uses all donors. Counts are k/n with n the donors rated.")

late_diff <- mean(copath$`LATE-NC stage 2–3`[groups$`Braak V–VI, dementia`], na.rm = TRUE) -
  mean(copath$`LATE-NC stage 2–3`[groups$`Braak V–VI, no dementia`], na.rm = TRUE)
late_resid <- mean(copath$`LATE-NC stage 2–3`[groups$`Residual ≤ 0`], na.rm = TRUE) -
  mean(copath$`LATE-NC stage 2–3`[groups$`Residual > 0`], na.rm = TRUE)
lewy_diff <- mean(copath$`Lewy body disease, limbic or neocortical`[groups$`Braak V–VI, dementia`]) -
  mean(copath$`Lewy body disease, limbic or neocortical`[groups$`Braak V–VI, no dementia`])
arterio_resid <- mean(copath$`Arteriolosclerosis severe`[groups$`Residual ≤ 0`]) -
  mean(copath$`Arteriolosclerosis severe`[groups$`Residual > 0`])
caa_apoe <- mean(copath$`CAA moderate–severe`[groups$`ε4 carrier`]) -
  mean(copath$`CAA moderate–severe`[groups$`ε4 noncarrier`])
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#| label: copath-model

# Source fields, verified before use: donor metadata `LATE` and `Arteriolosclerosis`.
late_levels <- c("Not Identified", "LATE Stage 1", "LATE Stage 2", "LATE Stage 3", "Unclassifiable")
art_levels <- c("Mild", "Moderate", "Severe")
stopifnot("LATE and Arteriolosclerosis columns present" =
            all(c("LATE", "Arteriolosclerosis") %in% names(donor_meta)),
          "only the documented LATE levels occur" = all(donor$late %in% late_levels),
          "only the documented arteriolosclerosis levels occur" = all(donor$arteriolo %in% art_levels))
late_counts <- table(factor(casi_set$late, levels = late_levels))
art_counts <- table(factor(casi_set$arteriolo, levels = art_levels))

# Same two indicators as the co-pathology table, not a larger set.
cp_set <- casi_set %>%
  filter(late != "Unclassifiable") %>%
  mutate(late_2_3 = late %in% c("LATE Stage 2", "LATE Stage 3"),
         arteriolo_severe = arteriolo == "Severe")
cp_base <- lm(CASI_FORMULA, data = cp_set)
cp_sens <- update(cp_base, . ~ . + late_2_3 + arteriolo_severe)
stopifnot("both models use the identical complete-case donors" =
            nobs(cp_base) == nobs(cp_sens) && nobs(cp_base) == nrow(cp_set))
cross <- table(LATE_2_3 = cp_set$late_2_3, arteriolo_severe = cp_set$arteriolo_severe)
stopifnot("no empty co-pathology cell" = all(cross > 0))

term_ci <- function(m, term) fmt_ci(coef(m)[term], confint(m)[term, 1], confint(m)[term, 2], 1)
tibble(
  Term = c("CPS (per unit)", "LATE-NC stage 2–3", "Arteriolosclerosis severe", "R² (adjusted)", "Residual SD"),
  Baseline = c(term_ci(cp_base, "cps"), "–", "–",
               sprintf("%.2f (%.2f)", summary(cp_base)$r.squared, summary(cp_base)$adj.r.squared),
               sprintf("%.1f", sigma(cp_base))),
  `+ LATE + arteriolosclerosis` = c(term_ci(cp_sens, "cps"), term_ci(cp_sens, "late_2_3TRUE"),
               term_ci(cp_sens, "arteriolo_severeTRUE"),
               sprintf("%.2f (%.2f)", summary(cp_sens)$r.squared, summary(cp_sens)$adj.r.squared),
               sprintf("%.1f", sigma(cp_sens)))
) %>%
  show_table(caption = sprintf("Baseline vs co-pathology-adjusted CASI model on the identical %d ACT donors (CASI points, 95%% CI). Age, sex, education and CASI-to-death interval are in both models and not shown.", nobs(cp_base)))

cp_resid_cor <- cor(residuals(cp_base), residuals(cp_sens))
stopifnot("the CPS coefficient changes by < 10% with co-pathology added" =
            abs(coef(cp_sens)["cps"] / coef(cp_base)["cps"] - 1) < 0.10)
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
