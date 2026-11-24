# QNP pipeline reuse audit

Goal: run the QNP pipeline for any SEA-AD donor from configuration, with no per-donor code edits:
download → particles → multistain alignment → compression and figures → stacking coordinate.
The audit was started on 2026-11-16, while running the second donor, H20.33.043 (MTG).
Each entry gives where the assumption lived, what breaks for another donor, and the fix or the reason it was deferred.

## How a new donor runs now

```bash
# 1. data (env wsi-pipeline, PYTHONPATH=src/seaad)
python -m qnp.download --region middle-temporal-gyrus --donor <D>
python -m qnp.montage  --region middle-temporal-gyrus --donor <D>
```

2. config: add two lines under `qnp_section_inputs` in `configs/project_config.yaml`.

```yaml
  <D>_MTG: {donor: "<D>", region: "middle-temporal-gyrus"}
```

```bash
# 3. pipeline (env ad-resilience-xiv, PYTHONPATH=src:src/config_setup:src/preprocess:src/scripts)
J=<D>_MTG; C=configs/project_config.yaml
python src/scripts/prepare_qnp_section_particles.py --config $C --job $J --stage inventory
python src/scripts/prepare_qnp_section_particles.py --config $C --job $J --stage particles --no-binned-figures
python src/scripts/align_qnp_multistain.py          --config $C --job $J
S="python src/scripts/scan_atlas_planes_for_qnp.py --config $C --job $J --stage"
$S roles; $S compress
python -m visualization.cli figures --modality qnp --job $J --compressed
$S slabs; $S profile --level h1.5; $S profile --level h1.0   # stop here; full level on request
$S profile --level 500um; $S passengers; $S stack-report
```

Every value is resolved by `config_setup.qnp_jobs.resolve_qnp_job`:
- `qnp_section_defaults` holds the shared settings and the path templates;
- the job block wins over the defaults;
- slide stems, the stains present and `annotation_misfit` are read from the data.

## Findings

| # | Where | Assumption | What breaks for another donor | Status |
|---|---|---|---|---|
| 1 | `src/visualization/cli.py` `QNP_JOB` (imported by `representation_variants.py`) | One job name for all QNP figures | Figures for any other donor need a code edit | **Fixed.** `--job` on `ng-build`, `figures` and `validate`, and on `representation_variants --job`. The constant is only the default, so H20.33.034 outputs are unchanged. |
| 2 | `prepare_qnp_section_particles.py` | Always writes the binned-particle figures `particles_<tag>.png` | Unwanted output (the compressed figures replace them) | **Fixed.** `--no-binned-figures`. Default behaviour unchanged. |
| 3 | `qnp_section_inputs.H20.33.034_MTG.annotation_misfit` | Hand-listed misfit annotations | A new donor needs manual inspection. H20.33.043 has none, H20.33.034 has GFAP and AT8 | **Fixed.** Derived from `inventory.json` (`annotation_on_tissue_min`) unless the block sets it. Provenance is recorded in `_annotation_misfit_source`. |
| 4 | Every representation path in `atlas_scan`, plus `ribbon_points`, `export.*`, `output_dir` | Paths spelled out per donor, including block names (`A07` NeuN vs `A7` others) | Copy-paste per donor; one wrong block name gives a silent wrong file | **Fixed.** Templates `{aligned_dir}/{stem[NEUN]}…` and `{structural}`. Stems come from the `.svs` names. `{plane_id}` is left for `--stage export`. |
| 5 | `scan_atlas_planes_for_qnp.py`, `align_qnp_multistain.py`, `prepare_qnp_section_particles.py`, `visualization/scene.py` | `config["qnp_section_inputs"][job]` read raw | Thin blocks would not work | **Fixed.** All four call `resolve_qnp_job`. The H20.33.034 block resolves to exactly its original values (test). |
| 6 | Frozen `rigid_families.py` / `rigid_summary.py` (`~/tmp/qnp-sts-landmark/frozen_copy/`) | H20.33.034 paths, the stain list (NEUN, LFB, AT), the output dir and the binned 1000/500 µm levels | No stacking driver for any other donor | **Fixed.** New stages in `scan_atlas_planes_for_qnp.py` (`roles`, `compress`, `slabs`, `profile`, `passengers`, `stack-report`), with the computation in `preprocess/qnp_stacking.py`. Roles are computed, not listed. `--families` gives the per-reflection-family analysis (off by default). |
| 7 | `scan_atlas_planes_for_qnp.py --stage export` | Exports only the coarse scan's own best poses (`plane_scores.csv`) | Cannot export a pose found by the stacking driver | **Deferred** (no xIV in this chat). Next step: let `--stage export` read a pose row from `profile_<level>.csv`. |
| 8 | `prepare`/`qnp_particles.tissue_mask` | Slide-edge lines are thin and removed by the opening | H20.33.043 NeuN and ASYN have a ~0.45 mm full-width black scanner vignette at the bottom edge. It joins the tissue component and became false tissue particles | **Fixed.** `remove_border_bands`: contiguous edge rows/cols ≥ 95 % mask, up to 1.5 mm deep, are cleared. A no-op on all six H20.33.034 slides. |
| 9 | `align_qnp_multistain.py` | Every stain aligned on gray/white features | The glial GFAP gray/white map drags the fit to a wrong pose: on H20.33.034 the saved edge was mirrored; on H20.33.043 θ 202°, scale at its 0.85 bound, ribbon Dice 0.13 | **Fixed for new donors.** Per-stain `align_features` (`gmwm` or `tissue`). The defaults use tissue outline for non-structural stains. An explicit stain list (H20.33.034) replaces the defaults, so that donor is unchanged; its known-bad GFAP edge stays until it is rerun with the defaults. |
| 10 | `align_qnp_multistain.py` | Reflection taken from the best score only | A mirrored section would pass silently | **Fixed.** Each edge also refines the best opposite-flip pose and records its score gap, white-matter Dice and ribbon Dice (`reflection_check.confirmed`). |
| 11 | Role QC (prompt step 7) | Roles judged by eye | Not repeatable | **Fixed.** `--stage roles`: stored Otsu separability, a periodic stripe index (spectral peak share at 0.4–1.5 mm periods; the stripes sit at 1.0 mm), layer-1 p(WM) via the reference HALO L1 in the local frame, and a glial list. Thresholds are in config. `stacking.roles.override` records a user decision. It reproduces H20.33.034's roles exactly. |
| 12 | `particle_approximation.optimize_particles` pair guard (4 M pairs) | Sections ≤ ~2,000 particles at 500 µm | H20.33.043 NeuN has 1,985 particles at 500 µm (3.94 M pairs), close to the guard; larger blocks would fail | **Fixed (config).** `stacking.compression.max_pairs` (16 M ≈ 4,000 particles, 128 MB per float64 matrix). **GPU path: not worth it.** The CPU float64 fit takes 4–12 s per section and level, run once and reused. |
| 13 | `figures_qnp.figure_a` | Reference stain only; mass scale tied to a bin size | No per-section comparison figures | **Fixed.** Takes a `stain=` argument (file suffix for non-reference stains) and `mass_refs`. `compressed_representation()` builds the h 1.0 / h 1.5 panels. H20.33.034 baseline output unchanged. |
| 14 | `visualization/ng_scenes.py` `HYPOTHESES` | The two H20.33.034 xIV pose hypotheses (A, B), hard-coded fit dirs | `figure_c`, `ng-build` and `render-complete` on another donor point at H20.33.034 fits | **Deferred.** Needs xIV fits for the new donor; derive from `transform_graph.json` / stacking `report.json` when xIV is run. |
| 15 | `representation_variants.py` (`preflight`, `build`, `render-complete`) | H20.33.034 pilot inputs, `HYPOTHESES` and MERFISH case | Runs only for the reference donor | **Partly fixed** (`--job`). Its stages still assume the H20.33.034 xIV fits (#14). The compressed figures for new donors go through `cli figures --compressed` instead. |
| 16 | `compare_qnp_iv_structural.py`, `run_qnp_od_registration_pilot.py`, `render_qnp_od_registration_qc.py` | Hard-coded `H20.33.034_MTG` job, scan dir and NG state | Pilot scripts for one donor | **Deferred.** One-off pilots, not part of the per-donor pipeline. |
| 17 | `seaad/qnp/montage.py` `ROTATE_180` | Two H20.33.034 HIP slides listed as stored upside down | Other donors' upside-down scans render wrong in the montage (display only) | **Deferred.** The file carries uncommitted work from another task. Detect from the annotation/tissue fit instead of listing. |
| 18 | `atlas_scan.dense_search` (H20.33.034 block) | Planes around that donor's two basins | Not applicable to other donors | **Kept per donor** (not in the defaults). |
| 20 | Stacking search window (frozen scripts, skill step 3) | The unconstrained 70 mm in-plane window is enough for the section to find its region | On H20.33.043 every anchor lands medially on the insula/operculum (ribbon on MTG 1 %) | **Option added, not decided.** `--region-prior` (`stacking.region_prior`: MTG/STG/ITG + 6 mm) is an explicit, reported prior with its own output folder. On H20.33.043 it gives no consistent basin (anchors at y −49 / −57 / −24.5, scores 4–7 δ worse). |
| 21 | Escalation logic (new) | — | (found while building) Flatness judged on narrow escalation ranges marked good anchors flat; windows touching the searched range were truncated | **Fixed.** Flatness is judged on the first level only; profiles extend past edge windows (`range_extensions_mm`). |
| 19 | `structural: true` for NEUN, LFB, AT (defaults) | The multistain consensus uses these three | On H20.33.043 AT8 fails the role QC (separability 0.56), yet it still enters `consensus_*` | **Deferred.** The stacking driver works per anchor and does not use the consensus. The consensus feeds only the coarse `--stage scan` and xIV export. Revisit when exporting for xIV (use `roles.json` anchors). |

## Regression (2026-11-16)

The new driver on H20.33.034 (roles from QC: NeuN, LFB, AT8; compressed h 1.5 → h 1.0):
- best depths: NeuN −27.5, LFB −28.0, AT8 −28.0;
- pose: flip True for all, θ 126–129°, scale 1.04–1.13;
- the h 1.0 window union is −27.0 … −28.5 with δ 0.018, containing the recorded −27.5 … −28. **Pass.**

## Tests

- `src/tests/test_qnp_jobs.py`: a job built from a synthetic donor ID and slide folder; inventory-derived misfit; an explicit block wins and replaces the stain list; minimal blocks are tolerated; the H20.33.034 block resolves to itself.
- `src/tests/test_qnp_stacking.py`:
  - compression conserves mass and lowers the error;
  - Otsu separability;
  - the stripe index detects synthetic 1 mm stripes but not a straight boundary;
  - role assignment, δ, windows and escalation depths;
  - border-band removal;
  - the per-depth GPU search recovers the matching depth of a tiny synthetic section.
