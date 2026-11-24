# Mapping 2-D tissue sections onto a 3-D atlas

This document is the reference for the section-to-atlas mapping in this repository. It has
three parts: the method (§1), the workflow for running it (§2), and where each operation
lives in the code (§3). Results, calibration evidence and regression references are in
[section_mapping_findings.md](section_mapping_findings.md). Historical reports are in
[archive/section_mapping/](archive/section_mapping/).

The method places partial 2-D sections (SEA-AD QNP histology now; spatial-omics sections
later) in a 3-D labelled atlas (AHRAv2 on MNI ICBM 2009b symmetric, 0.5 mm). It does so in
three steps:

1. Estimate each section's **stacking coordinate**: its depth along the atlas AP axis, with a
   noise-scaled window.
2. Apply anatomical priors explicitly, and report them as priors.
3. Refine the in-plane placement with deformable xIV-LDDMM.

The section is the **target** (fixed; it defines the support). The atlas is the **template**
(moving; it is π-mapped and support-censored). These roles are never swapped: the support
censoring exists to protect incomplete data.

---

## 1. Method

### 1.1 Representation

A section or an atlas plane is a feature-valued discrete measure

    μ = Σ_i ν_i δ_{x_i},   x_i ∈ R² (mm),   ν_i ∈ R^C_{≥0} (mm² of each class).

**Structural features.** The structural features have C = 2: gray- and white-matter area.

- *Section side:* a per-stain optical-density feature is split at its Otsu threshold into a
  soft membership p(WM). A particle carries area·(1−p) and area·p.
- *Atlas side:* label tables give gray = every non-zero label that is neither white matter
  nor dropped (ventricles, choroid plexus), and white = a configured code list.
- The correspondence between the two sides is fixed a priori (identity on gray/white), not
  learned.

**Anatomical channels.** One-hot atlas channels, in this order: `STG_gray_matter`,
`MTG_gray_matter`, `ITG_gray_matter`, `other_gray_matter`, `white_matter`. They partition the
non-dropped labels. A fixed 0/1 projection M maps them to other feature spaces (ν ↦ ν Mᵀ):

| projection | maps to |
|---|---|
| `gmwm` | the 2 structural channels; reproduces them exactly (this is checked) |
| `mtg_gm` | MTG gray matter alone |
| `anatomical` | the identity (all five channels) |

**Kernel inner product.** Measures are compared through

    ⟨μ_X, μ_Y⟩_σ = Σ_ij exp(−|x_i − y_j|² / 2σ²) ⟨ν_i, ν_j⟩.

**Compression.** A section is compressed once per level and reused in every search:

- The compressed measure has ~1 particle per h mm. Its positions and masses are optimised
  jointly to minimise the varifold distance to the original (kernel exp(−d²/σ²), σ = max(1, h)).
- The search uses ~200 particles (h = 1.5 mm). Ties escalate to ~450 (h = 1.0) and then to the
  full 500 µm particles (~1,700).
- Below ~200 particles the plane ranking is biased.

**Atlas planes.** The 0.5 mm label volume is resliced nearest-neighbour on the plane
`origin + a·u + b·v` over a 70 mm window. Each block of 2×2 pixels is summed into one 1 mm
particle carrying class areas.

### 1.2 Anatomical localization

**Coordinates.** Atlas world mm, MNI RAS, +y anterior. The affine maps voxels to world, and
voxel-index order is never read.

**Planes.** A plane at pitch/yaw has the basis u = R(+x), v = R(+z), n = u×v, with R = R_z(yaw) R_x(pitch).

- At pitch = yaw = 0 this gives u = +x, v = +z, n = −y.
- Every plane passes through the target region's centroid (x, z) in one hemisphere, at depth y.
- A coronal plane's stacking coordinate is therefore its world y.
- The AP slab coordinate is s = −y (posterior-increasing, the SEA-AD slabbing direction).

**Slab measure (fast pass).** The mean of the class masses over parallel planes y ± 3 mm,
stepped 0.5 mm apart.

**Landing (QC only).** The labels at the nearest voxel under the placed section points. It
reports where annotation ribbons fall. It is a consistency check, not ground truth.

### 1.3 Geometric alignment

**Transform.** Section mm x maps to template (plane) mm p by

    p = A (x − τ),   A = s R(θ) F,   F = diag(1, −1) if flip else I.

- s is template mm per section mm. For atlas planes it absorbs shrinkage, template size and
  atrophy, so it is a nuisance parameter, not a measurement. Bounds are [0.8, 1.5].
- Template particles are pulled into the section frame by A⁻¹p + τ, with masses divided by s²
  (area is conserved).

**Support-censored score.** 0 means identical; 1 means no overlapping tissue.

    D_T(y) = Σ_i |ν_i| exp(−|y − x_i|² / 2σ_f²)                    (section density, σ_f = 0.5 mm)
    w(y)   = min(1, D_T(y) / (c · median_i D_T(x_i))),   c = 0.25   (support weight)
    score  = mean_σ  |w·μ_S − μ_T|²_σ / |μ_T|²_σ,   σ ∈ {2, 1, 0.5} mm

- Template particles with w ≤ 10⁻³ are dropped.
- For particles at spacing h, the SDs below h/2 are dropped.
- The weight censors **absent** tissue (block edges, holes, tears). It does not censor tissue
  that is **present but wrong** (scanner stripes, layer 1 read as white matter). Feature QC
  handles that case (§1.4).

**Search.** The search runs in two steps:

1. FFT cross-correlation of Gaussian-blurred rasters (1 mm grid, 1.5 mm blur) ranks the
   translations for every (θ, s, flip). The grid is 15° rotations × s ∈ {0.9, 1.05, 1.2, 1.35}
   × both flips.
2. The best distinct candidates (separated by > 3 mm or > 20°, or with a different flip) are
   refined on the exact score.

Refinement uses Nelder–Mead on the CPU (reference) or batched projected Adam on the GPU (many
starts and depths per call). The two paths share every constant and agree to ~2×10⁻⁶.

**Pose algebra after a deformable fit.** A least-squares similarity read off the shot atlas
particles, x ↦ cRx + b, is composed with the start pose (A₀, τ₀) into one pose:
A = (cR)⁻¹A₀, τ = τ₀ + A₀⁻¹b.

**Common post-fit score (ranking xIV fits).** The native xIV loss is normalised per run and
cannot rank runs. Fitted poses are therefore re-scored in the section's fixed frame:

    R = Σ_σ w_σ |μ_A − μ_T|²_σ / |μ_T|²_σ,   w_σ ∝ 1/σ,   σ ∈ {8, 4, 2, 1} mm

- The atlas measure is carried into the section frame with areas pushed forward.
- It is weighted by the section's fixed 0.5 mm footprint.
- R_anat, used only with an annotation, scores atlas MTG against the annotated tissue,
  uncensored.

### 1.4 Stacking

1. **Roles (feature QC).** A section is an **anchor** when its gray/white map is clean:
   - Otsu separability ≥ 0.75;
   - periodic stripe energy ≤ 0.3;
   - layer-1 mean p(WM) ≤ 0.5;
   - it is not a glial stain.

   Otherwise it is a **passenger**. A user override is reported as an override.
2. **Fast pass (slabs).** Each anchor gets one global search per 6 mm search slab (3 mm
   stride) over the target region's AP extent ± 5 mm. The top 3 distinct slabs are kept, plus
   the best one inside the region extent.
3. **Depth profile.** For each anchor, a *fresh* global search runs on every coronal plane
   0.5 mm apart around the kept slabs, giving f_a(y). A pose is never warm-started from another
   model.
4. **Noise scale δ.** For each anchor and plane, take the score relative to that anchor's best
   plane, then the population SD across anchors. δ is the median of that SD over the 20 planes
   with the lowest mean relative score.
5. **Window.** The window is all depths within δ of an anchor's best. On the first (widest)
   level, a profile with median − best ≤ 2δ is *flat*, and its anchor is demoted to passenger.
   Later levels inherit this flatness.
6. **Escalation.** Each level searches only inside the previous windows, padded by one step.
   - It adds the previous best pose as an extra start.
   - It extends by two planes wherever a live window touches the scanned range.
   - Stop when the windows stop shrinking. The level whose window was reported is the
     *deciding level*.
7. **Passengers.** A passenger inherits the host anchor's pose and window through the
   section↔section alignment (rigid; tissue outline; both flips). This is checked by a direct
   outline fit to the host.
8. **Order.** Pairwise outline scores give at most a weak order. Per-slice spacing below the
   window width cannot be estimated from the atlas, and the report says so.

**Decision rules.**

- Candidates within δ are tied.
- Distant candidates are decided only at full resolution, never on compressed scores.
- Model changes (tilt, scale, features) rerun the global search.
- A saved pose is never inherited without checking both flips.

### 1.5 Anatomical priors (opt-in, reported as priors)

**Region prior (in-plane).** The admissible atlas support is the set of cells within 6 mm of
MTG, STG or ITG gray matter. The search sees only that tissue, so placements elsewhere have
nothing to match and no overlap threshold is needed. ITG stays in the support: an STG/MTG-only
support moved a supported H20.33.034 placement.

**Landmark-slab prior (AP).** SEA-AD sampled MTG at the first appearance of the LGN (Gabitto
et al. 2024), in 4 mm coronal slabs cut anterior to posterior (Latimer et al. 2023).

- *Landmark.* The landmark s_L is the most anterior coronal plane where the AHRAv2 `LG` label
  (one hemisphere) reaches 1 mm². This is y = −22.0, to voxel precision.
- *Candidate slabs.* With thickness h = 4, an unknown phase δ ∈ [0, h) and offsets j ∈ {−1, 0, +1}:

      S_j(δ) = [s_L − δ + j·h,  s_L − δ + (j+1)·h]

- *Search interval.* The union over all phases and offsets, ± 1 mm: y −13 … −31.
- *Effect on the search.* The first profile level scans every plane of the interval.
  Escalation is clamped to it. Flatness is inherited from the unconstrained run, because a
  narrow range inflates δ. `windows_at_prior_bound` names the anchors whose evidence pushes
  against the bound.

**Admissible vs unconstrained (`candidates`).** Both best placements are scored on one
*data-only* objective: the stacking score against the full, uncropped coronal plane, with δ
from the unconstrained run. The admissible pose is also refined locally on that objective.

| status | rule (gap = admissible − unconstrained data-only score) |
|---|---|
| prior inactive | same placement: depth within one step and ribbon points within 1 mm |
| image-supported | gap ≤ 0 |
| tie | 0 < gap ≤ δ |
| prior-driven | gap > δ |

Job verdict:

- **Supported by image geometry** when every anchor is inactive or supported, the admissible
  windows intersect, and nothing is flagged.
- **Primarily prior-driven** when any anchor is prior-driven.
- **Image ties; the prior selects** when the anchors only tie.
- **Unresolved** otherwise.

The diagnostics carry no thresholds and decide nothing:

- ribbon landing;
- exposed and sulcal pia → atlas outer surface;
- section gray/white → atlas gray/white;
- the tissue share inside the support;
- a local ±10° pitch × yaw check.

### 1.6 Deformable registration (xIV-LDDMM)

The deformable step is handed to the vendored xIV-LDDMM (`vendor/xIV-LDDMM-Particle`; see
[software/xiv_lddmm_upstream.md](software/xiv_lddmm_upstream.md)). It is driven by
`framework_script_AtlasToSectionCrossModalBoundary.py`.

The objective is

    L = γ·H(p, q) + D_var(φ·S censored by the target density, T; π_ST) + KL(π_ST) + R.

**Inputs (`export`).**

- *Source:* the atlas plane at 0.5 mm, uncompressed, in plane coordinates, with `nu_Anat`.
- *Target:* the section placed into the plane by the rigid pose (positions p = A(x − τ),
  masses × s²), optionally with `nu_Annot`.
- Both are already in the rigid-scan frame (`center_mode: shared_target`, start `r000_f0`).

**Where the plane and pose come from.** One writer serves two pose sources:

- *Stacking hand-off (`--section S`).*
  - The plane is the coronal plane at S's stacking coordinate.
  - The pose is S's best pose in the run's `report.json`. A passenger takes its host anchor's
    pose: both sections are in the QNP local frame, related by the rigid alignment edge M, so
    slide → plane is A(M x − τ).
  - The target is S's own full-level (500 µm) particles.
- *Legacy (`--plane`).* A row of the tilted-plane scan table, with the consensus target.

**Gate (stacking hand-off).** Without `--candidate`, a placement must be *eligible for automatic
export*: nothing recorded may contest it. It is **not** eligible when:

- the host anchor has search-boundary flags;
- its window is not contiguous;
- the anchors' windows do not intersect;
- any `candidates.json` of the job has a verdict other than "supported by image geometry";
- the run uses anatomical priors but has no `candidates.json` of its own (the prior's effect is
  then unassessed).

Otherwise the export refuses and lists the concerns. The user must then name the placement:

- `--candidate report`: the run's report pose;
- `--candidate unconstrained`, `admissible` or `admissible-refined`: from the prior run's
  `candidates.json`.

**Eligibility is not established anatomical correspondence.** The atlas is a population
template, and every check here is a consistency check. The export record (`stacking_handoff`)
keeps the two apart:

- `selection`: automatic or explicit, with the concerns that were overridden;
- `anatomical_correspondence`: what the data-only comparison says (for example "supported by
  image geometry", or "not assessed").

**Consistency checks at export time.**

- *Score:* the selected pose must reproduce, on the CPU, the score it was selected on
  (|Δ| ≤ 10⁻⁴). This catches a wrong transform direction, plane or level.
- *Rigid edge:* the slide → local edge must be rigid.
- *Composition:* slide → local → plane must agree with the composed affine.
- *Mass:* the target area equals the section area × s².

**Model.**

- One stage, `transform: affine+diffeo`.
- Kernels: RKHS 8/4/2 mm, varifold 8/4/2/1 mm.
- `c_nonrigid` 1, γ 0.1, `dens_sigma_mm` 0.5, float32.
- In 2-D the support weight is fixed: λ is inert.
- xIV is a single-start local fit without flips. It refines a pose from §1.4; it never
  replaces the search.

**Ranking.** Fits are ranked by `scores.json`, which uses fixed scales and a fixed 0.5 mm
footprint, or by the common score (§1.3). They are never ranked by the native loss.

**Uninformative slices** are not warped on their own evidence. They inherit the anchor's map.

---

## 2. Workflow (SEA-AD QNP)

**Setup.**

- Environment: `ad-resilience-xiv` (torch + pykeops, GPU first).
- Every command runs with `PYTHONPATH=src:src/config_setup:src/preprocess:src/scripts`.
- The settings are in `configs/project_config.yaml`: `qnp_section_defaults` plus one job
  block under `qnp_section_inputs`. A new donor needs only `{donor, region}`.

```bash
C="--config configs/project_config.yaml --job H20.33.043_MTG"
python src/scripts/prepare_qnp_section_particles.py $C --stage inventory   # slides, HALO fit-to-tissue
python src/scripts/prepare_qnp_section_particles.py $C --stage particles   # gray/white + OD particles, maps
python src/scripts/align_qnp_multistain.py $C                              # every slide -> the NeuN local frame
python src/scripts/map_qnp_sections.py $C --stage stack                    # roles ... report (unconstrained)
# optional priors (each in its own sub-folder), then the data-only comparison:
python src/scripts/map_qnp_sections.py $C --stage landmark                 # LGN, candidate slabs, figures
for L in h1.5 h1.0 500um; do
  python src/scripts/map_qnp_sections.py $C --region-prior --landmark-slab --stage profile --level $L; done
python src/scripts/map_qnp_sections.py $C --region-prior --landmark-slab --stage report
python src/scripts/map_qnp_sections.py $C --region-prior --landmark-slab --stage candidates
# stacking -> xIV hand-off (an anchor; a passenger takes its host's pose):
python src/scripts/map_qnp_sections.py $C --stage export --section NEUN          # automatic only if eligible
python src/scripts/map_qnp_sections.py $C --region-prior --landmark-slab \
    --stage export --section NEUN --candidate admissible                        # explicit choice (e.g. H20.33.043)
# add a registrations.<name> record whose source/target/qc paths are the ones printed in the export record, then:
python src/scripts/framework_script_AtlasToSectionCrossModalBoundary.py --config configs/project_config.yaml \
    --registration <name> --stage support-check          # then --stage single
python src/scripts/map_qnp_sections.py $C [same prior flags] --stage evaluate --section NEUN \
    [--candidate admissible] --plane p+000_w+000_y-024.5 --fit-dir <init dir>
# legacy: a plane of the tilted-plane scan table, consensus target
python src/scripts/map_qnp_sections.py $C --stage export --plane p+000_w+000_y-027.5
```

`scan_atlas_planes_for_qnp.py` is a compatibility alias for `map_qnp_sections.py`, with the
same flags. `stack-report` is an alias of `report`.

### Stages of `map_qnp_sections.py`

| stage | does | main outputs (`stacking.output_dir[/<prior folder>]`) |
|---|---|---|
| `roles` | feature QC → anchor / passenger | `roles.json`, `roles_gray_white_maps.png` |
| `compress` | joint compression per level (resumable) | `<aligned>/compressed/*_h1.5*.npz`, `compression.json` |
| `slabs` | fast pass over search slabs | `slabs.json`, `slabs_h1.5.csv` |
| `profile --level L` | depth profile, δ, windows, escalation | `profile_L.json`, `profile_L.csv` |
| `passengers` | inherit via the section↔section edge; direct outline check | `passengers.json` |
| `report` | report and figures | `report.json`, `REPORT.md`, `depth_profiles.png`, `best_poses.png` |
| `stack` | roles → compress → slabs → all levels → passengers → report | all of the above |
| `landmark` | landmark-slab prior and its figures | `landmark_slab.json`, `slab_3d.png`, `slab_coronal_montage.png`, `slab_sagittal.png` |
| `candidates` | admissible vs unconstrained, data only | `candidates.json`, `candidates.png` |
| `scan`, `summarize` | tilted-plane scan of the consensus (CPU; `--dense` re-check) | `atlas_scan.output_dir/plane_scores.csv`, `summary.json` |
| `export --section S [--candidate c]` | stacking → xIV hand-off (gated; §1.6) | `export.stacking_dir/<run>/S_<plane>[_c]/` archives, `<run folder>/export_S_<plane>[_c].json` |
| `export --plane` | legacy: xIV inputs for one scanned plane (consensus) | archives under `registration_inputs/`, `export_<plane>.json` |
| `evaluate` | ribbon landing before/after xIV (`--section` for a stacking export) | `<fit dir>/anatomical_evaluation.json`, `atlas_correspondence.png` |
| `graph` | slides → QNP frame → plane hypotheses → MNI | `transform_graph.json` |
| `atlas-features` | anatomical-channel checks and figures | `atlas_features/` |

The prior folders are `region_prior/`, `landmark_slab/` and `region_prior__landmark_slab/`.
Priors never change the unconstrained run in the base folder.

### QC before trusting a placement

- **Roles:** look at `roles_gray_white_maps.png`. Stripes, a white layer 1 or a glial stain
  mean passenger.
- **Profiles:** check `depth_profiles.png`. The window should shrink with the level (`stop_escalating`).
- **Flags:** check the flags in `best_poses.png` and `REPORT.md`: an AP optimum on the range
  or prior bound, or a scale at its bound.
- **Priors:** read `windows_at_prior_bound` and the `candidates.json` verdict before calling a
  placement anatomical.
- **Hand-off:** a refused export is a result, not an error to work around. Pick `--candidate`
  only as a stated choice, and report it with the fit.
- **xIV:**
  - judge `registration_t1.00.png` and `qc_warp.png` visually first;
  - then check `scores.json`, π_ST near-diagonal, mass-ratio p50 and support-weighted area
    ratio in [0.5, 2], and the loss trajectory;
  - "local minimum reached" just after a large drop can be a float32 plateau.

### Another ROI, donor-native MRI, or another modality

- **Another ROI:** give the job its own `atlas_scan.fine_regions` (the first entry is the
  target) and, optionally, its own `stacking.landmark_slab` block. `applies_to_regions` stops
  the MTG rule from being inherited.
- **Another atlas or a donor MRI segmentation:** change `annotation_nifti` / `terminology_csv`
  and the code lists. Nothing in `section_mapping` assumes AHRAv2.
- **Spatial omics:** the section stays the target and keeps its cell-type channels. Search and
  stacking use the a-priori gray/white collapse; xIV uses a rectangular π_ST.

---

## 3. Implementation map

General methods live in `src/section_mapping/` (no dataset knowledge). SEA-AD specifics live in
`src/scripts/` and `src/preprocess/qnp_particles.py`.

| operation | where |
|---|---|
| kernel inner products, compression, rasterisation | `section_mapping/measures.py`: `gauss_inners`, `grid_initialization`, `compress`, `splat` |
| label tables, anatomical channels, projections | `section_mapping/atlas.py`: `lookup_tables`, `anatomical_tables`, `projection_matrix`, `project`, `check_gmwm_reproduces` |
| atlas volume + terminology check | `atlas.load_atlas` (config glue: `qnp_atlas_planes.atlas_setup`) |
| plane geometry, reslicing, plane / slab particles | `atlas.plane_basis`, `plane_geometry`, `reslice_labels`, `label_class_masses`, `plane_particles`, `plane_measure`, `slab_measure`, `atlas_plane` |
| region prior support | `atlas.region_support_mask` |
| landing of section points on labels | `atlas.classify_points`, `landing_fractions` (counts), `landing` (masses) |
| landmark and slab prior | `atlas.locate_landmark`, `build_slab_prior`, `landmark_slab_prior`, `clamp_ys`, `slab_consistency` |
| transform and pose I/O | `section_mapping/alignment.py`: `Similarity2D` (`as_row` / `from_row`) |
| support-censored score | `alignment.FootprintScorer`; GPU `alignment_gpu.TorchFootprintScorer` |
| FFT candidates, refinement, search | `alignment.fft_candidates`, `refine`, `search`, `distinct_poses`; GPU `fft_candidates_gpu`, `refine_batch`, `search_gpu`, `search_planes_gpu` |
| pose composition, common post-fit score | `alignment.similarity_lsq`, `compose`, `CommonScore` |
| roles QC | `section_mapping/stacking.py`: `otsu_separability`, `stripe_index`, `assign_role` |
| depth search, δ, windows, escalation | `stacking.search_depths`, `noise_delta`, `window`, `summarize_level`, `escalation_ys`, `range_extension`, `windows_at_bound` |
| decision rules, placement diagnostics | `stacking.search_boundary_flags`, `same_placement`, `placement_status`, `placement_verdict`, `boundary_distance_map`, `gray_white_boundary_points`, `depth_inside_hull` |
| hand-off gate and candidate poses | `stacking.placement_concerns`, `candidate_pose` |
| QNP slides → particles, HALO annotation masses | `preprocess/qnp_particles.py` (`tissue_class_particles`, `annotated_tissue_masses`, `section_annotation_masses`) |
| job resolution | `config_setup/qnp_jobs.py` |
| section↔section alignment | `scripts/align_qnp_multistain.py` |
| workflow (stacking stages, priors, candidates) | `scripts/map_qnp_sections.py` |
| stacking → xIV hand-off (selection, checks) | `map_qnp_sections.select_placement`, `run_export_stacking` |
| xIV input writer (both pose sources), plane scan, evaluate, graph, atlas-feature QC | `scripts/qnp_atlas_planes.py` (`write_xiv_inputs`) |
| xIV registration driver | `scripts/framework_script_AtlasToSectionCrossModalBoundary.py`; trajectories `scripts/view_xiv_trajectory.py` |
| figures | `visualization/figures_mapping.py` (slab prior, roles, profiles, pose overlays, candidates) |
| tests | `src/tests/test_atlas.py`, `test_alignment.py`, `test_stacking.py`, `test_qnp_particles.py`, `test_qnp_jobs.py`; regression references against the saved runs: `test_mapping_regression.py` |

**Experiment, not part of the workflow.** `scripts/estimate_qnp_stacking_xiv.py` estimates the
stacking coordinate by xIV similarity fits, with and without HALO MTG supervision. Its
conclusion is in the findings doc.
