# Positive-only HALO anatomical supervision for xIV (QNP → AHRAv2)

The atlas→QNP xIV fit can carry a second image-varifold term. That term matches the
atlas MTG gray matter to the HALO-annotated MTG cortex of the section. Both terms act on
**one** deformation, so the geometry is shared and their gradients add.
**Nothing changes by default:** the anatomical weight is 0, and then the model, the
objective and the results are the original ones, bitwise (validated below).

Code:

- **Fork, reusable**, branch `feat/multi-term-varifold` in `vendor/xIV-LDDMM-Particle`,
  uncommitted (see `docs/software/xiv_lddmm_upstream.md`):
  - `xmodmap/distance/LossVarifoldNormCensored.py`: `LossVarifoldNormCensored` and
    `SmoothedIndicatorField`.
  - `xmodmap/model/CrossModalityBoundaryMultiTerm.py`: `CrossModalityBoundaryMultiTerm`
    and `VarifoldTerm`.
- **Project, SEA-AD specific:**
  - [`src/preprocess/qnp_annotation_features.py`](../src/preprocess/qnp_annotation_features.py)
    builds the annotation masses.
  - [`scan_atlas_planes_for_qnp.py`](../src/scripts/scan_atlas_planes_for_qnp.py)
    `--stage export` writes them as `nu_Annot`.
  - [`framework_script_AtlasToSectionCrossModalBoundary.py`](../src/scripts/framework_script_AtlasToSectionCrossModalBoundary.py)
    wires the term in.

Tests:

- [`src/tests/test_anatomical_supervision.py`](../src/tests/test_anatomical_supervision.py) (14).
- `vendor/xIV-LDDMM-Particle/unittests/test_multi_term_varifold.py` (19).

## Objective

    E = γ H(p, q) + λ_struct D²_struct(φ·S; π_ST) + KL(π_ST) + R(λ) + λ_anat D²_anat(φ·S)

    D²_anat = ½ | A · (φ·μ_S^MTG) − μ_T^annot |²_W

- **φ is shared.** One Hamiltonian, one set of momenta and one shooting give the deformed
  particles `(qx1, qw1)`. Both data terms are evaluated on them.
  `CrossModalityBoundaryMultiTerm.loss` returns `(H, data, pi, lambda, anatomical)`, and
  `Model.closure` sums them before one backward pass.
- **Source μ_S^MTG.** This is the fixed `mtg_gm` projection of `nu_Anat`
  (`docs/qnp_atlas_anatomical_features.md`). Nothing is modified or relearned. It is
  carried as a *fraction of each atlas particle's weight*, `ζ_A = nu_MTG / w_S`, so the
  term rides the same transported weights as the structural term:
  - rigid motions leave the weights unchanged;
  - the similarity scale multiplies them by the area change, so the density is unchanged.
- **Target μ_T^annot.** This is the annotated tissue area per QNP particle
  (`nu_Annot`). There is no π for this term: the map MTG gray ↔ annotated MTG cortex is
  fixed 1-to-1.
- **A, the annotation mask.** A fixed Gaussian-smoothed indicator of the annotated tissue:

      A(x) = min(1, Σ_j exp(−|x − y_j|²/2σ_A²) a_j / (2πσ_A²))

  It is 1 inside the annotation, ½ on a straight edge and 0 outside. It is built from the
  annotated masses alone. Atlas MTG where A = 0 is *unobserved*: it is neither matched nor
  penalised.
- **Kernels.** By default `sigma_var_mm` is the structural term's and σ_A =
  `dens_sigma_mm`, the structural support's width. No new numbers are introduced; both
  can be set independently.
- **Normalisation, `target` (default).** Each scale is divided by the annotated
  target's own norm, `β_σ = (0.6/σ)·2/|μ_T^annot|²_σ`.
  - D²_anat = Σ 0.6/σ at a **complete miss** (no atlas MTG inside A), whatever the start.
    That is the same value as the structural term at its own start.
  - D²_anat = 0 when the annotated tissue is fully explained.
  - So λ_anat = 1 means "a complete anatomical miss costs as much as the initial structural
    mismatch".
- **Normalisation, `initial`.** The structural term's per-start rule. It sets every
  start's initial D² to Σ 0.6/σ, so it cannot tell a correct start from a wrong one
  (fork test `test_initial_normalisation_cannot_rank_starts`). It is kept for comparison
  only.

### The 2-D tissue-support censoring (verified, unchanged)

The structural term's 2-D censoring was read in the code at the pin `67a7538`:

- `LossVarifoldNormBoundary_Partial.supportWeight` reduces to `densityWeight` in 2-D,
  because `definedSlicedSupport` returns zero normals when d < 3.
- `densityWeight` is the Gaussian density of the **target** particles (width
  `dens_sigma_mm`), evaluated at the *current deformed* source positions. It is then
  min-max normalised over the current source particles.
- `lamb` has no effect in 2-D and is held fixed.

So the censoring is **not learned**: it has no parameters, and it is a fixed function of
the target's tissue. Two refinements to "a fixed tissue mask":

- it is soft rather than binary;
- its min-max step depends on the current source configuration. If the source loses all
  overlap with the target, it becomes 0/0 and the data loss is NaN. This is upstream
  issue #6 in `docs/software/xiv_lddmm_upstream.md`; QNP sources always overlap.

None of this is changed. The anatomical term does **not** use the tissue support: A is
its only censor. The tissue mask and the HALO annotation mask stay separate objects in
separate terms.

## The annotation feature (QNP pipeline)

Source of the mask:

- `align_qnp_multistain.py` already writes `multistain_maps_32um_local.npz`. It holds
  `consensus_mask` (consensus tissue) and `reference_ribbon`: the union of the NeuN HALO
  layers L1…L5-6 on the same 32 µm grid. NeuN is the reference stain, so the ribbon
  needs no transform.
- **Annotated tissue = `consensus_mask & reference_ribbon`.** Ribbon pixels off the
  consensus tissue are dropped and reported (0.9% on H20.33.034).
- All layers are combined into one feature, `HALO_MTG_cortex_all_layers`. Per-layer
  features would need the layer index resampled onto the consensus grid; the saved maps
  carry only the union.

How the masses are built and checked:

- `annotated_tissue_masses` aggregates the ribbon with the *same* block aggregation as
  the consensus particles (`tissue_class_particles`, with the ribbon as a 0/1 value).
- It **refuses** a target whose positions or per-particle tissue masses it does not
  reproduce. On H20.33.034 the positions are equal at both 1 mm (459 particles) and
  0.5 mm (1614).

Export:

- `--stage export` writes `nu_Annot` (annotated mm² × scale², i.e. atlas-plane units like
  `nu_Sub`) and `annotation_feature_names` into the target archive. The record goes into
  metadata as `annotation_supervision`.
- This happens only when `atlas_scan.export.annotation_maps` is set. It is now a
  templated default in `qnp_section_defaults`.
- `X`, `nu_Sub`, `nu_All` and every original metadata key are **unchanged**. A 0.5 mm
  re-export of plane A matched the archives on disk array by array.
- A section whose reference stain has no ribbon gets `annotation_supervision.status =
  "no_annotation"` and no `nu_Annot`.

H20.33.034, plane A:

- annotated tissue 203.4 mm² of 392.4 mm² (51.8%), on 276 of 459 particles at 1 mm;
- 242.5 mm² in plane units;
- the ribbon (205.2 mm²) lies 99.1% on consensus tissue.

## Configuration

On a registration record (all optional):

```yaml
anatomical_supervision:
  weight: 0.0               # lambda_anat (0 = original objective; no term is built)
  structural_weight: 1.0    # lambda_struct (the old hard-coded data-loss weight)
  normalization: "target"   # or "initial"
  # sigma_var_mm: [...]     # default: model.sigma_var_mm
  # annotation_sigma_mm: x  # default: model.dens_sigma_mm
  # source_projection: "mtg_gm", target_key: "nu_Annot"
```

How the driver resolves it:

- **`weight > 0`, annotated archives:** builds `CrossModalityBoundaryMultiTerm`.
- **`weight == 0`:** builds the original `CrossModalityBoundary`.
- **`weight > 0`, section recorded as unannotated:** warns and runs the original
  objective.
- **`weight > 0`, stale archives** (no `nu_Annot` / `nu_Anat`): an error.
- The source's structural features must contain the MTG channel (gray/white, `gmwm` or
  `anatomical`): ζ_A is a fraction of the structural weight.

Outputs, in addition to the original ones:

- `scores.json`:
  - `native_losses.anatomical`: weighted;
  - `loss_components`: unweighted structural and anatomical D² at the start and end of
    each stage, plus the weights.
- `support_check.json`: an `anatomical` block, only when the term is active.
- `input_manifest.json`: `anatomical_supervision`.
- `anatomical_diagnostics.json`, written **after every other output**, so it cannot
  change them, whenever the archives carry the masses (also at weight 0):
  - target-normalised D²_anat at start and end, and its complete-miss value;
  - where annotated and unannotated tissue lands on the atlas channels after pull-back.
  - When the term is active, the annotated landing is fitted, not independent validation.

## Validation (H20.33.034 plane A, 1 mm, 2026-11-17)

The plane was re-exported at the coarse 1 mm level in two versions:

- **E1:** 702 atlas particles, the standard footprint restriction.
- **E2:** 2512 particles, the full 70 mm in-plane window, which adds STG, insula
  (long insular gyri, 17 mm²) and FuG.

The target is the 1000 µm consensus (459 particles). Transform: `affine`, i.e. a
similarity in 2-D (`rigid` where stated); 150 steps; float32; GPU; no diffeo.

- Scratch: `~/tmp/anatomical-varifold/`:
  - inputs: `export_1mm.py`, `export_report.json`;
  - config and fits: `scratch_config*.yaml`, `fits/`;
  - comparisons: `compare_weight0.json`, `summary_E1.json`, `degeneracy_E1.json`,
    `wrong_starts/*.json`.
- Each fit ran in a fresh process (support-check, then single).

### λ_anat = 0 reproduces the original exactly

| run | code | result |
|---|---|---|
| reference | driver **before** this change, fork at pristine `67a7538` | tissue 0.107764, gray/white 0.191899, native 4.933052 |
| new, no `anatomical_supervision` block | this change | **bitwise identical**: 28/28 arrays (optimized variables, deformation summaries, trajectory, pulled-back QC points), scores, `support_check.json` |
| new, `weight: 0` | this change | **bitwise identical** to both |

The fork's fresh-process test also passes: a toy fit with a weight-0 extra term and one
with no extra terms are bitwise equal to the `CrossModalityBoundary` fit (logs and end
positions).

### Gradients

- **Finite differences.** The float64 directional derivatives of the *summed* objective
  match autograd to 1e-5 relative. This holds for px, pw and π_ST, under the rigid and
  the similarity transforms (fork test).
- **Additivity.** The gradient of the total equals the base gradient plus the anatomical
  term's own gradient (1e-9 relative).
- **Rigid flows.** `pw` gets no gradient, because the weights do not move.

### Synthetic placements

Fork toy: three identical horizontal gray bands in white matter. Gray/white cannot tell
them apart.

- **Correct vs neighbouring gyrus.** The correct band gives D²_anat < ¼ of the value when
  the annotated band sits on STG or ITG. The structural term differs by < 5%.
- **Partial annotation.** With only the left half of the MTG band annotated, the censored
  term at the correct placement is < 0.2× the uncensored term. It is within 2× the
  value for the full annotation.
- **Unannotated tissue.** Changing the features *and* mass of unannotated target tissue
  changes the structural term and leaves the anatomical term bitwise equal. This was
  checked in the fork test and in the project test with real driver code.

Project toy (driver code, 1 mm grid):

- correct placement < 0.2 × complete miss;
- STG and ITG placements > 0.95 × complete miss.

### Real data, correct (reference) start, E1

| fit | annotated → atlas MTG | D²_anat end (miss 38.6) | structural D² end | tissue / gray-white score | similarity scale |
|---|---|---|---|---|---|
| similarity, λ_anat 0 | 77.1% | 4.82 | 4.89 | 0.108 / 0.192 | 1.040 |
| similarity, λ_anat 1 | 81.4% | 3.22 | 5.67 | 0.135 / 0.230 | **1.160** |
| similarity, λ_anat 5 | 82.8% | 3.04 | 6.03 | 0.134 / 0.233 | 1.153 |
| rigid, λ_anat 0 | 74.4% | 5.12 | 5.16 | 0.109 / 0.187 | 1 |
| rigid, λ_anat 1 | 77.1% | 4.69 | 5.37 | 0.107 / 0.190 | 1 |
| rigid, λ_anat 5 | 80.3% | 4.21 | 6.50 | 0.114 / 0.212 | 1 |

### Real data, anatomically wrong starts, E2

The section was translated so that its annotated tissue's centroid sits on another
region's centroid. Each row gives D²_anat at the start, the cosine between the anatomical
descent direction and the way back, and the share of annotated tissue on atlas MTG after
the fit.

| start (offset) | D²_anat at start / miss | cos | λ 0 | λ 1 | λ 5 |
|---|---|---|---|---|---|
| reference | 8.8 / 62.2 | n/a | 0.76 | 0.83 | 0.83 |
| STG (13.8 mm) | 35.8 / 61.9 | 0.99 | 0.39 | **0.83** | **0.83** |
| ITG (16.8 mm) | 76.2 / 79.5 | 0.93 | 0.24 | **0.83** | **0.83** |
| FuG (21.4 mm) | **= miss** (plateau) | no gradient | 0.24 | **0.83** | **0.83** |
| insula (38.5 mm) | **= miss** (plateau) | no gradient | 0.00 | 0.00 | 0.00 |

What this shows:

1. **The gradient exists only with overlap.** The term gives a useful gradient wherever
   atlas MTG partly overlaps the annotation: it points back to MTG (cos 0.93–0.99).
2. **Disjoint placements sit on a plateau.** There D²_anat equals its complete-miss
   value and the gradient is exactly 0. This follows from positive-only censoring:
   atlas MTG outside the annotation is unobserved, so it neither costs nor attracts.
   - The reach is *partial overlap + σ_A*, not the varifold kernel width.
   - From FuG the fit still recovered: the structural term's own motion brought atlas MTG
     into overlap, and the anatomical term took over.
   - From the insula, all three fits are identical and stay there.
3. **D²_anat ranks starts well.** After the λ = 0 fits, the wrong basins have tissue
   scores of 0.113–0.125 against 0.107 for the correct one: the structural score barely
   separates them. Their D²_anat values are 34.7–61.9 against 7.8.

## Uncensored option (`censoring: none`, opt-in, 2026-11-17)

`anatomical_supervision.censoring`:

- **`annotation` (default, unchanged):** the atlas MTG measure is multiplied by A.
- **`none`:** the ordinary varifold distance

      D²_anat = ½ |φ·μ_S^MTG − μ_T^annot|²_W

  This is `LossVarifoldNormCensored` without censors, i.e. xmodmap's `LossVarifoldNorm.__call__`.

What stays the same in `none` mode:

- **Target measure.** The HALO annotation still defines it: unannotated tissue carries no
  anatomical mass.
- **The model.** The kernels, the target normalisation, the fixed MTG→MTG map and the
  shared flow are all unchanged.
- **The structural term**, with its tissue support and its learned π_ST.

What changes: atlas MTG away from the annotation now costs. Unobserved atlas MTG is
penalised.

### Reporting

The target-normalised D² carries a factor of the joint unit-box scale s. That scale
changes with the start and with the atlas plane: the complete-miss value is 38.6 on E1
and 61.9 or 62.2 on E2. So `anatomical_diagnostics.json` reports both censoring modes
(`D2_anatomical_by_censoring`), each with `relative = D² / complete_miss`.

- `relative` is the 1/σ-weighted mean over kernels of |mismatch|² / |μ_T^annot|². It
  does not depend on s, so it is comparable across starts and planes.
- Censored: 1 = no atlas MTG inside the mask.
- Uncensored: a disjoint atlas MTG gives 1 + |μ_S^MTG|²/|μ_T^annot|² (> 1).
- `scores.json` `loss_components` adds each term's `complete_miss` and `censoring`.
- The existing `D2_anatomical_target_normalised` key keeps its censored meaning.

### QC figure

- The middle panel of row 3 draws the atlas MTG mass the term actually counts. For `none`
  that is all of it, and the title gives the part outside the annotation.
- The annotation contour is still drawn, as the target measure, labelled "not applied to
  atlas".

### Validation (H20.33.034 plane A, full window E2, 1 mm, similarity, λ_anat = 1)

Each fit ran in a fresh process. Scratch: `~/tmp/anatomical-varifold/censoring_compare.py`;
outputs in `fits/censoring/<start>_<mode>/`. Losses are unweighted. Gradients are for a
rigid translation of the atlas at the start (per mm); cos is the cosine between the
descent direction and the way back to the reference placement.

| start / censoring | D²_struct start → end | D²_anat start → end (relative) | anat. gradient at start | annotated → MTG | scale | rotation | translation (mm) | tissue / g-w score |
|---|---|---|---|---|---|---|---|---|
| MTG / annotation | 62.19 → 5.38 | 8.84 → 5.06 (0.14 → 0.08) | 2.98 | 82.8% | **1.159** | 0.18° | (2.85, 0.98) | 0.135 / 0.231 |
| MTG / none | 62.19 → 8.84 | 34.04 → 17.60 (0.55 → 0.28) | 8.12 | 72.2% | **0.889** | 6.87° | (−4.20, −1.08) | 0.105 / 0.224 |
| insula / annotation | 61.88 → 16.87 | 61.88 → 61.88 (1.00 → 1.00) | **0.000** | **0.0%** | 1.009 | 0.01° | (1.54, 2.14) | 0.115 / 0.331 |
| insula / none | 61.88 → 9.05 | 153.1 → 17.79 (2.48 → 0.29) | 0.045, **cos 0.9999** | **74.6%** | **0.897** | 0.20° | (27.94, 21.17) | 0.106 / 0.229 |

For reference, λ_anat = 0 on E2 lands 76.3% (MTG start; tissue 0.107) and 0.0% (insula
start).

### Findings

1. **The plateau is gone.** From the insula start the censored term is exactly flat
   (gradient 0, relative D² = 1). The uncensored term has a small but perfectly aimed
   gradient (cos 0.9999 towards MTG), smaller than the structural gradient there (0.65
   per mm).
   - The fit travels the full 38.5 mm: translation (27.9, 21.2) mm against an offset of
     (30.6, 23.3) mm.
   - It ends at 74.6% MTG, in the same basin as the uncensored fit from the correct start
     (72.2%, scale 0.889).
   - So with the existing multiscale kernels (8/4/2/1 mm), an uncensored MTG-to-MTG term
     can guide recovery from a disjoint start 38 mm away.
2. **It penalises unobserved atlas MTG and moves the preferred scale *down*.**
   - Every atlas MTG particle now needs annotated mass nearby. The atlas carries more MTG
     (317 mm² at scale 1) than is annotated (242.5 mm²). Under the similarity transform,
     mass follows area, so the fit shrinks the atlas until the masses roughly match:
     scale 0.889 / 0.897, atlas MTG 250 / 255 mm².
   - The censored term inflates instead: 1.159, with atlas MTG 426 mm², 232 mm² of it
     outside the annotation.
   - The uncensored fit still leaves 80–82 mm² of atlas MTG outside the annotation and
     pays for it. From the correct start this lowers MTG landing below λ = 0 (72.2% vs
     76.3%) and rotates the atlas by 6.9°.
   - In short, uncensored matching treats "unannotated" as "not MTG". That is exactly the
     positive-only assumption the censored term avoids. It is acceptable only where the
     annotation covers essentially all of the MTG cortex in the section.
3. **The two biases are mirror images.** Censored favours over-coverage (scale ↑);
   uncensored favours mass equality (scale ↓). Neither is neutral about scale under a
   similarity transform. Rigid fits avoid both biases; whether uncensored + rigid
   recovers equally well was not run.

Checks:

- float64 finite differences of the combined objective with the uncensored term pass,
  for rigid and similarity (fork test).
- A fork toy test and a project test confirm a non-zero, correctly signed gradient where
  the censored term is flat.
- λ_anat = 0 refits with the final code (fresh, `--overwrite`) are still bitwise equal
  to the pre-change reference (28/28 arrays and the scores), and so is the support check.
- The default censored λ = 1 refit is bitwise equal to its earlier run.

### Rigid check (uncensored, λ_anat = 1, E2, 1 mm; 2026-11-17)

The same runs with `model.transform: rigid`; nothing else changed. Scratch:
`fits/censoring/<start>_none_rigid/`, `anatomical_decomposition.json`,
`rigid_insula_profile.json`.

| start / transform | D²_struct start → end | D²_anat start → end (relative) | annotated → MTG | rotation | translation (mm) | scale | tissue / g-w |
|---|---|---|---|---|---|---|---|
| MTG / rigid | 62.19 → 6.44 | 34.04 → 27.38 (0.55 → 0.44) | **80.0%** | 0.01° | (−0.53, −1.08) | 1 | 0.109 / 0.214 |
| insula / rigid | 61.88 → 17.25 | 153.1 → 126.3 (2.48 → 2.04) | **14.1%** | 0.06° | (11.63, 21.80) of the needed (30.60, 23.33) | 1 | 0.083 / 0.314 |
| MTG / similarity (earlier) | 62.19 → 8.84 | 34.04 → 17.60 (0.55 → 0.28) | 72.2% | 6.87° | (−4.20, −1.08) | 0.889 | 0.105 / 0.224 |
| insula / similarity (earlier) | 61.88 → 9.05 | 153.1 → 17.79 (2.48 → 0.29) | 74.6% | 0.20° | (27.94, 21.17) | 0.897 | 0.106 / 0.229 |

**Decomposition**, D²_anat = ½ target-self + ½ atlas-self − cross, using the model's own
β-weighted kernel at the first and last flow frames:

| fit | ½ target-self | ½ atlas-self start → end | cross start → end | atlas MTG mass |
|---|---|---|---|---|
| MTG / rigid | 62.19 | 91.86 → 91.86 (≤ 4e-7 per kernel) | 120.0 → 126.7 | 317 → 317 |
| insula / rigid | 61.88 | 91.40 → 91.40 (≤ 2e-7) | 0.15 → 26.9 | 317 → 317 |
| MTG / similarity | 62.19 | 91.86 → **69.85** (−24%) | 120.0 → **114.4** | 317 → 251 |
| insula / similarity | 61.88 | 91.40 → 71.02 (−22%) | 0.15 → 115.1 | 317 → 255 |

### Findings

1. **Under rigid motion the uncensored term is pure overlap.**
   - Atlas-self and target-self are invariant, as expected (float32 noise ≤ 4e-7), so only
     the cross term moves: D²_anat = const − ⟨φ·μ_S^MTG, μ_T^annot⟩.
   - Under similarity, the MTG-start fit *lost* overlap (cross 120 → 114). It lowered D² by
     shrinking atlas-self by 24%. That confirms the scale/mass bias is the similarity
     fit's mechanism.
   - Rigid from the correct start keeps more annotated tissue on MTG (80.0%) than
     similarity (72.2%), λ = 0 (76.3%) or the censored rigid fit on E1 (77.1%). It does
     so with a better gray/white score than the uncensored similarity fit.
2. **The rigid fits do *not* converge to the same basin.**
   - From the insula start the rigid fit stopped after 11.6 of 30.6 mm in x: 14.1% MTG,
     relative D² still 2.04.
   - This is an optimizer stop, not a barrier:
     - Along the straight translation from that end pose to the reference pose (at the
       fit's own final π), the total objective falls monotonically, 143.6 → 48.7.
     - At the end pose the slope is downhill: −4.2/mm anatomical, −0.9/mm structural.
   - The cause is in the log. From iteration 4 on, every L-BFGS step overshoots to poses
     with no atlas/section overlap. There the structural term's min-max tissue support
     is 0/0, and the data loss is NaN (upstream issue #6).
   - Five backtracks in a row stay NaN. The step is effectively rejected, the loss
     repeats, and xmodmap's stopping rule reports "local minimum reached" (issues #5/#7).
     There were 17 NaN closures, against 1 in the similarity insula fit and 0 in both
     MTG-start fits.
3. **Answer to the question:**
   - The rigid uncensored objective has no mass bias and offers a continuous downhill
     path from the insula start to MTG.
   - But with the current optimizer and the structural support's NaN at zero overlap,
     the rigid fit does **not** reliably recover pose independently of the start.
   - The similarity fit recovered only together with its mass bias.
   - Making this reliable needs the NaN path fixed (a finite support weight when nothing
     overlaps, or a NaN-safe line search). That changes optimizer and support behaviour,
     so it is not done here.

QC figures: `registration_qc.png` in each fit directory.

### Zero-overlap fix and the rigid insula start (2026-11-17)

**Cause.** The trace runs through three pieces:

1. **The support.** The structural term's 2-D tissue support, `densityWeight`,
   min-max normalises the target density over the current atlas particles. The density
   kernel (0.5 mm) underflows to exactly 0 in float32 beyond ~7 mm. So a trial pose where
   no atlas particle comes within reach of the section has max = min = 0, and the support
   is 0/0 = NaN.
2. **The closure.** It turns the NaN into the float `inf`, with `inf` gradients.
3. **The line search.** Strong-Wolfe cubic interpolation on those values collapses the
   bracket to t = 0. Five such trials per step meant zero movement, an unchanged
   start-of-step loss, and the stopping rule's "Local minimum reached".

**Fix,** in the fork, branch `feat/multi-term-varifold`, uncommitted:

- **Support.** `densityWeight` returns all-0 weights when the density range is 0 and the
  maximum is 0. The censored atlas measure is then 0, and the structural term equals its
  target-self part. It returns all-1 weights when all densities are equal and positive.
  Otherwise it runs the original expression, with the same operations and the same
  autograd graph. There is no epsilon.
- **Optimizer.** The arithmetic is unchanged; the line search's evaluation limit (15 per
  step) did not prevent convergence once the trial values were finite.
- **Stop reporting** is now honest:
  - `sLBFGS.step_report` records accepted steps and non-finite trials;
  - `Model.stop_reason` separates `line_search_no_decrease` / `line_search_nonfinite` from
    `loss_change_below_tolerance` (convergence after an accepted step);
  - both go into `scores.json` (`optimizer_stops`) and the QC summary;
  - a non-finite start loss now stops cleanly instead of crashing.

**Unchanged where well defined.** Fresh refits are bitwise identical (28/28 arrays and
the scores):
- λ = 0 base and weight-0 records against the pre-change reference;
- the censored λ = 1 record;
- the rigid uncensored MTG-start fit, which never loses overlap.

**What the stop reasons revealed.** Every earlier fit (E1 λ 0 included) also stopped on
a no-progress step: 0 non-finite trials, max |grad| 28–243. That is the float32 stop of
upstream issue 5, previously labelled "local minimum". Only the label changed; changing
the behaviour would change every fit.

**Rigid uncensored, λ_anat = 1, E2, insula start, rerun:**

| start | D²_struct start → end | D²_anat start → end (relative) | total | rotation | translation rel. to reference (mm) | annotated → MTG | stop |
|---|---|---|---|---|---|---|---|
| MTG | 62.19 → 6.44 | 34.04 → 27.38 (0.55 → 0.440) | 33.88 | 0.007° | (−0.533, −1.077) | 80.0% | converged (iteration 4) |
| insula, before the fix | 61.88 → 17.25 | 153.1 → 126.3 (2.48 → 2.04) | 143.67 | 0.06° | (−18.97, −1.53) | 14.1% | 17 non-finite trials, then a no-progress stop |
| **insula, after the fix** | 61.88 → 6.36 | 153.1 → 27.23 (2.48 → **0.440**) | 33.68 | −0.011° | (−0.532, −1.085) | **80.0%** | converged (iteration 7), 0 non-finite |

- Zero-overlap trials are now finite: structural 77.59 = target-self, anatomical 153.27 =
  disjoint. The line search interpolates back from them.
- **Both starts reach the same pose:** 0.008 mm and 0.018° apart, with identical landing
  fractions and scores. The totals differ only through the start-dependent unit-box
  scale; the relative anatomical values are equal.
- So rigid uncensored MTG-to-MTG matching recovered the MTG basin from the insula start
  38 mm away, without the similarity mass bias.
- This is one plane, one donor and one wrong start.

### Structural feature map: learned vs fixed identity (2026-11-17)

`record.structural_feature_map`:

- `learned` (default, unchanged): the cross-modal π_ST is optimised with its KL
  regulariser. MERFISH records, whose feature spaces really differ, keep it.
- `identity`: π_ST is fixed to I_F, because QNP and atlas gray/white are the same
  classes. It is checked: the source and target feature names must be identical.

**xmodmap's same-modality model** (`SingleModality`) is the plain varifold term called
with π = `eye(F)`. It has no KL regulariser, no tissue support, no extra terms and no
normalisation in its examples. So it is reused as a *formulation*, not as a class, inside
`CrossModalityBoundaryMultiTerm`, keeping the tissue support, the anatomical term and the
shared flow.

Differences found, and how each was handled:

1. **The KL regulariser must be skipped, not zero-weighted.** For an identity, KL(π) is
   0·log 0 = NaN, and weight 0 × NaN is still NaN.
   - The fork's multi-term model now leaves a weight-0 π regulariser unevaluated (exact
     0), the same convention as weight-0 extra terms.
   - Identity mode always uses that model, even with no anatomical term.
   - π is the variable I_F itself, with preconditioner 1, so the loss sees exactly I_F,
     not (I/10⁴)·10⁴ in float32.
   - π is not in the optimised variables.
2. **The normalisation constant.** The structural term is scaled by its start mismatch
   under the *initial* π.
   - Learned mode starts from makePQ's uniform π, which explains the target poorly
     (target-normalised mismatch 0.79 at the reference start).
   - Normalising identity mode with I₂ instead (mismatch 0.08) raised the structural
     weight by 45 / 27 / 14 / 7× (8 / 4 / 2 / 1 mm kernels) at the reference start, and
     1.0–1.5× at the insula start. That silently changes λ_struct : λ_anat.
   - With that normalisation the identity fits ended at 75.4% (MTG start) and **0%**
     (insula start). From the insula the structural term found its own gray/white local
     minimum (structural descent cos −0.24 towards MTG), which the anatomical term could
     not overcome.
   - **Kept:** identity mode uses the same normalisation constant as learned mode (makePQ's
     initial π), so switching the map changes only π.

**Validation** (full window E2, 1 mm, rigid, uncensored anatomical term, λ_anat = 1, fresh
processes):

| map | start | D²_struct start → end | D²_anat start → end (relative) | annotated → MTG | rotation | translation rel. to reference (mm) | tissue score | g/w mismatch with I₂ at end |
|---|---|---|---|---|---|---|---|---|
| learned π | MTG | 62.19 → 6.43 | 34.04 → 27.38 (0.440) | 80.0% | 0.007° | (−0.533, −1.077) | 0.1088 | 0.110 |
| learned π | insula | 61.88 → 6.36 | 153.1 → 27.23 (0.440) | 80.0% | −0.011° | (−0.532, −1.085) | 0.1088 | 0.110 |
| **identity** | MTG | 6.18 → 7.70 | 34.04 → 27.94 (0.449) | 79.4% | 1.521° | (−0.679, −0.592) | **0.0829** | **0.099** |
| **identity** | insula | 55.67 → 7.61 | 153.1 → 27.78 (0.449) | 79.4% | 1.505° | (−0.678, −0.598) | **0.0829** | **0.099** |
| identity, normalised with I₂ | MTG | 62.19 → 61.37 | 34.04 → 31.40 (0.505) | 75.4% | 0.003° | (−0.125, −0.334) | 0.0856 | n/a |
| identity, normalised with I₂ | insula | 61.87 → 57.92 | 153.1 → 152.74 (2.469) | **0.0%** | −0.075° | (−24.45, −23.46) | 0.0890 | n/a |

Notes on the table:
- The structural start values differ between maps: under I₂ the reference start already
  matches gray/white, so with the same normalisation constant it starts at 6.18.
- "g/w mismatch with I₂ at end" is the target-normalised structural mismatch of the end
  pose evaluated with I₂ for both maps, so it is comparable. The `score_classes` in
  `scores.json` uses each fit's own π and is not comparable across maps.

Findings:

- **Same basin from both starts** under the fixed identity: 0.007 mm and 0.016° apart,
  79.4% MTG.
- **Better gray/white, essentially the same anatomy.** The identity pose is 0.5 mm and
  1.5° from the learned-π pose. It matches gray/white better (0.099 vs 0.110), with
  essentially the same anatomical fit (relative 0.449 vs 0.440) and a better tissue score
  (0.083 vs 0.109).
- **The learned π flattered itself.** It mixed classes (24% of atlas white explaining QNP
  gray), scoring 0.083 under its own π while its pose was worse for true gray↔gray,
  white↔white.
- **Backward compatibility,** fresh refits bitwise (28/28 and the scores):
  - λ = 0 base and weight-0 records against the pre-change reference;
  - the learned-π rigid MTG fit against its previous run.
  - Learned mode builds exactly the same model as before.
- **QC.** The feature-map panel shows the structural π and states "fixed identity I_2,
  not learned" or "learned (optimised)". The anatomical map ("fixed [[1]] (I_1), not
  learned") is written under it; `pi_ST.csv` states it too.

### Loss normalisation and records A / B (2026-11-17)

Records A and B now use the full-window archives and the identity map; the originals are
kept as `*_footprint_learned`. The normalisation audit, which shows that native losses
are not comparable across candidates and recommends a target-normalised score, is in
[`qnp_loss_normalisation_audit.md`](qnp_loss_normalisation_audit.md).

## Degenerate solutions checked

- **Escaping the mask is never rewarded.** With A applied before smoothing, moving atlas
  MTG out of the annotation takes the term *up* to its complete-miss value. That is a
  maximum over non-overlapping configurations (fork tests on the plateau and on
  zero-censor mass).
- **The scale is inflated under similarity.** Positive-only censoring is asymmetric:
  - atlas MTG that *under*-covers the annotation is penalised;
  - atlas MTG that *over*-covers it is censored, and so free.
  - Since the weights scale with area, enlarging the atlas raises coverage at constant
    density.
  - With λ_anat ≥ 1 the similarity scale rose from 1.04 to 1.16: 16% linear, 34% area,
    and visible in `registration_t1.00.png`.
  - The masked-MTG / annotated mass ratio rose from 0.72 to 0.80, and the tissue score
    worsened from 0.108 to 0.135.
  - Under a rigid transform the bias disappears: at λ_anat = 1 the anatomy improves
    (MTG 74% → 77%) with no structural cost.
  - **Recommendation:** use the anatomical term with `rigid`, or with a scale prior,
    until this is addressed.
- **Particle masses.** Atlas MTG is fully-MTG pixel area; the annotated QNP mass is
  tissue area times scale². Both are areas in plane units, so no π is needed.
  - At the correct start, the masked atlas MTG mass is 0.72 of the annotated mass.
  - The atlas MTG gray ribbon does not cover all of the annotated L1–L6 ribbon:
    annotated tissue also lands on atlas WM (11–14%) and ITG.
  - This leaves a residual D²_anat (3–5 against a miss of 38.6), and it is part of what
    drives the scale bias.
- **Normalisation.** Under `initial` normalisation every start costs the same, so the
  term would weigh a wrong start *less*. The default `target` normalisation is
  start-independent.
  - The *structural* term keeps its per-start normalisation, unchanged. So the effective
    ratio λ_anat : λ_struct still varies a little between starts, through the structural
    term's per-start scale.

## Registration QC figure (`registration_qc.png`)

Every fit now writes `registration_qc.png` and `registration_qc.json` as its **last**
outputs, after everything else, so neither the fit nor any earlier output can depend on
them. Existing fits get them without refitting:

```bash
PYTHONPATH=src:src/config_setup:src/preprocess python src/scripts/framework_script_AtlasToSectionCrossModalBoundary.py \
    --config <config> --registration <name> --stage qc [--init r000_f0]
```

Code:

- `registration_qc_payload`, `write_registration_qc` and `_similarity_fit` are in the
  driver.
- `registration_qc_figure` is in `src/scripts/view_xiv_trajectory.py`, next to the
  existing 2-D figure.
- It reuses xmodmap's disc sizing (`xmodmap.plot.utils.weight_to_radius`,
  `default_radius_scale`: disc area = particle mass) and the existing `_target_outline`.
- `registration_t1.00.png` is unchanged (byte-identical in the λ = 0 check).

Layout: three columns (atlas at t = 0 | QNP target | transformed atlas). All spatial
panels share one frame (mm, initialized target frame), one axis range and one
mass-to-disc-area scale.

| row | content |
|---|---|
| 1 structural | gray/white of the atlas, of the target, and of the transformed atlas on the target tissue outline. Gray/white are drawn in neutrals so no hue means two things. |
| 2 anatomical | atlas `nu_Anat` channels (STG orange, MTG blue, ITG aqua; other gray and WM neutral); the target's HALO annotation (filled = annotated area; rings = all tissue, so unannotated = unknown) with the annotation-mask contour A = ½; the transformed atlas in its anatomical colours with mask and outline. Without `nu_Annot` the channel panels are still drawn. |
| 3 what each term counts | **structural:** w·s(x) as filled discs, the full transformed atlas as rings. **Anatomical:** w·ζ_MTG·A(x), the same way, marked "in objective" or "NOT in objective, weight 0". **Landing:** annotated target tissue coloured by the atlas channel it lands on after pull-back, with fractions. |
| 4 numbers | the learned structural π_ST (annotated matrix), and a summary (also in the JSON). |

The summary (and JSON) contains:

- the initial transforms: the rigid-scan QNP→plane similarity (θ, scale, flip) and the
  xIV initialization matrix;
- the flow's similarity parameters: scale, rotation, translation and reflection, from a
  least-squares fit of start vs end particles, with the non-similarity residual (≈ 0 for
  rigid and similarity flows; 1.87 mm rms on the 0.5 mm diffeo fit);
- weighted and unweighted losses per stage, and the scores;
- mass and support totals per term;
- the feature maps actually used, and the mass-ratio proxy quantiles for diffeo stages.

### Data sources

Nothing is re-estimated:

- **positions and masses:** the first and last frames of `trajectory.pt`; the saved
  `atlas_deformationSummary.D` equals the last frame exactly;
- **end structural support:** the saved `relative_support_weight`;
- **start support, A and ζ_MTG:** the model's own loss objects evaluated at those
  positions;
- **landing:** the saved pull-back `Td`, which matches `anatomical_diagnostics.json`
  exactly.

The in-fit QC (live model) and `--stage qc` (rebuilt model; the unit box is checked
against `model_settings.pt`) agree in every number.

### Feature maps (checked in the model, not assumed)

| term | source features | source → target map |
|---|---|---|
| structural | `nu_Sub` = the fixed `gmwm` projection of `nu_Anat` | **π_ST is learned**: `pi_ST` is in `to_optimize` (except `target_feature_mode: total_mass`). It is initialised uniform and is not identity after fitting. E1 λ_anat 0: π² = [[1.018, 0.086], [0.192, 0.861]] (rigid). E1 λ_anat 1: [[0.978, 0.102], [0.252, 0.807]] (similarity). Row sums are not constrained. |
| anatomical | the fixed `mtg_gm` projection [0,1,0,0,0] | fixed [[1]], not optimised |

Because π_ST is learned and not identity, the figure shows it.

### Validated on

Existing outputs; no new optimisation beyond re-running the two λ = 0 fits for the
bitwise check:

- **H20.33.034 1 mm** (`~/tmp/anatomical-varifold/fits/`): similarity λ 0/1/5 and rigid
  λ 0/1/5. The weight-0 fits show the mask and landing while marked "NOT in objective".
- **0.5 mm affine+diffeo** fit (`~/tmp/atlas-anatomical-features/fits/projected_gmwm`):
  no annotation, so the channel panels are shown and the annotation panels say so.
- **Bitwise check rerun with the final code, including the QC hook:** pre-change driver +
  pristine fork vs new (no block / weight 0): 28/28 arrays and the scores identical.
- 4 tests in `test_anatomical_supervision.py`: similarity fit; figure with full, channels
  only, and structural only.

### Field of view

The QC draws the model's own atlas particles, so it can only show as much anatomy as the
exported source holds.

- **The default export is cropped.** It keeps atlas particles near the rigidly placed
  section (footprint weight > `support_min_weight`). That cuts away most of STG, the
  insula and the rest of the neighbouring cortex: 702 particles at 1 mm for plane A.
- **`export.restrict_source_to_footprint: false` keeps the whole `half_width_mm` window.**
  On plane A that is 2512 particles, including STG, insula (in other gray) and FuG.
  - The structural term's tissue support still censors everything off the section.
  - The flag changes the source, so it is a model choice. The default stays `true`.
  - It reproduces the scratch E2 export array for array, so the E2 fits are larger-FOV
    fits.
- **QC for the existing E2 fits is written without refitting**
  (`~/tmp/anatomical-varifold/qc_wide.py`):
  - the reference placement and the STG / ITG / FuG / insula starts, at λ_anat 0, 1 and 5;
  - files: `~/tmp/anatomical-varifold/fits/wrong_starts/<start>/w<λ>/registration_qc.png`;
  - every figure's landing fractions match `anatomical_diagnostics.json`.

### Not supported, not added

- Deformation grids, displacement fields and Jacobian maps were not added. The xmodmap
  plot module has no 2-D grid or displacement viewer: its viewers are particle viewers,
  plus the 3-D pyvista one.
- Non-linear change is visible as particle spacing and disc area (mass), and is
  summarised by the mass-ratio proxy.

## Stacking-coordinate evaluation (2026-11-18)

Baseline vs supervised (weight 1, `censoring: none`, similarity) over the whole MTG depth range, on both donors:
[`qnp_stacking_xiv_anatomical.md`](qnp_stacking_xiv_anatomical.md).

- **Effect on depth:** the term decides the gyrus, not the depth.
- **H20.33.034:** no improvement.
- **H20.33.043:** MTG placements were reached only by shrinking the atlas (scale 1.39-1.45, annotation partly
  outside the brain).

## Remaining limitations

- **The term is not a global search.** From starts disjoint from the annotation (e.g.
  insula) it gives no gradient. Use it to rank multistarts or plane hypotheses with
  D²_anat, or start within overlap.
- **The scale inflation under similarity is unresolved** (see above). It needs a scale
  prior, a two-sided penalty for atlas MTG on tissue that is *known* not to be MTG (no
  such labels exist), or rigid fits.
- **One donor, one plane.** Only H20.33.034 plane A was tested: coarse 1 mm, similarity
  and rigid, 150 steps.
  - Not run: 0.5 mm, full resolution, diffeo, tilted planes, slab order, and the second
    donor.
  - Float32 GPU fits depend on process history, so read differences < ~0.02 in tissue
    score between single runs as spread.
- **Supervision consumes the ribbon.** Once the term is active, the HALO ribbon
  supervises the fit. The ribbon-landing evaluations (`--stage evaluate`, and the
  annotated landing in `anatomical_diagnostics.json`) are then no longer independent
  checks. A held-out check would need, e.g., supervision by some layers and evaluation on
  others; per-layer masses are not exported yet.
- **The label definitions differ.** Atlas MTG is a population-template label with
  straight-cut gyral borders (`docs/qnp_atlas_anatomical_features.md`). The HALO ribbon
  is donor cortex, L1–L6 including some of what the atlas calls WM at 1 mm. Perfect
  agreement is not expected.
- **Figure and model class.**
  - The registration figure shows gray/white only. The anatomical overlap is reported
    numerically, not drawn.
  - `CrossModalityBoundaryMultiTerm` and the landmark branch's `CrossModalityBoundary`
    changes are not yet combined: a fit with both landmarks and extra terms needs them
    on one class.
