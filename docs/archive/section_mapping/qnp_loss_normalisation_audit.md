# Loss normalisation audit: structural and anatomical varifold terms (2026-11-17)

**Question.** Can xIV loss values be compared across candidate atlas planes and
initialisations? This matters before the global stacking search is restarted.

**Answer.** Not the native losses. The structural term's value is defined *relative to
its own start*, and both terms carry a factor of the internal unit-box scale. The
anatomical term's "relative" value, and the pipeline's fixed-footprint scores, are
comparable. A consistent normalisation is recommended below.

This is an audit only. Nothing in the objective was changed. The numbers come from the
driver's own `build_model` (no fitting) on H20.33.034, plane A full window at 0.5 mm,
`r000_f0`, varying one factor at a time. Script and output:
`~/tmp/anatomical-varifold/normalisation_audit.py`, `normalisation_audit.json`.

## The terms as implemented

Notation:

- `s`: the unit-box scale from `rescaleData(S, T)`. It is the largest side of the joint
  bounding box of the atlas source *and* the initialised target, in mm.
- Kernel widths are converted to unit-box units as `σ_u = σ_mm / s`.
- Masses (`w`) are in mm², and kernel values `exp(−|x−y|²/2σ_u²)` are the same in either
  unit. So every `|·|²_σ` below is independent of `s`.

**Structural term** (`LossVarifoldNormBoundary_Partial`, `normalize_across_scale`):

    D²_struct = Σ_σ β_σ ½ |c·φμ_S π − μ_T|²_σ,     β_σ = (0.6/σ_u) · 2 / N_σ
    N_σ = |c₀·μ_S π₀ − μ_T|²_σ          (the mismatch at the start)

- `c` is the min-max tissue support at the current pose; `c₀` is the support at the start.
- `π₀` is makePQ's initial π: uniform, times the mass ratio `Σw_T/Σw_S`. Since
  2026-11-17 it is kept as the normaliser for the fixed-identity map too.
- Hence `D²_struct(start) ≡ Σ_σ 0.6/σ_u = Σ_σ 0.6·s/σ_mm =: C` for **every** start.

**Anatomical term** (`LossVarifoldNormCensored`, `normalize_to_target`):

    D²_anat = Σ_σ (0.6/σ_u) · |A·φμ_S^MTG − μ_T^annot|²_σ / |μ_T^annot|²_σ = C · R_anat

- `R_anat` is the 1/σ-weighted mean of the target-normalised mismatches (the `relative`
  value already reported).
- `A` is the annotation censor; it is absent for `censoring: none`.

## Dependence of each normaliser

| factor | structural (`β`, start value `C`) | anatomical (`target`) |
|---|---|---|
| **initialisation** (pose) | **Yes, by construction.** The start value is always C, however good or bad the start. A perfect start and a 38 mm wrong start both begin at 61.9. The end value is C × (end mismatch / start mismatch): an improvement ratio, not a quality. N_σ itself moves little across poses (uniform π₀ explains any pose about equally badly); the support c₀ moves it slightly. | No, apart from `s`. |
| **atlas plane** | Through `s` (window extent), the atlas mass in the window (via π₀'s mass ratio), and support-weighted content. | Through `s` only. The annotated target mass varies slightly with the plane scan's scale² (A 242.5, B 245.1 mm²). |
| **atlas window / particle mass** | **Strong.** π₀ ∝ Σw_T/Σw_S: 0.33 for the footprint crop vs 0.094 for the full window. N/T_self is 0.26–0.47 (crop) vs 0.72–0.80 (full). | No (target-normalised). |
| **internal scaling `s`** | C ∝ s, multiplying every value. | C ∝ s, multiplying every value. |
| **feature map** | With learned π, the end value also reflects π adapting, i.e. class mixing. With identity, the normaliser is unchanged (same π₀); normalising with I₂ would raise β 7–45×. | Fixed 1-to-1; no dependence. |
| **particle resolution** | Small (1 mm vs 0.5 mm: N/T 0.74 vs 0.72, s 55.3 vs 55.0). | Small (R 0.547 vs 0.531). |

## Measured

N/T is the structural normaliser divided by the target self-norm, per kernel (8, 4, 2,
1 mm). "R_struct(I₂)" is the start structural mismatch with the identity map, target-
normalised and 1/σ-weighted: what an honest start score would be.

| candidate | s (mm) | C (= every structural start value) | atlas mass (support-counted at start), mm² | N/T (uniform π₀) | R_struct(I₂) at start | R_anat at start (uncensored) |
|---|---|---|---|---|---|---|
| A full, r000_f0 (reference) | 55.02 | 61.90 | 2493 (446) | 0.72 / 0.74 / 0.77 / 0.80 | **0.077** | **0.531** |
| A full, r090_f0 (rotated) | 55.00 | 61.88 | 2493 (408) | 0.74 … 0.82 | 0.588 | 1.373 |
| A full, r000_f1 (reflected) | 55.00 | 61.88 | 2493 (423) | 0.73 … 0.81 | 0.503 | 1.373 |
| A full, start on STG (13.8 mm) | 55.00 | 61.88 | 2493 (455) | 0.72 … 0.80 | 0.697 | 1.362 |
| A full, start on insula (38.5 mm) | 55.00 | 61.88 | 2493 (421) | 0.73 … 0.81 | 0.762 | 2.464 |
| B full, r000_f0 (other plane) | **70.00** | **78.75** | 2386 (454) | 0.70 … 0.79 | 0.081 | 2.039 |
| A footprint-cropped, r000_f0 | **34.52** | **38.84** | **708** (446) | **0.26 … 0.47** | 0.077 | 0.509 |
| A full at 1 mm, r000_f0 | 55.28 | 62.19 | 2512 (417) | 0.74 … 0.81 | 0.080 | 0.547 |

### What the table shows

- **The structural term cannot rank candidates.** Its start value is C everywhere,
  although the true gray/white mismatch differs 10× (0.077 vs 0.762). Final values are
  improvement ratios scaled by C.
- **`s` rescales both terms by candidate geometry.**
  - Plane B's window spans 70 mm against 55 mm for A, so every B loss is 1.27× an A loss
    for the same relative mismatch.
  - Cropping the window to the footprint gives 0.63×.
  - In these full-window candidates the atlas dominates the bounding box, so offsets
    barely change `s`. With a cropped atlas, a moved target would change it.
- **The structural normaliser also tracks the atlas mass in the window** (π₀'s mass
  ratio): ×1.7–2.8 between the cropped and full windows of the same plane.
- **`R_anat` and `R_struct(I₂)` behave like scores.** They depend only on the target's own
  norm (fixed for a section), not on `s`, mass scale or start. They rank the reference
  start best on both terms and separate the wrong starts (anatomically: 1.36 vs 2.46),
  and they agree across resolutions.
- **The pipeline's existing scores are already comparable.** `score_total_mass` and
  `score_classes` in `scores.json` (and the GPU stacking scorer) are target-normalised,
  in mm and footprint-restricted. They use different kernels (2 / 1 / 0.5 mm, a plain
  mean) and the fit's own π, so `score_classes` is not comparable between learned and
  identity maps.
- **The Hamiltonian** (not audited numerically) is also evaluated in the unit box. Its
  nonrigid coefficient `uCoeff = cS·ΣK(S,S)/(N²σ_u²)` depends on the atlas particle count
  and `s`, so for diffeomorphic fits the data/regularisation balance is
  candidate-dependent too. Rigid and similarity fits are barely affected.

## Recommendation

**1. For ranking and reporting, now** (no objective change): never compare native losses
across candidates. Rank by one target-normalised, mm-unit score per term, using the
objective's own kernels and weights:

    R_term = Σ_σ (1/σ_mm) · |μ_S^term − μ_T^term|²_σ / |μ_T^term|²_σ  /  Σ_σ (1/σ_mm)
    R = λ_struct · R_struct + λ_anat · R_anat

- **Structural:** the identity map, and a *fixed* tissue support. Use the section's own
  footprint, as `score_total_mass` does, not the min-max support, which is relative to
  the current atlas particles.
- **Anatomical:** the censoring actually used.
- **Properties:** the only normalisers are the section's own self-norms, which are
  constant across planes and initialisations. `R` is free of `s`, invariant to mass
  scale, and makes the feature map explicit. `R_anat` is the existing "relative" value,
  and `R_struct` is a re-weighted `score_classes` with I₂.
- Adding `R_struct` and `R` to `scores.json` and to `summarize`'s ranking is a reporting
  change.

**2. For the objective, next step** (proposed; changes fits, needs re-validation):

- Normalise the structural term like the anatomical one: `β_σ ∝ 1/|μ_T|²_σ`, the target
  self-norm, not the start mismatch.
- Take the 0.6/σ factor in mm with one constant per section: `c · 0.6/σ_mm`, with c, say,
  the reference `s` (55 mm) to keep today's magnitudes.
- Equivalently, fix the unit box from the target alone. The section is the same for
  every candidate, so `s` stops varying, which also stabilises the Hamiltonian
  coefficients.
- Then `E_data = c' (λ_struct R_struct + λ_anat R_anat)` with a candidate-independent c',
  and λ_struct : λ_anat means the same at every start.
- The relative weight of the structural term would change against the current runs:
  validated balances (λ_anat = 1, γ = 0.1) should be re-checked on the rigid MTG / insula
  pair before the stacking search uses it.

**3. Support normalisation** (proposed, for the same change):

- The min-max support is relative to the current atlas particles' maximum density. With
  a full window the maximum sits in the section interior and is effectively fixed.
  Otherwise it rescales with the overlap.
- Normalising by the section's own maximum self-density (a constant per section) would
  make the censoring candidate-independent. Where the two maxima coincide, it is
  identical to the current rule.

## Related changes on 2026-11-17

- **Records A and B** now use `structural_feature_map: identity`. Their inputs are new
  full-window archives with `nu_Anat` / `nu_Annot` in
  `data/derivatives/sea-ad/qnp/registration_inputs/H20.33.034_MTG/full_window/`, and they
  write to new `*_full_window` result directories. The anatomical term is not enabled on
  them (weight 0).
- **The original definitions** are kept as `*_footprint_learned` (old archives, old
  result directories). The cross-modal MERFISH records keep the learned map.
- **Export changes:** `atlas_scan.export.restrict_source_to_footprint: false` and
  `record_subdir: full_window` for the H20.33.034 job. Export records are now in
  `<scan output_dir>/full_window/`; the original records and archives are untouched.
- **Checks on the new archives:**
  - targets identical to the old archives apart from `nu_Annot`;
  - ribbon points identical;
  - the old atlas particles are a subset of the new ones with identical `nu_Sub` /
    `nu_All` (A: 2833 → 9973 particles; B: 3495 → 9543).
- Plane B's window holds no STG and only 238 mm² of MTG (319 on A).

## Follow-up (2026-11-18)

Recommendation 1 (rank by a target-normalised score, objective unchanged) is implemented for the
stacking-coordinate comparison in [`qnp_stacking_xiv_anatomical.md`](qnp_stacking_xiv_anatomical.md).
The plane A / B target-area difference above is exactly the start scale² of the two exports
(1.0919² vs 1.0977²): in QNP units both targets are identical (392.445 mm² tissue, 203.413 mm²
annotated), so that score is evaluated in the fixed QNP frame.
