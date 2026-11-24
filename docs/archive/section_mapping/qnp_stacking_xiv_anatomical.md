# QNP stacking coordinate with and without HALO MTG supervision (2026-11-18)

**Question.** Does the positive-only HALO MTG anatomical varifold term improve the estimated
stacking coordinate (coronal atlas depth) of the QNP sections of H20.33.034 and H20.33.043?

**Answer.** Not as configured.

- **H20.33.034:** the structural baseline already gives a consistent depth. All three anchors
  sit at y = −27.5 … −28.0. Supervision moves NeuN by 1 mm, worsens its gray/white fit and
  enlarges its scale. It does not raise the fraction of annotation that lands on MTG.
- **H20.33.043:** the baseline places every anchor on the insula/operculum (y −13 … −16, under
  1% of the annotation on MTG). Supervision pulls the two annotated sections onto MTG at
  y ≈ −9, but only by shrinking the atlas: scale 1.39–1.45, 27–29% of the annotated tissue
  outside the atlas brain, gray/white mismatch doubled. The registration QC shows these poses
  are not anatomically plausible.
- **Why it cannot help with depth:** the anatomical term mainly fixes *which gyrus* the section
  lies on, not *how deep* it lies. Atlas MTG runs the whole y-extent, so R_anat is nearly flat
  (≈ 0.2–0.6) over ~85 mm of depth.

Nothing in the xIV objective, normalisation, kernels or optimizer was changed. This is an
evaluation of the existing approach, plus a reusable procedure to run it.

## Procedure

Code:

- [`src/scripts/estimate_qnp_stacking_xiv.py`](../src/scripts/estimate_qnp_stacking_xiv.py)
  (stages);
- [`src/preprocess/qnp_stacking_xiv.py`](../src/preprocess/qnp_stacking_xiv.py)
  (computations);
- [`src/scripts/compare_qnp_stacking_xiv.py`](../src/scripts/compare_qnp_stacking_xiv.py)
  (both donors).

Config: `qnp_section_defaults.atlas_scan.xiv_stacking`, shared by both donors. Tests:
[`src/tests/test_qnp_stacking_xiv.py`](../src/tests/test_qnp_stacking_xiv.py) (7).

```bash
# env ad-resilience-xiv; PYTHONPATH=src:src/config_setup:src/preprocess:src/scripts
# prerequisites: scan_atlas_planes_for_qnp.py --stage roles, compress (anchors, h1.0 particles)
E="python src/scripts/estimate_qnp_stacking_xiv.py --config configs/project_config.yaml --job $J --stage"
$E sections; $E starts --level coarse
$E fit --level coarse --worker i --workers n      # n processes share the GPU (3 per donor here)
$E starts --level fine;  $E fit --level fine --worker i --workers n
$E report; $E qc
python src/scripts/estimate_qnp_stacking_xiv.py --config configs/project_config.yaml --stage compare \
    --jobs H20.33.034_MTG H20.33.043_MTG --out results/registrations/qnp_MTG_stacking_xiv_anatomical_comparison
```

`fit` is resumable: an existing result is kept unless `--overwrite` is given, and the number
kept is printed.

### Sections

- **Informative sections** are the anchors of the existing role QC (`roles.json`), the same
  rule on both donors:
  - H20.33.034: NeuN, LFB, AT8;
  - H20.33.043: NeuN, LFB, I6.
- **Target:** each anchor's own 500 µm gray/white particles in the donor's local QNP frame.
- **Supervision** uses only the slide's *own* HALO layer annotation, and only where it fits its
  tissue (inventory rule).
  - Annotated anchors: NeuN (both donors) and I6 (H20.33.043).
  - Unannotated: LFB (no annotation on either donor) and AT8 (H20.33.034, annotation misfits
    its tissue).
  - Unannotated tissue is unknown (mass 0); no labels are transferred between slides.
  - An unannotated section runs the baseline objective in the supervised condition, so its
    two conditions are the same fits.
- **Annotated mass per particle** = the annotated share of the 32 µm tissue pixels nearest that
  particle × its exact tissue mass. The 500 µm particles were binned at full resolution, which
  the 32 µm maps cannot reproduce exactly; this rule stays between 0 and the tissue mass by
  construction. Annotated areas:

  | section | annotated area | share of tissue |
  |---|---|---|
  | NeuN, H20.33.034 | 202.8 mm² | 52% |
  | NeuN, H20.33.043 | 168.2 mm² | 39% |
  | I6, H20.33.043 | 170.5 mm² | 40% |

- **Passengers** (034: ASYN, GFAP, I6; 043: AT8, ASYN, GFAP) are not fitted to the atlas. They
  keep the existing within-donor alignment, which places them in the anchors' local frame.

### Two conditions, one model

Both conditions use the same xIV fit:

- the registration driver's own `build_model` / `run_stages`;
- record A's model, with only `transform: affine` (a 2-D similarity: rotation, translation,
  isotropic scale);
- varifold 8/4/2/1 mm, RKHS 8/4/2 mm, γ 0.1, 150 steps, float32;
- fixed identity gray/white map, existing min-max tissue support, native normalisation;
- no scale prior.

| | baseline | supervised |
|---|---|---|
| structural term | identical | identical |
| anatomical term | weight 0 | weight 1, `censoring: none`, fixed MTG→MTG map |

**Atlas source.** The whole 70 mm window of the coronal plane at 1 mm (0.5 mm reslice summed
2×2), carrying gray/white and the one-hot anatomical channels.

### Search: coarse to fine, identical for both conditions

1. **Coarse.**
   - Depths: every plane along the atlas MTG y-extent ± 5 mm, at a 3 mm stride: 35 depths,
     +24.5 … −77.5. No shortlist.
   - Starts: at each depth, the existing global GPU 2-D search (15° × 4 scales × both flips,
     h1.0 particles) proposes up to 3 distinct refined poses.
   - Both conditions start from the same poses.
2. **Fine.**
   - Windows: ± 3 mm at 0.5 mm steps around each condition's 2 best distinct coarse minima.
     These are taken under both rankings (below) and unioned over conditions and anchors, so
     both conditions are refined at the same depths:

     | donor | coarse centres | fine depths |
     |---|---|---|
     | 034 | 9 | 101 |
     | 043 | 5 | 63 |

   - At every fine depth: one fresh scan pose (shared by both conditions), plus each
     condition's own best coarse pose of that window, as in the skill's escalation.
3. **Fits run:**

   | donor | finite fits | failed |
   |---|---|---|
   | 034 | 1,204 | 0 |
   | 043 | 1,112 | 2 (non-finite; recorded) |

   With 6 processes on the RTX 2080 SUPER, the coarse level took ~30 min and the fine level
   ~45 min.

### Ranking: a common score, not the native loss

Every fitted pose is scored *after* the fit, in the section's fixed local frame.

- **Atlas particles** are carried into that frame by the composed pose. Their areas are pushed
  forward (divided by scale²), as the rigid scan's `FootprintScorer` does.
- **Kernels** are the objective's own (8/4/2/1 mm), weighted 1/σ:

      R_term = Σσ wσ |μA − μT|²σ / |μT|²σ ,   wσ ∝ 1/σ
      R_struct: gray/white, identity map, atlas weighted by the section's fixed footprint (0.5 mm)
      R_anat:   atlas MTG gray vs annotated tissue, uncensored
      R_combined = R_struct + R_anat

- **Ranking per condition:**
  - baseline by R_struct;
  - supervised by R_combined (annotated sections) or R_struct (others).
- **Two kinds of objective.** xIV optimised its own native loss; R is only used to compare the
  fitted poses. A post-fit ranking of finitely many local fits does not guarantee the global
  infimum of R.

**The target is physically identical across candidates.** The plane A / plane B target-area
difference in the audit (annotated 242.5 vs 245.1 mm²) is exactly the start scale² of the two
exports:

- the archives store the section in atlas-plane units (QNP area × s_init²), with s = 1.0919
  vs 1.0977;
- in QNP units both hold the same 392.445 mm² of tissue and 203.413 mm² annotated.

Here the score is evaluated in QNP units, so every candidate of a section is compared with
bitwise the same target. The tests check that one configuration reached from starts of
different scale scores identically.

**Pose bookkeeping, checked on every fit:**

- the flow is a similarity (least-squares residual ≤ 3·10⁻⁵ mm);
- transported mass equals scale² (ratio 1.0000);
- the shot particles and the composed pose give the same R (≤ 10⁻⁴).

**Scale admissibility.** Many fits leave the scan's scale range [0.8, 1.5] (below). Each
condition is therefore ranked twice:

- over **plausible-scale fits** (final pose scale in [0.8, 1.5]; primary);
- over **all fits**.

This is a post-fit filter, not part of the objective.

**Windows and δ.**

- **Window:** all fine depths within δ of the best.
- **δ:** the skill's definition, the across-anchor spread of relative scores.

  | donor | δ R_struct | δ R_combined |
  |---|---|---|
  | 034 | 0.0095 | borrowed from R_struct: one annotated anchor only |
  | 043 | 0.0055 | 0.0062 (NeuN and I6) |

## Results (plausible-scale ranking)

Notation:

- **scale**: section mm → atlas mm;
- **θ**: rotation;
- **gray→gray / white→white**: fraction of section gray (white) area landing on atlas gray
  (white);
- **outside**: section tissue landing outside the atlas brain;
- **annot→MTG**: fraction of the annotated area landing on atlas MTG (nearest 0.5 mm voxel).
  - Under supervision, annot→MTG is what the fit optimised (training).
  - Under the baseline, and for the NeuN annotation viewed from the LFB/AT poses, it is an
    independent check (those fits never saw an annotation).

### H20.33.034

| section | condition | best y | window | R_struct | R_anat | scale | θ | gray→gray | white→white | outside | annot→MTG | runner-up |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| NeuN | baseline | −27.5 | −27.0 … −28.0 | 0.116 | 0.455 (not fitted) | 1.10 | 128° | 0.88 | 0.73 | 0.04 | 0.77 (independent) | −41.0 (3.7 δ) |
| NeuN | supervised | −26.5 | −26.5 … −27.0 | 0.152 | 0.222 | 1.19 | 124° | 0.83 | 0.64 | 0.07 | 0.74 (fitted) | −11.0 (3.0 δ) |
| LFB | both (no annotation) | −27.5 | −1.0 … −28.0, gaps | 0.107 | — | 1.03 | 129° | 0.84 | 0.83 | 0.03 | NeuN ribbon 0.79 | −1.5 (0.3 δ, scale 0.84) |
| AT8 | both (annotation misfit) | −28.0 | −27.5 … −28.5 | 0.138 | — | 1.09 | 126° | 0.85 | 0.69 | 0.03 | NeuN ribbon 0.74 | +12.0 (2.6 δ, scale 1.49) |

All poses are flip True.

**Consistency within the block:**

- baseline: best depths −27.5 / −27.5 / −28.0; common window −27.5 … −28.0.
- supervised: NeuN's window (−26.5 … −27.0) no longer overlaps AT8's, so there is no common
  window (spread 1.5 mm).

This baseline reproduces the earlier GPU stacking result (−27.5 / −28 / −28) with a different
fitter (xIV), unconstrained scale and the whole window.

LFB is nearly flat: a pose at y −1.5 (scale 0.84) is within 0.3 δ. Over *all* fits, LFB's best
is y −36.5 at scale 0.75. Its depth is set by the scale filter, not by the image.

### H20.33.043

| section | condition | best y | window | R_struct | R_anat | scale | θ | gray→gray | outside (tissue / annotation) | annot→MTG | runner-up |
|---|---|---|---|---|---|---|---|---|---|---|---|
| NeuN | baseline | −16.0 | −15.0 … −17.0 | 0.117 | 2.76 (not fitted) | 1.02 | 265° | 0.83 | 0.05 / 0.07 | 0.01 (independent) | −62.0 (7.6 δ, scale 0.82) |
| NeuN | supervised | −8.5 | −8.5 … −9.5 | 0.248 | 0.327 | 1.39 | 315° | 0.68 | 0.14 / 0.27 | 0.58 (fitted) | −16.0 (3.8 δ) |
| LFB | both (no annotation) | −14.0 | −13.0 … −16.5, gaps | 0.116 | — | 1.12 | 268° | 0.81 | 0.06 / — | NeuN ribbon 0.005 | −56.5 (7.5 δ) |
| I6 | baseline | −13.0 | −12.5 … −14.5 | 0.129 | 2.28 (not fitted) | 1.19 | 271° | 0.82 | 0.06 / 0.09 | 0.01 (independent) | −25.0 (5.1 δ) |
| I6 | supervised | −9.0 | −8.5 … −10.0 | 0.255 | 0.330 | 1.45 | 319° | 0.69 | 0.14 / 0.29 | 0.57 (fitted) | −60.5 (9.3 δ) |

**Consistency within the block:**

- baseline: −16 / −14 / −13; no common window (NeuN and I6 miss by 0.5 mm; LFB overlaps both).
  This reproduces the earlier GPU result (−15 / −14 / −13).
- supervised: NeuN and I6 agree (−8.5 … −9.5), but LFB stays at −14.
- The NeuN–I6 agreement is not independent evidence: I6's HALO layers are NeuN's, transferred
  by HALO's serial-section registration.

Registration QC (supervised NeuN, y −8.5):

- the atlas is shrunk (xIV scale 0.77) so that atlas MTG fits inside the annotated cortex;
- the section's lateral lobe then lies outside the atlas brain (23.6% of the annotated tissue
  by the figure's nearest-particle rule);
- gray/white structure is poorly matched.

The baseline pose (y −16) matches gray/white well. Its annotated cortex lands on insula/opercular
"other gray" (72%) and STG (18%).

**MTG placements with good structure exist, at scattered depths.** These are the best baseline
fits (never shown the annotation) that land ≥ 50% of the annotation on MTG, with scale in
[0.8, 1.5]:

| section | y | R_struct | annot→MTG | scale | supervised optimum: R_struct |
|---|---|---|---|---|---|
| NeuN | −62.0 | 0.159 | 0.71 | 0.82 | 0.248 |
| LFB | −38.5 | 0.174 | 0.76 | 0.95 | — |
| I6 | −25.0 | 0.157 | 0.73 | 0.99 | 0.230 |

All are 5–9 δ worse structurally than the insula pose.

**Selection-only ablation** (baseline fits ranked by R_combined, i.e. supervision used only to
pick among unsupervised fits):

- NeuN: y −7.5 (scale 1.05, annot→MTG 0.68);
- I6: y −38.5 (0.71);
- the two disagree by 31 mm.

## Assessment

1. **Supervision decides the gyrus, not the depth.**
   - Atlas MTG exists at every depth from y ≈ +20 to −72.
   - The supervised fit can always bring atlas MTG under the annotation: R_anat is ≈ 0.2–0.6 over
     the whole range, and rises to 1 only where the window holds no MTG.
   - The depth contrast of R_combined therefore comes from the structural term of the
     supervised fits.
   - On H20.33.034 the contrast is no better than the baseline's: runner-up at 3.0 δ vs 3.7 δ.
2. **The uncensored term moves the fit through scale.**
   - It counts all atlas MTG in the window against the annotated cortex.
   - Shrinking the atlas lowers its self term, so supervised fits shrink the atlas: median xIV
     scale 0.85 (034) and 0.72 (043); the worst 5% reach 0.07–0.44.
   - 33% / 59% of the coarse supervised fits end outside [0.8, 1.5] (baseline: 23–25%).
   - Where the section's anatomy disagrees with MTG under the structural term (043), the
     supervised optimum is reached by shrinking the atlas. It then pays with 27–29% of the
     annotation outside the brain and a doubled gray/white mismatch.
   - This is the mass-parity bias found earlier at a fixed plane (scale 0.89 on plane A),
     amplified by the whole-window MTG.
3. **H20.33.034:** supervision brings no improvement. The structural stack is already
   consistent and lands the (unseen) NeuN annotation 77% on MTG. Supervision trades gray/white
   fit for a lower R_anat without raising the MTG landing.
4. **H20.33.043:** the conflict is real, and supervision does not resolve it plausibly.
   - Gray/white strongly prefers an insula/opercular placement (5–9 δ better than any MTG
     placement).
   - The dissection label and the HALO annotation say MTG.
   - MTG placements with acceptable structure exist at several depths (−25, −38.5, −62), so
     even granting "on MTG", the depth is not identified.
   - The supervised optimum (y ≈ −9) depends on near-limit scaling and leaves tissue outside the
     brain.
5. **Ranking.** The audit's concern is real: the native losses here are not comparable.
   - Unit box 55 mm (034) vs 65–69 mm (043).
   - Final native totals range from 5.8 to 59 across leaders, unrelated to fit quality.
   - The common score handles comparison across candidates.
   - The remaining problem is the anatomical *objective*'s scale behaviour, not its
     normalisation.

## Limitations

- **xIV mostly polishes.**
  - 57% of all fits (1,312 of 2,316) stopped on line-search no-progress (the float32 issue #5),
    at a median of 33–47 closure evaluations.
  - Baseline fits changed R_struct by a median of only 0.001 from their start.
  - So xIV polishes the scan's starts more than it searches.
  - Coverage comes from the starts. These are GM/WM-derived and shared, and they include 3
    distinct poses per depth.
- **No guarantee of the global optimum.** The rankings are over finitely many local fits; they
  do not guarantee the global infimum of R.
- **One donor without an independent δ.** δ for R_combined on H20.33.034 is borrowed: only one
  annotated anchor.
- **Not tested here:**
  - atlas 1 mm only (no 0.5 mm confirmation);
  - coronal planes only;
  - passengers not re-fitted;
  - no per-layer holdout of the annotation.

## Outputs (new directories; nothing old was modified)

- `results/registrations/qnp_<job>_stacking_xiv_anatomical/`:
  - inputs and plans: `sections.json`, `starts_*.csv`, `fine_plan.json`;
  - fits: `fits_{coarse,fine}/<section>/<condition>/*.json`, one per fit, with native losses
    and components, stop reason, start/fit pose, xIV similarity, R scores, atlas MTG mass,
    anatomical self/cross, landing;
  - results: `report.json`, `REPORT.md`, `candidates.csv`, `profiles.png`;
  - QC: `qc/<section>/<condition>/<candidate>/init_r000_f0/` holds the driver's full outputs,
    incl. `registration_qc.png`. `qc_candidates.csv` shows that each refit reproduces its sweep
    pose (0–0.001 mm).
- `results/registrations/qnp_MTG_stacking_xiv_anatomical_comparison/`: `comparison.json`,
  `profiles_by_donor.png`.

## Possible next steps (not done; each changes the model or adds evidence)

- **Rigid supervision.** Run the supervised condition with `transform: rigid`. Rigid +
  uncensored had no mass bias at a fixed plane. The scale would then come from the baseline's
  structural fit.
- **Explicit 043 basins.** Compare the 043 basins explicitly (insula y −13 … −16 vs MTG
  candidates at −25 / −38.5 / −62) at full resolution, and against the dissection prior.
- **Tilted slab families** for 043: the coronal-only search missed an oblique candidate on 034
  before.
