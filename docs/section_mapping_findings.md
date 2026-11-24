# Section mapping: findings and regression references

Results of the section-to-atlas mapping. The method and workflow are in
[section_mapping.md](section_mapping.md); the full historical reports are in
[archive/section_mapping/](archive/section_mapping/).

## 1. Regression references

These are software regression references, not anatomical ground truth.
`src/tests/test_mapping_regression.py` checks them against the saved runs in
`results/registrations/qnp_<job>_stacking/` and skips when those are absent. It checks:

- the landmark and interval;
- every saved profile summary;
- the reference placements and verdicts;
- a re-run of `report`;
- the stacking → xIV hand-off.

Each hand-off pose must reproduce its recorded score.

The saved runs and the derived inputs they read are also frozen, read-only, at
`results/frozen/section_mapping_regression_20261124/`. That copy is on the research volume, not
in Git. Its SHA-256 manifest is in Git at
`archive/section_mapping/regression_evidence_20261124.sha256`, and its README explains how to
verify the copy and restore the live paths from it.

| Property | Value |
|---|---|
| AHRAv2 LGN first appearance | MNI y −22.0 mm (label `LG`, code 174, ≥ 1 mm², left hemisphere; voxel precision 0.5 mm) |
| LGN-guided search interval | y −13.0 … −31.0 mm (4 mm slabs, j ∈ {−1, 0, +1}, every phase, ± 1 mm; 24 candidate slabs) |
| Regional prior | atlas support within 6 mm of MTG, STG and ITG gray matter; ITG kept |
| H20.33.034, unconstrained, 500 µm | NeuN −27.5 … −28.0; LFB −27.5 … −28.0; AT −27.5 … −28.0; best −27.5 / −28 / −28 |
| H20.33.034, region + slab prior | prior inactive for all three anchors (gap 0 δ): verdict "supported by image geometry" |
| H20.33.043, unconstrained, 500 µm | NeuN −14 … −16; LFB −13 … −14; I6 −13 … −14 (superior temporal plane / STG / insula) |
| H20.33.043, region + slab prior, 500 µm | y −24.5 for all anchors (ribbon 74–77 % on MTG); +13.1 / +13.6 / +7.4 δ worse on the data-only score: verdict "primarily prior-driven", **unresolved** |
| Unconstrained mode | priors off by default; each prior run writes its own sub-folder |
| Stacking → xIV hand-off | H20.33.034 NEUN is eligible for automatic export at y −27.5 (score 0.161470; unconstrained run, and the region + slab run with its candidates.json; the slab-only run has no candidates.json and is refused); H20.33.043 is refused until `--candidate` is given: `admissible` y −24.5 (data-only 0.248470), `unconstrained` y −15.0 (0.176276) |

## 2. Results by donor (2026-11-19)

### H20.33.034 (reference donor)

- **Placement.** The block sits at coronal y ≈ −27.75 ± ~0.5 mm. The three anchors (NeuN, LFB,
  AT8) agree to within 0.5 mm.
- **Priors.** The slab prior and the region prior leave the poses and scores unchanged.
  Slab j = +1 (the slab posterior to the LGN slab) is consistent. That fits a fixed QNP slab
  next to a frozen LGN slab, but this has not been verified.
- **Ribbon check.** The (unseen) NeuN annotation lands 77 % on atlas MTG.
- **Passengers.** ASYN, GFAP and I6 inherit the NeuN pose. Their own depth profiles were flat.

### H20.33.043 (second donor)

- **Data-preferred placement.** Gray/white strongly prefers the superior temporal plane:
  - Heschl's gyrus / planum polare 45–52 %, STG 19 %, posterior insula 12–17 %, MTG 1 %;
  - y −13 … −16.
- **Admissible placement.** The landmark-slab and region priors give a placement on MTG at
  y −24.5. It is a stable local optimum of the data objective, but every geometric diagnostic
  prefers the excluded placement:
  - exposed pia → atlas surface: 1.0 vs 2.5 mm;
  - gray/white → atlas gray/white: 0.5–1.0 vs 1.0–1.1 mm;
  - freeing tilt improves the excluded placement by up to 5.9 δ.
- **Why the atlas cannot decide.** About 43 % of this section's pia lines a deep sulcus, which
  the population atlas does not represent; AHRAv2 has no open STS at the STG/MTG label edge.
  The atlas-to-section objective cannot certify the MTG placement, and it rules it out only
  weakly.
- **Status: unresolved.** The MTG placement is the anatomically admissible candidate, but it
  rests on the priors, not on image geometry. This waits for donor-specific MRI.

### HALO MTG supervision of the stacking coordinate (xIV experiment, 2026-11-18)

The question was whether a positive-only HALO MTG varifold term in xIV similarity fits
improves depth (`scripts/estimate_qnp_stacking_xiv.py`; 2,316 fits; ranked by the common
score). As configured, it does not:

- **It fixes the gyrus, not the depth.** Atlas MTG spans the whole AP range, so R_anat is
  nearly flat (≈ 0.2–0.6 over ~85 mm).
- **It distorts scale.** Uncensored and under a similarity, it shrinks the atlas (median xIV
  scale 0.85 / 0.72). It places 27–29 % of the annotated tissue outside the brain on
  H20.33.043.
- **The baseline already agrees.** The structural baseline reproduces the GPU stacking result
  with a different fitter: −27.5 / −27.5 / −28.0 on H20.33.034 and −16 / −14 / −13 on
  H20.33.043.
- **Use.** Do not use the term to pick depth. Rank xIV candidates by the common score and
  report the scale filter.
- **Report.** `archive/section_mapping/qnp_stacking_xiv_anatomical.md`.

## 3. Calibration evidence (reference donor H20.33.034, MTG block A7)

The reference case is six QNP stains against AHRAv2 at 500 µm, left hemisphere. It is one
donor only, so treat these numbers as calibrations, not laws.

### Particle count and compression
- **Section ↔ section rigid:** all 48 fits from N≈18 to 450 converge to the same pose after a 500 µm refinement (within 15 µm). Before refinement, joint compression at N≈120 lands closer than block binning at N≈450. HALO ribbon Dice at the coarse stage, joint vs binning: ASYN 0.956 vs 0.925, I6 0.928 vs 0.895.
- **Atlas plane ranking vs section N:** Spearman against the reference scan is 0.02–0.37 at N≤60, 0.53–0.66 at N≈120, and 0.91–0.92 at N≈200, where both candidate basins are in the top 2. At N≤120 the ranking is biased toward the coronal basin.
- **xIV:** compressing the atlas source (2,833 → 741 or fewer particles) cuts MTG landing from 85% to 76–78%, near rigid (74%). Compressing the section target to 148–251 particles keeps MTG at 82% (0.19–0.24 mm from the dense fit). But one jointly compressed target at 538 particles degraded (MTG 76%, 0.50 mm off), so the result is not monotone in N. Wall time only falls from 145 s to 124–138 s.
- **Resolution escalation inside the window (coronal, anchors):** the 0.018 window narrows from 1.5 to 1.0 mm going ~200 → ~450 section particles. Full resolution (~1,700) adds nothing. Native atlas voxels (~10,000 particles instead of ~2,500) change the scores by < 0.003 and leave the window unchanged.

### Feature matching (source = moving, target = fixed; π is d_source × d_target)

| Case | Source → target | Modality | d_source → d_target | Correspondence | Outcome |
|---|---|---|---|---|---|
| Section ↔ section rigid | stain → NeuN | single (tissue occupancy) | 1 → 1 | identity | HALO ASYN 0.980 / I6 0.932 |
| same | stain → NeuN | treated as single (gray/white surrogate) | 2 → 2 | identity | 0.952 / 0.943 |
| same | stain → NeuN | cross | 2 → 2 | learned π (closed-form NNLS per pose) | 0.956 / 0.942: no gain over identity |
| same | stain → NeuN | cross (per-stain k-means in OD-RGB; indices arbitrary) | K → K, K = 2–6 | learned π | ≈0.973 / ≈0.92: ≈ tissue only |
| same | stain → NeuN | cross (stain-specific OD bins) | 8 → 8 | identity | **fails**: I6 0.30, AT8 and LFB 10–23 mm off, LFB flipped |
| same | stain → NeuN | cross | 8 → 8 | learned π | 0.982 / 0.935: rescued, no gain |
| Atlas plane → section (scan, slabs, profiles, native) | AHRAv2 labels → QNP | cross, fixed a priori by label table | 2 → 2 (tissue: 1 → 1) | identity | gray/white localizes; tissue-only leaves 72% of planes within δ |
| Atlas plane → section | AHRAv2 → QNP | cross | 2 → K (K = 2–6), 2 → 8 OD | learned π 2 × d_target | inconsistent across stains (fixed-pose test only) |
| xIV deformable | AHRAv2 plane → QNP consensus | cross | 2 → 2 | learned π_ST | near-diagonal (row-normalised 0.94 / 0.06, 0.04 / 0.96): identity was nearly right |

Apart from the identity failure on OD bins, the section ↔ section differences are ≤ 0.03 Dice.

**Plane contrast vs feature setup** (atlas → section, all 297 planes at fixed gray/white-fitted poses; contrast = (median − best)/median, NeuN / LFB / AT8):

| Setup | NeuN | LFB | AT8 |
|---|---|---|---|
| 2 → 2 identity | 0.39 | 0.30 | 0.26 |
| 2 → 2 learned | 0.36 | 0.32 | 0.29 |
| 2 → K learned, K = 2 | 0.35 | 0.32 | 0.27 |
| 2 → K learned, K = 6 | 0.29 | 0.13 | 0.21 |
| 2 → 8 OD learned | 0.33 | 0.18 | 0.22 |

Learned π did not flatten the search at 2 → 2. Contrast falls as target channels outgrow what a 2-class template can explain. The best plane per stain moved between MTG and non-MTG planes. The poses were fitted on gray/white, which favours those variants, so this is suggestive only.

**Roles in all runs:** template = atlas (source), target = QNP. The rigid scorer weights atlas mass by the section's density footprint. xIV's `LossVarifoldNormBoundary_Partial` weights deformed atlas particles by min-max-normalised target density. In 2-D, xmodmap slab normals are zero, so λ (the learnable support parameter) is inert and held fixed (framework script docstring).
- **Gray/white map QC (Otsu separability):** NeuN 0.85, LFB 0.84, AT8 0.79 (the anchors). ASYN 0.57 (scanner stripes appear only in the derived map), I6 0.66 (layer 1 read as white), GFAP 0.70 (glial). The passengers' depth profiles were flat, and their best fits landed anywhere from y +2 to −58.

### Plane, tilt, slabs
- **Slab pass** (28 slabs × 6 sections): 3.5 min on 8 CPU workers. AT8 and NeuN show a clear minimum at the y −28 slab (0.15 vs 0.18–0.27). LFB is flat (0.13–0.15).
- **Free-tilt continuous fits:** the anchors land at y −27.6 to −27.9 (spread 0.26 mm). Forcing a shared tilt costs ≤ 0.016.
- **Coronal vs free tilt at full resolution:** costs +0.004 (AT8), +0.005 (NeuN), +0.012 (LFB), all < δ, and the depth is unchanged.
- **Oblique candidate** (pitch/yaw ≈ 13°/13°, y −62): ties AT8 and LFB at full resolution, loses NeuN (0.205 vs 0.158), and puts 0% of the gray matter on MTG (~60% fusiform). Coronal slabs never ranked it.
- **Sub-voxel depth** (0.125 mm, linear in depth): minima at −27.875 / −27.625 to −27.75 / −27.75, flat within ±0.25 mm (Δscore < 0.001). Interpolated planes score slightly better than voxel planes, which suggests a blur effect.

### Stacking
- Section-to-section outline residuals give the order AT8 → NeuN → ASYN/GFAP → I6, which matches the scan dates (a weak, independent signal). Adjacent sections differ by 0.06–0.14 mm at the boundary, so the whole stack is likely < 1 mm thick, below the atlas window (~1 mm).
- **Result:** block at coronal y ≈ −27.75 ± ~0.5 mm. The order and spacing of individual sections cannot be resolved from the atlas.

### xIV deformable step (basin A, start `r000_f0`, 2026-11 upstream)
- `affine+diffeo` on bcharlier `unpack` 67a7538 matches the legacy Hamiltonian, shooting and pull-back to ≤ 2e-14 (float64 probe).
- **float32:** the loss trajectories match legacy to 4 decimals up to closure 80. Then the new tree stops on a plateau at native loss 7.66, where legacy, by roundoff luck, went on to 5.32. The total-mass score is the same (0.0109 vs 0.0110). The class score is worse (0.041 vs 0.027). Mass-ratio p50 is 1.033 for both, hull ratio 1.06 for both. The legacy optimizer on the new tree stalls identically, so this is roundoff, not the line search.
- **float64:** passes the plateau (7.66 → 5.73 → 5.32 → 4.53 → 3.83 …). Final numbers: `docs/software/xiv_lddmm_upstream.md`.
- **Staged** rigid(30) → affine(30) → affine+diffeo(150): native loss 17.02, total-mass 0.0217, class 0.0944, mass-ratio and hull ≈ 1.000. This is near-rigid: after the rigid stage, the deformable stage's first step was dominated by π_ST and took no real step. The `quadratic_approx` line search ended the same way (17.01).

- **Tuning (float32; details in `docs/software/xiv_lddmm_upstream.md`):**
  - Larger kernels (RKHS 8/4/2 mm, varifold 8/4/2/1 mm) plus `c_nonrigid` 1 is the only change that improved both scores (class 0.0405 → 0.0257).
  - `gamma` 0.01 and staging added nothing; cropping the atlas to the section footprint + 3 mm changed nothing.
  - An atlas pre-shrunk to 0.85 was re-expanded to net ~1.03×, so the data do not support a smaller atlas.
  - All fits remain visually poor, with compression ridges at sulcal notches the smooth atlas plane lacks.
  - `dens_sigma_mm` 0.5 (support at particle spacing) is best so far: total-mass 0.0102 → 0.0068, class 0.0257 → 0.0201. Atlas tissue in notches is censored instead of compressed.
  - Reversed roles (QNP template, cropped atlas target): the section expands into atlas tissue the block lacks. Its mass grows with area, which is correct, but the distortion then sits on the data.
  - Caveat: these scores used each run's own support, which favours small `dens_sigma`. Scores now use a fixed 0.5 mm section footprint.

- **Pial correspondence (wide 6 mm atlas window):** median distance from section pia to atlas pia was 3.49 mm at rigid placement. It became 5.21 mm with support 0.5 mm, 3.02 mm with 2 mm, and 2.51 mm with 4 mm. The 90th percentile stays ~9 mm because the section's sulci have no atlas counterpart. Support 4 mm matches the lateral surface and gyral crown, but compresses tissue elsewhere (mass-ratio p5 0.11).

### Traps seen
- The saved GFAP→NeuN pose was mirrored (192°, flip). A global search gives 184°, unflipped, scale 0.995, with a 6× margin. The 0.85 scale at the bound was the symptom.
- Warm-starting a coronal profile from a tilted pose produced a false, jagged LFB profile; a fresh search fixed it.
- The AT8 and GFAP HALO annotations do not fit their own slides.
- The xIV run needs `--stage support-check` before `single`.
