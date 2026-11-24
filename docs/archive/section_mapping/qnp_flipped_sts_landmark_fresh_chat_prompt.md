# Fresh-chat prompt: flipped (STS) pose of QNP H20.33.034 MTG, fit with an STS-fundus landmark

Continue in `/cis/home/dpadova/Documents/git/ad-resilience`. Read these first:
- this file;
- `.claude/skills/section-to-template-mapping/SKILL.md` (the mapping procedure and its rules);
- `docs/software/xiv_lddmm_upstream.md` (the xIV state, tuning and pial results);
- the frozen report `~/tmp/qnp-pose-montage/REPORT.md` (the pose evidence so far).

The earlier prompt `docs/qnp_section_pose_montage_fresh_chat_prompt.md` covers the background. Do not rebuild existing machinery.

## Why this chat exists

Section H20.33.034 MTG (block A7) has two candidate poses in AHRAv2. Both are coronal, at y ≈ −28 mm.

- **Current** (flip True, θ ≈ 128°). The section's deep sulcus sits on the ITS and the cut lobe on ITG. Gray/white varifold score is better: consensus 0.158 vs 0.227 at 500 µm, 3.2δ (δ = 0.0217). NeuN and AT8 are also beyond δ; LFB is tied.
- **Flipped** (flip False, θ ≈ 206°). The deep sulcus sits on the STS, the pial lobe fills MTG, and the cut lobe lands on STG. Section pia lies much closer to atlas surfaces: median 0.5–0.7 mm vs 2.5 mm.

The user now favours the flipped pose, on two priors. Apply them explicitly as priors; neither is image evidence.
1. **Anatomy.** The STS is the only deep temporal sulcus with MTG on one bank. The HALO cortical layers run along one bank of the section's deep sulcus and stop at its fundus.
2. **Sampling protocol.** Latimer, C.S. et al. (2023), *Protocol for the Systematic Fixation, Circuit-Based Sampling, and Qualitative and Quantitative Neuropathological Analysis of Human Brain Tissue*, Methods Mol Biol 2561, doi:10.1007/978-1-0716-2655-9_1, Fig. 3. As the user reads it, the "MTG" QNP block spans the superior and middle temporal gyri. Posterior STG is possible if connectivity-guided sampling was used, but the user thinks this unlikely. The figure has not been checked in a chat, so cite it as the user's reading.

The rigid gray/white scorer is blind to pia (it censors to the section footprint and uses gm/wm labels only). The next test is therefore a deformable xIV fit of the flipped pose, guided by a landmark: the STS fundus in both atlas and section.

## What I want

1. **One landmark first (N = 1): the STS (STG–MTG sulcus) fundus.**
   - Define it in the section and in the atlas plane.
   - Add it to the xIV objective.
   - Fit the flipped pose. Compare against:
     - the same model without the landmark;
     - the current pose with the same model (no landmark, since the current pose has no STS correspondence).
2. **Design the landmark term so that it generalises to N landmarks**, each carrying an identifier feature. The user's suggestion is a point-feature varifold: landmark particles with a one-hot "fundus" (or landmark-id) feature channel. This will likely need changes to the xmodmap objective and its input arguments.
3. **Then N > 1** (later, after I review N = 1). Candidates: the STS mouth or MTG crown, and the end of the HALO ribbon.

The analysis in `~/tmp/qnp-pose-montage/` is **frozen**: read-only, with checksums in `MANIFEST.sha256`. Copy what you need into a new scratch dir, `/cis/home/dpadova/tmp/qnp-sts-landmark/`, and never edit the frozen files. We will later port the relevant code into this repo for repeated production use and QC. So write scratch code ready for that port:
- put the functions apart from the CLI;
- keep paths in one config block;
- give each output a stated provenance (inputs, pose and settings).

Its REPORT.md lists the porting candidates.

## The landmark: definitions and the known trap

**Section fundus.** Use `sulcus_evidence.section_sulcus_geometry(P, pia)` from the frozen dir:
- pia is the HALO ribbon at depth ≤ 100 µm;
- the deep sulcus is pia more than 2.5 mm inside the convex hull;
- the mouth is the nearest rim pia point;
- the fundus is the farthest point along the pia (1 mm graph links), 19.0 mm along the pia at scale 1.092.

A second, independent definition is where the HALO layer labels end. Report both, and use their disagreement (in mm) as the section landmark's uncertainty.

**Atlas fundus: the trap.** In AHRAv2 the STS is a one-voxel unlabelled cleft for only about 10 mm in from the lateral surface. Deeper than that it is only the MTG(30)|STG(29) label edge, and AHRAv2's gyral boundaries include straight plane cuts. By the label boundary, atlas STS depth is 23–28 mm along y −20…−50. So the atlas fundus is ill-defined. Candidates:
- (a) the end of the unlabelled cleft;
- (b) the deepest point of the MTG|STG boundary that is still gray matter;
- (c) where that boundary meets white matter (code 220 and the listed white-matter codes).

Show these candidates on the coronal slice at the fitted y, using a wide field of view, and **ask me which to use** before fitting. The fundus is a line along y in 3-D, so at each candidate y the landmark is one point in that plane.

## Design questions for the xIV landmark term (propose; I decide before the submodule changes)

The relevant code is in `vendor/xIV-LDDMM-Particle/xmodmap/`:
- `distance/LossVarifoldNorm.py`: a Gaussian-linear kernel with feature dot product; the `beta` per-scale normalisation is set by `set_normalization`.
- `distance/LossVarifoldNormBoundary*.py`: support censoring (`supportWeight`) and density weighting.
- `model/CrossModalityBoundary.py`: `loss(px, pw, qx, qw, pi_ST, zeta_S, lamb)` returns the H, data, π and λ terms. `Model.set_precond` asserts that the precond keys match the `loss` signature, so adding variables touches both.
- `deformation/Shooting.py`: `ShootingGrid(px0, pw0, qx0, qw0, qGrid, qGridw)` transports extra points through the flow without giving them momenta. This is the natural way to carry source landmarks.

The project wiring is `src/scripts/framework_script_AtlasToSectionCrossModalBoundary.py`. Its pieces:
- `build_model`, with the data loss at about line 202;
- `run_stages`;
- `pull_back`.

Its imports go through `src/preprocess/xmodmap_compat.py::load_xmodmap`.

Options to lay out, with a recommendation:

- **(A) A separate landmark varifold term.** A second `LossVarifoldNorm` on the landmark measures alone:
  - one Dirac per landmark;
  - a one-hot identity feature, so the feature dot product matches landmark k only to landmark k;
  - its own σ_L and weight;
  - no support censoring and no π.
  For one landmark at a single scale, ‖δ_φ(a) − δ_b‖²_σ = 2(1 − exp(−|φ(a) − b|²/2σ²)). This is a robust (Welsch) loss that is quadratic near zero and saturates beyond about σ. So σ_L must exceed the initial landmark gap: the deep sulcus starts about 1.5–4 mm from the STS boundary after the rigid refine. Multiscale σ (e.g. 8/4/2 mm) gives a coarse-to-fine pull.
- **(B) Landmark channels appended to the main data varifold** (extra ζ columns on S and T). This is closer to the user's phrasing, but it has three problems:
  - π_ST would mix the landmark channel with tissue classes, so π needs a block structure;
  - support censoring and density weighting would act on the landmarks;
  - the per-scale `beta` normalisation would be shared with tissue mass.
- **(C) A plain quadratic landmark penalty** Σ_k w_k|φ(a_k) − b_k|². This is the simplest baseline, but it is unbounded and so less forgiving of a wrong landmark.

My lean is (A): it is the point-feature varifold the user described, kept out of π and censoring, and it reduces to (C) near convergence. But present it and wait for the user's choice.

Where the code goes: build it first as a project-side subclass or wrapper loaded through `xmodmap_compat`, without editing the submodule. Port it to a fork branch (e.g. `feat/landmark-varifold` on `dmp2/xIV-LDDMM-Particle`) only when I ask. Add tests in the style of `src/tests/test_xmodmap_stages.py`. For a landmark that already coincides, the loss and gradient must be zero. With the landmark weight at 0, the fit must reproduce the no-landmark run.

## Starting poses (500 µm local refine; frozen `rigid/hyp_*_500um.csv`)

Similarity2D, section local (`qnp_local_mm`, the NeuN slide frame) → plane: moving = s·R(θ)·F·(x − τ), with F = diag(1, −1) when flip is set. Plane frame: u = +x, v = +z, origin at the left-MTG centroid (−57.88, y, −11.58).

| Family | Representation | y (mm) | score | θ (°) | scale | flip | τ (mm) |
|---|---|---|---|---|---|---|---|
| flipped | consensus | −28.0 | 0.2265 | 205.9 | 1.090 | False | (13.63, 12.58) |
| flipped | NeuN | −23.0 | 0.2589 | 234.7 | 0.912 | False | (8.20, 12.66) |
| flipped | LFB | −28.0 | 0.1863 | 203.3 | 1.031 | False | (13.46, 12.61) |
| flipped | AT8 | −28.0 | 0.2641 | 206.3 | 1.120 | False | (13.77, 12.54) |
| current | consensus | −28.0 | 0.1578 | 127.6 | 1.087 | True | (15.93, 14.97) |

The flipped family's within-δ window is y −21…−28 (consensus); the current family's is −27…−28. Treat y as a range: fit at least at −28 and −23, and do not claim one y.

The repo's exporter (`src/scripts/scan_atlas_planes_for_qnp.py --stage export`) only knows the scan's own poses. Export the flipped pose with a scratch script that reuses its functions, writing to `~/tmp/qnp-sts-landmark/inputs/`. Both families must use the same export window. The decision on that window is still open (see below); `support_sigma_mm` 6 keeps the atlas pia and the STG side.

## Suggested first steps

1. Copy the frozen helpers you need. Make a figure of the flipped pose at y −28 and −23 over a wide atlas field of view, showing:
   - the section fundus (both definitions);
   - the three atlas-fundus candidates.

   Show it to me and ask which atlas definition to use.
2. Write a short design note for options A/B/C, with the math, the code locations and the tests. Ask me to choose.
3. After my choices:
   - export both poses;
   - run the no-landmark baselines (variant H);
   - run the N = 1 landmark fits, sweeping the landmark weight and σ_L as small multiples (panel titles name the setting in words).

   Report for each fit:
   - landmark residual (mm);
   - section pia → atlas pia (median and p90);
   - tissue and gray/white mismatch;
   - ribbon landing on MTG;
   - compression (mass-ratio p5);
   - the registration figure.

   Do no harm: the landmark must not buy its residual with folding or extreme compression. Say so if it does.
4. Stop and show me before N > 1, and before any rigid-search change (e.g. adding the landmark to the y profile).

## Standing preferences

- **Do no harm** (memory and SKILL.md). Coronal planes unless tilt is justified. Report ties as ranges. Apply priors explicitly as priors.
- **Features: labels and clusters only.** Gray/white, region labels and landmark identities are fine; continuous feature fields such as cortical depth are not.
- **Figures:**
  - every axis labelled with units;
  - plain panel titles;
  - minimal text that is not a plot label;
  - in sweeps, panel titles name the setting in words;
  - follow the `dataviz` skill.

  Fixed palette: MTG #2a78d6, STG #eb6834, ITG #1baf7a, FuG #eda100, white matter #c4c2b9, other gray matter #e6e4dc, pia #e34948.
- **Scratch only under `~/tmp/<task>/`**, never system `/tmp`.
- GPU first; xIV in **float32**. Judge fits from `registration_t1.00.png` first, scores second.
- Do not commit, push or open PRs anywhere unless I ask. The working tree, including the `vendor/xIV-LDDMM-Particle` submodule, holds uncommitted work from several tasks; leave it alone.

## Environment

- Python: `/cis/home/dpadova/miniconda3/envs/ad-resilience-xiv/bin/python` (torch 2.7.1 cu128, pykeops 2.3, pyvista 0.49.2). GPU: RTX 2080 SUPER (FP64 runs at 1/32 rate).
- `PYTHONPATH=src:src/config_setup:src/preprocess:src/scripts`.
- xmodmap runs from the `vendor/xIV-LDDMM-Particle` submodule (bcharlier, branch `unpack`, 67a7538).
- Atlas codes (AHRAv2 compressed, HOMBA terminology): MTG 30, STG 29, ITG 31, FuG 2032, forebrain white matter 220. Code 34 is planum temporale.
- Section inputs:
  - `data/derivatives/sea-ad/qnp/particles/middle-temporal-gyrus/H20.33.034/aligned/` (consensus and per-stain tissue classes at 1000um and 500um);
  - the HALO ribbon `data/derivatives/sea-ad/qnp/registration_inputs/H20.33.034_MTG/p+000_w+000_y-027.5/qnp_ribbon_points_in_plane.npz` (`depth_from_pia_um`; ≤ 100 µm marks pia).
- Earlier xIV scratch to adapt: `~/tmp/xiv-lddmm-migration/pia/` (wide-window export, `pial_distance.py`, `pia_figure.py`, `rigid_pose_figure.py`).

## Open decisions (ask me; do not decide)

- Atlas STS-fundus definition (a/b/c above), and the landmark-term design (A/B/C).
- Default export window (`support_sigma_mm` 6?) and xIV support width (a 2–4 mm sweep was proposed).
- Whether and when to port the frozen scratch code and the landmark term into the repo or the fork.
- Whether to commit the earlier xIV migration work.
