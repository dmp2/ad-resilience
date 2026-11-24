# Fresh-chat prompt: QNP MTG section pose and stacking, checked visually against AHRAv2 montages

Continue in `/cis/home/dpadova/Documents/git/ad-resilience`. Read this file, then
`.claude/skills/section-to-template-mapping/SKILL.md` (the mapping procedure and its rules)
and `docs/software/xiv_lddmm_upstream.md` (the xIV state, tuning and pial results). Do not
rebuild existing machinery.

## What I want

I doubt the global pose of section H20.33.034 MTG (block A7) in the atlas. With a wide
atlas field of view, the atlas plane shows a deep sulcus *superiorly* (higher z). The
section has a deep sulcus and a small tissue area. The section may need flipping about
the z-axis so that its deep sulcus also lies superiorly. A changed pose means a new
stacking-coordinate search (SKILL.md: never warm-start from an old model's pose).

I want to check this the way I would by hand, coarse to fine:

1. **Coarse montage.** Coronal atlas slices along the A-P extent of MTG, with the
   surrounding temporal cortex (STG, ITG, fusiform) and white matter colour-coded and
   labelled, in a wide field of view. Look at how the MTG shape changes along A-P, and
   where the section could sit.
2. **Block estimate.** Show the section beside or over those slices in candidate poses,
   including the current one and its z-flip. Decide visually whether the output makes
   sense before any finer comparison.
3. **Fine montage.** Within the chosen block, a finer atlas montage, for visual
   inspection of the section's placement in atlas coordinates.
4. Only then, the finer stacking-coordinate estimate and the deformable (xIV) step.

The aim is a better picture of the overall MTG shape and of the section's placement, so
that automated search results can be judged against it.

## Standing preferences

- **Do no harm** (see memory and SKILL.md). Coronal planes unless tilt is justified;
  report ties as ranges; apply priors (e.g. "block dissected as MTG") explicitly as priors.
- **Features: labels and clusters only** (gray/white, atlas region labels). No continuous
  feature fields such as cortical depth.
- **Figures:** every axis labelled with units; plain panel titles; minimal text that is
  not a plot label. In sweeps, panel titles name the setting that changed, in words.
  Follow the `dataviz` skill (fixed categorical colour order).
- **Scratch work only under `~/tmp/<task>/`** (e.g. `/cis/home/dpadova/tmp/qnp-pose-montage/`),
  never system `/tmp`. The previous task's scratch is in `~/tmp/xiv-lddmm-migration/`.
- GPU first; xIV in **float32**.
- Do not commit, and do not push or open PRs anywhere, unless I ask. The working tree
  holds much uncommitted work from several tasks; leave it alone.

## Environment

- Python: `/cis/home/dpadova/miniconda3/envs/ad-resilience-xiv/bin/python` (torch 2.7.1
  cu128, pykeops 2.3, pyvista 0.49.2). GPU: RTX 2080 SUPER (FP64 is 1/32 rate).
- `PYTHONPATH=src:src/config_setup` (tests also need `src/preprocess`).
- xmodmap runs from the `vendor/xIV-LDDMM-Particle` submodule (bcharlier, branch `unpack`,
  67a7538), through `src/preprocess/xmodmap_compat.py::load_xmodmap`.

## Current state of the pose and stacking

- Rigid plane scan: `src/scripts/scan_atlas_planes_for_qnp.py` (job `H20.33.034_MTG` in
  `configs/project_config.yaml`). Results: `results/registrations/qnp_H20.33.034_MTG_atlas_plane_scan/`
  (`plane_scores.csv`, `REPORT.md`, `summary.json`). Helpers there: `atlas_setup`,
  `plane_geometry`, `plane_particles` (reslice AHRAv2 into in-plane particles with coarse
  gray/white or fine region features).
- Best plane `p+000_w+000_y-027.5` (coronal, y = −27.5 mm): similarity θ 128.3°,
  **flip = True**, scale 1.092, score 0.1695.
  - At the neighbouring planes p−15 and w+15, the opposite flip scores 0.04–0.05 worse
    (δ ≈ 0.018). For this plane, `best_other_flip_score` was not recorded.
  - The z-flip question is a reflection plus rotation, so the scan's search did cover it.
    But the scan scored with footprint censoring on a 1 mm atlas plane, so it could not
    use pial or sulcal information. Treat the flip as unresolved.
- Basin B (p+15 w+15 y −62.5) puts the block on fusiform. The user has deprioritised it
  given the MTG dissection.
- Two-loop stacking (memory note `qnp-particle-mapping-guidance`): anchors AT/NeuN/LFB at
  y ≈ −27.75 ± ~0.5 mm; passengers ASYN/GFAP/I6 placed via QNP↔QNP fits.
- Atlas labels: AHRAv2 compressed codes via HOMBA terminology. MTG = 30, STG = 29,
  ITG = 31, FuG = 2032; forebrain white matter = 220. Code 34 is planum temporale, not
  STG. The plane scan's `fine_regions` block in the config lists the regions in use.
- The section's HALO cortical ribbon (depth from pia) is at
  `data/derivatives/sea-ad/qnp/registration_inputs/H20.33.034_MTG/p+000_w+000_y-027.5/qnp_ribbon_points_in_plane.npz`.
  Points with depth ≤ 100 µm mark the section's pia.

## What the last chat established (xIV step; details in docs/software/xiv_lddmm_upstream.md)

- Default xIV model is variant H: RKHS 8/4/2 mm, varifold 8/4/2/1 mm, `c_nonrigid` 1,
  `gamma` 0.1, `dens_sigma_mm` 0.5, one `affine+diffeo` stage.
- The default export crops the atlas plane to the section footprint (`support_sigma_mm` 2),
  which removes the atlas pia. With 6 mm, the window keeps it (scratch export in
  `~/tmp/xiv-lddmm-migration/pia/inputs/`). **Use a wide field of view for any visual
  judgement.**
- At the current rigid pose the white-matter bands line up, and the section's right edge
  is a block cut. But the section's upward white-matter finger has no atlas counterpart
  at this plane.
- Pial correspondence (section pia → nearest atlas pia, median): 3.5 mm rigid. It is
  2.5 mm with support 4 mm, at the cost of strong compression. The section's sulci have
  no atlas pia within reach, either because the population atlas lacks them or because
  the pose is wrong. That ambiguity is this chat's question.
- Each fit writes `registration_t1.00.png` (labelled source | target | deformed with
  the target outline) and `qc_warp.png`. Scores are "tissue mismatch" and
  "gray/white mismatch" (fixed 0.5 mm section footprint; 0 = identical).
- Useful scratch scripts to adapt: `~/tmp/xiv-lddmm-migration/sweep_figure.py`
  (small multiples), `pia/rigid_pose_figure.py` (pose over the wide atlas window),
  `pia/pial_distance.py` and `pia/pia_figure.py` (pial correspondence).

## Open decisions from the last chat (ask me; do not decide)

- Default export window (`support_sigma_mm` 6?) and default xIV support width
  (2–4 mm sweep on the wide window was proposed).
- Whether to commit the xIV migration work. It is uncommitted; nothing went to bcharlier.
  Fix branches are on the fork `dmp2/xIV-LDDMM-Particle`; no PRs are open.

## Suggested first steps

1. Find MTG's A-P extent in AHRAv2 (code 30, left hemisphere). Build the coarse coronal
   montage (e.g. every 5 mm) with MTG/STG/ITG/FuG/white matter colour-coded, a wide
   FOV, axes in mm, and the y value as each panel's title.
2. Add the section in its current pose and its z-flip as an inset or overlay at the
   current y, and show me before going further.
3. After I pick a block, make the finer montage (e.g. 1 mm) and only then rerun the
   rigid search for the chosen pose family, and the stacking procedure in SKILL.md.
