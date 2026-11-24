# Fresh-chat prompt: QNP stacking-coordinate estimation on a second SEA-AD donor (MTG), made reusable

Continue in `/cis/home/dpadova/Documents/git/ad-resilience`. Read these first:
- this file;
- `.claude/skills/section-to-template-mapping/SKILL.md` (the procedure, defaults and decision rules) and its `evidence.md`;
- the memories on the QNP bucket, particle mapping and do-no-harm (they load automatically).

## Why this chat exists

The full pipeline has run on one donor only, H20.33.034 (MTG, block A7). The defaults in the skill were
calibrated on that donor. Running the same procedure on a second donor tests whether they transfer.

**This donor is the first of many.** A second goal, as important as the result itself, is to make the procedure
reusable: run it per donor from configuration, with no per-donor code edits.
- Find every hard-coded donor or job name, path, block name, stain list and threshold you meet.
- Fix them in the repo code as you go. The audit is a deliverable (see "Reuse audit" below).

**Running pose for H20.33.034 (user's decision, 2026-11-16).** Use the **current** pose; the flipped STS
alternative is not pursued. The current pose is coronal, flip True, θ ≈ 128°, scale ≈ 1.09, at y −27.5 … −28.
The record is in `~/tmp/qnp-sts-landmark/REPORT.md`. Do not reopen it here.

## What I want

For one new donor's MTG QNP set:
1. download it with the existing tools;
2. make the original-scale particle representations and the compressed ones (image-varifold joint compression);
3. save the representation comparison figures (visual QC);
4. run the rigid/similarity stacking-coordinate estimation (skill procedure, steps 1–7), with its QC figures;
5. leave the workflow reusable for the next donor.

No xIV in this chat unless I ask.

## Orientation and reflections

**Within a donor: orientation must be consistent.**
- Every stain's section is brought into the NeuN slide frame (`qnp_local_mm`). Where HALO annotations exist and fit
  their slide, the section must also agree with them, so all sections share one orientation.
- The multistain alignment allows a reflection, because a section can be mounted face down. Verify each chosen
  reflection with an independent check: ribbon Dice against the NeuN annotations, and white-matter Dice.
- Never inherit a stored pose unchecked: on H20.33.034 a saved GFAP→NeuN pose was mirrored and wrong.

**Against the atlas: reflection is an ordinary nuisance parameter.**
- The global 2-D search includes both reflections among its FFT candidates, and the best pose wins, as in the
  original scan.
- The **per-reflection-family analysis is optional and off by default.** That analysis held the flip fixed and
  ran a separate depth profile per family, then reported the gap between families. It was built for the STS vs ITS
  question on H20.33.034, where the section orientation was unknown.
- Expose it as a driver option (e.g. `--families`), and run it only if I ask, or if the result looks mirrored.

## Steps, with stop points

**0. Choose the donor (stop and ask me).**
- List MTG donors with `python -m qnp.download --region middle-temporal-gyrus --dry-run` (env `wsi-pipeline`,
  `PYTHONPATH=src/seaad`). Do a full listing only one folder level at a time; see the bucket memory.
- Propose 2–3 candidates that have all 6 stains (NEUN, LFB, AT, I6, GFAP, ASYN), with their sizes. Ask me to pick.
- Download MTG only. The shared disk is about 90% full (9.6 TB free). H20.33.034 with all 4 regions was 36.6 GB.
  Ask before fetching other regions.

**1. Download and montage.**
- `python -m qnp.download --region middle-temporal-gyrus --donor <D>` writes to
  `data/raw/sea-ad/qnp/middle-temporal-gyrus/<D>/` and resumes on rerun.
- `python -m qnp.montage --region middle-temporal-gyrus --donor <D>`, then look at the montage.

**2. Config: per-donor block, generated, not hand-copied.**
- Add one new block, `qnp_section_inputs.<D>_MTG`, to `configs/project_config.yaml`. Change nothing else in that
  file: it holds other uncommitted work.
- The `H20.33.034_MTG` block mixes three kinds of field, and they should be separated:
  - **shared settings:** bins, alignment and search parameters, and the atlas codes;
  - **donor-derived values:** stains present, block names, `annotation_misfit`;
  - **paths built from the donor ID:** for example, every representation path in `atlas_scan`.
- Propose a structure that separates them: shared defaults plus a thin per-donor block, or a small generator that
  writes the block from the donor ID and the inventory. Ask me before restructuring the existing H20.33.034 block.
- Derive the donor-specific values from the slides. Never copy them. For example, H20.33.034's NeuN block was "A07"
  and the others "A7".

**3. Inventory (gate 1; stop and show me).**
- Run `src/scripts/prepare_qnp_section_particles.py --job <D>_MTG --stage inventory` (env `ad-resilience-xiv`,
  `PYTHONPATH=src:src/config_setup:src/preprocess`).
- Report:
  - which HALO annotations fit their own slide (on H20.33.034, GFAP and AT8 did not);
  - non-standard layer names;
  - slide units.
- Prefer computing the annotation-misfit list from the inventory's on-tissue fraction (`annotation_on_tissue_min`)
  over listing it by hand in config.

**4. Particles (gate 2).**
- Run `--stage particles`: 500 µm (and 1000 µm, if downstream code still needs it) tissue-class particles, OD bins,
  32 µm maps and ribbon points, with QC in `results/qc/qnp/<D>_MTG/`.
- **Skip the binned-particle figures** (`particles_<tag>.png`, written by `particle_qc_figure`): the image-varifold
  compression has replaced binning as the representation we look at. If the stage always writes them, add an option
  to turn them off.

**5. Multistain alignment (gate 3).**
- Run `src/scripts/align_qnp_multistain.py --job <D>_MTG`. It aligns every stain into `qnp_local_mm` (the NeuN frame).
- Check orientation consistency as above, and report the ribbon and white-matter Dice checks.

**6. Compression and comparison figures (stop and show me).**
- Joint compression with `preprocess.particle_approximation.optimize_particles` (positions and masses), once per
  section: h 1.5 mm (about 200 particles) and h 1.0 mm (about 450). Save the compressed archives for reuse by the
  search.
  - It runs on CPU in float64, with a pair guard; it took 5–25 s per section.
  - Note in the audit whether a GPU path is worth it.
- Comparison figures, on the model of the optimised variant `results/visualization_prototypes/qnp/iv_structural/figA_scale_space_H20.33.034_MTG.png`:
  - measured histology | p(WM) | compressed ~450 | compressed ~200;
  - **no binned panels.**
- Use `visualization.figures_qnp.figure_a(config, job, out_dir, representation=...)`.
- **Hard-coding to fix:** `src/visualization/cli.py` sets `QNP_JOB = "H20.33.034_MTG"`, and
  `representation_variants.py` imports it. Make the job an argument, keeping today's default so H20.33.034 outputs
  don't change.
- Write the figures with the donor in the file name, beside H20.33.034's.

**7. Feature QC → roles.**
- Make and look at each section's gray/white map.
- Anchor: Otsu separability ≳ 0.75, no scanner stripes, L1 not read as white matter, not a glial stain.
- Passenger: everything else, or a section whose depth profile is flat.
- On H20.33.034: anchors NeuN, LFB, AT8; passengers ASYN (stripes), I6 (L1 read as WM), GFAP (glial). Do not assume
  the same here.
- Compute these checks in code, so the next donor gets the same report automatically.

**8. Stacking coordinate (skill steps 3–7; GPU).**
- Fast pass: coronal slabs, then a depth profile per anchor, with a **fresh global 2-D search at every depth**
  (FFT candidates over rotations × scales × reflections).
- Then:
  - δ (across-anchor noise) and within-δ windows;
  - escalate ~200 → ~450 → full 500 µm, each level conditioned on the previous one (`extra_starts`);
  - passengers placed by a QNP↔QNP outline fit to an anchor.
- Tools: `preprocess/section_alignment_gpu.py` (`search_planes_gpu`, `refine_batch`, `TorchFootprintScorer`) and the
  `scan_atlas_planes_for_qnp.py` helpers (`atlas_setup`, `plane_geometry`, `plane_particles`, `landing_fractions`).
- **The repo has no driver for steps 3–5. Write one as reusable repo code**, configured by job name. For example,
  a `--stage stack` in `scan_atlas_planes_for_qnp.py`, or a new `src/scripts/stack_qnp_sections.py`; propose which.
  - Starting material: `rigid_families.py` and `rigid_summary.py` in `~/tmp/qnp-sts-landmark/frozen_copy/`
    (a writable copy; never edit `~/tmp/qnp-pose-montage/`, which is frozen).
  - Those scripts hard-code H20.33.034 paths, the stain list and the output dir. Generalise them.
  - The per-family option from "Orientation and reflections" lives here, off by default.
- Coronal planes only, unless freeing tilt improves every anchor by more than δ at full resolution.
- Stop escalating when the window stops shrinking. Do not claim per-slide z.
- Stop and show me before escalating to full resolution.
- **Regression check:** run the new driver on H20.33.034 too. It must reproduce that donor's window, y −27.5 … −28
  (flip True), within δ.

**9. QC figures and report.**
- Depth profiles per anchor, with the δ band.
- Best poses over a wide-FOV coronal atlas slice, showing:
  - section tissue outline;
  - gray/white boundary;
  - pia (ribbon depth ≤ 100 µm);
  - atlas classes in the fixed palette.
- Ribbon landing fractions.
- Per section, the skill's "Report per section" fields: role, stacking coordinate and window, the deciding level,
  transform, score, landing (QC only), and the priors used.
- Compare with H20.33.034's window. A second donor's MTG block need not sit at the same y.

## Reuse audit (deliverable)

Keep `docs/qnp_pipeline_reuse_audit.md` (or a scratch draft, if you prefer to show it first). For each hard-coded
or donor-specific assumption, record:
- where it is;
- what breaks for another donor;
- the fix made, or why it was deferred.

Already known:
- `QNP_JOB` in `src/visualization/cli.py`;
- the binned-particle figures in the particle stage;
- hand-listed `annotation_misfit`;
- block-name differences (A07 vs A7);
- the frozen scripts' hard-coded paths and stains;
- `scan_atlas_planes_for_qnp.py --stage export`, which only exports the scan's own poses;
- representation paths in `atlas_scan` spelled out per donor.

Add tests for the generalised pieces, in the style of the existing `src/tests/` (e.g. a job built from a synthetic
donor ID; the driver on a tiny synthetic section).

## Code changes and git

- Repo code changes are wanted here, but keep each change focused. Before you start, list the files you plan to
  touch.
- Several files already carry uncommitted work from other tasks: `git status` shows, among others,
  `src/seaad/qnp/montage.py`, `src/preprocess/xmodmap_compat.py` and the framework scripts. Before editing one of
  them, tell me, and don't disturb the existing changes.
- Every existing H20.33.034 output path and default must keep working.
- Exploratory or one-off work goes under `~/tmp/qnp-<D>-stacking/`.
- **Do not commit, push or open PRs unless I ask.** Leave the `vendor/xIV-LDDMM-Particle` submodule alone.

## Standing preferences

- **Do no harm.** Coronal unless tilt is justified. Report ties as ranges. Apply priors explicitly as priors.
  Decide between distant candidates only at full resolution.
- **Features: labels and clusters only** (gray/white, regions). No continuous fields.
- **Figures** (follow the `dataviz` skill):
  - every axis labelled with units;
  - plain panel titles; in sweeps, titles name the setting in words;
  - minimal non-label text.

  Fixed palette: MTG #2a78d6, STG #eb6834, ITG #1baf7a, FuG #eda100, white matter #c4c2b9, other gray matter #e6e4dc, pia #e34948.
- **Scratch only under `~/tmp/<task>/`**, never system `/tmp`. GPU first.

## Environment

- Particles, alignment, search: `/cis/home/dpadova/miniconda3/envs/ad-resilience-xiv/bin/python`
  (torch 2.7.1 cu128, pykeops 2.3, tifffile, imagecodecs, skimage), with
  `PYTHONPATH=src:src/config_setup:src/preprocess:src/scripts`. GPU: RTX 2080 SUPER.
- Download and montage: env `wsi-pipeline`, `PYTHONPATH=src/seaad`.
- Atlas: AHRAv2 compressed codes (HOMBA terminology): MTG 30, STG 29, ITG 31, FuG 2032, forebrain white matter 220,
  with the white-matter and drop code lists in `qnp_section_inputs.H20.33.034_MTG.atlas_scan`.

### Traps from H20.33.034

- `skimage.draw.polygon` hangs on HALO polygons; use PIL.
- Linear-OD Otsu fails on GFAP; use log-OD.
- LFB slides have no HALO masks or annotations.
- Some `.svs` files carry a glass-label series; don't render it.
- Warm-starting a depth profile from a neighbouring pose gave a false jagged profile; use a fresh search per depth.
- Compressed scores biased the ranking toward one basin, so decide between distant candidates at 500 µm.

## Open decisions (ask me; do not decide)

- Which donor; whether to download other regions.
- Config structure (shared defaults plus per-donor block, or a generator), and whether to touch the H20.33.034 block.
- Where the stacking driver lives (a new stage or a new script).
- Anchor and passenger roles, if the gray/white maps are borderline.
- Whether to escalate past ~450 particles, whether to free tilt, and whether to run the per-family reflection analysis.
