# Fresh-chat prompt: `feat/landmark-varifold` dev branch on the xIV-LDDMM fork (featured landmarks + drawing)

Continue in `/cis/home/dpadova/Documents/git/ad-resilience`. Read these first:
- this file;
- `docs/software/xiv_lddmm_upstream.md`: the submodule pin, remotes, known issues and contribution workflow;
- the scratch prototype this branch ports (do not redesign it):
  - `~/tmp/qnp-sts-landmark/xiv_landmarks.py`, the implementation;
  - `~/tmp/qnp-sts-landmark/tests/test_xmodmap_landmarks.py`, 7 passing tests;
  - `~/tmp/qnp-sts-landmark/notes/landmark_term_design.md`, the math and options A/B/C;
  - `~/tmp/qnp-sts-landmark/REPORT.md`, the full-scale results and code status.

## Why this chat exists

A landmark term was prototyped project-side for the atlas → section fit (H20.33.034 MTG, the STS fundus). It worked as
designed:
- one STS-fundus landmark, end of the atlas label edge, σ 16/8/4 mm, weight 0.3× the starting data loss;
- the fundi went from 7.9 mm apart to 0.34 mm;
- tissue and gray/white mismatch stayed at the no-landmark level (within noise);
- nothing folded.

The user wants it as a feature of the fork: a dev branch on `dmp2/xIV-LDDMM-Particle` that adds **featured landmarks**
(points carrying a landmark-id feature) to the registration objective, and **draws them** in the registration figures.

## The design (decided by the user, 2026-11-16: option A)

A separate point-feature varifold on landmark Diracs, outside the data term:

E_L = w_L Σ_j β_j · ½ ‖μ_a − ν_b‖²_{σ_j},  with  k_σ((x, e), (y, f)) = exp(−|x − y|²/2σ²) ⟨e, f⟩
- μ_a = Σ_k c_k δ_{φ(a_k)} ⊗ e_{id_k}: source landmarks, transported passively by the flow (`ShootingGrid`;
  no momenta, so H is unchanged).
- ν_b = Σ_k c_k δ_{b_k} ⊗ e_{id_k}: target landmarks, fixed.
- Ids are one-hot, so different ids never attract. Points sharing an id match as a set, with no ordering needed.
  Mass is normalised to 1 per id on each side.
- β_j = 1, fixed. **Not** `set_normalization`, which divides by the starting discrepancy.
- No π, no support censoring, no density weight.
- Closed form for one landmark per id: E = w_L Σ_j β_j (1 − exp(−d²/2σ_j²)). It is quadratic near 0 and bounded far
  away (a robust loss). Loss and gradient are exactly 0 at coincidence.
- Weight: the project used w_L = ratio × D0 / E0, with D0 and E0 the data and unit landmark losses at the start.
  Keep that normalisation **project-side**: upstream takes a plain `weight`.

## What I want on the branch

1. **Loss.** `xmodmap/distance/LossLandmarkVarifold.py`.
   - Port `LandmarkVarifold`; make it dimension-agnostic (2-D and 3-D).
   - Dense torch is fine for N ≲ 10³; say whether a KeOps path is worth adding.
   - Keep `get_params()` consistent with the other losses (`check_resume` compares them).
2. **Model.** Add optional `landmarkLoss=None, source_landmarks=None` to `CrossModalityBoundary.__init__`.
   - With `landmarkLoss=None`, or weight 0: use plain `Shooting`, so the fit is **bitwise identical** to today's.
     The prototype used `ShootingGrid` even at weight 0. That changed the float32 summation order, and the full-scale
     fit diverged from closure 9 on. A repeated baseline is bitwise deterministic, so this is the only source.
   - With landmarks: `ShootingGrid` carries them, and the landmark term is added.
   - **Decide with me:** return 4 terms when there are no landmarks and 5 with them, or always 5? `print_log`,
     `check_resume` and the project's `scores.json` read the tuple by position.
   - Keep the `loss` signature, so `set_precond` and the precond keys are unchanged. Landmarks are attributes, not
     optimised variables.
3. **Drawing.**
   - `xmodmap/plot/plot2d.py::plot_registration_2d` and `plot3d.py::plot_registration_3d` gain optional
     `landmarks_S`, `landmarks_T`, `landmark_ids` and a landmark trajectory (the `ShootingGrid` states).
   - Draw source landmarks on the source panel, target landmarks on the target panel, and both on the deformed panel
     at time t (the slider moves them).
   - One marker shape per id, ink colour with a white halo, and a legend entry per id. Optionally a thin path from
     start to t on the deformed panel.
   - With no landmarks, the output is unchanged.
4. **Tests.**
   - Port the 7 scratch tests: closed form, no attraction across ids, zero loss and gradient at coincidence, weight-0
     bitwise reproduction, transport of a landmark sitting on a particle (two stages), and pull toward the target.
   - Add a 3-D case, an N > 1 case with mixed ids, and a smoke test that the plots render with landmarks
     (Agg / off-screen).
   - Follow the submodule's test layout (`tests/`, `unittests/`).
5. **Project side** (after the branch works; do not edit the superproject without asking):
   - `src/scripts/framework_script_AtlasToSectionCrossModalBoundary.py` gains an optional `model.landmarks` block:
     `npz`, keys, ids, `sigma_mm`, `ratio` or `weight`.
   - Outputs: `landmarks.json` (start and end residual in mm, per id).
   - `view_xiv_trajectory.write_trajectory` / `render_registration` carry the landmark frames, so
     `registration_t1.00.png` shows them.
   - The scratch `run_fit.py` / `xiv_landmarks.py` show the wiring.

## Verification

- Every new and existing test passes, and `src/tests/test_xmodmap_stages.py` still passes.
- A no-landmark run of the QNP baseline is bitwise identical before and after the branch (`fits/flipped_y-28.0/H_baseline`
  in the scratch dir: 224 closures, tissue 0.0567, gray/white 0.0983).
- The scratch landmark fit, re-run through the branch, reproduces the prototype within the noise floor
  (tissue ±0.0015, gray/white ±0.002): flipped y −28, definition b, σ 16/8/4, ratio 0.3 gave residual 0.34 mm,
  tissue 0.0570, gray/white 0.0999.
  - The inputs are in `~/tmp/qnp-sts-landmark/inputs/flipped_y-28.0/`, which has `landmarks.npz`.
  - This pose is now only a test case: the running pose for H20.33.034 is the current one.
- The registration figure shows the landmarks in all three panels; look at it.

## Git workflow (`docs/software/xiv_lddmm_upstream.md`, "Contributing")

- In `vendor/xIV-LDDMM-Particle`, branch `feat/landmark-varifold` from **`origin/unpack` (67a7538)**, not from the
  local fix branches. Remotes: `origin` is bcharlier (upstream of record), `fork` is dmp2.
- **Trap:** at 67a7538, `import xmodmap` fails on torch 2.7.1 (`_to_scalar`), and `CrossModalityBoundary` skips
  `Model.__init__`. The project bridges both in `src/preprocess/xmodmap_compat.py`; see the fix branches
  `fix/torch-to-scalar-fallback` and `fix/boundary-model-init`. Run tests through `load_xmodmap` / `instantiate_model`,
  or say how the branch should depend on the fixes. Do not merge the fix branches into the feature branch without asking.
- Commit on the feature branch in the submodule only.
- **Do not push, open a PR, or move the superproject's submodule pointer until I ask.**
- The superproject working tree holds uncommitted work from several tasks; leave it alone.

## Out of scope (note, don't do)

- xmodmap's stopping rule ("local minimum" when two L-BFGS steps start at the same loss) ended one landmark fit at
  step 6 with data loss 13.7. Upstream issue #5 covers it. The scratch `lbfgs_restarts` option is untested; leave it
  for a separate branch or issue.
- Landmark weight normalisation (ratio × D0/E0) stays project-side.
- N > 1 anatomy (STS mouth, end of the HALO ribbon) and any pose decision.

## Standing preferences

- Do no harm: the feature must be inert when unused.
- Figures follow the `dataviz` skill: labelled axes with units, plain titles, minimal non-label text.
- float32 by default; GPU first.
- Scratch only under `~/tmp/<task>/`.

## Environment

- `/cis/home/dpadova/miniconda3/envs/ad-resilience-xiv/bin/python` (torch 2.7.1 cu128, pykeops 2.3,
  pyvista 0.49.2 / vtk 9.7.2), with `PYTHONPATH=src:src/config_setup:src/preprocess:src/scripts`.
- GPU: RTX 2080 SUPER.

## Open decisions (ask me)

- Return arity of `loss` (4/5 vs always 5).
- Dense torch vs KeOps for the landmark kernel.
- Whether drawing goes in the same branch or a second one.
- When to push and PR (to the fork only, or on to `bcharlier:unpack`).
