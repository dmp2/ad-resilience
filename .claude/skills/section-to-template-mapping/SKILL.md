---
name: section-to-template-mapping
description: Map 2-D tissue sections (QNP histology now, spatial-omics sections later) onto a 3-D template atlas such as AHRAv2. Covers the particle/varifold representation, locating each section's stacking coordinate, tie and prior rules, GPU scoring, and the hand-off to deformable xIV. Use when registering a new donor's sections to the atlas or estimating where sections sit in it.
---

# Section → template mapping

**Do no harm.** Add a parameter, feature or decision only when the data justify it. Otherwise
keep the simpler model and report a range.

| need | read |
|---|---|
| Method: math, conventions, decision rules | `docs/section_mapping.md` §1 |
| Commands, stages, outputs, QC | `docs/section_mapping.md` §2 |
| Where each operation lives in the code | `docs/section_mapping.md` §3 |
| Calibration numbers, donor results, regression references | `docs/section_mapping_findings.md` |
| xIV runtime, precision, kernel tuning, pial correspondence | `docs/software/xiv_lddmm_upstream.md` |
| Historical reports (old file names) | `docs/archive/section_mapping/` |

## Run it

The environment is `ad-resilience-xiv`. Set `PYTHONPATH=src:src/config_setup:src/preprocess:src/scripts`.
A new donor needs only `{donor, region}` under `qnp_section_inputs`.

```bash
C="--config configs/project_config.yaml --job <donor>_MTG"
python src/scripts/prepare_qnp_section_particles.py $C --stage inventory   # then --stage particles
python src/scripts/align_qnp_multistain.py $C
python src/scripts/map_qnp_sections.py $C --stage stack      # roles, compress, slabs, profiles, passengers, report
python src/scripts/map_qnp_sections.py $C --stage landmark   # optional LGN slab prior; then profile/report/candidates
                                                             # with --region-prior --landmark-slab
python src/scripts/map_qnp_sections.py $C --stage export --section NEUN   # stacking -> xIV inputs (gated)
# then a registrations.<name> record on the printed paths, framework_script_... --stage support-check / single,
# and map_qnp_sections.py --stage evaluate --section NEUN --plane <id> --fit-dir <init dir>
```

**Hand-off gate.** `export --section` exports automatically only an *eligible* placement. A
placement is not eligible when it has:

- a search-boundary flag;
- a non-contiguous window;
- anchors whose windows do not intersect;
- any `candidates.json` verdict other than "supported by image geometry";
- a prior run without its own `candidates.json`.

Otherwise the export refuses and lists why.

Eligibility only means nothing recorded contests the placement. It is **not** an established
anatomical correspondence. Report the record's `anatomical_correspondence` field alongside any fit.

Never work around a refusal silently. Choose `--candidate report|unconstrained|admissible|admissible-refined`
only as a stated decision, and report it.

- A passenger exports with its host anchor's pose. It is not warped on its own evidence: carry
  it through the anchor's map, or fit it rigid.
- Every export re-scores its pose and records the slide → local → plane composition and the
  mass scaling (`stacking_handoff` in the record).
- `export --plane` (tilted-plane scan table, consensus target) is the legacy path.

Library code is in `src/section_mapping/`:

- `measures`: representation;
- `atlas`: labels, planes, landmark and slab;
- `alignment` and `alignment_gpu`: score and search;
- `stacking`: profiles, δ, windows, decision rules.

Never reimplement these in a script.

## Defaults that matter (change only with evidence)

- **Roles:**
  - template (source, moved, censored) = atlas;
  - target (fixed, defines the support) = section;
  - section ↔ section: target = the anchor.
  - Never swap them.
- **Features:**
  - gray/white areas with an a-priori identity correspondence;
  - one-hot anatomical channels (`nu_Anat`) for xIV;
  - learned π only for arbitrarily indexed channels (OD bins, clusters, omics types).
- **Score:**
  - support-censored multiscale varifold, SDs 2/1/0.5 mm, normalised by the target self-term;
  - drop SDs < h/2 for compressed particles;
  - the support censors absent tissue, not wrong tissue.
- **Search:**
  - coronal planes (tilt 0), similarity + flip, scale 0.8–1.5;
  - fresh global search at every depth;
  - particles ~200 → ~450 → full 500 µm;
  - stop when the window stops shrinking.
- **Passengers** (dirty gray/white maps or flat profiles) inherit an anchor's pose; they are
  never placed on their own evidence.
- **xIV:**
  - one stage `affine+diffeo`, RKHS 8/4/2, varifold 8/4/2/1, `dens_sigma_mm` 0.5, float32;
  - source uncompressed;
  - run `--stage support-check` before `single`.

## Decision rules

- **δ:** the across-anchor spread of relative scores, re-estimated per case and level.
  Candidates within δ tie.
- **Distant candidates:** decide only at full resolution, never on compressed scores.
- **Model changes** (tilt, scale, features): rerun the global search. Never warm-start.
  Check both flips before inheriting a saved pose.
- **Priors** (`--region-prior`, `--landmark-slab`) are reported as priors:
  - always follow them with `--stage candidates` (data-only comparison);
  - read `windows_at_prior_bound`;
  - keep ITG in the region support.
- **HALO MTG supervision** fixes the gyrus, not the depth. Do not use it to pick depth.
- **Rank xIV fits** with `scores.json` or the common score (`alignment.CommonScore`), never the
  native loss. Judge `registration_t1.00.png` visually first.
- **H20.33.043 is unresolved** (prior-driven MTG at y −24.5 vs a data-preferred superior
  temporal plane). Do not try to resolve it without donor MRI.

## Report per section

- role (anchor or passenger);
- stacking coordinate and window;
- deciding level;
- in-plane transform;
- score;
- flags;
- landing fractions (QC only);
- priors used and their status.

For xIV, also report the model or stages, the dtype, `scores.json` and the stop reason.
