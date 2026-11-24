# Anatomical atlas features for QNP registration

The AHRAv2 atlas side of the QNP pipeline now has a canonical, one-hot anatomical
feature representation. The gray/white features it used before are a fixed projection
of it. **Nothing changes by default:** the plane scan, the stacking stages and xIV
still match gray/white. The new channels are carried along and can be selected
explicitly.

Code: [`src/preprocess/atlas_features.py`](../src/preprocess/atlas_features.py).
Wired into [`src/scripts/scan_atlas_planes_for_qnp.py`](../src/scripts/scan_atlas_planes_for_qnp.py)
(`atlas_setup`, `--stage export`, `--stage atlas-features`) and
[`src/scripts/framework_script_AtlasToSectionCrossModalBoundary.py`](../src/scripts/framework_script_AtlasToSectionCrossModalBoundary.py)
(`source_feature_projection`). Config: `qnp_section_defaults.atlas_scan.anatomical_features`.
Tests: [`src/tests/test_atlas_features.py`](../src/tests/test_atlas_features.py).

## Channels

| # | channel | AHRAv2 codes |
|---|---|---|
| 0 | `STG_gray_matter` | 29 |
| 1 | `MTG_gray_matter` | 30 |
| 2 | `ITG_gray_matter` | 31 |
| 3 | `other_gray_matter` | every other non-zero code that is neither white matter nor dropped |
| 4 | `white_matter` | `atlas_scan.white_matter_codes` (one shared channel) |

Dropped codes (ventricles, choroid plexus, `drop_codes`) and background belong to no
channel, as before.

- **The gray/white/dropped rule is unchanged.** The tables are built by the existing
  `section_alignment.lookup_tables`. A named region keeps only its gray codes, so a
  code listed both as a region and as white matter stays white.
- **The gray regions are configurable.** `anatomical_features.gray_regions` names
  entries of `fine_regions`, so their codes go through the existing terminology check.
- **Other gray matter is kept, not excluded.** It includes the fusiform gyrus, the
  insula, the opercula, Heschl's gyrus, planum temporale and polare, the temporal
  pole, the hippocampus, deep gray and cerebellar cortex. FuG stays a `fine_region`
  for landing fractions, but it is not its own anatomical channel.
- **The mass is area.** A particle's channel masses are the mm² of its pixels in each
  channel. At pixel level the vectors are one-hot, and their sum equals the old
  gray + white mass.

## Projections (fixed 0/1 matrices, nothing learned)

| name | output channels | matrix rows over (STG, MTG, ITG, other, WM) |
|---|---|---|
| `gmwm` (default) | gray_matter, white_matter | (1,1,1,1,0), (0,0,0,0,1) |
| `mtg_gm` | MTG_gray_matter | (0,1,0,0,0) |
| `anatomical` | all five | identity |

The projected masses are `nu @ M.T`.

## Where the representation flows

- **`atlas_setup`** builds `atlas["anatomical"]` next to `coarse` and `fine`. It
  refuses to run unless the `gmwm` projection equals the original gray/white tables
  for every label code. Every gray/white consumer (scan, stacking, GPU scorer)
  therefore stays exactly as it was.
- **`--stage export`** writes, in the source archive, `nu_Anat` (float32, one-hot
  areas, the same particles and support restriction as `nu_Sub`) and
  `anatomical_feature_names`. Metadata `anatomical_features` holds every projection
  matrix. `X`, `nu_Sub`, `nu_All` and all original metadata keys are unchanged. Export
  aborts if the `gmwm` projection of `nu_Anat` differs from `nu_Sub`.
- **xIV:** a registration record may set
  `source_feature_projection: gmwm | mtg_gm | anatomical`.
  - Unset (the default) is the old behaviour: `feature_key`, i.e. gray/white.
  - `gmwm` must equal `nu_Sub` exactly, or the run refuses.
  - The other two give a 1×2 or 5×2 `pi_ST` through the *existing* cross-modal term.
  - The objective is not changed, and no new learned mapping is introduced. The
    projection and the source feature names are written to `input_manifest.json`.
  - After a projection, source particles with zero projected mass are dropped. This
    is the pipeline's existing rule that a particle exists only where it carries
    mass. Without it, `mtg_gm` gives NaN from step 0, because makePQ's feature
    composition is 0/0. `gmwm` and `anatomical` drop nothing. With
    `center_mode: independent`, a dropped set moves the source centre; QNP records
    use `shared_target`, so their centre does not move.
- **Coordinates, weights and transforms:** nothing new. The channels come from the
  same nearest-neighbour reslice, plane basis, particles and support mask as the
  gray/white masses, so positions and total weights are identical by construction.
  The tests and checks below confirm this.

## Validation (H20.33.034 MTG, 2026-11-17)

`--stage atlas-features` writes the files below to `<scan output_dir>/atlas_features/`:

- `atlas_features.json` holds the checks.
- `anatomical_channels.png` shows the five channels on coronal y = −60, −45, −27.5,
  −15 and 0 mm.
- `feature_projections.png` shows the three projections at y = −27.5.

1. **The channels reproduce the segmentation.**
   - Over the whole 500 µm volume, each channel's voxel set equals an independent
     `np.isin` over the configured codes, with 0 voxels differing.
   - The STG, MTG and ITG voxel counts equal Allen's
     `ahrav2_gyral_parcellation_volumes_500um.csv` (226,006 / 415,406 / 262,774).
   - No non-zero, non-dropped voxel is unassigned.
   - The `gmwm` projection differs from the old gray/white class in 0 voxels.
2. **The plane particles are exact.** On all five figure planes (0.5 mm), the
   particle positions are identical across tables and every particle is one-hot. All
   three projections reproduce their reference (old gray/white, old `fine["MTG"]`,
   identity) with maximum absolute difference 0.0.
3. **The export is bitwise identical.** Plane A (`p+000_w+000_y-027.5`) was
   re-exported to scratch and compared array by array with the archive on disk:
   - Source, target and QC archives: every original array is identical in dtype and
     value, and every original metadata key is equal.
   - The only additions are `nu_Anat`, `anatomical_feature_names` and the
     `anatomical_features` metadata.
   - Source areas after support restriction: STG 24.75, MTG 314.5, ITG 164.0,
     other gray 1.0, WM 204.0 mm².
4. **The registration result is reproduced.** See "xIV reproduction" below.

### xIV reproduction

Setup:

- Registration `ahrav2_tissue_plane_to_qnp_H20.33.034_MTG_A`, init `r000_f0`, the
  config model (affine+diffeo, 150 steps, float32, GPU).
- Each run was support-check followed by single, each in a fresh process with the
  same call sequence (float32 fits depend on process history).
- Scratch: `~/tmp/atlas-anatomical-features/` (`scratch_config.yaml`, `fits/`,
  `compare_gmwm.json`, `reexport_comparison.json`).

| run | source features | result |
|---|---|---|
| baseline | archive on disk, `nu_Sub` (unchanged path) | tissue 0.055014, gray/white 0.080545, native loss 0.710433 |
| projected | re-exported archive, `source_feature_projection: gmwm` | **bitwise identical**: optimized variables, atlas and target deformation summaries, pulled-back ribbon points (max \|Δ\| = 0), π_ST, scores |

**Smoke runs.** These are 10 steps only: a feasibility check, not results, and not
comparable to the 150-step fits.

- `anatomical`: 5×2 π_ST, finite. MTG/ITG rows lean gray (0.95/0.07, 0.93/0.09) and
  WM leans white (0.02/0.98). The STG and other-gray rows stay at their initial value
  (0.33): they carry 25 and 1 mm² after support restriction.
- `mtg_gm`: 1×2 π_ST, finite, mass-ratio flag. A source with only MTG gray against
  the whole QNP section leaves most target tissue unexplained, so its tissue score of
  0.30 is expected.

## Unresolved atlas-label ambiguities

- **Superior temporal plane and pole.**
  - In the HOMBA terminology, TTG (33), planum temporale (34), planum polare (36) and
    temporal pole (35) are *siblings* of STG under the temporal lobe, not
    descendants. They are therefore in `other_gray_matter`.
  - Many anatomical conventions count the superior temporal plane as part of STG.
  - If a target's "STG" includes the plane, add those codes to `fine_regions.STG`.
    This is a user decision, and the terminology check will need a matching
    `expected_name` scheme.
- **Unnamed codes.** Codes 32 (42 voxels, beside FuG/ITG at y ≈ −45) and 248 (22
  voxels, y ≈ −80, between forebrain WM 220 and optic radiation 251) have no
  terminology row.
  - By the existing rule both are gray, so both go to `other_gray_matter`.
  - Code 248's neighbours suggest an unnamed white-matter tract. It is far from the
    MTG planes and was left as is to preserve gm/wm behaviour.
- **Plane-cut gyral boundaries.** AHRAv2 gyral borders are straight cuts (e.g. the
  vertical STG edge at x ≈ −65 on y = −27.5, and MTG/ITG at z ≈ −17), not sulcal
  fundi. Channel boundaries inherit this, so a one-hot STG/MTG/ITG split is
  population-template anatomy, not a sulcal landmark.
- **`other_gray_matter` is heterogeneous.** It mixes neighbouring cortex (insula,
  opercula, FuG, perirhinal) with deep gray (putamen, claustrum) and, on posterior
  planes, cerebellar cortex. If a projection ever needs "neighbouring cortex only",
  split this channel; do not reinterpret it.
- **Unknown codes in a target.** Codes not in `fine_regions` cannot be named
  channels without a terminology check entry.

## Region-specific white matter (documented, not implemented)

AHRAv2 has **no gyrus-specific white matter**. Of the 4.76 M white-matter voxels,
4.10 M are code 220, "forebrain white matter (unspecified region)". The rest are
named tracts (corpus callosum, fornix, external sagittal stratum 251 = optic
radiation, peduncles, …). None is STG, MTG or ITG white matter.

Region-specific WM would therefore have to be *derived*. For example:

- assign each code-220 voxel to the nearest gyral gray label, within a depth limit; or
- use a geodesic Voronoi split through white matter.

That is a modelling choice, not an atlas label. It is not implemented: the
representation keeps one shared `white_matter` channel. A future split would add
channels after `white_matter` and extend `gmwm` to sum them back into white, so the
default stays exact.

## How to use

```bash
# checks + figures for a job
PYTHONPATH=src:src/config_setup:src/preprocess python src/scripts/scan_atlas_planes_for_qnp.py \
    --config configs/project_config.yaml --job H20.33.034_MTG --stage atlas-features
# re-export a plane: now also writes nu_Anat (X / nu_Sub / nu_All unchanged)
... --stage export --plane p+000_w+000_y-027.5
```

To try anatomical source features in xIV, add `source_feature_projection: anatomical`
(or `mtg_gm`) to a registration record. Do this only after re-exporting its source
archive. It is not the default, and the results are not validated.
