# QNP stacking: the LGN slab prior

An optional anatomical prior on the AP (stacking) coordinate of the QNP stacking search
(`scan_atlas_planes_for_qnp.py`, stacking stages). It is off by default, and the
unconstrained search does not change.

## Rule

- **Source.** Gabitto et al. 2024 (Methods): "MTG was sampled at the level of first
  appearance of the lateral geniculate nucleus", which corresponds to the intermediate
  subdivision of A21.
- **Slabs.** Latimer et al. 2023: 4 mm coronal slabs, cut anterior → posterior, with fixed
  and frozen slabs alternating.
- **Which tissue.** The LGN rule is stated for the frozen molecular sample. QNP comes from the
  fixed STG–MTG block, so the slabs on either side stay candidates.

## Geometry (`src/preprocess/anatomical_slab.py`)

- **Coordinates.** Atlas world mm, MNI RAS, with +y anterior. Slab coordinate `s = −y`
  (posterior-increasing).
- **Landmark.** The most anterior coronal plane where AHRAv2 `LG` (code 174, left hemisphere)
  has a cross-section ≥ `min_area_mm2`. The landmark is found from the world coordinates of
  the label voxels through the affine, never from voxel-index order. AHRAv2 gives
  **y = −22.0 mm**. The label starts abruptly at 13.75 mm², a plane-cut label edge, so the
  precision is one voxel (0.5 mm).
- **Candidate slabs.**
  `S_j(δ) = [s_L − δ + j h, s_L − δ + (j+1) h]`, with h = 4 mm, δ ∈ [0, h) in 0.5 mm steps,
  and j ∈ {−1, 0, +1}.
- **Search interval.** The union over all δ and j, ± `extension_mm` (1 mm):
  **y −13.0 … −31.0 mm** (18 mm). The full MTG extent searched without the prior is 102 mm.
- **Three concepts.** The tissue slab, the atlas `sampling_regions` (MTG/STG) inside it, and
  the QNP block. The QNP block is never constructed: its position is what the search
  estimates.

## Run

```bash
PYTHONPATH=src:src/config_setup:src/preprocess:src/scripts \
python src/scripts/scan_atlas_planes_for_qnp.py --config configs/project_config.yaml \
    --job H20.33.034_MTG --landmark-slab --stage slabs   # then: profile --level h1.5, profile --level h1.0, stack-report
python src/scripts/scan_atlas_planes_for_qnp.py ... --stage landmark   # landmark_slab.json + figures
```

- Use the `ad-resilience-xiv` env.
- `roles` and `compress` are shared with the unconstrained run.
- Add `--region-prior` for the in-plane prior as well.
- Output goes to `<stacking.output_dir>/landmark_slab/`, or to `region_prior__landmark_slab/`
  when both priors are on.

What changes under the prior:

- `profile`, first level: scans every 0.5 mm plane of the interval. No slab shortlist is
  needed.
- Escalation levels and range extensions: clamped to the interval.
- `windows_at_prior_bound` names the anchors whose window reaches the bound, i.e. where the
  image evidence pushes against the prior.
- **Flatness (anchor → passenger) is inherited from the unconstrained first level.**
  - In an 18 mm range, δ (the top-20-plane SD) is inflated: 0.015 vs 0.0095 on H20.33.034.
  - With the inflated δ, LFB was spuriously demoted.
  - Without an unconstrained profile, flatness is not judged.
- `stack-report` adds `landmark_slab` to `report.json` and `REPORT.md`. For each anchor, it
  lists the candidate slabs (j, δ range) whose best plane is within δ of the best slab.

## Adjust

All settings are in `atlas_scan.stacking.landmark_slab` in `configs/project_config.yaml`:

| What | Setting |
|---|---|
| Landmark position | `landmark.y_mm: <MNI y>` (manual; replaces the label rule) or `min_area_mm2` |
| Narrow the slab set (e.g. a known slab) | `slab_offsets: [0]` or `[1]` |
| Slab thickness | `slab_thickness_mm` |
| Margin beyond the nominal slabs | `extension_mm` |
| Other ROI | its own `landmark_slab` block in the job (`applies_to_regions` guards against inheriting the MTG rule); no block = no constraint |

Check the landmark with `slab_coronal_montage.png` before trusting it.

## Figures (in the output folder)

- `slab_3d.png`: brain, nominal slabs (δ = h/2), MTG/STG within the interval, LGN.
- `slab_coronal_montage.png`: planes 2 mm apart around the LGN; labels, nominal slab per
  panel, L/R marked.
- `slab_sagittal.png`: through the LGN and through MTG. It shows the slab bands, each j's
  cover over all phases, the search bounds, and the stacking windows when a report exists.
- `best_poses.png` and `depth_profiles.png` (from `stack-report`): the registered sections
  over the labelled atlas plane, and the profiles with the LGN line and the bounds.

## Results (2026-11-19, h1.5 → h1.0)

| Donor | Prior | Windows (y mm) | Consistent slabs | Notes |
|---|---|---|---|---|
| H20.33.034 | slab | NeuN −27.0…−28.0, LFB/AT −27.5…−28.0 | j +1 (δ 0–3) | Same poses and scores as unconstrained. |
| H20.33.043 | slab | −13…−16 | j −1 (δ 2–3.5) | Still the insula pose. LFB and I6 reach the anterior bound. |
| H20.33.043 | region + slab | −24.0…−25.0, all anchors | j 0 or j +1 | Ribbon 75% on MTG. Prior-driven: 7–14 δ worse on the data-only score at 500 µm (see below). |

For H20.33.034, j = +1 is the slab posterior to the LGN slab. That fits a fixed QNP slab
next to a frozen LGN slab, but this has not been verified.

## Admissibility and full-resolution comparison (2026-11-19, follow-up)

### Admissible set

The admissible set has two parts:

- **Landmark slab:** the LGN slab AP interval.
- **Region prior:** `--region-prior`, an atlas support within 6 mm of MTG/STG/ITG gray
  matter.

How it excludes placements:

- The search only sees atlas tissue inside the admissible set. A placement elsewhere, such as
  the temporal plane or insula, has nothing to match, so no overlap threshold is needed.
- The prior is a constraint, not a weighted term. Its effect on a placement's score is
  reported separately in `candidates.json` as `prior_effect_on_score`.

Why ITG stays in the support:

- An STG/MTG-only support (6 mm) moved H20.33.034's LFB and AT off their current candidate,
  by 5–6 δ on the data-only score. A stricter prior is not evidence, so ITG stays.
- That run is archived in `~/tmp/landmark-slab/fullres/stg_mtg_support/`.
- For H20.33.043, both supports give the same admissible placement.

### Stage `candidates`

Run it with the priors on:

```bash
... --region-prior --landmark-slab --stage candidates   # after profile --level 500um in both folders
```

For each anchor, it compares two placements:

- the unconstrained best;
- the admissible best, as searched and after local refinement on the full atlas.

All comparisons use one **data-only** objective: the stacking score against the full,
uncropped coronal plane. δ comes from the unconstrained profile.

Diagnostics, none of which decides anything on a threshold:

- ribbon landing by region (MTG, STG, ITG, temporal plane = TTG/PLT/PLP, insula);
- QNP pia → atlas outer surface, split into exposed pia (on the section hull) and sulcal pia
  (> 2.5 mm inside it);
- QNP gray/white boundary → atlas gray/white boundary;
- share of section tissue inside the admissible support;
- pose;
- a local pitch × yaw check (±10°).

Status per anchor:

| Status | Meaning |
|---|---|
| prior inactive | Same placement in both runs. |
| image-supported | The admissible placement scores as well or better. |
| tie | The gap is within δ. |
| prior-driven | The gap is greater than δ. |

The verdict also takes into account whether the anchors' windows intersect and whether any
search-boundary flags are raised.

Outputs: `candidates.json` and `candidates.png`. The new overlay style is also used in
`best_poses.png`:

- atlas fills are muted, with thin grey atlas boundaries (outer surface solid, gray/white
  dotted);
- transformed QNP boundaries are heavy black (outline solid, gray/white dashed), with the
  pia in red;
- the admissible support is drawn as a green dash-dot line;
- flags appear in red: an AP optimum on the searched range or on the prior bound, and scale
  at its bound.

### Results at 500 µm (δ ≈ 0.0055)

**H20.33.034:** prior inactive for NeuN, LFB and AT. The admissible and unconstrained
placements are identical (y −27.5 / −28 / −28, gap 0 δ). The verdict is "supported by image
geometry", and the candidate is unchanged.

**H20.33.043:** primarily prior-driven for 3/3 anchors.

| | Data-preferred (excluded) | Admissible (MTG) |
|---|---|---|
| y (mm) | −15 / −14 / −13.5 | −24.5 (all anchors) |
| Data-only score | 0.176 / 0.172 / 0.181 | 0.249 / 0.247 / 0.221 (+13.1 / +13.6 / +7.4 δ) |
| Ribbon landing | temporal plane 45–52% (Heschl's 38%, planum polare 14%), STG 19%, insula 12–17%, MTG 1% | MTG 74–77%, ITG 4–5%, STG 1–4% |
| Tissue in admissible support | 45–47% | 100% |
| Exposed pia → atlas surface, p50 | 1.0–1.1 mm | 2.5–2.7 mm |
| Sulcal pia → atlas surface, p50 | 5.5–8.6 mm | 7.9–9.2 mm |
| Gray/white → atlas gray/white, p50 | 0.5–1.0 mm | 1.0–1.1 mm |
| θ / scale | 265–269° / 1.06–1.19 | 310–316° / 0.995–1.03 (flip True for both) |
| Tilt check | pitch 10° improves NeuN by 5.9 δ and LFB by 3.7 δ | ≤ 1.3 δ |

Reading the H20.33.043 results:

- The MTG placement is a stable local optimum of the data objective: refining it on the full
  atlas leaves it on MTG.
- Every geometric diagnostic prefers the excluded placement, and so does freeing tilt.
- The data-preferred placement is the **superior temporal plane** (the dorsal STG surface
  inside the Sylvian fissure) plus the posterior insula, not the insula alone.
- About 43% of this section's pia lines a deep sulcus (about 22% in H20.33.034). That sulcal
  pia is far from any atlas surface in both placements. At MTG, the sulcus lies on the
  STG/MTG label edge, where AHRAv2 has no open STS.
- The population atlas does not represent the section's dominant feature. The atlas-to-section
  objective cannot certify the MTG placement and rules it out only weakly.
- **Not resolved.** The MTG placement (y −24.5) is the anatomically admissible candidate, but
  it rests on the priors, not on image geometry.
