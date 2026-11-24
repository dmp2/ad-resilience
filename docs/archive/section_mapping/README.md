# Archived section-mapping reports (history, not the current method)

These are the experimental reports and hand-off prompts written while the method was being
developed (2026-11). They hold the full evidence behind
[../../section_mapping_findings.md](../../section_mapping_findings.md). They describe the code
as it was at the time. For the current method, workflow and code map, see
[../../section_mapping.md](../../section_mapping.md).

| report | topic |
|---|---|
| `qnp_landmark_slab_prior.md` | LGN slab prior, admissibility, full-resolution candidate comparison (both donors) |
| `qnp_stacking_xiv_anatomical.md` | stacking coordinate by xIV similarity fits, with and without HALO MTG supervision |
| `qnp_anatomical_supervision.md` | the positive-only HALO MTG xIV term (fork `feat/multi-term-varifold`) |
| `qnp_atlas_anatomical_features.md` | the one-hot anatomical atlas channels and their projections |
| `qnp_loss_normalisation_audit.md` | why native xIV losses cannot rank runs; the common score |
| `qnp_pipeline_reuse_audit.md` | the pre-consolidation reuse audit |
| `*_fresh_chat_prompt.md` | hand-off prompts for the pose-montage, second-donor, flipped-STS and landmark-varifold sessions |
| `regression_evidence_20261124.sha256` | SHA-256 manifest of the frozen, read-only regression evidence at `results/frozen/section_mapping_regression_20261124/` (research volume, not in Git; see its `README.md`) |
| `section_mapping_pre_consolidation.tar.gz` | the code, tests, skill and config as they were before the consolidation (provenance for the saved runs), plus the one-off old-vs-new equivalence scripts; see its `README.md` |

The reports use the old file names. This table maps them to the current ones:

| old | current |
|---|---|
| `src/scripts/scan_atlas_planes_for_qnp.py` | `src/scripts/map_qnp_sections.py` (stacking stages) and `src/scripts/qnp_atlas_planes.py` (scan, export, evaluate, graph, atlas-features); the old name remains as an alias |
| `--stage stack-report` | `--stage report` (old name accepted) |
| `src/preprocess/section_alignment.py` | `src/section_mapping/alignment.py`; the reslicing and label tables moved to `src/section_mapping/atlas.py` |
| `src/preprocess/section_alignment_gpu.py` | `src/section_mapping/alignment_gpu.py` (`_distinct` is now `alignment.distinct_poses`) |
| `src/preprocess/qnp_stacking.py` | `src/section_mapping/stacking.py`; compression is in `measures.py`, plane and slab measures in `atlas.py`, `level_sigmas` in `alignment.py` |
| `src/preprocess/anatomical_slab.py` | `src/section_mapping/atlas.py` (landmark and slab section) |
| `src/preprocess/atlas_features.py` | `src/section_mapping/atlas.py` (label tables section) |
| `src/preprocess/qnp_stacking_xiv.py` | poses and common score in `alignment.py`, `atlas_plane` and `landing` in `atlas.py`, annotation masses in `preprocess/qnp_particles.py`; `SectionMeasure` in the experiment script |
| `src/preprocess/qnp_annotation_features.py` | `src/preprocess/qnp_particles.py` |
| `src/scripts/compare_qnp_stacking_xiv.py` | `estimate_qnp_stacking_xiv.py --stage compare` |
| `src/visualization/figures_slab.py` | `src/visualization/figures_mapping.py` |
| `.claude/skills/section-to-template-mapping/evidence.md` | `docs/section_mapping_findings.md` §3 |
