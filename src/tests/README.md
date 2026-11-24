Run the project-extension tests from the package root after installing dependencies:

```bash
PYTHONPATH=src pytest -q tests/test_project_extensions.py
```

A separate end-to-end test should be run on Dalet with the pinned xmodmap repository,
one legacy particle case, and one known registration output. These unit tests do not
validate PyKeOps kernels or the numerical optimizer.

Section-to-atlas mapping (`src/section_mapping`, `src/scripts/map_qnp_sections.py`), one
file per module; the GPU tests fall back to CPU torch:

| file | covers |
|---|---|
| `test_atlas.py` | label tables, reslicing, landing, anatomical channels and projections, landmark and slab prior; the last channel test reads the real AHRAv2 volume (skipped when absent) |
| `test_alignment.py` | transform, support-censored score, CPU search, GPU twin (agrees to ~1e-6), pose algebra, common post-fit score |
| `test_stacking.py` | compression, roles QC, depth search, delta and windows, level summary, flags and verdict, placement diagnostics |
| `test_qnp_particles.py` | QNP slide particles, multistain OD export, HALO annotation masses |
| `test_qnp_jobs.py` | job resolution against `qnp_section_defaults` |
| `test_mapping_regression.py` | regression references against the saved two-donor runs (LGN landmark, profiles, verdicts, `report` re-run, stacking → xIV hand-off scores); skipped without `results/` and the atlas |

```bash
PYTHONPATH=src:src/config_setup:src/preprocess:src/scripts pytest -q src/tests/test_atlas.py \
    src/tests/test_alignment.py src/tests/test_stacking.py src/tests/test_qnp_particles.py src/tests/test_qnp_jobs.py src/tests/test_mapping_regression.py
```
