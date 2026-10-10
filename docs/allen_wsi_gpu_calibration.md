# WSI annotation GPU calibration

The production annotation scheduler uses `wsi-upsample-seg-v1`, calibrated through
`upsample_between_slices(mode="seg", parallel=False)`. Allen direct-flow pilots and
old worklists are historical artifacts and cannot authorize this backend.

The solver arguments are frozen in `configs/allen_dense_pairwise_wsi_seg.json`
(SHA `6442fa60eb8db8f8b9056c377359ca2725e2d0418a9a477f0feeebae8cdb6ae1`).
The launcher uses the new schedule and a separate WSI production output directory.
Calibration writes diagnostics only; it does not launch production densification.

Activate the `wsi-pipeline` environment and run from the repository root:

```bash
PYTHONPATH=src:. python scripts/calibrate_allen_wsi_gpu.py \
  --source-output data/derivatives/allen/specimen_708424/annotations_dense_annotation_wsi_test_20261010_173135_pair_0075-0143_200um_groups_31_265297118
```

The source output must contain an existing full preflight, endpoint-pair table, and
materialized anchors matching the current authoritative sources and solver SHA.
Resource preparation reads cached channel counts and registered geometry metadata;
only selected pilots read endpoint images. Use `--prepare-only` to rescore without
pilots, or `--select-channels 32 49 65` / `--select-quantiles .25 .5 .75` to inspect
candidate selection. The default diagnostics directory is
`results/diagnostics/allen_wsi_gpu_calibration`.

Pilots run serially in fresh processes. The sequence establishes instrumented
4-channel success and 94-channel OOM evidence, probes 32/49/65 channels, then
refines between the highest success and lowest OOM to a five-channel bracket.
An 83-channel probe follows a 65-channel success. Candidate ties use pair ID;
automatic probes remain below a verified OOM score. Non-OOM failures stop the
sequence for investigation. Existing reports must have the current full identity.

For an explicit instrumented probe beyond successful coverage, run
`scripts/benchmark_allen_pair_registration.py` with `--pair`, `--device cuda:0`,
`--configuration corrected-production`, `--source-output`, `--resources` pointing
to the new preflight, and `--report` under diagnostics. Exit status is 0 for
success, 2 for OOM, and 1 for another failure. Review existing non-OOM failures
before rerunning calibration. The old 94-channel failure lacks the complete
identity and is retained as supporting evidence only; the new instrumented
failure establishes the production bound.

The structural score is a byte-scaled proxy, not a peak-VRAM reconstruction. It
models five-plane synthetic endpoints, active optimization allowances, channel
trajectories, transforms/Jacobians, FFT/velocity work, and nested retention.
There are two outer symmetric calls, each with forward/backward optimization.
The pinned optimizer returns velocities/weights on CPU and coordinate grids on
the solver device; WSI trajectories return on CPU. Those host terms are reported
separately. The first outer result persists during the second call; CUDA axes
retention is included without counting every returned tensor as CUDA memory.
Physical gap affects host/output work and runtime, not the CUDA score.

Fit at least three successful distinct scores against peak reserved VRAM. OOM
reports do not enter regression. Admission uses the fit plus positive residual
allowance, a 1.15 safety factor, and an availability reserve of max(15% capacity,
1 GiB). Every production GPU pair must satisfy:

```text
safety_adjusted_prediction <= admission_budget
structural_score <= largest_successful_calibration_score
structural_score < lowest_observed_oom_score, if present
```

The identity includes backend/mode, both repository commits, relevant backend
source hashes, solver SHA, shape/axes/dtype, score version, and GPU model/capacity.
Schedule generation, launch, and initialized workers reject stale identities.
CPU/GPU worklists reconcile to the full inventory with no frozen device counts;
historical completion metadata remains separate. A missing or unusable fit
blocks schedule generation. Unmeasured and memory-excluded pairs go to CPU;
non-OOM numerical failures remain unresolved without matching CPU success.
