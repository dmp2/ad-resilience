# Dense registered-histology Allen annotations

`preprocess.densify_allen_annotations` implements two-sided, Nissl-driven
diffeomorphic interpolation of categorical Allen annotations using two
endpoint-conditioned WSI trajectories. It writes only in registered-histology
space; MRI transfer is deliberately excluded.

## Canonical physical-z sampling and LDDMM time

The authoritative 2,846-position Allen/Ding physical-section lattice is both
the source of real anchor coordinates and the dense output z axis. For every
canonical coordinate `z` strictly between consecutive valid semantic anchors
`z0 < z1`, the requested pseudotime is

```text
t = (z - z0) / (z1 - z0).
```

Only the annotation-supported range is filled. Observed endpoints are copied
literally from the stored final registered anchors and are never regenerated
from a trajectory. `dense.zarr/z_um` stores the authoritative coordinates
explicitly as `float64`; the dense group arrays retain 2,846 z planes.

`nt` has a separate role: it is the number of temporal intervals used to
estimate and numerically integrate each LDDMM velocity trajectory. The initial
configured value is `nt=10`. It can be increased, for example to 20, for a
single-pair expert comparison without changing the requested canonical output
coordinates.

The arbitrary-time source-flow evaluator integrates the fitted piecewise
velocity to requested `t` using the same Euler step, inverse-pullback
composition, and transform-domain resampling helpers as the pinned WSI
implementation. It does not linearly interpolate transform maps. The right
endpoint-conditioned trajectory is evaluated at complementary time `1-t`.
At exact stored times `t=k/nt`, a regression requires the evaluator to
reproduce WSI's stored source-side `phi_I[k]` and corresponding warped Nissl
state to numerical tolerance.

The old generic WSI condition `nt >= max canonical-index gap` applies to its
fill-a-preexisting-grid upsampler. It is not a scientific or implementation
requirement of this Allen branch. Likewise, `nt=ceil(delta_z_um/20)`, the
20-um/state policy, the 6,719-position pseudosection union, and the assumption
that output planes must be stored integer trajectory states are not used.

## Execution

Use the environment containing the pinned WSI pipeline, EM-LDDMM, Zarr,
SciPy, tifffile, PyTorch, and this project's `src` tree.

Materialize and validate observed anchors and write the pair plan without
running registration:

```bash
PYTHONPATH=src python -m preprocess.densify_allen_annotations --anchors-only
```

Run one representative pair with the configured `nt=10`:

```bash
PYTHONPATH=src python -m preprocess.densify_allen_annotations \
  --dataset data/derivatives/allen/specimen_708424/histology_symmetric_nissl_native_200um_section_aligned \
  --registration-run results/allen/specimen_708424/emlddmm/native-200um-clean/HIST_NISSL_SYMMETRIC_SECTION_ALIGNED_to_MRI_7T_WHOLE \
  --pair 1097-1105
```

An expert can override temporal discretization for one explicit pair. Use a
dedicated output for comparisons:

```bash
PYTHONPATH=src python -m preprocess.densify_allen_annotations \
  --pair 1097-1105 --nt 20 \
  --output data/derivatives/allen/specimen_708424/annotations_dense_registered_histology_200um_nt20
```

A run without `--pair` processes all missing pairs serially and assembles the
combined product, but it should not be launched until the single-pair review
is accepted. Completed pairs are restart checkpoints. `--overwrite` is
intentionally valid only with an explicit `--pair`; it never replaces
authoritative anchors.

## Numerical and semantic conventions

Geometry retains WSI's two outer calls. Each endpoint-conditioned call performs
its existing forward/reverse fit and symmetric-velocity construction. Left
memberships use the left evaluator at `t`; right memberships use the right
one at `1-t`. The two trajectories are not collapsed into one geodesic.

Endpoint weights are the corresponding final-A2d-placed prepared Nissl
mask-channel planes; annotation occupancy is never a weight. The accepted
final-A2d pullback, semantic availability, one-hot categorical transport, and
graphic-group precedence are unchanged.

Each pair's local Allen IDs become one-hot scalar fields. Those fields are
warped continuously and fused as `(1-t) * left + t * right`; integer IDs are
never interpolated and Jacobians are not used. Exact membership ties prefer
larger left membership, then larger right membership, then the smallest
original Allen ID.

Allen returned no Hotspots layer for sections 111, 179, 2737, and 2797.
These remain conservatively `UNAVAILABLE`, not valid-empty background. All
420 returned layers are `LABELED`; current inputs contain no `VALID_EMPTY`
layer.

## Representative capacity result

Pair `1097-1105` has endpoint z coordinates -16,275 and -15,875 um
(delta-z 400 um). With `nt=10`, it filled the seven intervening canonical
positions for each of four graphic groups. Pair processing took 110.60
seconds; the complete timed command took 2:12.47 and peaked at 1,062,436 KiB
RSS on CPU. Across all 11 stored times for both endpoint-conditioned flows,
the map and warped-Nissl maximum absolute errors were zero.

No full production run has been launched.

## Products

The default root is
`data/derivatives/allen/specimen_708424/annotations_dense_registered_histology_200um/`:

- `anchors.zarr`: final-placed Nissl, endpoint weights, and per-group hard
  annotation anchors;
- `metadata/anchors.tsv`: section-by-group semantic and transform provenance;
- `metadata/endpoint_pairs.tsv`: every required pair's endpoint coordinates,
  canonical gap/interior count, and configured LDDMM `nt`;
- `dense.zarr/z_um`: the authoritative 2,846-position physical-z array;
- `dense.zarr/groups/<group-id>`: chunked dense `uint32` group volumes;
- `dense.zarr/semantic_state`: group-by-canonical-z `uint8` provenance;
- `metadata/section_group_provenance.tsv`: readable state table after full
  assembly;
- `dense.zarr/combined`: ordered nonzero-overwrite composite after full
  assembly;
- `metadata/pairs/*.json`: map checks, temporal configuration, categorical
  vocabulary, runtime/memory, and restart status.

State codes are 0 `UNSUPPORTED`, 1 `OBSERVED_LABELS`, 2
`OBSERVED_VALID_EMPTY`, and 3 `INFERRED`. Thus zero-valued background and
unsupported group/position combinations remain distinguishable.
