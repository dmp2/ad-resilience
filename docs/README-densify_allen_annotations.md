# Dense registered-histology Allen annotations

`preprocess.densify_allen_annotations` implements two-sided, Nissl-driven
diffeomorphic interpolation of categorical Allen annotations using two
endpoint-conditioned WSI trajectories. It writes only in registered-histology
space; MRI transfer is deliberately excluded.

## Canonical physical-z sampling and LDDMM time

For the current specimen, the authoritative Allen/Ding physical-section
lattice contains 2,846 positions. Its length and coordinates are read from the
selected Nissl derivative metadata and `physical_sections.tsv`; they are not
fixed implementation constants. The lattice is both the source of real anchor
coordinates and the dense output z axis. For every canonical coordinate `z`
strictly between consecutive valid semantic anchors
`z0 < z1`, the requested pseudotime is

```text
t = (z - z0) / (z1 - z0).
```

Only the annotation-supported range is filled. Observed endpoints are copied
literally from the stored final registered anchors and are never regenerated
from a trajectory. The authoritative coordinates remain explicit in
`metadata/physical_sections.tsv`; every dense group retains the full number of
positions declared by that selected dataset (2,846 for the current specimen).

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

### Dataset-derived geometry and groups

The implementation derives the canonical plane count, image shape, serial
spacing, and pixel spacing from the selected Nissl derivative and validates
them against its physical-section table and accepted numerical package. It
derives graphic-group IDs and precedence order from the linked source
annotation catalog, and derives annotation section and image counts from the
annotation derivative metadata and manifests. The current values (2,846
planes, 522 by 730 pixels, 50-um serial spacing, 200-um pixels, and four graphic
groups) therefore remain unchanged without constraining another valid example
to those values.

When `--annotations` is omitted, the command finds the unique sibling
annotation derivative whose metadata names the selected Nissl derivative as
its parent. No match or multiple matches fail clearly and require an explicit
`--annotations` path. The no-argument `--dataset`, `--registration-run`, and
`--output` locations remain convenience defaults for specimen 708424. For a
different example, pass its dataset, accepted registration run, and output root
explicitly; the geometry and group inventory then come from those inputs.

### Storage policy

The default primary output is a lossless plain TIFF series:

```text
--output-format tiff                 # default
--tiff-compression deflate           # default, lossless zlib/Deflate
--tiff-compression none              # optional, lossless uncompressed TIFF
```

Use `--output-format zarr` to select the existing chunked multidimensional
backend instead. A run writes exactly one primary dense backend; selecting TIFF
does not also write `dense.zarr`, and selecting Zarr does not write the TIFF
series. The internal `anchors.zarr` cache remains common to both backends and is
not a second dense output.

TIFF is the default because section-wise access matches the histology workflow,
each canonical plane is independently readable, interoperability is broad,
restart/recovery is straightforward, and basic use requires no specialized Zarr
reader. The TIFFs contain categorical Allen structure IDs as `uint32`, not RGB
or palette/display images. Lossy compression, including JPEG, is forbidden.
Scientific coordinates and semantics remain authoritative in TSV/JSON rather
than TIFF tags.

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
`data/derivatives/allen/specimen_708424/annotations_dense_registered_histology_200um/`.
A default TIFF run uses this layout:

```text
annotations_dense_registered_histology_200um/
├── dense_tiff/
│   ├── groups/
│   │   └── <graphic-group-id>/
│   │       ├── 000000.tif
│   │       ├── ...
│   │       └── 002845.tif
│   └── combined/
│       ├── 000000.tif
│       ├── ...
│       └── 002845.tif
├── metadata/
│   ├── physical_sections.tsv
│   ├── anchors.tsv
│   ├── endpoint_pairs.tsv
│   ├── section_group_provenance.tsv
│   ├── dense_tiff_manifest.tsv
│   └── pairs/tiff/*.json
├── provenance.json
└── README.md
```

Each group directory contains exactly one six-digit, zero-padded TIFF per
canonical position, including unsupported positions. Filenames encode the
canonical positional index, never a rounded z coordinate. Unsupported planes
are ordinary zero-valued `uint32` rasters, while their state remains
`UNSUPPORTED`/`UNAVAILABLE` in metadata. Therefore TIFF pixel value 0 alone does
not determine availability or valid categorical background: consumers **must**
consult `metadata/section_group_provenance.tsv` and
`metadata/dense_tiff_manifest.tsv`. The same warning applies to combined TIFFs,
which retain the established ordered nonzero-overwrite group precedence but do
not replace per-group provenance.

Each TIFF is written to a temporary file in its destination directory, verified,
and atomically committed with `os.replace`. Only a final six-digit `.tif` name is
a completed plane; temporary files are ignored on restart. Pair checkpoints are
backend-qualified under `metadata/pairs/tiff/` or `metadata/pairs/zarr/` and
record `output_format`, so an old Zarr checkpoint cannot complete a TIFF run.
Without `--overwrite`, incompatible existing planes or store metadata fail
clearly. `--overwrite` remains limited to one explicit pair and never replaces
the observed anchor cache.

`metadata/dense_tiff_manifest.tsv` has one row per completed TIFF with its
relative path, canonical index, physical z, graphic group/product type,
semantic/evidence state, dtype, shape, and SHA256. The existing project SHA256
helper is reused. The copied `metadata/physical_sections.tsv` is the
authoritative filename-index-to-physical-z mapping.

Products shared by both backends include:

- `anchors.zarr`: final-placed Nissl, endpoint weights, and per-group hard
  annotation anchors;
- `metadata/anchors.tsv`: section-by-group semantic and transform provenance;
- `metadata/endpoint_pairs.tsv`: pair endpoints, canonical gaps, and LDDMM `nt`;
- `metadata/section_group_provenance.tsv`: readable dense state table;
- `metadata/pairs/<output-format>/*.json`: pair map checks, temporal settings,
  categorical vocabulary, runtime/memory, output format, and restart status.

With `--output-format zarr`, the preserved optional products are
`dense.zarr/z_um`, `dense.zarr/groups/<group-id>`,
`dense.zarr/semantic_state`, and `dense.zarr/combined`. Existing Zarr products
are left in place when TIFF is selected, and existing TIFF products are left in
place when Zarr is selected.

State codes remain 0 `UNSUPPORTED`, 1 `OBSERVED_LABELS`, 2
`OBSERVED_VALID_EMPTY`, and 3 `INFERRED`. No availability state is inferred from
raster values.
