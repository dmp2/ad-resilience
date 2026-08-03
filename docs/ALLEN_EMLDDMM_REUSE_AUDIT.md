# Allen direct-7T EM-LDDMM reuse boundary

The Allen workflow is intentionally a thin adapter around pinned EM-LDDMM and
the adjacent WSI tissue-pipeline orchestration. It does not replace image
support, padding, mixture estimation, contrast fitting, interpolation,
multiscale optimization, or transformation writers.

## Capability matrix

| Capability | Decision | Implementation boundary |
|---|---|---|
| Frozen Allen manifest parsing | Project adapter | Select Nissl/PV, validate counts, IDs, paths, dimensions, spacing, and hashes |
| Physical mixed-stain ordering | Project adapter | Complete slots 36-2881; one row per 50-um cutting position |
| 32-to-200-um section resampling | Project adapter | Centered pixel grids, antialiasing, exact source/prepared pixel transform |
| JSON sidecars | Reuse upstream | Call pinned `histsetup.generate_sidecars`; assert sizes, axes, origins, and units |
| `samples.tsv` | Project adapter | Write from `physical_sections.tsv`; upstream helper has mouse-specific metadata and emits `missing` |
| Common serial canvas | Reuse with validated argument | Audit upstream 95th-percentile heuristic; pass explicit maximum `xJ` only because the real default crops |
| Image interpolation and padding | Reuse upstream | `emlddmm.load_slices` only |
| `W0` | Reuse upstream | Operationally first loaded channel greater than zero; diagnostics only |
| `WM`, `WA`, `WB` | Reuse upstream | Pinned Gaussian-mixture E step; never estimated by project preprocessing |
| Per-section contrast | Reuse upstream | `slice_matching=True`, `order=1`, multiscale `local_contrast=[[]]` |
| Multiscale registration | Reuse upstream | `emlddmm_multiscale` with z-preserving `downJ` |
| Transformation outputs | Reuse upstream | Pinned transform and QC writers |
| WSI workflow | Adapt narrowly | Reuse backend resolution, staging, configuration, and output patterns without its external-mask assumptions |
| Direct MRI identity/geometry | Project gate | Inventory basename, headers, archive transforms, literature evidence, and refuse unresolved geometry |
| MRI VTK creation | Thin project adapter | Canonical axis permutation/flip, verified 200-um centered axes, pinned VTK writer |
| Graph naming | Project validation | Unique `HIST_ALL`, `HIST_NISSL`, and `HIST_PV` spaces; reject duplicate edges |
| Nissl annotation support | Deferred isolated adapter | Separate from intensity `W0`; activated only after transform-direction test |
| Stain-specific reconstruction | Project orchestration | Shared saved geometry, separate Nissl/PV intensities and support/source products |

## Reasons direct reuse is not always possible

`histsetup.generate_sidecars()` is directly reusable because one-based Allen
grid-index filenames and `max_slice=2846` reproduce the required centered
domain. Its `make_samples_tsv()` is not the final metadata authority: the
pinned implementation writes mouse-specific participant/species values and
uses `missing`, whereas the Allen derivative requires participant `708424`,
species `Homo sapiens`, and status `present|absent`.

The loader's automatic in-plane canvas is also not accepted blindly. It uses a
95th percentile rather than the maximum, and the real prepared stack would be
cropped. Passing the audited centered maximum axes through the documented
`xJ` parameter keeps all interpolation, padding, and `W0` behavior upstream.

The stock transformation-graph runner executes tuples sequentially and
hard-codes `full_outputs=False`. The project runner calls the same pinned core
with `full_outputs=True` only for the 30-slot numerical pilot, then uses
`full_outputs=False` for the full stack.

## Explicit non-reuse

The workflow does not use or implement:

- external Nissl, PV, or MRI tissue masks;
- positive intensity offsets or alternative support estimation;
- custom Gaussian mixtures or `WM/WA/WB` updates;
- stain normalization before the mixed pilot;
- a substitute MRI reconstruction or unsupported contrast label;
- 500-um canonical histology inputs;
- merged Nissl/PV reconstruction intensities.

## Pins

```text
EM-LDDMM          864990e0619fcdfb3e22e05298291f439f1b6f3d
WSI pipeline      d4d118a47d08700c8c30cf852b855e14e411bbdf
```

Any upstream defect that would require a local patch must first be reproduced
by a focused test and documented against these exact commits.
