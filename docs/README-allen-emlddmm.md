# Allen 708424 direct-7T mixed-stain EM-LDDMM runbook

Observed-left, canonical symmetric, and preserve-source-grid histology layers
are complete. The selected whole-brain `T1_rot.mgz` has passed the 200-um
geometry and pinned-loader gate recorded in
`data/derivatives/allen/specimen_708424/mri_7t_whole/mri_provenance.json`.
Registration execution remains disabled only until a reviewed cross-space
similitude or rigid initialization is recorded.

See `ALLEN_EMLDDMM_AUDIT.md` for evidence and exact gate status, and
`data/derivatives/allen/specimen_708424/emlddmm_7t/metadata/physical_sections.tsv`
for the canonical centered serial cutting lattice. The preparation command is its sole
writer; the lattice utility is a read-only raw-provenance validator.
The derivative sidecar identifies this contract as
`prepared_metadata_schema: allen-emlddmm-serial-v3`.

## Environment pins

```text
EM-LDDMM:          configs/emlddmm-upstream-commit.txt
WSI orchestration: configs/wsi-tissue-pipeline-upstream-commit.txt
```

The focused commands use `pylddmm_env3.10` and `PYTHONPATH=src`.

## Prepare or reverify the complete dataset

```bash
bash scripts/run_allen_emlddmm.sh prepare-left

PYTHONPATH=src conda run -n pylddmm_env3.10 \
  python -m preprocess.prepare_allen_emlddmm_inputs \
  --verify-existing
```

To refresh only the canonical TSV/JSON metadata without rewriting images,
sidecars, transforms, or symlinks:

```bash
PYTHONPATH=src conda run -n pylddmm_env3.10 \
  python -m preprocess.prepare_allen_emlddmm_inputs \
  --refresh-metadata
```

The refresh stages and validates the complete candidate set, records current
output hashes, atomically replaces each metadata file, and restores the prior
files if a replacement fails.

Expected full counts:

```text
HIST_ALL:   2846 rows, 928 present
HIST_NISSL: 2846 rows, 641 present
HIST_PV:    2846 rows, 287 present
```

Prepared images are 200-um lossless RGB TIFFs. Each resampled section retains
the source pixel origin and is embedded in the same common canvas. The common
canvas receives one global translation; no section is translated to its own
center. Z remains the complete 50-um cutting lattice; neither stain is
compressed to its nominal retained interval.

## Serial geometry and provenance

The tissue was serially cut into contiguous 50-um sections. Section thickness
and section-center pitch are therefore both 50 um: consecutive section faces
touch while their centers remain 50 um apart. Allen `SectionImage.section_number`
is the authoritative whole-specimen serial coordinate. Sections 36 through 2881
produce 2,846 physical positions, a 142.25-mm center span, and a 142.30-mm
outer-face span. The centered nominal coordinate is:

```text
serial_z_center_mm = (allen_section_number - 1458.5) * 0.05
```

The 200-um Nissl and 400-um PV values are nominal observation-series sampling
intervals, not physical placement rules. Missing stains and unavailable sections
remain `unobserved` lattice rows with null image fields; no zero-valued image is
created. `anatomical_z_status` remains `not_established` until a validated MRI
alignment supplies anatomical placement.

Present rows retain `allen_section_image_id`, the raw manifest's canonical
`allen_data_set_id`, source and prepared relative paths, and hashes. The six
B1--B6 labels are secondary reconstruction-table observation envelopes only.
Their endpoints are the first and last matched observations, not physical slab
faces. Direct observation assignment, empty-row envelope context, and physical z
are separate fields. PV section 2130 (image 146699677) remains an observed PV
section at its Allen serial position with unresolved block assignment.

## Review the original and prepared full stacks

```bash
PYTHONPATH=src conda run -n pylddmm_env3.10 \
  python -m preprocess.visualize_allen_emlddmm_stack \
  --stage original \
  --data-dir data/raw/allen/specimen_708424 \
  --dataset data/derivatives/allen/specimen_708424/emlddmm_7t \
  --output results/qc/allen_emlddmm_original_stack_overview.png

PYTHONPATH=src conda run -n pylddmm_env3.10 \
  python -m preprocess.visualize_allen_emlddmm_stack \
  --dataset data/derivatives/allen/specimen_708424/emlddmm_7t \
  --output results/qc/allen_emlddmm_prepared_stack_overview.png
```

The two PNGs provide before-and-after views with central Nissl and PV sections,
sparse full x-z profiles, stain occupancy, and exact integer section-gap
distributions. The original profile keeps each native source image at source
x=0 and remains the unchanged source-frame overview. The prepared profile uses
the single globally translated common canvas and equal millimetre aspect; it
does not center sections individually. The QC labels the 104.2-by-72.8-mm
prepared canvas center extent separately from tissue support; no complete
accepted tissue-support mask is currently available. Both retain the unchanged
2,846-slot z lattice. OpenNeuro comparison availability is determined from
local annex payload presence and remains unvalidated until its geometry and
coordinate relationship pass review. Direct 7T comparison remains unavailable
while the MRI provenance gate is blocked. The
provisional 200-um MRI assumption used for this preparation QC does not clear
the registration provenance gate.

## Prepare and audit the bounded pilot inputs

```bash
bash scripts/run_allen_emlddmm.sh prepare-pilot

PYTHONPATH=src conda run -n pylddmm_env3.10 \
  python -m preprocess.audit_allen_emlddmm_loader \
  --dataset \
  data/derivatives/allen/specimen_708424/emlddmm_7t/pilot/slots-1448-1477
```

The pilot contains 30 consecutive canonical z coordinates, eight Nissl and
four PV images. It is not recentered.

The automatic loader canvas is rejected because it crops real prepared
sections. The recorded common canvas is passed through the pinned upstream
`xJ` argument; interpolation and `W0` remain upstream behavior. Because the
pinned loader ignores per-image in-plane `SpaceOrigin`, every TIFF already has
the same canvas and the same content origin. Loader centering therefore applies
only the intended single translation to the whole volume.

## Whole-brain 7T T1 gate

The immutable BrainSpan archive is already repository-controlled. Inventorying
it is read-only; the reviewed primary target is `T1_rot.mgz`:

```bash
PYTHONPATH=src conda run -n pylddmm_env3.10 \
  python -m preprocess.prepare_allen_7t_mri --inspect-only
```

Materialize the target outside raw data:

```bash
PYTHONPATH=src conda run -n pylddmm_env3.10 \
  python -m preprocess.prepare_allen_7t_mri
```

The only permitted registration input is then:

```text
data/derivatives/allen/specimen_708424/mri_7t_whole/
  T1_rot_space-MRI_7T_WHOLE_desc-header-corrected.nii
```

The command corrects the placeholder 1-mm MGH coordinate metadata to the
documented 0.2-mm spacing, then performs a lossless NIfTI container
normalization required by the pinned loader's greater-than-2-GiB MGH limit. It
preserves array shape, datatype, every voxel value, direction cosines,
orientation, frame count, and physical volume center. The single authoritative
audit record is
`mri_provenance.json`; verification is idempotent:

```bash
PYTHONPATH=src conda run -n pylddmm_env3.10 \
  python -m preprocess.prepare_allen_7t_mri --verify-existing
```

## Canonical symmetric source and established preparation

Create layer 2 from the observed prepared-left derivative:

```bash
PYTHONPATH=src conda run -n pylddmm_env3.10 \
  python -m preprocess.build_allen_symmetric_histology
```

The builder derives the bilateral grid and stores its one shared transform in
`histology_symmetric/metadata/symmetry.json`. Each 522-by-365 observed image
becomes one 522-by-730 bilateral observation; section and sample counts do not
double. The common origin mask distinguishes observed left from synthetically
reflected right and is not a default exclusion mask.

Nissl sections retain the established projected raw tissue masks. PV sections
instead derive tissue support on the prepared 522-by-365 grid by combining
local-entropy and darkness masks with independent Otsu thresholds, excluding
exact-black canvas padding, closing small gaps, filling small holes, and retaining
components relative to the largest.
The fixed parameters, thresholds, component areas, and per-section QC results
are recorded in `metadata/symmetry.json`. A deterministic early/middle/late PV
QC montage is written to `qc/pv_segmentation_montage.png`; a failed PV
plausibility or colored-fiducial check aborts the build.

Create layer 3 through the existing preparation module:

```bash
PYTHONPATH=src conda run -n pylddmm_env3.10 \
  python -m preprocess.prepare_allen_emlddmm_inputs \
  --source-dataset \
    data/derivatives/allen/specimen_708424/histology_symmetric \
  --output-dir \
    data/derivatives/allen/specimen_708424/emlddmm_7t_symmetric \
  --preserve-source-grid
```

Preserve-source-grid mode fails if the source cannot be copied without crop,
recanvas, reflection, or interpolation. It produces all three positional views
(`HIST_ALL`, `HIST_NISSL`, and `HIST_PV`) on the unchanged 2,846-row lattice.

Visualize the completed layer-3 stack in the registration environment:

```bash
PYTHONPATH=src conda run -n pylddmm_env3.10 \
  python -m preprocess.visualize_allen_emlddmm_stack \
  --dataset data/derivatives/allen/specimen_708424/emlddmm_7t_symmetric \
  --stage prepared \
  --output results/qc/allen_emlddmm_symmetric_prepared_stack_overview.png
```

The future, non-executable registration contract is:

```text
HIST_SYMMETRIC/HIST_ALL -> MRI_7T_WHOLE/7T_T1
```

It is stored in
`configs/emlddmm/allen_708424_hist_symmetric_to_mri7t_t1.json`. The pinned
adapter explicitly uses `I=MRI` and `J=histology` internally. Registration
outputs are reserved under `results/`, separate from all three derivatives.

## Mixed and fallback pilots

After MRI verification and initialization review:

The initialization is a finite homogeneous 4x4 text matrix with a same-stem
JSON review record. The record must contain the selected MRI basename, the
exact target view, the matrix SHA-256, and `"review_status": "accepted"`.
The runner rejects mismatches and any nonempty output root.

```bash
bash scripts/run_allen_emlddmm.sh pilot-all VERIFIED_MRI.vtk ACCEPTED_INITIAL_AFFINE.txt
bash scripts/run_allen_emlddmm.sh pilot-nissl VERIFIED_MRI.vtk ACCEPTED_INITIAL_AFFINE.txt
bash scripts/run_allen_emlddmm.sh pilot-pv VERIFIED_MRI.vtk ACCEPTED_INITIAL_AFFINE.txt
```

The runner enforces:

```python
slice_matching = True
order = 1
local_contrast = [[]]
full_outputs = True
n_draw = 0
```

It retains the short pilot's upstream `W0`, `WM`, `WA`, `WB`, contrast
coefficients, and transforms for finite-value inspection. It does not estimate
or replace those fields.

## Full registration

Only after the mixed pilot passes:

```bash
bash scripts/run_allen_emlddmm.sh full-all VERIFIED_MRI.vtk ACCEPTED_INITIAL_AFFINE.txt
```

The full runner performs a 70%-of-available-memory preflight, preserves the
200-um canonical inputs, and uses:

```python
full_outputs = False
n_draw = 0
```

Standard geometric outputs and transformation files are written; complete
voxelwise mixture arrays are not retained.

## Stop conditions

Do not run a real pilot or full optimization until all preceding gates pass.
Specifically, the current blocked MRI provenance record is an intentional hard
stop.

External tissue masks, alternative released reconstructions, custom support
weights, custom Gaussian mixtures, stain normalization, dense annotation
propagation, and xIV export are not preparation or pilot prerequisites.

Before applying transformations to Nissl annotations, run the separate
point/categorical direction test. Nissl and PV may share the registration
geometry, but their reconstructed intensity, support, and source-section
volumes remain separate.
