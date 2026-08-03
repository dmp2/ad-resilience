# Configured xIV-LDDMM particle workflow

This package modernizes and documents the three colleague-provided scripts without
modifying the copies under `legacy/`.

## Coordinate-space conclusion

- OpenNeuro `ds003590` reconstructs the Allen Human Brain Atlas donor's MRI, Nissl, and
  parvalbumin data in subject space and also supplies mapping to MNI space. This supports
  treating the Ding material and ds003590 as the same donor/reference lineage, but it
  does **not** justify assuming every raw and reconstructed file has the same voxel grid.
  Run `inspect_spaces.py` and retain the provided transforms.
- The Allen Human Reference Atlas 3-D, 2020 is a 141-structure parcellation drawn on
  **ICBM 2009b Nonlinear Symmetric**, a population-average MNI152 template. Its
  parcellation and ontology were adapted from Ding's prior 2-D atlas, but it is not the
  native Ding donor volume.

Accordingly, the setup file uses three different space names: `ding_raw_native`,
`ding_openneuro_subject`, and `mni_icbm2009b_symmetric`.

## Files

- `config/project_config.yaml`: the single machine/project setup file.
- `src/particle_utils.py`: affine-aware particle construction, block aggregation,
  metadata, and mass checks.
- `src/xmodmap_compat.py`: thin adapters to repository-native input, entropy, VTK,
  `makePQ`, and output functions.
- `XMODMAP_REUSE_AUDIT.md`: function-by-function comparison and reuse decisions.
- `src/makeADNIsegParticles.py`: generic target segmentation converter; historical
  filename retained.
- `src/makeHighFieldTemplateParticles.py`: generic source/atlas converter at one or
  more aggregation scales.
- `src/framework_script_HighFieldToLowFieldIndividual.py`: configurable single-modality
  xIV-LDDMM registration.
- `src/inspect_spaces.py`: NIfTI grid/affine report.
- `legacy/`: untouched uploaded scripts.

## First setup on Dalet

1. Place this folder at the desired project location.
2. Edit only `project.root` in `config/project_config.yaml`.
3. Clone or copy the exact xIV-LDDMM repository to the configured
   `software.xiv_lddmm_repository`. For reproduction, prefer the colleague's exact
   commit or directory over silently substituting the current public branch.
4. Create a Python environment and install `requirements.txt`.
5. Put downloaded data under the configured dataset roots, or change those paths in the
   same YAML file.
6. Create or identify **label NIfTI volumes**. The three supplied scripts do not consume
   raw MRI, dMRI, Nissl, parvalbumin, or transcriptomic intensities directly.

## Recommended first commands

Inspect likely same-donor files:

```bash
python src/inspect_spaces.py \
  --output results/qc/ding_openneuro_geometry.json \
  /path/to/ding_mri.nii.gz \
  /path/to/ds003590_subject_space_mri.nii.gz
```

Create source particles after enabling/filling the job:

```bash
python src/makeHighFieldTemplateParticles.py \
  --config config/project_config.yaml \
  --job ding_subject_source
```

Create target particles:

```bash
python src/makeADNIsegParticles.py \
  --config config/project_config.yaml \
  --job example_target
```

Run a single-modality registration:

```bash
python src/framework_script_HighFieldToLowFieldIndividual.py \
  --config config/project_config.yaml \
  --registration ding_subject_test
```

## Important model boundary

`SingleModality` uses an identity feature mapping. Source and target feature columns
must have the same order and biological meaning. An Allen HRA annotation with all 141
labels cannot simply be paired with an unrelated SEA-AD molecular feature matrix.
The mouse Allen/BarSeq example points toward `CrossModalityBoundary`, learned feature
mapping, and partial support; that is the next module after this geometric baseline is
reproduced.

## Behavioral changes from the legacy scripts

- Full NIfTI affine is used by default instead of diagonal spacing plus zero-centering.
- Arbitrary aggregation factors replace hard-coded nested factor-five slicing.
- Particle mass is explicitly conserved and checked.
- NPZ arrays are loaded by key, not archive order.
- Absolute `/cis/home/...` paths are removed.
- External `vtkFunctions.py` is no longer required; active VTK output delegates to
  `xmodmap.io.getOutput` through an explicit XYZ/YXZ adapter.
- The unsupported public-branch call `print_log(return_HD=True)` is replaced by reading
  the final recorded losses from `loss.log`.
- Every particle archive stores source geometry, coordinate-space name, feature names,
  and construction diagnostics.


## xmodmap-native versus project-extension preprocessing

The workflow now makes the implementation boundary explicit with
`construction_backend`:

- `xmodmap_native` calls `xmodmap.io.getInput.makeFromSingleChannelImage` or
  `makeBinsFromMultiChannelImage`. Use it for factor-1, centered-coordinate legacy
  reproduction.
- `project_affine_block` is the narrow Allen/OpenNeuro extension for complete NIfTI
  affines, anisotropic geometry, arbitrary 3-D block aggregation, masks, metadata, and
  mass conservation.

The second backend is not a replacement for xIV-LDDMM. Registration, loss functions,
Hamiltonian shooting, model initialization (`makePQ`), entropy, and VTK output remain
repository-native. See `XMODMAP_REUSE_AUDIT.md` for the detailed crosswalk.

### Important output correction

The public xmodmap function named `getJacobian` computes the ratio of deformed to
original particle mass. It does not differentiate the spatial map to calculate
`det(Dphi)`. New outputs therefore use `mass_ratio_proxy`; the historical `jac` key is
retained only for compatibility.

## Dense scalar images: T1, T2, and reconstructed histology

The additional colleague files cover two opposite operations:

1. `makeT2Particles_255bin.py` converts a scalar T2 image to one particle per nonzero
   voxel with a 256-column one-hot intensity feature.
2. `makeParticlesToT2seg_yxie.py` interpolates already warped scalar or segmentation
   particles back onto a dense NIfTI grid.

Their untouched versions are retained under `legacy/`. The configurable replacements
are:

- `src/makeDenseImageParticles.py`
- `src/dense_image_utils.py`
- `src/reconstructParticlesToNifti.py`

Create dense-image particles after filling a named YAML job:

```bash
python src/makeDenseImageParticles.py \
  --config config/project_config.yaml \
  --job ding_7t_t1_example
```

Reconstruct a scalar or categorical volume:

```bash
python src/reconstructParticlesToNifti.py \
  --config config/project_config.yaml \
  --job ding_7t_t1_roundtrip_example
```

### Which feature key should be used?

- `nu_bin` is the preferred starting point for **same-contrast image registration**.
  Its feature masses sum to tissue volume, so brightness changes composition without
  changing total geometric mass.
- `nu_intensity` stores `[mass, mass × transformed intensity]`. It is compact and lets
  the image be reconstructed after particle coordinates are warped. Do not feed it into
  a registration loss automatically: xIV-LDDMM's per-particle total weight would then
  depend on intensity.
- `nu_T2` is an optional compatibility alias for `nu_intensity`; it is written only by
  jobs that request `write_legacy_t2_alias: true`.

The legacy 256-bin representation is supported, but arbitrary MRI intensities are rarely
true integers from 0 to 255. The modern default robustly maps selected intensities to
`[0,1]` and uses fewer, linearly interpolated bins. The original intensity range is saved
so reconstructed scalar images can be returned approximately to their input units.

When two images will be compared through the same bin features, their bin definitions and
normalization policy must be shared. After initial QC, a common `fixed_range` is often more
interpretable than independently fitting robust quantiles to each image.

### Scope boundary

The generic path applies to one **3-D scalar** image at a time: T1-weighted MRI,
T2-weighted MRI, b0, FA, MD, a scalar quantitative map, or a reconstructed stain
intensity. It does not make different contrasts biologically equivalent. Raw 4-D dMRI,
vector-valued data, tensors, RGB microscopy, and thousands of gene channels require a
purpose-built feature representation rather than silently flattening them into scalar
bins.

At Allen 7T resolution, one particle per voxel can be extremely large. Start with a
validated ROI, a brain/tissue mask, and a block factor such as 5. Estimate memory before
enabling 64- or 256-bin features across a whole hemisphere.

## Tests

Project-only extensions can be checked without running xIV-LDDMM optimization:

```bash
PYTHONPATH=src pytest -q tests/test_project_extensions.py
```

The synthetic tests cover 3-D categorical mass conservation, dense-bin mass
conservation, and the XYZ/YXZ VTK adapter. A full Dalet validation still requires the
pinned xmodmap repository, its PyKeOps environment, and a known colleague case.
