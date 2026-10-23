# xmodmap reuse audit for the Kavli Allen–SEA-AD workflow

## Executive conclusion

The colleague scripts are **not an independent alternative implementation of
xIV-LDDMM**.  Their registration framework imports xmodmap's varifold losses,
Hamiltonian, shooting equations, optimizer model, coordinate normalization, VTK
summaries, entropy, and historical `getJacobian` output.  Its contribution is chiefly
application-specific orchestration: paths, label definitions, source/target choices,
parameters, output naming, cohort looping, and a few preprocessing/postprocessing
extensions.

The particle-construction scripts overlap substantially with
`xmodmap.io.getInput`, but the relative quality depends on the operation:

- Use **xmodmap natively** for canonical model setup, loss/deformation classes,
  entropy, output formatting, and legacy factor-1 categorical or scalar-bin
  reproduction.
- Retain a **thin project extension** for complete NIfTI-affine coordinates,
  self-describing metadata, explicit archive keys, arbitrary 3-D block aggregation,
  masks and robust intensity transforms, and NIfTI reconstruction.  Those functions
  are absent or materially weaker in the public `unpack` branch.
- Treat the colleague's `EmpiricalDistributions_yxie` reconstruction as potentially
  valuable, but it cannot yet be judged fully because that dependency was not supplied.

The production rule is therefore:

> xmodmap owns the mathematical registration machinery and any equivalent I/O primitive;
> this project adds only the neuroimaging geometry, provenance, scale aggregation, and
> reconstruction capabilities that xmodmap does not provide robustly.

## Function-level crosswalk

| Goal | xmodmap implementation | Colleague implementation | Assessment | Current project decision |
|---|---|---|---|---|
| Load NPZ/PT particles | `xmodmap.io.getInput.getFromFile` | Called directly by the registration framework | Same basic goal. xmodmap relies on dictionary/archive order and `featIndex`, which is fragile when files gain metadata or arrays. | Keep explicit key-based loading for new files; use xmodmap loader only for legacy archives when the key order is known. |
| Categorical image to particles | `makeFromSingleChannelImage` | `makeADNIsegParticles.py`; native-resolution branch of `makeHighFieldTemplateParticles.py` | Conceptually the same: centered voxel coordinates plus one-hot label features. The colleague adds fine-to-coarse biological groupings and voxel-volume weighting. Neither original implementation preserves the complete NIfTI affine. | `construction_backend: xmodmap_native` for factor-1 legacy reproduction; `project_affine_block` for Allen/OpenNeuro production. |
| Coarse categorical particles | `downsampleParticles` | `downsampleBrainSegs` | xmodmap's public helper is fixed factor-five and documented/implemented for 2-D image bins. The colleague performs 3-D 5×5×5 aggregation and preserves mixed labels in each block. | Retain the project generalization of the colleague's method: arbitrary 3-D block factor, mass conservation, full affine coordinates. |
| Scalar/dense image to intensity-bin particles | `makeBinsFromMultiChannelImage`; `makeBinsFromImageValues` | `makeT2Particles_255bin.py` | The T2 script is a narrower 3-D uint8 specialization. xmodmap is more generic across scalar/multichannel inputs and bin counts. Both historical versions center index coordinates and ignore the full affine. | Use xmodmap native binning for exact legacy experiments; use the affine/block extension for Allen 7T data, masks, robust transforms, and coarse particles. |
| Compact scalar transport features | No clear native `[mass, mass×intensity]` creator in `xmodmap.io.getInput` | Expected by the T2 reconstruction script | This is separate from histogram features and useful for reconstructing a scalar image after points move. | Retain `nu_intensity=[mass, mass×transformed intensity]` as a project extension; do not automatically use it as a registration feature. |
| Particle entropy | `xmodmap.io.getOutput.getEntropy` | Reimplemented in several colleague scripts | Same formula. | Use xmodmap's function. |
| VTK point/particle output | `writeVTK`; `writeParticleVTK` | External `vtkFunctions.writeVTK` and direct xmodmap output calls | Same broad goal. xmodmap's writer explicitly accepts historical `YXZ` and swaps the first two columns when writing XYZ. This is hazardous for ordinary affine-world XYZ arrays unless adapted. | Use xmodmap output functions through a thin coordinate-convention adapter; no second VTK writer in active code. |
| Source/target normalization and model initialization | `preprocess.makePQ_legacy.makePQ`; `rescaleData`; `resizeData` | Registration script calls `makePQ` | Already repository-native in the colleague code. | Use `makePQ` directly; do not duplicate its weight/composition and unit-box setup. |
| Deformation, loss, optimization | `Hamiltonian`, `Shooting`, `ShootingBackwards`, `LossVarifoldNorm`, `SingleModality`, cross-modal classes | Imported and parameterized by the framework | The colleague does not replace this mathematics. | Keep xmodmap as the single implementation. |
| “Jacobian” output | `getOutput.getJacobian` | Saved as `jac` and labeled Jacobian | The public function computes `sum(nu_D)/sum(nu_S)`. It is a particle mass ratio, not a numerical determinant of the spatial derivative `det(Dφ)`. | Save `mass_ratio_proxy`; retain `jac` only as a compatibility alias. A true geometric Jacobian requires a separate validated derivative calculation. |
| Dense scalar/segmentation reconstruction | No NIfTI reconstruction routine found in `xmodmap.io.getOutput` | `makeParticlesToT2seg_yxie.py` using `EmpiricalDistributions_yxie` | Genuine postprocessing extension. The colleague version offers Gaussian assignment, KNN support, and morphology, but depends on an unshared module and writes a synthetic affine. | Retain reference-grid reconstruction as a project extension. Compare against the colleague's implementation after obtaining `EmpiricalDistributions_yxie`. |
| Coordinate-space provenance | None in the helper NPZ convention | Implicit in filenames and reoriented directories | Insufficient for Allen native, OpenNeuro subject, MNI HRA, and SEA-AD hierarchies. | Retain coordinate-space registry, full affine metadata, geometry QC, and explicit transformations. |
| Machine/project configuration | Hard-coded paths in examples and colleague scripts | Hard-coded `/cis/home/...` paths | Neither is portable. | Retain YAML configuration and resolved run manifests. |

## Detailed technical findings

### 1. Categorical particles: mostly the same idea

`makeFromSingleChannelImage` and the colleague's target converter both:

1. load a labeled image;
2. make one coordinate per voxel;
3. remove configured background labels;
4. encode retained labels as one-hot feature vectors.

The colleague's target converter improves this for its use case by defining two related
feature matrices:

- `nu_All`: fine labels;
- `nu_Sub`: grouped anatomical regions.

That hierarchy is not supplied automatically by xmodmap and remains scientifically
useful.  The new wrapper therefore calls the xmodmap factor-1 constructor and applies
only the project-specific fine-to-coarse feature crosswalk afterward.

### 2. The colleague's 3-D factor-five aggregation is a real improvement

The high-field source script does not merely stride through every fifth voxel.  It
collects all 125 voxels in each 5×5×5 block and accumulates their label masses.  A
boundary particle can therefore carry, for example, 70% hippocampal and 30% entorhinal
feature mass.

The public xmodmap `downsampleParticles` helper is fixed to a factor of five and is
constructed around a 2-D binned image.  `makeFromSingleChannelImage(ds=5)` and
`makeBinsFromMultiChannelImage(ds=5)` instead use strided samples, not the colleague's
3-D mass-preserving block summary.  The latter function also constructs coordinates
from the downsampled array using the unmodified `res` argument, so using `ds>1` without
manually correcting spacing contracts the represented field of view.

The production block aggregator is therefore justified, but it should be understood as
a generalization of the colleague's useful method—not as a replacement for xIV-LDDMM.

### 3. T2 256-bin creation is largely duplicated functionality

The colleague's `makeT2Particles255` creates one feature column for each integer from
0 through 255 and gives each retained voxel a one-hot bin weighted by voxel volume.
`makeBinsFromMultiChannelImage` already creates hard intensity bins, supports arbitrary
numbers of bins and channels, and weights bins by spatial resolution.

The colleague specialization is easier to read for its exact use case and explicitly
removes zero-valued background, but it is not a more general algorithm.  The configured
legacy backend now calls xmodmap's function and applies the requested mask afterward.

For Allen MRI, neither historical implementation is sufficient by itself because raw
MRI intensities are not intrinsically standardized 8-bit tissue classes.  The project
extension adds masking, saved intensity transforms, optional linear bin assignment,
full affine geometry, and block aggregation.

### 4. VTK reuse requires an adapter, not blind delegation

`xmodmap.io.getOutput.writeVTK` documents its input as `YXZ` and writes coordinates in
the order column 1, column 0, column 2.  This convention is compatible with some image
array pipelines but would swap left/right or anterior/posterior axes if an ordinary
world-XYZ array were passed directly.

The active code now uses xmodmap's writer through `xmodmap_compat.write_vtk_xyz`:

- for ordinary `XYZ`, the adapter pre-swaps X and Y;
- xmodmap performs its historical swap;
- the VTK file receives the intended XYZ coordinates.

A configuration switch retains the historical `xmodmap_yxz` convention for legacy
files known to use it.

### 5. `getJacobian` must not be interpreted as a true deformation Jacobian

The public implementation is algebraically:

```python
j = sum(nu_D, axis=-1) / sum(nu_S, axis=-1)
```

It does not evaluate spatial derivatives of the map.  Depending on the model's particle
mass evolution, this ratio may be a useful local expansion/contraction-related summary,
but it is not automatically equal to `det(Dφ)`.  The revised workflow therefore calls
it `mass_ratio_proxy` in new outputs.  Any biological claim about local tissue
expansion or contraction should eventually use a validated geometric-Jacobian method
and compare it with this mass-ratio quantity.

### 6. Dense reconstruction is outside xmodmap's public output layer

The public output module provides VTK/NPZ summaries, entropy, mass-ratio calculations,
and related visualization utilities.  It does not provide a NIfTI reference-grid
reconstruction from deformed particles.

The colleague's reconstruction script is therefore not redundant.  It uses separate
mass and mass×intensity interpolation, which is mathematically sensible:

```text
reconstructed intensity = interpolated(mass × intensity) / interpolated(mass)
```

Its likely advantage is the unshared `EmpiricalDistributions_yxie` implementation of
localized Gaussian/KNN assignment and support morphology.  Its weaknesses are
hard-coded paths/parameters and a generated affine that does not preserve a reference
NIfTI's full geometry.  The project reconstruction uses an explicit reference grid and
full affine, but should be benchmarked rather than presumed superior.

## Backends in the revised package

### `xmodmap_native`

Use for:

- reproducing the repository/colleague centered-coordinate factor-1 input behavior;
- validating that our environment and feature conventions match xmodmap;
- legacy integer T2 binning at native resolution.

Restrictions are intentional:

- categorical images must be approximately isotropic because the public function takes
  one scalar resolution;
- dense/categorical native paths use `ds=1`;
- coordinates are legacy centered, not full NIfTI affine-world coordinates.

### `project_affine_block`

Use for:

- Ding/OpenNeuro/Allen subject or MNI data where physical coordinates matter;
- anisotropic, rotated, translated, or sheared NIfTI geometry;
- 3-D block aggregation with mass conservation;
- masks, robust intensity transforms, linear intensity bins, and self-describing output.

This backend should be viewed as a narrowly scoped neuroimaging extension around
xmodmap, not a competing registration library.

## Recommended validation experiments

1. **Categorical factor-1 equivalence:** run an isotropic synthetic segmentation through
   xmodmap native and project legacy-centered factor-1 paths; compare coordinates,
   feature masses, and ordering exactly.
2. **T2 256-bin equivalence:** compare xmodmap binning with the colleague script on a
   small integer 0–255 image after matching masks and voxel weights.
3. **Factor-five distinction:** compare stride sampling against 3-D block aggregation at
   anatomical boundaries; verify total mass and field of view.
4. **VTK orientation:** write asymmetric landmark points and verify their axes in
   ParaView against the NIfTI world coordinates.
5. **Mass ratio versus true Jacobian:** use a synthetic known isotropic scale map and
   independently calculate `det(Dφ)`; compare with xmodmap's particle mass ratio.
6. **Reconstruction benchmark:** after obtaining `EmpiricalDistributions_yxie`, compare
   reference-grid RMSE, edge behavior, support masks, runtime, and GPU/CPU memory.

## Repository paths audited

- `xmodmap/io/getInput.py`
- `xmodmap/io/getOutput.py`
- `xmodmap/preprocess/preprocess.py`
- `xmodmap/preprocess/makePQ_legacy.py`
- `xmodmap/deformation/`
- `xmodmap/distance/`
- `xmodmap/model/`
- `examples/runtime/framework_script_densSupport_halfBrainBarSeqD076_1L_ToAllen.py`
- representative single- and cross-modality examples

## xIV outputs accompanying the Omics and QNP prototypes

The reference is `vendor/xIV-LDDMM-Particle/examples/framework_script_2DAllenToBarSeqCells.py`.
The cross-modal pipeline now writes its atlas and target summaries through
`src/preprocess/xmodmap_compat.py`, including the reference fields and compatibility aliases.

To export complete bundles for existing figure A–C contexts without rerunning optimization:

```bash
PYTHONPATH=src:src/seaad:src/preprocess:src/config_setup:src/scripts \
  ~/miniconda3/envs/ad-resilience-xiv/bin/python -m visualization.cli xiv-outputs --modality all
```

Use the xIV environment (PyTorch and PyKeOps), rather than the rendering-only environment.
The exporter uses the vendored xIV VTK writers. `--modality omics` or `qnp` restricts the export.
It replaces only its derived bundles; the fitted registration directories remain untouched.

Output paths relative to `results/visualization_prototypes/`:

| Bundle | Saved registration context |
|---|---|
| `omics/xiv_outputs/baseline/` | `ahrav2_mtg_plane_to_strip_1194111462/init_r000_f0` |
| `omics/xiv_outputs/iv_modality/` | `omics/iv_modality/registration/init_r000_f0`, when completed |
| `qnp/xiv_outputs/pose_A/` | `ahrav2_tissue_plane_to_qnp_H20.33.034_MTG_A/init_r000_f0` |
| `qnp/xiv_outputs/pose_B/` | `ahrav2_tissue_plane_to_qnp_H20.33.034_MTG_B/init_r000_f0` |

Each bundle contains:

- `originalAtlas.vtk`, `originalTarget.vtk`: exact copies from the saved fit.
- `atlas_nu_D.vtk`, `atlas_nu_Dpi.vtk`: transported atlas and mapped target features.
- `atlas_deformationSummary.pt`: `D`, `nu_D`, `nu_Dpi`, `jac`, `mass_ratio_proxy`, and preserved support diagnostics.
- `atlas_deformationSummary.vtk`: `Weight_AtlasFeatures`, `MaxVal_AtlasFeatures`,
  `Weight_TargetFeatures`, `MaxVal_TargetFeatures`, `Jacobian`, `Entropy`,
  `MassRatioProxy`, and `RelativeSupportWeight` when available.
- `target_nu_Td.vtk`: inverse-transported target features.
- `target_deformationSummary.pt`: `Td`, `nu_Td`, `jac`, `mass_ratio_proxy`.
- `target_deformationSummary.vtk`: `Weight_TargetFeatures`, `MaxVal_TargetFeatures`,
  `Jacobian`, `Entropy`, `MassRatioProxy`.
- `input_manifest.json`, `transform_chain.json`, and `manifest.json` with source/output
  checksums, tensor-preservation checks, and VTK read-back results.

`xiv_outputs_all.json` indexes the full export (restricted exports use `xiv_outputs_omics.json`
or `xiv_outputs_qnp.json`). Figures A/B show measured or compressed inputs; figure C uses
the registration context. The bundles accompany those families without implying a separate
registration for each figure or a registration between the two donors. QNP representation
variants share the same two atlas pose fits; input compression does not create a new atlas fit.

Coordinates remain in physical mm in the fit's centered, initialized frame. Use the copied
transform chain to place them in a section or atlas frame. Feature maxima are one-based.
Atlas summary entropy is computed on `nu_Dpi`; target summary entropy uses `nu_Td`,
with upstream `getEntropy` (natural logarithm). Measures are saved before support censoring.
The upstream spelling `Weight_TargetFeaturs` is corrected to `Weight_TargetFeatures`.

Upstream `getJacobian` computes the transported particle mass ratio
`sum(nu_deformed) / sum(nu_initial)`. `Jacobian` and `jac` are compatibility aliases
for `MassRatioProxy` and `mass_ratio_proxy`; they do not denote a determinant calculated
from spatial derivatives.

Validation reads back every VTK file, checks finite values, compares coordinates and summary
scalars to the saved tensors, and verifies tensor equality and alias equality after PT reload.
Focused tests cover mapped entropy, uncensored feature mass, one-based maxima, single-particle
shapes, and target output suffixes:

```bash
PYTHONPATH=src:src/preprocess:src/config_setup ~/miniconda3/envs/ad-resilience-xiv/bin/python -m pytest src/tests/test_project_extensions.py -q
```

The export command lives in `src/visualization/cli.py`; read-back validation lives in
`src/visualization/validate.py`. No separate xIV output modules are needed. The particle
VTKs preserve upstream writeParticleVTK behavior, including its negative entropy sign;
summary VTKs call upstream getEntropy, matching the reference example's positive Shannon entropy.
