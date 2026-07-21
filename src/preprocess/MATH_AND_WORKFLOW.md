# Mathematical and workflow walkthrough

## 1. From segmentation voxels to particles

For selected voxel centers `x_i` and labels `l_i`, the native-resolution segmentation
becomes

\[
\mu = \sum_i \Delta V\,\delta_{x_i}\otimes e_{l_i},
\]

where `ΔV` is physical voxel volume and `e_l` is a one-hot feature vector. At block
factor `f`, feature mass is summed within each `f³` block. This preserves regional
volume and represents mixed boundary blocks.

`nu_All` contains fine label masses. `nu_Sub` contains sums over configured groups.

## 2. Coordinate systems

The default `affine_world` mode applies the complete NIfTI affine. This is necessary to
preserve orientation and physical translation across Ding native, OpenNeuro subject,
and MNI template data. `legacy_centered` exists only to reproduce the old scripts.

Same donor does not imply same voxel grid. Use the geometry inspector and documented
transforms.

## 3. xIV-LDDMM normalization

For each particle,

\[
w_i=\sum_f \nu_{if},\qquad \zeta_{if}=\nu_{if}/w_i.
\]

The repository-native `xmodmap.preprocess.makePQ_legacy.makePQ` computes these weights/compositions and jointly transforms source and target coordinates by

\[
\tilde x=(x-m)/s,
\]

where `m` is the minimum of the joint bounding box and `s` its largest side length.
Kernel widths in the config therefore refer to normalized spatial fractions.

## 4. Deformation model

Initial spatial and mass momenta `(p_x, p_w)` generate a Hamiltonian flow. The controls
include a smooth multiscale velocity field plus global rotation, translation, and—when
`isotropic_scale: true`—one common scale component.

The objective combines deformation energy and multiscale feature-valued varifold
mismatch:

\[
L=\gamma H + D_{\mathrm{varifold}}.
\]

Smaller `gamma` permits more deformation relative to data mismatch. Smaller kernel
scales permit finer local fitting but can fit noise or segmentation artifacts.

## 5. Outputs

- `atlas_deformationSummary.pt/.vtk`: forward-deformed source, feature masses, and xmodmap particle mass-ratio proxy.
- `target_deformationSummary*.pt/.vtk`: target carried backward through the estimated
  inverse flow.
- `optimized_variables.pt`: optimized momenta and state.
- `checkpoint.pt`: optimizer state for resume.
- `loss.png`, `logloss.png`: objective trajectories.
- `input_manifest.json`: particle files and their saved metadata.
- `COMPLETED.json`: final loss values.

The public xmodmap function historically named `getJacobian` is exactly

\[
r_i=\frac{\sum_f\nu^D_{if}}{\sum_f\nu^S_{if}}.
\]

It is a transported particle mass ratio. It does **not** calculate spatial derivatives
of the deformation and therefore is not, by itself, the geometric determinant
`det(Dphi)`. New outputs call it `mass_ratio_proxy`; `jac` remains only as a legacy key.
Entropy is computed with xmodmap's native `getEntropy`. Both quantities require
validation before biological interpretation.

## 6. Dense scalar images as particles

For selected scalar-image voxels with original values `I_i`, we first save an explicit
monotone transform

\[
\tilde I_i=\operatorname{clip}\left(\frac{I_i-a}{b-a},0,1\right),
\]

where `a` and `b` are normally robust quantiles. This bounds the numerical feature
range but is not a substitute for MRI harmonization or histological stain normalization.

For a spatial block `B_j`, the compact reconstruction moments are

\[
M_j=\sum_{i\in B_j}\Delta V,
\qquad
Q_j=\sum_{i\in B_j}\Delta V\,\tilde I_i.
\]

They are stored as `nu_intensity[j] = [M_j, Q_j]`. The block's mass-weighted mean
transformed intensity is `Q_j/M_j`. After deformation, kernel interpolation separately
reconstructs the numerator and denominator and divides them.

For registration, intensity is represented by a non-negative bin distribution
`nu_bin`. In hard assignment,

\[
\nu_{jk}=\Delta V\sum_{i\in B_j}\mathbf 1\{b(\tilde I_i)=k\}.
\]

Linear assignment divides each voxel's mass between its two nearest bin centers. In both
cases,

\[
\sum_k \nu_{jk}=M_j,
\]

so total particle weight reflects tissue volume rather than brightness. This is why
`nu_bin` is usually safer than `[M,Q]` as the registration feature matrix.

Intensity-bin matching is only meaningful for comparable, appropriately normalized
contrasts. Equal numeric values in T1, T2, Nissl, and parvalbumin images do not imply
equal tissue states.

## 7. Particle-to-grid reconstruction

For grid location `g` and particle positions `x_j`, Gaussian k-nearest interpolation uses

\[
K_{gj}=\exp\left(-\frac{\lVert g-x_j\rVert^2}{2\sigma^2}\right)
\]

within a finite support radius. Scalar intensity is reconstructed by

\[
\hat I(g)=
\frac{\sum_j K_{gj}Q_j}{\sum_j K_{gj}M_j}.
\]

Categorical features are interpolated columnwise and the maximum-mass label is selected.
The saved mass and support images expose where interpolation is actually supported. A
reference NIfTI grid is preferred so shape, affine, and orientation are not invented by
the reconstruction script.
## 8. Repository-native versus project-extension operations

The active implementation delegates the following operations to xmodmap:

- factor-1 legacy categorical image conversion;
- factor-1 legacy scalar intensity binning;
- `makePQ` weight/composition and unit-box initialization;
- Hamiltonian, shooting, varifold losses, and model optimization;
- entropy and VTK output.

The project retains only operations that are absent or materially insufficient for the
Allen/SEA-AD geometry: full NIfTI affine coordinates, explicit metadata and keys,
arbitrary 3-D block aggregation, masks and saved intensity transforms, and NIfTI
reference-grid reconstruction. `XMODMAP_REUSE_AUDIT.md` gives the function-level
rationale.

