#!/usr/bin/env python3
"""Configurable, documented single-modality particle registration with xIV-LDDMM.

Scientific role
===============
This script estimates a smooth map from a source/atlas particle measure to a target
particle measure when both use the *same feature vocabulary*.  For example, source and
target may both encode the same three anatomical groupings.  It is not yet the
cross-modal Allen-to-transcriptomics model, which needs an optimized source-to-target
feature map and possibly partial-support estimation.

Particle notation
-----------------
For source positions ``S`` and feature masses ``nu_S``:

    w_S[i]       = sum_f nu_S[i, f]
    zeta_S[i, f] = nu_S[i, f] / w_S[i]

``w_S`` is total mass at a particle and ``zeta_S`` is its conditional feature
composition.  xIV-LDDMM jointly rescales source and target coordinates into a unit box,
then optimizes initial momenta ``p_x`` and ``p_w``.  Hamiltonian shooting integrates
those momenta to generate a diffeomorphic flow and global rotation, translation, and
isotropic scale controls.

Objective
---------
The optimized objective is approximately

    L = gamma * H(p, q) + D_varifold(phi . source, target),

where ``H`` penalizes deformation complexity and ``D_varifold`` is a multiscale kernel
distance over positions and feature masses.  ``sigmaRKHS`` sets deformation-field
length scales; ``sigmaVar`` sets comparison length scales.  Because ``makePQ`` first
normalizes the joint spatial extent to roughly [0, 1], these sigmas are fractions of
the combined source-target bounding-box scale, not millimeters.

Coordinate-space warning
------------------------
The Allen Human Reference Atlas 3-D annotation is in MNI ICBM 2009b nonlinear
symmetric template space.  Ding/OpenNeuro subject-space data are not automatically in
that template space.  This script checks saved metadata and refuses mismatches unless
the configuration explicitly allows them.  Allowing a mismatch does not prove that a
registration is scientifically valid.

Usage::

    python src/framework_script_HighFieldToLowFieldIndividual.py \
        --config config/project_config.yaml \
        --registration ding_subject_test
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch

from preprocess.particle_utils import load_particle_archive
from config_setup.project_config import load_config, resolve_path, write_resolved_config
from preprocess.xmodmap_compat import (
    load_xmodmap,
    write_particle_vtk_xyz,
    write_vtk_xyz,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Project YAML configuration.")
    parser.add_argument(
        "--registration", required=True, help="Named record under registrations."
    )
    parser.add_argument(
        "--target", help="Override configured target particle NPZ for a one-off run."
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace a completed output directory."
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume from checkpoint.pt when present."
    )
    return parser.parse_args()


def _registration_record(config: dict[str, Any], name: str) -> dict[str, Any]:
    records = config.get("registrations", {})
    if name not in records:
        available = ", ".join(sorted(records)) or "<none>"
        raise KeyError(f"Unknown registration '{name}'. Available: {available}")
    return dict(records[name])


def _torch_dtype(name: str) -> torch.dtype:
    choices = {"float32": torch.float32, "float64": torch.float64}
    if name not in choices:
        raise ValueError(f"Unsupported torch dtype '{name}'. Use one of {sorted(choices)}.")
    return choices[name]


def _safe_feature_composition(nu: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return total mass ``w`` and row-normalized composition ``zeta`` without NaNs."""

    w = nu.sum(dim=-1, keepdim=True)
    zeta = torch.zeros_like(nu)
    positive = torch.squeeze(w > 0, dim=-1)
    zeta[positive] = nu[positive] / w[positive]
    return w, zeta


def _check_particles(
    source_X: np.ndarray,
    source_nu: np.ndarray,
    target_X: np.ndarray,
    target_nu: np.ndarray,
    source_metadata: dict[str, Any],
    target_metadata: dict[str, Any],
    allow_space_mismatch: bool,
) -> None:
    """Fail early on common input errors that otherwise create opaque optimizer failures."""

    for name, X, nu in (
        ("source", source_X, source_nu),
        ("target", target_X, target_nu),
    ):
        if X.ndim != 2 or X.shape[1] != 3:
            raise ValueError(f"{name} positions need shape [N, 3]; got {X.shape}.")
        if nu.ndim != 2 or nu.shape[0] != X.shape[0]:
            raise ValueError(f"{name} features {nu.shape} do not match positions {X.shape}.")
        if np.any(~np.isfinite(X)) or np.any(~np.isfinite(nu)):
            raise ValueError(f"{name} particles contain NaN or infinite values.")
        if np.any(nu < 0):
            raise ValueError(f"{name} particle feature masses must be non-negative.")
        if np.any(np.sum(nu, axis=1) <= 0):
            raise ValueError(f"{name} contains particles with zero total mass.")

    # SingleModality uses an identity source-to-target feature map, so feature columns
    # must have the same number and, scientifically, the same meaning and order.
    if source_nu.shape[1] != target_nu.shape[1]:
        raise ValueError(
            "Single-modality registration requires equal feature dimensions: "
            f"source has {source_nu.shape[1]}, target has {target_nu.shape[1]}. "
            "Use a documented crosswalk or the cross-modal xIV model instead."
        )

    source_space = source_metadata.get("coordinate_space", "unspecified")
    target_space = target_metadata.get("coordinate_space", "unspecified")
    if source_space != target_space:
        message = (
            f"Particle coordinate spaces differ: source='{source_space}', "
            f"target='{target_space}'."
        )
        if allow_space_mismatch:
            print(f"WARNING: {message} Continuing only because allow_space_mismatch=true.")
        else:
            raise ValueError(
                message
                + " Put both particle sets in a documented common physical frame, or "
                "explicitly allow the mismatch for an exploratory registration."
            )


def _save_original(writeVTK, S, nu_S, T, nu_T, savedir: Path) -> None:
    """Write source and target particles before optimization for visual QC."""

    source_weight = torch.sum(nu_S, dim=-1)
    target_weight = torch.sum(nu_T, dim=-1)
    source_label = torch.argmax(nu_S, dim=-1) + 1.0
    target_label = torch.argmax(nu_T, dim=-1) + 1.0
    writeVTK(
        S,
        [source_weight.cpu().numpy(), source_label.cpu().numpy()],
        ["Weight", "Max_Region"],
        str(savedir / "originalAtlas.vtk"),
    )
    writeVTK(
        T,
        [target_weight.cpu().numpy(), target_label.cpu().numpy()],
        ["Weight", "Max_Region"],
        str(savedir / "originalTarget.vtk"),
    )


def _save_atlas(
    writeParticleVTK,
    writeVTK,
    getMassRatio,
    resizeData,
    qx1,
    qw1,
    zeta_source,
    zeta_source_report,
    scale,
    minimum,
    nu_source_initial,
    savedir: Path,
) -> None:
    """Save the forward-deformed source and deformation-derived summaries.

    ``qx1`` is in normalized coordinates.  ``resizeData`` returns physical coordinates.
    Feature composition is transported with each source particle while total particle
    mass ``qw1`` evolves under the flow.  Recombining them gives the deformed measure
    ``nu_D = qw1 * zeta_source``.
    """

    D = resizeData(qx1, scale, minimum)
    nu_D = torch.squeeze(qw1)[..., None] * zeta_source
    nu_D_report = torch.squeeze(qw1)[..., None] * zeta_source_report

    # Despite its historical function name ``getJacobian``, the public xmodmap
    # implementation returns sum(nu_D)/sum(nu_S) at each particle.  This is a
    # transported mass ratio, not det(D phi) computed from spatial derivatives.
    mass_ratio_proxy = getMassRatio(D, nu_source_initial, nu_D)

    writeParticleVTK(
        D,
        nu_D,
        str(savedir / "atlas_nu_D.vtk"),
        norm=True,
        condense=False,
        featNames=None,
        sW=None,
    )
    summary = {
        "D": D,
        "nu_D": nu_D,
        "nu_D_report": nu_D_report,
        "mass_ratio_proxy": mass_ratio_proxy,
        "jac": mass_ratio_proxy,  # legacy compatibility only
    }
    torch.save(summary, savedir / "atlas_deformationSummary.pt")
    writeVTK(
        D,
        [
            np.sum(nu_D.cpu().numpy(), axis=-1),
            np.argmax(nu_D.cpu().numpy(), axis=-1) + 1,
            np.argmax(nu_D_report.cpu().numpy(), axis=-1) + 1,
            np.squeeze(mass_ratio_proxy.cpu().numpy()),
        ],
        [
            "Weight_AtlasFeatures",
            "MaxVal_AtlasFeatures",
            "MaxVal_AtlasFeatures_Report",
            "MassRatioProxy",
        ],
        str(savedir / "atlas_deformationSummary.vtk"),
    )


def _save_target(
    writeParticleVTK,
    writeVTK,
    getMassRatio,
    getEntropy,
    resizeData,
    target_deformed,
    target_weight_deformed,
    target_composition,
    target_weight_initial,
    scale,
    minimum,
    savedir: Path,
    suffix: str = "",
) -> None:
    """Save target particles carried backward by the inverse estimated flow."""

    target_deformed = resizeData(target_deformed, scale, minimum)
    nu_target_deformed = torch.squeeze(target_weight_deformed)[..., None] * target_composition
    nu_target_initial = target_weight_initial * target_composition
    mass_ratio_proxy = getMassRatio(target_deformed, nu_target_initial, nu_target_deformed)

    writeParticleVTK(
        target_deformed,
        nu_target_deformed,
        str(savedir / f"target_nu_Td{suffix}.vtk"),
        norm=True,
        condense=False,
        featNames=None,
        sW=None,
    )
    torch.save(
        {
            "Td": target_deformed,
            "nu_Td": nu_target_deformed,
            "mass_ratio_proxy": mass_ratio_proxy,
            "jac": mass_ratio_proxy,  # legacy compatibility only
        },
        savedir / f"target_deformationSummary{suffix}.pt",
    )
    writeVTK(
        target_deformed,
        [
            np.sum(nu_target_deformed.cpu().numpy(), axis=-1),
            np.argmax(nu_target_deformed.cpu().numpy(), axis=-1) + 1,
            np.squeeze(mass_ratio_proxy.cpu().numpy()),
            np.squeeze(getEntropy(nu_target_deformed.cpu().numpy())),
        ],
        ["Weight_TargetFeatures", "MaxVal_TargetFeatures", "MassRatioProxy", "Entropy"],
        str(savedir / f"target_deformationSummary{suffix}.vtk"),
    )


def run_registration(
    config: dict[str, Any],
    registration_name: str,
    target_override: str | None,
    overwrite: bool,
    resume: bool,
) -> tuple[float, float, Path]:
    record = _registration_record(config, registration_name)
    model = dict(record.get("model", {}))

    source_file = resolve_path(config, record["source_particles"])
    target_file = resolve_path(config, target_override or record["target_particles"])
    output_root = resolve_path(config, record["output_dir"])
    output_dir = output_root / target_file.stem
    complete_marker = output_dir / "COMPLETED.json"

    if complete_marker.exists() and not overwrite:
        result = json.loads(complete_marker.read_text(encoding="utf-8"))
        return float(result["final_hamiltonian_loss"]), float(result["final_data_loss"]), output_dir
    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_key = str(record.get("feature_key", "nu_Sub"))
    report_feature_key = str(record.get("report_feature_key", "nu_All"))
    source_X, source_nu, source_metadata = load_particle_archive(source_file, feature_key)
    _, source_nu_report, _ = load_particle_archive(source_file, report_feature_key)
    target_X, target_nu, target_metadata = load_particle_archive(target_file, feature_key)

    # A second target representation can carry a denser particle cloud through the
    # inverse deformation for visualization.  It does not change the optimization.
    full_target_file = resolve_path(config, record.get("full_target_particles", target_file))
    full_target_X, full_target_nu, full_target_metadata = load_particle_archive(
        full_target_file, feature_key
    )

    _check_particles(
        source_X,
        source_nu,
        target_X,
        target_nu,
        source_metadata,
        target_metadata,
        bool(record.get("allow_space_mismatch", False)),
    )

    api = load_xmodmap(config)
    xmodmap = api.xmodmap
    getEntropy = api.get_entropy
    getMassRatio = api.get_mass_ratio
    resizeData = api.resize_data
    vtk_convention = str(record.get("vtk_coordinate_convention", "xyz"))

    def writeVTK(points, arrays, names, filename, *args, **kwargs):
        write_vtk_xyz(
            api,
            points,
            arrays,
            names,
            filename,
            coordinate_convention=vtk_convention,
        )

    def writeParticleVTK(
        points, feature_mass, filename, norm=True, condense=False,
        featNames=None, sW=None, *args, **kwargs
    ):
        write_particle_vtk_xyz(
            api,
            points,
            feature_mass,
            filename,
            coordinate_convention=vtk_convention,
            norm=norm,
            condense=condense,
            feature_names=featNames,
            support_weights=sW,
        )

    seed = int(model.get("seed", 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    dtype = _torch_dtype(str(model.get("torch_dtype", "float32")))
    device_name = str(model.get("device", "cpu"))
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested ({device_name}) but CUDA is unavailable.")
    device = torch.device(device_name)

    S = torch.as_tensor(source_X, dtype=dtype, device=device)
    nu_S = torch.as_tensor(source_nu, dtype=dtype, device=device)
    nu_S_report = torch.as_tensor(source_nu_report, dtype=dtype, device=device)
    T = torch.as_tensor(target_X, dtype=dtype, device=device)
    nu_T = torch.as_tensor(target_nu, dtype=dtype, device=device)
    full_T = torch.as_tensor(full_target_X, dtype=dtype, device=device)
    full_nu_T = torch.as_tensor(full_target_nu, dtype=dtype, device=device)

    # Optional barycenter centering is kept configurable for legacy reproduction.  It
    # is normally unnecessary because makePQ jointly translates/scales coordinates.
    if bool(model.get("center_source", False)):
        source_center = torch.mean(S, dim=0)
        S = S - source_center
    else:
        source_center = torch.zeros(3, dtype=dtype, device=device)
    if bool(model.get("center_target", False)):
        target_center = torch.mean(T, dim=0)
        T = T - target_center
        full_T = full_T - target_center
    else:
        target_center = torch.zeros(3, dtype=dtype, device=device)

    _save_original(writeVTK, S, nu_S, T, nu_T, output_dir)

    # Reporting features may be finer than optimization features, but must occur on the
    # same source particles.  Their normalized proportions are passively transported.
    _, zeta_source_report = _safe_feature_composition(nu_S_report)
    full_target_weight, full_target_composition = _safe_feature_composition(full_nu_T)

    # Use the repository-native state/weight initialization used by both the
    # colleague framework and the Allen/BarSeq runtime example.  Although this
    # single-modality model does not optimize Pi_ST or support lambda, makePQ also
    # supplies the canonical particle weights, feature compositions, and joint
    # source-target unit-box normalization.
    (
        w_S,
        w_T,
        zeta_S,
        zeta_T,
        _q0,
        _p0,
        _num_source,
        S_tilde,
        T_tilde,
        scale,
        minimum,
        _pi_st_initial,
        _lambda_initial,
    ) = api.make_pq(
        S,
        nu_S,
        T,
        nu_T,
        lambInit=torch.tensor(0.5, dtype=dtype, device=device),
    )
    if not torch.isfinite(scale) or scale <= 0:
        raise ValueError("xmodmap.makePQ returned a degenerate joint spatial scale.")

    sigma_rkhs = [float(value) for value in model.get("sigma_rkhs", [0.2, 0.1, 0.05])]
    sigma_var = [float(value) for value in model.get("sigma_var", [0.2, 0.05, 0.02])]
    steps = int(model.get("steps", 250))
    c_rotation = float(model.get("c_rotation", 1.0))
    c_translation = float(model.get("c_translation", 1.0))
    c_nonrigid = float(model.get("c_nonrigid", 5.0))
    gamma = float(model.get("gamma", 0.01))
    effective_dimension = int(model.get("effective_dimension", 3))
    single_scale = bool(model.get("isotropic_scale", True))
    shooting_steps = int(model.get("shooting_time_steps", 10))

    # Save all normalization and model constants needed to interpret or replay the run.
    torch.save(
        {
            "s": scale,
            "m": minimum,
            "cA": c_rotation,
            "cT": c_translation,
            "cS": c_nonrigid,
            "sigmaRKHS": sigma_rkhs,
            "sigmaVar": sigma_var,
            "dimEff": effective_dimension,
            "single": single_scale,
            "source_center": source_center,
            "target_center": target_center,
            "gamma": gamma,
            "steps": steps,
            "shooting_time_steps": shooting_steps,
        },
        output_dir / "model_settings.pt",
    )
    write_resolved_config(config, output_dir / "resolved_project_config.json")
    (output_dir / "input_manifest.json").write_text(
        json.dumps(
            {
                "source_particles": str(source_file),
                "target_particles": str(target_file),
                "full_target_particles": str(full_target_file),
                "feature_key": feature_key,
                "report_feature_key": report_feature_key,
                "source_metadata": source_metadata,
                "target_metadata": target_metadata,
                "full_target_metadata": full_target_metadata,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    full_T_tilde = (full_T - minimum) / scale

    # The varifold distance compares two feature-valued measures over several Gaussian
    # spatial scales.  normalize_across_scale chooses beta coefficients so that no one
    # scale dominates merely because of its numerical magnitude.
    data_loss = xmodmap.distance.LossVarifoldNorm(sigma_var, w_T, zeta_T, T_tilde)
    identity_features = torch.eye(zeta_S.shape[-1], dtype=dtype, device=device)
    data_loss.normalize_across_scale(S_tilde, w_S, zeta_S, identity_features)
    data_loss.weight = float(model.get("data_loss_weight", 1.0))

    # Hamiltonian controls include a smooth kernel velocity field plus global rotation,
    # translation, and (when single=True) one isotropic scale degree of freedom.
    hamiltonian = xmodmap.deformation.Hamiltonian(
        sigma_rkhs,
        S_tilde,
        cA=c_rotation,
        cS=c_nonrigid,
        cT=c_translation,
        dimEff=effective_dimension,
        single=single_scale,
    )
    hamiltonian.weight = gamma
    shooting = xmodmap.deformation.Shooting(
        sigma_rkhs,
        S_tilde,
        cA=c_rotation,
        cS=c_nonrigid,
        cT=c_translation,
        dimEff=effective_dimension,
        single=single_scale,
        nt=shooting_steps,
    )

    # p_x and p_w are the only optimized variables.  q_x and q_w begin at the source
    # positions and masses and are evolved by shooting.  zeta_S is fixed because this
    # is a single-modality model: anatomical feature identity is preserved.
    variables = {
        "px": torch.zeros_like(S_tilde).requires_grad_(True),
        "pw": torch.zeros_like(w_S).requires_grad_(True),
        "qx": S_tilde.clone().detach().requires_grad_(True),
        "qw": w_S.clone().detach().requires_grad_(True),
        "zeta_S": zeta_S,
    }
    k_scale = torch.tensor(float(model.get("precondition_scale", 1.0)), dtype=dtype, device=device)
    preconditioner = {
        "px": torch.rsqrt(k_scale),
        # Division by w_S makes momentum scaling less sensitive to particle mass.
        "pw": torch.rsqrt(k_scale) / effective_dimension / w_S,
    }

    loss = xmodmap.model.SingleModality(hamiltonian, shooting, data_loss)
    loss.init(variables, ["px", "pw"], precond=preconditioner, savedir=str(output_dir))
    checkpoint = output_dir / "checkpoint.pt"
    if resume and checkpoint.is_file():
        loss.resume(variables, str(checkpoint))
    loss.optimize(steps)

    optimized = loss.get_variables_optimized()
    torch.save(optimized, output_dir / "optimized_variables.pt")
    px1, pw1, qx1, qw1 = shooting(
        optimized["px"], optimized["pw"], optimized["qx"], optimized["qw"]
    )[-1]

    # Integrate the target through the reverse flow.  This gives a useful inverse-space
    # QC view, although exact numerical inverse consistency should be assessed rather
    # than assumed.
    shooting_back = xmodmap.deformation.ShootingBackwards(
        sigma_rkhs,
        S_tilde,
        cA=c_rotation,
        cS=c_nonrigid,
        cT=c_translation,
        dimEff=effective_dimension,
        single=single_scale,
        nt=shooting_steps,
    )
    _, _, _, _, target_deformed, target_weight_deformed = shooting_back(
        px1, pw1, qx1, qw1, T_tilde, w_T
    )[-1]

    _save_atlas(
        writeParticleVTK,
        writeVTK,
        getMassRatio,
        resizeData,
        qx1.detach(),
        qw1.detach(),
        zeta_S,
        zeta_source_report,
        scale,
        minimum,
        nu_S,
        output_dir,
    )
    _save_target(
        writeParticleVTK,
        writeVTK,
        getMassRatio,
        getEntropy,
        resizeData,
        target_deformed.detach(),
        target_weight_deformed.detach(),
        zeta_T,
        w_T,
        scale,
        minimum,
        output_dir,
    )

    _, _, _, _, full_target_deformed, full_target_weight_deformed = shooting_back(
        px1, pw1, qx1, qw1, full_T_tilde, full_target_weight
    )[-1]
    _save_target(
        writeParticleVTK,
        writeVTK,
        getMassRatio,
        getEntropy,
        resizeData,
        full_target_deformed.detach(),
        full_target_weight_deformed.detach(),
        full_target_composition,
        full_target_weight,
        scale,
        minimum,
        output_dir,
        suffix="_full",
    )

    figure = loss.print_log()
    figure.savefig(output_dir / "loss.png", dpi=300, bbox_inches="tight")
    figure = loss.print_log(logScale=True)
    figure.savefig(output_dir / "logloss.png", dpi=300, bbox_inches="tight")

    # The public unpack branch does not support print_log(return_HD=True).  Loss values
    # are available directly from Model.log, which stores [Hamiltonian, data] per step.
    if not loss.log:
        raise RuntimeError("Optimization finished without a recorded loss.")
    final_hamiltonian = float(loss.log[-1][0].item())
    final_data = float(loss.log[-1][1].item())
    completion = {
        "registration": registration_name,
        "final_hamiltonian_loss": final_hamiltonian,
        "final_data_loss": final_data,
        "iterations_recorded": len(loss.log),
    }
    complete_marker.write_text(json.dumps(completion, indent=2) + "\n", encoding="utf-8")
    return final_hamiltonian, final_data, output_dir


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    hamiltonian, data, output = run_registration(
        config,
        args.registration,
        args.target,
        args.overwrite,
        args.resume,
    )
    print(
        json.dumps(
            {
                "output_dir": str(output),
                "final_hamiltonian_loss": hamiltonian,
                "final_data_loss": data,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
