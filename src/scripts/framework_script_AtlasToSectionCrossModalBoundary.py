#!/usr/bin/env python3
"""Atlas plane -> MERFISH section registration with xIV-LDDMM and support censoring.

Scientific role
===============
The source is a provisional 2-D plane cut from a 3-D atlas label; the target is a
binned MERFISH section.  The two measures have different feature spaces (atlas
regions vs cell classes) and different supports (the atlas plane covers much more
tissue than the section).  The model therefore follows
``vendor/xIV-LDDMM-Particle/examples/framework_script_densSupport_halfBrainBarSeqToAllen.py``:

    L = gamma*H(p, q) + D_var(phi . S restricted by support, T; pi_ST)
        + gamma*0.1/log(F) * KL(pi_ST) + gamma*R(lambda),

with ``xmodmap.model.CrossModalityBoundary``.  ``LossVarifoldNormBoundary_Partial``
multiplies each deformed source particle by a support weight computed from the local
density of target particles (min-max normalized over the current source particles).
In 2-D, xmodmap's slab normals are zero, so this density weight is the whole support
model and ``lambda`` has no effect; ``lambda`` is therefore held fixed.

What this script is *not*
-------------------------
With one atlas feature, ``pi_ST`` is a single-region aggregate, not a regional feature
law.  The atlas plane is a hand-chosen hypothesis, so a good fit does not locate the
section in MNI space.  Level 2 (geometric plausibility) is never certified here; the
outputs support a manual assessment.

Stages
------
``support-check``  evaluate the density support and data-loss gradients at the
                   reference start, after a 0.5 mm shift, and after a small
                   deformation; fail loudly on non-finite or constant weights.
``single``         fit one start (``--init``, default ``reference_initialization``).
``multistart``     fit every configured start and rank them with a fixed-scale
                   score, because xmodmap's training loss normalizes each start
                   differently.

Initialization ids ``rDDD_fK``: the centered target is reflected (``y -> -y``) when
K = 1, then rotated by DDD degrees.  This is discrete preprocessing, not part of the
diffeomorphic flow, and is recorded in ``transform_chain.json``.

Usage::

    PYTHONPATH=src:src/config_setup python src/scripts/framework_script_AtlasToSectionCrossModalBoundary.py \
        --config configs/project_config.yaml \
        --registration ahrav2_mtg_plane_to_strip_1194111462 --stage support-check
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
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial import ConvexHull

from config_setup.project_config import load_config, resolve_path, write_resolved_config
from preprocess.particle_utils import load_particle_archive, varifold_score_mm
from preprocess.xmodmap_compat import load_xmodmap, write_particle_vtk_xyz, write_vtk_xyz
from scripts.framework_script_HighFieldToLowFieldIndividual import (
    _registration_record,
    _save_original,
    _save_target,
    _torch_dtype,
)

DEPTH_BINS_MM = [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 5.0)]
CLASS_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="Project YAML configuration.")
    parser.add_argument("--registration", required=True, help="Named record under registrations.")
    parser.add_argument("--stage", required=True, choices=["support-check", "single", "multistart"])
    parser.add_argument("--init", help="Initialization id for --stage single (default: reference).")
    parser.add_argument("--steps", type=int, help="Override model.steps (short feasibility runs).")
    parser.add_argument("--overwrite", action="store_true", help="Replace a completed initialization.")
    return parser.parse_args()


# --------------------------------------------------------------------------------------
# Inputs and the discrete initial transform
# --------------------------------------------------------------------------------------

def initial_transform(init_id: str) -> np.ndarray:
    """2x2 matrix M for ``rDDD_fK``: reflect y when K = 1, then rotate by DDD degrees."""

    rotation, reflection = init_id.split("_")
    angle = np.deg2rad(float(rotation[1:]))
    R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    F = np.diag([1.0, -1.0]) if reflection == "f1" else np.eye(2)
    return R @ F


def load_inputs(config: dict, record: dict) -> dict[str, Any]:
    """Load 2-D source/target measures, center each, and read the QC strip cells."""

    source_file = resolve_path(config, record["source_particles"])
    target_file = resolve_path(config, record["target_particles"])
    S, nu_S, source_meta = load_particle_archive(source_file, record.get("feature_key", "nu_Sub"))
    T, nu_T, target_meta = load_particle_archive(target_file, record.get("feature_key", "nu_Sub"))
    with np.load(target_file, allow_pickle=False) as archive:
        target_names = [str(v) for v in archive["coarse_feature_names"]]
    with np.load(source_file, allow_pickle=False) as archive:
        source_names = [str(v) for v in archive["coarse_feature_names"]]
    for name, X in (("source", S), ("target", T)):
        if not np.allclose(X[:, 2], 0):
            raise ValueError(f"{name} particles are not planar (z != 0); prepare them first.")
    if source_meta.get("coordinate_space") != "mni_icbm2009b_symmetric_plane":
        raise ValueError(f"Unexpected source space {source_meta.get('coordinate_space')!r}.")
    if target_meta.get("coordinate_space") != "seaad_section_mm":
        raise ValueError(f"Unexpected target space {target_meta.get('coordinate_space')!r}.")

    mode = record.get("target_feature_mode", "classes")
    if mode == "total_mass":
        nu_T = nu_T.sum(axis=1, keepdims=True)
        target_names = ["total_mass"]
    elif mode != "classes":
        raise ValueError("target_feature_mode must be 'classes' or 'total_mass'.")

    with np.load(resolve_path(config, record["qc_strip_cells"]), allow_pickle=False) as archive:
        strip_X, strip_depth = archive["X"][:, :2].astype(np.float64), archive["depth_from_pia_um"].astype(np.float64)

    S2, T2 = S[:, :2].astype(np.float64), T[:, :2].astype(np.float64)
    source_center, target_center = S2.mean(axis=0), T2.mean(axis=0)
    return {
        "source_file": source_file, "target_file": target_file,
        "source_meta": source_meta, "target_meta": target_meta,
        "S": S2 - source_center, "nu_S": nu_S.astype(np.float64),
        "T": T2 - target_center, "nu_T": nu_T.astype(np.float64),
        "source_center": source_center, "target_center": target_center,
        "strip_X": strip_X - target_center, "strip_depth_um": strip_depth,
        "source_names": source_names, "target_names": target_names, "mode": mode,
    }


# --------------------------------------------------------------------------------------
# Model construction (shared by every stage)
# --------------------------------------------------------------------------------------

def build_model(api, inputs: dict, model: dict, init_id: str, device, dtype) -> dict[str, Any]:
    xmodmap = api.xmodmap
    M = initial_transform(init_id)
    as_tensor = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=dtype, device=device)
    S, nu_S = as_tensor(inputs["S"]), as_tensor(inputs["nu_S"])
    T, nu_T = as_tensor(inputs["T"] @ M.T), as_tensor(inputs["nu_T"])
    csqpi, csqlamb = float(model["csqpi"]), float(model["csqlamb"])
    lamb_init = torch.tensor(float(model["lamb_init"]), dtype=dtype, device=device)

    (w_S, w_T, zeta_S, zeta_T, _q0, _p0, _numS, S_tilde, T_tilde, s, m, pi_init, _lamb0) = api.make_pq(
        S, nu_S, T, nu_T, Csqpi=csqpi, lambInit=lamb_init, Csqlamb=csqlamb
    )
    s_mm = float(s)
    if not np.isfinite(s_mm) or s_mm <= 0:
        raise ValueError("makePQ returned a degenerate joint scale.")
    to_unit = lambda values: [float(v) / s_mm for v in values]
    sigma_var, sigma_rkhs = to_unit(model["sigma_var_mm"]), to_unit(model["sigma_rkhs_mm"])
    dens_sigma = float(model["dens_sigma_mm"]) / s_mm
    gamma = float(model["gamma"])
    geometry_only = inputs["mode"] == "total_mass"
    if geometry_only:
        pi_init = torch.ones((1, 1), dtype=dtype, device=device)

    data_loss = xmodmap.distance.LossVarifoldNormBoundary_Partial(sigma_var, w_T, zeta_T, T_tilde, dens_sigma)
    data_loss.normalize_across_scale(S_tilde, w_S, zeta_S, pi_init, lamb_init)
    data_loss.weight = 1.0

    if geometry_only:
        # F = 1: no feature law to regularize; pi is fixed to identity below.
        pi_loss = xmodmap.distance.PiRegularizationSystemDummy(zeta_S, nu_T)
        pi_loss.weight = 0.0
    else:
        pi_loss = xmodmap.distance.PiRegularizationSystem(zeta_S, nu_T)
        pi_loss.weight = gamma * 0.1 / float(np.log(nu_T.shape[-1]))
    lamb_loss = xmodmap.distance.SupportRestrictionReg(torch.tensor(float(model["eta0"]), dtype=dtype, device=device))
    lamb_loss.weight = gamma

    deformation = dict(cA=float(model["c_rotation"]), cS=float(model["c_nonrigid"]), cT=float(model["c_translation"]),
                       dimEff=2, single=False)
    nt = int(model.get("shooting_time_steps", 10))
    hamiltonian = xmodmap.deformation.Hamiltonian(sigma_rkhs, S_tilde, **deformation)
    hamiltonian.weight = gamma
    shooting = xmodmap.deformation.Shooting(sigma_rkhs, S_tilde, nt=nt, **deformation)
    shooting_back = xmodmap.deformation.ShootingBackwards(sigma_rkhs, S_tilde, nt=nt, **deformation)

    k_scale = torch.tensor(float(model.get("precondition_scale", 1.0)), dtype=dtype, device=device)
    variables = {
        "qx": S_tilde.clone().detach().requires_grad_(True),
        "qw": w_S.clone().detach().requires_grad_(True),
        "px": torch.zeros_like(S_tilde).requires_grad_(True),
        "pw": torch.zeros_like(w_S).requires_grad_(True),
        # Square-root parameterization: the loss sees (pi_ST * csqpi)^2 = pi.
        "pi_ST": ((1.0 / csqpi) * torch.sqrt(pi_init)).clone().detach().requires_grad_(not geometry_only),
        "zeta_S": zeta_S,
        # lambda is inert for 2-D density support; fixed at lamb_init, not optimized.
        "lamb": (lamb_init.sqrt() / csqlamb).clone().detach(),
    }
    precond = {"px": torch.rsqrt(k_scale), "pw": torch.rsqrt(k_scale) / 2.0 / w_S, "pi_ST": csqpi, "lamb": csqlamb}
    to_optimize = ["px", "pw"] if geometry_only else ["px", "pw", "pi_ST"]
    return {
        "M": M, "S": S, "T": T, "nu_S": nu_S, "nu_T": nu_T, "w_S": w_S, "w_T": w_T,
        "zeta_S": zeta_S, "zeta_T": zeta_T, "S_tilde": S_tilde, "T_tilde": T_tilde, "s": s, "m": m,
        "pi_init": pi_init, "lamb_init": lamb_init, "data_loss": data_loss, "pi_loss": pi_loss,
        "lamb_loss": lamb_loss, "hamiltonian": hamiltonian, "shooting": shooting, "shooting_back": shooting_back,
        "variables": variables, "precond": precond, "to_optimize": to_optimize,
        "sigma_var_unit": sigma_var, "sigma_rkhs_unit": sigma_rkhs, "dens_sigma_unit": dens_sigma,
        "geometry_only": geometry_only,
    }


def raw_density(data_loss, qx: torch.Tensor) -> torch.Tensor:
    """Unnormalized target density at qx (what densityWeight min-max normalizes)."""

    ones = torch.ones((qx.shape[0], 1), dtype=qx.dtype, device=qx.device)
    return data_loss.Kd(qx, data_loss.Ttilde, ones, data_loss.w_T)


# --------------------------------------------------------------------------------------
# Stage: support-check
# --------------------------------------------------------------------------------------

def support_check(built: dict, output_dir: Path) -> dict[str, Any]:
    s = float(built["s"])
    S0 = built["S_tilde"].detach()
    center = S0.mean(dim=0, keepdim=True)
    shift = torch.tensor([[0.5 / s, 0.0]], dtype=S0.dtype, device=S0.device)
    wavelength = 10.0 / s  # 10 mm
    bump = (0.5 / s) * torch.sin(2.0 * np.pi * (S0 - center).flip(-1) / wavelength)
    cases = {
        "initial": S0,
        "rigid_shift_0.5mm": S0 + shift,
        "scale_1.1_plus_0.5mm_sinusoid": center + 1.1 * (S0 - center) + bump,
    }
    preconditioned = {key: value * built["precond"].get(key, 1.0) for key, value in built["variables"].items()}
    report: dict[str, Any] = {"weights_are": "relative (min-max over current source particles), not coverage probabilities"}
    failures = []
    for name, positions in cases.items():
        qx = positions.clone().requires_grad_(True)
        qw = built["w_S"].clone().detach().requires_grad_(True)
        density = raw_density(built["data_loss"], qx.detach())
        weights = built["data_loss"].supportWeight(qx.detach(), preconditioned["lamb"] ** 2)
        loss = built["data_loss"](qx, qw, built["zeta_S"], preconditioned["pi_ST"].detach(), preconditioned["lamb"])
        loss.backward()
        d = density.detach().cpu().numpy().ravel()
        w = weights.detach().cpu().numpy().ravel()
        gx, gw = qx.grad.detach().cpu().numpy(), qw.grad.detach().cpu().numpy()
        entry = {
            "density_min": float(d.min()), "density_max": float(d.max()),
            "density_relative_range": float((d.max() - d.min()) / max(abs(d.max()), 1e-30)),
            "weight_min": float(w.min()), "weight_max": float(w.max()), "weight_std": float(w.std()),
            "weight_fraction_above_0.5": float(np.mean(w > 0.5)),
            "data_loss": float(loss.detach()),
            "grad_qx_norm": float(np.linalg.norm(gx)), "grad_qw_norm": float(np.linalg.norm(gw)),
        }
        checks = {
            "weights_finite": bool(np.all(np.isfinite(w))),
            "density_range_nonzero": entry["density_relative_range"] > 1e-6,
            "weights_vary": entry["weight_std"] > 0,
            "loss_finite": bool(np.isfinite(entry["data_loss"])),
            "gradients_finite": bool(np.all(np.isfinite(gx)) and np.all(np.isfinite(gw))),
            "gradients_nonzero": entry["grad_qx_norm"] > 0 and entry["grad_qw_norm"] > 0,
        }
        entry["checks"] = checks
        failures += [f"{name}: {key}" for key, ok in checks.items() if not ok]
        report[name] = entry
    report["failures"] = failures
    (output_dir / "support_check.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


# --------------------------------------------------------------------------------------
# Stage: fit one initialization
# --------------------------------------------------------------------------------------

def _pad3(points: torch.Tensor | np.ndarray):
    if torch.is_tensor(points):
        return torch.cat([points, torch.zeros_like(points[:, :1])], dim=1) if points.shape[1] == 2 else points
    points = np.asarray(points)
    return np.column_stack([points, np.zeros(len(points))]) if points.shape[1] == 2 else points


def _writers(api, convention: str):
    def writeVTK(points, arrays, names, filename, *args, **kwargs):
        write_vtk_xyz(api, _pad3(points), arrays, names, filename, coordinate_convention=convention)

    def writeParticleVTK(points, feature_mass, filename, norm=True, condense=False, featNames=None, sW=None,
                         *args, **kwargs):
        write_particle_vtk_xyz(api, _pad3(points), feature_mass, filename, coordinate_convention=convention,
                               norm=norm, condense=condense, feature_names=featNames, support_weights=sW)

    return writeVTK, writeParticleVTK


def _write_matrix_csv(path: Path, matrix: np.ndarray, rows: list[str], columns: list[str], comment: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as stream:
        stream.write(f"# {comment}\n")
        writer = csv.writer(stream)
        writer.writerow(["source_feature", *columns])
        for name, values in zip(rows, matrix):
            writer.writerow([name, *[f"{v:.6g}" for v in values]])


def _hull_area(points: np.ndarray) -> float | None:
    return float(ConvexHull(points).volume) if len(points) >= 3 else None


def _qc_warp(path, D_canon, nu_D_weight, support, T_canon, nu_T, target_names, strip_back, S_plane, title):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
    ax = axes[0]
    dominant = np.argmax(nu_T, axis=1)
    for index, name in enumerate(target_names):
        sel = dominant == index
        if np.any(sel):
            ax.scatter(T_canon[sel, 0], T_canon[sel, 1], s=14, marker="s", color=CLASS_COLORS[index % 7],
                       linewidths=0, alpha=0.5, label=f"target: {name}")
    shown = ax.scatter(D_canon[:, 0], D_canon[:, 1], s=5, c=support, cmap="Greys", vmin=0, vmax=1,
                       edgecolors="#1a1a19", linewidths=0.2, label="deformed atlas (shade = relative support)")
    fig.colorbar(shown, ax=ax, shrink=0.6, label="relative support weight")
    ax.set_aspect("equal")
    ax.set_title("Deformed atlas plane in the section frame (mm)", fontsize=10)
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.08), fontsize=7, frameon=False, ncol=2)

    ax = axes[1]
    ax.scatter(S_plane[:, 0], S_plane[:, 1], s=5, marker="s", color="#c3c2b7", linewidths=0, label="atlas plane (MNI)")
    shown = ax.scatter(strip_back[:, 0], strip_back[:, 1], s=1, c=strip_back[:, 2], cmap="viridis", linewidths=0,
                       label="strip cells pulled back")
    fig.colorbar(shown, ax=ax, shrink=0.6, label="depth from pia (mm)")
    ax.set_aspect("equal")
    ax.set_title("Strip cells pulled back onto the atlas plane", fontsize=10)
    ax.set_xlabel("MNI x (mm)")
    ax.set_ylabel("MNI z (mm)")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.08), fontsize=7, frameon=False)
    fig.suptitle(title, fontsize=11)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fit_initialization(api, config, record, inputs, model, init_id, steps, output_dir, device, dtype) -> dict:
    xmodmap = api.xmodmap
    convention = str(record.get("vtk_coordinate_convention", "xyz"))
    writeVTK, writeParticleVTK = _writers(api, convention)
    built = build_model(api, inputs, model, init_id, device, dtype)
    s, m = built["s"], built["m"]

    _save_original(writeVTK, built["S"], built["nu_S"], built["T"], built["nu_T"], output_dir)
    loss = xmodmap.model.CrossModalityBoundary(
        built["hamiltonian"], built["shooting"], built["data_loss"], built["pi_loss"], built["lamb_loss"]
    )
    loss.init(built["variables"], built["to_optimize"], precond=built["precond"], savedir=str(output_dir))
    loss.optimize(steps)
    if not loss.log:
        raise RuntimeError("Optimization finished without a recorded loss.")

    optimized = loss.get_variables_optimized()
    torch.save({k: (v.detach() if torch.is_tensor(v) else v) for k, v in optimized.items()},
               output_dir / "optimized_variables.pt")
    # Keep the graph on the shot state: ShootingBackwards differentiates the Hamiltonian.
    shot = built["shooting"](optimized["px"], optimized["pw"], optimized["qx"], optimized["qw"])[-1]
    px1, pw1 = shot[0], shot[1]
    qx1, qw1 = shot[2].detach(), shot[3].detach()
    pi = (optimized["pi_ST"].detach() ** 2)
    support = built["data_loss"].supportWeight(qx1, optimized["lamb"].detach() ** 2).detach()
    density = raw_density(built["data_loss"], qx1).detach()

    # Deformed source in the (initialized) target frame, mm.
    D = api.resize_data(qx1, s, m)
    nu_D = torch.squeeze(qw1)[..., None] * built["zeta_S"]
    nu_Dpi = nu_D @ pi
    mass_ratio = api.get_mass_ratio(D, built["nu_S"], nu_D)
    writeParticleVTK(D, nu_D, str(output_dir / "atlas_nu_D.vtk"), norm=True, featNames=inputs["source_names"])
    writeParticleVTK(D, nu_Dpi, str(output_dir / "atlas_nu_Dpi.vtk"), norm=True, featNames=inputs["target_names"])
    writeVTK(D, [nu_D.sum(-1).cpu().numpy(), nu_Dpi.sum(-1).cpu().numpy(),
                 np.argmax(nu_Dpi.cpu().numpy(), axis=-1) + 1, np.squeeze(mass_ratio.cpu().numpy()),
                 np.squeeze(support.cpu().numpy())],
             ["Weight_AtlasFeatures", "Weight_TargetFeatures", "MaxVal_TargetFeatures", "MassRatioProxy",
              "RelativeSupportWeight"], str(output_dir / "atlas_deformationSummary.vtk"))
    torch.save({"D": D, "nu_D": nu_D, "nu_Dpi": nu_Dpi, "mass_ratio_proxy": mass_ratio,
                "relative_support_weight": support, "raw_density": density},
               output_dir / "atlas_deformationSummary.pt")

    # Target bins and strip cells carried back to the atlas plane.
    _, _, _, _, Td, wTd = built["shooting_back"](px1, pw1, shot[2], shot[3], built["T_tilde"], built["w_T"])[-1]
    _save_target(writeParticleVTK, writeVTK, api.get_mass_ratio, api.get_entropy, api.resize_data,
                 Td.detach(), wTd.detach(), built["zeta_T"], built["w_T"], s, m, output_dir)
    strip_init = torch.as_tensor(inputs["strip_X"] @ built["M"].T, dtype=dtype, device=device)
    strip_tilde = (strip_init - m) / s
    strip_w = torch.ones((len(strip_tilde), 1), dtype=dtype, device=device)
    _, _, _, _, strip_back, _ = built["shooting_back"](px1, pw1, shot[2], shot[3], strip_tilde, strip_w)[-1]
    strip_back_plane = api.resize_data(strip_back.detach(), s, m).cpu().numpy() + inputs["source_center"]

    # ---- canonical-frame scores (fixed scales; comparable across starts and models) ----
    M = built["M"]
    D_canon = D.cpu().numpy() @ M  # undo the discrete transform (M orthogonal)
    sw = support.cpu().numpy()
    nu_Dpi_np = nu_Dpi.cpu().numpy()
    T_canon, nu_T_np = inputs["T"], inputs["nu_T"]
    sigmas = [float(v) for v in model.get("score_sigmas_mm", [2.0, 1.0, 0.5])]
    scores = {
        "score_total_mass": varifold_score_mm(D_canon, nu_Dpi_np.sum(1, keepdims=True) * sw,
                                              T_canon, nu_T_np.sum(1, keepdims=True), sigmas),
        "score_classes": (None if built["geometry_only"]
                          else varifold_score_mm(D_canon, nu_Dpi_np * sw, T_canon, nu_T_np, sigmas)),
        "score_sigmas_mm": sigmas,
        "definition": "mean over sigma of |mu_src - mu_tgt|^2 / |mu_tgt|^2; source = support-weighted, pi-mapped "
                      "deformed atlas; canonical (centered, untransformed) target frame",
    }
    final = [float(v) for v in loss.log[-1]]
    scores["native_losses"] = dict(zip(["hamiltonian", "data", "pi", "lambda"], final))
    scores["native_total_loss"] = float(sum(final))
    (output_dir / "scores.json").write_text(json.dumps(scores, indent=2) + "\n", encoding="utf-8")

    # ---- pi, in interpretable form ----
    pi_np = pi.cpu().numpy()
    _write_matrix_csv(output_dir / "pi_ST.csv", pi_np, inputs["source_names"], inputs["target_names"],
                      "learned non-negative cross-modal coefficients (squared pi_ST); row sums are not constrained"
                      + ("; fixed to identity (geometry-only)" if built["geometry_only"] else ""))
    row_sum = pi_np.sum(axis=1, keepdims=True)
    _write_matrix_csv(output_dir / "pi_ST_row_composition.csv", pi_np / np.where(row_sum > 0, row_sum, 1.0),
                      inputs["source_names"], inputs["target_names"],
                      "row-normalized pi; single-region aggregate; not a regional feature law")

    # ---- plausibility diagnostics (numbers only; Level 2 is a manual judgement) ----
    S_np, ratio = built["S"].cpu().numpy(), np.squeeze(mass_ratio.cpu().numpy())
    supported = sw.ravel() > 0.5
    support_area = float((nu_D.sum(-1).cpu().numpy() * sw.ravel()).sum())
    target_area = float(nu_T_np.sum())
    plane_keys = {tuple(k) for k in np.round((S_np + inputs["source_center"]) / 0.5).astype(int).tolist()}
    strip_keys = np.round(strip_back_plane / 0.5).astype(int)
    inside = np.array([tuple(k) in plane_keys for k in strip_keys.tolist()])
    depth_mm = inputs["strip_depth_um"] / 1000.0
    depth_profile = {}
    for low, high in DEPTH_BINS_MM:
        sel = (depth_mm >= low) & (depth_mm < high)
        depth_profile[f"{low:g}-{high:g}mm"] = {
            "cells": int(sel.sum()), "fraction_in_rasterized_label": float(inside[sel].mean()) if sel.any() else None,
        }
    quantiles = lambda values: dict(zip(["p5", "p50", "p95"], np.percentile(values, [5, 50, 95]).tolist()))
    plausibility = {
        "mass_ratio_proxy_all": quantiles(ratio),
        "mass_ratio_proxy_supported": quantiles(ratio[supported]) if supported.any() else None,
        "supported_particles_fraction": float(supported.mean()),
        "support_weighted_area_mm2": support_area,
        "target_area_mm2": target_area,
        "support_weighted_area_ratio": support_area / target_area,
        "hull_area_ratio_supported": (None if supported.sum() < 3 else
                                      _hull_area(D.cpu().numpy()[supported]) / _hull_area(S_np[supported])),
        "depth_stratified_post_fit_diagnostic": {
            "note": "not independent validation: the fit sees strip positions and classes, not depth; "
                    "'inside' = within the rasterized 0.5 mm label pixels of the atlas plane",
            "bins": depth_profile,
        },
        "relative_support_weight_note": "min-max normalized over current source particles; not coverage probability",
    }
    flags = []
    med = plausibility["mass_ratio_proxy_supported"] or plausibility["mass_ratio_proxy_all"]
    if not 0.5 <= med["p50"] <= 2.0 or med["p5"] < 0.25 or med["p95"] > 4.0:
        flags.append("mass_ratio_proxy outside [0.5, 2] median or [0.25, 4] tails")
    if not 0.5 <= plausibility["support_weighted_area_ratio"] <= 2.0:
        flags.append("support-weighted area ratio outside [0.5, 2]")
    (output_dir / "plausibility.json").write_text(json.dumps(plausibility, indent=2) + "\n", encoding="utf-8")

    # ---- transform chain ----
    chain = {
        "source_plane": inputs["source_meta"].get("plane"),
        "source_plane_to_mni": "MNI = origin + u * u_axis + v * v_axis",
        "source_center_plane_mm": inputs["source_center"].tolist(),
        "target_raw_to_mm_scale": inputs["target_meta"].get("coordinate_scale_from_raw"),
        "target_a_cell_mm2": inputs["target_meta"].get("a_cell_mm2"),
        "target_center_section_mm": inputs["target_center"].tolist(),
        "initialization": init_id,
        "initial_transform_matrix_M": M.tolist(),
        "initial_transform_convention": "T_init = (T_section_mm - target_center) @ M.T; reflection is "
                                        "preprocessing, not part of the LDDMM flow",
        "unit_box": {"s_mm": float(s), "m_mm": m.detach().cpu().numpy().tolist(),
                     "convention": "X_tilde = (X - m) / s"},
        "sigma_mm_to_unit": {"sigma_var": built["sigma_var_unit"], "sigma_rkhs": built["sigma_rkhs_unit"],
                             "dens_sigma": built["dens_sigma_unit"]},
        "optimized_variables": str(output_dir / "optimized_variables.pt"),
        "forward": "Shooting(px, pw, qx, qw) moves the atlas plane into T_init; D = X_tilde * s + m",
        "inverse": "ShootingBackwards(px1, pw1, qx1, qw1, T_tilde, w_T) carries target points to the atlas plane",
    }
    (output_dir / "transform_chain.json").write_text(json.dumps(chain, indent=2) + "\n", encoding="utf-8")
    torch.save({"s": s, "m": m, "model": model, "steps": steps, "init": init_id,
                "sigma_var_unit": built["sigma_var_unit"], "sigma_rkhs_unit": built["sigma_rkhs_unit"],
                "dens_sigma_unit": built["dens_sigma_unit"]}, output_dir / "model_settings.pt")

    for name, log_scale in (("loss.png", False), ("logloss.png", True)):
        figure = loss.print_log(logScale=log_scale)
        figure.savefig(output_dir / name, dpi=200, bbox_inches="tight")
        plt.close(figure)
    strip_back_qc = np.column_stack([strip_back_plane, depth_mm])
    _qc_warp(output_dir / "qc_warp.png", D_canon + inputs["target_center"], None, sw.ravel(),
             T_canon + inputs["target_center"], nu_T_np, inputs["target_names"], strip_back_qc,
             S_np + inputs["source_center"], f"{record['_name']} / {init_id}: total-mass score "
             f"{scores['score_total_mass']:.3f}")

    # ---- Level 1: finite outputs that reload ----
    reloaded = [torch.load(output_dir / f, weights_only=False) for f in
                ("optimized_variables.pt", "atlas_deformationSummary.pt", "target_deformationSummary.pt")]
    finite = all(bool(torch.isfinite(v).all()) for blob in reloaded for v in blob.values() if torch.is_tensor(v))
    finite &= bool(np.isfinite(strip_back_plane).all()) and bool(np.isfinite(pi_np).all())
    completion = {
        "registration": record["_name"], "initialization": init_id, "steps": steps,
        "closure_evaluations_logged": len(loss.log),
        "level1_passed": bool(finite),
        "numeric_flags": flags,
        "level2_assessment": "manual - see qc_warp.png, plausibility.json, and the geometry-only baseline",
        **{k: scores[k] for k in ("score_total_mass", "score_classes", "native_total_loss")},
    }
    (output_dir / "COMPLETED.json").write_text(json.dumps(completion, indent=2) + "\n", encoding="utf-8")
    return completion


def summarize(output_root: Path) -> None:
    """Rebuild initializations.csv and summary.json from every finished start."""

    rows = []
    for marker in sorted(output_root.glob("init_*/COMPLETED.json")):
        result = json.loads(marker.read_text(encoding="utf-8"))
        rows.append({"directory": marker.parent.name, **{k: result[k] for k in (
            "initialization", "steps", "level1_passed", "score_total_mass", "score_classes", "native_total_loss")},
            "numeric_flags": "; ".join(result["numeric_flags"])})
    if not rows:
        return
    with open(output_root / "initializations.csv", "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    ranked = sorted((r for r in rows if r["level1_passed"]),
                    key=lambda r: (r["score_total_mass"], r["score_classes"] if r["score_classes"] is not None else 0))
    summary = {"ranking_score": "score_total_mass (ties: score_classes); native losses are not comparable",
               "best": ranked[0] if ranked else None, "number_finished": len(rows)}
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    record = _registration_record(config, args.registration)
    record["_name"] = args.registration
    model = dict(record["model"])
    steps = int(args.steps if args.steps is not None else model.get("steps", 150))

    seed = int(model.get("seed", 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    api = load_xmodmap(config)  # legacy xmodmap sets the default tensor type on import
    dtype = _torch_dtype(str(model.get("torch_dtype", "float32")))
    device = torch.device(str(model.get("device", "cpu")))
    default_device = torch.empty(0).device
    if device.type != default_device.type:
        raise RuntimeError(f"model.device={device} but legacy xmodmap made {default_device} the default; its "
                           "internal allocations would land on a different device.")

    output_root = resolve_path(config, record["output_dir"])
    output_root.mkdir(parents=True, exist_ok=True)
    write_resolved_config(config, output_root / "resolved_project_config.json")
    inputs = load_inputs(config, record)
    manifest = {"source_particles": str(inputs["source_file"]), "target_particles": str(inputs["target_file"]),
                "target_feature_mode": inputs["mode"], "source_metadata": inputs["source_meta"],
                "target_metadata": inputs["target_meta"]}
    (output_root / "input_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    if args.stage == "support-check":
        built = build_model(api, inputs, model, record["reference_initialization"], device, dtype)
        report = support_check(built, output_root)
        print(json.dumps(report, indent=2))
        if report["failures"]:
            raise SystemExit(f"Support check failed: {report['failures']}")
        return

    inits = record["initializations"] if args.stage == "multistart" else [args.init or record["reference_initialization"]]
    if not (output_root / "support_check.json").is_file() or json.loads(
            (output_root / "support_check.json").read_text(encoding="utf-8"))["failures"]:
        raise SystemExit("Run --stage support-check first (and make it pass).")
    for init_id in inits:
        suffix = "" if args.steps is None else f"_steps{steps}"
        output_dir = output_root / f"init_{init_id}{suffix}"
        if (output_dir / "COMPLETED.json").exists() and not args.overwrite:
            print(f"Skipping completed {output_dir}")
            continue
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True)
        (output_dir / "input_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        result = fit_initialization(api, config, record, inputs, model, init_id, steps, output_dir, device, dtype)
        print(json.dumps(result, indent=2))
    summarize(output_root)


if __name__ == "__main__":
    main()
