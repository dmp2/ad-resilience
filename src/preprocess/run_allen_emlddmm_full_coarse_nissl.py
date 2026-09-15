"""Staged, memory-bounded coarse MRI-to-symmetric-Nissl EM-LDDMM workflow."""
from __future__ import annotations

import argparse, copy, csv, gc, hashlib, json, math, numbers, os, resource, shutil, sys, time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import scipy.ndimage as ndi
import tifffile
import torch
from PIL import Image

from preprocess.build_allen_symmetric_histology import (
    _json, _write_image_sidecar, _write_rows,
)
from preprocess.prepare_allen_emlddmm_inputs import accepted_loader_axes
from preprocess.run_allen_emlddmm import load_physical_rows, load_pinned_mri_image, pinned_emlddmm, sha256_file
from preprocess.visualize_allen_emlddmm_stack import (
    _coordinate_cell_edges,
    _render_saved_transform_stack_overview,
    _uniform_representative_positions,
)
from preprocess.visualize_allen_annotations import (
    _combined_display_map,
    _internal_boundary_mask,
    _label_colors,
    _label_names,
    _merged_colors,
    _multiscale,
)

PROJECT = Path(__file__).resolve().parents[2]
BASELINE_OUTPUT = PROJECT / "results/allen/specimen_708424/emlddmm/full-coarse/HIST_NISSL_to_MRI_7T_WHOLE_eA1e6"
OUTPUT = Path(os.environ.get("EMLDDMM_OUTPUT_ROOT", BASELINE_OUTPUT))
RUN_TMP = Path(os.environ.get("EMLDDMM_RUN_TMP", "/invalid/run-tmp"))
PEAK_FILE = Path(os.environ.get("EMLDDMM_RSS_PEAK_FILE", RUN_TMP / "process_group_peak_rss_kib"))
DEFAULT_DATASET = PROJECT / "data/derivatives/allen/specimen_708424/emlddmm_7t_symmetric"
OBSERVED_LEFT_DATASET = PROJECT / "data/derivatives/allen/specimen_708424/emlddmm_7t"
DATASET = DEFAULT_DATASET
ANNOTATION_DATASET = DATASET
VIEW = DATASET / "inputs/views/HIST_NISSL"
MRI = PROJECT / "data/derivatives/allen/specimen_708424/mri_7t_whole/T1_rot_space-MRI_7T_WHOLE_desc-header-corrected.nii"
MRI_PROV = PROJECT / "data/derivatives/allen/specimen_708424/mri_7t_whole/mri_provenance.json"
INITIAL_A = PROJECT / "results/qc/allen_708424_mri7t_to_symmetric_nissl_initial_similitude.txt"
CHECKPOINTS = OUTPUT / "checkpoints"
ATLAS_DIR = OUTPUT / "section_alignment_atlas_free"
REG_DIR = OUTPUT / "registration"
POST_DIR = OUTPUT / "postprocessed_qc"
ANNOTATION_DIR = OUTPUT / "annotations_on_coarse_mri"
ANNOTATION_ZARR = PROJECT / "data/derivatives/allen/specimen_708424/annotations_ome_zarr"
CORRECTED_NISSL = PROJECT / "data/derivatives/allen/specimen_708424/histology_linear_nissl"
FLAGGED_ALLEN = (1082, 1089, 2055, 2059, 2238, 2242, 2466)
DEFORMATION_DIRECTION = (
    "phi maps MRI_7T_WHOLE output physical coordinates to the pre-affine "
    "MRI-domain coordinates used by the saved sampling chain; the saved "
    "global affine A then maps phi(x) into HIST_SYMMETRIC coordinates"
)
PIN = "864990e0619fcdfb3e22e05298291f439f1b6f3d"
ORIGINAL_A_SHA256 = "3a03af802d8ddfb347801a934080e7db809960c57c00e67e8366f1ec6ffd17eb"


BASE_EXAMPLE_STANDARD = {
    "downI": [[4, 4, 4], [2, 2, 2], [1, 1, 1]],
    "downJ": [[1, 4, 4], [1, 2, 2], [1, 1, 1]],
    "n_iter": [100, 50, 40],
    "v_start": [0, 0, 0],
    "ev": [1e-2, 1e-2, 1e-2],
    "slice_matching": [True, True, True],
    "slice_matching_start": [0, 0, 0],
    "slice_deformation": False,
    "a": 1000.0, # spatial scale of the deformation
    "dv": 2000.0, # voxel size to sample the deformation on
    "eA": 1e7,
    "eA2d": 1e5,
    "Amode": 2,
    "rigid_procrustes": True,
    "auto_stepsize_v": 0,
    "local_contrast": [[1, 32, 32], [1, 32, 32], [1, 32, 32]],
    "order": [1, 1, 1],
    "n_e_step": [3, 3, 3],
    "priors": None,
    "update_priors": True,
    "muA": None,
    "muB": None,
    "update_muA": False,
    "update_muB": False,
    "sigmaM": None,
    "sigmaA": None,
    "sigmaB": None,
    "n_draw": [0, 0, 0],
    "full_outputs": False,
}

PROFILE_OVERRIDES = {
    "example-standard-sigmaR5e4-a2000-dv4000-lc188-linear-no-v": {
        "a": 2000.0,
        "dv": 4000.0,
        "eA": 1e6,
        "Amode": 0,
        "sigmaR": 5e4,
        "n_iter": [100, 50, 40],
        "v_start": [101, 51, 41],
        "local_contrast": [[1, 8, 8], [1, 8, 8], [1, 8, 8]],
        "muA": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        "muB": [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
    },
    "example-standard-sigmaR5e5": {"sigmaR": 5e5},
    "example-standard-sigmaR5e5-contrast-stable-fine100": {
        "sigmaR": 5e5,
        "n_iter": [100, 50, 40],
        "local_contrast": [[1, 32, 45], [1, 65, 91], [1, 65, 91]],
    },
    "example-standard-sigmaR5e5-contrast-stable-fine100": {
        "sigmaR": 5e5,
        # Preserve the exact completed-profile values here.
        "n_iter": [100, 50, 40],
        "local_contrast": [
            [1, 32, 45],
            [1, 65, 91],
            [1, 65, 91],
        ],
    },
    "example-standard-sigmaR5e4-a2000-dv4000": {
            "a": 2000.0,
            "dv": 4000.0,
            "eA": 1e6,
            "sigmaR": 5e4,
            "n_iter": [100, 50, 40],
            "local_contrast": [[1, 16, 16], [1, 16, 16], [1, 16, 16]],
            "muA": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], # estimate of the intensitiy of artifacts (black)
            "muB": [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], # estimate of the intensity of the background (white)
         }, 
    "example-standard-sigmaR5e4-a2000-dv4000-lc188": {
                "a": 2000.0,
                "dv": 4000.0,
                "eA": 1e6,
                "sigmaR": 5e4,
                "n_iter": [100, 50, 40],
                "local_contrast": [[1, 8, 8], [1, 8, 8], [1, 8, 8]],
                "muA": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], # estimate of the intensitiy of artifacts (black)
                "muB": [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], # estimate of the intensity of the background (white)
            }, 
    "example-standard-sigmaR5e4-a1500-dv3000": {
        "a": 1500.0,
        "dv": 3000.0,
        "eA": 1e6,
        "sigmaR": 5e4,
        "n_iter": [100, 50, 40],
        "local_contrast": [[1, 16, 16], [1, 16, 16], [1, 16, 16]],
        "muA": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], # estimate of the intensitiy of artifacts (black)
        "muB": [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], # estimate of the intensity of the background (white)
     }, 
    "example-standard-sigmaR5e6": {"sigmaR": 5e6},
    }

LOCAL_CONTRAST_TARGET_SHAPES = ((32, 45), (65, 91), (130, 182))

_THREE_LEVEL_SCALARS = (
    "a", "dv", 
    "eA", "eA2d", "Amode", "rigid_procrustes",
    "slice_deformation", "auto_stepsize_v", "priors", "update_priors",
    "update_muA", "update_muB", "sigmaM", "sigmaA", "sigmaB",
    "full_outputs", "sigmaR",
)


def _normalize_mixture_means(name: str, value: Any) -> list:
    """Return a validated three-level schedule of three-channel GMM means."""
    if value is None:
        return [None, None, None]
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{name} must contain exactly three levels")
    normalized = copy.deepcopy(value)
    for level, vector in enumerate(normalized):
        if vector is None:
            continue
        if not isinstance(vector, list) or len(vector) != 3:
            raise ValueError(f"{name}[{level}] must contain exactly three channels")
        if any(
            isinstance(component, bool)
            or not isinstance(component, numbers.Real)
            or not math.isfinite(float(component))
            for component in vector
        ):
            raise ValueError(
                f"{name}[{level}] must contain three finite numeric values"
            )
    return normalized


def resolve_registration_profile(name: str) -> dict:
    """Return validated, wrapper-ready optimizer parameters for one profile."""
    try:
        override = PROFILE_OVERRIDES[name]
    except KeyError as exc:
        raise ValueError(f"Unknown registration profile: {name}") from exc
    resolved = copy.deepcopy(BASE_EXAMPLE_STANDARD)
    resolved.update(copy.deepcopy(override))
    for key in _THREE_LEVEL_SCALARS:
        value = resolved[key]
        resolved[key] = [copy.deepcopy(value) for _ in range(3)]
    for key in ("muA", "muB"):
        resolved[key] = _normalize_mixture_means(key, resolved[key])
    for key, value in resolved.items():
        if not isinstance(value, list) or len(value) != 3:
            raise ValueError(f"{key} must contain exactly three levels")
    for key in ("downI", "downJ"):
        for level, value in enumerate(resolved[key]):
            if not isinstance(value, list) or len(value) != 3:
                raise ValueError(f"{key}[{level}] must contain three dimensions")
    for level, (value, target_shape) in enumerate(zip(
        resolved["local_contrast"], LOCAL_CONTRAST_TARGET_SHAPES, strict=True
    )):
        if not isinstance(value, list) or len(value) != 3:
            raise ValueError(f"local_contrast[{level}] must contain three dimensions")
        if any(type(dimension) is not int or dimension <= 0 for dimension in value):
            raise ValueError(
                f"local_contrast[{level}] dimensions must be positive integers"
            )
        if value[0] != 1:
            raise ValueError(
                f"local_contrast[{level}] must not span multiple serial sections"
            )
        if any(
            block > target
            for block, target in zip(value[1:], target_shape, strict=True)
        ):
            raise ValueError(
                f"local_contrast[{level}] in-plane blocks exceed target dimensions"
            )
    if any(value[0] != 1 for value in resolved["downJ"]):
        raise ValueError("Histology serial lattice must not be downsampled")
    if resolved["order"] != [1, 1, 1]:
        raise ValueError("Local contrast requires first-order contrast")
    return resolved


def resolve_registration_execution(name: str, *, native_qc: bool = False) -> dict:
    """Resolve model parameters plus optional native drawing execution settings."""
    resolved = resolve_registration_profile(name)
    resolved["full_outputs"] = [False, False, True]
    if native_qc:
        resolved["n_draw"] = [0, 0, 10]
    return resolved


def registration_output_root(name: str, *, native_qc: bool = False) -> Path:
    """Derive a profile output root without touching the filesystem."""
    resolve_registration_profile(name)
    suffix = f"_{name}" + ("_native-qc" if native_qc else "")
    return BASELINE_OUTPUT.with_name(f"{BASELINE_OUTPUT.name}{suffix}")


NATIVE_QC_FIGURES = (
    ("figJ", "01_reconstructed_nissl.png"),
    ("figI", "02_transformed_mri.png"),
    ("figfI", "03_contrast_predicted_nissl.png"),
    ("figErr", "04_prediction_error.png"),
    ("figW", "05_gmm_weights.png"),
    ("figV", "06_velocity.png"),
    ("figE", "07_energy.png"),
    ("figA", "08_transform_updates.png"),
)
FULL_OUTPUT_RELEASE_KEYS = (
    "WM", "WA", "WB", "W0", "muA", "muB",
    "sigmaA", "sigmaB", "sigmaM", "coeffs",
)


def _save_final_effective_match_weight(
    final: dict, observed: np.ndarray, expected_full_shape: tuple[int, ...],
    output: Path,
) -> Path:
    """Retain only final-scale EM-LDDMM matching probability within W0."""
    matching = finite("final WM", final["WM"])
    support = finite("final W0", final["W0"])
    observed = np.asarray(observed, dtype=np.int64)
    if matching.shape != expected_full_shape or support.shape != expected_full_shape:
        raise RuntimeError(
            f"Final WM/W0 shapes differ: {matching.shape}, {support.shape}, "
            f"expected {expected_full_shape}"
        )
    if observed.shape != (641,) or np.unique(observed).size != 641:
        raise RuntimeError("Final effective match weight requires 641 unique rows")
    effective = np.asarray(
        matching[observed] * support[observed], dtype=np.float32
    )
    expected_observed_shape = (641, *expected_full_shape[1:])
    if effective.shape != expected_observed_shape:
        raise RuntimeError(
            f"Final effective match weight shape is {effective.shape}, "
            f"expected {expected_observed_shape}"
        )
    if not np.all(np.isfinite(effective)) or np.any(effective < 0.0):
        raise RuntimeError("Final effective match weight is nonfinite or negative")
    if output.exists():
        raise RuntimeError(f"Effective match weight output exists: {output}")
    np.save(output, effective)
    return output


def save_native_qc_figures(final: dict) -> None:
    """Save the final scale's latest native draw without altering its figures."""
    expected = {key for key, _ in NATIVE_QC_FIGURES}
    returned = {key for key in final if key.startswith("fig")}
    if returned != expected:
        raise RuntimeError(
            f"Native figure keys differ: expected {sorted(expected)}, got {sorted(returned)}"
        )
    missing_full = [key for key in FULL_OUTPUT_RELEASE_KEYS if key not in final]
    if missing_full:
        raise RuntimeError(f"Missing final full-output entries: {missing_full}")
    output_dir = REG_DIR / "native_qc"
    if output_dir.exists():
        raise RuntimeError(f"Native-QC output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    figures = [final[key] for key, _ in NATIVE_QC_FIGURES]
    try:
        for (key, filename), figure in zip(NATIVE_QC_FIGURES, figures, strict=True):
            figure.savefig(output_dir / filename, dpi=150)
    finally:
        for figure in figures:
            plt.close(figure)
    for key in FULL_OUTPUT_RELEASE_KEYS:
        del final[key]


def canonical_registration_config(config: dict) -> str:
    return json.dumps(config, sort_keys=True, separators=(",", ":"))


def registration_config_sha256(config: dict) -> str:
    canonical = canonical_registration_config(config).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def configure_output_root(path: Path) -> None:
    global OUTPUT, CHECKPOINTS, ATLAS_DIR, REG_DIR, POST_DIR, ANNOTATION_DIR
    OUTPUT = path
    CHECKPOINTS = OUTPUT / "checkpoints"
    ATLAS_DIR = OUTPUT / "section_alignment_atlas_free"
    REG_DIR = OUTPUT / "registration"
    POST_DIR = OUTPUT / "postprocessed_qc"
    ANNOTATION_DIR = OUTPUT / "annotations_on_coarse_mri"


def configure_linear_inputs(dataset: Path, initial_affine: Path) -> None:
    """Opt into a non-default observed-left dataset without changing legacy defaults."""
    global DATASET, VIEW, INITIAL_A
    DATASET = dataset.resolve()
    VIEW = DATASET / "inputs/views/HIST_NISSL"
    INITIAL_A = initial_affine.resolve()
    for required in (
        DATASET / "metadata/physical_sections.tsv",
        VIEW / "samples.tsv",
        INITIAL_A,
    ):
        if not required.exists():
            raise FileNotFoundError(f"Linear-run input is missing: {required}")


def now() -> str: return datetime.now(timezone.utc).isoformat()
def peak_rss_kib() -> int:
    own = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    try: group = int(PEAK_FILE.read_text().strip())
    except Exception: group = 0
    return max(own, group)
def current_rss_kib() -> int:
    """Return current process RSS from Linux procfs, distinct from peak RSS."""
    for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
        if line.startswith("VmRSS:"):
            fields = line.split()
            if len(fields) != 3 or fields[2] != "kB":
                break
            return int(fields[1])
    raise RuntimeError("Could not read current VmRSS from /proc/self/status")
def atomic_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)
def checksum(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024), b""): h.update(chunk)
    return h.hexdigest()
def finite(name: str, x: Any) -> np.ndarray:
    a=x.detach().cpu().numpy() if isinstance(x,torch.Tensor) else np.asarray(x)
    if not np.all(np.isfinite(a)): raise RuntimeError(f"{name} contains nonfinite values")
    return a
def read_samples() -> list[dict[str,str]]:
    with (VIEW/"samples.tsv").open(newline="",encoding="utf-8") as f: return list(csv.DictReader(f,delimiter="\t"))
def load_context():
    if (PROJECT/"configs/emlddmm-upstream-commit.txt").read_text().strip()!=PIN: raise RuntimeError("pin file mismatch")
    em=pinned_emlddmm(); rows=load_physical_rows(DATASET); samples=read_samples()
    if len(rows)!=2846 or len(samples)!=2846: raise RuntimeError("expected 2,846 manifest rows")
    canvas=json.loads((DATASET/"metadata/loader_canvas_audit.json").read_text())
    axes=accepted_loader_axes(rows,canvas)
    observed=np.asarray([i for i,(r,s) in enumerate(zip(rows,samples)) if s["status"]=="present" and r["stain"]=="nissl"],dtype=np.int64)
    if observed.size!=641: raise RuntimeError(f"expected 641 Nissl observations, got {observed.size}")
    return em,rows,samples,axes,observed

def block_axis(axis: np.ndarray, factor: int) -> np.ndarray:
    n=len(axis)//factor
    return np.asarray(axis[:n*factor],dtype=np.float64).reshape(n,factor).mean(1)


def _preserves_source_grid() -> bool:
    audit = json.loads((DATASET / "metadata/loader_canvas_audit.json").read_text())
    return audit.get("preserve_source_grid") is True


def coarse_spatial_axes(axes, factor: int = 4, *, preserve_source_grid: bool = False) -> tuple[np.ndarray, np.ndarray]:
    spatial = tuple(np.asarray(axis, dtype=np.float64) for axis in axes[1:])
    if preserve_source_grid:
        for axis in spatial:
            if axis.ndim != 1 or len(axis) < 2 or not np.all(np.diff(axis) > 0):
                raise RuntimeError("Preserved source-grid axes must increase")
            if not np.allclose(np.diff(axis), np.diff(axis)[0], atol=1e-6, rtol=0.0):
                raise RuntimeError("Preserved source-grid axes must be uniform")
        return spatial
    coarse = block_axis(spatial[0], factor), block_axis(spatial[1], factor)
    shape = tuple(map(len, coarse))
    expected = {"emlddmm_7t": (130, 91), "emlddmm_7t_symmetric": (130, 182)}.get(DATASET.name)
    if expected is not None and shape != expected:
        raise RuntimeError(f"{DATASET.name} blocked spatial axes are {shape}, expected {expected}")
    return coarse
def read_section(sample: dict[str, str], em=None, spatial_axes=None, *, preserve_source_grid: bool = False) -> tuple[np.ndarray, np.ndarray]:
    path=VIEW/sample["sample_id"]
    raw=tifffile.imread(path)
    if raw.dtype==np.uint8: image=raw[...,:3].astype(np.float64)/255.0
    else:
        image=raw[...,:3].astype(np.float64); image/=np.mean(np.abs(image.reshape(-1,3)),axis=0)
    image=image.transpose(2,0,1)
    if preserve_source_grid:
        if spatial_axes is None or image.shape[1:] != tuple(map(len, spatial_axes)):
            raise RuntimeError("Preserved TIFF raster differs from its physical axes")
    elif em is not None and spatial_axes is not None:
        source_y=np.arange(image.shape[1],dtype=np.float64)*200.0-(image.shape[1]-1)*100.0
        source_x=np.arange(image.shape[2],dtype=np.float64)*200.0-(image.shape[2]-1)*100.0
        query=torch.stack(torch.meshgrid(torch.as_tensor(spatial_axes[0]),torch.as_tensor(spatial_axes[1]),indexing="ij"))
        image=em.interp([source_y,source_x],torch.as_tensor(image),query,interp2d=True,padding_mode="zeros").numpy()
    support_path = DATASET / "support/nissl" / path.name
    if support_path.is_file():
        support = tifffile.imread(support_path).astype(np.float32)
        if support.shape != image.shape[1:]:
            raise RuntimeError("Stored validity support differs from Nissl raster")
    else:
        support=(image[0]>0).astype(np.float32)
    return image.astype(np.float32),support
def downsample_section(image: np.ndarray,support: np.ndarray,factor: int=4):
    c,h,w=image.shape; nh,nw=h//factor,w//factor; h4,w4=nh*factor,nw*factor
    sup=support[:h4,:w4].reshape(nh,factor,nw,factor).sum((1,3))
    num=(image[:,:h4,:w4]*support[None,:h4,:w4]).reshape(c,nh,factor,nw,factor).sum((2,4))
    out=np.zeros((c,nh,nw),np.float32); positive=sup>0; out[:,positive]=num[:,positive]/sup[positive]
    return out,(sup/(factor*factor)).astype(np.float32)
def stream_stack(samples, indices, total_rows, shape, em=None, spatial_axes=None, *, preserve_source_grid: bool = False):
    J=np.zeros((3,total_rows,*shape),np.float32); W=np.zeros((total_rows,*shape),np.float32)
    for count,index in enumerate(indices,1):
        image, support = read_section(samples[index], em, spatial_axes, preserve_source_grid=preserve_source_grid)
        if not preserve_source_grid:
            image, support = downsample_section(image, support)
        if image.shape!=(3,*shape) or support.shape!=shape: raise RuntimeError(f"downsample shape error at row {index}")
        destination=count-1 if total_rows==len(indices) else index
        J[:,destination]=image; W[destination]=support
        if count%50==0: print(f"streamed {count}/{len(indices)} sections",flush=True)
    return J,W

def rigid_frame(mats: np.ndarray):
    u,_,vt=np.linalg.svd(mats[:,:2,:2].mean(0)); q=u@vt
    if np.linalg.det(q)<0: u[:,-1]*=-1; q=u@vt
    B=np.eye(3); B[:2,:2]=q; B[:2,2]=np.median(mats[:,:2,2],axis=0); return B

def residuals(path,rows,indices,mats,B):
    R=np.linalg.inv(B)[None]@mats[indices]; err=float(np.max(np.abs(B[None]@R-mats[indices])))
    tx,ty=R[:,0,2],R[:,1,2]; rot=np.degrees(np.arctan2(R[:,1,0],R[:,0,0]))
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f,delimiter="\t"); w.writerow(["physical_index","allen_section","serial_z_um","tx_um","ty_um","rotation_deg"])
        for i,a,b,c in zip(indices,tx,ty,rot): w.writerow([i,rows[i]["allen_section_number"],float(rows[i]["serial_z_center_mm"])*1000,a,b,c])
    return {"tx_um":[float(tx.min()),float(tx.max())],"ty_um":[float(ty.min()),float(ty.max())],"rotation_deg":[float(rot.min()),float(rot.max())],"recomposition_max_abs_error":err}

def mini_equivalence(em,rows,samples,axes,observed,*,preserve_source_grid=False):
    chosen=observed[[0,len(observed)//2,-1]]
    if preserve_source_grid:
        for index in chosen:
            image, support = read_section(samples[index], spatial_axes=[axes[1], axes[2]], preserve_source_grid=True)
            if image.shape != (3, len(axes[1]), len(axes[2])) or support.shape != image.shape[1:]:
                raise RuntimeError("Preserved source raster/axis validation failed")
        return {"physical_indices": chosen.tolist(), "image_max_abs_error": 0.0, "support_max_abs_error": 0.0, "equivalent": True, "method": "direct_preserved_raster"}
    mini=RUN_TMP/"support-equivalence-mini-view"; mini.mkdir()
    with (mini/"samples.tsv").open("w",encoding="utf-8",newline="") as f:
        w=csv.writer(f,delimiter="\t"); w.writerow(["sample_id","participant_id","species","status"])
        for i in chosen:
            s=samples[i]; w.writerow([s["sample_id"],s["participant_id"],s["species"],"present"])
            stem=Path(s["sample_id"]).stem
            for suffix in (".tif",".json"):
                src=(VIEW/(stem+suffix)).resolve(); os.symlink(os.path.relpath(src,mini),mini/(stem+suffix))
    obj=em.Image(space="HIST_SYMMETRIC_MINI",name="HIST_NISSL",fpath=str(mini),x=[axes[0][chosen],axes[1],axes[2]])
    image_error=0.0; support_error=0.0
    for j,i in enumerate(chosen):
        image,support=read_section(samples[i],em,[axes[1],axes[2]]); image_error=max(image_error,float(np.max(np.abs(obj.data[:,j]-image))))
        support_error=max(support_error,float(np.max(np.abs(obj.mask[j]-support))))
    result={"physical_indices":chosen.tolist(),"image_max_abs_error":image_error,"support_max_abs_error":support_error,"equivalent":image_error<=1e-6 and support_error==0.0}
    del obj; plt.close("all"); shutil.rmtree(mini)
    if not result["equivalent"]: raise RuntimeError(f"streamed support equivalence failed: {result}")
    return result

def validate_rigid(mats):
    hom=float(np.max(np.abs(mats[:,2]-np.array([0,0,1])))); ortho=float(np.max(np.abs(np.swapaxes(mats[:,:2,:2],1,2)@mats[:,:2,:2]-np.eye(2))))
    det=np.linalg.det(mats[:,:2,:2]); return {"homogeneous_row_max_error":hom,"orthogonality_max_error":ortho,"determinant_range":[float(det.min()),float(det.max())]}

def atlas_free():
    start=time.monotonic(); em,rows,samples,axes,observed=load_context(); ATLAS_DIR.mkdir(parents=True,exist_ok=True)
    preserved = _preserves_source_grid()
    audit=mini_equivalence(em,rows,samples,axes,observed,preserve_source_grid=preserved)
    coarse_row_axis, coarse_column_axis = coarse_spatial_axes(axes, preserve_source_grid=preserved)
    coarse_shape = (len(coarse_row_axis), len(coarse_column_axis))
    J,W=stream_stack(samples,observed,len(observed),shape=coarse_shape,em=em,spatial_axes=[axes[1],axes[2]],preserve_source_grid=preserved); x=[np.asarray(axes[0][observed]),coarse_row_axis,coarse_column_axis]
    assert J.shape==(3,641,*coarse_shape); assert W.shape==(641,*coarse_shape); assert tuple(map(len,x))==(641,*coarse_shape)
    gaps=np.diff(x[0]); before={"stage":"atlas-free","status":"running","shapes":{"J":list(J.shape),"W":list(W.shape)},"coordinate_lengths":list(map(len,x)),"spacings_um":{"observed_serial_gaps":{"min":float(gaps.min()),"median":float(np.median(gaps)),"max":float(gaps.max())},"row":float(np.diff(x[1]).mean()),"column":float(np.diff(x[2]).mean())},"pre_downsample":([1,1] if preserved else [4,4]),"optimizer_downI":[1,2,2],"optimizer_downJ":[1,2,2],"effective_inplane_spacing_um":float(np.diff(x[1]).mean()*2.0),"support_equivalence":audit,"start_time":now(),"peak_process_group_rss_kib":peak_rss_kib()}
    atomic_json(CHECKPOINTS/"atlas-free.json",before); print(json.dumps(before,indent=2),flush=True)
    draw=em.draw
    try: em.draw=False; out=em.atlas_free_reconstruction(J=J,xJ=x,W=W,n_steps=10,eA2d=2e4,downI=[1,2,2],downJ=[1,2,2])
    finally: em.draw=draw; plt.close("all")
    P=finite("atlas A2d",out["A2d"]).astype(np.float64)
    if P.shape!=(641,3,3): raise RuntimeError(f"unexpected A2d {P.shape}")
    B=rigid_frame(P); full=np.repeat(B[None],2846,axis=0); full[observed]=P
    p_obs=ATLAS_DIR/"observed_A2d.npy"; p_full=ATLAS_DIR/"expanded_2846_A2d.npy"; p_idx=ATLAS_DIR/"observed_physical_indices.npy"; p_b=ATLAS_DIR/"common_bookkeeping_frame.txt"
    np.save(p_obs,P); np.save(p_full,full); np.save(p_idx,observed); np.savetxt(p_b,B)
    ranges=residuals(ATLAS_DIR/"observed_residual_transforms.tsv",rows,observed,full,B); residual_plot(ATLAS_DIR/"observed_residual_transform_summary.png",axes,observed,full,B); rigid=validate_rigid(P)
    placement=bool(np.array_equal(np.load(p_idx),observed) and np.array_equal(x[0],axes[0][observed]) and np.allclose(full[observed],P))
    if not placement or ranges["recomposition_max_abs_error"]>1e-8 or rigid["homogeneous_row_max_error"]>1e-5: raise RuntimeError("atlas matrix validation failed")
    original_figure = ATLAS_DIR / "original_stack_emlddmm_orthogonal.png"
    transformed_figure = ATLAS_DIR / "atlas_free_transformed_stack_emlddmm_orthogonal.png"
    _emlddmm_stack_draw_qc(
        em, J * W[None], W, x, original_figure,
        "Observed-left Nissl before atlas-free section alignment",
    )
    residual = np.linalg.inv(B)[None] @ P
    transformed_numerator = np.zeros_like(J, dtype=np.float32)
    transformed_support = np.zeros_like(W, dtype=np.float32)
    for order in range(len(observed)):
        transformed, transformed_weights = _warp_saved_section(
            J[:, order], W[order], residual[order],
            coarse_row_axis, coarse_column_axis,
        )
        transformed_numerator[:, order] = transformed * transformed_weights[None]
        transformed_support[order] = transformed_weights
    _emlddmm_stack_draw_qc(
        em, transformed_numerator, transformed_support, x, transformed_figure,
        "Observed-left Nissl after atlas-free section alignment",
    )
    qc_png_validation = _validate_pngs([original_figure, transformed_figure])
    elapsed=time.monotonic()-start
    done={**before,"status":"complete","completion_time":now(),"elapsed_seconds":elapsed,"peak_process_group_rss_kib":peak_rss_kib(),"outputs":{"observed_A2d":str(p_obs),"expanded_A2d":str(p_full),"observed_indices":str(p_idx),"bookkeeping_frame":str(p_b),"residuals":str(ATLAS_DIR/"observed_residual_transforms.tsv")},"qc_outputs":{"original_stack_emlddmm_orthogonal":str(original_figure),"atlas_free_transformed_stack_emlddmm_orthogonal":str(transformed_figure)},"checksums":{str(p):checksum(p) for p in (p_obs,p_full,p_idx,p_b)},"qc_png_validation":qc_png_validation,"matrix_validation":{"finite":True,"observed_shape":list(P.shape),"expanded_shape":list(full.shape),"rigid":rigid,"ranges":ranges,"placement_exact":placement}}
    atomic_json(CHECKPOINTS/"atlas-free.json",done); print(json.dumps(done,indent=2),flush=True)

@dataclass
class LightImage:
    space:str; name:str; data:np.ndarray; x:list[np.ndarray]; title:str; names:list[str]
    def fnames(self): return self.names

def load_checkpoint(stage, checkpoint_root=None, *, accepted_statuses=("complete",)):
    checkpoint_root = CHECKPOINTS if checkpoint_root is None else checkpoint_root
    p=checkpoint_root/f"{stage}.json"; d=json.loads(p.read_text())
    if d.get("status") not in accepted_statuses:
        raise RuntimeError(f"{stage} checkpoint incomplete")
    for path,digest in d.get("checksums",{}).items():
        if checksum(Path(path))!=digest: raise RuntimeError(f"checkpoint checksum mismatch: {path}")
    return d

def profile_capture(code,histories):
    def fn(frame,event,arg):
        if event=="return" and frame.f_code is code and "Esave" in frame.f_locals:
            histories.append(np.asarray([[float(finite("energy",v)) for v in row] for row in frame.f_locals["Esave"]]))
        return fn
    return fn

def validate_registration_atlas_inputs(rows, axes, observed):
    """Validate the authoritative atlas-free registration initialization."""
    current_checkpoint = CHECKPOINTS / "atlas-free.json"
    if current_checkpoint.is_file():
        source_atlas = ATLAS_DIR
        source_checkpoints = CHECKPOINTS
    else:
        source_atlas = BASELINE_OUTPUT / "section_alignment_atlas_free"
        source_checkpoints = BASELINE_OUTPUT / "checkpoints"
    checkpoint = load_checkpoint("atlas-free", source_checkpoints)
    paths = {
        "observed_A2d": source_atlas / "observed_A2d.npy",
        "expanded_A2d": source_atlas / "expanded_2846_A2d.npy",
        "observed_indices": source_atlas / "observed_physical_indices.npy",
        "bookkeeping_frame": source_atlas / "common_bookkeeping_frame.txt",
    }
    source_hashes = {str(path): checksum(path) for path in paths.values()}
    if any(
        checkpoint.get("checksums", {}).get(path) != digest
        for path, digest in source_hashes.items()
    ):
        raise RuntimeError("Atlas-free source checksum differs from checkpoint")
    checkpoint_path = source_checkpoints / "atlas-free.json"
    checkpoint_hash = checksum(checkpoint_path)
    integrity_hashes = {
        **source_hashes,
        str(checkpoint_path): checkpoint_hash,
    }
    saved_observed = finite(
        "observed atlas A2d", np.load(paths["observed_A2d"])
    ).astype(np.float64)
    expanded = finite(
        "expanded atlas A2d", np.load(paths["expanded_A2d"])
    ).astype(np.float64)
    saved_indices = np.asarray(
        np.load(paths["observed_indices"]), dtype=np.int64
    )
    baseline = finite(
        "atlas bookkeeping frame", np.loadtxt(paths["bookkeeping_frame"])
    ).astype(np.float64)
    if saved_observed.shape != (641, 3, 3):
        raise RuntimeError(f"Observed atlas A2d shape is {saved_observed.shape}")
    if expanded.shape != (2846, 3, 3):
        raise RuntimeError(f"Expanded atlas A2d shape is {expanded.shape}")
    if saved_indices.shape != (641,):
        raise RuntimeError(f"Observed index shape is {saved_indices.shape}")
    if baseline.shape != (3, 3):
        raise RuntimeError(f"Bookkeeping frame shape is {baseline.shape}")
    if not np.array_equal(saved_indices, observed):
        raise RuntimeError("Saved observed indices differ from manifest mapping")
    if not np.array_equal(expanded[observed], saved_observed):
        raise RuntimeError("Expanded observed matrices differ from saved matrices")
    unsupported = np.ones(2846, dtype=bool)
    unsupported[observed] = False
    if not np.array_equal(
        expanded[unsupported],
        np.broadcast_to(baseline, expanded[unsupported].shape),
    ):
        raise RuntimeError("Unsupported rows differ from bookkeeping frame")
    allen = np.asarray(
        [int(rows[index]["allen_section_number"]) for index in observed]
    )
    serial = np.asarray(axes[0], dtype=np.float64)
    table_serial = np.asarray(
        [
            float(rows[index]["serial_z_center_mm"]) * 1000.0
            for index in observed
        ]
    )
    if np.unique(saved_indices).size != 641 or not np.allclose(
        serial[observed], table_serial, atol=1e-8, rtol=0.0
    ):
        raise RuntimeError("Observed Allen/serial mapping validation failed")
    report = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_status": checkpoint["status"],
        "source_paths": {key: str(path) for key, path in paths.items()},
        "source_checksums": source_hashes,
        "checkpoint_sha256": checkpoint_hash,
        "observed_count": 641,
        "expanded_count": 2846,
        "observed_shape": list(saved_observed.shape),
        "expanded_shape": list(expanded.shape),
        "observed_placement_exact": True,
        "unsupported_rows_equal_bookkeeping_frame": True,
        "manifest_index_mapping_exact": True,
        "serial_coordinate_mapping_exact": True,
        "serial_coordinate_tolerance_um": 1e-8,
        "allen_section_range": [int(allen.min()), int(allen.max())],
        "serial_coordinate_range_um": [
            float(table_serial.min()), float(table_serial.max())
        ],
    }
    return expanded, baseline, integrity_hashes, report


def flagged_section_residual_comparison(
    rows, axes, observed, initial, final, baseline
):
    flagged = (1082, 1089, 2055, 2059, 2238, 2242, 2466)
    by_allen = {
        int(rows[index]["allen_section_number"]): (order, int(index))
        for order, index in enumerate(observed)
    }
    inverse_baseline = np.linalg.inv(baseline)
    records = []
    for allen in flagged:
        if allen not in by_allen:
            raise RuntimeError(f"Flagged Allen section {allen} is absent")
        observed_order, physical_index = by_allen[allen]
        initial_residual = inverse_baseline @ initial[physical_index]
        final_residual = inverse_baseline @ final[physical_index]
        initial_error = float(
            np.max(
                np.abs(
                    baseline @ initial_residual - initial[physical_index]
                )
            )
        )
        final_error = float(
            np.max(
                np.abs(baseline @ final_residual - final[physical_index])
            )
        )
        if max(initial_error, final_error) > 1e-8:
            raise RuntimeError(
                f"Flagged-section recomposition failed for Allen {allen}"
            )
        def components(matrix):
            return {
                "row_translation_um": float(matrix[0, 2]),
                "column_translation_um": float(matrix[1, 2]),
                "rotation_deg": float(
                    np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0]))
                ),
            }
        initial_components = components(initial_residual)
        final_components = components(final_residual)
        records.append(
            {
                "allen_section": allen,
                "observed_order": observed_order,
                "physical_index": physical_index,
                "serial_um": float(np.asarray(axes[0])[physical_index]),
                "initial_residual_matrix": initial_residual.tolist(),
                "final_residual_matrix": final_residual.tolist(),
                "initial": initial_components,
                "final": final_components,
                "final_minus_initial": {
                    key: final_components[key] - initial_components[key]
                    for key in initial_components
                },
                "initial_recomposition_max_abs_error": initial_error,
                "final_recomposition_max_abs_error": final_error,
            }
        )
    return records


def write_native_transform_manifest(paths):
    """Write a deterministic checksum inventory for pinned native outputs."""
    manifest = REG_DIR / "native_transform_checksums.tsv"
    temporary = manifest.with_suffix(".tsv.tmp")
    entries = []
    for path in sorted(paths, key=lambda item: item.relative_to(REG_DIR).as_posix()):
        relative = path.relative_to(REG_DIR).as_posix()
        entries.append((relative, path.stat().st_size, checksum(path)))
    if not entries:
        raise RuntimeError("Pinned transform writer produced no native files")
    with temporary.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(["relative_path", "size_bytes", "sha256"])
        writer.writerows(entries)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, manifest)
    return manifest, entries


def _registration_initial_affine(path: Path, dataset: Path, *, preserve_source_grid: bool) -> np.ndarray:
    if path.suffix == ".npz":
        with np.load(path) as saved:
            if "A" not in saved.files:
                raise RuntimeError("Initialization NPZ does not contain A")
            affine = finite("initial affine", saved["A"]).astype(np.float64)
    else:
        affine = finite("initial affine", np.loadtxt(path)).astype(np.float64)
    if affine.shape != (4, 4):
        raise RuntimeError(f"Initial affine shape is {affine.shape}")
    if not preserve_source_grid:
        if checksum(path) != ORIGINAL_A_SHA256:
            raise RuntimeError("Original accepted affine checksum changed")
        return affine
    symmetry = json.loads((dataset / "metadata/symmetry.json").read_text())
    lineage = Path(symmetry["source_dataset"])
    if not lineage.is_absolute():
        lineage = PROJECT / lineage
    inherited = np.asarray(json.loads((lineage / "metadata/linear_restack.json").read_text())["global_affine_mri_um_to_histology_um"], dtype=np.float64)
    if inherited.shape != (4, 4) or not np.array_equal(affine, inherited):
        raise RuntimeError("Initial affine differs from symmetric-source lineage")
    return affine


def registration(profile: str, *, native_qc: bool = False):
    start = time.monotonic()
    optimizer_parameters = resolve_registration_execution(profile, native_qc=native_qc)
    effective_config_sha256 = registration_config_sha256(optimizer_parameters)
    em, rows, samples, axes, observed = load_context()
    initial_A2d, baseline, atlas_hashes, atlas_validation = (
        validate_registration_atlas_inputs(rows, axes, observed)
    )
    if REG_DIR.exists() and any(REG_DIR.iterdir()):
        raise RuntimeError(f"Registration output directory is not empty: {REG_DIR}")
    REG_DIR.mkdir(parents=True, exist_ok=True)

    prov = json.loads(MRI_PROV.read_text())
    native = load_pinned_mri_image(em, mri_path=MRI, provenance=prov)
    xI, I = em.downsample_image_domain(native.x, native.data, [4, 4, 4])
    I = np.asarray(I, dtype=np.float32)
    xI = [np.asarray(value) for value in xI]
    if I.shape[1:] != (237, 284, 254):
        raise RuntimeError(f"Downsampled MRI shape is {I.shape[1:]}")
    mri = LightImage(
        "MRI_7T_WHOLE", "7T_T1", I, xI, "volume", [str(MRI)]
    )
    rss_before_release = current_rss_kib()
    del native
    gc.collect()
    rss_after_release = current_rss_kib()

    preserved = _preserves_source_grid()
    coarse_row_axis, coarse_column_axis = coarse_spatial_axes(axes, preserve_source_grid=preserved)
    coarse_shape = (len(coarse_row_axis), len(coarse_column_axis))
    J, W = stream_stack(
        samples, observed, 2846, shape=coarse_shape,
        em=em, spatial_axes=[axes[1], axes[2]], preserve_source_grid=preserved,
    )
    xJ = [
        np.asarray(axes[0]),
        coarse_row_axis,
        coarse_column_axis,
    ]
    if J.shape != (3, 2846, *coarse_shape):
        raise RuntimeError(f"Histology shape is {J.shape}")
    if W.shape != (2846, *coarse_shape):
        raise RuntimeError(f"Support shape is {W.shape}")
    serial_pitch = float(abs(xJ[0][1] - xJ[0][0]))
    si = [float(abs(value[1] - value[0])) for value in xI]
    sj = [
        serial_pitch,
        float(abs(xJ[1][1] - xJ[1][0])),
        float(abs(xJ[2][1] - xJ[2][0])),
    ]
    if not np.allclose(si, [800.0] * 3, atol=1e-3):
        raise RuntimeError(f"MRI spacing assertion failed: {si}")
    if not np.isclose(serial_pitch, 50.0, atol=1e-8):
        raise RuntimeError(f"Serial lattice pitch changed: {serial_pitch}")
    if preserved:
        expected_spacing = float(json.loads((DATASET / "metadata/loader_canvas_audit.json").read_text())["target_spacing_um"])
        if not np.allclose(sj[1:], [expected_spacing] * 2, atol=1e-6, rtol=0.0):
            raise RuntimeError(f"Preserved histology spacing assertion failed: {sj}")
    elif not np.allclose(sj, [50.0, 800.0, 800.0], atol=1e-3):
        raise RuntimeError(f"Histology spacing assertion failed: {sj}")
    names = [Path(sample["sample_id"]).stem for sample in samples]
    hist = LightImage(
        "HIST_SYMMETRIC", "HIST_NISSL", J, xJ, "slice_dataset", names
    )
    initial_A_sha256 = checksum(INITIAL_A)
    A = _registration_initial_affine(INITIAL_A, DATASET, preserve_source_grid=preserved)

    cfg = dict(
        I=I,
        xI=[xI],
        J=J,
        xJ=[xJ],
        W0=W,
        A=A,
        A2d=initial_A2d,
        v=None,
        dtype=torch.float32,
        device="cpu",
        **optimizer_parameters,
    )
    effective_scales = {
        "MRI_um": [
            [800.0 * factor for factor in level]
            for level in optimizer_parameters["downI"]
        ],
        "histology_um": [
            [serial_pitch, sj[1] * level[1], sj[2] * level[2]]
            for level in optimizer_parameters["downJ"]
        ],
    }
    supported_rows = int(np.count_nonzero(np.any(W > 0, axis=(1, 2))))
    provenance = {
        "profile_name": profile,
        "optimizer_parameters": optimizer_parameters,
        "external_pre_downsampling": {
            "MRI": [4, 4, 4],
            "histology": ([1, 1, 1] if preserved else [1, 4, 4]),
        },
        "effective_scales": effective_scales,
        "pinned_emlddmm_commit": PIN,
        "original_affine": {
            "source_path": str(INITIAL_A),
            "sha256": initial_A_sha256,
        },
        "atlas_free_expanded_A2d": {
            "source_path": atlas_validation["source_paths"]["expanded_A2d"],
            "sha256": atlas_validation["source_checksums"][
                atlas_validation["source_paths"]["expanded_A2d"]
            ],
        },
        "input_datasets": {
            "dataset": str(DATASET),
            "histology_view": str(VIEW),
            "mri": str(MRI),
            "mri_provenance": str(MRI_PROV),
        },
        "working_shapes": {
            "I": list(I.shape),
            "J": list(J.shape),
            "W0": list(W.shape),
        },
        "working_spacings_um": {"xI": si, "xJ": sj},
        "W0_support": {
            "sum": float(np.sum(W, dtype=np.float64)),
            "supported_row_count": supported_rows,
            "zero_row_count": int(W.shape[0] - supported_rows),
        },
        "initial_velocity": "implicit_zero",
        "configuration_sha256": effective_config_sha256,
    }
    atomic_json(REG_DIR / "effective_registration_config.json", provenance)
    rss_before_optimizer = current_rss_kib()
    running = {
        "stage": "registration",
        "status": "running",
        "shapes": {
            "I": list(I.shape),
            "J": list(J.shape),
            "W0": list(W.shape),
        },
        "coordinate_lengths": {
            "xI": list(map(len, xI)),
            "xJ": list(map(len, xJ)),
        },
        "spacings_um": {"xI": si, "xJ": sj},
        "spacing_validation_passed": True,
        "shape_validation_passed": True,
        "atlas_free_validation": atlas_validation,
        "pre_downsample": {"MRI": [4, 4, 4], "histology": ([1, 1, 1] if preserved else [1, 4, 4])},
        "registration_profile": profile,
        "effective_config_sha256": effective_config_sha256,
        "effective_spacings_um": {
            "level1_I": effective_scales["MRI_um"][0],
            "level2_I": effective_scales["MRI_um"][1],
            "level3_I": effective_scales["MRI_um"][2],
            "level1_J": effective_scales["histology_um"][0],
            "level2_J": effective_scales["histology_um"][1],
            "level3_J": effective_scales["histology_um"][2],
        },
        "rss_current_kib": {
            "before_native_mri_release": rss_before_release,
            "after_native_mri_release": rss_after_release,
            "immediately_before_emlddmm": rss_before_optimizer,
        },
        "peak_process_group_rss_kib_before_emlddmm": peak_rss_kib(),
        "start_time": now(),
        "postprocessing_invoked": False,
    }
    atomic_json(CHECKPOINTS / "registration.json", running)
    print(json.dumps(running, indent=2), flush=True)

    histories = []
    sys.setprofile(profile_capture(em.emlddmm.__code__, histories))
    try:
        outputs = em.emlddmm_multiscale(**cfg)
    finally:
        sys.setprofile(None)
    if len(outputs) != 3 or len(histories) != 3:
        raise RuntimeError("Missing multiscale outputs or raw Esave histories")
    histories = [
        finite(f"raw Esave level {level}", history)
        for level, history in enumerate(histories, 1)
    ]
    final = outputs[-1]
    _save_final_effective_match_weight(
        final,
        observed,
        tuple(W.shape),
        REG_DIR / "final_observed_effective_match_weight.npy",
    )
    if native_qc:
        save_native_qc_figures(final)
    else:
        for key in FULL_OUTPUT_RELEASE_KEYS:
            del final[key]
    Aout = finite("final A", final["A"])
    final_A2d = finite("final A2d", final["A2d"])
    velocity = finite("final velocity", final["v"])
    xv = [
        finite(f"xv{axis}", value)
        for axis, value in enumerate(final["xv"])
    ]
    if Aout.shape != (4, 4):
        raise RuntimeError(f"Final affine shape is {Aout.shape}")
    if final_A2d.shape != (2846, 3, 3):
        raise RuntimeError(f"Final A2d shape is {final_A2d.shape}")
    if len(xv) != 3 or any(value.ndim != 1 for value in xv):
        raise RuntimeError("Velocity coordinate vectors are invalid")

    before_native = {
        path.resolve()
        for path in REG_DIR.rglob("*")
        if path.is_file()
    }
    em.write_transform_outputs(str(REG_DIR), final, mri, hist)
    after_native = {
        path.resolve()
        for path in REG_DIR.rglob("*")
        if path.is_file()
    }
    native_paths = after_native - before_native
    section_matrix_count = sum(
        path.name.endswith("_matrix.txt") for path in native_paths
    )
    affine_count = sum(path.name == "A.txt" for path in native_paths)
    velocity_count = sum(path.name == "velocity.vtk" for path in native_paths)
    if (
        section_matrix_count != 2846
        or affine_count != 1
        or velocity_count != 1
    ):
        raise RuntimeError(
            "Pinned writer output validation failed: "
            f"A2d={section_matrix_count}, A={affine_count}, "
            f"velocity={velocity_count}"
        )
    manifest, native_entries = write_native_transform_manifest(native_paths)

    numerical = REG_DIR / "full_coarse_numerical_outputs.npz"
    np.savez_compressed(
        numerical,
        A=Aout,
        A2d=final_A2d,
        v=velocity,
        xv0=xv[0],
        xv1=xv[1],
        xv2=xv[2],
        xI0=xI[0],
        xI1=xI[1],
        xI2=xI[2],
        xJ0=xJ[0],
        xJ1=xJ[1],
        xJ2=xJ[2],
        observed=observed,
    )
    energy_paths = []
    for level, history in enumerate(histories, 1):
        path = REG_DIR / f"raw_Esave_level-{level}.npy"
        np.save(path, history)
        energy_paths.append(path)

    flagged = flagged_section_residual_comparison(
        rows, axes, observed, initial_A2d, final_A2d, baseline
    )
    final_atlas_hashes = {
        path: checksum(Path(path)) for path in atlas_hashes
    }
    atlas_unchanged = final_atlas_hashes == atlas_hashes
    if not atlas_unchanged:
        raise RuntimeError("Atlas-free initialization changed during registration")
    elapsed = time.monotonic() - start
    done = {
        **running,
        "status": "complete",
        "completion_time": now(),
        "elapsed_seconds": elapsed,
        "peak_process_group_rss_kib": peak_rss_kib(),
        "atlas_free_hashes_after": final_atlas_hashes,
        "atlas_free_hashes_unchanged": atlas_unchanged,
        "flagged_section_residuals": flagged,
        "outputs": {
            "numerical": str(numerical),
            "raw_Esave": [str(path) for path in energy_paths],
            "transform_root": str(REG_DIR),
            "native_checksum_manifest": str(manifest),
        },
        "native_outputs": {
            "file_count": len(native_entries),
            "section_matrix_count": section_matrix_count,
            "global_affine_count": affine_count,
            "velocity_vtk_count": velocity_count,
            "manifest_sha256": checksum(manifest),
        },
        "checksums": {
            str(path): checksum(path)
            for path in [numerical, *energy_paths, manifest]
        },
        "postprocessing_invoked": False,
    }
    atomic_json(CHECKPOINTS / "registration.json", done)
    print(json.dumps(done, indent=2), flush=True)

def geometry_audit(source,nifti):
    shape=tuple(nifti.shape[:3]); affine=np.asarray(nifti.affine); points=[(i,j,k) for i in (0,shape[0]-1) for j in (0,shape[1]-1) for k in (0,shape[2]-1)]; points.append(tuple((np.asarray(shape)-1)//2)); errors=[]
    if source.data.shape[1:]!=shape: raise RuntimeError("MRI array order mismatch")
    for p in points: errors.append(float(np.max(np.abs((affine@[*p,1])[:3]*1000-np.asarray([source.x[a][p[a]] for a in range(3)])))))
    if max(errors)>1e-3: raise RuntimeError("MRI affine/loader mismatch")
    return {"points_checked":9,"max_abs_error_um":max(errors),"shape":list(shape)}

def jacobian_chunks(em,xv,v,chunk=24):
    xt=[torch.as_tensor(x,dtype=torch.float32) for x in xv]; phi=em.v_to_phii(xt,-torch.as_tensor(v,dtype=torch.float32).flip(0)).cpu().numpy(); spacing=[float(x[1]-x[0]) for x in xv]
    minimum=math.inf; maximum=-math.inf; nonpositive=0; count=0
    for start in range(0,phi.shape[1],chunk):
        lo=max(0,start-1); hi=min(phi.shape[1],start+chunk+1); block=phi[:,lo:hi]
        derivatives=[]
        for component in range(3): derivatives.append(np.gradient(block[component],*spacing,edge_order=1))
        interior=slice(start-lo,min(start+chunk,phi.shape[1])-lo)
        a,b,c=derivatives[0][0][interior],derivatives[0][1][interior],derivatives[0][2][interior]
        d,e,f=derivatives[1][0][interior],derivatives[1][1][interior],derivatives[1][2][interior]
        g,h,i=derivatives[2][0][interior],derivatives[2][1][interior],derivatives[2][2][interior]
        det=a*(e*i-f*h)-b*(d*i-f*g)+c*(d*h-e*g)
        if not np.all(np.isfinite(det)): raise RuntimeError("nonfinite Jacobian")
        minimum=min(minimum,float(det.min())); maximum=max(maximum,float(det.max())); nonpositive+=int(np.count_nonzero(det<=0)); count+=det.size
        del derivatives,det
    return phi,{"min":minimum,"max":maximum,"nonpositive":nonpositive,"count":count}

def _legacy_postprocess_DO_NOT_USE():
    start=time.monotonic(); load_checkpoint("atlas-free"); load_checkpoint("registration"); em,rows,samples,axes,observed=load_context(); POST_DIR.mkdir(parents=True,exist_ok=True)
    z=np.load(REG_DIR/"full_coarse_numerical_outputs.npz"); A=z["A"]; P=z["A2d"]; v=z["v"]; xv=[z[f"xv{i}"] for i in range(3)]
    prov=json.loads(MRI_PROV.read_text()); source=load_pinned_mri_image(em,mri_path=MRI,provenance=prov); ni=nib.load(str(MRI)); geometry=geometry_audit(source,ni)
    B=P[np.setdiff1d(np.arange(2846),observed)[0]]; ranges=residuals(POST_DIR/"observed_residual_transforms.tsv",rows,observed,P,B); residual_plot(POST_DIR/"observed_residual_transform_summary.png",axes,observed,P,B)
    # Compact QC and support-aware full reconstruction reuse streamed 800-um histology.
    coarse_row_axis, coarse_column_axis = coarse_spatial_axes(axes)
    coarse_shape = (len(coarse_row_axis), len(coarse_column_axis))
    J,W=stream_stack(samples,observed,2846,shape=coarse_shape,em=em,spatial_axes=[axes[1],axes[2]]); xJ=[axes[0],coarse_row_axis,coarse_column_axis]
    bi=np.linalg.inv(B); corners=np.array([[xJ[1][0],xJ[2][0],1],[xJ[1][0],xJ[2][-1],1],[xJ[1][-1],xJ[2][0],1],[xJ[1][-1],xJ[2][-1],1]]).T; rc=bi@corners
    yr=np.linspace(rc[0].min(),rc[0].max(),130); xr=np.linspace(rc[1].min(),rc[1].max(),182); RR,CC=np.meshgrid(yr,xr,indexing="ij")
    num=np.memmap(RUN_TMP/"reg_num",mode="w+",dtype=np.float32,shape=J.shape); sup=np.memmap(RUN_TMP/"reg_sup",mode="w+",dtype=np.float32,shape=W.shape); num[:]=0; sup[:]=0
    for idx in observed:
        qy=P[idx,0,0]*RR+P[idx,0,1]*CC+P[idx,0,2]; qx=P[idx,1,0]*RR+P[idx,1,1]*CC+P[idx,1,2]; iy=(qy-xJ[1][0])/(xJ[1][1]-xJ[1][0]); ix=(qx-xJ[2][0])/(xJ[2][1]-xJ[2][0]); sup[idx]=ndi.map_coordinates(W[idx],[iy,ix],order=1,mode="constant",cval=0,prefilter=False)
        for c in range(3): num[c,idx]=ndi.map_coordinates(J[c,idx]*W[idx],[iy,ix],order=1,mode="constant",cval=0,prefilter=False)
    phi,jac=jacobian_chunks(em,xv,v); XV=np.stack(np.meshgrid(*xv,indexing="ij")); disp=phi-XV; shape=tuple(source.data.shape[1:]); recon=np.memmap(RUN_TMP/"recon",mode="w+",dtype=np.float32,shape=(*shape,3)); frac=np.memmap(RUN_TMP/"support",mode="w+",dtype=np.float32,shape=shape); support_min=math.inf; support_max=-math.inf; support_nonempty=False; zero_violation=False
    for start1 in range(0,shape[1],8):
        stop=min(start1+8,shape[1]); X=np.stack(np.meshgrid(source.x[0],source.x[1][start1:stop],source.x[2],indexing="ij")); iv=[(X[k]-xv[k][0])/(xv[k][1]-xv[k][0]) for k in range(3)]; flow=X+np.stack([ndi.map_coordinates(disp[k],iv,order=1,mode="nearest",prefilter=False) for k in range(3)]); q=(A[:3,:3]@flow.reshape(3,-1)+A[:3,3,None]).reshape(flow.shape); ih=[(q[0]-xJ[0][0])/(xJ[0][1]-xJ[0][0]),(q[1]-yr[0])/(yr[1]-yr[0]),(q[2]-xr[0])/(xr[1]-xr[0])]; s=np.clip(ndi.map_coordinates(sup,ih,order=1,mode="constant",cval=0,prefilter=False),0,1); out=np.zeros((*s.shape,3),np.float32); positive=s>0
        for c in range(3): sampled=ndi.map_coordinates(num[c],ih,order=1,mode="constant",cval=0,prefilter=False); out[...,c][positive]=sampled[positive]/s[positive]
        if not np.all(np.isfinite(s)) or not np.all(np.isfinite(out)): raise RuntimeError("nonfinite reconstruction slab")
        support_min=min(support_min,float(s.min())); support_max=max(support_max,float(s.max())); support_nonempty=support_nonempty or bool(np.any(positive)); zero_violation=zero_violation or bool(np.any(out[~positive]!=0))
        recon[:,start1:stop]=out; frac[:,start1:stop]=s; recon.flush(); frac.flush(); print(f"postprocess slab {start1}:{stop}",flush=True)
    rh=ni.header.copy(); rh.set_data_dtype(np.float32); sh=ni.header.copy(); sh.set_data_dtype(np.float32); rp=OUTPUT/"nissl_support_weighted_reconstruction_on_mri_grid.nii"; sp=OUTPUT/"nissl_fractional_support_on_mri_grid.nii"; nib.save(nib.Nifti1Image(recon,ni.affine,header=rh),str(rp)); nib.save(nib.Nifti1Image(frac,ni.affine,header=sh),str(sp))
    compact_qc=compact_post_qc(source,recon,frac,axes,observed,A)
    if not support_nonempty or support_min<0 or support_max>1+1e-6: raise RuntimeError("invalid fractional support volume")
    if zero_violation: raise RuntimeError("nonzero reconstruction outside support")
    deformation={"velocity_max_um":float(np.linalg.norm(v,axis=1).max()),"jacobian":jac}; atomic_json(POST_DIR/"velocity_jacobian_summary.json",deformation)
    done={"stage":"postprocess","status":"review_required","elapsed_seconds":time.monotonic()-start,"peak_process_group_rss_kib":peak_rss_kib(),"geometry":geometry,"residual_ranges":ranges,"reconstruction":{"intensity":str(rp),"fractional_support":str(sp)},"deformation":deformation,"compact_qc":str(compact_qc),"production_refinement_launched":False}; atomic_json(CHECKPOINTS/"postprocess.json",done); atomic_json(OUTPUT/"full_coarse_summary.json",done); print(json.dumps(done,indent=2),flush=True)

def residual_plot(path: Path, axes, observed, mats, baseline):
    r=np.linalg.inv(baseline)[None]@mats[observed]; serial=np.asarray(axes[0])[observed]
    values=(r[:,0,2],r[:,1,2],np.degrees(np.arctan2(r[:,1,0],r[:,0,0])))
    labels=("row translation (um)","column translation (um)","rotation (degrees)")
    fig,panels=plt.subplots(3,1,figsize=(12,7),sharex=True)
    for panel,value,label in zip(panels,values,labels): panel.plot(serial,value,lw=.7); panel.set_ylabel(label)
    panels[-1].set_xlabel("original serial coordinate (um)"); fig.tight_layout(); fig.savefig(path,dpi=150); plt.close(fig)

def compact_post_qc(source,recon,fraction,axes,observed,A):
    reps=observed[np.linspace(0,len(observed)-1,9).round().astype(int)]; inv=np.linalg.inv(A)
    fig,p=plt.subplots(3,len(reps),figsize=(27,9),facecolor="white")
    for col,index in enumerate(reps):
        point=inv@np.asarray([float(axes[0][index]),0,0,1]); plane=int(np.argmin(np.abs(np.asarray(source.x[1])-point[1])))
        m=np.asarray(source.data[0,::4,plane,::4],np.float32); r=np.asarray(recon[::4,plane,::4],np.float32); s=np.asarray(fraction[::4,plane,::4],np.float32)
        lo,hi=np.percentile(m,[1,99]); md=np.clip((m-lo)/(hi-lo+1e-8),0,1); overlay=.5*np.clip(r,0,1)+.5*md[...,None]
        p[0,col].imshow(md,cmap="gray",origin="lower"); p[1,col].imshow(np.clip(r,0,1),origin="lower"); p[2,col].imshow(np.clip(overlay,0,1),origin="lower")
        if np.any(s>.01): p[1,col].contour(s,levels=[.01],colors="red",linewidths=.3)
        p[0,col].set_title(str(index)); [p[row,col].axis("off") for row in range(3)]
    p[0,0].set_ylabel("MRI"); p[1,0].set_ylabel("support-weighted Nissl"); p[2,0].set_ylabel("overlay")
    fig.tight_layout(); path=POST_DIR/"full_extent_compact_mri_nissl_qc.png"; fig.savefig(path,dpi=150); plt.close(fig)
    for level in (1,2):
        history=np.load(REG_DIR/f"raw_Esave_level-{level}.npy"); fig,ax=plt.subplots(figsize=(8,4))
        for column,label in enumerate(("E","matching","regularization")[:history.shape[1]]): ax.plot(history[:,column],label=label)
        ax.set(xlabel="iteration",title=f"Raw Esave level {level}"); ax.legend(); fig.tight_layout(); fig.savefig(POST_DIR/f"objective_history_level-{level}.png",dpi=150); plt.close(fig)
    return path



def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if path.exists() or temporary.exists():
        raise RuntimeError(f"Refusing to overwrite output: {path}")
    with temporary.open("x", encoding="utf-8", newline="") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _atomic_nifti(path: Path, data, affine: np.ndarray, dtype) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp{path.suffix}")
    if path.exists() or temporary.exists():
        raise RuntimeError(f"Refusing to overwrite output: {path}")
    header = nib.Nifti1Header()
    header.set_data_dtype(dtype)
    header.set_xyzt_units("mm")
    nib.save(nib.Nifti1Image(data, affine, header=header), str(temporary))
    os.replace(temporary, path)


def _atomic_figure(path: Path, figure, *, dpi: int = 160) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp{path.suffix}")
    if path.exists() or temporary.exists():
        raise RuntimeError(f"Refusing to overwrite output: {path}")
    figure.savefig(temporary, dpi=dpi, bbox_inches="tight")
    os.replace(temporary, path)
    plt.close(figure)


def _emlddmm_stack_draw_qc(
    em, numerator, support, spatial_axes, output: Path, title: str
) -> Path:
    """Render one supported RGB stack with the pinned EM-LDDMM drawer."""
    weighted = np.asarray(numerator, dtype=np.float32)
    weights = np.asarray(support, dtype=np.float32)
    axes = [np.asarray(axis, dtype=np.float64) for axis in spatial_axes]
    if weighted.ndim != 4 or weighted.shape[0] != 3:
        raise RuntimeError("Stack numerator must have shape (3, serial, row, column)")
    if weights.shape != weighted.shape[1:]:
        raise RuntimeError("Stack support does not match numerator shape")
    if len(axes) != 3 or tuple(map(len, axes)) != weights.shape:
        raise RuntimeError("Stack axes do not match supported image shape")
    if (
        not np.all(np.isfinite(weighted))
        or not np.all(np.isfinite(weights))
        or any(not np.all(np.isfinite(axis)) for axis in axes)
    ):
        raise RuntimeError("Stack QC inputs contain nonfinite values")
    if np.any(weights < 0.0):
        raise RuntimeError("Stack support must be nonnegative")
    positive = weights > 0.0
    image = np.zeros_like(weighted, dtype=np.float32)
    image[:, positive] = weighted[:, positive] / weights[positive]
    figure, _ = em.draw(
        image,
        xJ=axes,
        n_slices=5,
        disp=False,
        interpolation="none",
        vmin=0,
        vmax=1,
    )
    figure.suptitle(title)
    _atomic_figure(output, figure)
    return output


def _source_inventory(checkpoints: dict[str, Any]) -> tuple[dict[str, str], dict[str, Any]]:
    registration = checkpoints["registration"]
    atlas = checkpoints["atlas-free"]
    paths: list[Path] = [
        CHECKPOINTS / "atlas-free.json",
        CHECKPOINTS / "registration.json",
        REG_DIR / "full_coarse_numerical_outputs.npz",
        REG_DIR / "raw_Esave_level-1.npy",
        REG_DIR / "raw_Esave_level-2.npy",
        REG_DIR / "native_transform_checksums.tsv",
        ATLAS_DIR / "observed_A2d.npy",
        ATLAS_DIR / "expanded_2846_A2d.npy",
        ATLAS_DIR / "observed_physical_indices.npy",
        ATLAS_DIR / "common_bookkeeping_frame.txt",
    ]
    for key in ("numerical", "native_checksum_manifest"):
        value = registration.get("outputs", {}).get(key)
        if value:
            paths.append(Path(value))
    paths.extend(Path(value) for value in registration.get("outputs", {}).get("raw_Esave", []))
    for value in atlas.get("outputs", {}).values():
        candidate = Path(value)
        if candidate.is_file():
            paths.append(candidate)
    manifest = REG_DIR / "native_transform_checksums.tsv"
    with manifest.open(encoding="utf-8", newline="") as stream:
        native_rows = list(csv.DictReader(stream, delimiter="\t"))
    if len(native_rows) != 2848:
        raise RuntimeError(f"Native transform manifest has {len(native_rows)} rows")
    native_report = []
    for row in native_rows:
        relative = row.get("relative_path", "")
        candidate = (REG_DIR / relative).resolve()
        try:
            candidate.relative_to(REG_DIR.resolve())
        except ValueError as exc:
            raise RuntimeError(f"Unsafe native transform path: {relative}") from exc
        if not candidate.is_file() or candidate.is_symlink():
            raise RuntimeError(f"Missing native transform: {candidate}")
        digest = checksum(candidate)
        if digest != row.get("sha256"):
            raise RuntimeError(f"Native transform checksum mismatch: {candidate}")
        if candidate.stat().st_size != int(row["size_bytes"]):
            raise RuntimeError(f"Native transform size mismatch: {candidate}")
        paths.append(candidate)
        native_report.append(relative)
    unique = sorted(set(paths), key=lambda item: str(item))
    hashes = {str(path): checksum(path) for path in unique}
    for checkpoint in checkpoints.values():
        for value, digest in checkpoint.get("checksums", {}).items():
            if hashes.get(value, checksum(Path(value))) != digest:
                raise RuntimeError(f"Checkpoint checksum mismatch: {value}")
    return hashes, {
        "native_transform_count": len(native_report),
        "native_transform_manifest": str(manifest),
        "native_manifest_verified": True,
    }


def _resolve_recorded_checkpoint_path(
    checkpoint_path: Path, recorded: str, *, description: str
) -> Path:
    if not isinstance(recorded, str) or not recorded:
        raise RuntimeError(f"Registration checkpoint has no recorded {description} path")
    path = Path(recorded).expanduser()
    if not path.is_absolute():
        path = checkpoint_path.parent.parent / path
    path = path.resolve()
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"Recorded {description} is not a regular file: {path}")
    return path


def _registration_output_value(registration: dict[str, Any], key: str) -> Any:
    outputs = registration.get("outputs")
    if isinstance(outputs, dict) and key in outputs:
        return outputs[key]
    return registration.get(key)


def _native_postprocess_sources(
    registration: dict[str, Any],
) -> tuple[Path, list[Path], Path | None, dict[str, str], dict[str, Any]]:
    """Resolve and hash the native product exactly as recorded by its checkpoint."""
    checkpoint_path = CHECKPOINTS / "registration.json"
    status = registration.get("status")
    if status not in {"complete", "complete_through_scale"}:
        raise RuntimeError("Registration checkpoint is not a usable completed product")
    numerical = _resolve_recorded_checkpoint_path(
        checkpoint_path, _registration_output_value(registration, "numerical"),
        description="numerical package",
    )
    raw_recorded = _registration_output_value(registration, "raw_Esave")
    if not isinstance(raw_recorded, list) or not raw_recorded:
        raise RuntimeError("Registration checkpoint has no recorded raw_Esave list")
    raw_paths = [
        _resolve_recorded_checkpoint_path(
            checkpoint_path, value, description=f"raw_Esave level {level}"
        )
        for level, value in enumerate(raw_recorded, 1)
    ]
    if status == "complete_through_scale":
        completed = registration.get("completed_scale_number")
        if not isinstance(completed, int) or completed < 1:
            raise RuntimeError("Through-scale checkpoint has no valid completed scale")
        if len(raw_paths) != completed:
            raise RuntimeError(
                "Recorded raw_Esave count does not match completed scale number"
            )
    effective_recorded = _registration_output_value(
        registration, "effective_match_weight"
    )
    if effective_recorded is None and status == "complete":
        effective_recorded = _registration_output_value(
            registration, "final_effective_match_weight"
        )
    if effective_recorded is None:
        if status != "complete_through_scale":
            raise RuntimeError(
                "Complete registration checkpoint has no effective match weight"
            )
        effective_path = None
    else:
        effective_path = _resolve_recorded_checkpoint_path(
            checkpoint_path, effective_recorded,
            description="effective match weight",
        )
    provenance_recorded = _registration_output_value(registration, "provenance")
    provenance_path = (
        _resolve_recorded_checkpoint_path(
            checkpoint_path, provenance_recorded,
            description="registration provenance",
        )
        if provenance_recorded is not None else None
    )
    if provenance_path is not None:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        for recorded, expected in provenance.get("checksums", {}).items():
            candidate = _resolve_recorded_checkpoint_path(
                checkpoint_path, recorded, description="provenance checksum source"
            )
            if checksum(candidate) != expected:
                raise RuntimeError(
                    f"Registration-product checksum mismatch: {candidate}"
                )
    product_recorded = _registration_output_value(
        registration, "transform_output_root"
    ) or registration.get("through_scale_product")
    product_root = (
        Path(product_recorded).expanduser()
        if isinstance(product_recorded, str) and product_recorded
        else numerical.parent
    )
    if not product_root.is_absolute():
        product_root = checkpoint_path.parent.parent / product_root
    product_root = product_root.resolve()
    if not product_root.is_dir() or product_root.is_symlink():
        raise RuntimeError(f"Recorded registration product is not a directory: {product_root}")
    product_files = []
    for path in product_root.rglob("*"):
        if path.is_symlink():
            raise RuntimeError(f"Registration product contains a symlink: {path}")
        if path.is_file():
            product_files.append(path.resolve())
    checkpoint_files = [
        path.resolve()
        for path in CHECKPOINTS.glob("registration*")
        if path.is_file() and not path.is_symlink()
    ]
    protected = sorted(
        set([numerical, *raw_paths, *product_files, *checkpoint_files]
            + ([effective_path] if effective_path is not None else [])
            + ([provenance_path] if provenance_path is not None else [])),
        key=str,
    )
    source_hashes = {str(path): checksum(path) for path in protected}
    validation = {
        "checkpoint_status": status,
        "registration_checkpoint": str(checkpoint_path),
        "numerical_package": str(numerical),
        "raw_objective_histories": [str(path) for path in raw_paths],
        "effective_match_weight": (
            str(effective_path) if effective_path is not None else None
        ),
        "effective_match_weight_missing_is_legitimate": (
            status == "complete_through_scale" and effective_path is None
        ),
        "registration_product_root": str(product_root),
        "protected_source_file_count": len(source_hashes),
        "numerical_package_verified": True,
        "recorded_paths_used": True,
    }
    return numerical, raw_paths, effective_path, source_hashes, validation


def _coarse_affine(
    xI: list[np.ndarray], *,
    expected_shape: tuple[int, int, int] | None = (237, 284, 254),
    grid_name: str = "coarse",
) -> tuple[np.ndarray, dict[str, Any]]:
    if expected_shape is not None and tuple(map(len, xI)) != expected_shape:
        raise RuntimeError(f"Unexpected {grid_name} MRI axes: {tuple(map(len, xI))}")
    affine = np.eye(4, dtype=np.float64)
    spacings = []
    for axis, coordinates in enumerate(xI):
        coordinates = finite(f"xI{axis}", coordinates).astype(np.float64)
        steps = np.diff(coordinates)
        if not np.allclose(steps, steps[0], atol=1e-6, rtol=0.0):
            raise RuntimeError(f"xI{axis} is not uniform")
        affine[axis, axis] = steps[0] / 1000.0
        affine[axis, 3] = coordinates[0] / 1000.0
        spacings.append(float(steps[0]))
    checks = [
        (i, j, k)
        for i in (0, len(xI[0]) - 1)
        for j in (0, len(xI[1]) - 1)
        for k in (0, len(xI[2]) - 1)
    ]
    checks.append(tuple(((np.asarray(tuple(map(len, xI))) - 1) // 2).tolist()))
    errors = []
    for index in checks:
        measured = (affine @ np.asarray([*index, 1.0]))[:3] * 1000.0
        expected = np.asarray([xI[axis][index[axis]] for axis in range(3)])
        errors.append(float(np.max(np.abs(measured - expected))))
    if max(errors) > 1e-6:
        raise RuntimeError(f"{grid_name.capitalize()} MRI affine coordinate audit failed")
    return affine, {
        "shape": list(map(len, xI)),
        "axes_um": [
            {"first": float(axis[0]), "last": float(axis[-1]), "length": len(axis)}
            for axis in xI
        ],
        "signed_spacings_um": spacings,
        "affine_mm": affine.tolist(),
        "points_checked": 9,
        "corner_and_center_max_abs_error_um": max(errors),
    }


def _final_residual_diagnostics(
    rows, axes, observed, final: np.ndarray, *,
    comparison_initializer: str = "atlas_free",
) -> tuple[np.ndarray, dict[str, Any], list[dict[str, Any]]]:
    baseline, unsupported = _final_a2d_baseline(final, observed)
    residual = np.linalg.inv(baseline)[None] @ final[observed]
    error = float(np.max(np.abs(baseline[None] @ residual - final[observed])))
    if error > 1e-7:
        raise RuntimeError("Final baseline-factored A2d recomposition failed")
    serial = np.asarray(axes[0], dtype=np.float64)[observed]
    values = {
        "row_translation_um": residual[:, 0, 2],
        "column_translation_um": residual[:, 1, 2],
        "rotation_deg": np.degrees(np.arctan2(residual[:, 1, 0], residual[:, 0, 0])),
    }
    jump_score = np.zeros(len(observed), dtype=np.float64)
    for value in values.values():
        scale = np.median(np.abs(np.diff(value))) + 1e-12
        jump_score[1:] += np.abs(np.diff(value)) / scale
    largest = np.argsort(jump_score[1:])[-10:] + 1
    flagged_by_allen = {
        int(rows[index]["allen_section_number"]): position
        for position, index in enumerate(observed)
    }
    if any(value not in flagged_by_allen for value in FLAGGED_ALLEN):
        raise RuntimeError("A flagged Allen section is absent from observed Nissl")
    path = POST_DIR / "final_observed_residual_transforms.tsv"
    lines = [
        "physical_index\tallen_section\tserial_z_um\trow_translation_um\t"
        "column_translation_um\trotation_deg\n"
    ]
    for position, index in enumerate(observed):
        lines.append(
            f"{int(index)}\t{rows[index]['allen_section_number']}\t{serial[position]:.9g}\t"
            f"{values['row_translation_um'][position]:.12g}\t"
            f"{values['column_translation_um'][position]:.12g}\t"
            f"{values['rotation_deg'][position]:.12g}\n"
        )
    _atomic_text(path, "".join(lines))
    figure, panels = plt.subplots(3, 1, figsize=(13, 8), sharex=True)
    for panel, (name, value) in zip(panels, values.items(), strict=True):
        panel.plot(serial / 1000.0, value, lw=0.75)
        for allen in FLAGGED_ALLEN:
            panel.axvline(
                serial[flagged_by_allen[allen]] / 1000.0,
                color="#d62728", lw=0.55, alpha=0.7,
            )
        for position in largest:
            panel.axvline(serial[position] / 1000.0, color="#ff7f0e", lw=0.4, alpha=0.45)
        panel.set_ylabel(name.replace("_", " "))
        panel.grid(alpha=0.15)
    panels[-1].set_xlabel("canonical anterior-to-posterior serial coordinate (mm)")
    figure.suptitle("Final section residuals after removal of the shared unsupported-row baseline")
    _atomic_figure(POST_DIR / "final_section_residual_traces.png", figure)
    if comparison_initializer == "identity":
        initial_residual = np.broadcast_to(
            np.eye(3, dtype=np.float64), (len(observed), 3, 3)
        )
        initial_baseline = "identity per physical serial position"
    elif comparison_initializer == "atlas_free":
        atlas_full = finite(
            "atlas-free expanded A2d",
            np.load(ATLAS_DIR / "expanded_2846_A2d.npy"),
        )
        atlas_baseline = finite(
            "atlas-free bookkeeping frame",
            np.loadtxt(ATLAS_DIR / "common_bookkeeping_frame.txt"),
        )
        initial_residual = np.linalg.inv(atlas_baseline)[None] @ atlas_full[observed]
        initial_baseline = "atlas-free common_bookkeeping_frame"
    else:
        raise ValueError(f"Unknown residual comparison initializer: {comparison_initializer}")
    comparisons = []
    for allen in FLAGGED_ALLEN:
        position = flagged_by_allen[allen]
        def components(matrix):
            return {
                "row_translation_um": float(matrix[0, 2]),
                "column_translation_um": float(matrix[1, 2]),
                "rotation_deg": float(np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0]))),
            }
        initial = components(initial_residual[position])
        final_value = components(residual[position])
        comparisons.append({
            "allen_section": allen,
            "physical_index": int(observed[position]),
            "serial_z_um": float(serial[position]),
            "initial_baseline": initial_baseline,
            "final_baseline": "common matrix shared by 2,205 unsupported final A2d rows",
            "initial": initial,
            "final": final_value,
            "gauge_invariant_final_minus_initial": {
                key: final_value[key] - initial[key] for key in initial
            },
        })
    report = {
        "baseline_matrix": baseline.tolist(),
        "unsupported_count": int(unsupported.sum()),
        "unsupported_rows_exactly_equal": True,
        "recomposition_max_abs_error": error,
        "largest_adjacent_jump_physical_indices": observed[largest].astype(int).tolist(),
        "flagged_allen_sections": list(FLAGGED_ALLEN),
        "comparison_initializer": comparison_initializer,
        "comparison_initial_baseline": initial_baseline,
        "tsv": str(path),
        "figure": str(POST_DIR / "final_section_residual_traces.png"),
    }
    return baseline, report, comparisons


def _final_a2d_baseline(
    final: np.ndarray, observed: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the exact common final-A2d matrix on unsupported physical rows."""
    unsupported = np.ones(len(final), dtype=bool)
    unsupported[np.asarray(observed, dtype=np.int64)] = False
    unsupported_matrices = np.asarray(final)[unsupported]
    if not len(unsupported_matrices):
        raise RuntimeError("Final A2d has no unsupported rows for baseline recovery")
    baseline = unsupported_matrices[0].copy()
    if not np.array_equal(
        unsupported_matrices, np.broadcast_to(baseline, unsupported_matrices.shape)
    ):
        raise RuntimeError("Unsupported final A2d rows do not share one exact baseline")
    return baseline, unsupported


def _objective_figure(
    raw_history_paths: list[Path] | None = None,
) -> dict[str, Any]:
    paths = (
        [REG_DIR / f"raw_Esave_level-{level}.npy" for level in (1, 2)]
        if raw_history_paths is None else list(raw_history_paths)
    )
    if not paths:
        raise RuntimeError("No raw objective histories were recorded")
    figure, panels = plt.subplots(
        len(paths), 1, figsize=(10, 3.5 * len(paths)), squeeze=False
    )
    shapes = []
    for level, path in enumerate(paths, 1):
        history = finite(f"raw Esave level {level}", np.load(path))
        if history.ndim == 1:
            history = history[:, None]
        shapes.append(list(history.shape))
        panel = panels[level - 1, 0]
        for column in range(history.shape[1]):
            panel.plot(history[:, column], label=f"saved component {column}")
        panel.set_title(f"Raw saved Esave level {level}")
        panel.set_xlabel("saved iteration")
        panel.set_ylabel("raw objective value")
        panel.legend(fontsize=7)
        panel.grid(alpha=0.15)
    figure.suptitle("Saved raw EM-LDDMM objective histories (no monotonicity requirement)")
    output = POST_DIR / "objective_histories.png"
    _atomic_figure(output, figure)
    return {"path": str(output), "raw_shapes": shapes, "values_modified": False}


def _deformation_products(em, xv, v) -> tuple[np.ndarray, dict[str, Any]]:
    xt = [torch.as_tensor(axis, dtype=torch.float32) for axis in xv]
    phi = finite(
        "saved-velocity deformation",
        em.v_to_phii(xt, -torch.as_tensor(v, dtype=torch.float32).flip(0)),
    ).astype(np.float32)
    if phi.shape[0] != 3 or tuple(phi.shape[1:]) != tuple(map(len, xv)):
        raise RuntimeError(f"Unexpected deformation shape {phi.shape}")
    shape = phi.shape[1:]
    displacement = np.memmap(
        RUN_TMP / "displacement_magnitude.dat", mode="w+", dtype=np.float32, shape=shape
    )
    jacobian = np.memmap(
        RUN_TMP / "jacobian_determinant.dat", mode="w+", dtype=np.float32, shape=shape
    )
    spacing = [float(axis[1] - axis[0]) for axis in xv]
    for start in range(0, shape[0], 24):
        stop = min(start + 24, shape[0])
        for local, absolute in enumerate(range(start, stop)):
            d0 = phi[0, absolute] - float(xv[0][absolute])
            d1 = phi[1, absolute] - np.asarray(xv[1])[:, None]
            d2 = phi[2, absolute] - np.asarray(xv[2])[None, :]
            displacement[absolute] = np.sqrt(d0 * d0 + d1 * d1 + d2 * d2)
        displacement.flush()
    for start in range(0, shape[0], 24):
        lo, hi = max(0, start - 1), min(shape[0], start + 25)
        block = phi[:, lo:hi]
        derivatives = []
        for component in range(3):
            derivatives.extend(np.gradient(block[component], *spacing, edge_order=1))
        interior = slice(start - lo, min(start + 24, shape[0]) - lo)
        a, b, c, d, e, f, g, h, i = [value[interior] for value in derivatives]
        determinant = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
        if not np.all(np.isfinite(determinant)):
            raise RuntimeError("Nonfinite Jacobian determinant")
        jacobian[start:min(start + 24, shape[0])] = determinant
        jacobian.flush()
        del derivatives, determinant, block
    if not np.all(np.isfinite(displacement)) or not np.all(np.isfinite(jacobian)):
        raise RuntimeError("Nonfinite deformation diagnostic")
    nonpositive = int(np.count_nonzero(jacobian <= 0.0))
    if nonpositive:
        raise RuntimeError(f"Final deformation has {nonpositive} nonpositive Jacobians")
    velocity_magnitude = np.sqrt(np.sum(np.asarray(v, dtype=np.float64) ** 2, axis=1))
    def stats(array, percentiles=(0, 1, 5, 25, 50, 75, 95, 99, 100)):
        values = np.percentile(np.asarray(array), percentiles)
        return {
            "percentiles": {str(key): float(value) for key, value in zip(percentiles, values)},
            "minimum": float(np.min(array)),
            "maximum": float(np.max(array)),
        }
    report = {
        "map_direction": DEFORMATION_DIRECTION,
        "displacement_definition": "phi(x) - x for the stated phi direction",
        "grid": "saved velocity grid",
        "shape": list(shape),
        "velocity_magnitude_um": stats(velocity_magnitude),
        "displacement_magnitude_um": stats(displacement),
        "jacobian_determinant": {**stats(jacobian), "nonpositive_count": nonpositive},
        "interpretation": (
            "Registration-model deformation; contributions may include ex vivo processing, "
            "histological distortion, missing data, image contrast, and registration error. "
            "It is not a uniquely identifiable measurement of biological shrinkage."
        ),
    }
    figure, panels = plt.subplots(3, 3, figsize=(13, 11))
    centers = [length // 2 for length in shape]
    fields = [
        (displacement, "displacement magnitude", "magma", None),
        (jacobian, "Jacobian determinant", "coolwarm", 1.0),
    ]
    for row, (field, title, cmap, center) in enumerate(fields):
        views = [field[centers[0]], field[:, centers[1]], field[:, :, centers[2]]]
        for column, view in enumerate(views):
            kwargs = {}
            if center is not None:
                radius = max(abs(float(np.percentile(field, 1)) - center),
                             abs(float(np.percentile(field, 99)) - center))
                kwargs.update(vmin=center - radius, vmax=center + radius)
            panels[row, column].imshow(view.T, origin="lower", cmap=cmap, **kwargs)
            panels[row, column].set_title(f"{title} — axis {column}")
            panels[row, column].axis("off")
    skips = max(1, min(shape) // 20)
    for column in range(3):
        panel = panels[2, column]
        if column == 0:
            yy, xx = np.meshgrid(xv[1][::skips], xv[2][::skips], indexing="ij")
            panel.quiver(xx, yy, phi[2, centers[0], ::skips, ::skips] - xx,
                         phi[1, centers[0], ::skips, ::skips] - yy, angles="xy", scale_units="xy", scale=0.05)
        elif column == 1:
            yy, xx = np.meshgrid(xv[0][::skips], xv[2][::skips], indexing="ij")
            panel.quiver(xx, yy, phi[2, ::skips, centers[1], ::skips] - xx,
                         phi[0, ::skips, centers[1], ::skips] - yy, angles="xy", scale_units="xy", scale=0.05)
        else:
            yy, xx = np.meshgrid(xv[0][::skips], xv[1][::skips], indexing="ij")
            panel.quiver(xx, yy, phi[1, ::skips, ::skips, centers[2]] - xx,
                         phi[0, ::skips, ::skips, centers[2]] - yy, angles="xy", scale_units="xy", scale=0.05)
        panel.set_title(f"displacement vectors (20x) — axis {column}")
    figure.suptitle("Saved deformation diagnostics\n" + DEFORMATION_DIRECTION, fontsize=10)
    output = POST_DIR / "deformation_orthogonal_overview.png"
    _atomic_figure(output, figure)
    report["figure"] = str(output)
    return phi, report


def _registered_frame_axes(
    xJ: list[np.ndarray], baseline: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    inverse = np.linalg.inv(baseline)
    corners = np.asarray([
        [xJ[1][0], xJ[2][0], 1.0],
        [xJ[1][0], xJ[2][-1], 1.0],
        [xJ[1][-1], xJ[2][0], 1.0],
        [xJ[1][-1], xJ[2][-1], 1.0],
    ]).T
    registered = inverse @ corners
    return (
        np.linspace(float(registered[0].min()), float(registered[0].max()), len(xJ[1])),
        np.linspace(float(registered[1].min()), float(registered[1].max()), len(xJ[2])),
    )


def _warp_nissl_sections(samples, observed, axes, xJ, final, row_um, column_um):
    numerator = np.memmap(
        RUN_TMP / "registered_nissl_numerator.dat", mode="w+", dtype=np.float32,
        shape=(3, 2846, len(row_um), len(column_um)),
    )
    support = np.memmap(
        RUN_TMP / "registered_nissl_support.dat", mode="w+", dtype=np.float32,
        shape=(2846, len(row_um), len(column_um)),
    )
    numerator[:] = 0.0
    support[:] = 0.0
    rr, cc = np.meshgrid(row_um, column_um, indexing="ij")
    preserved = _preserves_source_grid()
    for count, index in enumerate(observed, 1):
        image, raw_support = read_section(samples[index], spatial_axes=[axes[1], axes[2]], preserve_source_grid=preserved)
        if not preserved:
            image, raw_support = downsample_section(image, raw_support)
        matrix = final[index]
        source_row = matrix[0, 0] * rr + matrix[0, 1] * cc + matrix[0, 2]
        source_column = matrix[1, 0] * rr + matrix[1, 1] * cc + matrix[1, 2]
        iy = (source_row - xJ[1][0]) / (xJ[1][1] - xJ[1][0])
        ix = (source_column - xJ[2][0]) / (xJ[2][1] - xJ[2][0])
        transformed_support = ndi.map_coordinates(
            raw_support, [iy, ix], order=1, mode="constant", cval=0.0, prefilter=False
        ).astype(np.float32)
        support[index] = transformed_support
        for channel in range(3):
            numerator[channel, index] = ndi.map_coordinates(
                image[channel] * raw_support, [iy, ix], order=1,
                mode="constant", cval=0.0, prefilter=False,
            )
        if count % 50 == 0:
            print(f"warped saved Nissl section {count}/{len(observed)}", flush=True)
    numerator.flush()
    support.flush()
    return numerator, support


def _warp_effective_match_weight(
    effective_match_weight, observed, xJ, final, row_um, column_um
):
    expected = (len(observed), len(xJ[1]), len(xJ[2]))
    weights = np.asarray(effective_match_weight, dtype=np.float32)
    if weights.shape != expected:
        raise RuntimeError(
            f"Effective match weight has shape {weights.shape}, expected {expected}"
        )
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise RuntimeError("Effective match weight must be finite and nonnegative")
    warped = np.memmap(
        RUN_TMP / "registered_effective_match_weight.dat",
        mode="w+", dtype=np.float32,
        shape=(len(xJ[0]), len(row_um), len(column_um)),
    )
    warped[:] = 0.0
    rr, cc = np.meshgrid(row_um, column_um, indexing="ij")
    for source_index, physical_index in enumerate(observed):
        matrix = final[physical_index]
        source_row = matrix[0, 0] * rr + matrix[0, 1] * cc + matrix[0, 2]
        source_column = matrix[1, 0] * rr + matrix[1, 1] * cc + matrix[1, 2]
        iy = (source_row - xJ[1][0]) / (xJ[1][1] - xJ[1][0])
        ix = (source_column - xJ[2][0]) / (xJ[2][1] - xJ[2][0])
        warped[physical_index] = ndi.map_coordinates(
            weights[source_index], [iy, ix], order=1, mode="constant",
            cval=0.0, prefilter=False,
        )
    warped.flush()
    return warped


def _validate_published_affine(A, rows, observed, row_um, column_um, mri_center_um):
    """Verify the MRI-to-restack affine used by the saved sampling chain."""
    affine = np.asarray(A, dtype=np.float64)
    center = np.asarray(mri_center_um, dtype=np.float64)
    mapped = affine[:3, :3] @ center + affine[:3, 3]
    serial_um = np.asarray([
        float(rows[index]["serial_z_center_mm"]) * 1000.0 for index in observed
    ])
    extents = (serial_um, np.asarray(row_um), np.asarray(column_um))
    if any(not axis[0] <= value <= axis[-1] for axis, value in zip(extents, mapped)):
        raise RuntimeError(
            f"MRI center maps outside published registered stack: {mapped.tolist()}"
        )
    plane_normal = np.linalg.solve(affine[:3, :3].T, [1.0, 0.0, 0.0])
    plane_normal /= np.linalg.norm(plane_normal)
    if not np.allclose(plane_normal[:2], 0.0, atol=1e-6, rtol=0.0):
        raise RuntimeError(
            "MRI midsagittal plane is not column-aligned in published coordinates: "
            f"{plane_normal.tolist()}"
        )
    return affine


def _publish_corrected_nissl(
    rows, observed, numerator, support, row_um, column_um, A, baseline
):
    """Persist the section warps already computed by `_warp_nissl_sections`."""
    if CORRECTED_NISSL.exists():
        raise RuntimeError(f"Corrected Nissl derivative exists: {CORRECTED_NISSL}")
    staging = Path(
        __import__("tempfile").mkdtemp(
            prefix=f".{CORRECTED_NISSL.name}.", dir=CORRECTED_NISSL.parent
        )
    )
    try:
        (staging / "metadata").mkdir()
        images = staging / "images" / "nissl"
        images.mkdir(parents=True)
        present = set(map(int, observed))
        output_rows = []
        spacing = float(column_um[1] - column_um[0])
        for index, original in enumerate(rows):
            row = dict(original)
            if index in present:
                positive = support[index] > 0
                corrected = np.zeros((len(row_um), len(column_um), 3), np.uint8)
                for channel in range(3):
                    values = np.zeros_like(support[index])
                    values[positive] = numerator[channel, index][positive] / support[index][positive]
                    corrected[..., channel] = np.rint(
                        np.clip(values, 0.0, 1.0) * 255.0
                    ).astype(np.uint8)
                output = images / f"allen_708424_nissl_{int(row['allen_section_number']):04d}.tif"
                Image.fromarray(corrected).save(
                    output, format="TIFF", compression="tiff_deflate"
                )
                _write_image_sidecar(
                    output,
                    shape_yx=corrected.shape[:2],
                    origin_xy_um=(float(column_um[0]), float(row_um[0])),
                    z_um=float(row["serial_z_center_mm"]) * 1000.0,
                    pixel_size_um=spacing,
                )
                row["prepared_relative_path"] = output.relative_to(staging).as_posix()
                row["prepared_sha256"] = sha256_file(output)
            else:
                for field in (
                    "stain", "allen_section_image_id", "allen_data_set_id",
                    "source_relative_path", "prepared_relative_path",
                    "source_sha256", "prepared_sha256",
                    "nominal_series_interval_um",
                ):
                    row[field] = ""
                row["image_present"] = "false"
                row["observation_class"] = "unobserved"
            output_rows.append(row)
        fields = list(rows[0])
        _write_rows(staging / "metadata" / "physical_sections.tsv", output_rows, fields)
        published_affine = _validate_published_affine(
            A,
            rows,
            observed,
            row_um,
            column_um,
            np.asarray(json.loads(MRI_PROV.read_text())["physical_center_mm"])
            * 1000.0,
        )
        _json(staging / "metadata" / "linear_restack.json", {
            "source_dataset": str(DATASET),
            "global_affine_mri_um_to_histology_um": (
                published_affine
            ).tolist(),
            "restack_axes_um": {
                "row": [float(row_um[0]), float(row_um[-1]), len(row_um)],
                "column": [float(column_um[0]), float(column_um[-1]), len(column_um)],
            },
        })
        _json(staging / "dataset.json", {
            "dataset": "Allen specimen 708424 corrected observed-left Nissl restack",
            "specimen_id": 708424,
            "space_name": "HIST_LINEAR_NISSL",
            "source_layer": str(DATASET),
            "physical_position_count": len(output_rows),
            "present_image_count": len(present),
            "nissl_count": len(present),
            "pv_count": 0,
            "prepared_canvas_shape_yx": [len(row_um), len(column_um)],
        })
        os.replace(staging, CORRECTED_NISSL)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _sample_chain_slabs(
    A, phi, xv, xI, xJ, row_um, column_um, numerator, support,
    effective_match_weight,
):
    shape = tuple(map(len, xI))
    reconstruction = np.memmap(
        RUN_TMP / "nissl_reconstruction.dat", mode="w+", dtype=np.float32,
        shape=(*shape, 3),
    )
    propagated = np.memmap(
        RUN_TMP / "nissl_support_on_mri.dat", mode="w+", dtype=np.float32, shape=shape
    )
    propagated_effective = (
        np.memmap(
            RUN_TMP / "effective_match_weight_on_mri.dat", mode="w+",
            dtype=np.float32, shape=shape,
        )
        if effective_match_weight is not None else None
    )
    for start in range(0, shape[1], 8):
        stop = min(start + 8, shape[1])
        X = np.stack(np.meshgrid(xI[0], xI[1][start:stop], xI[2], indexing="ij"))
        indices = [(X[axis] - xv[axis][0]) / (xv[axis][1] - xv[axis][0]) for axis in range(3)]
        flow = np.stack([
            ndi.map_coordinates(phi[axis], indices, order=1, mode="nearest", prefilter=False)
            for axis in range(3)
        ])
        query = (A[:3, :3] @ flow.reshape(3, -1) + A[:3, 3, None]).reshape(flow.shape)
        hist_indices = [
            (query[0] - xJ[0][0]) / (xJ[0][1] - xJ[0][0]),
            (query[1] - row_um[0]) / (row_um[1] - row_um[0]),
            (query[2] - column_um[0]) / (column_um[1] - column_um[0]),
        ]
        sampled_support = np.clip(ndi.map_coordinates(
            support, hist_indices, order=1, mode="constant", cval=0.0, prefilter=False
        ), 0.0, 1.0).astype(np.float32)
        sampled_effective = None
        if effective_match_weight is not None:
            sampled_effective = np.clip(ndi.map_coordinates(
                effective_match_weight, hist_indices, order=1, mode="constant",
                cval=0.0, prefilter=False,
            ), 0.0, 1.0).astype(np.float32)
            sampled_effective[sampled_support <= 0.0] = 0.0
        output = np.zeros((*sampled_support.shape, 3), dtype=np.float32)
        positive = sampled_support > 0.0
        for channel in range(3):
            sampled = ndi.map_coordinates(
                numerator[channel], hist_indices, order=1,
                mode="constant", cval=0.0, prefilter=False,
            )
            output[..., channel][positive] = sampled[positive] / sampled_support[positive]
        if (not np.all(np.isfinite(output))
                or not np.all(np.isfinite(sampled_support))
                or (sampled_effective is not None
                    and not np.all(np.isfinite(sampled_effective)))):
            raise RuntimeError("Nonfinite Nissl reconstruction slab")
        if np.any(output[~positive] != 0.0):
            raise RuntimeError("Nissl intensity is nonzero outside propagated support")
        reconstruction[:, start:stop] = output
        propagated[:, start:stop] = sampled_support
        if propagated_effective is not None:
            propagated_effective[:, start:stop] = sampled_effective
        reconstruction.flush()
        propagated.flush()
        if propagated_effective is not None:
            propagated_effective.flush()
        print(f"sampled saved Nissl MRI slab {start}:{stop}", flush=True)
    if not np.any(propagated > 0) or not np.any(reconstruction != 0):
        raise RuntimeError("Saved-output Nissl reconstruction is blank")
    return reconstruction, propagated, propagated_effective


def _mri_nissl_figures(
    mri, xI, reconstruction, support, effective_match_weight,
    mri_midline_um: float,
    *,
    grid_name: str = "coarse",
):
    axis0 = np.asarray(xI[0], dtype=np.float64)
    if not np.all(np.diff(axis0) > 0.0):
        raise RuntimeError("MRI axis 0 must be strictly increasing")
    if not axis0[0] <= mri_midline_um <= axis0[-1]:
        raise RuntimeError("MRI midsagittal coordinate is outside axis 0")
    midline_display_position = float(
        np.interp(mri_midline_um, axis0, np.arange(len(axis0)))
    )
    occupied = np.flatnonzero(np.any(support > 0.0, axis=(0, 2)))
    positions = occupied[
        _uniform_representative_positions(len(occupied), 9)]
    figure, panels = plt.subplots(4, len(positions), figsize=(27, 11))
    for column, position in enumerate(positions):
        base = np.asarray(mri[0, :, position, :], dtype=np.float32)
        lo, hi = np.percentile(base, [1, 99])
        base = np.clip((base - lo) / (hi - lo + 1e-8), 0, 1)
        nissl = np.asarray(reconstruction[:, position, :])
        effective = np.asarray(
            support[:, position, :]
            if effective_match_weight is None
            else effective_match_weight[:, position, :]
        )
        display_nissl = np.clip(nissl, 0.0, 1.0)
        departure_from_white = np.sqrt(
            np.mean((1.0 - display_nissl) ** 2, axis=-1)
        )
        alpha = (
            np.clip(effective, 0.0, 1.0) * departure_from_white
        )[..., None]
        nissl_false_color = np.array([1.0, 0.0, 0.7], dtype=np.float32)
        overlay = (
            (1.0 - alpha) * base[..., None]
            + alpha * nissl_false_color
        )
        for row, (image, cmap) in enumerate(
            ((base, "gray"), (nissl, None), (effective, "viridis"), (overlay, None))
        ):
            display = image.transpose(1, 0, 2) if image.ndim == 3 else image.T
            panels[row, column].imshow(
                display, origin="lower", cmap=cmap)
            panels[row, column].axis("off")
        for row in (0, 1, 3):
            panels[row, column].axvline(
                midline_display_position, color="cyan", linestyle="--",
                linewidth=0.8,
            )
        panels[0, column].set_title(f"axis-1 {position}\n{xI[1][position] / 1000:.1f} mm")
    weight_label = (
        "Propagated Nissl support"
        if effective_match_weight is None
        else "Effective match weight (WM × W0)"
    )
    overlay_label = (
        "MRI + support-weighted Nissl"
        if effective_match_weight is None
        else "MRI + weighted Nissl"
    )
    row_labels = ("MRI", "Registered Nissl", weight_label, overlay_label)
    for row, label in enumerate(row_labels):
        box = panels[row, 0].get_position()
        figure.text(0.005, (box.y0 + box.y1) / 2.0, label,
                    rotation=90, va="center", ha="left")
    figure.text(
        0.995, 0.01, "cyan dashed line = MRI midsagittal plane",
        color="cyan", ha="right", va="bottom",
    )
    figure.subplots_adjust(left=0.075, bottom=0.06, top=0.90)
    figure.suptitle(
        "Sparse observed-section reconstruction / sparse support transported onto "
        "the MRI grid\nOrthogonal appearance is not a direct anatomical registration "
        "QC prior to densification"
    )
    overview = POST_DIR / "mri_nissl_registration_overview.png"
    _atomic_figure(overview, figure)
    figure, panels = plt.subplots(2, 3, figsize=(14, 9))
    centers = [len(axis) // 2 for axis in xI]
    for column in range(3):
        if column == 0:
            nview, mview = reconstruction[centers[0]], mri[0, centers[0]]
            extent = [xI[2][0] / 1000, xI[2][-1] / 1000, xI[1][0] / 1000, xI[1][-1] / 1000]
        elif column == 1:
            nview, mview = reconstruction[:, centers[1]], mri[0, :, centers[1]]
            extent = [xI[2][0] / 1000, xI[2][-1] / 1000, xI[0][0] / 1000, xI[0][-1] / 1000]
        else:
            nview, mview = reconstruction[:, :, centers[2]], mri[0, :, :, centers[2]]
            extent = [xI[1][0] / 1000, xI[1][-1] / 1000, xI[0][0] / 1000, xI[0][-1] / 1000]
        panels[0, column].imshow(np.moveaxis(nview, -1, 0).transpose(1, 2, 0)
                                 if nview.shape[-1] == 3 else nview,
                                 origin="lower", extent=extent, aspect="auto")
        panels[1, column].imshow(mview.T, origin="lower", extent=extent, aspect="auto", cmap="gray")
        panels[0, column].set_title(f"registered Nissl — axis {column}")
        panels[1, column].set_title(f"MRI — axis {column}")
    figure.suptitle(
        f"{grid_name.capitalize()} MRI-grid sparse observed-section reconstruction "
        "/ sparse transported support\nOrthogonal appearance is not a direct "
        "anatomical registration QC prior to densification"
    )
    orthogonal = POST_DIR / "registered_nissl_orthogonal_overview.png"
    _atomic_figure(orthogonal, figure)
    return {"registration_overview": str(overview), "orthogonal_overview": str(orthogonal)}



class _StoredOmeMetadata:
    """Minimal group facade for established pure OME metadata helpers."""
    def __init__(self, path: Path):
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.attrs = payload.get("attributes", {})


def _annotation_sources(rows, samples, axes, observed) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]], list[tuple[int, tuple[int, ...]]]]:
    inventory_path = ANNOTATION_DATASET / "metadata/annotations.tsv"
    symmetry_path = ANNOTATION_DATASET / "metadata/symmetry.json"
    canvas_path = DATASET / "metadata/loader_canvas_audit.json"
    symmetry = json.loads(symmetry_path.read_text())
    canvas = json.loads(canvas_path.read_text())
    if symmetry.get("symmetric_space") != "HIST_SYMMETRIC":
        raise RuntimeError("Annotation derivative is not in HIST_SYMMETRIC")
    if symmetry.get("image_operation") != "exact_reflection_and_union_without_interpolation":
        raise RuntimeError("Unexpected symmetrization history")
    expected_shape = [len(axes[1]), len(axes[2])]
    if (
        symmetry.get("bilateral_shape_yx") != expected_shape
        or symmetry.get("pixel_size_um") != canvas.get("target_spacing_um")
    ):
        raise RuntimeError("Unexpected symmetric annotation geometry")
    if ANNOTATION_DATASET.resolve() != DATASET.resolve():
        parent_recorded = symmetry.get("parent_nissl_derivative")
        if (
            not isinstance(parent_recorded, str)
            or Path(parent_recorded).resolve() != DATASET.resolve()
        ):
            raise RuntimeError(
                "Annotation derivative has the wrong parent Nissl derivative"
            )
        if (
            symmetry.get("parent_nissl_dataset_json_sha256")
            != checksum(DATASET / "dataset.json")
        ):
            raise RuntimeError("Annotation parent Nissl checksum differs")
    if not np.allclose(symmetry.get("bilateral_origin_xy_um"), [axes[2][0], axes[1][0]]):
        raise RuntimeError("Symmetric derivative origin differs from loader axes")
    with inventory_path.open(encoding="utf-8", newline="") as stream:
        inventory = list(csv.DictReader(stream, delimiter="\t"))
    by_section: dict[int, list[dict[str, str]]] = {}
    for row in inventory:
        by_section.setdefault(int(row["section_number"]), []).append(row)
    if len(by_section) != 106:
        raise RuntimeError(f"Expected 106 annotation sections, found {len(by_section)}")
    observed_set = set(map(int, observed))
    physical_by_allen = {int(row["allen_section_number"]): index for index, row in enumerate(rows)}
    audits = []
    sources = {}
    merged_color_layers = []
    label_metadata: dict[int, dict[str, Any]] = {}
    for allen in sorted(by_section):
        if allen not in physical_by_allen:
            raise RuntimeError(f"Annotated Allen section {allen} is absent from physical manifest")
        index = physical_by_allen[allen]
        if index not in observed_set or rows[index]["stain"] != "nissl":
            raise RuntimeError(f"Annotated Allen section {allen} has no observed Nissl transform")
        prepared = VIEW / samples[index]["sample_id"]
        with Image.open(prepared) as image:
            prepared_shape = [image.height, image.width]
        if prepared_shape != expected_shape:
            raise RuntimeError(f"Prepared Nissl geometry mismatch for Allen {allen}")
        package_path = ANNOTATION_ZARR / f"section-{allen:04d}.ome.zarr"
        root = _StoredOmeMetadata(package_path / "zarr.json")
        labels_container = _StoredOmeMetadata(package_path / "labels/zarr.json")
        if dict(root.attrs).get("allen", {}).get("section_number") != allen:
            raise RuntimeError(f"OME Allen identity mismatch for section {allen}")
        names = _label_names(labels_container)
        expected_names = [f"group-{row['graphic_group_id']}" for row in by_section[allen]]
        if names != expected_names:
            raise RuntimeError(f"Declared annotation order mismatch for Allen {allen}")
        arrays = []
        colors_for_section = []
        paths = []
        for record, name in zip(by_section[allen], names, strict=True):
            if record.get("sampling") != "categorical_nearest_neighbor":
                raise RuntimeError(f"Noncategorical annotation sampling for Allen {allen}")
            if record.get("right_origin") != "synthetically_reflected":
                raise RuntimeError(f"Unexpected hemisphere convention for Allen {allen}")
            path = (ANNOTATION_DATASET / record["path"]).resolve()
            try:
                path.relative_to(ANNOTATION_DATASET.resolve())
            except ValueError as exc:
                raise RuntimeError(f"Unsafe annotation path: {path}") from exc
            if checksum(path) != record["sha256"]:
                raise RuntimeError(f"Annotation checksum mismatch: {path}")
            labels = tifffile.imread(path)
            if labels.shape != tuple(expected_shape) or not np.issubdtype(labels.dtype, np.integer):
                raise RuntimeError(f"Annotation raster geometry/dtype mismatch: {path}")
            arrays.append(labels.astype(np.uint32, copy=False))
            paths.append(str(path))
            group = _StoredOmeMetadata(
                package_path / "labels" / name / "zarr.json"
            )
            group_colors = _label_colors(group, f"label group {name}")
            colors_for_section.append(group_colors)
            attrs = dict(group.attrs)["ome"]["image-label"]
            for item in attrs.get("properties", []):
                value = int(item["label-value"])
                current = {
                    "label_id": value,
                    "name": str(item.get("name", "")),
                    "acronym": str(item.get("acronym", "")),
                }
                if value in label_metadata and label_metadata[value] != current:
                    raise RuntimeError(f"Conflicting label metadata for ID {value}")
                label_metadata[value] = current
        combined = _combined_display_map(arrays)
        merged_color_layers.append(_merged_colors(colors_for_section))
        nissl_image, nissl_support = read_section(samples[index])
        if nissl_image.shape[1:] != combined.shape or nissl_support.shape != combined.shape:
            raise RuntimeError(f"Annotation/Nissl shape mismatch for Allen {allen}")
        audits.append({
            "allen_section": allen,
            "physical_index": int(index),
            "serial_z_um": float(axes[0][index]),
            "hemisphere_and_symmetrization": "observed left plus exact synthetic right reflection",
            "shape_yx": expected_shape,
            "axes": ["row/y", "column/x"],
            "origin_yx_um": [float(axes[1][0]), float(axes[2][0])],
            "axis_direction": ["increasing", "increasing"],
            "crop_and_padding": "shared preserved canonical symmetric source grid",
            "reflection": symmetry["image_operation"],
            "resampling": "prepared 200-um grid; annotation categorical_nearest_neighbor",
            "annotation_paths": paths,
            "prepared_nissl_path": str(prepared),
            "identity_proof_passed": True,
        })
        sources[index] = {"labels": combined, "support": nissl_support.astype(np.uint8)}
    all_colors = _merged_colors(merged_color_layers)
    return audits, sources, all_colors, label_metadata


def _annotation_transport(
    rows, samples, axes, observed, final, A, phi, xv, xI, xJ, row_um,
    column_um, affine, *, grid_name: str = "coarse",
):
    audits, sources, colors, label_metadata = _annotation_sources(rows, samples, axes, observed)
    labels_stack = np.memmap(
        RUN_TMP / "registered_annotation_labels.dat", mode="w+", dtype=np.uint32,
        shape=(2846, len(row_um), len(column_um)),
    )
    support_stack = np.memmap(
        RUN_TMP / "registered_annotation_support.dat", mode="w+", dtype=np.uint8,
        shape=labels_stack.shape,
    )
    labels_stack[:] = 0
    support_stack[:] = 0
    rr, cc = np.meshgrid(row_um, column_um, indexing="ij")
    source_ids = set()
    for index, source in sources.items():
        labels = source["labels"]
        support = source["support"]
        coarse_y = (xJ[1] - axes[1][0]) / (axes[1][1] - axes[1][0])
        coarse_x = (xJ[2] - axes[2][0]) / (axes[2][1] - axes[2][0])
        gy, gx = np.meshgrid(coarse_y, coarse_x, indexing="ij")
        coarse_labels = ndi.map_coordinates(
            labels, [gy, gx], order=0, mode="constant", cval=0, prefilter=False
        ).astype(np.uint32)
        coarse_support = ndi.map_coordinates(
            support, [gy, gx], order=0, mode="constant", cval=0, prefilter=False
        ).astype(np.uint8)
        matrix = final[index]
        source_row = matrix[0, 0] * rr + matrix[0, 1] * cc + matrix[0, 2]
        source_column = matrix[1, 0] * rr + matrix[1, 1] * cc + matrix[1, 2]
        iy = (source_row - xJ[1][0]) / (xJ[1][1] - xJ[1][0])
        ix = (source_column - xJ[2][0]) / (xJ[2][1] - xJ[2][0])
        transported_support = ndi.map_coordinates(
            coarse_support, [iy, ix], order=0, mode="constant", cval=0, prefilter=False
        ).astype(np.uint8)
        transported_labels = ndi.map_coordinates(
            coarse_labels, [iy, ix], order=0, mode="constant", cval=0, prefilter=False
        ).astype(np.uint32)
        transported_labels[transported_support == 0] = 0
        labels_stack[index] = transported_labels
        support_stack[index] = transported_support
        source_ids.update(int(value) for value in np.unique(labels) if value != 0)
    shape = tuple(map(len, xI))
    output_labels = np.memmap(
        RUN_TMP / "annotation_labels_on_mri.dat", mode="w+", dtype=np.uint32, shape=shape
    )
    output_support = np.memmap(
        RUN_TMP / "annotation_support_on_mri.dat", mode="w+", dtype=np.uint8, shape=shape
    )
    for start in range(0, shape[1], 8):
        stop = min(start + 8, shape[1])
        X = np.stack(np.meshgrid(xI[0], xI[1][start:stop], xI[2], indexing="ij"))
        indices = [(X[axis] - xv[axis][0]) / (xv[axis][1] - xv[axis][0]) for axis in range(3)]
        flow = np.stack([
            ndi.map_coordinates(phi[axis], indices, order=1, mode="nearest", prefilter=False)
            for axis in range(3)
        ])
        query = (A[:3, :3] @ flow.reshape(3, -1) + A[:3, 3, None]).reshape(flow.shape)
        hist_indices = [
            (query[0] - xJ[0][0]) / (xJ[0][1] - xJ[0][0]),
            (query[1] - row_um[0]) / (row_um[1] - row_um[0]),
            (query[2] - column_um[0]) / (column_um[1] - column_um[0]),
        ]
        slab_support = ndi.map_coordinates(
            support_stack, hist_indices, order=0, mode="constant", cval=0, prefilter=False
        ).astype(np.uint8)
        slab_labels = ndi.map_coordinates(
            labels_stack, hist_indices, order=0, mode="constant", cval=0, prefilter=False
        ).astype(np.uint32)
        slab_labels[slab_support == 0] = 0
        output_labels[:, start:stop] = slab_labels
        output_support[:, start:stop] = slab_support
        output_labels.flush()
        output_support.flush()
    output_ids = {int(value) for value in np.unique(output_labels) if value != 0}
    if not np.any(output_support) or not output_ids:
        raise RuntimeError("Sparse observed-plane annotation transport is empty")
    if not output_ids.issubset(source_ids):
        raise RuntimeError("Annotation transport created an unknown label ID")
    if np.any(output_labels[np.asarray(output_support) == 0] != 0):
        raise RuntimeError("Annotation labels are nonzero outside direct support")
    label_path = ANNOTATION_DIR / (
        f"combined_annotation_labels_on_{grid_name}_mri_grid.nii"
    )
    support_path = ANNOTATION_DIR / (
        f"annotation_support_on_{grid_name}_mri_grid.nii"
    )
    _atomic_nifti(label_path, output_labels, affine, np.uint32)
    _atomic_nifti(support_path, output_support, affine, np.uint8)
    lookup_lines = ["label_id\tname\tacronym\tr\tg\tb\ta\n"]
    color_by_id = dict(colors)
    for value in sorted(source_ids):
        metadata = label_metadata.get(value, {"name": "", "acronym": ""})
        rgba = color_by_id.get(value)
        if rgba is None:
            raise RuntimeError(f"No established color for source label {value}")
        lookup_lines.append(
            f"{value}\t{metadata['name']}\t{metadata['acronym']}\t"
            + "\t".join(map(str, rgba)) + "\n"
        )
    lookup_path = ANNOTATION_DIR / "annotation_label_lookup.tsv"
    _atomic_text(lookup_path, "".join(lookup_lines))
    per_label = {
        str(value): int(np.count_nonzero(output_labels == value)) for value in sorted(output_ids)
    }
    report = {
        "derivative_kind": "sparse_direct_observed_planes",
        "dense_segmentation": False,
        "nearest_annotated_section_extrusion_produced": False,
        "source_observed_plane_count": len(audits),
        "source_observed_physical_indices": [item["physical_index"] for item in audits],
        "identity_audits": audits,
        "source_label_ids": sorted(source_ids),
        "output_label_ids": sorted(output_ids),
        "output_ids_subset_of_source": True,
        "direct_observation_supported_voxels": int(np.count_nonzero(output_support)),
        "direct_observation_coverage_fraction": float(np.count_nonzero(output_support) / output_support.size),
        "per_label_voxel_counts": per_label,
        "zero_outside_direct_support": True,
        "interpolation": "nearest-neighbor for labels and binary support",
        "labels": str(label_path),
        "support": str(support_path),
        "lookup": str(lookup_path),
    }
    report_path = ANNOTATION_DIR / "annotation_transport_report.json"
    atomic_json(report_path, report)
    report["report"] = str(report_path)
    return output_labels, output_support, colors, report


def _annotation_qc(
    mri, xI, labels, support, colors, *, grid_name: str = "coarse"
):
    occupied_positions = np.flatnonzero(np.any(support != 0, axis=(0, 2)))
    if occupied_positions.size < 1:
        raise RuntimeError("No supported annotation planes available for QC")
    chosen = occupied_positions[_uniform_representative_positions(len(occupied_positions), 9)]
    color_lookup = dict(colors)
    figure, panels = plt.subplots(4, len(chosen), figsize=(27, 11))
    for column, position in enumerate(chosen):
        base = np.asarray(mri[0, :, position, :], dtype=np.float32)
        lo, hi = np.percentile(base, [1, 99])
        gray = np.clip((base - lo) / (hi - lo + 1e-8), 0, 1)
        plane = np.asarray(labels[:, position, :], dtype=np.uint32)
        direct = np.asarray(support[:, position, :], dtype=np.uint8)
        rgb = np.zeros((*plane.shape, 3), dtype=np.uint8)
        for value in np.unique(plane):
            if value:
                rgba = color_lookup[int(value)]
                rgb[plane == value] = rgba[:3]
        boundary = _internal_boundary_mask(plane)
        overlay = np.repeat((gray * 255).astype(np.uint8)[..., None], 3, axis=2)
        overlay[boundary & (direct != 0)] = np.asarray([255, 0, 255], dtype=np.uint8)
        panels[0, column].imshow(gray.T, origin="lower", cmap="gray")
        panels[1, column].imshow(rgb.transpose(1, 0, 2), origin="lower")
        panels[2, column].imshow(overlay.transpose(1, 0, 2), origin="lower")
        panels[3, column].imshow(direct.T, origin="lower", cmap="gray", vmin=0, vmax=1)
        panels[0, column].set_title(f"direct plane {position}\n{xI[1][position]/1000:.1f} mm")
        for row in range(4):
            panels[row, column].axis("off")
    for row, title in enumerate(("MRI", "categorical labels", "boundaries over MRI", "direct support")):
        panels[row, 0].set_ylabel(title)
    figure.suptitle(
        f"Sparse directly observed annotation planes on {grid_name} MRI — "
        "manual boundary review required"
    )
    output = ANNOTATION_DIR / "annotation_boundaries_on_mri_overview.png"
    _atomic_figure(output, figure)
    return str(output)


def _validate_pngs(paths: list[Path]) -> dict[str, Any]:
    report = {}
    for path in paths:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            array = np.asarray(image.convert("RGB"))
        if array.size == 0 or np.all(array == array.reshape(-1, 3)[0]):
            raise RuntimeError(f"Blank QC figure: {path}")
        report[str(path)] = {"size_px": [int(array.shape[1]), int(array.shape[0])], "nonblank": True}
    return report


def postprocess(*, native_resolution: bool = False):
    start = time.monotonic()
    grid_name = "native" if native_resolution else "coarse"
    annotations_enabled = (
        ANNOTATION_DATASET / "metadata" / "annotations.tsv"
    ).is_file()
    if native_resolution and annotations_enabled:
        symmetry_path = ANNOTATION_DATASET / "metadata/symmetry.json"
        if not symmetry_path.is_file():
            annotations_enabled = False
        else:
            symmetry = json.loads(symmetry_path.read_text(encoding="utf-8"))
            canvas = json.loads(
                (DATASET / "metadata/loader_canvas_audit.json").read_text(
                    encoding="utf-8"
                )
            )
            annotations_enabled = (
                symmetry.get("pixel_size_um") == 200.0
                and symmetry.get("bilateral_shape_yx")
                == canvas.get("accepted_canvas_shape_yx")
                and symmetry.get("bilateral_origin_xy_um")
                == [
                    canvas.get("global_translation_xy_um", [None, None])[0],
                    canvas.get("global_translation_xy_um", [None, None])[1],
                ]
            )
    if POST_DIR.exists() or (annotations_enabled and ANNOTATION_DIR.exists()):
        raise RuntimeError("Refusing to overwrite an existing postprocess output directory")
    POST_DIR.mkdir(parents=True)
    if annotations_enabled:
        ANNOTATION_DIR.mkdir(parents=True)
    if native_resolution:
        registration = load_checkpoint(
            "registration",
            accepted_statuses=("complete", "complete_through_scale"),
        )
        checkpoints = {"registration": registration}
        (
            numerical_path,
            raw_history_paths,
            effective_match_weight_path,
            before_hashes,
            native_validation,
        ) = _native_postprocess_sources(registration)
    else:
        checkpoints = {
            "atlas-free": load_checkpoint("atlas-free"),
            "registration": load_checkpoint("registration"),
        }
        before_hashes, native_validation = _source_inventory(checkpoints)
        numerical_path = REG_DIR / "full_coarse_numerical_outputs.npz"
        raw_history_paths = None
        effective_match_weight_path = (
            REG_DIR / "final_observed_effective_match_weight.npy"
        )
    em, rows, samples, axes, observed = load_context()
    if len(observed) != 641 or len(rows) - len(observed) != 2205:
        raise RuntimeError("Observed/unsupported Nissl counts changed")
    with np.load(numerical_path) as saved:
        required = {"A", "A2d", "v", "xv0", "xv1", "xv2",
                    "xI0", "xI1", "xI2", "xJ0", "xJ1", "xJ2", "observed"}
        if set(saved.files) != required:
            raise RuntimeError(f"Saved numerical package keys changed: {saved.files}")
        A = finite("A", saved["A"]).astype(np.float64)
        final = finite("A2d", saved["A2d"]).astype(np.float64)
        v = finite("v", saved["v"]).astype(np.float32)
        xv = [finite(f"xv{axis}", saved[f"xv{axis}"]).astype(np.float64) for axis in range(3)]
        xI = [finite(f"xI{axis}", saved[f"xI{axis}"]).astype(np.float64) for axis in range(3)]
        xJ = [finite(f"xJ{axis}", saved[f"xJ{axis}"]).astype(np.float64) for axis in range(3)]
        saved_observed = np.asarray(saved["observed"], dtype=np.int64)
    if A.shape != (4, 4) or final.shape != (2846, 3, 3):
        raise RuntimeError("Saved affine or section-transform shape changed")
    if not np.array_equal(saved_observed, observed):
        raise RuntimeError("Saved observed indices differ from canonical manifest")
    preserved = _preserves_source_grid()
    expected_coarse_axes = (
        tuple(np.asarray(axis) for axis in axes[1:])
        if native_resolution
        else coarse_spatial_axes(axes, preserve_source_grid=preserved)
    )
    expected_xJ_shape = (2846, *(len(axis) for axis in expected_coarse_axes))
    if tuple(map(len, xJ)) != expected_xJ_shape:
        raise RuntimeError("Saved histology axes changed")
    expected_mri_spacing = [200] * 3 if native_resolution else [800] * 3
    if not np.allclose(
        np.abs([np.diff(axis).mean() for axis in xI]),
        expected_mri_spacing, atol=1e-3,
    ):
        raise RuntimeError("Saved MRI working-grid spacing changed")
    saved_histology_spacing = np.asarray(
        [np.diff(axis).mean() for axis in xJ], dtype=np.float64
    )
    if preserved:
        source_spacing = float(
            json.loads(
                (DATASET / "metadata" / "loader_canvas_audit.json").read_text()
            )["target_spacing_um"]
        )
        expected_histology_spacing = [50.0, source_spacing, source_spacing]
    else:
        expected_histology_spacing = [50.0, 800.0, 800.0]
    if not np.allclose(
        saved_histology_spacing, expected_histology_spacing, atol=1e-6, rtol=0.0
    ):
        raise RuntimeError("Saved histology working grid changed")
    affine, geometry = _coarse_affine(
        xI,
        expected_shape=None if native_resolution else (237, 284, 254),
        grid_name=grid_name,
    )
    baseline, residual_report, comparisons = _final_residual_diagnostics(
        rows, axes, observed, final,
        comparison_initializer=("identity" if native_resolution else "atlas_free"),
    )
    objective_report = _objective_figure(raw_history_paths)
    phi, deformation_report = _deformation_products(em, xv, v)
    row_um, column_um = _registered_frame_axes(xJ, baseline)
    numerator, section_support = _warp_nissl_sections(
        samples, observed, axes, xJ, final, row_um, column_um
    )
    effective_match_weight = (
        _warp_effective_match_weight(
            np.load(effective_match_weight_path, mmap_mode="r"),
            observed, xJ, final, row_um, column_um,
        )
        if effective_match_weight_path is not None else None
    )
    velocity_max = float(np.max(np.abs(v)))
    observed_left = DATASET.resolve() == OBSERVED_LEFT_DATASET.resolve()
    registered_histology_figure = None
    if observed_left:
        registered_histology_figure = POST_DIR / (
            "observed_left_nissl_mri_affine_A2d_v0_"
            "registered_histology_frame_emlddmm_draw.png"
        )
        _emlddmm_stack_draw_qc(
            em,
            numerator[:, observed],
            section_support[observed],
            [xJ[0][observed], row_um, column_um],
            registered_histology_figure,
            "Observed-left registered Nissl only (final A2d; no MRI sampled)\n"
            "Frame/orientation check, not direct MRI/Nissl anatomical QC",
        )
    if observed_left and velocity_max > 1e-7:
        raise RuntimeError(
            f"Observed-left linear registration saved |v|max={velocity_max:.6g}"
        )
    if observed_left:
        _publish_corrected_nissl(
            rows, observed, numerator, section_support, row_um, column_um, A,
            baseline,
        )
    reconstruction, nissl_support, mri_effective_match_weight = _sample_chain_slabs(
        A, phi, xv, xI, xJ, row_um, column_um, numerator, section_support,
        effective_match_weight,
    )
    nissl_path = POST_DIR / f"nissl_reconstruction_on_{grid_name}_mri_grid.nii"
    nissl_support_path = POST_DIR / f"nissl_support_on_{grid_name}_mri_grid.nii"
    _atomic_nifti(nissl_path, reconstruction, affine, np.float32)
    _atomic_nifti(nissl_support_path, nissl_support, affine, np.float32)
    provenance = json.loads(MRI_PROV.read_text())
    native = load_pinned_mri_image(em, mri_path=MRI, provenance=provenance)
    if native_resolution:
        loaded_xI, mri = native.x, native.data
    else:
        loaded_xI, mri = em.downsample_image_domain(
            native.x, native.data, [4, 4, 4]
        )
    loaded_xI = [np.asarray(axis) for axis in loaded_xI]
    mri = finite(f"{grid_name} MRI", mri).astype(np.float32)
    del native
    gc.collect()
    if any(not np.allclose(left, right) for left, right in zip(loaded_xI, xI, strict=True)):
        raise RuntimeError(
            f"Reconstructed {grid_name} MRI axes differ from saved registration axes"
        )
    nissl_figures = _mri_nissl_figures(
        mri,
        xI,
        reconstruction,
        nissl_support,
        mri_effective_match_weight,
        float(provenance["physical_center_mm"][0]) * 1000.0,
        grid_name=grid_name,
    )
    figures = [
        POST_DIR / "mri_nissl_registration_overview.png",
        POST_DIR / "registered_nissl_orthogonal_overview.png",
        POST_DIR / "objective_histories.png",
        POST_DIR / "final_section_residual_traces.png",
        POST_DIR / "deformation_orthogonal_overview.png",
    ]
    if registered_histology_figure is not None:
        figures.append(registered_histology_figure)
    manual_review_items = [
        "sparse observed-section support transport on the MRI grid (not direct anatomical QC)",
        "orientation and physical extents",
        "deformation interpretation in the stated map direction",
    ]
    annotation_report = None
    if annotations_enabled:
        annotation_labels, annotation_support, colors, annotation_report = (
            _annotation_transport(
                rows, samples, axes, observed, final, A, phi, xv, xI, xJ,
                row_um, column_um, affine, grid_name=grid_name,
            )
        )
        annotation_qc = _annotation_qc(
            mri, xI, annotation_labels, annotation_support, colors,
            grid_name=grid_name,
        )
        annotation_report["boundary_qc"] = annotation_qc
        atomic_json(Path(annotation_report["report"]), {
            key: value for key, value in annotation_report.items() if key != "report"
        })
        figures.append(ANNOTATION_DIR / "annotation_boundaries_on_mri_overview.png")
        manual_review_items.insert(1, "annotation boundary alignment")
    png_validation = _validate_pngs(figures)
    after_hashes = {path: checksum(Path(path)) for path in before_hashes}
    if before_hashes != after_hashes:
        changed = sorted(path for path in before_hashes if before_hashes[path] != after_hashes[path])
        raise RuntimeError(f"Authoritative registration sources changed: {changed}")
    elapsed = time.monotonic() - start
    report = {
        "stage": "postprocess",
        "status": "review_required",
        "saved_transform_reconstruction_and_transport_only": True,
        "source_hashes_before": before_hashes,
        "source_hashes_after": after_hashes,
        "source_transforms_and_checkpoints_unchanged": True,
        "native_transform_validation": native_validation,
        "geometry": geometry,
        "shapes": {
            f"{grid_name}_mri": list(map(len, xI)),
            "histology_working_stack": [3, *expected_xJ_shape],
            "histology_support": list(expected_xJ_shape),
        },
        "spacings_um": {f"{grid_name}_mri": [float(np.diff(axis).mean()) for axis in xI],
                        "histology": [float(np.diff(axis).mean()) for axis in xJ]},
        "deformation": deformation_report,
        "objective_histories": objective_report,
        "final_residuals": residual_report,
        "flagged_section_gauge_invariant_comparisons": comparisons,
        "nissl": {
            "reconstruction": str(nissl_path),
            "description": (
                "Sparse observed-section reconstruction / sparse support transported "
                "onto the MRI grid. Orthogonal appearance is not direct anatomical "
                "registration QC prior to densification."
            ),
            "support": str(nissl_support_path),
            **nissl_figures,
            **(
                {"registered_histology_frame_emlddmm_draw": str(registered_histology_figure)}
                if registered_histology_figure is not None else {}
            ),
            "finite": True,
            "nonempty": True,
            "zero_intensity_outside_support": True,
        },
        "png_validation": png_validation,
        "manual_visual_review": {
            "required": True,
            "items": manual_review_items,
        },
        "elapsed_seconds": elapsed,
        "peak_process_group_rss_kib": peak_rss_kib(),
        "optimization_invoked_by_helper": False,
        "atlas_free_invoked_by_helper": False,
        "registration_invoked_by_helper": False,
        "source_rasterization_invoked_by_helper": False,
        "refinement_invoked_by_helper": False,
        "production_refinement_launched": False,
    }
    if annotation_report is not None:
        report["annotations"] = annotation_report
    report_path = POST_DIR / "postprocess_report.json"
    atomic_json(report_path, report)
    checkpoint = {**report, "report": str(report_path)}
    atomic_json(CHECKPOINTS / "postprocess.json", checkpoint)
    atomic_json(
        OUTPUT / (
            "native_resolution_summary.json"
            if native_resolution else "full_coarse_summary.json"
        ),
        checkpoint,
    )
    print(json.dumps(checkpoint, indent=2), flush=True)

def _section_sample_indices(
    transform, row_um, column_um, *, source_row_um=None, source_column_um=None,
):
    """Return the physical-coordinate pullback shared by image and label warps."""
    source_row_um = row_um if source_row_um is None else source_row_um
    source_column_um = column_um if source_column_um is None else source_column_um
    rr, cc = np.meshgrid(row_um, column_um, indexing="ij")
    source_row = (
        transform[0, 0] * rr + transform[0, 1] * cc + transform[0, 2]
    )
    source_column = (
        transform[1, 0] * rr + transform[1, 1] * cc + transform[1, 2]
    )
    iy = (source_row - source_row_um[0]) / (
        source_row_um[1] - source_row_um[0]
    )
    ix = (source_column - source_column_um[0]) / (
        source_column_um[1] - source_column_um[0]
    )
    return iy, ix


def _warp_saved_section(
    image, support, transform, row_um, column_um,
    *, source_row_um=None, source_column_um=None,
):
    """Sample one prepared section into its baseline-factored registered frame."""
    iy, ix = _section_sample_indices(
        transform, row_um, column_um,
        source_row_um=source_row_um, source_column_um=source_column_um,
    )
    rr_shape = (len(row_um), len(column_um))
    transformed_support = ndi.map_coordinates(
        support, [iy, ix], order=1, mode="constant", cval=0.0,
        prefilter=False,
    ).astype(np.float32)
    transformed = np.zeros((image.shape[0], *rr_shape), dtype=np.float32)
    positive = transformed_support > 0.0
    for channel in range(3):
        numerator = ndi.map_coordinates(
            image[channel] * support,
            [iy, ix],
            order=1,
            mode="constant",
            cval=0.0,
            prefilter=False,
        )
        transformed[channel, positive] = (
            numerator[positive] / transformed_support[positive]
        )
    return transformed, transformed_support


def _publish_saved_overview(source: Path, destination: Path) -> None:
    """Publish one completed file atomically, including across filesystems."""
    if destination.exists():
        raise RuntimeError(f"Refusing to overwrite overview output: {destination}")
    if os.stat(source.parent).st_dev == os.stat(destination.parent).st_dev:
        os.replace(source, destination)
        return
    local = destination.parent / f".{destination.name}.{os.getpid()}.tmp"
    if local.exists():
        raise RuntimeError(f"Destination-local temporary file exists: {local}")
    try:
        with source.open("rb") as incoming, local.open("xb") as outgoing:
            shutil.copyfileobj(incoming, outgoing, length=1024 * 1024)
            outgoing.flush()
            os.fsync(outgoing.fileno())
        os.replace(local, destination)
    finally:
        if local.exists():
            local.unlink()


def _saved_atlas_free_stack_overview() -> dict[str, Any]:
    """Render saved-transform atlas-free QC without invoking a pipeline stage."""
    png = ATLAS_DIR / "atlas_free_transformed_stack_overview.png"
    report_path = ATLAS_DIR / "atlas_free_transformed_stack_overview.json"
    if png.exists() or report_path.exists():
        raise RuntimeError("Saved-transform overview output already exists")
    checkpoint_path = CHECKPOINTS / "atlas-free.json"
    registration_checkpoint = CHECKPOINTS / "registration.json"
    sources = {
        "observed_A2d": ATLAS_DIR / "observed_A2d.npy",
        "bookkeeping_frame": ATLAS_DIR / "common_bookkeeping_frame.txt",
        "observed_indices": ATLAS_DIR / "observed_physical_indices.npy",
        "atlas_free_checkpoint": checkpoint_path,
    }
    registration_before = registration_checkpoint.exists()
    before_hashes = {name: checksum(path) for name, path in sources.items()}
    checkpoint = load_checkpoint("atlas-free")
    if checkpoint.get("status") != "complete":
        raise RuntimeError("Atlas-free checkpoint is not complete")

    em, rows, samples, axes, expected_observed = load_context()
    observed = np.load(sources["observed_indices"]).astype(np.int64)
    matrices = finite("observed A2d", np.load(sources["observed_A2d"]))
    baseline = finite("bookkeeping frame", np.loadtxt(sources["bookkeeping_frame"]))
    if matrices.shape != (641, 3, 3) or observed.shape != (641,):
        raise RuntimeError("Unexpected saved atlas-free transform dimensions")
    if not np.array_equal(observed, expected_observed):
        raise RuntimeError("Saved observed indices do not match the Nissl manifest")
    if np.unique(observed).size != 641 or observed.min() < 0 or observed.max() >= 2846:
        raise RuntimeError("Saved observed indices are invalid")

    residual = np.linalg.inv(baseline)[None] @ matrices
    recomposition_error = float(
        np.max(np.abs(baseline[None] @ residual - matrices))
    )
    if recomposition_error > 1e-8:
        raise RuntimeError("Baseline-factored transforms do not recompose")
    row_translation = residual[:, 0, 2]
    column_translation = residual[:, 1, 2]
    rotation = np.degrees(
        np.arctan2(residual[:, 1, 0], residual[:, 0, 0])
    )
    serial = np.asarray(axes[0], dtype=np.float64)
    observed_serial = serial[observed]
    preserved = _preserves_source_grid()
    row_um, column_um = coarse_spatial_axes(axes, preserve_source_grid=preserved)
    coarse_shape = (len(row_um), len(column_um))
    representative_positions = _uniform_representative_positions(641, 9)
    representative_lookup = {
        int(position): slot
        for slot, position in enumerate(representative_positions)
    }
    representative_original = [None] * len(representative_positions)
    representative_transformed = [None] * len(representative_positions)

    original_serial_row_num = np.zeros((3, 2846, coarse_shape[0]), np.float32)
    transformed_serial_row_num = np.zeros((3, 2846, coarse_shape[0]), np.float32)
    original_serial_column_num = np.zeros((3, 2846, coarse_shape[1]), np.float32)
    transformed_serial_column_num = np.zeros((3, 2846, coarse_shape[1]), np.float32)
    original_serial_row_den = np.zeros((2846, coarse_shape[0]), np.float32)
    transformed_serial_row_den = np.zeros((2846, coarse_shape[0]), np.float32)
    original_serial_column_den = np.zeros((2846, coarse_shape[1]), np.float32)
    transformed_serial_column_den = np.zeros((2846, coarse_shape[1]), np.float32)

    for order, physical_index in enumerate(observed):
        image, support = read_section(
            samples[physical_index], em, [axes[1], axes[2]],
            preserve_source_grid=preserved,
        )
        if not preserved:
            image, support = downsample_section(image, support)
        if image.shape != (3, *coarse_shape) or support.shape != coarse_shape:
            raise RuntimeError(
                f"Unexpected streamed section shape at {physical_index}"
            )
        transformed, transformed_support = _warp_saved_section(
            image, support, residual[order], row_um, column_um
        )
        weighted = image * support[None]
        transformed_weighted = transformed * transformed_support[None]
        original_serial_row_num[:, physical_index] = weighted.sum(axis=2)
        transformed_serial_row_num[:, physical_index] = (
            transformed_weighted.sum(axis=2)
        )
        original_serial_column_num[:, physical_index] = weighted.sum(axis=1)
        transformed_serial_column_num[:, physical_index] = (
            transformed_weighted.sum(axis=1)
        )
        original_serial_row_den[physical_index] = support.sum(axis=1)
        transformed_serial_row_den[physical_index] = (
            transformed_support.sum(axis=1)
        )
        original_serial_column_den[physical_index] = support.sum(axis=0)
        transformed_serial_column_den[physical_index] = (
            transformed_support.sum(axis=0)
        )
        if order in representative_lookup:
            slot = representative_lookup[order]
            representative_original[slot] = image.copy()
            representative_transformed[slot] = transformed.copy()
        if (order + 1) % 50 == 0:
            print(f"overview streamed {order + 1}/641", flush=True)

    if any(image is None for image in representative_original):
        raise RuntimeError("A representative original section was not retained")
    if any(image is None for image in representative_transformed):
        raise RuntimeError("A representative transformed section was not retained")
    expected_shapes = {
        "original_serial_row_num": (3, 2846, coarse_shape[0]),
        "transformed_serial_row_num": (3, 2846, coarse_shape[0]),
        "original_serial_column_num": (3, 2846, coarse_shape[1]),
        "transformed_serial_column_num": (3, 2846, coarse_shape[1]),
        "original_serial_row_den": (2846, coarse_shape[0]),
        "transformed_serial_row_den": (2846, coarse_shape[0]),
        "original_serial_column_den": (2846, coarse_shape[1]),
        "transformed_serial_column_den": (2846, coarse_shape[1]),
    }
    for name, expected in expected_shapes.items():
        if locals()[name].shape != expected:
            raise RuntimeError(
                f"{name} has shape {locals()[name].shape}, expected {expected}"
            )

    jump_order = np.argsort(np.abs(np.diff(row_translation)))[-3:][::-1]
    jump_records = []
    marked_orders = set()
    for left in jump_order:
        right = int(left + 1)
        marked_orders.update((int(left), right))
        jump_records.append(
            {
                "left_allen_section": int(
                    rows[observed[left]]["allen_section_number"]
                ),
                "right_allen_section": int(
                    rows[observed[right]]["allen_section_number"]
                ),
                "serial_gap_um": float(
                    observed_serial[right] - observed_serial[left]
                ),
                "absolute_row_translation_jump_um": float(
                    abs(row_translation[right] - row_translation[left])
                ),
            }
        )
    allen_to_order = {
        int(rows[index]["allen_section_number"]): order
        for order, index in enumerate(observed)
    }
    for allen_section in (1082, 1089):
        if allen_section not in allen_to_order:
            raise RuntimeError(f"Required marked Allen section {allen_section} is absent")
        marked_orders.add(allen_to_order[allen_section])
    marked_serial = {
        str(int(rows[observed[order]]["allen_section_number"])): float(
            observed_serial[order]
        )
        for order in sorted(marked_orders)
    }

    representatives = []
    titles = []
    for position in representative_positions:
        physical_index = int(observed[position])
        allen = int(rows[physical_index]["allen_section_number"])
        z_um = float(serial[physical_index])
        titles.append(f"Allen {allen}, z={z_um / 1000.0:.3f} mm")
        representatives.append(
            {
                "observed_order": int(position),
                "physical_index": physical_index,
                "allen_section": allen,
                "serial_um": z_um,
            }
        )
    projections = {
        "original_serial_row": (
            original_serial_row_num, original_serial_row_den, row_um
        ),
        "transformed_serial_row": (
            transformed_serial_row_num, transformed_serial_row_den, row_um
        ),
        "original_serial_column": (
            original_serial_column_num, original_serial_column_den, column_um
        ),
        "transformed_serial_column": (
            transformed_serial_column_num,
            transformed_serial_column_den,
            column_um,
        ),
    }
    traces = {
        "row_translation_um": row_translation,
        "column_translation_um": column_translation,
        "rotation_deg": rotation,
    }

    run_tmp = Path(os.environ["TMPDIR"]).resolve()
    approved_tmp = Path(
        "/cis/home/dpadova/.cache/ad-resilience/tmp"
    ).resolve()
    if run_tmp != approved_tmp and approved_tmp not in run_tmp.parents:
        raise RuntimeError("Overview TMPDIR is outside the approved home cache")
    private = Path(
        __import__("tempfile").mkdtemp(prefix="atlas-overview-", dir=run_tmp)
    ).resolve()
    if approved_tmp not in private.parents:
        raise RuntimeError("Resolved overview temporary directory escaped cache")
    tmp_png = private / png.name
    tmp_json = private / report_path.name
    try:
        _, projection_shapes = _render_saved_transform_stack_overview(
            representative_original=representative_original,
            representative_transformed=representative_transformed,
            representative_titles=titles,
            projections=projections,
            serial_um=serial,
            trace_serial_um=observed_serial,
            traces=traces,
            marked_serial_um=marked_serial,
            output=tmp_png,
        )
        after_hashes = {
            name: checksum(path) for name, path in sources.items()
        }
        registration_after = registration_checkpoint.exists()
        source_unchanged = all(
            before_hashes[name] == after_hashes[name]
            for name in ("observed_A2d", "bookkeeping_frame", "observed_indices")
        )
        checkpoint_unchanged = (
            before_hashes["atlas_free_checkpoint"]
            == after_hashes["atlas_free_checkpoint"]
        )
        if not source_unchanged or not checkpoint_unchanged:
            raise RuntimeError("Authoritative atlas-free inputs changed during QC")
        serial_edges = _coordinate_cell_edges(serial)
        row_edges = _coordinate_cell_edges(row_um)
        column_edges = _coordinate_cell_edges(column_um)
        report = {
            "created_at": now(),
            "outputs": {"png": str(png), "json": str(report_path)},
            "source_paths": {name: str(path) for name, path in sources.items()},
            "source_checksums_before": before_hashes,
            "source_checksums_after": after_hashes,
            "representative_sections": representatives,
            "projection_numerator_shapes": {
                key: list(expected_shapes[key])
                for key in expected_shapes if key.endswith("_num")
            },
            "projection_denominator_shapes": {
                key: list(expected_shapes[key])
                for key in expected_shapes if key.endswith("_den")
            },
            "normalized_projection_shapes": projection_shapes,
            "projection_orientation": {
                "storage": "(serial, spatial-coordinate), RGB channel first for numerators",
                "display": "projection.T",
                "horizontal_axis": "anterior-to-posterior serial position",
                "vertical_axes": ["row position", "column position"],
                "zero_support": "masked neutral background",
            },
            "physical_extents_um": {
                "serial_cell_edges": [
                    float(serial_edges[0]), float(serial_edges[-1])
                ],
                "row_cell_edges": [
                    float(row_edges[0]), float(row_edges[-1])
                ],
                "column_cell_edges": [
                    float(column_edges[0]), float(column_edges[-1])
                ],
            },
            "marked_allen_sections": sorted(
                int(value) for value in marked_serial
            ),
            "largest_adjacent_row_translation_jumps": jump_records,
            "residual_ranges": {
                "row_translation_um": [
                    float(row_translation.min()), float(row_translation.max())
                ],
                "column_translation_um": [
                    float(column_translation.min()),
                    float(column_translation.max()),
                ],
                "rotation_deg": [
                    float(rotation.min()), float(rotation.max())
                ],
            },
            "baseline_recomposition_max_abs_error": recomposition_error,
            "optimization_invoked_by_helper": False,
            "registration_invoked_by_helper": False,
            "registration_checkpoint_present_before": registration_before,
            "registration_checkpoint_present_after": registration_after,
            "source_transforms_unchanged": source_unchanged,
            "atlas_free_checkpoint_unchanged": checkpoint_unchanged,
        }
        with tmp_json.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        _publish_saved_overview(tmp_png, png)
        try:
            _publish_saved_overview(tmp_json, report_path)
        except BaseException:
            if png.exists():
                png.unlink()
            raise
    finally:
        shutil.rmtree(private)
        plt.close("all")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["atlas-free", "registration", "postprocess"])
    parser.add_argument("--profile")
    parser.add_argument("--native-qc", action="store_true")
    parser.add_argument("--input-dataset", type=Path)
    parser.add_argument("--initial-affine", type=Path)
    parser.add_argument("--output-root", type=Path)
    inspection = parser.add_mutually_exclusive_group()
    inspection.add_argument("--print-effective-config", action="store_true")
    inspection.add_argument("--print-output-root", action="store_true")
    args = parser.parse_args()

    if (args.input_dataset is None) != (args.initial_affine is None):
        parser.error("--input-dataset and --initial-affine must be supplied together")
    if args.input_dataset is not None:
        configure_linear_inputs(args.input_dataset, args.initial_affine)
    if args.output_root is not None:
        configure_output_root(args.output_root.resolve())

    if args.print_effective_config or args.print_output_root:
        if args.stage != "registration" or args.profile is None:
            parser.error("inspection options require registration --profile NAME")
        parameters = resolve_registration_execution(
            args.profile, native_qc=args.native_qc
        )
        output_root = registration_output_root(
            args.profile, native_qc=args.native_qc
        )
        if args.print_output_root:
            print(output_root)
        else:
            print(canonical_registration_config(parameters))
            print(f"output_root: {output_root}")
            print(f"sha256: {registration_config_sha256(parameters)}")
        return

    if args.native_qc and (args.stage != "registration" or args.profile is None):
        parser.error("--native-qc requires registration --profile NAME")
    if args.stage == "registration" and args.profile is not None:
        resolve_registration_execution(args.profile, native_qc=args.native_qc)
        if args.output_root is None:
            configure_output_root(
                registration_output_root(args.profile, native_qc=args.native_qc)
            )
        completed = CHECKPOINTS / "registration.json"
        if (
            completed.is_file()
            and json.loads(completed.read_text()).get("status") == "complete"
        ):
            parser.error(f"registration profile is already complete: {args.profile}")
    elif args.stage == "registration":
        completed = BASELINE_OUTPUT / "checkpoints/registration.json"
        if completed.is_file() and json.loads(completed.read_text()).get("status") == "complete":
            parser.error("baseline registration is already complete; refusing overwrite")
    elif args.profile is not None:
        parser.error("--profile is valid only for registration")
    if args.stage == "atlas-free" and args.output_root is None and args.input_dataset is not None:
        parser.error("Observed-left atlas-free runs require an explicit --output-root")

    torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", "8")))
    torch.set_num_interop_threads(1)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    try:
        if args.stage == "registration":
            registration(args.profile, native_qc=args.native_qc)
        else:
            {"atlas-free": atlas_free, "postprocess": postprocess}[args.stage]()
    except BaseException as exc:
        atomic_json(
            CHECKPOINTS / f"{args.stage}.json",
            {
                "stage": args.stage,
                "status": "failed",
                "failure": repr(exc),
                "failure_time": now(),
                "peak_process_group_rss_kib": peak_rss_kib(),
                "production_refinement_launched": False,
            },
        )
        raise


if __name__ == "__main__":
    main()
