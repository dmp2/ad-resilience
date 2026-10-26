"""Visual EM-LDDMM tuning used by the 1 mm Allen registration notebook.

No numerical registration scores: inspect the saved EM-LDDMM draw figures and
bidirectional overlays before choosing a candidate. Runs resume from saved files.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

# Avoid fragmentation on the 8 GB GPU during 1 mm backpropagation.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import matplotlib.pyplot as plt
import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
EMLDDMM_HOME = PROJECT.parent / "emlddmm"
sys.path.insert(0, str(EMLDDMM_HOME))
import emlddmm

ROOT = PROJECT / "results/allen_7T_SS_to_AHRAv2_registration"
TUNING = ROOT / "tuning_1mm"
TEMPLATE = (
    PROJECT
    / "data/raw/allen/ahrav2/2025/ahrav2_icbm2009b_symmetric_template_500um_1mm.nii.gz"
)
TARGET = (
    PROJECT
    / "data/derivatives/allen/specimen_708424/mri_7t_whole/T1_rot_space-MRI_7T_WHOLE_desc-header-corrected_1mm.nii.gz"
)
LABELS = PROJECT / "data/raw/allen/ahrav2/2025/ahrav2_brodmann_annotations_500um.nii.gz"


def numpy(value):
    return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


def normalize(image):
    image = np.asarray(image, dtype=np.float32)
    lo, hi = np.percentile(image[np.isfinite(image)], [0.5, 99.5])
    if hi <= lo:
        raise ValueError("Image has no intensity range")
    return ((np.clip(image, lo, hi) - lo) / (hi - lo)).astype(np.float32)


MASK_NAMES = ("labels", "convex_hull", "synthstrip", "synthstrip_nocsf")


def load_inputs(atlas_mask="labels"):
    """Load once; preserve read_data's centered coordinates and channel order."""
    xI, I, _, _ = emlddmm.read_data(str(TEMPLATE))
    xJ, J, _, _ = emlddmm.read_data(str(TARGET))
    I, J = normalize(I), normalize(J)
    template_nii = nib.load(TEMPLATE)
    labels_nii = nib.load(LABELS)
    # Native NIfTI geometry establishes the mask-to-template mapping; centered
    # registration coordinates are used only after this resampling.
    S = np.asarray(
        resample_from_to(labels_nii, template_nii, order=0).dataobj, dtype=np.int32
    )[None]
    label_mask = (S != 0).astype(np.float32)
    base = dict(xI=xI, I=I, xJ=xJ, J=J, S=S, label_mask=label_mask)
    return use_atlas_mask(base, atlas_mask)


def use_atlas_mask(data, atlas_mask):
    """Switch atlas masks without reloading or renormalizing either image."""
    if atlas_mask not in MASK_NAMES:
        raise ValueError(f"Unknown atlas mask: {atlas_mask}")
    if atlas_mask == "labels":
        brain_mask = data["label_mask"]
    else:
        mask_path = TUNING / "masks" / f"{atlas_mask}_mask.nii.gz"
        mask_nii = nib.load(mask_path)
        template_nii = nib.load(TEMPLATE)
        brain_mask = (
            np.asarray(resample_from_to(mask_nii, template_nii, order=0).dataobj) > 0
        )[None].astype(np.float32)
    result = dict(data)
    result.update(
        I_brain=data["I"] * brain_mask,
        brain_mask=brain_mask,
        atlas_mask=atlas_mask,
        atlas_mask_sha256=hashlib.sha256(
            brain_mask.astype(np.uint8).tobytes()
        ).hexdigest(),
    )
    return result


def prepare_convex_hull(data):
    """Build one true 3D hull, including voxel extent, on the 1 mm atlas grid."""
    from skimage.morphology import convex_hull_image

    directory = TUNING / "masks"
    directory.mkdir(parents=True, exist_ok=True)
    mask_path = directory / "convex_hull_mask.nii.gz"
    if not mask_path.exists():
        mask = convex_hull_image(data["label_mask"][0] > 0)
        reference = nib.load(TEMPLATE)
        nib.save(nib.Nifti1Image(mask.astype(np.uint8), reference.affine), mask_path)
        image = np.asarray(reference.dataobj, dtype=np.float32) * mask
        nib.save(
            nib.Nifti1Image(image, reference.affine),
            directory / "convex_hull_atlas.nii.gz",
        )
    return mask_path


def review_atlas_mask(data):
    directory = TUNING / "masks"
    directory.mkdir(parents=True, exist_ok=True)
    name = data["atlas_mask"]
    plot_image(
        data["I_brain"],
        data["xI"],
        f"Atlas input: {name}",
        directory / f"{name}_input.png",
    )
    gray = np.repeat(data["I"], 3, axis=0)
    alpha = data["brain_mask"] * 0.3
    overlay = gray * (1 - alpha)
    overlay[1] += alpha[0]
    plot_image(
        overlay,
        data["xI"],
        f"{name}: mask green over original atlas",
        directory / f"{name}_overlay.png",
    )


def initial_affine(data):
    A = np.eye(4, dtype=np.float32)
    A[:3, :3] *= 0.9
    ci = np.array([(x[0] + x[-1]) / 2 for x in data["xI"]])
    cj = np.array([(x[0] + x[-1]) / 2 for x in data["xJ"]])
    A[:3, 3] = cj - A[:3, :3] @ ci
    return A


def common_config(data):
    return dict(
        device="cuda:0",
        dtype="float32",
        nt=10,
        dv=2.0,
        sigmaM=0.5,
        sigmaB=1.0,
        sigmaA=2.5,
        muB=0.0,
        muA=float(np.quantile(data["J"], 0.999)),
        update_muB=False,
        update_muA=False,
        order=1,
        full_outputs=True,
        n_draw=50,
        W0=(data["J"][0] > 0).astype(np.float32),
    )


def config_for(data, kind, *, eA=250.0, a=10.0, sigmaR=50.0, local=False):
    config = common_config(data)
    config.update(
        downI=[[4, 4, 4], [2, 2, 2]],
        downJ=[[4, 4, 4], [2, 2, 2]],
        n_iter=[100, 50],
        v_start=[0, 0],
        Amode=0,
        eA=eA,
        ev=30.0,
        a=a,
        sigmaR=sigmaR,
        local_contrast=[[16, 16, 16], [32, 32, 32]] if local else None,
    )
    if kind == "similitude":
        config.update(
            n_iter=[200, 100],
            v_start=[1000, 1000],
            Amode=2,
            A=initial_affine(data),
            local_contrast=None,
        )
    return config


def fresh_nocsf_config(data, stage):
    """Current no-CSF workflow: fresh similitude, then unrestricted affine/diffeos."""
    if data.get("atlas_mask") != "synthstrip_nocsf":
        raise ValueError("The fresh workflow uses the SynthStrip no-CSF atlas mask")
    if stage == "similitude":
        config = config_for(data, "similitude", eA=250.0, a=5.0, sigmaR=200.0)
    elif stage == "deformation":
        config = config_for(
            data, "deformation", eA=125.0, a=5.0, sigmaR=200.0, local=True
        )
    else:
        raise ValueError("Stage must be similitude or deformation")
    config.update(W0=None, ev=60.0)
    return config


def json_config(config):
    # W0 is derived from the target; large arrays are stored separately.
    return {
        k: (numpy(v).tolist() if torch.is_tensor(v) or isinstance(v, np.ndarray) else v)
        for k, v in config.items()
        if k not in ("W0", "v")
    }


def load_result(name):
    with np.load(TUNING / name / "transform.npz") as z:
        return dict(A=z["A"], v=z["v"], xv=[z[f"xv{i}"] for i in range(3)])


def load_config(name):
    """Read the parameter record; resume_config prepares arrays and weighting."""
    return json.loads((TUNING / name / "config.json").read_text())["parameters"]


def plot_image(image, axes, title, path):
    fig, _ = emlddmm.draw(
        numpy(image),
        [numpy(x) for x in axes],
        n_slices=7,
        vmin=0.0,
        vmax=1.0,
        cmap="gray",
    )
    fig.set_size_inches(16, 7)
    fig.suptitle(title)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def visual_review(data, result, directory):
    """Same seven slices in each direction for every candidate; no scores."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    xI, xJ, I, J = (data[k] for k in ("xI", "xJ", "I_brain", "J"))
    backward = emlddmm.compose_sequence(
        [
            emlddmm.Transform(result["A"], direction="b"),
            emlddmm.Transform(result["v"], domain=result["xv"], direction="b"),
        ],
        xJ,
    )
    atlas_in_target = numpy(emlddmm.apply_transform_float(xI, I, backward))
    labels_in_target = numpy(
        emlddmm.apply_transform_int(
            xI, data["S"], backward, double=False, padding_mode="zeros"
        )
    )
    del backward
    forward = emlddmm.compose_sequence(
        [
            emlddmm.Transform(result["v"], domain=result["xv"], direction="f"),
            emlddmm.Transform(result["A"], direction="f"),
        ],
        xI,
    )
    target_in_atlas = numpy(emlddmm.apply_transform_float(xJ, J, forward))
    del forward
    identity = emlddmm.compose_sequence([emlddmm.Transform(np.eye(4))], xJ)
    unregistered_atlas = numpy(emlddmm.apply_transform_float(xI, I, identity))
    del identity
    identity = emlddmm.compose_sequence([emlddmm.Transform(np.eye(4))], xI)
    unregistered_target = numpy(emlddmm.apply_transform_float(xJ, J, identity))
    del identity
    name = directory.name
    # emlddmm.draw maps two channels to magenta/green consistently.
    for image, axes, suffix, title in [
        (
            np.concatenate((unregistered_atlas, J)),
            xJ,
            "target_before",
            "Atlas magenta / target green: before",
        ),
        (
            np.concatenate((atlas_in_target, J)),
            xJ,
            "target_after",
            "Atlas magenta / target green: registered",
        ),
        (
            np.concatenate((I, unregistered_target)),
            xI,
            "atlas_before",
            "Atlas magenta / target green: before",
        ),
        (
            np.concatenate((I, target_in_atlas)),
            xI,
            "atlas_after",
            "Atlas magenta / target green: registered",
        ),
    ]:
        plot_image(image, axes, f"{name}: {title}", directory / f"{suffix}.png")
    draw_label_overlay(data, labels_in_target, directory)


def draw_label_overlay(data, labels_in_target, directory):
    """Use the same atlas-wide label palette even when labels disappear in a run."""
    directory = Path(directory)
    labels = numpy(labels_in_target)[0]
    unique = np.unique(data["S"])
    # Preserve the established colors, while supporting tiny labels present in
    # the original 0.5 mm atlas but absent from its 1 mm nearest-neighbor sample.
    maximum = int(max(np.max(unique), np.max(labels)))
    colors = np.random.default_rng(0).random((maximum + 1, 3)).astype(np.float32)
    colors[unique] = (
        np.random.default_rng(0).random((len(unique), 3)).astype(np.float32)
    )
    colors[0] = 0
    rgb = colors[labels].transpose(3, 0, 1, 2)
    alpha = 0.35 * (labels != 0)[None]
    overlay = np.repeat(data["J"], 3, axis=0) * (1 - alpha) + rgb * alpha
    plot_image(
        overlay,
        data["xJ"],
        f"{directory.name}: atlas labels in target space",
        directory / "labels_in_target.png",
    )


def refresh_label_overlay(data, result, directory):
    backward = emlddmm.compose_sequence(
        [
            emlddmm.Transform(result["A"], direction="b"),
            emlddmm.Transform(result["v"], domain=result["xv"], direction="b"),
        ],
        data["xJ"],
    )
    labels = emlddmm.apply_transform_int(
        data["xI"], data["S"], backward, double=False, padding_mode="zeros"
    )
    draw_label_overlay(data, labels, directory)


def review_cerebellum_masks(base, candidates, *, output_path=None):
    """Same posterior/inferior target-space slices for each mask trial."""
    positions = (-20.0, 0.0, 20.0)
    fig, axes = plt.subplots(
        len(candidates),
        len(positions),
        figsize=(12, 4 * len(candidates)),
        squeeze=False,
    )
    xJ = base["xJ"]
    keep_y = (xJ[1] >= -110.0) & (xJ[1] <= -20.0)
    keep_z = (xJ[2] >= -100.0) & (xJ[2] <= 40.0)
    y, z = xJ[1][keep_y], xJ[2][keep_z]
    for row, (label, specification) in enumerate(candidates.items()):
        if isinstance(specification, str):
            mask_name, candidate = label, specification
        else:
            mask_name, candidate = specification
        data = use_atlas_mask(base, mask_name)
        result = load_result(candidate)
        backward = emlddmm.compose_sequence(
            [
                emlddmm.Transform(result["A"], direction="b"),
                emlddmm.Transform(result["v"], domain=result["xv"], direction="b"),
            ],
            xJ,
        )
        atlas = numpy(
            emlddmm.apply_transform_float(data["xI"], data["I_brain"], backward)
        )[0]
        del backward, result
        for column, position in enumerate(positions):
            index = np.argmin(np.abs(xJ[0] - position))
            a = atlas[index][np.ix_(keep_y, keep_z)]
            j = base["J"][0, index][np.ix_(keep_y, keep_z)]
            rgb = np.clip(np.stack((a, j, a), axis=-1), 0.0, 1.0)
            ax = axes[row, column]
            ax.imshow(rgb, extent=[z[0], z[-1], y[-1], y[0]], interpolation="nearest")
            ax.set_title(f"{label}\nCentered LR = {xJ[0][index]:g} mm", fontsize=10)
            ax.set_xlabel("Centered superior/inferior coordinate (mm)")
            ax.set_ylabel("Centered anterior/posterior coordinate (mm)")
        del atlas
    fig.suptitle(
        "Posterior/inferior target-space comparison: atlas magenta, target green"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = (
        Path(output_path)
        if output_path is not None
        else TUNING / "masks/cerebellum_comparison.png"
    )
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def run_candidate(name, data, config, *, parent=None, affine_only=False):
    directory = TUNING / name
    directory.mkdir(parents=True, exist_ok=True)
    config = dict(config)
    if parent:
        previous = load_result(parent)
        config["A"] = previous["A"]
        if not affine_only:
            config["v"] = previous["v"]
    metadata = dict(
        atlas_mask=data.get("atlas_mask", "labels"),
        atlas_mask_sha256=data.get("atlas_mask_sha256"),
        parameters=json_config(config),
        target_weighting="all" if config.get("W0") is None else "positive",
        parent=parent,
        initialization="affine only" if affine_only else "affine and velocity",
        inputs=dict(template=str(TEMPLATE), target=str(TARGET), labels=str(LABELS)),
        emlddmm_commit=subprocess.check_output(
            ["git", "-C", str(EMLDDMM_HOME), "rev-parse", "HEAD"], text=True
        ).strip(),
    )
    if (directory / "transform.npz").exists():
        saved = json.loads((directory / "config.json").read_text())
        saved.setdefault("target_weighting", "positive")
        saved.setdefault("atlas_mask", "labels")
        if saved["atlas_mask"] == "labels":
            saved.setdefault("atlas_mask_sha256", metadata["atlas_mask_sha256"])
        for key in (
            "atlas_mask",
            "atlas_mask_sha256",
            "target_weighting",
            "parameters",
            "parent",
            "initialization",
            "inputs",
            "emlddmm_commit",
        ):
            if saved[key] != metadata[key]:
                raise ValueError(
                    f"{name}: saved {key} differs; use a new candidate name"
                )
        print("Using saved candidate:", name, flush=True)
        result = load_result(name)
        if not all(
            (directory / (stem + ".png")).exists()
            for stem in (
                "target_before",
                "target_after",
                "atlas_before",
                "atlas_after",
                "labels_in_target",
            )
        ):
            visual_review(data, result, directory)
        return result
    (directory / "config.json").write_text(json.dumps(metadata, indent=2))
    print("Running:", name, flush=True)
    start = time.monotonic()
    with (directory / "optimizer.log").open("w") as log, contextlib.redirect_stdout(
        log
    ), contextlib.redirect_stderr(log):
        outputs = emlddmm.emlddmm_multiscale(
            xI=[data["xI"]], I=data["I_brain"], xJ=[data["xJ"]], J=data["J"], **config
        )
        for stage, output in enumerate(outputs):
            for key, value in output.items():
                if key.startswith("fig"):
                    value.savefig(directory / f"stage{stage}_{key}.png", dpi=100)
        final = outputs[-1]
        result = dict(
            A=numpy(final["A"]), v=numpy(final["v"]), xv=[numpy(x) for x in final["xv"]]
        )
        np.savez_compressed(
            directory / "transform.npz",
            A=result["A"],
            v=result["v"],
            **{f"xv{i}": x for i, x in enumerate(result["xv"])},
        )
        del outputs, final
        plt.close("all")
        gc.collect()
        torch.cuda.empty_cache()
        visual_review(data, result, directory)
    metadata["elapsed_seconds"] = time.monotonic() - start
    (directory / "config.json").write_text(json.dumps(metadata, indent=2))
    print("Finished:", name, "seconds:", round(metadata["elapsed_seconds"]), flush=True)
    return result


def resume_config(data, parent, *, scales, iterations, half_steps=False):
    config = load_config(parent)
    metadata = json.loads((TUNING / parent / "config.json").read_text())
    weights = (
        None
        if metadata.get("target_weighting", "positive") == "all"
        else (data["J"][0] > 0).astype(np.float32)
    )
    config.update(
        W0=weights,
        downI=[[s] * 3 for s in scales],
        downJ=[[s] * 3 for s in scales],
        n_iter=iterations,
        v_start=[0] * len(scales),
        Amode=0,
    )
    if config["local_contrast"] is not None:
        config["local_contrast"] = [[int(64 / s)] * 3 for s in scales]
    if half_steps:
        config["eA"] *= 0.5
        config["ev"] *= 0.5
    config["A"] = np.asarray(config["A"], dtype=np.float32)
    return config


def export_selected(data, name, *, select=True, mask_zero_background=False):
    """Export original labels/images, with an optional exact-zero-MRI companion."""
    result = load_result(name)
    directory = TUNING / name
    metadata = json.loads((directory / "config.json").read_text())
    if metadata.get("atlas_mask", "labels") != data.get("atlas_mask", "labels"):
        raise ValueError("Export must use the candidate's atlas mask")
    if metadata.get("atlas_mask_sha256") is not None and metadata[
        "atlas_mask_sha256"
    ] != data.get("atlas_mask_sha256"):
        raise ValueError("Export mask differs from the saved candidate mask")
    xS, S, _, _ = emlddmm.read_data(str(LABELS))
    backward = emlddmm.compose_sequence(
        [
            emlddmm.Transform(result["A"], direction="b"),
            emlddmm.Transform(result["v"], domain=result["xv"], direction="b"),
        ],
        data["xJ"],
    )
    transformed_labels = numpy(
        emlddmm.apply_transform_int(xS, S, backward, double=False, padding_mode="zeros")
    ).astype(np.int32)
    transformed_atlas = numpy(
        emlddmm.apply_transform_float(data["xI"], data["I_brain"], backward)
    )
    del backward, S
    forward = emlddmm.compose_sequence(
        [
            emlddmm.Transform(result["v"], domain=result["xv"], direction="f"),
            emlddmm.Transform(result["A"], direction="f"),
        ],
        data["xI"],
    )
    transformed_target = numpy(
        emlddmm.apply_transform_float(data["xJ"], data["J"], forward)
    )
    for filename, image, reference in [
        ("atlas_labels_to_target.nii.gz", transformed_labels, TARGET),
        ("atlas_image_to_target.nii.gz", transformed_atlas, TARGET),
        ("target_image_to_atlas.nii.gz", transformed_target, TEMPLATE),
    ]:
        nii = nib.load(reference)
        nib.save(nib.Nifti1Image(image[0], nii.affine), directory / filename)
    label_outputs = {"raw": "atlas_labels_to_target.nii.gz"}
    if mask_zero_background:
        # This is output support cleanup only. Keep W0=None during registration.
        raw_target = np.asarray(nib.load(TARGET).dataobj)
        clean_labels = transformed_labels[0].copy()
        clean_labels[raw_target == 0] = 0
        clean_name = "atlas_labels_to_target_no_zero_background.nii.gz"
        nib.save(
            nib.Nifti1Image(clean_labels, nib.load(TARGET).affine),
            directory / clean_name,
        )
        label_outputs["zero_background_removed"] = clean_name
    selected_config = load_config(name)
    weighting = json.loads((directory / "config.json").read_text()).get(
        "target_weighting", "positive"
    )
    if weighting == "all":
        selected_config["W0"] = None
    selection = dict(
        candidate=name,
        label_outputs=label_outputs,
        label_resampling="nearest neighbor; zero padding outside atlas field of view",
        output_background_cleanup=(
            "labels set to zero only where unnormalized target MRI equals zero"
            if mask_zero_background
            else None
        ),
        atlas_mask=data.get("atlas_mask", "labels"),
        atlas_mask_sha256=data.get("atlas_mask_sha256"),
        parameters=selected_config,
        target_weighting=json.loads((directory / "config.json").read_text()).get(
            "target_weighting", "positive"
        ),
        review="Visual review of EM-LDDMM draw output and bidirectional overlays; no numerical scores.",
    )
    native_record = directory / "native_label_export.json"
    if native_record.exists():
        selection["native_label_export"] = json.loads(native_record.read_text())
    (directory / "export.json").write_text(json.dumps(selection, indent=2))
    if select:
        (TUNING / "selected.json").write_text(json.dumps(selection, indent=2))


def review_label_boundaries(data, name):
    """Render edge disagreements without computing registration scores."""
    directory = TUNING / name
    labels = np.asarray(nib.load(directory / "atlas_labels_to_target.nii.gz").dataobj)
    raw_target = np.asarray(nib.load(TARGET).dataobj)
    result = load_result(name)
    backward = emlddmm.compose_sequence(
        [
            emlddmm.Transform(result["A"], direction="b"),
            emlddmm.Transform(result["v"], domain=result["xv"], direction="b"),
        ],
        data["xJ"],
    )
    support = (
        numpy(
            emlddmm.apply_transform_int(
                data["xI"],
                data["brain_mask"].astype(np.uint8),
                backward,
                double=False,
                padding_mode="zeros",
            )
        )[0]
        > 0
    )
    for flag, title, filename in [
        (
            (labels != 0) & (raw_target == 0),
            "Labels over exactly zero target MRI (orange)",
            "labels_over_zero_mri.png",
        ),
        (
            (labels != 0) & ~support,
            "Labels outside transformed no-CSF atlas mask (orange)",
            "labels_outside_atlas_support.png",
        ),
    ]:
        overlay = np.repeat(data["J"], 3, axis=0) * 0.75
        overlay[:, flag] = np.array([1.0, 0.45, 0.0], dtype=np.float32)[:, None]
        plot_image(overlay, data["xJ"], f"{name}: {title}", directory / filename)
    cleaned_path = directory / "atlas_labels_to_target_no_zero_background.nii.gz"
    if cleaned_path.exists():
        clean = np.asarray(nib.load(cleaned_path).dataobj, dtype=np.int32)[None]
        # Use the same atlas-wide palette; retain the raw overlay alongside it.
        clean_directory = directory / "zero_background_removed"
        clean_directory.mkdir(exist_ok=True)
        draw_label_overlay(data, clean, clean_directory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase",
        choices=[
            "similitude",
            "deformation",
            "local",
            "refine",
            "long",
            "continue",
            "export",
        ],
    )
    parser.add_argument("--parent")
    parser.add_argument("--name")
    parser.add_argument("--atlas-mask", choices=MASK_NAMES)
    args = parser.parse_args()
    mask_name = args.atlas_mask
    if mask_name is None and args.parent:
        parent_metadata = json.loads((TUNING / args.parent / "config.json").read_text())
        mask_name = parent_metadata.get("atlas_mask", "labels")
    data = load_inputs(mask_name or "labels")
    if args.phase == "similitude":
        for step in (250.0, 500.0):
            run_candidate(
                f"similitude_eA{step:g}", data, config_for(data, "similitude", eA=step)
            )
    elif args.phase == "deformation":
        if not args.parent:
            parser.error("--parent must name the visually selected similitude")
        for a in (5.0, 10.0):
            for sr in (50.0, 100.0):
                run_candidate(
                    f"deform_a{a:g}_sigmaR{sr:g}",
                    data,
                    config_for(data, "deformation", a=a, sigmaR=sr),
                    parent=args.parent,
                    affine_only=True,
                )
    elif args.phase == "local":
        if not args.parent:
            parser.error("--parent must name the visually preferred deformation")
        previous = load_config(args.parent)
        # Fair contrast comparison: restart from the same similitude, not a fitted deformation.
        metadata = json.loads((TUNING / args.parent / "config.json").read_text())
        run_candidate(
            args.name or args.parent + "_local",
            data,
            config_for(
                data,
                "deformation",
                a=previous["a"],
                sigmaR=previous["sigmaR"],
                local=True,
            ),
            parent=metadata["parent"],
            affine_only=True,
        )
    elif args.phase in ("refine", "long", "continue"):
        if not args.parent:
            parser.error("--parent is required")
        scales, iterations = {
            "refine": ([1], [100]),
            "long": ([4, 2, 1], [1000, 500, 300]),
            "continue": ([1], [500]),
        }[args.phase]
        config = resume_config(
            data,
            args.parent,
            scales=scales,
            iterations=iterations,
            half_steps=args.phase == "refine",
        )
        run_candidate(
            args.name or args.parent + "_" + args.phase,
            data,
            config,
            parent=args.parent,
        )
    else:
        if not args.parent:
            parser.error("--parent is required")
        export_selected(data, args.parent)


if __name__ == "__main__":
    main()
