"""Sample the fitted atlas labels directly on the original 0.2 mm target grid."""

import json
from pathlib import Path
import tempfile

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch

import tune_allen_1mm_registration as tuning

NATIVE_TARGET = tuning.TARGET.with_name(
    "T1_rot_space-MRI_7T_WHOLE_desc-header-corrected.nii"
)


def export_native_labels(name, *, device="cuda:0", slab_size=4):
    directory = tuning.TUNING / name
    reference = nib.load(NATIVE_TARGET)
    fitted_reference = nib.load(tuning.TARGET)
    raw_path = directory / "atlas_labels_to_target_native_0p2mm.nii.gz"
    clean_path = (
        directory / "atlas_labels_to_target_native_0p2mm_no_zero_background.nii.gz"
    )
    record_path = directory / "native_label_export.json"
    result = tuning.load_result(name)
    atlas = nib.load(tuning.LABELS)
    source = np.asarray(atlas.dataobj)
    if source.min() < 0 or source.max() > np.iinfo(np.uint16).max:
        raise ValueError("Label IDs cannot be stored losslessly as uint16")
    # Match read_data's centered source axes exactly, without loading float64 MRI.
    axes = [
        (np.arange(n) - (n - 1) / 2) * d
        for n, d in zip(source.shape, atlas.header.get_zooms()[:3])
    ]
    source_tensor = torch.as_tensor(source[None].astype(np.float32), device=device)
    # Native voxel -> world -> fitted target voxel -> centered fitted coordinates.
    # Recentring the native FOV independently would introduce a subvoxel shift.
    voxel_to_fit = np.linalg.inv(fitted_reference.affine) @ reference.affine
    spacing = np.asarray(fitted_reference.header.get_zooms()[:3])
    centered = np.eye(4)
    centered[:3, :3] = np.diag(spacing)
    centered[:3, 3] = -(np.asarray(fitted_reference.shape) - 1) * spacing / 2
    native_to_centered = centered @ voxel_to_fit
    geometry = torch.as_tensor(native_to_centered, dtype=torch.float32, device=device)
    transforms = [
        tuning.emlddmm.Transform(result["A"], direction="b", device=device),
        tuning.emlddmm.Transform(
            result["v"], domain=result["xv"], direction="b", device=device
        ),
    ]
    target_proxy = reference.dataobj
    with tempfile.TemporaryDirectory(prefix="allen_native_labels_", dir="/tmp") as tmp:
        raw = np.memmap(
            Path(tmp) / "raw.dat",
            dtype=np.uint16,
            mode="w+",
            shape=reference.shape,
            order="F",
        )
        clean = np.memmap(
            Path(tmp) / "clean.dat",
            dtype=np.uint16,
            mode="w+",
            shape=reference.shape,
            order="F",
        )
        with torch.no_grad():
            for start in range(0, reference.shape[2], slab_size):
                stop = min(start + slab_size, reference.shape[2])
                coordinates = torch.stack(
                    torch.meshgrid(
                        torch.arange(
                            reference.shape[0], device=device, dtype=torch.float32
                        ),
                        torch.arange(
                            reference.shape[1], device=device, dtype=torch.float32
                        ),
                        torch.arange(start, stop, device=device, dtype=torch.float32),
                        indexing="ij",
                    )
                )
                coordinates = (
                    torch.einsum("ij,jabc->iabc", geometry[:3, :3], coordinates)
                    + geometry[:3, 3, None, None, None]
                )
                backward = tuning.emlddmm.compose_sequence(transforms, coordinates)
                labels = (
                    tuning.emlddmm.interp(
                        axes,
                        source_tensor,
                        backward,
                        mode="nearest",
                        padding_mode="zeros",
                    )[0]
                    .cpu()
                    .numpy()
                    .astype(np.uint16)
                )
                raw[:, :, start:stop] = labels
                labels[np.asarray(target_proxy[:, :, start:stop]) == 0] = 0
                clean[:, :, start:stop] = labels
                if start % 100 == 0:
                    print(
                        f"Native label export: {stop}/{reference.shape[2]} slices",
                        flush=True,
                    )
        for values, path in ((raw, raw_path), (clean, clean_path)):
            header = reference.header.copy()
            header.set_data_dtype(np.uint16)
            image = nib.Nifti1Image(values, reference.affine, header)
            image.set_qform(reference.get_qform(), int(reference.header["qform_code"]))
            image.set_sform(reference.get_sform(), int(reference.header["sform_code"]))
            nib.save(image, path)
            print(f"Saved {path.name}", flush=True)
        # Read only representative planes; never expand the entire native MRI.
        fig, plots = plt.subplots(2, 3, figsize=(15, 10))
        palette = np.random.default_rng(0).random((int(source.max()) + 1, 3))
        registered_source = np.unique(
            np.asarray(
                nib.processing.resample_from_to(
                    atlas, nib.load(tuning.TEMPLATE), order=0
                ).dataobj,
                dtype=np.int32,
            )
        )
        palette[registered_source] = np.random.default_rng(0).random(
            (len(registered_source), 3)
        )
        for column, axis in enumerate(range(3)):
            index = reference.shape[axis] // 2
            selection = [slice(None)] * 3
            selection[axis] = index
            selection = tuple(selection)
            mri = np.asarray(target_proxy[selection])
            positive = mri[mri > 0]
            maximum = np.percentile(positive, 99.5)
            for row, values in enumerate((raw, clean)):
                plane = np.asarray(values[selection])
                ax = plots[row, column]
                ax.imshow(mri.T, origin="lower", cmap="gray", vmin=0, vmax=maximum)
                rgba = np.concatenate(
                    (palette[plane], (0.35 * (plane != 0))[..., None]), axis=-1
                )
                ax.imshow(
                    rgba.transpose(1, 0, 2), origin="lower", interpolation="nearest"
                )
                ax.set_title(
                    f'{("Raw", "Zero background removed")[row]}: {("sagittal", "coronal", "axial")[axis]} voxel {index}'
                )
                ax.axis("off")
        fig.suptitle(f"{name}: original 0.2 mm target grid")
        fig.tight_layout()
        fig.savefig(directory / "native_0p2mm_label_overlay.png", dpi=160)
        plt.close(fig)
    record = dict(
        candidate=name,
        target=str(NATIVE_TARGET),
        source_labels=str(tuning.LABELS),
        target_shape=list(reference.shape),
        target_spacing_mm=list(map(float, reference.header.get_zooms()[:3])),
        target_affine=reference.affine.tolist(),
        native_voxel_to_registration_centered_mm=native_to_centered.tolist(),
        raw=str(raw_path),
        zero_background_removed=str(clean_path),
        resampling="Original atlas labels sampled directly by nearest neighbor, with zero padding; saved affine and integrated velocity unchanged.",
        cleanup="Companion labels zeroed only where original native target MRI equals exactly zero.",
        limitation="0.2 mm output grid; atlas annotation resolution remains 0.5 mm and registration was fitted at 1 mm.",
    )
    record_path.write_text(json.dumps(record, indent=2))
    for path in (directory / "export.json", tuning.TUNING / "selected.json"):
        if path.exists():
            metadata = json.loads(path.read_text())
            if metadata.get("candidate") == name:
                metadata["native_label_export"] = record
                path.write_text(json.dumps(metadata, indent=2))
    return record


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate",
        default=json.loads((tuning.TUNING / "selected.json").read_text())["candidate"],
    )
    args = parser.parse_args()
    torch.set_num_threads(6)
    export_native_labels(args.candidate)
