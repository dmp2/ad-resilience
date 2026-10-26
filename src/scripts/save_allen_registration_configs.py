"""Save the selected optimization lineage and a graph-compatible replay config."""

import json
from pathlib import Path
import shutil

import nibabel as nib
import numpy as np

import tune_allen_1mm_registration as tuning


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def save_configs():
    selection = json.loads((tuning.TUNING / "selected.json").read_text())
    selected = selection["candidate"]
    chain = []
    name = selected
    while name:
        record = json.loads((tuning.TUNING / name / "config.json").read_text())
        chain.append((name, record))
        name = record.get("parent")
    chain.reverse()
    if any(
        record["inputs"]["template"] != str(tuning.TEMPLATE)
        or record["inputs"]["target"] != str(tuning.TARGET)
        or record["target_weighting"] != "all"
        for _, record in chain
    ):
        raise ValueError(
            "This graph replay requires the selected all-voxel 1 mm lineage"
        )
    destination = tuning.ROOT / "final_registration_configs"
    destination.mkdir(exist_ok=True)
    inputs = destination / "inputs"
    inputs.mkdir(exist_ok=True)
    data = tuning.load_inputs(selection["atlas_mask"])
    if data["atlas_mask_sha256"] != selection["atlas_mask_sha256"]:
        raise ValueError("Current atlas mask differs from the selected run")
    atlas_path = inputs / "atlas_normalized_synthstrip_nocsf_1mm.nii.gz"
    target_path = inputs / "target_normalized_1mm.nii.gz"
    for path, array, reference in (
        (atlas_path, data["I_brain"], tuning.TEMPLATE),
        (target_path, data["J"], tuning.TARGET),
    ):
        reference = nib.load(reference)
        nib.save(nib.Nifti1Image(array[0].astype(np.float32), reference.affine), path)
    # Each sequence member is a multiscale optimization, not a new spatial edge.
    stages = []
    schedule = []
    for index, (name, record) in enumerate(chain):
        parameters = dict(record["parameters"])
        parameters.pop("full_outputs", None)  # graph CLI supplies this explicitly
        parameters.pop("W0", None)  # graph CLI supplies an all-ones target mask
        parameters["A"] = [parameters["A"]]  # one matrix; not four multiscale entries
        config_path = destination / f"{index + 1:02d}_{name}.json"
        write_json(config_path, parameters)
        for level in range(len(parameters["downI"])):
            values = {}
            for key, value in parameters.items():
                values[key] = (
                    value[level]
                    if isinstance(value, list) and len(value) > 1
                    else (value[0] if isinstance(value, list) else value)
                )
            schedule.append(values)
        stages.append(
            dict(
                candidate=name,
                config=str(config_path),
                parent=record.get("parent"),
                initialization=record["initialization"],
                checkpoint=str(tuning.TUNING / name / "transform.npz"),
                recorded_run=record,
            )
        )
    # Carry A/v internally across all eight scales. Only the first A is explicit.
    merged = {}
    for key in schedule[0]:
        if key == "A":
            merged[key] = [schedule[0][key]]
        else:
            merged[key] = [step[key] for step in schedule]
    config_file = tuning.ROOT / "atlas_to_target_config.json"
    if (
        config_file.exists()
        and not (destination / "previous_atlas_to_target_config.json").exists()
    ):
        shutil.copy2(config_file, destination / "previous_atlas_to_target_config.json")
    write_json(config_file, merged)
    graph_file = tuning.ROOT / "transformation_graph_config.json"
    graph_config = dict(
        output=str(tuning.ROOT / "final_registration_graph_output"),
        space_image_path=[
            ["Atlas", "image", str(atlas_path)],
            ["Atlas", "labels", str(tuning.LABELS)],
            ["Target", "image", str(target_path)],
        ],
        registrations=[[["Atlas", "image"], ["Target", "image"]]],
        configs=[str(config_file)],
        transform_all=True,
    )
    write_json(graph_file, graph_config)
    manifest = dict(
        selected_candidate=selected,
        stages=stages,
        replay_config=str(config_file),
        transformation_graph_config=str(graph_file),
        runner=str(
            Path(__file__).with_name("run_allen_registration_graph.py").resolve()
        ),
        inputs=dict(
            template=str(tuning.TEMPLATE),
            target=str(tuning.TARGET),
            labels=str(tuning.LABELS),
            atlas_mask=str(tuning.TUNING / "masks/synthstrip_nocsf_mask.nii.gz"),
            atlas_mask_sha256=selection["atlas_mask_sha256"],
        ),
        preprocessing="Float32 clipping/normalization at original 0.5/99.5 percentiles, then atlas no-CSF mask; stored as prepared 1 mm images. No target mask or positive-only weighting.",
        coordinates="Original centered 1 mm physical coordinates; equal template/target downsampling.",
        initialization="Fresh centered 0.9 scale. Similitude velocity is zero because deformation is disabled; subsequent levels carry the estimated affine and velocity.",
        selected_transform=str(tuning.TUNING / selected / "transform.npz"),
        native_label_export=selection.get("native_label_export"),
        excluded_trials=["fresh_nocsf_0p5mm_cpu_pilot"],
        note="This JSON describes a rerun of the selected protocol. The existing transform.npz remains the saved result; a rerun is not a bitwise identity guarantee. Graph reconstructions use the 1 mm target grid; original 0.2 mm exports are recorded separately.",
    )
    write_json(destination / "sequence.json", manifest)
    (destination / "README.md").write_text(
        """# Selected Allen registration configuration set

`../atlas_to_target_config.json` replays the selected four optimization blocks as eight scales on the prepared 1 mm inputs. The individual numbered configs and `sequence.json` preserve the original block boundaries, actual initialization affines, checkpoint references, and original run records. Continuation blocks require the parent velocity from the referenced checkpoint when executed separately.

`../transformation_graph_config.json` uses the standard graph schema with one Atlas-to-Target registration and `transform_all=true`. Absolute paths allow launching it from any directory. The 0.5 mm CPU trial is excluded because it was not promoted.

Run with the repository adapter, which preserves prepared-image intensities, loads NIfTI labels as integers, and enables the CUDA expandable allocator:

```bash
/cis/home/dpadova/miniconda3/envs/wsi-pipeline/bin/python src/scripts/run_allen_registration_graph.py --infile results/allen_7T_SS_to_AHRAv2_registration/transformation_graph_config.json
```

This command performs the full optimization again and writes into `final_registration_graph_output`; it is not run when saving configs. The upstream graph CLI normalizes VTK inputs again and passes an unsupported `normalize` option to its NIfTI loader, so use the adapter for these prepared NIfTI inputs. The adapter leaves the optimizer and graph construction unchanged. It supplies all-ones target weights, equivalent to the selected registration's default `W0=None`.

Graph reconstruction is on the prepared 1 mm target grid. `sequence.json` points to the already-exported original 0.2 mm target label maps and the selected saved transform. Output cleanup of exactly-zero MRI background is a separate export operation, not an optimizer setting.
"""
    )
    return dict(
        config_file=str(config_file),
        transformation_config_file=str(graph_file),
        sequence_file=str(destination / "sequence.json"),
        selected_candidate=selected,
    )


if __name__ == "__main__":
    print(json.dumps(save_configs(), indent=2))
