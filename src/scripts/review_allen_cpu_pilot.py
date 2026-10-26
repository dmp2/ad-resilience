"""Matched 0.5 mm anatomical detail views for the CPU continuation."""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
import tune_allen_1mm_registration as tuning
from refine_allen_0p5mm_cpu import prepare_inputs


def review_detail(data, parent, candidate):
    directory = tuning.TUNING / candidate
    results = [tuning.load_result(name) for name in (parent, candidate)]
    outputs = []
    # Each plane is evaluated directly; no full-volume coordinate field is needed.
    panels = [
        ("Cerebellum: sagittal x=-20 mm", 0, -20, (-90, -20), (-55, 15)),
        ("Cerebellum: sagittal x=0 mm", 0, 0, (-90, -20), (-55, 15)),
        ("Cerebellum: sagittal x=20 mm", 0, 20, (-90, -20), (-55, 15)),
        ("Cortex: sagittal x=40 mm", 0, 40, (-65, 70), (10, 55)),
        ("Cortex: coronal y=0 mm", 1, 0, (-80, 80), (10, 60)),
        ("Cortex: axial z=25 mm", 2, 25, (-80, 80), (-90, 70)),
    ]
    for region, panel_group in (("cerebellum", panels[:3]), ("cortex", panels[3:])):
        fig, plots = plt.subplots(2, 3, figsize=(16, 10), squeeze=False)
        for row, (name, result) in enumerate(zip((parent, candidate), results)):
            transforms = [
                tuning.emlddmm.Transform(result["A"], direction="b"),
                tuning.emlddmm.Transform(
                    result["v"], domain=result["xv"], direction="b"
                ),
            ]
            for col, (title, axis, position, range_a, range_b) in enumerate(
                panel_group
            ):
                varying = [a for a in range(3) if a != axis]
                grids = [None] * 3
                grids[axis] = np.array([position], dtype=np.float32)
                for a, limits in zip(varying, (range_a, range_b)):
                    grids[a] = data["xJ"][a][
                        (data["xJ"][a] >= limits[0]) & (data["xJ"][a] <= limits[1])
                    ]
                points = torch.stack(
                    torch.meshgrid([torch.as_tensor(x) for x in grids], indexing="ij")
                )
                with torch.no_grad():
                    backward = tuning.emlddmm.compose_sequence(transforms, points)
                    atlas = tuning.numpy(
                        tuning.emlddmm.apply_transform_float(
                            data["xI"], data["I_brain"], backward
                        )
                    )[0].squeeze(axis)
                    target = tuning.numpy(
                        tuning.emlddmm.apply_transform_float(
                            data["xJ"], data["J"], points
                        )
                    )[0].squeeze(axis)
                rgb = np.clip(np.stack((atlas, target, atlas), axis=-1), 0, 1)
                plot = plots[row, col]
                plot.imshow(
                    rgb.transpose(1, 0, 2),
                    origin="lower",
                    extent=[
                        grids[varying[0]][0],
                        grids[varying[0]][-1],
                        grids[varying[1]][0],
                        grids[varying[1]][-1],
                    ],
                    interpolation="nearest",
                )
                plot.set_title(f'{("1 mm parent", "0.5 mm CPU pilot")[row]}\n{title}')
                plot.set_xlabel("Centered coordinate (mm)")
                plot.set_ylabel("Centered coordinate (mm)")
        fig.suptitle("Matched 0.5 mm image samples: atlas magenta / target green")
        fig.tight_layout()
        output = directory / f"{region}_detail_comparison.png"
        fig.savefig(output, dpi=150)
        plt.close(fig)
        outputs.append(output)
    return outputs


if __name__ == "__main__":
    torch.set_num_threads(4)
    data, _ = prepare_inputs()
    for path in review_detail(
        data, "fresh_nocsf_boundary_refine", "fresh_nocsf_0p5mm_cpu_pilot"
    ):
        print(path, flush=True)
