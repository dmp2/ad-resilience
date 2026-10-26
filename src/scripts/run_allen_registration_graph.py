"""Run the transformation graph on the saved, already-normalized Allen inputs."""

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np
import tune_allen_1mm_registration as tuning
import transformation_graph_v01 as graph_module


class PreparedImage(tuning.emlddmm.Image):
    """Preserve stored intensities and identify NIfTI labels by graph image name."""

    def __init__(self, space, name, fpath, mask=None, x=None):
        self.space, self.name, self.path = space, name, fpath
        self.x, self.data, self.title, self.names = tuning.emlddmm.read_data(fpath, x=x)
        self.annotation = name.lower() == "labels"
        if self.annotation:
            self.data = self.data.astype(np.int32)
            self.title = "atlas_labels"
        else:
            self.data = self.data.astype(np.float32)
        self.mask = (
            np.ones_like(self.data[0], dtype=np.float32) if mask is True else mask
        )


def main():
    graph_module.emlddmm.Image = PreparedImage
    graph_module.main()


if __name__ == "__main__":
    main()
