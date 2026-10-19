"""Reproducible visualization of SEA-AD spatial omics and QNP products.

``scene``        what is drawn, in which coordinate frame, and how it got there
                 (loaders over existing derivatives; no new statistics).
``style``        Matplotlib conventions shared by every static figure.
``neuroglancer`` an adapter that turns scene layers into Neuroglancer assets using
                 wsi_pipeline's writers/server plus neuroglancer's annotation writer.
``figures_*``    the three figure families (scale space, feature maps, correspondence).
``cli``          ``python -m visualization.cli`` entry point.

Run from the project root with
``PYTHONPATH=src:src/seaad:src/preprocess:src/config_setup:src/scripts`` in the
``wsi-pipeline`` conda env (it has neuroglancer, wsi_pipeline, h5py and tifffile).
"""
