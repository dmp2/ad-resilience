# Download OpenNeuro ds003590

This script uses DataLad/git-annex to clone and retrieve the exact OpenNeuro
snapshot `ds003590` version `1.0.2`.

Keep this dataset separate from the raw Allen section-image directory created by
`download_allen.py`. A simple project layout is:

```text
data/
├── allen_sections/          # download_allen.py output
└── openneuro/
    └── ds003590/            # this script's output
```

## Install DataLad

Using conda or mamba is the simplest option:

```bash
mamba create -n allen-recon -c conda-forge python=3.11 datalad git-annex
mamba activate allen-recon
```

## Recommended download

This retrieves the source 7T MRI, the revision-1 reconstructed volumes, and the
MNI mapping:

```bash
python download_openneuro_ds003590.py \
    --data-dir /path/to/data/openneuro/ds003590
```

The command is restartable. Rerunning it retrieves only content that is still
missing.

## Retrieve both reconstruction versions

```bash
python download_openneuro_ds003590.py \
    --data-dir /path/to/data/openneuro/ds003590 \
    --content all-recon
```

This retrieves:

```text
sub-01/ses-7T/anat/
derivatives/historecon/
derivatives/historecon-revision-1/
derivatives/mni-mapping/
```

## Other useful modes

Clone only the lightweight metadata and file index:

```bash
python download_openneuro_ds003590.py \
    --data-dir /path/to/data/openneuro/ds003590 \
    --content metadata
```

Retrieve the complete dataset:

```bash
python download_openneuro_ds003590.py \
    --data-dir /path/to/data/openneuro/ds003590 \
    --content all
```

Retrieve only a specific path:

```bash
python download_openneuro_ds003590.py \
    --data-dir /path/to/data/openneuro/ds003590 \
    --path derivatives/historecon-revision-1/sub-01
```

Run a full git-annex checksum check after downloading:

```bash
python download_openneuro_ds003590.py \
    --data-dir /path/to/data/openneuro/ds003590 \
    --content all-recon \
    --verify
```

The script writes a small provenance record beside the dataset, named
`ds003590.download.json`. Do not put analysis outputs inside the downloaded
DataLad dataset; use a separate project `derivatives/` or `results/` directory.
