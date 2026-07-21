# Allen histology downloader

Python 3.10+ translation of `database/preprocessing/download_allen.m` from `acasamitjana/3dhirest`.

The script uses the documented Allen REST endpoints directly. It does not require AllenSDK.

## Install

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -r requirements-download_allen.txt
```

## Run

```bash
python download_allen.py --data-dir /path/to/Allen/downloads
```

Default constants match the MATLAB script:

- specimen ID: `708424`
- atlas provenance ID: `265297126`
- effective downsample: `5`
- SVG groups: `31,113753816,141667008,265297118`
- treatment ID `3`: Nissl
- treatment ID `16`: IHC/parvalbumin

## Useful commands

Metadata only:

```bash
python download_allen.py --data-dir ./allen_downloads --metadata-only
```

One section from each stain, without masks:

```bash
python download_allen.py \
  --data-dir ./allen_downloads \
  --limit 1 \
  --skip-masks
```

Generate masks from existing images:

```bash
python download_allen.py \
  --data-dir ./allen_downloads \
  --skip-downloads
```

Validate all manifest-tracked files without network access:

```bash
python download_allen.py \
  --data-dir ./allen_downloads \
  --validate-only
```

Refresh metadata while retaining the prior raw response:

```bash
python download_allen.py \
  --data-dir ./allen_downloads \
  --refresh-metadata \
  --metadata-only
```

## Output layout

```text
<DATA_DIR>/
├── manifest.csv
├── secInfo.json
├── secInfo.mat
├── raw/
│   ├── metadata/section_datasets_specimen_708424_<timestamp>.json
│   ├── nissl/images/image_####.jpg
│   └── ihc/images/image_####.jpg
├── nissl/
│   ├── images_orig/image_####.jpg
│   ├── labels_orig/seg_####.svg
│   └── masks_orig/image_####.png
└── ihc/
    ├── images_orig/image_####.jpg
    └── masks_orig/slice_###.png
```

Files under `raw/` preserve the bytes returned by the Allen API. The `images_orig` files are the locally halved JPEGs expected by the original 3dhirest workflow.

## Manifest behavior

`manifest.csv` records:

- source URL and request parameters
- retrieval timestamp and selected HTTP headers
- file size and SHA-256
- section ID, section number, and stain
- source SHA-256 for generated images and masks
- whether a file was downloaded, generated, or adopted from an existing directory

The first two columns are `url` and `path`. Rows with a non-empty URL describe acquired files and can be selected for a DataLad `addurls` workflow.

On restart, existing tracked files are hash-checked before reuse. A mismatch stops the run. `--overwrite` is required to replace a conflicting file.

## Fixes relative to MATLAB

- Creates all output directories.
- Replaces undefined `NISSL_DIR` and `it_slice` variables.
- Corrects the malformed SVG request.
- Uses structured JSON metadata instead of line-by-line XML parsing.
- Retains raw API responses rather than saving only transformed images.
- Separates acquisition records from derived preprocessing records.
- Adds retries, atomic writes, SHA-256 validation, restart support, and offline manifest validation.
