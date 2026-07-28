# Allen section OME-Zarr rasterization

This command converts the 106 expert-annotated Allen Nissl plates into an
analysis-ready derivative. It does not alter or replace the authoritative Allen
JPEG, SVG, ontology, or canonical raw metadata.

## Environment

Create the dedicated Python 3.12 production environment without changing the
project's existing xIV environment:

```bash
conda env create -f configs/environment-rasterize-allen.yml
conda activate ad-resilience-allen-zarr
```

For development and the independent section-111 validation only:

```bash
python -m pip install -r configs/requirements-rasterize-allen-dev.txt
```

CairoSVG is not needed for production rasterization or read-only derivative
validation.

## Run

From the repository root:

```bash
python src/preprocessing/rasterize_allen_annotations.py \
  --data-dir data/raw/allen/specimen_708424
```

The default derivative is written under:

```text
data/derivatives/allen/specimen_708424/annotations_ome_zarr/
```

Every `section-NNNN.ome.zarr` contains the RGB image at `0` and one true 2-D
`uint32` label array at `labels/group-ID/0` for each group present in the source
SVG. Missing source groups are recorded as absent and are not materialized as
empty arrays.

Useful bounded runs include:

```bash
python src/preprocessing/rasterize_allen_annotations.py \
  --data-dir data/raw/allen/specimen_708424 \
  --section-number 111 --write-qc

python src/preprocessing/rasterize_allen_annotations.py \
  --data-dir data/raw/allen/specimen_708424 \
  --workers 4
```

A filtered run can resume an initialized derivative but cannot initialize a new
source snapshot. Existing valid packages are deeply verified and skipped.
`--overwrite` replaces packages only within the same frozen source snapshot.
Any raw inventory, raw manifest, or structures-table drift requires a new
`--output-dir`.

## Read-only validation

```bash
python src/preprocessing/rasterize_allen_annotations.py \
  --data-dir data/raw/allen/specimen_708424 \
  --verify-existing
```

This reads the raw and derivative files, recomputes source and package hashes,
opens every Zarr array, and validates NGFF metadata, dimensions, physical
spacing, codecs, label IDs, ontology properties, and manifest rows. It does not
render, repair, update metadata, generate QC, or write files.

## Boundary and overlap semantics

Paths are painted in SVG document order. The Allen `order` attribute is recorded
for validation and provenance but never controls painting. The operational
boundary policy is **Skia non-antialiased target-grid fill**; no pixel-center
containment equivalence is claimed. SVG strokes are visual outlines and are not
included as categorical fills.

Within each graphic-group layer, a later path overwrites an earlier path.
Conflicting coverage by different IDs and repeated coverage by the same ID are
tracked as separate unique-pixel masks. The derivative manifest's
`within_layer_conflicting_pixel_count_sum` sums independent per-layer conflict
counts; it is not a package-wide union of physical pixels.

## Independent section-111 gate

Run this development-only gate before a full-corpus production run:

```bash
ALLEN_SECTION111_VALIDATION=1 \
PYTHONPATH=src/preprocessing \
pytest -q src/tests/test_rasterize_allen_annotations.py \
  -k section_111_independent_cairosvg
```

The test renders section 111 independently with CairoSVG. It requires exact
occupancy agreement away from a one-target-pixel boundary band, bounds agreement
within one pixel, and no systematic alignment discrepancy.
