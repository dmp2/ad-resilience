# Allen 708424 raw acquisition

This workflow freezes the raw, publicly exposed Allen data used by the inherited
3DHiResT pipeline. It acquires exact Allen JPEG response bytes, the 106 annotated
atlas SVGs, and Allen structure graph 16. It does not rasterize SVGs, create
OME/NIfTI products, reconstruct a volume, register MRI, propagate labels, or
produce EM-LDDMM sidecars.

## Corpus accounting

The accepted Allen API inventory observed on **July 28, 2026** is distinct from
the corpus reported by Ding et al.:

| Series | Allen API | Published | Currently unavailable |
|---|---:|---:|---:|
| Nissl | 641 | 679 | 38 |
| PV/parvalbumin | 287 | 339 | 52 |
| SMI-32/NFP | 0 | 338 | 338 |
| Total | 928 | 1,356 | 428 |

The inherited pipeline's `ihc/` directory means PV/parvalbumin. It must not be
renamed destructively. Treatment 5 is Allen's `IHC:SMI-32` vocabulary term, but
the accepted snapshot has no specimen-associated SMI-32 SectionDataSet.

A bounded official-source search on July 28, 2026 found no qualifying SMI-32
provider. This is a dated result, not a permanent-unavailability claim. The
search checked Allen/BrainSpan atlas pages and RMA/legacy image records, Ding
paper links and supplements, official archives, 3DHiResT, and OpenNeuro. A future
source is admitted only after donor, treatment, provenance, section identity,
reproducible download, and reuse evidence are verified. Its observed count is
reported against the published denominator of 338; a partial release is valid
partial coverage and count equality alone does not prove completeness.

| Candidate | Organization / URL | Donor and stain evidence | Count / identity / resolution | Reuse and download finding |
|---|---|---|---|---|
| Allen RMA and atlas | [Allen API](https://api.brain-map.org/api/v2/) | specimen 708424; NISSL and IHC:Parvalbumin datasets; global IHC:SMI-32 term | 641 Nissl, 287 PV, no specimen SMI-32 dataset; SectionImage IDs and section numbers; Allen image service | official API and terms; Nissl/PV reproducibly downloadable; SMI-32 unavailable in snapshot |
| BrainSpan / Allen atlas pages and legacy records | Allen Institute | bounded search for the same donor and SMI-32/NFP | no authoritative downloadable SMI-32 section series identified | no qualifying provider found |
| Ding paper and supplements | [Ding et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC5054943/) | published stain totals for the atlas donor | 679/339/338; paper accounting, not a section download manifest | publication supports denominators, not a reproducible raw SMI-32 provider |
| 3DHiResT and OpenNeuro | project repositories/archives | derivative/reconstruction provenance examined | inherited selection exposes 641 Nissl and 287 PV; no qualifying raw SMI-32 serial source | useful derivatives, not an admitted SMI-32 raw provider |
| Other official archives | Allen/BrainSpan-linked archives | donor, stain, provenance, and section identity required | no candidate met all requirements in the bounded search | reassess when official holdings change |

## What is acquired

- `nissl/images_orig/`: 641 Nissl JPEGs.
- `ihc/images_orig/`: 287 PV JPEGs (legacy path).
- `nissl/labels_orig/`: one SVG for each of 106 atlas plates.
- `ontology/structure_graph_16.json`: the untouched Allen ontology response.
- `nissl/masks_orig/` and `ihc/masks_orig/`: retained legacy mask capability,
  but the wrapper skips mask generation. No SMI-32 mask threshold is assumed.

Each SVG request includes all four graphic groups in one response:

| ID | Allen label | Purpose |
|---:|---|---|
| 31 | Atlas - Developing Human | broader developing-human anatomy |
| 113753816 | Atlas - Developing Human Sulci | sulcal/gyral annotations |
| 141667008 | Atlas - Developing Human Hotspots | hotspot annotations |
| 265297118 | Atlas - Developing Human Brodmann | modified Brodmann annotations |

A group can legitimately be absent on an individual plate. The manifest stores
only present IDs as a semicolon-delimited value. Per-group path and unique-ID
counts are validation results, not TSV-embedded JSON.

Graph 16 (Developing Human Brain Atlas) is verified through atlas 265297126.
It is required because SVG paths carry numeric `structure_id` values. Any value
not present in graph 16 is a validation error; `structures.tsv` is never padded
with invented rows.

## Metadata

Only three files under `metadata/` are canonical:

- `dataset.json`: shared facts, accepted API inventory/digest, treatment and
  provider status, published coverage, graphic-group catalog, spatial sampling,
  provenance, and software version.
- `manifest.tsv`: one row per real JPEG, SVG, raw ontology JSON, mask, or later
  derivative. Metadata files and compatibility exports are excluded.
- `structures.tsv`: one row per structure flattened strictly from graph 16.

Root `secInfo.json` and `secInfo.mat` remain noncanonical 3DHiResT/MATLAB
compatibility exports; their IHC arrays continue to mean PV. Superseded metadata
TSVs are moved into a timestamped quarantine only after canonical output passes
validation. Raw sections and SVGs remain in local 2-D coordinates. No origin,
3-D orientation, AP coordinate, or MRI affine is invented.

## Run and resume

From any working directory:

```bash
bash scripts/download_allen_sections.sh
```

The wrapper derives the repository root, takes a nonblocking writer lock, uses
Allen-direct/downsample level 5, selects all currently available accepted series,
requests all SVG groups, acquires the ontology, skips masks, writes logs and an
atomic status file, invokes offline validation, and reports disk usage. Rerun the
same command to resume. Verified files are hash-checked and skipped; unregistered
files are byte-compared with Allen, adopted when identical, or quarantined and
replaced when different.

For a partial/test run, use the Python CLI directly, for example
`--series nissl --limit 2`. `--limit` limits work only; it never truncates the
accepted inventory or canonical expected counts. Explicit `--series smi32`
currently exits with the dated provider-availability explanation.

## API inventory changes

The live inventory is queried before any acquisition directory creation,
quarantine, migration, or canonical metadata write. If treatment identity,
dataset IDs, or counts differ, the program exits 3 with
`API_INVENTORY_CHANGED`. The wrapper saves the observed inventory report but
never accepts it automatically.

After scientific review, re-query and accept only the exact displayed digest:

```bash
python src/download_data/download_allen.py \
  --data-dir data/raw/allen/specimen_708424 \
  --inventory-json reviewed-inventory.json \
  --accept-api-inventory-sha256 SHA256_FROM_REVIEW
```

## Read-only validation

```bash
python src/download_data/download_allen.py \
  --data-dir data/raw/allen/specimen_708424 \
  --validate-only
```

Optional compact JSON report:

```bash
python src/download_data/download_allen.py \
  --data-dir data/raw/allen/specimen_708424 \
  --validate-only \
  --validation-json results/status/allen_validation.json
```

Without `--validation-json`, validation performs no network calls, directory
creation, repair, quarantine, or timestamp update. API acquisition can pass at
928/928 while published coverage is correctly reported as incomplete at
928/1,356 (38 Nissl, 52 PV, and 338 SMI-32 unavailable).
