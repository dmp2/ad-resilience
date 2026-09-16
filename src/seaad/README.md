# SEA-AD MTG Docker package

This package turns your SEA-AD MTG crosswalk workflow into a Docker-runnable CLI.

It preserves the existing analysis steps from `seaad_mtg_pipeline.txt`:

1. local `.h5ad` donor crosswalk
2. spatial section manifest
3. spatial-neuropath linkage
4. `.h5ad` crosswalk probe
5. ranked crosswalk candidate report

## What changed

- `seaad_mtg_crosswalk.py` was converted from a hardcoded-path script into a real CLI.
- A single entrypoint command, `seaad-mtg`, was added.
- A `pipeline` command now runs the full workflow end-to-end.
- The container is designed so you can mount your Windows SEA-AD directory at runtime rather than baking large data files into the image.

## Expected data layout

Point `--data-dir` at the folder that contains your files, for example:

- `SEAAD_MTG_MERFISH.2024-12-11.h5ad`
- `68debdfdd1b8e9f8fd64dab0_sea-ad_cohort_donor_metadata_072524.xlsx`
- `68debdfd4748b7546943a7b4_sea-ad_cohort_harmonized_cognitive_scores_20241213.xlsx`
- `68debdfdae5f82b97af2fb0f_sea-ad_cohort_mri_volumetrics.xlsx`
- `68debdfd24606956df13f2dd_sea-ad_all_mtg_quant_neuropath_bydonorid_081122.csv`
- `68debdff5b8003454786ea29_sea-ad_cohort_mtg-tissue_extractions-luminex_data.xlsx`
- optionally `SpecimenMetadata.csv`

Your Windows path appears to be:

`C:\Users\dpado\Documents\jhu\postdoc\grants\Kavli\SEA-AD`

## Build

From the folder containing this Dockerfile:

```bash
docker build -t seaad-mtg .
```

## Run the full pipeline

### PowerShell

```powershell
docker run --rm -it `
  --mount type=bind,source="C:\Users\dpado\Documents\jhu\postdoc\grants\Kavli\SEA-AD",target=/work/data `
  seaad-mtg pipeline `
  --data-dir /work/data
```

### CMD

```bat
docker run --rm -it ^
  --mount type=bind,source="C:\Users\dpado\Documents\jhu\postdoc\grants\Kavli\SEA-AD",target=/work/data ^
  seaad-mtg pipeline ^
  --data-dir /work/data
```

## Run only the local crosswalk step

```powershell
docker run --rm -it `
  --mount type=bind,source="C:\Users\dpado\Documents\jhu\postdoc\grants\Kavli\SEA-AD",target=/work/data `
  seaad-mtg crosswalk `
  --data-dir /work/data
```

## Override filenames

If any file names differ, override them explicitly:

```powershell
docker run --rm -it `
  --mount type=bind,source="C:\Users\dpado\Documents\jhu\postdoc\grants\Kavli\SEA-AD",target=/work/data `
  seaad-mtg pipeline `
  --data-dir /work/data `
  --h5ad SEAAD_MTG_MERFISH.2024-12-11.h5ad `
  --specimen-metadata "" 
```

Passing an empty string for `--specimen-metadata` disables that optional file.

## Manual outputs

By default, outputs are written into the mounted working directory, usually `/work/data`, under:

- `seaad_mtg_spatial_crosswalk_outputs`
- `seaad_spatial_section_manifest`
- `seaad_mtg_linkage`
- `seaad_mtg_h5ad_crosswalk_probe`
- `seaad_crosswalk_ranked`

## Notes

- The S3-based steps require network access from inside the container.
- The `.h5ad` step opens the AnnData object in backed mode, so the file stays on disk and is not fully loaded into RAM.
- If donor-column detection fails for the `.h5ad`, re-run with `--obs-donor-column <column_name>`.

## CLI help

```bash
docker run --rm seaad-mtg --help
docker run --rm seaad-mtg pipeline --help
docker run --rm seaad-mtg crosswalk --help
```


## Docker Compose

A single `docker-compose.yml` is included for both Windows and WSL2 use.
The container paths stay fixed:

- input data mount: `/work/data`
- output/work mount: `/work/work`

Only the host-side path changes.

### Option A: launch from PowerShell or CMD

Copy `.env.windows.example` to `.env`, then run:

```powershell
copy .env.windows.example .env
docker compose up --build
```

### Option B: launch from WSL2

Copy `.env.wsl.example` to `.env`, then run:

```bash
cp .env.wsl.example .env
docker compose up --build
```

### Run a different command

By default the compose service runs the full pipeline. To run only the local crosswalk step, set:

```env
SEAAD_COMMAND=crosswalk
```

and then run `docker compose up --build`, or override ad hoc:

```bash
docker compose run --rm seaad-mtg crosswalk --data-dir /work/data --work-dir /work/work
```

### Notes on paths

- If you launch compose from Windows, use a Windows path like `C:\Users\...` in `.env`.
- If you launch compose from WSL2, use the mounted Linux path like `/mnt/c/Users/...` in `.env`.
- Since your `which docker` in WSL points to Docker Desktop's Windows-managed binary, WSL2 should work fine as long as Docker Desktop file sharing is enabled for that drive.
