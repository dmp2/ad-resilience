"""Download the raw SEA-AD quantitative-neuropathology whole-slide images and HALO outputs.

For every slide folder s3://sea-ad-quantitative-neuropathology/<region>/<donor>/<slide>/
this fetches the raw image and its HALO layer boundaries,
    <slide>.svs, <slide>.annotations  -> data/raw/sea-ad/qnp/<region>/<donor>/
and the HALO analysis masks with their job-ID crosswalk,
    analysis_masks/*                  -> data/raw/sea-ad/qnp/<region>/<donor>/analysis_masks/
and records every listed key in data/raw/sea-ad/qnp/manifest.csv. Files already present
at their listed size are skipped, so an interrupted run resumes where it stopped.

The full set is ~2,000 slides of ~1 GB each; use --dry-run to see the total first.
The `middle-temporal-gyrus-and-superior-temporal-gyrus/` prefix is not listed: it holds
only DeepZoom viewer tiles of the MTG slides, no raw images.
"""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from acquisition.fetch_seaad_omics_inventory import find_project_root, list_folders, list_prefix
from acquisition.s3_download import PlannedDownload, SelectionItem, download_one

BUCKET = "https://sea-ad-quantitative-neuropathology.s3.us-west-2.amazonaws.com/"
REGIONS = (
    "middle-temporal-gyrus",
    "hippocampus-medial-entorhinal-cortex",
    "dorsolateral-prefrontal-cortex",
    "primary-visual-cortex-extrastriate-occipital-cortex",
)
RAW_ROOT = Path("data/raw/sea-ad/qnp")


def list_slides() -> list[dict]:
    """Every slide-level file and analysis mask in the bucket (~1 min).

    Lists one folder level at a time so the DeepZoom tile pyramids beside each
    slide (`downsampled/`, thousands of keys per slide) are never enumerated.
    """
    with ThreadPoolExecutor(16) as pool:
        def flatmap(fn, prefixes, *args):
            return [y for ys in pool.map(lambda p: fn(p, *args), prefixes) for y in ys]

        donor_dirs = flatmap(list_folders, [f"{r}/" for r in REGIONS], BUCKET)
        slide_dirs = flatmap(list_folders, donor_dirs, BUCKET)
        return (flatmap(list_prefix, slide_dirs, "/", BUCKET)
                + flatmap(list_prefix, [f"{s}analysis_masks/" for s in slide_dirs], None, BUCKET))


def plan(obj: dict, root: Path) -> PlannedDownload:
    region, donor, _slide, *rest = obj["key"].split("/")
    local = RAW_ROOT / region / donor / Path(*rest)
    dest = root / local
    if not dest.exists():
        action = "download"
    elif dest.stat().st_size == obj["size_bytes"]:
        action = "skip_verified"
    else:
        raise SystemExit(f"{dest} exists with a size other than the listed {obj['size_bytes']} bytes")
    item = SelectionItem("QNP", local.suffix.lstrip("."), region, obj["key"], obj["size_bytes"], str(local))
    return PlannedDownload(item, dest, action)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--region", action="append", choices=REGIONS,
                        help="Restrict to this region prefix; repeatable. Default: all")
    parser.add_argument("--donor", action="append", help="Restrict to this donor ID; repeatable")
    parser.add_argument("--jobs", type=int, default=4, help="Parallel transfers (default 4)")
    parser.add_argument("--dry-run", action="store_true", help="List and size only")
    args = parser.parse_args(argv)

    root = find_project_root(Path.cwd())
    objects = list_slides()
    manifest = root / RAW_ROOT / "manifest.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["key", "size_bytes", "last_modified"])
        writer.writeheader()
        writer.writerows(objects)
    print(f"{len(objects)} files ({sum(o['size_bytes'] for o in objects) / 1e9:.1f} GB) listed -> {manifest}")

    selected = [o for o in objects if (not args.region or o["key"].split("/")[0] in args.region)
                and (not args.donor or o["key"].split("/")[1] in args.donor)]
    if not selected:
        raise SystemExit("No files match the --region/--donor selection")
    todo = [p for p in (plan(o, root) for o in selected) if p.action == "download"]
    print(f"{len(selected)} selected; {len(todo)} to download "
          f"({sum(p.item.size_bytes for p in todo) / 1e9:.1f} GB), {len(selected) - len(todo)} already present")
    if args.dry_run:
        return 0

    def fetch(numbered):
        i, planned = numbered
        download_one(planned, BUCKET)
        print(f"[{i}/{len(todo)}] {planned.destination.relative_to(root)}", flush=True)

    with ThreadPoolExecutor(args.jobs) as pool:
        list(pool.map(fetch, enumerate(todo, start=1)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
