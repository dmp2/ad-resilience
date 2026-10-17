"""Render one montage per donor and region from the downloaded QNP whole-slide images.

Reads the lowest-resolution pyramid level (~1/16 of full resolution) of every .svs in
data/raw/sea-ad/qnp/<region>/<donor>/, outlines its HALO analysis regions (cortical
layers or hippocampal subfields; excluded holes dashed), and writes
data/derivatives/sea-ad/qnp/montages/<region>/<donor>.png. The slide-label image
embedded in some .svs files is never read.

Needs tifffile + imagecodecs (the Aperio JPEG tiles), e.g. the wsi-pipeline env.
"""

from __future__ import annotations

import argparse
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import tifffile

from acquisition.fetch_seaad_omics_inventory import find_project_root
from qnp.download import RAW_ROOT, REGIONS

OUTPUT_ROOT = Path("data/derivatives/sea-ad/qnp/montages")
STAINS = {"NEUN": "NeuN", "LFB": "H&E-LFB", "AT": "AT8 + pTDP-43",
          "I6": "Iba1 + 6E10", "GFAP": "GFAP", "ASYN": "α-synuclein"}
ORDER = list(STAINS)


def stain(svs: Path) -> str:
    return svs.stem.split("-")[-1].upper()


def overview(svs: Path) -> tuple[np.ndarray, float]:
    """Smallest pyramid level and its scale relative to the baseline pixels."""
    with tifffile.TiffFile(svs) as tif:
        image = tif.series[0].levels[-1].asarray()
        return image, image.shape[1] / tif.series[0].shape[1]


def outlines(svs: Path) -> list[tuple[str, np.ndarray, bool]]:
    """(analysis region, baseline-pixel polygon, is excluded hole) for one slide.

    MTG slides keep every layer in <slide>.annotations; the other regions write one
    <slide>_<region>_analysis.annotations per analysis region under analysis_masks/.
    """
    files = [svs.with_suffix(".annotations"),
             *sorted((svs.parent / "analysis_masks").glob(f"{svs.stem}_*.annotations"))]
    polygons = []
    for path in filter(Path.exists, files):
        for annotation in ET.parse(path).getroot().iter("Annotation"):
            for region in annotation.iter("Region"):
                xy = np.array([(float(v.get("X")), float(v.get("Y"))) for v in region.iter("V")])
                if len(xy):
                    polygons.append((annotation.get("Name"), np.vstack([xy, xy[:1]]),
                                     region.get("NegativeROA") == "1"))
    return polygons


def make_montage(donor_dir: Path, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    slides = sorted(donor_dir.glob("*.svs"),
                    key=lambda p: (ORDER.index(stain(p)) if stain(p) in STAINS else len(ORDER), p.name))
    ncols = min(3, len(slides))
    nrows = math.ceil(len(slides) / ncols)
    polygons = {svs: outlines(svs) for svs in slides}
    names = sorted({name for ps in polygons.values() for name, _, _ in ps})
    colors = dict(zip(names, plt.get_cmap("tab20").colors))

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows + 0.6), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, svs in zip(axes.flat, slides):
        image, scale = overview(svs)
        ax.imshow(image)
        for name, xy, hole in polygons[svs]:
            ax.plot(*(xy * scale).T, color=colors[name], lw=0.6, ls="--" if hole else "-")
        ax.set_title(STAINS.get(stain(svs), svs.stem), fontsize=11)
    fig.suptitle(f"SEA-AD QNP  ·  {donor_dir.name}  ·  {donor_dir.parent.name}", fontsize=14)
    if names:
        fig.legend([plt.Line2D([], [], color=colors[n]) for n in names], names,
                   loc="lower center", ncol=min(len(names), 12), frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.04 if names else 0, 1, 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--region", action="append", choices=REGIONS,
                        help="Restrict to this region prefix; repeatable. Default: all downloaded")
    parser.add_argument("--donor", action="append", help="Restrict to this donor ID; repeatable")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    root = find_project_root(Path.cwd())
    donor_dirs = sorted(d for d in (root / RAW_ROOT).glob("*/*") if d.is_dir() and any(d.glob("*.svs"))
                        and (not args.region or d.parent.name in args.region)
                        and (not args.donor or d.name in args.donor))
    if not donor_dirs:
        raise SystemExit(f"No downloaded slides match the selection under {root / RAW_ROOT}")
    for donor_dir in donor_dirs:
        out = root / OUTPUT_ROOT / donor_dir.parent.name / f"{donor_dir.name}.png"
        if out.exists() and not args.overwrite:
            print(f"exists, skipped: {out.relative_to(root)}")
            continue
        make_montage(donor_dir, out)
        print(f"wrote {out.relative_to(root)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
