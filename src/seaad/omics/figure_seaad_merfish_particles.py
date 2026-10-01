#!/usr/bin/env python3
"""Render SEA-AD MERFISH sections and a donor montage from xIV particles."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple, Sequence

import h5py
import numpy as np

from omics.export_seaad_merfish_section_for_xiv import _decode_array, export_section_for_xiv
from omics.visualize_seaad_merfish_section import _obs_column_length, _read_obs_chunk


REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_ROOT = REPO_ROOT / "data/derivatives/sea-ad/merfish"
RAW_ROOT = REPO_ROOT / "data/raw/sea-ad/merfish"
STEM = "full_section_particles"


def _directory_name(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", value) or value in {".", ".."}:
        raise ValueError(f"Unsafe donor or section identifier: {value!r}")
    return value


class SourceSpec(NamedTuple):
    region: str
    donor: str | None
    section: str | None
    section_field: str | None
    coordinate_key: str
    feature_field: str


SINGLE_SPECIMENS = {
    "1444201261_MEC_mapped.h5ad": SourceSpec(
        "MEC", "H24.30.005", "1444201261", None, "spatial", "Subclass_scANVI"),
    "1444211893_HPF_mapped.h5ad": SourceSpec(
        "HPF", "H24.30.005", "1444211893", None, "spatial", "Subclass_scANVI"),
}
DONOR_MAPPING_URL = "https://community.brain-map.org/t/donor-mapping-for-merfish-mec-hpf-specimens/5020"


def _source_spec(path: Path) -> SourceSpec:
    if path.name in SINGLE_SPECIMENS:
        return SINGLE_SPECIMENS[path.name]
    match = re.fullmatch(r"SEAAD_([A-Za-z0-9]+)_MERFISH\..+\.h5ad", path.name)
    if match is None:
        raise ValueError(f"Unknown MERFISH source filename: {path.name}")
    return SourceSpec(match.group(1), None, None, "Specimen Barcode",
                      "X_spatial_raw", "Subclass")


def _sections_for_donor(h5ad: Path, donor: str, chunk_rows: int) -> list[str]:
    """Find all barcodes without loading the expression matrix."""
    spec = _source_spec(h5ad)
    if spec.section is not None:
        return [spec.section] if donor == spec.donor else []
    barcodes: set[str] = set()
    with h5py.File(h5ad, "r") as handle:
        obs = handle["obs"]
        n = _obs_column_length(obs, "Donor ID")
        for start in range(0, n, chunk_rows):
            stop = min(n, start + chunk_rows)
            donors = _read_obs_chunk(obs, "Donor ID", start, stop)
            matches = donors == donor
            if np.any(matches):
                sections = _read_obs_chunk(obs, "Specimen Barcode", start, stop)
                barcodes.update(str(value) for value in sections[matches]
                                if value is not None and str(value) not in {"", "nan"})
    return sorted(barcodes)


def _subclass_colors(h5ad: Path, labels: list[str], field: str) -> tuple[list[str], str]:
    """Use the release palette when its category order matches the xIV channels."""
    import matplotlib.colors as mcolors

    with h5py.File(h5ad, "r") as f:
        column = f[f"obs/{field}"]
        categories = [str(v) for v in _decode_array(column["categories"][...])]
        colors_key = f"{field}_colors"
        if colors_key in f["uns"]:
            colors = [str(v) for v in _decode_array(f["uns"][colors_key][...])]
            if categories == labels[: len(categories)] and len(colors) == len(categories):
                if all(mcolors.is_color_like(color) for color in colors):
                    if len(labels) == len(colors) + 1 and labels[-1] == "__missing__":
                        colors.append("#8b9097")
                    return [mcolors.to_hex(color) for color in colors], f"H5AD uns/{colors_key}"

    colors = []
    for label in labels:
        if label == "__missing__":
            colors.append("#8b9097")
        else:
            hue = int.from_bytes(hashlib.sha256(label.encode()).digest()[:8], "big") / 2**64
            colors.append(mcolors.to_hex(mcolors.hsv_to_rgb((hue, 0.68, 0.75))))
    return colors, "stable label-based HSV"



def _render(Z: np.ndarray, nu: np.ndarray, labels: list[str], colors: list[str],
            donor: str, section: str, spec: SourceSpec, directory: Path, pdf: bool) -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    channel = np.argmax(nu, axis=1)
    counts = np.bincount(channel, minlength=len(labels))
    present = [i for i, count in enumerate(counts) if count]
    x, y = Z[:, 0], Z[:, 1]
    xpad, ypad = max(float(np.ptp(x)) * 0.025, 1), max(float(np.ptp(y)) * 0.025, 1)

    fig = plt.figure(figsize=(14, 9), facecolor="white")
    ax = fig.add_axes((0.035, 0.09, 0.70, 0.79))
    legend = fig.add_axes((0.76, 0.10, 0.22, 0.78))
    for i in sorted(present, key=lambda j: (-counts[j], j)):
        mask = channel == i
        ax.scatter(x[mask], y[mask], s=3.5, c=colors[i], alpha=0.92,
                   linewidths=0, rasterized=True)
    ax.set_xlim(float(x.min() - xpad), float(x.max() + xpad))
    ax.set_ylim(float(y.min() - ypad), float(y.max() + ypad))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_facecolor("#f8f9fb")

    legend.axis("off")
    legend.text(0, 1, "Subclass", va="top", fontsize=13, fontweight="bold")
    legend.text(1, 1, "Cells", va="top", ha="right", fontsize=9, color="#59636e")
    order = sorted(present, key=lambda i: labels[i].casefold())
    step = min(0.034, 0.91 / max(len(order), 1))
    for row, i in enumerate(order):
        y_pos = 0.95 - row * step
        legend.scatter([0.025], [y_pos], s=40, c=colors[i], linewidths=0,
                       transform=legend.transAxes, clip_on=False)
        legend.text(0.07, y_pos, labels[i], va="center", fontsize=8.4,
                    transform=legend.transAxes)
        legend.text(0.99, y_pos, f"{counts[i]:,}", va="center", ha="right",
                    fontsize=8.1, color="#59636e", transform=legend.transAxes)

    fig.text(0.035, 0.955, f"SEA-AD {spec.region} MERFISH", fontsize=18, fontweight="bold")
    fig.text(0.035, 0.916, f"Donor {donor}  ·  Specimen {section}  ·  {len(Z):,} cells",
             fontsize=11, color="#4b5563")
    fig.text(0.035, 0.045,
             f"Full, undeformed xIV-LDDMM input particle cloud · {spec.coordinate_key} coordinates",
             fontsize=9, color="#59636e")
    for suffix in (("png", "pdf") if pdf else ("png",)):
        fig.savefig(directory / f"{STEM}.{suffix}", dpi=300, facecolor="white")
    plt.close(fig)
    if int(counts.sum()) != len(Z):
        raise AssertionError("Rendered particle count differs from the xIV NPZ")
    return len(present)


def _check_vtk(path: Path, Z: np.ndarray) -> None:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("POINTS "):
                count = int(line.split()[1])
                points = np.asarray([[float(v) for v in next(handle).split()]
                                     for _ in range(count)])
                if count != len(Z) or not np.allclose(points, Z, rtol=0, atol=1e-5):
                    raise AssertionError("VTK coordinates differ from the xIV NPZ")
                return
    raise ValueError(f"No VTK POINTS found in {path}")


def _update_catalog(path: Path, donor: str, spec: SourceSpec, h5ad: Path, labels: list[str],
                    colors: list[str], palette_source: str, section: str,
                    entry: dict[str, object]) -> None:
    try:
        source = str(h5ad.resolve().relative_to(REPO_ROOT))
    except ValueError:
        source = str(h5ad.resolve())
    shared = {
        "schema_version": 1,
        "donor_id": donor,
        "region": spec.region,
        "hemisphere": None,
        "source_h5ad": source,
        "coordinates": f"{spec.coordinate_key} (2D); xIV z=0 is padding, not anatomical depth",
        "feature": spec.feature_field,
        "subclass_colors": dict(zip(labels, colors)),
        "palette_source": palette_source,
    }
    if spec.donor is not None:
        shared["donor_id_source"] = DONOR_MAPPING_URL
    catalog = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
        **shared, "sections": {}
    }
    for key, expected in shared.items():
        if catalog.get(key) != expected:
            raise ValueError(f"Existing catalog has a different {key}: {path}")
    catalog["sections"][section] = entry
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     prefix=".sections-", suffix=".json", delete=False) as handle:
        temp_path = Path(handle.name)
        json.dump(catalog, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    os.replace(temp_path, path)


def make_figure(h5ad: Path, donor: str, section: str, output_root: Path,
                chunk_rows: int, overwrite: bool, pdf: bool = False,
                save_npz: bool = True) -> Path:
    spec = _source_spec(h5ad)
    if spec.donor is not None and (donor, section) != (spec.donor, spec.section):
        raise ValueError(f"Source {h5ad.name} belongs to {spec.donor}/{spec.section}")
    directory = output_root / _directory_name(donor) / spec.region / _directory_name(section)
    catalog_path = directory.parent / "sections.json"
    files = [directory / f"{STEM}.{ext}" for ext in ("npz", "json", "vtk", "png", "pdf")]
    files.append(directory / "full_section_figure.json")
    existing_section = (json.loads(catalog_path.read_text(encoding="utf-8"))
                        .get("sections", {}).get(section) if catalog_path.exists() else None)
    if not overwrite and (any(path.exists() for path in files) or existing_section):
        raise FileExistsError(f"Derivative exists in {directory}; use --overwrite")
    directory.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="seaad_merfish_") as staging:
        npz, _, meta = export_section_for_xiv(
            h5ad_path=h5ad, section=section,
            donor=donor if spec.section_field is not None else None,
            output=Path(staging) / "particles.npz",
            section_field=spec.section_field, donor_field="Donor ID",
            coordinate_key=spec.coordinate_key, selection="all",
            selection_field="Depth from pia", feature_field=spec.feature_field, genes=None,
            missing_feature="keep", row_normalize_gene_features=False,
            weight_field=None, coordinate_scale=1.0, source_units="unknown",
            output_units="unknown", center="none", dimensions=3, z_value=0.0,
            dtype_name="float32", chunk_rows=chunk_rows, compress=True,
            overwrite=False,
        )
        labels = list(meta["features"]["feature_labels"])
        colors, palette_source = _subclass_colors(h5ad, labels, spec.feature_field)

        for relative in ("src/config_setup", "src/preprocess", "vendor/xIV-LDDMM-Particle"):
            sys.path.insert(0, str(REPO_ROOT / relative))
        from xmodmap.io.getInput import readParticleApproximation
        from xmodmap.io.getOutput import writeParticleVTK
        from xmodmap_compat import write_particle_vtk_xyz

        Z_t, nu_t = readParticleApproximation(str(npz))
        Z, nu = Z_t.detach().cpu().numpy(), nu_t.detach().cpu().numpy()
        if nu.shape[1] != len(labels) or len(colors) != len(labels):
            raise AssertionError("Subclass labels and colors differ from particle channels")
        vtk = directory / f"{STEM}.vtk"
        write_particle_vtk_xyz(SimpleNamespace(write_particle_vtk=writeParticleVTK),
                               Z, nu, vtk, condense=True, coordinate_convention="xyz")
        _check_vtk(vtk, Z)
        n_subclasses = _render(Z, nu, labels, colors, donor, section, spec, directory, pdf)
        excluded = meta["coordinates"]["n_cells_dropped_nonfinite_coordinates"]
        if len(Z) + excluded != meta["selection"]["n_cells_matching_section"]:
            raise AssertionError("Some section cells were not accounted for")
        if save_npz:
            saved_npz = directory / f"{STEM}.npz"
            shutil.copy2(npz, saved_npz)
            with np.load(saved_npz) as archive:
                if len(archive["Z"]) != len(Z) or len(archive["nu_Z"]) != len(Z):
                    raise AssertionError("Saved NPZ particle count differs from the figure")

    for legacy in (f"{STEM}.json", "full_section_figure.json"):
        (directory / legacy).unlink(missing_ok=True)
    if not pdf:
        (directory / f"{STEM}.pdf").unlink(missing_ok=True)
    if not save_npz:
        (directory / f"{STEM}.npz").unlink(missing_ok=True)

    outputs = {"png": f"{section}/{STEM}.png", "vtk": f"{section}/{STEM}.vtk"}
    if save_npz:
        outputs["npz"] = f"{section}/{STEM}.npz"
    if pdf:
        outputs["pdf"] = f"{section}/{STEM}.pdf"
    _update_catalog(catalog_path, donor, spec, h5ad, labels, colors, palette_source,
                    section, {"n_particles": len(Z),
                              "n_excluded_invalid_coordinates": excluded,
                              "n_subclasses_present": n_subclasses,
                              "files": outputs})
    return directory


def _make_montage(donor_dir: Path, donor: str) -> Path:
    """Tile the existing full-section PNGs; their legends and labels stay intact."""
    from PIL import Image, ImageDraw, ImageFont

    panels: list[Path] = []
    for catalog_path in sorted(donor_dir.glob("*/sections.json")):
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        if catalog["donor_id"] != donor:
            raise ValueError(f"Catalog donor differs from montage donor: {catalog_path}")
        for section in sorted(catalog["sections"]):
            relative = catalog["sections"][section]["files"]["png"]
            panel = catalog_path.parent / relative
            if not panel.is_file():
                raise FileNotFoundError(panel)
            panels.append(panel)
    if not panels:
        raise ValueError(f"No section figures found for donor {donor}")

    width, gap = 1800, 32
    with Image.open(panels[0]) as first:
        height = round(width * first.height / first.width)
    columns = min(2, len(panels))
    rows = (len(panels) + columns - 1) // columns
    header = 90
    canvas = Image.new("RGB", (columns * width + (columns + 1) * gap,
                               rows * height + (rows + 1) * gap + header), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 42)
    except OSError:
        font = ImageFont.load_default()
    draw.text((gap, 22), f"SEA-AD MERFISH · Donor {donor} · {len(panels)} sections",
              fill="#263238", font=font)
    for index, panel in enumerate(panels):
        with Image.open(panel) as source:
            image = source.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
        row, column = divmod(index, columns)
        x = gap + column * (width + gap)
        if row == rows - 1 and len(panels) % columns == 1:
            x += (width + gap) // 2
        canvas.paste(image, (x, header + gap + row * (height + gap)))
    output = donor_dir / "merfish_montage.png"
    canvas.save(output, optimize=True)
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5ad", type=Path, action="append",
                        help="MERFISH H5AD; repeat for more ROIs. Batch default: all local MERFISH H5ADs")
    parser.add_argument("--donor", required=True, help="Exact Donor ID")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--section", help="Exact Specimen Barcode")
    selection.add_argument("--all-sections", action="store_true",
                           help="Make every section for this donor, then a subject montage")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--chunk-rows", type=int, default=250_000)
    parser.add_argument("--pdf", action="store_true", help="Also save PDF section figures")
    parser.add_argument("--no-npz", action="store_true",
                        help="Omit registration-ready xIV NPZ files from the derivatives")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.chunk_rows < 1:
        parser.error("--chunk-rows must be positive")
    sources = args.h5ad or sorted([*RAW_ROOT.glob("SEAAD_*_MERFISH*.h5ad"),
                                   *RAW_ROOT.glob("*/*_mapped.h5ad")])
    if not sources:
        parser.error("No MERFISH H5AD found; pass --h5ad")
    if args.section and len(sources) != 1:
        parser.error("--section requires exactly one --h5ad")
    regions = []
    for source in sources:
        if not source.is_file():
            parser.error(f"H5AD not found: {source}")
        try:
            regions.append(_source_spec(source).region)
        except ValueError as exc:
            parser.error(str(exc))
    if len(regions) != len(set(regions)):
        parser.error("Pass only one MERFISH H5AD per ROI")

    if args.section:
        path = make_figure(sources[0], args.donor, args.section, args.output_root,
                           args.chunk_rows, args.overwrite, args.pdf, not args.no_npz)
        print(f"Wrote MERFISH section derivatives: {path}")
        return 0

    for source in sources:
        spec = _source_spec(source)
        sections = _sections_for_donor(source, args.donor, args.chunk_rows)
        if not sections:
            continue
        region = spec.region
        print(f"{region}: {len(sections)} sections for donor {args.donor}", flush=True)
        for section in sections:
            directory = args.output_root / args.donor / region / section
            catalog_path = directory.parent / "sections.json"
            entry = (json.loads(catalog_path.read_text(encoding="utf-8"))
                     .get("sections", {}).get(section) if catalog_path.exists() else None)
            needed = {"png", "vtk"}
            if not args.no_npz:
                needed.add("npz")
            if args.pdf:
                needed.add("pdf")
            if (not args.overwrite and entry and
                    needed.issubset(entry.get("files", {})) and
                    all((catalog_path.parent / entry["files"][ext]).is_file()
                        for ext in needed)):
                print(f"  Reusing {section}", flush=True)
                continue
            make_figure(source, args.donor, section, args.output_root,
                        args.chunk_rows, args.overwrite, args.pdf, not args.no_npz)
            print(f"  Wrote {section}", flush=True)
    montage = _make_montage(args.output_root / args.donor, args.donor)
    print(f"Wrote subject montage: {montage}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
