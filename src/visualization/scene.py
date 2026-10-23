"""Scene description shared by the static figures and the Neuroglancer exporter.

A scene layer records *what* is drawn (coordinates and feature values), *where* (a
named coordinate frame and its units), and *how it got there* (a status and a
provenance string).  Loaders read existing project derivatives only; nothing here
optimises, registers or computes new statistics.

Status vocabulary
=================
``measured``     provider/instrument observation in its native frame
``transformed``  a measured observation moved by a recorded, fitted transform
``derived``      deterministic aggregation of measured observations (binning, means)
``inferred``     the output of an optimisation (xIV deformation): an estimate

Frames (all lengths in mm unless a name says otherwise)
=======================================================
``merfish_section_mm``          SEA-AD ``X_spatial_raw`` x 1e-3 (provider um).  The provider
                                does not record anatomical orientation for these axes.
``merfish_layer_polygon_um``    frame of ``uns/layer_polygons``: (-y_raw, x_raw) in um,
                                established by cell-label agreement (``merfish_layer_polygons``).
``qnp_slide_mm:<slide>``        x = column * px, y = row * px (y down), origin at the
                                top-left baseline pixel (``qnp_particles``).
``qnp_local_mm``                the NeuN slide's ``qnp_slide_mm`` (within-QNP reference).
``atlas_plane_mm:<plane_id>``   in-plane (u, v) mm of a resliced AHRAv2 plane;
                                world = origin + u * u_axis + v * v_axis (MNI ICBM 2009b).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
STATUSES = ("measured", "transformed", "derived", "inferred")


# --------------------------------------------------------------------------------------
# Transforms and layers
# --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Affine2D:
    """``target = M @ [x, y, 1]`` between two named frames (same length units)."""

    matrix: np.ndarray
    source: str
    target: str
    kind: str = "affine"
    status: str = "accepted"
    provenance: str = ""

    def apply(self, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, dtype=np.float64)[:, :2]
        return xy @ self.matrix[:2, :2].T + self.matrix[:2, 2]

    def inverse(self) -> "Affine2D":
        return Affine2D(np.linalg.inv(self.matrix), self.target, self.source, self.kind,
                        self.status, f"inverse of [{self.provenance}]")

    def then(self, other: "Affine2D") -> "Affine2D":
        """``other`` applied after ``self``; frames must chain."""

        if self.target != other.source:
            raise ValueError(f"Cannot chain {self.source}->{self.target} with {other.source}->{other.target}")
        return Affine2D(other.matrix @ self.matrix, self.source, other.target, f"{self.kind} then {other.kind}",
                        f"{self.status}; {other.status}", f"[{self.provenance}] then [{other.provenance}]")

    def in_units(self, factor: float, suffix: str) -> "Affine2D":
        """The same map with both frames rescaled by ``factor`` (e.g. mm -> um = 1000)."""

        S = np.diag([factor, factor, 1.0])
        rename = lambda name: name.replace("_mm", f"_{suffix}")  # noqa: E731
        return Affine2D(S @ self.matrix @ np.linalg.inv(S), rename(self.source), rename(self.target), self.kind,
                        self.status, self.provenance)

    @classmethod
    def identity(cls, frame: str) -> "Affine2D":
        return cls(np.eye(3), frame, frame, "identity", "exact", "identity")

    @classmethod
    def from_similarity(cls, similarity: Any, source: str, target: str, status: str, provenance: str) -> "Affine2D":
        """``Similarity2D`` (moving = A (section - tau)) as section -> moving."""

        A = np.asarray(similarity.matrix, float)
        M = np.eye(3)
        M[:2, :2] = A
        M[:2, 2] = -A @ np.asarray(similarity.tau, float)
        return cls(M, source, target, "similarity (+reflection)", status, provenance)

    def describe(self) -> dict[str, Any]:
        return {"source": self.source, "target": self.target, "kind": self.kind, "status": self.status,
                "provenance": self.provenance, "matrix_3x3": np.round(self.matrix, 12).tolist()}


@dataclass
class Categorical:
    codes: np.ndarray            # int, -1 = missing
    names: tuple[str, ...]
    colors: tuple[str, ...] = ()


@dataclass
class Scalar:
    values: np.ndarray           # float, NaN = missing
    units: str
    description: str = ""


@dataclass
class Points:
    name: str
    frame: str
    xy: np.ndarray               # (N, 2) in frame units
    status: str
    provenance: str
    categories: dict[str, Categorical] = field(default_factory=dict)
    scalars: dict[str, Scalar] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(f"Unknown status {self.status!r}")
        self.xy = np.asarray(self.xy, dtype=np.float64)[:, :2]


@dataclass
class Polygons:
    name: str
    frame: str
    rings: list[np.ndarray]      # each (K, 2) in frame units
    labels: np.ndarray           # int index into label_names
    holes: np.ndarray            # bool
    label_names: tuple[str, ...]
    status: str
    provenance: str
    meta: dict[str, Any] = field(default_factory=dict)

    def transformed(self, transform: Affine2D, name: str | None = None) -> "Polygons":
        if transform.source != self.frame:
            raise ValueError(f"{self.name} is in {self.frame}, transform starts at {transform.source}")
        return Polygons(name or self.name, transform.target, [transform.apply(r) for r in self.rings],
                        self.labels.copy(), self.holes.copy(), self.label_names, "transformed",
                        f"{self.provenance}; moved by {transform.kind} ({transform.provenance})", dict(self.meta))


@dataclass
class Raster:
    """A bounded in-memory raster; pixel (row, col) has its centre at ``origin + (col, row) * px``."""

    name: str
    frame: str
    array: np.ndarray
    origin: tuple[float, float]  # (x, y) of pixel (0, 0) centre
    px: tuple[float, float]      # (x, y) pixel size
    status: str
    provenance: str
    units: str = ""

    @property
    def extent(self) -> list[float]:
        """Matplotlib ``imshow`` extent (left, right, bottom, top) for y-down frames."""

        h, w = self.array.shape[:2]
        x0, y0 = self.origin[0] - 0.5 * self.px[0], self.origin[1] - 0.5 * self.px[1]
        return [x0, x0 + w * self.px[0], y0 + h * self.px[1], y0]


def _json_scalar(value: Any) -> Any:
    return value.item() if hasattr(value, "item") else value


# --------------------------------------------------------------------------------------
# Config helpers
# --------------------------------------------------------------------------------------

def load_project_config(path: str | Path = "configs/project_config.yaml") -> dict:
    from config_setup.project_config import load_config

    return load_config(PROJECT / path if not Path(path).is_absolute() else path)


def resolve(config: Mapping, value: str | Path) -> Path:
    from config_setup.project_config import resolve_path

    return resolve_path(config, value)


# --------------------------------------------------------------------------------------
# MERFISH
# --------------------------------------------------------------------------------------

CLASS3 = {  # three-way fold of the seven project classes (dataviz all-pairs cap = 3 hues)
    "neurons": ("excitatory", "inhibitory"),
    "oligodendrocyte lineage": ("oligodendrocyte", "opc"),
    "other non-neuronal": ("astrocyte", "microglia_pvm", "vascular"),
}
MERFISH_LAYERS = ("L1", "L2/3", "L4", "L5", "L6")


def load_merfish_cells(h5ad: str | Path, barcode: str, groups: Mapping[str, Sequence[str]],
                       genes: Sequence[str] = (), chunk_rows: int = 200_000, check_rows: int = 300,
                       gene_pools: Mapping[str, tuple[str, str]] | None = None) -> Points:
    """One MERFISH section as measured cell points in ``merfish_section_mm``.

    Reads only the section's rows (the repo's chunked h5ad helpers).  In the 2024-12-11
    release ``X`` holds raw spot counts although ``uns/X_normalization`` says
    "ln(Spots per 10K +1)": rows are integer and sum to ``Number of spots``.  That is
    checked on the first ``check_rows`` cells, and genes are exposed both as raw counts
    (``count:G``) and as ``lognorm:G`` = ln(1 + 1e4 * count / spots in the cell).
    ``gene_pools`` maps a name to (var-name prefix, "sum" | "mean"): the matching columns
    are pooled per cell (e.g. the mean of the blank control probes = one probe-equivalent).
    """

    import h5py

    from omics.visualize_seaad_merfish_section import (
        _extract_gene_values, _extract_obs_rows, _extract_obsm_rows, _selected_indices, _var_names)

    h5ad = Path(h5ad)
    with h5py.File(h5ad, "r") as f:
        obs = f["obs"]
        n_obs = f["obsm"]["X_spatial_raw"].shape[0]
        rows = _selected_indices(obs, "Specimen Barcode", str(barcode), chunk_rows)
        if not len(rows):
            raise ValueError(f"No cells with Specimen Barcode {barcode}")
        xy_um = _extract_obsm_rows(f["obsm"], "X_spatial_raw", rows, n_obs, chunk_rows)[:, :2].astype(np.float64)
        text = lambda name: np.array([v if isinstance(v, str) else "" for v in  # noqa: E731
                                      _extract_obs_rows(obs, name, rows, n_obs, chunk_rows)], dtype=object)
        subclass, layer = text("Subclass"), text("Layer annotation")
        sections, donors = set(text("Section")), set(text("Donor ID"))
        depth = np.asarray(_extract_obs_rows(obs, "Depth from pia", rows, n_obs, chunk_rows), dtype=np.float64)
        spots = np.asarray(_extract_obs_rows(obs, "Number of spots", rows, n_obs, chunk_rows), dtype=np.float64)
        normalization = f["uns"]["X_normalization"][()]
        normalization = normalization.decode() if isinstance(normalization, bytes) else str(normalization)
        sample = np.vstack([f["X"][int(i)] for i in rows[:check_rows]])
        x_check = {"rows_checked": int(len(sample)), "integer_valued": bool(np.all(sample == np.round(sample))),
                   "row_sum_equals_number_of_spots": bool(np.allclose(sample.sum(1), spots[:check_rows])),
                   "uns_X_normalization": normalization}
        if not (x_check["integer_valued"] and x_check["row_sum_equals_number_of_spots"]):
            raise ValueError(f"X is not raw counts as expected: {x_check}")
        var = list(_var_names(f))
        gene_values, x_sums = {}, {}
        for gene in genes:
            if gene not in var:
                raise KeyError(f"{gene} is not in the MERFISH panel")
            gene_values[gene] = np.asarray(_extract_gene_values(f, var.index(gene), rows, n_obs, chunk_rows), float)
        for name, (prefix, how) in (gene_pools or {}).items():
            members = [i for i, v in enumerate(var) if v.startswith(prefix)]
            if not members or how not in ("sum", "mean"):
                raise KeyError(f"Bad pool {name!r}: prefix {prefix!r} ({len(members)} columns), {how!r}")
            total = sum(np.asarray(_extract_gene_values(f, i, rows, n_obs, chunk_rows), float) for i in members)
            gene_values[name] = total / len(members) if how == "mean" else total
            x_sums[name] = {"prefix": prefix, "n_columns": len(members), "pool": how}

    if len(sections) != 1 or len(donors) != 1:
        raise ValueError(f"Barcode {barcode} maps to sections {sections} / donors {donors}")
    class_names = tuple(groups)
    member = {label: i for i, (_, labels) in enumerate(groups.items()) for label in labels}
    class_codes = np.array([member.get(s, -1) for s in subclass], dtype=np.int64)
    class3_names = tuple(CLASS3)
    class3_of = {c: i for i, (_, cs) in enumerate(CLASS3.items()) for c in cs}
    class3_codes = np.array([class3_of[class_names[c]] if c >= 0 else -1 for c in class_codes], dtype=np.int64)
    sub_names = tuple(sorted(set(subclass) - {""}))
    layer_codes = np.array([MERFISH_LAYERS.index(v) if v in MERFISH_LAYERS else -1 for v in layer], dtype=np.int64)

    points = Points(
        name=f"MERFISH cells {barcode}", frame="merfish_section_mm", xy=xy_um * 1e-3, status="measured",
        provenance=f"{h5ad.name} obsm/X_spatial_raw (um) x 1e-3; rows with Specimen Barcode {barcode}",
        categories={
            "subclass": Categorical(np.array([sub_names.index(s) if s else -1 for s in subclass]), sub_names),
            "class": Categorical(class_codes, class_names),
            "class3": Categorical(class3_codes, class3_names),
            "layer_annotation": Categorical(layer_codes, MERFISH_LAYERS),
        },
        scalars={"depth_from_pia_um": Scalar(depth, "um", "provider 'Depth from pia' (strip cells only)"),
                 "spots": Scalar(spots, "spots", "provider 'Number of spots' (all probes incl. blanks)"),
                 **{f"count:{g}": Scalar(v, "spots", f"{g} raw spot count (X)") for g, v in gene_values.items()},
                 **{f"lognorm:{g}": Scalar(np.log1p(1e4 * v / np.maximum(spots, 1)), "ln(1 + spots per 10K)",
                                           f"{g}: ln(1 + 1e4 x count / spots in cell), computed here")
                    for g, v in gene_values.items()}},
        meta={"barcode": str(barcode), "section": sections.pop(), "donor": donors.pop(), "h5ad_rows": rows,
              "n_cells": int(len(rows)), "x_check": x_check, "gene_pools": x_sums},
    )
    return points


def merfish_layer_polygons(h5ad: str | Path, cells: Points, min_agreement: float = 0.9) -> tuple[Polygons, Affine2D, dict]:
    """Provider layer polygons for the cells' section, and their transform to ``merfish_section_mm``.

    The polygon frame is not documented.  It is identified here, not assumed: every
    axis permutation/sign is tried and the one under which cell ``Layer annotation``
    labels agree with polygon membership is kept (the others give ~0).  Each polygon is
    named by the majority label of the cells inside it.  Fails below ``min_agreement``.
    """

    import h5py
    from matplotlib.path import Path as MplPath

    section = cells.meta["section"]
    donor = cells.meta["donor"]
    with h5py.File(h5ad, "r") as f:
        group = f["uns"]["layer_polygons"]
        key = section if section in group else f"{donor}.{section.split('.')[-1]}"
        if key not in group:
            raise KeyError(f"No layer polygons for {section} (tried {key})")
        raw = group[key][()]
    geometry = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
    rings_um = [np.asarray(g["coordinates"], float) for g in geometry["geometries"] if g["type"] == "Polygon"]

    layer = cells.categories["layer_annotation"]
    annotated = layer.codes >= 0
    xy_um = cells.xy[annotated] * 1e3
    candidates = {}
    for sx in (1, -1):
        for sy in (1, -1):
            for swap in (False, True):
                M = np.array([[0, sx], [sy, 0]]) if swap else np.array([[sx, 0], [0, sy]])
                candidates[(sx, sy, swap)] = M  # polygon frame = M @ raw
    best, agreements = None, {}
    for key_, M in candidates.items():
        P = xy_um @ M.T
        inside = np.array([MplPath(r).contains_points(P) for r in rings_um]).T
        votes = []
        for k in range(len(rings_um)):
            labels = layer.codes[annotated][inside[:, k]]
            votes.append(np.bincount(labels, minlength=len(layer.names)).argmax() if len(labels) else -1)
        assigned = np.where(inside.any(1), np.asarray(votes)[inside.argmax(1)], -1)
        agreement = float(np.mean(assigned == layer.codes[annotated]))
        agreements[str(M.tolist())] = agreement
        if best is None or agreement > best[1]:
            best = (key_, agreement, M, votes, float(inside.any(1).mean()))
    (sx, sy, swap), agreement, M, votes, coverage = best
    if agreement < min_agreement or sorted(votes) != list(range(len(rings_um))):
        raise ValueError(f"Layer polygons do not match cell labels (agreement {agreement:.3f}, votes {votes})")
    polygons = Polygons(
        name=f"provider layer polygons {key}", frame="merfish_layer_polygon_um", rings=rings_um,
        labels=np.asarray(votes), holes=np.zeros(len(rings_um), bool), label_names=layer.names,
        status="measured", provenance=f"{Path(h5ad).name} uns/layer_polygons['{key}']")
    to_section = np.eye(3)
    to_section[:2, :2] = np.linalg.inv(M) * 1e-3  # raw_um = M^-1 polygon_um; mm = um * 1e-3
    transform = Affine2D(to_section, "merfish_layer_polygon_um", "merfish_section_mm", "axis permutation + um->mm",
                         "exact (identified by label agreement)",
                         f"polygon = {M.tolist()} @ X_spatial_raw; identified from {annotated.sum()} labelled cells")
    check = {"polygon_key": key, "frame_matrix_polygon_from_raw": M.tolist(), "label_agreement": agreement,
             "annotated_cells_inside_any_polygon": coverage, "annotated_cells": int(annotated.sum()),
             "polygon_layer_names": [layer.names[v] for v in votes],
             "agreement_by_candidate_frame": agreements,
             "runner_up_agreement": sorted(agreements.values())[-2]}
    return polygons, transform, check


def bin_merfish(cells: Points, bin_mm: float, gene_keys: Sequence[str] = ()) -> Points:
    """Scale-space representation: the project's ``bin_particles_2d`` on cell points.

    Mass = cell count per class (the same aggregation as ``prepare_atlas_plane_and_section``,
    before its area conversion).  Gene scalars are per-bin means over cells.  This is a
    deterministic binning, *not* a varifold-optimised compression.
    """

    from preprocess.particle_utils import bin_particles_2d

    klass = cells.categories["class"]
    keep = klass.codes >= 0
    onehot = np.eye(len(klass.names))[klass.codes[keep]]
    gene_arrays = [np.nan_to_num(cells.scalars[g].values[keep])[:, None] for g in gene_keys]
    X, binned, counts = bin_particles_2d(cells.xy[keep], [onehot, *gene_arrays], bin_mm)
    nu = binned[0]
    scalars = {"cells_per_bin": Scalar(counts.astype(float), "cells"),
               "cells_per_mm2": Scalar(counts / bin_mm**2, "cells / mm^2")}
    for g, arr in zip(gene_keys, binned[1:]):
        scalars[g] = Scalar(arr[:, 0] / counts, cells.scalars[g].units, f"bin mean of {g}")
    for i, name in enumerate(klass.names):
        scalars[f"fraction:{name}"] = Scalar(nu[:, i] / counts, "fraction of cells")
    grid = np.floor(cells.xy[keep] / bin_mm).astype(np.int64)
    _, first = np.unique(grid, axis=0, return_index=True)
    return Points(
        name=f"{cells.name} binned {bin_mm * 1000:g} um", frame=cells.frame, xy=X[:, :2], status="derived",
        provenance=f"preprocess.particle_utils.bin_particles_2d(bin_mm={bin_mm}) of {cells.name}",
        scalars=scalars, meta={"bin_mm": bin_mm, "nu_class": nu, "class_names": klass.names,
                               "grid_cells": grid[first], "n_cells": int(keep.sum())})


def load_merfish_xiv(fit_dir: str | Path) -> dict[str, Any]:
    """Atlas-plane particles of the MERFISH pilot fit, placed in ``merfish_section_mm``.

    ``rigid``: the initial barycentre placement (source - source_centre + target_centre,
    M = I); ``deformed``: xIV's ``atlas_deformationSummary`` D + target_centre.  The fit
    targeted the 50-bin depth strip only and is under-determined (MTG-only label).
    """

    import torch

    fit_dir = Path(fit_dir)
    chain = json.loads((fit_dir / "transform_chain.json").read_text())
    manifest = json.loads((fit_dir / "input_manifest.json").read_text())
    if np.abs(np.asarray(chain["initial_transform_matrix_M"]) - np.eye(2)).max() > 0:
        raise NotImplementedError("Only M = I initialisations are supported")
    with np.load(manifest["source_particles"]) as archive:
        plane_xy = archive["X"][:, :2].astype(np.float64)
    summary = torch.load(fit_dir / "atlas_deformationSummary.pt", map_location="cpu", weights_only=False)
    D = summary["D"].double().numpy()
    src_c, tgt_c = np.asarray(chain["source_center_plane_mm"]), np.asarray(chain["target_center_section_mm"])
    rigid = Points("AHRAv2 MTG plane, initial placement", "merfish_section_mm", plane_xy - src_c + tgt_c,
                   "transformed", f"{manifest['source_particles']} centred by source/target barycentres (M = I)")
    deformed = Points("AHRAv2 MTG plane after xIV", "merfish_section_mm", D + tgt_c, "inferred",
                      f"{fit_dir}/atlas_deformationSummary.pt D + target_center_section_mm",
                      scalars={"mass_ratio_proxy": Scalar(summary["mass_ratio_proxy"].double().numpy(), "ratio")})
    return {"rigid": rigid, "deformed": deformed, "chain": chain, "manifest": manifest,
            "scores": json.loads((fit_dir / "scores.json").read_text()) if (fit_dir / "scores.json").exists() else {},
            "plane_xy": plane_xy}


# --------------------------------------------------------------------------------------
# QNP
# --------------------------------------------------------------------------------------

QNP_LAYERS = ("L1", "L2", "L3", "L4", "L5-6")
QNP_TABLE_LAYERS = {"Layer1": "L1", "Layer2": "L2", "Layer3": "L3", "Layer4": "L4", "Layer5-6": "L5-6"}


def qnp_job(config: Mapping, job: str = "H20.33.034_MTG") -> dict:
    spec = dict(config["qnp_section_inputs"][job])
    slide_dir = resolve(config, spec["slide_dir"])
    slides = {}
    for path in sorted(slide_dir.glob("*.svs")):
        stain = path.stem.split("-")[-1].upper()
        slides[stain] = path
    spec.update(_name=job, _slides=slides, _particle_dir=resolve(config, spec["particle_dir"]),
                _aligned_dir=resolve(config, spec["alignment"]["output_dir"]),
                _scan_dir=resolve(config, spec["atlas_scan"]["output_dir"]))
    return spec


def slide_frame(path: Path) -> str:
    return f"qnp_slide_mm:{Path(path).stem}"


def svs_geometry(path: str | Path) -> dict[str, Any]:
    """Aperio level shapes and baseline MPP without reading pixels."""

    import tifffile

    from qnp_particles import _aperio_header

    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        header = _aperio_header(tif.pages[0].description)
        shapes = [tuple(level.shape[:2]) for level in series.levels]
    return {"mpp_um": float(header["MPP"]), "level_shapes": shapes, "app_mag": header.get("AppMag")}


def load_qnp_transforms(job: Mapping) -> dict[str, Affine2D]:
    """Within-QNP rigid transforms (slide mm -> qnp_local_mm), with their accepted/unreliable status."""

    path = job["_aligned_dir"] / "transforms.json"
    record = json.loads(path.read_text())
    reference = record["reference_stain"]
    out = {reference: Affine2D(np.eye(3), slide_frame(job["_slides"][reference]), "qnp_local_mm", "identity",
                               "reference", f"{path.name}: reference stain")}
    graph = json.loads((job["_scan_dir"] / "transform_graph.json").read_text())
    status = {e["from"].split(":")[-1]: e.get("status", "") for e in graph["edges"] if e["to"] == "qnp_local_mm"}
    for stain, edge in record["edges"].items():
        slide = edge["slide"]
        out[stain] = Affine2D(np.asarray(edge["slide_mm_to_local_mm"], float), f"qnp_slide_mm:{slide}", "qnp_local_mm",
                              "2-D rigid (+reflection)", status.get(slide, "accepted"),
                              f"{path.relative_to(PROJECT)} edges.{stain} (score {edge['score_fine']:.3f})")
    return out


def load_halo_polygons(annotation_path: str | Path, mpp_um: float) -> Polygons:
    """HALO layer polygons (baseline pixels) in the slide's mm frame."""

    from qnp_particles import read_halo_layers

    annotation_path = Path(annotation_path)
    layers = read_halo_layers(annotation_path)
    rings, labels, holes = [], [], []
    for name in QNP_LAYERS:
        for xy, hole in layers.get(name, []):
            rings.append(xy * mpp_um * 1e-3)
            labels.append(QNP_LAYERS.index(name))
            holes.append(hole)
    return Polygons(f"HALO layers {annotation_path.stem}", f"qnp_slide_mm:{annotation_path.stem}", rings,
                    np.asarray(labels), np.asarray(holes, bool), QNP_LAYERS, "measured",
                    f"{annotation_path.name} (HALO XML, baseline px x {mpp_um} um)")


def load_qnp_map(job: Mapping, stain: str) -> dict[str, Raster]:
    """The 32 um overlay maps of one slide (``prepare_qnp_section_particles``), slide frame."""

    from prepare_qnp_section_particles import MAP_REDUCTION

    slide = job["_slides"][stain]
    path = job["_particle_dir"] / f"{slide.stem}_maps_32um.npz"
    maps = dict(np.load(path))
    py, px = (float(v) for v in maps["px_mm"])
    # Map pixel k averages level pixels 4k..4k+3; level pixel j is centred at j * px_level.
    centre = (MAP_REDUCTION - 1) / 2 / MAP_REDUCTION
    origin = (centre * px, centre * py)
    out = {}
    for key, units in (("rgb", "RGB (smallest Aperio level, block mean)"), ("p_wm", "p(white matter)"),
                       ("mask", "tissue"), ("layers", "HALO layer index (0 = none)")):
        if key in maps:
            out[key] = Raster(f"{slide.stem} {key}", slide_frame(slide), maps[key], origin, (px, py),
                              "derived" if key != "rgb" else "measured", f"{path.relative_to(PROJECT)}:{key}", units)
    return out


def load_qnp_particles(path: str | Path, frame: str) -> Points:
    """Tissue-class particles (gray/white area masses, mm^2) as derived points."""

    path = Path(path)
    with np.load(path) as archive:
        X, nu = archive["X"][:, :2].astype(np.float64), archive["nu_Sub"].astype(np.float64)
        meta = json.loads(str(archive["metadata_json"]))
    area = nu.sum(1)
    return Points(path.stem, frame, X, "derived", str(path.relative_to(PROJECT)),
                  scalars={"area_mm2": Scalar(area, "mm^2"),
                           "white_fraction": Scalar(nu[:, 1] / np.maximum(area, 1e-12), "fraction of area")},
                  meta={"bin_mm": float(meta.get("particles", {}).get("bin_mm_actual", np.nan)), "metadata": meta})


def load_ribbon_points(job: Mapping) -> Points:
    """NeuN HALO ribbon points with approximate depth from pia (QC coordinate)."""

    path = resolve_ribbon(job)
    with np.load(path) as archive:
        X, depth, layer = archive["X"][:, :2], archive["depth_from_pia_um"], archive["layer_index"]
        frame = str(archive["coordinate_space"])
    if frame != "qnp_slide_mm":
        raise ValueError(f"Unexpected ribbon frame {frame}")
    return Points("NeuN HALO ribbon points", "qnp_local_mm", X, "derived", str(path.relative_to(PROJECT)),
                  categories={"layer": Categorical(layer.astype(int) - 1, QNP_LAYERS)},
                  scalars={"depth_from_pia_um": Scalar(depth.astype(float), "um",
                                                       "distance to the layer-1 outer edge (QC, not laminar depth)")})


def resolve_ribbon(job: Mapping) -> Path:
    return PROJECT / job["atlas_scan"]["ribbon_points"]


def load_plane_hypothesis(job: Mapping, plane_id: str, fit_dir: str | Path | None) -> dict[str, Any]:
    """One atlas-plane pose hypothesis: local -> plane similarity and the ribbon in the plane."""

    from section_alignment import Similarity2D

    export = json.loads((job["_scan_dir"] / f"export_{plane_id}.json").read_text())
    similarity = Similarity2D.from_dict(export["similarity_qnp_local_to_plane"])
    frame = f"atlas_plane_mm:{plane_id}"
    to_plane = Affine2D.from_similarity(similarity, "qnp_local_mm", frame, "hypothesis (scan confidence set)",
                                        f"export_{plane_id}.json similarity_qnp_local_to_plane")
    out = {"plane_id": plane_id, "plane": export["plane"], "export": export, "similarity": similarity,
           "local_to_plane": to_plane, "frame": frame}
    with np.load(export["paths"]["qc_points_npz"]) as archive:
        out["ribbon_rigid_plane"] = archive["X"][:, :2].astype(np.float64)
    if fit_dir is not None:
        fit_dir = Path(fit_dir)
        with np.load(fit_dir / "qc_points_pulled_back.npz") as archive:
            out["ribbon_xiv_plane"] = archive["X_plane"][:, :2].astype(np.float64)
            out["ribbon_depth_um"] = archive["depth"].astype(np.float64)
        out["evaluation"] = json.loads((fit_dir / "anatomical_evaluation.json").read_text())
        out["fit_dir"] = fit_dir
    return out


def atlas_plane_classes(config: Mapping, job: Mapping, plane: Mapping, spacing_mm: float = 0.25) -> dict[str, Any]:
    """Resliced AHRAv2 labels on one plane, grouped into the scan's fine classes, plus the T1."""

    import nibabel as nib
    from scipy import ndimage as ndi

    from scan_atlas_planes_for_qnp import FINE_ORDER_TAIL, atlas_setup
    from section_alignment import reslice_labels

    scan = job["atlas_scan"]
    atlas = atlas_setup(config, scan)
    origin, u, v = (np.asarray(plane[k], float) for k in ("origin_mm", "u_axis", "v_axis"))
    grid, axis = reslice_labels(atlas["labels"], atlas["affine"], origin, u, v, float(scan["half_width_mm"]), spacing_mm)
    names = list(scan["fine_regions"]) + list(FINE_ORDER_TAIL)
    classes = np.full(grid.shape, -1, dtype=np.int16)
    for index, name in enumerate(names):
        table = atlas["fine"][name]
        classes[(grid < len(table)) & table[np.clip(grid, 0, len(table) - 1)]] = index
    template = nib.load(str(resolve(config, scan["template_nifti"])))
    aa, bb = np.meshgrid(axis, axis, indexing="xy")
    world = origin + aa[..., None] * u + bb[..., None] * v
    ijk = (world.reshape(-1, 3) - template.affine[:3, 3]) @ np.linalg.inv(template.affine[:3, :3]).T
    t1 = ndi.map_coordinates(np.asanyarray(template.dataobj).astype(np.float32), ijk.T, order=1).reshape(aa.shape)
    frame = f"atlas_plane_mm:{plane['plane_id']}"
    prov = f"section_alignment.reslice_labels of {scan['annotation_nifti']} at {plane['plane_id']} ({spacing_mm} mm)"
    return {"classes": Raster("AHRAv2 classes", frame, classes, (axis[0], axis[0]), (spacing_mm, spacing_mm),
                              "measured", prov, "class index"),
            "labels": Raster("AHRAv2 labels", frame, grid, (axis[0], axis[0]), (spacing_mm, spacing_mm), "measured", prov),
            "t1": Raster("MNI T1", frame, t1, (axis[0], axis[0]), (spacing_mm, spacing_mm), "measured",
                         f"{scan['template_nifti']} resliced (linear)"),
            "class_names": names, "axis": axis, "names": atlas["names"]}


def load_qnp_layer_measures(config: Mapping, donor: str, measures: Sequence[str]) -> dict[str, dict[str, float]]:
    """Provider per-layer QNP values (June-2026 release, ``MTG`` tab): {measure: {layer: value}}."""

    from omics import xlsx

    path = resolve(config, "data/raw/sea-ad/JGN327NUXRZSHEV88TN_6fdd18ff3e0e4699d43c5ad89a23cf0462a42853d1b8970c589bdbf01f0fedcf_specimen_metadata/6a4579d1d67150b0e69ad044_sea-ad-quantitative-neuropathology-063026.xlsx")
    rows = [r for r in xlsx.read_records(path, "MTG") if r["Donor ID"] == donor]
    out: dict[str, dict[str, float]] = {m: {} for m in measures}
    for row in rows:
        layer = QNP_TABLE_LAYERS.get(row["analysis region"])
        if layer is None:
            continue
        for m in measures:
            value = row.get(m, "")
            out[m][layer] = float(value) if value not in ("", None) else float("nan")
    for m, values in out.items():
        if set(values) != set(QNP_LAYERS):
            raise ValueError(f"{m}: layers {sorted(values)} for {donor}")
    return out


QNP_TABLE_SOURCE = ("SEA-AD QNP release 2026-06-30 (sea-ad-quantitative-neuropathology-063026.xlsx), "
                    "tab 'MTG', one row per donor x analysis region")


def build_mtg_multimodal_3d(config: Mapping, rotations: Mapping[str, float] | None = None,
                            include_donor_sections: bool = False) -> dict[str, Any]:
    """Read existing MTG assets; add display-only slabs in MNI world mm.

    Display z means the coronal plane normal (MNI -Y), not MNI superior Z.
    Optional donor sections inherit the reviewed section's affine without fitting.
    """
    from scan_atlas_planes_for_qnp import atlas_setup
    from skimage.measure import marching_cubes

    rotations = dict(rotations or {})
    job = qnp_job(config)
    atlas = atlas_setup(config, job['atlas_scan'])
    mask = atlas['fine']['MTG'][atlas['labels']]
    # Hemisphere selection in world coordinates, using the atlas's own affine.
    indices = np.argwhere(mask)
    world = indices @ atlas['affine'][:3, :3].T + atlas['affine'][:3, 3]
    mask[tuple(indices[world[:, 0] >= 0].T)] = False
    vertices, faces, _, _ = marching_cubes(mask.astype(np.uint8), level=.5)
    vertices = vertices @ atlas['affine'][:3, :3].T + atlas['affine'][:3, 3]
    plane_id = 'p+000_w+000_y-027.5'
    hyp = load_plane_hypothesis(job, plane_id, None)
    transforms = load_qnp_transforms(job)
    order = ('NEUN', 'LFB', 'AT', 'ASYN', 'I6', 'GFAP')
    if set(rotations) - set(order):
        raise ValueError('Unknown QNP stain in display rotations')
    sources = [PROJECT / 'configs/project_config.yaml',
               resolve(config, job['atlas_scan']['annotation_nifti']),
               resolve(config, job['atlas_scan']['terminology_csv']),
               job['_aligned_dir'] / 'transforms.json', job['_scan_dir'] / 'transform_graph.json',
               job['_scan_dir'] / f'export_{plane_id}.json',
               Path(hyp['export']['paths']['qc_points_npz'])]
    layers = []

    def add(name, modality, xy, plane, offset, color, meta):
        u, v, n = (np.asarray(plane[k], float) for k in ('u_axis', 'v_axis', 'normal'))
        if not (np.allclose(u, [1,0,0]) and np.allclose(v, [0,0,1]) and np.allclose(n, [0,-1,0])):
            raise ValueError(f'{name}: expected stored coronal plane (X,Z; caudal normal)')
        origin = np.asarray(plane['origin_mm'], float)
        xyz = origin + xy[:, :1] * u + xy[:, 1:2] * v + offset * n
        if not np.isfinite(xyz).all() or not len(xyz):
            raise ValueError(f'Invalid geometry: {name}')
        layers.append(dict(name=name, modality=modality, xyz=xyz, normal=n, color=color,
                           metadata=dict(meta, plane=plane, orientation='coronal; rostrocaudal stacking',
                                         z_offset_mm=float(offset),
                                         thickness_mm=.005, rendered_points=len(xyz),
                                         bounds_world_mm=[xyz.min(0).tolist(), xyz.max(0).tolist()])))

    for i, stain in enumerate(order):
        path = job['_particle_dir'] / f"{job['_slides'][stain].stem}_tissue_classes_500um.npz"
        points = load_qnp_particles(path, slide_frame(job['_slides'][stain]))
        chain = transforms[stain].then(hyp['local_to_plane'])
        xy = chain.apply(points.xy)
        angle = float(rotations.get(stain, 0.))
        if not np.isfinite(angle):
            raise ValueError('Display rotations must be finite')
        pivot = xy.mean(0)
        theta = np.deg2rad(angle)
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        xy = (xy - pivot) @ R.T + pivot
        add(stain, 'QNP', xy, hyp['plane'], (i - (len(order)-1)/2)*.020,
            ('#86b6ef', '#5598e7', '#2a78d6', '#1c5cab', '#0d366b', '#5e53a5')[i],
            dict(source=str(path), transform=chain.describe(), rotation_degrees=angle,
                 rotation_pivot_plane_mm=pivot.tolist(), order=i, source_points=len(xy)))
        sources.append(path)

    placement_path = PROJECT / 'results/analysis/omics_atlas_affine_initialization/initial_affine_candidates.json'
    placement = json.loads(placement_path.read_text())
    selected = placement['selected']
    to_plane = Affine2D(np.asarray(selected['inverse']), 'merfish_section_mm', 'atlas_plane_mm:coronal_y-30')
    manifest_path = PROJECT / 'results/registrations/ahrav2_mtg_plane_to_strip_1194111462/init_r000_f0/input_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    omics_plane = manifest['source_metadata']['plane']
    base = PROJECT / 'data/derivatives/sea-ad/merfish/H20.33.001/MTG'
    inventory_path = base / 'sections.json'
    inventory = json.loads(inventory_path.read_text())
    barcodes = sorted(inventory['sections']) if include_donor_sections else ['1194111462']
    for i, barcode in enumerate(barcodes):
        path = base / barcode / 'full_section_particles.npz'
        # Original full-section export uses Z in provider micrometres (scene validation convention).
        with np.load(path, allow_pickle=False) as archive:
            xy = archive['Z'][:, :2].astype(float) * 1e-3
        count = len(xy)
        xy = xy[::max(1, int(np.ceil(count/12000)))]
        add(barcode, 'OMICS', to_plane.apply(xy), omics_plane, (i-(len(barcodes)-1)/2)*5.,
            ('#eb6834', '#bc4720', '#f29a60')[i],
            dict(source=str(path), transform=to_plane.describe(), selected_placement=selected,
                 placement_status=('reviewed provisional C5' if barcode == '1194111462' else
                                   'unvalidated transfer of 1194111462 display affine'),
                 rotation_degrees=0., order=i, source_points=count, input_units='um'))
        sources.append(path)
    sources += [placement_path, manifest_path, inventory_path]
    return dict(vertices=vertices, faces=faces, layers=layers, sources=sources,
                frame='MNI ICBM2009b symmetric world mm',
                display_z='plane normal = MNI -Y; in-plane axes = MNI X and Z',
                spacing_mm=dict(QNP=.020, OMICS=5.), thickness_mm=.005,
                include_donor_sections=include_donor_sections)


def add_mtg_biological_detail(scene: dict, config: Mapping) -> None:
    """Attach existing RGB maps, measured class3 cells, and provider layer outlines.

    Original scene coordinates and transforms are left intact. No representation
    is recomputed; the existing provider-polygon frame check is reused as-is.
    """
    job = qnp_job(config)
    rgb_sources = []
    h5ad = resolve(config, 'data/raw/sea-ad/merfish/SEAAD_MTG_MERFISH.2024-12-11.h5ad')
    grouping = config['feature_groupings']['seaad_merfish_subclass_classes']['groups']
    production = json.loads((job['_aligned_dir'] / 'transforms.json').read_text())

    def embed(xy, layer):
        meta = layer['metadata']
        plane = meta['plane']
        return (np.asarray(plane['origin_mm']) + xy[:, :1]*np.asarray(plane['u_axis'])
                + xy[:, 1:2]*np.asarray(plane['v_axis'])
                + meta['z_offset_mm']*layer['normal'])

    for layer in scene['layers']:
        meta = layer['metadata']
        transform = Affine2D(np.asarray(meta['transform']['matrix_3x3']),
                             meta['transform']['source'], meta['transform']['target'])
        if layer['modality'] == 'QNP':
            maps = load_qnp_map(job, layer['name'])
            raster, mask = maps['rgb'], maps['mask'].array.astype(bool)
            rows, cols = np.nonzero(mask)
            r0, r1 = max(0, rows.min()-2), min(mask.shape[0], rows.max()+3)
            c0, c1 = max(0, cols.min()-2), min(mask.shape[1], cols.max()+3)
            rgba = np.dstack((raster.array[r0:r1,c0:c1], mask[r0:r1,c0:c1]*255)).astype(np.uint8)
            corners = np.array([[c0-.5,r0-.5],[c1-.5,r0-.5],
                                [c1-.5,r1-.5],[c0-.5,r1-.5]])
            native = corners*np.asarray(raster.px) + np.asarray(raster.origin)
            xy = transform.apply(native)
            angle = np.deg2rad(meta['rotation_degrees'])
            rotation = np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
            pivot = np.asarray(meta['rotation_pivot_plane_mm'])
            xy = (xy-pivot)@rotation.T+pivot
            layer.update(rgb_quad=embed(xy,layer), rgba=rgba)
            path = job['_particle_dir'] / f"{job['_slides'][layer['name']].stem}_maps_32um.npz"
            rgb_sources.append(path)
            edge = production['edges'].get(layer['name'])
            meta['biological_representation'] = dict(
                source=str(path), representation='existing 32um RGB block-mean map from coarsest Aperio level',
                transparency='existing tissue mask; no new segmentation', shape_rgba=list(rgba.shape),
                native_crop_rows_cols=[int(r0),int(r1),int(c0),int(c1)],
                raster_origin_mm=list(raster.origin), raster_pixel_mm=list(raster.px),
                texture_world_corners_mm=layer['rgb_quad'].tolist(),
                production_theta_degrees=edge['transform']['theta_deg'] if edge else 0.,
                production_reflection=edge['transform']['flip'] if edge else False,
                additional_orientation_degrees=0.,
                orientation_note='Saved chain applied once; GFAP alignment remains unreliable' if layer['name']=='GFAP'
                                 else 'Saved chain applied once; no additional 180-degree adjustment')
        else:
            cells = load_merfish_cells(h5ad, layer['name'], grouping)
            try:
                polygons, to_section, check = merfish_layer_polygons(h5ad, cells)
            except (KeyError, ValueError) as exc:
                # Missing or label-incompatible provider polygons are not displayed.
                # Never transfer another section's annotations to fill this gap.
                polygons, to_section, check = None, None, dict(unavailable=str(exc))
            world = embed(transform.apply(cells.xy),layer)
            stride = max(1,int(np.ceil(len(cells.xy)/12000)))
            error = float(np.max(np.abs(world[::stride]-layer['xyz'])))
            if error > 1e-5:
                raise ValueError(f"{layer['name']}: measured cells do not reproduce saved display placement")
            polygon_world = ([embed(transform.apply(to_section.apply(ring)),layer) for ring in polygons.rings]
                             if polygons is not None else [])
            layer.update(cell_xyz=world, cell_class3=cells.categories['class3'].codes,
                         polygon_world=polygon_world,
                         polygon_labels=polygons.labels if polygons is not None else [],
                         layer_names=polygons.label_names if polygons is not None else ())
            meta['biological_representation'] = dict(
                source=str(h5ad), representation='all measured cells, existing CLASS3 grouping; no new bins/clustering',
                cells=len(world), class_names=list(cells.categories['class3'].names),
                class_counts={name:int(np.sum(cells.categories['class3'].codes==i))
                              for i,name in enumerate(cells.categories['class3'].names)},
                polygons=polygons.provenance if polygons is not None else 'unavailable for this section',
                polygon_to_section=to_section.describe() if to_section is not None else None,
                provider_frame_check=check, sampled_coordinate_agreement_mm=error)
    scene['sources'].extend([*rgb_sources,h5ad])
