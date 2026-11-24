"""Anatomical localization: atlas labels, feature channels, planes, landing and the landmark-slab prior.

Everything that turns a labelled 3-D atlas into measures on a 2-D plane, or answers "where is
this point / plane in the atlas", lives here.

Coordinates
===========
Atlas world mm (NIfTI/MNI RAS, +y anterior); the affine maps voxels to world and nothing here
reads voxel-index order.  A plane is ``world = origin + a u + b v`` with in-plane coordinates
(a, b) mm.  Coronal planes (pitch = yaw = 0) have u = +x, v = +z, normal n = -y, and are centred
on the target region's centroid (one hemisphere) at depth y, so a coronal plane's *stacking
coordinate* is its world y.  The AP slab coordinate is s = -y (posterior-increasing).

Label tables and feature channels
=================================
A *table* is a boolean lookup label value -> member.  Gray matter is every non-zero label that
is neither white matter nor dropped (ventricles, choroid plexus): an anatomical correspondence
fixed in advance, not learned.  Three table sets share that rule:

``coarse``      gray_matter, white_matter (the 2-channel structural features);
``fine``        each named region's gray codes, other_gray_matter, white_matter;
``anatomical``  one-hot channels <region>_gray_matter for the chosen gray regions
                (default STG, MTG, ITG), other_gray_matter, white_matter.

A fixed 0/1 *projection* ``M`` [C_out, C_anat] maps anatomical masses to another feature space
(``nu @ M.T``): ``gmwm`` (reproduces ``coarse`` exactly, checked), ``mtg_gm`` and ``anatomical``
(identity).  Matching a projected atlas measure to a target with another feature space is the
registration's job (xIV pi_ST); nothing here is learned.

Plane measures
==============
Labels are resliced nearest-neighbour on a regular in-plane grid, and ``aggregate^2`` pixels
are summed into one particle carrying per-channel areas (mm^2).  A *slab* measure is the mean
over parallel planes.  The *region support* (dissection-region prior) keeps only cells within
``margin_mm`` of named regions.

Landmark-slab prior
===================
``locate_landmark`` returns the first meaningful appearance of an atlas label moving
anterior -> posterior: the most anterior coronal plane whose cross-section is at least
``min_area_mm2`` (one hemisphere); a manual ``y_mm`` replaces the rule.  With landmark s_L,
thickness h, phase delta in [0, h) and offset j::

    S_j(delta) = [s_L - delta + j h,  s_L - delta + (j + 1) h]

j = 0 is the landmark-containing slab, j = -1 / +1 its anterior / posterior neighbours.  Over
all phases S_j covers (s_L + (j-1)h, s_L + (j+1)h]; the search interval is the union over the
chosen offsets, widened by ``extension_mm``.  Three concepts are kept apart: the tissue slab,
the atlas sampling regions inside it (display only), and the section block (never constructed:
its position is what the search estimates).  ROI-specific values come from the caller's config.
"""

from __future__ import annotations

import csv
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import ndimage as ndi

FINE_ORDER_TAIL = ("other_gray_matter", "white_matter")
LANDING_CLASSES = ("outside", "dropped")
DEFAULT_GRAY_REGIONS = ("STG", "MTG", "ITG")
OTHER_GRAY = "other_gray_matter"
WHITE = "white_matter"
GMWM_NAMES = ("gray_matter", "white_matter")
PROJECTIONS = ("gmwm", "mtg_gm", "anatomical")
DEFAULT_PROJECTION = "gmwm"
AP_CONVENTION = ("atlas world mm, NIfTI/MNI RAS (+y anterior); slab coordinate s = -y "
                 "(increases posteriorly, the slabbing direction)")


# --------------------------------------------------------------------------------------
# Label tables
# --------------------------------------------------------------------------------------


def lookup_tables(max_label: int, white_matter: Sequence[int], drop: Sequence[int],
                  fine_regions: Mapping[str, Sequence[int]]) -> tuple[dict, dict]:
    """Boolean tables for coarse (gray, white) and fine (named regions, other gray, white)."""

    size = max_label + 1
    white = np.zeros(size, bool); white[list(white_matter)] = True
    dropped = np.zeros(size, bool); dropped[list(drop)] = True; dropped[0] = True
    gray = ~white & ~dropped
    coarse = {"gray_matter": gray, "white_matter": white}
    fine: dict[str, np.ndarray] = {}
    named = np.zeros(size, bool)
    for name, codes in fine_regions.items():
        table = np.zeros(size, bool); table[list(codes)] = True
        fine[name] = table & gray
        named |= table
    fine["other_gray_matter"] = gray & ~named
    fine["white_matter"] = white
    return coarse, fine


def code_tables(size: int, named_codes: Mapping[str, Sequence[int]]) -> dict[str, np.ndarray]:
    """Plain code-set tables (e.g. QC structures such as the insula); codes >= size are ignored."""

    out = {}
    for name, codes in (named_codes or {}).items():
        table = np.zeros(int(size), bool)
        table[[int(c) for c in codes if int(c) < size]] = True
        out[name] = table
    return out


def channel_name(region: str) -> str:
    return f"{region}_gray_matter"


def anatomical_tables(max_label: int, white_matter: Sequence[int], drop: Sequence[int],
                      fine_regions: Mapping[str, Sequence[int]],
                      gray_regions: Sequence[str] = DEFAULT_GRAY_REGIONS) -> dict[str, np.ndarray]:
    """Ordered one-hot tables of the anatomical channels (built with ``lookup_tables``)."""

    missing = [r for r in gray_regions if r not in fine_regions]
    if missing:
        raise KeyError(f"anatomical gray regions {missing} are not in fine_regions {list(fine_regions)}")
    _, fine = lookup_tables(max_label, white_matter, drop, {r: fine_regions[r] for r in gray_regions})
    tables = {channel_name(r): fine[r] for r in gray_regions}
    tables[OTHER_GRAY] = fine[OTHER_GRAY]
    tables[WHITE] = fine[WHITE]
    check_partition(tables)
    return tables


def check_partition(tables: Mapping[str, np.ndarray]) -> None:
    """Every label value belongs to at most one channel (one-hot)."""

    counts = np.sum([np.asarray(t, bool) for t in tables.values()], axis=0)
    overlap = np.flatnonzero(counts > 1)
    if overlap.size:
        raise ValueError(f"label codes in more than one anatomical channel: {overlap[:20].tolist()}")


def label_channel_index(tables: Mapping[str, np.ndarray]) -> np.ndarray:
    """Label value -> channel index (-1: background or dropped)."""

    index = np.full(len(next(iter(tables.values()))), -1, np.int16)
    for k, table in enumerate(tables.values()):
        index[np.asarray(table, bool)] = k
    return index


def projection_matrix(name: str, channel_names: Sequence[str]) -> tuple[tuple[str, ...], np.ndarray]:
    """``(output_names, M)`` with ``M`` [C_out, C_in] for a named fixed projection."""

    names = list(channel_names)
    if name not in PROJECTIONS:
        raise ValueError(f"unknown feature projection {name!r}; choose one of {PROJECTIONS}")
    if WHITE not in names:
        raise ValueError(f"anatomical channels {names} have no {WHITE!r}")
    if name == "anatomical":
        return tuple(names), np.eye(len(names))
    if name == "gmwm":
        M = np.zeros((2, len(names)))
        M[0, [k for k, n in enumerate(names) if n != WHITE]] = 1.0
        M[1, names.index(WHITE)] = 1.0
        return GMWM_NAMES, M
    target = channel_name("MTG")
    if target not in names:
        raise ValueError(f"projection 'mtg_gm' needs a {target!r} channel; channels are {names}")
    M = np.zeros((1, len(names)))
    M[0, names.index(target)] = 1.0
    return (target,), M


def project(nu: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Projected masses ``nu @ M.T`` (keeps the input dtype)."""

    nu = np.asarray(nu)
    return nu @ np.asarray(matrix, dtype=nu.dtype).T


def projected_tables(tables: Mapping[str, np.ndarray], name: str) -> dict[str, np.ndarray]:
    """Label-level projection: the boolean table of each projected channel."""

    out_names, M = projection_matrix(name, list(tables))
    stacked = np.stack([np.asarray(t, bool) for t in tables.values()])
    return {n: (row[:, None] * stacked).sum(0) > 0 for n, row in zip(out_names, M)}


def check_gmwm_reproduces(tables: Mapping[str, np.ndarray], coarse: Mapping[str, np.ndarray]) -> None:
    """The ``gmwm`` projection equals the original gray/white tables code for code."""

    projected = projected_tables(tables, "gmwm")
    if list(projected) != list(coarse):
        raise ValueError(f"gmwm channels {list(projected)} != original {list(coarse)}")
    for name in coarse:
        differ = np.flatnonzero(projected[name] != np.asarray(coarse[name], bool))
        if differ.size:
            raise ValueError(f"gmwm projection differs from the original {name} table at codes {differ[:20].tolist()}")


def projection_metadata(channel_names: Sequence[str]) -> dict[str, dict]:
    """JSON-ready record of every projection for archive metadata."""

    out = {}
    for name in PROJECTIONS:
        names, M = projection_matrix(name, channel_names)
        out[name] = {"output_names": list(names), "matrix": M.astype(int).tolist()}
    return out


# --------------------------------------------------------------------------------------
# Atlas volume
# --------------------------------------------------------------------------------------


def load_atlas(annotation_nifti, terminology_csv, *, white_matter_codes: Sequence[int], drop_codes: Sequence[int],
               fine_regions: Mapping[str, Mapping[str, Any]], gray_regions: Sequence[str] = DEFAULT_GRAY_REGIONS,
               hemisphere: str = "left") -> dict[str, Any]:
    """Label volume, decoded names and the three table sets; the first fine region is the target.

    ``fine_regions``: {name: {codes, expected_name}}; every code is checked against the
    terminology.  ``centroid`` is the target region's centroid in one hemisphere (the in-plane
    centre of every plane); ``region_extent_y`` its world-y extent.
    """

    import nibabel as nib

    image = nib.load(str(annotation_nifti))
    labels = np.asanyarray(image.dataobj).astype(np.int32)
    affine = np.asarray(image.affine, float)
    with open(terminology_csv, newline="", encoding="utf-8") as stream:
        names = {int(r["annotation_value"]): r["name"] for r in csv.DictReader(stream) if r["annotation_value"].strip()}
    failures = []
    for region, spec in fine_regions.items():
        for code in spec["codes"]:
            if names.get(int(code)) != spec["expected_name"]:
                failures.append(f"{region}: code {code} decodes to {names.get(int(code))!r}, expected {spec['expected_name']!r}")
    if failures:
        raise ValueError("Terminology check failed:\n  " + "\n  ".join(failures))
    codes = {k: v["codes"] for k, v in fine_regions.items()}
    coarse, fine = lookup_tables(int(labels.max()), white_matter_codes, drop_codes, codes)
    # The gmwm projection of the anatomical channels must equal ``coarse`` code for code.
    anatomical = anatomical_tables(int(labels.max()), white_matter_codes, drop_codes, codes, gray_regions)
    check_gmwm_reproduces(anatomical, coarse)
    hemisphere_sign = -1.0 if hemisphere == "left" else 1.0
    ijk = np.argwhere(fine[next(iter(fine_regions))][labels])
    world = ijk @ affine[:3, :3].T + affine[:3, 3]
    world = world[np.sign(world[:, 0]) == hemisphere_sign]
    return {"labels": labels, "affine": affine, "coarse": coarse, "fine": fine, "anatomical": anatomical,
            "names": names, "centroid": world.mean(0),
            "region_extent_y": [float(world[:, 1].min()), float(world[:, 1].max())]}


def region_y_extent(atlas: Mapping) -> tuple[float, float]:
    return tuple(float(v) for v in atlas["region_extent_y"])


# --------------------------------------------------------------------------------------
# Planes and reslicing
# --------------------------------------------------------------------------------------


def plane_basis(pitch_deg: float, yaw_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Orthonormal (u, v, n) of a coronal plane tilted by pitch (about x) then yaw (about z).

    pitch = yaw = 0 gives u = +x, v = +z, n = u x v = -y.
    """

    p, y = np.deg2rad(pitch_deg), np.deg2rad(yaw_deg)
    Rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
    Rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    R = Rz @ Rx
    u, v = R @ np.array([1.0, 0, 0]), R @ np.array([0, 0, 1.0])
    return u, v, np.cross(u, v)


def plane_geometry(atlas: Mapping, pitch: float, yaw: float, y0: float):
    """(origin, u, v, n) of the plane through the target centroid's (x, z) at depth y0."""

    u, v, n = plane_basis(pitch, yaw)
    origin = np.array([atlas["centroid"][0], float(y0), atlas["centroid"][2]])
    return origin, u, v, n


def plane_id(pitch: float, yaw: float, y0: float) -> str:
    return f"p{pitch:+04.0f}_w{yaw:+04.0f}_y{y0:+06.1f}"


def plane_metadata(origin, u, v, n, pitch: float, yaw: float, y0: float) -> dict[str, Any]:
    return {"origin_mm": np.asarray(origin).tolist(), "u_axis": np.asarray(u).tolist(), "v_axis": np.asarray(v).tolist(),
            "normal": np.asarray(n).tolist(), "pitch_deg": pitch, "yaw_deg": yaw, "y0_mm": y0,
            "plane_id": plane_id(pitch, yaw, y0),
            "inverse": "world = origin + plane_X[:, 0] * u_axis + plane_X[:, 1] * v_axis"}


def reslice_labels(labels: np.ndarray, affine: np.ndarray, origin: Sequence[float], u: np.ndarray,
                   v: np.ndarray, half_width_mm: float, spacing_mm: float) -> tuple[np.ndarray, np.ndarray]:
    """Nearest-neighbour label grid on ``origin + a u + b v``; returns (grid[b, a], axis)."""

    axis = np.arange(-half_width_mm, half_width_mm + 1e-9, spacing_mm)
    aa, bb = np.meshgrid(axis, axis, indexing="xy")
    world = np.asarray(origin, float) + aa[..., None] * u + bb[..., None] * v
    ijk = (world.reshape(-1, 3) - affine[:3, 3]) @ np.linalg.inv(affine[:3, :3]).T
    grid = ndi.map_coordinates(labels, ijk.T, order=0, mode="constant", cval=0)
    return grid.reshape(aa.shape), axis


def label_class_masses(grid: np.ndarray, axis: np.ndarray, spacing_mm: float, classes: Mapping[str, np.ndarray],
                       aggregate: int = 1, keep_empty: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Particles (a, b, 0) with per-class area masses (mm^2), summed over aggregate^2 pixels.

    ``classes`` maps a feature name to a boolean lookup table over label values.  Cells
    without any class mass are dropped unless ``keep_empty`` (the full grid, row-major).
    """

    n = (len(axis) // aggregate) * aggregate
    grid = grid[:n, :n]
    masses = []
    for table in classes.values():
        member = table[np.clip(grid, 0, len(table) - 1)] & (grid < len(table))
        masses.append(member.reshape(n // aggregate, aggregate, n // aggregate, aggregate).sum((1, 3)))
    nu = np.stack(masses, -1).reshape(-1, len(classes)).astype(np.float64) * spacing_mm ** 2
    centers = axis[:n].reshape(-1, aggregate).mean(1)
    aa, bb = np.meshgrid(centers, centers, indexing="xy")
    X = np.column_stack([aa.ravel(), bb.ravel(), np.zeros(aa.size)])
    if keep_empty:
        return X, nu
    keep = nu.sum(1) > 0
    return X[keep], nu[keep]


def plane_particles(atlas: Mapping, scan: Mapping, origin, u, v, spacing: float, aggregate: int, table: str,
                    keep_empty: bool = False):
    """Atlas plane ``origin + a u + b v`` as particles carrying ``atlas[table]`` areas.

    ``scan`` supplies the in-plane window ``half_width_mm``.
    """

    grid, axis = reslice_labels(atlas["labels"], atlas["affine"], origin, u, v, float(scan["half_width_mm"]), spacing)
    return label_class_masses(grid, axis, spacing, atlas[table], aggregate, keep_empty)


def _scan_grid(scan: Mapping) -> tuple[float, int]:
    """(reslice spacing, pixels per particle side) for particles of ``scan_bin_mm``."""

    spacing = float(scan["reslice_mm"])
    return spacing, int(round(float(scan["scan_bin_mm"]) / spacing))


def coronal_grid(atlas: Mapping, scan: Mapping, y: float, spacing: float, aggregate: int, table: str = "coarse"):
    """Full-grid (unfiltered) class masses of the coronal plane at y: (X[n, 2], nu[n, C])."""

    origin, u, v, _ = plane_geometry(atlas, 0.0, 0.0, y)
    X, nu = plane_particles(atlas, scan, origin, u, v, spacing, aggregate, table, keep_empty=True)
    return X[:, :2], nu


def region_support_mask(atlas: Mapping, scan: Mapping, y: float, spacing: float, aggregate: int,
                        prior: Mapping[str, Any]) -> np.ndarray:
    """Grid cells within ``margin_mm`` of the prior's atlas regions (dissection-region prior)."""

    _, fine = coronal_grid(atlas, scan, y, spacing, aggregate, table="fine")
    names = list(atlas["fine"])
    present = fine[:, [names.index(r) for r in prior["regions"]]].sum(1) > 0
    side = int(round(np.sqrt(len(present))))
    grid = present.reshape(side, side)
    cell = spacing * aggregate
    distance = ndi.distance_transform_edt(~grid) * cell
    return (distance <= float(prior.get("margin_mm", 6.0))).ravel()


def plane_measure(atlas: Mapping, scan: Mapping, y: float, prior: Mapping[str, Any] | None = None):
    """Coronal plane at y as gray/white particles of ``scan_bin_mm`` (positions 2-D).

    ``prior`` ({regions, margin_mm}) keeps only atlas cells within margin_mm of those fine
    regions: an explicit dissection-region prior (the section can only be placed there).
    """

    spacing, aggregate = _scan_grid(scan)
    X, nu = coronal_grid(atlas, scan, y, spacing, aggregate)
    keep = nu.sum(1) > 0
    if prior:
        keep &= region_support_mask(atlas, scan, y, spacing, aggregate, prior)
    return X[keep], nu[keep]


def slab_measure(atlas: Mapping, scan: Mapping, y_centre: float, half_thickness_mm: float,
                 plane_step_mm: float, prior: Mapping[str, Any] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Mean class masses over the parallel coronal planes y_centre +- half_thickness."""

    spacing, aggregate = _scan_grid(scan)
    ys = np.arange(y_centre - half_thickness_mm, y_centre + half_thickness_mm + 1e-9, plane_step_mm)
    total, X, inside = None, None, None
    for y in ys:
        X, nu = coronal_grid(atlas, scan, y, spacing, aggregate)
        total = nu if total is None else total + nu
        if prior:
            m = region_support_mask(atlas, scan, y, spacing, aggregate, prior)
            inside = m if inside is None else inside | m
    total = total / len(ys)
    keep = total.sum(1) > 0
    if prior:
        keep &= inside
    return X[keep], total[keep]


def atlas_plane(atlas: Mapping, scan: Mapping, y: float, bin_mm: float, anat_channel: str = "MTG_gray_matter"
                ) -> dict[str, Any]:
    """Coronal plane at y: the whole ``half_width_mm`` window at ``bin_mm``, gray/white + anatomical.

    Both feature sets come from one reslice, so positions coincide; the gmwm projection of the
    anatomical masses equals the gray/white masses (checked).  ``anat`` is ``anat_channel`` alone.
    """

    spacing = float(scan["reslice_mm"])
    aggregate = int(round(float(bin_mm) / spacing))
    if abs(aggregate * spacing - float(bin_mm)) > 1e-9:
        raise ValueError(f"atlas_bin_mm {bin_mm} is not a multiple of reslice_mm {spacing}")
    origin, u, v, n = plane_geometry(atlas, 0.0, 0.0, y)
    grid, axis = reslice_labels(atlas["labels"], atlas["affine"], origin, u, v, float(scan["half_width_mm"]), spacing)
    P, nu_gmwm = label_class_masses(grid, axis, spacing, atlas["coarse"], aggregate)
    P2, nu_anat = label_class_masses(grid, axis, spacing, atlas["anatomical"], aggregate)
    names = list(atlas["anatomical"])
    if not np.array_equal(P, P2) or not np.array_equal(project(nu_anat, projection_matrix("gmwm", names)[1]), nu_gmwm):
        raise RuntimeError("anatomical plane particles do not reproduce the gray/white particles")
    return {"y_mm": float(y), "X": P[:, :2].astype(np.float64), "nu": nu_gmwm, "nu_anat": nu_anat,
            "anat_names": names, "anat": nu_anat[:, names.index(anat_channel)][:, None], "anat_channel": anat_channel,
            "origin": origin, "u": u, "v": v, "normal": n, "bin_mm": float(bin_mm),
            "plane_meta": plane_metadata(origin, u, v, n, 0.0, 0.0, float(y))}


# --------------------------------------------------------------------------------------
# Label lookup at world points (landing; QC only)
# --------------------------------------------------------------------------------------


def voxel_labels(atlas: Mapping, world: np.ndarray) -> np.ndarray:
    """Nearest-voxel label code at world points (0 outside the volume)."""

    labels, affine = atlas["labels"], atlas["affine"]
    ijk = np.rint((world - affine[:3, 3]) @ np.linalg.inv(affine[:3, :3]).T).astype(int)
    inside = np.all((ijk >= 0) & (ijk < np.array(labels.shape)), axis=1)
    codes = np.zeros(len(world), int)
    codes[inside] = labels[tuple(ijk[inside].T)]
    return codes


def _assign(index: np.ndarray, codes: np.ndarray, tables: Sequence[np.ndarray]) -> np.ndarray:
    """index[k] = position of the last table containing codes[k] (later tables win)."""

    for k, table in enumerate(tables):
        valid = codes < len(table)
        member = np.zeros(len(codes), bool)
        member[valid] = table[codes[valid]]
        index[member] = k
    return index


def classify_points(atlas: Mapping, world: np.ndarray, region_names: Sequence[str]) -> np.ndarray:
    """Index into region_names + ('outside', 'dropped') for each world point (nearest voxel).

    ``region_names`` index ``atlas["fine"]``; label 0 is 'outside', any other unlisted code 'dropped'.
    """

    codes = voxel_labels(atlas, world)
    out = np.full(len(world), len(region_names) + 1)  # dropped by default
    out[codes == 0] = len(region_names)
    return _assign(out, codes, [atlas["fine"][name] for name in region_names])


def to_world(origin, u, v, plane_points: np.ndarray) -> np.ndarray:
    return origin + plane_points[:, :1] * u + plane_points[:, 1:2] * v


def landing_fractions(atlas: Mapping, transform, origin, u, v, points_local: np.ndarray,
                      region_names: Sequence[str]) -> dict[str, float]:
    """Share of section points (moved by ``transform``) on each named region, outside, dropped."""

    index = classify_points(atlas, to_world(origin, u, v, transform.section_to_moving(points_local)), region_names)
    names = list(region_names) + list(LANDING_CLASSES)
    counts = np.bincount(index, minlength=len(names)) / max(len(index), 1)
    return dict(zip(names, counts.tolist()))


def landing(plane: Mapping[str, Any], atlas: Mapping, pose, X_section: np.ndarray, masses: np.ndarray,
            channel_tables: Mapping[str, np.ndarray]) -> dict[str, float]:
    """Mass fractions of section particles landing on each channel (nearest voxel).

    ``outside`` = label 0, a dropped code, or any code in no channel.
    """

    world = to_world(plane["origin"], plane["u"], plane["v"], pose.section_to_moving(X_section))
    codes = voxel_labels(atlas, world)
    names = list(channel_tables)
    index = _assign(np.full(len(world), len(names)), codes, [channel_tables[n] for n in names])
    m = np.asarray(masses, float).ravel()
    total = m.sum()
    return {n: float(m[index == k].sum() / total) if total > 0 else 0.0 for k, n in enumerate([*names, "outside"])}


# --------------------------------------------------------------------------------------
# Landmark
# --------------------------------------------------------------------------------------


def posterior_mm(y_mm):
    """Slab coordinate s (posterior-increasing) of an atlas world y (anterior-increasing)."""

    return -np.asarray(y_mm, float)


def world_y_mm(s_mm):
    return -np.asarray(s_mm, float)


def label_coronal_profile(labels: np.ndarray, affine: np.ndarray, codes: Sequence[int],
                          hemisphere: str | None = None, plane_step_mm: float | None = None) -> dict[str, Any]:
    """Cross-sectional area (mm^2) of the label set per coronal plane, in world y.

    Voxel centres are binned by world y into planes ``plane_step_mm`` apart on the voxel
    lattice (default: the voxel spacing along world y); area = voxels x voxel volume / step.
    ``hemisphere`` ('left' / 'right') keeps world x < 0 / > 0.
    """

    affine = np.asarray(affine, float)
    ijk = np.argwhere(np.isin(labels, np.asarray(codes, int)))
    if not len(ijk):
        return {"y_mm": np.zeros(0), "area_mm2": np.zeros(0), "world": np.zeros((0, 3)), "step_mm": None}
    world = ijk @ affine[:3, :3].T + affine[:3, 3]
    if hemisphere is not None:
        world = world[np.sign(world[:, 0]) == (-1.0 if hemisphere == "left" else 1.0)]
    # World-y spacing between adjacent voxel planes: the largest |dy| of one index step.
    step = float(plane_step_mm or np.abs(affine[1, :3]).max())
    origin = float(affine[1, 3])
    plane = np.round((world[:, 1] - origin) / step).astype(np.int64)
    keys, counts = np.unique(plane, return_counts=True)
    voxel_volume = abs(float(np.linalg.det(affine[:3, :3])))
    return {"y_mm": origin + keys * step, "area_mm2": counts * voxel_volume / step,
            "world": world, "plane_index": plane, "keys": keys, "step_mm": step}


def locate_landmark(labels: np.ndarray, affine: np.ndarray, spec: Mapping[str, Any],
                    hemisphere: str | None = None) -> dict[str, Any]:
    """The landmark plane from ``spec``: manual ``y_mm``, or the label's first appearance.

    ``spec``: {name, codes, rule: "first_appearance", min_area_mm2, [y_mm]}.  Returns world y,
    slab coordinate s, how it was obtained, the area profile (most anterior first) and the
    centroid of the landmark cross-section on that plane (for figures).
    """

    name = spec.get("name", "landmark")
    profile = (label_coronal_profile(labels, affine, spec["codes"], hemisphere)
               if spec.get("codes") else None)
    rows = []
    if profile is not None and len(profile["y_mm"]):
        order = np.argsort(-profile["y_mm"])  # anterior -> posterior
        rows = [[float(profile["y_mm"][k]), float(profile["area_mm2"][k])] for k in order]
    if spec.get("y_mm") is not None:
        y = float(spec["y_mm"])
        out = {"name": name, "y_mm": y, "s_mm": float(posterior_mm(y)), "method": "manual (landmark.y_mm)",
               "codes": list(spec.get("codes", [])), "hemisphere": hemisphere}
        if rows:
            out["label_first_appearance_y_mm_for_comparison"] = rows[0][0]
        out["profile_y_area_mm2"] = rows
        out["centroid_mm"] = None
        return out
    rule = spec.get("rule", "first_appearance")
    if rule != "first_appearance":
        raise NotImplementedError(f"landmark rule {rule!r}: only 'first_appearance' or a manual y_mm")
    if not rows:
        raise ValueError(f"{name}: label codes {list(spec.get('codes', []))} are absent from the atlas "
                         f"({hemisphere or 'both'} hemisphere); set landmark.y_mm manually")
    min_area = float(spec.get("min_area_mm2", 1.0))
    first = next((r for r in rows if r[1] >= min_area), None)
    if first is None:
        raise ValueError(f"{name}: no coronal plane reaches {min_area} mm^2")
    y = first[0]
    key = profile["keys"][np.argmin(np.abs(profile["y_mm"] - y))]
    centroid = profile["world"][profile["plane_index"] == key].mean(0)
    return {"name": name, "y_mm": y, "s_mm": float(posterior_mm(y)),
            "method": f"first appearance of atlas label(s) {list(spec['codes'])} moving anterior -> posterior: "
                      f"most anterior coronal plane with cross-section >= {min_area} mm^2",
            "codes": list(spec["codes"]), "hemisphere": hemisphere, "min_area_mm2": min_area,
            "first_any_voxel_y_mm": rows[0][0], "area_at_landmark_mm2": first[1],
            "precision_mm": profile["step_mm"], "centroid_mm": centroid.tolist(),
            "profile_y_area_mm2": rows}


# --------------------------------------------------------------------------------------
# Slabs
# --------------------------------------------------------------------------------------


def phases(thickness_mm: float, step_mm: float) -> np.ndarray:
    """delta in [0, h) on a step grid (the unknown cutting phase)."""

    return np.round(np.arange(0.0, float(thickness_mm) - 1e-9, float(step_mm)), 6)


def slab_bounds(s_landmark: float, thickness_mm: float, j: int, delta_mm: float) -> tuple[float, float]:
    """[s0, s1] of S_j(delta)."""

    s0 = float(s_landmark) - float(delta_mm) + int(j) * float(thickness_mm)
    return s0, s0 + float(thickness_mm)


def candidate_slabs(s_landmark: float, thickness_mm: float, offsets: Sequence[int], phase_step_mm: float) -> list[dict]:
    """Every S_j(delta): offset, phase, s bounds and world-y bounds [anterior, posterior]."""

    out = []
    for j in offsets:
        for delta in phases(thickness_mm, phase_step_mm):
            s0, s1 = slab_bounds(s_landmark, thickness_mm, j, delta)
            out.append({"j": int(j), "delta_mm": float(delta), "s_mm": [s0, s1],
                        "y_mm": [float(world_y_mm(s0)), float(world_y_mm(s1))]})
    return out


def offset_cover(s_landmark: float, thickness_mm: float, j: int) -> tuple[float, float]:
    """s range covered by S_j over all phases: (s_L + (j-1)h, s_L + (j+1)h]."""

    h = float(thickness_mm)
    return float(s_landmark) + (int(j) - 1) * h, float(s_landmark) + (int(j) + 1) * h


def search_interval(s_landmark: float, thickness_mm: float, offsets: Sequence[int],
                    extension_mm: float = 0.0) -> dict[str, list[float]]:
    """Union of the candidate slabs over all phases and offsets, widened by ``extension_mm``."""

    lo = offset_cover(s_landmark, thickness_mm, min(offsets))[0] - float(extension_mm)
    hi = offset_cover(s_landmark, thickness_mm, max(offsets))[1] + float(extension_mm)
    return {"s_mm": [lo, hi], "y_mm": [float(world_y_mm(lo)), float(world_y_mm(hi))]}


def build_slab_prior(landmark: Mapping[str, Any], cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Landmark + candidate slabs + search interval: the JSON-ready record."""

    h = float(cfg.get("slab_thickness_mm", 4.0))
    offsets = [int(j) for j in cfg.get("slab_offsets", (-1, 0, 1))]
    step = float(cfg.get("phase_step_mm", 0.5))
    ext = float(cfg.get("extension_mm", 0.0))
    s_L = float(landmark["s_mm"])
    return {"ap_convention": AP_CONVENTION, "landmark": dict(landmark), "slab_thickness_mm": h,
            "slab_offsets": offsets, "phase_step_mm": step, "extension_mm": ext,
            "slab_formula": "S_j(delta) = [s_L - delta + j h, s_L - delta + (j+1) h], 0 <= delta < h",
            "offset_cover_y_mm": {str(j): [float(world_y_mm(v)) for v in offset_cover(s_L, h, j)] for j in offsets},
            "search_interval": search_interval(s_L, h, offsets, ext),
            "candidates": candidate_slabs(s_L, h, offsets, step)}


def landmark_slab_prior(atlas: Mapping, cfg: Mapping[str, Any], hemisphere: str) -> dict[str, Any]:
    """Check the landmark code against the terminology, locate it and build the slab prior."""

    spec = cfg["landmark"]
    if spec.get("codes") and spec.get("expected_name"):
        decoded = atlas["names"].get(int(spec["codes"][0]))
        if decoded != spec["expected_name"]:
            raise ValueError(f"landmark {spec.get('name')}: code {spec['codes'][0]} decodes to {decoded!r}, "
                             f"expected {spec['expected_name']!r}")
    record = build_slab_prior(locate_landmark(atlas["labels"], atlas["affine"], spec, hemisphere), cfg)
    record.update(sampling_regions=list(cfg.get("sampling_regions", [])),
                  concepts={"slab": "nominal tissue slab S_j(delta) of the dissection protocol",
                            "sampling_region": "atlas sampling_regions inside the slab (display/QC only)",
                            "qnp_block": "not constructed: its position is the registration estimate"})
    return record


def clamp_ys(ys: Sequence[float], interval_y: Sequence[float], tol: float = 1e-6) -> list[float]:
    """Planes inside the search interval ([anterior y, posterior y])."""

    hi, lo = max(interval_y), min(interval_y)
    return [y for y in ys if lo - tol <= y <= hi + tol]


def slab_consistency(profile: Mapping[float, float], candidates: Sequence[Mapping[str, Any]],
                     delta_noise: float | None) -> dict[str, Any]:
    """Best profile score inside each candidate slab and the slabs within ``delta_noise`` of the best.

    ``profile``: {y: score} of one section (lower is better).  A plane on a shared boundary
    belongs to both slabs (closed intervals): at atlas resolution it cannot be assigned.
    """

    ys = np.array(sorted(profile), float)
    scores = np.array([profile[y] for y in ys])
    rows = []
    for c in candidates:
        lo, hi = min(c["y_mm"]), max(c["y_mm"])
        inside = (ys >= lo - 1e-6) & (ys <= hi + 1e-6)
        if not inside.any():
            continue
        k = np.flatnonzero(inside)[np.argmin(scores[inside])]
        rows.append({"j": c["j"], "delta_mm": c["delta_mm"], "best_y_mm": float(ys[k]), "best_score": float(scores[k]),
                     "planes_scored": int(inside.sum())})
    if not rows:
        return {"slabs": [], "consistent": {}}
    best = min(r["best_score"] for r in rows)
    tol = 0.0 if delta_noise is None else float(delta_noise)
    consistent: dict[str, list[float]] = {}
    for r in rows:
        r["within_delta"] = bool(r["best_score"] <= best + tol + 1e-12)
        if r["within_delta"]:
            consistent.setdefault(str(r["j"]), []).append(r["delta_mm"])
    return {"slabs": rows, "consistent": {j: [min(d), max(d)] for j, d in consistent.items()},
            "rule": "candidate slabs whose best plane scores within delta of the best slab"}
