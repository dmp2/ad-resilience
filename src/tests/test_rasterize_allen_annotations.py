from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import allen_svg_renderer as renderer
import rasterize_allen_annotations as raster


CURRENT_INVENTORY = (
    "5a46a8e85bbd26ada68fa438bd067da3044ab3e1d064905d9427e811d7e8101f"
)
CURRENT_MANIFEST = (
    "c444caa3997e66beba998d3ccd801dbfb3441c06cd7b3a53ee6b7baec8891fa7"
)
CURRENT_STRUCTURES = (
    "b12afeed3e1f11b7ee6aced03cd7f7875b103d5984d168c0fcca50235fa6da57"
)
RAW = Path(__file__).resolve().parents[2] / "data/raw/allen/specimen_708424"


def write_svg(path: Path, body: str, width: int = 12, height: int = 10) -> None:
    path.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}">{body}</svg>'
    )


def synthetic_snapshot(groups=((31, "Anatomy"),)) -> raster.SourceSnapshot:
    return raster.SourceSnapshot("a" * 64, "b" * 64, "c" * 64, 1, 2, 3, 16, groups)


def test_current_repaired_snapshot_and_section_mapping():
    snapshot, sections, structures, _ = raster.load_source(RAW)
    assert snapshot.inventory_sha256 == CURRENT_INVENTORY
    assert snapshot.manifest_sha256 == CURRENT_MANIFEST
    assert snapshot.structures_sha256 == CURRENT_STRUCTURES
    assert len(sections) == 106
    assert len(structures) == 3317
    section = sections[0]
    assert section.section_number == 111
    assert section.graphic_groups == (265297118, 31, 113753816)
    assert (section.width_px, section.height_px) == (908, 1356)
    assert section.pixel_size_um == 31.072


def test_standard_transform_parser_and_composition():
    matrix = renderer.parse_transform(
        "translate(3,4) scale(2) rotate(90)", np=np
    )
    point = matrix @ np.asarray([1.0, 0.0, 1.0])
    assert np.allclose(point, [3.0, 6.0, 1.0])
    centered = renderer.parse_transform("rotate(90 2 3)", np=np)
    assert np.allclose(centered @ np.asarray([3.0, 3.0, 1.0]), [2.0, 4.0, 1.0])
    with pytest.raises(RuntimeError, match="Unsupported"):
        renderer.parse_transform("translate(1) nonsense(2)", np=np)


def test_document_order_transforms_fill_rules_and_overlap_metrics(tmp_path):
    svg = tmp_path / "test.svg"
    write_svg(
        svg,
        """
        <g graphic_group_label_id="31" graphic_group_label="Anatomy"
           transform="translate(1 0)" fill-rule="evenodd">
          <path structure_id="1" order="99"
                d="M0 0 H7 V7 H0 Z M2 2 H5 V5 H2 Z"/>
          <g transform="translate(3 0)">
            <path structure_id="2" order="1" d="M0 1 H7 V6 H0 Z"/>
            <path structure_id="2" order="1" d="M1 2 H4 V5 H1 Z"/>
          </g>
        </g>
        """,
    )
    result = renderer.rasterize_svg(svg, 12, 10, {31: "Anatomy"})[0]
    assert result.labels.shape == (10, 12)
    assert result.labels.dtype == np.uint32
    assert set(np.unique(result.labels)) == {0, 1, 2}
    assert result.conflicting_overlap_pixel_count > 0
    assert result.repeated_same_id_pixel_count > 0
    assert result.svg_order_nondecreasing is False
    # The later ID 2 path wins despite its lower Allen order value.
    assert result.labels[1, 4] == 2


def test_non_antialiased_half_pixel_policy_is_locked_to_skia(tmp_path):
    integer = tmp_path / "integer.svg"
    half = tmp_path / "half.svg"
    group = (
        '<g graphic_group_label_id="31" graphic_group_label="Anatomy">'
        '<path structure_id="1" d="{path}"/></g>'
    )
    write_svg(integer, group.format(path="M2 2 H6 V6 H2 Z"), 8, 8)
    write_svg(half, group.format(path="M2.5 2.5 H6.5 V6.5 H2.5 Z"), 8, 8)
    a = renderer.rasterize_svg(integer, 8, 8, {31: "Anatomy"})[0].labels
    b = renderer.rasterize_svg(half, 8, 8, {31: "Anatomy"})[0].labels
    # These are operational Skia goldens, not a mathematical containment claim.
    assert np.argwhere(a).tolist() == [
        [2, 2], [2, 3], [2, 4], [2, 5],
        [3, 2], [3, 3], [3, 4], [3, 5],
        [4, 2], [4, 3], [4, 4], [4, 5],
        [5, 2], [5, 3], [5, 4], [5, 5],
    ]
    assert np.argwhere(b).tolist() == [
        [3, 3], [3, 4], [3, 5], [3, 6],
        [4, 3], [4, 4], [4, 5], [4, 6],
        [5, 3], [5, 4], [5, 5], [5, 6],
        [6, 3], [6, 4], [6, 5], [6, 6],
    ]


def test_ome_zarr_true_2d_labels_codec_and_compact_ontology(tmp_path):
    data = tmp_path / "raw"
    data.mkdir()
    image_path = data / "image.jpg"
    svg_path = data / "labels.svg"
    Image.new("RGB", (12, 10), (20, 40, 60)).save(
        image_path, quality=100, subsampling=0
    )
    write_svg(
        svg_path,
        '<g graphic_group_label_id="31" graphic_group_label="Anatomy">'
        '<path structure_id="1" order="5" style="stroke:black;fill:#fff" '
        'd="M1 1 H9 V8 H1 Z"/></g>',
    )
    section = raster.SectionSpec(
        7,
        "image.jpg",
        raster.sha_file(image_path),
        "labels.svg",
        raster.sha_file(svg_path),
        (31,),
        12,
        10,
        2.5,
    )
    snapshot = synthetic_snapshot()
    structures = {1: raster.Structure(1, "A", "Area", (1, 2, 3, 255))}
    package = tmp_path / section.package_name
    raster.write_package(package, data, section, snapshot, structures)
    metrics = raster.validate_package(
        package, data, section, snapshot, structures
    )
    assert metrics["unique_structure_count"] == 1
    root_array = json.loads((package / "0/zarr.json").read_text())
    label_array = json.loads(
        (package / "labels/group-31/0/zarr.json").read_text()
    )
    assert root_array["shape"] == [3, 10, 12]
    assert root_array["dimension_names"] == ["c", "y", "x"]
    assert root_array["codecs"] == raster.CODEC_PIPELINES["uint8"]
    assert label_array["shape"] == [10, 12]
    assert label_array["dimension_names"] == ["y", "x"]
    assert label_array["codecs"] == raster.CODEC_PIPELINES["uint32"]
    attrs = json.loads(
        (package / "labels/group-31/zarr.json").read_text()
    )["attributes"]["ome"]["image-label"]
    assert attrs["source"] == {"image": "../../"}
    assert attrs["colors"] == [{"label-value": 1, "rgba": [1, 2, 3, 255]}]
    assert attrs["properties"] == [
        {"label-value": 1, "name": "Area", "acronym": "A"}
    ]
    assert "parent_structure_id" not in json.dumps(attrs)
    assert "structure_id_path" not in json.dumps(attrs)
    from ome_zarr.io import parse_url
    from ome_zarr.reader import Reader

    nodes = list(Reader(parse_url(str(package), mode="r"))())
    assert len(nodes) == 3  # image, labels container, and one label image


def test_tree_hash_is_path_sensitive_deterministic_and_rejects_symlinks(tmp_path):
    root = tmp_path / "tree"
    root.mkdir()
    (root / "a").write_text("one")
    (root / "b").write_text("two")
    first = raster.tree_sha256(root)
    assert first == raster.tree_sha256(root)
    (root / "b").rename(root / "c")
    assert first != raster.tree_sha256(root)
    (root / "link").symlink_to(root / "a")
    with pytest.raises(RuntimeError, match="Symlink"):
        raster.tree_sha256(root)


def test_snapshot_drift_and_filtered_initialization_are_refused(tmp_path):
    output = tmp_path / "derivative"
    snapshot = synthetic_snapshot()
    with pytest.raises(RuntimeError, match="Filtered runs"):
        raster.ensure_snapshot(output, snapshot, filtered=True, verify_only=False)
    raster.ensure_snapshot(output, snapshot, filtered=False, verify_only=False)
    changed = raster.SourceSnapshot(
        "d" * 64,
        snapshot.manifest_sha256,
        snapshot.structures_sha256,
        snapshot.specimen_id,
        snapshot.donor_id,
        snapshot.atlas_id,
        snapshot.structure_graph_id,
        snapshot.graphic_groups,
    )
    with pytest.raises(RuntimeError, match="SOURCE_SNAPSHOT_DRIFT"):
        raster.ensure_snapshot(output, changed, filtered=False, verify_only=False)


@pytest.mark.skipif(
    os.environ.get("ALLEN_SECTION111_VALIDATION") != "1",
    reason="development-only CairoSVG validation gate",
)
def test_section_111_independent_cairosvg():
    cairosvg = pytest.importorskip("cairosvg")
    import io

    snapshot, sections, _, _ = raster.load_source(RAW)
    section = next(item for item in sections if item.section_number == 111)
    layers = renderer.rasterize_svg(
        RAW / section.svg_path,
        section.width_px,
        section.height_px,
        dict(snapshot.graphic_groups),
    )
    skia_occupied = np.zeros((section.height_px, section.width_px), dtype=bool)
    for layer in layers:
        skia_occupied |= layer.labels != 0
    png = cairosvg.svg2png(
        url=str(RAW / section.svg_path),
        output_width=section.width_px,
        output_height=section.height_px,
    )
    cairo_rgba = np.asarray(Image.open(io.BytesIO(png)).convert("RGBA"))
    cairo_occupied = cairo_rgba[..., 3] != 0
    def boundary_of(mask):
        boundary = np.zeros(mask.shape, dtype=bool)
        boundary[1:, :] |= mask[1:, :] != mask[:-1, :]
        boundary[:, 1:] |= mask[:, 1:] != mask[:, :-1]
        return boundary

    boundary = boundary_of(cairo_occupied) | boundary_of(skia_occupied)
    band = boundary.copy()
    padded = np.pad(boundary, 1)
    for y in (-1, 0, 1):
        for x in (-1, 0, 1):
            band |= padded[
                1 + y : 1 + y + boundary.shape[0],
                1 + x : 1 + x + boundary.shape[1],
            ]
    mismatch = skia_occupied != cairo_occupied
    assert not np.any(mismatch & ~band)
    for occupied in (skia_occupied, cairo_occupied):
        ys, xs = np.nonzero(occupied)
        assert ys.size
        bounds = (xs.min(), ys.min(), xs.max(), ys.max())
        if occupied is skia_occupied:
            skia_bounds = bounds
        else:
            cairo_bounds = bounds
    assert max(abs(a - b) for a, b in zip(skia_bounds, cairo_bounds)) <= 1
