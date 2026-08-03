from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
import zarr
from PIL import Image

import visualize_allen_annotations as visual


FROZEN_ALL_GROUP_PACKAGE = (
    Path(__file__).resolve().parents[2]
    / "data/derivatives/allen/specimen_708424/annotations_ome_zarr"
    / "section-1532.ome.zarr"
)


def _multiscales(name: str, path: str, axes: tuple[str, ...]):
    return [
        {
            "name": name,
            "axes": [
                {"name": axis, "type": "channel" if axis == "c" else "space"}
                for axis in axes
            ],
            "datasets": [{"path": path}],
        }
    ]


def write_package(path: Path, include_sparse_color: bool = True) -> Path:
    root = zarr.open_group(str(path), mode="w", zarr_format=3)
    root.attrs.update(
        {
            "ome": {
                "version": "0.5",
                "multiscales": _multiscales("Nissl", "image", ("c", "y", "x")),
            },
            "allen": {"section_number": 7},
        }
    )
    rgb = np.zeros((3, 12, 8), dtype=np.uint8)
    rgb[0] = 20
    rgb[1] = 40
    rgb[2] = 60
    root.create_array("image", data=rgb, chunks=rgb.shape)

    labels = root.create_group("labels")
    labels.attrs.update(
        {"ome": {"version": "0.5", "labels": ["group-second", "group-first"]}}
    )

    second = labels.create_group("group-second")
    second_colors = [{"label-value": 1, "rgba": [200, 100, 50, 255]}]
    if include_sparse_color:
        second_colors.append({"label-value": 2, "rgba": [5, 220, 80, 255]})
    second.attrs.update(
        {
            "ome": {
                "version": "0.5",
                "multiscales": _multiscales("Second", "labels", ("y", "x")),
                "image-label": {"version": "0.5", "colors": second_colors},
            },
            "allen": {"graphic_group_name": "Second in metadata"},
        }
    )
    second_data = np.zeros((12, 8), dtype=np.uint32)
    second_data[2:10, 1:7] = 1
    second_data[0, 7] = 2
    second.create_array("labels", data=second_data, chunks=second_data.shape)

    first = labels.create_group("group-first")
    first.attrs.update(
        {
            "ome": {
                "version": "0.5",
                "multiscales": _multiscales("First", "0", ("y", "x")),
                "image-label": {
                    "version": "0.5",
                    "colors": [{"label-value": 3, "rgba": [90, 110, 230, 255]}],
                },
            },
            "allen": {"graphic_group_name": "First on disk"},
        }
    )
    first_data = np.zeros((12, 8), dtype=np.uint32)
    first_data[4:8, 2:6] = 3
    first.create_array("0", data=first_data, chunks=first_data.shape)
    return path


def test_metadata_order_colors_and_sparse_label_are_preserved(tmp_path):
    package = write_package(tmp_path / "section-0007.ome.zarr")

    root = zarr.open_group(str(package), mode="r")
    rgb = np.transpose(np.asarray(root["image"][:]), (1, 2, 0))
    expected_nissl = Image.fromarray(rgb).resize((80, 120), Image.Resampling.LANCZOS)
    second_group = root["labels/group-second"]
    second_labels = np.asarray(second_group["labels"][:])
    second_colors = visual._label_colors(second_group, "label group group-second")
    expected_second = visual._label_thumbnail(
        second_labels, second_colors, (80, 120), "label group group-second"
    )
    first_group = root["labels/group-first"]
    first_labels = np.asarray(first_group["0"][:])
    first_colors = visual._label_colors(first_group, "label group group-first")
    expected_first = visual._label_thumbnail(
        first_labels, first_colors, (80, 120), "label group group-first"
    )

    heading, panels = visual.load_panels(package, panel_width=80)

    assert heading == "Allen section 0007 annotations"
    assert [panel.title for panel in panels] == [
        "Nissl reference",
        "Second in metadata",
        "First on disk",
        visual.COMBINED_PREVIEW_TITLE,
        visual.BOUNDARY_OVERLAY_TITLE,
    ]
    assert panels[0].image.size == (80, 120)
    assert np.array_equal(np.asarray(panels[0].image), np.asarray(expected_nissl))
    assert np.array_equal(np.asarray(panels[1].image), np.asarray(expected_second))
    assert np.array_equal(np.asarray(panels[2].image), np.asarray(expected_first))
    assert np.all(np.asarray(panels[0].image)[60, 40] == [20, 40, 60])

    second = np.asarray(panels[1].image)
    assert np.any(np.all(second == [200, 100, 50], axis=2))
    assert np.any(np.all(second == [5, 220, 80], axis=2))


@pytest.mark.skipif(
    not FROZEN_ALL_GROUP_PACKAGE.is_dir(),
    reason="frozen all-four-group Allen derivative is not available",
)
def test_frozen_all_group_original_panel_buffers_are_unchanged():
    root = zarr.open_group(str(FROZEN_ALL_GROUP_PACKAGE), mode="r")
    rgb, width, height = visual._rgb_image(root)
    panel_width = 64
    panel_height = round(height * panel_width / width)
    expected = [
        Image.fromarray(rgb).resize(
            (panel_width, panel_height), Image.Resampling.LANCZOS
        )
    ]
    labels_group = root["labels"]
    names = visual._label_names(labels_group)
    assert names == [
        "group-265297118",
        "group-31",
        "group-113753816",
        "group-141667008",
    ]
    for name in names:
        group = labels_group[name]
        dataset_path, _ = visual._multiscale(group, f"label group {name}")
        labels = visual._read_array(group, dataset_path, f"label group {name}")
        colors = visual._label_colors(group, f"label group {name}")
        expected.append(
            visual._label_thumbnail(
                labels,
                colors,
                (panel_width, panel_height),
                f"Label group {name}",
            )
        )

    _, panels = visual.load_panels(FROZEN_ALL_GROUP_PACKAGE, panel_width=panel_width)

    assert len(panels) == len(expected) + 2
    for old_buffer, new_panel in zip(
        expected, panels[: len(expected)], strict=True
    ):
        assert np.array_equal(np.asarray(old_buffer), np.asarray(new_panel.image))


def test_combined_display_uses_declared_order_and_is_deterministic():
    earlier = np.array([[0, 1, 1], [0, 1, 0]], dtype=np.uint32)
    later = np.array([[0, 0, 2], [3, 0, 0]], dtype=np.uint32)

    combined = visual._combined_display_map([earlier, later])

    assert np.array_equal(
        combined,
        np.array([[0, 1, 2], [3, 1, 0]], dtype=np.uint32),
    )
    assert np.array_equal(combined, visual._combined_display_map([earlier, later]))


def test_nearest_resize_and_symmetric_internal_boundaries():
    labels = np.array([[0, 1], [2, 2]], dtype=np.uint32)
    resized = visual._resize_labels_nearest(labels, (4, 4))
    assert np.array_equal(
        resized,
        np.array(
            [
                [0, 0, 1, 1],
                [0, 0, 1, 1],
                [2, 2, 2, 2],
                [2, 2, 2, 2],
            ],
            dtype=np.uint32,
        ),
    )

    boundary = visual._internal_boundary_mask(resized)
    expected = np.zeros((4, 4), dtype=bool)
    expected[1, 2:] = True
    expected[2, 2:] = True
    assert np.array_equal(boundary, expected)
    assert not boundary[0, 1]
    assert not boundary[1, 1]

    colors = [(1, (50, 60, 70, 255)), (2, (50, 60, 70, 255))]
    first = visual._combined_display_thumbnail(labels, colors, (4, 4))
    second = visual._combined_display_thumbnail(labels, colors, (4, 4))
    rendered = np.asarray(first)
    assert np.array_equal(rendered, np.asarray(second))
    assert np.all(rendered[boundary] == visual.BOUNDARY_COLOR)
    assert np.all(rendered[0, 2] == [50, 60, 70])
    assert np.all(rendered[3, 0] == [50, 60, 70])


def test_native_boundary_overlay_preserves_existing_magenta_semantics():
    rgb = np.full((3, 4, 3), 17, dtype=np.uint8)
    labels = np.array(
        [[0, 1, 1, 0], [0, 1, 2, 0], [0, 0, 2, 0]], dtype=np.uint32
    )
    expected = rgb.copy()
    edge = np.zeros(labels.shape, dtype=bool)
    edge[1:, :] |= labels[1:, :] != labels[:-1, :]
    edge[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    expected[edge & (labels != 0)] = visual.OVERLAY_BOUNDARY_COLOR

    rendered = visual._boundary_overlay(rgb, labels, (4, 3))

    assert np.array_equal(np.asarray(rendered), expected)


def test_montage_grid_default_output_and_overwrite_protection(tmp_path):
    package = write_package(tmp_path / "section-0007.ome.zarr")
    montage, titles = visual.render_montage(package, panel_width=80)
    assert titles == (
        "Second in metadata",
        "First on disk",
        visual.COMBINED_PREVIEW_TITLE,
        visual.BOUNDARY_OVERLAY_TITLE,
    )
    assert montage.width > 2 * 80
    assert montage.height > 2 * 120

    destination, written_titles = visual.write_montage(package, panel_width=80)
    assert destination == tmp_path / "thumbnails/section-0007.png"
    assert written_titles == titles
    with Image.open(destination) as image:
        assert image.format == "PNG"
        assert image.size == montage.size

    with pytest.raises(RuntimeError, match="already exists"):
        visual.write_montage(package, panel_width=80)
    visual.write_montage(package, panel_width=80, overwrite=True)


def test_missing_color_metadata_is_reported(tmp_path):
    package = write_package(
        tmp_path / "section-0007.ome.zarr", include_sparse_color=False
    )
    with pytest.raises(RuntimeError, match=r"without color metadata: \[2\]"):
        visual.load_panels(package, panel_width=80)


def test_invalid_panel_width_and_output_inside_package_are_rejected(tmp_path):
    package = write_package(tmp_path / "section-0007.ome.zarr")
    with pytest.raises(RuntimeError, match="at least"):
        visual.render_montage(package, panel_width=32)
    with pytest.raises(RuntimeError, match="inside"):
        visual.write_montage(
            package,
            output=package / "thumbnail.png",
            panel_width=80,
        )



def write_bilateral_derivative(path: Path, source_package: Path) -> Path:
    metadata = path / "metadata"
    image_dir = path / "images/nissl"
    metadata.mkdir(parents=True)
    image_dir.mkdir(parents=True)
    root = zarr.open_group(str(source_package), mode="r")
    rgb = np.transpose(np.asarray(root["image"][:]), (1, 2, 0))
    bilateral_rgb = np.concatenate((rgb[:, ::-1], rgb), axis=1)
    image_relative = "images/nissl/section-0007.tif"
    Image.fromarray(bilateral_rgb).save(path / image_relative)

    with (metadata / "physical_sections.tsv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "allen_section_number",
                "image_present",
                "prepared_relative_path",
            ],
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerow(
            {
                "allen_section_number": "7",
                "image_present": "true",
                "prepared_relative_path": image_relative,
            }
        )

    with (metadata / "annotations.tsv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["section_number", "graphic_group_id", "path"],
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for group_id, array_name in (("second", "labels"), ("first", "0")):
            labels = np.asarray(root[f"labels/group-{group_id}/{array_name}"][:])
            bilateral = np.concatenate((labels[:, ::-1], labels), axis=1)
            relative = f"annotations/group-{group_id}/section-0007.tif"
            destination = path / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(bilateral).save(destination)
            writer.writerow(
                {
                    "section_number": "7",
                    "graphic_group_id": group_id,
                    "path": relative,
                }
            )
    return path


def test_bilateral_derivative_reuses_zarr_metadata_and_preserves_symmetry(tmp_path):
    annotations_zarr = tmp_path / "annotations-zarr"
    package = write_package(annotations_zarr / "section-0007.ome.zarr")
    dataset = write_bilateral_derivative(tmp_path / "histology-symmetric", package)

    heading, panels = visual.load_bilateral_panels(
        dataset, 7, annotations_zarr=annotations_zarr, panel_width=80
    )

    assert heading == "Allen section 0007 bilateral annotations"
    assert [panel.title for panel in panels] == [
        "Nissl reference",
        "Second in metadata",
        "First on disk",
        visual.COMBINED_PREVIEW_TITLE,
        visual.BOUNDARY_OVERLAY_TITLE,
        "Medial seam close-up",
    ]
    assert all(panel.image.size == (80, 60) for panel in panels)
    nissl = np.asarray(panels[0].image)
    np.testing.assert_array_equal(nissl[:, :40], nissl[:, 40:][:, ::-1])

    destination, titles = visual.write_bilateral_montage(
        dataset,
        7,
        annotations_zarr=annotations_zarr,
        panel_width=80,
    )
    assert destination == dataset / "thumbnails/section-0007.png"
    assert titles == tuple(panel.title for panel in panels[1:])
    with Image.open(destination) as image:
        assert image.format == "PNG"


def test_bilateral_derivative_rejects_group_and_shape_mismatches(tmp_path):
    annotations_zarr = tmp_path / "annotations-zarr"
    package = write_package(annotations_zarr / "section-0007.ome.zarr")
    dataset = write_bilateral_derivative(tmp_path / "histology-symmetric", package)
    inventory = dataset / "metadata/annotations.tsv"
    rows = list(csv.DictReader(inventory.open(encoding="utf-8"), delimiter="\t"))
    rows.pop()
    with inventory.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["section_number", "graphic_group_id", "path"],
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(RuntimeError, match="groups do not match"):
        visual.load_bilateral_panels(dataset, 7, annotations_zarr=annotations_zarr)

    dataset = write_bilateral_derivative(tmp_path / "histology-shape", package)
    Image.fromarray(np.zeros((12, 15), dtype=np.uint32)).save(
        dataset / "annotations/group-first/section-0007.tif"
    )
    with pytest.raises(RuntimeError, match="does not match section image shape"):
        visual.load_bilateral_panels(dataset, 7, annotations_zarr=annotations_zarr)
