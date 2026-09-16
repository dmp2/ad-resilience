"""A minimal, dependency-free reader for the one thing this project needs from XLSX.

`openpyxl` is a declared dependency of the package but is absent from the conda
environments this repository is actually run in, and a published supplementary
workbook is the only authoritative machine-readable form of some gene-set
definitions. An `.xlsx` file is a ZIP of XML parts, so the small amount of it
that a flat marker table uses can be read with the standard library alone.

Deliberately narrow: it reads cell *values* as strings, in row order, from one
named worksheet. It does not evaluate formulas, apply number formats, resolve
dates, or handle streamed/huge sheets. Anything it cannot read, it refuses to
guess at.
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

_MAIN = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
_REL = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
_PKG_REL = "{http://schemas.openxmlformats.org/package/2006/relationships}"


class XlsxError(RuntimeError):
    """The workbook does not have the shape this reader requires."""


def _column_index(reference: str) -> int:
    """``"C7"`` -> 2. Column letters only; the row number is ignored."""
    index = 0
    for character in reference:
        if not character.isalpha():
            break
        index = index * 26 + (ord(character.upper()) - ord("A") + 1)
    if index == 0:
        raise XlsxError(f"Cell reference without a column: {reference!r}")
    return index - 1


def sheet_names(path: Path) -> list[str]:
    with zipfile.ZipFile(path) as archive:
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        return [sheet.get("name", "") for sheet in workbook.iter(f"{_MAIN}sheet")]


def _sheet_part(archive: zipfile.ZipFile, sheet: str) -> str:
    workbook = ET.fromstring(archive.read("xl/workbook.xml"))
    relationships = {
        node.get("Id"): node.get("Target", "")
        for node in ET.fromstring(archive.read("xl/_rels/workbook.xml.rels")).iter(
            f"{_PKG_REL}Relationship"
        )
    }
    for node in workbook.iter(f"{_MAIN}sheet"):
        if node.get("name") != sheet:
            continue
        target = relationships.get(node.get(f"{_REL}id", ""), "")
        if not target:
            raise XlsxError(f"Sheet {sheet!r} has no resolvable relationship target")
        target = target.lstrip("/")
        return target if target.startswith("xl/") else f"xl/{target}"
    raise XlsxError(f"Sheet {sheet!r} is absent; present: {sheet_names(Path(archive.filename))}")


def _shared_strings(archive: zipfile.ZipFile) -> list[str]:
    try:
        raw = archive.read("xl/sharedStrings.xml")
    except KeyError:
        return []
    return [
        "".join(text.text or "" for text in item.iter(f"{_MAIN}t"))
        for item in ET.fromstring(raw).iter(f"{_MAIN}si")
    ]


def read_sheet(path: Path, sheet: str) -> list[list[str]]:
    """Return one worksheet as a list of rows of cell strings.

    Rows are padded to the width of the widest row, so a ragged sheet still
    yields a rectangle and a missing cell is an empty string rather than a
    silently shifted value.
    """
    with zipfile.ZipFile(path) as archive:
        strings = _shared_strings(archive)
        worksheet = ET.fromstring(archive.read(_sheet_part(archive, sheet)))

    rows: list[dict[int, str]] = []
    for row_node in worksheet.iter(f"{_MAIN}row"):
        cells: dict[int, str] = {}
        position = 0
        for cell in row_node.iter(f"{_MAIN}c"):
            reference = cell.get("r")
            column = _column_index(reference) if reference else position
            position = column + 1
            kind = cell.get("t")
            inline = cell.find(f"{_MAIN}is")
            value_node = cell.find(f"{_MAIN}v")
            if kind == "s" and value_node is not None and value_node.text is not None:
                value = strings[int(value_node.text)]
            elif inline is not None:
                value = "".join(text.text or "" for text in inline.iter(f"{_MAIN}t"))
            elif value_node is not None:
                value = value_node.text or ""
            else:
                value = ""
            cells[column] = value
        rows.append(cells)

    width = max((max(cells) + 1 for cells in rows if cells), default=0)
    return [[cells.get(column, "") for column in range(width)] for cells in rows]


def read_records(path: Path, sheet: str) -> list[dict[str, str]]:
    """Read a worksheet whose first row is a header, as a list of dicts."""
    table = read_sheet(path, sheet)
    if not table:
        raise XlsxError(f"Sheet {sheet!r} in {path.name} is empty")
    header = [name.strip() for name in table[0]]
    return [dict(zip(header, row)) for row in table[1:]]
