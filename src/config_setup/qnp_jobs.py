"""Resolve one QNP job (``qnp_section_inputs.<job>``) against shared per-project defaults.

A job block in ``qnp_section_inputs`` may be *thin* (just ``donor`` and ``region``) or fully
explicit (the original ``H20.33.034_MTG`` block).  ``resolve_qnp_job`` returns the same
mapping the scripts always used, built as

1. ``qnp_section_defaults`` deep-merged under the job block (the job wins; lists are
   replaced, never concatenated);
2. donor-derived values read from the data, not copied into the config:
   * ``stains``: restricted to the stains that have an ``.svs`` in ``slide_dir`` when the
     stain list comes from the defaults (an explicit list is kept and checked later);
   * ``_slide_stems``: stain -> slide stem (e.g. ``H20.33.034-A07-NEUN``), which carries the
     block name as the slide is actually named (``A07`` vs ``A7``);
   * ``annotation_misfit``: from ``<qc_dir>/inventory.json`` (slides whose HALO ribbon lies
     less than ``annotation_on_tissue_min`` on their own tissue) unless the job sets it;
3. string templates filled in every value: ``{donor}``, ``{region}``, ``{job}``,
   ``{slide_dir}``, ``{particle_dir}``, ``{qc_dir}``, ``{aligned_dir}``, ``{structural}``
   (structural stains joined by ``_``), ``{reference_stem}`` and ``{stem[STAIN]}``.
   Unknown fields (e.g. ``{plane_id}``, filled later by --stage export) are left as is.

Keys starting with ``_`` are provenance added here and are never read from YAML.
"""

from __future__ import annotations

import copy
import json
import string
from pathlib import Path
from typing import Any, Mapping

from config_setup.project_config import ConfigError, resolve_path

DEFAULTS_KEY = "qnp_section_defaults"
JOBS_KEY = "qnp_section_inputs"


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """``override`` on top of ``base``; nested mappings merge, everything else is replaced."""

    out = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


class _Missing(dict):
    """format_map context that leaves unknown fields (and their format specs) untouched."""

    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


class _Formatter(string.Formatter):
    def get_field(self, field_name, args, kwargs):
        try:
            return super().get_field(field_name, args, kwargs)
        except (KeyError, AttributeError, IndexError, TypeError):
            return "{" + field_name + "}", field_name


def _fill(value: Any, context: Mapping[str, Any]) -> Any:
    if isinstance(value, str):
        return _Formatter().vformat(value, (), context) if "{" in value else value
    if isinstance(value, Mapping):
        return {k: _fill(v, context) for k, v in value.items()}
    if isinstance(value, list):
        return [_fill(v, context) for v in value]
    return value


def slide_stems(slide_dir: Path) -> dict[str, str]:
    """Stain code (last ``-`` field, upper case) -> slide stem, for every ``.svs`` in the folder."""

    return {path.stem.split("-")[-1].upper(): path.stem for path in sorted(slide_dir.glob("*.svs"))}


def misfit_from_inventory(qc_dir: Path) -> list[str] | None:
    path = qc_dir / "inventory.json"
    if not path.is_file():
        return None
    readiness = json.loads(path.read_text())["readiness"]
    return list(readiness.get("slides_with_misfit_annotations", []))


def resolve_qnp_job(config: Mapping[str, Any], name: str) -> dict[str, Any]:
    """The fully specified job ``name`` with ``_name`` and provenance keys added."""

    jobs = config.get(JOBS_KEY, {})
    if name not in jobs:
        raise ConfigError(f"Unknown QNP job {name!r}. Available: {', '.join(sorted(jobs)) or '<none>'}")
    block = dict(jobs[name])
    defaults = dict(config.get(DEFAULTS_KEY, {}))
    job = deep_merge(defaults, block)
    if "stains" in block:  # an explicit stain list replaces the default list (no per-stain merge)
        job["stains"] = copy.deepcopy(dict(block["stains"]))
    job["_name"] = name
    job.setdefault("donor", name.rsplit("_", 1)[0])

    # Missing keys are tolerated (minimal blocks in tests and one-off scripts); whatever a
    # stage needs and does not find fails there, with the stage's own message.
    base = _Missing(donor=job["donor"], job=name)
    if "region" in job:
        base["region"] = job["region"]
    for key in ("slide_dir", "particle_dir", "qc_dir"):
        if key in job:
            job[key] = _fill(job[key], base)
            base[key] = job[key]

    stems: dict[str, str] = {}
    if "slide_dir" in job:
        slide_dir = resolve_path(config, job["slide_dir"])
        stems = slide_stems(slide_dir) if slide_dir.is_dir() else {}
    if "stains" not in block and stems:
        job["_stains_absent"] = sorted(set(job["stains"]) - set(stems))
        job["stains"] = {s: c for s, c in job["stains"].items() if s in stems}
    job["_slide_stems"] = {s: stems[s] for s in job.get("stains", {}) if s in stems}

    if "annotation_misfit" in block:
        job["_annotation_misfit_source"] = "config"
    else:
        derived = misfit_from_inventory(resolve_path(config, job["qc_dir"])) if "qc_dir" in job else None
        job["annotation_misfit"] = derived or []
        job["_annotation_misfit_source"] = "inventory.json" if derived is not None else "none (inventory not run)"

    reference = job.get("reference_stain", "NEUN")
    structural = [s for s, c in job.get("stains", {}).items() if c.get("structural")]
    base["stem"] = dict(stems)  # a stain without a slide leaves '{stem[X]}' unfilled (visible in any error)
    base["structural"] = "_".join(structural)
    if reference in stems:
        base["reference_stem"] = stems[reference]
    alignment = job.get("alignment", {})
    if "output_dir" in alignment or "particle_dir" in job:
        base["aligned_dir"] = _fill(alignment.get("output_dir", job.get("particle_dir", "") + "/aligned"), base)
    return _fill(job, base)
