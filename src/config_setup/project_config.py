"""Configuration helpers for the Kavli xIV-LDDMM workflow.

The original scripts contained absolute paths tied to a colleague's workstation.  This
module centralizes path handling so that a move to another machine should require
editing only one YAML file.

Path rule
---------
Every relative path in the YAML file is interpreted relative to ``project.root``.
Absolute paths are left unchanged.  ``~`` and environment variables such as
``$HOME`` are expanded.

The module deliberately performs no xIV-LDDMM imports.  This lets a script read the
configuration, insert the configured repository into ``sys.path``, and only then
import ``xmodmap``.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any, Mapping

import yaml


class ConfigError(ValueError):
    """Raised when the setup YAML is missing a required or inconsistent value."""


def _expand_text(value: str) -> str:
    """Expand ``~`` and ``$ENVIRONMENT_VARIABLE`` references in a string."""

    return os.path.expandvars(os.path.expanduser(value))


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load and minimally validate a project YAML file.

    Parameters
    ----------
    config_path:
        Path to the YAML setup file.

    Returns
    -------
    dict
        A deep-copied mutable dictionary with two added private entries:
        ``_config_path`` and ``_project_root``.
    """

    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file does not exist: {path}")

    with path.open("r", encoding="utf-8") as stream:
        loaded = yaml.safe_load(stream)

    if not isinstance(loaded, Mapping):
        raise ConfigError("The top level of the YAML file must be a mapping.")

    config = copy.deepcopy(dict(loaded))
    project = config.get("project", {})
    root_text = project.get("root")
    if not root_text:
        raise ConfigError("Set project.root in the YAML configuration.")

    root = Path(_expand_text(str(root_text)))
    if not root.is_absolute():
        # A relative project.root is interpreted relative to the YAML file itself.
        root = path.parent / root
    root = root.resolve()

    config["_config_path"] = str(path)
    config["_project_root"] = str(root)
    return config


def project_root(config: Mapping[str, Any]) -> Path:
    """Return the resolved project root."""

    return Path(str(config["_project_root"]))


def resolve_path(config: Mapping[str, Any], value: str | Path) -> Path:
    """Resolve a configured path against ``project.root``.

    This function does not require the path to exist.  Call ``require_file`` or
    ``require_dir`` when existence is part of the operation's contract.
    """

    path = Path(_expand_text(str(value)))
    if not path.is_absolute():
        path = project_root(config) / path
    return path.resolve()


def require_file(config: Mapping[str, Any], value: str | Path, label: str) -> Path:
    """Resolve a path and raise an informative error unless it is a file."""

    path = resolve_path(config, value)
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist or is not a file: {path}")
    return path


def require_dir(config: Mapping[str, Any], value: str | Path, label: str) -> Path:
    """Resolve a path and raise an informative error unless it is a directory."""

    path = resolve_path(config, value)
    if not path.is_dir():
        raise FileNotFoundError(f"{label} does not exist or is not a directory: {path}")
    return path


def get_named(config: Mapping[str, Any], section: str, name: str) -> dict[str, Any]:
    """Return a named configuration record with a useful error on misspelling."""

    records = config.get(section, {})
    if name not in records:
        available = ", ".join(sorted(records)) or "<none>"
        raise ConfigError(
            f"No entry named '{name}' in '{section}'. Available entries: {available}"
        )
    record = records[name]
    if not isinstance(record, Mapping):
        raise ConfigError(f"Configuration entry {section}.{name} must be a mapping.")
    return copy.deepcopy(dict(record))


def write_resolved_config(config: Mapping[str, Any], destination: str | Path) -> None:
    """Write a JSON snapshot of the configuration used for a result.

    A result should remain interpretable even after the main YAML file changes.  The
    snapshot is JSON rather than YAML so that it is unambiguous and easy to inspect
    from Python, R, MATLAB, or the command line.
    """

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as stream:
        json.dump(dict(config), stream, indent=2, sort_keys=True, default=str)
