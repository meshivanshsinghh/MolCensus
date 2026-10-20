"""Load the YAML config and resolve every path relative to the repository root."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

# src/molcensus/config.py -> repository root is two levels above the package.
REPO_ROOT = Path(__file__).resolve().parents[2]


def load_dotenv(env_file: Path) -> None:
    """Read KEY=VALUE lines from a .env file without overriding variables already set."""
    if not env_file.exists():
        return
    for raw_line in env_file.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def expand(value: str) -> str:
    """Expand ~ and ${VARS}. An unset variable is returned as an empty string."""
    expanded = os.path.expandvars(os.path.expanduser(value))
    return "" if "$" in expanded else expanded


def resolve_path(value: str | None) -> Path | None:
    """Turn a config value into an absolute path. Relative paths start at the repo root."""
    if not value:
        return None
    text = expand(str(value))
    if not text:
        return None
    path = Path(text)
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def _merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        both_dicts = isinstance(value, dict) and isinstance(base.get(key), dict)
        merged[key] = _merge(base[key], value) if both_dicts else value
    return merged


def load_config(config_file: str | Path) -> dict[str, Any]:
    """Load the YAML config. Every entry under `paths` becomes an absolute Path.

    A config can start with `extends: other.yaml` and list only the settings it changes.
    """
    load_dotenv(REPO_ROOT / ".env")
    config_path = resolve_path(str(config_file))
    if config_path is None or not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    with open(config_path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if "extends" in config:
        with open(config_path.parent / config.pop("extends"), encoding="utf-8") as handle:
            config = _merge(yaml.safe_load(handle), config)

    config["paths"] = {name: resolve_path(value) for name, value in config.get("paths", {}).items()}
    config["papers"] = [int(paper) for paper in config.get("papers", [])]
    config["config_file"] = config_path
    config["repo_root"] = REPO_ROOT
    return config
