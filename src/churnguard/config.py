"""Configuration loading.

All stages read `params.yaml`. Paths in the file are relative to the project root,
which is the directory holding `params.yaml` (overridable with CHURNGUARD_ROOT).
"""

from __future__ import annotations

import copy
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    env = os.getenv("CHURNGUARD_ROOT")
    if env:
        return Path(env).resolve()
    here = Path.cwd().resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "params.yaml").exists():
            return candidate
    # Fallback: repository root relative to this file (src/churnguard/config.py)
    return Path(__file__).resolve().parents[2]


@lru_cache(maxsize=4)
def _load(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_params(path: str | Path | None = None) -> dict[str, Any]:
    """Return a deep copy of the parameters so callers can override safely."""
    path = Path(path) if path else project_root() / "params.yaml"
    return copy.deepcopy(_load(str(path)))


def resolve(relative: str | Path) -> Path:
    """Resolve a path from params.yaml against the project root."""
    p = Path(relative)
    return p if p.is_absolute() else project_root() / p
