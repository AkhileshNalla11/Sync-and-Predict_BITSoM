"""Config loading shared by every pipeline stage."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load config.yaml (defaults to the repo root copy)."""
    path = Path(path) if path else ROOT / "config.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(relative: str | Path) -> Path:
    """Resolve a config path relative to the repo root, creating the directory if needed."""
    p = ROOT / relative
    p.mkdir(parents=True, exist_ok=True)
    return p


def steps_per_hour(cfg: dict[str, Any]) -> int:
    return 60 // cfg["sampling"]["interval_minutes"]
