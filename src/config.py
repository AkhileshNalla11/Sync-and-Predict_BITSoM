"""Config loading shared by every pipeline stage."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load config.yaml (defaults to the repo root copy)."""
    path = Path(path) if path else ROOT / "config.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def runtime_config() -> dict[str, Any]:
    """Use the full pipeline outputs if present, else the committed demo bundle (e.g. on a cloud host).

    Set TWIN_USE_DEMO_BUNDLE=1 to force the bundle. Used by the dashboard and the FHIR API.
    """
    cfg = load_config()
    local = ROOT / cfg["paths"]["processed_dir"] / "series.npz"
    bundle = ROOT / "demo_bundle" / "processed" / "series.npz"
    if os.environ.get("TWIN_USE_DEMO_BUNDLE") == "1" or (not local.exists() and bundle.exists()):
        cfg["paths"] = {"raw_dir": "demo_bundle/raw", "processed_dir": "demo_bundle/processed",
                        "models_dir": "demo_bundle/processed/models", "docs_dir": cfg["paths"]["docs_dir"]}
    return cfg


def resolve_path(relative: str | Path) -> Path:
    """Resolve a config path relative to the repo root, creating the directory if needed."""
    p = ROOT / relative
    p.mkdir(parents=True, exist_ok=True)
    return p


def steps_per_hour(cfg: dict[str, Any]) -> int:
    return 60 // cfg["sampling"]["interval_minutes"]
