from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config and resolve an optional local `inherits` field."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    parent_name = cfg.pop("inherits", None)
    if parent_name is None:
        return cfg

    parent_path = config_path.parent / parent_name
    parent_cfg = load_config(parent_path)
    return _deep_update(parent_cfg, cfg)


def summarize_config(cfg: dict[str, Any]) -> str:
    name = cfg.get("experiment_name", "unnamed_experiment")
    model = cfg.get("model", {}).get("name", "unknown_model")
    return f"{name} ({model})"

