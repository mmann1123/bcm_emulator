"""YAML config loader -> dataclass-like namespace."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


class ConfigNamespace:
    """Recursive namespace from nested dict. Supports attribute access."""

    def __init__(self, d: dict):
        for k, v in d.items():
            if isinstance(v, dict):
                setattr(self, k, ConfigNamespace(v))
            else:
                setattr(self, k, v)

    def __repr__(self):
        items = ", ".join(f"{k}={v!r}" for k, v in self.__dict__.items())
        return f"Config({items})"

    def to_dict(self) -> dict:
        out = {}
        for k, v in self.__dict__.items():
            if isinstance(v, ConfigNamespace):
                out[k] = v.to_dict()
            else:
                out[k] = v
        return out


def load_config(path: str) -> ConfigNamespace:
    """Load YAML config file and return as ConfigNamespace."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    cfg = ConfigNamespace(raw)

    # Ensure output directories exist
    for dir_attr in ["checkpoint_dir", "output_dir"]:
        if hasattr(cfg.paths, dir_attr):
            Path(getattr(cfg.paths, dir_attr)).mkdir(parents=True, exist_ok=True)

    return cfg
