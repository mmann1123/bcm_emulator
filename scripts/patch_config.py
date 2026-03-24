#!/usr/bin/env python3
"""Surgical YAML config patcher for tuning sweep.

Usage:
    python scripts/patch_config.py config.yaml training.loss_weights.aet_initial 2.0
    python scripts/patch_config.py config.yaml training.loss_weights.extreme_vars "[aet,pck]"

Supports:
    - Dotted key paths (e.g. training.loss_weights.aet_initial)
    - Scalar values (float, int, bool, string)
    - Simple lists written as [a,b,c]
    - Preserves all other config values and comments where possible
"""

import sys
import re


def parse_value(val_str: str):
    """Parse a string value into the appropriate Python type."""
    # List: [aet,pck] or ["aet","pck"]
    if val_str.startswith("[") and val_str.endswith("]"):
        inner = val_str[1:-1]
        items = [v.strip().strip('"').strip("'") for v in inner.split(",") if v.strip()]
        return items

    # Bool
    if val_str.lower() == "true":
        return True
    if val_str.lower() == "false":
        return False

    # Null
    if val_str.lower() in ("null", "none", "~"):
        return None

    # Int
    try:
        return int(val_str)
    except ValueError:
        pass

    # Float
    try:
        return float(val_str)
    except ValueError:
        pass

    # String
    return val_str


def set_nested(data: dict, key_path: str, value):
    """Set a nested dict value using dotted key path."""
    keys = key_path.split(".")
    d = data
    for k in keys[:-1]:
        if k not in d:
            raise KeyError(f"Key not found in config: '{k}' (in path '{key_path}')")
        d = d[k]
    final_key = keys[-1]
    if final_key not in d:
        raise KeyError(f"Key not found in config: '{final_key}' (in path '{key_path}')")
    old_val = d[final_key]
    d[final_key] = value
    return old_val


def main():
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} config.yaml key.path value")
        sys.exit(1)

    config_path, key_path, val_str = sys.argv[1], sys.argv[2], sys.argv[3]
    value = parse_value(val_str)

    # Use ruamel.yaml if available for comment-preserving round-trip
    try:
        from ruamel.yaml import YAML
        yaml = YAML()
        yaml.preserve_quotes = True
        yaml.width = 4096  # prevent line wrapping

        with open(config_path, "r") as f:
            data = yaml.load(f)

        old_val = set_nested(data, key_path, value)

        with open(config_path, "w") as f:
            yaml.dump(data, f)

        print(f"  patched {key_path}: {old_val!r} → {value!r}")

    except ImportError:
        # Fallback: PyYAML (loses comments)
        import yaml

        with open(config_path, "r") as f:
            data = yaml.safe_load(f)

        old_val = set_nested(data, key_path, value)

        with open(config_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

        print(f"  patched {key_path}: {old_val!r} → {value!r}  (ruamel.yaml not installed — comments lost)")


if __name__ == "__main__":
    main()
