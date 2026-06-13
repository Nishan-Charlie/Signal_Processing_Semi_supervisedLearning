"""Configuration system: YAML-backed, dot-accessible, with inheritance and CLI overrides.

Configs support a ``_base_`` key for composition (a config can inherit one or more
parent configs and override selected fields), and command-line overrides of the form
``key.subkey=value`` so any field can be swept without editing files.
"""

from __future__ import annotations

import ast
import copy
import os
from typing import Any, Dict, List, MutableMapping, Sequence

import yaml


class Config(dict):
    """A dict that also supports attribute access and recursive wrapping."""

    def __getattr__(self, item: str) -> Any:
        try:
            value = self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc
        return Config(value) if isinstance(value, dict) else value

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def get(self, key: str, default: Any = None) -> Any:  # type: ignore[override]
        value = super().get(key, default)
        return Config(value) if isinstance(value, dict) else value

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k, v in self.items():
            out[k] = v.to_dict() if isinstance(v, Config) else v
        return out


def _deep_merge(base: MutableMapping, override: MutableMapping) -> Dict[str, Any]:
    """Recursively merge ``override`` into ``base`` (override wins on conflicts)."""
    result = copy.deepcopy(dict(base))
    for key, value in override.items():
        if key in result and isinstance(result[key], MutableMapping) and isinstance(value, MutableMapping):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_config(path: str) -> Config:
    """Load a YAML config, resolving ``_base_`` inheritance relative to the file."""
    raw = _load_yaml(path)
    bases = raw.pop("_base_", [])
    if isinstance(bases, str):
        bases = [bases]

    merged: Dict[str, Any] = {}
    for base in bases:
        base_path = base if os.path.isabs(base) else os.path.join(os.path.dirname(path), base)
        merged = _deep_merge(merged, load_config(base_path))
    merged = _deep_merge(merged, raw)
    return Config(merged)


def save_config(config: Config | Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    data = config.to_dict() if isinstance(config, Config) else config
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, sort_keys=False, default_flow_style=False)


def _parse_value(value: str) -> Any:
    """Parse a CLI string into a Python literal where possible (int/float/bool/list)."""
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def merge_overrides(config: Config, overrides: Sequence[str]) -> Config:
    """Apply ``a.b.c=value`` style overrides to a config in place and return it.

    Example: ``["method.threshold=0.95", "optim.lr=1e-3", "seed=7"]``.
    """
    cfg = copy.deepcopy(config.to_dict())
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override '{item}' must be of the form key.subkey=value")
        key, value = item.split("=", 1)
        parsed = _parse_value(value)
        node: Dict[str, Any] = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                raise ValueError(f"Cannot set '{key}': '{part}' is not a mapping.")
        node[parts[-1]] = parsed
    return Config(cfg)
