import copy
import pathlib
from typing import Any, Dict, Iterable

import yaml


def _deep_update(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in update.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def _parse_override(raw: str) -> tuple[list[str], Any]:
    if "=" not in raw:
        raise ValueError(f"Override must be key=value, got: {raw}")
    key, value = raw.split("=", 1)
    key_path = [part for part in key.strip().split(".") if part]
    if not key_path:
        raise ValueError(f"Invalid override key: {raw}")
    parsed_value = yaml.safe_load(value)
    return key_path, parsed_value


def _set_nested(cfg: Dict[str, Any], key_path: list[str], value: Any) -> None:
    cursor = cfg
    for part in key_path[:-1]:
        if part not in cursor or not isinstance(cursor[part], dict):
            cursor[part] = {}
        cursor = cursor[part]
    cursor[key_path[-1]] = value


def load_config(path: str | pathlib.Path, overrides: Iterable[str] | None = None) -> Dict[str, Any]:
    path = pathlib.Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg = copy.deepcopy(cfg)
    for raw in overrides or []:
        key_path, value = _parse_override(raw)
        _set_nested(cfg, key_path, value)
    return cfg


def merge_dicts(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for d in dicts:
        _deep_update(out, d)
    return out
