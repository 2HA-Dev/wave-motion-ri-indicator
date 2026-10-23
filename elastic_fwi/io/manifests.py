from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def read_manifest(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing manifest: {path}")
    if not path.is_file():
        raise ValueError(f"Manifest path must be a file: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Manifest must contain a JSON object: {path}")
    return payload


def write_manifest(path: str | Path, payload: Mapping[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_to_jsonable(dict(payload)), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return path


def resolve_manifest_path(
    manifest_path: str | Path,
    relative_or_absolute_path: str | Path,
    *,
    must_exist: bool = False,
) -> Path:
    """Resolve a path stored in a manifest relative to common manifest roots."""
    manifest_path = Path(manifest_path)
    raw = Path(relative_or_absolute_path)

    if raw.is_absolute():
        resolved = raw
    else:
        candidates = [raw, manifest_path.parent / raw, manifest_path.parent.parent / raw]
        resolved = candidates[0]
        for candidate in candidates:
            if candidate.exists():
                resolved = candidate
                break
        else:
            resolved = manifest_path.parent / raw

    if must_exist and not resolved.exists():
        raise FileNotFoundError(f"Path referenced by manifest does not exist: {resolved}")

    return resolved.resolve() if resolved.exists() else resolved


def _lookup_dotted_key(payload: Mapping[str, Any], dotted_key: str) -> Any:
    current: Any = payload
    for part in str(dotted_key).split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise KeyError(dotted_key)
        current = current[part]
    return current


def require_manifest_keys(manifest: Mapping[str, Any], keys: Iterable[str]) -> dict[str, Any]:
    """Require top-level or dotted keys and return their resolved values."""
    missing: list[str] = []
    resolved: dict[str, Any] = {}

    for key in keys:
        key_str = str(key)
        try:
            resolved[key_str] = _lookup_dotted_key(manifest, key_str)
        except KeyError:
            missing.append(key_str)

    if missing:
        raise ValueError(f"Missing required manifest keys: {missing}")

    return resolved
