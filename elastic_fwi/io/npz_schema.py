from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

_ALLOWED_KINDS = {
    "vector_array_residual_npz",
    "vector_array_residual_npz_minimal",
    "vector_array_residual_series",
}


def _to_scalar_string(value: Any) -> str:
    arr = np.asarray(value)
    if arr.shape == ():
        return str(arr.item())
    if arr.size == 1:
        return str(arr.reshape(-1)[0])
    return str(value)


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


def _as_finite_array(value: Any, *, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values.")
    return arr


def _parse_metadata(data: Any) -> dict[str, Any]:
    if data is None:
        return {}
    try:
        text = _to_scalar_string(data)
        if not text:
            return {}
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            raise ValueError("metadata_json must encode a JSON object.")
        return parsed
    except Exception as exc:
        raise ValueError("Could not parse metadata_json from NPZ file.") from exc


def validate_vector_array_residual_npz(path: str | Path) -> dict[str, Any]:
    """Validate a vector-array residual NPZ and return a compact schema summary.

    Required fields are ``times`` and ``values``. Existing full residual exports
    with ``kind='vector_array_residual_series'`` are accepted, as are minimal
    files written by :func:`write_vector_array_residual_npz`.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing vector-array residual NPZ: {path}")
    if not path.is_file():
        raise ValueError(f"NPZ path must be a file: {path}")

    with np.load(path, allow_pickle=False) as data:
        files = set(data.files)
        missing = [key for key in ("times", "values") if key not in files]
        if missing:
            raise ValueError(f"Missing required NPZ keys in {path}: {missing}")

        kind = _to_scalar_string(data["kind"]) if "kind" in files else "vector_array_residual_npz"
        if kind not in _ALLOWED_KINDS:
            raise ValueError(f"Unexpected vector-array residual NPZ kind in {path}: {kind!r}")

        times = _as_finite_array(data["times"], name="times").reshape(-1)
        values = _as_finite_array(data["values"], name="values")

        if values.ndim != 3 or values.shape[-1] != 2:
            raise ValueError(f"{path}: values must have shape (num_steps, num_receivers, 2).")
        if times.shape != (values.shape[0],):
            raise ValueError(f"{path}: times length must match values.shape[0].")
        if times.size and not np.all(np.diff(times) >= 0.0):
            raise ValueError(f"{path}: times must be non-decreasing.")

        receiver_coordinates_shape: tuple[int, ...] | None = None
        if "receiver_coordinates" in files:
            receiver_coordinates = _as_finite_array(
                data["receiver_coordinates"],
                name="receiver_coordinates",
            )
            if receiver_coordinates.shape != (values.shape[1], 2):
                raise ValueError(
                    f"{path}: receiver_coordinates must have shape (num_receivers, 2)."
                )
            receiver_coordinates_shape = tuple(int(v) for v in receiver_coordinates.shape)

        metadata_keys: list[str] = []
        if "metadata_json" in files:
            metadata_keys = sorted(_parse_metadata(data["metadata_json"]).keys())

        return {
            "kind": kind,
            "path": str(path),
            "num_steps": int(values.shape[0]),
            "num_receivers": int(values.shape[1]),
            "num_components": int(values.shape[2]),
            "times_shape": tuple(int(v) for v in times.shape),
            "values_shape": tuple(int(v) for v in values.shape),
            "has_receiver_coordinates": receiver_coordinates_shape is not None,
            "receiver_coordinates_shape": receiver_coordinates_shape,
            "has_metadata": bool(metadata_keys),
            "metadata_keys": metadata_keys,
        }


def load_vector_array_residual_npz(path: str | Path) -> dict[str, Any]:
    """Load a validated vector-array residual NPZ using the shared schema."""
    path = Path(path)
    validation = validate_vector_array_residual_npz(path)

    with np.load(path, allow_pickle=False) as data:
        files = set(data.files)
        metadata = _parse_metadata(data["metadata_json"]) if "metadata_json" in files else {}
        receiver_coordinates = (
            _as_finite_array(data["receiver_coordinates"], name="receiver_coordinates")
            if "receiver_coordinates" in files
            else None
        )

        payload: dict[str, Any] = {
            "path": str(path),
            "kind": validation["kind"],
            "times": _as_finite_array(data["times"], name="times").reshape(-1),
            "values": _as_finite_array(data["values"], name="values"),
            "receiver_coordinates": receiver_coordinates,
            "metadata": metadata,
            "validation": validation,
        }

        for key in ("label", "name", "array_name", "role"):
            if key in files:
                payload[key] = _to_scalar_string(data[key])

        return payload


def write_vector_array_residual_npz(
    path: str | Path,
    times: Any,
    values: Any,
    metadata: Mapping[str, Any] | None = None,
    *,
    receiver_coordinates: Any | None = None,
    label: str = "residual",
) -> Path:
    """Write a minimal pickle-free vector-array residual NPZ."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    times_arr = _as_finite_array(times, name="times").reshape(-1)
    values_arr = _as_finite_array(values, name="values")

    if values_arr.ndim != 3 or values_arr.shape[-1] != 2:
        raise ValueError("values must have shape (num_steps, num_receivers, 2).")
    if times_arr.shape != (values_arr.shape[0],):
        raise ValueError("times length must match values.shape[0].")
    if times_arr.size and not np.all(np.diff(times_arr) >= 0.0):
        raise ValueError("times must be non-decreasing.")

    arrays: dict[str, Any] = {
        "kind": np.asarray("vector_array_residual_npz", dtype=np.str_),
        "label": np.asarray(str(label), dtype=np.str_),
        "times": np.asarray(times_arr, dtype=np.float64),
        "values": np.asarray(values_arr, dtype=np.float64),
        "num_steps": np.asarray(values_arr.shape[0], dtype=np.int64),
        "num_receivers": np.asarray(values_arr.shape[1], dtype=np.int64),
        "num_components": np.asarray(values_arr.shape[2], dtype=np.int64),
        "components": np.asarray(["x", "y"], dtype=np.str_),
        "metadata_json": np.asarray(
            json.dumps(_to_jsonable(dict(metadata or {})), sort_keys=True),
            dtype=np.str_,
        ),
    }

    if receiver_coordinates is not None:
        receiver_coordinates_arr = _as_finite_array(
            receiver_coordinates,
            name="receiver_coordinates",
        )
        if receiver_coordinates_arr.shape != (values_arr.shape[1], 2):
            raise ValueError("receiver_coordinates must have shape (num_receivers, 2).")
        arrays["receiver_coordinates"] = np.asarray(receiver_coordinates_arr, dtype=np.float64)

    np.savez(path, **arrays)
    validate_vector_array_residual_npz(path)
    return path
