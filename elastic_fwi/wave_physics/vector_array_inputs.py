from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def _to_scalar_string(value: Any) -> str:
    arr = np.asarray(value)
    if arr.shape == ():
        return str(arr.item())
    if arr.size == 1:
        return str(arr.reshape(-1)[0])
    return str(value)


def _as_finite_array(value: Any, *, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values.")
    return arr


def resolve_manifest_path(raw_path: str | Path, *, manifest_dir: Path) -> Path:
    path = Path(raw_path)

    if path.is_absolute():
        return path

    candidates = [
        path,
        manifest_dir / path,
        manifest_dir.parent / path,
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return path


def load_vector_array_residual_series_npz(path: str | Path) -> dict[str, Any]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Missing vector-array residual NPZ: {path}")

    with np.load(path, allow_pickle=False) as data:
        required = [
            "kind",
            "label",
            "times",
            "values",
            "reference_values",
            "observed_values",
        ]
        missing = [key for key in required if key not in data.files]
        if missing:
            raise ValueError(f"Missing keys in {path}: {missing}")

        kind = _to_scalar_string(data["kind"])
        if kind != "vector_array_residual_series":
            raise ValueError(f"Unexpected NPZ kind in {path}: {kind!r}")

        times = _as_finite_array(data["times"], name=f"{path}:times").reshape(-1)
        values = _as_finite_array(data["values"], name=f"{path}:values")
        reference_values = _as_finite_array(
            data["reference_values"],
            name=f"{path}:reference_values",
        )
        observed_values = _as_finite_array(
            data["observed_values"],
            name=f"{path}:observed_values",
        )

        if values.ndim != 3 or values.shape[-1] != 2:
            raise ValueError(
                f"{path}: values must have shape (num_steps, num_receivers, 2)."
            )

        if reference_values.shape != values.shape:
            raise ValueError(f"{path}: reference_values shape mismatch.")

        if observed_values.shape != values.shape:
            raise ValueError(f"{path}: observed_values shape mismatch.")

        if times.shape != (values.shape[0],):
            raise ValueError(f"{path}: times length does not match values.")

        return {
            "path": str(path),
            "kind": kind,
            "label": _to_scalar_string(data["label"]),
            "name": _to_scalar_string(data["name"]) if "name" in data.files else "",
            "array_name": _to_scalar_string(data["array_name"])
            if "array_name" in data.files
            else "",
            "role": _to_scalar_string(data["role"]) if "role" in data.files else "",
            "times": times,
            "values": values,
            "reference_values": reference_values,
            "observed_values": observed_values,
            "num_steps": int(values.shape[0]),
            "num_receivers": int(values.shape[1]),
            "num_components": int(values.shape[2]),
        }


def _candidate_residual_npz_dicts(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []

    if isinstance(manifest.get("residual_npz"), dict):
        candidates.append(manifest["residual_npz"])

    files = manifest.get("files")
    if isinstance(files, dict) and isinstance(files.get("residual_npz"), dict):
        candidates.append(files["residual_npz"])

    exports = manifest.get("exports")
    if isinstance(exports, dict):
        vector_array = exports.get("vector_array")
        if isinstance(vector_array, dict):
            if isinstance(vector_array.get("residual_npz"), dict):
                candidates.append(vector_array["residual_npz"])

            va_files = vector_array.get("files")
            if isinstance(va_files, dict) and isinstance(va_files.get("residual_npz"), dict):
                candidates.append(va_files["residual_npz"])

    return candidates


def resolve_vector_array_residual_npz_paths(
    manifest_path: str | Path,
) -> dict[str, Path]:
    manifest_path = Path(manifest_path)

    if not str(manifest_path):
        raise ValueError("Empty manifest path.")

    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing inverse/vector-array manifest: {manifest_path}")

    if not manifest_path.is_file():
        raise ValueError(f"Manifest path must be a file, got: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    for candidate in _candidate_residual_npz_dicts(manifest):
        if "reflection" in candidate and "transmission" in candidate:
            return {
                "reflection": resolve_manifest_path(
                    candidate["reflection"],
                    manifest_dir=manifest_path.parent,
                ),
                "transmission": resolve_manifest_path(
                    candidate["transmission"],
                    manifest_dir=manifest_path.parent,
                ),
            }

    raise ValueError(
        "Could not find reflection/transmission vector-array residual NPZ paths "
        f"in manifest: {manifest_path}"
    )


def load_vector_array_residual_pair_from_manifest(
    manifest_path: str | Path,
) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    paths = resolve_vector_array_residual_npz_paths(manifest_path)

    reflection = load_vector_array_residual_series_npz(paths["reflection"])
    transmission = load_vector_array_residual_series_npz(paths["transmission"])

    if reflection["values"].shape[-1] != 2 or transmission["values"].shape[-1] != 2:
        raise ValueError("Residual values must contain two vector components.")

    if not np.allclose(reflection["times"], transmission["times"], rtol=0.0, atol=1.0e-14):
        raise ValueError("Reflection and transmission residuals must share the same times.")

    return {
        "kind": "vector_array_residual_pair_from_manifest",
        "manifest_path": str(manifest_path),
        "times": reflection["times"],
        "reflection": reflection,
        "transmission": transmission,
        "metadata": {
            "input_manifest": str(manifest_path),
            "reflection_npz": str(paths["reflection"]),
            "transmission_npz": str(paths["transmission"]),
            "source_type": "vector_array_residual_npz",
            "residual_convention": "values already contain reference_minus_observed residuals",
            "num_steps": int(reflection["values"].shape[0]),
            "num_reflection_receivers": int(reflection["values"].shape[1]),
            "num_transmission_receivers": int(transmission["values"].shape[1]),
            "num_components": 2,
        },
    }
