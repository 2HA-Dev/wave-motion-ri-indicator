from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from elastic_fwi.inversion.vector_array_residuals import (
    VectorArrayResidualPair,
    VectorArrayResidualSeries,
)
from elastic_fwi.observations.vector_array import VectorArrayObservationTimeSeries


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, Path):
        return str(value)

    return value


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def export_vector_array_time_series_npz(
    *,
    path: str | Path,
    series: VectorArrayObservationTimeSeries,
    label: str,
) -> Path:
    """Export one vector-array observation time series to NPZ.

    The NPZ is pickle-free and contains compact arrays only.
    """
    path = Path(path)
    _ensure_parent(path)

    centers = np.asarray(series.spec.centers, dtype=np.float64)

    np.savez(
        path,
        kind=np.asarray("vector_array_observation_time_series", dtype=np.str_),
        label=np.asarray(str(label), dtype=np.str_),
        observation_name=np.asarray(series.spec.name, dtype=np.str_),
        array_name=np.asarray(series.spec.array.name, dtype=np.str_),
        role=np.asarray(series.spec.role, dtype=np.str_),
        sigma=np.asarray(series.spec.sigma, dtype=np.float64),
        quadrature_degree=np.asarray(series.spec.quadrature_degree, dtype=np.int64),
        num_steps=np.asarray(series.num_steps, dtype=np.int64),
        num_receivers=np.asarray(series.num_receivers, dtype=np.int64),
        steps=np.asarray(series.steps, dtype=np.int64),
        times=np.asarray(series.times, dtype=np.float64),
        centers=centers,
        values=np.asarray(series.values, dtype=np.float64),
        components=np.asarray(["x", "y"], dtype=np.str_),
    )

    return path


def export_vector_array_residual_series_npz(
    *,
    path: str | Path,
    residual: VectorArrayResidualSeries,
    label: str,
) -> Path:
    """Export one vector-array residual series to NPZ."""
    path = Path(path)
    _ensure_parent(path)

    np.savez(
        path,
        kind=np.asarray("vector_array_residual_series", dtype=np.str_),
        label=np.asarray(str(label), dtype=np.str_),
        name=np.asarray(residual.name, dtype=np.str_),
        array_name=np.asarray(residual.array_name, dtype=np.str_),
        role=np.asarray(residual.role, dtype=np.str_),
        scale=np.asarray(residual.scale, dtype=np.float64),
        num_steps=np.asarray(residual.num_steps, dtype=np.int64),
        num_receivers=np.asarray(residual.num_receivers, dtype=np.int64),
        num_components=np.asarray(residual.num_components, dtype=np.int64),
        steps=np.asarray(residual.steps, dtype=np.int64),
        times=np.asarray(residual.times, dtype=np.float64),
        values=np.asarray(residual.values, dtype=np.float64),
        reference_values=np.asarray(residual.reference_values, dtype=np.float64),
        observed_values=np.asarray(residual.observed_values, dtype=np.float64),
        components=np.asarray(["x", "y"], dtype=np.str_),
        max_abs_residual=np.asarray(residual.max_abs_residual, dtype=np.float64),
        l2_norm=np.asarray(residual.l2_norm, dtype=np.float64),
        l2_norm_squared=np.asarray(residual.l2_norm_squared, dtype=np.float64),
    )

    return path


def export_vector_array_residual_pair_npz(
    *,
    output_dir: str | Path,
    residuals: VectorArrayResidualPair,
) -> dict[str, str]:
    """Export reflection/transmission residuals as separate NPZ files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result: dict[str, str] = {}

    if residuals.reflection is not None:
        result["reflection"] = str(
            export_vector_array_residual_series_npz(
                path=output_dir / "vector_array_reflection_residuals.npz",
                residual=residuals.reflection,
                label="reflection",
            )
        )

    if residuals.transmission is not None:
        result["transmission"] = str(
            export_vector_array_residual_series_npz(
                path=output_dir / "vector_array_transmission_residuals.npz",
                residual=residuals.transmission,
                label="transmission",
            )
        )

    return result


def export_vector_array_residual_pair_csv(
    *,
    path: str | Path,
    residuals: VectorArrayResidualPair,
) -> Path:
    """Export vector residuals in long CSV format.

    Columns:
    time, step, array, role, receiver_id, component, residual, reference, observed.
    """
    path = Path(path)
    _ensure_parent(path)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "time",
                "step",
                "array",
                "role",
                "receiver_id",
                "component",
                "residual",
                "reference",
                "observed",
            ],
        )
        writer.writeheader()

        for residual in (residuals.reflection, residuals.transmission):
            if residual is None:
                continue

            for k in range(residual.num_steps):
                for j in range(residual.num_receivers):
                    for c, component in enumerate(("x", "y")):
                        writer.writerow(
                            {
                                "time": float(residual.times[k]),
                                "step": int(residual.steps[k]),
                                "array": residual.array_name,
                                "role": residual.role,
                                "receiver_id": int(j),
                                "component": component,
                                "residual": float(residual.values[k, j, c]),
                                "reference": float(residual.reference_values[k, j, c]),
                                "observed": float(residual.observed_values[k, j, c]),
                            }
                        )

    return path


def write_vector_array_manifest(
    *,
    path: str | Path,
    payload: dict[str, Any],
) -> Path:
    """Write a JSON manifest for vector-array observation artifacts.

    The manifest is enriched with tracking/provenance sidecars so that
    vector-array residual exports satisfy strict reproducibility checks.
    """
    path = Path(path)
    _ensure_parent(path)

    manifest = dict(payload)

    try:
        from elastic_fwi.tracking.tracked_manifest import enrich_manifest_with_tracking

        manifest = enrich_manifest_with_tracking(
            manifest,
            output_dir=path.parent,
            args={
                "manifest_kind": "vector_array_export_manifest",
                "manifest_path": str(path),
            },
        )
    except Exception as exc:
        manifest["tracking_error"] = repr(exc)

    with path.open("w", encoding="utf-8") as handle:
        json.dump(_to_jsonable(manifest), handle, indent=2, sort_keys=True)

    return path


def build_vector_array_export_manifest(
    *,
    residuals: VectorArrayResidualPair,
    files: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a compact manifest for vector-array exports."""
    manifest: dict[str, Any] = {
        "kind": "vector_array_export_manifest",
        "observation_kind": "vector_array",
        "residuals": residuals.as_dict(),
        "files": files,
        "scientific_role": "primary_vector_array_observation_chain",
    }

    if extra:
        manifest["extra"] = extra

    return manifest
