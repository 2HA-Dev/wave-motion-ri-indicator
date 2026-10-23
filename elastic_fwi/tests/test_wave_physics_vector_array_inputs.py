from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from elastic_fwi.wave_physics.vector_array_inputs import (
    load_vector_array_residual_pair_from_manifest,
    load_vector_array_residual_series_npz,
    resolve_vector_array_residual_npz_paths,
)


def _write_residual_npz(path: Path, *, label: str, values: np.ndarray) -> None:
    times = np.linspace(0.0, 1.0, values.shape[0])
    np.savez(
        path,
        kind=np.asarray("vector_array_residual_series"),
        label=np.asarray(label),
        name=np.asarray(f"{label}_residual"),
        array_name=np.asarray(f"{label}_array"),
        role=np.asarray(label),
        scale=np.asarray(1.0),
        num_steps=np.asarray(values.shape[0]),
        num_receivers=np.asarray(values.shape[1]),
        num_components=np.asarray(values.shape[2]),
        steps=np.arange(values.shape[0]),
        times=times,
        values=values,
        reference_values=values,
        observed_values=np.zeros_like(values),
        components=np.asarray(["x", "y"]),
        max_abs_residual=np.asarray(float(np.max(np.abs(values)))),
        l2_norm=np.asarray(float(np.linalg.norm(values))),
        l2_norm_squared=np.asarray(float(np.sum(values * values))),
    )


def test_load_vector_array_residual_series_npz(tmp_path):
    path = tmp_path / "reflection.npz"
    values = np.ones((4, 3, 2), dtype=float)
    _write_residual_npz(path, label="reflection", values=values)

    loaded = load_vector_array_residual_series_npz(path)

    assert loaded["label"] == "reflection"
    assert loaded["num_steps"] == 4
    assert loaded["num_receivers"] == 3
    assert loaded["num_components"] == 2
    assert np.allclose(loaded["values"], values)


def test_resolve_vector_array_residual_paths_from_inverse_manifest(tmp_path):
    vector_dir = tmp_path / "vector_array"
    vector_dir.mkdir()

    reflection = vector_dir / "vector_array_reflection_residuals.npz"
    transmission = vector_dir / "vector_array_transmission_residuals.npz"

    _write_residual_npz(reflection, label="reflection", values=np.ones((5, 2, 2)))
    _write_residual_npz(transmission, label="transmission", values=2.0 * np.ones((5, 3, 2)))

    manifest = {
        "kind": "inverse_time_series_xdmf",
        "exports": {
            "vector_array": {
                "residual_npz": {
                    "reflection": str(reflection),
                    "transmission": str(transmission),
                }
            }
        },
    }

    manifest_path = tmp_path / "inverse_time_series_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    paths = resolve_vector_array_residual_npz_paths(manifest_path)

    assert paths["reflection"] == reflection
    assert paths["transmission"] == transmission


def test_load_pair_from_manifest(tmp_path):
    reflection = tmp_path / "reflection.npz"
    transmission = tmp_path / "transmission.npz"

    _write_residual_npz(reflection, label="reflection", values=np.ones((6, 2, 2)))
    _write_residual_npz(transmission, label="transmission", values=2.0 * np.ones((6, 4, 2)))

    manifest = {
        "kind": "vector_array_export_manifest",
        "files": {
            "residual_npz": {
                "reflection": str(reflection),
                "transmission": str(transmission),
            }
        },
    }

    manifest_path = tmp_path / "vector_array_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    pair = load_vector_array_residual_pair_from_manifest(manifest_path)

    assert pair["kind"] == "vector_array_residual_pair_from_manifest"
    assert pair["times"].shape == (6,)
    assert pair["reflection"]["values"].shape == (6, 2, 2)
    assert pair["transmission"]["values"].shape == (6, 4, 2)
    assert pair["metadata"]["num_components"] == 2
