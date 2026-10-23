from __future__ import annotations

import json
from pathlib import Path

from elastic_fwi.diagnostics.check_vector_array_export_manifest import (
    check_vector_array_export_manifest,
)


def test_check_vector_array_export_manifest_accepts_valid_minimal_manifest(tmp_path: Path):
    vector_dir = tmp_path / "vector_array"
    vector_dir.mkdir()

    files = {
        "manifest": vector_dir / "vector_array_manifest.json",
        "residual_csv": vector_dir / "vector_array_residuals.csv",
        "reflection_npz": vector_dir / "vector_array_reflection_residuals.npz",
        "transmission_npz": vector_dir / "vector_array_transmission_residuals.npz",
        "sim_ref": vector_dir / "simulated_reflection_signals.npz",
        "obs_ref": vector_dir / "observed_reflection_signals.npz",
        "sim_trans": vector_dir / "simulated_transmission_signals.npz",
        "obs_trans": vector_dir / "observed_transmission_signals.npz",
        "fig_png": vector_dir / "vector_array_residuals.png",
        "fig_pdf": vector_dir / "vector_array_residuals.pdf",
        "fig_summary": vector_dir / "vector_array_residuals_summary.json",
    }

    for path in files.values():
        path.write_text("placeholder", encoding="utf-8")

    manifest = {
        "observation_kind": "vector_array",
        "resolved_receiver_sigma": 0.25,
        "vector_array_comparison": {
            "max_abs_residual": 1.0,
            "residuals": {
                "reflection": {"num_receivers": 3},
                "transmission": {"num_receivers": 3},
            },
        },
        "adjoint_rhs": {
            "kind": "vector_array_full_adjoint_rhs_result",
            "max_rhs_norm": 2.0,
        },
        "exports": {
            "vector_array": {
                "manifest": str(files["manifest"]),
                "residual_csv": str(files["residual_csv"]),
                "residual_npz": {
                    "reflection": str(files["reflection_npz"]),
                    "transmission": str(files["transmission_npz"]),
                },
                "signals": {
                    "simulated_reflection": str(files["sim_ref"]),
                    "observed_reflection": str(files["obs_ref"]),
                    "simulated_transmission": str(files["sim_trans"]),
                    "observed_transmission": str(files["obs_trans"]),
                },
                "residual_figures": {
                    "png": str(files["fig_png"]),
                    "pdf": str(files["fig_pdf"]),
                    "summary": str(files["fig_summary"]),
                },
            }
        },
    }

    manifest_path = tmp_path / "inverse_time_series_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = check_vector_array_export_manifest(manifest_path)

    assert result["status"] == "ok"
    assert result["observation_kind"] == "vector_array"
    assert result["resolved_receiver_sigma"] == 0.25
    assert result["gamma1_num_receivers"] == 3
    assert result["gamma2_num_receivers"] == 3
    assert result["file_checks"]["residual_figure_png"] is True
    assert result["file_checks"]["residual_figure_pdf"] is True
    assert result["file_checks"]["residual_figure_summary"] is True
    assert result["errors"] == []


def test_check_vector_array_export_manifest_rejects_scalar_energy_manifest(tmp_path: Path):
    manifest = {
        "observation_kind": "scalar_energy",
        "resolved_receiver_sigma": None,
        "vector_array_comparison": None,
        "adjoint_rhs": {
            "kind": "full_adjoint_observation_rhs_result",
            "max_rhs_norm": 0.0,
        },
        "exports": {
            "vector_array": None,
        },
    }

    manifest_path = tmp_path / "inverse_time_series_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = check_vector_array_export_manifest(manifest_path)

    assert result["status"] == "fail"
    assert result["errors"]
