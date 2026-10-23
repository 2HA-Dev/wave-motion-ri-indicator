from __future__ import annotations

import json
import subprocess

import pytest
import sys
from pathlib import Path
from mpi4py import MPI


def test_inverse_export_with_internal_gaussian_receiver_has_nonzero_adjoint_rhs(tmp_path):
    if MPI.COMM_WORLD.size > 1:
        pytest.skip(
            "This test launches the diagnostic as a subprocess and is sequential-only; "
            "nested MPI initialization is not supported under mpirun pytest."
        )

    output_dir = tmp_path / "inverse_internal_gaussian_receiver"

    cmd = [
        sys.executable,
        "-m",
        "elastic_fwi.diagnostics.export_inverse_time_series_xdmf",
        "--mesh-kind",
        "structured",
        "--domain-length-x",
        "10.0",
        "--domain-length-y",
        "10.0",
        "--absorbing-layer",
        "--absorbing-layer-width",
        "1.0",
        "--absorbing-layer-strength",
        "6.0",
        "--absorbing-layer-polynomial-order",
        "2.0",
        "--absorbing-layer-min-cells",
        "8",
        "--auto-place-probes-inside-absorbing-layer",
        "--source-kind",
        "internal_gaussian_array",
        "--source-array-start-x",
        "3.0",
        "--source-array-end-x",
        "7.0",
        "--source-array-num-elements",
        "5",
        "--source-sigma",
        "0.20",
        "--source-direction-x",
        "0.0",
        "--source-direction-y",
        "1.0",
        "--receiver-kind",
        "internal_gaussian_array",
        "--receiver-array-start-x",
        "3.0",
        "--receiver-array-end-x",
        "7.0",
        "--receiver-array-num-elements",
        "5",
        "--receiver-array-element-length",
        "0.25",
        "--simulated-material-kind",
        "healthy",
        "--observed-material-kind",
        "localized_disk_defect",
        "--defect-center-x",
        "5.0",
        "--defect-center-y",
        "5.0",
        "--defect-radius",
        "0.30",
        "--contrast-factor",
        "0.5",
        "--nx",
        "48",
        "--ny",
        "48",
        "--dt",
        "0.05",
        "--num-steps",
        "100",
        "--source-frequency",
        "1.0",
        "--every-k",
        "10",
        "--output-dir",
        str(output_dir),
    ]

    subprocess.run(cmd, check=True)

    manifests = sorted(output_dir.glob("*manifest*.json"))
    assert manifests

    manifest = json.loads(manifests[0].read_text())

    assert manifest["status"] == "ok"

    text = json.dumps(manifest)
    assert "internal_gaussian_array" in text
    assert "gamma2_internal_gaussian_array_transmission" in text

    max_residual = manifest["comparison"]["values"]["max_abs_residual"]
    adjoint_rhs_norm = manifest["adjoint_rhs"]["max_rhs_norm"]

    assert max_residual > 0.0
    assert adjoint_rhs_norm > 0.0
