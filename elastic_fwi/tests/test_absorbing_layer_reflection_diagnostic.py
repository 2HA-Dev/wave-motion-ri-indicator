from __future__ import annotations

import json
from pathlib import Path

from mpi4py import MPI

from elastic_fwi.diagnostics.check_absorbing_layer_reflection import (
    run_absorbing_layer_reflection_check,
)


def test_absorbing_layer_reflection_check_writes_artifacts(tmp_path: Path) -> None:
    summary = run_absorbing_layer_reflection_check(
        output_dir=tmp_path,
        nx=12,
        ny=12,
        dt=0.05,
        num_steps=120,
        every_k=2,
        acceptance_ratio=1.0,
        comm=MPI.COMM_WORLD,
    )

    assert summary["kind"] == "absorbing_layer_reflection_check"
    assert summary["status"] in {"ok", "warning"}
    assert summary["absorbing_layer"]["enabled"] is True
    assert summary["time"]["final_time"] == 6.0
    assert summary["time"]["num_snapshots"] == 60
    assert summary["energy"]["late_energy_suppression_ratio"] >= 0.0

    if MPI.COMM_WORLD.rank == 0:
        json_path = tmp_path / "absorbing_layer_reflection_summary.json"
        csv_path = tmp_path / "absorbing_layer_reflection_energy.csv"
        png_path = tmp_path / "absorbing_layer_reflection_energy.png"

        assert json_path.exists()
        assert csv_path.exists()
        assert png_path.exists()

        loaded = json.loads(json_path.read_text())
        assert loaded["kind"] == "absorbing_layer_reflection_check"
        assert loaded["absorbing_layer"]["enabled"] is True
