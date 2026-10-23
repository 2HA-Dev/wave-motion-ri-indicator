from __future__ import annotations

import numpy as np
from mpi4py import MPI

from elastic_fwi.diagnostics.check_material_gradient_directional import (
    run_material_gradient_directional_check,
)


def test_material_gradient_directional_check_reports_ok():
    report = run_material_gradient_directional_check(
        nx=4,
        ny=4,
        degree=1,
        theta=0.25,
        dt=0.1,
        num_steps=5,
        comm=MPI.COMM_WORLD,
    )

    assert report["kind"] == "material_gradient_directional_check"
    assert report["status"] == "ok"
    assert report["mpi_size"] == MPI.COMM_WORLD.size
    assert report["config"]["observation_mode"] == "classical_full"

    checks = report["checks"]

    assert checks["pipeline_status_ok"]
    assert checks["simulated_snapshots_stored"]
    assert checks["snapshot_count_matches_num_steps"]
    assert checks["adjoint_num_steps_match"]
    assert checks["gradient_num_steps_match"]
    assert checks["direction_nonzero"]
    assert checks["perturbation_mass_shape_valid"]
    assert checks["perturbation_stiffness_shape_valid"]
    assert checks["gradient_value_finite"]
    assert checks["gradient_records_finite"]
    assert checks["directional_forcing_nonzero"]

    values = report["values"]

    assert np.isfinite(values["gradient_value"])
    assert np.isfinite(values["gradient_abs"])
    assert np.isfinite(values["max_abs_contribution"])
    assert np.isfinite(values["max_forcing_norm"])
    assert np.isfinite(values["max_adjoint_norm"])
    assert np.isfinite(values["max_observation_residual"])

    assert values["max_forcing_norm"] > 0.0
    assert values["max_adjoint_norm"] > 0.0
    assert values["max_observation_residual"] > 0.0

    gradient = report["gradient"]

    assert gradient["kind"] == "material_directional_gradient_result"
    assert gradient["num_steps"] == 5
    assert len(gradient["records"]) == 5
    assert np.isfinite(gradient["value"])
    assert gradient["max_forcing_norm"] > 0.0

    perturbation = report["perturbation"]

    assert perturbation["kind"] == "material_perturbation_matrices"
    assert perturbation["mass"]["rows"] == perturbation["mass"]["cols"]
    assert perturbation["stiffness"]["rows"] == perturbation["stiffness"]["cols"]
    assert perturbation["mass"]["rows"] > 0
    assert perturbation["stiffness"]["rows"] > 0


def test_material_gradient_directional_check_accepts_density_direction():
    report = run_material_gradient_directional_check(
        nx=4,
        ny=4,
        degree=1,
        theta=0.25,
        dt=0.1,
        num_steps=5,
        direction_rho=1.0,
        direction_C11=0.0,
        direction_C22=0.0,
        direction_C12=0.0,
        direction_C66=0.0,
        comm=MPI.COMM_WORLD,
    )

    assert report["status"] == "ok"
    assert report["checks"]["gradient_value_finite"]
    assert report["checks"]["directional_forcing_nonzero"]
    assert np.isfinite(report["values"]["gradient_value"])
    assert report["values"]["max_forcing_norm"] > 0.0


def test_material_gradient_directional_check_accepts_signed_direction():
    report = run_material_gradient_directional_check(
        nx=4,
        ny=4,
        degree=1,
        theta=0.25,
        dt=0.1,
        num_steps=5,
        direction_rho=-1.0,
        direction_C11=-0.1,
        direction_C22=0.2,
        direction_C12=-0.03,
        direction_C66=0.04,
        comm=MPI.COMM_WORLD,
    )

    assert report["status"] == "ok"
    assert report["checks"]["gradient_value_finite"]
    assert report["checks"]["gradient_records_finite"]
    assert report["checks"]["directional_forcing_nonzero"]
    assert np.isfinite(report["values"]["gradient_value"])
