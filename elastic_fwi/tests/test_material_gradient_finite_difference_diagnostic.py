from __future__ import annotations

import numpy as np
from mpi4py import MPI

from elastic_fwi.diagnostics.check_material_gradient_finite_difference import (
    run_material_gradient_finite_difference_check,
)


def test_material_gradient_finite_difference_check_reports_ok():
    report = run_material_gradient_finite_difference_check(
        nx=4,
        ny=4,
        degree=1,
        theta=0.25,
        dt=0.1,
        num_steps=5,
        eps_values=(1.0e-2, 5.0e-3, 2.5e-3),
        comm=MPI.COMM_WORLD,
    )

    assert report["kind"] == "material_gradient_finite_difference_check"
    assert report["status"] == "ok"
    assert report["mpi_size"] == MPI.COMM_WORLD.size

    checks = report["checks"]

    assert checks["directional_gradient_status_ok"]
    assert checks["adjoint_gradient_finite"]
    assert checks["baseline_cost_finite"]
    assert checks["baseline_cost_nonnegative"]
    assert checks["finite_difference_records_finite"]
    assert checks["comparison_status_ok"]
    assert checks["finite_difference_close_enough"]
    assert checks["directional_sign_consistent"]
    assert checks["second_order_remainder_improves"]
    assert report["config"]["observation_mode"] == "classical_full"

    values = report["values"]

    assert np.isfinite(values["J0"])
    assert np.isfinite(values["adjoint_gradient"])
    assert np.isfinite(values["best_absolute_error"])
    assert np.isfinite(values["best_relative_error"])

    assert values["J0"] >= 0.0
    assert values["best_absolute_error"] <= report["config"]["absolute_tolerance"]

    records = report["records"]

    assert len(records) == 3

    for record in records:
        assert record["comparison_status"] == "ok"
        assert np.isfinite(record["J0"])
        assert np.isfinite(record["J_eps"])
        assert np.isfinite(record["finite_difference"])
        assert np.isfinite(record["adjoint_gradient"])
        assert np.isfinite(record["absolute_error"])
        assert np.isfinite(record["relative_error"])
        assert np.isfinite(record["first_order_remainder"])
        assert np.isfinite(record["second_order_remainder"])
        assert record["directional_sign_consistent"]

    errors = [record["absolute_error"] for record in records]

    assert errors[-1] <= errors[0]


def test_material_gradient_finite_difference_check_accepts_density_direction():
    report = run_material_gradient_finite_difference_check(
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
        eps_values=(1.0e-2, 5.0e-3, 2.5e-3),
        comm=MPI.COMM_WORLD,
    )

    assert report["status"] == "ok"
    assert report["checks"]["adjoint_gradient_finite"]
    assert report["checks"]["finite_difference_records_finite"]
    assert report["checks"]["comparison_status_ok"]
    assert report["checks"]["finite_difference_close_enough"]
    assert np.isfinite(report["values"]["adjoint_gradient"])


def test_material_gradient_finite_difference_check_accepts_signed_small_direction():
    report = run_material_gradient_finite_difference_check(
        nx=4,
        ny=4,
        degree=1,
        theta=0.25,
        dt=0.1,
        num_steps=5,
        direction_rho=-0.1,
        direction_C11=-0.1,
        direction_C22=0.2,
        direction_C12=-0.03,
        direction_C66=0.04,
        eps_values=(1.0e-2, 5.0e-3, 2.5e-3),
        comm=MPI.COMM_WORLD,
    )

    assert report["status"] == "ok"
    assert report["checks"]["adjoint_gradient_finite"]
    assert report["checks"]["finite_difference_records_finite"]
    assert report["checks"]["comparison_status_ok"]
    assert np.isfinite(report["values"]["adjoint_gradient"])
