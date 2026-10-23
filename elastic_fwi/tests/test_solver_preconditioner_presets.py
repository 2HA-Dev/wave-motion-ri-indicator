from __future__ import annotations

import pytest

pytest.importorskip("dolfinx")

from elastic_fwi.diagnostics.check_solver_preconditioners import (
    check_solver_preconditioners,
)
from elastic_fwi.solvers.newmark import LinearSolveConfig
from elastic_fwi.solvers.solver_presets import (
    get_solver_preset,
    list_solver_presets,
)


def test_linear_solve_config_documents_default_cg_jacobi():
    config = LinearSolveConfig()

    assert config.ksp_type == "cg"
    assert config.pc_type == "jacobi"
    assert config.pc_factor_mat_solver_type is None


def test_solver_preset_contracts():
    names = list_solver_presets()

    assert "default_cg_jacobi" in names
    assert "cg_gamg" in names
    assert "cg_hypre" in names
    assert "gmres_ilu" in names
    assert "preonly_lu" in names
    assert "preonly_lu_mumps" in names

    mumps = get_solver_preset("preonly_lu_mumps")
    assert mumps.config.ksp_type == "preonly"
    assert mumps.config.pc_type == "lu"
    assert mumps.config.pc_factor_mat_solver_type == "mumps"

    with pytest.raises(ValueError):
        get_solver_preset("unknown")


def test_default_solver_preconditioner_diagnostic_passes():
    report = check_solver_preconditioners(
        presets=["default_cg_jacobi"],
        nx=2,
        ny=2,
        dt=0.1,
        theta=0.25,
        residual_tol=1.0e-8,
    )

    assert report["status"] == "ok"
    assert report["default_solver_contract"]["ksp_type"] == "cg"
    assert report["default_solver_contract"]["pc_type"] == "jacobi"

    result = report["results"][0]
    assert result["preset"]["name"] == "default_cg_jacobi"
    assert result["status"] == "ok"
    assert result["converged"]


def test_single_non_default_solver_preset_can_pass_without_default():
    report = check_solver_preconditioners(
        presets=["preonly_lu"],
        nx=2,
        ny=2,
        dt=0.1,
        theta=0.25,
        residual_tol=1.0e-8,
    )

    assert report["status"] == "ok"
    assert report["results"][0]["preset"]["name"] == "preonly_lu"
    assert report["results"][0]["status"] == "ok"


def test_cg_gamg_preset_can_pass_without_default():
    report = check_solver_preconditioners(
        presets=["cg_gamg"],
        nx=2,
        ny=2,
        dt=0.1,
        theta=0.25,
        residual_tol=1.0e-8,
    )

    assert report["status"] == "ok"
    assert report["results"][0]["preset"]["name"] == "cg_gamg"
    assert report["results"][0]["status"] == "ok"
