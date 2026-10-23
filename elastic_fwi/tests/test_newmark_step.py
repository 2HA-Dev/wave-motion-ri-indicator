import numpy as np
import pytest

pytest.importorskip("dolfinx")

from mpi4py import MPI
from dolfinx import mesh

from elastic_fwi.materials.dg0_fields import homogeneous_dg0_material_functions
from elastic_fwi.materials.orthotropic import LEGACY_HEALTHY_MATERIAL
from elastic_fwi.solvers.elastic_operator import (
    ElasticOperatorConfig,
    assemble_elastic_operator_matrices,
    create_vector_lagrange_space,
)
from elastic_fwi.solvers.newmark import (
    LinearSolveConfig,
    ReusableEffectiveLinearSolver,
    build_newmark_theta_rhs,
    create_constant_state_vector,
    create_zero_state_vector,
    linear_residual_norm,
    newmark_theta_step,
    solve_effective_system,
)


def _create_matrices():
    msh = mesh.create_unit_square(MPI.COMM_WORLD, 2, 2)
    V = create_vector_lagrange_space(msh, degree=1)
    materials = homogeneous_dg0_material_functions(msh, LEGACY_HEALTHY_MATERIAL)
    config = ElasticOperatorConfig(theta=0.25, dt=0.1)
    return assemble_elastic_operator_matrices(V, materials, config)


def _constant_force(matrices, value):
    force = matrices.effective.createVecLeft()
    force.set(value)
    return force


def _vector_difference_norm(left, right):
    diff = left.copy()
    diff.axpy(-1.0, right)
    norm = float(diff.norm())
    diff.destroy()
    return norm


def test_linear_solve_config_validation():
    config = LinearSolveConfig(ksp_type="cg", pc_type="jacobi", rtol=1.0e-10)

    assert config.ksp_type == "cg"
    assert config.pc_type == "jacobi"
    assert config.rtol == pytest.approx(1.0e-10)

    with pytest.raises(ValueError):
        LinearSolveConfig(ksp_type="", pc_type="jacobi")

    with pytest.raises(ValueError):
        LinearSolveConfig(ksp_type="cg", pc_type="")

    with pytest.raises(ValueError):
        LinearSolveConfig(rtol=-1.0)

    with pytest.raises(ValueError):
        LinearSolveConfig(max_it=0)


def test_newmark_rhs_zero_states_is_dt2_force():
    matrices = _create_matrices()

    try:
        u_current = create_zero_state_vector(matrices)
        u_previous = create_zero_state_vector(matrices)
        force = _constant_force(matrices, value=2.0)

        rhs = build_newmark_theta_rhs(
            matrices,
            u_current,
            u_previous,
            force,
        )

        expected = force.copy()
        expected.scale(matrices.config.dt**2)

        error = _vector_difference_norm(rhs, expected)

        assert error <= 1.0e-13

    finally:
        for name in ("u_current", "u_previous", "force", "rhs", "expected"):
            vector = locals().get(name)
            if vector is not None:
                vector.destroy()
        matrices.destroy()


def test_newmark_rhs_uses_current_and_previous_states():
    matrices = _create_matrices()

    try:
        u_current = create_constant_state_vector(matrices, value=1.0)
        u_previous = create_zero_state_vector(matrices)
        force = _constant_force(matrices, value=0.0)

        rhs = build_newmark_theta_rhs(
            matrices,
            u_current,
            u_previous,
            force,
        )

        norm = float(rhs.norm())

        assert np.isfinite(norm)
        assert norm > 0.0

    finally:
        for name in ("u_current", "u_previous", "force", "rhs"):
            vector = locals().get(name)
            if vector is not None:
                vector.destroy()
        matrices.destroy()


def test_newmark_theta_step_solves_effective_system():
    matrices = _create_matrices()

    try:
        u_current = create_zero_state_vector(matrices)
        u_previous = create_zero_state_vector(matrices)
        force = _constant_force(matrices, value=1.0)

        result = newmark_theta_step(
            matrices,
            u_current,
            u_previous,
            force,
            LinearSolveConfig(rtol=1.0e-10, atol=1.0e-12, max_it=1000),
        )

        residual_norm = linear_residual_norm(
            matrices.effective,
            result.displacement,
            result.rhs,
        )

        solution_norm = float(result.displacement.norm())

        assert result.solve_report.converged
        assert np.isfinite(result.solve_report.residual_norm)
        assert np.isfinite(residual_norm)
        assert residual_norm <= 1.0e-9
        assert solution_norm > 0.0

    finally:
        for name in ("u_current", "u_previous", "force"):
            vector = locals().get(name)
            if vector is not None:
                vector.destroy()
        result = locals().get("result")
        if result is not None:
            result.destroy()
        matrices.destroy()


def test_reusable_effective_linear_solver_matches_one_shot_solve():
    matrices = _create_matrices()
    reusable_solver = None

    try:
        u_current = create_zero_state_vector(matrices)
        u_previous = create_zero_state_vector(matrices)
        force = _constant_force(matrices, value=1.0)
        rhs = build_newmark_theta_rhs(
            matrices,
            u_current,
            u_previous,
            force,
        )
        config = LinearSolveConfig(
            ksp_type="cg",
            pc_type="jacobi",
            rtol=1.0e-10,
            atol=1.0e-12,
            max_it=1000,
        )

        one_shot_solution, one_shot_report = solve_effective_system(
            matrices.effective,
            rhs,
            config,
        )

        reusable_solver = ReusableEffectiveLinearSolver(
            matrices.effective,
            config,
        )
        reusable_solution, reusable_report = reusable_solver.solve(rhs)

        diff_norm = _vector_difference_norm(reusable_solution, one_shot_solution)
        reference_norm = max(float(one_shot_solution.norm()), 1.0)

        assert reusable_solution.getSize() == one_shot_solution.getSize()
        assert np.all(np.isfinite(reusable_solution.getArray(readonly=True)))
        assert one_shot_report.converged
        assert reusable_report.converged
        assert np.isfinite(one_shot_report.residual_norm)
        assert np.isfinite(reusable_report.residual_norm)
        assert diff_norm / reference_norm <= 1.0e-12

        reusable_solver.destroy()
        with pytest.raises(RuntimeError):
            reusable_solver.solve(rhs)

    finally:
        if reusable_solver is not None:
            reusable_solver.destroy()

        for name in (
            "u_current",
            "u_previous",
            "force",
            "rhs",
            "one_shot_solution",
            "reusable_solution",
        ):
            vector = locals().get(name)
            if vector is not None:
                vector.destroy()

        matrices.destroy()

