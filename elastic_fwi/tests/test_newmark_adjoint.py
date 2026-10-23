import json

import numpy as np
import pytest
from mpi4py import MPI
from petsc4py import PETSc

from elastic_fwi.inversion.adjoint_observation_sources import (
    AdjointObservationRHSRecord,
    AdjointObservationRHSResult,
    FullAdjointObservationRHSResult,
)
from elastic_fwi.inversion.newmark_adjoint import (
    newmark_middle_operator_action,
    solve_newmark_discrete_adjoint_state,
)
from elastic_fwi.solvers.elastic_operator import (
    ElasticOperatorConfig,
    ElasticOperatorMatrices,
)
from elastic_fwi.solvers.newmark import LinearSolveConfig


def _identity_matrix(global_size=8):
    comm = MPI.COMM_WORLD
    matrix = PETSc.Mat().createAIJ([global_size, global_size], comm=comm)
    matrix.setUp()

    row_start, row_end = matrix.getOwnershipRange()

    for row in range(row_start, row_end):
        matrix.setValue(row, row, 1.0)

    matrix.assemble()

    return matrix


def _zero_matrix(global_size=8):
    comm = MPI.COMM_WORLD
    matrix = PETSc.Mat().createAIJ([global_size, global_size], comm=comm)
    matrix.setUp()
    matrix.assemble()
    return matrix


def _constant_vector(matrix, value):
    vector = matrix.createVecRight()
    vector.set(float(value))
    vector.assemble()
    return vector


def _identity_newmark_matrices(global_size=8):
    mass = _identity_matrix(global_size)
    stiffness = _zero_matrix(global_size)
    effective = _identity_matrix(global_size)

    return ElasticOperatorMatrices(
        mass=mass,
        stiffness=stiffness,
        effective=effective,
        config=ElasticOperatorConfig(theta=0.25, dt=0.1),
    )


def _rhs_result(matrix, values):
    reflection_vectors = []
    transmission_vectors = []
    reflection_records = []
    transmission_records = []

    for index, value in enumerate(values):
        step = index + 1
        time = 0.1 * step

        rhs_ref = _constant_vector(matrix, value)
        rhs_trans = _constant_vector(matrix, 0.0)

        reflection_vectors.append(rhs_ref)
        transmission_vectors.append(rhs_trans)

        reflection_records.append(
            AdjointObservationRHSRecord(
                step=step,
                time=time,
                tag=1,
                component="vector",
                residual_value=float(value),
                rhs_norm=float(rhs_ref.norm()),
                rhs_size_global=int(rhs_ref.getSize()),
                rhs_size_local=int(rhs_ref.getLocalSize()),
            )
        )

        transmission_records.append(
            AdjointObservationRHSRecord(
                step=step,
                time=time,
                tag=2,
                component="vector",
                residual_value=0.0,
                rhs_norm=float(rhs_trans.norm()),
                rhs_size_global=int(rhs_trans.getSize()),
                rhs_size_local=int(rhs_trans.getLocalSize()),
            )
        )

    return FullAdjointObservationRHSResult(
        reflection=AdjointObservationRHSResult(
            records=tuple(reflection_records),
            rhs_vectors=tuple(reflection_vectors),
        ),
        transmission=AdjointObservationRHSResult(
            records=tuple(transmission_records),
            rhs_vectors=tuple(transmission_vectors),
        ),
    )


def test_newmark_middle_operator_action_for_identity_mass_zero_stiffness():
    matrices = _identity_newmark_matrices(global_size=8)
    vector = _constant_vector(matrices.effective, 3.0)

    action = newmark_middle_operator_action(matrices, vector)

    try:
        assert np.allclose(action.getArray(readonly=True), 6.0)

    finally:
        action.destroy()
        vector.destroy()
        matrices.destroy()


def test_newmark_discrete_adjoint_one_step_identity_recurrence():
    matrices = _identity_newmark_matrices(global_size=8)
    rhs = _rhs_result(matrices.effective, values=[3.0])

    result = solve_newmark_discrete_adjoint_state(
        matrices,
        rhs,
        solve_config=LinearSolveConfig(
            ksp_type="cg",
            pc_type="jacobi",
            rtol=1.0e-12,
            atol=1.0e-14,
            max_it=100,
        ),
    )

    try:
        assert result.num_steps == 1
        assert result.all_ksp_converged
        assert result.max_linear_residual_norm < 1.0e-10

        equation = result.equation_adjoint_vectors[0]

        assert np.allclose(equation.getArray(readonly=True), 3.0)
        assert np.allclose(result.initial_current_adjoint.getArray(readonly=True), 6.0)
        assert np.allclose(result.initial_previous_adjoint.getArray(readonly=True), -3.0)

        report = result.as_dict()

        assert report["kind"] == "newmark_adjoint_state_result"
        assert report["records"][0]["step"] == 1
        assert report["records"][0]["solve_index"] == 1

        payload = json.dumps(report, sort_keys=True)

        assert '"newmark_adjoint_state_result"' in payload

    finally:
        result.destroy()
        rhs.destroy()
        matrices.destroy()


def test_newmark_discrete_adjoint_two_step_identity_recurrence():
    matrices = _identity_newmark_matrices(global_size=8)
    rhs = _rhs_result(matrices.effective, values=[1.0, 2.0])

    result = solve_newmark_discrete_adjoint_state(
        matrices,
        rhs,
        solve_config=LinearSolveConfig(
            ksp_type="cg",
            pc_type="jacobi",
            rtol=1.0e-12,
            atol=1.0e-14,
            max_it=100,
        ),
    )

    try:
        assert result.num_steps == 2
        assert result.all_ksp_converged

        # For M=I, K=0:
        # u1 = 2 u0 - u_{-1}
        # u2 = 2 u1 - u0 = 3 u0 - 2 u_{-1}
        #
        # With g1=1 and g2=2:
        # dJ/du0     = 2*g1 + 3*g2 = 8
        # dJ/du_{-1} = -g1 - 2*g2  = -5
        assert np.allclose(result.equation_adjoint_vectors[1].getArray(readonly=True), 2.0)
        assert np.allclose(result.equation_adjoint_vectors[0].getArray(readonly=True), 5.0)

        assert np.allclose(result.initial_current_adjoint.getArray(readonly=True), 8.0)
        assert np.allclose(result.initial_previous_adjoint.getArray(readonly=True), -5.0)

        report = result.as_dict()

        assert report["records"][0]["step"] == 1
        assert report["records"][0]["solve_index"] == 2
        assert report["records"][1]["step"] == 2
        assert report["records"][1]["solve_index"] == 1

    finally:
        result.destroy()
        rhs.destroy()
        matrices.destroy()


def test_newmark_discrete_adjoint_rejects_wrong_rhs_size():
    matrices = _identity_newmark_matrices(global_size=8)
    bad_matrix = _identity_matrix(global_size=4)
    rhs = _rhs_result(bad_matrix, values=[1.0])

    try:
        with pytest.raises(ValueError):
            solve_newmark_discrete_adjoint_state(matrices, rhs)

    finally:
        rhs.destroy()
        bad_matrix.destroy()
        matrices.destroy()


def test_newmark_discrete_adjoint_reuse_matches_old_path():
    matrices = _identity_newmark_matrices(global_size=8)
    rhs = _rhs_result(matrices.effective, values=[1.0, 2.0, 3.0])
    solve_config = LinearSolveConfig(
        ksp_type="cg",
        pc_type="jacobi",
        rtol=1.0e-12,
        atol=1.0e-14,
        max_it=100,
    )

    old = solve_newmark_discrete_adjoint_state(
        matrices,
        rhs,
        solve_config=solve_config,
        reuse_effective_solver=False,
    )
    new = solve_newmark_discrete_adjoint_state(
        matrices,
        rhs,
        solve_config=solve_config,
        reuse_effective_solver=True,
    )

    try:
        assert old.num_steps == new.num_steps == 3
        assert old.all_ksp_converged
        assert new.all_ksp_converged

        for vec_old, vec_new in zip(
            old.equation_adjoint_vectors,
            new.equation_adjoint_vectors,
        ):
            diff = vec_new.copy()
            try:
                diff.axpy(-1.0, vec_old)
                diff_norm = float(diff.norm())
                ref_norm = max(float(vec_old.norm()), 1.0)
                assert diff_norm / ref_norm <= 1.0e-12
            finally:
                diff.destroy()

        for vec_old, vec_new in (
            (old.initial_current_adjoint, new.initial_current_adjoint),
            (old.initial_previous_adjoint, new.initial_previous_adjoint),
        ):
            diff = vec_new.copy()
            try:
                diff.axpy(-1.0, vec_old)
                diff_norm = float(diff.norm())
                ref_norm = max(float(vec_old.norm()), 1.0)
                assert diff_norm / ref_norm <= 1.0e-12
            finally:
                diff.destroy()

        for rec_old, rec_new in zip(old.records, new.records):
            assert rec_old.step == rec_new.step
            assert rec_old.time == rec_new.time
            assert rec_old.solve_index == rec_new.solve_index
            assert rec_old.ksp_converged == rec_new.ksp_converged
            assert rec_old.ksp_reason == rec_new.ksp_reason
            assert rec_old.ksp_iterations == rec_new.ksp_iterations
            assert np.isfinite(rec_old.linear_residual_relative)
            assert np.isfinite(rec_new.linear_residual_relative)
            assert abs(
                rec_old.linear_residual_relative
                - rec_new.linear_residual_relative
            ) <= 1.0e-12

    finally:
        old.destroy()
        new.destroy()
        rhs.destroy()
        matrices.destroy()
