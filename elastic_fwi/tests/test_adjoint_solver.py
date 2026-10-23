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
from elastic_fwi.inversion.adjoint_solver import solve_adjoint_effective_rhs_series
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


def _owned_filled_vector(matrix, value):
    vector = matrix.createVecRight()
    vector.set(0.0)

    start, end = vector.getOwnershipRange()

    for row in range(start, end):
        vector.setValue(row, float(value))

    vector.assemble()

    return vector


def _rhs_result(matrix, num_steps=3):
    reflection_vectors = []
    transmission_vectors = []
    reflection_records = []
    transmission_records = []

    for index in range(num_steps):
        step = index + 1
        time = 0.1 * step

        rhs_ref = _owned_filled_vector(matrix, float(step))
        rhs_trans = _owned_filled_vector(matrix, 0.1 * float(step))

        reflection_vectors.append(rhs_ref)
        transmission_vectors.append(rhs_trans)

        reflection_records.append(
            AdjointObservationRHSRecord(
                step=step,
                time=time,
                tag=1,
                component="vector",
                residual_value=float(step),
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
                residual_value=0.1 * float(step),
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


def test_adjoint_effective_solver_solves_identity_system_in_time_order():
    matrix = _identity_matrix()
    rhs = _rhs_result(matrix, num_steps=3)

    result = solve_adjoint_effective_rhs_series(
        matrix,
        rhs,
        solve_config=LinearSolveConfig(
            ksp_type="cg",
            pc_type="jacobi",
            rtol=1.0e-12,
            atol=1.0e-14,
            max_it=100,
        ),
        reverse_time=False,
    )

    try:
        assert result.num_steps == 3
        assert result.all_ksp_converged
        assert result.max_solution_norm > 0.0
        assert result.max_linear_residual_norm < 1.0e-10

        for index, vector in enumerate(result.adjoint_vectors):
            step = index + 1
            expected = 1.1 * float(step)

            local_values = vector.getArray(readonly=True)

            assert np.allclose(local_values, expected)

        report = result.as_dict()

        assert report["kind"] == "adjoint_effective_solve_result"
        assert report["reverse_time"] is False
        assert report["records"][0]["solve_index"] == 1
        assert report["records"][1]["solve_index"] == 2
        assert report["records"][2]["solve_index"] == 3

        payload = json.dumps(report, sort_keys=True)

        assert '"adjoint_effective_solve_result"' in payload

    finally:
        result.destroy()
        rhs.destroy()
        matrix.destroy()


def test_adjoint_effective_solver_records_reverse_solve_order():
    matrix = _identity_matrix()
    rhs = _rhs_result(matrix, num_steps=3)

    result = solve_adjoint_effective_rhs_series(
        matrix,
        rhs,
        solve_config=LinearSolveConfig(
            ksp_type="cg",
            pc_type="jacobi",
            rtol=1.0e-12,
            atol=1.0e-14,
            max_it=100,
        ),
        reverse_time=True,
    )

    try:
        assert result.num_steps == 3
        assert result.all_ksp_converged

        report = result.as_dict()

        assert report["reverse_time"] is True

        # Records are stored in physical time order, while solve_index
        # records the actual backward solve order.
        assert report["records"][0]["step"] == 1
        assert report["records"][0]["solve_index"] == 3
        assert report["records"][1]["step"] == 2
        assert report["records"][1]["solve_index"] == 2
        assert report["records"][2]["step"] == 3
        assert report["records"][2]["solve_index"] == 1

    finally:
        result.destroy()
        rhs.destroy()
        matrix.destroy()


def test_adjoint_effective_solver_rejects_mismatched_rhs_sizes():
    matrix = _identity_matrix(global_size=8)
    rhs = _rhs_result(matrix, num_steps=2)

    small_matrix = _identity_matrix(global_size=4)
    bad_vector = _owned_filled_vector(small_matrix, 1.0)

    bad_transmission = AdjointObservationRHSResult(
        records=rhs.transmission.records,
        rhs_vectors=(bad_vector, rhs.transmission.rhs_vectors[1]),
    )

    bad_rhs = FullAdjointObservationRHSResult(
        reflection=rhs.reflection,
        transmission=bad_transmission,
    )

    try:
        with pytest.raises(ValueError):
            solve_adjoint_effective_rhs_series(matrix, bad_rhs)

    finally:
        bad_vector.destroy()
        small_matrix.destroy()
        rhs.destroy()
        matrix.destroy()


def test_adjoint_effective_solver_reuse_matches_old_path():
    matrix = _identity_matrix(global_size=8)
    rhs = _rhs_result(matrix, num_steps=4)
    solve_config = LinearSolveConfig(
        ksp_type="cg",
        pc_type="jacobi",
        rtol=1.0e-12,
        atol=1.0e-14,
        max_it=100,
    )

    old = solve_adjoint_effective_rhs_series(
        matrix,
        rhs,
        solve_config=solve_config,
        reverse_time=True,
        reuse_effective_solver=False,
    )
    new = solve_adjoint_effective_rhs_series(
        matrix,
        rhs,
        solve_config=solve_config,
        reverse_time=True,
        reuse_effective_solver=True,
    )

    try:
        assert old.num_steps == new.num_steps == 4
        assert old.all_ksp_converged
        assert new.all_ksp_converged

        for vec_old, vec_new in zip(old.adjoint_vectors, new.adjoint_vectors):
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
        matrix.destroy()
