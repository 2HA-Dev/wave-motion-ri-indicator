from __future__ import annotations

import numpy as np
import pytest
from dolfinx import fem, mesh
from mpi4py import MPI

from elastic_fwi.inversion import (
    assemble_vector_array_adjoint_rhs_series,
    assemble_vector_array_adjoint_rhs_vector,
)
from elastic_fwi.inversion.vector_array_residuals import VectorArrayResidualSeries
from elastic_fwi.observations import (
    VectorArrayObservationTimeSeries,
    build_vector_array_observation_operator,
    vector_array_observation,
)
from elastic_fwi.transducers import build_line_transducer_array


def _create_vector_space(nx: int = 8, ny: int = 8):
    domain = mesh.create_rectangle(
        MPI.COMM_WORLD,
        points=((0.0, 0.0), (10.0, 10.0)),
        n=(nx, ny),
        cell_type=mesh.CellType.triangle,
    )
    V = fem.functionspace(domain, ("Lagrange", 1, (2,)))
    return domain, V


def _constant_vector_function(V, values=(2.0, -3.0)):
    u = fem.Function(V)
    vx, vy = float(values[0]), float(values[1])

    def expr(x):
        out = np.zeros((2, x.shape[1]), dtype=float)
        out[0, :] = vx
        out[1, :] = vy
        return out

    u.interpolate(expr)
    u.x.scatter_forward()
    return u


def _make_spec(num_receivers: int = 3):
    array = build_line_transducer_array(
        name="gamma2",
        role="transmission",
        start_x=3.0,
        start_y=9.0,
        end_x=7.0,
        end_y=9.0,
        num_elements=num_receivers,
        sigma=0.25,
        tag=2,
    )
    return vector_array_observation(array=array)


def _make_residual_series(spec, values):
    values = np.asarray(values, dtype=float)
    reference = values.copy()
    observed = np.zeros_like(values)
    return VectorArrayResidualSeries(
        name="gamma2_residual",
        array_name=spec.array.name,
        role=spec.role,
        steps=np.arange(values.shape[0], dtype=np.int64),
        times=0.1 * np.arange(values.shape[0], dtype=float),
        values=values,
        reference_values=reference,
        observed_values=observed,
    )


def _global_form_dot_function(rhs, u):
    """Compute RHS(u) with a PETSc Vec dot product."""
    u.x.scatter_forward()
    return float(rhs.dot(u.x.petsc_vec))


def test_vector_array_adjoint_rhs_nonzero_for_nonzero_residual():
    _, V = _create_vector_space()
    spec = _make_spec(num_receivers=3)

    residual = np.array(
        [
            [1.0, 0.0],
            [0.0, 2.0],
            [-1.0, 1.0],
        ],
        dtype=float,
    )

    rhs = assemble_vector_array_adjoint_rhs_vector(V, spec, residual)

    try:
        assert rhs.getSize() > 0
        assert rhs.norm() > 0.0
    finally:
        rhs.destroy()


def test_vector_array_adjoint_rhs_zero_for_zero_residual():
    _, V = _create_vector_space()
    spec = _make_spec(num_receivers=3)

    residual = np.zeros((3, 2), dtype=float)

    rhs = assemble_vector_array_adjoint_rhs_vector(V, spec, residual)

    try:
        assert rhs.norm() == pytest.approx(0.0)
    finally:
        rhs.destroy()


def test_vector_array_adjoint_rhs_scales_linearly():
    _, V = _create_vector_space()
    spec = _make_spec(num_receivers=3)

    residual = np.array(
        [
            [1.0, 0.5],
            [0.0, -1.0],
            [2.0, 1.0],
        ],
        dtype=float,
    )

    rhs1 = assemble_vector_array_adjoint_rhs_vector(V, spec, residual, scale=1.0)
    rhs10 = assemble_vector_array_adjoint_rhs_vector(V, spec, residual, scale=10.0)

    try:
        diff = rhs10.copy()
        diff.axpy(-10.0, rhs1)
        assert diff.norm() == pytest.approx(0.0, abs=1.0e-12)
        diff.destroy()
    finally:
        rhs1.destroy()
        rhs10.destroy()


def test_vector_array_adjoint_rhs_satisfies_discrete_duality_on_constant_field():
    _, V = _create_vector_space()
    spec = _make_spec(num_receivers=4)
    operator = build_vector_array_observation_operator(V, spec)

    u = _constant_vector_function(V, values=(2.0, -3.0))
    Hu = operator.evaluate_function(u)

    residual = np.array(
        [
            [1.0, 0.0],
            [0.0, 2.0],
            [-1.0, 1.0],
            [0.5, -0.5],
        ],
        dtype=float,
    )

    rhs = assemble_vector_array_adjoint_rhs_vector(V, spec, residual)

    try:
        left = float(np.sum(Hu * residual))
        right = _global_form_dot_function(rhs, u)
        assert right == pytest.approx(left, rel=1.0e-10, abs=1.0e-10)
    finally:
        rhs.destroy()


def test_vector_array_adjoint_rhs_series_builds_one_vector_per_step():
    _, V = _create_vector_space()
    spec = _make_spec(num_receivers=2)

    residual_series = _make_residual_series(
        spec,
        values=[
            [[1.0, 0.0], [0.0, 1.0]],
            [[2.0, 0.0], [0.0, 2.0]],
            [[0.0, 0.0], [0.0, 0.0]],
        ],
    )

    result = assemble_vector_array_adjoint_rhs_series(
        V,
        spec,
        residual_series,
    )

    try:
        assert result.num_steps == 3
        assert len(result.rhs_vectors) == 3
        assert result.records[0].rhs_norm > 0.0
        assert result.records[1].rhs_norm > result.records[0].rhs_norm
        assert result.records[2].rhs_norm == pytest.approx(0.0)
        assert result.max_rhs_norm == pytest.approx(result.records[1].rhs_norm)
        assert result.as_dict()["kind"] == "vector_array_adjoint_rhs_result"
    finally:
        result.destroy()


def test_vector_array_adjoint_rhs_rejects_wrong_residual_shape():
    _, V = _create_vector_space()
    spec = _make_spec(num_receivers=3)

    with pytest.raises(ValueError):
        assemble_vector_array_adjoint_rhs_vector(
            V,
            spec,
            np.zeros((3,), dtype=float),
        )

    with pytest.raises(ValueError):
        assemble_vector_array_adjoint_rhs_vector(
            V,
            spec,
            np.zeros((4, 2), dtype=float),
        )


def test_vector_array_adjoint_rhs_series_rejects_wrong_number_of_receivers():
    _, V = _create_vector_space()
    spec = _make_spec(num_receivers=3)

    bad_series = _make_residual_series(
        spec,
        values=[
            [[1.0, 0.0], [0.0, 1.0]],
        ],
    )

    with pytest.raises(ValueError):
        assemble_vector_array_adjoint_rhs_series(V, spec, bad_series)
