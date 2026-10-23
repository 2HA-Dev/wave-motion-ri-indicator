from __future__ import annotations

import numpy as np
import pytest
from dolfinx import fem, mesh
from mpi4py import MPI

from elastic_fwi.observations import (
    build_vector_array_observation_operator,
    evaluate_vector_array_observation_time_series,
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
    return u


def test_vector_array_observation_recovers_constant_vector_field():
    _, V = _create_vector_space()

    array = build_line_transducer_array(
        name="gamma2",
        role="transmission",
        start_x=3.0,
        start_y=9.0,
        end_x=7.0,
        end_y=9.0,
        num_elements=3,
        sigma=0.25,
        tag=2,
    )
    spec = vector_array_observation(array=array)
    operator = build_vector_array_observation_operator(V, spec)

    u = _constant_vector_function(V, values=(2.0, -3.0))
    values = operator.evaluate_function(u)

    assert values.shape == (3, 2)
    np.testing.assert_allclose(values[:, 0], 2.0, rtol=1.0e-10, atol=1.0e-10)
    np.testing.assert_allclose(values[:, 1], -3.0, rtol=1.0e-10, atol=1.0e-10)


def test_vector_array_observation_has_positive_normalizations():
    _, V = _create_vector_space()

    array = build_line_transducer_array(
        name="gamma1",
        role="source_reflection",
        start_x=3.0,
        start_y=1.0,
        end_x=7.0,
        end_y=1.0,
        num_elements=5,
        sigma=0.25,
        tag=1,
    )
    spec = vector_array_observation(array=array)
    operator = build_vector_array_observation_operator(V, spec)

    assert operator.num_receivers == 5
    assert operator.normalizations.shape == (5,)
    assert np.all(np.isfinite(operator.normalizations))
    assert np.all(operator.normalizations > 0.0)


def test_vector_array_observation_time_series_shape_and_values():
    _, V = _create_vector_space()

    array = build_line_transducer_array(
        name="gamma2",
        role="transmission",
        start_x=3.0,
        start_y=9.0,
        end_x=7.0,
        end_y=9.0,
        num_elements=4,
        sigma=0.25,
        tag=2,
    )
    spec = vector_array_observation(array=array)

    u0 = _constant_vector_function(V, values=(1.0, 2.0))
    u1 = _constant_vector_function(V, values=(3.0, 4.0))

    snapshots = [
        (0, 0.0, u0.x.petsc_vec.copy()),
        (1, 0.1, u1.x.petsc_vec.copy()),
    ]

    try:
        series = evaluate_vector_array_observation_time_series(V, snapshots, spec)
    finally:
        for _, _, vec in snapshots:
            vec.destroy()

    assert series.values.shape == (2, 4, 2)
    assert series.steps.tolist() == [0, 1]
    np.testing.assert_allclose(series.times, [0.0, 0.1])
    np.testing.assert_allclose(series.values[0, :, 0], 1.0, rtol=1.0e-10, atol=1.0e-10)
    np.testing.assert_allclose(series.values[0, :, 1], 2.0, rtol=1.0e-10, atol=1.0e-10)
    np.testing.assert_allclose(series.values[1, :, 0], 3.0, rtol=1.0e-10, atol=1.0e-10)
    np.testing.assert_allclose(series.values[1, :, 1], 4.0, rtol=1.0e-10, atol=1.0e-10)


def test_vector_array_observation_manifest_is_json_friendly():
    _, V = _create_vector_space()

    array = build_line_transducer_array(
        name="gamma1",
        role="source_reflection",
        start_x=3.0,
        start_y=1.0,
        end_x=7.0,
        end_y=1.0,
        num_elements=2,
        sigma=0.25,
        tag=1,
    )
    spec = vector_array_observation(array=array)
    operator = build_vector_array_observation_operator(V, spec)

    data = operator.as_dict()

    assert data["kind"] == "vector_array_observation_operator"
    assert data["spec"]["kind"] == "vector_array_observation"
    assert data["num_receivers"] == 2
    assert data["sigma"] == pytest.approx(0.25)
    assert data["spec"]["returns"] == "vector"
    assert data["spec"]["components"] == ["x", "y"]


def test_vector_array_observation_requires_sigma():
    array = build_line_transducer_array(
        name="gamma2",
        role="transmission",
        start_x=3.0,
        start_y=9.0,
        end_x=7.0,
        end_y=9.0,
        num_elements=3,
        sigma=None,
        tag=2,
    )

    with pytest.raises(ValueError):
        vector_array_observation(array=array)


def test_vector_array_observation_rejects_invalid_sigma():
    array = build_line_transducer_array(
        name="gamma2",
        role="transmission",
        start_x=3.0,
        start_y=9.0,
        end_x=7.0,
        end_y=9.0,
        num_elements=3,
        sigma=0.25,
        tag=2,
    )

    with pytest.raises(ValueError):
        vector_array_observation(array=array, sigma=0.0)
