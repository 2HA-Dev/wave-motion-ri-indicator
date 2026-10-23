from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from dolfinx import fem, mesh
from mpi4py import MPI

from elastic_fwi.pipelines.vector_array_observation_comparison import (
    run_vector_array_observation_comparison_from_direct_results,
)
from elastic_fwi.transducers import build_pitch_catch_transducer_arrays
from elastic_fwi.observations import vector_array_observation


def _create_vector_space(nx: int = 8, ny: int = 8):
    domain = mesh.create_rectangle(
        MPI.COMM_WORLD,
        points=((0.0, 0.0), (10.0, 10.0)),
        n=(nx, ny),
        cell_type=mesh.CellType.triangle,
    )
    V = fem.functionspace(domain, ("Lagrange", 1, (2,)))
    return domain, V


def _constant_vector_function(V, values):
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


def _direct_result_from_values(V, values_by_step):
    snapshots = []
    owned_vectors = []

    for step, values in enumerate(values_by_step):
        u = _constant_vector_function(V, values)
        vec = u.x.petsc_vec.copy()
        snapshots.append((step, 0.1 * step, vec))
        owned_vectors.append(vec)

    return SimpleNamespace(
        function_space=V,
        displacement_snapshots=tuple(snapshots),
        _owned_vectors=tuple(owned_vectors),
    )


def _destroy_result(result):
    for vec in result._owned_vectors:
        vec.destroy()


def test_vector_array_observation_comparison_pipeline_builds_residuals_and_rhs():
    _, V = _create_vector_space()

    gamma1, gamma2 = build_pitch_catch_transducer_arrays(
        domain_length_y=10.0,
        absorbing_layer_width=1.0,
        start_x=3.0,
        end_x=7.0,
        num_elements=3,
        sigma=0.25,
    )

    reflection_spec = vector_array_observation(array=gamma1)
    transmission_spec = vector_array_observation(array=gamma2)

    simulated = _direct_result_from_values(
        V,
        values_by_step=[
            (1.0, 2.0),
            (2.0, 4.0),
            (3.0, 6.0),
        ],
    )
    observed = _direct_result_from_values(
        V,
        values_by_step=[
            (0.5, 1.0),
            (1.0, 2.0),
            (1.5, 3.0),
        ],
    )

    try:
        result = run_vector_array_observation_comparison_from_direct_results(
            simulated=simulated,
            observed=observed,
            reflection_spec=reflection_spec,
            transmission_spec=transmission_spec,
        )

        try:
            assert result.num_steps == 3
            assert result.max_abs_residual == pytest.approx(3.0)
            assert result.max_rhs_norm > 0.0

            assert result.simulated_reflection.values.shape == (3, 3, 2)
            assert result.observed_reflection.values.shape == (3, 3, 2)
            assert result.simulated_transmission.values.shape == (3, 3, 2)
            assert result.observed_transmission.values.shape == (3, 3, 2)

            assert result.residuals.reflection is not None
            assert result.residuals.transmission is not None

            np.testing.assert_allclose(
                result.residuals.reflection.values[0, :, 0],
                0.5,
                rtol=1.0e-10,
                atol=1.0e-10,
            )
            np.testing.assert_allclose(
                result.residuals.reflection.values[0, :, 1],
                1.0,
                rtol=1.0e-10,
                atol=1.0e-10,
            )
            np.testing.assert_allclose(
                result.residuals.reflection.values[2, :, 0],
                1.5,
                rtol=1.0e-10,
                atol=1.0e-10,
            )
            np.testing.assert_allclose(
                result.residuals.reflection.values[2, :, 1],
                3.0,
                rtol=1.0e-10,
                atol=1.0e-10,
            )

            data = result.as_dict()
            assert data["kind"] == "vector_array_observation_comparison_result"
            assert data["residual_time_order"] == "chronological"
            assert data["adjoint_solver_expected_order"] == "solver_runs_backward_in_time"
        finally:
            result.destroy()
    finally:
        _destroy_result(simulated)
        _destroy_result(observed)


def test_vector_array_observation_comparison_rejects_missing_attributes():
    _, V = _create_vector_space()

    gamma1, gamma2 = build_pitch_catch_transducer_arrays(
        domain_length_y=10.0,
        absorbing_layer_width=1.0,
        start_x=3.0,
        end_x=7.0,
        num_elements=2,
        sigma=0.25,
    )

    reflection_spec = vector_array_observation(array=gamma1)
    transmission_spec = vector_array_observation(array=gamma2)

    simulated = SimpleNamespace(function_space=V, displacement_snapshots=())
    observed = SimpleNamespace(function_space=V, displacement_snapshots=())

    with pytest.raises(ValueError):
        run_vector_array_observation_comparison_from_direct_results(
            simulated=simulated,
            observed=observed,
            reflection_spec=reflection_spec,
            transmission_spec=transmission_spec,
        )
