from __future__ import annotations

import numpy as np
import pytest
from dolfinx import fem, mesh
from mpi4py import MPI

from elastic_fwi.inversion import assemble_full_vector_array_adjoint_rhs
from elastic_fwi.inversion.vector_array_residuals import VectorArrayResidualSeries
from elastic_fwi.observations import VectorArrayObservationTimeSeries, vector_array_observation
from elastic_fwi.transducers import (
    build_line_transducer_array,
    build_pitch_catch_transducer_arrays,
)


def _create_vector_space(nx: int = 6, ny: int = 6):
    domain = mesh.create_rectangle(
        MPI.COMM_WORLD,
        points=((0.0, 0.0), (10.0, 10.0)),
        n=(nx, ny),
        cell_type=mesh.CellType.triangle,
    )
    V = fem.functionspace(domain, ("Lagrange", 1, (2,)))
    return domain, V


def _make_residual(spec, values):
    values = np.asarray(values, dtype=float)
    return VectorArrayResidualSeries(
        name=f"{spec.array.name}_residual",
        array_name=spec.array.name,
        role=spec.role,
        steps=np.arange(values.shape[0], dtype=np.int64),
        times=0.1 * np.arange(values.shape[0], dtype=float),
        values=values,
        reference_values=values.copy(),
        observed_values=np.zeros_like(values),
    )


def test_full_vector_array_adjoint_rhs_has_newmark_compatible_interface():
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

    reflection_residual = _make_residual(
        reflection_spec,
        values=[
            [[1.0, 0.0], [0.0, 1.0], [0.5, -0.5]],
            [[2.0, 0.0], [0.0, 2.0], [1.0, -1.0]],
        ],
    )
    transmission_residual = _make_residual(
        transmission_spec,
        values=[
            [[0.0, 1.0], [1.0, 0.0], [-0.5, 0.5]],
            [[0.0, 2.0], [2.0, 0.0], [-1.0, 1.0]],
        ],
    )

    rhs = assemble_full_vector_array_adjoint_rhs(
        V,
        reflection_spec=reflection_spec,
        reflection_residual=reflection_residual,
        transmission_spec=transmission_spec,
        transmission_residual=transmission_residual,
    )

    try:
        assert rhs.num_steps == 2
        assert rhs.reflection.num_steps == 2
        assert rhs.transmission.num_steps == 2
        assert len(rhs.reflection.rhs_vectors) == 2
        assert len(rhs.transmission.rhs_vectors) == 2
        assert rhs.max_rhs_norm > 0.0

        combined0 = rhs.reflection.rhs_vectors[0].copy()
        combined0.axpy(1.0, rhs.transmission.rhs_vectors[0])
        try:
            assert combined0.norm() > 0.0
            assert combined0.getSize() == rhs.reflection.rhs_vectors[0].getSize()
        finally:
            combined0.destroy()

        data = rhs.as_dict()
        assert data["kind"] == "vector_array_full_adjoint_rhs_result"
        assert data["num_steps"] == 2
        assert data["reflection"]["kind"] == "vector_array_adjoint_rhs_result"
        assert data["transmission"]["kind"] == "vector_array_adjoint_rhs_result"
    finally:
        rhs.destroy()


def test_full_vector_array_adjoint_rhs_rejects_different_lengths():
    _, V = _create_vector_space()

    gamma1 = build_line_transducer_array(
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
    gamma2 = build_line_transducer_array(
        name="gamma2",
        role="transmission",
        start_x=3.0,
        start_y=9.0,
        end_x=7.0,
        end_y=9.0,
        num_elements=2,
        sigma=0.25,
        tag=2,
    )

    reflection_spec = vector_array_observation(array=gamma1)
    transmission_spec = vector_array_observation(array=gamma2)

    reflection_residual = _make_residual(
        reflection_spec,
        values=[
            [[1.0, 0.0], [0.0, 1.0]],
            [[2.0, 0.0], [0.0, 2.0]],
        ],
    )
    transmission_residual = _make_residual(
        transmission_spec,
        values=[
            [[0.0, 1.0], [1.0, 0.0]],
        ],
    )

    with pytest.raises(ValueError):
        assemble_full_vector_array_adjoint_rhs(
            V,
            reflection_spec=reflection_spec,
            reflection_residual=reflection_residual,
            transmission_spec=transmission_spec,
            transmission_residual=transmission_residual,
        )
