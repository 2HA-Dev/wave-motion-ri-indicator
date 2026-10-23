import json

import numpy as np
import pytest
from mpi4py import MPI
from petsc4py import PETSc

from elastic_fwi.inversion.material_gradient import (
    compute_newmark_material_directional_gradient,
    material_effective_operator_action,
    material_middle_operator_action,
    material_newmark_step_forcing,
)
from elastic_fwi.solvers.elastic_operator import ElasticOperatorConfig


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


def test_material_effective_and_middle_actions_for_delta_mass_identity():
    delta_mass = _identity_matrix(global_size=8)
    delta_stiffness = _zero_matrix(global_size=8)
    config = ElasticOperatorConfig(theta=0.25, dt=0.1)

    vector = _constant_vector(delta_mass, 3.0)

    effective = material_effective_operator_action(
        delta_mass,
        delta_stiffness,
        config,
        vector,
    )
    middle = material_middle_operator_action(
        delta_mass,
        delta_stiffness,
        config,
        vector,
    )

    try:
        assert np.allclose(effective.getArray(readonly=True), 3.0)
        assert np.allclose(middle.getArray(readonly=True), 6.0)

    finally:
        effective.destroy()
        middle.destroy()
        vector.destroy()
        delta_mass.destroy()
        delta_stiffness.destroy()


def test_material_newmark_step_forcing_identity_delta_mass():
    delta_mass = _identity_matrix(global_size=8)
    delta_stiffness = _zero_matrix(global_size=8)
    config = ElasticOperatorConfig(theta=0.25, dt=0.1)

    next_state = _constant_vector(delta_mass, 2.0)
    current_state = _constant_vector(delta_mass, 1.0)
    previous_state = _constant_vector(delta_mass, 0.5)

    forcing = material_newmark_step_forcing(
        delta_mass,
        delta_stiffness,
        config,
        next_state=next_state,
        current_state=current_state,
        previous_state=previous_state,
    )

    try:
        # δB u^n - δA(u^{n+1}+u^{n-1})
        # = 2*1 - (2 + 0.5) = -0.5
        assert np.allclose(forcing.getArray(readonly=True), -0.5)

    finally:
        forcing.destroy()
        next_state.destroy()
        current_state.destroy()
        previous_state.destroy()
        delta_mass.destroy()
        delta_stiffness.destroy()


def test_material_directional_gradient_identity_delta_mass():
    delta_mass = _identity_matrix(global_size=8)
    delta_stiffness = _zero_matrix(global_size=8)
    config = ElasticOperatorConfig(theta=0.25, dt=0.1)

    u1 = _constant_vector(delta_mass, 1.0)
    u2 = _constant_vector(delta_mass, 2.0)

    mu1 = _constant_vector(delta_mass, 10.0)
    mu2 = _constant_vector(delta_mass, 20.0)

    result = compute_newmark_material_directional_gradient(
        delta_mass,
        delta_stiffness,
        config,
        direct_snapshots=(u1, u2),
        equation_adjoint_vectors=(mu1, mu2),
    )

    try:
        # Step 1:
        # u^1=1, u^0=0, u^-1=0
        # forcing = 0 - 1 = -1
        # contribution = global_size * 10 * (-1) = -80
        #
        # Step 2:
        # u^2=2, u^1=1, u^0=0
        # forcing = 2*1 - 2 = 0
        # contribution = 0
        assert result.num_steps == 2
        assert np.isclose(result.value, -80.0)
        assert np.isclose(result.records[0].contribution, -80.0)
        assert np.isclose(result.records[1].contribution, 0.0)

        report = result.as_dict()

        assert report["kind"] == "material_directional_gradient_result"
        assert report["num_steps"] == 2
        assert np.isclose(report["value"], -80.0)

        payload = json.dumps(report, sort_keys=True)

        assert '"material_directional_gradient_result"' in payload

    finally:
        u1.destroy()
        u2.destroy()
        mu1.destroy()
        mu2.destroy()
        delta_mass.destroy()
        delta_stiffness.destroy()


def test_material_directional_gradient_zero_direction_is_zero():
    delta_mass = _zero_matrix(global_size=8)
    delta_stiffness = _zero_matrix(global_size=8)
    config = ElasticOperatorConfig(theta=0.25, dt=0.1)

    u1 = _constant_vector(delta_mass, 1.0)
    u2 = _constant_vector(delta_mass, 2.0)

    mu1 = _constant_vector(delta_mass, 10.0)
    mu2 = _constant_vector(delta_mass, 20.0)

    result = compute_newmark_material_directional_gradient(
        delta_mass,
        delta_stiffness,
        config,
        direct_snapshots=(u1, u2),
        equation_adjoint_vectors=(mu1, mu2),
    )

    try:
        assert result.num_steps == 2
        assert np.isclose(result.value, 0.0)
        assert result.max_forcing_norm == 0.0

    finally:
        u1.destroy()
        u2.destroy()
        mu1.destroy()
        mu2.destroy()
        delta_mass.destroy()
        delta_stiffness.destroy()


def test_material_directional_gradient_rejects_length_mismatch():
    delta_mass = _identity_matrix(global_size=8)
    delta_stiffness = _zero_matrix(global_size=8)
    config = ElasticOperatorConfig(theta=0.25, dt=0.1)

    u1 = _constant_vector(delta_mass, 1.0)
    mu1 = _constant_vector(delta_mass, 10.0)
    mu2 = _constant_vector(delta_mass, 20.0)

    try:
        with pytest.raises(ValueError):
            compute_newmark_material_directional_gradient(
                delta_mass,
                delta_stiffness,
                config,
                direct_snapshots=(u1,),
                equation_adjoint_vectors=(mu1, mu2),
            )

    finally:
        u1.destroy()
        mu1.destroy()
        mu2.destroy()
        delta_mass.destroy()
        delta_stiffness.destroy()
