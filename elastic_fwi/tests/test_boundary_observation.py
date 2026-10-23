import numpy as np
import pytest

pytest.importorskip("dolfinx")

from mpi4py import MPI
from dolfinx import fem, mesh

from elastic_fwi.geometry.boundary_tags import (
    build_boundary_meshtags,
    unit_square_pitch_catch_boundary_specs,
)
from elastic_fwi.observations.boundary_observation import (
    BoundaryObservationSpec,
    boundary_inner_product,
    boundary_l2_misfit_cost,
    boundary_l2_misfit_norm_squared,
    boundary_l2_norm_squared,
    gamma1_reflection_observation,
    gamma2_transmission_observation,
)
from elastic_fwi.observations.residuals import (
    boundary_residual_report,
)


def _create_mesh_and_tags():
    msh = mesh.create_unit_square(MPI.COMM_WORLD, 4, 4)
    facet_tags = build_boundary_meshtags(msh, unit_square_pitch_catch_boundary_specs())
    return msh, facet_tags


def _create_vector_space(msh):
    return fem.functionspace(msh, ("Lagrange", 1, (msh.geometry.dim,)))


def _constant_vector_function(V, values, name):
    function = fem.Function(V)
    function.name = name

    vx, vy = values

    def expr(x):
        return np.vstack(
            (
                np.full(x.shape[1], vx, dtype=float),
                np.full(x.shape[1], vy, dtype=float),
            )
        )

    function.interpolate(expr)
    function.x.scatter_forward()
    return function


def test_boundary_l2_norm_squared_for_constant_vector_field():
    msh, facet_tags = _create_mesh_and_tags()
    V = _create_vector_space(msh)

    u = _constant_vector_function(V, (2.0, -1.0), "u")

    gamma1_vector = gamma1_reflection_observation(component="vector")
    gamma1_x = gamma1_reflection_observation(component="x")
    gamma1_y = gamma1_reflection_observation(component="y")

    norm_vector = boundary_l2_norm_squared(u, facet_tags, gamma1_vector)
    norm_x = boundary_l2_norm_squared(u, facet_tags, gamma1_x)
    norm_y = boundary_l2_norm_squared(u, facet_tags, gamma1_y)

    assert norm_vector == pytest.approx(5.0)
    assert norm_x == pytest.approx(4.0)
    assert norm_y == pytest.approx(1.0)


def test_boundary_inner_product_for_constant_vector_fields():
    msh, facet_tags = _create_mesh_and_tags()
    V = _create_vector_space(msh)

    u = _constant_vector_function(V, (2.0, -1.0), "u")
    v = _constant_vector_function(V, (1.0, 3.0), "v")

    gamma2_vector = gamma2_transmission_observation(component="vector")
    gamma2_x = gamma2_transmission_observation(component="x")
    gamma2_y = gamma2_transmission_observation(component="y")

    inner_vector = boundary_inner_product(u, v, facet_tags, gamma2_vector)
    inner_x = boundary_inner_product(u, v, facet_tags, gamma2_x)
    inner_y = boundary_inner_product(u, v, facet_tags, gamma2_y)

    assert inner_vector == pytest.approx(-1.0)
    assert inner_x == pytest.approx(2.0)
    assert inner_y == pytest.approx(-3.0)


def test_boundary_misfit_norm_and_cost():
    msh, facet_tags = _create_mesh_and_tags()
    V = _create_vector_space(msh)

    simulated = _constant_vector_function(V, (2.0, -1.0), "simulated")
    observed = _constant_vector_function(V, (1.0, 1.0), "observed")

    gamma1_vector = gamma1_reflection_observation(component="vector")
    gamma1_x = gamma1_reflection_observation(component="x")
    gamma1_y = gamma1_reflection_observation(component="y")

    misfit_vector = boundary_l2_misfit_norm_squared(
        simulated,
        observed,
        facet_tags,
        gamma1_vector,
    )

    misfit_x = boundary_l2_misfit_norm_squared(
        simulated,
        observed,
        facet_tags,
        gamma1_x,
    )

    misfit_y = boundary_l2_misfit_norm_squared(
        simulated,
        observed,
        facet_tags,
        gamma1_y,
    )

    cost_vector = boundary_l2_misfit_cost(
        simulated,
        observed,
        facet_tags,
        gamma1_vector,
    )

    assert misfit_vector == pytest.approx(5.0)
    assert misfit_x == pytest.approx(1.0)
    assert misfit_y == pytest.approx(4.0)
    assert cost_vector == pytest.approx(2.5)


def test_boundary_residual_wrappers_and_report():
    msh, facet_tags = _create_mesh_and_tags()
    V = _create_vector_space(msh)

    simulated = _constant_vector_function(V, (2.0, -1.0), "simulated")
    observed = _constant_vector_function(V, (1.0, 1.0), "observed")

    observation = gamma2_transmission_observation(component="vector")

    report = boundary_residual_report(
        simulated,
        observed,
        facet_tags,
        observation,
    )

    assert report["observation"]["name"] == "gamma2_transmission"
    assert report["observation"]["tag"] == 2
    assert report["observation"]["component"] == "vector"
    assert report["norm_squared"] == pytest.approx(5.0)
    assert report["cost"] == pytest.approx(2.5)


def test_invalid_boundary_observation_spec_is_rejected():
    with pytest.raises(ValueError):
        BoundaryObservationSpec(name="", tag=1)

    with pytest.raises(ValueError):
        BoundaryObservationSpec(name="bad_tag", tag=0)

    with pytest.raises(ValueError):
        BoundaryObservationSpec(name="bad_component", tag=1, component="normal")
