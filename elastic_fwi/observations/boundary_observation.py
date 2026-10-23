from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import ufl
from dolfinx import fem
from mpi4py import MPI


_ALLOWED_COMPONENTS = {"vector", "x", "y"}


@dataclass(frozen=True)
class BoundaryObservationSpec:
    """
    Boundary observation specification.

    Parameters
    ----------
    name:
        Stable observation identifier.
    tag:
        Boundary tag where the observation is evaluated.
    component:
        Observed component:
            "vector" -> full vector displacement
            "x"      -> x-component
            "y"      -> y-component
    """

    name: str
    tag: int
    component: str = "vector"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Observation name must be non-empty.")

        if self.tag <= 0:
            raise ValueError("Observation boundary tag must be a strictly positive integer.")

        if self.component not in _ALLOWED_COMPONENTS:
            raise ValueError(
                f"Unsupported observation component {self.component!r}. "
                f"Allowed components are {sorted(_ALLOWED_COMPONENTS)}."
            )


def gamma1_reflection_observation(
    *,
    tag: int = 1,
    component: str = "vector",
) -> BoundaryObservationSpec:
    return BoundaryObservationSpec(
        name="gamma1_reflection",
        tag=tag,
        component=component,
    )


def gamma2_transmission_observation(
    *,
    tag: int = 2,
    component: str = "vector",
) -> BoundaryObservationSpec:
    return BoundaryObservationSpec(
        name="gamma2_transmission",
        tag=tag,
        component=component,
    )


def _component_inner(left, right, component: str):
    if component == "vector":
        return ufl.inner(left, right)

    if component == "x":
        return left[0] * right[0]

    if component == "y":
        return left[1] * right[1]

    raise ValueError(f"Unsupported observation component {component!r}.")


def _assemble_global_scalar(msh, expression) -> float:
    local_value = fem.assemble_scalar(fem.form(expression))
    global_value = msh.comm.allreduce(local_value, op=MPI.SUM)
    return float(np.real_if_close(global_value))


def boundary_inner_product(
    left,
    right,
    facet_tags,
    observation: BoundaryObservationSpec,
) -> float:
    """
    Compute the L2 boundary inner product on one tagged boundary.
    """

    msh = left.function_space.mesh
    ds = ufl.Measure("ds", domain=msh, subdomain_data=facet_tags)

    expression = _component_inner(left, right, observation.component) * ds(observation.tag)

    return _assemble_global_scalar(msh, expression)


def boundary_l2_norm_squared(
    function,
    facet_tags,
    observation: BoundaryObservationSpec,
) -> float:
    """
    Compute ||function||^2 on the observed boundary.
    """

    return boundary_inner_product(function, function, facet_tags, observation)


def boundary_l2_misfit_norm_squared(
    simulated,
    observed,
    facet_tags,
    observation: BoundaryObservationSpec,
) -> float:
    """
    Compute ||simulated - observed||^2 on the observed boundary.
    """

    residual = simulated - observed
    msh = simulated.function_space.mesh
    ds = ufl.Measure("ds", domain=msh, subdomain_data=facet_tags)

    expression = _component_inner(residual, residual, observation.component) * ds(observation.tag)

    return _assemble_global_scalar(msh, expression)


def boundary_l2_misfit_cost(
    simulated,
    observed,
    facet_tags,
    observation: BoundaryObservationSpec,
) -> float:
    """
    Compute 0.5 * ||simulated - observed||^2 on the observed boundary.
    """

    return 0.5 * boundary_l2_misfit_norm_squared(
        simulated,
        observed,
        facet_tags,
        observation,
    )
