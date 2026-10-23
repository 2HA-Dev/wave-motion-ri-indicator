from __future__ import annotations

from dataclasses import asdict

from elastic_fwi.observations.boundary_observation import (
    BoundaryObservationSpec,
    boundary_l2_misfit_cost,
    boundary_l2_misfit_norm_squared,
)


def boundary_residual_norm_squared(
    simulated,
    observed,
    facet_tags,
    observation: BoundaryObservationSpec,
) -> float:
    return boundary_l2_misfit_norm_squared(
        simulated,
        observed,
        facet_tags,
        observation,
    )


def boundary_residual_cost(
    simulated,
    observed,
    facet_tags,
    observation: BoundaryObservationSpec,
) -> float:
    return boundary_l2_misfit_cost(
        simulated,
        observed,
        facet_tags,
        observation,
    )


def boundary_residual_report(
    simulated,
    observed,
    facet_tags,
    observation: BoundaryObservationSpec,
) -> dict[str, object]:
    norm_squared = boundary_residual_norm_squared(
        simulated,
        observed,
        facet_tags,
        observation,
    )

    return {
        "observation": asdict(observation),
        "norm_squared": norm_squared,
        "cost": 0.5 * norm_squared,
    }
