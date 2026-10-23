from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from dolfinx import fem
from petsc4py import PETSc

from elastic_fwi.observations.boundary_observation import (
    BoundaryObservationSpec,
    boundary_l2_norm_squared,
)


@dataclass(frozen=True)
class BoundaryObservationTimeSeriesRecord:
    """One boundary observation value at one time step."""

    step: int
    time: float
    observation_name: str
    tag: int
    component: str
    norm_squared: float

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "step": self.step,
            "time": self.time,
            "observation_name": self.observation_name,
            "tag": self.tag,
            "component": self.component,
            "norm_squared": self.norm_squared,
        }


def displacement_vector_to_function(
    V,
    vector: PETSc.Vec,
    *,
    name: str = "u_snapshot",
) -> fem.Function:
    """
    Convert a PETSc displacement vector with the same layout as V into a Function.

    Only owned entries are copied first; ghost values are then updated with
    scatter_forward().
    """

    function = fem.Function(V)
    function.name = name

    owned_size = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
    local_values = np.asarray(vector.getArray(readonly=True))

    if local_values.size != owned_size:
        raise ValueError(
            f"Incompatible local vector size: got {local_values.size}, "
            f"expected {owned_size}."
        )

    function.x.array[:owned_size] = local_values

    if function.x.array.size > owned_size:
        function.x.array[owned_size:] = 0.0

    function.x.scatter_forward()

    return function


def evaluate_boundary_observation_from_vector(
    V,
    vector: PETSc.Vec,
    facet_tags,
    observation: BoundaryObservationSpec,
    *,
    name: str = "u_snapshot",
) -> float:
    function = displacement_vector_to_function(V, vector, name=name)

    return boundary_l2_norm_squared(
        function,
        facet_tags,
        observation,
    )


def evaluate_boundary_observation_time_series(
    V,
    snapshots: list[tuple[int, float, PETSc.Vec]],
    facet_tags,
    observation: BoundaryObservationSpec,
) -> list[BoundaryObservationTimeSeriesRecord]:
    records: list[BoundaryObservationTimeSeriesRecord] = []

    for step, time, vector in snapshots:
        norm_squared = evaluate_boundary_observation_from_vector(
            V,
            vector,
            facet_tags,
            observation,
            name=f"{observation.name}_step_{step}",
        )

        records.append(
            BoundaryObservationTimeSeriesRecord(
                step=int(step),
                time=float(time),
                observation_name=observation.name,
                tag=int(observation.tag),
                component=observation.component,
                norm_squared=float(norm_squared),
            )
        )

    return records
