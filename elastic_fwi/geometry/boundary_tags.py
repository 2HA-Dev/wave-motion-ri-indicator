from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np
import ufl
from dolfinx import fem, mesh
from mpi4py import MPI
from petsc4py import PETSc


@dataclass(frozen=True)
class BoundarySegment2D:
    """
    Coordinate-based 2-D boundary segment.

    A segment is defined by:
        x[fixed_axis] = fixed_value
        interval_min <= x[interval_axis] <= interval_max

    Axes:
        0 -> x
        1 -> y
    """

    name: str
    tag: int
    fixed_axis: int
    fixed_value: float
    interval_axis: int
    interval_min: float
    interval_max: float
    tol: float = 1.0e-10

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Boundary segment name must be non-empty.")

        if self.tag <= 0:
            raise ValueError("Boundary tag must be a strictly positive integer.")

        if self.fixed_axis not in (0, 1):
            raise ValueError("fixed_axis must be 0 or 1.")

        if self.interval_axis not in (0, 1):
            raise ValueError("interval_axis must be 0 or 1.")

        if self.fixed_axis == self.interval_axis:
            raise ValueError("fixed_axis and interval_axis must be distinct.")

        values = (
            float(self.fixed_value),
            float(self.interval_min),
            float(self.interval_max),
            float(self.tol),
        )

        if not all(isfinite(value) for value in values):
            raise ValueError("Boundary segment coordinates and tolerance must be finite.")

        if self.interval_min > self.interval_max:
            raise ValueError("interval_min must be <= interval_max.")

        if self.tol < 0.0:
            raise ValueError("Tolerance must be non-negative.")

        object.__setattr__(self, "fixed_value", float(self.fixed_value))
        object.__setattr__(self, "interval_min", float(self.interval_min))
        object.__setattr__(self, "interval_max", float(self.interval_max))
        object.__setattr__(self, "tol", float(self.tol))

    def marker(self, x: np.ndarray) -> np.ndarray:
        fixed = np.isclose(
            x[self.fixed_axis],
            self.fixed_value,
            rtol=0.0,
            atol=self.tol,
        )

        inside_interval = np.logical_and(
            x[self.interval_axis] >= self.interval_min - self.tol,
            x[self.interval_axis] <= self.interval_max + self.tol,
        )

        return np.logical_and(fixed, inside_interval)


def unit_square_pitch_catch_boundary_specs(
    *,
    reflection_tag: int = 1,
    transmission_tag: int = 2,
    left_tag: int = 3,
    right_tag: int = 4,
    tol: float = 1.0e-10,
) -> tuple[BoundarySegment2D, ...]:
    """
    Boundary tags for a unit-square pitch-catch surrogate.

    Convention:
        Gamma_1 / reflection    : y = 0
        Gamma_2 / transmission  : y = 1
        left lateral boundary   : x = 0
        right lateral boundary  : x = 1
    """

    return (
        BoundarySegment2D(
            name="gamma1_reflection",
            tag=reflection_tag,
            fixed_axis=1,
            fixed_value=0.0,
            interval_axis=0,
            interval_min=0.0,
            interval_max=1.0,
            tol=tol,
        ),
        BoundarySegment2D(
            name="gamma2_transmission",
            tag=transmission_tag,
            fixed_axis=1,
            fixed_value=1.0,
            interval_axis=0,
            interval_min=0.0,
            interval_max=1.0,
            tol=tol,
        ),
        BoundarySegment2D(
            name="left_lateral",
            tag=left_tag,
            fixed_axis=0,
            fixed_value=0.0,
            interval_axis=1,
            interval_min=0.0,
            interval_max=1.0,
            tol=tol,
        ),
        BoundarySegment2D(
            name="right_lateral",
            tag=right_tag,
            fixed_axis=0,
            fixed_value=1.0,
            interval_axis=1,
            interval_min=0.0,
            interval_max=1.0,
            tol=tol,
        ),
    )


def legacy_pitch_catch_boundary_specs(
    *,
    reflection_tag: int = 1,
    transmission_tag: int = 2,
    left_tag: int = 3,
    right_tag: int = 4,
    tol: float = 1.0e-10,
) -> tuple[BoundarySegment2D, ...]:
    """
    Legacy C++ coordinate convention on [75, 105] x [75, 105].

    Convention:
        Gamma_1 / reflection    : y = 75
        Gamma_2 / transmission  : y = 105
        lateral boundaries      : x = 75 and x = 105
    """

    return (
        BoundarySegment2D(
            name="gamma1_reflection",
            tag=reflection_tag,
            fixed_axis=1,
            fixed_value=75.0,
            interval_axis=0,
            interval_min=75.0,
            interval_max=105.0,
            tol=tol,
        ),
        BoundarySegment2D(
            name="gamma2_transmission",
            tag=transmission_tag,
            fixed_axis=1,
            fixed_value=105.0,
            interval_axis=0,
            interval_min=75.0,
            interval_max=105.0,
            tol=tol,
        ),
        BoundarySegment2D(
            name="left_lateral",
            tag=left_tag,
            fixed_axis=0,
            fixed_value=75.0,
            interval_axis=1,
            interval_min=75.0,
            interval_max=105.0,
            tol=tol,
        ),
        BoundarySegment2D(
            name="right_lateral",
            tag=right_tag,
            fixed_axis=0,
            fixed_value=105.0,
            interval_axis=1,
            interval_min=75.0,
            interval_max=105.0,
            tol=tol,
        ),
    )


def validate_boundary_specs(specs: tuple[BoundarySegment2D, ...]) -> None:
    if len(specs) == 0:
        raise ValueError("At least one boundary segment must be provided.")

    names = [spec.name for spec in specs]
    if len(names) != len(set(names)):
        raise ValueError("Boundary segment names must be unique.")

    # Repeated tag values are intentionally allowed.
    # They represent a union of disjoint boundary patches belonging to the same
    # physical source/receiver, e.g. a multi-element transducer array.


def build_boundary_meshtags(msh, specs: tuple[BoundarySegment2D, ...]):
    """
    Locate exterior facets and build DOLFINx MeshTags.

    The returned MeshTags object marks facets with the integer tags provided in
    specs.
    """

    validate_boundary_specs(specs)

    tdim = msh.topology.dim
    fdim = tdim - 1

    msh.topology.create_connectivity(fdim, tdim)
    msh.topology.create_connectivity(tdim, fdim)

    all_facets: list[np.ndarray] = []
    all_values: list[np.ndarray] = []

    for spec in specs:
        facets = mesh.locate_entities_boundary(msh, fdim, spec.marker)
        facets = np.asarray(facets, dtype=np.int32)
        values = np.full(facets.shape, spec.tag, dtype=np.int32)

        all_facets.append(facets)
        all_values.append(values)

    if all_facets:
        facets = np.concatenate(all_facets).astype(np.int32)
        values = np.concatenate(all_values).astype(np.int32)
    else:
        facets = np.array([], dtype=np.int32)
        values = np.array([], dtype=np.int32)

    if facets.size > 0:
        unique_facets, counts = np.unique(facets, return_counts=True)
        if np.any(counts > 1):
            duplicated = unique_facets[counts > 1]
            raise ValueError(f"Some local facets are matched by several boundary specs: {duplicated}.")

        order = np.argsort(facets)
        facets = facets[order]
        values = values[order]

    return mesh.meshtags(msh, fdim, facets, values)


def boundary_tag_counts_global(
    facet_tags,
    tags: tuple[int, ...],
    comm: MPI.Comm,
) -> dict[int, int]:
    counts: dict[int, int] = {}

    for tag in tags:
        local_count = int(np.count_nonzero(facet_tags.values == tag))
        global_count = int(comm.allreduce(local_count, op=MPI.SUM))
        counts[int(tag)] = global_count

    return counts


def boundary_measure_by_tag(msh, facet_tags, tag: int) -> float:
    ds = ufl.Measure("ds", domain=msh, subdomain_data=facet_tags)
    one = fem.Constant(msh, PETSc.ScalarType(1.0))
    local_value = fem.assemble_scalar(fem.form(one * ds(int(tag))))
    return float(msh.comm.allreduce(local_value, op=MPI.SUM))


def boundary_measures_by_tag(msh, facet_tags, tags: tuple[int, ...]) -> dict[int, float]:
    return {
        int(tag): boundary_measure_by_tag(msh, facet_tags, int(tag))
        for tag in tags
    }


def rectangular_pitch_catch_boundary_specs(
    *,
    domain_length_x: float,
    domain_length_y: float,
    source_segment_xmin: float,
    source_segment_xmax: float,
    receiver_segment_xmin: float,
    receiver_segment_xmax: float,
    reflection_tag: int = 1,
    transmission_tag: int = 2,
    left_tag: int = 3,
    right_tag: int = 4,
    tol: float = 1.0e-10,
) -> tuple[BoundarySegment2D, ...]:
    """
    Boundary tags for a rectangular pitch-catch configuration.

    Convention:
        Gamma_1 / source-reflection : y = 0,  x in [source_segment_xmin, source_segment_xmax]
        Gamma_2 / receiver          : y = Ly, x in [receiver_segment_xmin, receiver_segment_xmax]
        left lateral boundary       : x = 0
        right lateral boundary      : x = Lx
    """

    Lx = float(domain_length_x)
    Ly = float(domain_length_y)
    sx0 = float(source_segment_xmin)
    sx1 = float(source_segment_xmax)
    rx0 = float(receiver_segment_xmin)
    rx1 = float(receiver_segment_xmax)

    values = (Lx, Ly, sx0, sx1, rx0, rx1, float(tol))
    if not all(isfinite(value) for value in values):
        raise ValueError("Rectangular pitch-catch coordinates must be finite.")

    if Lx <= 0.0 or Ly <= 0.0:
        raise ValueError("domain_length_x and domain_length_y must be strictly positive.")

    if sx0 < 0.0 or sx1 > Lx or sx0 > sx1:
        raise ValueError("Source segment must satisfy 0 <= xmin <= xmax <= domain_length_x.")

    if rx0 < 0.0 or rx1 > Lx or rx0 > rx1:
        raise ValueError("Receiver segment must satisfy 0 <= xmin <= xmax <= domain_length_x.")

    return (
        BoundarySegment2D(
            name="gamma1_reflection",
            tag=reflection_tag,
            fixed_axis=1,
            fixed_value=0.0,
            interval_axis=0,
            interval_min=sx0,
            interval_max=sx1,
            tol=tol,
        ),
        BoundarySegment2D(
            name="gamma2_transmission",
            tag=transmission_tag,
            fixed_axis=1,
            fixed_value=Ly,
            interval_axis=0,
            interval_min=rx0,
            interval_max=rx1,
            tol=tol,
        ),
        BoundarySegment2D(
            name="left_lateral",
            tag=left_tag,
            fixed_axis=0,
            fixed_value=0.0,
            interval_axis=1,
            interval_min=0.0,
            interval_max=Ly,
            tol=tol,
        ),
        BoundarySegment2D(
            name="right_lateral",
            tag=right_tag,
            fixed_axis=0,
            fixed_value=Lx,
            interval_axis=1,
            interval_min=0.0,
            interval_max=Ly,
            tol=tol,
        ),
    )


def rectangular_pitch_catch_boundary_specs(
    *,
    domain_length_x: float,
    domain_length_y: float,
    source_segment_xmin: float,
    source_segment_xmax: float,
    receiver_segment_xmin: float,
    receiver_segment_xmax: float,
    reflection_tag: int = 1,
    transmission_tag: int = 2,
    left_tag: int = 3,
    right_tag: int = 4,
    tol: float = 1.0e-10,
) -> tuple[BoundarySegment2D, ...]:
    """
    Boundary tags for a rectangular pitch-catch configuration.

    Convention:
        Gamma_1 / source-reflection : y = 0,  x in [source_segment_xmin, source_segment_xmax]
        Gamma_2 / receiver          : y = Ly, x in [receiver_segment_xmin, receiver_segment_xmax]
        left lateral boundary       : x = 0
        right lateral boundary      : x = Lx
    """

    Lx = float(domain_length_x)
    Ly = float(domain_length_y)
    sx0 = float(source_segment_xmin)
    sx1 = float(source_segment_xmax)
    rx0 = float(receiver_segment_xmin)
    rx1 = float(receiver_segment_xmax)

    values = (Lx, Ly, sx0, sx1, rx0, rx1, float(tol))
    if not all(isfinite(value) for value in values):
        raise ValueError("Rectangular pitch-catch coordinates must be finite.")

    if Lx <= 0.0 or Ly <= 0.0:
        raise ValueError("domain_length_x and domain_length_y must be strictly positive.")

    if sx0 < 0.0 or sx1 > Lx or sx0 > sx1:
        raise ValueError("Source segment must satisfy 0 <= xmin <= xmax <= domain_length_x.")

    if rx0 < 0.0 or rx1 > Lx or rx0 > rx1:
        raise ValueError("Receiver segment must satisfy 0 <= xmin <= xmax <= domain_length_x.")

    return (
        BoundarySegment2D(
            name="gamma1_reflection",
            tag=reflection_tag,
            fixed_axis=1,
            fixed_value=0.0,
            interval_axis=0,
            interval_min=sx0,
            interval_max=sx1,
            tol=tol,
        ),
        BoundarySegment2D(
            name="gamma2_transmission",
            tag=transmission_tag,
            fixed_axis=1,
            fixed_value=Ly,
            interval_axis=0,
            interval_min=rx0,
            interval_max=rx1,
            tol=tol,
        ),
        BoundarySegment2D(
            name="left_lateral",
            tag=left_tag,
            fixed_axis=0,
            fixed_value=0.0,
            interval_axis=1,
            interval_min=0.0,
            interval_max=Ly,
            tol=tol,
        ),
        BoundarySegment2D(
            name="right_lateral",
            tag=right_tag,
            fixed_axis=0,
            fixed_value=Lx,
            interval_axis=1,
            interval_min=0.0,
            interval_max=Ly,
            tol=tol,
        ),
    )
