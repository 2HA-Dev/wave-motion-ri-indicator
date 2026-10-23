from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil, isfinite, sqrt
from typing import Any

import numpy as np
from dolfinx import fem

from elastic_fwi.geometry.cell_centers import local_cell_midpoints_2d


@dataclass(frozen=True)
class AbsorbingLayerPlan:
    kind: str
    enabled: bool
    domain_length_x: float
    domain_length_y: float
    source_frequency: float
    rho: float
    C11: float
    C22: float
    C66: float
    c_min_estimate: float
    c_max_estimate: float
    lambda_min: float
    h_reference: float
    width: float
    width_wavelengths: float
    min_cells: int
    max_fraction: float
    sigma_max: float
    strength: float
    polynomial_order: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _positive_float(value: float, name: str) -> float:
    value = float(value)
    if not isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be strictly positive and finite.")
    return value


def _nonnegative_float(value: float, name: str) -> float:
    value = float(value)
    if not isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be non-negative and finite.")
    return value


def _estimate_wave_speed_bounds(
    *,
    rho: float,
    C11: float,
    C22: float,
    C66: float,
) -> tuple[float, float]:
    rho = _positive_float(rho, "rho")
    C11 = _positive_float(C11, "C11")
    C22 = _positive_float(C22, "C22")
    C66 = _positive_float(C66, "C66")

    c_min = sqrt(min(C11, C22, C66) / rho)
    c_max = sqrt(max(C11, C22, C66) / rho)

    return float(c_min), float(c_max)


def plan_absorbing_layer(
    *,
    domain_length_x: float,
    domain_length_y: float,
    source_frequency: float,
    rho: float,
    C11: float,
    C22: float,
    C66: float,
    h_reference: float,
    width: float | None = None,
    width_wavelengths: float = 1.5,
    min_cells: int = 6,
    max_fraction: float = 0.45,
    strength: float = 3.0,
    polynomial_order: float = 2.0,
) -> AbsorbingLayerPlan:
    domain_length_x = _positive_float(domain_length_x, "domain_length_x")
    domain_length_y = _positive_float(domain_length_y, "domain_length_y")
    source_frequency = _positive_float(source_frequency, "source_frequency")
    h_reference = _positive_float(h_reference, "h_reference")
    width_wavelengths = _positive_float(width_wavelengths, "width_wavelengths")
    strength = _positive_float(strength, "strength")
    polynomial_order = _positive_float(polynomial_order, "polynomial_order")
    max_fraction = _positive_float(max_fraction, "max_fraction")
    min_cells = int(min_cells)

    if min_cells <= 0:
        raise ValueError("min_cells must be strictly positive.")

    if max_fraction >= 0.5:
        raise ValueError("max_fraction must be smaller than 0.5.")

    rho = _positive_float(rho, "rho")
    C11 = _positive_float(C11, "C11")
    C22 = _positive_float(C22, "C22")
    C66 = _positive_float(C66, "C66")

    c_min, c_max = _estimate_wave_speed_bounds(
        rho=rho,
        C11=C11,
        C22=C22,
        C66=C66,
    )

    lambda_min = c_min / source_frequency

    automatic_width = max(
        width_wavelengths * lambda_min,
        float(min_cells) * h_reference,
    )

    if width is None:
        selected_width = automatic_width
    else:
        selected_width = _positive_float(width, "width")

    max_width = max_fraction * min(domain_length_x, domain_length_y)
    selected_width = min(selected_width, max_width)

    sigma_max = strength * c_max / selected_width

    return AbsorbingLayerPlan(
        kind="absorbing_sponge_layer",
        enabled=True,
        domain_length_x=domain_length_x,
        domain_length_y=domain_length_y,
        source_frequency=source_frequency,
        rho=rho,
        C11=C11,
        C22=C22,
        C66=C66,
        c_min_estimate=c_min,
        c_max_estimate=c_max,
        lambda_min=lambda_min,
        h_reference=h_reference,
        width=selected_width,
        width_wavelengths=width_wavelengths,
        min_cells=min_cells,
        max_fraction=max_fraction,
        sigma_max=sigma_max,
        strength=strength,
        polynomial_order=polynomial_order,
    )


def create_absorbing_layer_dg0(
    msh,
    *,
    domain_length_x: float,
    domain_length_y: float,
    plan: AbsorbingLayerPlan,
):
    """
    Create a DG0 damping coefficient sigma(x).

    sigma(x)=0 inside the physical region and increases polynomially inside
    the absorbing layer near the outer boundary.
    """

    V0 = fem.functionspace(msh, ("DG", 0))
    sigma = fem.Function(V0)
    sigma.name = "absorbing_sigma"

    midpoints = local_cell_midpoints_2d(msh, include_ghosts=True)

    if midpoints.shape[0] != sigma.x.array.shape[0]:
        raise RuntimeError(
            "DG0 damping field and local cell midpoint array have inconsistent sizes."
        )

    x = midpoints[:, 0]
    y = midpoints[:, 1]

    distance_to_boundary = np.minimum.reduce(
        [
            x,
            domain_length_x - x,
            y,
            domain_length_y - y,
        ]
    )

    s = np.clip((plan.width - distance_to_boundary) / plan.width, 0.0, 1.0)
    values = plan.sigma_max * s**plan.polynomial_order

    sigma.x.array[:] = values
    sigma.x.scatter_forward()

    return sigma


def absorbing_layer_summary_from_function(sigma) -> dict[str, Any]:
    from mpi4py import MPI

    values = np.asarray(sigma.x.array, dtype=float)
    comm = sigma.function_space.mesh.comm

    local_min = float(np.min(values)) if values.size else float("inf")
    local_max = float(np.max(values)) if values.size else 0.0
    local_nonzero = int(np.count_nonzero(values > 0.0))
    local_size = int(values.size)

    global_min = float(comm.allreduce(local_min, op=MPI.MIN))
    global_max = float(comm.allreduce(local_max, op=MPI.MAX))
    global_nonzero = int(comm.allreduce(local_nonzero, op=MPI.SUM))
    global_size = int(comm.allreduce(local_size, op=MPI.SUM))

    if global_size == 0:
        global_min = 0.0

    return {
        "kind": "absorbing_sponge_layer_field",
        "global_min": global_min,
        "global_max": global_max,
        "global_nonzero_cells": global_nonzero,
        "global_cells": global_size,
    }
