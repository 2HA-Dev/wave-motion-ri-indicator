"""Shear-strain factor maps behind the C66 kernels, and their Cauchy--Schwarz split.

Both C66 kernels of the accompanying article pair two shear strains over the
time window,

    S_classical(x) = int_0^T gamma12(u)   gamma12(lambda)   dt,   (classical)
    S_RI(x)        = int_0^T gamma12(W1)  gamma12(Lambda2)  dt,   (residual interaction)

so each is an inner product in time of two factors.  Cauchy--Schwarz in the time
variable gives, pointwise in space,

    |S(x)| = rho(x) * sqrt(A_first(x)) * sqrt(A_second(x)),
    A_f(x) = int_0^T gamma12(f)^2 dt,      0 <= rho(x) <= 1,

which separates three distinct causes for a kernel to be large somewhere: the
first factor is energetic there, the second factor is energetic there, and the
two are aligned in time there.

The mirror-symmetry theorem explains why both kernels vanish on the axis through
the defect, but not why the classical pattern drifts along e2 while the
residual-interaction pattern stays centred.  The proposed mechanism is that the
classical kernel multiplies a factor focused at the defect, ``gamma12(lambda)``,
by one that is not, ``gamma12(u)``: the reference forward field illuminates the
whole path from the emitting array and carries no scale of its own in x2, so the
product inherits the geometry of the beam.  The residual-interaction kernel
multiplies two back-propagated factors, both focused.  This module measures the
three ingredients separately, so that each part of the mechanism can be checked
numerically.

Nothing here solves a PDE.  The wavefield snapshots are read as exported.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Sequence

import numpy as np


class ShearFactorError(ValueError):
    """Raised when a shear-factor computation cannot be carried out."""


@dataclass(frozen=True)
class ShearStrainCells:
    """Per-cell shear strain of a P1 vector field, with cell geometry."""

    shear_strain: np.ndarray
    areas: np.ndarray
    centroids: np.ndarray


@dataclass(frozen=True)
class FactorSpread:
    """Where a non-negative map sits along x2, and how far it spreads."""

    total_mass: float
    centroid_x1: float
    centroid_x2: float
    spread_x1: float
    spread_x2: float
    anisotropy_x2_over_x1: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def triangle_shear_strain(
    points: np.ndarray,
    triangles: np.ndarray,
    displacement: np.ndarray,
) -> ShearStrainCells:
    """Per-triangle engineering shear strain of a P1 vector displacement.

    On a P1 triangle the displacement gradient is constant, so

        gamma12 = d(u1)/dx2 + d(u2)/dx1

    is one value per cell and needs no projection.  Returned areas are the
    triangle areas, which serve as the DG0 cell measures of the L2 sensitivity
    density.
    """

    points = np.asarray(points, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.int64)
    displacement = np.asarray(displacement, dtype=np.float64)

    if points.ndim != 2 or points.shape[1] < 2:
        raise ShearFactorError("points must have shape (n, >=2).")
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ShearFactorError("triangles must have shape (m, 3).")
    if displacement.shape[0] != points.shape[0]:
        raise ShearFactorError("displacement and points must have the same length.")
    if displacement.ndim != 2 or displacement.shape[1] < 2:
        raise ShearFactorError("displacement must have shape (n, >=2).")

    coordinates = points[:, :2]
    values = displacement[:, :2]

    p0 = coordinates[triangles[:, 0]]
    p1 = coordinates[triangles[:, 1]]
    p2 = coordinates[triangles[:, 2]]

    edge1 = p1 - p0
    edge2 = p2 - p0

    # Signed doubled area; its absolute value gives the triangle measure.
    determinant = edge1[:, 0] * edge2[:, 1] - edge1[:, 1] * edge2[:, 0]
    if np.any(determinant == 0.0):
        raise ShearFactorError("Degenerate triangle with zero area encountered.")

    u0 = values[triangles[:, 0]]
    u1 = values[triangles[:, 1]]
    u2 = values[triangles[:, 2]]

    d1 = u1 - u0
    d2 = u2 - u0

    # Inverse of J = [edge1, edge2] applied to the nodal differences gives the
    # constant gradient: d(component)/dx = (dcomp1 * e2y - dcomp2 * e1y) / det,
    #                    d(component)/dy = (dcomp2 * e1x - dcomp1 * e2x) / det.
    du1_dx2 = (d2[:, 0] * edge1[:, 0] - d1[:, 0] * edge2[:, 0]) / determinant
    du2_dx1 = (d1[:, 1] * edge2[:, 1] - d2[:, 1] * edge1[:, 1]) / determinant

    shear = du1_dx2 + du2_dx1
    areas = 0.5 * np.abs(determinant)
    centroids = (p0 + p1 + p2) / 3.0

    return ShearStrainCells(shear_strain=shear, areas=areas, centroids=centroids)


def cauchy_schwarz_split(
    pair_integral: np.ndarray,
    first_energy: np.ndarray,
    second_energy: np.ndarray,
    *,
    relative_floor: float = 1.0e-12,
) -> np.ndarray:
    """Temporal coherence rho = |S| / sqrt(A_first * A_second), clipped to [0, 1].

    Cells where either factor carries no energy are reported as ``nan`` rather
    than as an arbitrary ratio: the coherence is undefined there.
    """

    pair_integral = np.asarray(pair_integral, dtype=np.float64)
    first_energy = np.asarray(first_energy, dtype=np.float64)
    second_energy = np.asarray(second_energy, dtype=np.float64)

    if not (pair_integral.shape == first_energy.shape == second_energy.shape):
        raise ShearFactorError("All three maps must have the same shape.")
    if np.any(first_energy < 0.0) or np.any(second_energy < 0.0):
        raise ShearFactorError("Energy maps must be non-negative.")

    denominator = np.sqrt(first_energy * second_energy)
    scale = float(np.max(denominator)) if denominator.size else 0.0
    threshold = relative_floor * scale

    coherence = np.full(pair_integral.shape, np.nan, dtype=np.float64)
    usable = denominator > threshold
    coherence[usable] = np.abs(pair_integral[usable]) / denominator[usable]

    # Cauchy--Schwarz guarantees rho <= 1; round-off can exceed it marginally.
    return np.clip(coherence, 0.0, 1.0)


def factor_spread(
    centroids: np.ndarray,
    weights: np.ndarray,
    areas: np.ndarray,
) -> FactorSpread:
    """Area-weighted centroid and standard deviation of a non-negative map.

    ``spread_x2`` is the quantity the mechanism is about: a factor focused at the
    defect has a small one, an illumination spanning the whole path between the
    arrays has a large one.
    """

    centroids = np.asarray(centroids, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    areas = np.asarray(areas, dtype=np.float64)

    if centroids.ndim != 2 or centroids.shape[1] < 2:
        raise ShearFactorError("centroids must have shape (n, >=2).")
    if weights.shape[0] != centroids.shape[0] or areas.shape[0] != centroids.shape[0]:
        raise ShearFactorError("weights, areas and centroids must agree in length.")
    if np.any(weights < 0.0):
        raise ShearFactorError("weights must be non-negative.")

    mass = weights * areas
    total = float(np.sum(mass, dtype=np.float64))
    if not math.isfinite(total) or total <= 0.0:
        raise ShearFactorError("The map carries no mass; spread is undefined.")

    mean_x1 = float(np.sum(mass * centroids[:, 0], dtype=np.float64) / total)
    mean_x2 = float(np.sum(mass * centroids[:, 1], dtype=np.float64) / total)

    var_x1 = float(
        np.sum(mass * (centroids[:, 0] - mean_x1) ** 2, dtype=np.float64) / total
    )
    var_x2 = float(
        np.sum(mass * (centroids[:, 1] - mean_x2) ** 2, dtype=np.float64) / total
    )

    spread_x1 = math.sqrt(max(var_x1, 0.0))
    spread_x2 = math.sqrt(max(var_x2, 0.0))

    return FactorSpread(
        total_mass=total,
        centroid_x1=mean_x1,
        centroid_x2=mean_x2,
        spread_x1=spread_x1,
        spread_x2=spread_x2,
        anisotropy_x2_over_x1=(
            spread_x2 / spread_x1 if spread_x1 > 0.0 else math.nan
        ),
    )


def marginal_profile_x2(
    centroids: np.ndarray,
    weights: np.ndarray,
    areas: np.ndarray,
    *,
    num_bins: int = 60,
    x2_min: float | None = None,
    x2_max: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Mass of a map per x2 slice, for plotting the extent along the beam."""

    centroids = np.asarray(centroids, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    areas = np.asarray(areas, dtype=np.float64)

    if num_bins < 2:
        raise ShearFactorError("num_bins must be at least 2.")

    lower = float(np.min(centroids[:, 1])) if x2_min is None else float(x2_min)
    upper = float(np.max(centroids[:, 1])) if x2_max is None else float(x2_max)
    if not upper > lower:
        raise ShearFactorError("x2_max must exceed x2_min.")

    edges = np.linspace(lower, upper, int(num_bins) + 1)
    index = np.clip(np.digitize(centroids[:, 1], edges) - 1, 0, int(num_bins) - 1)

    mass = weights * areas
    totals = np.bincount(index, weights=mass, minlength=int(num_bins))
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, totals


def correlation_against_reference(
    computed: np.ndarray,
    reference: np.ndarray,
    areas: np.ndarray,
) -> dict[str, float]:
    """Area-weighted correlation between a recomputed map and an exported one.

    Used to confirm that the shear factors are being paired in the right time
    convention: the recomputed kernel must reproduce the exported kernel, up to
    the overall sign of the gradient convention.
    """

    computed = np.asarray(computed, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    areas = np.asarray(areas, dtype=np.float64)

    if not (computed.shape == reference.shape == areas.shape):
        raise ShearFactorError("All inputs must have the same shape.")

    total = float(np.sum(areas, dtype=np.float64))
    if total <= 0.0:
        raise ShearFactorError("Areas must sum to a positive measure.")

    mean_computed = float(np.sum(areas * computed) / total)
    mean_reference = float(np.sum(areas * reference) / total)

    dc = computed - mean_computed
    dr = reference - mean_reference

    covariance = float(np.sum(areas * dc * dr) / total)
    var_computed = float(np.sum(areas * dc * dc) / total)
    var_reference = float(np.sum(areas * dr * dr) / total)

    denominator = math.sqrt(var_computed * var_reference)
    correlation = covariance / denominator if denominator > 0.0 else math.nan

    # Best least-squares scale mapping the computed map onto the reference.
    scale = (
        float(np.sum(areas * computed * reference) / np.sum(areas * computed * computed))
        if var_computed > 0.0
        else math.nan
    )

    return {
        "correlation": correlation,
        "abs_correlation": abs(correlation) if math.isfinite(correlation) else math.nan,
        "best_fit_scale": scale,
        "computed_rms": math.sqrt(max(var_computed, 0.0)),
        "reference_rms": math.sqrt(max(var_reference, 0.0)),
    }


def accumulate_pair(
    shear_first: Sequence[np.ndarray],
    shear_second: Sequence[np.ndarray],
    dt: float,
) -> dict[str, np.ndarray]:
    """Time-integrate one shear pair into its kernel and two energy maps."""

    if len(shear_first) != len(shear_second):
        raise ShearFactorError("Both shear sequences must have the same length.")
    if len(shear_first) == 0:
        raise ShearFactorError("At least one snapshot is required.")
    if dt <= 0.0:
        raise ShearFactorError("dt must be positive.")

    size = np.asarray(shear_first[0]).shape
    pair = np.zeros(size, dtype=np.float64)
    energy_first = np.zeros(size, dtype=np.float64)
    energy_second = np.zeros(size, dtype=np.float64)

    for first, second in zip(shear_first, shear_second, strict=True):
        first = np.asarray(first, dtype=np.float64)
        second = np.asarray(second, dtype=np.float64)
        pair += dt * first * second
        energy_first += dt * first * first
        energy_second += dt * second * second

    return {
        "pair_integral": pair,
        "first_energy": energy_first,
        "second_energy": energy_second,
    }
