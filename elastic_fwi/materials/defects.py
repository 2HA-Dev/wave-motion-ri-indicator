from __future__ import annotations

from dataclasses import dataclass
from math import hypot, isfinite
from typing import Iterable

import numpy as np

from elastic_fwi.materials.orthotropic import (
    LEGACY_HEALTHY_MATERIAL,
    OrthotropicPlaneStrainConstants,
)


Point2D = tuple[float, float]


@dataclass(frozen=True)
class CircularDefectSpec:
    """
    Analytical specification of a circular 2-D penetrable inclusion.

    This class does not depend on FEniCSx. It only describes the geometry and
    contrast of a defect. The DG0/FEniCSx layer decides how cells are marked
    from this analytical description.
    """

    name: str
    center: Point2D
    radius: float
    material_factor: float | None = None
    norm_order: int = 2
    rho_factor: float | None = None
    stiffness_factor: float | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("The defect name must be a non-empty string.")

        if len(self.center) != 2:
            raise ValueError("The defect center must contain exactly two coordinates.")

        cx = float(self.center[0])
        cy = float(self.center[1])
        radius = float(self.radius)

        material_factor = (
            None
            if self.material_factor is None
            else float(self.material_factor)
        )
        rho_factor = (
            None
            if self.rho_factor is None
            else float(self.rho_factor)
        )
        stiffness_factor = (
            None
            if self.stiffness_factor is None
            else float(self.stiffness_factor)
        )

        if not (isfinite(cx) and isfinite(cy)):
            raise ValueError("The defect center coordinates must be finite.")

        if not isfinite(radius) or radius <= 0.0:
            raise ValueError("The defect radius must be a strictly positive finite value.")

        component_mode = (
            rho_factor is not None
            or stiffness_factor is not None
        )

        if material_factor is not None and component_mode:
            raise ValueError(
                "Use either material_factor or the pair "
                "rho_factor/stiffness_factor, not both."
            )

        if not component_mode:
            if material_factor is None:
                material_factor = 0.8

            if not isfinite(material_factor) or material_factor <= 0.0:
                raise ValueError(
                    "The material factor must be a strictly positive finite value."
                )

            rho_factor = material_factor
            stiffness_factor = material_factor
        else:
            if rho_factor is None or stiffness_factor is None:
                raise ValueError(
                    "rho_factor and stiffness_factor must be provided together."
                )

            if not isfinite(rho_factor) or rho_factor <= 0.0:
                raise ValueError(
                    "The density factor must be a strictly positive finite value."
                )

            if not isfinite(stiffness_factor) or stiffness_factor <= 0.0:
                raise ValueError(
                    "The stiffness factor must be a strictly positive finite value."
                )

        if self.norm_order not in (1, 2):
            raise ValueError("Only norm_order=1 or norm_order=2 is supported.")

        object.__setattr__(self, "center", (cx, cy))
        object.__setattr__(self, "radius", radius)
        object.__setattr__(self, "material_factor", material_factor)
        object.__setattr__(self, "rho_factor", rho_factor)
        object.__setattr__(self, "stiffness_factor", stiffness_factor)

    @property
    def center_x(self) -> float:
        return self.center[0]

    @property
    def center_y(self) -> float:
        return self.center[1]

    def distance_to(self, point: Point2D) -> float:
        if len(point) != 2:
            raise ValueError("A 2-D point must contain exactly two coordinates.")

        x = float(point[0])
        y = float(point[1])

        if not (isfinite(x) and isfinite(y)):
            raise ValueError("Point coordinates must be finite.")

        dx = abs(x - self.center_x)
        dy = abs(y - self.center_y)

        if self.norm_order == 1:
            return dx + dy

        return hypot(dx, dy)

    def contains_point(self, point: Point2D, *, tol: float = 0.0) -> bool:
        tol = float(tol)

        if not isfinite(tol) or tol < 0.0:
            raise ValueError("The tolerance must be a non-negative finite value.")

        return self.distance_to(point) <= self.radius + tol

    def material_constants(
        self,
        healthy_material: OrthotropicPlaneStrainConstants = LEGACY_HEALTHY_MATERIAL,
    ) -> OrthotropicPlaneStrainConstants:
        return healthy_material.scaled_components(
            rho_factor=self.rho_factor,
            stiffness_factor=self.stiffness_factor,
        )


@dataclass(frozen=True)
class DefectScenario:
    """
    Collection of analytical defects.

    The ordering of the defects is preserved because the legacy C++
    generator applies the inclusions sequentially.
    """

    name: str
    defects: tuple[CircularDefectSpec, ...]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("The scenario name must be a non-empty string.")

        defects = tuple(self.defects)
        names = [defect.name for defect in defects]

        if len(names) != len(set(names)):
            raise ValueError("Defect names must be unique inside a scenario.")

        object.__setattr__(self, "defects", defects)

    def active_defects_at_point(
        self,
        point: Point2D,
        *,
        tol: float = 0.0,
    ) -> tuple[CircularDefectSpec, ...]:
        return tuple(defect for defect in self.defects if defect.contains_point(point, tol=tol))

    def first_defect_at_point(
        self,
        point: Point2D,
        *,
        tol: float = 0.0,
    ) -> CircularDefectSpec | None:
        for defect in self.defects:
            if defect.contains_point(point, tol=tol):
                return defect
        return None

    def last_defect_at_point(
        self,
        point: Point2D,
        *,
        tol: float = 0.0,
    ) -> CircularDefectSpec | None:
        selected = None
        for defect in self.defects:
            if defect.contains_point(point, tol=tol):
                selected = defect
        return selected


def ensure_defect_tuple(defects: Iterable[CircularDefectSpec]) -> tuple[CircularDefectSpec, ...]:
    return tuple(defects)


def _as_points_array(points: object) -> np.ndarray:
    array = np.asarray(points, dtype=float)

    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError("Points must be provided as an array of shape (n_points, 2).")

    if not np.all(np.isfinite(array)):
        raise ValueError("All point coordinates must be finite.")

    return array


def _validate_tolerance(tol: float) -> float:
    tol = float(tol)

    if not isfinite(tol) or tol < 0.0:
        raise ValueError("The tolerance must be a non-negative finite value.")

    return tol


def defect_membership_mask(
    points: object,
    defect: CircularDefectSpec,
    *,
    tol: float = 0.0,
) -> np.ndarray:
    """
    Return a boolean mask indicating which points belong to one defect.
    """

    array = _as_points_array(points)
    tol = _validate_tolerance(tol)

    dx = np.abs(array[:, 0] - defect.center_x)
    dy = np.abs(array[:, 1] - defect.center_y)

    if defect.norm_order == 1:
        distance = dx + dy
    elif defect.norm_order == 2:
        distance = np.hypot(dx, dy)
    else:
        raise ValueError("Only norm_order=1 or norm_order=2 is supported.")

    return distance <= defect.radius + tol


def assign_defect_indices_to_points(
    points: object,
    scenario: DefectScenario,
    *,
    tol: float = 0.0,
    conflict_policy: str = "last",
) -> np.ndarray:
    """
    Assign one defect index to each point.

    Returns
    -------
    np.ndarray
        Integer array of shape (n_points,). The value -1 denotes the healthy
        background. Values 0, 1, ... denote indices in scenario.defects.

    conflict_policy
    ---------------
    "last":
        Later defects overwrite earlier ones. This matches the sequential
        assignment logic used by the legacy C++ data generator.
    "first":
        First matching defect is kept.
    "error":
        Raise an error if at least one point belongs to several defects.
    """

    array = _as_points_array(points)
    tol = _validate_tolerance(tol)

    if conflict_policy not in {"first", "last", "error"}:
        raise ValueError("conflict_policy must be one of: 'first', 'last', 'error'.")

    indices = np.full(array.shape[0], -1, dtype=np.int64)
    hit_count = np.zeros(array.shape[0], dtype=np.int64)

    for defect_index, defect in enumerate(scenario.defects):
        mask = defect_membership_mask(array, defect, tol=tol)
        hit_count[mask] += 1

        if conflict_policy == "first":
            indices[np.logical_and(mask, indices == -1)] = defect_index
        else:
            indices[mask] = defect_index

    if conflict_policy == "error" and np.any(hit_count > 1):
        raise ValueError("At least one point belongs to several defects.")

    return indices


def assign_material_component_factors_to_points(
    points: object,
    scenario: DefectScenario,
    *,
    background_rho_factor: float = 1.0,
    background_stiffness_factor: float = 1.0,
    tol: float = 0.0,
    conflict_policy: str = "last",
) -> tuple[np.ndarray, np.ndarray]:
    """Assign density and stiffness scaling factors independently."""

    background_rho_factor = float(background_rho_factor)
    background_stiffness_factor = float(background_stiffness_factor)

    if (
        not isfinite(background_rho_factor)
        or background_rho_factor <= 0.0
    ):
        raise ValueError(
            "The background density factor must be a strictly positive "
            "finite value."
        )

    if (
        not isfinite(background_stiffness_factor)
        or background_stiffness_factor <= 0.0
    ):
        raise ValueError(
            "The background stiffness factor must be a strictly positive "
            "finite value."
        )

    indices = assign_defect_indices_to_points(
        points,
        scenario,
        tol=tol,
        conflict_policy=conflict_policy,
    )

    rho_factors = np.full(
        indices.shape,
        background_rho_factor,
        dtype=float,
    )
    stiffness_factors = np.full(
        indices.shape,
        background_stiffness_factor,
        dtype=float,
    )

    for defect_index, defect in enumerate(scenario.defects):
        mask = indices == defect_index
        rho_factors[mask] = defect.rho_factor
        stiffness_factors[mask] = defect.stiffness_factor

    return rho_factors, stiffness_factors


def assign_material_factors_to_points(
    points: object,
    scenario: DefectScenario,
    *,
    background_factor: float = 1.0,
    tol: float = 0.0,
    conflict_policy: str = "last",
) -> np.ndarray:
    """
    Assign material scaling factors to points.

    The healthy background receives background_factor. Points inside a defect
    receive the defect material_factor.
    """

    background_factor = float(background_factor)

    if not isfinite(background_factor) or background_factor <= 0.0:
        raise ValueError("The background factor must be a strictly positive finite value.")

    indices = assign_defect_indices_to_points(
        points,
        scenario,
        tol=tol,
        conflict_policy=conflict_policy,
    )

    factors = np.full(indices.shape, background_factor, dtype=float)

    for defect_index, defect in enumerate(scenario.defects):
        if defect.material_factor is None:
            raise ValueError(
                "assign_material_factors_to_points only supports defects "
                "with one common material_factor. Use "
                "assign_material_component_factors_to_points for independent "
                "density and stiffness contrasts."
            )

        factors[indices == defect_index] = defect.material_factor

    return factors
