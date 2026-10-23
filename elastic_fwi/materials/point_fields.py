from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np

from elastic_fwi.materials.defects import (
    DefectScenario,
    assign_defect_indices_to_points,
    assign_material_component_factors_to_points,
)
from elastic_fwi.materials.orthotropic import (
    LEGACY_HEALTHY_MATERIAL,
    OrthotropicPlaneStrainConstants,
)


@dataclass(frozen=True)
class OrthotropicPointFields:
    """
    Material fields evaluated at a finite set of 2-D points.

    The intended first use is to evaluate material constants at cell centres
    before transferring them to DG0 functions in FEniCSx.
    """

    rho: np.ndarray
    C11: np.ndarray
    C22: np.ndarray
    C12: np.ndarray
    C66: np.ndarray

    def __post_init__(self) -> None:
        arrays = {
            "rho": np.asarray(self.rho, dtype=float),
            "C11": np.asarray(self.C11, dtype=float),
            "C22": np.asarray(self.C22, dtype=float),
            "C12": np.asarray(self.C12, dtype=float),
            "C66": np.asarray(self.C66, dtype=float),
        }

        shapes = {name: array.shape for name, array in arrays.items()}

        if any(array.ndim != 1 for array in arrays.values()):
            raise ValueError("All material fields must be one-dimensional arrays.")

        if len(set(shapes.values())) != 1:
            raise ValueError(f"All material fields must have the same shape, got {shapes}.")

        for name, array in arrays.items():
            if not np.all(np.isfinite(array)):
                raise ValueError(f"Material field {name} contains non-finite values.")

        if np.any(arrays["rho"] <= 0.0):
            raise ValueError("The density field must be strictly positive.")

        object.__setattr__(self, "rho", arrays["rho"])
        object.__setattr__(self, "C11", arrays["C11"])
        object.__setattr__(self, "C22", arrays["C22"])
        object.__setattr__(self, "C12", arrays["C12"])
        object.__setattr__(self, "C66", arrays["C66"])

    @property
    def size(self) -> int:
        return int(self.rho.size)

    def as_dict(self) -> dict[str, np.ndarray]:
        return {
            "rho": self.rho,
            "C11": self.C11,
            "C22": self.C22,
            "C12": self.C12,
            "C66": self.C66,
        }

    def stacked(self) -> np.ndarray:
        """
        Return an array of shape (n_points, 5) with columns
        rho, C11, C22, C12, C66.
        """
        return np.column_stack((self.rho, self.C11, self.C22, self.C12, self.C66))


def _validate_material(material: OrthotropicPlaneStrainConstants) -> OrthotropicPlaneStrainConstants:
    values = material.as_tuple()

    if not all(isfinite(float(value)) for value in values):
        raise ValueError("Material constants must be finite.")

    if material.rho <= 0.0:
        raise ValueError("The density must be strictly positive.")

    return material


def homogeneous_material_fields_at_points(
    points: object,
    material: OrthotropicPlaneStrainConstants = LEGACY_HEALTHY_MATERIAL,
) -> OrthotropicPointFields:
    """
    Build homogeneous material fields on a point cloud.
    """

    material = _validate_material(material)
    points_array = np.asarray(points, dtype=float)

    if points_array.ndim != 2 or points_array.shape[1] != 2:
        raise ValueError("Points must be provided as an array of shape (n_points, 2).")

    if not np.all(np.isfinite(points_array)):
        raise ValueError("All point coordinates must be finite.")

    n_points = points_array.shape[0]

    return OrthotropicPointFields(
        rho=np.full(n_points, material.rho, dtype=float),
        C11=np.full(n_points, material.C11, dtype=float),
        C22=np.full(n_points, material.C22, dtype=float),
        C12=np.full(n_points, material.C12, dtype=float),
        C66=np.full(n_points, material.C66, dtype=float),
    )


def material_fields_from_factors(
    factors: object,
    material: OrthotropicPlaneStrainConstants = LEGACY_HEALTHY_MATERIAL,
) -> OrthotropicPointFields:
    """
    Build material fields by multiplying a healthy material by pointwise factors.
    """

    material = _validate_material(material)
    factor_array = np.asarray(factors, dtype=float)

    if factor_array.ndim != 1:
        raise ValueError("Material factors must be a one-dimensional array.")

    if not np.all(np.isfinite(factor_array)):
        raise ValueError("Material factors must be finite.")

    if np.any(factor_array <= 0.0):
        raise ValueError("Material factors must be strictly positive.")

    return material_fields_from_component_factors(
        factor_array,
        factor_array,
        material,
    )


def material_fields_from_component_factors(
    rho_factors: object,
    stiffness_factors: object,
    material: OrthotropicPlaneStrainConstants = LEGACY_HEALTHY_MATERIAL,
) -> OrthotropicPointFields:
    """Build fields from independent density and stiffness factors."""

    material = _validate_material(material)
    rho_factor_array = np.asarray(rho_factors, dtype=float)
    stiffness_factor_array = np.asarray(stiffness_factors, dtype=float)

    if rho_factor_array.ndim != 1:
        raise ValueError(
            "Density factors must be a one-dimensional array."
        )
    if stiffness_factor_array.ndim != 1:
        raise ValueError(
            "Stiffness factors must be a one-dimensional array."
        )
    if rho_factor_array.shape != stiffness_factor_array.shape:
        raise ValueError(
            "Density and stiffness factor arrays must have the same shape."
        )
    if not np.all(np.isfinite(rho_factor_array)):
        raise ValueError("Density factors must be finite.")
    if not np.all(np.isfinite(stiffness_factor_array)):
        raise ValueError("Stiffness factors must be finite.")
    if np.any(rho_factor_array <= 0.0):
        raise ValueError("Density factors must be strictly positive.")
    if np.any(stiffness_factor_array <= 0.0):
        raise ValueError("Stiffness factors must be strictly positive.")

    return OrthotropicPointFields(
        rho=material.rho * rho_factor_array,
        C11=material.C11 * stiffness_factor_array,
        C22=material.C22 * stiffness_factor_array,
        C12=material.C12 * stiffness_factor_array,
        C66=material.C66 * stiffness_factor_array,
    )


def defective_material_fields_at_points(
    points: object,
    scenario: DefectScenario,
    *,
    healthy_material: OrthotropicPlaneStrainConstants = LEGACY_HEALTHY_MATERIAL,
    background_factor: float = 1.0,
    tol: float = 0.0,
    conflict_policy: str = "last",
) -> OrthotropicPointFields:
    """
    Build material fields on a point cloud from an analytical defect scenario.
    """

    rho_factors, stiffness_factors = (
        assign_material_component_factors_to_points(
            points,
            scenario,
            background_rho_factor=background_factor,
            background_stiffness_factor=background_factor,
            tol=tol,
            conflict_policy=conflict_policy,
        )
    )

    return material_fields_from_component_factors(
        rho_factors,
        stiffness_factors,
        healthy_material,
    )


def material_indices_and_fields_at_points(
    points: object,
    scenario: DefectScenario,
    *,
    healthy_material: OrthotropicPlaneStrainConstants = LEGACY_HEALTHY_MATERIAL,
    background_factor: float = 1.0,
    tol: float = 0.0,
    conflict_policy: str = "last",
) -> tuple[np.ndarray, OrthotropicPointFields]:
    """
    Return both defect indices and material fields.

    This is useful for diagnostics and DG0 tagging.
    """

    indices = assign_defect_indices_to_points(
        points,
        scenario,
        tol=tol,
        conflict_policy=conflict_policy,
    )

    fields = defective_material_fields_at_points(
        points,
        scenario,
        healthy_material=healthy_material,
        background_factor=background_factor,
        tol=tol,
        conflict_policy=conflict_policy,
    )

    return indices, fields
