from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class OrthotropicPlaneStrainConstants:
    """
    Orthotropic 2-D plane-strain material constants.

    Convention used in the object-oriented code:
        rho : density
        C11 : stiffness coefficient along x
        C22 : stiffness coefficient along y
        C12 : coupling stiffness
        C66 : in-plane shear stiffness

    Plane strain, not plane stress:
        the five constants are the corresponding sub-block of the
        three-dimensional stiffness tensor, taken without the reduction
        Q_ij = C_ij - C_i3 C_3j / C_33 that plane stress would require.
        ``OrthotropicPlaneStressConstants`` is a deprecated alias of this
        class, with identical values.

    Legacy C++ convention:
        the historical code names the shear-like coefficient C33.
        In this package, this quantity is represented as C66.
    """

    rho: float
    C11: float
    C22: float
    C12: float
    C66: float

    def scaled(self, factor: float) -> "OrthotropicPlaneStrainConstants":
        """Scale density and stiffnesses by one common legacy factor."""
        return self.scaled_components(
            rho_factor=factor,
            stiffness_factor=factor,
        )

    def scaled_components(
        self,
        *,
        rho_factor: float,
        stiffness_factor: float,
    ) -> "OrthotropicPlaneStrainConstants":
        """Scale density and all elastic stiffnesses independently."""
        rho_factor = float(rho_factor)
        stiffness_factor = float(stiffness_factor)

        if not isfinite(rho_factor) or rho_factor <= 0.0:
            raise ValueError(
                "The density scaling factor must be strictly positive and finite."
            )
        if not isfinite(stiffness_factor) or stiffness_factor <= 0.0:
            raise ValueError(
                "The stiffness scaling factor must be strictly positive and finite."
            )

        return OrthotropicPlaneStrainConstants(
            rho=self.rho * rho_factor,
            C11=self.C11 * stiffness_factor,
            C22=self.C22 * stiffness_factor,
            C12=self.C12 * stiffness_factor,
            C66=self.C66 * stiffness_factor,
        )

    def as_tuple(self) -> tuple[float, float, float, float, float]:
        return (self.rho, self.C11, self.C22, self.C12, self.C66)


LEGACY_HEALTHY_MATERIAL = OrthotropicPlaneStrainConstants(
    rho=4.5,
    C11=16.24,
    C22=18.07,
    C12=6.9,
    C66=4.67,
)


def legacy_defect_material(factor: float = 0.8) -> OrthotropicPlaneStrainConstants:
    """
    Return the defect material used by the historical C++ data generator.

    The default factor 0.8 corresponds to a -20% contrast with respect to the
    healthy material.
    """
    return LEGACY_HEALTHY_MATERIAL.scaled(factor)


# Deprecated alias kept for backward compatibility.  Same class, same values.
OrthotropicPlaneStressConstants = OrthotropicPlaneStrainConstants
