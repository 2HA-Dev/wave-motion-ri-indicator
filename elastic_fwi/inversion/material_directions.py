from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from dolfinx import fem
from petsc4py import PETSc

from elastic_fwi.materials.dg0_fields import (
    create_named_dg0_function,
    create_scalar_dg0_space,
)
import ufl

from elastic_fwi.solvers.elastic_operator import assemble_matrix_from_form


@dataclass(frozen=True)
class HomogeneousMaterialDirection:
    rho: float = 0.0
    C11: float = 0.0
    C22: float = 0.0
    C12: float = 0.0
    C66: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "rho": float(self.rho),
            "C11": float(self.C11),
            "C22": float(self.C22),
            "C12": float(self.C12),
            "C66": float(self.C66),
        }


@dataclass(frozen=True)
class MaterialPerturbationMatrices:
    mass: PETSc.Mat
    stiffness: PETSc.Mat
    metadata: dict[str, Any]

    def destroy(self) -> None:
        self.mass.destroy()
        self.stiffness.destroy()

    def as_dict(self) -> dict[str, Any]:
        rows_m, cols_m = self.mass.getSize()
        rows_k, cols_k = self.stiffness.getSize()

        local_rows_m, local_cols_m = self.mass.getLocalSize()
        local_rows_k, local_cols_k = self.stiffness.getLocalSize()

        return {
            "kind": "material_perturbation_matrices",
            "metadata": dict(self.metadata),
            "mass": {
                "rows": int(rows_m),
                "cols": int(cols_m),
                "local_rows": int(local_rows_m),
                "local_cols": int(local_cols_m),
            },
            "stiffness": {
                "rows": int(rows_k),
                "cols": int(cols_k),
                "local_rows": int(local_rows_k),
                "local_cols": int(local_cols_k),
            },
        }


def _fill_constant_dg0(function: fem.Function, value: float) -> None:
    function.x.array[:] = float(value)
    function.x.scatter_forward()


def homogeneous_material_direction_fields(
    V,
    direction: HomogeneousMaterialDirection,
) -> dict[str, fem.Function]:
    """
    Build DG0 material-direction fields on the same mesh as V.

    The fields are allowed to be zero or signed. They represent a direction
    δrho, δC11, δC22, δC12, δC66, not a physical positive material.
    """

    msh = V.mesh
    Q0 = create_scalar_dg0_space(msh)

    fields = {
        "rho": create_named_dg0_function(Q0, "delta_rho"),
        "C11": create_named_dg0_function(Q0, "delta_C11"),
        "C22": create_named_dg0_function(Q0, "delta_C22"),
        "C12": create_named_dg0_function(Q0, "delta_C12"),
        "C66": create_named_dg0_function(Q0, "delta_C66"),
    }

    values = direction.as_dict()

    for key, function in fields.items():
        _fill_constant_dg0(function, values[key])

    return fields


def material_direction_mass_form(V, delta_rho: fem.Function):
    """
    Mass bilinear form for a material direction δrho.

    This deliberately bypasses material admissibility validation because
    δrho is a signed direction, not a physical density field.
    """

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    return fem.form(delta_rho * ufl.inner(u, v) * ufl.dx)


def material_direction_stiffness_form(
    V,
    delta_C11: fem.Function,
    delta_C22: fem.Function,
    delta_C12: fem.Function,
    delta_C66: fem.Function,
):
    """
    Orthotropic stiffness bilinear form for material directions.

    The direction fields may be zero or signed.
    """

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    eps_u_xx = u[0].dx(0)
    eps_u_yy = u[1].dx(1)
    gamma_u_xy = u[0].dx(1) + u[1].dx(0)

    eps_v_xx = v[0].dx(0)
    eps_v_yy = v[1].dx(1)
    gamma_v_xy = v[0].dx(1) + v[1].dx(0)

    energy_density = (
        delta_C11 * eps_u_xx * eps_v_xx
        + delta_C22 * eps_u_yy * eps_v_yy
        + delta_C12 * (eps_u_xx * eps_v_yy + eps_u_yy * eps_v_xx)
        + delta_C66 * gamma_u_xy * gamma_v_xy
    )

    return fem.form(energy_density * ufl.dx)


def assemble_material_perturbation_matrices(
    V,
    direction_fields: dict[str, fem.Function],
    *,
    metadata: dict[str, Any] | None = None,
) -> MaterialPerturbationMatrices:
    """
    Assemble δM and δK from DG0 material-direction fields.

    No positivity validation is applied here because these fields are
    directional perturbations, not admissible material coefficients.
    """

    required = {"rho", "C11", "C22", "C12", "C66"}
    missing = sorted(required.difference(direction_fields))

    if missing:
        raise ValueError(f"Missing material direction fields: {missing}")

    delta_mass = assemble_matrix_from_form(
        material_direction_mass_form(
            V,
            direction_fields["rho"],
        )
    )

    delta_stiffness = assemble_matrix_from_form(
        material_direction_stiffness_form(
            V,
            direction_fields["C11"],
            direction_fields["C22"],
            direction_fields["C12"],
            direction_fields["C66"],
        )
    )

    return MaterialPerturbationMatrices(
        mass=delta_mass,
        stiffness=delta_stiffness,
        metadata=dict(metadata or {}),
    )


def assemble_homogeneous_material_direction_matrices(
    V,
    direction: HomogeneousMaterialDirection,
) -> MaterialPerturbationMatrices:
    fields = homogeneous_material_direction_fields(V, direction)

    return assemble_material_perturbation_matrices(
        V,
        fields,
        metadata={
            "kind": "homogeneous_material_direction",
            "direction": direction.as_dict(),
        },
    )
