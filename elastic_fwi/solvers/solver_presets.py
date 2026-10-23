from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from elastic_fwi.solvers.newmark import LinearSolveConfig


@dataclass(frozen=True)
class SolverPreset:
    name: str
    config: LinearSolveConfig
    scientific_role: str
    expected_use: str
    optional_dependency: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "ksp_type": self.config.ksp_type,
            "pc_type": self.config.pc_type,
            "pc_factor_mat_solver_type": self.config.pc_factor_mat_solver_type,
            "rtol": self.config.rtol,
            "atol": self.config.atol,
            "max_it": self.config.max_it,
            "scientific_role": self.scientific_role,
            "expected_use": self.expected_use,
            "optional_dependency": self.optional_dependency,
        }


SOLVER_PRESETS: dict[str, SolverPreset] = {
    "default_cg_jacobi": SolverPreset(
        name="default_cg_jacobi",
        config=LinearSolveConfig(ksp_type="cg", pc_type="jacobi"),
        scientific_role="repository_default_reference",
        expected_use="transparent baseline for SPD Newmark effective systems",
    ),
    "cg_gamg": SolverPreset(
        name="cg_gamg",
        config=LinearSolveConfig(ksp_type="cg", pc_type="gamg"),
        scientific_role="scalable_spd_candidate",
        expected_use="larger SPD systems when PETSc GAMG is effective",
    ),
    "cg_hypre": SolverPreset(
        name="cg_hypre",
        config=LinearSolveConfig(ksp_type="cg", pc_type="hypre"),
        scientific_role="scalable_spd_candidate",
        expected_use="larger SPD systems when PETSc was built with HYPRE",
        optional_dependency="hypre",
    ),
    "gmres_ilu": SolverPreset(
        name="gmres_ilu",
        config=LinearSolveConfig(ksp_type="gmres", pc_type="ilu"),
        scientific_role="historical_cpp_like_candidate",
        expected_use="sequential or small MPI cases; close to GMRES+ILU spirit",
    ),
    "preonly_lu": SolverPreset(
        name="preonly_lu",
        config=LinearSolveConfig(ksp_type="preonly", pc_type="lu"),
        scientific_role="direct_reference",
        expected_use="robust small/medium reference solve",
    ),
    "preonly_lu_mumps": SolverPreset(
        name="preonly_lu_mumps",
        config=LinearSolveConfig(
            ksp_type="preonly",
            pc_type="lu",
            pc_factor_mat_solver_type="mumps",
        ),
        scientific_role="parallel_direct_reference",
        expected_use="robust MPI reference solve if PETSc was built with MUMPS",
        optional_dependency="mumps",
    ),
}


def get_solver_preset(name: str) -> SolverPreset:
    try:
        return SOLVER_PRESETS[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown solver preset {name!r}. "
            f"Available presets: {sorted(SOLVER_PRESETS)}"
        ) from exc


def list_solver_presets() -> list[str]:
    return sorted(SOLVER_PRESETS)
