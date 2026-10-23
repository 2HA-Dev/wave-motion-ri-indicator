from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np
from mpi4py import MPI

from elastic_fwi.inversion.material_directions import HomogeneousMaterialDirection
from elastic_fwi.gradient_observation_modes import (
    TRUE_GRADIENT_OBSERVATION_MODES,
    resolve_gradient_observation_mode,
    select_observation_functional_value,
)
from elastic_fwi.postprocessing.gradient_validation import taylor_order_estimates
from elastic_fwi.inversion.observation_functionals import ObservationFunctionalWeights
from elastic_fwi.materials.orthotropic import LEGACY_HEALTHY_MATERIAL
from elastic_fwi.pipelines.direct_observation import DirectObservationPipelineConfig
from elastic_fwi.pipelines.observation_comparison import (
    DirectObservationComparisonConfig,
    run_direct_observation_comparison,
)
from elastic_fwi.diagnostics.check_material_gradient_directional import (
    run_material_gradient_directional_check,
)


def _perturbed_constants(
    direction: HomogeneousMaterialDirection,
    eps: float,
) -> dict[str, float]:
    return {
        "rho": LEGACY_HEALTHY_MATERIAL.rho + eps * direction.rho,
        "C11": LEGACY_HEALTHY_MATERIAL.C11 + eps * direction.C11,
        "C22": LEGACY_HEALTHY_MATERIAL.C22 + eps * direction.C22,
        "C12": LEGACY_HEALTHY_MATERIAL.C12 + eps * direction.C12,
        "C66": LEGACY_HEALTHY_MATERIAL.C66 + eps * direction.C66,
    }


def _direct_config(
    *,
    nx: int,
    ny: int,
    degree: int,
    theta: float,
    dt: float,
    num_steps: int,
    material_kind: str,
    contrast_factor: float,
    defect_center_x: float,
    defect_center_y: float,
    defect_radius: float,
    constants: dict[str, float] | None = None,
    ksp_type: str,
    pc_type: str,
    ksp_rtol: float,
    ksp_atol: float,
    ksp_max_it: int,
    reuse_effective_solver: bool = True,
) -> DirectObservationPipelineConfig:
    if constants is None:
        constants = {
            "rho": LEGACY_HEALTHY_MATERIAL.rho,
            "C11": LEGACY_HEALTHY_MATERIAL.C11,
            "C22": LEGACY_HEALTHY_MATERIAL.C22,
            "C12": LEGACY_HEALTHY_MATERIAL.C12,
            "C66": LEGACY_HEALTHY_MATERIAL.C66,
        }

    return DirectObservationPipelineConfig(
        nx=nx,
        ny=ny,
        degree=degree,
        theta=theta,
        dt=dt,
        num_steps=num_steps,
        material_kind=material_kind,
        contrast_factor=contrast_factor,
        custom_rho=constants["rho"],
        custom_C11=constants["C11"],
        custom_C22=constants["C22"],
        custom_C12=constants["C12"],
        custom_C66=constants["C66"],
        defect_center_x=defect_center_x,
        defect_center_y=defect_center_y,
        defect_radius=defect_radius,
        ksp_type=ksp_type,
        pc_type=pc_type,
        ksp_rtol=ksp_rtol,
        ksp_atol=ksp_atol,
        ksp_max_it=ksp_max_it,
        reuse_effective_solver=reuse_effective_solver,
    )


def run_material_gradient_finite_difference_check(
    *,
    nx: int = 4,
    ny: int = 4,
    degree: int = 1,
    theta: float = 0.25,
    dt: float = 0.1,
    num_steps: int = 5,
    observation_mode: str = "classical_full",
    observed_material_kind: str = "localized_disk_defect",
    contrast_factor: float = 0.8,
    defect_center_x: float = 0.5,
    defect_center_y: float = 0.5,
    defect_radius: float = 0.35,
    direction_rho: float = 0.0,
    direction_C11: float = 1.0,
    direction_C22: float = 0.0,
    direction_C12: float = 0.0,
    direction_C66: float = 0.0,
    eps_values: tuple[float, ...] = (1.0e-2, 5.0e-3, 2.5e-3),
    absolute_tolerance: float = 5.0e-9,
    relative_tolerance: float = 5.0e-1,
    ksp_type: str = "cg",
    pc_type: str = "jacobi",
    ksp_rtol: float = 1.0e-10,
    ksp_atol: float = 1.0e-12,
    ksp_max_it: int = 1000,
    reuse_direct_effective_solver: bool = True,
    reuse_adjoint_effective_solver: bool = True,
    comm: MPI.Comm = MPI.COMM_WORLD,
) -> dict[str, Any]:
    mode = resolve_gradient_observation_mode(observation_mode)
    weights = ObservationFunctionalWeights(
        reflection=mode.reflection_weight,
        transmission=mode.transmission_weight,
    )
    direction = HomogeneousMaterialDirection(
        rho=direction_rho,
        C11=direction_C11,
        C22=direction_C22,
        C12=direction_C12,
        C66=direction_C66,
    )

    gradient_report = run_material_gradient_directional_check(
        nx=nx,
        ny=ny,
        degree=degree,
        theta=theta,
        dt=dt,
        num_steps=num_steps,
        observation_mode=mode.name,
        simulated_material_kind="healthy",
        observed_material_kind=observed_material_kind,
        contrast_factor=contrast_factor,
        defect_center_x=defect_center_x,
        defect_center_y=defect_center_y,
        defect_radius=defect_radius,
        direction_rho=direction_rho,
        direction_C11=direction_C11,
        direction_C22=direction_C22,
        direction_C12=direction_C12,
        direction_C66=direction_C66,
        ksp_type=ksp_type,
        pc_type=pc_type,
        ksp_rtol=ksp_rtol,
        ksp_atol=ksp_atol,
        ksp_max_it=ksp_max_it,
        reuse_direct_effective_solver=reuse_direct_effective_solver,
        reuse_adjoint_effective_solver=reuse_adjoint_effective_solver,
        comm=comm,
    )

    adjoint_gradient = float(gradient_report["values"]["gradient_value"])
    j0 = select_observation_functional_value(
        gradient_report["pipeline"]["comparison"]["functionals"][
            "simulated_vs_observed"
        ],
        mode,
    )

    observed_config = _direct_config(
        nx=nx,
        ny=ny,
        degree=degree,
        theta=theta,
        dt=dt,
        num_steps=num_steps,
        material_kind=observed_material_kind,
        contrast_factor=contrast_factor,
        defect_center_x=defect_center_x,
        defect_center_y=defect_center_y,
        defect_radius=defect_radius,
        ksp_type=ksp_type,
        pc_type=pc_type,
        ksp_rtol=ksp_rtol,
        ksp_atol=ksp_atol,
        ksp_max_it=ksp_max_it,
        reuse_effective_solver=reuse_direct_effective_solver,
    )

    records: list[dict[str, Any]] = []

    for eps in eps_values:
        eps = float(eps)

        if eps <= 0.0 or not np.isfinite(eps):
            raise ValueError("All eps values must be finite and strictly positive.")

        constants_eps = _perturbed_constants(direction, eps)

        simulated_eps_config = _direct_config(
            nx=nx,
            ny=ny,
            degree=degree,
            theta=theta,
            dt=dt,
            num_steps=num_steps,
            material_kind="custom_homogeneous",
            contrast_factor=contrast_factor,
            defect_center_x=defect_center_x,
            defect_center_y=defect_center_y,
            defect_radius=defect_radius,
            constants=constants_eps,
            ksp_type=ksp_type,
            pc_type=pc_type,
            ksp_rtol=ksp_rtol,
            ksp_atol=ksp_atol,
            ksp_max_it=ksp_max_it,
            reuse_effective_solver=reuse_direct_effective_solver,
        )

        comparison = run_direct_observation_comparison(
            DirectObservationComparisonConfig(
                simulated=simulated_eps_config,
                observed=observed_config,
                weights=weights,
            ),
            comm=comm,
        )

        try:
            j_eps = select_observation_functional_value(
                comparison.simulated_vs_observed,
                mode,
            )
            finite_difference = float((j_eps - j0) / eps)
            absolute_error = float(abs(finite_difference - adjoint_gradient))
            denominator = max(1.0, abs(finite_difference), abs(adjoint_gradient))
            relative_error = float(absolute_error / denominator)
            first_order_remainder = float(abs(j_eps - j0))
            second_order_remainder = float(
                abs(j_eps - j0 - eps * adjoint_gradient)
            )
            sign_scale = max(abs(finite_difference), abs(adjoint_gradient))
            sign_consistent = bool(
                sign_scale <= absolute_tolerance
                or finite_difference == 0.0
                or adjoint_gradient == 0.0
                or np.signbit(finite_difference) == np.signbit(adjoint_gradient)
            )

            records.append(
                {
                    "eps": eps,
                    "J0": j0,
                    "J_eps": j_eps,
                    "finite_difference": finite_difference,
                    "adjoint_gradient": adjoint_gradient,
                    "absolute_error": absolute_error,
                    "relative_error": relative_error,
                    "first_order_remainder": first_order_remainder,
                    "second_order_remainder": second_order_remainder,
                    "directional_sign_consistent": sign_consistent,
                    "constants": constants_eps,
                    "comparison_status": comparison.status,
                }
            )

        finally:
            comparison.simulated.destroy()
            comparison.observed.destroy()

    finite_records = bool(
        all(
            np.isfinite(record["finite_difference"])
            and np.isfinite(record["absolute_error"])
            and np.isfinite(record["relative_error"])
            and np.isfinite(record["first_order_remainder"])
            and np.isfinite(record["second_order_remainder"])
            for record in records
        )
    )

    comparison_status_ok = bool(
        all(record["comparison_status"] == "ok" for record in records)
    )

    best_absolute_error = float(min(record["absolute_error"] for record in records))
    best_relative_error = float(min(record["relative_error"] for record in records))

    close_enough = bool(
        best_absolute_error <= absolute_tolerance
        or best_relative_error <= relative_tolerance
    )
    order_estimates = taylor_order_estimates(records)
    finite_orders = [value for value in order_estimates if np.isfinite(value)]
    best_taylor_order = max(finite_orders) if finite_orders else float("nan")
    directional_sign_consistent = bool(
        all(record["directional_sign_consistent"] for record in records)
    )
    second_order_improves = bool(
        min(record["second_order_remainder"] for record in records)
        <= min(record["first_order_remainder"] for record in records)
    )

    checks = {
        "directional_gradient_status_ok": gradient_report["status"] == "ok",
        "adjoint_gradient_finite": bool(np.isfinite(adjoint_gradient)),
        "baseline_cost_finite": bool(np.isfinite(j0)),
        "baseline_cost_nonnegative": bool(j0 >= 0.0),
        "finite_difference_records_finite": finite_records,
        "comparison_status_ok": comparison_status_ok,
        "best_absolute_error_within_tolerance": bool(
            best_absolute_error <= absolute_tolerance
        ),
        "best_relative_error_within_tolerance": bool(
            best_relative_error <= relative_tolerance
        ),
        "finite_difference_close_enough": close_enough,
        "directional_sign_consistent": directional_sign_consistent,
        "second_order_remainder_improves": second_order_improves,
        "taylor_order_estimates_finite_when_defined": bool(
            all(np.isfinite(value) for value in finite_orders)
        ),
    }

    status_ok = bool(
        checks["directional_gradient_status_ok"]
        and checks["adjoint_gradient_finite"]
        and checks["baseline_cost_finite"]
        and checks["baseline_cost_nonnegative"]
        and checks["finite_difference_records_finite"]
        and checks["comparison_status_ok"]
        and checks["finite_difference_close_enough"]
        and checks["directional_sign_consistent"]
        and checks["second_order_remainder_improves"]
    )

    return {
        "kind": "material_gradient_finite_difference_check",
        "status": "ok" if status_ok else "failed",
        "mpi_size": comm.size,
        "checks": checks,
        "config": {
            "nx": nx,
            "ny": ny,
            "degree": degree,
            "theta": theta,
            "dt": dt,
            "num_steps": num_steps,
            "observation_mode": mode.name,
            "observation_mode_definition": mode.as_dict(),
            "observed_material_kind": observed_material_kind,
            "contrast_factor": contrast_factor,
            "defect_center_x": defect_center_x,
            "defect_center_y": defect_center_y,
            "defect_radius": defect_radius,
            "direction": direction.as_dict(),
            "eps_values": list(eps_values),
            "absolute_tolerance": absolute_tolerance,
            "relative_tolerance": relative_tolerance,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": ksp_rtol,
            "ksp_atol": ksp_atol,
            "ksp_max_it": ksp_max_it,
            "reuse_direct_effective_solver": reuse_direct_effective_solver,
            "reuse_adjoint_effective_solver": reuse_adjoint_effective_solver,
        },
        "values": {
            "J0": j0,
            "adjoint_gradient": adjoint_gradient,
            "best_absolute_error": best_absolute_error,
            "best_relative_error": best_relative_error,
            "best_taylor_order": best_taylor_order,
        },
        "taylor_order_estimates": order_estimates,
        "records": records,
        "directional_gradient": gradient_report,
    }


def _parse_eps_values(raw: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in raw.split(",") if item.strip())

    if not values:
        raise ValueError("At least one epsilon value is required.")

    return values


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check material adjoint gradient by finite differences."
    )

    parser.add_argument("--nx", type=int, default=4)
    parser.add_argument("--ny", type=int, default=4)
    parser.add_argument("--degree", type=int, default=1)
    parser.add_argument("--theta", type=float, default=0.25)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--num-steps", type=int, default=5)
    parser.add_argument(
        "--observation-mode",
        choices=TRUE_GRADIENT_OBSERVATION_MODES,
        default="classical_full",
    )

    parser.add_argument("--observed-material-kind", type=str, default="localized_disk_defect")
    parser.add_argument("--contrast-factor", type=float, default=0.8)
    parser.add_argument("--defect-center-x", type=float, default=0.5)
    parser.add_argument("--defect-center-y", type=float, default=0.5)
    parser.add_argument("--defect-radius", type=float, default=0.35)

    parser.add_argument("--direction-rho", type=float, default=0.0)
    parser.add_argument("--direction-C11", type=float, default=1.0)
    parser.add_argument("--direction-C22", type=float, default=0.0)
    parser.add_argument("--direction-C12", type=float, default=0.0)
    parser.add_argument("--direction-C66", type=float, default=0.0)

    parser.add_argument("--eps-values", type=str, default="1e-2,5e-3,2.5e-3")
    parser.add_argument("--absolute-tolerance", type=float, default=5.0e-9)
    parser.add_argument("--relative-tolerance", type=float, default=5.0e-1)

    parser.add_argument("--ksp-type", type=str, default="cg")
    parser.add_argument("--pc-type", type=str, default="jacobi")
    parser.add_argument("--ksp-rtol", type=float, default=1.0e-10)
    parser.add_argument("--ksp-atol", type=float, default=1.0e-12)
    parser.add_argument("--ksp-max-it", type=int, default=1000)

    reuse_group = parser.add_mutually_exclusive_group()
    reuse_group.add_argument(
        "--reuse-effective-solver",
        dest="reuse_effective_solver",
        action="store_true",
        default=True,
        help="Reuse direct and adjoint effective Newmark solvers. Default: enabled.",
    )
    reuse_group.add_argument(
        "--no-reuse-effective-solver",
        dest="reuse_effective_solver",
        action="store_false",
        help="Use legacy one-shot effective solvers in direct and adjoint paths.",
    )

    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--json", action="store_true")

    args = parser.parse_args()
    comm = MPI.COMM_WORLD

    report = run_material_gradient_finite_difference_check(
        nx=args.nx,
        ny=args.ny,
        degree=args.degree,
        theta=args.theta,
        dt=args.dt,
        num_steps=args.num_steps,
        observation_mode=args.observation_mode,
        observed_material_kind=args.observed_material_kind,
        contrast_factor=args.contrast_factor,
        defect_center_x=args.defect_center_x,
        defect_center_y=args.defect_center_y,
        defect_radius=args.defect_radius,
        direction_rho=args.direction_rho,
        direction_C11=args.direction_C11,
        direction_C22=args.direction_C22,
        direction_C12=args.direction_C12,
        direction_C66=args.direction_C66,
        eps_values=_parse_eps_values(args.eps_values),
        absolute_tolerance=args.absolute_tolerance,
        relative_tolerance=args.relative_tolerance,
        ksp_type=args.ksp_type,
        pc_type=args.pc_type,
        ksp_rtol=args.ksp_rtol,
        ksp_atol=args.ksp_atol,
        ksp_max_it=args.ksp_max_it,
        reuse_direct_effective_solver=args.reuse_effective_solver,
        reuse_adjoint_effective_solver=args.reuse_effective_solver,
        comm=comm,
    )

    if comm.rank == 0 and not args.quiet:
        if args.json:
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            status = "OK" if report["status"] == "ok" else "FAILED"
            print(
                f"[MATERIAL GRADIENT FD CHECK] {status} "
                f"for mesh={args.nx}x{args.ny}, degree={args.degree}, "
                f"theta={args.theta}, dt={args.dt}, steps={args.num_steps}, "
                f"mode={args.observation_mode}, "
                f"gradient={report['values']['adjoint_gradient']:.6e}, "
                f"best_abs={report['values']['best_absolute_error']:.6e}, "
                f"best_rel={report['values']['best_relative_error']:.6e}, "
                f"MPI size={comm.size}"
            )

    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
