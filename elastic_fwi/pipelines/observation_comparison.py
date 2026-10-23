from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from mpi4py import MPI

from elastic_fwi.inversion.observation_functionals import (
    ObservationFunctionalValues,
    ObservationFunctionalWeights,
    compute_observation_functionals,
)
from elastic_fwi.inversion.observation_residual_series import (
    ObservationResidualData,
    compute_observation_residual_data,
)
from elastic_fwi.pipelines.direct_observation import (
    DirectObservationPipelineConfig,
    DirectObservationPipelineResult,
    build_direct_observation_spatial_context,
    run_direct_observation_pipeline,
    validate_direct_observation_common_mesh_compatibility,
)


@dataclass(frozen=True)
class DirectObservationComparisonConfig:
    """
    Configuration for comparing two direct observation pipelines.

    simulated:
        Forward result interpreted as the current numerical prediction.

    observed:
        Forward result interpreted as synthetic reference data.

    weights:
        Reflection/transmission weights used in J_weighted.

    mesh_policy:
        ``shared_observed_refined`` (default) builds the physical-experiment
        mesh once from the observed/defective configuration and reuses the
        same MPI partition and finite-element space for the healthy direct
        solve. ``independent`` is retained only for explicit legacy studies.
    """

    simulated: DirectObservationPipelineConfig = field(
        default_factory=lambda: DirectObservationPipelineConfig(
            nx=4,
            ny=4,
            material_kind="healthy",
        )
    )
    observed: DirectObservationPipelineConfig = field(
        default_factory=lambda: DirectObservationPipelineConfig(
            nx=4,
            ny=4,
            material_kind="localized_disk_defect",
        )
    )
    weights: ObservationFunctionalWeights = field(
        default_factory=ObservationFunctionalWeights
    )
    mesh_policy: str = "shared_observed_refined"

    def __post_init__(self) -> None:
        policy = str(self.mesh_policy)
        allowed = {"shared_observed_refined", "independent"}
        if policy not in allowed:
            raise ValueError(
                "mesh_policy must be one of "
                f"{sorted(allowed)}, got {policy!r}."
            )
        object.__setattr__(self, "mesh_policy", policy)

    def as_dict(self) -> dict[str, Any]:
        return {
            "simulated": self.simulated.as_dict(),
            "observed": self.observed.as_dict(),
            "weights": self.weights.as_dict(),
            "mesh_policy": self.mesh_policy,
        }


@dataclass(frozen=True)
class DirectObservationComparisonResult:
    config: DirectObservationComparisonConfig
    simulated: DirectObservationPipelineResult
    observed: DirectObservationPipelineResult
    residuals: ObservationResidualData
    simulated_vs_observed: ObservationFunctionalValues
    observed_vs_simulated: ObservationFunctionalValues
    checks: dict[str, bool]
    status: str
    mpi_size: int
    common_mesh_metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        simulated_values = self.simulated.data.stacked_values()
        observed_values = self.observed.data.stacked_values()
        difference = observed_values - simulated_values

        simulated_defect_cells = int(
            self.simulated.material_metadata.get("global_defect_cells", 0)
        )
        observed_defect_cells = int(
            self.observed.material_metadata.get("global_defect_cells", 0)
        )

        return {
            "kind": "direct_observation_comparison_result",
            "status": self.status,
            "mpi_size": int(self.mpi_size),
            "common_mesh": dict(self.common_mesh_metadata),
            "checks": dict(self.checks),
            "config": self.config.as_dict(),
            "values": {
                "max_abs_observation_difference": float(np.max(np.abs(difference))),
                "max_abs_residual": self.residuals.max_abs_residual,
                "simulated_max_reflection": self.simulated.data.reflection.max_value,
                "simulated_max_transmission": self.simulated.data.transmission.max_value,
                "observed_max_reflection": self.observed.data.reflection.max_value,
                "observed_max_transmission": self.observed.data.transmission.max_value,
                "simulated_defect_cells": simulated_defect_cells,
                "observed_defect_cells": observed_defect_cells,
            },
            "functionals": {
                "simulated_vs_observed": self.simulated_vs_observed.as_dict(),
                "observed_vs_simulated": self.observed_vs_simulated.as_dict(),
            },
            "residuals": self.residuals.as_dict(),
            "simulated": self.simulated.as_dict(),
            "observed": self.observed.as_dict(),
        }


def _comparison_checks(
    *,
    simulated: DirectObservationPipelineResult,
    observed: DirectObservationPipelineResult,
    residuals: ObservationResidualData,
    simulated_vs_observed: ObservationFunctionalValues,
    observed_vs_simulated: ObservationFunctionalValues,
) -> dict[str, bool]:
    simulated_values = simulated.data.stacked_values()
    observed_values = observed.data.stacked_values()
    residual_values = residuals.stacked_values()

    expected_residual_values = simulated_values - observed_values

    return {
        "simulated_values_finite": bool(np.all(np.isfinite(simulated_values))),
        "observed_values_finite": bool(np.all(np.isfinite(observed_values))),
        "residual_values_finite": bool(np.all(np.isfinite(residual_values))),
        "simulated_values_nonnegative": bool(np.min(simulated_values) >= 0.0),
        "observed_values_nonnegative": bool(np.min(observed_values) >= 0.0),
        "num_steps_match": bool(simulated.data.num_steps == observed.data.num_steps),
        "residual_num_steps_match": bool(
            residuals.num_steps == simulated.data.num_steps
        ),
        "times_match": bool(
            np.allclose(simulated.data.times, observed.data.times, rtol=0.0, atol=0.0)
        ),
        "residual_times_match": bool(
            np.allclose(residuals.times, simulated.data.times, rtol=0.0, atol=0.0)
        ),
        "reflection_tags": bool(
            simulated.data.reflection.tag == 1 and observed.data.reflection.tag == 1
        ),
        "transmission_tags": bool(
            simulated.data.transmission.tag == 2
            and observed.data.transmission.tag == 2
        ),
        "residual_tags": bool(
            residuals.reflection.tag == 1 and residuals.transmission.tag == 2
        ),
        "residual_definition": bool(
            np.allclose(
                residual_values,
                expected_residual_values,
                rtol=0.0,
                atol=0.0,
            )
        ),
        "residual_functionals_match": bool(
            np.isclose(
                residuals.functionals.reflection,
                simulated_vs_observed.reflection,
                rtol=0.0,
                atol=0.0,
            )
            and np.isclose(
                residuals.functionals.transmission,
                simulated_vs_observed.transmission,
                rtol=0.0,
                atol=0.0,
            )
            and np.isclose(
                residuals.functionals.full,
                simulated_vs_observed.full,
                rtol=0.0,
                atol=0.0,
            )
            and np.isclose(
                residuals.functionals.weighted,
                simulated_vs_observed.weighted,
                rtol=0.0,
                atol=0.0,
            )
        ),
        "misfits_finite": bool(
            np.isfinite(simulated_vs_observed.reflection)
            and np.isfinite(simulated_vs_observed.transmission)
            and np.isfinite(simulated_vs_observed.full)
            and np.isfinite(simulated_vs_observed.weighted)
        ),
        "misfits_nonnegative": bool(
            simulated_vs_observed.reflection >= 0.0
            and simulated_vs_observed.transmission >= 0.0
            and simulated_vs_observed.full >= 0.0
            and simulated_vs_observed.weighted >= 0.0
        ),
        "full_is_sum": bool(
            np.isclose(
                simulated_vs_observed.full,
                simulated_vs_observed.reflection
                + simulated_vs_observed.transmission,
                rtol=1.0e-14,
                atol=1.0e-14,
            )
        ),
        "weighted_formula": bool(
            np.isclose(
                simulated_vs_observed.weighted,
                simulated_vs_observed.reflection_weight
                * simulated_vs_observed.reflection
                + simulated_vs_observed.transmission_weight
                * simulated_vs_observed.transmission,
                rtol=1.0e-14,
                atol=1.0e-14,
            )
        ),
        "symmetry_of_squared_misfit": bool(
            np.isclose(
                simulated_vs_observed.full,
                observed_vs_simulated.full,
                rtol=1.0e-14,
                atol=1.0e-14,
            )
            and np.isclose(
                simulated_vs_observed.weighted,
                observed_vs_simulated.weighted,
                rtol=1.0e-14,
                atol=1.0e-14,
            )
        ),
    }


def run_direct_observation_comparison(
    config: DirectObservationComparisonConfig | None = None,
    *,
    comm: MPI.Comm = MPI.COMM_WORLD,
) -> DirectObservationComparisonResult:
    """
    Run two direct observation pipelines and compare their Gamma1/Gamma2 data.

    By default both wave problems use one shared physical-experiment mesh
    built from the observed/defective configuration. The independent-mesh
    branch is available only through ``mesh_policy='independent'``.

    This is a direct-only block:
    no adjoint, no gradient, no reconstruction algorithm.
    """

    if config is None:
        config = DirectObservationComparisonConfig()

    if config.mesh_policy == "shared_observed_refined":
        return run_common_mesh_direct_observation_comparison(
            config,
            comm=comm,
        )

    simulated = run_direct_observation_pipeline(config.simulated, comm=comm)
    observed = run_direct_observation_pipeline(config.observed, comm=comm)

    simulated_vs_observed = compute_observation_functionals(
        simulated.data,
        observed.data,
        weights=config.weights,
    )

    observed_vs_simulated = compute_observation_functionals(
        observed.data,
        simulated.data,
        weights=config.weights,
    )

    residuals = compute_observation_residual_data(
        simulated.data,
        observed.data,
        weights=config.weights,
    )

    checks = _comparison_checks(
        simulated=simulated,
        observed=observed,
        residuals=residuals,
        simulated_vs_observed=simulated_vs_observed,
        observed_vs_simulated=observed_vs_simulated,
    )

    return DirectObservationComparisonResult(
        config=config,
        simulated=simulated,
        observed=observed,
        residuals=residuals,
        simulated_vs_observed=simulated_vs_observed,
        observed_vs_simulated=observed_vs_simulated,
        checks=checks,
        status="ok" if all(checks.values()) else "failed",
        mpi_size=comm.size,
    )

def run_common_mesh_direct_observation_comparison(
    config: DirectObservationComparisonConfig | None = None,
    *,
    comm: MPI.Comm = MPI.COMM_WORLD,
) -> DirectObservationComparisonResult:
    """
    Run two direct solves on one mesh, one MPI partition and one FE space.

    The shared spatial context is built from the observed configuration.
    Material fields and operator matrices are assembled independently.
    """

    if config is None:
        config = DirectObservationComparisonConfig()

    validate_direct_observation_common_mesh_compatibility(
        config.observed,
        config.simulated,
    )

    spatial_context = build_direct_observation_spatial_context(
        config.observed,
        comm=comm,
    )

    simulated = run_direct_observation_pipeline(
        config.simulated,
        comm=comm,
        spatial_context=spatial_context,
    )

    observed = run_direct_observation_pipeline(
        config.observed,
        comm=comm,
        spatial_context=spatial_context,
    )

    simulated_vs_observed = compute_observation_functionals(
        simulated.data,
        observed.data,
        weights=config.weights,
    )

    observed_vs_simulated = compute_observation_functionals(
        observed.data,
        simulated.data,
        weights=config.weights,
    )

    residuals = compute_observation_residual_data(
        simulated.data,
        observed.data,
        weights=config.weights,
    )

    checks = _comparison_checks(
        simulated=simulated,
        observed=observed,
        residuals=residuals,
        simulated_vs_observed=simulated_vs_observed,
        observed_vs_simulated=observed_vs_simulated,
    )

    shared_function_space = bool(
        simulated.function_space is observed.function_space
        and simulated.function_space is spatial_context.function_space
    )

    simulated_mesh = getattr(simulated.function_space, "mesh", None)
    observed_mesh = getattr(observed.function_space, "mesh", None)

    shared_mesh_object = bool(
        simulated_mesh is observed_mesh
        and simulated_mesh is spatial_context.mesh
    )

    checks = {
        **checks,
        "common_mesh_enabled": True,
        "mesh_created_once": True,
        "shared_function_space": shared_function_space,
        "shared_mesh_object": shared_mesh_object,
        "same_mpi_partition": True,
    }

    common_mesh_metadata = {
        "enabled": True,
        "common_mesh": True,
        "mesh_created_once": True,
        "mesh_source": "observed_configuration",
        "refinement_semantics": "physical_experiment_geometry",
        "all_direct_wave_problems_share_spatial_context": True,
        "shared_mesh_object": shared_mesh_object,
        "shared_function_space": shared_function_space,
        "same_mpi_partition": True,
        "same_export_times": bool(checks["times_match"]),
        "mpi_size": int(comm.size),
        "refinement_disks": list(
            spatial_context.mesh_metadata["refinement_disks"]
        ),
    }

    return DirectObservationComparisonResult(
        config=config,
        simulated=simulated,
        observed=observed,
        residuals=residuals,
        simulated_vs_observed=simulated_vs_observed,
        observed_vs_simulated=observed_vs_simulated,
        checks=checks,
        status="ok" if all(checks.values()) else "failed",
        mpi_size=comm.size,
        common_mesh_metadata=common_mesh_metadata,
    )
