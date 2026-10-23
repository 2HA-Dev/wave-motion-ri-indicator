from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from elastic_fwi.inversion import (
    VectorArrayFullAdjointRHSResult,
    VectorArrayResidualPair,
    assemble_full_vector_array_adjoint_rhs,
    build_vector_array_residual_pair,
    compute_vector_array_residual_series,
)
from elastic_fwi.observations import (
    VectorArrayObservationSpec,
    VectorArrayObservationTimeSeries,
    evaluate_vector_array_observation_time_series,
)


@dataclass(frozen=True)
class VectorArrayObservationComparisonResult:
    """Vector-array comparison between two direct simulations.

    The two direct results are interpreted as:

    - simulated/reference: usually healthy material;
    - observed/defective: usually defect material.

    Residual convention:

        residual = simulated - observed.

    The residuals are stored in chronological time order. They are not reversed
    here because solve_newmark_discrete_adjoint_state already traverses RHS
    vectors from final time to initial time.
    """

    simulated_reflection: VectorArrayObservationTimeSeries
    observed_reflection: VectorArrayObservationTimeSeries
    simulated_transmission: VectorArrayObservationTimeSeries
    observed_transmission: VectorArrayObservationTimeSeries
    residuals: VectorArrayResidualPair
    adjoint_rhs: VectorArrayFullAdjointRHSResult

    def destroy(self) -> None:
        self.adjoint_rhs.destroy()

    @property
    def num_steps(self) -> int:
        if self.residuals.reflection is not None:
            return self.residuals.reflection.num_steps
        if self.residuals.transmission is not None:
            return self.residuals.transmission.num_steps
        return 0

    @property
    def max_abs_residual(self) -> float:
        return self.residuals.max_abs_residual

    @property
    def max_rhs_norm(self) -> float:
        return self.adjoint_rhs.max_rhs_norm

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": "vector_array_observation_comparison_result",
            "num_steps": int(self.num_steps),
            "max_abs_residual": float(self.max_abs_residual),
            "max_rhs_norm": float(self.max_rhs_norm),
            "simulated_reflection": self.simulated_reflection.as_dict(),
            "observed_reflection": self.observed_reflection.as_dict(),
            "simulated_transmission": self.simulated_transmission.as_dict(),
            "observed_transmission": self.observed_transmission.as_dict(),
            "residuals": self.residuals.as_dict(),
            "adjoint_rhs": self.adjoint_rhs.as_dict(),
            "residual_time_order": "chronological",
            "adjoint_solver_expected_order": "solver_runs_backward_in_time",
        }


def _get_required_attribute(obj: Any, name: str) -> Any:
    if not hasattr(obj, name):
        raise ValueError(f"Object must expose attribute {name!r}.")
    value = getattr(obj, name)
    if value is None:
        raise ValueError(f"Attribute {name!r} must not be None.")
    return value


def _get_function_space_and_snapshots(result: Any):
    V = _get_required_attribute(result, "function_space")
    snapshots = tuple(_get_required_attribute(result, "displacement_snapshots"))
    if len(snapshots) == 0:
        raise ValueError("Direct result must contain displacement_snapshots.")

    # Diagnostics/tests may already provide explicit (step, time, vector)
    # triplets. Preserve them verbatim. Production direct results instead store
    # raw vectors and expose the authoritative physical times in
    # time_loop_records.
    if all(isinstance(snapshot, tuple) and len(snapshot) == 3 for snapshot in snapshots):
        return V, snapshots

    records = tuple(_get_required_attribute(result, "time_loop_records"))
    if len(records) != len(snapshots):
        raise ValueError(
            "Direct result time_loop_records and displacement_snapshots "
            "must have the same length."
        )

    timed_snapshots = tuple(
        (
            int(record["step"]),
            float(record["time"]),
            snapshot,
        )
        for record, snapshot in zip(records, snapshots)
    )
    return V, timed_snapshots


def run_vector_array_observation_comparison_from_direct_results(
    *,
    simulated,
    observed,
    reflection_spec: VectorArrayObservationSpec,
    transmission_spec: VectorArrayObservationSpec,
    reflection_scale: float = 1.0,
    transmission_scale: float = 1.0,
) -> VectorArrayObservationComparisonResult:
    """Build vector-array residuals and adjoint RHS from two direct results.

    This is the minimal bridge between the direct pipeline and the
    vector-array adjoint source.
    """
    V_sim, simulated_snapshots = _get_function_space_and_snapshots(simulated)
    V_obs, observed_snapshots = _get_function_space_and_snapshots(observed)

    if len(simulated_snapshots) != len(observed_snapshots):
        raise ValueError("Simulated and observed snapshots must have the same length.")

    simulated_reflection = evaluate_vector_array_observation_time_series(
        V_sim,
        list(simulated_snapshots),
        reflection_spec,
    )
    observed_reflection = evaluate_vector_array_observation_time_series(
        V_obs,
        list(observed_snapshots),
        reflection_spec,
    )

    simulated_transmission = evaluate_vector_array_observation_time_series(
        V_sim,
        list(simulated_snapshots),
        transmission_spec,
    )
    observed_transmission = evaluate_vector_array_observation_time_series(
        V_obs,
        list(observed_snapshots),
        transmission_spec,
    )

    reflection_residual = compute_vector_array_residual_series(
        reference=simulated_reflection,
        observed=observed_reflection,
        name=f"{reflection_spec.array.name}_vector_array_reflection_residual",
    )

    transmission_residual = compute_vector_array_residual_series(
        reference=simulated_transmission,
        observed=observed_transmission,
        name=f"{transmission_spec.array.name}_vector_array_transmission_residual",
    )

    residual_pair = build_vector_array_residual_pair(
        reflection=reflection_residual,
        transmission=transmission_residual,
    )

    adjoint_rhs = assemble_full_vector_array_adjoint_rhs(
        V_sim,
        reflection_spec=reflection_spec,
        reflection_residual=reflection_residual,
        transmission_spec=transmission_spec,
        transmission_residual=transmission_residual,
        reflection_scale=reflection_scale,
        transmission_scale=transmission_scale,
    )

    return VectorArrayObservationComparisonResult(
        simulated_reflection=simulated_reflection,
        observed_reflection=observed_reflection,
        simulated_transmission=simulated_transmission,
        observed_transmission=observed_transmission,
        residuals=residual_pair,
        adjoint_rhs=adjoint_rhs,
    )
