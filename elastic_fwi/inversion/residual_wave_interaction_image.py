from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from petsc4py import PETSc

from elastic_fwi.inversion.direct_adjoint_interaction_image import (
    DirectAdjointInteractionImage,
    compute_direct_adjoint_interaction_image_dg0,
)
from elastic_fwi.inversion.vector_array_adjoint_sources import (
    VectorArrayAdjointRHSRecord,
    VectorArrayAdjointRHSResult,
    VectorArrayFullAdjointRHSResult,
)


ResidualWaveRole = Literal["reflection", "transmission"]


@dataclass(frozen=True)
class SingleSourceVectorArrayFullRHS:
    """Single-source RHS wrapper for residual-wave imaging.

    The existing vector-array adjoint RHS contains two source families:

        H_Gamma1^T r_Gamma1 + H_Gamma2^T r_Gamma2.

    The residual-interaction protocol requires two separate wave solves
    using the same residual sign convention as the classical adjoint:

        wave_Gamma1: source = + H_Gamma1^T r_Gamma1,
        wave_Gamma2: source = + H_Gamma2^T r_Gamma2.

    This object keeps one family active and replaces the other by zero
    vectors, so that the interface expected by the adjoint solver is unchanged.
    """

    keep: ResidualWaveRole
    full_rhs: VectorArrayFullAdjointRHSResult
    zero_rhs: VectorArrayAdjointRHSResult
    active_rhs: VectorArrayAdjointRHSResult | None = None
    active_scale: float = 1.0

    def destroy_zero_rhs(self) -> None:
        self.zero_rhs.destroy()
        if self.active_rhs is not None:
            self.active_rhs.destroy()

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": "single_source_vector_array_full_rhs",
            "keep": self.keep,
            "active_scale": float(self.active_scale),
            "num_steps": self.full_rhs.num_steps,
            "max_rhs_norm": self.full_rhs.max_rhs_norm,
            "zero_rhs": self.zero_rhs.as_dict(),
        }


def _zero_like_vector(vector: PETSc.Vec) -> PETSc.Vec:
    zero = vector.copy()
    zero.set(0.0)
    zero.assemble()
    return zero


def _zero_record(record: VectorArrayAdjointRHSRecord) -> VectorArrayAdjointRHSRecord:
    return VectorArrayAdjointRHSRecord(
        step=record.step,
        time=record.time,
        observation_name=record.observation_name,
        array_name=record.array_name,
        role=f"zero_{record.role}",
        residual_max_abs=0.0,
        residual_l2_norm=0.0,
        rhs_norm=0.0,
        rhs_size_global=record.rhs_size_global,
        rhs_size_local=record.rhs_size_local,
    )


def _scaled_like_vector(vector: PETSc.Vec, scale: float) -> PETSc.Vec:
    scaled = vector.copy()
    scaled.scale(float(scale))
    scaled.assemble()
    return scaled


def _scaled_record(
    record: VectorArrayAdjointRHSRecord,
    *,
    scale: float,
) -> VectorArrayAdjointRHSRecord:
    abs_scale = abs(float(scale))
    return VectorArrayAdjointRHSRecord(
        step=record.step,
        time=record.time,
        observation_name=record.observation_name,
        array_name=record.array_name,
        role=f"{record.role}_scaled_{float(scale):g}",
        residual_max_abs=abs_scale * record.residual_max_abs,
        residual_l2_norm=abs_scale * record.residual_l2_norm,
        rhs_norm=abs_scale * record.rhs_norm,
        rhs_size_global=record.rhs_size_global,
        rhs_size_local=record.rhs_size_local,
    )


def make_zero_vector_array_rhs_result(
    rhs: VectorArrayAdjointRHSResult,
) -> VectorArrayAdjointRHSResult:
    records = tuple(_zero_record(record) for record in rhs.records)
    vectors = tuple(_zero_like_vector(vector) for vector in rhs.rhs_vectors)

    return VectorArrayAdjointRHSResult(
        records=records,
        rhs_vectors=vectors,
    )


def make_scaled_vector_array_rhs_result(
    rhs: VectorArrayAdjointRHSResult,
    *,
    scale: float,
) -> VectorArrayAdjointRHSResult:
    records = tuple(_scaled_record(record, scale=scale) for record in rhs.records)
    vectors = tuple(_scaled_like_vector(vector, scale) for vector in rhs.rhs_vectors)

    return VectorArrayAdjointRHSResult(
        records=records,
        rhs_vectors=vectors,
    )


def make_single_source_vector_array_full_rhs(
    rhs: VectorArrayFullAdjointRHSResult,
    *,
    keep: ResidualWaveRole,
) -> SingleSourceVectorArrayFullRHS:
    """Keep only Gamma1/reflection or Gamma2/transmission RHS active."""

    if keep == "reflection":
        zero_transmission = make_zero_vector_array_rhs_result(rhs.transmission)
        full_rhs = VectorArrayFullAdjointRHSResult(
            reflection=rhs.reflection,
            transmission=zero_transmission,
        )
        return SingleSourceVectorArrayFullRHS(
            keep=keep,
            full_rhs=full_rhs,
            zero_rhs=zero_transmission,
        )

    if keep == "transmission":
        zero_reflection = make_zero_vector_array_rhs_result(rhs.reflection)

        # Sign convention: use the same positive residual sign as the
        # classical adjoint, r_i = H_i u_ref - u_obs,i.  The legacy C++ code
        # applies an additional minus sign, which can be applied externally if
        # required; exported maps keep the coherent FWI sign.
        full_rhs = VectorArrayFullAdjointRHSResult(
            reflection=zero_reflection,
            transmission=rhs.transmission,
        )
        return SingleSourceVectorArrayFullRHS(
            keep=keep,
            full_rhs=full_rhs,
            zero_rhs=zero_reflection,
            active_scale=1.0,
        )

    raise ValueError("keep must be 'reflection' or 'transmission'.")


def compute_residual_wave_interaction_image_dg0(
    *,
    V,
    reflection_residual_wave_snapshots,
    transmission_residual_wave_snapshots,
    dt: float,
    eps: float = 1.0e-12,
) -> DirectAdjointInteractionImage:
    """Compute residual-wave interaction images.

    This uses the same algebraic imaging formulas as the direct-adjoint
    interaction image, but replaces the pair

        direct wavefield, adjoint wavefield

    by

        Gamma1 residual wavefield, Gamma2 residual wavefield.
    """

    result = compute_direct_adjoint_interaction_image_dg0(
        V=V,
        direct_snapshots=reflection_residual_wave_snapshots,
        adjoint_snapshots=transmission_residual_wave_snapshots,
        dt=dt,
        eps=eps,
    )

    for name, field in result.fields.items():
        field.name = f"residual_wave_{name}"

    metadata = {
        **result.metadata,
        "kind": "residual_wave_interaction_image_dg0",
        "imaging_mode": "residual_wave_interaction",
        "first_wavefield": "reflection_residual_wavefield",
        "second_wavefield": "transmission_residual_wavefield",
        "source_family_1": "H_Gamma1^T r_Gamma1",
        "source_family_2": "+H_Gamma2^T r_Gamma2",
    }

    return DirectAdjointInteractionImage(
        fields=result.fields,
        metadata=metadata,
    )
