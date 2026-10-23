from __future__ import annotations

from elastic_fwi.inversion.residual_wave_interaction_image import (
    make_single_source_vector_array_full_rhs,
)


def test_residual_wave_interaction_module_exports_names():
    from elastic_fwi.inversion import compute_residual_wave_interaction_image_dg0

    assert compute_residual_wave_interaction_image_dg0 is not None


def test_residual_wave_interaction_rhs_factory_rejects_invalid_keep():
    class DummyFullRHS:
        pass

    try:
        make_single_source_vector_array_full_rhs(DummyFullRHS(), keep="invalid")  # type: ignore[arg-type]
    except ValueError as exc:
        assert "keep must be" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid residual-wave RHS role.")

from elastic_fwi.inversion.residual_wave_interaction_image import (
    make_single_source_vector_array_full_rhs,
)
from elastic_fwi.inversion.vector_array_adjoint_sources import (
    VectorArrayAdjointRHSRecord,
    VectorArrayAdjointRHSResult,
    VectorArrayFullAdjointRHSResult,
)


class _FakeVec:
    def __init__(self, value):
        self.value = float(value)
        self.destroyed = False

    def copy(self):
        return _FakeVec(self.value)

    def set(self, value):
        self.value = float(value)

    def scale(self, value):
        self.value *= float(value)

    def assemble(self):
        pass

    def norm(self):
        return abs(self.value)

    def getSize(self):
        return 1

    def getLocalSize(self):
        return 1

    def destroy(self):
        self.destroyed = True


def _rhs_result(role: str, value: float) -> VectorArrayAdjointRHSResult:
    vector = _FakeVec(value)
    record = VectorArrayAdjointRHSRecord(
        step=0,
        time=0.0,
        observation_name=f"{role}_observation",
        array_name=f"{role}_array",
        role=role,
        residual_max_abs=abs(value),
        residual_l2_norm=abs(value),
        rhs_norm=abs(value),
        rhs_size_global=1,
        rhs_size_local=1,
    )
    return VectorArrayAdjointRHSResult(
        records=(record,),
        rhs_vectors=(vector,),
    )


def test_transmission_residual_wave_rhs_uses_positive_sign():
    full_rhs = VectorArrayFullAdjointRHSResult(
        reflection=_rhs_result("reflection", 2.0),
        transmission=_rhs_result("transmission", 3.0),
    )

    split = make_single_source_vector_array_full_rhs(
        full_rhs,
        keep="transmission",
    )

    assert split.keep == "transmission"
    assert split.active_scale == 1.0
    assert split.full_rhs.reflection.rhs_vectors[0].value == 0.0
    assert split.full_rhs.transmission.rhs_vectors[0].value == 3.0
    assert split.full_rhs.transmission.records[0].role == "transmission"

    split.destroy_zero_rhs()
    full_rhs.destroy()


def test_reflection_residual_wave_rhs_keeps_positive_legacy_cpp_sign():
    full_rhs = VectorArrayFullAdjointRHSResult(
        reflection=_rhs_result("reflection", 2.0),
        transmission=_rhs_result("transmission", 3.0),
    )

    split = make_single_source_vector_array_full_rhs(
        full_rhs,
        keep="reflection",
    )

    assert split.keep == "reflection"
    assert split.active_scale == 1.0
    assert split.full_rhs.reflection.rhs_vectors[0].value == 2.0
    assert split.full_rhs.transmission.rhs_vectors[0].value == 0.0

    split.destroy_zero_rhs()
    full_rhs.destroy()
