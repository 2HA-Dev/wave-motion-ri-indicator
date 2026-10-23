import numpy as np
import pytest

pytest.importorskip("dolfinx")

from mpi4py import MPI
from dolfinx import fem, mesh

from elastic_fwi.geometry.boundary_tags import (
    build_boundary_meshtags,
    unit_square_pitch_catch_boundary_specs,
)
from elastic_fwi.sources.traction import (
    BoundaryTractionSource,
    ConstantTimeSignal,
    HannSinePulse,
    assemble_boundary_traction_vector,
    boundary_traction_l2_norm_squared,
    boundary_traction_work,
    gamma1_vertical_traction_source,
)


def _create_mesh_and_tags():
    msh = mesh.create_unit_square(MPI.COMM_WORLD, 4, 4)
    facet_tags = build_boundary_meshtags(msh, unit_square_pitch_catch_boundary_specs())
    return msh, facet_tags


def _create_vector_space(msh):
    return fem.functionspace(msh, ("Lagrange", 1, (msh.geometry.dim,)))


def _constant_vector_function(V, values, name):
    function = fem.Function(V)
    function.name = name

    vx, vy = values

    def expr(x):
        return np.vstack(
            (
                np.full(x.shape[1], vx, dtype=float),
                np.full(x.shape[1], vy, dtype=float),
            )
        )

    function.interpolate(expr)
    function.x.scatter_forward()
    return function


def test_constant_time_signal():
    signal = ConstantTimeSignal(3.0)

    assert signal.value(0.0) == pytest.approx(3.0)
    assert signal.value(10.0) == pytest.approx(3.0)


def test_hann_sine_pulse_window_and_value():
    pulse = HannSinePulse(frequency=1.0, amplitude=2.0, t0=0.0, num_periods=1.0)

    assert pulse.duration == pytest.approx(1.0)
    assert pulse.t_end == pytest.approx(1.0)

    assert pulse.value(-0.1) == pytest.approx(0.0)
    assert pulse.value(0.0) == pytest.approx(0.0)
    assert pulse.value(1.0) == pytest.approx(0.0)
    assert pulse.value(1.1) == pytest.approx(0.0)

    expected = 2.0 * np.sin(2.0 * np.pi * 0.25) * np.sin(np.pi * 0.25) ** 2
    assert pulse.value(0.25) == pytest.approx(expected)


def test_gamma1_vertical_traction_source_values():
    source = gamma1_vertical_traction_source(
        tag=1,
        amplitude=2.0,
        frequency=1.0,
        t0=0.0,
        num_periods=1.0,
    )

    assert source.name == "gamma1_vertical_hann_sine"
    assert source.tag == 1
    assert source.direction == pytest.approx((0.0, 1.0))
    assert source.value(-0.1) == pytest.approx((0.0, 0.0))

    expected_scalar = 2.0 * np.sin(2.0 * np.pi * 0.25) * np.sin(np.pi * 0.25) ** 2
    assert source.value(0.25) == pytest.approx((0.0, expected_scalar))


def test_boundary_traction_work_for_constant_vector_field():
    msh, facet_tags = _create_mesh_and_tags()
    V = _create_vector_space(msh)

    test_function = _constant_vector_function(V, (2.0, -1.0), "u")

    source = BoundaryTractionSource(
        name="constant_vertical",
        tag=1,
        direction=(0.0, 1.0),
        time_signal=ConstantTimeSignal(3.0),
    )

    work = boundary_traction_work(test_function, facet_tags, source, t=0.0)

    assert work == pytest.approx(-3.0)


def test_boundary_traction_l2_norm_squared():
    msh, facet_tags = _create_mesh_and_tags()
    V = _create_vector_space(msh)

    source = BoundaryTractionSource(
        name="constant_vertical",
        tag=1,
        direction=(0.0, 2.0),
        time_signal=ConstantTimeSignal(3.0),
    )

    norm_squared = boundary_traction_l2_norm_squared(V, facet_tags, source, t=0.0)

    assert norm_squared == pytest.approx(36.0)


def test_assemble_boundary_traction_vector_has_expected_size_and_positive_norm():
    msh, facet_tags = _create_mesh_and_tags()
    V = _create_vector_space(msh)

    source = BoundaryTractionSource(
        name="constant_vertical",
        tag=1,
        direction=(0.0, 1.0),
        time_signal=ConstantTimeSignal(3.0),
    )

    vector = assemble_boundary_traction_vector(V, facet_tags, source, t=0.0)

    expected_global_size = V.dofmap.index_map.size_global * V.dofmap.index_map_bs

    assert vector.getSize() == expected_global_size
    assert vector.norm() > 0.0


def test_invalid_signals_and_sources_are_rejected():
    with pytest.raises(ValueError):
        ConstantTimeSignal(np.nan)

    with pytest.raises(ValueError):
        HannSinePulse(frequency=0.0)

    with pytest.raises(ValueError):
        HannSinePulse(frequency=1.0, num_periods=0.0)

    with pytest.raises(ValueError):
        BoundaryTractionSource(
            name="",
            tag=1,
            direction=(0.0, 1.0),
            time_signal=ConstantTimeSignal(1.0),
        )

    with pytest.raises(ValueError):
        BoundaryTractionSource(
            name="bad_tag",
            tag=0,
            direction=(0.0, 1.0),
            time_signal=ConstantTimeSignal(1.0),
        )

    with pytest.raises(ValueError):
        BoundaryTractionSource(
            name="bad_direction",
            tag=1,
            direction=(0.0, 0.0),
            time_signal=ConstantTimeSignal(1.0),
        )
