from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, pi, sin
from typing import Any

import numpy as np
import ufl
from dolfinx import fem
from mpi4py import MPI
from petsc4py import PETSc


@dataclass(frozen=True)
class ConstantTimeSignal:
    """Constant scalar time signal."""

    amplitude: float

    def __post_init__(self) -> None:
        amplitude = float(self.amplitude)

        if not isfinite(amplitude):
            raise ValueError("Signal amplitude must be finite.")

        object.__setattr__(self, "amplitude", amplitude)

    def value(self, t: float) -> float:
        t = float(t)

        if not isfinite(t):
            raise ValueError("Time must be finite.")

        return self.amplitude


@dataclass(frozen=True)
class HannSinePulse:
    """
    Hann-windowed sinusoidal pulse.

    The signal is

        A sin(2 pi f tau) sin^2(pi tau / T)

    for tau = t - t0 in [0, T], and zero outside.
    """

    frequency: float
    amplitude: float = 1.0
    t0: float = 0.0
    num_periods: float = 1.0

    def __post_init__(self) -> None:
        frequency = float(self.frequency)
        amplitude = float(self.amplitude)
        t0 = float(self.t0)
        num_periods = float(self.num_periods)

        if not isfinite(frequency) or frequency <= 0.0:
            raise ValueError("Pulse frequency must be strictly positive and finite.")

        if not isfinite(amplitude):
            raise ValueError("Pulse amplitude must be finite.")

        if not isfinite(t0):
            raise ValueError("Pulse initial time must be finite.")

        if not isfinite(num_periods) or num_periods <= 0.0:
            raise ValueError("Pulse number of periods must be strictly positive and finite.")

        object.__setattr__(self, "frequency", frequency)
        object.__setattr__(self, "amplitude", amplitude)
        object.__setattr__(self, "t0", t0)
        object.__setattr__(self, "num_periods", num_periods)

    @property
    def duration(self) -> float:
        return self.num_periods / self.frequency

    @property
    def t_end(self) -> float:
        return self.t0 + self.duration

    def value(self, t: float) -> float:
        t = float(t)

        if not isfinite(t):
            raise ValueError("Time must be finite.")

        tau = t - self.t0

        if tau < 0.0 or tau > self.duration:
            return 0.0

        if tau == 0.0 or tau == self.duration:
            return 0.0

        window = sin(pi * tau / self.duration) ** 2
        carrier = sin(2.0 * pi * self.frequency * tau)

        return self.amplitude * carrier * window


def evaluate_time_signal(signal: Any, t: float) -> float:
    if hasattr(signal, "value"):
        value = signal.value(t)
    elif callable(signal):
        value = signal(t)
    else:
        raise TypeError("A time signal must expose a value(t) method or be callable.")

    value = float(value)

    if not isfinite(value):
        raise ValueError("Time signal value must be finite.")

    return value


@dataclass(frozen=True)
class BoundaryTractionSource:
    """
    Weak boundary traction source.

    It represents

        int_{Gamma_tag} g(t) . v ds,

    with g(t) = signal(t) * direction.
    """

    name: str
    tag: int
    direction: tuple[float, float]
    time_signal: Any

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Source name must be non-empty.")

        if self.tag <= 0:
            raise ValueError("Boundary source tag must be a strictly positive integer.")

        if len(self.direction) != 2:
            raise ValueError("A 2-D traction direction must contain exactly two components.")

        direction = (float(self.direction[0]), float(self.direction[1]))

        if not all(isfinite(value) for value in direction):
            raise ValueError("Traction direction components must be finite.")

        if direction[0] == 0.0 and direction[1] == 0.0:
            raise ValueError("Traction direction must not be the zero vector.")

        evaluate_time_signal(self.time_signal, 0.0)

        object.__setattr__(self, "direction", direction)

    def scalar_amplitude(self, t: float) -> float:
        return evaluate_time_signal(self.time_signal, t)

    def value(self, t: float) -> tuple[float, float]:
        amplitude = self.scalar_amplitude(t)
        return (
            amplitude * self.direction[0],
            amplitude * self.direction[1],
        )


def gamma1_vertical_traction_source(
    *,
    tag: int = 1,
    amplitude: float = 1.0,
    frequency: float = 1.0,
    t0: float = 0.0,
    num_periods: float = 1.0,
    name: str = "gamma1_vertical_hann_sine",
) -> BoundaryTractionSource:
    """
    Vertical pulse on Gamma1, compatible with the legacy C++ code.

    The source of the legacy C++ code is vertical. This function gives the weak
    boundary-traction analogue on Gamma1.
    """

    return BoundaryTractionSource(
        name=name,
        tag=tag,
        direction=(0.0, 1.0),
        time_signal=HannSinePulse(
            frequency=frequency,
            amplitude=amplitude,
            t0=t0,
            num_periods=num_periods,
        ),
    )


def _source_constant(msh, source: BoundaryTractionSource, t: float) -> fem.Constant:
    value = np.asarray(source.value(t), dtype=PETSc.ScalarType)
    return fem.Constant(msh, value)


def _assemble_global_scalar(msh, expression) -> float:
    local_value = fem.assemble_scalar(fem.form(expression))
    global_value = msh.comm.allreduce(local_value, op=MPI.SUM)
    return float(np.real_if_close(global_value))


def boundary_traction_work(
    test_function,
    facet_tags,
    source: BoundaryTractionSource,
    t: float,
) -> float:
    """
    Assemble int_{Gamma_tag} g(t) . test_function ds.
    """

    msh = test_function.function_space.mesh
    ds = ufl.Measure("ds", domain=msh, subdomain_data=facet_tags)
    g = _source_constant(msh, source, t)

    expression = ufl.dot(g, test_function) * ds(source.tag)

    return _assemble_global_scalar(msh, expression)


def boundary_traction_l2_norm_squared(
    V,
    facet_tags,
    source: BoundaryTractionSource,
    t: float,
) -> float:
    """
    Assemble int_{Gamma_tag} |g(t)|^2 ds.
    """

    msh = V.mesh
    ds = ufl.Measure("ds", domain=msh, subdomain_data=facet_tags)
    g = _source_constant(msh, source, t)

    expression = ufl.dot(g, g) * ds(source.tag)

    return _assemble_global_scalar(msh, expression)


def assemble_boundary_traction_vector(
    V,
    facet_tags,
    source: BoundaryTractionSource,
    t: float,
):
    """
    Assemble the PETSc RHS vector associated with int_{Gamma_tag} g(t) . v ds.
    """

    from dolfinx.fem import petsc as fem_petsc

    msh = V.mesh
    v = ufl.TestFunction(V)
    ds = ufl.Measure("ds", domain=msh, subdomain_data=facet_tags)
    g = _source_constant(msh, source, t)

    form = fem.form(ufl.dot(g, v) * ds(source.tag))
    vector = fem_petsc.assemble_vector(form)

    vector.ghostUpdate(
        addv=PETSc.InsertMode.ADD_VALUES,
        mode=PETSc.ScatterMode.REVERSE,
    )

    return vector
