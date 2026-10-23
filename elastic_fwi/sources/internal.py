from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any

import numpy as np
import ufl
from dolfinx import fem
from dolfinx.fem import petsc as fem_petsc
from petsc4py import PETSc

from elastic_fwi.sources.traction import HannSinePulse, evaluate_time_signal


def _finite_float(value: Any, name: str) -> float:
    out = float(value)
    if not isfinite(out):
        raise ValueError(f"{name} must be finite.")
    return out


def _positive_float(value: Any, name: str) -> float:
    out = _finite_float(value, name)
    if out <= 0.0:
        raise ValueError(f"{name} must be strictly positive.")
    return out


@dataclass(frozen=True)
class InternalDiskSource:
    """
    Regularized internal point source over a disk.

    Weak form:
        ∫_Ω chi_disk(x) g(t) · v dx

    This is the stable FEM analogue of a point source.
    """

    name: str
    center_x: float
    center_y: float
    radius: float
    direction: tuple[float, float]
    time_signal: Any

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("InternalDiskSource name must be non-empty.")

        center_x = _finite_float(self.center_x, "center_x")
        center_y = _finite_float(self.center_y, "center_y")
        radius = _positive_float(self.radius, "radius")

        if len(self.direction) != 2:
            raise ValueError("direction must contain exactly two components.")

        direction = (float(self.direction[0]), float(self.direction[1]))
        if not all(isfinite(value) for value in direction):
            raise ValueError("direction components must be finite.")

        if direction[0] == 0.0 and direction[1] == 0.0:
            raise ValueError("direction must not be the zero vector.")

        evaluate_time_signal(self.time_signal, 0.0)

        object.__setattr__(self, "center_x", center_x)
        object.__setattr__(self, "center_y", center_y)
        object.__setattr__(self, "radius", radius)
        object.__setattr__(self, "direction", direction)

    def scalar_amplitude(self, t: float) -> float:
        return evaluate_time_signal(self.time_signal, t)

    def value(self, t: float) -> tuple[float, float]:
        amplitude = self.scalar_amplitude(t)
        return (
            amplitude * self.direction[0],
            amplitude * self.direction[1],
        )

    def assemble_vector(self, V, facet_tags, t: float):
        msh = V.mesh
        v = ufl.TestFunction(V)
        x = ufl.SpatialCoordinate(msh)

        r2 = float(self.radius) ** 2
        distance2 = (x[0] - self.center_x) ** 2 + (x[1] - self.center_y) ** 2
        chi = ufl.conditional(ufl.le(distance2, r2), 1.0, 0.0)

        g = fem.Constant(msh, np.asarray(self.value(t), dtype=PETSc.ScalarType))
        form = fem.form(ufl.dot(g, v) * chi * ufl.dx)

        vector = fem_petsc.assemble_vector(form)
        vector.ghostUpdate(
            addv=PETSc.InsertMode.ADD_VALUES,
            mode=PETSc.ScatterMode.REVERSE,
        )
        return vector

    def metadata(self) -> dict[str, Any]:
        return {
            "kind": "internal_disk",
            "name": self.name,
            "center_x": self.center_x,
            "center_y": self.center_y,
            "radius": self.radius,
            "direction": list(self.direction),
        }


@dataclass(frozen=True)
class InternalGaussianSource:
    """
    Smooth internal source.

    Weak form:
        ∫_Ω exp(-|x-x0|²/(2 sigma²)) g(t) · v dx
    """

    name: str
    center_x: float
    center_y: float
    sigma: float
    direction: tuple[float, float]
    time_signal: Any

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("InternalGaussianSource name must be non-empty.")

        center_x = _finite_float(self.center_x, "center_x")
        center_y = _finite_float(self.center_y, "center_y")
        sigma = _positive_float(self.sigma, "sigma")

        if len(self.direction) != 2:
            raise ValueError("direction must contain exactly two components.")

        direction = (float(self.direction[0]), float(self.direction[1]))
        if not all(isfinite(value) for value in direction):
            raise ValueError("direction components must be finite.")

        if direction[0] == 0.0 and direction[1] == 0.0:
            raise ValueError("direction must not be the zero vector.")

        evaluate_time_signal(self.time_signal, 0.0)

        object.__setattr__(self, "center_x", center_x)
        object.__setattr__(self, "center_y", center_y)
        object.__setattr__(self, "sigma", sigma)
        object.__setattr__(self, "direction", direction)

    def scalar_amplitude(self, t: float) -> float:
        return evaluate_time_signal(self.time_signal, t)

    def value(self, t: float) -> tuple[float, float]:
        amplitude = self.scalar_amplitude(t)
        return (
            amplitude * self.direction[0],
            amplitude * self.direction[1],
        )

    def assemble_vector(self, V, facet_tags, t: float):
        msh = V.mesh
        v = ufl.TestFunction(V)
        x = ufl.SpatialCoordinate(msh)

        sigma2 = float(self.sigma) ** 2
        distance2 = (x[0] - self.center_x) ** 2 + (x[1] - self.center_y) ** 2
        weight = ufl.exp(-distance2 / (2.0 * sigma2))

        g = fem.Constant(msh, np.asarray(self.value(t), dtype=PETSc.ScalarType))
        form = fem.form(ufl.dot(g, v) * weight * ufl.dx)

        vector = fem_petsc.assemble_vector(form)
        vector.ghostUpdate(
            addv=PETSc.InsertMode.ADD_VALUES,
            mode=PETSc.ScatterMode.REVERSE,
        )
        return vector

    def metadata(self) -> dict[str, Any]:
        return {
            "kind": "internal_gaussian",
            "name": self.name,
            "center_x": self.center_x,
            "center_y": self.center_y,
            "sigma": self.sigma,
            "direction": list(self.direction),
        }


def internal_disk_hann_source(
    *,
    center_x: float,
    center_y: float,
    radius: float,
    direction: tuple[float, float] = (0.0, 1.0),
    amplitude: float = 1.0,
    frequency: float = 1.0,
    t0: float = 0.0,
    num_periods: float = 1.0,
    name: str = "internal_disk_hann_sine",
) -> InternalDiskSource:
    return InternalDiskSource(
        name=name,
        center_x=center_x,
        center_y=center_y,
        radius=radius,
        direction=direction,
        time_signal=HannSinePulse(
            frequency=frequency,
            amplitude=amplitude,
            t0=t0,
            num_periods=num_periods,
        ),
    )


def internal_gaussian_hann_source(
    *,
    center_x: float,
    center_y: float,
    sigma: float,
    direction: tuple[float, float] = (0.0, 1.0),
    amplitude: float = 1.0,
    frequency: float = 1.0,
    t0: float = 0.0,
    num_periods: float = 1.0,
    name: str = "internal_gaussian_hann_sine",
) -> InternalGaussianSource:
    return InternalGaussianSource(
        name=name,
        center_x=center_x,
        center_y=center_y,
        sigma=sigma,
        direction=direction,
        time_signal=HannSinePulse(
            frequency=frequency,
            amplitude=amplitude,
            t0=t0,
            num_periods=num_periods,
        ),
    )



@dataclass(frozen=True)
class InternalGaussianArraySource:
    """
    Smooth internal multi-element source.

    Weak form:
        ∫_Ω [sum_i exp(-|x-x_i|²/(2 sigma²))] g(t) · v dx

    This models an NDT-like internal probe made of several synchronized
    Gaussian source elements distributed along a segment.
    """

    name: str
    points: tuple[tuple[float, float], ...]
    sigma: float
    direction: tuple[float, float]
    time_signal: Any

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("InternalGaussianArraySource name must be non-empty.")

        if len(self.points) <= 0:
            raise ValueError("points must contain at least one source element.")

        points: list[tuple[float, float]] = []
        for index, point in enumerate(self.points):
            if len(point) != 2:
                raise ValueError(f"points[{index}] must contain exactly two coordinates.")
            x = _finite_float(point[0], f"points[{index}][0]")
            y = _finite_float(point[1], f"points[{index}][1]")
            points.append((x, y))

        sigma = _positive_float(self.sigma, "sigma")

        if len(self.direction) != 2:
            raise ValueError("direction must contain exactly two components.")

        direction = (float(self.direction[0]), float(self.direction[1]))
        if not all(isfinite(value) for value in direction):
            raise ValueError("direction components must be finite.")

        if direction[0] == 0.0 and direction[1] == 0.0:
            raise ValueError("direction must not be the zero vector.")

        evaluate_time_signal(self.time_signal, 0.0)

        object.__setattr__(self, "points", tuple(points))
        object.__setattr__(self, "sigma", sigma)
        object.__setattr__(self, "direction", direction)

    @property
    def num_elements(self) -> int:
        return len(self.points)

    @property
    def start(self) -> tuple[float, float]:
        return self.points[0]

    @property
    def end(self) -> tuple[float, float]:
        return self.points[-1]

    @property
    def center_x(self) -> float:
        return 0.5 * (self.start[0] + self.end[0])

    @property
    def center_y(self) -> float:
        return 0.5 * (self.start[1] + self.end[1])

    def scalar_amplitude(self, t: float) -> float:
        return evaluate_time_signal(self.time_signal, t)

    def value(self, t: float) -> tuple[float, float]:
        amplitude = self.scalar_amplitude(t)
        return (
            amplitude * self.direction[0],
            amplitude * self.direction[1],
        )

    def assemble_vector(self, V, facet_tags, t: float):
        msh = V.mesh
        v = ufl.TestFunction(V)
        x = ufl.SpatialCoordinate(msh)

        sigma2 = float(self.sigma) ** 2
        weight = 0.0

        for center_x, center_y in self.points:
            distance2 = (x[0] - center_x) ** 2 + (x[1] - center_y) ** 2
            weight = weight + ufl.exp(-distance2 / (2.0 * sigma2))

        g = fem.Constant(msh, np.asarray(self.value(t), dtype=PETSc.ScalarType))
        form = fem.form(ufl.dot(g, v) * weight * ufl.dx)

        vector = fem_petsc.assemble_vector(form)
        vector.ghostUpdate(
            addv=PETSc.InsertMode.ADD_VALUES,
            mode=PETSc.ScatterMode.REVERSE,
        )
        return vector

    def metadata(self) -> dict[str, Any]:
        return {
            "kind": "internal_gaussian_array",
            "name": self.name,
            "num_elements": self.num_elements,
            "start_x": self.start[0],
            "start_y": self.start[1],
            "end_x": self.end[0],
            "end_y": self.end[1],
            "center_x": self.center_x,
            "center_y": self.center_y,
            "sigma": self.sigma,
            "direction": list(self.direction),
            "points": [
                {"x": float(px), "y": float(py)}
                for px, py in self.points
            ],
        }


def internal_gaussian_array_hann_source(
    *,
    start_x: float,
    start_y: float,
    end_x: float,
    end_y: float,
    num_elements: int,
    sigma: float,
    direction: tuple[float, float] = (0.0, 1.0),
    amplitude: float = 1.0,
    frequency: float = 1.0,
    t0: float = 0.0,
    num_periods: float = 1.0,
    name: str = "internal_gaussian_array_hann_sine",
) -> InternalGaussianArraySource:
    num_elements = int(num_elements)
    if num_elements <= 0:
        raise ValueError("num_elements must be strictly positive.")

    start_x = _finite_float(start_x, "start_x")
    start_y = _finite_float(start_y, "start_y")
    end_x = _finite_float(end_x, "end_x")
    end_y = _finite_float(end_y, "end_y")

    if num_elements == 1:
        points = ((start_x, start_y),)
    else:
        xs = np.linspace(start_x, end_x, num_elements)
        ys = np.linspace(start_y, end_y, num_elements)
        points = tuple((float(x), float(y)) for x, y in zip(xs, ys, strict=True))

    return InternalGaussianArraySource(
        name=name,
        points=points,
        sigma=sigma,
        direction=direction,
        time_signal=HannSinePulse(
            frequency=frequency,
            amplitude=amplitude,
            t0=t0,
            num_periods=num_periods,
        ),
    )
