from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any

import numpy as np
import ufl
from dolfinx import fem
from mpi4py import MPI
from petsc4py import PETSc

from elastic_fwi.observations.time_series import displacement_vector_to_function
from elastic_fwi.transducers import TransducerArraySpec


@dataclass(frozen=True)
class VectorArrayObservationSpec:
    """Regularized vector observation on a transducer array.

    For receiver j,

        H_j(u)(t) = int_Omega psi_j(x) u(x,t) dx / int_Omega psi_j(x) dx,

    with H_j(u)(t) in R^2.

    This observable is vector-valued and is intended to represent ux, uy
    measured by a finite-size transducer.
    """

    name: str
    array: TransducerArraySpec
    sigma: float
    quadrature_degree: int = 10

    @property
    def num_receivers(self) -> int:
        return int(self.array.num_elements)

    @property
    def centers(self) -> np.ndarray:
        return self.array.centers

    @property
    def tag(self) -> int | None:
        return self.array.tag

    @property
    def role(self) -> str:
        return self.array.role

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": "vector_array_observation",
            "name": self.name,
            "array": self.array.as_manifest_dict(),
            "sigma": float(self.sigma),
            "quadrature_degree": int(self.quadrature_degree),
            "num_receivers": self.num_receivers,
            "returns": "vector",
            "components": ["x", "y"],
        }


@dataclass(frozen=True)
class VectorArrayObservationTimeSeriesRecord:
    step: int
    time: float
    observation_name: str
    array_name: str
    role: str
    values: np.ndarray

    def as_dict(self) -> dict[str, Any]:
        values = np.asarray(self.values, dtype=float)
        return {
            "step": int(self.step),
            "time": float(self.time),
            "observation_name": self.observation_name,
            "array_name": self.array_name,
            "role": self.role,
            "values": values.tolist(),
            "shape": list(values.shape),
        }


@dataclass(frozen=True)
class VectorArrayObservationTimeSeries:
    spec: VectorArrayObservationSpec
    times: np.ndarray
    steps: np.ndarray
    values: np.ndarray

    @property
    def num_steps(self) -> int:
        return int(self.values.shape[0])

    @property
    def num_receivers(self) -> int:
        return int(self.values.shape[1])

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": "vector_array_observation_time_series",
            "spec": self.spec.as_dict(),
            "num_steps": self.num_steps,
            "num_receivers": self.num_receivers,
            "shape": list(self.values.shape),
            "times": self.times.tolist(),
            "steps": self.steps.tolist(),
        }

    def records(self) -> list[VectorArrayObservationTimeSeriesRecord]:
        return [
            VectorArrayObservationTimeSeriesRecord(
                step=int(self.steps[k]),
                time=float(self.times[k]),
                observation_name=self.spec.name,
                array_name=self.spec.array.name,
                role=self.spec.role,
                values=self.values[k],
            )
            for k in range(self.num_steps)
        ]


def vector_array_observation(
    *,
    array: TransducerArraySpec,
    sigma: float | None = None,
    name: str | None = None,
    quadrature_degree: int = 10,
) -> VectorArrayObservationSpec:
    """Create a vector-array observation spec from a transducer array."""
    effective_sigma = array.sigma if sigma is None else sigma

    if effective_sigma is None:
        raise ValueError("sigma must be provided either explicitly or in the array spec.")

    effective_sigma = float(effective_sigma)
    if not isfinite(effective_sigma) or effective_sigma <= 0.0:
        raise ValueError("sigma must be strictly positive.")

    quadrature_degree = int(quadrature_degree)
    if quadrature_degree < 1:
        raise ValueError("quadrature_degree must be >= 1.")

    observation_name = name
    if observation_name is None:
        observation_name = f"{array.name}_vector_array_{array.role}"

    return VectorArrayObservationSpec(
        name=str(observation_name),
        array=array.with_sigma(effective_sigma),
        sigma=effective_sigma,
        quadrature_degree=quadrature_degree,
    )


def _gaussian_weight(center: np.ndarray, sigma: float, mesh):
    x = ufl.SpatialCoordinate(mesh)
    sigma2 = float(sigma) ** 2
    r2 = (x[0] - float(center[0])) ** 2 + (x[1] - float(center[1])) ** 2
    return ufl.exp(-0.5 * r2 / sigma2)


class VectorArrayObservationOperator:
    """Prebuilt regularized vector-array observation operator.

    Cached objects:
    - receiver centers;
    - Gaussian UFL weights;
    - global normalizations int psi_j dx;
    - one dx measure with fixed quadrature metadata.

    The same quadrature measure is used for the numerator and denominator.
    This is essential for exact recovery of constant vector fields.
    """

    def __init__(self, V, spec: VectorArrayObservationSpec):
        self.V = V
        self.mesh = V.mesh
        self.spec = spec
        self.centers = np.asarray(spec.centers, dtype=float)
        self.sigma = float(spec.sigma)
        self.quadrature_degree = int(spec.quadrature_degree)

        if self.centers.ndim != 2 or self.centers.shape[1] != 2:
            raise ValueError("centers must have shape (num_receivers, 2).")

        if self.centers.shape[0] <= 0:
            raise ValueError("At least one receiver is required.")

        if not isfinite(self.sigma) or self.sigma <= 0.0:
            raise ValueError("sigma must be strictly positive.")

        if self.quadrature_degree < 1:
            raise ValueError("quadrature_degree must be >= 1.")

        self.dx = ufl.Measure(
            "dx",
            domain=self.mesh,
            metadata={"quadrature_degree": self.quadrature_degree},
        )

        self.weights = tuple(
            _gaussian_weight(center, self.sigma, self.mesh)
            for center in self.centers
        )

        self.normalizations = self._assemble_normalizations()

    @property
    def num_receivers(self) -> int:
        return int(self.centers.shape[0])

    def _assemble_normalizations(self) -> np.ndarray:
        values = np.zeros(self.num_receivers, dtype=float)

        for j, weight in enumerate(self.weights):
            form = fem.form(weight * self.dx)
            local_value = fem.assemble_scalar(form)
            global_value = self.mesh.comm.allreduce(float(local_value), op=MPI.SUM)

            if not np.isfinite(global_value) or global_value <= 0.0:
                raise RuntimeError(
                    f"Invalid normalization for receiver {j}: {global_value}"
                )

            values[j] = float(global_value)

        return values

    def evaluate_function(self, u_h: fem.Function) -> np.ndarray:
        """Evaluate H(u) with shape (num_receivers, 2)."""
        if u_h.function_space.mesh is not self.mesh:
            raise ValueError("u_h must be defined on the same mesh as the operator.")

        u_h.x.scatter_forward()

        values = np.zeros((self.num_receivers, 2), dtype=float)

        for j, weight in enumerate(self.weights):
            inv_norm = 1.0 / float(self.normalizations[j])

            for component in range(2):
                form = fem.form(inv_norm * weight * u_h[component] * self.dx)
                local_value = fem.assemble_scalar(form)
                global_value = self.mesh.comm.allreduce(float(local_value), op=MPI.SUM)
                values[j, component] = float(global_value)

        return values

    def evaluate_vector(
        self,
        vector: PETSc.Vec,
        *,
        name: str = "vector_array_observation_state",
    ) -> np.ndarray:
        function = displacement_vector_to_function(self.V, vector, name=name)
        function.x.scatter_forward()
        return self.evaluate_function(function)

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": "vector_array_observation_operator",
            "spec": self.spec.as_dict(),
            "num_receivers": self.num_receivers,
            "centers": self.centers.tolist(),
            "sigma": float(self.sigma),
            "quadrature_degree": int(self.quadrature_degree),
            "normalizations": self.normalizations.tolist(),
        }


def build_vector_array_observation_operator(
    V,
    spec: VectorArrayObservationSpec,
) -> VectorArrayObservationOperator:
    return VectorArrayObservationOperator(V, spec)


def evaluate_vector_array_observation_from_vector(
    V,
    vector: PETSc.Vec,
    spec: VectorArrayObservationSpec,
    *,
    name: str = "vector_array_observation_state",
) -> np.ndarray:
    operator = build_vector_array_observation_operator(V, spec)
    return operator.evaluate_vector(vector, name=name)


def _normalize_snapshot_item(index: int, snapshot):
    """Return (step, time, vector_like) from either a raw snapshot or a triplet.

    Supported formats:
    - (step, time, vector_like)
    - raw vector_like

    The direct time-loop pipeline stores raw displacement snapshots, whereas
    unit tests and some diagnostics may use explicit triplets.
    """
    if isinstance(snapshot, tuple) and len(snapshot) == 3:
        step, time, vector = snapshot
        return int(step), float(time), vector

    return int(index), float(index), snapshot


def _snapshot_to_observation_function(V, snapshot, *, name: str) -> fem.Function:
    """Convert a PETSc Vec or array-like snapshot to a dolfinx Function."""
    if isinstance(snapshot, PETSc.Vec):
        function = displacement_vector_to_function(V, snapshot, name=name)
        function.x.scatter_forward()
        return function

    function = fem.Function(V)
    function.name = name

    array = np.asarray(snapshot, dtype=float).reshape(-1)
    local_size = function.x.array.size

    if array.size == local_size:
        function.x.array[:] = array
    elif array.size >= local_size:
        function.x.array[:] = array[:local_size]
    else:
        raise ValueError(
            "Snapshot array is too small for the local Function vector: "
            f"got {array.size}, expected at least {local_size}."
        )

    function.x.scatter_forward()
    return function


def evaluate_vector_array_observation_time_series(
    V,
    snapshots,
    spec: VectorArrayObservationSpec,
) -> VectorArrayObservationTimeSeries:
    """Evaluate a vector-array observation on a displacement snapshot series.

    Returned values have shape:

        (num_steps, num_receivers, 2)

    Accepted snapshot formats:
    - explicit triplets (step, time, vector_like);
    - raw displacement snapshots as stored by the direct time-loop pipeline.
    """
    snapshots = tuple(snapshots)
    operator = build_vector_array_observation_operator(V, spec)

    steps = np.zeros(len(snapshots), dtype=np.int64)
    times = np.zeros(len(snapshots), dtype=float)
    values = np.zeros((len(snapshots), operator.num_receivers, 2), dtype=float)

    for k, snapshot in enumerate(snapshots):
        step, time, vector = _normalize_snapshot_item(k, snapshot)

        function = _snapshot_to_observation_function(
            V,
            vector,
            name=f"{spec.name}_step_{step}",
        )

        steps[k] = int(step)
        times[k] = float(time)
        values[k, :, :] = operator.evaluate_function(function)

    return VectorArrayObservationTimeSeries(
        spec=spec,
        steps=steps,
        times=times,
        values=values,
    )
