from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence

import numpy as np


ArrayRole = Literal[
    "source",
    "reflection",
    "transmission",
    "source_reflection",
]


@dataclass(frozen=True)
class TransducerArraySpec:
    """Geometric description of a transducer array.

    Two geometries are supported:
    - uniform linear array between start and end;
    - explicit array defined by centers_override.

    Physical convention:
    - Gamma1: role="source_reflection", emitting array and reflection receiver array;
    - Gamma2: role="transmission", transmission receiver array.
    """

    name: str
    role: ArrayRole
    start: tuple[float, float]
    end: tuple[float, float]
    num_elements: int
    sigma: float | None = None
    tag: int | None = None
    centers_override: tuple[tuple[float, float], ...] | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name must be a non-empty string.")

        allowed_roles = {"source", "reflection", "transmission", "source_reflection"}
        if self.role not in allowed_roles:
            raise ValueError(f"role must be one of {sorted(allowed_roles)}.")

        if self.num_elements < 1:
            raise ValueError("num_elements must be >= 1.")

        sx, sy = self.start
        ex, ey = self.end
        values = (sx, sy, ex, ey)
        if not all(np.isfinite(values)):
            raise ValueError("start and end coordinates must be finite.")

        if self.sigma is not None:
            if not np.isfinite(self.sigma) or self.sigma <= 0.0:
                raise ValueError("sigma must be strictly positive when provided.")

        if self.tag is not None and self.tag <= 0:
            raise ValueError("tag must be strictly positive when provided.")

        if self.centers_override is not None:
            centers = np.asarray(self.centers_override, dtype=float)
            if centers.ndim != 2 or centers.shape[1] != 2:
                raise ValueError("centers_override must have shape (num_elements, 2).")
            if centers.shape[0] != self.num_elements:
                raise ValueError(
                    "centers_override length must match num_elements."
                )
            if not np.all(np.isfinite(centers)):
                raise ValueError("centers_override coordinates must be finite.")

    @property
    def centers(self) -> np.ndarray:
        """Return transducer centers with shape (num_elements, 2)."""
        if self.centers_override is not None:
            return np.asarray(self.centers_override, dtype=float)

        start = np.asarray(self.start, dtype=float)
        end = np.asarray(self.end, dtype=float)

        if self.num_elements == 1:
            return (0.5 * (start + end))[None, :]

        weights = np.linspace(0.0, 1.0, self.num_elements)
        return (1.0 - weights)[:, None] * start[None, :] + weights[:, None] * end[None, :]

    @property
    def spacing(self) -> float:
        """Return representative center-to-center spacing."""
        if self.num_elements <= 1:
            return 0.0

        centers = self.centers
        distances = np.linalg.norm(np.diff(centers, axis=0), axis=1)
        positive = distances[distances > 0.0]
        if positive.size == 0:
            return 0.0
        return float(np.mean(positive))

    def with_sigma(self, sigma: float) -> "TransducerArraySpec":
        return TransducerArraySpec(
            name=self.name,
            role=self.role,
            start=self.start,
            end=self.end,
            num_elements=self.num_elements,
            sigma=float(sigma),
            tag=self.tag,
            centers_override=self.centers_override,
        )

    def with_y(self, y: float) -> "TransducerArraySpec":
        if self.centers_override is not None:
            centers = tuple((float(x), float(y)) for x, _ in self.centers_override)
        else:
            centers = None

        return TransducerArraySpec(
            name=self.name,
            role=self.role,
            start=(self.start[0], float(y)),
            end=(self.end[0], float(y)),
            num_elements=self.num_elements,
            sigma=self.sigma,
            tag=self.tag,
            centers_override=centers,
        )

    def as_manifest_dict(self) -> dict[str, Any]:
        centers = self.centers
        return {
            "name": self.name,
            "role": self.role,
            "geometry": "explicit_centers" if self.centers_override is not None else "line",
            "start": [float(self.start[0]), float(self.start[1])],
            "end": [float(self.end[0]), float(self.end[1])],
            "num_elements": int(self.num_elements),
            "sigma": None if self.sigma is None else float(self.sigma),
            "tag": self.tag,
            "spacing": float(self.spacing),
            "centers": centers.tolist(),
        }


def build_line_transducer_array(
    *,
    name: str,
    role: ArrayRole,
    start_x: float,
    start_y: float,
    end_x: float,
    end_y: float,
    num_elements: int,
    sigma: float | None = None,
    tag: int | None = None,
) -> TransducerArraySpec:
    """Build a line array between two points."""
    return TransducerArraySpec(
        name=name,
        role=role,
        start=(float(start_x), float(start_y)),
        end=(float(end_x), float(end_y)),
        num_elements=int(num_elements),
        sigma=None if sigma is None else float(sigma),
        tag=tag,
    )


def _line_centers(
    start: tuple[float, float],
    end: tuple[float, float],
    num_elements: int,
) -> np.ndarray:
    if num_elements < 1:
        raise ValueError("num_elements must be >= 1.")

    start_arr = np.asarray(start, dtype=float)
    end_arr = np.asarray(end, dtype=float)

    if num_elements == 1:
        return (0.5 * (start_arr + end_arr))[None, :]

    weights = np.linspace(0.0, 1.0, int(num_elements))
    return (1.0 - weights)[:, None] * start_arr[None, :] + weights[:, None] * end_arr[None, :]


def build_polyline_transducer_array(
    *,
    name: str,
    role: ArrayRole,
    vertices: Sequence[tuple[float, float]],
    num_elements_per_segment: Sequence[int],
    sigma: float | None = None,
    tag: int | None = None,
    drop_duplicate_vertices: bool = True,
) -> TransducerArraySpec:
    """Build a multi-segment transducer array from an ordered polyline.

    Example for Gamma2 surrounding a square except the bottom side:

        vertices = [(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin)]

    This gives the left, top and right sides.
    """
    if len(vertices) < 2:
        raise ValueError("vertices must contain at least two points.")

    if len(num_elements_per_segment) != len(vertices) - 1:
        raise ValueError(
            "num_elements_per_segment must have length len(vertices) - 1."
        )

    all_centers: list[np.ndarray] = []

    for segment_index, n in enumerate(num_elements_per_segment):
        n_int = int(n)
        if n_int < 1:
            raise ValueError("each segment must have at least one transducer.")

        segment_centers = _line_centers(
            tuple(map(float, vertices[segment_index])),
            tuple(map(float, vertices[segment_index + 1])),
            n_int,
        )

        if drop_duplicate_vertices and all_centers:
            previous_last = all_centers[-1][-1]
            current_first = segment_centers[0]
            if np.allclose(previous_last, current_first):
                segment_centers = segment_centers[1:]

        if segment_centers.size:
            all_centers.append(segment_centers)

    centers = np.vstack(all_centers)
    centers_tuple = tuple((float(x), float(y)) for x, y in centers)

    return TransducerArraySpec(
        name=name,
        role=role,
        start=tuple(map(float, vertices[0])),
        end=tuple(map(float, vertices[-1])),
        num_elements=len(centers_tuple),
        sigma=None if sigma is None else float(sigma),
        tag=tag,
        centers_override=centers_tuple,
    )


def auto_place_array_inside_absorbing_layer(
    *,
    name: str,
    role: ArrayRole,
    side: Literal["bottom", "top"],
    domain_length_y: float,
    absorbing_layer_width: float,
    start_x: float,
    end_x: float,
    num_elements: int,
    sigma: float | None = None,
    tag: int | None = None,
) -> TransducerArraySpec:
    """Place a horizontal transducer array at the inner edge of a sponge layer.

    For a domain [0, Ly] with absorbing layer width X:
    - bottom array is placed at y = X ;
    - top array is placed at y = Ly - X.
    """
    if not np.isfinite(domain_length_y) or domain_length_y <= 0.0:
        raise ValueError("domain_length_y must be strictly positive.")

    if not np.isfinite(absorbing_layer_width) or absorbing_layer_width < 0.0:
        raise ValueError("absorbing_layer_width must be non-negative.")

    if 2.0 * absorbing_layer_width >= domain_length_y:
        raise ValueError(
            "absorbing_layer_width is too large: the useful domain would be empty."
        )

    if side == "bottom":
        y = absorbing_layer_width
    elif side == "top":
        y = domain_length_y - absorbing_layer_width
    else:
        raise ValueError("side must be either 'bottom' or 'top'.")

    return build_line_transducer_array(
        name=name,
        role=role,
        start_x=start_x,
        start_y=y,
        end_x=end_x,
        end_y=y,
        num_elements=num_elements,
        sigma=sigma,
        tag=tag,
    )


def build_pitch_catch_transducer_arrays(
    *,
    domain_length_y: float,
    absorbing_layer_width: float,
    start_x: float,
    end_x: float,
    num_elements: int,
    sigma: float | None = None,
    gamma1_tag: int | None = 1,
    gamma2_tag: int | None = 2,
) -> tuple[TransducerArraySpec, TransducerArraySpec]:
    """Build the default pitch-catch pair Gamma1/Gamma2.

    Gamma1 is placed at the bottom inner edge of the absorbing layer.
    Gamma2 is placed at the top inner edge of the absorbing layer.
    """
    gamma1 = auto_place_array_inside_absorbing_layer(
        name="gamma1",
        role="source_reflection",
        side="bottom",
        domain_length_y=domain_length_y,
        absorbing_layer_width=absorbing_layer_width,
        start_x=start_x,
        end_x=end_x,
        num_elements=num_elements,
        sigma=sigma,
        tag=gamma1_tag,
    )

    gamma2 = auto_place_array_inside_absorbing_layer(
        name="gamma2",
        role="transmission",
        side="top",
        domain_length_y=domain_length_y,
        absorbing_layer_width=absorbing_layer_width,
        start_x=start_x,
        end_x=end_x,
        num_elements=num_elements,
        sigma=sigma,
        tag=gamma2_tag,
    )

    return gamma1, gamma2


def recommended_sigma_from_mesh_size(h: float, factor: float = 2.0) -> float:
    """Return the recommended receiver Gaussian width sigma_rec ~= factor*h."""
    if not np.isfinite(h) or h <= 0.0:
        raise ValueError("h must be strictly positive.")
    if not np.isfinite(factor) or factor <= 0.0:
        raise ValueError("factor must be strictly positive.")
    return float(factor * h)
