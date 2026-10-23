from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any

import numpy as np

from elastic_fwi.geometry.boundary_tags import BoundarySegment2D


_ALLOWED_SIDES = {"bottom", "top", "left", "right"}


def _as_float(value: Any, name: str) -> float:
    out = float(value)
    if not isfinite(out):
        raise ValueError(f"{name} must be finite.")
    return out


def _as_positive_float(value: Any, name: str) -> float:
    out = _as_float(value, name)
    if out <= 0.0:
        raise ValueError(f"{name} must be strictly positive.")
    return out


def _validate_side(side: str) -> str:
    side = str(side)
    if side not in _ALLOWED_SIDES:
        raise ValueError(f"side must be one of {sorted(_ALLOWED_SIDES)}.")
    return side


def _boundary_axes_and_limits(
    *,
    side: str,
    domain_length_x: float,
    domain_length_y: float,
) -> tuple[int, float, int, float, float]:
    side = _validate_side(side)
    Lx = _as_positive_float(domain_length_x, "domain_length_x")
    Ly = _as_positive_float(domain_length_y, "domain_length_y")

    if side == "bottom":
        return 1, 0.0, 0, 0.0, Lx

    if side == "top":
        return 1, Ly, 0, 0.0, Lx

    if side == "left":
        return 0, 0.0, 1, 0.0, Ly

    if side == "right":
        return 0, Lx, 1, 0.0, Ly

    raise ValueError(f"Unsupported side: {side}")


@dataclass(frozen=True)
class BoundarySegmentSource:
    """
    Continuous boundary segment source or receiver.

    The segment is centered along one boundary side and has a finite length.
    """

    name: str
    side: str
    center: float
    length: float
    tag: int
    tol: float = 1.0e-10

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("BoundarySegmentSource name must be non-empty.")

        side = _validate_side(self.side)
        center = _as_float(self.center, "center")
        length = _as_positive_float(self.length, "length")
        tol = float(self.tol)

        if self.tag <= 0:
            raise ValueError("tag must be a strictly positive integer.")

        if not isfinite(tol) or tol < 0.0:
            raise ValueError("tol must be finite and non-negative.")

        object.__setattr__(self, "side", side)
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "length", length)
        object.__setattr__(self, "tol", tol)

    def boundary_specs(
        self,
        *,
        domain_length_x: float,
        domain_length_y: float,
    ) -> tuple[BoundarySegment2D, ...]:
        fixed_axis, fixed_value, interval_axis, interval_min_domain, interval_max_domain = (
            _boundary_axes_and_limits(
                side=self.side,
                domain_length_x=domain_length_x,
                domain_length_y=domain_length_y,
            )
        )

        a = self.center - 0.5 * self.length
        b = self.center + 0.5 * self.length

        if a < interval_min_domain or b > interval_max_domain:
            raise ValueError(
                f"{self.name} segment [{a}, {b}] is outside boundary interval "
                f"[{interval_min_domain}, {interval_max_domain}]."
            )

        return (
            BoundarySegment2D(
                name=self.name,
                tag=self.tag,
                fixed_axis=fixed_axis,
                fixed_value=fixed_value,
                interval_axis=interval_axis,
                interval_min=a,
                interval_max=b,
                tol=self.tol,
            ),
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "kind": "boundary_segment",
            "name": self.name,
            "side": self.side,
            "center": self.center,
            "length": self.length,
            "tag": self.tag,
            "tol": self.tol,
        }


@dataclass(frozen=True)
class BoundaryPointArraySource:
    """
    Boundary point-array source or receiver.

    Each point is represented by a small boundary patch. All patches share the
    same physical tag, so existing boundary observation/source operators work on
    the union of all elements.
    """

    name: str
    side: str
    start: float
    end: float
    num_elements: int
    element_length: float
    tag: int
    tol: float = 1.0e-10

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("BoundaryPointArraySource name must be non-empty.")

        side = _validate_side(self.side)
        start = _as_float(self.start, "start")
        end = _as_float(self.end, "end")
        element_length = _as_positive_float(self.element_length, "element_length")
        num_elements = int(self.num_elements)
        tol = float(self.tol)

        if num_elements <= 0:
            raise ValueError("num_elements must be strictly positive.")

        if end < start:
            raise ValueError("end must be >= start.")

        if self.tag <= 0:
            raise ValueError("tag must be a strictly positive integer.")

        if not isfinite(tol) or tol < 0.0:
            raise ValueError("tol must be finite and non-negative.")

        object.__setattr__(self, "side", side)
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)
        object.__setattr__(self, "num_elements", num_elements)
        object.__setattr__(self, "element_length", element_length)
        object.__setattr__(self, "tol", tol)

    @property
    def centers(self) -> tuple[float, ...]:
        if self.num_elements == 1:
            return (0.5 * (self.start + self.end),)

        return tuple(
            float(value)
            for value in np.linspace(self.start, self.end, self.num_elements)
        )

    def boundary_specs(
        self,
        *,
        domain_length_x: float,
        domain_length_y: float,
    ) -> tuple[BoundarySegment2D, ...]:
        fixed_axis, fixed_value, interval_axis, interval_min_domain, interval_max_domain = (
            _boundary_axes_and_limits(
                side=self.side,
                domain_length_x=domain_length_x,
                domain_length_y=domain_length_y,
            )
        )

        specs: list[BoundarySegment2D] = []

        for index, center in enumerate(self.centers):
            a = center - 0.5 * self.element_length
            b = center + 0.5 * self.element_length

            if a < interval_min_domain or b > interval_max_domain:
                raise ValueError(
                    f"{self.name}[{index}] patch [{a}, {b}] is outside boundary interval "
                    f"[{interval_min_domain}, {interval_max_domain}]."
                )

            specs.append(
                BoundarySegment2D(
                    name=f"{self.name}_{index:04d}",
                    tag=self.tag,
                    fixed_axis=fixed_axis,
                    fixed_value=fixed_value,
                    interval_axis=interval_axis,
                    interval_min=a,
                    interval_max=b,
                    tol=self.tol,
                )
            )

        return tuple(specs)

    def metadata(self) -> dict[str, Any]:
        return {
            "kind": "boundary_point_array",
            "name": self.name,
            "side": self.side,
            "start": self.start,
            "end": self.end,
            "num_elements": self.num_elements,
            "element_length": self.element_length,
            "centers": list(self.centers),
            "tag": self.tag,
            "tol": self.tol,
        }


def lateral_boundary_specs(
    *,
    domain_length_x: float,
    domain_length_y: float,
    left_tag: int = 3,
    right_tag: int = 4,
    tol: float = 1.0e-10,
) -> tuple[BoundarySegment2D, ...]:
    Lx = _as_positive_float(domain_length_x, "domain_length_x")
    Ly = _as_positive_float(domain_length_y, "domain_length_y")

    return (
        BoundarySegment2D(
            name="left_lateral",
            tag=left_tag,
            fixed_axis=0,
            fixed_value=0.0,
            interval_axis=1,
            interval_min=0.0,
            interval_max=Ly,
            tol=tol,
        ),
        BoundarySegment2D(
            name="right_lateral",
            tag=right_tag,
            fixed_axis=0,
            fixed_value=Lx,
            interval_axis=1,
            interval_min=0.0,
            interval_max=Ly,
            tol=tol,
        ),
    )


def default_element_length_for_side(
    *,
    side: str,
    domain_length_x: float,
    domain_length_y: float,
    nx: int,
    ny: int,
) -> float:
    side = _validate_side(side)

    # Boundary point arrays are represented by small boundary patches.
    # With DOLFINx facet marking, a patch of length h centered on a mesh node may
    # contain no full boundary facet. A default length 2h is therefore safer.
    if side in {"bottom", "top"}:
        return 2.0 * float(domain_length_x) / float(nx)

    return 2.0 * float(domain_length_y) / float(ny)


def build_boundary_entity(
    *,
    kind: str,
    name: str,
    side: str,
    tag: int,
    center: float | None,
    length: float | None,
    array_start: float | None,
    array_end: float | None,
    array_num_elements: int,
    array_element_length: float | None,
    fallback_interval_min: float,
    fallback_interval_max: float,
    default_element_length: float,
) -> BoundarySegmentSource | BoundaryPointArraySource:
    kind = str(kind)

    if kind == "boundary_segment":
        if center is None:
            center = 0.5 * (float(fallback_interval_min) + float(fallback_interval_max))
        if length is None:
            length = float(fallback_interval_max) - float(fallback_interval_min)

        return BoundarySegmentSource(
            name=name,
            side=side,
            center=float(center),
            length=float(length),
            tag=tag,
        )

    if kind == "boundary_point_array":
        if array_start is None:
            array_start = float(fallback_interval_min)
        if array_end is None:
            array_end = float(fallback_interval_max)
        if array_element_length is None:
            array_element_length = float(default_element_length)

        return BoundaryPointArraySource(
            name=name,
            side=side,
            start=float(array_start),
            end=float(array_end),
            num_elements=int(array_num_elements),
            element_length=float(array_element_length),
            tag=tag,
        )

    raise ValueError("kind must be one of: boundary_segment, boundary_point_array.")
