from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil, isfinite, sqrt
from typing import Any


@dataclass(frozen=True)
class WavelengthMeshSizingResult:
    kind: str
    domain_length_x: float
    domain_length_y: float
    source_frequency: float
    rho: float
    C11: float
    C22: float
    C12: float
    C66: float
    c_min_estimate: float
    c_max_estimate: float
    lambda_min: float
    target_ppw: float
    near_target_ppw: float
    target_h: float
    near_h: float
    nx: int
    ny: int
    mesh_far_h: float
    mesh_near_h: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _positive_float(value: float, name: str) -> float:
    value = float(value)
    if not isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be strictly positive and finite.")
    return value


def _positive_int(value: int, name: str) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be strictly positive.")
    return value


def estimate_elastic_wave_speed_bounds(
    *,
    rho: float,
    C11: float,
    C22: float,
    C66: float,
) -> tuple[float, float]:
    rho = _positive_float(rho, "rho")
    C11 = _positive_float(C11, "C11")
    C22 = _positive_float(C22, "C22")
    C66 = _positive_float(C66, "C66")

    c_min = sqrt(min(C11, C22, C66) / rho)
    c_max = sqrt(max(C11, C22, C66) / rho)

    return float(c_min), float(c_max)


def plan_wavelength_mesh_sizing(
    *,
    domain_length_x: float,
    domain_length_y: float,
    source_frequency: float,
    rho: float,
    C11: float,
    C22: float,
    C12: float,
    C66: float,
    target_ppw: float = 10.0,
    near_target_ppw: float = 20.0,
) -> WavelengthMeshSizingResult:
    domain_length_x = _positive_float(domain_length_x, "domain_length_x")
    domain_length_y = _positive_float(domain_length_y, "domain_length_y")
    source_frequency = _positive_float(source_frequency, "source_frequency")
    target_ppw = _positive_float(target_ppw, "target_ppw")
    near_target_ppw = _positive_float(near_target_ppw, "near_target_ppw")

    if near_target_ppw < target_ppw:
        raise ValueError("near_target_ppw must be greater than or equal to target_ppw.")

    rho = _positive_float(rho, "rho")
    C11 = _positive_float(C11, "C11")
    C22 = _positive_float(C22, "C22")
    C12 = float(C12)
    C66 = _positive_float(C66, "C66")

    c_min, c_max = estimate_elastic_wave_speed_bounds(
        rho=rho,
        C11=C11,
        C22=C22,
        C66=C66,
    )

    lambda_min = c_min / source_frequency
    target_h = lambda_min / target_ppw
    near_h = lambda_min / near_target_ppw

    nx = _positive_int(ceil(domain_length_x / target_h), "nx")
    ny = _positive_int(ceil(domain_length_y / target_h), "ny")

    return WavelengthMeshSizingResult(
        kind="wavelength_mesh_sizing",
        domain_length_x=domain_length_x,
        domain_length_y=domain_length_y,
        source_frequency=source_frequency,
        rho=rho,
        C11=C11,
        C22=C22,
        C12=C12,
        C66=C66,
        c_min_estimate=c_min,
        c_max_estimate=c_max,
        lambda_min=lambda_min,
        target_ppw=target_ppw,
        near_target_ppw=near_target_ppw,
        target_h=target_h,
        near_h=near_h,
        nx=nx,
        ny=ny,
        mesh_far_h=target_h,
        mesh_near_h=near_h,
    )
