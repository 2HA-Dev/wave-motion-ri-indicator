from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_RHO = 4.5
DEFAULT_C11 = 16.24
DEFAULT_C22 = 18.07
DEFAULT_C12 = 6.9
DEFAULT_C66 = 4.67
DEFAULT_CFL_MAX = 2.0


@dataclass(frozen=True)
class SamplingConditionResult:
    kind: str
    status: str
    recommended_status: str
    nx: int
    ny: int
    dt: float
    source_frequency: float
    domain_length_x: float
    domain_length_y: float
    hx: float
    hy: float
    h: float
    rho: float
    C11: float
    C22: float
    C12: float
    C66: float
    c_min_estimate: float
    c_max_estimate: float
    lambda_min: float
    points_per_wavelength: float
    time_steps_per_period: float
    cfl: float
    shannon_min_points_per_wavelength: float
    shannon_min_time_steps_per_period: float
    recommended_min_points_per_wavelength: float
    recommended_min_time_steps_per_period: float
    cfl_max: float
    shannon_spatial_ok: bool
    shannon_temporal_ok: bool
    cfl_ok: bool
    recommended_spatial_ok: bool
    recommended_temporal_ok: bool
    strict_recommended: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _positive_float(value: float, name: str) -> float:
    value = float(value)

    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")

    return value


def _positive_int(value: int, name: str) -> int:
    value = int(value)

    if value <= 0:
        raise ValueError(f"{name} must be positive.")

    return value


def estimate_wave_speeds(
    *,
    rho: float,
    C11: float,
    C22: float,
    C66: float,
) -> tuple[float, float]:
    rho = _positive_float(rho, "rho")
    stiffnesses = np.asarray([C11, C22, C66], dtype=float)

    if not np.all(np.isfinite(stiffnesses)):
        raise ValueError("Elastic stiffnesses must be finite.")

    if np.any(stiffnesses <= 0.0):
        raise ValueError("Elastic stiffnesses used for speed estimates must be positive.")

    c_min = float(math.sqrt(float(np.min(stiffnesses)) / rho))
    c_max = float(math.sqrt(float(np.max(stiffnesses)) / rho))

    return c_min, c_max


def check_sampling_conditions(
    *,
    nx: int,
    ny: int,
    dt: float,
    source_frequency: float,
    domain_length_x: float = 1.0,
    domain_length_y: float = 1.0,
    rho: float = DEFAULT_RHO,
    C11: float = DEFAULT_C11,
    C22: float = DEFAULT_C22,
    C12: float = DEFAULT_C12,
    C66: float = DEFAULT_C66,
    shannon_min_points_per_wavelength: float = 2.0,
    shannon_min_time_steps_per_period: float = 2.0,
    recommended_min_points_per_wavelength: float = 10.0,
    recommended_min_time_steps_per_period: float = 10.0,
    cfl_max: float = DEFAULT_CFL_MAX,
    strict_recommended: bool = False,
) -> SamplingConditionResult:
    nx = _positive_int(nx, "nx")
    ny = _positive_int(ny, "ny")
    dt = _positive_float(dt, "dt")
    source_frequency = _positive_float(source_frequency, "source_frequency")
    domain_length_x = _positive_float(domain_length_x, "domain_length_x")
    domain_length_y = _positive_float(domain_length_y, "domain_length_y")
    rho = _positive_float(rho, "rho")
    C11 = _positive_float(C11, "C11")
    C22 = _positive_float(C22, "C22")
    C12 = float(C12)
    C66 = _positive_float(C66, "C66")

    shannon_min_points_per_wavelength = _positive_float(
        shannon_min_points_per_wavelength,
        "shannon_min_points_per_wavelength",
    )
    shannon_min_time_steps_per_period = _positive_float(
        shannon_min_time_steps_per_period,
        "shannon_min_time_steps_per_period",
    )
    recommended_min_points_per_wavelength = _positive_float(
        recommended_min_points_per_wavelength,
        "recommended_min_points_per_wavelength",
    )
    recommended_min_time_steps_per_period = _positive_float(
        recommended_min_time_steps_per_period,
        "recommended_min_time_steps_per_period",
    )
    cfl_max = _positive_float(cfl_max, "cfl_max")

    hx = domain_length_x / float(nx)
    hy = domain_length_y / float(ny)
    h = max(hx, hy)

    c_min, c_max = estimate_wave_speeds(
        rho=rho,
        C11=C11,
        C22=C22,
        C66=C66,
    )

    lambda_min = c_min / source_frequency
    points_per_wavelength = lambda_min / h
    time_steps_per_period = 1.0 / (source_frequency * dt)
    cfl = c_max * dt / h

    shannon_spatial_ok = points_per_wavelength >= shannon_min_points_per_wavelength
    shannon_temporal_ok = (
        time_steps_per_period >= shannon_min_time_steps_per_period
    )
    cfl_ok = cfl <= cfl_max

    recommended_spatial_ok = (
        points_per_wavelength >= recommended_min_points_per_wavelength
    )
    recommended_temporal_ok = (
        time_steps_per_period >= recommended_min_time_steps_per_period
    )

    status_ok = shannon_spatial_ok and shannon_temporal_ok and cfl_ok
    recommended_ok = recommended_spatial_ok and recommended_temporal_ok

    if strict_recommended:
        status_ok = status_ok and recommended_ok

    return SamplingConditionResult(
        kind="elastic_sampling_conditions",
        status="ok" if status_ok else "failed",
        recommended_status="ok" if recommended_ok else "warning",
        nx=nx,
        ny=ny,
        dt=dt,
        source_frequency=source_frequency,
        domain_length_x=domain_length_x,
        domain_length_y=domain_length_y,
        hx=hx,
        hy=hy,
        h=h,
        rho=rho,
        C11=C11,
        C22=C22,
        C12=C12,
        C66=C66,
        c_min_estimate=c_min,
        c_max_estimate=c_max,
        lambda_min=lambda_min,
        points_per_wavelength=points_per_wavelength,
        time_steps_per_period=time_steps_per_period,
        cfl=cfl,
        shannon_min_points_per_wavelength=shannon_min_points_per_wavelength,
        shannon_min_time_steps_per_period=shannon_min_time_steps_per_period,
        recommended_min_points_per_wavelength=recommended_min_points_per_wavelength,
        recommended_min_time_steps_per_period=recommended_min_time_steps_per_period,
        cfl_max=cfl_max,
        shannon_spatial_ok=shannon_spatial_ok,
        shannon_temporal_ok=shannon_temporal_ok,
        cfl_ok=cfl_ok,
        recommended_spatial_ok=recommended_spatial_ok,
        recommended_temporal_ok=recommended_temporal_ok,
        strict_recommended=bool(strict_recommended),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check spatial/temporal sampling and CFL-like conditions."
    )

    parser.add_argument("--nx", type=int, required=True)
    parser.add_argument("--ny", type=int, required=True)
    parser.add_argument("--dt", type=float, required=True)
    parser.add_argument("--source-frequency", type=float, required=True)

    parser.add_argument("--domain-length-x", type=float, default=1.0)
    parser.add_argument("--domain-length-y", type=float, default=1.0)

    parser.add_argument("--rho", type=float, default=DEFAULT_RHO)
    parser.add_argument("--C11", type=float, default=DEFAULT_C11)
    parser.add_argument("--C22", type=float, default=DEFAULT_C22)
    parser.add_argument("--C12", type=float, default=DEFAULT_C12)
    parser.add_argument("--C66", type=float, default=DEFAULT_C66)

    parser.add_argument("--shannon-min-ppw", type=float, default=2.0)
    parser.add_argument("--shannon-min-time-steps", type=float, default=2.0)
    parser.add_argument("--recommended-min-ppw", type=float, default=10.0)
    parser.add_argument("--recommended-min-time-steps", type=float, default=10.0)
    parser.add_argument("--cfl-max", type=float, default=DEFAULT_CFL_MAX)

    parser.add_argument("--strict-recommended", action="store_true")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--quiet", action="store_true")

    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    result = check_sampling_conditions(
        nx=args.nx,
        ny=args.ny,
        dt=args.dt,
        source_frequency=args.source_frequency,
        domain_length_x=args.domain_length_x,
        domain_length_y=args.domain_length_y,
        rho=args.rho,
        C11=args.C11,
        C22=args.C22,
        C12=args.C12,
        C66=args.C66,
        shannon_min_points_per_wavelength=args.shannon_min_ppw,
        shannon_min_time_steps_per_period=args.shannon_min_time_steps,
        recommended_min_points_per_wavelength=args.recommended_min_ppw,
        recommended_min_time_steps_per_period=args.recommended_min_time_steps,
        cfl_max=args.cfl_max,
        strict_recommended=args.strict_recommended,
    )

    payload = result.to_dict()

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)

    if not args.quiet:
        print("[SAMPLING CONDITIONS]", result.status)
        print(f"  nx, ny                 : {result.nx}, {result.ny}")
        print(f"  h                      : {result.h:.6e}")
        print(f"  c_min_estimate          : {result.c_min_estimate:.6e}")
        print(f"  c_max_estimate          : {result.c_max_estimate:.6e}")
        print(f"  lambda_min              : {result.lambda_min:.6e}")
        print(f"  points_per_wavelength   : {result.points_per_wavelength:.6e}")
        print(f"  time_steps_per_period   : {result.time_steps_per_period:.6e}")
        print(f"  CFL                     : {result.cfl:.6e}")
        print(f"  recommended_status      : {result.recommended_status}")

    return 0 if result.status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
