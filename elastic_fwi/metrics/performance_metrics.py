from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PerformanceSummary:
    elapsed_seconds: float
    num_time_steps: int | None = None
    num_dofs: int | None = None
    mpi_size: int = 1
    disk_usage_mb: float | None = None

    @property
    def seconds_per_step(self) -> float | None:
        if self.num_time_steps is None or self.num_time_steps <= 0:
            return None
        return float(self.elapsed_seconds / self.num_time_steps)

    def as_dict(self) -> dict[str, Any]:
        return {
            "elapsed_seconds": float(self.elapsed_seconds),
            "num_time_steps": None if self.num_time_steps is None else int(self.num_time_steps),
            "num_dofs": None if self.num_dofs is None else int(self.num_dofs),
            "mpi_size": int(self.mpi_size),
            "seconds_per_step": self.seconds_per_step,
            "disk_usage_mb": None if self.disk_usage_mb is None else float(self.disk_usage_mb),
        }


def compute_disk_usage_mb(path: str | Path) -> float:
    root = Path(path)
    if not root.exists():
        return 0.0
    if root.is_file():
        return float(root.stat().st_size) / (1024.0 * 1024.0)

    total_bytes = 0
    for item in root.rglob("*"):
        if item.is_file():
            try:
                total_bytes += item.stat().st_size
            except OSError:
                continue

    return float(total_bytes) / (1024.0 * 1024.0)


def _optional_int(value: int | float | str | None) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except Exception:
        return None


def summarize_performance(
    *,
    elapsed_seconds: float,
    num_time_steps: int | None = None,
    num_dofs: int | None = None,
    mpi_size: int = 1,
    disk_usage_mb: float | None = None,
) -> PerformanceSummary:
    if elapsed_seconds < 0.0:
        raise ValueError("elapsed_seconds must be non-negative.")
    if num_time_steps is not None and num_time_steps < 0:
        raise ValueError("num_time_steps must be non-negative when provided.")
    if num_dofs is not None and num_dofs < 0:
        raise ValueError("num_dofs must be non-negative when provided.")
    if mpi_size <= 0:
        raise ValueError("mpi_size must be positive.")
    if disk_usage_mb is not None and disk_usage_mb < 0.0:
        raise ValueError("disk_usage_mb must be non-negative when provided.")

    return PerformanceSummary(
        elapsed_seconds=float(elapsed_seconds),
        num_time_steps=_optional_int(num_time_steps),
        num_dofs=_optional_int(num_dofs),
        mpi_size=int(mpi_size),
        disk_usage_mb=None if disk_usage_mb is None else float(disk_usage_mb),
    )
