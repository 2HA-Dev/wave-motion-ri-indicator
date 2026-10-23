from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class OutputConfig:
    output_dir: Path
    clean: bool = False
    reuse_existing: bool = False
    write_figures: bool = True
    figure_format: str = "png"


@dataclass(frozen=True)
class RunPreset:
    name: str
    description: str
    quick: bool
    allow_heavy_required: bool


@dataclass(frozen=True)
class CampaignPreset:
    name: str
    domain_length: float
    nx: int
    ny: int
    num_steps: int
    every_k: int
    dt: float
    source_frequency: float
    defect_radius: float
    observation_kind: str = "vector_array"
