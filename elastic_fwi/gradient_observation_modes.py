from __future__ import annotations

"""Observation-mode conventions for finite-difference gradient validation."""

from dataclasses import dataclass
from typing import Any, Mapping


TRUE_GRADIENT_OBSERVATION_MODES = (
    "classical_full",
    "standard_reflection_only",
    "standard_transmission_only",
)
RESIDUAL_INTERACTION_MODE = "residual_interaction"


@dataclass(frozen=True)
class GradientObservationMode:
    """Weights and scalar functional associated with a true gradient export."""

    name: str
    reflection_weight: float
    transmission_weight: float
    functional_key: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "reflection_weight": float(self.reflection_weight),
            "transmission_weight": float(self.transmission_weight),
            "functional_key": self.functional_key,
        }


_MODES = {
    "classical_full": GradientObservationMode(
        name="classical_full",
        reflection_weight=1.0,
        transmission_weight=1.0,
        functional_key="full",
    ),
    "standard_reflection_only": GradientObservationMode(
        name="standard_reflection_only",
        reflection_weight=1.0,
        transmission_weight=0.0,
        functional_key="reflection",
    ),
    "standard_transmission_only": GradientObservationMode(
        name="standard_transmission_only",
        reflection_weight=0.0,
        transmission_weight=1.0,
        functional_key="transmission",
    ),
}


def resolve_gradient_observation_mode(name: str) -> GradientObservationMode:
    """Resolve a true-gradient observation mode and reject interaction kernels."""

    normalized = str(name).strip().lower()
    if normalized == RESIDUAL_INTERACTION_MODE:
        raise ValueError(
            "residual_interaction is not a demonstrated derivative of a scalar "
            "functional and must not be subjected to a Taylor gradient test."
        )
    try:
        return _MODES[normalized]
    except KeyError as exc:
        raise ValueError(
            f"Unknown observation mode {name!r}; expected one of "
            f"{list(TRUE_GRADIENT_OBSERVATION_MODES)}."
        ) from exc


def select_observation_functional_value(
    values: Mapping[str, Any] | Any,
    mode: str | GradientObservationMode,
) -> float:
    """Return the scalar cost differentiated by the selected gradient mode."""

    resolved = (
        mode if isinstance(mode, GradientObservationMode)
        else resolve_gradient_observation_mode(mode)
    )
    if isinstance(values, Mapping):
        value = values[resolved.functional_key]
    else:
        value = getattr(values, resolved.functional_key)
    return float(value)
