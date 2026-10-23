"""Wave-physics utilities.

The absorbing-layer helpers depend on FEniCSx. They are imported lazily through
this compatibility block so that NumPy-only wave-physics utilities remain
importable in lightweight environments.
"""

try:  # pragma: no cover - exercised in FEniCSx environments
    from elastic_fwi.wave_physics.absorbing_layers import (
        AbsorbingLayerPlan,
        absorbing_layer_summary_from_function,
        create_absorbing_layer_dg0,
        plan_absorbing_layer,
    )
except Exception:  # pragma: no cover - allows NumPy-only tests without dolfinx
    AbsorbingLayerPlan = None  # type: ignore[assignment]
    absorbing_layer_summary_from_function = None  # type: ignore[assignment]
    create_absorbing_layer_dg0 = None  # type: ignore[assignment]
    plan_absorbing_layer = None  # type: ignore[assignment]

from elastic_fwi.wave_physics.energy_flux import (
    compute_component_energy_flux_proxy,
    compute_directional_energy_proxy,
    summarize_energy_flux_proxy,
)
from elastic_fwi.wave_physics.modal_proxies import (
    apparent_mode_proxy,
    component_dominance_ratio,
    longitudinal_transverse_projection,
    mean_projected_longitudinal_energy,
    mean_projected_transverse_energy,
)
from elastic_fwi.wave_physics.polarization_proxies import (
    compute_hodogram_data,
    compute_orientation_angles,
    compute_orientation_histogram,
    compute_polarization_proxies,
    compute_windowed_polarization_proxies,
)

__all__ = [
    "plan_absorbing_layer",
    "create_absorbing_layer_dg0",
    "absorbing_layer_summary_from_function",
    "AbsorbingLayerPlan",
    "compute_component_energy_flux_proxy",
    "compute_directional_energy_proxy",
    "summarize_energy_flux_proxy",
    "apparent_mode_proxy",
    "component_dominance_ratio",
    "longitudinal_transverse_projection",
    "mean_projected_longitudinal_energy",
    "mean_projected_transverse_energy",
    "compute_hodogram_data",
    "compute_orientation_angles",
    "compute_orientation_histogram",
    "compute_polarization_proxies",
    "compute_windowed_polarization_proxies",
]
