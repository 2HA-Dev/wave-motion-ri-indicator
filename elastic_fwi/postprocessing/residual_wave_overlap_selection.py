from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class TemporalOverlapSelection:
    metric_mode: str
    activity_threshold: float
    maximum_total_energy: float
    overlap_selection: dict[str, Any]
    localization_selection: dict[str, Any]
    closest_peak_selection: dict[str, Any]
    num_snapshots: int
    num_active_snapshots: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _finite_float(
    row: dict[str, Any],
    key: str,
) -> float:
    try:
        value = float(row[key])
    except KeyError as exc:
        raise ValueError(
            f"Missing metric field {key!r}."
        ) from exc
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Metric field {key!r} is not numeric."
        ) from exc

    if not np.isfinite(value):
        raise ValueError(
            f"Metric field {key!r} is not finite."
        )

    return value


def _nonnegative_float(
    row: dict[str, Any],
    key: str,
) -> float:
    value = _finite_float(row, key)

    if value < 0.0:
        raise ValueError(
            f"Metric field {key!r} must be non-negative."
        )

    return value


def _positive_float(
    row: dict[str, Any],
    key: str,
) -> float:
    value = _finite_float(row, key)

    if value <= 0.0:
        raise ValueError(
            f"Metric field {key!r} must be strictly positive."
        )

    return value


def _safe_ratio(
    numerator: float,
    denominator: float,
) -> float:
    if denominator <= 0.0:
        return 0.0

    value = numerator / denominator

    if not np.isfinite(value):
        raise RuntimeError(
            "Computed a non-finite ratio."
        )

    return float(value)


def enrich_metric_row(
    row: dict[str, Any],
    *,
    maximum_total_energy: float,
    consistency_rtol: float = 1.0e-12,
    consistency_atol: float = 1.0e-15,
) -> dict[str, Any]:
    if not np.isfinite(maximum_total_energy):
        raise ValueError(
            "maximum_total_energy must be finite."
        )

    if maximum_total_energy <= 0.0:
        raise ValueError(
            "maximum_total_energy must be strictly positive."
        )

    area_total = _positive_float(
        row,
        "area_total",
    )

    area_defect = _nonnegative_float(
        row,
        "area_defect",
    )

    area_ring = _nonnegative_float(
        row,
        "area_ring",
    )

    area_outside = _nonnegative_float(
        row,
        "area_outside",
    )

    energy_total = _nonnegative_float(
        row,
        "energy_total",
    )

    energy_defect = _nonnegative_float(
        row,
        "energy_defect",
    )

    energy_ring = _nonnegative_float(
        row,
        "energy_ring",
    )

    energy_outside = _nonnegative_float(
        row,
        "energy_outside",
    )

    area_sum = (
        area_defect
        + area_ring
        + area_outside
    )

    if not np.isclose(
        area_sum,
        area_total,
        rtol=consistency_rtol,
        atol=consistency_atol,
    ):
        raise ValueError(
            "Defect, ring and outside areas do not sum "
            "to the total area."
        )

    energy_sum = (
        energy_defect
        + energy_ring
        + energy_outside
    )

    if not np.isclose(
        energy_sum,
        energy_total,
        rtol=consistency_rtol,
        atol=consistency_atol,
    ):
        raise ValueError(
            "Defect, ring and outside energies do not sum "
            "to the total energy."
        )

    area_defect_ring = (
        area_defect
        + area_ring
    )

    energy_defect_ring = (
        energy_defect
        + energy_ring
    )

    area_fraction_defect = (
        area_defect / area_total
    )

    area_fraction_defect_ring = (
        area_defect_ring / area_total
    )

    energy_fraction_defect = _safe_ratio(
        energy_defect,
        energy_total,
    )

    energy_fraction_defect_ring = (
        _safe_ratio(
            energy_defect_ring,
            energy_total,
        )
    )

    mean_density_global = _safe_ratio(
        energy_total,
        area_total,
    )

    mean_density_defect = _safe_ratio(
        energy_defect,
        area_defect,
    )

    mean_density_defect_ring = (
        _safe_ratio(
            energy_defect_ring,
            area_defect_ring,
        )
    )

    mean_density_outside = _safe_ratio(
        energy_outside,
        area_outside,
    )

    defect_enrichment = _safe_ratio(
        energy_fraction_defect,
        area_fraction_defect,
    )

    defect_ring_enrichment = _safe_ratio(
        energy_fraction_defect_ring,
        area_fraction_defect_ring,
    )

    defect_to_outside = _safe_ratio(
        mean_density_defect,
        mean_density_outside,
    )

    defect_ring_to_outside = _safe_ratio(
        mean_density_defect_ring,
        mean_density_outside,
    )

    existing_fraction = row.get(
        "energy_fraction_defect"
    )

    if existing_fraction is not None:
        if not np.isclose(
            float(existing_fraction),
            energy_fraction_defect,
            rtol=consistency_rtol,
            atol=consistency_atol,
        ):
            raise ValueError(
                "Stored defect-energy fraction is inconsistent "
                "with the raw energies."
            )

    existing_combined_fraction = row.get(
        "energy_fraction_defect_and_ring"
    )

    if existing_combined_fraction is not None:
        if not np.isclose(
            float(existing_combined_fraction),
            energy_fraction_defect_ring,
            rtol=consistency_rtol,
            atol=consistency_atol,
        ):
            raise ValueError(
                "Stored defect-plus-ring fraction is "
                "inconsistent with the raw energies."
            )

    enriched = dict(row)

    enriched.update(
        {
            "area_defect_and_ring": (
                area_defect_ring
            ),
            "energy_defect_and_ring": (
                energy_defect_ring
            ),
            "area_fraction_defect": (
                area_fraction_defect
            ),
            "area_fraction_defect_and_ring": (
                area_fraction_defect_ring
            ),
            "relative_total_energy": (
                energy_total
                / maximum_total_energy
            ),
            "mean_interaction_density_global": (
                mean_density_global
            ),
            "mean_interaction_density_defect": (
                mean_density_defect
            ),
            "mean_interaction_density_defect_and_ring": (
                mean_density_defect_ring
            ),
            "mean_interaction_density_outside": (
                mean_density_outside
            ),
            "defect_energy_enrichment": (
                defect_enrichment
            ),
            "defect_ring_energy_enrichment": (
                defect_ring_enrichment
            ),
            "defect_to_outside_density_ratio": (
                defect_to_outside
            ),
            "defect_ring_to_outside_density_ratio": (
                defect_ring_to_outside
            ),
        }
    )

    for key, value in enriched.items():
        if isinstance(value, float):
            if not np.isfinite(value):
                raise RuntimeError(
                    f"Enriched metric {key!r} is not finite."
                )

    return enriched


def _selection_projection(
    row: dict[str, Any],
) -> dict[str, Any]:
    keys = (
        "time_index",
        "time_value",
        "metric_mode",
        "energy_total",
        "relative_total_energy",
        "energy_defect",
        "energy_defect_and_ring",
        "energy_fraction_defect",
        "energy_fraction_defect_and_ring",
        "area_fraction_defect",
        "area_fraction_defect_and_ring",
        "defect_energy_enrichment",
        "defect_ring_energy_enrichment",
        "defect_to_outside_density_ratio",
        "defect_ring_to_outside_density_ratio",
        "distance_peak_to_defect_center",
        "distance_centroid_to_defect_center",
        "global_max_inside_defect",
        "global_max_inside_ring",
        "peak_to_artifact_ratio",
        "peak_defined",
    )

    return {
        key: row[key]
        for key in keys
    }


def select_temporal_overlap_states(
    rows: Iterable[dict[str, Any]],
    *,
    metric_mode: str,
    activity_threshold: float = 0.10,
) -> tuple[
    TemporalOverlapSelection,
    list[dict[str, Any]],
]:
    if not np.isfinite(activity_threshold):
        raise ValueError(
            "activity_threshold must be finite."
        )

    if not 0.0 < activity_threshold <= 1.0:
        raise ValueError(
            "activity_threshold must lie in (0, 1]."
        )

    mode_rows = [
        dict(row)
        for row in rows
        if str(row.get("metric_mode"))
        == str(metric_mode)
    ]

    if not mode_rows:
        raise ValueError(
            f"No metric rows found for {metric_mode!r}."
        )

    mode_rows.sort(
        key=lambda row: (
            _finite_float(
                row,
                "time_value",
            ),
            int(row["time_index"]),
        )
    )

    indices = [
        int(row["time_index"])
        for row in mode_rows
    ]

    if len(indices) != len(set(indices)):
        raise ValueError(
            "Metric rows contain duplicated time indices."
        )

    maximum_total_energy = max(
        _nonnegative_float(
            row,
            "energy_total",
        )
        for row in mode_rows
    )

    if maximum_total_energy <= 0.0:
        raise ValueError(
            "All total interaction energies are zero."
        )

    enriched_rows = [
        enrich_metric_row(
            row,
            maximum_total_energy=(
                maximum_total_energy
            ),
        )
        for row in mode_rows
    ]

    overlap = max(
        enriched_rows,
        key=lambda row: (
            float(row["energy_total"]),
            -int(row["time_index"]),
        ),
    )

    active_rows = [
        row
        for row in enriched_rows
        if (
            float(
                row["relative_total_energy"]
            )
            >= activity_threshold
            and bool(row["peak_defined"])
        )
    ]

    if not active_rows:
        raise ValueError(
            "No active snapshot satisfies the temporal "
            "selection criterion."
        )

    localization = max(
        active_rows,
        key=lambda row: (
            float(
                row[
                    "defect_ring_energy_enrichment"
                ]
            ),
            -float(
                row[
                    "distance_peak_to_defect_center"
                ]
            ),
            float(row["energy_total"]),
            -int(row["time_index"]),
        ),
    )

    closest_peak = min(
        active_rows,
        key=lambda row: (
            float(
                row[
                    "distance_peak_to_defect_center"
                ]
            ),
            -float(
                row[
                    "defect_ring_energy_enrichment"
                ]
            ),
            -float(row["energy_total"]),
            int(row["time_index"]),
        ),
    )

    selection = TemporalOverlapSelection(
        metric_mode=str(metric_mode),
        activity_threshold=float(
            activity_threshold
        ),
        maximum_total_energy=float(
            maximum_total_energy
        ),
        overlap_selection=(
            _selection_projection(overlap)
        ),
        localization_selection=(
            _selection_projection(
                localization
            )
        ),
        closest_peak_selection=(
            _selection_projection(
                closest_peak
            )
        ),
        num_snapshots=len(enriched_rows),
        num_active_snapshots=len(active_rows),
    )

    return selection, enriched_rows


def audit_activity_thresholds(
    rows: Iterable[dict[str, Any]],
    *,
    metric_mode: str,
    thresholds: Iterable[float] = (
        0.01,
        0.05,
        0.10,
    ),
) -> dict[str, Any]:
    rows_materialized = [
        dict(row)
        for row in rows
    ]

    results: dict[str, Any] = {}
    localization_times: set[float] = set()
    closest_peak_times: set[float] = set()

    for threshold in thresholds:
        selection, _ = (
            select_temporal_overlap_states(
                rows_materialized,
                metric_mode=metric_mode,
                activity_threshold=float(
                    threshold
                ),
            )
        )

        result = selection.as_dict()

        results[
            f"{float(threshold):.6g}"
        ] = result

        localization_times.add(
            float(
                selection.localization_selection[
                    "time_value"
                ]
            )
        )

        closest_peak_times.add(
            float(
                selection.closest_peak_selection[
                    "time_value"
                ]
            )
        )

    return {
        "metric_mode": str(metric_mode),
        "threshold_results": results,
        "localization_times": sorted(
            localization_times
        ),
        "localization_time_stable": (
            len(localization_times) == 1
        ),
        "closest_peak_times": sorted(
            closest_peak_times
        ),
        "closest_peak_time_stable": (
            len(closest_peak_times) == 1
        ),
    }
