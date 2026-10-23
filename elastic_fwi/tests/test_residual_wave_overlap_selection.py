from __future__ import annotations

import copy

import pytest

from elastic_fwi.postprocessing.residual_wave_overlap_selection import (
    audit_activity_thresholds,
    enrich_metric_row,
    select_temporal_overlap_states,
)


def metric_row(
    *,
    index: int,
    time: float,
    total: float,
    defect: float,
    ring: float,
    peak_distance: float,
    peak_defined: bool = True,
    mode: str = "norm_product",
) -> dict:
    outside = total - defect - ring

    return {
        "time_index": index,
        "time_value": time,
        "metric_mode": mode,
        "area_total": 100.0,
        "area_defect": 1.0,
        "area_ring": 3.0,
        "area_outside": 96.0,
        "energy_total": total,
        "energy_defect": defect,
        "energy_ring": ring,
        "energy_outside": outside,
        "energy_fraction_defect": (
            defect / total
            if total > 0.0
            else 0.0
        ),
        "energy_fraction_defect_and_ring": (
            (defect + ring) / total
            if total > 0.0
            else 0.0
        ),
        "distance_peak_to_defect_center": (
            peak_distance
        ),
        "distance_centroid_to_defect_center": 1.0,
        "global_max_inside_defect": (
            peak_distance <= 0.25
        ),
        "global_max_inside_ring": (
            0.25 < peak_distance <= 0.5
        ),
        "peak_to_artifact_ratio": 1.0,
        "peak_defined": peak_defined,
    }


def test_uniform_distribution_has_unit_enrichment():
    row = metric_row(
        index=0,
        time=0.0,
        total=200.0,
        defect=2.0,
        ring=6.0,
        peak_distance=0.0,
    )

    enriched = enrich_metric_row(
        row,
        maximum_total_energy=200.0,
    )

    assert (
        enriched["defect_energy_enrichment"]
        == pytest.approx(1.0)
    )

    assert (
        enriched[
            "defect_ring_energy_enrichment"
        ]
        == pytest.approx(1.0)
    )

    assert (
        enriched[
            "defect_to_outside_density_ratio"
        ]
        == pytest.approx(1.0)
    )


def test_overlap_and_localization_selections_differ():
    rows = [
        metric_row(
            index=0,
            time=0.0,
            total=100.0,
            defect=1.0,
            ring=3.0,
            peak_distance=2.0,
        ),
        metric_row(
            index=1,
            time=1.0,
            total=90.0,
            defect=4.0,
            ring=14.0,
            peak_distance=0.1,
        ),
        metric_row(
            index=2,
            time=2.0,
            total=1.0,
            defect=0.2,
            ring=0.4,
            peak_distance=0.0,
        ),
    ]

    selection, enriched = (
        select_temporal_overlap_states(
            rows,
            metric_mode="norm_product",
            activity_threshold=0.10,
        )
    )

    assert len(enriched) == 3

    assert (
        selection.overlap_selection[
            "time_value"
        ]
        == 0.0
    )

    assert (
        selection.localization_selection[
            "time_value"
        ]
        == 1.0
    )

    assert selection.num_active_snapshots == 2


def test_inactive_high_enrichment_is_excluded():
    rows = [
        metric_row(
            index=0,
            time=0.0,
            total=100.0,
            defect=2.0,
            ring=6.0,
            peak_distance=1.0,
        ),
        metric_row(
            index=1,
            time=1.0,
            total=20.0,
            defect=2.0,
            ring=4.0,
            peak_distance=0.2,
        ),
        metric_row(
            index=2,
            time=2.0,
            total=0.5,
            defect=0.2,
            ring=0.2,
            peak_distance=0.0,
        ),
    ]

    selection, _ = (
        select_temporal_overlap_states(
            rows,
            metric_mode="norm_product",
            activity_threshold=0.10,
        )
    )

    assert (
        selection.localization_selection[
            "time_value"
        ]
        == 1.0
    )


def test_localization_tie_breaks_on_peak_distance():
    rows = [
        metric_row(
            index=0,
            time=0.0,
            total=100.0,
            defect=4.0,
            ring=12.0,
            peak_distance=0.4,
        ),
        metric_row(
            index=1,
            time=1.0,
            total=100.0,
            defect=4.0,
            ring=12.0,
            peak_distance=0.1,
        ),
    ]

    selection, _ = (
        select_temporal_overlap_states(
            rows,
            metric_mode="norm_product",
            activity_threshold=0.10,
        )
    )

    assert (
        selection.localization_selection[
            "time_value"
        ]
        == 1.0
    )


def test_inconsistent_stored_fraction_is_rejected():
    row = metric_row(
        index=0,
        time=0.0,
        total=100.0,
        defect=1.0,
        ring=3.0,
        peak_distance=1.0,
    )

    row["energy_fraction_defect"] = 0.5

    with pytest.raises(
        ValueError,
        match="inconsistent",
    ):
        enrich_metric_row(
            row,
            maximum_total_energy=100.0,
        )


def test_activity_threshold_audit_reports_stability():
    rows = [
        metric_row(
            index=0,
            time=0.0,
            total=100.0,
            defect=1.0,
            ring=3.0,
            peak_distance=2.0,
        ),
        metric_row(
            index=1,
            time=1.0,
            total=60.0,
            defect=5.0,
            ring=15.0,
            peak_distance=0.1,
        ),
    ]

    audit = audit_activity_thresholds(
        rows,
        metric_mode="norm_product",
        thresholds=(0.01, 0.05, 0.10),
    )

    assert (
        audit["localization_time_stable"]
        is True
    )

    assert audit["localization_times"] == [1.0]


def test_input_rows_are_not_mutated():
    rows = [
        metric_row(
            index=0,
            time=0.0,
            total=100.0,
            defect=1.0,
            ring=3.0,
            peak_distance=1.0,
        )
    ]

    reference = copy.deepcopy(rows)

    select_temporal_overlap_states(
        rows,
        metric_mode="norm_product",
        activity_threshold=0.10,
    )

    assert rows == reference
