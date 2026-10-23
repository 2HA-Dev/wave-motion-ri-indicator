from __future__ import annotations

from copy import deepcopy

import pytest

from elastic_fwi.postprocessing.scattered_wavefield_reconstruction import (
    validate_pre_interaction,
)


def _row(
    time: float,
    *,
    l2_scattered: float,
    max_norm_scattered: float,
    l2_healthy: float = 2.8e-2,
    max_norm_healthy: float = 6.2e-3,
    area: float = 5329.0,
) -> dict[str, float]:
    return {
        "time": time,
        "l2_scattered": l2_scattered,
        "l2_healthy": l2_healthy,
        "relative_l2": (
            l2_scattered / l2_healthy
            if l2_healthy > 0.0
            else 0.0
        ),
        "max_norm_scattered": max_norm_scattered,
        "max_norm_healthy": max_norm_healthy,
        "domain_area_integrated": area,
    }


def _campaign_prefix() -> list[dict[str, float]]:
    return [
        _row(
            0.0,
            l2_scattered=2.1314595452029295e-14,
            max_norm_scattered=5.91320875182043e-14,
            l2_healthy=1.443629631488905e-5,
            max_norm_healthy=5.23420948173035e-6,
        ),
        _row(
            0.5,
            l2_scattered=3.986901109118238e-13,
            max_norm_scattered=8.585327846010812e-13,
        ),
        _row(
            1.0,
            l2_scattered=1.0270317834095223e-12,
            max_norm_scattered=9.785794081522752e-13,
        ),
        _row(
            1.5,
            l2_scattered=1.5736877418403033e-12,
            max_norm_scattered=7.008893031377962e-13,
        ),
        _row(
            2.0,
            l2_scattered=2.2459687195846283e-12,
            max_norm_scattered=4.993034926475207e-12,
        ),
        _row(
            2.5,
            l2_scattered=1.8639445146990903e-8,
            max_norm_scattered=1.0056200892069752e-7,
        ),
        _row(
            3.0,
            l2_scattered=1.4085175426124974e-5,
            max_norm_scattered=5.679178932658581e-5,
        ),
    ]


def test_transition_snapshot_is_not_strict_preinteraction() -> None:
    rows = deepcopy(_campaign_prefix())

    result = validate_pre_interaction(
        rows,
        arrival_time=3.0,
        atol=1.0e-14,
        rtol=1.0e-10,
    )

    assert result["num_pre_arrival_candidates"] == 6
    assert result["num_pre_interaction_snapshots"] == 5
    assert result["num_transition_snapshots"] == 1
    assert result["transition_times"] == [2.5]
    assert result["strict_preinteraction_cutoff_time"] == 2.5

    assert rows[4]["pre_interaction"] is True
    assert rows[5]["pre_interaction"] is False
    assert rows[5]["pre_interaction_transition"] is True
    assert rows[5]["pre_interaction_pass"] is None
    assert rows[6]["pre_interaction"] is False
    assert rows[6]["pre_interaction_transition"] is False


def test_campaign_core_energy_passes_with_pointwise_warning() -> None:
    rows = deepcopy(_campaign_prefix())

    result = validate_pre_interaction(
        rows,
        arrival_time=3.0,
        atol=1.0e-14,
        rtol=1.0e-10,
    )

    assert result["status"] == "WARNING"
    assert result["reason"] == (
        "strict_preinteraction_energy_pass_"
        "pointwise_tolerance_exceeded"
    )

    assert result["num_l2_pass"] == 5
    assert result["num_l2_fail"] == 0
    assert result["num_pass"] == 5
    assert result["num_fail"] == 0
    assert result["num_pointwise_fail"] > 0

    assert all(
        row["pre_interaction_l2_pass"] is True
        for row in rows[:5]
    )


def test_strict_preinteraction_energy_failure_remains_fail() -> None:
    rows = [
        _row(
            0.0,
            l2_scattered=0.0,
            max_norm_scattered=0.0,
        ),
        _row(
            0.5,
            l2_scattered=1.0e-3,
            max_norm_scattered=1.0e-3,
        ),
        _row(
            1.0,
            l2_scattered=1.0e-2,
            max_norm_scattered=1.0e-2,
        ),
    ]

    result = validate_pre_interaction(
        rows,
        arrival_time=1.5,
        atol=1.0e-14,
        rtol=1.0e-10,
    )

    # t=1 is the transition sample; t=0.5 remains in the strict core.
    assert result["transition_times"] == [1.0]
    assert result["status"] == "FAIL"
    assert result["num_l2_fail"] == 1


def test_zero_background_passes_both_checks() -> None:
    rows = [
        _row(
            0.0,
            l2_scattered=0.0,
            max_norm_scattered=0.0,
        ),
        _row(
            0.5,
            l2_scattered=0.0,
            max_norm_scattered=0.0,
        ),
        _row(
            1.0,
            l2_scattered=1.0,
            max_norm_scattered=1.0,
        ),
    ]

    result = validate_pre_interaction(
        rows,
        arrival_time=1.0,
        atol=1.0e-14,
        rtol=1.0e-10,
    )

    assert result["status"] == "PASS"
    assert result["num_pre_interaction_snapshots"] == 1
    assert result["transition_times"] == [0.5]
    assert result["num_l2_fail"] == 0
    assert result["num_pointwise_fail"] == 0


def test_unavailable_arrival_is_warning() -> None:
    rows = deepcopy(_campaign_prefix())

    result = validate_pre_interaction(
        rows,
        arrival_time=None,
        atol=1.0e-14,
        rtol=1.0e-10,
    )

    assert result["status"] == "WARNING"
    assert result["reason"] == "arrival_time_unavailable"
    assert result["num_pre_interaction_snapshots"] == 0
    assert result["num_transition_snapshots"] == 0


@pytest.mark.parametrize(
    "invalid",
    [-1, 1.5, True],
)
def test_invalid_transition_guard_is_rejected(invalid) -> None:
    with pytest.raises(
        ValueError,
        match="non-negative integer",
    ):
        validate_pre_interaction(
            deepcopy(_campaign_prefix()),
            arrival_time=3.0,
            atol=1.0e-14,
            rtol=1.0e-10,
            transition_guard_snapshots=invalid,
        )


def test_zero_transition_guard_preserves_all_prearrival_samples() -> None:
    rows = deepcopy(_campaign_prefix())

    result = validate_pre_interaction(
        rows,
        arrival_time=3.0,
        atol=1.0e-14,
        rtol=1.0e-10,
        transition_guard_snapshots=0,
    )

    assert result["num_pre_interaction_snapshots"] == 6
    assert result["num_transition_snapshots"] == 0
    assert result["status"] == "FAIL"
    assert result["num_l2_fail"] == 1


def test_single_prearrival_sample_is_never_consumed_by_guard() -> None:
    rows = [
        _row(
            0.0,
            l2_scattered=1.0e-3,
            max_norm_scattered=1.0e-3,
        ),
        _row(
            1.0,
            l2_scattered=1.0,
            max_norm_scattered=1.0,
        ),
    ]

    result = validate_pre_interaction(
        rows,
        arrival_time=1.0,
        atol=1.0e-14,
        rtol=1.0e-10,
    )

    assert result["num_pre_arrival_candidates"] == 1
    assert result["num_pre_interaction_snapshots"] == 1
    assert result["num_transition_snapshots"] == 0
    assert result["transition_guard_snapshots_applied"] == 0
    assert result["status"] == "FAIL"

    assert rows[0]["pre_interaction"] is True
    assert rows[0]["pre_interaction_transition"] is False
