from __future__ import annotations

from elastic_fwi.diagnostics.export_inverse_time_series_xdmf import (
    _select_vector_time_series_items,
)


def test_article_ready_selection_includes_initial_regular_and_final_states() -> None:
    snapshots = tuple(f"u{step}" for step in range(1, 24))
    records = tuple(
        {"step": step, "time": 0.025 * step}
        for step in range(1, 24)
    )

    items = _select_vector_time_series_items(
        snapshots=snapshots,
        time_records=records,
        dt=0.025,
        every_k=10,
        export_initial_state=True,
        export_final_state=True,
        initial_snapshot=None,
    )

    assert [item[0] for item in items] == [0, 10, 20, 23]
    assert [item[1] for item in items] == [0.0, 0.25, 0.5, 0.5750000000000001]


def test_article_ready_T20_selection_has_81_volumetric_states() -> None:
    snapshots = tuple(range(1, 801))
    records = tuple(
        {"step": step, "time": 0.025 * step}
        for step in range(1, 801)
    )

    items = _select_vector_time_series_items(
        snapshots=snapshots,
        time_records=records,
        dt=0.025,
        every_k=10,
        export_initial_state=True,
        export_final_state=True,
        initial_snapshot=None,
    )

    assert len(items) == 81
    assert items[0][:2] == (0, 0.0)
    assert items[1][:2] == (10, 0.25)
    assert items[-1][:2] == (800, 20.0)
