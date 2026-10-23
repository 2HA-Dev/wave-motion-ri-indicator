from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from elastic_fwi.diagnostics.plot_residual_wave_overlap import (
    _prepare_output_directory,
    _selected_snapshot_indices,
    build_argument_parser,
    rasterize_nodal_values,
)


def test_select_full_time_interval():
    times = np.asarray(
        [0.0, 0.5, 1.0, 1.5],
        dtype=np.float64,
    )

    indices = _selected_snapshot_indices(
        times,
        start_time=0.5,
        end_time=1.0,
        single_time=None,
    )

    assert indices == [1, 2]


def test_select_single_time():
    times = np.asarray(
        [0.0, 0.5, 1.0],
        dtype=np.float64,
    )

    indices = _selected_snapshot_indices(
        times,
        start_time=None,
        end_time=None,
        single_time=0.5,
    )

    assert indices == [1]


def test_rasterize_nonnegative_values():
    points = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [0.5, 0.5],
        ],
        dtype=np.float64,
    )

    values = np.asarray(
        [0.0, 1.0, 2.0, 3.0, 4.0],
        dtype=np.float64,
    )

    grid, extent = rasterize_nodal_values(
        points,
        values,
        resolution=100,
        signed=False,
    )

    assert grid.shape == (100, 100)
    assert np.all(np.isfinite(grid))
    assert float(np.max(grid)) == 4.0
    assert extent == (0.0, 1.0, 0.0, 1.0)


def test_rasterize_signed_values():
    points = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [0.5, 0.5],
        ],
        dtype=np.float64,
    )

    values = np.asarray(
        [-1.0, 1.0, -2.0, 2.0, 0.0],
        dtype=np.float64,
    )

    grid, _ = rasterize_nodal_values(
        points,
        values,
        resolution=100,
        signed=True,
    )

    assert grid.shape == (100, 100)
    assert np.all(np.isfinite(grid))
    assert float(np.min(grid)) < 0.0
    assert float(np.max(grid)) > 0.0


def test_existing_outputs_require_overwrite(
    tmp_path: Path,
):
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()

    existing = (
        output_dir
        / "residual_wave_overlap_metrics.csv"
    )

    existing.write_text(
        "existing\n",
        encoding="utf-8",
    )

    with pytest.raises(
        FileExistsError,
        match="--overwrite",
    ):
        _prepare_output_directory(
            output_dir,
            overwrite=False,
        )

    paths = _prepare_output_directory(
        output_dir,
        overwrite=True,
    )

    assert not existing.exists()

    assert (
        paths["metrics_csv"]
        == existing
    )


def test_cli_required_arguments():
    parser = build_argument_parser()

    with pytest.raises(SystemExit):
        parser.parse_args([])


def _temporal_metric_row(
    *,
    mode: str,
    index: int,
    time: float,
    total: float,
    defect: float,
    ring: float,
    peak_distance: float,
) -> dict:
    outside = total - defect - ring

    return {
        "case_id": "synthetic",
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
        ),
        "energy_fraction_defect_and_ring": (
            (defect + ring) / total
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
        "peak_defined": True,
    }


def test_dual_temporal_selection_bundle():
    from elastic_fwi.diagnostics.plot_residual_wave_overlap import (
        _compute_temporal_selections,
    )

    rows = []

    for mode in (
        "absolute_dot",
        "norm_product",
    ):
        rows.extend(
            [
                _temporal_metric_row(
                    mode=mode,
                    index=0,
                    time=0.0,
                    total=100.0,
                    defect=1.0,
                    ring=3.0,
                    peak_distance=2.0,
                ),
                _temporal_metric_row(
                    mode=mode,
                    index=1,
                    time=1.0,
                    total=60.0,
                    defect=5.0,
                    ring=15.0,
                    peak_distance=0.1,
                ),
            ]
        )

    result = _compute_temporal_selections(
        rows,
        primary_mode="norm_product",
        activity_threshold=0.10,
    )

    primary = result[
        "primary_selection"
    ]

    assert (
        primary["overlap_selection"][
            "time_value"
        ]
        == 0.0
    )

    assert (
        primary["localization_selection"][
            "time_value"
        ]
        == 1.0
    )

    assert len(result["rows"]) == 4

    overlap_rows = [
        row
        for row in result["rows"]
        if row[
            "selected_for_overlap_figure"
        ]
    ]

    localization_rows = [
        row
        for row in result["rows"]
        if row[
            "selected_for_localization_figure"
        ]
    ]

    assert len(overlap_rows) == 2
    assert len(localization_rows) == 2

    assert {
        row["time_value"]
        for row in overlap_rows
    } == {0.0}

    assert {
        row["time_value"]
        for row in localization_rows
    } == {1.0}


def test_localization_activity_threshold_cli_default():
    parser = build_argument_parser()

    args = parser.parse_args(
        [
            "--geometry",
            "line",
            "--reflection-field",
            "reflection.pvd",
            "--transmission-field",
            "transmission.pvd",
            "--output-dir",
            "outputs",
            "--case-id",
            "line",
            "--defect-center-x",
            "36.5",
            "--defect-center-y",
            "36.5",
            "--defect-radius",
            "0.25",
            "--ring-width",
            "0.25",
        ]
    )

    assert (
        args.localization_activity_threshold
        == pytest.approx(0.10)
    )
    assert args.geometry == "line"



def test_temporal_diagnostics_outputs_are_written(
    tmp_path: Path,
):
    from elastic_fwi.diagnostics.plot_residual_wave_overlap import (
        _compute_temporal_selections,
        _produce_temporal_diagnostics,
    )

    rows = []

    for mode in (
        "absolute_dot",
        "norm_product",
    ):
        rows.extend(
            [
                _temporal_metric_row(
                    mode=mode,
                    index=0,
                    time=0.0,
                    total=100.0,
                    defect=1.0,
                    ring=3.0,
                    peak_distance=2.0,
                ),
                _temporal_metric_row(
                    mode=mode,
                    index=1,
                    time=1.0,
                    total=60.0,
                    defect=5.0,
                    ring=15.0,
                    peak_distance=0.1,
                ),
            ]
        )

    temporal = _compute_temporal_selections(
        rows,
        primary_mode="norm_product",
        activity_threshold=0.10,
    )

    png = tmp_path / "temporal.png"
    pdf = tmp_path / "temporal.pdf"

    metadata = _produce_temporal_diagnostics(
        rows=temporal["rows"],
        primary_mode="norm_product",
        primary_selection=(
            temporal["primary_selection"]
        ),
        output_png=png,
        output_pdf=pdf,
        figure_dpi=100,
    )

    assert png.is_file()
    assert pdf.is_file()
    assert png.stat().st_size > 0
    assert pdf.stat().st_size > 0

    assert metadata["overlap_time"] == 0.0
    assert metadata["localization_time"] == 1.0
    assert metadata["num_snapshots"] == 2
