from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from elastic_fwi.postprocessing.vtk_residual_wave_io import (
    RawResidualWaveSnapshot,
    assemble_residual_wave_pair,
    build_residual_wave_assembly_plan,
    read_pvd_series,
    select_pvd_snapshot,
    validate_pvd_pair_times,
)


def raw_snapshot(
    *,
    field_name: str,
    points: np.ndarray | None = None,
    field: np.ndarray | None = None,
) -> RawResidualWaveSnapshot:
    if points is None:
        points = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )

    triangles = np.asarray(
        [
            [0, 1, 2],
            [3, 4, 5],
        ],
        dtype=np.int64,
    )

    if field is None:
        field = np.asarray(
            [
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [4.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )

    return RawResidualWaveSnapshot(
        path=Path(f"{field_name}.pvtu"),
        field_name=field_name,
        points=np.ascontiguousarray(points),
        triangles_owned=triangles,
        field=np.ascontiguousarray(field),
        number_of_raw_points=points.shape[0],
        number_of_raw_cells=triangles.shape[0],
        number_of_owned_cells=triangles.shape[0],
        owned_cell_type_ids=(69,),
        available_point_arrays=(field_name,),
    )


def test_read_pvd_series_and_select_time(
    tmp_path: Path,
):
    first = tmp_path / "snapshot_000.pvtu"
    second = tmp_path / "snapshot_001.pvtu"

    first.write_text(
        "<VTKFile/>",
        encoding="utf-8",
    )

    second.write_text(
        "<VTKFile/>",
        encoding="utf-8",
    )

    pvd = tmp_path / "series.pvd"

    pvd.write_text(
        """<?xml version="1.0"?>
<VTKFile type="Collection" version="0.1">
  <Collection>
    <DataSet timestep="0.0" file="snapshot_000.pvtu"/>
    <DataSet timestep="0.5" file="snapshot_001.pvtu"/>
  </Collection>
</VTKFile>
""",
        encoding="utf-8",
    )

    series = read_pvd_series(pvd)

    assert len(series.snapshots) == 2

    assert np.array_equal(
        series.times,
        np.asarray([0.0, 0.5]),
    )

    selected = select_pvd_snapshot(
        series,
        requested_time=0.5,
    )

    assert selected.index == 1
    assert selected.time == 0.5
    assert selected.path == second.resolve()


def test_non_increasing_pvd_times_are_rejected(
    tmp_path: Path,
):
    dataset = tmp_path / "snapshot.pvtu"

    dataset.write_text(
        "<VTKFile/>",
        encoding="utf-8",
    )

    pvd = tmp_path / "series.pvd"

    pvd.write_text(
        """<VTKFile type="Collection">
<Collection>
<DataSet timestep="0.5" file="snapshot.pvtu"/>
<DataSet timestep="0.0" file="snapshot.pvtu"/>
</Collection>
</VTKFile>
""",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="strictly increasing",
    ):
        read_pvd_series(pvd)


def test_pvd_pair_time_mismatch_is_rejected(
    tmp_path: Path,
):
    dataset = tmp_path / "snapshot.pvtu"

    dataset.write_text(
        "<VTKFile/>",
        encoding="utf-8",
    )

    left_path = tmp_path / "left.pvd"
    right_path = tmp_path / "right.pvd"

    left_path.write_text(
        """<VTKFile><Collection>
<DataSet timestep="0.0" file="snapshot.pvtu"/>
</Collection></VTKFile>
""",
        encoding="utf-8",
    )

    right_path.write_text(
        """<VTKFile><Collection>
<DataSet timestep="0.5" file="snapshot.pvtu"/>
</Collection></VTKFile>
""",
        encoding="utf-8",
    )

    left = read_pvd_series(left_path)
    right = read_pvd_series(right_path)

    with pytest.raises(
        ValueError,
        match="timesteps differ",
    ):
        validate_pvd_pair_times(
            left,
            right,
        )


def test_duplicate_points_are_assembled_consistently():
    reflection = raw_snapshot(
        field_name="reflection",
    )

    transmission = raw_snapshot(
        field_name="transmission",
        field=2.0 * reflection.field,
    )

    plan = build_residual_wave_assembly_plan(
        reflection,
        transmission,
    )

    assembled = assemble_residual_wave_pair(
        plan,
        reflection,
        transmission,
    )

    assert plan.num_global_points == 4
    assert plan.num_global_triangles == 2
    assert plan.duplicate_used_points == 2

    assert assembled.reflection_field.shape == (
        4,
        3,
    )

    assert assembled.transmission_field.shape == (
        4,
        3,
    )

    assert (
        assembled.reflection_duplicate_max_discrepancy
        == 0.0
    )

    assert (
        assembled.transmission_duplicate_max_discrepancy
        == 0.0
    )

    assert np.array_equal(
        assembled.transmission_field,
        2.0 * assembled.reflection_field,
    )


def test_inconsistent_duplicate_values_are_rejected():
    reflection = raw_snapshot(
        field_name="reflection",
    )

    transmission_field = (
        2.0 * reflection.field
    )

    transmission_field[3, 0] += 1.0e-3

    transmission = raw_snapshot(
        field_name="transmission",
        field=transmission_field,
    )

    plan = build_residual_wave_assembly_plan(
        reflection,
        transmission,
    )

    with pytest.raises(
        ValueError,
        match="Inconsistent duplicated nodal values",
    ):
        assemble_residual_wave_pair(
            plan,
            reflection,
            transmission,
            duplicate_rtol=1.0e-12,
            duplicate_atol=0.0,
        )


def test_reflection_transmission_mesh_mismatch_is_rejected():
    reflection = raw_snapshot(
        field_name="reflection",
    )

    points = reflection.points.copy()
    points[0, 0] = 0.25

    transmission = raw_snapshot(
        field_name="transmission",
        points=points,
    )

    with pytest.raises(
        ValueError,
        match="coordinates differ",
    ):
        build_residual_wave_assembly_plan(
            reflection,
            transmission,
        )


def test_significant_mesh_z_coordinate_is_rejected():
    points = raw_snapshot(
        field_name="template",
    ).points.copy()

    points[2, 2] = 1.0e-3

    reflection = raw_snapshot(
        field_name="reflection",
        points=points,
    )

    transmission = raw_snapshot(
        field_name="transmission",
        points=points,
    )

    with pytest.raises(
        ValueError,
        match="significant z coordinate",
    ):
        build_residual_wave_assembly_plan(
            reflection,
            transmission,
        )
