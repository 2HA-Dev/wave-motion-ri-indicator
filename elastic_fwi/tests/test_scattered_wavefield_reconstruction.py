from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import meshio
import numpy as np
import pytest

from elastic_fwi.postprocessing.scattered_wavefield_reconstruction import (
    AnalysisBox,
    FieldSelectionError,
    MeshCompatibilityError,
    PVDSeriesError,
    ScatteredWavefieldConfig,
    ScatteredWavefieldInputs,
    dataset_piece_paths,
    pair_pvd_series,
    parse_pvd_series,
    run_scattered_wavefield_analysis,
    validate_mesh_pair,
    validate_pre_interaction,
    validate_pre_detection_relative,
    write_scattered_wavefield_outputs,
)


def _write_pvtu(path: Path, piece_names: list[str]) -> None:
    root = ET.Element("VTKFile", {"type": "PUnstructuredGrid", "version": "1.0"})
    grid = ET.SubElement(root, "PUnstructuredGrid", {"GhostLevel": "0"})
    point_data = ET.SubElement(grid, "PPointData")
    ET.SubElement(
        point_data,
        "PDataArray",
        {
            "type": "Float64",
            "Name": "field",
            "NumberOfComponents": "3",
        },
    )
    points = ET.SubElement(grid, "PPoints")
    ET.SubElement(
        points,
        "PDataArray",
        {"type": "Float64", "NumberOfComponents": "3"},
    )
    for name in piece_names:
        ET.SubElement(grid, "Piece", {"Source": name})
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def _write_pvd(path: Path, datasets: list[tuple[float, str]]) -> None:
    root = ET.Element("VTKFile", {"type": "Collection", "version": "1.0"})
    collection = ET.SubElement(root, "Collection")
    for time_value, dataset in datasets:
        ET.SubElement(
            collection,
            "DataSet",
            {
                "timestep": str(time_value),
                "part": "0",
                "file": dataset,
            },
        )
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def _piece_mesh(
    points: np.ndarray,
    triangle: np.ndarray,
    values: np.ndarray,
    field_name: str,
) -> meshio.Mesh:
    vector3 = np.zeros((values.shape[0], 3), dtype=float)
    vector3[:, :2] = values
    return meshio.Mesh(
        points=np.column_stack([points, np.zeros(points.shape[0])]),
        cells=[("triangle", triangle)],
        point_data={field_name: vector3},
    )


def _write_series(
    root: Path,
    *,
    prefix: str,
    field_name: str,
    times: tuple[float, ...],
    differences: tuple[tuple[float, float], ...],
    defect: bool,
    multipart: bool = True,
    geometry_shift: float = 0.0,
    nonfinite: bool = False,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    datasets: list[tuple[float, str]] = []
    base_points = np.asarray(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=float
    )
    if geometry_shift:
        base_points = base_points.copy()
        base_points[0, 0] += geometry_shift
    triangles = [np.asarray([[0, 1, 2]], dtype=int), np.asarray([[0, 2, 3]], dtype=int)]
    for snapshot_index, (time_value, difference) in enumerate(zip(times, differences, strict=True)):
        healthy_values = np.column_stack(
            [
                np.full(4, 2.0 + time_value),
                np.full(4, -1.0 + 0.5 * time_value),
            ]
        )
        values = healthy_values.copy()
        if defect:
            values += np.asarray(difference)[None, :]
        if nonfinite and snapshot_index == len(times) - 1:
            values[0, 0] = np.nan

        if multipart:
            piece_names: list[str] = []
            for piece_index, triangle in enumerate(triangles):
                piece_name = f"{prefix}_{snapshot_index:03d}_piece{piece_index}.vtu"
                meshio.write(
                    root / piece_name,
                    _piece_mesh(base_points, triangle, values, field_name),
                    binary=False,
                )
                piece_names.append(piece_name)
            dataset_name = f"{prefix}_{snapshot_index:03d}.pvtu"
            _write_pvtu(root / dataset_name, piece_names)
        else:
            dataset_name = f"{prefix}_{snapshot_index:03d}.vtu"
            meshio.write(
                root / dataset_name,
                meshio.Mesh(
                    points=np.column_stack([base_points, np.zeros(4)]),
                    cells=[("triangle", np.asarray([[0, 1, 2], [0, 2, 3]], dtype=int))],
                    point_data={
                        field_name: np.column_stack([values, np.zeros(4)])
                    },
                ),
                binary=False,
            )
        datasets.append((time_value, dataset_name))
    pvd = root / f"{prefix}.pvd"
    _write_pvd(pvd, datasets)
    return pvd


def _config(tmp_path: Path, *, geometry_shift: float = 0.0, nonfinite: bool = False) -> ScatteredWavefieldConfig:
    times = (0.0, 1.0, 2.0)
    differences = ((0.0, 0.0), (1.0, 2.0), (2.0, 0.0))
    healthy = _write_series(
        tmp_path / "healthy",
        prefix="healthy",
        field_name="healthy_u",
        times=times,
        differences=differences,
        defect=False,
    )
    defect = _write_series(
        tmp_path / "defect",
        prefix="defect",
        field_name="defect_u",
        times=times,
        differences=differences,
        defect=True,
        geometry_shift=geometry_shift,
        nonfinite=nonfinite,
    )
    return ScatteredWavefieldConfig(
        inputs=ScatteredWavefieldInputs(healthy_pvd=healthy, defect_pvd=defect),
        output_dir=tmp_path / "out",
        geometry_kind="line",
        healthy_field="healthy_u",
        defect_field="defect_u",
        defect_center=(0.5, 0.5),
        defect_radius=0.2,
        analysis_box=AnalysisBox(0.0, 1.0, 0.0, 1.0),
        zone_width=0.2,
        pre_atol=1.0e-14,
        pre_rtol=1.0e-12,
        detection_fraction=0.1,
        zone_detection_fraction=0.1,
        raster_resolution=64,
        vtk_export_mode="selected",
    )


def test_parse_and_pair_pvd_series(tmp_path: Path) -> None:
    config = _config(tmp_path)
    healthy = parse_pvd_series(config.inputs.healthy_pvd)
    defect = parse_pvd_series(config.inputs.defect_pvd)
    pairs = pair_pvd_series(healthy, defect, atol=1.0e-12, rtol=1.0e-12)
    assert [pair.time for pair in pairs] == [0.0, 1.0, 2.0]
    assert len(dataset_piece_paths(pairs[0].healthy_dataset)) == 2


def test_exact_scattered_energy_and_subtraction(tmp_path: Path) -> None:
    result = run_scattered_wavefield_analysis(_config(tmp_path))
    rows = result.metrics
    assert rows[0]["energy_scattered"] == pytest.approx(0.0, abs=1.0e-14)
    assert rows[1]["energy_scattered"] == pytest.approx(5.0, rel=1.0e-12)
    assert rows[1]["l2_scattered"] == pytest.approx(np.sqrt(5.0))
    assert rows[1]["max_abs_ux_scattered"] == pytest.approx(1.0)
    assert rows[1]["max_abs_uy_scattered"] == pytest.approx(2.0)
    assert rows[1]["max_norm_scattered"] == pytest.approx(np.sqrt(5.0))
    assert result.validation["status"] == "PASS"
    assert result.validation["validation_mode"] == "relative_pre_detection"
    assert result.validation["time_semantics"] == (
        "threshold_detection_not_physical_first_arrival"
    )
    assert result.characteristic_times["interaction_arrival_time"] == pytest.approx(1.0)
    assert result.characteristic_times["selected_detection_time"] == pytest.approx(1.0)
    assert result.characteristic_times["time_semantics"] == (
        "threshold_detection_not_physical_first_arrival"
    )


def test_identical_fields_are_zero_and_finite(tmp_path: Path) -> None:
    config = _config(tmp_path)
    defect_pvd = _write_series(
        tmp_path / "identical",
        prefix="identical",
        field_name="defect_u",
        times=(0.0, 1.0, 2.0),
        differences=((0.0, 0.0),) * 3,
        defect=True,
    )
    config = ScatteredWavefieldConfig(
        **{**config.__dict__, "inputs": ScatteredWavefieldInputs(config.inputs.healthy_pvd, defect_pvd)}
    )
    result = run_scattered_wavefield_analysis(config)
    assert all(row["energy_scattered"] == pytest.approx(0.0) for row in result.metrics)
    assert result.validation["status"] == "WARNING"
    assert result.validation["reason"] == "threshold_detection_time_unavailable"
    assert result.validation["validation_mode"] == "relative_pre_detection"
    assert result.validation["time_semantics"] == (
        "threshold_detection_not_physical_first_arrival"
    )


def test_time_mismatch_is_rejected(tmp_path: Path) -> None:
    config = _config(tmp_path)
    defect = _write_series(
        tmp_path / "time_bad",
        prefix="time_bad",
        field_name="defect_u",
        times=(0.0, 1.1, 2.0),
        differences=((0.0, 0.0), (1.0, 2.0), (2.0, 0.0)),
        defect=True,
    )
    bad = ScatteredWavefieldConfig(
        **{**config.__dict__, "inputs": ScatteredWavefieldInputs(config.inputs.healthy_pvd, defect)}
    )
    with pytest.raises(PVDSeriesError, match="Time mismatch"):
        run_scattered_wavefield_analysis(bad)


def test_geometry_mismatch_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(MeshCompatibilityError, match="coordinates differ"):
        run_scattered_wavefield_analysis(_config(tmp_path, geometry_shift=0.01))


def test_equivalent_triangle_topology_accepts_different_cell_blocks(
    tmp_path: Path,
) -> None:
    points = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=float,
    )
    healthy = meshio.Mesh(
        points=points,
        cells=[
            ("line", np.asarray([[0, 1]], dtype=int)),
            ("triangle", np.asarray([[0, 1, 2]], dtype=int)),
            ("triangle", np.asarray([[0, 2, 3]], dtype=int)),
        ],
    )
    defect = meshio.Mesh(
        points=points.copy(),
        cells=[
            ("triangle", np.asarray([[3, 2, 0], [2, 1, 0]], dtype=int)),
            ("vertex", np.asarray([[0]], dtype=int)),
        ],
    )

    validate_mesh_pair(
        healthy,
        defect,
        healthy_piece=tmp_path / "healthy.vtu",
        defect_piece=tmp_path / "defect.vtu",
        atol=1.0e-14,
        rtol=1.0e-12,
    )


def test_nonfinite_field_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(FieldSelectionError, match="NaN or infinity"):
        run_scattered_wavefield_analysis(_config(tmp_path, nonfinite=True))


def test_missing_field_is_rejected(tmp_path: Path) -> None:
    config = _config(tmp_path)
    bad = ScatteredWavefieldConfig(
        **{**config.__dict__, "healthy_field": "missing"}
    )
    with pytest.raises(FieldSelectionError, match="absent"):
        run_scattered_wavefield_analysis(bad)


def test_manual_preinteraction_failure_is_detailed() -> None:
    rows = [
        {
            "time": 0.0,
            "max_norm_scattered": 1.0,
            "max_norm_healthy": 1.0,
            "l2_scattered": 1.0,
            "l2_healthy": 1.0,
            "domain_area_integrated": 1.0,
            "relative_l2": 1.0,
        },
        {
            "time": 1.0,
            "max_norm_scattered": 2.0,
            "max_norm_healthy": 1.0,
            "l2_scattered": 2.0,
            "l2_healthy": 1.0,
            "domain_area_integrated": 1.0,
            "relative_l2": 2.0,
        },
    ]
    validation = validate_pre_interaction(
        rows, arrival_time=1.0, atol=1.0e-12, rtol=1.0e-12
    )
    assert validation["status"] == "FAIL"
    assert validation["num_fail"] == 1
    assert rows[0]["pre_interaction_pass"] is False



def test_relative_pre_detection_validation_passes_on_negligible_background() -> None:
    rows = [
        {
            "time": 0.0,
            "energy_scattered": 1.0e-7,
            "l2_scattered": np.sqrt(1.0e-7),
            "max_norm_scattered": 5.0e-4,
        },
        {
            "time": 1.0,
            "energy_scattered": 5.0e-8,
            "l2_scattered": np.sqrt(5.0e-8),
            "max_norm_scattered": 4.0e-4,
        },
        {
            "time": 2.0,
            "energy_scattered": 1.0,
            "l2_scattered": 1.0,
            "max_norm_scattered": 1.0,
        },
    ]

    validation = validate_pre_detection_relative(
        rows,
        detection_time=2.0,
        detection_fraction=1.0e-6,
        transition_guard_snapshots=0,
    )

    assert validation["status"] == "PASS"
    assert validation["validation_mode"] == "relative_pre_detection"
    assert validation["relative_energy_pass"] is True
    assert validation["relative_l2_pass"] is True
    assert validation["time_semantics"] == (
        "threshold_detection_not_physical_first_arrival"
    )


def test_relative_pre_detection_validation_fails_on_large_background() -> None:
    rows = [
        {
            "time": 0.0,
            "energy_scattered": 1.0e-3,
            "l2_scattered": np.sqrt(1.0e-3),
            "max_norm_scattered": 2.0e-2,
        },
        {
            "time": 1.0,
            "energy_scattered": 1.0,
            "l2_scattered": 1.0,
            "max_norm_scattered": 1.0,
        },
    ]

    validation = validate_pre_detection_relative(
        rows,
        detection_time=1.0,
        detection_fraction=1.0e-6,
        transition_guard_snapshots=0,
    )

    assert validation["status"] == "FAIL"
    assert validation["relative_energy_pass"] is False
    assert validation["relative_l2_pass"] is False


def test_relative_pointwise_failure_is_secondary_only() -> None:
    rows = [
        {
            "time": 0.0,
            "energy_scattered": 1.0e-8,
            "l2_scattered": 1.0e-4,
            "max_norm_scattered": 2.0e-3,
        },
        {
            "time": 1.0,
            "energy_scattered": 1.0,
            "l2_scattered": 1.0,
            "max_norm_scattered": 1.0,
        },
    ]

    validation = validate_pre_detection_relative(
        rows,
        detection_time=1.0,
        detection_fraction=1.0e-6,
        transition_guard_snapshots=0,
    )

    assert validation["relative_energy_pass"] is True
    assert validation["relative_l2_pass"] is True
    assert validation["relative_pointwise_pass"] is False

    # Linf is explicitly secondary.
    assert validation["status"] == "PASS"
    assert validation["secondary_pointwise_status"] == "WARNING"



def test_outputs_include_csv_figures_json_and_scattered_pvd(tmp_path: Path) -> None:
    config = _config(tmp_path)
    result = run_scattered_wavefield_analysis(config)
    outputs = write_scattered_wavefield_outputs(
        result,
        config,
        command=["python", "synthetic"],
        repository_root=tmp_path,
    )
    assert outputs["metrics_csv"].is_file()
    assert outputs["summary_json"].is_file()
    assert outputs["manifest_json"].is_file()
    assert outputs["scattered_pvd"].is_file()
    header = outputs["metrics_csv"].read_text(encoding="utf-8").splitlines()[0]
    assert "analysis_mode" in header
    assert "geometry_kind" in header
    summary = json.loads(outputs["summary_json"].read_text(encoding="utf-8"))
    manifest = json.loads(outputs["manifest_json"].read_text(encoding="utf-8"))
    assert summary["analysis_mode"] == "line-only"
    assert summary["geometry_kind"] == "line"
    assert manifest["analysis_mode"] == "line-only"
    assert manifest["geometry_kind"] == "line"
    assert any(path.suffix == ".png" for key, path in outputs.items() if key.startswith("figure_"))
    parsed = parse_pvd_series(outputs["scattered_pvd"])
    assert len(parsed.entries) >= 2
    first_piece = dataset_piece_paths(parsed.entries[0].dataset_path)[0]
    mesh = meshio.read(first_piece)
    assert set(mesh.point_data) == {
        "u_scattered",
        "u_scattered_x",
        "u_scattered_y",
        "u_scattered_norm",
    }



def test_invalid_geometry_kind_is_rejected(tmp_path: Path) -> None:
    config = _config(tmp_path)
    invalid = ScatteredWavefieldConfig(
        **{**config.__dict__, "geometry_kind": "line-box"}
    )
    with pytest.raises(ValueError, match="geometry_kind"):
        invalid.validate()


def test_metrics_csv_is_reproducible(tmp_path: Path) -> None:
    config_a = _config(tmp_path / "a")
    result_a = run_scattered_wavefield_analysis(config_a)
    outputs_a = write_scattered_wavefield_outputs(
        result_a,
        config_a,
        command=["python", "synthetic"],
        repository_root=tmp_path,
    )
    config_b = _config(tmp_path / "b")
    result_b = run_scattered_wavefield_analysis(config_b)
    outputs_b = write_scattered_wavefield_outputs(
        result_b,
        config_b,
        command=["python", "synthetic"],
        repository_root=tmp_path,
    )
    text_a = outputs_a["metrics_csv"].read_text(encoding="utf-8")
    text_b = outputs_b["metrics_csv"].read_text(encoding="utf-8")
    # Source paths differ; numeric columns and ordering must remain identical.
    rows_a = list(__import__("csv").DictReader(text_a.splitlines()))
    rows_b = list(__import__("csv").DictReader(text_b.splitlines()))
    for rows in (rows_a, rows_b):
        for row in rows:
            row.pop("healthy_dataset")
            row.pop("defect_dataset")
    assert rows_a == rows_b


def test_cli_parser_accepts_explicit_line_geometry() -> None:
    script = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "revision_wave_motion"
        / "reconstruct_scattered_wavefield.py"
    )
    spec = importlib.util.spec_from_file_location("scattered_wavefield_cli", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = module.build_parser().parse_args(
        [
            "--geometry",
            "line",
            "--healthy-pvd",
            "healthy.pvd",
            "--defect-pvd",
            "defect.pvd",
            "--output-dir",
            "out",
        ]
    )
    assert args.geometry_kind == "line"
