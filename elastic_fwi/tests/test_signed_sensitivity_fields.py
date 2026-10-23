from __future__ import annotations

import csv
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import types

import numpy as np
import pytest

from elastic_fwi.postprocessing.signed_sensitivity_fields import (
    _resolve_analysis_mode,
    compare_article_safe,
    compute_signed_field_metrics,
    reconstruct_article_safe,
)
from elastic_fwi.postprocessing.vtk_cell_fields import (
    _cell_geometry_2d,
    iter_vtk_cell_field_chunks,
)


def _basic_geometry() -> tuple[np.ndarray, np.ndarray]:
    centers = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=np.float64,
    )
    measures = np.ones(4, dtype=np.float64)
    return centers, measures


def test_positive_constant_field() -> None:
    centers, measures = _basic_geometry()
    metrics = compute_signed_field_metrics(
        np.full(4, 2.0),
        centers,
        measures,
        zero_tolerance=1.0e-14,
    )

    assert metrics["metric_status"] == "OK"
    assert metrics["negative_magnitude_integral"] == pytest.approx(0.0)
    assert metrics["positive_mass_fraction"] == pytest.approx(1.0)
    assert metrics["negative_mass_fraction"] == pytest.approx(0.0)
    assert metrics["signed_balance_index"] == pytest.approx(1.0)
    assert metrics["positive_barycenter_x"] == pytest.approx(0.5)
    assert metrics["positive_barycenter_y"] == pytest.approx(0.5)
    assert metrics["negative_moment_status"] == "UNDEFINED_ZERO_MASS"


def test_negative_constant_field() -> None:
    centers, measures = _basic_geometry()
    metrics = compute_signed_field_metrics(
        np.full(4, -3.0),
        centers,
        measures,
    )

    assert metrics["positive_integral"] == pytest.approx(0.0)
    assert metrics["negative_mass_fraction"] == pytest.approx(1.0)
    assert metrics["signed_balance_index"] == pytest.approx(-1.0)
    assert metrics["negative_barycenter_x"] == pytest.approx(0.5)
    assert metrics["negative_barycenter_y"] == pytest.approx(0.5)
    assert metrics["positive_moment_status"] == "UNDEFINED_ZERO_MASS"


def test_antisymmetric_field_has_zero_balance_and_distinct_barycenters() -> None:
    centers = np.asarray([[-1.0, 0.0], [1.0, 0.0]], dtype=np.float64)
    measures = np.ones(2, dtype=np.float64)
    metrics = compute_signed_field_metrics(
        np.asarray([-2.0, 2.0]),
        centers,
        measures,
    )

    assert metrics["signed_integral"] == pytest.approx(0.0)
    assert metrics["positive_integral"] == pytest.approx(
        metrics["negative_magnitude_integral"]
    )
    assert metrics["signed_balance_index"] == pytest.approx(0.0)
    assert metrics["positive_barycenter_x"] == pytest.approx(1.0)
    assert metrics["negative_barycenter_x"] == pytest.approx(-1.0)
    assert metrics["positive_negative_barycenter_distance"] == pytest.approx(2.0)


def test_zero_field_has_explicit_undefined_moments() -> None:
    centers, measures = _basic_geometry()
    metrics = compute_signed_field_metrics(
        np.zeros(4),
        centers,
        measures,
        zero_tolerance=1.0e-12,
    )

    assert metrics["metric_status"] == "ZERO_FIELD"
    assert metrics["absolute_integral"] == pytest.approx(0.0)
    assert np.isnan(metrics["signed_balance_index"])
    assert np.isnan(metrics["positive_barycenter_x"])
    assert np.isnan(metrics["negative_barycenter_x"])
    assert metrics["absolute_moment_status"] == "UNDEFINED_ZERO_MASS"


def test_article_safe_reconstruction_and_weighted_errors() -> None:
    signed = np.asarray([-4.0, 2.0, 0.0, 1.0])
    reference = np.asarray([1.0, 0.5, 0.0, 0.25])
    measures = np.asarray([1.0, 2.0, 3.0, 4.0])
    reconstructed = reconstruct_article_safe(signed)
    comparison = compare_article_safe(
        reconstructed,
        reference,
        measures,
        rtol=1.0e-14,
        atol=1.0e-14,
    )

    np.testing.assert_allclose(reconstructed, reference)
    assert comparison["article_safe_allclose"] is True
    assert comparison["article_safe_max_abs_error"] == pytest.approx(0.0)
    assert comparison["article_safe_l2_error"] == pytest.approx(0.0)
    assert comparison["article_safe_relative_l2_error"] == pytest.approx(0.0)


def test_scaling_preserves_geometry_and_swaps_signs_for_negative_factor() -> None:
    centers = np.asarray(
        [[-2.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [3.0, 0.0]],
        dtype=np.float64,
    )
    measures = np.asarray([1.0, 2.0, 1.0, 3.0], dtype=np.float64)
    values = np.asarray([-1.0, -2.0, 3.0, 1.0], dtype=np.float64)
    base = compute_signed_field_metrics(values, centers, measures)
    positive_scale = compute_signed_field_metrics(5.0 * values, centers, measures)
    negative_scale = compute_signed_field_metrics(-2.0 * values, centers, measures)

    assert positive_scale["signed_integral"] == pytest.approx(
        5.0 * base["signed_integral"]
    )
    assert positive_scale["positive_integral"] == pytest.approx(
        5.0 * base["positive_integral"]
    )
    assert positive_scale["positive_barycenter_x"] == pytest.approx(
        base["positive_barycenter_x"]
    )
    assert positive_scale["negative_barycenter_x"] == pytest.approx(
        base["negative_barycenter_x"]
    )
    np.testing.assert_allclose(
        reconstruct_article_safe(5.0 * values),
        reconstruct_article_safe(values),
    )

    assert negative_scale["signed_integral"] == pytest.approx(
        -2.0 * base["signed_integral"]
    )
    assert negative_scale["positive_integral"] == pytest.approx(
        2.0 * base["negative_magnitude_integral"]
    )
    assert negative_scale["negative_magnitude_integral"] == pytest.approx(
        2.0 * base["positive_integral"]
    )
    assert negative_scale["positive_barycenter_x"] == pytest.approx(
        base["negative_barycenter_x"]
    )
    assert negative_scale["negative_barycenter_x"] == pytest.approx(
        base["positive_barycenter_x"]
    )
    np.testing.assert_allclose(
        reconstruct_article_safe(-2.0 * values),
        reconstruct_article_safe(values),
    )


def test_vtk_lagrange_triangle_with_three_nodes_is_supported() -> None:
    points = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    connectivity = np.asarray([[0, 1, 2]], dtype=np.int64)

    centers, measures = _cell_geometry_2d(
        points=points,
        connectivity=connectivity,
        cell_type="VTK_LAGRANGE_TRIANGLE",
    )

    np.testing.assert_allclose(
        centers,
        np.asarray([[2.0 / 3.0, 1.0 / 3.0]], dtype=np.float64),
    )
    np.testing.assert_allclose(
        measures,
        np.asarray([1.0], dtype=np.float64),
    )


def test_area_weighting_is_not_an_unweighted_cell_average() -> None:
    centers = np.asarray([[0.0, 0.0], [2.0, 0.0]], dtype=np.float64)
    measures = np.asarray([1.0, 3.0], dtype=np.float64)
    metrics = compute_signed_field_metrics(
        np.ones(2),
        centers,
        measures,
    )

    assert metrics["positive_barycenter_x"] == pytest.approx(1.5)
    assert metrics["positive_integral"] == pytest.approx(4.0)


def _meshio_is_available() -> bool:
    return importlib.util.find_spec("meshio") is not None


def _fake_meshio_module() -> types.ModuleType:
    module = types.ModuleType("meshio")

    def read(path: str | Path):
        with np.load(path, allow_pickle=False) as payload:
            points = np.asarray(payload["points"], dtype=np.float64)
            cells = np.asarray(payload["cells"], dtype=np.int64)
            values = np.asarray(payload["values"], dtype=np.float64)
            field_name = str(payload["field_name"].item())
        return types.SimpleNamespace(
            points=points,
            cells=[types.SimpleNamespace(type="triangle", data=cells)],
            cell_data={field_name: [values]},
        )

    module.read = read
    return module


def _write_fake_meshio_file(path: Path) -> None:
    path.write_text(
        """import types
import numpy as np

def read(path):
    with np.load(path, allow_pickle=False) as payload:
        points = np.asarray(payload['points'], dtype=np.float64)
        cells = np.asarray(payload['cells'], dtype=np.int64)
        values = np.asarray(payload['values'], dtype=np.float64)
        field_name = str(payload['field_name'].item())
    return types.SimpleNamespace(
        points=points,
        cells=[types.SimpleNamespace(type='triangle', data=cells)],
        cell_data={field_name: [values]},
    )
""",
        encoding="utf-8",
    )


def _write_vtu(path: Path, values: np.ndarray) -> None:
    points = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [2.0, 2.0, 0.0],
        ],
        dtype=np.float64,
    )
    cells = [("triangle", np.asarray([[0, 1, 2], [1, 3, 2]], dtype=np.int64))]
    if _meshio_is_available():
        import meshio

        meshio.write(
            path,
            meshio.Mesh(
                points=points,
                cells=cells,
                cell_data={"gradient_rho": [np.asarray(values, dtype=np.float64)]},
            ),
            binary=False,
        )
    else:
        with path.open("wb") as handle:
            np.savez(
                handle,
                points=points,
                cells=cells[0][1],
                values=np.asarray(values, dtype=np.float64),
                field_name=np.asarray("gradient_rho"),
            )


def _write_pvd(path: Path, vtu_name: str) -> None:
    path.write_text(
        "\n".join(
            [
                '<?xml version="1.0"?>',
                '<VTKFile type="Collection" version="1.0" byte_order="LittleEndian">',
                "  <Collection>",
                f'    <DataSet timestep="0" group="" part="0" file="{vtu_name}"/>',
                "  </Collection>",
                "</VTKFile>",
            ]
        ),
        encoding="utf-8",
    )


def _build_cli_case(root: Path) -> tuple[Path, Path]:
    raw_dir = (
        root
        / "inverse"
        / "gradient_combinations_raw_signed"
        / "classical_full"
        / "vtk"
    )
    safe_dir = (
        root
        / "inverse"
        / "gradient_combinations"
        / "classical_full"
        / "vtk"
    )
    raw_dir.mkdir(parents=True)
    safe_dir.mkdir(parents=True)
    raw_vtu = raw_dir / "raw.vtu"
    safe_vtu = safe_dir / "safe.vtu"
    raw_pvd = raw_dir / "material_parameter_gradients_gradient_rho.pvd"
    safe_pvd = safe_dir / "material_parameter_gradients_gradient_rho.pvd"
    raw_values = np.asarray([-4.0, 2.0])
    _write_vtu(raw_vtu, raw_values)
    _write_vtu(safe_vtu, np.abs(raw_values) / np.max(np.abs(raw_values)))
    _write_pvd(raw_pvd, raw_vtu.name)
    _write_pvd(safe_pvd, safe_vtu.name)

    manifest = root / "gradient_combination_campaign_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "kind": "gradient_combination_campaign",
                "status": "ok",
                "output_dir": str(root),
                "config": {
                    "receiver_array_geometry": "line",
                    "defect_center_x": 0.5,
                    "defect_center_y": 0.5,
                    "defect_radius": 0.2,
                },
                "raw_signed_validation": {
                    "found": {
                        "classical_full": {"gradient_rho": str(raw_pvd)}
                    }
                },
                "validation": {
                    "found": {
                        "classical_full": {"gradient_rho": str(safe_pvd)}
                    }
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return manifest, raw_pvd


def test_vtk_reader_exposes_exact_triangle_areas(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "case"
    _, raw_pvd = _build_cli_case(root)
    if not _meshio_is_available():
        monkeypatch.setitem(sys.modules, "meshio", _fake_meshio_module())
    chunks = list(iter_vtk_cell_field_chunks(raw_pvd, "gradient_rho"))

    assert len(chunks) == 1
    np.testing.assert_allclose(chunks[0].cell_measures, np.asarray([1.0, 2.0]))


def test_cli_creates_csv_manifest_figures_and_honors_quiet(tmp_path: Path) -> None:
    root = tmp_path / "case"
    _build_cli_case(root)
    output_dir = tmp_path / "analysis"
    script = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "revision_wave_motion"
        / "analyze_signed_sensitivity_fields.py"
    )
    env = os.environ.copy()
    if not _meshio_is_available():
        fake_deps = tmp_path / "fake_dependencies"
        fake_deps.mkdir()
        _write_fake_meshio_file(fake_deps / "meshio.py")
        env["PYTHONPATH"] = os.pathsep.join(
            [str(fake_deps), str(Path(__file__).resolve().parents[2]), env.get("PYTHONPATH", "")]
        )
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--input-dir",
            str(root),
            "--output-dir",
            str(output_dir),
            "--analysis-mode",
            "line-only",
            "--quiet",
        ],
        check=False,
        text=True,
        capture_output=True,
        env=env,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == ""
    csv_path = output_dir / "signed_sensitivity_metrics.csv"
    manifest_path = output_dir / "signed_sensitivity_manifest.json"
    assert csv_path.exists()
    assert manifest_path.exists()
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    assert len(rows) == 1
    assert rows[0]["status"] == "OK"
    assert rows[0]["article_safe_allclose"] == "True"
    assert Path(rows[0]["signed_figure_png"]).exists()
    assert Path(rows[0]["signed_figure_pdf"]).exists()
    assert Path(rows[0]["reconstructed_article_safe_png"]).exists()
    assert Path(rows[0]["reconstructed_article_safe_pdf"]).exists()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["analysis_mode"] == "line-only"
    assert payload["geometry_ids"] == ["line"]
    assert payload["summary"]["analysis_mode"] == "line-only"
    assert payload["summary"]["num_discovered"] == 1
    assert payload["summary"]["num_article_safe_allclose"] == 1


def test_analysis_mode_contract() -> None:
    assert _resolve_analysis_mode("auto", ["line"]) == "line-only"
    assert _resolve_analysis_mode("auto", ["line", "box"]) == "line-box"
    assert _resolve_analysis_mode("line-only", ["line"]) == "line-only"
    with pytest.raises(ValueError, match="exclusively line"):
        _resolve_analysis_mode("line-only", ["line", "box"])
    with pytest.raises(ValueError, match="requires both line and box"):
        _resolve_analysis_mode("line-box", ["line"])
