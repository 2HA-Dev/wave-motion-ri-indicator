from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from elastic_fwi.postprocessing.single_defect_method_comparison import (
    CardInput,
    CardKey,
    compute_raw_signed_metrics_streaming,
)
from elastic_fwi.postprocessing.vtk_cell_fields import CellFieldChunk


def _extractor_module():
    script = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "revision_wave_motion"
        / "extract_gradient_image_quality_metrics.py"
    )
    spec = importlib.util.spec_from_file_location("p1_metric_extractor", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_article_safe_metrics_use_exact_cell_areas() -> None:
    module = _extractor_module()
    result = module.compute_metrics(
        centers=np.asarray([[0.0, 0.0], [1.0, 0.0]]),
        values=np.asarray([1.0, 0.5]),
        cell_measures=np.asarray([0.1, 1.0]),
        defect_center=(0.0, 0.0),
        defect_radius=0.2,
        false_max_threshold_fraction=0.5,
        false_max_min_distance_factor=1.0,
    )

    assert result["energy_in_defect"] == pytest.approx(0.1)
    assert result["energy_outside_defect"] == pytest.approx(0.25)
    assert result["energy_fraction_in_defect"] == pytest.approx(0.1 / 0.35)
    assert result["half_max_area"] == pytest.approx(1.1)
    assert result["mean_in_defect"] == pytest.approx(1.0)
    assert result["mean_outside_defect"] == pytest.approx(0.5)
    assert result["spatial_measure"] == "exact_cell_area_weighted"


def test_raw_signed_metrics_use_exact_cell_areas(tmp_path: Path) -> None:
    card = CardInput(
        key=CardKey("line", "raw_signed", "classical_full", "rho"),
        manifest_path=tmp_path / "manifest.json",
        declared_path="dummy.pvd",
        pvd_path=tmp_path / "dummy.pvd",
        expected_field_name="gradient_rho",
    )
    chunk = CellFieldChunk(
        source_piece=tmp_path / "piece.vtu",
        cell_type="triangle",
        field_name="gradient_rho",
        centers=np.asarray([[0.0, 0.0], [1.0, 0.0]]),
        values=np.asarray([1.0, -0.5]),
        cell_measures=np.asarray([0.1, 1.0]),
        num_cells_before_ghost_filter=2,
        num_ghost_cells_removed=0,
    )

    with patch(
        "elastic_fwi.postprocessing.single_defect_method_comparison.iter_vtk_cell_field_chunks",
        return_value=iter((chunk,)),
    ):
        result = compute_raw_signed_metrics_streaming(
            card,
            defect_center=(0.0, 0.0),
            defect_radius=0.2,
            max_false_max_candidates=10,
        )

    assert result["positive_sum"] == pytest.approx(0.1)
    assert result["negative_sum_abs"] == pytest.approx(0.5)
    assert result["positive_quadratic_energy"] == pytest.approx(0.1)
    assert result["negative_quadratic_energy"] == pytest.approx(0.25)
    assert result["energy_in_defect"] == pytest.approx(0.1)
    assert result["energy_outside_defect"] == pytest.approx(0.25)
    assert result["energy_fraction_in_defect"] == pytest.approx(0.1 / 0.35)
    assert result["half_max_area"] == pytest.approx(1.1)
    assert result["spatial_measure"] == "exact_cell_area_weighted"


def test_legacy_cell_integrals_are_converted_to_cell_average_density() -> None:
    from types import SimpleNamespace
    from elastic_fwi.postprocessing.dg0_l2_reexport import _cell_density_blocks

    # Two triangles with areas 0.5 and 1.0; stored legacy values are integrals
    # of the constant density 2.0, hence [1.0, 2.0].
    mesh = SimpleNamespace(
        points=np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [2.0, 0.0, 0.0],
                [2.0, 1.0, 0.0],
            ]
        ),
        cells=[
            SimpleNamespace(
                type="triangle",
                data=np.asarray([[0, 1, 2], [1, 3, 4]], dtype=np.int64),
            )
        ],
        cell_data={"gradient_rho": [np.asarray([1.0, 1.0])]},
    )
    # Both triangles actually have area 0.5, so the density is 2 everywhere.
    densities = _cell_density_blocks(mesh, "gradient_rho", piece=Path("piece.vtu"))
    assert len(densities) == 1
    assert np.allclose(densities[0], 2.0)
