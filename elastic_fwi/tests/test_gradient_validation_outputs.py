from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from elastic_fwi.gradient_observation_modes import (
    resolve_gradient_observation_mode,
    select_observation_functional_value,
)
from elastic_fwi.postprocessing import gradient_validation as gv
from elastic_fwi.postprocessing.gradient_validation import (
    PartitionGeometryChunk,
    canonical_cell_geometry_hashes,
    inspect_vtk_partition_integrity,
    taylor_order_estimates,
    validate_gradient_decomposition,
)
from elastic_fwi.postprocessing.vtk_cell_fields import (
    CellFieldChunk,
    VTKDataset,
)


def _field_chunk(values: np.ndarray) -> CellFieldChunk:
    values = np.asarray(values, dtype=np.float64)
    n = values.size
    return CellFieldChunk(
        source_piece=Path("piece.vtu"),
        cell_type="triangle",
        field_name="gradient_C66",
        centers=np.column_stack((np.arange(n, dtype=float), np.zeros(n))),
        values=values,
        cell_measures=np.arange(1, n + 1, dtype=np.float64),
        num_cells_before_ghost_filter=n,
        num_ghost_cells_removed=0,
    )


def test_observation_modes_select_true_scalar_functionals() -> None:
    values = {"reflection": 2.0, "transmission": 3.0, "full": 5.0}

    full = resolve_gradient_observation_mode("classical_full")
    reflection = resolve_gradient_observation_mode("standard_reflection_only")
    transmission = resolve_gradient_observation_mode("standard_transmission_only")

    assert full.reflection_weight == 1.0
    assert full.transmission_weight == 1.0
    assert select_observation_functional_value(values, full) == pytest.approx(5.0)
    assert select_observation_functional_value(values, reflection) == pytest.approx(2.0)
    assert select_observation_functional_value(values, transmission) == pytest.approx(3.0)

    with pytest.raises(ValueError, match="must not be subjected"):
        resolve_gradient_observation_mode("residual_interaction")


def test_geometry_hash_is_invariant_to_triangle_vertex_order() -> None:
    points = np.asarray(
        [[0.0, 0.0], [2.0, 0.0], [0.0, 1.0]],
        dtype=np.float64,
    )
    first = np.asarray([[0, 1, 2]], dtype=np.int64)
    permuted = np.asarray([[2, 0, 1]], dtype=np.int64)

    h1a, h2a = canonical_cell_geometry_hashes(
        points=points,
        connectivity=first,
        cell_type="VTK_LAGRANGE_TRIANGLE",
        coordinate_tolerance=1.0e-12,
    )
    h1b, h2b = canonical_cell_geometry_hashes(
        points=points,
        connectivity=permuted,
        cell_type="VTK_LAGRANGE_TRIANGLE",
        coordinate_tolerance=1.0e-12,
    )

    np.testing.assert_array_equal(h1a, h1b)
    np.testing.assert_array_equal(h2a, h2b)


def test_partition_integrity_uses_geometry_when_original_ids_are_piece_local(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = VTKDataset(
        requested_path=Path("field.pvd"),
        dataset_path=Path("field.pvtu"),
        piece_paths=(Path("p0.vtu"), Path("p1.vtu")),
        time_index=0,
    )
    monkeypatch.setattr(gv, "resolve_vtk_dataset", lambda *args, **kwargs: dataset)

    chunks = (
        PartitionGeometryChunk(
            source_piece=Path("p0.vtu"),
            cell_type="triangle",
            geometry_hash_primary=np.asarray([1, 2], dtype=np.uint64),
            geometry_hash_secondary=np.asarray([11, 12], dtype=np.uint64),
            cell_measures=np.asarray([1.0, 1.0]),
            original_cell_ids=np.asarray([0, 1]),
            num_cells_before_ghost_filter=2,
            num_ghost_cells_removed=0,
            ghost_array_present=False,
        ),
        PartitionGeometryChunk(
            source_piece=Path("p1.vtu"),
            cell_type="triangle",
            geometry_hash_primary=np.asarray([3, 4], dtype=np.uint64),
            geometry_hash_secondary=np.asarray([13, 14], dtype=np.uint64),
            cell_measures=np.asarray([1.0, 1.0]),
            original_cell_ids=np.asarray([0, 1]),
            num_cells_before_ghost_filter=2,
            num_ghost_cells_removed=0,
            ghost_array_present=False,
        ),
    )
    monkeypatch.setattr(
        gv,
        "iter_vtk_partition_geometry_chunks",
        lambda *args, **kwargs: iter(chunks),
    )

    report = inspect_vtk_partition_integrity(
        "field.pvd",
        expected_num_cells=4,
        expected_area=4.0,
    )

    assert report["status"] == "ok"
    assert report["duplicate_geometry_cells"] == 0
    assert report["original_cell_id_duplicate_cells"] == 2
    assert report["original_cell_id_scope"] == "piece_local_or_nonunique"


def test_partition_integrity_rejects_duplicate_geometry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = VTKDataset(
        requested_path=Path("field.pvd"),
        dataset_path=Path("field.pvtu"),
        piece_paths=(Path("p0.vtu"),),
        time_index=0,
    )
    monkeypatch.setattr(gv, "resolve_vtk_dataset", lambda *args, **kwargs: dataset)
    chunk = PartitionGeometryChunk(
        source_piece=Path("p0.vtu"),
        cell_type="triangle",
        geometry_hash_primary=np.asarray([7, 7], dtype=np.uint64),
        geometry_hash_secondary=np.asarray([9, 9], dtype=np.uint64),
        cell_measures=np.asarray([1.0, 1.0]),
        original_cell_ids=np.asarray([0, 1]),
        num_cells_before_ghost_filter=2,
        num_ghost_cells_removed=0,
        ghost_array_present=True,
    )
    monkeypatch.setattr(
        gv,
        "iter_vtk_partition_geometry_chunks",
        lambda *args, **kwargs: iter((chunk,)),
    )

    report = inspect_vtk_partition_integrity("field.pvd")

    assert report["status"] == "failed"
    assert report["duplicate_geometry_cells"] == 1
    assert report["num_unique_geometry_cells"] == 1


def test_gradient_decomposition_is_checked_pointwise_and_area_weighted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fields = {
        "classical": np.asarray([3.0, 1.0, -2.0]),
        "reflection": np.asarray([1.0, 2.0, -3.0]),
        "transmission": np.asarray([2.0, -1.0, 1.0]),
    }

    def iterator(path, field_name):
        del field_name
        yield _field_chunk(fields[str(path)])

    monkeypatch.setattr(gv, "iter_vtk_cell_field_chunks", iterator)
    report = validate_gradient_decomposition(
        classical_path="classical",
        reflection_path="reflection",
        transmission_path="transmission",
        field_name="gradient_C66",
        rtol=0.0,
        atol=0.0,
    )

    assert report["status"] == "ok"
    assert report["pointwise_allclose"] is True
    assert report["strict_pointwise_allclose"] is True
    assert report["solver_aware_validation"] is True
    assert report["acceptance_status"] == "OK_STRICT"
    assert report["relative_linf_error"] == pytest.approx(0.0)
    assert report["max_abs_residual"] == pytest.approx(0.0)
    assert report["residual_l2_error"] == pytest.approx(0.0)
    assert report["spatial_representation"] == "DG0_L2_cell_average_density"
    assert report["norm_measure"] == "cell_area_weighted"


def test_gradient_decomposition_accepts_solver_scale_residual_but_reports_strict_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fields = {
        "classical": np.asarray([1.0, -2.0, 0.5]),
        "reflection": np.asarray([0.4, -0.5, 0.2]),
        "transmission": np.asarray([0.6000002, -1.5000001, 0.3000001]),
    }

    def iterator(path, field_name):
        del field_name
        yield _field_chunk(fields[str(path)])

    monkeypatch.setattr(gv, "iter_vtk_cell_field_chunks", iterator)
    report = validate_gradient_decomposition(
        classical_path="classical",
        reflection_path="reflection",
        transmission_path="transmission",
        field_name="gradient_C66",
        rtol=1.0e-10,
        atol=1.0e-18,
        acceptance_relative_l2_tolerance=1.0e-6,
        acceptance_relative_linf_tolerance=1.0e-6,
    )

    assert report["status"] == "ok"
    assert report["pointwise_allclose"] is False
    assert report["strict_pointwise_allclose"] is False
    assert report["solver_aware_validation"] is True
    assert report["acceptance_status"] == "OK_SOLVER_TOLERANCE"
    assert report["relative_l2_error"] < 1.0e-6
    assert report["relative_linf_error"] < 1.0e-6
    assert "solver-aware" in report["message"]


def test_gradient_decomposition_detects_local_error_despite_small_integral(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fields = {
        "classical": np.asarray([1.0, -1.0]),
        "reflection": np.asarray([0.0, 0.0]),
        "transmission": np.asarray([0.9, -0.9]),
    }

    def iterator(path, field_name):
        del field_name
        chunk = _field_chunk(fields[str(path)])
        chunk = CellFieldChunk(
            **{**chunk.__dict__, "cell_measures": np.ones(2, dtype=np.float64)}
        )
        yield chunk

    monkeypatch.setattr(gv, "iter_vtk_cell_field_chunks", iterator)
    report = validate_gradient_decomposition(
        classical_path="classical",
        reflection_path="reflection",
        transmission_path="transmission",
        field_name="gradient_C66",
        rtol=0.0,
        atol=1.0e-12,
    )

    assert report["status"] == "failed"
    assert report["strict_pointwise_allclose"] is False
    assert report["solver_aware_validation"] is False
    assert report["acceptance_status"] == "FAILED"
    assert report["residual_signed_integral"] == pytest.approx(0.0)
    assert report["max_abs_residual"] == pytest.approx(0.1)
    assert report["residual_l2_error"] > 0.0


def test_taylor_order_estimate_recovers_quadratic_remainder() -> None:
    records = [
        {"eps": eps, "second_order_remainder": 3.0 * eps**2}
        for eps in (1.0e-1, 5.0e-2, 2.5e-2)
    ]
    orders = taylor_order_estimates(records)
    np.testing.assert_allclose(orders, np.asarray([2.0, 2.0]))


def test_t20_suite_preflight_reports_missing_campaigns(tmp_path: Path) -> None:
    from scripts.revision_wave_motion.validate_T20_gradient_suite import (
        _campaign_preflight,
    )

    line = tmp_path / "line"
    box = tmp_path / "box"
    line.mkdir()
    (line / "gradient_combination_campaign_manifest.json").write_text(
        "{}", encoding="utf-8"
    )

    available, missing = _campaign_preflight({"line": line, "box": box})

    assert set(available) == {"line"}
    assert len(missing) == 1
    assert missing[0]["campaign_label"] == "box"
    assert missing[0]["reason"] == "campaign_directory_missing"


def test_gradient_decomposition_converts_legacy_cell_integrals_to_l2_density(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    measures = np.asarray([0.1, 2.0, 0.5], dtype=np.float64)
    reflection_density = np.asarray([1.0, -2.0, 3.0])
    transmission_density = np.asarray([4.0, 1.0, -1.0])
    classical_density = reflection_density + transmission_density
    fields = {
        "classical": classical_density * measures,
        "reflection": reflection_density * measures,
        "transmission": transmission_density * measures,
    }

    def iterator(path, field_name):
        del field_name
        values = fields[str(path)]
        n = values.size
        yield CellFieldChunk(
            source_piece=Path("piece.vtu"),
            cell_type="triangle",
            field_name="gradient_C66",
            centers=np.column_stack((np.arange(n, dtype=float), np.zeros(n))),
            values=values,
            cell_measures=measures,
            num_cells_before_ghost_filter=n,
            num_ghost_cells_removed=0,
        )

    monkeypatch.setattr(gv, "iter_vtk_cell_field_chunks", iterator)
    report = validate_gradient_decomposition(
        classical_path="classical",
        reflection_path="reflection",
        transmission_path="transmission",
        field_name="gradient_C66",
        rtol=0.0,
        atol=0.0,
        input_semantics="cell-integral",
    )

    assert report["status"] == "ok"
    assert report["strict_pointwise_allclose"] is True
    assert report["input_semantics"] == "cell-integral"
    assert report["spatial_representation"] == "DG0_L2_cell_average_density"
    assert report["residual_l2_error"] == pytest.approx(0.0)
