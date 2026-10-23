from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from elastic_fwi.postprocessing.dg0_l2_sensitivity_validation import (
    DefectGeometry,
    _expected_20_contract,
    compute_field_representation_metrics,
)
from elastic_fwi.postprocessing.signed_sensitivity_fields import DiscoveredSignedField
from elastic_fwi.postprocessing.vtk_cell_fields import CellFieldChunk


def _chunk(values: np.ndarray, measures: np.ndarray) -> CellFieldChunk:
    return CellFieldChunk(
        source_piece=Path("piece.vtu"),
        cell_type="triangle",
        field_name="gradient_rho",
        centers=np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64),
        values=np.asarray(values, dtype=np.float64),
        cell_measures=np.asarray(measures, dtype=np.float64),
        num_cells_before_ghost_filter=2,
        num_ghost_cells_removed=0,
    )


def test_cell_integral_audit_recovers_l2_density_peak_at_defect() -> None:
    measures = np.asarray([0.1, 1.0])
    density = np.asarray([10.0, 2.0])
    cell_integrals = density * measures

    report = compute_field_representation_metrics(
        [_chunk(cell_integrals, measures)],
        defect=DefectGeometry(0.0, 0.0, 0.2),
        input_semantics="cell-integral",
    )

    assert report.integrated_peak_x == pytest.approx(1.0)
    assert report.integrated_peak_distance_to_defect == pytest.approx(1.0)
    assert report.density_peak_x == pytest.approx(0.0)
    assert report.density_peak_distance_to_defect == pytest.approx(0.0)
    assert report.density_peak_inside_defect is True
    assert report.density_inside_outside_peak_ratio == pytest.approx(5.0)
    assert report.cell_measure_ratio == pytest.approx(10.0)


def test_corrected_l2_density_semantics_preserve_same_physical_density() -> None:
    measures = np.asarray([0.1, 1.0])
    density = np.asarray([10.0, 2.0])

    report = compute_field_representation_metrics(
        [_chunk(density, measures)],
        defect=DefectGeometry(0.0, 0.0, 0.2),
        input_semantics="l2-density",
    )

    assert report.integrated_peak_x == pytest.approx(1.0)
    assert report.density_peak_x == pytest.approx(0.0)
    assert report.density_peak_abs == pytest.approx(10.0)
    assert report.integrated_peak_abs == pytest.approx(2.0)


def test_expected_20_map_contract_requires_four_modalities_times_five_parameters() -> None:
    modalities = (
        "classical_full",
        "standard_reflection_only",
        "standard_transmission_only",
        "residual_interaction",
    )
    parameters = ("rho", "C11", "C22", "C12", "C66")
    fields = tuple(
        DiscoveredSignedField(
            geometry_id="line",
            campaign_id="campaign",
            field_id=f"campaign__{mode}__{parameter}",
            parameter_name=parameter,
            observation_mode=mode,
            input_path=Path(f"{mode}_{parameter}.pvd"),
            article_safe_candidates=(),
            manifest_path=None,
        )
        for mode in modalities
        for parameter in parameters
    )

    report = _expected_20_contract(fields)

    assert report["status"] == "ok"
    assert report["num_fields"] == 20
    assert report["missing"] == []
    assert report["extra"] == []
    assert report["duplicates"] == 0


def test_campaign_validation_processes_exactly_20_maps(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import elastic_fwi.postprocessing.dg0_l2_sensitivity_validation as module
    from elastic_fwi.postprocessing.signed_sensitivity_fields import DiscoveryResult

    modalities = (
        "classical_full",
        "standard_reflection_only",
        "standard_transmission_only",
        "residual_interaction",
    )
    parameters = ("rho", "C11", "C22", "C12", "C66")
    fields = tuple(
        DiscoveredSignedField(
            geometry_id="line",
            campaign_id="campaign",
            field_id=f"campaign__{mode}__{parameter}",
            parameter_name=parameter,
            observation_mode=mode,
            input_path=tmp_path / f"{mode}_{parameter}.pvd",
            article_safe_candidates=(),
            manifest_path=tmp_path / "manifest.json",
            configuration={
                "defect_center_x": 0.0,
                "defect_center_y": 0.0,
                "defect_radius": 0.2,
            },
        )
        for mode in modalities
        for parameter in parameters
    )
    monkeypatch.setattr(
        module,
        "discover_signed_fields",
        lambda config: DiscoveryResult(
            fields=fields,
            manifests=(tmp_path / "manifest.json",),
            ignored=(),
            warnings=(),
        ),
    )
    monkeypatch.setattr(
        module,
        "iter_vtk_cell_field_chunks",
        lambda *args, **kwargs: iter(
            (_chunk(np.asarray([1.0, 2.0]), np.asarray([0.1, 1.0])),)
        ),
    )

    report = module.validate_campaign_20_maps(
        campaign_dir=tmp_path,
        output_dir=tmp_path / "out",
        manifest_path=tmp_path / "manifest.json",
        input_semantics="cell-integral",
    )

    assert report["status"] == "ok"
    assert report["num_processed"] == 20
    assert report["num_ok"] == 20
    assert Path(report["csv_path"]).is_file()
    assert Path(report["json_path"]).is_file()
