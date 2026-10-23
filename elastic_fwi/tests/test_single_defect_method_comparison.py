from __future__ import annotations

import csv
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil

import meshio
import numpy as np
import pytest

from elastic_fwi.postprocessing.single_defect_method_comparison import (
    CardContractError,
    CardInput,
    CardKey,
    ComparisonConfig,
    ComparisonInputs,
    EXPECTED_METHODS,
    EXPECTED_PROJECT_PROPERTIES,
    ReflectionConsistencyError,
    SourceMetricError,
    apply_multi_criteria_ranking,
    check_reflection_only_consistency,
    compute_raw_signed_metrics_streaming,
    discover_comparison_inputs,
    read_source_metrics_csv,
    relative_gain,
    run_single_defect_method_comparison,
    validate_card_contract,
    write_comparison_outputs,
)
from elastic_fwi.postprocessing.vtk_cell_fields import (
    VTKCellFieldError,
    inspect_vtk_cell_field,
    iter_vtk_cell_field_chunks,
    resolve_vtk_dataset,
)


SOURCE_COLUMNS = [
    "campaign_name",
    "combination",
    "expected_field",
    "actual_field",
    "pvd_path",
    "data_path",
    "num_cells",
    "global_peak_value",
    "global_peak_x",
    "global_peak_y",
    "localization_error",
    "peak_in_defect",
    "peak_outside_defect",
    "peak_to_artifact_ratio",
    "mean_in_defect",
    "mean_outside_defect",
    "contrast_defect_to_background",
    "energy_in_defect",
    "energy_outside_defect",
    "energy_fraction_in_defect",
    "half_max_threshold",
    "half_max_area",
    "false_max_threshold",
    "false_maxima_count",
]


def _field_names() -> list[str]:
    return [
        *(f"gradient_{prop}" for prop in EXPECTED_PROJECT_PROPERTIES),
        *(f"residual_interaction_{prop}" for prop in EXPECTED_PROJECT_PROPERTIES),
        "material_gradient_proxy",
    ]


def _write_vtu(
    path: Path,
    *,
    x_offset: float,
    field_values: dict[str, float],
) -> None:
    points = np.asarray(
        [
            [x_offset + 0.0, 0.0, 0.0],
            [x_offset + 1.0, 0.0, 0.0],
            [x_offset + 0.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    cells = [("triangle", np.asarray([[0, 1, 2]], dtype=np.int64))]
    cell_data = {
        name: [np.asarray([value], dtype=float)] for name, value in field_values.items()
    }
    meshio.write(
        path,
        meshio.Mesh(points=points, cells=cells, cell_data=cell_data),
        binary=False,
    )


def _write_pvtu(path: Path, piece_names: list[str], fields: list[str]) -> None:
    pdata = "\n".join(
        f'      <PDataArray type="Float64" Name="{name}" NumberOfComponents="1"/>'
        for name in fields
    )
    pieces = "\n".join(f'    <Piece Source="{name}"/>' for name in piece_names)
    path.write_text(
        "\n".join(
            [
                '<?xml version="1.0"?>',
                '<VTKFile type="PUnstructuredGrid" version="1.0" byte_order="LittleEndian">',
                '  <PUnstructuredGrid GhostLevel="0">',
                '    <PCellData Scalars="scalars">',
                pdata,
                '    </PCellData>',
                '    <PPoints>',
                '      <PDataArray type="Float64" NumberOfComponents="3"/>',
                '    </PPoints>',
                pieces,
                '  </PUnstructuredGrid>',
                '</VTKFile>',
            ]
        ),
        encoding="utf-8",
    )


def _write_pvd(path: Path, dataset_name: str) -> None:
    path.write_text(
        "\n".join(
            [
                '<?xml version="1.0"?>',
                '<VTKFile type="Collection" version="1.0" byte_order="LittleEndian">',
                '  <Collection>',
                f'    <DataSet timestep="0" group="" part="0" file="{dataset_name}"/>',
                '  </Collection>',
                '</VTKFile>',
            ]
        ),
        encoding="utf-8",
    )


def _build_vtk_bundle(root: Path, *, positive_only: bool = False, negative_only: bool = False, zero: bool = False) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    fields = _field_names()
    if zero:
        first_value, second_value = 0.0, 0.0
    elif positive_only:
        first_value, second_value = 2.0, 1.0
    elif negative_only:
        first_value, second_value = -2.0, -1.0
    else:
        first_value, second_value = 2.0, -1.0
    first = {name: first_value for name in fields}
    second = {name: second_value for name in fields}
    _write_vtu(root / "piece0.vtu", x_offset=0.0, field_values=first)
    _write_vtu(root / "piece1.vtu", x_offset=2.0, field_values=second)
    _write_pvtu(root / "field.pvtu", ["piece0.vtu", "piece1.vtu"], fields)
    _write_pvd(root / "field.pvd", "field.pvtu")
    return root / "field.pvd"


def _required_fields(method: str) -> list[str]:
    prefix = "residual_interaction_" if method == "residual_interaction" else "gradient_"
    return [f"{prefix}{prop}" for prop in EXPECTED_PROJECT_PROPERTIES]


def _write_manifest(
    path: Path,
    *,
    pvd_path: Path,
    add_proxy: bool = False,
    remove: tuple[str, str, str] | None = None,
    duplicate: tuple[str, str, str] | None = None,
) -> None:
    sections: dict[str, dict] = {}
    for representation, section_name, combinations_key in (
        ("article_safe", "validation", "gradient_combinations"),
        ("raw_signed", "raw_signed_validation", "gradient_combinations_raw_signed"),
    ):
        found: dict[str, dict[str, str]] = {}
        required_by: dict[str, list[str]] = {}
        for method in EXPECTED_METHODS:
            required = _required_fields(method)
            required_by[method] = required
            fields = {field: str(pvd_path) for field in required}
            if remove == (representation, method, required[0]):
                fields.pop(required[0])
            if duplicate == (representation, method, required[0]):
                duplicate_name = (
                    f"residual_interaction_{EXPECTED_PROJECT_PROPERTIES[0]}"
                    if method != "residual_interaction"
                    else f"gradient_{EXPECTED_PROJECT_PROPERTIES[0]}"
                )
                fields[duplicate_name] = str(pvd_path)
            found[method] = fields
        if add_proxy:
            found["material_gradient_proxy"] = {
                "material_gradient_proxy": str(pvd_path)
            }
        sections[section_name] = {
            "combinations_key": combinations_key,
            "found": found,
            "missing": [],
            "num_expected_images": 20,
            "num_found_images": 20,
            "num_missing_images": 0,
            "required_combinations": list(EXPECTED_METHODS),
            "required_fields_by_combination": required_by,
            "status": "ok",
        }
    payload = {
        "kind": "gradient_combination_campaign",
        "status": "ok",
        "config": {
            "defect_center_x": 1.0 / 3.0,
            "defect_center_y": 1.0 / 3.0,
            "defect_radius": 0.6,
        },
        **sections,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _source_row(geometry: str, method: str, prop: str, pvd_path: Path) -> dict[str, str]:
    field = (
        f"residual_interaction_{prop}"
        if method == "residual_interaction"
        else f"gradient_{prop}"
    )
    method_index = EXPECTED_METHODS.index(method)
    property_index = EXPECTED_PROJECT_PROPERTIES.index(prop)
    base = 1.0 + 0.1 * method_index + 0.01 * property_index
    if geometry == "box" and method != "standard_reflection_only":
        base += 0.2
    localization = 1.0 / base
    energy_fraction = 0.1 * base
    par = 2.0 * base
    contrast = 3.0 * base
    false_count = 20 - method_index - property_index
    return {
        "campaign_name": f"synthetic_{geometry}",
        "combination": method,
        "expected_field": field,
        "actual_field": field,
        "pvd_path": str(pvd_path),
        "data_path": str(pvd_path.with_suffix(".pvtu")),
        "num_cells": "2",
        "global_peak_value": "1.0",
        "global_peak_x": "0.33333333333333331",
        "global_peak_y": "0.33333333333333331",
        "localization_error": format(localization, ".17g"),
        "peak_in_defect": format(base, ".17g"),
        "peak_outside_defect": "0.5",
        "peak_to_artifact_ratio": format(par, ".17g"),
        "mean_in_defect": format(base, ".17g"),
        "mean_outside_defect": "0.1",
        "contrast_defect_to_background": format(contrast, ".17g"),
        "energy_in_defect": format(base * base, ".17g"),
        "energy_outside_defect": format(10.0 / base, ".17g"),
        "energy_fraction_in_defect": format(energy_fraction, ".17g"),
        "half_max_threshold": "0.5",
        "half_max_area": format(1.0 / base, ".17g"),
        "false_max_threshold": "0.5",
        "false_maxima_count": str(false_count),
    }


def _write_metrics_csv(path: Path, *, geometry: str, pvd_path: Path, duplicate: bool = False) -> None:
    rows = [
        _source_row(geometry, method, prop, pvd_path)
        for method in EXPECTED_METHODS
        for prop in EXPECTED_PROJECT_PROPERTIES
    ]
    if duplicate:
        rows.append(dict(rows[0]))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SOURCE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _build_case(tmp_path: Path, *, add_proxy: bool = False) -> dict[str, Path]:
    pvd = _build_vtk_bundle(tmp_path / "vtk")
    line_manifest = tmp_path / "line_manifest.json"
    box_manifest = tmp_path / "box_manifest.json"
    _write_manifest(line_manifest, pvd_path=pvd, add_proxy=add_proxy)
    _write_manifest(box_manifest, pvd_path=pvd, add_proxy=add_proxy)
    line_csv = tmp_path / "line_metrics.csv"
    box_csv = tmp_path / "box_metrics.csv"
    _write_metrics_csv(line_csv, geometry="line", pvd_path=pvd)
    _write_metrics_csv(box_csv, geometry="box", pvd_path=pvd)
    return {
        "pvd": pvd,
        "line_manifest": line_manifest,
        "box_manifest": box_manifest,
        "line_csv": line_csv,
        "box_csv": box_csv,
    }


def _config(case: dict[str, Path], output_dir: Path, **kwargs) -> ComparisonConfig:
    return ComparisonConfig(
        inputs=ComparisonInputs(
            line_metrics_csv=case["line_csv"],
            box_metrics_csv=case["box_csv"],
            line_manifest=case["line_manifest"],
            box_manifest=case["box_manifest"],
        ),
        output_dir=output_dir,
        execution_timestamp="2026-07-03T00:00:00+00:00",
        **kwargs,
    )


def _card(pvd: Path, *, field: str = "gradient_rho") -> CardInput:
    return CardInput(
        key=CardKey("line", "raw_signed", "classical_full", "rho"),
        manifest_path=pvd.parent / "manifest.json",
        declared_path=str(pvd),
        pvd_path=pvd,
        expected_field_name=field,
    )


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()




@pytest.fixture(scope="module")
def pipeline_bundle(tmp_path_factory):
    root = tmp_path_factory.mktemp("single_defect_pipeline")
    case = _build_case(root)
    hashes_before = {name: _hash(path) for name, path in case.items() if path.is_file()}
    config = _config(case, root / "base_output")
    result = run_single_defect_method_comparison(config)
    hashes_after = {name: _hash(path) for name, path in case.items() if path.is_file()}
    return {
        "root": root,
        "case": case,
        "config": config,
        "result": result,
        "hashes_before": hashes_before,
        "hashes_after": hashes_after,
    }


def test_discovery_exactly_80_keys(tmp_path: Path) -> None:
    case = _build_case(tmp_path)
    cards, methods, properties, _, _, _ = discover_comparison_inputs(
        case["line_manifest"], case["box_manifest"]
    )
    assert len(cards) == 80
    assert len({card.key for card in cards}) == 80
    assert methods == EXPECTED_METHODS
    assert properties == EXPECTED_PROJECT_PROPERTIES


def test_missing_card_is_reported(tmp_path: Path) -> None:
    case = _build_case(tmp_path)
    _write_manifest(
        case["line_manifest"],
        pvd_path=case["pvd"],
        remove=("raw_signed", "classical_full", "gradient_rho"),
    )
    with pytest.raises(CardContractError, match="missing properties"):
        discover_comparison_inputs(case["line_manifest"], case["box_manifest"])


def test_duplicate_logical_card_is_reported(tmp_path: Path) -> None:
    case = _build_case(tmp_path)
    _write_manifest(
        case["line_manifest"],
        pvd_path=case["pvd"],
        duplicate=("article_safe", "classical_full", "gradient_rho"),
    )
    with pytest.raises(CardContractError, match="Duplicate logical card"):
        discover_comparison_inputs(case["line_manifest"], case["box_manifest"])


def test_material_gradient_proxy_is_explicitly_excluded(tmp_path: Path) -> None:
    case = _build_case(tmp_path, add_proxy=True)
    cards, _, _, exclusions, _, _ = discover_comparison_inputs(
        case["line_manifest"], case["box_manifest"]
    )
    assert len(cards) == 80
    assert exclusions
    assert all("material_gradient_proxy" in item["name"] for item in exclusions)
    assert all("material_gradient_proxy" not in card.expected_field_name for card in cards)


def test_four_methods_are_identified_from_manifests(tmp_path: Path) -> None:
    case = _build_case(tmp_path)
    _, methods, properties, _, _, _ = discover_comparison_inputs(
        case["line_manifest"], case["box_manifest"]
    )
    assert methods == EXPECTED_METHODS
    assert properties == ("rho", "C11", "C22", "C12", "C66")


def test_representations_remain_separate(tmp_path: Path) -> None:
    case = _build_case(tmp_path)
    cards, *_ = discover_comparison_inputs(case["line_manifest"], case["box_manifest"])
    assert sum(card.key.representation == "article_safe" for card in cards) == 40
    assert sum(card.key.representation == "raw_signed" for card in cards) == 40


def test_source_metrics_duplicate_is_rejected(tmp_path: Path) -> None:
    pvd = _build_vtk_bundle(tmp_path / "vtk")
    csv_path = tmp_path / "duplicate.csv"
    _write_metrics_csv(csv_path, geometry="line", pvd_path=pvd, duplicate=True)
    with pytest.raises(SourceMetricError, match="Duplicate source metric key"):
        read_source_metrics_csv(csv_path, geometry="line")


def test_source_metric_strings_are_preserved_exactly(tmp_path: Path) -> None:
    case = _build_case(tmp_path)
    records = read_source_metrics_csv(case["line_csv"], geometry="line")
    key = CardKey("line", "article_safe", "classical_full", "rho")
    assert records[key].exact_metrics["global_peak_x"] == "0.33333333333333331"


def test_streaming_signed_statistics_mixed_signs(tmp_path: Path) -> None:
    pvd = _build_vtk_bundle(tmp_path / "vtk")
    metrics = compute_raw_signed_metrics_streaming(
        _card(pvd),
        defect_center=(1.0 / 3.0, 1.0 / 3.0),
        defect_radius=0.6,
        expected_num_cells=2,
    )
    assert metrics["signed_minimum"] == pytest.approx(-1.0)
    assert metrics["signed_maximum"] == pytest.approx(2.0)
    assert metrics["positive_sum"] == pytest.approx(1.0)
    assert metrics["negative_sum_abs"] == pytest.approx(0.5)
    assert metrics["positive_cell_fraction"] == pytest.approx(0.5)
    assert metrics["negative_cell_fraction"] == pytest.approx(0.5)
    assert metrics["positive_quadratic_energy"] == pytest.approx(2.0)
    assert metrics["negative_quadratic_energy"] == pytest.approx(0.5)
    assert metrics["positive_to_negative_sum_ratio"] == pytest.approx(2.0)
    assert metrics["max_positive_in_defect"] == pytest.approx(2.0)
    assert metrics["min_negative_outside_defect"] == pytest.approx(-1.0)


def test_streaming_signed_statistics_entirely_positive(tmp_path: Path) -> None:
    pvd = _build_vtk_bundle(tmp_path / "vtk", positive_only=True)
    metrics = compute_raw_signed_metrics_streaming(
        _card(pvd), defect_center=(1 / 3, 1 / 3), defect_radius=0.6, expected_num_cells=2
    )
    assert metrics["negative_cell_fraction"] == 0.0
    assert metrics["positive_to_negative_sum_ratio"] is None
    assert metrics["positive_to_negative_ratio_status"] == "undefined_zero_denominator"


def test_streaming_signed_statistics_entirely_negative(tmp_path: Path) -> None:
    pvd = _build_vtk_bundle(tmp_path / "vtk", negative_only=True)
    metrics = compute_raw_signed_metrics_streaming(
        _card(pvd), defect_center=(1 / 3, 1 / 3), defect_radius=0.6, expected_num_cells=2
    )
    assert metrics["positive_cell_fraction"] == 0.0
    assert metrics["positive_to_negative_sum_ratio"] == 0.0


def test_streaming_signed_statistics_zero_map(tmp_path: Path) -> None:
    pvd = _build_vtk_bundle(tmp_path / "vtk", zero=True)
    metrics = compute_raw_signed_metrics_streaming(
        _card(pvd), defect_center=(1 / 3, 1 / 3), defect_radius=0.6, expected_num_cells=2
    )
    assert metrics["metric_status"] == "zero_image"
    assert metrics["positive_to_negative_sum_ratio"] == 0.0
    assert metrics["false_maxima_count"] == 0
    assert metrics["half_max_area"] == 0.0


def test_relative_gain_higher_is_better() -> None:
    assert relative_gain(2.0, 3.0, direction="higher_is_better", epsilon=1e-15) == pytest.approx(0.5)


def test_relative_gain_lower_is_better() -> None:
    assert relative_gain(2.0, 1.0, direction="lower_is_better", epsilon=1e-15) == pytest.approx(0.5)


def test_multi_criteria_ranking_is_deterministic() -> None:
    rows = []
    for geometry_index, geometry in enumerate(("line", "box")):
        for method_index, method in enumerate(EXPECTED_METHODS):
            for prop in EXPECTED_PROJECT_PROPERTIES:
                base = 1.0 + geometry_index + method_index
                rows.append(
                    {
                        "geometry": geometry,
                        "method": method,
                        "property": prop,
                        "article_safe__localization_error": 1.0 / base,
                        "article_safe__energy_fraction_in_defect": base,
                        "article_safe__peak_to_artifact_ratio": base,
                        "article_safe__contrast_defect_to_background": base,
                        "article_safe__false_maxima_count": 10.0 / base,
                    }
                )
    first = apply_multi_criteria_ranking(rows, weights={
        "localization_error": 0.25,
        "energy_fraction_in_defect": 0.25,
        "peak_to_artifact_ratio": 0.2,
        "contrast_defect_to_background": 0.15,
        "false_maxima_count": 0.15,
    }, atol=1e-14, rtol=1e-12)
    rows_copy = [{k: v for k, v in row.items() if not k.startswith("ranking__")} for row in rows]
    second = apply_multi_criteria_ranking(rows_copy, weights={
        "localization_error": 0.25,
        "energy_fraction_in_defect": 0.25,
        "peak_to_artifact_ratio": 0.2,
        "contrast_defect_to_background": 0.15,
        "false_maxima_count": 0.15,
    }, atol=1e-14, rtol=1e-12)
    assert first == second


def test_reflection_only_control_passes(pipeline_bundle) -> None:
    result = pipeline_bundle["result"]
    assert result.reflection_control["status"] == "ok"
    assert result.reflection_control["num_failures"] == 0


def test_reflection_only_control_failure_is_detailed(pipeline_bundle) -> None:
    result = pipeline_bundle["result"]
    rows = [dict(row) for row in result.consolidated_rows]
    target = next(
        row for row in rows
        if row["geometry"] == "box"
        and row["method"] == "standard_reflection_only"
        and row["property"] == "rho"
    )
    target["article_safe__localization_error"] = "99"
    with pytest.raises(ReflectionConsistencyError, match="property=rho.*representation=article_safe.*metric=localization_error"):
        check_reflection_only_consistency(rows, rtol=1e-12, atol=1e-14, raise_on_failure=True)


def test_incremental_single_vtu_read(tmp_path: Path) -> None:
    pvd = _build_vtk_bundle(tmp_path / "vtk")
    single = pvd.parent / "piece0.vtu"
    chunks = list(iter_vtk_cell_field_chunks(single, "gradient_rho"))
    assert len(chunks) == 1
    assert chunks[0].num_cells == 1
    assert chunks[0].values.tolist() == [2.0]


def test_incremental_multipart_pvtu_read(tmp_path: Path) -> None:
    pvd = _build_vtk_bundle(tmp_path / "vtk")
    dataset = resolve_vtk_dataset(pvd)
    assert len(dataset.piece_paths) == 2
    inspection = inspect_vtk_cell_field(pvd, "gradient_rho", expected_num_cells=2)
    assert inspection.num_pieces == 2
    assert inspection.num_cells == 2
    chunks = list(iter_vtk_cell_field_chunks(pvd, "gradient_rho"))
    assert [chunk.source_piece.name for chunk in chunks] == ["piece0.vtu", "piece1.vtu"]


def test_vtk_reader_rejects_ambiguous_field_request(tmp_path: Path) -> None:
    pvd = _build_vtk_bundle(tmp_path / "vtk")
    with pytest.raises(VTKCellFieldError, match="Ambiguous"):
        list(iter_vtk_cell_field_chunks(pvd, ("gradient_rho", "gradient_C11")))


def test_full_pipeline_produces_40_rows_and_does_not_modify_inputs(pipeline_bundle) -> None:
    result = pipeline_bundle["result"]
    assert len(result.cards) == 80
    assert len(result.consolidated_rows) == 40
    assert len({(r["geometry"], r["method"], r["property"]) for r in result.consolidated_rows}) == 40
    assert pipeline_bundle["hashes_before"] == pipeline_bundle["hashes_after"]


def test_csv_json_latex_and_figures_are_serialized(tmp_path: Path, pipeline_bundle) -> None:
    result = pipeline_bundle["result"]
    config = replace(pipeline_bundle["config"], output_dir=tmp_path / "out")
    paths = write_comparison_outputs(result, config, command=["python", "synthetic"])
    for name in ("csv", "latex", "summary", "manifest", "log", "figure_1", "figure_2", "figure_3"):
        assert paths[name].exists(), name
    with paths["csv"].open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 40
    assert "article_safe__localization_error" in rows[0]
    assert "raw_signed__positive_sum" in rows[0]
    assert "ranking__composite_score" in rows[0]
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    assert summary["num_cards_validated"] == 80
    assert summary["num_consolidated_rows"] == 40
    assert "\\begin{longtable}" in paths["latex"].read_text(encoding="utf-8")


def test_reproducible_core_outputs_at_identical_inputs(tmp_path: Path, pipeline_bundle) -> None:
    output = tmp_path / "out"
    config = replace(pipeline_bundle["config"], output_dir=output, overwrite=True)
    result = pipeline_bundle["result"]
    first_paths = write_comparison_outputs(result, config, command=["python", "synthetic"])
    first_bytes = {
        name: first_paths[name].read_bytes() for name in ("csv", "latex", "summary")
    }
    second_paths = write_comparison_outputs(result, config, command=["python", "synthetic"])
    second_bytes = {
        name: second_paths[name].read_bytes() for name in ("csv", "latex", "summary")
    }
    assert first_bytes == second_bytes
