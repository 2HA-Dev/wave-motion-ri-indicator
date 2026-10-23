from __future__ import annotations

import dataclasses
import inspect
import math
from pathlib import Path

from elastic_fwi.campaigns.run_gradient_combination_campaign import (
    _config_for_args,
    build_parser,
)
from elastic_fwi.diagnostics.export_direct_time_series_xdmf import (
    export_direct_time_series_xdmf,
)
from elastic_fwi.diagnostics.export_inverse_time_series_xdmf import (
    export_inverse_time_series_xdmf,
)
from elastic_fwi.pipelines.direct_observation import (
    DirectObservationPipelineConfig,
    _build_gmsh_refinement_disks,
)


LINE_PRESET = (
    "article_L120_gmsh_p1_ppw15_near40_"
    "multidefect_T42_no_absorbing"
)

BOX_PRESET = (
    "article_L120_gmsh_p1_ppw15_near40_"
    "multidefect_box_except_gamma1_T42_no_absorbing"
)

SCENARIO_PATH = Path(
    "configs/defects/"
    "article_L120_multidefect_5_elastic_only.json"
)


def _preset_config(preset: str) -> dict[str, object]:
    args = build_parser().parse_args(
        [
            "--output-dir",
            "/tmp/article_L120_multidefect_preset_test",
            "--preset",
            preset,
        ]
    )
    return _config_for_args(args)


def _pipeline_config(
    campaign_config: dict[str, object],
    *,
    material_kind: str,
) -> DirectObservationPipelineConfig:
    valid_names = {
        field.name
        for field in dataclasses.fields(DirectObservationPipelineConfig)
    }
    filtered = {
        key: value
        for key, value in campaign_config.items()
        if key in valid_names
    }
    filtered["material_kind"] = material_kind
    return DirectObservationPipelineConfig(**filtered)


def test_exporters_expose_L120_mesh_extension_arguments() -> None:
    for function in (
        export_direct_time_series_xdmf,
        export_inverse_time_series_xdmf,
    ):
        parameters = inspect.signature(function).parameters
        assert "mesh_refine_from_defects_json" in parameters
        assert "mesh_near_stiffness_factor" in parameters
        assert parameters["mesh_refine_from_defects_json"].default is False
        assert parameters["mesh_near_stiffness_factor"].default == 1.0


def test_article_L120_line_preset_is_locked() -> None:
    config = _preset_config(LINE_PRESET)

    assert SCENARIO_PATH.is_file()
    assert config["degree"] == 1
    assert config["mesh_kind"] == "gmsh_refined"
    assert config["mesh_auto_from_wavelength"] is True
    assert config["mesh_target_ppw"] == 15.0
    assert config["mesh_near_target_ppw"] == 40.0
    assert config["mesh_near_stiffness_factor"] == 0.5
    assert config["mesh_refine_defect"] is True
    assert config["mesh_refine_from_defects_json"] is True
    assert config["mesh_refine_defect_radius"] is None
    assert config["nx"] == 1767
    assert config["ny"] == 1767

    assert config["domain_length_x"] == 120.0
    assert config["domain_length_y"] == 120.0
    assert config["absorbing_layer"] is False
    assert config["absorbing_layer_width"] is None

    assert config["source_array_num_elements"] == 32
    assert config["receiver_array_num_elements"] == 32
    assert math.isclose(config["source_sigma"], 10.0 / 31.0)
    assert math.isclose(config["receiver_sigma"], 10.0 / 31.0)

    assert (
        config["source_array_start_x"],
        config["source_array_start_y"],
    ) == (50.0, 50.0)
    assert (
        config["source_array_end_x"],
        config["source_array_end_y"],
    ) == (70.0, 50.0)
    assert (
        config["receiver_array_start_x"],
        config["receiver_array_start_y"],
    ) == (50.0, 70.0)
    assert (
        config["receiver_array_end_x"],
        config["receiver_array_end_y"],
    ) == (70.0, 70.0)
    assert config["receiver_array_geometry"] == "line"

    assert config["simulated_material_kind"] == "healthy"
    assert config["observed_material_kind"] == "multi_disk_defects"
    assert config["defects_json"] == str(SCENARIO_PATH)
    assert config["defect_background_factor"] == 1.0
    assert config["defect_conflict_policy"] == "error"

    assert config["dt"] == 0.05
    assert config["num_steps"] == 840
    assert math.isclose(config["dt"] * config["num_steps"], 42.0)
    assert config["every_k"] == 10


def test_article_L120_box_preset_is_locked() -> None:
    config = _preset_config(BOX_PRESET)

    assert config["receiver_array_geometry"] == "box_except_gamma1"
    assert config["receiver_box_xmin"] == 50.0
    assert config["receiver_box_xmax"] == 70.0
    assert config["receiver_box_ymin"] == 50.0
    assert config["receiver_box_ymax"] == 70.0
    assert config["receiver_box_left_num_elements"] == 32
    assert config["receiver_box_top_num_elements"] == 32
    assert config["receiver_box_right_num_elements"] == 32

    assert (
        config["receiver_array_start_x"],
        config["receiver_array_start_y"],
    ) == (50.0, 70.0)
    assert (
        config["receiver_array_end_x"],
        config["receiver_array_end_y"],
    ) == (70.0, 70.0)

    assert config["mesh_refine_from_defects_json"] is True
    assert config["mesh_near_stiffness_factor"] == 0.5
    assert config["observed_material_kind"] == "multi_disk_defects"
    assert math.isclose(config["dt"] * config["num_steps"], 42.0)


def test_L120_presets_build_identical_healthy_and_defective_mesh_plans() -> None:
    for preset in (LINE_PRESET, BOX_PRESET):
        campaign_config = _preset_config(preset)

        healthy = _pipeline_config(
            campaign_config,
            material_kind="healthy",
        )
        defective = _pipeline_config(
            campaign_config,
            material_kind="multi_disk_defects",
        )

        healthy_disks = _build_gmsh_refinement_disks(healthy)
        defective_disks = _build_gmsh_refinement_disks(defective)

        assert healthy_disks == defective_disks
        assert len(healthy_disks) == 5
        assert math.isclose(healthy.mesh_far_h, 0.0679142523997161)
        assert math.isclose(healthy.mesh_near_h, 0.018008485654145256)
        assert math.isclose(defective.mesh_far_h, healthy.mesh_far_h)
        assert math.isclose(defective.mesh_near_h, healthy.mesh_near_h)

        assert [disk.radius for disk in healthy_disks] == [
            1.5,
            1.2000000000000002,
            2.0999999999999996,
            2.4000000000000004,
            1.7999999999999998,
        ]


def test_article_L120_quick_mode_only_shortens_time_loop() -> None:
    args = build_parser().parse_args(
        [
            "--output-dir",
            "/tmp/article_L120_multidefect_quick_test",
            "--preset",
            LINE_PRESET,
            "--quick",
        ]
    )
    config = _config_for_args(args)

    assert config["num_steps"] == 10
    assert config["every_k"] == 1
    assert config["domain_length_x"] == 120.0
    assert config["domain_length_y"] == 120.0
    assert config["mesh_near_stiffness_factor"] == 0.5
    assert config["mesh_refine_from_defects_json"] is True
    assert config["defects_json"] == str(SCENARIO_PATH)
