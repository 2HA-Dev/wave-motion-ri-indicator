from __future__ import annotations

import math

from elastic_fwi.campaigns.run_gradient_combination_campaign import (
    _config_for_args,
    build_parser,
)
from elastic_fwi.materials.orthotropic import LEGACY_HEALTHY_MATERIAL
from elastic_fwi.meshing import plan_wavelength_mesh_sizing


def test_article_L73_gmsh_p1_ppw15_near40_box_except_gamma1_T26_no_absorbing_preset() -> None:
    preset = "article_L73_gmsh_p1_ppw15_near40_box_except_gamma1_T26_no_absorbing"

    args = build_parser().parse_args(
        [
            "--output-dir",
            "/tmp/article_L73_gmsh_box_test",
            "--preset",
            preset,
        ]
    )
    config = _config_for_args(args)

    assert config["degree"] == 1
    assert config["mesh_kind"] == "gmsh_refined"
    assert config["mesh_auto_from_wavelength"] is True
    assert config["mesh_target_ppw"] == 15.0
    assert config["mesh_near_target_ppw"] == 40.0
    assert config["mesh_refine_defect"] is True
    assert config["mesh_refine_defect_radius"] == 0.75

    assert config["domain_length_x"] == 73.0
    assert config["domain_length_y"] == 73.0
    assert config["absorbing_layer"] is False
    assert config["absorbing_layer_width"] is None

    assert (config["source_array_start_x"], config["source_array_start_y"]) == (30.0, 30.0)
    assert (config["source_array_end_x"], config["source_array_end_y"]) == (43.0, 30.0)

    assert config["receiver_array_geometry"] == "box_except_gamma1"
    assert config["receiver_box_xmin"] == 30.0
    assert config["receiver_box_xmax"] == 43.0
    assert config["receiver_box_ymin"] == 30.0
    assert config["receiver_box_ymax"] == 43.0
    assert config["receiver_box_left_num_elements"] == 21
    assert config["receiver_box_top_num_elements"] == 21
    assert config["receiver_box_right_num_elements"] == 21

    assert (config["receiver_array_start_x"], config["receiver_array_start_y"]) == (30.0, 43.0)
    assert (config["receiver_array_end_x"], config["receiver_array_end_y"]) == (43.0, 43.0)

    assert (config["defect_center_x"], config["defect_center_y"]) == (36.5, 36.5)
    assert config["defect_radius"] == 0.25
    assert config["contrast_factor"] == 0.75
    assert math.isclose(config["dt"] * config["num_steps"], 26.0)

    plan = plan_wavelength_mesh_sizing(
        domain_length_x=config["domain_length_x"],
        domain_length_y=config["domain_length_y"],
        source_frequency=config["source_frequency"],
        rho=LEGACY_HEALTHY_MATERIAL.rho,
        C11=LEGACY_HEALTHY_MATERIAL.C11,
        C22=LEGACY_HEALTHY_MATERIAL.C22,
        C12=LEGACY_HEALTHY_MATERIAL.C12,
        C66=LEGACY_HEALTHY_MATERIAL.C66,
        target_ppw=config["mesh_target_ppw"],
        near_target_ppw=config["mesh_near_target_ppw"],
    )

    assert math.isclose(plan.lambda_min, 1.0187137859957416)
    assert math.isclose(plan.mesh_far_h, 0.0679142523997161)
    assert math.isclose(plan.mesh_near_h, 0.025467844649893538)
    assert math.isclose(
        2.0 * config["defect_radius"] / plan.mesh_near_h,
        19.632599730110655,
    )
