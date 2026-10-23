from __future__ import annotations

import math
from pathlib import Path

import pytest

from elastic_fwi.pipelines.direct_observation import (
    DirectObservationPipelineConfig,
    _build_gmsh_refinement_disks,
)


SCENARIO_PATH = Path(
    "configs/defects/"
    "article_L120_multidefect_5_elastic_only.json"
)


def _common_config() -> dict[str, object]:
    return {
        "nx": 16,
        "ny": 16,
        "degree": 1,
        "mesh_kind": "gmsh_refined",
        "mesh_auto_from_wavelength": True,
        "mesh_target_ppw": 15.0,
        "mesh_near_target_ppw": 40.0,
        "mesh_near_stiffness_factor": 0.5,
        "mesh_refine_defect": True,
        "mesh_refine_from_defects_json": True,
        "mesh_refine_defect_radius": None,
        "domain_length_x": 120.0,
        "domain_length_y": 120.0,
        "source_frequency": 1.0,
        "defects_json": str(SCENARIO_PATH),
        "defect_background_factor": 1.0,
        "defect_conflict_policy": "error",
    }


def test_L120_near_h_uses_softest_stiffness_factor() -> None:
    config = DirectObservationPipelineConfig(
        **_common_config(),
        material_kind="healthy",
    )

    assert math.isclose(
        config.mesh_far_h,
        0.0679142523997161,
    )
    assert math.isclose(
        config.mesh_near_h,
        0.018008485654145256,
    )

    sizing = config.wavelength_mesh_sizing
    assert sizing is not None
    assert math.isclose(
        sizing["base_mesh_near_h"],
        0.025467844649893538,
    )
    assert math.isclose(
        sizing["mesh_near_h"],
        config.mesh_near_h,
    )
    assert sizing["mesh_near_stiffness_factor"] == 0.5


def test_L120_healthy_and_defective_meshes_use_identical_disks() -> None:
    healthy = DirectObservationPipelineConfig(
        **_common_config(),
        material_kind="healthy",
    )
    defective = DirectObservationPipelineConfig(
        **_common_config(),
        material_kind="multi_disk_defects",
    )

    healthy_disks = _build_gmsh_refinement_disks(healthy)
    defective_disks = _build_gmsh_refinement_disks(defective)

    assert healthy_disks == defective_disks
    assert len(healthy_disks) == 5

    expected = [
        (60.0, 60.0, 1.5),
        (56.0, 56.0, 1.2),
        (64.0, 56.0, 2.1),
        (56.0, 64.0, 2.4),
        (64.0, 64.0, 1.8),
    ]

    for disk, (x, y, radius) in zip(
        healthy_disks,
        expected,
        strict=True,
    ):
        assert math.isclose(disk.center_x, x)
        assert math.isclose(disk.center_y, y)
        assert math.isclose(disk.radius, radius)
        assert math.isclose(
            disk.target_h,
            0.018008485654145256,
        )


def test_mesh_extension_defaults_preserve_legacy_behavior() -> None:
    config = DirectObservationPipelineConfig(
        mesh_kind="gmsh_refined",
        mesh_auto_from_wavelength=True,
        mesh_target_ppw=15.0,
        mesh_near_target_ppw=40.0,
        mesh_refine_defect=True,
        material_kind="localized_disk_defect",
        defect_center_x=36.5,
        defect_center_y=36.5,
        defect_radius=0.25,
        domain_length_x=73.0,
        domain_length_y=73.0,
        source_frequency=1.0,
    )

    assert config.mesh_refine_from_defects_json is False
    assert config.mesh_near_stiffness_factor == 1.0
    assert math.isclose(
        config.mesh_near_h,
        0.025467844649893538,
    )

    disks = _build_gmsh_refinement_disks(config)
    assert len(disks) == 1
    assert disks[0].name == "defect"
    assert math.isclose(disks[0].radius, 0.75)


def test_explicit_near_h_overrides_stiffness_adjustment() -> None:
    config = DirectObservationPipelineConfig(
        **{
            **_common_config(),
            "mesh_near_h": 0.02,
        },
        material_kind="healthy",
    )

    assert config.mesh_near_h == 0.02
    disks = _build_gmsh_refinement_disks(config)
    assert all(disk.target_h == 0.02 for disk in disks)


def test_mesh_refine_from_json_requires_defect_refinement() -> None:
    with pytest.raises(
        ValueError,
        match="requires mesh_refine_defect=True",
    ):
        DirectObservationPipelineConfig(
            mesh_refine_from_defects_json=True,
            defects_json=str(SCENARIO_PATH),
        )


def test_mesh_refine_from_json_requires_json_path() -> None:
    with pytest.raises(
        ValueError,
        match="requires defects_json",
    ):
        DirectObservationPipelineConfig(
            mesh_refine_defect=True,
            mesh_refine_from_defects_json=True,
        )


def test_mesh_near_stiffness_factor_must_be_positive() -> None:
    with pytest.raises(
        ValueError,
        match="mesh_near_stiffness_factor",
    ):
        DirectObservationPipelineConfig(
            mesh_near_stiffness_factor=0.0,
        )
