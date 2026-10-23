from __future__ import annotations

from elastic_fwi.materials.defects import CircularDefectSpec, DefectScenario
from elastic_fwi.materials.orthotropic import (
    LEGACY_HEALTHY_MATERIAL,
    OrthotropicPlaneStrainConstants,
    legacy_defect_material,
)


LEGACY_DEFECT_MATERIAL_FACTOR = 0.8
LEGACY_DEFECT_NORM_ORDER = 2


LEGACY_DEFECTS: tuple[CircularDefectSpec, ...] = (
    CircularDefectSpec(
        name="legacy_defect_1",
        center=(82.0, 82.0),
        radius=0.2,
        material_factor=LEGACY_DEFECT_MATERIAL_FACTOR,
        norm_order=LEGACY_DEFECT_NORM_ORDER,
    ),
    CircularDefectSpec(
        name="legacy_defect_2",
        center=(82.0, 98.0),
        radius=0.4,
        material_factor=LEGACY_DEFECT_MATERIAL_FACTOR,
        norm_order=LEGACY_DEFECT_NORM_ORDER,
    ),
    CircularDefectSpec(
        name="legacy_defect_3",
        center=(90.0, 90.0),
        radius=0.28,
        material_factor=LEGACY_DEFECT_MATERIAL_FACTOR,
        norm_order=LEGACY_DEFECT_NORM_ORDER,
    ),
    CircularDefectSpec(
        name="legacy_defect_4",
        center=(98.0, 82.0),
        radius=0.24,
        material_factor=LEGACY_DEFECT_MATERIAL_FACTOR,
        norm_order=LEGACY_DEFECT_NORM_ORDER,
    ),
    CircularDefectSpec(
        name="legacy_defect_5",
        center=(98.0, 98.0),
        radius=0.36,
        material_factor=LEGACY_DEFECT_MATERIAL_FACTOR,
        norm_order=LEGACY_DEFECT_NORM_ORDER,
    ),
)


LEGACY_DEFECT_SCENARIO = DefectScenario(
    name="legacy_cpp_five_circular_defects",
    defects=LEGACY_DEFECTS,
)


def legacy_defect_specs() -> tuple[CircularDefectSpec, ...]:
    return LEGACY_DEFECTS


def legacy_defect_scenario() -> DefectScenario:
    return LEGACY_DEFECT_SCENARIO


def legacy_healthy_material() -> OrthotropicPlaneStrainConstants:
    return LEGACY_HEALTHY_MATERIAL


def legacy_defective_material() -> OrthotropicPlaneStrainConstants:
    return legacy_defect_material(LEGACY_DEFECT_MATERIAL_FACTOR)
