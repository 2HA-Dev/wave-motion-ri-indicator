import pytest

from elastic_fwi.materials.orthotropic import (
    LEGACY_HEALTHY_MATERIAL,
    OrthotropicPlaneStrainConstants,
    legacy_defect_material,
)


def test_legacy_healthy_material_values():
    mat = LEGACY_HEALTHY_MATERIAL

    assert mat.rho == pytest.approx(4.5)
    assert mat.C11 == pytest.approx(16.24)
    assert mat.C22 == pytest.approx(18.07)
    assert mat.C12 == pytest.approx(6.9)
    assert mat.C66 == pytest.approx(4.67)


def test_legacy_defect_material_default_is_twenty_percent_reduction():
    defect = legacy_defect_material()

    assert defect.rho == pytest.approx(0.8 * 4.5)
    assert defect.C11 == pytest.approx(0.8 * 16.24)
    assert defect.C22 == pytest.approx(0.8 * 18.07)
    assert defect.C12 == pytest.approx(0.8 * 6.9)
    assert defect.C66 == pytest.approx(0.8 * 4.67)


def test_material_scaling_rejects_non_positive_factor():
    mat = OrthotropicPlaneStrainConstants(
        rho=1.0,
        C11=2.0,
        C22=3.0,
        C12=0.5,
        C66=0.8,
    )

    with pytest.raises(ValueError):
        mat.scaled(0.0)

    with pytest.raises(ValueError):
        mat.scaled(-1.0)
