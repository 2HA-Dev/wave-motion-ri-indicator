import pytest

pytest.importorskip("dolfinx")

from elastic_fwi.diagnostics.check_material_dg0 import run_material_dg0_check


def test_material_dg0_diagnostic_report_is_ok():
    report = run_material_dg0_check(nx=2, ny=2)

    assert report["status"] == "ok"
    assert report["expected_names_ok"]
    assert report["homogeneous_constant_ok"]
    assert report["defective_constant_ok"]
    assert report["defective_from_indices_constant_ok"]
    assert report["global_indices_all_zero"]
    assert report["all_positive"]

    for block_name in ("homogeneous", "defective", "defective_from_indices"):
        block = report[block_name]
        assert set(block) == {"rho", "C11", "C22", "C12", "C66"}

        for stats in block.values():
            assert stats["global_count"] > 0
            assert stats["global_nonfinite"] == 0
            assert stats["global_min"] is not None
            assert stats["global_max"] is not None
