import pytest

pytest.importorskip("dolfinx")

from elastic_fwi.diagnostics.check_traction_source import run_traction_source_check


def test_traction_source_diagnostic_report_is_ok():
    report = run_traction_source_check(nx=4, ny=4)

    assert report["status"] == "ok"

    for check_name, check_value in report["checks"].items():
        assert check_value, check_name

    values = report["values"]

    assert values["constant_work"] == pytest.approx(-3.0)
    assert values["stronger_norm"] == pytest.approx(36.0)

    assert values["pulse_value_before"] == pytest.approx([0.0, 0.0])
    assert values["pulse_value_start"] == pytest.approx([0.0, 0.0])
    assert values["pulse_value_mid"] == pytest.approx([0.0, 1.0])
    assert values["pulse_value_end"] == pytest.approx([0.0, 0.0])
    assert values["pulse_value_after"] == pytest.approx([0.0, 0.0])

    assert values["pulse_norm_mid"] == pytest.approx(1.0)

    assert values["rhs_global_size"] == values["expected_global_size"]
    assert values["rhs_norm"] > 0.0
