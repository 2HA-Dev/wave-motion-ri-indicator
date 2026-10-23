import pytest

pytest.importorskip("dolfinx")

from elastic_fwi.diagnostics.check_boundary_tags import run_boundary_tags_check


def test_boundary_tags_diagnostic_report_is_ok():
    report = run_boundary_tags_check(nx=4, ny=4)

    assert report["status"] == "ok"
    assert report["tags_present"]
    assert report["counts_positive"]
    assert report["measures_ok"]

    assert report["expected_tags"] == [1, 2, 3, 4]

    for tag in (1, 2, 3, 4):
        assert report["counts"][tag] > 0
        assert report["measures"][tag] == pytest.approx(1.0)
