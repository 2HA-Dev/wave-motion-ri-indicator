import json

import numpy as np
import pytest

pytest.importorskip("dolfinx")

from elastic_fwi.diagnostics.check_localized_defect_functionals import (
    run_localized_defect_functionals_check,
)


def test_localized_defect_functionals_diagnostic_report_is_ok():
    report = run_localized_defect_functionals_check(
        nx=4,
        ny=4,
        degree=1,
        theta=0.25,
        dt=0.1,
        num_steps=5,
        defect_center_x=0.5,
        defect_center_y=0.5,
        defect_radius=0.35,
        contrast_factor=0.8,
        reflection_weight=2.0,
        transmission_weight=3.0,
    )

    assert report["status"] == "ok"

    for check_name, check_value in report["checks"].items():
        assert check_value, check_name

    assert report["config"]["global_defect_cells"] > 0

    values = report["functionals"]["healthy_vs_localized"]

    assert values["reflection"] > 0.0
    assert values["transmission"] > 0.0
    assert values["full"] == pytest.approx(
        values["reflection"] + values["transmission"]
    )
    assert values["weighted"] == pytest.approx(
        2.0 * values["reflection"] + 3.0 * values["transmission"]
    )

    assert np.isfinite(report["values"]["max_abs_observation_difference"])
    assert report["values"]["max_abs_observation_difference"] > 0.0

    assert report["healthy_data"]["reflection"]["tag"] == 1
    assert report["healthy_data"]["transmission"]["tag"] == 2
    assert report["localized_data"]["reflection"]["tag"] == 1
    assert report["localized_data"]["transmission"]["tag"] == 2


def test_localized_defect_functionals_report_is_json_serializable():
    report = run_localized_defect_functionals_check(
        nx=4,
        ny=4,
        degree=1,
        theta=0.25,
        dt=0.1,
        num_steps=5,
        defect_radius=0.35,
        contrast_factor=0.8,
    )

    payload = json.dumps(report, sort_keys=True)

    assert '"status": "ok"' in payload
    assert '"healthy_vs_localized"' in payload
    assert '"localized_disk_defect"' in payload
