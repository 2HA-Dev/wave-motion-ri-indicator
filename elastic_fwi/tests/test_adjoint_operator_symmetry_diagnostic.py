import json

import numpy as np
import pytest

pytest.importorskip("dolfinx")

from elastic_fwi.diagnostics.check_adjoint_operator_symmetry import (
    run_adjoint_operator_symmetry_check,
)


def test_adjoint_operator_symmetry_diagnostic_healthy_material():
    report = run_adjoint_operator_symmetry_check(
        nx=4,
        ny=4,
        degree=1,
        theta=0.25,
        dt=0.1,
        material_kind="healthy",
        tolerance=1.0e-10,
    )

    assert report["status"] == "ok"

    for check_name, check_value in report["checks"].items():
        assert check_value, check_name

    assert report["kind"] == "adjoint_operator_symmetry_check"
    assert report["values"]["effective_rows"] == report["values"]["effective_cols"]
    assert report["values"]["effective_rows"] > 0
    assert report["values"]["effective_frobenius_norm"] > 0.0

    for matrix_name in ("mass", "stiffness", "effective"):
        matrix = report["matrices"][matrix_name]

        assert matrix["square"] is True
        assert matrix["local_shape_valid"] is True
        assert matrix["finite"] is True
        assert matrix["symmetric_in_duality"] is True
        assert matrix["duality_abs"] <= matrix["tolerance"] * max(
            abs(matrix["duality_left_y_Ax"]),
            abs(matrix["duality_right_x_ATy"]),
            1.0,
        )
        assert np.isfinite(matrix["duality_rel"])


def test_adjoint_operator_symmetry_diagnostic_localized_material():
    report = run_adjoint_operator_symmetry_check(
        nx=4,
        ny=4,
        degree=1,
        theta=0.25,
        dt=0.1,
        material_kind="localized_disk_defect",
        contrast_factor=0.8,
        defect_radius=0.35,
        tolerance=1.0e-10,
    )

    assert report["status"] == "ok"

    for check_name, check_value in report["checks"].items():
        assert check_value, check_name

    assert report["material"]["kind"] == "localized_disk_defect"
    assert report["material"]["global_defect_cells"] > 0
    assert report["matrices"]["effective"]["symmetric_in_duality"] is True


def test_adjoint_operator_symmetry_report_is_json_serializable():
    report = run_adjoint_operator_symmetry_check(
        nx=4,
        ny=4,
        degree=1,
        theta=0.25,
        dt=0.1,
        material_kind="healthy",
        tolerance=1.0e-10,
    )

    payload = json.dumps(report, sort_keys=True)

    assert '"adjoint_operator_symmetry_check"' in payload
    assert '"effective_symmetric_in_duality"' in payload
    assert '"mass"' in payload
    assert '"stiffness"' in payload
