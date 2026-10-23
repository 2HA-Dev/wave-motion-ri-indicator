import json

import numpy as np
import pytest

pytest.importorskip("dolfinx")

from elastic_fwi.pipelines.adjoint_newmark import (
    AdjointNewmarkPipelineConfig,
    run_adjoint_newmark_pipeline,
)


def test_adjoint_newmark_pipeline_default_case():
    result = run_adjoint_newmark_pipeline(
        AdjointNewmarkPipelineConfig(
            nx=4,
            ny=4,
            degree=1,
            theta=0.25,
            dt=0.1,
            num_steps=5,
            simulated_material_kind="healthy",
            observed_material_kind="localized_disk_defect",
            contrast_factor=0.8,
            defect_radius=0.35,
            ksp_type="cg",
            pc_type="jacobi",
            ksp_rtol=1.0e-10,
            ksp_atol=1.0e-12,
            ksp_max_it=1000,
        )
    )

    try:
        assert result.mpi_size >= 1

        report = result.as_dict()

        assert report["kind"] == "adjoint_newmark_pipeline_result"

        assert report["comparison"]["status"] == "ok"
        assert report["comparison"]["common_mesh"]["common_mesh"] is True
        assert report["comparison"]["common_mesh"]["shared_mesh_object"] is True
        assert report["comparison"]["common_mesh"]["shared_function_space"] is True
        assert report["comparison"]["common_mesh"]["same_mpi_partition"] is True
        assert report["comparison"]["simulated"]["state"][
            "displacement_snapshots_stored"
        ] is True
        assert report["comparison"]["simulated"]["state"][
            "num_displacement_snapshots"
        ] == 5

        assert report["comparison"]["values"]["observed_defect_cells"] > 0

        assert report["adjoint_rhs"]["kind"] == "full_adjoint_observation_rhs_result"
        assert report["adjoint_rhs"]["num_steps"] == 5
        assert report["adjoint_rhs"]["max_rhs_norm"] > 0.0

        assert report["newmark_adjoint"]["kind"] == "newmark_adjoint_state_result"
        assert report["newmark_adjoint"]["num_steps"] == 5
        assert report["newmark_adjoint"]["reverse_time"] is True
        assert report["newmark_adjoint"]["all_ksp_converged"] is True
        assert report["newmark_adjoint"]["max_equation_adjoint_norm"] > 0.0
        assert report["newmark_adjoint"]["max_linear_residual_norm"] <= 1.0e-8

        records = report["newmark_adjoint"]["records"]

        assert records[0]["step"] == 1
        assert records[0]["solve_index"] == 5
        assert records[-1]["step"] == 5
        assert records[-1]["solve_index"] == 1

        for record in records:
            assert record["ksp_converged"] is True
            assert record["ksp_iterations"] > 0
            assert np.isfinite(record["observation_rhs_norm"])
            assert np.isfinite(record["accumulated_rhs_norm"])
            assert np.isfinite(record["equation_adjoint_norm"])
            assert np.isfinite(record["linear_residual_norm"])
            assert record["linear_residual_norm"] <= 1.0e-8

        assert report["matrix"]["rows"] == report["matrix"]["cols"]
        assert report["matrix"]["rows"] > 0
        assert report["matrix"]["local_rows"] >= 0
        assert report["matrix"]["local_cols"] >= 0

        assert report["values"]["max_abs_residual"] > 0.0
        assert report["values"]["max_rhs_norm"] > 0.0
        assert report["values"]["max_equation_adjoint_norm"] > 0.0
        assert np.isfinite(report["values"]["max_linear_residual_norm"])
        assert np.isfinite(report["values"]["initial_current_adjoint_norm"])
        assert np.isfinite(report["values"]["initial_previous_adjoint_norm"])

    finally:
        result.destroy()


def test_adjoint_newmark_pipeline_report_is_json_serializable():
    result = run_adjoint_newmark_pipeline(
        AdjointNewmarkPipelineConfig(
            nx=4,
            ny=4,
            degree=1,
            theta=0.25,
            dt=0.1,
            num_steps=5,
        )
    )

    try:
        payload = json.dumps(result.as_dict(), sort_keys=True)

        assert '"adjoint_newmark_pipeline_result"' in payload
        assert '"direct_observation_comparison_result"' in payload
        assert '"full_adjoint_observation_rhs_result"' in payload
        assert '"newmark_adjoint_state_result"' in payload

    finally:
        result.destroy()
