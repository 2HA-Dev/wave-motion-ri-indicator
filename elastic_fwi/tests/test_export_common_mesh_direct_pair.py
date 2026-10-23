import json
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest
from mpi4py import MPI

pytest.importorskip("dolfinx")

from elastic_fwi.diagnostics.export_common_mesh_direct_pair import (
    export_common_mesh_direct_pair,
)


def test_common_mesh_direct_pair_exports_two_compatible_pvd_series(
    tmp_path: Path,
):
    template = tmp_path / "template.json"
    template.write_text(
        json.dumps(
            {
                "kind": "test_template",
                "nx": 4,
                "ny": 4,
                "degree": 1,
                "mesh_kind": "structured",
                "domain_length_x": 1.0,
                "domain_length_y": 1.0,
                "theta": 0.25,
                "dt": 0.1,
                "num_steps": 3,
                "every_k": 1,
                "source_frequency": 1.0,
                "source_num_periods": 1.0,
                "ksp_type": "cg",
                "pc_type": "jacobi",
                "ksp_rtol": 1.0e-10,
                "ksp_atol": 1.0e-12,
                "ksp_max_it": 1000,
                "simulated_material_kind": "healthy",
                "observed_material_kind": "localized_disk_defect",
                "defect_center_x": 0.5,
                "defect_center_y": 0.5,
                "defect_radius": 0.35,
                "contrast_factor": 0.8,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    output_dir = tmp_path / "pair"

    report = export_common_mesh_direct_pair(
        template_manifest=template,
        output_dir=output_dir,
    )

    assert report["status"] == "ok"

    statuses = MPI.COMM_WORLD.allgather(report["status"])
    assert statuses == ["ok"] * MPI.COMM_WORLD.size

    output_dirs = MPI.COMM_WORLD.allgather(
        report["output_dir"]
    )
    assert len(set(output_dirs)) == 1

    assert report["common_mesh"] is True
    assert report["mesh_created_once"] is True
    assert report["shared_mesh_object"] is True
    assert report["shared_function_space"] is True
    assert report["same_mpi_partition"] is True
    assert report["direct_solves"] == 2
    assert report["adjoint_solves"] == 0
    assert report["num_snapshots_exported"] == 3
    assert report["exported_times"] == pytest.approx(
        [0.0, 0.1, 0.2]
    )

    simulated_pvd = Path(
        report["exports"]["direct_simulated"]["pvd"]
    )
    observed_pvd = Path(
        report["exports"]["direct_observed"]["pvd"]
    )
    manifest = Path(report["manifest"])

    assert simulated_pvd.is_file()
    assert observed_pvd.is_file()
    assert manifest.is_file()

    simulated_datasets = (
        ET.parse(simulated_pvd)
        .getroot()
        .findall(".//DataSet")
    )
    observed_datasets = (
        ET.parse(observed_pvd)
        .getroot()
        .findall(".//DataSet")
    )

    assert len(simulated_datasets) == 3
    assert len(observed_datasets) == 3

    simulated_times = [
        float(item.attrib["timestep"])
        for item in simulated_datasets
    ]
    observed_times = [
        float(item.attrib["timestep"])
        for item in observed_datasets
    ]

    assert simulated_times == observed_times

    payload = json.loads(manifest.read_text(encoding="utf-8"))

    assert payload["status"] == "ok"
    assert payload["checks"]["same_export_times"] is True
    assert payload["checks"]["simulated_pvd_exists"] is True
    assert payload["checks"]["observed_pvd_exists"] is True
    assert payload["mesh"]["fingerprint"]["global_owned_cells"] > 0
