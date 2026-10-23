import json
from pathlib import Path

from mpi4py import MPI

from elastic_fwi.materials.defect_json import load_defect_scenario_json
from elastic_fwi.pipelines.direct_observation import (
    DirectObservationPipelineConfig,
    run_direct_observation_pipeline,
)


def _write_defects_json(path: Path) -> Path:
    data = {
        "name": "test_multi_defects",
        "background_factor": 1.0,
        "conflict_policy": "last",
        "defects": [
            {
                "name": "A",
                "center_x": 5.0,
                "center_y": 5.0,
                "radius": 1.0,
                "material_factor": 0.8,
            },
            {
                "name": "B",
                "center_x": 3.0,
                "center_y": 7.0,
                "radius": 0.6,
                "material_factor": 0.7,
            },
        ],
    }

    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_load_defect_scenario_json(tmp_path: Path):
    json_path = _write_defects_json(tmp_path / "defects.json")

    scenario, metadata = load_defect_scenario_json(json_path)

    assert scenario.name == "test_multi_defects"
    assert len(scenario.defects) == 2
    assert metadata["num_defects"] == 2
    assert metadata["defects"][0]["center_x"] == 5.0
    assert metadata["defects"][1]["material_factor"] == 0.7


def test_direct_pipeline_accepts_multi_disk_defects(tmp_path: Path):
    comm = MPI.COMM_WORLD

    json_path = _write_defects_json(tmp_path / "defects.json")

    config = DirectObservationPipelineConfig(
        nx=10,
        ny=10,
        degree=1,
        domain_length_x=10.0,
        domain_length_y=10.0,
        source_segment_xmin=4.0,
        source_segment_xmax=6.0,
        receiver_segment_xmin=4.0,
        receiver_segment_xmax=6.0,
        material_kind="multi_disk_defects",
        defects_json=str(json_path),
        defect_background_factor=1.0,
        defect_conflict_policy="last",
        dt=0.05,
        num_steps=3,
        source_frequency=1.0,
        store_displacement_snapshots=True,
    )

    result = run_direct_observation_pipeline(config=config, comm=comm)

    assert result.material_metadata["kind"] == "multi_disk_defects"
    assert result.material_metadata["scenario"]["num_defects"] == 2
    assert result.material_metadata["scenario"]["defects"][0]["name"] == "A"
    assert result.material_metadata["scenario"]["defects"][1]["name"] == "B"
    assert result.material_metadata["conflict_policy"] == "last"

    assert result.has_displacement_snapshots
    assert result.num_displacement_snapshots == 3
