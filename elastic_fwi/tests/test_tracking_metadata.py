from __future__ import annotations

import argparse
import json
from pathlib import Path

from elastic_fwi.tracking.run_metadata import (
    SCHEMA_VERSION,
    build_tracking_metadata,
    stable_hash,
    write_tracking_sidecars,
)
from elastic_fwi.tracking.timers import TimingRecorder
from elastic_fwi.tracking.tracked_manifest import enrich_manifest_with_tracking


def test_stable_hash_is_reproducible():
    assert stable_hash("abc") == stable_hash("abc")
    assert stable_hash("abc") != stable_hash("abcd")


def test_build_tracking_metadata_contains_core_fields(tmp_path):
    args = argparse.Namespace(quick=True, seed=0)

    tracking = build_tracking_metadata(
        output_dir=tmp_path,
        args=args,
    )

    assert tracking["schema_version"] == SCHEMA_VERSION
    assert tracking["command"]["command_line_hash"]
    assert tracking["environment"]["python_executable"]
    assert "cpu_count" in tracking["hardware"]
    assert tracking["parallel_runtime"]["size"] >= 1
    assert tracking["args"]["quick"] is True


def test_write_tracking_sidecars_creates_expected_files(tmp_path):
    tracking = build_tracking_metadata(
        output_dir=tmp_path,
        args=argparse.Namespace(quick=True),
    )

    files = write_tracking_sidecars(
        output_dir=tmp_path,
        tracking=tracking,
    )

    for path in files.values():
        assert Path(path).exists()

    command_payload = json.loads((tmp_path / "raw" / "command_line.json").read_text())
    assert command_payload["command_line_hash"] == tracking["command"]["command_line_hash"]


def test_enrich_manifest_with_tracking_creates_tracking_block_and_sidecars(tmp_path):
    manifest = {
        "kind": "test_campaign",
        "status": "ok",
        "output_dir": str(tmp_path),
        "metrics_summary_csv": str(tmp_path / "metrics_summary.csv"),
        "figures": {},
        "num_metric_rows": 0,
    }

    enriched = enrich_manifest_with_tracking(
        manifest,
        output_dir=tmp_path,
        args=argparse.Namespace(quick=True),
    )

    assert enriched["tracking"]["schema_version"] == SCHEMA_VERSION
    assert Path(enriched["tracking"]["files"]["command_line_txt"]).exists()
    assert Path(enriched["tracking"]["files"]["environment_json"]).exists()
    assert Path(enriched["tracking"]["files"]["hardware_json"]).exists()
    assert Path(enriched["tracking"]["files"]["git_info_json"]).exists()
    assert Path(enriched["tracking"]["files"]["parallel_runtime_json"]).exists()
    assert Path(enriched["tracking"]["files"]["run_log"]).exists()


def test_timing_recorder_writes_csv(tmp_path):
    recorder = TimingRecorder()

    with recorder.section("section_a"):
        value = 1 + 1

    assert value == 2
    assert recorder.rows
    assert recorder.rows[0]["section"] == "section_a"
    assert recorder.rows[0]["status"] == "ok"

    path = recorder.write_csv(tmp_path / "timings.csv")
    assert path.exists()
    assert "section_a" in path.read_text()


def test_enrich_manifest_writes_timings_csv(tmp_path):
    import argparse
    from pathlib import Path

    from elastic_fwi.tracking.tracked_manifest import enrich_manifest_with_tracking

    manifest = {
        "kind": "timing_campaign",
        "status": "ok",
        "output_dir": str(tmp_path),
        "metrics_summary_csv": str(tmp_path / "metrics_summary.csv"),
        "figures": {},
        "num_metric_rows": 0,
    }

    enriched = enrich_manifest_with_tracking(
        manifest,
        output_dir=tmp_path,
        args=argparse.Namespace(quick=True),
        elapsed_seconds=1.25,
    )

    assert "timings_csv" in enriched
    timings_path = Path(enriched["timings_csv"])
    assert timings_path.exists()

    text = timings_path.read_text()
    assert "total" in text
    assert "tracking_enrichment" in text

    files = enriched["tracking"]["files"]
    assert Path(files["timings_csv"]).exists()


def test_write_failed_manifest_creates_traceback_and_tracking(tmp_path):
    import argparse
    from pathlib import Path

    from elastic_fwi.tracking.tracked_manifest import write_failed_manifest

    try:
        raise RuntimeError("synthetic failure for tracking test")
    except RuntimeError as exc:
        failed = write_failed_manifest(
            output_dir=tmp_path,
            kind="failed_tracking_test",
            error=exc,
            args=argparse.Namespace(quick=True),
        )

    assert failed["status"] == "failed"
    assert failed["kind"] == "failed_tracking_test"
    assert failed["error_type"] == "RuntimeError"
    assert "synthetic failure" in failed["error_message"]

    manifest_path = tmp_path / "campaign_manifest.json"
    assert manifest_path.exists()

    traceback_path = Path(failed["traceback_file"])
    assert traceback_path.exists()
    assert "RuntimeError" in traceback_path.read_text()

    files = failed["tracking"]["files"]
    assert Path(files["command_line_txt"]).exists()
    assert Path(files["environment_json"]).exists()
    assert Path(files["hardware_json"]).exists()
    assert Path(files["git_info_json"]).exists()
    assert Path(files["parallel_runtime_json"]).exists()
    assert Path(files["run_log"]).exists()
    assert Path(files["timings_csv"]).exists()


def test_enrich_manifest_writes_timings_csv(tmp_path):
    import argparse
    from pathlib import Path

    from elastic_fwi.tracking.tracked_manifest import enrich_manifest_with_tracking

    manifest = {
        "kind": "timing_campaign",
        "status": "ok",
        "output_dir": str(tmp_path),
        "metrics_summary_csv": str(tmp_path / "metrics_summary.csv"),
        "figures": {},
        "num_metric_rows": 0,
    }

    enriched = enrich_manifest_with_tracking(
        manifest,
        output_dir=tmp_path,
        args=argparse.Namespace(quick=True),
        elapsed_seconds=1.25,
    )

    assert "timings_csv" in enriched
    timings_path = Path(enriched["timings_csv"])
    assert timings_path.exists()

    text = timings_path.read_text()
    assert "total" in text
    assert "tracking_enrichment" in text

    files = enriched["tracking"]["files"]
    assert Path(files["timings_csv"]).exists()


def test_write_failed_manifest_creates_traceback_and_tracking(tmp_path):
    import argparse
    from pathlib import Path

    from elastic_fwi.tracking.tracked_manifest import write_failed_manifest

    try:
        raise RuntimeError("synthetic failure for tracking test")
    except RuntimeError as exc:
        failed = write_failed_manifest(
            output_dir=tmp_path,
            kind="failed_tracking_test",
            error=exc,
            args=argparse.Namespace(quick=True),
        )

    assert failed["status"] == "failed"
    assert failed["kind"] == "failed_tracking_test"
    assert failed["error_type"] == "RuntimeError"
    assert "synthetic failure" in failed["error_message"]

    manifest_path = tmp_path / "campaign_manifest.json"
    assert manifest_path.exists()

    traceback_path = Path(failed["traceback_file"])
    assert traceback_path.exists()
    assert "RuntimeError" in traceback_path.read_text()

    files = failed["tracking"]["files"]
    assert Path(files["command_line_txt"]).exists()
    assert Path(files["environment_json"]).exists()
    assert Path(files["hardware_json"]).exists()
    assert Path(files["git_info_json"]).exists()
    assert Path(files["parallel_runtime_json"]).exists()
    assert Path(files["run_log"]).exists()
    assert Path(files["timings_csv"]).exists()
