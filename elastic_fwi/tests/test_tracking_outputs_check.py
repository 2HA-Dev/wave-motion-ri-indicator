from __future__ import annotations

import argparse
import json
from pathlib import Path

from elastic_fwi.diagnostics.check_tracking_outputs import (
    build_tracking_outputs_report,
)
from elastic_fwi.tracking.tracked_manifest import enrich_manifest_with_tracking


def test_tracking_outputs_report_detects_tracked_manifest(tmp_path):
    manifest = {
        "kind": "tracked_case",
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

    manifest_path = tmp_path / "campaign_manifest.json"
    manifest_path.write_text(json.dumps(enriched), encoding="utf-8")

    report = build_tracking_outputs_report(root=tmp_path, require_tracking=True)

    assert report["status"] == "ok"
    assert report["num_manifests"] == 1
    assert report["num_ok"] == 1
    assert report["num_failed"] == 0


def test_tracking_outputs_report_detects_missing_tracking(tmp_path):
    manifest_path = tmp_path / "campaign_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "kind": "legacy_manifest",
                "status": "ok",
            }
        ),
        encoding="utf-8",
    )

    report = build_tracking_outputs_report(root=tmp_path, require_tracking=True)

    assert report["status"] == "failed"
    assert report["num_missing_tracking"] == 1
    assert report["records"][0]["reason"] == "missing_tracking"
