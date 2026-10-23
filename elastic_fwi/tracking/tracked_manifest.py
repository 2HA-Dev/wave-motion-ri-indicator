from __future__ import annotations

import argparse
import csv
import json
import time
import traceback
from pathlib import Path
from typing import Any, Mapping

from elastic_fwi.tracking.run_logger import append_run_log
from elastic_fwi.tracking.run_metadata import (
    build_tracking_metadata,
    to_jsonable,
    utc_now_iso,
    write_json,
    write_tracking_sidecars,
)


def _write_timings_csv(
    *,
    output_dir: str | Path,
    rows: list[dict[str, Any]],
) -> Path:
    output_dir = Path(output_dir)
    path = output_dir / "timings.csv"
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "section",
        "elapsed_seconds",
        "status",
    ]

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    "section": row.get("section", "unknown"),
                    "elapsed_seconds": float(row.get("elapsed_seconds", 0.0)),
                    "status": row.get("status", "ok"),
                }
            )

    return path


def _default_timing_rows(
    *,
    elapsed_seconds: float | None,
    tracking_elapsed_seconds: float,
    status: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    rows.append(
        {
            "section": "tracking_enrichment",
            "elapsed_seconds": float(tracking_elapsed_seconds),
            "status": "ok",
        }
    )

    if elapsed_seconds is not None:
        rows.insert(
            0,
            {
                "section": "total",
                "elapsed_seconds": float(elapsed_seconds),
                "status": status,
            },
        )

    return rows


def enrich_manifest_with_tracking(
    manifest: Mapping[str, Any],
    *,
    output_dir: str | Path,
    args: argparse.Namespace | Mapping[str, Any] | None = None,
    comm: Any | None = None,
    started_at_utc: str | None = None,
    finished_at_utc: str | None = None,
    elapsed_seconds: float | None = None,
    timing_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Enrich a manifest with reproducibility metadata and write sidecars.

    This function is intentionally lightweight. It does not run numerical
    computations. It only writes provenance files next to the manifest.
    """
    t0 = time.perf_counter()
    payload = dict(manifest)

    status = str(payload.get("status", "unknown"))
    output_dir = Path(output_dir)

    started = started_at_utc or utc_now_iso()
    finished = finished_at_utc or utc_now_iso()

    tracking = build_tracking_metadata(
        output_dir=output_dir,
        args=args,
        comm=comm,
        started_at_utc=started,
        finished_at_utc=finished,
        elapsed_seconds=elapsed_seconds,
        status=status,
    )

    sidecars = write_tracking_sidecars(
        output_dir=output_dir,
        tracking=tracking,
    )
    tracking["files"].update(sidecars)

    tracking_elapsed = time.perf_counter() - t0

    if timing_rows is None:
        timing_rows = _default_timing_rows(
            elapsed_seconds=elapsed_seconds,
            tracking_elapsed_seconds=tracking_elapsed,
            status=status,
        )
    else:
        timing_rows = list(timing_rows)
        timing_rows.append(
            {
                "section": "tracking_enrichment",
                "elapsed_seconds": tracking_elapsed,
                "status": "ok",
            }
        )

    timings_csv = _write_timings_csv(
        output_dir=output_dir,
        rows=timing_rows,
    )

    tracking["files"]["timings_csv"] = str(timings_csv)
    payload["timings_csv"] = str(timings_csv)
    payload["tracking"] = tracking

    append_run_log(
        output_dir,
        [
            f"manifest kind={payload.get('kind')}",
            f"manifest status={payload.get('status')}",
            f"command_line_hash={tracking['command']['command_line_hash']}",
            f"timings_csv={timings_csv}",
        ],
    )

    return payload


def write_failed_manifest(
    *,
    output_dir: str | Path,
    kind: str,
    error: BaseException,
    args: argparse.Namespace | Mapping[str, Any] | None = None,
    comm: Any | None = None,
    manifest_filename: str = "campaign_manifest.json",
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Write a failed manifest with traceback and tracking sidecars.

    Intended for campaign/diagnostic entry points. It gives a reproducible
    failure record instead of leaving an empty output directory after an
    exception.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    traceback_path = output_dir / "logs" / "traceback.txt"
    traceback_path.parent.mkdir(parents=True, exist_ok=True)
    traceback_path.write_text(
        "".join(traceback.format_exception(type(error), error, error.__traceback__)),
        encoding="utf-8",
    )

    payload: dict[str, Any] = {
        "kind": kind,
        "status": "failed",
        "output_dir": str(output_dir),
        "created_at_utc": utc_now_iso(),
        "error_type": type(error).__name__,
        "error_message": str(error),
        "traceback_file": str(traceback_path),
    }

    if extra:
        payload.update(dict(extra))

    payload = enrich_manifest_with_tracking(
        payload,
        output_dir=output_dir,
        args=args,
        comm=comm,
        timing_rows=[
            {
                "section": "failed_run",
                "elapsed_seconds": 0.0,
                "status": "failed",
            }
        ],
    )

    manifest_path = output_dir / manifest_filename
    write_json(manifest_path, payload)

    append_run_log(
        output_dir,
        [
            "failed manifest written",
            f"error_type={type(error).__name__}",
            f"error_message={str(error)}",
            f"traceback_file={traceback_path}",
        ],
    )

    return to_jsonable(payload)
