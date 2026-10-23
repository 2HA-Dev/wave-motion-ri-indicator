from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REQUIRED_TRACKING_FILES = (
    "command_line_txt",
    "command_line_json",
    "environment_json",
    "hardware_json",
    "git_info_json",
    "parallel_runtime_json",
    "run_log",
    "timings_csv",
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _manifest_has_tracking(path: Path) -> dict[str, Any]:
    try:
        payload = _read_json(path)
    except Exception as exc:
        return {
            "manifest": str(path),
            "status": "failed",
            "reason": "unreadable_json",
            "error": repr(exc),
        }

    tracking = payload.get("tracking")

    if not isinstance(tracking, dict):
        return {
            "manifest": str(path),
            "status": "failed",
            "reason": "missing_tracking",
            "kind": payload.get("kind"),
            "manifest_status": payload.get("status"),
        }

    if tracking.get("status") == "skipped_non_root_rank":
        return {
            "manifest": str(path),
            "status": "ok",
            "reason": "skipped_non_root_rank",
            "kind": payload.get("kind"),
            "manifest_status": payload.get("status"),
        }

    files = tracking.get("files")
    if not isinstance(files, dict):
        return {
            "manifest": str(path),
            "status": "failed",
            "reason": "missing_tracking_files",
            "kind": payload.get("kind"),
            "manifest_status": payload.get("status"),
        }

    missing_keys = [key for key in REQUIRED_TRACKING_FILES if key not in files]
    missing_paths = [
        key
        for key in REQUIRED_TRACKING_FILES
        if key in files and not Path(str(files[key])).exists()
    ]

    status = "ok" if not missing_keys and not missing_paths else "failed"

    return {
        "manifest": str(path),
        "status": status,
        "reason": "ok" if status == "ok" else "tracking_sidecar_missing",
        "kind": payload.get("kind"),
        "manifest_status": payload.get("status"),
        "schema_version": tracking.get("schema_version"),
        "missing_keys": missing_keys,
        "missing_paths": missing_paths,
        "command_line_hash": tracking.get("command", {}).get("command_line_hash"),
    }


def find_manifests(root: Path) -> list[Path]:
    patterns = (
        "*manifest*.json",
        "article_claims_summary.json",
        "check_article_pipeline_ready.json",
    )

    paths: set[Path] = set()
    for pattern in patterns:
        paths.update(root.rglob(pattern))

    # Exclude sidecar JSON files and obvious non-manifest configs.
    excluded_names = {
        "command_line.json",
        "environment.json",
        "hardware.json",
        "git_info.json",
        "parallel_runtime.json",
    }

    return sorted(path for path in paths if path.name not in excluded_names)


def build_tracking_outputs_report(
    *,
    root: str | Path,
    require_tracking: bool = False,
) -> dict[str, Any]:
    root = Path(root)
    manifests = find_manifests(root)

    records = [_manifest_has_tracking(path) for path in manifests]

    failed = [record for record in records if record["status"] != "ok"]
    missing_tracking = [
        record for record in records if record.get("reason") == "missing_tracking"
    ]

    status = "ok"
    if failed:
        status = "warning"
    if require_tracking and missing_tracking:
        status = "failed"

    return {
        "kind": "tracking_outputs_check",
        "status": status,
        "root": str(root),
        "require_tracking": bool(require_tracking),
        "num_manifests": len(records),
        "num_ok": sum(1 for record in records if record["status"] == "ok"),
        "num_failed": len(failed),
        "num_missing_tracking": len(missing_tracking),
        "records": records,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check tracking/provenance sidecars for JSON manifests."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs"),
        help="Root directory to scan.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional output JSON report path.",
    )
    parser.add_argument(
        "--require-tracking",
        action="store_true",
        help="Return failed if any manifest has no tracking block.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return non-zero if status is not ok.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    report = build_tracking_outputs_report(
        root=args.root,
        require_tracking=args.require_tracking,
    )

    print("[TRACKING OUTPUTS CHECK]", report["status"].upper())
    print("root =", report["root"])
    print("num_manifests =", report["num_manifests"])
    print("num_ok =", report["num_ok"])
    print("num_failed =", report["num_failed"])
    print("num_missing_tracking =", report["num_missing_tracking"])

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(report, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print("output_json =", args.output_json)

    if args.strict and report["status"] != "ok":
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
