from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_run_log(
    output_dir: str | Path,
    lines: str | Iterable[str],
    *,
    filename: str = "run.log",
) -> Path:
    output_dir = Path(output_dir)
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    path = logs_dir / filename

    if isinstance(lines, str):
        payload = [lines]
    else:
        payload = [str(line) for line in lines]

    with path.open("a", encoding="utf-8") as handle:
        for line in payload:
            handle.write(f"{utc_now_iso()} {line.rstrip()}\n")

    return path
