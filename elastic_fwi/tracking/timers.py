from __future__ import annotations

import csv
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


class TimingRecorder:
    def __init__(self) -> None:
        self.rows: list[dict[str, object]] = []

    @contextmanager
    def section(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        status = "ok"
        try:
            yield
        except Exception:
            status = "failed"
            raise
        finally:
            elapsed = time.perf_counter() - start
            self.rows.append(
                {
                    "section": name,
                    "elapsed_seconds": elapsed,
                    "status": status,
                }
            )

    def add(self, *, section: str, elapsed_seconds: float, status: str = "ok") -> None:
        self.rows.append(
            {
                "section": section,
                "elapsed_seconds": float(elapsed_seconds),
                "status": status,
            }
        )

    def total_elapsed_seconds(self) -> float:
        values = [
            float(row["elapsed_seconds"])
            for row in self.rows
            if row.get("section") == "total"
        ]
        if values:
            return values[-1]
        return float(sum(float(row["elapsed_seconds"]) for row in self.rows))

    def write_csv(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=("section", "elapsed_seconds", "status"),
            )
            writer.writeheader()
            for row in self.rows:
                writer.writerow(row)

        return path


@contextmanager
def timed_total(recorder: TimingRecorder, *, section: str = "total") -> Iterator[None]:
    with recorder.section(section):
        yield
