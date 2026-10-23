from __future__ import annotations

from elastic_fwi.metrics.performance_metrics import compute_disk_usage_mb, summarize_performance


def test_performance_summary_supports_nullable_pde_fields(tmp_path):
    (tmp_path / "a.txt").write_text("abc", encoding="utf-8")

    summary = summarize_performance(
        elapsed_seconds=0.12,
        num_time_steps=None,
        num_dofs=None,
        mpi_size=1,
        disk_usage_mb=compute_disk_usage_mb(tmp_path),
    )

    payload = summary.as_dict()
    assert payload["elapsed_seconds"] == 0.12
    assert payload["num_time_steps"] is None
    assert payload["num_dofs"] is None
    assert payload["mpi_size"] == 1
    assert payload["seconds_per_step"] is None
    assert payload["disk_usage_mb"] > 0.0


def test_performance_summary_seconds_per_step_with_pde_fields():
    summary = summarize_performance(
        elapsed_seconds=12.5,
        num_time_steps=80,
        num_dofs=4802,
        mpi_size=4,
        disk_usage_mb=124.7,
    ).as_dict()

    assert summary["num_time_steps"] == 80
    assert summary["num_dofs"] == 4802
    assert summary["mpi_size"] == 4
    assert summary["seconds_per_step"] == 12.5 / 80
    assert summary["disk_usage_mb"] == 124.7
