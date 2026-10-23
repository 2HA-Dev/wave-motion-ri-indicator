from __future__ import annotations

import numpy as np

from elastic_fwi.metrics.detection_metrics import binary_detection_metrics
from elastic_fwi.metrics.performance_metrics import summarize_performance


def test_binary_detection_metrics_basic():
    pred = np.array([True, True, False, False])
    ref = np.array([True, False, True, False])

    metrics = binary_detection_metrics(pred, ref)

    assert metrics.num_true_positive == 1
    assert metrics.num_false_positive == 1
    assert metrics.num_false_negative == 1
    assert metrics.precision == 0.5
    assert metrics.recall == 0.5
    assert metrics.f1_score == 0.5


def test_performance_summary_basic():
    summary = summarize_performance(
        elapsed_seconds=10.0,
        num_time_steps=5,
        num_dofs=100,
        mpi_size=2,
    )

    assert summary.seconds_per_step == 2.0
    assert summary.as_dict()["mpi_size"] == 2
