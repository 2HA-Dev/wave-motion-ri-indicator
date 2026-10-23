from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DetectionMetrics:
    num_true_positive: int
    num_false_positive: int
    num_false_negative: int

    @property
    def precision(self) -> float:
        denom = self.num_true_positive + self.num_false_positive
        return float(self.num_true_positive / denom) if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.num_true_positive + self.num_false_negative
        return float(self.num_true_positive / denom) if denom else 0.0

    @property
    def f1_score(self) -> float:
        p = self.precision
        r = self.recall
        return float(2.0 * p * r / (p + r)) if (p + r) else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "num_true_positive": self.num_true_positive,
            "num_false_positive": self.num_false_positive,
            "num_false_negative": self.num_false_negative,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
        }


def binary_detection_metrics(
    predicted_mask: np.ndarray,
    reference_mask: np.ndarray,
) -> DetectionMetrics:
    pred = np.asarray(predicted_mask, dtype=bool)
    ref = np.asarray(reference_mask, dtype=bool)

    if pred.shape != ref.shape:
        raise ValueError("predicted_mask and reference_mask must have the same shape.")

    tp = int(np.count_nonzero(pred & ref))
    fp = int(np.count_nonzero(pred & ~ref))
    fn = int(np.count_nonzero(~pred & ref))

    return DetectionMetrics(
        num_true_positive=tp,
        num_false_positive=fp,
        num_false_negative=fn,
    )
