from __future__ import annotations

import pandas as pd
import pytest

from vpncat.errors import PipelineInvariantError
from vpncat.paper_statistics import paired_method_intervals, per_class_metrics


def _method_frame(predictions: tuple[str, ...]) -> pd.DataFrame:
    rows = []
    labels = ("A", "B", "A", "B")
    for domain in ("inner", "outer"):
        for pair_id, true_label, prediction in zip(
            ("p1", "p2", "p3", "p4"), labels, predictions, strict=True
        ):
            probability = (0.8, 0.2) if prediction == "A" else (0.2, 0.8)
            rows.append(
                {
                    "pair_id": pair_id,
                    "test_domain": domain,
                    "true_label": true_label,
                    "prediction": prediction,
                    "class_probabilities": probability,
                }
            )
    return pd.DataFrame(rows)


def test_paired_method_intervals_are_deterministic_and_left_minus_right() -> None:
    left = _method_frame(("A", "B", "A", "B"))
    right = _method_frame(("A", "A", "A", "B"))
    first = paired_method_intervals(left, right, ("A", "B"), domain="outer")
    second = paired_method_intervals(left, right, ("A", "B"), domain="outer")
    assert first == second
    assert all(row["delta_estimate"] > 0 for row in first)
    assert all(row["pair_count"] == 4 for row in first)


def test_paired_method_intervals_reject_different_pair_sets() -> None:
    left = _method_frame(("A", "B", "A", "B"))
    right = _method_frame(("A", "A", "A", "B"))
    right.loc[right["pair_id"].eq("p4"), "pair_id"] = "different"
    with pytest.raises(PipelineInvariantError, match="identities"):
        paired_method_intervals(left, right, ("A", "B"), domain="outer")


def test_per_class_metrics_preserve_frozen_class_order_and_zero_support() -> None:
    frame = _method_frame(("A", "B", "A", "B"))
    frame["class_probabilities"] = frame["class_probabilities"].map(
        lambda values: (*values, 0.0)
    )
    result = per_class_metrics(frame, ("A", "B", "C"), domain="outer")
    assert result["class_name"].tolist() == ["A", "B", "C"]
    assert result["support"].tolist() == [2, 2, 0]
    assert result.loc[result["class_name"].eq("C"), "f1"].item() == 0.0
