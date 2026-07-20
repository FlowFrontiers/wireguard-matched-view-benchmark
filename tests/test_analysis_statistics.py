from __future__ import annotations

import pandas as pd
import pytest

from vpncat.analysis_statistics import (
    compute_analysis_metrics,
    paired_bootstrap_intervals,
)
from vpncat.errors import PipelineInvariantError


def _frame() -> pd.DataFrame:
    rows = []
    for domain, predictions, probabilities in (
        ("inner", ("A", "B", "A", "B"), ((0.9, 0.1), (0.1, 0.9), (0.8, 0.2), (0.2, 0.8))),
        ("outer", ("A", "A", "A", "B"), ((0.8, 0.2), (0.6, 0.4), (0.7, 0.3), (0.3, 0.7))),
    ):
        for pair_id, true, prediction, probability in zip(
            ("a", "b", "c", "d"),
            ("A", "B", "A", "B"),
            predictions,
            probabilities,
            strict=True,
        ):
            rows.append(
                {
                    "pair_id": pair_id,
                    "test_domain": domain,
                    "true_label": true,
                    "prediction": prediction,
                    "class_probabilities": probability,
                }
            )
    return pd.DataFrame(rows)


def test_metrics_and_paired_bootstrap_are_deterministic() -> None:
    frame = _frame()
    metrics = compute_analysis_metrics(frame, ("A", "B"))
    assert metrics["inner"]["macro_f1"] == 1.0
    assert metrics["outer"]["accuracy"] == 0.75
    first = paired_bootstrap_intervals(
        frame,
        ("A", "B"),
        metrics=("balanced_accuracy", "macro_f1"),
        replicates=100,
        confidence_level=0.95,
        seed=42,
    )
    second = paired_bootstrap_intervals(
        frame,
        ("A", "B"),
        metrics=("balanced_accuracy", "macro_f1"),
        replicates=100,
        confidence_level=0.95,
        seed=42,
    )
    assert first == second
    assert all(row["gap_estimate"] <= 0 for row in first)
    assert all(row["replicates"] == 100 for row in first)


def test_bootstrap_rejects_unpaired_views() -> None:
    frame = _frame()
    frame.loc[frame["test_domain"].eq("outer") & frame["pair_id"].eq("a"), "pair_id"] = "z"
    with pytest.raises(PipelineInvariantError, match="not paired"):
        paired_bootstrap_intervals(
            frame,
            ("A", "B"),
            metrics=("macro_f1",),
            replicates=10,
            confidence_level=0.95,
            seed=42,
        )
