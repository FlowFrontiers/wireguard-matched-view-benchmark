from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score

from vpncat.temporal_bootstrap import (
    build_time_clusters,
    cluster_bootstrap_difference,
    session_preserving_multiplicities,
)


def test_time_cluster_bootstrap_is_paired_and_session_preserving() -> None:
    pair_times = pd.DataFrame(
        {
            "pair_id": [f"s1:{i}" for i in range(4)] + [f"s2:{i}" for i in range(4)],
            "session": [1] * 4 + [2] * 4,
            "bidirectional_first_seen_ms": [0, 1, 3_600_000, 3_600_001] * 2,
        }
    )
    clusters, by_session = build_time_clusters(pair_times, block_hours=1)
    multiplicities = session_preserving_multiplicities(
        by_session, replicates=200, seed=42
    )
    assert multiplicities.shape == (200, 4)
    for indices in by_session.values():
        np.testing.assert_array_equal(
            multiplicities[:, indices].sum(axis=1), np.full(200, len(indices))
        )

    frame = pd.DataFrame(
        {
            "pair_id": pair_times["pair_id"],
            "true_label": ["A", "A", "B", "B"] * 2,
            "a_prediction": ["A", "A", "B", "B"] * 2,
            "b_prediction": ["B", "A", "A", "B"] * 2,
        }
    )
    rows = cluster_bootstrap_difference(
        frame,
        clusters,
        multiplicities,
        a_column="a_prediction",
        b_column="b_prediction",
        classes=("A", "B"),
    )
    by_metric = {row["metric"]: row for row in rows}
    assert by_metric["balanced_accuracy"]["a_estimate"] == 1.0
    assert by_metric["balanced_accuracy"]["b_estimate"] == 0.5
    assert by_metric["balanced_accuracy"]["delta_estimate"] == 0.5
    assert by_metric["macro_f1"]["delta_ci_low"] <= 0.5
    assert by_metric["macro_f1"]["delta_ci_high"] >= 0.5


def test_cluster_interval_matches_explicit_pair_expansion() -> None:
    clusters = pd.DataFrame(
        {
            "pair_id": ["p0", "p1", "p2", "p3"],
            "session": [1, 1, 2, 2],
            "local_block": [0, 1, 0, 1],
            "cluster_index": [0, 1, 2, 3],
        }
    )
    frame = pd.DataFrame(
        {
            "pair_id": ["p0", "p1", "p2", "p3"],
            "true_label": ["A", "B", "A", "B"],
            "a_prediction": ["A", "B", "A", "B"],
            "b_prediction": ["B", "B", "A", "A"],
        }
    )
    multiplicities = np.asarray([[2, 0, 0, 2]], dtype=np.int16)
    rows = cluster_bootstrap_difference(
        frame,
        clusters,
        multiplicities,
        a_column="a_prediction",
        b_column="b_prediction",
        classes=("A", "B"),
    )

    expanded = frame.iloc[[0, 0, 3, 3]]
    expected = {
        "balanced_accuracy": balanced_accuracy_score(
            expanded["true_label"], expanded["a_prediction"]
        )
        - balanced_accuracy_score(expanded["true_label"], expanded["b_prediction"]),
        "macro_f1": f1_score(
            expanded["true_label"],
            expanded["a_prediction"],
            labels=["A", "B"],
            average="macro",
            zero_division=0,
        )
        - f1_score(
            expanded["true_label"],
            expanded["b_prediction"],
            labels=["A", "B"],
            average="macro",
            zero_division=0,
        ),
    }
    for row in rows:
        assert row["delta_ci_low"] == expected[row["metric"]]
        assert row["delta_ci_high"] == expected[row["metric"]]
