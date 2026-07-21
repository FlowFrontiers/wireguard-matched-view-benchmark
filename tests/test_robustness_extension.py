from __future__ import annotations

import pandas as pd

from vpncat.robustness_extension import build_robustness_macros


def test_robustness_macros_are_derived_from_evidence() -> None:
    endpoint_rows = []
    for fold, value in enumerate((0.987, 0.988, 0.989, 0.988, 0.989), start=1):
        endpoint_rows.append(
            {
                "protocol": "primary",
                "split": f"fold_{fold}",
                "endpoint_key": "remote_ip",
                "target_seen_fraction": value,
            }
        )
    for split, value in (("s1_to_s2", 0.844), ("s2_to_s1", 0.824)):
        endpoint_rows.append(
            {
                "protocol": "cross_session",
                "split": split,
                "endpoint_key": "remote_ip",
                "target_seen_fraction": value,
            }
        )
    excluded = pd.DataFrame(
        {
            "application_category": ["A", "B", "C", "D", "E", "F"],
            "eligible_flow_count": [10, 19, 1, 24, 84, 2],
        }
    )
    comparison_intervals = {
        "primary__sequential_splt__cnn1d__train_inner": (-0.018, -0.006),
        "cnn1d_minus_lstm": (0.041, 0.075),
        "cnn1d_minus_transformer": (0.058, 0.072),
        "dann_minus_plain_cnn1d": (0.004, 0.015),
        "dann_minus_supervised_outer_cnn1d": (-0.010, -0.003),
    }
    temporal_rows = []
    for index in range(9):
        comparison_id = (
            "primary__sequential_splt__cnn1d__train_inner"
            if index == 0
            else f"primary_{index}"
        )
        low, high = comparison_intervals.get(comparison_id, (-0.2, -0.1))
        temporal_rows.append(
            {
                "comparison_type": "view_gap",
                "comparison_id": comparison_id,
                "metric": "macro_f1",
                "block_hours": 2,
                "delta_ci_low": low,
                "delta_ci_high": high,
            }
        )
    for comparison_id, (low, high) in comparison_intervals.items():
        if comparison_id.startswith("primary__"):
            continue
        temporal_rows.append(
            {
                "comparison_type": "method_difference",
                "comparison_id": comparison_id,
                "metric": "macro_f1",
                "block_hours": 2,
                "delta_ci_low": low,
                "delta_ci_high": high,
            }
        )

    macros = build_robustness_macros(
        pd.DataFrame(endpoint_rows), excluded, pd.DataFrame(temporal_rows)
    )
    assert "\\newcommand{\\ExcludedBelowSupportFlowCount}{140}" in macros
    assert "\\newcommand{\\PrimaryRemoteIPOverlapLow}{98.7\\%}" in macros
    assert "\\newcommand{\\CrossSessionRemoteIPOverlapLow}{82.4\\%}" in macros
    assert "\\newcommand{\\DANNTemporalGainHigh}{0.015}" in macros
