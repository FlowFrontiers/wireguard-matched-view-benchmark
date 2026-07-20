from pathlib import Path

import pandas as pd

from vpncat.matched_packets import aggregate_partition


def test_prefix_keeps_matched_membership_but_uses_each_view_order(tmp_path: Path) -> None:
    assigned = pd.DataFrame(
        {
            "flow_id": [0, 0, 0],
            "inner_idx": [1, 2, 3],
            "outer_idx": [10, 12, 11],
            "direction": [0, 1, 0],
            "inner_time_ms": [0.0, 1.0, 2.0],
            "outer_time_ms": [0.0, 2.0, 1.0],
            "inner_size": [100.0, 200.0, 300.0],
            "outer_size": [112.0, 208.0, 304.0],
        }
    )
    path = tmp_path / "assigned.parquet"
    assigned.to_parquet(path, index=False)

    result, reordered_pairs, reordered_flows = aggregate_partition(path, maximum_length=3)
    row = result.iloc[0]
    assert reordered_pairs == 1
    assert reordered_flows == 1
    assert row["inner_size"] == [100.0, 200.0, 300.0]
    assert row["outer_size"] == [112.0, 304.0, 208.0]
    assert row["inner_direction"] == [0, 1, 0]
    assert row["outer_direction"] == [0, 0, 1]
    assert row["inner_iat_ms"] == [0.0, 1.0, 1.0]
    assert row["outer_iat_ms"] == [0.0, 1.0, 1.0]
