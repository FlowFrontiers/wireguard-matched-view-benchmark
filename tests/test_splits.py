import pandas as pd
import pytest

from vpncat.errors import PipelineInvariantError
from vpncat.splits import build_split_manifest


def _pair_index(per_stratum: int = 20) -> pd.DataFrame:
    rows = []
    for session in (1, 2):
        for category in ("Web", "Network"):
            for index in range(per_stratum):
                rows.append(
                    {
                        "pair_id": f"s{session}:{category}:{index}",
                        "session": session,
                        "application_category": category,
                    }
                )
    return pd.DataFrame(rows)


def test_split_manifest_is_deterministic_and_pair_disjoint() -> None:
    pairs = _pair_index()
    first = build_split_manifest(
        pairs, folds=5, validation_fraction=0.10, random_seed=42
    )
    second = build_split_manifest(
        pairs, folds=5, validation_fraction=0.10, random_seed=42
    )

    pd.testing.assert_frame_equal(first, second)
    role_columns = [f"role_fold_{fold}" for fold in range(1, 6)]
    assert first[role_columns].eq("test").sum(axis=1).eq(1).all()
    for column in role_columns:
        assert set(first[column]) == {"train", "validation", "test"}


def test_split_manifest_rejects_small_class_session_strata() -> None:
    with pytest.raises(PipelineInvariantError, match="cannot support"):
        build_split_manifest(
            _pair_index(per_stratum=4),
            folds=5,
            validation_fraction=0.10,
            random_seed=42,
        )
