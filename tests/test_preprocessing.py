from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vpncat.errors import PipelineInvariantError
from vpncat.folds import materialize_fold_index
from vpncat.preprocessing import (
    build_statistical_observations,
    fit_fold_preprocessing,
    pair_id_digest,
)
from vpncat.schema import CANONICAL_STATS


def _inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    pair_ids = tuple(f"s1:{index}" for index in range(8))
    labels = ("A", "A", "B", "B", "A", "B", "A", "B")
    canonical = pd.DataFrame(
        {
            "pair_id": pair_ids,
            "session": [1] * len(pair_ids),
            "application_category": labels,
        }
    )
    for domain, offset in (("inner", 0.0), ("outer", 1_000.0)):
        for column, name in enumerate(CANONICAL_STATS):
            canonical[f"{domain}_{name}"] = (
                np.arange(len(canonical), dtype=float) + offset + column
            )
    canonical.loc[1, "inner_packet_rate"] = np.nan
    canonical.loc[4, "inner_packet_rate"] = 99_999.0
    canonical.loc[6, "inner_packet_rate"] = 88_888.0

    split = canonical.loc[:, ["pair_id", "session", "application_category"]].copy()
    split["role_fold_1"] = [
        "train",
        "train",
        "train",
        "train",
        "validation",
        "validation",
        "test",
        "test",
    ]
    return canonical, split


def test_fold_index_is_pair_disjoint_and_canonical_ordered() -> None:
    canonical, split = _inputs()
    fold = materialize_fold_index(canonical, split.sample(frac=1.0), fold=1)

    assert fold.pair_ids == tuple(canonical["pair_id"])
    assert fold.pair_ids_for("train") == ("s1:0", "s1:1", "s1:2", "s1:3")
    assert set(fold.pair_ids_for("train")).isdisjoint(fold.pair_ids_for("validation"))
    assert set(fold.pair_ids_for("train")).isdisjoint(fold.pair_ids_for("test"))
    assert set(fold.pair_ids_for("validation")).isdisjoint(fold.pair_ids_for("test"))


def test_preprocessing_fit_ignores_nontraining_and_opposite_domain_values() -> None:
    canonical, split = _inputs()
    fold = materialize_fold_index(canonical, split, fold=1)
    inner = build_statistical_observations(
        canonical,
        domain="inner",
        representation="matched_flow_stats",
        prefix_length=50,
    )
    first = fit_fold_preprocessing(inner, fold)

    poisoned = canonical.copy()
    nontraining = np.concatenate((fold.validation_positions, fold.test_positions))
    poisoned.loc[nontraining, [f"inner_{name}" for name in CANONICAL_STATS]] = 1e30
    poisoned.loc[:, [f"outer_{name}" for name in CANONICAL_STATS]] = -1e30
    second = fit_fold_preprocessing(
        build_statistical_observations(
            poisoned,
            domain="inner",
            representation="matched_flow_stats",
            prefix_length=50,
        ),
        fold,
    )

    np.testing.assert_array_equal(first.medians, second.medians)
    np.testing.assert_array_equal(first.class_weights, second.class_weights)
    assert first.classes == second.classes == ("A", "B")
    assert first.fit_pair_count == 4
    assert first.fit_pair_ids_sha256 == pair_id_digest(fold.pair_ids_for("train"))
    assert first.to_dict()["scaling"] == "none"


def test_inner_fitted_medians_apply_unchanged_to_outer_test_view() -> None:
    canonical, split = _inputs()
    fold = materialize_fold_index(canonical, split, fold=1)
    inner = build_statistical_observations(
        canonical,
        domain="inner",
        representation="matched_flow_stats",
        prefix_length=50,
    )
    outer = build_statistical_observations(
        canonical,
        domain="outer",
        representation="matched_flow_stats",
        prefix_length=50,
    )
    state = fit_fold_preprocessing(inner, fold)
    packet_rate = state.feature_names.index("packet_rate")
    outer.values[fold.test_positions[0], packet_rate] = np.nan

    transformed = state.transform_features(outer)
    assert transformed[fold.test_positions[0], packet_rate] == state.medians[packet_rate]
    assert np.isfinite(transformed).all()
    np.testing.assert_array_equal(
        state.sample_weights(np.asarray(fold.labels)[fold.train_positions]),
        np.ones(len(fold.train_positions)),
    )


def test_preprocessing_rejects_misalignment_and_unknown_labels() -> None:
    canonical, split = _inputs()
    fold = materialize_fold_index(canonical, split, fold=1)
    observations = build_statistical_observations(
        canonical,
        domain="inner",
        representation="matched_flow_stats",
        prefix_length=50,
    )
    state = fit_fold_preprocessing(observations, fold)

    misaligned = canonical.iloc[::-1].reset_index(drop=True)
    with pytest.raises(PipelineInvariantError, match="not aligned"):
        fit_fold_preprocessing(
            build_statistical_observations(
                misaligned,
                domain="inner",
                representation="matched_flow_stats",
                prefix_length=50,
            ),
            fold,
        )
    with pytest.raises(PipelineInvariantError, match="absent"):
        state.encode_labels(("A", "unknown"))


def test_fold_index_rejects_split_metadata_disagreement() -> None:
    canonical, split = _inputs()
    split.loc[0, "application_category"] = "wrong"
    with pytest.raises(PipelineInvariantError, match="disagree"):
        materialize_fold_index(canonical, split, fold=1)
