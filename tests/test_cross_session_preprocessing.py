from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vpncat.cross_session import CrossSessionRunSpec, load_cross_session_config
from vpncat.cross_session_data import (
    prepare_cross_session_classical,
    prepare_cross_session_neural,
)
from vpncat.cross_session_index import materialize_cross_session_index
from vpncat.cross_session_preprocessing import (
    fit_cross_session_preprocessing,
    fit_cross_session_targets,
)
from vpncat.cross_session_preprocessing_audit import (
    load_cross_session_preprocessing_config,
)
from vpncat.preprocessing import StatisticalObservations
from vpncat.schema import CANONICAL_STATS


def _metadata_and_split() -> tuple[pd.DataFrame, pd.DataFrame]:
    metadata = pd.DataFrame(
        {
            "pair_id": [f"pair:{index}" for index in range(12)],
            "session": [1] * 6 + [2] * 6,
            "application_category": ["A", "B", "A", "B", "A", "B"] * 2,
        }
    )
    split = metadata.copy()
    split["role_train_session_1"] = [
        "train", "train", "train", "train", "validation", "validation",
        "test", "test", "test", "test", "test", "test",
    ]
    split["role_train_session_2"] = [
        "test", "test", "test", "test", "test", "test",
        "train", "train", "train", "train", "validation", "validation",
    ]
    return metadata, split


def _run(*, family: str, representation: str, model: str) -> CrossSessionRunSpec:
    return CrossSessionRunSpec(
        protocol="cross_session",
        experiment_id=f"{representation}__{model}",
        representation=representation,
        model=model,
        family=family,
        seed=42,
        train_session=1,
        test_session=2,
        train_domain="inner",
        test_domains=("inner", "outer"),
    )


def _write_inputs(tmp_path: Path):
    metadata, split = _metadata_and_split()
    frame = metadata.copy()
    for column_index, name in enumerate(CANONICAL_STATS):
        base = np.arange(len(frame), dtype=np.float64) + column_index + 1
        frame[f"inner_{name}"] = base
        frame[f"outer_{name}"] = base + 100
    for domain, offset in (("inner", 0.0), ("outer", 100.0)):
        frame[f"{domain}_direction"] = [np.asarray([0, 1, 0])] * len(frame)
        frame[f"{domain}_size"] = [
            np.asarray([100 + index + offset, 200, 50]) for index in range(len(frame))
        ]
        frame[f"{domain}_iat_ms"] = [
            np.asarray([0, 1 + index + offset, 2]) for index in range(len(frame))
        ]
    canonical = tmp_path / "canonical.parquet"
    split_path = tmp_path / "cross_split.csv"
    frame.to_parquet(canonical, index=False)
    split.to_csv(split_path, index=False)
    base = load_cross_session_config(
        Path(__file__).parents[1] / "configs" / "cross_session.yaml"
    )
    primary = replace(base.primary, canonical_path=canonical)
    return replace(base, primary=primary, split_path=split_path), canonical


def test_cross_session_fitting_ignores_validation_test_and_opposite_domain() -> None:
    metadata, split = _metadata_and_split()
    index = materialize_cross_session_index(metadata, split, train_session=1)
    values = np.arange(24, dtype=np.float64).reshape(12, 2)
    observations = StatisticalObservations(
        pair_ids=index.pair_ids,
        domain="inner",
        representation="matched_flow_stats",
        feature_names=("one", "two"),
        values=values,
    )
    baseline = fit_cross_session_preprocessing(observations, index)
    poisoned = values.copy()
    poisoned[np.concatenate([index.validation_positions, index.test_positions])] = 1e30
    observed = fit_cross_session_preprocessing(
        replace(observations, values=poisoned), index
    )
    assert baseline.to_dict() == observed.to_dict()

    poisoned_labels = list(index.labels)
    for position in np.concatenate([index.validation_positions, index.test_positions]):
        poisoned_labels[int(position)] = "POISON"
    assert fit_cross_session_targets(index).to_dict() == fit_cross_session_targets(
        replace(index, labels=tuple(poisoned_labels))
    ).to_dict()


def test_cross_session_classical_materializer_uses_source_only(tmp_path: Path) -> None:
    config, canonical = _write_inputs(tmp_path)
    run = _run(
        family="classical",
        representation="matched_flow_stats",
        model="random_forest",
    )
    baseline = prepare_cross_session_classical(config, run)
    poisoned = pd.read_parquet(canonical)
    inner_columns = [f"inner_{name}" for name in CANONICAL_STATS]
    outer_columns = [f"outer_{name}" for name in CANONICAL_STATS]
    poisoned.loc[4:, inner_columns] = 1e30
    poisoned.loc[:, outer_columns] = -1e30
    poisoned_path = tmp_path / "poisoned.parquet"
    poisoned.to_parquet(poisoned_path, index=False)
    observed = prepare_cross_session_classical(
        replace(config, primary=replace(config.primary, canonical_path=poisoned_path)),
        run,
    )
    assert baseline.state.to_dict() == observed.state.to_dict()
    np.testing.assert_array_equal(baseline.training_values, observed.training_values)
    assert len(observed.training_values) == 4
    assert len(observed.validation_values) == 2
    assert len(observed.test_values["inner"]) == 6


@pytest.mark.parametrize("train_session", [1, 2])
def test_cross_session_flattened_materializer_is_fit_free(
    tmp_path: Path,
    train_session: int,
) -> None:
    config, _ = _write_inputs(tmp_path)
    run = replace(
        _run(
            family="classical",
            representation="flattened_splt",
            model="random_forest",
        ),
        train_session=train_session,
        test_session=3 - train_session,
    )
    prepared = prepare_cross_session_classical(config, run)
    expected_train = 4
    expected_validation = 2
    expected_test = 6
    assert len(prepared.training_values) == expected_train
    assert len(prepared.validation_values) == expected_validation
    assert len(prepared.test_values["inner"]) == expected_test
    assert prepared.training_values.shape[1] == 150
    assert prepared.state.fit_pair_count == expected_train


def test_cross_session_neural_materializer_preserves_paired_tests(tmp_path: Path) -> None:
    config, _ = _write_inputs(tmp_path)
    prepared = prepare_cross_session_neural(
        config,
        _run(family="neural", representation="sequential_splt", model="cnn1d"),
        channels=("direction", "size", "iat_ms"),
    )
    assert len(prepared.training.pair_ids) == 4
    assert len(prepared.validation.pair_ids) == 2
    assert len(prepared.tests["inner"].pair_ids) == 6
    assert prepared.tests["inner"].pair_ids == prepared.tests["outer"].pair_ids
    assert set(prepared.training.pair_ids).isdisjoint(prepared.tests["inner"].pair_ids)


def test_cross_session_artifact_directory_relocates_preprocessing_audit(
    tmp_path: Path,
) -> None:
    config = load_cross_session_preprocessing_config(
        Path(__file__).parents[1] / "configs" / "cross_session_preprocessing.yaml",
        artifact_dir=tmp_path,
    )
    assert config.audit_output == tmp_path.resolve() / "cross_session_preprocessing_audit.json"
    assert config.cross_session.split_path == tmp_path.resolve() / (
        "cross_session_split_manifest.csv"
    )
