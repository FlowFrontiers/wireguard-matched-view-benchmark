from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

import vpncat.neural_data as neural_data
from vpncat.errors import PipelineInvariantError
from vpncat.experiment import RunSpec
from vpncat.models.neural import build_neural_model, trainable_parameter_count
from vpncat.neural_config import FROZEN_TOPOLOGIES, load_neural_config
from vpncat.neural_data import prepare_neural_development, prepare_neural_run


def _config_path() -> Path:
    return Path(__file__).parents[1] / "configs" / "neural.yaml"


def _run() -> RunSpec:
    return RunSpec(
        protocol="primary",
        experiment_id="sequential_splt__cnn1d",
        representation="sequential_splt",
        model="cnn1d",
        family="neural",
        fold=1,
        seed=42,
        train_domain="inner",
        test_domains=("inner", "outer"),
    )


def _write_inputs(tmp_path: Path) -> tuple[Path, Path]:
    row_count = 8
    frame = pd.DataFrame(
        {
            "pair_id": [f"s1:{index}" for index in range(row_count)],
            "session": [1] * row_count,
            "application_category": ["A", "A", "B", "B", "A", "B", "A", "B"],
        }
    )
    for domain, offset in (("inner", 0.0), ("outer", 10.0)):
        frame[f"{domain}_direction"] = [np.asarray([0, 1, 0], dtype=np.int8)] * row_count
        frame[f"{domain}_size"] = [
            np.asarray([100.0 + index + offset, 200.0, 50.0], dtype=np.float64)
            for index in range(row_count)
        ]
        frame[f"{domain}_iat_ms"] = [
            np.asarray([0.0, 1.0 + index + offset, 2.0], dtype=np.float64)
            for index in range(row_count)
        ]
    canonical_path = tmp_path / "canonical.parquet"
    frame.to_parquet(canonical_path, index=False)

    split = frame.loc[:, ["pair_id", "session", "application_category"]].copy()
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
    split_path = tmp_path / "split.csv"
    split.to_csv(split_path, index=False)
    return canonical_path, split_path


def test_neural_configuration_freezes_topologies_and_twelve_trials() -> None:
    config = load_neural_config(_config_path())
    assert config.topologies == FROZEN_TOPOLOGIES
    assert config.primary_prefix_length == 50
    assert config.maximum_prefix_length == 80
    assert len(config.trials) == 12
    assert config.development_fold == 1
    assert config.development_train_domain == "inner"
    assert config.development_seed == 42


@pytest.mark.parametrize("model_name", ["cnn1d", "lstm", "transformer"])
def test_neural_models_ignore_padded_values_and_emit_logits(model_name: str) -> None:
    torch.manual_seed(42)
    model = build_neural_model(
        model_name,
        feature_count=3,
        class_count=14,
        width=32,
        dropout=0.2,
        maximum_length=80,
        topology=FROZEN_TOPOLOGIES[model_name],
    )
    model.eval()
    values = torch.randn(3, 8, 3)
    mask = torch.tensor(
        [
            [True] * 8,
            [True] * 5 + [False] * 3,
            [True] * 2 + [False] * 6,
        ]
    )
    values[~mask] = 0.0
    poisoned = values.clone()
    poisoned[~mask] = 1e6
    with torch.no_grad():
        baseline = model(values, mask)
        observed = model(poisoned, mask)

    assert baseline.shape == (3, 14)
    assert torch.isfinite(baseline).all()
    torch.testing.assert_close(baseline, observed, rtol=0.0, atol=1e-6)
    assert trainable_parameter_count(model) > 0


def test_neural_preparation_uses_source_view_for_train_and_validation(
    tmp_path: Path,
) -> None:
    canonical_path, split_path = _write_inputs(tmp_path)
    baseline = prepare_neural_run(
        canonical_path,
        split_path,
        _run(),
        prefix_length=5,
        channels=("direction", "size", "iat_ms"),
    )

    poisoned = pd.read_parquet(canonical_path)
    for row in range(len(poisoned)):
        poisoned.at[row, "outer_size"] = np.asarray([9e9, 9e9, 9e9])
    for row in (6, 7):
        poisoned.at[row, "inner_size"] = np.asarray([8e9, 8e9, 8e9])
    poisoned_path = tmp_path / "poisoned.parquet"
    poisoned.to_parquet(poisoned_path, index=False)
    observed = prepare_neural_run(
        poisoned_path,
        split_path,
        _run(),
        prefix_length=5,
        channels=("direction", "size", "iat_ms"),
    )

    assert baseline.state.to_dict() == observed.state.to_dict()
    np.testing.assert_array_equal(baseline.training.values, observed.training.values)
    np.testing.assert_array_equal(baseline.validation.values, observed.validation.values)
    assert baseline.training.pair_ids == ("s1:0", "s1:1", "s1:2", "s1:3")
    assert baseline.validation.pair_ids == ("s1:4", "s1:5")
    assert baseline.tests["inner"].pair_ids == baseline.tests["outer"].pair_ids


def test_neural_model_rejects_topology_drift() -> None:
    changed = dict(FROZEN_TOPOLOGIES["cnn1d"])
    changed["kernels"] = [3, 5, 7]
    with pytest.raises(PipelineInvariantError, match="topology"):
        build_neural_model(
            "cnn1d",
            feature_count=3,
            class_count=14,
            width=64,
            dropout=0.2,
            maximum_length=80,
            topology=changed,
        )


def test_development_materializes_only_train_and_validation_source_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical_path, split_path = _write_inputs(tmp_path)
    poisoned = pd.read_parquet(canonical_path)
    for row in range(len(poisoned)):
        poisoned.at[row, "outer_size"] = np.asarray([-1.0, -1.0, -1.0])
    poisoned.to_parquet(canonical_path, index=False)

    materialized_rows: list[int] = []
    original = neural_data.build_sequential_splt

    def observed_builder(*args, **kwargs):
        materialized_rows.append(len(args[0]))
        return original(*args, **kwargs)

    monkeypatch.setattr(neural_data, "build_sequential_splt", observed_builder)
    development = prepare_neural_development(
        canonical_path,
        split_path,
        _run(),
        prefix_length=5,
        channels=("direction", "size", "iat_ms"),
    )

    assert materialized_rows == [6]
    assert len(development.training.pair_ids) == 4
    assert len(development.validation.pair_ids) == 2
