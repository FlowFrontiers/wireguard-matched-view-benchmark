from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from vpncat.errors import PipelineInvariantError
from vpncat.experiment import RunSpec
from vpncat.folds import FoldIndex
from vpncat.models.neural import build_neural_model
from vpncat.neural_config import (
    FROZEN_OPTIMIZER,
    FROZEN_TOPOLOGIES,
    FROZEN_TRAINING,
    load_neural_config,
)
from vpncat.neural_data import NeuralSubset, PreparedNeuralDevelopment
from vpncat.neural_training import (
    predict_neural_probabilities,
    seed_neural_execution,
    train_neural_model,
)
from vpncat.neural_tuning import _publish_trial, _select_trial, _selection_payload
from vpncat.preprocessing import FoldTargetState


def _state() -> FoldTargetState:
    return FoldTargetState(
        fold=1,
        classes=("A", "B"),
        class_weights=np.asarray([1.0, 1.0]),
        fit_pair_count=24,
        fit_pair_ids_sha256="synthetic-fit-hash",
    )


def _subset(prefix: str, count: int, *, offset: int) -> NeuralSubset:
    rng = np.random.default_rng(100 + offset)
    targets = np.arange(count, dtype=np.int64) % 2
    values = rng.normal(0, 0.05, size=(count, 8, 3)).astype(np.float32)
    values[:, :, 0] += np.where(targets[:, None] == 0, -1.0, 1.0)
    mask = np.ones((count, 8), dtype=bool)
    mask[::3, 6:] = False
    values[~mask] = 0.0
    positions = np.arange(offset, offset + count, dtype=np.int64)
    return NeuralSubset(
        pair_ids=tuple(f"{prefix}:{index}" for index in range(count)),
        positions=positions,
        values=values,
        mask=mask,
        targets=targets,
    )


def _training_policy() -> dict[str, object]:
    policy = dict(FROZEN_TRAINING)
    policy.update(
        {
            "maximum_epochs": 4,
            "early_stopping_patience": 3,
            "scheduler_patience": 1,
        }
    )
    return policy


def _fold() -> FoldIndex:
    pair_ids = tuple(f"pair:{index}" for index in range(36))
    roles = ("train",) * 24 + ("validation",) * 12
    return FoldIndex(
        fold=1,
        pair_ids=pair_ids,
        sessions=np.ones(36, dtype=np.int16),
        labels=tuple("A" if index % 2 == 0 else "B" for index in range(36)),
        roles=roles,
        train_positions=np.arange(24, dtype=np.int64),
        validation_positions=np.arange(24, 36, dtype=np.int64),
        test_positions=np.asarray([], dtype=np.int64),
    )


def _train_once():
    seed_neural_execution(42)
    model = build_neural_model(
        "cnn1d",
        feature_count=3,
        class_count=2,
        width=8,
        dropout=0.2,
        maximum_length=80,
        topology=FROZEN_TOPOLOGIES["cnn1d"],
    )
    return train_neural_model(
        model,
        _subset("train", 24, offset=0),
        _subset("validation", 12, offset=24),
        _state(),
        learning_rate=0.001,
        batch_size=8,
        seed=42,
        optimizer_policy=FROZEN_OPTIMIZER,
        training_policy=_training_policy(),
        device_name="cpu",
    )


def test_neural_training_is_deterministic_and_restores_best_epoch() -> None:
    first = _train_once()
    second = _train_once()

    pd.testing.assert_frame_equal(first.history, second.history, check_exact=True)
    assert first.best_validation_macro_f1 == first.history["validation_macro_f1"].max()
    assert first.best_epoch == int(
        first.history.loc[first.history["validation_macro_f1"].idxmax(), "epoch"]
    )
    for name, value in first.model.state_dict().items():
        torch.testing.assert_close(value, second.model.state_dict()[name], rtol=0, atol=0)


def test_neural_prediction_applies_softmax_exactly_once_in_canonical_order() -> None:
    subset = _subset("test", 5, offset=30)

    class FixedLogits(torch.nn.Module):
        def forward(self, values, mask):
            del mask
            return values[:, 0, :2]

    expected = torch.softmax(torch.from_numpy(subset.values[:, 0, :2]), dim=1).numpy()
    observed = predict_neural_probabilities(
        FixedLogits(),
        subset,
        batch_size=2,
        class_count=2,
        device_name="cpu",
    )

    np.testing.assert_allclose(observed, expected, rtol=0, atol=0)
    np.testing.assert_allclose(observed.sum(axis=1), 1.0, rtol=0, atol=1e-7)


def test_trial_selection_uses_macro_f1_then_lowest_trial_id() -> None:
    manifests = [
        {
            "identity": {"trial": {"id": 2}},
            "result": {"best_validation_macro_f1": 0.8},
        },
        {
            "identity": {"trial": {"id": 1}},
            "result": {"best_validation_macro_f1": 0.8},
        },
        {
            "identity": {"trial": {"id": 3}},
            "result": {"best_validation_macro_f1": 0.7},
        },
    ]
    assert _select_trial(manifests)["identity"]["trial"]["id"] == 1


def test_tuning_selection_rejects_mixed_execution_environments() -> None:
    config = load_neural_config(Path(__file__).parents[1] / "configs" / "neural.yaml")
    manifests = [
        {
            "identity": {"trial": {"id": trial_id}},
            "result": {"best_validation_macro_f1": 0.5, "device": device},
            "environment": {"torch": "test", "platform": platform},
        }
        for trial_id, device, platform in (
            (1, "cpu", "macOS"),
            (2, "cuda", "Linux"),
        )
    ]
    with pytest.raises(PipelineInvariantError, match="mix execution environments"):
        _selection_payload(
            "cnn1d",
            manifests,
            neural=config,
            input_hashes={"canonical": "test"},
            provenance={"revision": "test", "dirty": False},
        )


def test_trial_publication_is_atomic_and_resumable(tmp_path: Path) -> None:
    config = load_neural_config(Path(__file__).parents[1] / "configs" / "neural.yaml")
    policy = _training_policy()
    policy["maximum_epochs"] = 2
    config = replace(config, training=policy, tuning_output_root=tmp_path)
    trial = replace(config.trials[0], batch_size=8, width=8)
    development = PreparedNeuralDevelopment(
        fold=_fold(),
        state=_state(),
        channels=config.channels,
        training=_subset("train", 24, offset=0),
        validation=_subset("validation", 12, offset=24),
    )
    run = RunSpec(
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
    inputs = {"canonical": "synthetic"}
    provenance = {"revision": "abc123", "dirty": False, "status_available": True}
    expected_data = {
        "training_pair_count": 24,
        "validation_pair_count": 12,
        "fit_pair_ids_sha256": "synthetic-fit-hash",
        "prefix_length": 50,
        "channels": list(config.channels),
        "class_order": ["A", "B"],
        "class_weights": [1.0, 1.0],
        "test_views_materialized": False,
    }
    first = _publish_trial(
        tmp_path / "cnn1d",
        model_name="cnn1d",
        trial=trial,
        neural=config,
        development=development,
        run=run,
        input_hashes=inputs,
        expected_data=expected_data,
        provenance=provenance,
        device_name="cpu",
    )
    second = _publish_trial(
        tmp_path / "cnn1d",
        model_name="cnn1d",
        trial=trial,
        neural=config,
        development=development,
        run=run,
        input_hashes=inputs,
        expected_data=expected_data,
        provenance=provenance,
        device_name="cpu",
    )

    assert first == second
    manifest_path = tmp_path / "cnn1d" / "trial_01" / "trial.json"
    assert manifest_path.is_file()
    assert not list((tmp_path / "cnn1d").glob(".trial_01-*"))

    tampered = json.loads(manifest_path.read_text(encoding="utf-8"))
    tampered["result"]["parameter_count"] += 1
    manifest_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(PipelineInvariantError, match="parameter count"):
        _publish_trial(
            tmp_path / "cnn1d",
            model_name="cnn1d",
            trial=trial,
            neural=config,
            development=development,
            run=run,
            input_hashes=inputs,
            expected_data=expected_data,
            provenance=provenance,
            device_name="cpu",
        )
