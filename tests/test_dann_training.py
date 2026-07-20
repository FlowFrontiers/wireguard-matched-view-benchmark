from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import torch

from vpncat.dann_data import UnlabeledNeuralSubset
from vpncat.dann_training import (
    DANN_HISTORY_COLUMNS,
    build_paired_dann_loader,
    predict_dann_probabilities,
    train_dann_model,
)
from vpncat.errors import PipelineInvariantError
from vpncat.models.dann import (
    FROZEN_DOMAIN_HEAD,
    build_dann_model,
    gradient_reverse,
    logistic_grl_coefficient,
)
from vpncat.neural_config import FROZEN_OPTIMIZER, FROZEN_TOPOLOGIES, FROZEN_TRAINING
from vpncat.neural_data import NeuralSubset
from vpncat.neural_training import seed_neural_execution
from vpncat.preprocessing import FoldTargetState


def _state() -> FoldTargetState:
    return FoldTargetState(
        fold=1,
        classes=("A", "B"),
        class_weights=np.asarray([1.0, 1.0]),
        fit_pair_count=24,
        fit_pair_ids_sha256="synthetic-fit-hash",
    )


def _source(prefix: str, count: int, *, offset: int) -> NeuralSubset:
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


def _adaptation(source: NeuralSubset) -> UnlabeledNeuralSubset:
    values = source.values.copy()
    values[source.mask, 1] += 0.25
    return UnlabeledNeuralSubset(
        pair_ids=source.pair_ids,
        positions=source.positions.copy(),
        values=values,
        mask=source.mask.copy(),
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


def _gradient_reversal_policy() -> dict[str, object]:
    return {
        "schedule": "logistic",
        "gamma": 10.0,
        "start": 0.0,
        "end": 1.0,
    }


def _model():
    return build_dann_model(
        feature_count=3,
        class_count=2,
        width=8,
        dropout=0.2,
        maximum_length=80,
        topology=FROZEN_TOPOLOGIES["cnn1d"],
        domain_head=FROZEN_DOMAIN_HEAD,
    )


def _train_once():
    source = _source("train", 24, offset=0)
    seed_neural_execution(42)
    model = _model()
    return train_dann_model(
        model,
        source,
        _adaptation(source),
        _source("validation", 12, offset=24),
        _state(),
        learning_rate=0.001,
        batch_size=8,
        seed=42,
        domain_loss_weight=1.0,
        gradient_reversal=_gradient_reversal_policy(),
        optimizer_policy=FROZEN_OPTIMIZER,
        training_policy=_training_policy(),
        device_name="cpu",
    )


def test_gradient_reversal_is_identity_forward_and_reverses_scaled_gradient() -> None:
    values = torch.tensor([1.0, -2.0, 3.0], requires_grad=True)
    observed = gradient_reverse(values, 0.25)
    torch.testing.assert_close(observed, values, rtol=0.0, atol=0.0)
    observed.sum().backward()
    torch.testing.assert_close(
        values.grad,
        torch.full_like(values, -0.25),
        rtol=0.0,
        atol=0.0,
    )


def test_logistic_gradient_reversal_schedule_is_standard_and_monotonic() -> None:
    observed = [
        logistic_grl_coefficient(p, gamma=10.0, start=0.0, end=1.0)
        for p in np.linspace(0.0, 1.0, 101)
    ]
    assert observed[0] == 0.0
    assert observed[-1] == pytest.approx(2.0 / (1.0 + np.exp(-10.0)) - 1.0)
    assert all(
        left < right for left, right in zip(observed, observed[1:], strict=False)
    )
    with pytest.raises(PipelineInvariantError, match="schedule"):
        logistic_grl_coefficient(1.01, gamma=10.0, start=0.0, end=1.0)


def test_dann_model_reuses_cnn_logits_and_ignores_padded_poison() -> None:
    seed_neural_execution(42)
    model = _model().eval()
    values = torch.randn(4, 8, 3)
    mask = torch.tensor(
        [
            [True] * 8,
            [True] * 6 + [False] * 2,
            [True] * 4 + [False] * 4,
            [True] * 2 + [False] * 6,
        ]
    )
    values[~mask] = 0.0
    poisoned = values.clone()
    poisoned[~mask] = 1e6
    with torch.no_grad():
        class_logits, domain_logits = model(values, mask, grl_coefficient=0.5)
        poisoned_class, poisoned_domain = model(
            poisoned, mask, grl_coefficient=0.5
        )
        backbone_logits = model.backbone(values, mask)
    torch.testing.assert_close(class_logits, backbone_logits, rtol=0.0, atol=0.0)
    torch.testing.assert_close(class_logits, poisoned_class, rtol=0.0, atol=1e-6)
    torch.testing.assert_close(domain_logits, poisoned_domain, rtol=0.0, atol=1e-6)
    assert class_logits.shape == (4, 2)
    assert domain_logits.shape == (4,)


def test_dann_domain_head_must_match_selected_backbone() -> None:
    seed_neural_execution(42)
    model = _model()
    with pytest.raises(PipelineInvariantError, match="backbone selection"):
        type(model)(
            model.backbone,
            width=16,
            dropout=0.2,
            domain_head=FROZEN_DOMAIN_HEAD,
        )


def test_paired_loader_uses_one_permutation_for_both_views() -> None:
    source = _source("pair", 13, offset=0)
    source_values = np.zeros_like(source.values)
    source_values[:, 0, 0] = np.arange(13)
    source = replace(source, values=source_values)
    adaptation = _adaptation(source)
    adaptation_values = adaptation.values.copy()
    adaptation_values[:, 0, 0] += 100.0
    adaptation = replace(adaptation, values=adaptation_values)
    loader = build_paired_dann_loader(
        source,
        adaptation,
        batch_size=4,
        shuffle=True,
        seed=42,
        workers=0,
    )
    seen: list[int] = []
    for source_values, _, _, target_values, _, row_indices in loader:
        torch.testing.assert_close(
            source_values[:, 0, 0],
            row_indices.to(dtype=torch.float32),
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            target_values[:, 0, 0],
            row_indices.to(dtype=torch.float32) + 100.0,
            rtol=0.0,
            atol=0.0,
        )
        seen.extend(row_indices.tolist())
    assert sorted(seen) == list(range(13))


def test_dann_training_is_deterministic_and_source_validation_selects_checkpoint() -> None:
    first = _train_once()
    second = _train_once()
    pd.testing.assert_frame_equal(first.history, second.history, check_exact=True)
    assert tuple(first.history.columns) == DANN_HISTORY_COLUMNS
    assert first.best_validation_macro_f1 == first.history["validation_macro_f1"].max()
    assert first.best_epoch == int(
        first.history.loc[first.history["validation_macro_f1"].idxmax(), "epoch"]
    )
    assert first.parameter_count > first.backbone_parameter_count > 0
    assert first.history["grl_coefficient_start"].iloc[0] == 0.0
    assert first.history["grl_coefficient_end"].is_monotonic_increasing
    for name, value in first.model.state_dict().items():
        torch.testing.assert_close(value, second.model.state_dict()[name], rtol=0, atol=0)


def test_dann_prediction_bypasses_domain_head_and_softmaxes_once() -> None:
    subset = _source("test", 5, offset=30)

    class FixedClassifier(torch.nn.Module):
        def classify(self, values, mask):
            del mask
            return values[:, 0, :2]

    expected = torch.softmax(torch.from_numpy(subset.values[:, 0, :2]), dim=1).numpy()
    observed = predict_dann_probabilities(
        FixedClassifier(),
        subset,
        batch_size=2,
        class_count=2,
        device_name="cpu",
    )
    np.testing.assert_allclose(observed, expected, rtol=0, atol=0)
    np.testing.assert_allclose(observed.sum(axis=1), 1.0, rtol=0, atol=1e-7)


def test_dann_trainer_rejects_pairing_and_policy_drift() -> None:
    source = _source("train", 8, offset=0)
    adaptation = _adaptation(source)
    validation = _source("validation", 4, offset=8)
    with pytest.raises(PipelineInvariantError, match="mispaired"):
        train_dann_model(
            _model(),
            source,
            replace(adaptation, pair_ids=tuple(reversed(adaptation.pair_ids))),
            validation,
            _state(),
            learning_rate=0.001,
            batch_size=4,
            seed=42,
            domain_loss_weight=1.0,
            gradient_reversal=_gradient_reversal_policy(),
            optimizer_policy=FROZEN_OPTIMIZER,
            training_policy=_training_policy(),
            device_name="cpu",
        )
    changed_policy = _gradient_reversal_policy()
    changed_policy["gamma"] = 5.0
    with pytest.raises(PipelineInvariantError, match="policy"):
        train_dann_model(
            _model(),
            source,
            adaptation,
            validation,
            _state(),
            learning_rate=0.001,
            batch_size=4,
            seed=42,
            domain_loss_weight=1.0,
            gradient_reversal=changed_policy,
            optimizer_policy=FROZEN_OPTIMIZER,
            training_policy=_training_policy(),
            device_name="cpu",
        )
