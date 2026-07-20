from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vpncat.errors import PipelineInvariantError
from vpncat.experiment import RunSpec
from vpncat.models import classical
from vpncat.models.classical import fit_classical_model
from vpncat.preprocessing import FoldTargetState
from vpncat.primary_data import prepare_classical_run
from vpncat.primary_runner import build_prediction_frame
from vpncat.schema import CANONICAL_STATS


def _run(*, model: str = "random_forest") -> RunSpec:
    return RunSpec(
        protocol="primary",
        experiment_id=f"matched_flow_stats__{model}",
        representation="matched_flow_stats",
        model=model,
        family="classical",
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
    for column_index, name in enumerate(CANONICAL_STATS):
        base = np.arange(row_count, dtype=np.float64) + column_index + 1
        frame[f"inner_{name}"] = base
        frame[f"outer_{name}"] = base + 0.5
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


def _small_rf_parameters() -> dict[str, object]:
    return {
        "n_estimators": 10,
        "criterion": "gini",
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "bootstrap": True,
        "class_weight": "balanced",
        "n_jobs": 1,
        "random_state": "run_seed",
    }


def test_preparation_excludes_validation_test_and_opposite_domain(tmp_path: Path) -> None:
    canonical_path, split_path = _write_inputs(tmp_path)
    baseline = prepare_classical_run(
        canonical_path,
        split_path,
        _run(),
        prefix_length=50,
    )

    poisoned = pd.read_parquet(canonical_path)
    poisoned.loc[4:, [f"inner_{name}" for name in CANONICAL_STATS]] = 1e30
    poisoned.loc[:, [f"outer_{name}" for name in CANONICAL_STATS]] = -1e30
    poisoned_path = tmp_path / "poisoned.parquet"
    poisoned.to_parquet(poisoned_path, index=False)
    observed = prepare_classical_run(
        poisoned_path,
        split_path,
        _run(),
        prefix_length=50,
    )

    assert baseline.state.to_dict() == observed.state.to_dict()
    np.testing.assert_array_equal(baseline.training_values, observed.training_values)
    assert len(observed.training_values) == 4


def test_random_forest_fit_and_dual_view_predictions(tmp_path: Path) -> None:
    canonical_path, split_path = _write_inputs(tmp_path)
    run = _run()
    prepared = prepare_classical_run(
        canonical_path,
        split_path,
        run,
        prefix_length=50,
    )
    fitted = fit_classical_model(
        run.model,
        _small_rf_parameters(),
        prepared.state,
        prepared.training_values,
        prepared.training_targets,
        prepared.training_labels,
        seed=run.seed,
    )
    probabilities = {
        domain: fitted.predict_probabilities(prepared.test_values[domain])
        for domain in run.test_domains
    }
    predictions = build_prediction_frame(run, prepared, probabilities)

    assert len(predictions) == 4
    assert set(predictions["test_domain"]) == {"inner", "outer"}
    inner_ids = set(predictions.loc[predictions["test_domain"] == "inner", "pair_id"])
    outer_ids = set(predictions.loc[predictions["test_domain"] == "outer", "pair_id"])
    assert inner_ids == outer_ids == set(prepared.fold.pair_ids_for("test"))
    assert fitted.recorded_hyperparameters["random_state"] == 42
    assert fitted.recorded_hyperparameters["class_weight_policy"] == (
        "training-fold-balanced"
    )


def test_xgboost_dependency_failure_is_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    state = FoldTargetState(
        fold=1,
        classes=("A", "B"),
        class_weights=np.asarray([1.0, 1.0]),
        fit_pair_count=4,
        fit_pair_ids_sha256="unused",
    )

    def unavailable() -> type:
        raise RuntimeError("install the 'classical' optional dependency")

    monkeypatch.setattr(classical, "_xgboost_classifier", unavailable)
    parameters = {
        "random_state": "run_seed",
        "sample_weight": "balanced",
    }
    with pytest.raises(RuntimeError, match="optional dependency"):
        fit_classical_model(
            "xgboost",
            parameters,
            state,
            np.asarray([[0.0], [1.0], [2.0], [3.0]]),
            np.asarray([0, 0, 1, 1]),
            np.asarray(["A", "A", "B", "B"]),
            seed=42,
        )


def test_classical_preparation_rejects_neural_run(tmp_path: Path) -> None:
    canonical_path, split_path = _write_inputs(tmp_path)
    neural = replace(_run(), family="neural")
    with pytest.raises(PipelineInvariantError, match="incompatible run"):
        prepare_classical_run(
            canonical_path,
            split_path,
            neural,
            prefix_length=50,
        )
