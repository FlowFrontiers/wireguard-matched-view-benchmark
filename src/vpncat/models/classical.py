from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from vpncat.errors import PipelineInvariantError
from vpncat.preprocessing import FoldPreprocessingState, FoldTargetState

TargetState = FoldPreprocessingState | FoldTargetState


@dataclass(frozen=True)
class FittedClassicalModel:
    estimator: Any
    recorded_hyperparameters: dict[str, Any]

    def predict_probabilities(self, values: np.ndarray) -> np.ndarray:
        probabilities = np.asarray(self.estimator.predict_proba(values), dtype=np.float64)
        expected_classes = np.arange(len(self.estimator.classes_), dtype=np.int64)
        if not np.array_equal(np.asarray(self.estimator.classes_), expected_classes):
            raise PipelineInvariantError("Estimator class order differs from encoded class order")
        if probabilities.shape != (len(values), len(expected_classes)):
            raise PipelineInvariantError("Estimator returned an invalid probability matrix shape")
        if not np.isfinite(probabilities).all():
            raise PipelineInvariantError("Estimator returned non-finite probabilities")
        if (probabilities < 0).any() or (probabilities > 1).any():
            raise PipelineInvariantError("Estimator probabilities lie outside [0, 1]")
        if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=1e-6, atol=1e-8):
            raise PipelineInvariantError("Estimator probabilities do not sum to one")
        return probabilities


def _xgboost_classifier() -> type:
    try:
        from xgboost import XGBClassifier
    except ImportError as error:
        raise RuntimeError(
            "XGBoost is required for this run; install the 'classical' optional dependency"
        ) from error
    return XGBClassifier


def _validate_fit_arrays(
    values: np.ndarray,
    targets: np.ndarray,
    labels: np.ndarray,
    *,
    class_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(values)
    targets = np.asarray(targets, dtype=np.int64)
    labels = np.asarray(labels, dtype=object).astype(str)
    if values.ndim != 2 or len(values) == 0:
        raise PipelineInvariantError("Classical training features must be a non-empty matrix")
    if not np.isfinite(values).all():
        raise PipelineInvariantError("Classical training features contain non-finite values")
    if targets.shape != (len(values),) or labels.shape != (len(values),):
        raise PipelineInvariantError("Classical training features and labels are misaligned")
    if not np.array_equal(np.unique(targets), np.arange(class_count, dtype=np.int64)):
        raise PipelineInvariantError("Classical training data does not cover every encoded class")
    return values, targets, labels


def fit_classical_model(
    model_name: str,
    hyperparameters: dict[str, Any],
    state: TargetState,
    values: np.ndarray,
    targets: np.ndarray,
    training_labels: np.ndarray,
    *,
    seed: int,
) -> FittedClassicalModel:
    """Fit one frozen classical model with exactly one balancing mechanism."""
    values, targets, training_labels = _validate_fit_arrays(
        values,
        targets,
        training_labels,
        class_count=len(state.classes),
    )
    parameters = dict(hyperparameters)
    if parameters.pop("random_state", None) != "run_seed":
        raise PipelineInvariantError("Classical random_state must be bound to the run seed")
    parameters["random_state"] = seed

    if model_name == "random_forest":
        if parameters.pop("class_weight", None) != "balanced":
            raise PipelineInvariantError("Random Forest must use frozen balanced class weights")
        class_weights = state.class_weight_dict()
        estimator = RandomForestClassifier(class_weight=class_weights, **parameters)
        estimator.fit(values, targets)
        recorded = {
            **parameters,
            "class_weight": {str(key): value for key, value in class_weights.items()},
            "class_weight_policy": "training-fold-balanced",
        }
    elif model_name == "xgboost":
        if parameters.pop("sample_weight", None) != "balanced":
            raise PipelineInvariantError("XGBoost must use frozen balanced sample weights")
        estimator = _xgboost_classifier()(**parameters)
        estimator.fit(values, targets, sample_weight=state.sample_weights(training_labels))
        recorded = {
            **parameters,
            "sample_weight_policy": "training-fold-balanced",
        }
    else:
        raise PipelineInvariantError(f"Unsupported classical model: {model_name}")

    expected_classes = np.arange(len(state.classes), dtype=np.int64)
    if not np.array_equal(np.asarray(estimator.classes_), expected_classes):
        raise PipelineInvariantError("Fitted estimator does not expose the complete class order")
    return FittedClassicalModel(
        estimator=estimator,
        recorded_hyperparameters=recorded,
    )
