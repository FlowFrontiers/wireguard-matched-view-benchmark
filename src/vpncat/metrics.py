from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
)

from vpncat.errors import PipelineInvariantError
from vpncat.experiment import RunSpec
from vpncat.folds import FoldIndex

PREDICTION_COLUMNS = (
    "run_id",
    "protocol",
    "representation",
    "model",
    "pair_id",
    "session",
    "train_domain",
    "test_domain",
    "fold",
    "seed",
    "true_label",
    "prediction",
    "class_probabilities",
)
METRIC_NAMES = (
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "weighted_f1",
    "macro_ovr_average_precision",
)


def _probability_matrix(frame: pd.DataFrame, classes: tuple[str, ...]) -> np.ndarray:
    try:
        probabilities = np.asarray(frame["class_probabilities"].tolist(), dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise PipelineInvariantError("Class probabilities do not form a numeric matrix") from error
    if probabilities.shape != (len(frame), len(classes)):
        raise PipelineInvariantError(
            "Class-probability shape disagrees with row count or class vocabulary"
        )
    if not np.isfinite(probabilities).all():
        raise PipelineInvariantError("Class probabilities contain non-finite values")
    if (probabilities < 0).any() or (probabilities > 1).any():
        raise PipelineInvariantError("Class probabilities must lie in [0, 1]")
    if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=1e-6, atol=1e-8):
        raise PipelineInvariantError("Class probabilities must sum to one per row")
    return probabilities


def validate_predictions(
    frame: pd.DataFrame,
    *,
    run: RunSpec,
    fold: FoldIndex,
    classes: tuple[str, ...],
) -> np.ndarray:
    """Validate complete paired held-out predictions for one trained model."""
    missing = set(PREDICTION_COLUMNS) - set(frame.columns)
    extra = set(frame.columns) - set(PREDICTION_COLUMNS)
    if missing or extra:
        raise PipelineInvariantError(
            f"Prediction schema differs: missing={sorted(missing)}, extra={sorted(extra)}"
        )
    if len(classes) < 2 or len(set(classes)) != len(classes):
        raise PipelineInvariantError("Prediction class vocabulary must be unique")
    constants: dict[str, object] = {
        "run_id": run.run_id,
        "protocol": run.protocol,
        "representation": run.representation,
        "model": run.model,
        "train_domain": run.train_domain,
        "fold": run.fold,
        "seed": run.seed,
    }
    for column, expected in constants.items():
        observed = set(frame[column])
        if observed != {expected}:
            raise PipelineInvariantError(
                f"Prediction {column} values {observed} disagree with run value {expected}"
            )
    if tuple(sorted(frame["test_domain"].astype(str).unique())) != tuple(
        sorted(run.test_domains)
    ):
        raise PipelineInvariantError("Prediction test domains are incomplete")
    if frame.duplicated(["pair_id", "test_domain"]).any():
        raise PipelineInvariantError("Predictions contain duplicate pair/domain rows")

    expected_positions = fold.test_positions
    expected_ids = set(fold.pair_ids_for("test"))
    expected_sessions = {
        fold.pair_ids[index]: int(fold.sessions[index]) for index in expected_positions
    }
    expected_labels = {
        fold.pair_ids[index]: fold.labels[index] for index in expected_positions
    }
    for domain in run.test_domains:
        selected = frame.loc[frame["test_domain"] == domain]
        observed_ids = set(selected["pair_id"].astype(str))
        if observed_ids != expected_ids or len(selected) != len(expected_ids):
            raise PipelineInvariantError(
                f"{domain} predictions do not exactly cover held-out pair IDs"
            )
        observed_sessions = selected.set_index("pair_id")["session"].astype(int).to_dict()
        observed_labels = selected.set_index("pair_id")["true_label"].astype(str).to_dict()
        if observed_sessions != expected_sessions:
            raise PipelineInvariantError(
                f"{domain} prediction sessions disagree with canonical data"
            )
        if observed_labels != expected_labels:
            raise PipelineInvariantError(f"{domain} prediction labels disagree with canonical data")
        if set(observed_labels.values()) != set(classes):
            raise PipelineInvariantError(
                f"{domain} held-out predictions do not cover every class"
            )

    probabilities = _probability_matrix(frame, classes)
    predictions = np.asarray(classes, dtype=object)[np.argmax(probabilities, axis=1)]
    if not np.array_equal(predictions.astype(str), frame["prediction"].astype(str)):
        raise PipelineInvariantError("Prediction labels disagree with probability argmax")
    unknown_true = set(frame["true_label"].astype(str)) - set(classes)
    if unknown_true:
        raise PipelineInvariantError(f"Prediction labels are outside vocabulary: {unknown_true}")
    return probabilities


def compute_metrics(
    frame: pd.DataFrame,
    *,
    run: RunSpec,
    fold: FoldIndex,
    classes: tuple[str, ...],
) -> dict[str, dict[str, float]]:
    probabilities = validate_predictions(frame, run=run, fold=fold, classes=classes)
    metrics: dict[str, dict[str, float]] = {}
    for domain in run.test_domains:
        selected_mask = frame["test_domain"].eq(domain).to_numpy()
        selected = frame.loc[selected_mask]
        domain_probabilities = probabilities[selected_mask]
        y_true = selected["true_label"].astype(str).to_numpy()
        y_pred = selected["prediction"].astype(str).to_numpy()
        class_indices = {label: index for index, label in enumerate(classes)}
        y_binary = np.zeros((len(y_true), len(classes)), dtype=np.int8)
        y_binary[
            np.arange(len(y_true)),
            np.asarray([class_indices[label] for label in y_true], dtype=np.int64),
        ] = 1
        values = {
            "accuracy": accuracy_score(y_true, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "macro_f1": f1_score(
                y_true,
                y_pred,
                labels=list(classes),
                average="macro",
                zero_division=0,
            ),
            "weighted_f1": f1_score(
                y_true,
                y_pred,
                labels=list(classes),
                average="weighted",
                zero_division=0,
            ),
            "macro_ovr_average_precision": average_precision_score(
                y_binary,
                domain_probabilities,
                average="macro",
            ),
        }
        metrics[domain] = {name: float(values[name]) for name in METRIC_NAMES}
    return metrics


def metrics_long_frame(
    metrics: dict[str, dict[str, float]],
    *,
    run: RunSpec,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for test_domain in run.test_domains:
        if set(metrics.get(test_domain, {})) != set(METRIC_NAMES):
            raise PipelineInvariantError(f"Metrics are incomplete for {test_domain}")
        for metric in METRIC_NAMES:
            rows.append(
                {
                    "run_id": run.run_id,
                    "protocol": run.protocol,
                    "experiment_id": run.experiment_id,
                    "representation": run.representation,
                    "model": run.model,
                    "family": run.family,
                    "fold": run.fold,
                    "seed": run.seed,
                    "train_domain": run.train_domain,
                    "test_domain": test_domain,
                    "metric": metric,
                    "value": metrics[test_domain][metric],
                }
            )
    return pd.DataFrame(rows)
