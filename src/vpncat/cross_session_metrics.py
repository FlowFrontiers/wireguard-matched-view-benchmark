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

from vpncat.cross_session import CrossSessionRunSpec
from vpncat.cross_session_index import CrossSessionIndex
from vpncat.errors import PipelineInvariantError
from vpncat.metrics import METRIC_NAMES

CROSS_SESSION_PREDICTION_COLUMNS = (
    "run_id",
    "protocol",
    "representation",
    "model",
    "pair_id",
    "session",
    "train_session",
    "test_session",
    "train_domain",
    "test_domain",
    "seed",
    "true_label",
    "prediction",
    "class_probabilities",
)


def _probability_matrix(frame: pd.DataFrame, classes: tuple[str, ...]) -> np.ndarray:
    try:
        probabilities = np.asarray(
            frame["class_probabilities"].tolist(), dtype=np.float64
        )
    except (TypeError, ValueError) as error:
        raise PipelineInvariantError(
            "Cross-session probabilities do not form a numeric matrix"
        ) from error
    if probabilities.shape != (len(frame), len(classes)):
        raise PipelineInvariantError("Cross-session probability shape is invalid")
    if not np.isfinite(probabilities).all():
        raise PipelineInvariantError("Cross-session probabilities are not finite")
    if (probabilities < 0).any() or (probabilities > 1).any():
        raise PipelineInvariantError("Cross-session probabilities must lie in [0, 1]")
    if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=1e-6, atol=1e-8):
        raise PipelineInvariantError("Cross-session probabilities do not sum to one")
    return probabilities


def validate_cross_session_predictions(
    frame: pd.DataFrame,
    *,
    run: CrossSessionRunSpec,
    index: CrossSessionIndex,
    classes: tuple[str, ...],
) -> np.ndarray:
    missing = set(CROSS_SESSION_PREDICTION_COLUMNS) - set(frame.columns)
    extra = set(frame.columns) - set(CROSS_SESSION_PREDICTION_COLUMNS)
    if missing or extra:
        raise PipelineInvariantError(
            f"Cross-session prediction schema differs: missing={sorted(missing)}, "
            f"extra={sorted(extra)}"
        )
    if len(classes) < 2 or len(set(classes)) != len(classes):
        raise PipelineInvariantError("Cross-session class vocabulary must be unique")
    constants: dict[str, object] = {
        "run_id": run.run_id,
        "protocol": run.protocol,
        "representation": run.representation,
        "model": run.model,
        "train_session": run.train_session,
        "test_session": run.test_session,
        "train_domain": run.train_domain,
        "seed": run.seed,
    }
    for column, expected in constants.items():
        if set(frame[column]) != {expected}:
            raise PipelineInvariantError(
                f"Cross-session prediction {column} disagrees with run identity"
            )
    if tuple(sorted(frame["test_domain"].astype(str).unique())) != tuple(
        sorted(run.test_domains)
    ):
        raise PipelineInvariantError("Cross-session prediction domains are incomplete")
    if frame.duplicated(["pair_id", "test_domain"]).any():
        raise PipelineInvariantError("Cross-session predictions contain duplicate rows")

    expected_ids = set(index.pair_ids_for("test"))
    expected_sessions = {
        index.pair_ids[position]: int(index.sessions[position])
        for position in index.test_positions
    }
    expected_labels = {
        index.pair_ids[position]: index.labels[position]
        for position in index.test_positions
    }
    for domain in run.test_domains:
        selected = frame.loc[frame["test_domain"] == domain]
        observed_ids = set(selected["pair_id"].astype(str))
        if observed_ids != expected_ids or len(selected) != len(expected_ids):
            raise PipelineInvariantError(
                f"Cross-session {domain} predictions do not exactly cover target pairs"
            )
        sessions = selected.set_index("pair_id")["session"].astype(int).to_dict()
        labels = selected.set_index("pair_id")["true_label"].astype(str).to_dict()
        if sessions != expected_sessions or set(sessions.values()) != {run.test_session}:
            raise PipelineInvariantError("Cross-session prediction sessions are invalid")
        if labels != expected_labels:
            raise PipelineInvariantError("Cross-session prediction labels are invalid")
        if set(labels.values()) != set(classes):
            raise PipelineInvariantError("Cross-session test view omits one or more classes")

    probabilities = _probability_matrix(frame, classes)
    predictions = np.asarray(classes, dtype=object)[np.argmax(probabilities, axis=1)]
    if not np.array_equal(predictions.astype(str), frame["prediction"].astype(str)):
        raise PipelineInvariantError("Cross-session predictions disagree with argmax")
    if set(frame["true_label"].astype(str)) - set(classes):
        raise PipelineInvariantError("Cross-session true labels are outside vocabulary")
    return probabilities


def compute_cross_session_metrics(
    frame: pd.DataFrame,
    *,
    run: CrossSessionRunSpec,
    index: CrossSessionIndex,
    classes: tuple[str, ...],
) -> dict[str, dict[str, float]]:
    probabilities = validate_cross_session_predictions(
        frame, run=run, index=index, classes=classes
    )
    metrics: dict[str, dict[str, float]] = {}
    class_indices = {label: position for position, label in enumerate(classes)}
    for domain in run.test_domains:
        selected_mask = frame["test_domain"].eq(domain).to_numpy()
        selected = frame.loc[selected_mask]
        y_true = selected["true_label"].astype(str).to_numpy()
        y_pred = selected["prediction"].astype(str).to_numpy()
        y_binary = np.zeros((len(y_true), len(classes)), dtype=np.int8)
        y_binary[
            np.arange(len(y_true)),
            np.asarray([class_indices[label] for label in y_true], dtype=np.int64),
        ] = 1
        values = {
            "accuracy": accuracy_score(y_true, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "macro_f1": f1_score(
                y_true, y_pred, labels=list(classes), average="macro", zero_division=0
            ),
            "weighted_f1": f1_score(
                y_true, y_pred, labels=list(classes), average="weighted", zero_division=0
            ),
            "macro_ovr_average_precision": average_precision_score(
                y_binary, probabilities[selected_mask], average="macro"
            ),
        }
        metrics[domain] = {name: float(values[name]) for name in METRIC_NAMES}
    return metrics


def cross_session_metrics_long_frame(
    metrics: dict[str, dict[str, float]],
    *,
    run: CrossSessionRunSpec,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for domain in run.test_domains:
        if set(metrics.get(domain, {})) != set(METRIC_NAMES):
            raise PipelineInvariantError(
                f"Cross-session metrics are incomplete for {domain}"
            )
        for metric in METRIC_NAMES:
            rows.append(
                {
                    "run_id": run.run_id,
                    "protocol": run.protocol,
                    "experiment_id": run.experiment_id,
                    "representation": run.representation,
                    "model": run.model,
                    "family": run.family,
                    "train_session": run.train_session,
                    "test_session": run.test_session,
                    "seed": run.seed,
                    "train_domain": run.train_domain,
                    "test_domain": domain,
                    "metric": metric,
                    "value": metrics[domain][metric],
                }
            )
    return pd.DataFrame(rows)
