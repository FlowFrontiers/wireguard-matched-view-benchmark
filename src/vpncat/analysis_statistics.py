from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
)

from vpncat.errors import PipelineInvariantError
from vpncat.metrics import METRIC_NAMES


def _arrays(frame: pd.DataFrame, classes: tuple[str, ...]):
    required = {"pair_id", "test_domain", "true_label", "prediction", "class_probabilities"}
    if missing := required - set(frame.columns):
        raise PipelineInvariantError(f"Analysis predictions omit columns: {sorted(missing)}")
    try:
        probabilities = np.asarray(frame["class_probabilities"].tolist(), dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise PipelineInvariantError("Analysis probabilities are not numeric") from error
    if probabilities.shape != (len(frame), len(classes)) or not np.isfinite(probabilities).all():
        raise PipelineInvariantError("Analysis probability matrix is invalid")
    if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=1e-6, atol=1e-8):
        raise PipelineInvariantError("Analysis probabilities do not sum to one")
    expected = np.asarray(classes, dtype=object)[np.argmax(probabilities, axis=1)]
    if not np.array_equal(expected.astype(str), frame["prediction"].astype(str)):
        raise PipelineInvariantError("Analysis prediction labels disagree with argmax")
    return frame["true_label"].astype(str).to_numpy(), expected.astype(str), probabilities


def _metric_values(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    classes: tuple[str, ...],
) -> dict[str, float]:
    class_index = {label: index for index, label in enumerate(classes)}
    binary = np.zeros((len(y_true), len(classes)), dtype=np.int8)
    binary[np.arange(len(y_true)), [class_index[label] for label in y_true]] = 1
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=list(classes), average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_true, y_pred, labels=list(classes), average="weighted", zero_division=0)
        ),
        "macro_ovr_average_precision": float(
            average_precision_score(binary, probabilities, average="macro")
        ),
    }


def compute_analysis_metrics(
    frame: pd.DataFrame,
    classes: tuple[str, ...],
) -> dict[str, dict[str, float]]:
    if tuple(sorted(frame["test_domain"].astype(str).unique())) != ("inner", "outer"):
        raise PipelineInvariantError("Analysis predictions require paired inner/outer views")
    if frame.duplicated(["pair_id", "test_domain"]).any():
        raise PipelineInvariantError("Analysis predictions duplicate pair/domain rows")
    result = {}
    for domain in ("inner", "outer"):
        selected = frame.loc[frame["test_domain"].eq(domain)].sort_values("pair_id")
        y_true, y_pred, probabilities = _arrays(selected, classes)
        result[domain] = _metric_values(y_true, y_pred, probabilities, classes)
    if set(result["inner"]) != set(METRIC_NAMES):
        raise PipelineInvariantError("Analysis metric set differs from the freeze")
    return result


def paired_bootstrap_intervals(
    frame: pd.DataFrame,
    classes: tuple[str, ...],
    *,
    metrics: tuple[str, ...],
    replicates: int,
    confidence_level: float,
    seed: int,
) -> list[dict[str, float | str | int]]:
    views = {
        domain: frame.loc[frame["test_domain"].eq(domain)]
        .sort_values("pair_id")
        .reset_index(drop=True)
        for domain in ("inner", "outer")
    }
    if not np.array_equal(views["inner"]["pair_id"], views["outer"]["pair_id"]):
        raise PipelineInvariantError("Bootstrap views are not paired by pair_id")
    if not np.array_equal(views["inner"]["true_label"], views["outer"]["true_label"]):
        raise PipelineInvariantError("Bootstrap paired labels differ")
    arrays = {domain: _arrays(view, classes) for domain, view in views.items()}
    if not metrics or any(
        metric not in {"balanced_accuracy", "macro_f1"} for metric in metrics
    ):
        raise PipelineInvariantError("Bootstrap metrics differ from primary metrics")
    n = len(views["inner"])
    class_index = {label: index for index, label in enumerate(classes)}
    class_count = len(classes)
    true_indices = np.asarray(
        [class_index[label] for label in arrays["inner"][0]], dtype=np.int64
    )
    prediction_indices = {
        domain: np.asarray(
            [class_index[label] for label in arrays[domain][1]], dtype=np.int64
        )
        for domain in ("inner", "outer")
    }
    joint_codes = (
        true_indices * class_count * class_count
        + prediction_indices["inner"] * class_count
        + prediction_indices["outer"]
    )
    joint_counts = np.bincount(
        joint_codes, minlength=class_count**3
    ).astype(np.int64)
    active = np.flatnonzero(joint_counts)
    probabilities = joint_counts[active].astype(np.float64) / n
    true_state = active // (class_count * class_count)
    inner_state = (active // class_count) % class_count
    outer_state = active % class_count

    def confusion_metric(counts: np.ndarray, predicted: np.ndarray, metric: str) -> float:
        confusion = np.bincount(
            true_state * class_count + predicted,
            weights=counts,
            minlength=class_count * class_count,
        ).reshape(class_count, class_count)
        true_positive = np.diag(confusion)
        false_negative = confusion.sum(axis=1) - true_positive
        if metric == "balanced_accuracy":
            return float(
                np.divide(
                    true_positive,
                    true_positive + false_negative,
                    out=np.zeros(class_count),
                    where=(true_positive + false_negative) > 0,
                ).mean()
            )
        false_positive = confusion.sum(axis=0) - true_positive
        denominator = 2 * true_positive + false_positive + false_negative
        return float(
            np.divide(
                2 * true_positive,
                denominator,
                out=np.zeros(class_count),
                where=denominator > 0,
            ).mean()
        )

    rng = np.random.default_rng(seed)
    samples = {metric: {"inner": [], "outer": [], "gap": []} for metric in metrics}
    for _ in range(replicates):
        bootstrap_counts = rng.multinomial(n, probabilities)
        for metric in metrics:
            values = {
                "inner": confusion_metric(bootstrap_counts, inner_state, metric),
                "outer": confusion_metric(bootstrap_counts, outer_state, metric),
            }
            for domain in ("inner", "outer"):
                samples[metric][domain].append(values[domain])
            samples[metric]["gap"].append(values["outer"] - values["inner"])
    alpha = (1.0 - confidence_level) / 2.0
    point = compute_analysis_metrics(frame, classes)
    rows = []
    for metric in metrics:
        row: dict[str, float | str | int] = {"metric": metric, "replicates": replicates}
        for domain in ("inner", "outer"):
            values = np.asarray(samples[metric][domain])
            row[f"{domain}_estimate"] = point[domain][metric]
            row[f"{domain}_ci_low"] = float(np.quantile(values, alpha))
            row[f"{domain}_ci_high"] = float(np.quantile(values, 1.0 - alpha))
        gaps = np.asarray(samples[metric]["gap"])
        row["gap_estimate"] = point["outer"][metric] - point["inner"][metric]
        row["gap_ci_low"] = float(np.quantile(gaps, alpha))
        row["gap_ci_high"] = float(np.quantile(gaps, 1.0 - alpha))
        rows.append(row)
    return rows
