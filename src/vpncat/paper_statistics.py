from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)

from vpncat.errors import PipelineInvariantError

PRIMARY_METRICS = ("balanced_accuracy", "macro_f1")


def validated_domain_frame(
    frame: pd.DataFrame,
    classes: tuple[str, ...],
    *,
    domain: str,
) -> pd.DataFrame:
    """Return one pair-unique domain after validating labels and probabilities."""
    required = {
        "pair_id",
        "test_domain",
        "true_label",
        "prediction",
        "class_probabilities",
    }
    if missing := required - set(frame.columns):
        raise PipelineInvariantError(f"Paper predictions omit columns: {sorted(missing)}")
    selected = frame.loc[frame["test_domain"].eq(domain)].sort_values("pair_id", ignore_index=True)
    if selected.empty or selected["pair_id"].duplicated().any():
        raise PipelineInvariantError(f"Paper predictions are not pair-unique: {domain}")
    try:
        probabilities = np.asarray(selected["class_probabilities"].tolist(), dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise PipelineInvariantError("Paper probabilities are not numeric") from error
    if probabilities.shape != (len(selected), len(classes)):
        raise PipelineInvariantError("Paper probability shape differs from class order")
    if (
        not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or (probabilities > 1).any()
        or not np.allclose(probabilities.sum(axis=1), 1.0, rtol=1e-6, atol=1e-8)
    ):
        raise PipelineInvariantError("Paper probabilities violate the simplex")
    expected = np.asarray(classes, dtype=object)[np.argmax(probabilities, axis=1)]
    if not np.array_equal(expected.astype(str), selected["prediction"].astype(str)):
        raise PipelineInvariantError("Paper predictions disagree with probability argmax")
    if not set(selected["true_label"].astype(str)) <= set(classes):
        raise PipelineInvariantError("Paper labels differ from the class order")
    return selected


def metric_values(
    frame: pd.DataFrame,
    classes: tuple[str, ...],
    *,
    domain: str,
) -> dict[str, float]:
    selected = validated_domain_frame(frame, classes, domain=domain)
    true = selected["true_label"].astype(str).to_numpy()
    predicted = selected["prediction"].astype(str).to_numpy()
    probabilities = np.asarray(selected["class_probabilities"].tolist(), dtype=np.float64)
    class_index = {label: index for index, label in enumerate(classes)}
    binary = np.zeros((len(true), len(classes)), dtype=np.int8)
    binary[np.arange(len(true)), [class_index[label] for label in true]] = 1
    return {
        "accuracy": float(accuracy_score(true, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(true, predicted)),
        "macro_f1": float(
            f1_score(true, predicted, labels=list(classes), average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(
                true,
                predicted,
                labels=list(classes),
                average="weighted",
                zero_division=0,
            )
        ),
        "macro_ovr_average_precision": float(
            average_precision_score(binary, probabilities, average="macro")
        ),
    }


def per_class_metrics(
    frame: pd.DataFrame,
    classes: tuple[str, ...],
    *,
    domain: str,
) -> pd.DataFrame:
    selected = validated_domain_frame(frame, classes, domain=domain)
    precision, recall, f1, support = precision_recall_fscore_support(
        selected["true_label"].astype(str),
        selected["prediction"].astype(str),
        labels=list(classes),
        zero_division=0,
    )
    return pd.DataFrame(
        {
            "class_index": np.arange(len(classes), dtype=np.int64),
            "class_name": classes,
            "support": support.astype(np.int64),
            "precision": precision.astype(np.float64),
            "recall": recall.astype(np.float64),
            "f1": f1.astype(np.float64),
        }
    )


def _confusion_metric(
    counts: np.ndarray,
    true_indices: np.ndarray,
    predicted_indices: np.ndarray,
    *,
    class_count: int,
    metric: str,
) -> float:
    confusion = np.zeros((class_count, class_count), dtype=np.int64)
    np.add.at(confusion, (true_indices, predicted_indices), counts)
    true_support = confusion.sum(axis=1)
    predicted_support = confusion.sum(axis=0)
    true_positive = np.diag(confusion)
    if metric == "balanced_accuracy":
        return float(
            np.divide(
                true_positive,
                true_support,
                out=np.zeros(class_count),
                where=true_support > 0,
            ).mean()
        )
    if metric == "macro_f1":
        denominator = true_support + predicted_support
        return float(
            np.divide(
                2 * true_positive,
                denominator,
                out=np.zeros(class_count),
                where=denominator > 0,
            ).mean()
        )
    raise PipelineInvariantError(f"Unsupported paired paper metric: {metric}")


def paired_method_intervals(
    left: pd.DataFrame,
    right: pd.DataFrame,
    classes: tuple[str, ...],
    *,
    domain: str,
    metrics: Iterable[str] = PRIMARY_METRICS,
    replicates: int = 1_000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> list[dict[str, float | int | str]]:
    """Compute left-minus-right intervals with one multinomial pair bootstrap."""
    metric_names = tuple(metrics)
    if not metric_names or any(metric not in PRIMARY_METRICS for metric in metric_names):
        raise PipelineInvariantError("Paired method metrics differ from the paper freeze")
    if replicates != 1_000 or confidence_level != 0.95 or seed != 42:
        raise PipelineInvariantError("Paired method bootstrap policy differs from the freeze")
    left_domain = validated_domain_frame(left, classes, domain=domain)
    right_domain = validated_domain_frame(right, classes, domain=domain)
    identity = ("pair_id", "true_label")
    try:
        pd.testing.assert_frame_equal(
            left_domain.loc[:, identity],
            right_domain.loc[:, identity],
            check_dtype=False,
            check_exact=True,
        )
    except AssertionError as error:
        raise PipelineInvariantError("Paired method identities or labels differ") from error

    class_index = {label: index for index, label in enumerate(classes)}
    class_count = len(classes)
    true_indices = left_domain["true_label"].astype(str).map(class_index).to_numpy()
    left_indices = left_domain["prediction"].astype(str).map(class_index).to_numpy()
    right_indices = right_domain["prediction"].astype(str).map(class_index).to_numpy()
    if any(pd.isna(values).any() for values in (true_indices, left_indices, right_indices)):
        raise PipelineInvariantError("Paired method labels differ from the class order")
    true_indices = true_indices.astype(np.int64)
    left_indices = left_indices.astype(np.int64)
    right_indices = right_indices.astype(np.int64)

    state = true_indices * class_count**2 + left_indices * class_count + right_indices
    full_counts = np.bincount(state, minlength=class_count**3)
    observed_states = np.flatnonzero(full_counts)
    counts = full_counts[observed_states]
    state_true = observed_states // (class_count**2)
    state_remainder = observed_states % (class_count**2)
    state_left = state_remainder // class_count
    state_right = state_remainder % class_count
    pair_count = int(counts.sum())
    probabilities = counts / pair_count

    point: dict[str, tuple[float, float]] = {}
    samples = {metric: np.empty(replicates, dtype=np.float64) for metric in metric_names}
    for metric in metric_names:
        point[metric] = (
            _confusion_metric(
                counts,
                state_true,
                state_left,
                class_count=class_count,
                metric=metric,
            ),
            _confusion_metric(
                counts,
                state_true,
                state_right,
                class_count=class_count,
                metric=metric,
            ),
        )
    rng = np.random.default_rng(seed)
    for replicate in range(replicates):
        sampled_counts = rng.multinomial(pair_count, probabilities)
        for metric in metric_names:
            samples[metric][replicate] = _confusion_metric(
                sampled_counts,
                state_true,
                state_left,
                class_count=class_count,
                metric=metric,
            ) - _confusion_metric(
                sampled_counts,
                state_true,
                state_right,
                class_count=class_count,
                metric=metric,
            )
    alpha = (1.0 - confidence_level) / 2.0
    rows: list[dict[str, float | int | str]] = []
    for metric in metric_names:
        left_value, right_value = point[metric]
        rows.append(
            {
                "test_domain": domain,
                "metric": metric,
                "pair_count": pair_count,
                "replicates": replicates,
                "left_estimate": left_value,
                "right_estimate": right_value,
                "delta_estimate": left_value - right_value,
                "delta_ci_low": float(np.quantile(samples[metric], alpha)),
                "delta_ci_high": float(np.quantile(samples[metric], 1.0 - alpha)),
            }
        )
    return rows
