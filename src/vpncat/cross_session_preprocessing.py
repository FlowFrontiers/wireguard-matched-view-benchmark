from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from vpncat.cross_session_index import CrossSessionIndex
from vpncat.errors import PipelineInvariantError
from vpncat.preprocessing import StatisticalObservations, pair_id_digest


@dataclass(frozen=True)
class CrossSessionTargetState:
    train_session: int
    classes: tuple[str, ...]
    class_weights: np.ndarray
    fit_pair_count: int
    fit_pair_ids_sha256: str

    def encode_labels(self, labels) -> np.ndarray:
        mapping = {label: index for index, label in enumerate(self.classes)}
        encoded = np.asarray([mapping.get(str(label), -1) for label in labels], dtype=np.int64)
        if (encoded < 0).any():
            raise PipelineInvariantError("Cross-session labels are absent from source training")
        return encoded

    def sample_weights(self, labels) -> np.ndarray:
        return self.class_weights[self.encode_labels(labels)]

    def class_weight_dict(self) -> dict[int, float]:
        return {index: float(weight) for index, weight in enumerate(self.class_weights)}

    def to_dict(self) -> dict[str, Any]:
        return {
            "train_session": self.train_session,
            "classes": list(self.classes),
            "class_weights": self.class_weights.tolist(),
            "class_weighting": "balanced",
            "fit_pair_count": self.fit_pair_count,
            "fit_pair_ids_sha256": self.fit_pair_ids_sha256,
        }


@dataclass(frozen=True)
class CrossSessionPreprocessingState:
    train_session: int
    train_domain: str
    representation: str
    feature_names: tuple[str, ...]
    medians: np.ndarray
    classes: tuple[str, ...]
    class_weights: np.ndarray
    fit_pair_count: int
    fit_pair_ids_sha256: str

    def transform_features(self, observations: StatisticalObservations) -> np.ndarray:
        if observations.representation != self.representation:
            raise PipelineInvariantError("Cross-session representation mismatch")
        if observations.feature_names != self.feature_names:
            raise PipelineInvariantError("Cross-session feature schema mismatch")
        values = np.asarray(observations.values, dtype=np.float64).copy()
        if values.ndim != 2 or values.shape[1] != len(self.feature_names):
            raise PipelineInvariantError("Cross-session feature matrix shape is invalid")
        if len(values) != len(observations.pair_ids):
            raise PipelineInvariantError("Cross-session feature rows and pair IDs differ")
        if np.isinf(values).any():
            raise PipelineInvariantError("Cross-session features contain infinity")
        missing_rows, missing_columns = np.where(np.isnan(values))
        values[missing_rows, missing_columns] = self.medians[missing_columns]
        if not np.isfinite(values).all():
            raise PipelineInvariantError("Cross-session imputation is not finite")
        return values

    def target_state(self) -> CrossSessionTargetState:
        return CrossSessionTargetState(
            train_session=self.train_session,
            classes=self.classes,
            class_weights=self.class_weights,
            fit_pair_count=self.fit_pair_count,
            fit_pair_ids_sha256=self.fit_pair_ids_sha256,
        )

    def encode_labels(self, labels) -> np.ndarray:
        return self.target_state().encode_labels(labels)

    def sample_weights(self, labels) -> np.ndarray:
        return self.target_state().sample_weights(labels)

    def class_weight_dict(self) -> dict[int, float]:
        return self.target_state().class_weight_dict()

    def to_dict(self) -> dict[str, Any]:
        return {
            "train_session": self.train_session,
            "train_domain": self.train_domain,
            "representation": self.representation,
            "feature_names": list(self.feature_names),
            "imputation": "training-median",
            "scaling": "none",
            "medians": self.medians.tolist(),
            "classes": list(self.classes),
            "class_weights": self.class_weights.tolist(),
            "class_weighting": "balanced",
            "fit_pair_count": self.fit_pair_count,
            "fit_pair_ids_sha256": self.fit_pair_ids_sha256,
        }


def fit_cross_session_targets(index: CrossSessionIndex) -> CrossSessionTargetState:
    labels = np.asarray(index.labels, dtype=object)[index.train_positions].astype(str)
    classes, counts = np.unique(labels, return_counts=True)
    if len(classes) < 2 or (counts <= 0).any():
        raise PipelineInvariantError("Cross-session source labels cannot be balanced")
    weights = len(labels) / (len(classes) * counts.astype(np.float64))
    pair_ids = index.pair_ids_for("train")
    return CrossSessionTargetState(
        train_session=index.train_session,
        classes=tuple(classes.astype(str)),
        class_weights=weights,
        fit_pair_count=len(pair_ids),
        fit_pair_ids_sha256=pair_id_digest(pair_ids),
    )


def fit_cross_session_preprocessing(
    observations: StatisticalObservations,
    index: CrossSessionIndex,
) -> CrossSessionPreprocessingState:
    if observations.pair_ids != index.pair_ids:
        raise PipelineInvariantError("Cross-session observations are misaligned")
    values = np.asarray(observations.values, dtype=np.float64)
    if values.ndim != 2 or values.shape != (
        len(index.pair_ids),
        len(observations.feature_names),
    ):
        raise PipelineInvariantError("Cross-session fitting matrix shape is invalid")
    if np.isinf(values).any():
        raise PipelineInvariantError("Cross-session fitting features contain infinity")
    training = values[index.train_positions]
    medians = np.empty(training.shape[1], dtype=np.float64)
    for column in range(training.shape[1]):
        finite = training[np.isfinite(training[:, column]), column]
        if len(finite) == 0:
            raise PipelineInvariantError("Cross-session training feature is entirely missing")
        medians[column] = float(np.median(finite))
    targets = fit_cross_session_targets(index)
    return CrossSessionPreprocessingState(
        train_session=index.train_session,
        train_domain=observations.domain,
        representation=observations.representation,
        feature_names=observations.feature_names,
        medians=medians,
        classes=targets.classes,
        class_weights=targets.class_weights,
        fit_pair_count=targets.fit_pair_count,
        fit_pair_ids_sha256=targets.fit_pair_ids_sha256,
    )
