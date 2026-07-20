from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from vpncat.errors import PipelineInvariantError
from vpncat.features import build_matched_flow_stats, build_prefix_stats
from vpncat.folds import FoldIndex

STATISTICAL_REPRESENTATIONS = frozenset({"matched_flow_stats", "prefix_stats"})


def pair_id_digest(pair_ids: tuple[str, ...]) -> str:
    """Hash an unordered pair-ID set using an unambiguous length-prefixed encoding."""
    digest = hashlib.sha256()
    for pair_id in sorted(pair_ids):
        encoded = pair_id.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="big"))
        digest.update(encoded)
    return digest.hexdigest()


@dataclass(frozen=True)
class StatisticalObservations:
    pair_ids: tuple[str, ...]
    domain: str
    representation: str
    feature_names: tuple[str, ...]
    values: np.ndarray


@dataclass(frozen=True)
class FoldTargetState:
    fold: int
    classes: tuple[str, ...]
    class_weights: np.ndarray
    fit_pair_count: int
    fit_pair_ids_sha256: str

    def encode_labels(self, labels: tuple[str, ...] | np.ndarray) -> np.ndarray:
        mapping = {label: index for index, label in enumerate(self.classes)}
        encoded = np.asarray([mapping.get(str(label), -1) for label in labels], dtype=np.int64)
        if (encoded < 0).any():
            unknown = sorted(
                {
                    str(label)
                    for label, value in zip(labels, encoded, strict=True)
                    if value < 0
                }
            )
            raise PipelineInvariantError(f"Labels were absent from training data: {unknown}")
        return encoded

    def sample_weights(self, labels: tuple[str, ...] | np.ndarray) -> np.ndarray:
        return self.class_weights[self.encode_labels(labels)]

    def class_weight_dict(self) -> dict[int, float]:
        return {index: float(weight) for index, weight in enumerate(self.class_weights)}

    def to_dict(self) -> dict[str, Any]:
        return {
            "fold": self.fold,
            "classes": list(self.classes),
            "class_weights": self.class_weights.tolist(),
            "class_weighting": "balanced",
            "fit_pair_count": self.fit_pair_count,
            "fit_pair_ids_sha256": self.fit_pair_ids_sha256,
        }


@dataclass(frozen=True)
class FoldPreprocessingState:
    fold: int
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
            raise PipelineInvariantError(
                "Preprocessor representation does not match observations"
            )
        if observations.feature_names != self.feature_names:
            raise PipelineInvariantError("Preprocessor feature schema does not match observations")
        values = np.asarray(observations.values, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(self.feature_names):
            raise PipelineInvariantError("Statistical feature matrix has an invalid shape")
        if np.isinf(values).any():
            raise PipelineInvariantError("Statistical feature matrix contains infinite values")
        transformed = values.copy()
        missing_rows, missing_columns = np.where(np.isnan(transformed))
        transformed[missing_rows, missing_columns] = self.medians[missing_columns]
        if not np.isfinite(transformed).all():
            raise PipelineInvariantError("Median imputation did not produce finite features")
        return transformed

    def encode_labels(self, labels: tuple[str, ...] | np.ndarray) -> np.ndarray:
        return self.target_state().encode_labels(labels)

    def sample_weights(self, labels: tuple[str, ...] | np.ndarray) -> np.ndarray:
        return self.target_state().sample_weights(labels)

    def class_weight_dict(self) -> dict[int, float]:
        return self.target_state().class_weight_dict()

    def target_state(self) -> FoldTargetState:
        return FoldTargetState(
            fold=self.fold,
            classes=self.classes,
            class_weights=self.class_weights,
            fit_pair_count=self.fit_pair_count,
            fit_pair_ids_sha256=self.fit_pair_ids_sha256,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "fold": self.fold,
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


def build_statistical_observations(
    frame: pd.DataFrame,
    *,
    domain: str,
    representation: str,
    prefix_length: int,
) -> StatisticalObservations:
    if "pair_id" not in frame:
        raise PipelineInvariantError("Canonical feature frame has no pair_id column")
    pair_ids = tuple(frame["pair_id"].astype(str))
    if len(set(pair_ids)) != len(pair_ids):
        raise PipelineInvariantError("Statistical observations contain duplicate pair IDs")
    if representation == "matched_flow_stats":
        matrix = build_matched_flow_stats(frame, domain=domain)
    elif representation == "prefix_stats":
        matrix = build_prefix_stats(frame, domain=domain, prefix_length=prefix_length)
    else:
        raise PipelineInvariantError(
            f"Unsupported fitted statistical representation: {representation}"
        )
    return StatisticalObservations(
        pair_ids=pair_ids,
        domain=domain,
        representation=representation,
        feature_names=matrix.feature_names,
        values=matrix.values,
    )


def fit_fold_preprocessing(
    observations: StatisticalObservations,
    fold_index: FoldIndex,
) -> FoldPreprocessingState:
    """Fit all data-dependent preprocessing from the fold's training pairs only."""
    if observations.pair_ids != fold_index.pair_ids:
        raise PipelineInvariantError(
            "Observation rows are not aligned to the canonical fold index"
        )
    if observations.representation not in STATISTICAL_REPRESENTATIONS:
        raise PipelineInvariantError(
            f"Unsupported fitted representation: {observations.representation}"
        )
    values = np.asarray(observations.values, dtype=np.float64)
    if values.ndim != 2 or values.shape != (
        len(fold_index.pair_ids),
        len(observations.feature_names),
    ):
        raise PipelineInvariantError("Observation matrix shape disagrees with fold metadata")
    if np.isinf(values).any():
        raise PipelineInvariantError("Training observations contain infinite values")

    training_positions = fold_index.train_positions
    training_values = values[training_positions]
    medians = np.empty(training_values.shape[1], dtype=np.float64)
    for column in range(training_values.shape[1]):
        finite = training_values[np.isfinite(training_values[:, column]), column]
        if len(finite) == 0:
            raise PipelineInvariantError(
                f"Training feature is entirely missing: {observations.feature_names[column]}"
            )
        medians[column] = float(np.median(finite))

    targets = fit_fold_targets(fold_index)
    return FoldPreprocessingState(
        fold=fold_index.fold,
        train_domain=observations.domain,
        representation=observations.representation,
        feature_names=observations.feature_names,
        medians=medians,
        classes=targets.classes,
        class_weights=targets.class_weights,
        fit_pair_count=targets.fit_pair_count,
        fit_pair_ids_sha256=targets.fit_pair_ids_sha256,
    )


def fit_fold_targets(fold_index: FoldIndex) -> FoldTargetState:
    """Fit label encoding and balanced weights from training pairs only."""
    training_labels = np.asarray(fold_index.labels, dtype=object)[
        fold_index.train_positions
    ].astype(str)
    classes, counts = np.unique(training_labels, return_counts=True)
    if len(classes) < 2 or (counts <= 0).any():
        raise PipelineInvariantError("Training labels cannot support balanced class weighting")
    class_weights = len(training_labels) / (len(classes) * counts.astype(np.float64))
    training_pair_ids = fold_index.pair_ids_for("train")
    return FoldTargetState(
        fold=fold_index.fold,
        classes=tuple(classes.astype(str)),
        class_weights=class_weights,
        fit_pair_count=len(training_pair_ids),
        fit_pair_ids_sha256=pair_id_digest(training_pair_ids),
    )
