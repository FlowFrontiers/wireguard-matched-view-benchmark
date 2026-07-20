from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from vpncat.errors import PipelineInvariantError
from vpncat.experiment import DOMAINS, RunSpec
from vpncat.features import build_sequential_splt
from vpncat.folds import FoldIndex, materialize_fold_index
from vpncat.preprocessing import FoldTargetState, fit_fold_targets
from vpncat.schema import SEQUENCE_COLUMNS


@dataclass(frozen=True)
class NeuralSubset:
    pair_ids: tuple[str, ...]
    positions: np.ndarray
    values: np.ndarray
    mask: np.ndarray
    targets: np.ndarray


@dataclass(frozen=True)
class PreparedNeuralRun:
    fold: FoldIndex
    state: FoldTargetState
    channels: tuple[str, ...]
    training: NeuralSubset
    validation: NeuralSubset
    tests: dict[str, NeuralSubset]


@dataclass(frozen=True)
class PreparedNeuralDevelopment:
    fold: FoldIndex
    state: FoldTargetState
    channels: tuple[str, ...]
    training: NeuralSubset
    validation: NeuralSubset


def _subset(
    fold: FoldIndex,
    state: FoldTargetState,
    values: np.ndarray,
    mask: np.ndarray,
    positions: np.ndarray,
) -> NeuralSubset:
    positions = np.asarray(positions, dtype=np.int64)
    labels = np.asarray(fold.labels, dtype=object)[positions].astype(str)
    return NeuralSubset(
        pair_ids=tuple(fold.pair_ids[position] for position in positions),
        positions=positions,
        values=np.asarray(values[positions], dtype=np.float32),
        mask=np.asarray(mask[positions], dtype=bool),
        targets=state.encode_labels(labels),
    )


def _materialized_subset(
    fold: FoldIndex,
    state: FoldTargetState,
    values: np.ndarray,
    mask: np.ndarray,
    positions: np.ndarray,
) -> NeuralSubset:
    positions = np.asarray(positions, dtype=np.int64)
    if len(values) != len(positions) or len(mask) != len(positions):
        raise PipelineInvariantError("Materialized neural subset rows are misaligned")
    labels = np.asarray(fold.labels, dtype=object)[positions].astype(str)
    return NeuralSubset(
        pair_ids=tuple(fold.pair_ids[position] for position in positions),
        positions=positions,
        values=np.asarray(values, dtype=np.float32),
        mask=np.asarray(mask, dtype=bool),
        targets=state.encode_labels(labels),
    )


def prepare_neural_run(
    canonical_path: Path,
    split_path: Path,
    run: RunSpec,
    *,
    prefix_length: int,
    channels: tuple[str, ...],
) -> PreparedNeuralRun:
    """Materialize source-only train/validation tensors and paired held-out views."""
    if (
        run.family != "neural"
        or run.representation != "sequential_splt"
        or run.train_domain not in DOMAINS
    ):
        raise PipelineInvariantError("Neural preparation received an incompatible run")
    metadata_columns = ("pair_id", "session", "application_category")
    columns = (*metadata_columns, *SEQUENCE_COLUMNS)
    frame = pq.read_table(canonical_path, columns=list(columns)).to_pandas()
    if tuple(frame.columns) != columns:
        raise PipelineInvariantError("Canonical neural columns were not loaded in contract order")
    split = pd.read_csv(split_path)
    fold = materialize_fold_index(
        frame.loc[:, list(metadata_columns)],
        split,
        fold=run.fold,
    )
    if tuple(frame["pair_id"].astype(str)) != fold.pair_ids:
        raise PipelineInvariantError("Canonical neural rows and fold index are misaligned")

    matrices = {
        domain: build_sequential_splt(
            frame,
            domain=domain,
            prefix_length=prefix_length,
            channels=channels,
        )
        for domain in DOMAINS
    }
    for domain, matrix in matrices.items():
        expected_shape = (len(fold.pair_ids), prefix_length, len(channels))
        if matrix.values.shape != expected_shape or matrix.mask.shape != expected_shape[:2]:
            raise PipelineInvariantError(f"{domain} sequential tensor shape is invalid")
        if not np.isfinite(matrix.values).all():
            raise PipelineInvariantError(f"{domain} sequential tensor contains non-finite values")
        if not np.all(matrix.values[~matrix.mask] == 0):
            raise PipelineInvariantError(f"{domain} sequential padding is not zero")

    state = fit_fold_targets(fold)
    source = matrices[run.train_domain]
    training = _subset(
        fold,
        state,
        source.values,
        source.mask,
        fold.train_positions,
    )
    validation = _subset(
        fold,
        state,
        source.values,
        source.mask,
        fold.validation_positions,
    )
    if set(training.pair_ids) & set(validation.pair_ids):
        raise PipelineInvariantError("Neural training and validation pair IDs overlap")
    return PreparedNeuralRun(
        fold=fold,
        state=state,
        channels=channels,
        training=training,
        validation=validation,
        tests={
            domain: _subset(
                fold,
                state,
                matrices[domain].values,
                matrices[domain].mask,
                fold.test_positions,
            )
            for domain in run.test_domains
        },
    )


def prepare_neural_development(
    canonical_path: Path,
    split_path: Path,
    run: RunSpec,
    *,
    prefix_length: int,
    channels: tuple[str, ...],
) -> PreparedNeuralDevelopment:
    """Read only the source view needed for fold-1 tuning; never materialize test views."""
    if (
        run.family != "neural"
        or run.representation != "sequential_splt"
        or run.train_domain not in DOMAINS
    ):
        raise PipelineInvariantError("Neural development received an incompatible run")
    metadata_columns = ("pair_id", "session", "application_category")
    source_columns = tuple(
        f"{run.train_domain}_{channel}"
        for channel in ("direction", "size", "iat_ms")
    )
    columns = (*metadata_columns, *source_columns)
    frame = pq.read_table(canonical_path, columns=list(columns)).to_pandas()
    if tuple(frame.columns) != columns:
        raise PipelineInvariantError("Neural development columns violate source-only I/O")
    split = pd.read_csv(split_path)
    fold = materialize_fold_index(
        frame.loc[:, list(metadata_columns)],
        split,
        fold=run.fold,
    )
    if tuple(frame["pair_id"].astype(str)) != fold.pair_ids:
        raise PipelineInvariantError("Neural development rows and fold index are misaligned")

    development_positions = np.concatenate(
        [fold.train_positions, fold.validation_positions]
    )
    development_frame = frame.iloc[development_positions].reset_index(drop=True)
    matrix = build_sequential_splt(
        development_frame,
        domain=run.train_domain,
        prefix_length=prefix_length,
        channels=channels,
    )
    expected_shape = (len(development_positions), prefix_length, len(channels))
    if matrix.values.shape != expected_shape or matrix.mask.shape != expected_shape[:2]:
        raise PipelineInvariantError("Neural development tensor shape is invalid")
    if not np.isfinite(matrix.values).all() or not np.all(matrix.values[~matrix.mask] == 0):
        raise PipelineInvariantError("Neural development tensor values are invalid")

    state = fit_fold_targets(fold)
    training_count = len(fold.train_positions)
    training = _materialized_subset(
        fold,
        state,
        matrix.values[:training_count],
        matrix.mask[:training_count],
        fold.train_positions,
    )
    validation = _materialized_subset(
        fold,
        state,
        matrix.values[training_count:],
        matrix.mask[training_count:],
        fold.validation_positions,
    )
    if set(training.pair_ids) & set(validation.pair_ids):
        raise PipelineInvariantError("Neural development train/validation IDs overlap")
    return PreparedNeuralDevelopment(
        fold=fold,
        state=state,
        channels=channels,
        training=training,
        validation=validation,
    )
