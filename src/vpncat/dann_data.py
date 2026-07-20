from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from vpncat.dann import DANNConfig, DANNRunSpec
from vpncat.errors import PipelineInvariantError
from vpncat.features import build_sequential_splt
from vpncat.folds import FoldIndex, materialize_fold_index
from vpncat.neural_data import NeuralSubset
from vpncat.preprocessing import FoldTargetState, fit_fold_targets
from vpncat.schema import SEQUENCE_COLUMNS


@dataclass(frozen=True)
class UnlabeledNeuralSubset:
    pair_ids: tuple[str, ...]
    positions: np.ndarray
    values: np.ndarray
    mask: np.ndarray


@dataclass(frozen=True)
class PreparedDANNRun:
    fold: FoldIndex
    state: FoldTargetState
    channels: tuple[str, ...]
    source_training: NeuralSubset
    adaptation_training: UnlabeledNeuralSubset
    source_validation: NeuralSubset
    tests: dict[str, NeuralSubset]


def _labeled_subset(
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


def _unlabeled_subset(
    fold: FoldIndex,
    values: np.ndarray,
    mask: np.ndarray,
    positions: np.ndarray,
) -> UnlabeledNeuralSubset:
    positions = np.asarray(positions, dtype=np.int64)
    return UnlabeledNeuralSubset(
        pair_ids=tuple(fold.pair_ids[position] for position in positions),
        positions=positions,
        values=np.asarray(values[positions], dtype=np.float32),
        mask=np.asarray(mask[positions], dtype=bool),
    )


def prepare_dann_run(config: DANNConfig, run: DANNRunSpec) -> PreparedDANNRun:
    if (
        run.protocol != "dann"
        or run.model != config.model
        or run.backbone != config.backbone
        or run.source_domain != config.source_domain
        or run.adaptation_domain != config.adaptation_domain
    ):
        raise PipelineInvariantError("DANN run is incompatible with data contract")
    metadata_columns = ("pair_id", "session", "application_category")
    columns = (*metadata_columns, *SEQUENCE_COLUMNS)
    frame = pq.read_table(
        config.primary.canonical_path,
        columns=list(columns),
    ).to_pandas()
    if tuple(frame.columns) != columns:
        raise PipelineInvariantError("DANN canonical columns are out of contract order")
    split = pd.read_csv(config.primary.split_path)
    fold = materialize_fold_index(
        frame.loc[:, list(metadata_columns)],
        split,
        fold=run.fold,
    )
    if tuple(frame["pair_id"].astype(str)) != fold.pair_ids:
        raise PipelineInvariantError("DANN canonical rows and fold are misaligned")
    matrices = {
        domain: build_sequential_splt(
            frame,
            domain=domain,
            prefix_length=config.prefix_length,
            channels=config.channels,
        )
        for domain in run.test_domains
    }
    expected_shape = (
        len(fold.pair_ids),
        config.prefix_length,
        len(config.channels),
    )
    for domain, matrix in matrices.items():
        if matrix.values.shape != expected_shape or matrix.mask.shape != expected_shape[:2]:
            raise PipelineInvariantError(f"DANN {domain} tensor shape is invalid")
        if not np.isfinite(matrix.values).all() or not np.all(
            matrix.values[~matrix.mask] == 0
        ):
            raise PipelineInvariantError(f"DANN {domain} tensor values are invalid")

    state = fit_fold_targets(fold)
    source = matrices[run.source_domain]
    adaptation = matrices[run.adaptation_domain]
    source_training = _labeled_subset(
        fold,
        state,
        source.values,
        source.mask,
        fold.train_positions,
    )
    adaptation_training = _unlabeled_subset(
        fold,
        adaptation.values,
        adaptation.mask,
        fold.train_positions,
    )
    source_validation = _labeled_subset(
        fold,
        state,
        source.values,
        source.mask,
        fold.validation_positions,
    )
    if source_training.pair_ids != adaptation_training.pair_ids:
        raise PipelineInvariantError("DANN source and adaptation pairs are not identical")
    if not np.array_equal(
        source_training.positions, adaptation_training.positions
    ):
        raise PipelineInvariantError("DANN source and adaptation positions differ")
    forbidden = set(source_validation.pair_ids) | set(fold.pair_ids_for("test"))
    if set(adaptation_training.pair_ids) & forbidden:
        raise PipelineInvariantError("DANN adaptation data contains validation or test pairs")
    tests = {
        domain: _labeled_subset(
            fold,
            state,
            matrices[domain].values,
            matrices[domain].mask,
            fold.test_positions,
        )
        for domain in run.test_domains
    }
    if tests["inner"].pair_ids != tests["outer"].pair_ids:
        raise PipelineInvariantError("DANN paired test identities differ")
    return PreparedDANNRun(
        fold=fold,
        state=state,
        channels=config.channels,
        source_training=source_training,
        adaptation_training=adaptation_training,
        source_validation=source_validation,
        tests=tests,
    )
