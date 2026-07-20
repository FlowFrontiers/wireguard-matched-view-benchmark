from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from vpncat.cross_session import CrossSessionConfig, CrossSessionRunSpec
from vpncat.cross_session_index import CrossSessionIndex, materialize_cross_session_index
from vpncat.cross_session_preprocessing import (
    CrossSessionPreprocessingState,
    CrossSessionTargetState,
    fit_cross_session_preprocessing,
    fit_cross_session_targets,
)
from vpncat.errors import PipelineInvariantError
from vpncat.experiment import DOMAINS
from vpncat.features import build_flattened_splt, build_sequential_splt
from vpncat.neural_data import NeuralSubset
from vpncat.preprocessing import build_statistical_observations
from vpncat.primary_data import load_primary_frame
from vpncat.schema import SEQUENCE_COLUMNS

CrossState = CrossSessionPreprocessingState | CrossSessionTargetState


@dataclass(frozen=True)
class PreparedCrossSessionClassical:
    index: CrossSessionIndex
    state: CrossState
    feature_names: tuple[str, ...]
    training_values: np.ndarray
    training_targets: np.ndarray
    training_labels: np.ndarray
    validation_values: np.ndarray
    validation_targets: np.ndarray
    test_values: dict[str, np.ndarray]


@dataclass(frozen=True)
class PreparedCrossSessionNeural:
    index: CrossSessionIndex
    state: CrossSessionTargetState
    channels: tuple[str, ...]
    training: NeuralSubset
    validation: NeuralSubset
    tests: dict[str, NeuralSubset]


def _index(
    frame: pd.DataFrame,
    config: CrossSessionConfig,
    run: CrossSessionRunSpec,
) -> CrossSessionIndex:
    split = pd.read_csv(config.split_path)
    index = materialize_cross_session_index(
        frame.loc[:, ["pair_id", "session", "application_category"]],
        split,
        train_session=run.train_session,
    )
    if index.test_session != run.test_session:
        raise PipelineInvariantError("Cross-session run direction disagrees with split")
    if tuple(frame["pair_id"].astype(str)) != index.pair_ids:
        raise PipelineInvariantError("Cross-session canonical rows are misaligned")
    return index


def prepare_cross_session_classical(
    config: CrossSessionConfig,
    run: CrossSessionRunSpec,
) -> PreparedCrossSessionClassical:
    if run.family != "classical" or run.train_domain != "inner":
        raise PipelineInvariantError("Cross-session classical run is incompatible")
    frame = load_primary_frame(
        config.primary.canonical_path,
        representation=run.representation,
    )
    index = _index(frame, config, run)
    if run.representation in {"matched_flow_stats", "prefix_stats"}:
        observations = {
            domain: build_statistical_observations(
                frame,
                domain=domain,
                representation=run.representation,
                prefix_length=config.prefix_length,
            )
            for domain in DOMAINS
        }
        state: CrossState = fit_cross_session_preprocessing(
            observations[run.train_domain], index
        )
        feature_names = state.feature_names
        values = {
            domain: state.transform_features(observations[domain])
            for domain in DOMAINS
        }
    elif run.representation == "flattened_splt":
        matrices = {
            domain: build_flattened_splt(
                frame,
                domain=domain,
                prefix_length=config.prefix_length,
            )
            for domain in DOMAINS
        }
        state = fit_cross_session_targets(index)
        feature_names = matrices[run.train_domain].feature_names
        values = {domain: matrices[domain].values for domain in DOMAINS}
    else:
        raise PipelineInvariantError("Unsupported cross-session classical representation")
    for domain, matrix in values.items():
        if matrix.shape != (len(index.pair_ids), len(feature_names)):
            raise PipelineInvariantError(f"Cross-session {domain} matrix shape is invalid")
        if not np.isfinite(matrix).all():
            raise PipelineInvariantError(f"Cross-session {domain} matrix is not finite")
    labels = np.asarray(index.labels, dtype=object).astype(str)
    train_labels = labels[index.train_positions]
    validation_labels = labels[index.validation_positions]
    source = values[run.train_domain]
    return PreparedCrossSessionClassical(
        index=index,
        state=state,
        feature_names=feature_names,
        training_values=np.asarray(source[index.train_positions]),
        training_targets=state.encode_labels(train_labels),
        training_labels=train_labels,
        validation_values=np.asarray(source[index.validation_positions]),
        validation_targets=state.encode_labels(validation_labels),
        test_values={
            domain: np.asarray(values[domain][index.test_positions])
            for domain in run.test_domains
        },
    )


def _neural_subset(
    index: CrossSessionIndex,
    state: CrossSessionTargetState,
    values: np.ndarray,
    mask: np.ndarray,
    positions: np.ndarray,
) -> NeuralSubset:
    labels = np.asarray(index.labels, dtype=object)[positions].astype(str)
    return NeuralSubset(
        pair_ids=tuple(index.pair_ids[position] for position in positions),
        positions=np.asarray(positions, dtype=np.int64),
        values=np.asarray(values[positions], dtype=np.float32),
        mask=np.asarray(mask[positions], dtype=bool),
        targets=state.encode_labels(labels),
    )


def prepare_cross_session_neural(
    config: CrossSessionConfig,
    run: CrossSessionRunSpec,
    *,
    channels: tuple[str, ...],
) -> PreparedCrossSessionNeural:
    if (
        run.family != "neural"
        or run.representation != "sequential_splt"
        or run.train_domain != "inner"
    ):
        raise PipelineInvariantError("Cross-session neural run is incompatible")
    columns = ("pair_id", "session", "application_category", *SEQUENCE_COLUMNS)
    frame = pq.read_table(
        config.primary.canonical_path,
        columns=list(columns),
    ).to_pandas()
    index = _index(frame, config, run)
    matrices = {
        domain: build_sequential_splt(
            frame,
            domain=domain,
            prefix_length=config.prefix_length,
            channels=channels,
        )
        for domain in DOMAINS
    }
    state = fit_cross_session_targets(index)
    source = matrices[run.train_domain]
    training = _neural_subset(
        index, state, source.values, source.mask, index.train_positions
    )
    validation = _neural_subset(
        index, state, source.values, source.mask, index.validation_positions
    )
    if set(training.pair_ids) & set(validation.pair_ids):
        raise PipelineInvariantError("Cross-session neural train/validation overlap")
    tests = {
        domain: _neural_subset(
            index,
            state,
            matrices[domain].values,
            matrices[domain].mask,
            index.test_positions,
        )
        for domain in run.test_domains
    }
    if tests["inner"].pair_ids != tests["outer"].pair_ids:
        raise PipelineInvariantError("Cross-session paired test identities differ")
    return PreparedCrossSessionNeural(
        index=index,
        state=state,
        channels=channels,
        training=training,
        validation=validation,
        tests=tests,
    )
