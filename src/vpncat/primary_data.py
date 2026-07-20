from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from vpncat.errors import PipelineInvariantError
from vpncat.experiment import DOMAINS, RunSpec
from vpncat.features import build_flattened_splt
from vpncat.folds import FoldIndex, materialize_fold_index
from vpncat.preprocessing import (
    FoldPreprocessingState,
    FoldTargetState,
    build_statistical_observations,
    fit_fold_preprocessing,
    fit_fold_targets,
)
from vpncat.schema import SEQUENCE_COLUMNS, STAT_COLUMNS

TargetState = FoldPreprocessingState | FoldTargetState


@dataclass(frozen=True)
class PreparedClassicalRun:
    fold: FoldIndex
    state: TargetState
    feature_names: tuple[str, ...]
    training_values: np.ndarray
    training_targets: np.ndarray
    training_labels: np.ndarray
    test_values: dict[str, np.ndarray]


def _required_columns(representation: str) -> tuple[str, ...]:
    metadata = ("pair_id", "session", "application_category")
    if representation == "matched_flow_stats":
        return (*metadata, *STAT_COLUMNS)
    if representation in {"prefix_stats", "flattened_splt"}:
        return (*metadata, *SEQUENCE_COLUMNS)
    raise PipelineInvariantError(
        f"Classical runner does not support representation: {representation}"
    )


def load_primary_frame(canonical_path: Path, *, representation: str) -> pd.DataFrame:
    columns = _required_columns(representation)
    frame = pq.read_table(canonical_path, columns=list(columns)).to_pandas()
    if tuple(frame.columns) != columns:
        raise PipelineInvariantError("Canonical feature columns were not loaded in contract order")
    return frame


def _build_feature_views(
    frame: pd.DataFrame,
    run: RunSpec,
    *,
    prefix_length: int,
    fold: FoldIndex,
) -> tuple[TargetState, tuple[str, ...], dict[str, np.ndarray]]:
    if run.representation in {"matched_flow_stats", "prefix_stats"}:
        observations = {
            domain: build_statistical_observations(
                frame,
                domain=domain,
                representation=run.representation,
                prefix_length=prefix_length,
            )
            for domain in DOMAINS
        }
        state = fit_fold_preprocessing(observations[run.train_domain], fold)
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
                prefix_length=prefix_length,
            )
            for domain in DOMAINS
        }
        state = fit_fold_targets(fold)
        feature_names = matrices[run.train_domain].feature_names
        values = {domain: matrices[domain].values for domain in DOMAINS}
    else:
        raise PipelineInvariantError(
            f"Classical runner does not support representation: {run.representation}"
        )

    for domain in DOMAINS:
        matrix = np.asarray(values[domain])
        if matrix.shape != (len(fold.pair_ids), len(feature_names)):
            raise PipelineInvariantError(f"{domain} feature matrix shape is invalid")
        if not np.isfinite(matrix).all():
            raise PipelineInvariantError(f"{domain} feature matrix contains non-finite values")
        values[domain] = matrix
    return state, feature_names, values


def prepare_classical_run(
    canonical_path: Path,
    split_path: Path,
    run: RunSpec,
    *,
    prefix_length: int,
) -> PreparedClassicalRun:
    """Materialize one run while making training-only row selection unavoidable."""
    if run.family != "classical" or run.train_domain not in DOMAINS:
        raise PipelineInvariantError("Classical preparation received an incompatible run")
    frame = load_primary_frame(canonical_path, representation=run.representation)
    split = pd.read_csv(split_path)
    fold = materialize_fold_index(
        frame.loc[:, ["pair_id", "session", "application_category"]],
        split,
        fold=run.fold,
    )
    if tuple(frame["pair_id"].astype(str)) != fold.pair_ids:
        raise PipelineInvariantError("Canonical feature rows and fold index are misaligned")

    state, feature_names, values = _build_feature_views(
        frame,
        run,
        prefix_length=prefix_length,
        fold=fold,
    )
    training_positions = fold.train_positions
    training_labels = np.asarray(fold.labels, dtype=object)[training_positions].astype(str)
    training_values = np.asarray(values[run.train_domain][training_positions])
    training_targets = state.encode_labels(training_labels)
    if len(training_values) != state.fit_pair_count:
        raise PipelineInvariantError("Training matrix includes non-training fold rows")

    return PreparedClassicalRun(
        fold=fold,
        state=state,
        feature_names=feature_names,
        training_values=training_values,
        training_targets=training_targets,
        training_labels=training_labels,
        test_values={
            domain: np.asarray(values[domain][fold.test_positions])
            for domain in run.test_domains
        },
    )
