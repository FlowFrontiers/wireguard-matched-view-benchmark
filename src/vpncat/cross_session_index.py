from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from vpncat.errors import PipelineInvariantError

ROLES = ("train", "validation", "test")


@dataclass(frozen=True)
class CrossSessionIndex:
    train_session: int
    test_session: int
    pair_ids: tuple[str, ...]
    sessions: np.ndarray
    labels: tuple[str, ...]
    roles: tuple[str, ...]
    train_positions: np.ndarray
    validation_positions: np.ndarray
    test_positions: np.ndarray

    def positions(self, role: str) -> np.ndarray:
        positions = {
            "train": self.train_positions,
            "validation": self.validation_positions,
            "test": self.test_positions,
        }
        if role not in positions:
            raise PipelineInvariantError(f"Unsupported cross-session role: {role}")
        return positions[role]

    def pair_ids_for(self, role: str) -> tuple[str, ...]:
        return tuple(self.pair_ids[index] for index in self.positions(role))


def materialize_cross_session_index(
    pair_metadata: pd.DataFrame,
    split_manifest: pd.DataFrame,
    *,
    train_session: int,
) -> CrossSessionIndex:
    metadata_columns = ("pair_id", "session", "application_category")
    if not set(metadata_columns) <= set(pair_metadata):
        raise PipelineInvariantError("Cross-session pair metadata is incomplete")
    role_column = f"role_train_session_{train_session}"
    if not {"pair_id", "session", "application_category", role_column} <= set(
        split_manifest
    ):
        raise PipelineInvariantError("Cross-session split columns are incomplete")
    if (
        pair_metadata["pair_id"].duplicated().any()
        or split_manifest["pair_id"].duplicated().any()
    ):
        raise PipelineInvariantError("Cross-session pair IDs are duplicated")
    canonical_ids = set(pair_metadata["pair_id"].astype(str))
    split_ids = set(split_manifest["pair_id"].astype(str))
    if canonical_ids != split_ids:
        raise PipelineInvariantError("Cross-session canonical and split coverage differs")
    aligned = pair_metadata.loc[:, metadata_columns].merge(
        split_manifest.loc[:, [*metadata_columns, role_column]],
        on="pair_id",
        how="left",
        sort=False,
        validate="one_to_one",
        suffixes=("_canonical", "_split"),
    )
    for column in ("session", "application_category"):
        if not aligned[f"{column}_canonical"].astype(str).equals(
            aligned[f"{column}_split"].astype(str)
        ):
            raise PipelineInvariantError(f"Cross-session {column} metadata disagrees")
    roles = aligned[role_column].astype(str)
    if set(roles) != set(ROLES):
        raise PipelineInvariantError("Cross-session roles are incomplete")
    sessions = aligned["session_canonical"].to_numpy(dtype=np.int16)
    observed_sessions = sorted(np.unique(sessions).tolist())
    if train_session not in observed_sessions or len(observed_sessions) != 2:
        raise PipelineInvariantError("Cross-session direction is invalid")
    test_session = next(value for value in observed_sessions if value != train_session)
    positions = {
        role: np.flatnonzero(roles.eq(role).to_numpy()).astype(np.int64)
        for role in ROLES
    }
    partition = np.concatenate([positions[role] for role in ROLES])
    if len(partition) != len(aligned) or not np.array_equal(
        np.sort(partition), np.arange(len(aligned))
    ):
        raise PipelineInvariantError("Cross-session roles do not partition canonical rows")
    if not np.isin(sessions[positions["train"]], [train_session]).all():
        raise PipelineInvariantError("Cross-session training rows leave source session")
    if not np.isin(sessions[positions["validation"]], [train_session]).all():
        raise PipelineInvariantError("Cross-session validation rows leave source session")
    if not np.isin(sessions[positions["test"]], [test_session]).all():
        raise PipelineInvariantError("Cross-session test rows leave target session")
    return CrossSessionIndex(
        train_session=train_session,
        test_session=test_session,
        pair_ids=tuple(aligned["pair_id"].astype(str)),
        sessions=sessions,
        labels=tuple(aligned["application_category_canonical"].astype(str)),
        roles=tuple(roles),
        train_positions=positions["train"],
        validation_positions=positions["validation"],
        test_positions=positions["test"],
    )
