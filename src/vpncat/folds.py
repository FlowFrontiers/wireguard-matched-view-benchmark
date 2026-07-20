from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from vpncat.errors import PipelineInvariantError

ROLES = ("train", "validation", "test")


@dataclass(frozen=True)
class FoldIndex:
    fold: int
    pair_ids: tuple[str, ...]
    sessions: np.ndarray
    labels: tuple[str, ...]
    roles: tuple[str, ...]
    train_positions: np.ndarray
    validation_positions: np.ndarray
    test_positions: np.ndarray

    def positions(self, role: str) -> np.ndarray:
        if role == "train":
            return self.train_positions
        if role == "validation":
            return self.validation_positions
        if role == "test":
            return self.test_positions
        raise PipelineInvariantError(f"Unsupported fold role: {role}")

    def pair_ids_for(self, role: str) -> tuple[str, ...]:
        return tuple(self.pair_ids[index] for index in self.positions(role))


def materialize_fold_index(
    pair_metadata: pd.DataFrame,
    split_manifest: pd.DataFrame,
    *,
    fold: int,
) -> FoldIndex:
    """Align one fixed fold to canonical row order without duplicating paired views."""
    if fold < 1:
        raise PipelineInvariantError("fold must be positive")
    metadata_columns = {"pair_id", "session", "application_category"}
    missing_metadata = metadata_columns - set(pair_metadata.columns)
    if missing_metadata:
        raise PipelineInvariantError(
            f"Pair metadata is missing columns: {sorted(missing_metadata)}"
        )
    role_column = f"role_fold_{fold}"
    missing_split = {"pair_id", role_column} - set(split_manifest.columns)
    if missing_split:
        raise PipelineInvariantError(
            f"Split manifest is missing columns: {sorted(missing_split)}"
        )
    if pair_metadata["pair_id"].duplicated().any():
        raise PipelineInvariantError("Canonical pair metadata contains duplicate pair_id values")
    if split_manifest["pair_id"].duplicated().any():
        raise PipelineInvariantError("Split manifest contains duplicate pair_id values")

    canonical_ids = set(pair_metadata["pair_id"].astype(str))
    split_ids = set(split_manifest["pair_id"].astype(str))
    if canonical_ids != split_ids:
        raise PipelineInvariantError(
            "Canonical and split pair coverage differs: "
            f"missing={len(canonical_ids - split_ids)}, unknown={len(split_ids - canonical_ids)}"
        )

    split_columns = ["pair_id", role_column]
    for column in ("session", "application_category"):
        if column in split_manifest:
            split_columns.append(column)
    aligned = pair_metadata.loc[:, ["pair_id", "session", "application_category"]].merge(
        split_manifest.loc[:, split_columns],
        on="pair_id",
        how="left",
        sort=False,
        validate="one_to_one",
        suffixes=("_canonical", "_split"),
    )
    for column in ("session", "application_category"):
        canonical_column = f"{column}_canonical"
        split_column = f"{column}_split"
        if canonical_column in aligned and not aligned[canonical_column].astype(str).equals(
            aligned[split_column].astype(str)
        ):
            raise PipelineInvariantError(
                f"Canonical and split {column} values disagree"
            )

    roles = aligned[role_column].astype(str)
    observed_roles = set(roles)
    if observed_roles != set(ROLES):
        raise PipelineInvariantError(
            f"Fold {fold} has roles {sorted(observed_roles)}, expected {sorted(ROLES)}"
        )
    positions = {
        role: np.flatnonzero(roles.eq(role).to_numpy()).astype(np.int64)
        for role in ROLES
    }
    combined = np.concatenate(list(positions.values()))
    if len(np.unique(combined)) != len(aligned) or len(combined) != len(aligned):
        raise PipelineInvariantError("Fold roles are not a disjoint partition of canonical pairs")

    session_column = "session_canonical" if "session_canonical" in aligned else "session"
    label_column = (
        "application_category_canonical"
        if "application_category_canonical" in aligned
        else "application_category"
    )
    return FoldIndex(
        fold=fold,
        pair_ids=tuple(aligned["pair_id"].astype(str)),
        sessions=aligned[session_column].to_numpy(dtype=np.int16),
        labels=tuple(aligned[label_column].astype(str)),
        roles=tuple(roles),
        train_positions=positions["train"],
        validation_positions=positions["validation"],
        test_positions=positions["test"],
    )
