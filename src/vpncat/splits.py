from __future__ import annotations

import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from vpncat.errors import PipelineInvariantError

ROLE_VALUES = frozenset({"train", "validation", "test"})


def build_split_manifest(
    pair_index: pd.DataFrame,
    *,
    folds: int,
    validation_fraction: float,
    random_seed: int,
) -> pd.DataFrame:
    """Assign deterministic pair-disjoint train/validation/test roles for every fold."""
    required = {"pair_id", "session", "application_category"}
    missing = required - set(pair_index.columns)
    if missing:
        raise PipelineInvariantError(f"Pair index is missing columns: {sorted(missing)}")
    if pair_index["pair_id"].duplicated().any():
        raise PipelineInvariantError("pair_id values must be unique before splitting")

    manifest = pair_index.loc[:, ["pair_id", "session", "application_category"]].copy()
    strata = manifest["session"].astype(str) + "::" + manifest["application_category"]
    support = strata.value_counts()
    if int(support.min()) < folds:
        too_small = support[support < folds].to_dict()
        raise PipelineInvariantError(
            f"Class-session strata cannot support {folds} folds: {too_small}"
        )

    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_seed)
    for fold, (training_positions, test_positions) in enumerate(
        splitter.split(manifest, strata), start=1
    ):
        role_column = f"role_fold_{fold}"
        manifest[role_column] = "train"
        manifest.loc[manifest.index[test_positions], role_column] = "test"

        training_strata = strata.iloc[training_positions]
        validation_splitter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=validation_fraction,
            random_state=random_seed + fold,
        )
        _, validation_relative = next(
            validation_splitter.split(training_positions, training_strata)
        )
        validation_positions = training_positions[validation_relative]
        manifest.loc[manifest.index[validation_positions], role_column] = "validation"

    validate_split_manifest(manifest, folds=folds)
    return manifest


def validate_split_manifest(manifest: pd.DataFrame, *, folds: int) -> None:
    if manifest["pair_id"].duplicated().any():
        raise PipelineInvariantError("Split manifest contains duplicate pair_id values")

    test_appearances = pd.Series(0, index=manifest.index, dtype="int16")
    for fold in range(1, folds + 1):
        column = f"role_fold_{fold}"
        if column not in manifest:
            raise PipelineInvariantError(f"Split manifest is missing {column}")
        observed = set(manifest[column].unique())
        if observed != ROLE_VALUES:
            raise PipelineInvariantError(
                f"{column} has roles {sorted(observed)}, expected {sorted(ROLE_VALUES)}"
            )
        test_appearances += manifest[column].eq("test").astype("int16")

    if not test_appearances.eq(1).all():
        raise PipelineInvariantError("Every pair must be a test pair in exactly one fold")
