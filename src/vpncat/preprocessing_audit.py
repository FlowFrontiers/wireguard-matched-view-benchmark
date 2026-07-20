from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from vpncat import __version__
from vpncat.config import PreprocessingConfig
from vpncat.errors import PipelineInvariantError
from vpncat.folds import FoldIndex, materialize_fold_index
from vpncat.hashing import sha256_file
from vpncat.preprocessing import (
    StatisticalObservations,
    build_statistical_observations,
    fit_fold_preprocessing,
    fit_fold_targets,
    pair_id_digest,
)
from vpncat.provenance import git_provenance
from vpncat.schema import CANONICAL_COLUMNS
from vpncat.splits import validate_split_manifest


def _verify_source_hashes(config: PreprocessingConfig) -> tuple[str, str]:
    canonical_hash = sha256_file(config.canonical_path)
    split_hash = sha256_file(config.split_path)
    with config.dataset_manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    expected_canonical = manifest.get("artifacts", {}).get("canonical_pairs", {}).get("sha256")
    expected_split = manifest.get("artifacts", {}).get("split_manifest", {}).get("sha256")
    if canonical_hash != expected_canonical:
        raise PipelineInvariantError("Canonical hash disagrees with dataset manifest")
    if split_hash != expected_split:
        raise PipelineInvariantError("Split hash disagrees with dataset manifest")
    return canonical_hash, split_hash


def _subset(
    observations: StatisticalObservations,
    positions: np.ndarray,
) -> StatisticalObservations:
    return StatisticalObservations(
        pair_ids=tuple(observations.pair_ids[index] for index in positions),
        domain=observations.domain,
        representation=observations.representation,
        feature_names=observations.feature_names,
        values=observations.values[positions],
    )


def _fold_summary(fold: FoldIndex) -> dict[str, Any]:
    role_ids = {role: fold.pair_ids_for(role) for role in ("train", "validation", "test")}
    if set(role_ids["train"]) & (set(role_ids["validation"]) | set(role_ids["test"])):
        raise PipelineInvariantError("Training pair IDs overlap validation or test roles")
    if set(role_ids["validation"]) & set(role_ids["test"]):
        raise PipelineInvariantError("Validation and test pair IDs overlap")
    targets = fit_fold_targets(fold)
    encoded = targets.encode_labels(fold.labels)
    if len(np.unique(encoded)) != len(targets.classes):
        raise PipelineInvariantError("Encoded labels do not cover the training vocabulary")
    training_labels = np.asarray(fold.labels, dtype=object)[fold.train_positions]
    training_weights = targets.sample_weights(training_labels)
    class_weight_totals = np.bincount(
        targets.encode_labels(training_labels),
        weights=training_weights,
        minlength=len(targets.classes),
    )
    if not np.allclose(class_weight_totals, class_weight_totals[0]):
        raise PipelineInvariantError("Balanced training weights do not equalize class totals")
    return {
        "role_counts": {role: len(pair_ids) for role, pair_ids in role_ids.items()},
        "role_pair_id_sha256": {
            role: pair_id_digest(pair_ids) for role, pair_ids in role_ids.items()
        },
        "pair_overlap": {
            "train_validation": 0,
            "train_test": 0,
            "validation_test": 0,
        },
        "targets": targets.to_dict(),
    }


def audit_preprocessing(config: PreprocessingConfig) -> dict[str, Any]:
    """Audit fold roles and every data-fitted preprocessing state over full data."""
    canonical_hash, split_hash = _verify_source_hashes(config)
    canonical = pq.read_table(
        config.canonical_path,
        columns=list(CANONICAL_COLUMNS),
    ).to_pandas()
    split = pd.read_csv(config.split_path)
    validate_split_manifest(split, folds=config.folds)
    pair_metadata = canonical.loc[
        :, ["pair_id", "session", "application_category"]
    ]
    folds = {
        fold: materialize_fold_index(pair_metadata, split, fold=fold)
        for fold in range(1, config.folds + 1)
    }
    fold_summaries = {str(fold): _fold_summary(index) for fold, index in folds.items()}

    fitted_states: dict[str, Any] = {}
    for representation in config.statistical_representations:
        observations = {
            domain: build_statistical_observations(
                canonical,
                domain=domain,
                representation=representation,
                prefix_length=config.prefix_length,
            )
            for domain in config.domains
        }
        representation_states: dict[str, Any] = {}
        for fold_number, fold in folds.items():
            fold_states: dict[str, Any] = {}
            expected_fit_hash = pair_id_digest(fold.pair_ids_for("train"))
            for train_domain in config.domains:
                state = fit_fold_preprocessing(observations[train_domain], fold)
                if state.fit_pair_ids_sha256 != expected_fit_hash:
                    raise PipelineInvariantError(
                        "Fitted-state pair hash is not the training-role hash"
                    )
                if state.fit_pair_count != len(fold.train_positions):
                    raise PipelineInvariantError(
                        "Fitted-state pair count is not the training-role count"
                    )
                transformed: dict[str, dict[str, int]] = {}
                for test_domain in config.domains:
                    role_counts: dict[str, int] = {}
                    for role in ("train", "validation", "test"):
                        role_observations = _subset(
                            observations[test_domain], fold.positions(role)
                        )
                        values = state.transform_features(role_observations)
                        if not np.isfinite(values).all():
                            raise PipelineInvariantError(
                                "Transformed statistical features are not finite"
                            )
                        role_counts[role] = len(values)
                    transformed[test_domain] = role_counts
                state_payload = state.to_dict()
                state_payload["fit_role"] = "train"
                state_payload["forbidden_fit_pair_overlap"] = {
                    "validation": 0,
                    "test": 0,
                }
                state_payload["transformed_rows"] = transformed
                fold_states[train_domain] = state_payload
            representation_states[str(fold_number)] = fold_states
        fitted_states[representation] = representation_states

    payload = {
        "audit_version": 1,
        "created_utc": datetime.now(UTC).isoformat(),
        "package_version": __version__,
        "git": git_provenance(config.project_root),
        "inputs": {
            "canonical_path": str(config.canonical_path),
            "canonical_sha256": canonical_hash,
            "split_path": str(config.split_path),
            "split_sha256": split_hash,
            "rows": len(canonical),
        },
        "configuration": {
            "folds": config.folds,
            "prefix_length": config.prefix_length,
            "statistical_representations": list(config.statistical_representations),
            "domains": list(config.domains),
            "imputation": "training-median",
            "scaling": "none",
            "class_weighting": "balanced",
            "fit_free_representations": ["flattened_splt", "sequential_splt"],
            "fit_free_transforms": ["direction-remap", "log1p-size", "log1p-iat"],
        },
        "folds": fold_summaries,
        "fitted_states": fitted_states,
        "status": "valid",
    }
    config.audit_output.parent.mkdir(parents=True, exist_ok=True)
    temporary = config.audit_output.with_suffix(config.audit_output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(config.audit_output)
    return payload
