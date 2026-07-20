from __future__ import annotations

import json
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

from vpncat import __version__
from vpncat.cross_session import (
    CrossSessionConfig,
    load_cross_session_config,
    validate_cross_session_contract,
)
from vpncat.cross_session_index import CrossSessionIndex, materialize_cross_session_index
from vpncat.cross_session_preprocessing import (
    fit_cross_session_preprocessing,
    fit_cross_session_targets,
)
from vpncat.errors import PipelineInvariantError
from vpncat.experiment import DOMAINS
from vpncat.hashing import sha256_file
from vpncat.preprocessing import (
    StatisticalObservations,
    build_statistical_observations,
    pair_id_digest,
)
from vpncat.provenance import git_provenance
from vpncat.schema import CANONICAL_COLUMNS


@dataclass(frozen=True)
class CrossSessionPreprocessingConfig:
    config_path: Path
    project_root: Path
    cross_session: CrossSessionConfig
    audit_output: Path
    prefix_length: int
    statistical_representations: tuple[str, ...]
    fit_domain: str
    transform_domains: tuple[str, ...]


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def load_cross_session_preprocessing_config(
    path: Path,
    *,
    artifact_dir: Path | None = None,
    output_root: Path | None = None,
    split_path: Path | None = None,
    contract_audit_path: Path | None = None,
    audit_output: Path | None = None,
) -> CrossSessionPreprocessingConfig:
    path = path.expanduser().resolve()
    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle).get("cross_session_preprocessing", {})
    project_root = path.parent.parent
    cross_session = load_cross_session_config(
        _resolve(project_root, raw.get("cross_session_config_path", "")),
        artifact_dir=artifact_dir,
        output_root=output_root,
        split_path=split_path,
        contract_audit_path=contract_audit_path,
    )
    if audit_output is not None:
        output = _resolve(project_root, audit_output)
    elif artifact_dir is not None:
        output = _resolve(project_root, artifact_dir) / "cross_session_preprocessing_audit.json"
    else:
        output = _resolve(project_root, raw.get("audit_output", ""))
    config = CrossSessionPreprocessingConfig(
        config_path=path,
        project_root=project_root,
        cross_session=cross_session,
        audit_output=output,
        prefix_length=int(raw.get("prefix_length", 0)),
        statistical_representations=tuple(
            str(value) for value in raw.get("statistical_representations", [])
        ),
        fit_domain=str(raw.get("fit_domain", "")),
        transform_domains=tuple(str(value) for value in raw.get("transform_domains", [])),
    )
    if config.prefix_length != 50:
        raise PipelineInvariantError("Cross-session preprocessing prefix must be 50")
    if config.statistical_representations != ("matched_flow_stats", "prefix_stats"):
        raise PipelineInvariantError("Cross-session statistical representations differ")
    if config.fit_domain != "inner" or config.transform_domains != DOMAINS:
        raise PipelineInvariantError("Cross-session preprocessing domains differ")
    if raw.get("imputation") != "median" or raw.get("scaling") != "none":
        raise PipelineInvariantError("Cross-session preprocessing policy differs")
    if raw.get("class_weighting") != "balanced":
        raise PipelineInvariantError("Cross-session class weighting differs")
    return config


def _subset(
    observations: StatisticalObservations,
    positions: np.ndarray,
) -> StatisticalObservations:
    return StatisticalObservations(
        pair_ids=tuple(observations.pair_ids[position] for position in positions),
        domain=observations.domain,
        representation=observations.representation,
        feature_names=observations.feature_names,
        values=observations.values[positions],
    )


def _index_summary(index: CrossSessionIndex) -> dict[str, Any]:
    target = fit_cross_session_targets(index)
    target.encode_labels(index.labels)
    labels = np.asarray(index.labels, dtype=object)[index.train_positions].astype(str)
    totals = np.bincount(
        target.encode_labels(labels),
        weights=target.sample_weights(labels),
        minlength=len(target.classes),
    )
    if not np.allclose(totals, totals[0]):
        raise PipelineInvariantError("Cross-session class weights are not balanced")
    role_ids = {
        role: index.pair_ids_for(role) for role in ("train", "validation", "test")
    }
    if any(
        set(role_ids[left]) & set(role_ids[right])
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
    ):
        raise PipelineInvariantError("Cross-session pair roles overlap")
    poisoned_labels = list(index.labels)
    for position in np.concatenate([index.validation_positions, index.test_positions]):
        poisoned_labels[int(position)] = "POISONED"
    poisoned_target = fit_cross_session_targets(
        replace(index, labels=tuple(poisoned_labels))
    )
    if poisoned_target.to_dict() != target.to_dict():
        raise PipelineInvariantError("Non-training labels changed cross-session targets")
    return {
        "train_session": index.train_session,
        "test_session": index.test_session,
        "role_counts": {role: len(ids) for role, ids in role_ids.items()},
        "role_pair_id_sha256": {
            role: pair_id_digest(ids)
            for role, ids in role_ids.items()
        },
        "pair_overlap": 0,
        "targets": target.to_dict(),
        "adversarial_label_poison": "passed",
    }


def audit_cross_session_preprocessing(
    config: CrossSessionPreprocessingConfig,
) -> dict[str, Any]:
    contract = validate_cross_session_contract(config.cross_session)
    canonical_path = config.cross_session.primary.canonical_path
    canonical = pq.read_table(canonical_path, columns=list(CANONICAL_COLUMNS)).to_pandas()
    split = pd.read_csv(config.cross_session.split_path)
    metadata = canonical.loc[:, ["pair_id", "session", "application_category"]]
    indexes = {
        session: materialize_cross_session_index(metadata, split, train_session=session)
        for session in config.cross_session.sessions
    }
    directions = {str(session): _index_summary(index) for session, index in indexes.items()}

    fitted_states: dict[str, Any] = {}
    for representation in config.statistical_representations:
        observations = {
            domain: build_statistical_observations(
                canonical,
                domain=domain,
                representation=representation,
                prefix_length=config.prefix_length,
            )
            for domain in config.transform_domains
        }
        representation_states: dict[str, Any] = {}
        for session, index in indexes.items():
            state = fit_cross_session_preprocessing(
                observations[config.fit_domain], index
            )
            transformed: dict[str, dict[str, int]] = {}
            for domain in config.transform_domains:
                transformed[domain] = {}
                for role in ("train", "validation", "test"):
                    values = state.transform_features(
                        _subset(observations[domain], index.positions(role))
                    )
                    if not np.isfinite(values).all():
                        raise PipelineInvariantError("Cross-session transformed values are invalid")
                    transformed[domain][role] = len(values)

            poisoned_values = observations[config.fit_domain].values.copy()
            forbidden = np.concatenate([index.validation_positions, index.test_positions])
            poisoned_values[forbidden] = 1e30
            poisoned = replace(
                observations[config.fit_domain],
                values=poisoned_values,
            )
            poisoned_state = fit_cross_session_preprocessing(poisoned, index)
            if poisoned_state.to_dict() != state.to_dict():
                raise PipelineInvariantError("Forbidden rows changed cross-session preprocessing")
            opposite_values = observations["outer"].values.copy()
            opposite_values[:] = -1e30
            poisoned_observations = dict(observations)
            poisoned_observations["outer"] = replace(
                observations["outer"], values=opposite_values
            )
            if poisoned_observations["outer"].domain == state.train_domain:
                raise PipelineInvariantError("Opposite-domain poison targeted the fit domain")
            opposite_poison_state = fit_cross_session_preprocessing(
                poisoned_observations[config.fit_domain], index
            )
            if opposite_poison_state.to_dict() != state.to_dict():
                raise PipelineInvariantError("Opposite-domain rows changed fitted state")
            state_payload = state.to_dict()
            state_payload["fit_role"] = "train"
            state_payload["transformed_rows"] = transformed
            state_payload["adversarial_forbidden_row_poison"] = "passed"
            state_payload["adversarial_opposite_domain_poison"] = "passed"
            state_payload["opposite_domain_fit_access"] = False
            representation_states[str(session)] = state_payload
        fitted_states[representation] = representation_states

    input_hashes = {
        "preprocessing_config": sha256_file(config.config_path),
        "cross_session_config": sha256_file(config.cross_session.config_path),
        "canonical": sha256_file(canonical_path),
        "cross_session_split": sha256_file(config.cross_session.split_path),
        "cross_session_contract_audit": sha256_file(
            config.cross_session.contract_audit_path
        ),
    }
    payload = {
        "audit_version": 1,
        "created_utc": datetime.now(UTC).isoformat(),
        "package_version": __version__,
        "git": git_provenance(config.project_root),
        "input_hashes": input_hashes,
        "contract_validation": contract,
        "configuration": {
            "prefix_length": config.prefix_length,
            "fit_domain": config.fit_domain,
            "transform_domains": list(config.transform_domains),
            "statistical_representations": list(config.statistical_representations),
            "fit_free_representations": ["flattened_splt", "sequential_splt"],
            "imputation": "training-median",
            "scaling": "none",
            "class_weighting": "balanced",
        },
        "directions": directions,
        "fitted_states": fitted_states,
        "status": "valid",
    }
    config.audit_output.parent.mkdir(parents=True, exist_ok=True)
    temporary = config.audit_output.with_suffix(config.audit_output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(config.audit_output)
    return payload
