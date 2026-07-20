from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq
import yaml

from vpncat import __version__
from vpncat.artifacts import verify_input_chain
from vpncat.errors import PipelineInvariantError
from vpncat.experiment import (
    PrimaryExperimentConfig,
    RunSpec,
    load_primary_experiment_config,
    select_primary_run,
)
from vpncat.folds import materialize_fold_index
from vpncat.hashing import sha256_file
from vpncat.neural_config import NeuralConfig, load_neural_config
from vpncat.preprocessing import pair_id_digest
from vpncat.provenance import git_provenance

ABLATION_MODELS = ("cnn1d", "transformer")
ALL_CHANNELS = ("direction", "size", "iat_ms")
PREFIX_LENGTHS = (10, 20, 50, 80)
CHANNEL_COMBINATIONS = (
    ("direction", ("direction",)),
    ("direction_size", ("direction", "size")),
    ("direction_timing", ("direction", "iat_ms")),
    ("size_timing", ("size", "iat_ms")),
    ("all", ALL_CHANNELS),
)


@dataclass(frozen=True)
class AblationObservation:
    observation_id: str
    prefix_length: int
    channels: tuple[str, ...]
    reuse_primary: bool


@dataclass(frozen=True)
class AblationRunSpec:
    protocol: str
    kind: str
    experiment_id: str
    observation_id: str
    representation: str
    model: str
    family: str
    fold: int
    seed: int
    train_domain: str
    test_domains: tuple[str, ...]
    prefix_length: int
    channels: tuple[str, ...]
    execution_mode: str

    @property
    def run_id(self) -> str:
        return (
            f"{self.protocol}__{self.experiment_id}__fold_{self.fold:02d}__"
            f"train_{self.train_domain}__seed_{self.seed:06d}"
        )

    @property
    def relative_output_dir(self) -> Path:
        return Path(
            self.model,
            self.observation_id,
            f"fold_{self.fold:02d}",
            f"train_{self.train_domain}",
            f"seed_{self.seed:06d}",
        )

    @property
    def is_primary_reference(self) -> bool:
        return self.execution_mode == "reuse_primary"

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol": self.protocol,
            "run_id": self.run_id,
            "kind": self.kind,
            "experiment_id": self.experiment_id,
            "observation_id": self.observation_id,
            "representation": self.representation,
            "model": self.model,
            "family": self.family,
            "fold": self.fold,
            "seed": self.seed,
            "train_domain": self.train_domain,
            "test_domains": list(self.test_domains),
            "prefix_length": self.prefix_length,
            "channels": list(self.channels),
            "execution_mode": self.execution_mode,
        }


@dataclass(frozen=True)
class AblationConfig:
    config_path: Path
    project_root: Path
    primary: PrimaryExperimentConfig
    neural: NeuralConfig
    contract_audit_path: Path
    output_root: Path
    kind: str
    protocol: str
    models: tuple[str, ...]
    folds: tuple[int, ...]
    seed: int
    train_domain: str
    test_domains: tuple[str, ...]
    observations: tuple[AblationObservation, ...]


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def load_ablation_config(
    path: Path,
    *,
    artifact_dir: Path | None = None,
    output_root: Path | None = None,
    tuning_output_root: Path | None = None,
    contract_audit_path: Path | None = None,
) -> AblationConfig:
    path = path.expanduser().resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")).get("ablation", {})
    root = path.parent.parent
    resolved_artifacts = (
        _resolve(root, artifact_dir) if artifact_dir is not None else None
    )
    primary = load_primary_experiment_config(
        _resolve(root, raw.get("primary_config_path", "")),
        artifact_dir=resolved_artifacts,
    )
    neural = load_neural_config(
        _resolve(root, raw.get("neural_config_path", "")),
        tuning_output_root=tuning_output_root,
    )
    kind = str(raw.get("kind", ""))
    if kind == "prefix_length":
        observations = tuple(
            AblationObservation(
                observation_id=f"n{int(length):03d}",
                prefix_length=int(length),
                channels=tuple(str(value) for value in raw.get("channels", [])),
                reuse_primary=int(length)
                == int(raw.get("primary_reference_prefix_length", -1)),
            )
            for length in raw.get("prefix_lengths", [])
        )
    elif kind == "channels":
        reference_id = str(raw.get("primary_reference_channel_id", ""))
        observations = tuple(
            AblationObservation(
                observation_id=str(item.get("id", "")),
                prefix_length=int(raw.get("prefix_length", 0)),
                channels=tuple(str(value) for value in item.get("channels", [])),
                reuse_primary=str(item.get("id", "")) == reference_id,
            )
            for item in raw.get("channel_combinations", [])
        )
    else:
        observations = ()
    default_audit_name = (
        "ablation_prefix_contract_audit.json"
        if kind == "prefix_length"
        else f"ablation_{kind}_contract_audit.json"
    )
    audit_path = (
        _resolve(root, contract_audit_path)
        if contract_audit_path is not None
        else (
            resolved_artifacts / default_audit_name
            if resolved_artifacts is not None
            else _resolve(root, raw.get("contract_audit_path", ""))
        )
    )
    config = AblationConfig(
        config_path=path,
        project_root=root,
        primary=primary,
        neural=neural,
        contract_audit_path=audit_path,
        output_root=_resolve(
            root,
            output_root if output_root is not None else raw.get("output_root", ""),
        ),
        kind=kind,
        protocol=str(raw.get("protocol", "")),
        models=tuple(str(value) for value in raw.get("models", [])),
        folds=tuple(int(value) for value in raw.get("folds", [])),
        seed=int(raw.get("seed", -1)),
        train_domain=str(raw.get("train_domain", "")),
        test_domains=tuple(str(value) for value in raw.get("test_domains", [])),
        observations=observations,
    )
    _validate_ablation_config(config, raw)
    return config


def _validate_ablation_config(config: AblationConfig, raw: dict[str, Any]) -> None:
    expected_protocol = {
        "prefix_length": "ablation_prefix",
        "channels": "ablation_channels",
    }.get(config.kind)
    if config.protocol != expected_protocol:
        raise PipelineInvariantError("Ablation protocol and kind disagree")
    if (
        config.models != ABLATION_MODELS
        or config.folds != config.primary.folds
        or config.seed != 42
        or config.train_domain != "inner"
        or config.test_domains != ("inner", "outer")
        or raw.get("primary_reference_seed") != 42
        or raw.get("primary_reference_policy") != "reuse_without_retraining"
        or raw.get("augmentation") is not False
    ):
        raise PipelineInvariantError("Ablation common policy differs from the freeze")
    observed = tuple(
        (item.observation_id, item.prefix_length, item.channels, item.reuse_primary)
        for item in config.observations
    )
    if config.kind == "prefix_length":
        expected = tuple(
            (f"n{length:03d}", length, ALL_CHANNELS, length == 50)
            for length in PREFIX_LENGTHS
        )
    else:
        expected = tuple(
            (identifier, 50, channels, identifier == "all")
            for identifier, channels in CHANNEL_COMBINATIONS
        )
    if observed != expected:
        raise PipelineInvariantError("Ablation observations differ from the freeze")
    if any(
        item.prefix_length > config.neural.maximum_prefix_length
        for item in config.observations
    ):
        raise PipelineInvariantError("Ablation prefix exceeds the neural maximum")


def enumerate_ablation_runs(config: AblationConfig) -> tuple[AblationRunSpec, ...]:
    runs = tuple(
        AblationRunSpec(
            protocol=config.protocol,
            kind=config.kind,
            experiment_id=f"{observation.observation_id}__{model}",
            observation_id=observation.observation_id,
            representation="sequential_splt",
            model=model,
            family="neural",
            fold=fold,
            seed=config.seed,
            train_domain=config.train_domain,
            test_domains=config.test_domains,
            prefix_length=observation.prefix_length,
            channels=observation.channels,
            execution_mode="reuse_primary" if observation.reuse_primary else "train",
        )
        for model in config.models
        for observation in config.observations
        for fold in config.folds
    )
    expected = 40 if config.kind == "prefix_length" else 50
    expected_references = 10
    if (
        len(runs) != expected
        or sum(run.is_primary_reference for run in runs) != expected_references
        or len({run.run_id for run in runs}) != len(runs)
        or len({run.relative_output_dir for run in runs}) != len(runs)
    ):
        raise PipelineInvariantError("Ablation matrix size or identity differs")
    return runs


def primary_reference_run(
    config: AblationConfig,
    run: AblationRunSpec,
) -> RunSpec:
    if not run.is_primary_reference:
        raise PipelineInvariantError("Executable ablation has no primary reference")
    return select_primary_run(
        config.primary,
        experiment_id=f"sequential_splt__{run.model}",
        fold=run.fold,
        train_domain="inner",
        seed=42,
    )


def _input_hashes(config: AblationConfig) -> dict[str, str]:
    primary_hashes = verify_input_chain(config.primary)
    primary_contract = json.loads(
        config.primary.contract_audit_path.read_text(encoding="utf-8")
    )
    if (
        primary_contract.get("status") != "valid"
        or primary_contract.get("input_hashes") != primary_hashes
    ):
        raise PipelineInvariantError("Ablations require a valid primary contract audit")
    return {
        **primary_hashes,
        "ablation_config": sha256_file(config.config_path),
        "neural_config": sha256_file(config.neural.config_path),
        "primary_contract_audit": sha256_file(config.primary.contract_audit_path),
    }


def _contract_rows(
    config: AblationConfig,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metadata = pq.read_table(
        config.primary.canonical_path,
        columns=["pair_id", "session", "application_category"],
    ).to_pandas()
    split = pd.read_csv(config.primary.split_path)
    fold_summaries: dict[str, Any] = {}
    for fold_number in config.folds:
        fold = materialize_fold_index(metadata, split, fold=fold_number)
        counts = {
            role: len(fold.positions(role))
            for role in ("train", "validation", "test")
        }
        hashes = {
            role: pair_id_digest(fold.pair_ids_for(role))
            for role in ("train", "validation", "test")
        }
        classes = sorted(set(fold.labels[position] for position in fold.train_positions))
        if any(
            set(fold.labels[position] for position in fold.positions(role))
            != set(classes)
            for role in ("validation", "test")
        ):
            raise PipelineInvariantError("Ablation fold role omits classes")
        fold_summaries[str(fold_number)] = {
            "role_counts": counts,
            "role_pair_ids_sha256": hashes,
            "classes": classes,
        }
    primary_contract = json.loads(
        config.primary.contract_audit_path.read_text(encoding="utf-8")
    )
    primary_rows = primary_contract.get("runs", [])
    rows: list[dict[str, Any]] = []
    for run in enumerate_ablation_runs(config):
        fold_summary = fold_summaries[str(run.fold)]
        row = {
            **run.to_dict(),
            "artifact_source": "primary" if run.is_primary_reference else "ablation",
            "training_pair_count": fold_summary["role_counts"]["train"],
            "validation_pair_count": fold_summary["role_counts"]["validation"],
            "test_pair_count": fold_summary["role_counts"]["test"],
            "training_pair_ids_sha256": fold_summary["role_pair_ids_sha256"]["train"],
            "validation_pair_ids_sha256": fold_summary["role_pair_ids_sha256"][
                "validation"
            ],
            "test_pair_ids_sha256": fold_summary["role_pair_ids_sha256"]["test"],
            "prediction_rows": fold_summary["role_counts"]["test"]
            * len(run.test_domains),
        }
        if run.is_primary_reference:
            reference = primary_reference_run(config, run)
            matches = [
                item for item in primary_rows if item.get("run_id") == reference.run_id
            ]
            if (
                len(matches) != 1
                or matches[0].get("relative_output_dir")
                != reference.relative_output_dir.as_posix()
            ):
                raise PipelineInvariantError("Ablation primary reference is not audited")
            row["artifact_relative_output_dir"] = (
                reference.relative_output_dir.as_posix()
            )
            row["primary_reference"] = {
                "run_id": reference.run_id,
                "relative_output_dir": reference.relative_output_dir.as_posix(),
            }
        else:
            row["artifact_relative_output_dir"] = run.relative_output_dir.as_posix()
            row["primary_reference"] = None
        rows.append(row)
    return fold_summaries, rows


def build_ablation_contract(
    config: AblationConfig,
    *,
    force: bool = False,
) -> dict[str, Any]:
    if config.contract_audit_path.exists() and not force:
        raise FileExistsError("Refusing to overwrite ablation contract audit")
    folds, rows = _contract_rows(config)
    reference_count = sum(row["execution_mode"] == "reuse_primary" for row in rows)
    payload = {
        "audit_version": 1,
        "created_utc": datetime.now(UTC).isoformat(),
        "package_version": __version__,
        "git": git_provenance(config.project_root),
        "input_hashes": _input_hashes(config),
        "protocol": {
            "kind": config.kind,
            "train_domain": config.train_domain,
            "test_domains": list(config.test_domains),
            "seed": config.seed,
            "augmentation": False,
            "primary_reference_policy": "reuse_without_retraining",
        },
        "folds": folds,
        "matrix": {
            "cells": len(rows),
            "training_runs": len(rows) - reference_count,
            "primary_references": reference_count,
            "prediction_groups": len(rows) * len(config.test_domains),
        },
        "runs": rows,
        "status": "valid",
    }
    config.contract_audit_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = config.contract_audit_path.with_suffix(
        config.contract_audit_path.suffix + ".tmp"
    )
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(config.contract_audit_path)
    return payload


def validate_ablation_contract(config: AblationConfig) -> dict[str, Any]:
    if not config.contract_audit_path.is_file():
        raise PipelineInvariantError("Ablation contract audit is missing")
    audit = json.loads(config.contract_audit_path.read_text(encoding="utf-8"))
    folds, rows = _contract_rows(config)
    reference_count = sum(row["execution_mode"] == "reuse_primary" for row in rows)
    expected_matrix = {
        "cells": len(rows),
        "training_runs": len(rows) - reference_count,
        "primary_references": reference_count,
        "prediction_groups": len(rows) * len(config.test_domains),
    }
    expected_protocol = {
        "kind": config.kind,
        "train_domain": config.train_domain,
        "test_domains": list(config.test_domains),
        "seed": config.seed,
        "augmentation": False,
        "primary_reference_policy": "reuse_without_retraining",
    }
    if (
        audit.get("audit_version") != 1
        or audit.get("status") != "valid"
        or audit.get("package_version") != __version__
        or audit.get("input_hashes") != _input_hashes(config)
        or audit.get("protocol") != expected_protocol
        or audit.get("folds") != folds
        or audit.get("matrix") != expected_matrix
        or audit.get("runs") != rows
    ):
        raise PipelineInvariantError("Ablation contract audit is stale")
    return {"status": "valid", "matrix": expected_matrix, "folds": folds}
