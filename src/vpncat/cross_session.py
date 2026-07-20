from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

from vpncat import __version__
from vpncat.errors import PipelineInvariantError
from vpncat.experiment import (
    DOMAINS,
    PrimaryExperimentConfig,
    load_primary_experiment_config,
)
from vpncat.hashing import sha256_file
from vpncat.preprocessing import pair_id_digest
from vpncat.provenance import git_provenance

METADATA_COLUMNS = ("pair_id", "session", "application_category")


@dataclass(frozen=True)
class CrossSessionConfig:
    config_path: Path
    project_root: Path
    primary: PrimaryExperimentConfig
    split_path: Path
    contract_audit_path: Path
    output_root: Path
    sessions: tuple[int, int]
    validation_fraction: float
    validation_seed: int
    prefix_length: int
    train_domain: str
    test_domains: tuple[str, ...]


@dataclass(frozen=True)
class CrossSessionRunSpec:
    protocol: str
    experiment_id: str
    representation: str
    model: str
    family: str
    seed: int
    train_session: int
    test_session: int
    train_domain: str
    test_domains: tuple[str, ...]

    @property
    def run_id(self) -> str:
        return (
            f"{self.protocol}__{self.experiment_id}__train_session_{self.train_session:02d}__"
            f"test_session_{self.test_session:02d}__train_{self.train_domain}__"
            f"seed_{self.seed:06d}"
        )

    @property
    def relative_output_dir(self) -> Path:
        return Path(
            self.representation,
            self.model,
            f"train_session_{self.train_session:02d}",
            f"test_session_{self.test_session:02d}",
            f"train_{self.train_domain}",
            f"seed_{self.seed:06d}",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol": self.protocol,
            "run_id": self.run_id,
            "experiment_id": self.experiment_id,
            "representation": self.representation,
            "model": self.model,
            "family": self.family,
            "seed": self.seed,
            "train_session": self.train_session,
            "test_session": self.test_session,
            "train_domain": self.train_domain,
            "test_domains": list(self.test_domains),
        }


def _resolve(project_root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def load_cross_session_config(
    path: Path,
    *,
    artifact_dir: Path | None = None,
    output_root: Path | None = None,
    split_path: Path | None = None,
    contract_audit_path: Path | None = None,
) -> CrossSessionConfig:
    path = path.expanduser().resolve()
    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle).get("cross_session", {})
    project_root = path.parent.parent
    resolved_artifact_dir = (
        _resolve(project_root, artifact_dir) if artifact_dir is not None else None
    )
    primary = load_primary_experiment_config(
        _resolve(project_root, raw.get("primary_config_path", "")),
        artifact_dir=resolved_artifact_dir,
    )

    def artifact(configured: str, override: Path | None, filename: str) -> Path:
        if override is not None:
            return _resolve(project_root, override)
        if resolved_artifact_dir is not None:
            return resolved_artifact_dir / filename
        return _resolve(project_root, raw.get(configured, ""))

    sessions = tuple(int(value) for value in raw.get("sessions", []))
    config = CrossSessionConfig(
        config_path=path,
        project_root=project_root,
        primary=primary,
        split_path=artifact(
            "split_path", split_path, "cross_session_split_manifest.csv"
        ),
        contract_audit_path=artifact(
            "contract_audit_path",
            contract_audit_path,
            "cross_session_contract_audit.json",
        ),
        output_root=_resolve(
            project_root,
            output_root if output_root is not None else raw.get("output_root", ""),
        ),
        sessions=sessions,
        validation_fraction=float(raw.get("validation_fraction", 0)),
        validation_seed=int(raw.get("validation_seed", -1)),
        prefix_length=int(raw.get("prefix_length", 0)),
        train_domain=str(raw.get("train_domain", "")),
        test_domains=tuple(str(value) for value in raw.get("test_domains", [])),
    )
    if raw.get("protocol") != "cross_session":
        raise PipelineInvariantError("Cross-session protocol name is invalid")
    if config.sessions != (1, 2):
        raise PipelineInvariantError("Cross-session protocol requires sessions [1, 2]")
    if config.validation_fraction != 0.1 or config.validation_seed != 2026:
        raise PipelineInvariantError("Cross-session validation policy differs from the freeze")
    if config.prefix_length != 50:
        raise PipelineInvariantError("Cross-session primary prefix must be 50")
    if config.train_domain != "inner" or config.test_domains != DOMAINS:
        raise PipelineInvariantError("Cross-session domains differ from the base protocol")
    if raw.get("augmentation") is not False:
        raise PipelineInvariantError("Cross-session augmentation is prohibited")
    if raw.get("outer_trained_references") != "deferred_until_primary_selection":
        raise PipelineInvariantError("Cross-session outer references must remain deferred")
    return config


def enumerate_cross_session_runs(
    config: CrossSessionConfig,
) -> tuple[CrossSessionRunSpec, ...]:
    runs = tuple(
        CrossSessionRunSpec(
            protocol="cross_session",
            experiment_id=item.experiment_id,
            representation=item.representation,
            model=item.model,
            family=item.family,
            seed=seed,
            train_session=train_session,
            test_session=next(
                session for session in config.sessions if session != train_session
            ),
            train_domain=config.train_domain,
            test_domains=config.test_domains,
        )
        for item in config.primary.configurations
        for train_session in config.sessions
        for seed in item.seeds
    )
    if len(runs) != 30:
        raise PipelineInvariantError(
            f"Cross-session base matrix must contain 30 runs, observed {len(runs)}"
        )
    if len({run.run_id for run in runs}) != len(runs):
        raise PipelineInvariantError("Cross-session run IDs are not unique")
    if len({run.relative_output_dir for run in runs}) != len(runs):
        raise PipelineInvariantError("Cross-session output paths are not unique")
    return runs


def select_cross_session_run(
    config: CrossSessionConfig,
    *,
    experiment_id: str,
    train_session: int,
    seed: int,
) -> CrossSessionRunSpec:
    matches = [
        run
        for run in enumerate_cross_session_runs(config)
        if run.experiment_id == experiment_id
        and run.train_session == train_session
        and run.seed == seed
    ]
    if len(matches) != 1:
        raise PipelineInvariantError(
            "Cross-session run selector did not resolve exactly one frozen run"
        )
    return matches[0]


def _score(pair_id: str, *, seed: int, train_session: int) -> str:
    value = f"{seed}:{train_session}:{pair_id}".encode()
    return hashlib.sha256(value).hexdigest()


def _validation_quotas(
    counts: dict[str, int],
    *,
    fraction: float,
) -> dict[str, int]:
    target = int(round(sum(counts.values()) * fraction))
    raw = {label: count * fraction for label, count in counts.items()}
    quotas = {label: int(np.floor(value)) for label, value in raw.items()}
    remaining = target - sum(quotas.values())
    order = sorted(raw, key=lambda label: (-(raw[label] - quotas[label]), label))
    for label in order[:remaining]:
        quotas[label] += 1
    for label, count in counts.items():
        if quotas[label] < 1 or quotas[label] >= count:
            raise PipelineInvariantError(
                f"Class {label} cannot support cross-session train/validation roles"
            )
    if sum(quotas.values()) != target:
        raise PipelineInvariantError("Cross-session validation quota total is invalid")
    return quotas


def build_cross_session_split(
    metadata: pd.DataFrame,
    config: CrossSessionConfig,
) -> pd.DataFrame:
    if tuple(metadata.columns) != METADATA_COLUMNS:
        raise PipelineInvariantError("Cross-session metadata schema is invalid")
    frame = metadata.copy()
    frame["pair_id"] = frame["pair_id"].astype(str)
    frame["session"] = frame["session"].astype(np.int64)
    frame["application_category"] = frame["application_category"].astype(str)
    if frame["pair_id"].duplicated().any():
        raise PipelineInvariantError("Cross-session metadata contains duplicate pair IDs")
    if set(frame["session"].astype(int)) != set(config.sessions):
        raise PipelineInvariantError("Cross-session metadata has unexpected sessions")
    frame = frame.sort_values("pair_id", kind="mergesort").reset_index(drop=True)

    for train_session in config.sessions:
        role_column = f"role_train_session_{train_session}"
        frame[role_column] = "test"
        source = frame["session"].astype(int) == train_session
        source_frame = frame.loc[source]
        counts = source_frame["application_category"].value_counts().to_dict()
        quotas = _validation_quotas(counts, fraction=config.validation_fraction)
        validation_ids: set[str] = set()
        for label, quota in sorted(quotas.items()):
            candidates = source_frame.loc[
                source_frame["application_category"] == label, "pair_id"
            ].tolist()
            ranked = sorted(
                candidates,
                key=lambda pair_id: (
                    _score(
                        pair_id,
                        seed=config.validation_seed,
                        train_session=train_session,
                    ),
                    pair_id,
                ),
            )
            validation_ids.update(ranked[:quota])
        frame.loc[source, role_column] = "train"
        frame.loc[frame["pair_id"].isin(validation_ids), role_column] = "validation"
    return frame.loc[
        :, [*METADATA_COLUMNS, *(f"role_train_session_{s}" for s in config.sessions)]
    ]


def _input_hashes(config: CrossSessionConfig) -> dict[str, str]:
    paths = {
        "cross_session_config": config.config_path,
        "primary_config": config.primary.config_path,
        "canonical": config.primary.canonical_path,
        "dataset_manifest": config.primary.dataset_manifest_path,
        "feature_audit": config.primary.feature_audit_path,
    }
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    dataset_manifest = json.loads(
        config.primary.dataset_manifest_path.read_text(encoding="utf-8")
    )
    feature_audit = json.loads(
        config.primary.feature_audit_path.read_text(encoding="utf-8")
    )
    if dataset_manifest["artifacts"]["canonical_pairs"]["sha256"] != hashes["canonical"]:
        raise PipelineInvariantError("Cross-session canonical hash disagrees with manifest")
    if feature_audit.get("status") != "valid" or feature_audit.get(
        "canonical", {}
    ).get("sha256") != hashes["canonical"]:
        raise PipelineInvariantError("Cross-session feature audit is stale")
    return hashes


def _role_summary(
    split: pd.DataFrame,
    config: CrossSessionConfig,
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for train_session in config.sessions:
        test_session = next(s for s in config.sessions if s != train_session)
        column = f"role_train_session_{train_session}"
        roles = split[column].astype(str)
        source = split["session"].astype(int) == train_session
        target = split["session"].astype(int) == test_session
        boundaries_valid = roles[source].isin({"train", "validation"}).all() and (
            roles[target] == "test"
        ).all()
        if not boundaries_valid:
            raise PipelineInvariantError("Cross-session roles violate session boundaries")
        expected_validation = int(round(int(source.sum()) * config.validation_fraction))
        if int((roles == "validation").sum()) != expected_validation:
            raise PipelineInvariantError("Cross-session validation count is not exactly 10%")
        role_counts = roles.value_counts().to_dict()
        classes_by_role = {
            role: sorted(split.loc[roles == role, "application_category"].unique())
            for role in ("train", "validation", "test")
        }
        if len({tuple(value) for value in classes_by_role.values()}) != 1:
            raise PipelineInvariantError("A cross-session role lacks one or more classes")
        summary[str(train_session)] = {
            "test_session": test_session,
            "counts": {role: int(role_counts[role]) for role in ("train", "validation", "test")},
            "pair_id_sha256": {
                role: pair_id_digest(tuple(split.loc[roles == role, "pair_id"].astype(str)))
                for role in ("train", "validation", "test")
            },
            "classes": classes_by_role["train"],
            "class_counts": {
                role: {
                    str(label): int(count)
                    for label, count in sorted(
                        split.loc[
                            roles == role, "application_category"
                        ].value_counts().to_dict().items()
                    )
                }
                for role in ("train", "validation", "test")
            },
        }
    return summary


def _run_rows(
    split: pd.DataFrame,
    config: CrossSessionConfig,
    role_summary: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for run in enumerate_cross_session_runs(config):
        summary = role_summary[str(run.train_session)]
        rows.append(
            {
                **run.to_dict(),
                "relative_output_dir": run.relative_output_dir.as_posix(),
                "training_pair_count": summary["counts"]["train"],
                "validation_pair_count": summary["counts"]["validation"],
                "test_pair_count": summary["counts"]["test"],
                "training_pair_ids_sha256": summary["pair_id_sha256"]["train"],
                "validation_pair_ids_sha256": summary["pair_id_sha256"]["validation"],
                "test_pair_ids_sha256": summary["pair_id_sha256"]["test"],
                "prediction_rows": summary["counts"]["test"] * len(run.test_domains),
            }
        )
    return rows


def build_cross_session_contract(
    config: CrossSessionConfig,
    *,
    force: bool = False,
) -> dict[str, Any]:
    targets = (config.split_path, config.contract_audit_path)
    if not force and any(path.exists() for path in targets):
        raise FileExistsError("Refusing to overwrite cross-session contract artifacts")
    metadata = pq.read_table(
        config.primary.canonical_path,
        columns=list(METADATA_COLUMNS),
    ).to_pandas()
    split = build_cross_session_split(metadata, config)
    input_hashes = _input_hashes(config)
    role_summary = _role_summary(split, config)
    config.split_path.parent.mkdir(parents=True, exist_ok=True)
    config.contract_audit_path.parent.mkdir(parents=True, exist_ok=True)
    staged_split = config.split_path.with_suffix(config.split_path.suffix + ".tmp")
    staged_audit = config.contract_audit_path.with_suffix(
        config.contract_audit_path.suffix + ".tmp"
    )
    try:
        split.to_csv(staged_split, index=False)
        family_counts = Counter(run.family for run in enumerate_cross_session_runs(config))
        payload = {
            "audit_version": 1,
            "created_utc": datetime.now(UTC).isoformat(),
            "package_version": __version__,
            "git": git_provenance(config.project_root),
            "input_hashes": input_hashes,
            "split_sha256": sha256_file(staged_split),
            "validation_policy": {
                "fraction": config.validation_fraction,
                "seed": config.validation_seed,
                "allocation": "class-stratified-largest-remainder",
                "selection": "seeded-sha256-rank",
            },
            "directions": role_summary,
            "matrix": {
                "training_runs": 30,
                "prediction_groups": 60,
                "family_counts": dict(sorted(family_counts.items())),
                "outer_trained_references": "deferred_until_primary_selection",
            },
            "runs": _run_rows(split, config, role_summary),
            "status": "valid",
        }
        staged_audit.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        staged_split.replace(config.split_path)
        staged_audit.replace(config.contract_audit_path)
    finally:
        staged_split.unlink(missing_ok=True)
        staged_audit.unlink(missing_ok=True)
    return payload


def validate_cross_session_contract(config: CrossSessionConfig) -> dict[str, Any]:
    if not config.split_path.is_file() or not config.contract_audit_path.is_file():
        raise PipelineInvariantError("Cross-session contract artifacts are missing")
    metadata = pq.read_table(
        config.primary.canonical_path,
        columns=list(METADATA_COLUMNS),
    ).to_pandas()
    expected_split = build_cross_session_split(metadata, config)
    observed_split = pd.read_csv(config.split_path)
    try:
        pd.testing.assert_frame_equal(observed_split, expected_split, check_exact=True)
    except AssertionError as error:
        raise PipelineInvariantError(
            "Cross-session split differs from deterministic build"
        ) from error
    audit = json.loads(config.contract_audit_path.read_text(encoding="utf-8"))
    role_summary = _role_summary(observed_split, config)
    if audit.get("status") != "valid" or audit.get("package_version") != __version__:
        raise PipelineInvariantError("Cross-session audit status or version is invalid")
    if audit.get("input_hashes") != _input_hashes(config):
        raise PipelineInvariantError("Cross-session audit input hashes are stale")
    if audit.get("split_sha256") != sha256_file(config.split_path):
        raise PipelineInvariantError("Cross-session split hash is stale")
    if audit.get("directions") != role_summary:
        raise PipelineInvariantError("Cross-session role summary is stale")
    if audit.get("runs") != _run_rows(observed_split, config, role_summary):
        raise PipelineInvariantError("Cross-session run contract is stale")
    return {
        "status": "valid",
        "split_sha256": audit["split_sha256"],
        "training_runs": len(audit["runs"]),
        "directions": role_summary,
    }
