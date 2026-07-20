from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from vpncat import __version__
from vpncat.ablations import (
    AblationConfig,
    enumerate_ablation_runs,
    load_ablation_config,
    primary_reference_run,
    validate_ablation_contract,
)
from vpncat.artifacts import verify_input_chain
from vpncat.cross_session import (
    CrossSessionConfig,
    enumerate_cross_session_runs,
    load_cross_session_config,
    validate_cross_session_contract,
)
from vpncat.dann import (
    DANNConfig,
    enumerate_dann_runs,
    load_dann_config,
    validate_dann_contract,
)
from vpncat.errors import PipelineInvariantError
from vpncat.experiment import (
    PrimaryExperimentConfig,
    enumerate_primary_runs,
    load_primary_experiment_config,
)
from vpncat.hashing import sha256_file
from vpncat.metrics import METRIC_NAMES
from vpncat.provenance import git_provenance

ANALYSIS_PRIMARY_METRICS = ("balanced_accuracy", "macro_f1")
ANALYSIS_PROTOCOLS = (
    "primary",
    "cross_session",
    "dann",
    "ablation_prefix",
    "ablation_channels",
)


@dataclass(frozen=True)
class AnalysisConfig:
    config_path: Path
    project_root: Path
    primary: PrimaryExperimentConfig
    cross_session: CrossSessionConfig
    dann: DANNConfig
    ablation_prefix: AblationConfig
    ablation_channels: AblationConfig
    contract_audit_path: Path
    output_root: Path
    metrics: tuple[str, ...]
    primary_metrics: tuple[str, ...]
    neural_seed_policy: str
    seed_dispersion: str
    bootstrap: dict[str, Any]


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def load_analysis_config(
    path: Path,
    *,
    artifact_dir: Path | None = None,
    output_root: Path | None = None,
    primary_output_root: Path | None = None,
    cross_session_output_root: Path | None = None,
    dann_output_root: Path | None = None,
    ablation_prefix_output_root: Path | None = None,
    ablation_channels_output_root: Path | None = None,
    tuning_output_root: Path | None = None,
    contract_audit_path: Path | None = None,
) -> AnalysisConfig:
    path = path.expanduser().resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")).get("analysis", {})
    root = path.parent.parent
    resolved_artifacts = (
        _resolve(root, artifact_dir) if artifact_dir is not None else None
    )
    primary = load_primary_experiment_config(
        _resolve(root, raw.get("primary_config_path", "")),
        artifact_dir=resolved_artifacts,
        output_root=primary_output_root,
    )
    cross_session = load_cross_session_config(
        _resolve(root, raw.get("cross_session_config_path", "")),
        artifact_dir=resolved_artifacts,
        output_root=cross_session_output_root,
    )
    dann = load_dann_config(
        _resolve(root, raw.get("dann_config_path", "")),
        artifact_dir=resolved_artifacts,
        output_root=dann_output_root,
        tuning_output_root=tuning_output_root,
    )
    ablation_prefix = load_ablation_config(
        _resolve(root, raw.get("ablation_prefix_config_path", "")),
        artifact_dir=resolved_artifacts,
        output_root=ablation_prefix_output_root,
        tuning_output_root=tuning_output_root,
    )
    ablation_channels = load_ablation_config(
        _resolve(root, raw.get("ablation_channels_config_path", "")),
        artifact_dir=resolved_artifacts,
        output_root=ablation_channels_output_root,
        tuning_output_root=tuning_output_root,
    )
    audit_path = (
        _resolve(root, contract_audit_path)
        if contract_audit_path is not None
        else (
            resolved_artifacts / "analysis_contract_audit.json"
            if resolved_artifacts is not None
            else _resolve(root, raw.get("contract_audit_path", ""))
        )
    )
    config = AnalysisConfig(
        config_path=path,
        project_root=root,
        primary=primary,
        cross_session=cross_session,
        dann=dann,
        ablation_prefix=ablation_prefix,
        ablation_channels=ablation_channels,
        contract_audit_path=audit_path,
        output_root=_resolve(
            root, output_root if output_root is not None else raw.get("output_root", "")
        ),
        metrics=tuple(str(value) for value in raw.get("metrics", [])),
        primary_metrics=tuple(str(value) for value in raw.get("primary_metrics", [])),
        neural_seed_policy=str(raw.get("neural_seed_policy", "")),
        seed_dispersion=str(raw.get("seed_dispersion", "")),
        bootstrap=dict(raw.get("bootstrap", {})),
    )
    _validate_analysis_config(config, raw)
    return config


def _validate_analysis_config(config: AnalysisConfig, raw: dict[str, Any]) -> None:
    if raw.get("protocol") != "analysis":
        raise PipelineInvariantError("Analysis protocol name differs from the freeze")
    if config.metrics != METRIC_NAMES or config.primary_metrics != ANALYSIS_PRIMARY_METRICS:
        raise PipelineInvariantError("Analysis metrics differ from the frozen metric set")
    if (
        config.neural_seed_policy != "mean_class_probabilities"
        or config.seed_dispersion != "separate"
    ):
        raise PipelineInvariantError("Analysis seed policy differs from the freeze")
    if config.bootstrap != {
        "resampling_unit": "pair_id",
        "paired_views": True,
        "replicates": 1000,
        "confidence_level": 0.95,
        "seed": 42,
    }:
        raise PipelineInvariantError("Analysis bootstrap policy differs from the freeze")
    if (
        config.cross_session.primary.config_path != config.primary.config_path
        or config.dann.primary.config_path != config.primary.config_path
        or config.ablation_prefix.primary.config_path != config.primary.config_path
        or config.ablation_channels.primary.config_path != config.primary.config_path
    ):
        raise PipelineInvariantError("Analysis protocols do not share one primary config")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_primary_contract(config: PrimaryExperimentConfig) -> dict[str, Any]:
    audit = _load_json(config.contract_audit_path)
    expected_runs = enumerate_primary_runs(config)
    expected_hashes = verify_input_chain(config)
    if (
        audit.get("status") != "valid"
        or audit.get("input_hashes") != expected_hashes
        or len(audit.get("runs", [])) != len(expected_runs)
    ):
        raise PipelineInvariantError("Primary contract audit is stale for analysis")
    audited = {row.get("run_id"): row for row in audit["runs"]}
    if len(audited) != len(expected_runs):
        raise PipelineInvariantError("Primary contract run identities are duplicated")
    for run in expected_runs:
        row = audited.get(run.run_id)
        if (
            row is None
            or row.get("relative_output_dir") != run.relative_output_dir.as_posix()
            or any(row.get(key) != value for key, value in run.to_dict().items())
        ):
            raise PipelineInvariantError("Primary contract run inventory differs")
    return audit


def _validated_audits(config: AnalysisConfig) -> dict[str, dict[str, Any]]:
    primary = _validate_primary_contract(config.primary)
    validate_cross_session_contract(config.cross_session)
    validate_dann_contract(config.dann)
    validate_ablation_contract(config.ablation_prefix)
    validate_ablation_contract(config.ablation_channels)
    return {
        "primary": primary,
        "cross_session": _load_json(config.cross_session.contract_audit_path),
        "dann": _load_json(config.dann.contract_audit_path),
        "ablation_prefix": _load_json(config.ablation_prefix.contract_audit_path),
        "ablation_channels": _load_json(config.ablation_channels.contract_audit_path),
    }


def _audit_rows(audit: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = audit.get("runs", [])
    indexed = {str(row.get("run_id")): row for row in rows}
    if len(indexed) != len(rows):
        raise PipelineInvariantError("Protocol contract contains duplicate run IDs")
    return indexed


def _physical_row(
    run: Any,
    *,
    protocol: str,
    output_group: str,
    relative_output_dir: str,
    prediction_rows: int,
) -> dict[str, Any]:
    return {
        "artifact_id": f"{protocol}:{run.run_id}",
        "protocol": protocol,
        "output_group": output_group,
        "relative_output_dir": relative_output_dir,
        "expected_prediction_rows": int(prediction_rows),
        "run": run.to_dict(),
    }


def enumerate_analysis_inventory(
    config: AnalysisConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    audits = _validated_audits(config)
    artifacts: list[dict[str, Any]] = []
    references: list[dict[str, Any]] = []

    primary_rows = _audit_rows(audits["primary"])
    for run in enumerate_primary_runs(config.primary):
        row = primary_rows[run.run_id]
        artifacts.append(
            _physical_row(
                run,
                protocol="primary",
                output_group="primary",
                relative_output_dir=run.relative_output_dir.as_posix(),
                prediction_rows=row["prediction_rows"],
            )
        )

    cross_rows = _audit_rows(audits["cross_session"])
    for run in enumerate_cross_session_runs(config.cross_session):
        row = cross_rows[run.run_id]
        artifacts.append(
            _physical_row(
                run,
                protocol="cross_session",
                output_group="cross_session",
                relative_output_dir=run.relative_output_dir.as_posix(),
                prediction_rows=row["prediction_rows"],
            )
        )

    dann_rows = _audit_rows(audits["dann"])
    for run in enumerate_dann_runs(config.dann):
        row = dann_rows[run.run_id]
        artifacts.append(
            _physical_row(
                run,
                protocol="dann",
                output_group="dann",
                relative_output_dir=run.relative_output_dir.as_posix(),
                prediction_rows=row["prediction_rows"],
            )
        )

    for ablation_config, protocol, output_group in (
        (config.ablation_prefix, "ablation_prefix", "ablation_prefix"),
        (config.ablation_channels, "ablation_channels", "ablation_channels"),
    ):
        rows = _audit_rows(audits[protocol])
        for run in enumerate_ablation_runs(ablation_config):
            row = rows[run.run_id]
            if run.is_primary_reference:
                reference = primary_reference_run(ablation_config, run)
                references.append(
                    {
                        "logical_artifact_id": f"{protocol}:{run.run_id}",
                        "logical_protocol": protocol,
                        "logical_run": run.to_dict(),
                        "physical_artifact_id": f"primary:{reference.run_id}",
                        "physical_run_id": reference.run_id,
                        "expected_prediction_rows": int(row["prediction_rows"]),
                    }
                )
                continue
            artifacts.append(
                _physical_row(
                    run,
                    protocol=protocol,
                    output_group=output_group,
                    relative_output_dir=run.relative_output_dir.as_posix(),
                    prediction_rows=row["prediction_rows"],
                )
            )

    artifact_ids = [row["artifact_id"] for row in artifacts]
    reference_ids = [row["logical_artifact_id"] for row in references]
    physical_ids = set(artifact_ids)
    physical_by_id = {row["artifact_id"]: row for row in artifacts}
    target_counts = Counter(row["physical_artifact_id"] for row in references)
    reference_protocols = {
        target: {
            row["logical_protocol"]
            for row in references
            if row["physical_artifact_id"] == target
        }
        for target in target_counts
    }
    if (
        len(artifacts) != 265
        or len(references) != 20
        or len(set(artifact_ids)) != len(artifact_ids)
        or len(
            {
                (row["output_group"], row["relative_output_dir"])
                for row in artifacts
            }
        )
        != len(artifacts)
        or len(set(reference_ids)) != len(reference_ids)
        or any(row["physical_artifact_id"] not in physical_ids for row in references)
        or set(target_counts.values()) != {2}
        or any(
            protocols != {"ablation_prefix", "ablation_channels"}
            for protocols in reference_protocols.values()
        )
        or any(
            row["expected_prediction_rows"]
            != physical_by_id[row["physical_artifact_id"]]["expected_prediction_rows"]
            for row in references
        )
    ):
        raise PipelineInvariantError("Analysis physical/reference inventory differs")
    return artifacts, references


def _input_hashes(config: AnalysisConfig) -> dict[str, str]:
    audits = _validated_audits(config)
    return {
        "analysis_config": sha256_file(config.config_path),
        "primary_contract_audit": sha256_file(config.primary.contract_audit_path),
        "cross_session_contract_audit": sha256_file(
            config.cross_session.contract_audit_path
        ),
        "dann_contract_audit": sha256_file(config.dann.contract_audit_path),
        "ablation_prefix_contract_audit": sha256_file(
            config.ablation_prefix.contract_audit_path
        ),
        "ablation_channels_contract_audit": sha256_file(
            config.ablation_channels.contract_audit_path
        ),
        "canonical": audits["primary"]["input_hashes"]["canonical"],
        "split_manifest": audits["primary"]["input_hashes"]["split_manifest"],
    }


def _matrix_summary(
    artifacts: list[dict[str, Any]], references: list[dict[str, Any]]
) -> dict[str, Any]:
    protocol_counts = {
        protocol: sum(row["protocol"] == protocol for row in artifacts)
        for protocol in ANALYSIS_PROTOCOLS
    }
    return {
        "physical_artifacts": len(artifacts),
        "logical_references": len(references),
        "unique_referenced_artifacts": len(
            {row["physical_artifact_id"] for row in references}
        ),
        "physical_prediction_rows": sum(
            row["expected_prediction_rows"] for row in artifacts
        ),
        "logical_reference_prediction_rows": sum(
            row["expected_prediction_rows"] for row in references
        ),
        "protocol_physical_counts": protocol_counts,
    }


def build_analysis_contract(
    config: AnalysisConfig,
    *,
    force: bool = False,
) -> dict[str, Any]:
    if config.contract_audit_path.exists() and not force:
        raise FileExistsError("Refusing to overwrite analysis contract audit")
    artifacts, references = enumerate_analysis_inventory(config)
    payload = {
        "audit_version": 1,
        "created_utc": datetime.now(UTC).isoformat(),
        "package_version": __version__,
        "git": git_provenance(config.project_root),
        "input_hashes": _input_hashes(config),
        "policy": {
            "metrics": list(config.metrics),
            "primary_metrics": list(config.primary_metrics),
            "neural_seed_policy": config.neural_seed_policy,
            "seed_dispersion": config.seed_dispersion,
            "bootstrap": config.bootstrap,
        },
        "matrix": _matrix_summary(artifacts, references),
        "physical_artifacts": artifacts,
        "logical_references": references,
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


def validate_analysis_contract(config: AnalysisConfig) -> dict[str, Any]:
    if not config.contract_audit_path.is_file():
        raise PipelineInvariantError("Analysis contract audit is missing")
    audit = _load_json(config.contract_audit_path)
    artifacts, references = enumerate_analysis_inventory(config)
    expected_policy = {
        "metrics": list(config.metrics),
        "primary_metrics": list(config.primary_metrics),
        "neural_seed_policy": config.neural_seed_policy,
        "seed_dispersion": config.seed_dispersion,
        "bootstrap": config.bootstrap,
    }
    if (
        audit.get("audit_version") != 1
        or audit.get("package_version") != __version__
        or audit.get("status") != "valid"
        or audit.get("input_hashes") != _input_hashes(config)
        or audit.get("policy") != expected_policy
        or audit.get("matrix") != _matrix_summary(artifacts, references)
        or audit.get("physical_artifacts") != artifacts
        or audit.get("logical_references") != references
    ):
        raise PipelineInvariantError("Analysis contract audit is stale")
    return {
        "status": "valid",
        "matrix": audit["matrix"],
        "policy": expected_policy,
    }
