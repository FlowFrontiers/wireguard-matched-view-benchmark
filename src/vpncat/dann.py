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
from vpncat.experiment import PrimaryExperimentConfig, load_primary_experiment_config
from vpncat.folds import materialize_fold_index
from vpncat.hashing import sha256_file
from vpncat.neural_config import NeuralConfig, load_neural_config
from vpncat.preprocessing import pair_id_digest
from vpncat.provenance import git_provenance

DANN_HISTORY_COLUMNS = (
    "epoch",
    "train_classification_loss",
    "train_domain_loss",
    "train_total_loss",
    "validation_loss",
    "validation_macro_f1",
    "learning_rate",
    "grl_coefficient_start",
    "grl_coefficient_end",
)


@dataclass(frozen=True)
class DANNRunSpec:
    protocol: str
    experiment_id: str
    representation: str
    model: str
    family: str
    backbone: str
    fold: int
    seed: int
    source_domain: str
    adaptation_domain: str
    test_domains: tuple[str, ...]

    @property
    def train_domain(self) -> str:
        """Expose the source domain under the shared prediction schema name."""
        return self.source_domain

    @property
    def run_id(self) -> str:
        return (
            f"{self.protocol}__{self.experiment_id}__fold_{self.fold:02d}__"
            f"source_{self.source_domain}__adapt_{self.adaptation_domain}__"
            f"seed_{self.seed:06d}"
        )

    @property
    def relative_output_dir(self) -> Path:
        return Path(
            self.model,
            f"fold_{self.fold:02d}",
            f"source_{self.source_domain}_adapt_{self.adaptation_domain}",
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
            "backbone": self.backbone,
            "fold": self.fold,
            "seed": self.seed,
            "source_domain": self.source_domain,
            "adaptation_domain": self.adaptation_domain,
            "test_domains": list(self.test_domains),
        }


@dataclass(frozen=True)
class DANNConfig:
    config_path: Path
    project_root: Path
    primary: PrimaryExperimentConfig
    neural: NeuralConfig
    contract_audit_path: Path
    output_root: Path
    representation: str
    model: str
    backbone: str
    folds: tuple[int, ...]
    seeds: tuple[int, ...]
    prefix_length: int
    channels: tuple[str, ...]
    source_domain: str
    adaptation_domain: str
    test_domains: tuple[str, ...]
    domain_loss_weight: float
    gradient_reversal: dict[str, Any]
    domain_head: dict[str, Any]


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def load_dann_config(
    path: Path,
    *,
    artifact_dir: Path | None = None,
    output_root: Path | None = None,
    tuning_output_root: Path | None = None,
    contract_audit_path: Path | None = None,
) -> DANNConfig:
    path = path.expanduser().resolve()
    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle).get("dann", {})
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
    audit_path = (
        _resolve(root, contract_audit_path)
        if contract_audit_path is not None
        else (
            resolved_artifacts / "dann_contract_audit.json"
            if resolved_artifacts is not None
            else _resolve(root, raw.get("contract_audit_path", ""))
        )
    )
    config = DANNConfig(
        config_path=path,
        project_root=root,
        primary=primary,
        neural=neural,
        contract_audit_path=audit_path,
        output_root=_resolve(
            root,
            output_root if output_root is not None else raw.get("output_root", ""),
        ),
        representation=str(raw.get("representation", "")),
        model=str(raw.get("model", "")),
        backbone=str(raw.get("backbone", "")),
        folds=tuple(int(value) for value in raw.get("folds", [])),
        seeds=tuple(int(value) for value in raw.get("seeds", [])),
        prefix_length=int(raw.get("prefix_length", 0)),
        channels=tuple(str(value) for value in raw.get("channels", [])),
        source_domain=str(raw.get("source_domain", "")),
        adaptation_domain=str(raw.get("adaptation_domain", "")),
        test_domains=tuple(str(value) for value in raw.get("test_domains", [])),
        domain_loss_weight=float(raw.get("domain_loss_weight", -1)),
        gradient_reversal=dict(raw.get("gradient_reversal", {})),
        domain_head=dict(raw.get("domain_head", {})),
    )
    _validate_dann_config(config, raw)
    return config


def _validate_dann_config(config: DANNConfig, raw: dict[str, Any]) -> None:
    if raw.get("protocol") != "dann":
        raise PipelineInvariantError("DANN protocol name is invalid")
    if (
        config.representation != "sequential_splt"
        or config.model != "dann_cnn1d"
        or config.backbone != "cnn1d"
    ):
        raise PipelineInvariantError("DANN representation or model differs from freeze")
    if config.folds != (1, 2, 3, 4, 5) or config.seeds != (42, 43, 44):
        raise PipelineInvariantError("DANN folds or seeds differ from freeze")
    if config.prefix_length != 50 or config.channels != config.neural.channels:
        raise PipelineInvariantError("DANN observations differ from primary CNN1D")
    if (
        config.source_domain != "inner"
        or config.adaptation_domain != "outer"
        or config.test_domains != ("inner", "outer")
    ):
        raise PipelineInvariantError("DANN domains differ from freeze")
    expected_literals = {
        "adaptation_pairs": "source_training_pairs_only",
        "adaptation_labels": "prohibited",
        "validation_domain": "inner",
        "validation_role": "validation",
        "source_loss": "weighted_cross_entropy",
        "domain_loss": "binary_cross_entropy_logits",
        "domain_batching": "paired_by_pair_id",
        "augmentation": False,
    }
    if any(raw.get(key) != value for key, value in expected_literals.items()):
        raise PipelineInvariantError("DANN leakage or optimization policy differs")
    if config.domain_loss_weight != 1.0:
        raise PipelineInvariantError("DANN domain loss weight differs from freeze")
    if config.gradient_reversal != {
        "schedule": "logistic",
        "gamma": 10.0,
        "start": 0.0,
        "end": 1.0,
    }:
        raise PipelineInvariantError("DANN gradient-reversal schedule differs")
    if config.domain_head != {
        "hidden_layers": 1,
        "hidden_width": "backbone_width",
        "activation": "gelu",
        "dropout": "selected_backbone_dropout",
    }:
        raise PipelineInvariantError("DANN domain head differs from freeze")
    if config.folds != config.primary.folds:
        raise PipelineInvariantError("DANN and primary folds differ")


def enumerate_dann_runs(config: DANNConfig) -> tuple[DANNRunSpec, ...]:
    runs = tuple(
        DANNRunSpec(
            protocol="dann",
            experiment_id="sequential_splt__dann_cnn1d",
            representation=config.representation,
            model=config.model,
            family="neural",
            backbone=config.backbone,
            fold=fold,
            seed=seed,
            source_domain=config.source_domain,
            adaptation_domain=config.adaptation_domain,
            test_domains=config.test_domains,
        )
        for fold in config.folds
        for seed in config.seeds
    )
    if len(runs) != 15 or len({run.run_id for run in runs}) != 15:
        raise PipelineInvariantError("DANN matrix must contain 15 unique runs")
    if len({run.relative_output_dir for run in runs}) != 15:
        raise PipelineInvariantError("DANN output paths are not unique")
    return runs


def _input_hashes(config: DANNConfig) -> dict[str, str]:
    primary_hashes = verify_input_chain(config.primary)
    primary_contract = json.loads(
        config.primary.contract_audit_path.read_text(encoding="utf-8")
    )
    if (
        primary_contract.get("status") != "valid"
        or primary_contract.get("input_hashes") != primary_hashes
    ):
        raise PipelineInvariantError("DANN requires a valid primary contract audit")
    return {
        **primary_hashes,
        "dann_config": sha256_file(config.config_path),
        "neural_config": sha256_file(config.neural.config_path),
        "primary_contract_audit": sha256_file(config.primary.contract_audit_path),
    }


def _fold_rows(config: DANNConfig) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metadata = pq.read_table(
        config.primary.canonical_path,
        columns=["pair_id", "session", "application_category"],
    ).to_pandas()
    split = pd.read_csv(config.primary.split_path)
    summaries: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for fold_number in config.folds:
        fold = materialize_fold_index(metadata, split, fold=fold_number)
        role_counts = {
            role: len(fold.positions(role))
            for role in ("train", "validation", "test")
        }
        role_hashes = {
            role: pair_id_digest(fold.pair_ids_for(role))
            for role in ("train", "validation", "test")
        }
        classes = sorted(set(fold.labels[position] for position in fold.train_positions))
        if any(
            set(fold.labels[position] for position in fold.positions(role))
            != set(classes)
            for role in ("validation", "test")
        ):
            raise PipelineInvariantError("DANN fold role omits one or more classes")
        summaries[str(fold_number)] = {
            "role_counts": role_counts,
            "role_pair_ids_sha256": role_hashes,
            "classes": classes,
            "adaptation_pair_count": role_counts["train"],
            "adaptation_pair_ids_sha256": role_hashes["train"],
            "adaptation_labels_exposed": False,
        }
    for run in enumerate_dann_runs(config):
        summary = summaries[str(run.fold)]
        rows.append(
            {
                **run.to_dict(),
                "relative_output_dir": run.relative_output_dir.as_posix(),
                "training_pair_count": summary["role_counts"]["train"],
                "validation_pair_count": summary["role_counts"]["validation"],
                "test_pair_count": summary["role_counts"]["test"],
                "training_pair_ids_sha256": summary["role_pair_ids_sha256"]["train"],
                "validation_pair_ids_sha256": summary["role_pair_ids_sha256"][
                    "validation"
                ],
                "test_pair_ids_sha256": summary["role_pair_ids_sha256"]["test"],
                "adaptation_pair_ids_sha256": summary[
                    "adaptation_pair_ids_sha256"
                ],
                "prediction_rows": summary["role_counts"]["test"]
                * len(run.test_domains),
            }
        )
    return summaries, rows


def _protocol_summary(config: DANNConfig) -> dict[str, Any]:
    return {
        "source_domain": config.source_domain,
        "adaptation_domain": config.adaptation_domain,
        "adaptation_pairs": "source_training_pairs_only",
        "adaptation_labels_exposed": False,
        "validation_domain": config.source_domain,
        "test_domains": list(config.test_domains),
        "gradient_reversal": config.gradient_reversal,
        "domain_loss_weight": config.domain_loss_weight,
        "augmentation": False,
    }


def build_dann_contract(config: DANNConfig, *, force: bool = False) -> dict[str, Any]:
    if config.contract_audit_path.exists() and not force:
        raise FileExistsError("Refusing to overwrite DANN contract audit")
    input_hashes = _input_hashes(config)
    folds, runs = _fold_rows(config)
    payload = {
        "audit_version": 1,
        "created_utc": datetime.now(UTC).isoformat(),
        "package_version": __version__,
        "git": git_provenance(config.project_root),
        "input_hashes": input_hashes,
        "protocol": _protocol_summary(config),
        "folds": folds,
        "matrix": {"training_runs": 15, "prediction_groups": 30},
        "runs": runs,
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


def validate_dann_contract(config: DANNConfig) -> dict[str, Any]:
    if not config.contract_audit_path.is_file():
        raise PipelineInvariantError("DANN contract audit is missing")
    audit = json.loads(config.contract_audit_path.read_text(encoding="utf-8"))
    folds, runs = _fold_rows(config)
    if (
        audit.get("audit_version") != 1
        or audit.get("status") != "valid"
        or audit.get("package_version") != __version__
        or audit.get("input_hashes") != _input_hashes(config)
        or audit.get("protocol") != _protocol_summary(config)
        or audit.get("folds") != folds
        or audit.get("matrix") != {"training_runs": 15, "prediction_groups": 30}
        or audit.get("runs") != runs
    ):
        raise PipelineInvariantError("DANN contract audit is stale")
    return {
        "status": "valid",
        "training_runs": len(runs),
        "folds": folds,
    }
