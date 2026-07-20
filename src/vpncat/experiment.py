from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from vpncat.errors import PipelineInvariantError

DOMAINS = ("inner", "outer")
PRIMARY_CONFIGURATIONS = frozenset(
    {
        ("matched_flow_stats", "random_forest"),
        ("matched_flow_stats", "xgboost"),
        ("prefix_stats", "random_forest"),
        ("prefix_stats", "xgboost"),
        ("flattened_splt", "random_forest"),
        ("flattened_splt", "xgboost"),
        ("sequential_splt", "cnn1d"),
        ("sequential_splt", "lstm"),
        ("sequential_splt", "transformer"),
    }
)
MODEL_FAMILIES = {
    "random_forest": "classical",
    "xgboost": "classical",
    "cnn1d": "neural",
    "lstm": "neural",
    "transformer": "neural",
}
FROZEN_CLASSICAL_HYPERPARAMETERS: dict[str, dict[str, Any]] = {
    "random_forest": {
        "n_estimators": 500,
        "criterion": "gini",
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "bootstrap": True,
        "class_weight": "balanced",
        "n_jobs": -1,
        "random_state": "run_seed",
    },
    "xgboost": {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 1,
        "gamma": 0.0,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "objective": "multi:softprob",
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "n_jobs": -1,
        "random_state": "run_seed",
        "verbosity": 0,
        "sample_weight": "balanced",
    },
}


@dataclass(frozen=True)
class ModelConfiguration:
    experiment_id: str
    representation: str
    model: str
    family: str
    seeds: tuple[int, ...]


@dataclass(frozen=True)
class PrimaryExperimentConfig:
    config_path: Path
    project_root: Path
    canonical_path: Path
    split_path: Path
    dataset_manifest_path: Path
    feature_audit_path: Path
    preprocessing_audit_path: Path
    contract_audit_path: Path
    output_root: Path
    folds: tuple[int, ...]
    prefix_length: int
    train_domains: tuple[str, ...]
    test_domains: tuple[str, ...]
    model_hyperparameters: dict[str, dict[str, Any]]
    configurations: tuple[ModelConfiguration, ...]


@dataclass(frozen=True)
class RunSpec:
    protocol: str
    experiment_id: str
    representation: str
    model: str
    family: str
    fold: int
    seed: int
    train_domain: str
    test_domains: tuple[str, ...]

    @property
    def run_id(self) -> str:
        return (
            f"{self.protocol}__{self.experiment_id}__fold_{self.fold:02d}__"
            f"train_{self.train_domain}__seed_{self.seed:06d}"
        )

    @property
    def relative_output_dir(self) -> Path:
        return Path(
            self.representation,
            self.model,
            f"fold_{self.fold:02d}",
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
            "fold": self.fold,
            "seed": self.seed,
            "train_domain": self.train_domain,
            "test_domains": list(self.test_domains),
        }


def _resolve(project_root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def load_primary_experiment_config(
    config_path: Path,
    *,
    artifact_dir: Path | None = None,
    output_root: Path | None = None,
    contract_audit_path: Path | None = None,
) -> PrimaryExperimentConfig:
    config_path = config_path.expanduser().resolve()
    with config_path.open(encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle)
    project_root = config_path.parent.parent
    experiment = raw.get("experiment", {})
    if experiment.get("protocol") != "primary":
        raise PipelineInvariantError("Primary configuration must use protocol=primary")
    required_paths = (
        "canonical_path",
        "split_path",
        "dataset_manifest_path",
        "feature_audit_path",
        "preprocessing_audit_path",
        "contract_audit_path",
        "output_root",
    )
    missing_paths = [name for name in required_paths if experiment.get(name) is None]
    if missing_paths:
        raise PipelineInvariantError(
            f"Primary configuration is missing paths: {missing_paths}"
        )
    configurations = tuple(
        ModelConfiguration(
            experiment_id=str(item["id"]),
            representation=str(item["representation"]),
            model=str(item["model"]),
            family=str(item["family"]),
            seeds=tuple(int(seed) for seed in item["seeds"]),
        )
        for item in experiment.get("configurations", [])
    )
    resolved_artifact_dir = (
        _resolve(project_root, artifact_dir) if artifact_dir is not None else None
    )

    def artifact_path(config_key: str, filename: str) -> Path:
        if resolved_artifact_dir is not None:
            return resolved_artifact_dir / filename
        return _resolve(project_root, experiment[config_key])

    config = PrimaryExperimentConfig(
        config_path=config_path,
        project_root=project_root,
        canonical_path=artifact_path("canonical_path", "canonical_pairs.parquet"),
        split_path=artifact_path("split_path", "split_manifest.csv"),
        dataset_manifest_path=artifact_path(
            "dataset_manifest_path", "dataset_manifest.json"
        ),
        feature_audit_path=artifact_path("feature_audit_path", "feature_audit.json"),
        preprocessing_audit_path=artifact_path(
            "preprocessing_audit_path", "preprocessing_audit.json"
        ),
        contract_audit_path=(
            _resolve(project_root, contract_audit_path)
            if contract_audit_path is not None
            else artifact_path(
                "contract_audit_path", "experiment_contract_audit.json"
            )
        ),
        output_root=_resolve(
            project_root,
            output_root if output_root is not None else experiment["output_root"],
        ),
        folds=tuple(int(fold) for fold in experiment.get("folds", [])),
        prefix_length=int(experiment.get("prefix_length", 0)),
        train_domains=tuple(str(value) for value in experiment.get("train_domains", [])),
        test_domains=tuple(str(value) for value in experiment.get("test_domains", [])),
        model_hyperparameters={
            str(model): dict(parameters)
            for model, parameters in experiment.get("model_hyperparameters", {}).items()
        },
        configurations=configurations,
    )
    _validate_primary_config(config, augmentation=experiment.get("augmentation"))
    return config


def _validate_primary_config(
    config: PrimaryExperimentConfig,
    *,
    augmentation: object,
) -> None:
    if config.folds != (1, 2, 3, 4, 5):
        raise PipelineInvariantError("Primary experiment requires folds [1, 2, 3, 4, 5]")
    if config.prefix_length != 50:
        raise PipelineInvariantError("Primary experiment requires prefix_length=50")
    if config.train_domains != DOMAINS or config.test_domains != DOMAINS:
        raise PipelineInvariantError("Primary experiment requires ordered domains [inner, outer]")
    if augmentation is not False:
        raise PipelineInvariantError("Data augmentation is prohibited in the primary matrix")
    for model, expected in FROZEN_CLASSICAL_HYPERPARAMETERS.items():
        if config.model_hyperparameters.get(model) != expected:
            raise PipelineInvariantError(
                f"{model} hyperparameters differ from the frozen inherited configuration"
            )
    observed = {(item.representation, item.model) for item in config.configurations}
    if observed != PRIMARY_CONFIGURATIONS or len(config.configurations) != len(
        PRIMARY_CONFIGURATIONS
    ):
        missing = sorted(PRIMARY_CONFIGURATIONS - observed)
        extra = sorted(observed - PRIMARY_CONFIGURATIONS)
        raise PipelineInvariantError(
            f"Primary matrix differs from the frozen nine configurations: "
            f"missing={missing}, extra={extra}"
        )
    ids = [item.experiment_id for item in config.configurations]
    if len(set(ids)) != len(ids):
        raise PipelineInvariantError("Primary configuration IDs must be unique")
    for item in config.configurations:
        expected_family = MODEL_FAMILIES[item.model]
        if item.family != expected_family:
            raise PipelineInvariantError(
                f"Model family mismatch for {item.model}: {item.family}"
            )
        expected_seed_count = 1 if item.family == "classical" else 3
        if len(item.seeds) != expected_seed_count or len(set(item.seeds)) != len(item.seeds):
            raise PipelineInvariantError(
                f"{item.experiment_id} requires {expected_seed_count} unique seeds"
            )
        if any(seed < 0 for seed in item.seeds):
            raise PipelineInvariantError("Random seeds must be nonnegative")


def enumerate_primary_runs(config: PrimaryExperimentConfig) -> tuple[RunSpec, ...]:
    runs = tuple(
        RunSpec(
            protocol="primary",
            experiment_id=item.experiment_id,
            representation=item.representation,
            model=item.model,
            family=item.family,
            fold=fold,
            seed=seed,
            train_domain=train_domain,
            test_domains=config.test_domains,
        )
        for item in config.configurations
        for fold in config.folds
        for train_domain in config.train_domains
        for seed in item.seeds
    )
    run_ids = [run.run_id for run in runs]
    output_dirs = [run.relative_output_dir for run in runs]
    if len(runs) != 150:
        raise PipelineInvariantError(f"Primary matrix must contain 150 runs, observed {len(runs)}")
    if len(set(run_ids)) != len(runs) or len(set(output_dirs)) != len(runs):
        raise PipelineInvariantError("Primary matrix contains duplicate run identities")
    return runs


def select_primary_run(
    config: PrimaryExperimentConfig,
    *,
    experiment_id: str,
    fold: int,
    train_domain: str,
    seed: int,
) -> RunSpec:
    matches = [
        run
        for run in enumerate_primary_runs(config)
        if run.experiment_id == experiment_id
        and run.fold == fold
        and run.train_domain == train_domain
        and run.seed == seed
    ]
    if len(matches) != 1:
        raise PipelineInvariantError(
            "Requested run is not a unique member of the frozen primary matrix"
        )
    return matches[0]
