from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from vpncat.errors import PipelineInvariantError

FROZEN_TOPOLOGIES: dict[str, dict[str, Any]] = {
    "cnn1d": {
        "kernels": [3, 7, 11],
        "residual_blocks": 1,
        "residual_kernel": 3,
        "activation": "gelu",
        "pooling": "masked_mean",
    },
    "lstm": {
        "layers": 2,
        "bidirectional": False,
        "pooling": "masked_mean",
    },
    "transformer": {
        "layers": 2,
        "attention_heads": 4,
        "feedforward_multiplier": 4,
        "activation": "gelu",
        "norm_first": True,
        "pooling": "masked_mean",
    },
}
FROZEN_OPTIMIZER = {
    "name": "adamw",
    "betas": [0.9, 0.999],
    "epsilon": 0.00000001,
    "amsgrad": False,
    "weight_decay": 0.0001,
    "gradient_clip_norm": 1.0,
}
FROZEN_TRAINING = {
    "maximum_epochs": 60,
    "early_stopping_metric": "macro_f1",
    "early_stopping_patience": 6,
    "early_stopping_min_delta": 0.0,
    "scheduler": "reduce_lr_on_plateau",
    "scheduler_metric": "validation_loss",
    "scheduler_factor": 0.5,
    "scheduler_patience": 3,
    "scheduler_threshold": 0.0001,
    "scheduler_threshold_mode": "rel",
    "scheduler_cooldown": 0,
    "scheduler_epsilon": 0.00000001,
    "minimum_learning_rate": 0.000001,
    "deterministic_algorithms": True,
    "mixed_precision": False,
    "training_shuffle": True,
    "drop_last": False,
    "data_loader_workers": 0,
}


@dataclass(frozen=True)
class NeuralTrial:
    trial_id: int
    learning_rate: float
    batch_size: int
    dropout: float
    width: int

    def to_dict(self) -> dict[str, int | float]:
        return {
            "id": self.trial_id,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "dropout": self.dropout,
            "width": self.width,
        }


@dataclass(frozen=True)
class NeuralConfig:
    config_path: Path
    project_root: Path
    primary_prefix_length: int
    maximum_prefix_length: int
    channels: tuple[str, ...]
    optimizer: dict[str, Any]
    training: dict[str, Any]
    development_fold: int
    development_train_domain: str
    development_seed: int
    selection_metric: str
    maximum_trials: int
    tuning_output_root: Path
    topologies: dict[str, dict[str, Any]]
    trials: tuple[NeuralTrial, ...]


def _resolve(project_root: Path, value: str | Path) -> Path:
    candidate = Path(value).expanduser()
    return candidate.resolve() if candidate.is_absolute() else (project_root / candidate).resolve()


def load_neural_config(
    path: Path,
    *,
    tuning_output_root: Path | None = None,
) -> NeuralConfig:
    path = path.expanduser().resolve()
    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle).get("neural", {})
    development = raw.get("development", {})
    project_root = path.parent.parent
    config = NeuralConfig(
        config_path=path,
        project_root=project_root,
        primary_prefix_length=int(raw.get("primary_prefix_length", 0)),
        maximum_prefix_length=int(raw.get("maximum_prefix_length", 0)),
        channels=tuple(str(channel) for channel in raw.get("channels", [])),
        optimizer=dict(raw.get("optimizer", {})),
        training=dict(raw.get("training", {})),
        development_fold=int(development.get("fold", 0)),
        development_train_domain=str(development.get("train_domain", "")),
        development_seed=int(development.get("seed", -1)),
        selection_metric=str(development.get("selection_metric", "")),
        maximum_trials=int(development.get("maximum_trials", 0)),
        tuning_output_root=_resolve(
            project_root,
            tuning_output_root
            if tuning_output_root is not None
            else development.get("output_root", ""),
        ),
        topologies={
            str(model): dict(topology)
            for model, topology in raw.get("topologies", {}).items()
        },
        trials=tuple(
            NeuralTrial(
                trial_id=int(item["id"]),
                learning_rate=float(item["learning_rate"]),
                batch_size=int(item["batch_size"]),
                dropout=float(item["dropout"]),
                width=int(item["width"]),
            )
            for item in raw.get("trials", [])
        ),
    )
    _validate_neural_config(config, raw=raw)
    return config


def _validate_neural_config(config: NeuralConfig, *, raw: dict[str, Any]) -> None:
    if raw.get("framework") != "pytorch":
        raise PipelineInvariantError("Neural experiments require framework=pytorch")
    if config.primary_prefix_length != 50 or config.maximum_prefix_length != 80:
        raise PipelineInvariantError("Neural sequence horizons must be primary=50 and maximum=80")
    if config.channels != ("direction", "size", "iat_ms"):
        raise PipelineInvariantError("Neural primary runs require all three ordered SPLT channels")
    if raw.get("loss") != "weighted_cross_entropy":
        raise PipelineInvariantError("Neural runs require weighted cross-entropy")
    if config.optimizer != FROZEN_OPTIMIZER or config.training != FROZEN_TRAINING:
        raise PipelineInvariantError("Neural optimizer or training policy differs from the freeze")
    if config.topologies != FROZEN_TOPOLOGIES:
        raise PipelineInvariantError("Neural topology definitions differ from the freeze")
    if (
        config.development_fold != 1
        or config.development_train_domain != "inner"
        or config.development_seed != 42
        or config.selection_metric != "macro_f1"
        or config.maximum_trials != 12
    ):
        raise PipelineInvariantError("Neural development protocol differs from the freeze")
    if len(config.trials) != config.maximum_trials:
        raise PipelineInvariantError("Neural search must contain exactly 12 trials")
    if tuple(trial.trial_id for trial in config.trials) != tuple(range(1, 13)):
        raise PipelineInvariantError("Neural trial IDs must be consecutive from 1 through 12")
    trial_parameters = {
        (trial.learning_rate, trial.batch_size, trial.dropout, trial.width)
        for trial in config.trials
    }
    if len(trial_parameters) != 12:
        raise PipelineInvariantError("Neural search contains duplicate trials")
    for trial in config.trials:
        if trial.learning_rate <= 0 or trial.batch_size <= 0:
            raise PipelineInvariantError("Neural learning rates and batch sizes must be positive")
        if not 0 <= trial.dropout < 1 or trial.width <= 0 or trial.width % 4:
            raise PipelineInvariantError("Neural dropout or width is invalid")
