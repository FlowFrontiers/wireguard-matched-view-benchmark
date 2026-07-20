from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from vpncat.artifacts import write_completed_run
from vpncat.errors import PipelineInvariantError
from vpncat.experiment import PrimaryExperimentConfig, RunSpec
from vpncat.hashing import sha256_file
from vpncat.models.neural import build_neural_model, trainable_parameter_count
from vpncat.neural_config import NeuralConfig
from vpncat.neural_data import PreparedNeuralRun, prepare_neural_run
from vpncat.neural_training import (
    predict_neural_probabilities,
    seed_neural_execution,
    train_neural_model,
)
from vpncat.neural_tuning import (
    SelectedNeuralConfiguration,
    load_selected_neural_configuration,
)
from vpncat.primary_runner import validate_contract_audit


def build_neural_prediction_frame(
    run: RunSpec,
    prepared: PreparedNeuralRun,
    probabilities: dict[str, np.ndarray],
) -> pd.DataFrame:
    classes = np.asarray(prepared.state.classes, dtype=object)
    rows: list[dict[str, Any]] = []
    for domain in run.test_domains:
        subset = prepared.tests[domain]
        domain_probabilities = np.asarray(probabilities[domain], dtype=np.float64)
        if domain_probabilities.shape != (len(subset.pair_ids), len(classes)):
            raise PipelineInvariantError(f"{domain} neural probability shape is invalid")
        predictions = classes[np.argmax(domain_probabilities, axis=1)]
        for offset, position in enumerate(subset.positions):
            rows.append(
                {
                    "run_id": run.run_id,
                    "protocol": run.protocol,
                    "representation": run.representation,
                    "model": run.model,
                    "pair_id": subset.pair_ids[offset],
                    "session": int(prepared.fold.sessions[position]),
                    "train_domain": run.train_domain,
                    "test_domain": domain,
                    "fold": run.fold,
                    "seed": run.seed,
                    "true_label": prepared.fold.labels[position],
                    "prediction": str(predictions[offset]),
                    "class_probabilities": domain_probabilities[offset].tolist(),
                }
            )
    return pd.DataFrame(rows)


def run_primary_neural(
    primary: PrimaryExperimentConfig,
    neural: NeuralConfig,
    run: RunSpec,
    *,
    device_name: str = "auto",
    selected: SelectedNeuralConfiguration | None = None,
) -> Path:
    """Train one selected neural configuration and publish both held-out views."""
    if (
        run.family != "neural"
        or run.representation != "sequential_splt"
        or run.model not in neural.topologies
    ):
        raise PipelineInvariantError("Neural runner received an incompatible primary run")
    validate_contract_audit(primary, run)
    if selected is None:
        selected = load_selected_neural_configuration(
            primary,
            neural,
            model_name=run.model,
        )
    elif selected.model != run.model:
        raise PipelineInvariantError("Selected neural configuration belongs to another model")
    prepared = prepare_neural_run(
        primary.canonical_path,
        primary.split_path,
        run,
        prefix_length=neural.primary_prefix_length,
        channels=neural.channels,
    )
    seed_neural_execution(run.seed)
    model = build_neural_model(
        run.model,
        feature_count=len(neural.channels),
        class_count=len(prepared.state.classes),
        width=selected.trial.width,
        dropout=selected.trial.dropout,
        maximum_length=neural.maximum_prefix_length,
        topology=neural.topologies[run.model],
    )
    expected_parameter_count = int(selected.result["parameter_count"])
    if trainable_parameter_count(model) != expected_parameter_count:
        raise PipelineInvariantError("Primary neural model differs from tuned parameter count")
    result = train_neural_model(
        model,
        prepared.training,
        prepared.validation,
        prepared.state,
        learning_rate=selected.trial.learning_rate,
        batch_size=selected.trial.batch_size,
        seed=run.seed,
        optimizer_policy=neural.optimizer,
        training_policy=neural.training,
        device_name=device_name,
    )
    probabilities = {
        domain: predict_neural_probabilities(
            result.model,
            prepared.tests[domain],
            batch_size=selected.trial.batch_size,
            class_count=len(prepared.state.classes),
            device_name=device_name,
        )
        for domain in run.test_domains
    }
    predictions = build_neural_prediction_frame(run, prepared, probabilities)
    model_hyperparameters = {
        "selected_trial": selected.trial.to_dict(),
        "topology": neural.topologies[run.model],
        "optimizer": neural.optimizer,
        "training_policy": neural.training,
        "parameter_count": result.parameter_count,
        "training_outcome": {
            "best_epoch": result.best_epoch,
            "best_validation_macro_f1": result.best_validation_macro_f1,
            "validation_loss_at_best_epoch": result.validation_loss_at_best_epoch,
            "epochs_completed": result.epochs_completed,
            "device": result.device,
        },
        "selection": {
            "metric": neural.selection_metric,
            "development_fold": neural.development_fold,
            "development_train_domain": neural.development_train_domain,
            "tuning_revision": selected.tuning_revision,
            "tuning_environment": selected.tuning_environment,
            "tuning_device": selected.tuning_device,
            "selected_result": selected.result,
        },
    }
    return write_completed_run(
        primary,
        run,
        prepared.fold,
        prepared.state,
        predictions,
        model_hyperparameters=model_hyperparameters,
        training_history=result.history,
        additional_input_hashes={
            "neural_config": sha256_file(neural.config_path),
            "neural_tuning_selection": selected.selected_sha256,
            "neural_tuning_manifest": selected.tuning_manifest_sha256,
        },
    )
