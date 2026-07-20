from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from vpncat.dann import DANNConfig, DANNRunSpec
from vpncat.dann_artifacts import (
    validate_dann_run_contract,
    write_completed_dann_run,
)
from vpncat.dann_data import PreparedDANNRun, prepare_dann_run
from vpncat.dann_training import predict_dann_probabilities, train_dann_model
from vpncat.errors import PipelineInvariantError
from vpncat.models.dann import build_dann_model
from vpncat.models.neural import trainable_parameter_count
from vpncat.neural_training import seed_neural_execution
from vpncat.neural_tuning import (
    SelectedNeuralConfiguration,
    load_selected_neural_configuration,
)
from vpncat.provenance import git_provenance


def build_dann_prediction_frame(
    run: DANNRunSpec,
    prepared: PreparedDANNRun,
    probabilities: dict[str, np.ndarray],
) -> pd.DataFrame:
    classes = np.asarray(prepared.state.classes, dtype=object)
    frames: list[pd.DataFrame] = []
    for domain in run.test_domains:
        subset = prepared.tests[domain]
        domain_probabilities = np.asarray(probabilities[domain], dtype=np.float64)
        if domain_probabilities.shape != (len(subset.pair_ids), len(classes)):
            raise PipelineInvariantError(f"DANN {domain} probability shape is invalid")
        predictions = classes[np.argmax(domain_probabilities, axis=1)]
        frames.append(
            pd.DataFrame(
                {
                    "run_id": run.run_id,
                    "protocol": run.protocol,
                    "representation": run.representation,
                    "model": run.model,
                    "pair_id": subset.pair_ids,
                    "session": prepared.fold.sessions[subset.positions].astype(int),
                    "train_domain": run.source_domain,
                    "test_domain": domain,
                    "fold": run.fold,
                    "seed": run.seed,
                    "true_label": [
                        prepared.fold.labels[position] for position in subset.positions
                    ],
                    "prediction": predictions.astype(str),
                    "class_probabilities": domain_probabilities.tolist(),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def run_dann(
    config: DANNConfig,
    run: DANNRunSpec,
    *,
    device_name: str = "auto",
    selected: SelectedNeuralConfiguration | None = None,
) -> Path:
    """Train one frozen DANN configuration and atomically publish both views."""
    if (
        run.protocol != "dann"
        or run.model != config.model
        or run.backbone != "cnn1d"
        or run.source_domain != config.source_domain
        or run.adaptation_domain != config.adaptation_domain
    ):
        raise PipelineInvariantError("DANN runner received an incompatible run")
    target = config.output_root / run.relative_output_dir
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite existing DANN run: {target}")
    provenance = git_provenance(config.project_root)
    if not provenance.get("status_available") or provenance.get("dirty"):
        raise PipelineInvariantError("DANN execution requires a clean Git revision")
    base_hashes = validate_dann_run_contract(config, run)
    if selected is None:
        selected = load_selected_neural_configuration(
            config.primary,
            config.neural,
            model_name="cnn1d",
        )
    elif selected.model != "cnn1d":
        raise PipelineInvariantError("DANN requires the selected CNN1D configuration")
    input_hashes = {
        **base_hashes,
        "neural_tuning_selection": selected.selected_sha256,
        "neural_tuning_manifest": selected.tuning_manifest_sha256,
    }
    prepared = prepare_dann_run(config, run)
    seed_neural_execution(run.seed)
    model = build_dann_model(
        feature_count=len(config.channels),
        class_count=len(prepared.state.classes),
        width=selected.trial.width,
        dropout=selected.trial.dropout,
        maximum_length=config.neural.maximum_prefix_length,
        topology=config.neural.topologies["cnn1d"],
        domain_head=config.domain_head,
    )
    expected_backbone_parameters = int(selected.result["parameter_count"])
    if trainable_parameter_count(model.backbone) != expected_backbone_parameters:
        raise PipelineInvariantError("DANN backbone differs from tuned CNN1D")
    result = train_dann_model(
        model,
        prepared.source_training,
        prepared.adaptation_training,
        prepared.source_validation,
        prepared.state,
        learning_rate=selected.trial.learning_rate,
        batch_size=selected.trial.batch_size,
        seed=run.seed,
        domain_loss_weight=config.domain_loss_weight,
        gradient_reversal=config.gradient_reversal,
        optimizer_policy=config.neural.optimizer,
        training_policy=config.neural.training,
        device_name=device_name,
    )
    probabilities = {
        domain: predict_dann_probabilities(
            result.model,
            prepared.tests[domain],
            batch_size=selected.trial.batch_size,
            class_count=len(prepared.state.classes),
            device_name=device_name,
        )
        for domain in run.test_domains
    }
    predictions = build_dann_prediction_frame(run, prepared, probabilities)
    model_hyperparameters = {
        "selected_trial": selected.trial.to_dict(),
        "backbone_topology": config.neural.topologies["cnn1d"],
        "domain_head": config.domain_head,
        "optimizer": config.neural.optimizer,
        "training_policy": config.neural.training,
        "domain_loss_weight": config.domain_loss_weight,
        "gradient_reversal": config.gradient_reversal,
        "parameter_count": result.parameter_count,
        "backbone_parameter_count": result.backbone_parameter_count,
        "training_outcome": {
            "best_epoch": result.best_epoch,
            "best_validation_macro_f1": result.best_validation_macro_f1,
            "validation_loss_at_best_epoch": result.validation_loss_at_best_epoch,
            "epochs_completed": result.epochs_completed,
            "device": result.device,
        },
        "selection": {
            "metric": config.neural.selection_metric,
            "development_fold": config.neural.development_fold,
            "development_train_domain": config.neural.development_train_domain,
            "tuning_revision": selected.tuning_revision,
            "tuning_environment": selected.tuning_environment,
            "tuning_device": selected.tuning_device,
            "selected_result": selected.result,
            "selected_path": str(selected.selected_path),
            "selected_sha256": selected.selected_sha256,
        },
    }
    return write_completed_dann_run(
        config,
        run,
        prepared.fold,
        prepared.state,
        predictions,
        model_hyperparameters=model_hyperparameters,
        training_history=result.history,
        input_hashes=input_hashes,
    )
