from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from vpncat.ablation_artifacts import (
    ablation_input_hashes,
    validate_ablation_run_contract,
    write_completed_ablation_run,
)
from vpncat.ablation_data import PreparedNeuralRun, prepare_ablation_run
from vpncat.ablations import AblationConfig, AblationRunSpec
from vpncat.errors import PipelineInvariantError
from vpncat.models.neural import build_neural_model, trainable_parameter_count
from vpncat.neural_training import (
    predict_neural_probabilities,
    seed_neural_execution,
    train_neural_model,
)
from vpncat.neural_tuning import (
    SelectedNeuralConfiguration,
    load_selected_neural_configuration,
)
from vpncat.provenance import git_provenance


def build_ablation_prediction_frame(
    run: AblationRunSpec,
    prepared: PreparedNeuralRun,
    probabilities: dict[str, np.ndarray],
) -> pd.DataFrame:
    classes = np.asarray(prepared.state.classes, dtype=object)
    frames: list[pd.DataFrame] = []
    for domain in run.test_domains:
        subset = prepared.tests[domain]
        domain_probabilities = np.asarray(probabilities[domain], dtype=np.float64)
        if domain_probabilities.shape != (len(subset.pair_ids), len(classes)):
            raise PipelineInvariantError(
                f"Ablation {domain} probability shape is invalid"
            )
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
                    "train_domain": run.train_domain,
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


def run_ablation(
    config: AblationConfig,
    run: AblationRunSpec,
    *,
    device_name: str = "auto",
    selected: SelectedNeuralConfiguration | None = None,
) -> Path:
    """Train one frozen ablation and atomically publish both held-out views."""
    if (
        run.is_primary_reference
        or run.family != "neural"
        or run.representation != "sequential_splt"
        or run.model not in config.models
        or run.train_domain != "inner"
        or run.seed != 42
    ):
        raise PipelineInvariantError("Ablation runner received an incompatible run")
    target = config.output_root / run.relative_output_dir
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite existing ablation run: {target}")
    provenance = git_provenance(config.project_root)
    if not provenance.get("status_available") or provenance.get("dirty"):
        raise PipelineInvariantError("Ablation execution requires a clean Git revision")
    base_hashes = validate_ablation_run_contract(config, run)
    if selected is None:
        selected = load_selected_neural_configuration(
            config.primary,
            config.neural,
            model_name=run.model,
        )
    elif selected.model != run.model:
        raise PipelineInvariantError("Ablation tuning selection belongs to another model")
    input_hashes = ablation_input_hashes(base_hashes, selected)
    prepared = prepare_ablation_run(config, run)
    seed_neural_execution(run.seed)
    model = build_neural_model(
        run.model,
        feature_count=len(run.channels),
        class_count=len(prepared.state.classes),
        width=selected.trial.width,
        dropout=selected.trial.dropout,
        maximum_length=config.neural.maximum_prefix_length,
        topology=config.neural.topologies[run.model],
    )
    parameter_count = trainable_parameter_count(model)
    if (
        run.channels == config.neural.channels
        and parameter_count != int(selected.result["parameter_count"])
    ):
        raise PipelineInvariantError("All-channel ablation differs from tuned architecture")
    result = train_neural_model(
        model,
        prepared.training,
        prepared.validation,
        prepared.state,
        learning_rate=selected.trial.learning_rate,
        batch_size=selected.trial.batch_size,
        seed=run.seed,
        optimizer_policy=config.neural.optimizer,
        training_policy=config.neural.training,
        device_name=device_name,
    )
    if result.parameter_count != parameter_count:
        raise PipelineInvariantError("Trained ablation parameter count changed")
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
    predictions = build_ablation_prediction_frame(run, prepared, probabilities)
    model_hyperparameters = {
        "selected_trial": selected.trial.to_dict(),
        "topology": config.neural.topologies[run.model],
        "optimizer": config.neural.optimizer,
        "training_policy": config.neural.training,
        "observation": {
            "prefix_length": run.prefix_length,
            "channels": list(run.channels),
        },
        "parameter_count": result.parameter_count,
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
        },
    }
    return write_completed_ablation_run(
        config,
        run,
        prepared.fold,
        prepared.state,
        predictions,
        model_hyperparameters=model_hyperparameters,
        training_history=result.history,
        input_hashes=input_hashes,
    )
