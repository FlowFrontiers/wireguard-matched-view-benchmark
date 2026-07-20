from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from vpncat.cross_session import CrossSessionRunSpec
from vpncat.cross_session_artifacts import (
    validate_cross_session_run_contract,
    write_completed_cross_session_run,
)
from vpncat.cross_session_data import (
    PreparedCrossSessionNeural,
    prepare_cross_session_neural,
)
from vpncat.cross_session_preprocessing_audit import CrossSessionPreprocessingConfig
from vpncat.errors import PipelineInvariantError
from vpncat.hashing import sha256_file
from vpncat.models.neural import build_neural_model, trainable_parameter_count
from vpncat.neural_config import NeuralConfig
from vpncat.neural_training import (
    predict_neural_probabilities,
    seed_neural_execution,
    train_neural_model,
)
from vpncat.neural_tuning import (
    SelectedNeuralConfiguration,
    load_selected_neural_configuration,
)


def cross_session_neural_input_hashes(
    neural: NeuralConfig,
    selected: SelectedNeuralConfiguration,
) -> dict[str, str]:
    return {
        "neural_config": sha256_file(neural.config_path),
        "neural_tuning_selection": selected.selected_sha256,
        "neural_tuning_manifest": selected.tuning_manifest_sha256,
    }


def build_cross_session_neural_prediction_frame(
    run: CrossSessionRunSpec,
    prepared: PreparedCrossSessionNeural,
    probabilities: dict[str, np.ndarray],
) -> pd.DataFrame:
    classes = np.asarray(prepared.state.classes, dtype=object)
    frames: list[pd.DataFrame] = []
    for domain in run.test_domains:
        subset = prepared.tests[domain]
        domain_probabilities = np.asarray(probabilities[domain], dtype=np.float64)
        if domain_probabilities.shape != (len(subset.pair_ids), len(classes)):
            raise PipelineInvariantError(
                f"Cross-session {domain} neural probability shape is invalid"
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
                    "session": prepared.index.sessions[subset.positions].astype(int),
                    "train_session": run.train_session,
                    "test_session": run.test_session,
                    "train_domain": run.train_domain,
                    "test_domain": domain,
                    "seed": run.seed,
                    "true_label": [
                        prepared.index.labels[position] for position in subset.positions
                    ],
                    "prediction": predictions.astype(str),
                    "class_probabilities": domain_probabilities.tolist(),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def run_cross_session_neural(
    config: CrossSessionPreprocessingConfig,
    neural: NeuralConfig,
    run: CrossSessionRunSpec,
    *,
    device_name: str = "auto",
    selected: SelectedNeuralConfiguration | None = None,
) -> Path:
    """Train one frozen neural configuration and publish both target views."""
    if (
        run.family != "neural"
        or run.representation != "sequential_splt"
        or run.model not in neural.topologies
        or run.train_domain != "inner"
    ):
        raise PipelineInvariantError("Cross-session neural runner is incompatible")
    target = config.cross_session.output_root / run.relative_output_dir
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite cross-session run: {target}")
    validate_cross_session_run_contract(config, run)
    if selected is None:
        selected = load_selected_neural_configuration(
            config.cross_session.primary,
            neural,
            model_name=run.model,
        )
    elif selected.model != run.model:
        raise PipelineInvariantError("Neural selection belongs to another model")
    prepared = prepare_cross_session_neural(
        config.cross_session,
        run,
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
        raise PipelineInvariantError(
            "Cross-session neural model differs from tuned parameter count"
        )
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
    predictions = build_cross_session_neural_prediction_frame(
        run, prepared, probabilities
    )
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
            "selected_path": str(selected.selected_path),
        },
    }
    return write_completed_cross_session_run(
        config,
        run,
        prepared.index,
        prepared.state,
        predictions,
        model_hyperparameters=model_hyperparameters,
        training_history=result.history,
        additional_input_hashes=cross_session_neural_input_hashes(neural, selected),
    )
