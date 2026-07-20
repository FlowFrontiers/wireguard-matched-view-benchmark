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
    PreparedCrossSessionClassical,
    prepare_cross_session_classical,
)
from vpncat.cross_session_preprocessing_audit import CrossSessionPreprocessingConfig
from vpncat.errors import PipelineInvariantError
from vpncat.models.classical import fit_classical_model


def build_cross_session_prediction_frame(
    run: CrossSessionRunSpec,
    prepared: PreparedCrossSessionClassical,
    probabilities: dict[str, np.ndarray],
) -> pd.DataFrame:
    classes = np.asarray(prepared.state.classes, dtype=object)
    positions = prepared.index.test_positions
    frames: list[pd.DataFrame] = []
    for domain in run.test_domains:
        domain_probabilities = np.asarray(probabilities[domain], dtype=np.float64)
        if domain_probabilities.shape != (len(positions), len(classes)):
            raise PipelineInvariantError(
                f"Cross-session {domain} probability matrix shape is invalid"
            )
        predictions = classes[np.argmax(domain_probabilities, axis=1)]
        frames.append(
            pd.DataFrame(
                {
                    "run_id": run.run_id,
                    "protocol": run.protocol,
                    "representation": run.representation,
                    "model": run.model,
                    "pair_id": [prepared.index.pair_ids[position] for position in positions],
                    "session": prepared.index.sessions[positions].astype(int),
                    "train_session": run.train_session,
                    "test_session": run.test_session,
                    "train_domain": run.train_domain,
                    "test_domain": domain,
                    "seed": run.seed,
                    "true_label": [prepared.index.labels[position] for position in positions],
                    "prediction": predictions.astype(str),
                    "class_probabilities": domain_probabilities.tolist(),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def run_cross_session_classical(
    config: CrossSessionPreprocessingConfig,
    run: CrossSessionRunSpec,
) -> Path:
    """Fit one source-session classical model and publish both target views."""
    if run.family != "classical" or run.train_domain != "inner":
        raise PipelineInvariantError("Cross-session classical runner is incompatible")
    target = config.cross_session.output_root / run.relative_output_dir
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite cross-session run: {target}")
    validate_cross_session_run_contract(config, run)
    prepared = prepare_cross_session_classical(config.cross_session, run)
    model = fit_classical_model(
        run.model,
        config.cross_session.primary.model_hyperparameters[run.model],
        prepared.state,
        prepared.training_values,
        prepared.training_targets,
        prepared.training_labels,
        seed=run.seed,
    )
    probabilities = {
        domain: model.predict_probabilities(prepared.test_values[domain])
        for domain in run.test_domains
    }
    predictions = build_cross_session_prediction_frame(run, prepared, probabilities)
    return write_completed_cross_session_run(
        config,
        run,
        prepared.index,
        prepared.state,
        predictions,
        model_hyperparameters=model.recorded_hyperparameters,
    )
