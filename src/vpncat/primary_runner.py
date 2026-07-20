from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from vpncat.artifacts import verify_input_chain, write_completed_run
from vpncat.errors import PipelineInvariantError
from vpncat.experiment import PrimaryExperimentConfig, RunSpec
from vpncat.hashing import sha256_file
from vpncat.models.classical import fit_classical_model
from vpncat.primary_data import PreparedClassicalRun, prepare_classical_run


def validate_contract_audit(config: PrimaryExperimentConfig, run: RunSpec) -> None:
    if not config.contract_audit_path.is_file():
        raise PipelineInvariantError("Experiment contract audit is missing")
    audit = json.loads(config.contract_audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "valid":
        raise PipelineInvariantError("Experiment contract audit is not valid")
    expected_hashes = verify_input_chain(config)
    if audit.get("input_hashes") != expected_hashes:
        raise PipelineInvariantError("Experiment contract audit is stale for current inputs")
    matching = [row for row in audit.get("runs", []) if row.get("run_id") == run.run_id]
    if len(matching) != 1:
        raise PipelineInvariantError("Run is absent or duplicated in experiment contract audit")
    row = matching[0]
    if row.get("relative_output_dir") != run.relative_output_dir.as_posix():
        raise PipelineInvariantError("Audited run output path disagrees with run identity")
    if sha256_file(config.config_path) != audit["input_hashes"]["primary_config"]:
        raise PipelineInvariantError("Primary configuration changed after contract audit")


def build_prediction_frame(
    run: RunSpec,
    prepared: PreparedClassicalRun,
    probabilities: dict[str, np.ndarray],
) -> pd.DataFrame:
    test_positions = prepared.fold.test_positions
    classes = np.asarray(prepared.state.classes, dtype=object)
    rows: list[dict[str, Any]] = []
    for domain in run.test_domains:
        domain_probabilities = np.asarray(probabilities[domain], dtype=np.float64)
        if domain_probabilities.shape != (len(test_positions), len(classes)):
            raise PipelineInvariantError(f"{domain} probability matrix shape is invalid")
        predictions = classes[np.argmax(domain_probabilities, axis=1)]
        for offset, position in enumerate(test_positions):
            rows.append(
                {
                    "run_id": run.run_id,
                    "protocol": run.protocol,
                    "representation": run.representation,
                    "model": run.model,
                    "pair_id": prepared.fold.pair_ids[position],
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


def run_primary_classical(config: PrimaryExperimentConfig, run: RunSpec) -> Path:
    """Fit one classical model and atomically publish both held-out test views."""
    if run.family != "classical":
        raise PipelineInvariantError("Classical runner cannot execute a neural run")
    validate_contract_audit(config, run)
    prepared = prepare_classical_run(
        config.canonical_path,
        config.split_path,
        run,
        prefix_length=config.prefix_length,
    )
    model = fit_classical_model(
        run.model,
        config.model_hyperparameters[run.model],
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
    predictions = build_prediction_frame(run, prepared, probabilities)
    return write_completed_run(
        config,
        run,
        prepared.fold,
        prepared.state,
        predictions,
        model_hyperparameters=model.recorded_hyperparameters,
    )
