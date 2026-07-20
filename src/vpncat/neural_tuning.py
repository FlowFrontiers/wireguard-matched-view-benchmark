from __future__ import annotations

import importlib.metadata
import json
import platform
import shutil
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from vpncat import __version__
from vpncat.artifacts import HISTORY_COLUMNS, verify_input_chain
from vpncat.errors import PipelineInvariantError
from vpncat.experiment import (
    PrimaryExperimentConfig,
    RunSpec,
    select_primary_run,
)
from vpncat.hashing import sha256_file
from vpncat.models.neural import build_neural_model, trainable_parameter_count
from vpncat.neural_config import NeuralConfig, NeuralTrial
from vpncat.neural_data import PreparedNeuralDevelopment, prepare_neural_development
from vpncat.neural_training import seed_neural_execution, train_neural_model
from vpncat.primary_runner import validate_contract_audit
from vpncat.provenance import git_provenance

NEURAL_MODELS = ("cnn1d", "lstm", "transformer")


@dataclass(frozen=True)
class SelectedNeuralConfiguration:
    model: str
    trial: NeuralTrial
    result: dict[str, Any]
    selected_path: Path
    selected_sha256: str
    tuning_manifest_sha256: str
    tuning_revision: str
    tuning_environment: dict[str, str]
    tuning_device: str


def _environment_versions() -> dict[str, str]:
    packages = ("numpy", "pandas", "pyarrow", "scikit-learn", "torch")
    versions = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        **{name: importlib.metadata.version(name) for name in packages},
        "cuda_runtime": str(torch.version.cuda),
        "cudnn": str(torch.backends.cudnn.version()),
    }
    if torch.cuda.is_available():
        versions["cuda_device"] = torch.cuda.get_device_name(torch.cuda.current_device())
    return versions


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    _write_json(temporary, payload)
    temporary.replace(path)


def _tuning_inputs(
    primary: PrimaryExperimentConfig,
    neural: NeuralConfig,
) -> dict[str, str]:
    return {
        **verify_input_chain(primary),
        "neural_config": sha256_file(neural.config_path),
    }


def _development_run(primary: PrimaryExperimentConfig, neural: NeuralConfig, model: str) -> RunSpec:
    if model not in NEURAL_MODELS:
        raise PipelineInvariantError(f"Unsupported neural tuning model: {model}")
    return select_primary_run(
        primary,
        experiment_id=f"sequential_splt__{model}",
        fold=neural.development_fold,
        train_domain=neural.development_train_domain,
        seed=neural.development_seed,
    )


def _validate_clean_provenance(primary: PrimaryExperimentConfig) -> dict[str, Any]:
    provenance = git_provenance(primary.project_root)
    if not provenance.get("status_available") or provenance.get("dirty"):
        raise PipelineInvariantError("Neural tuning requires a clean Git revision")
    return provenance


def _trial_identity(model: str, trial: NeuralTrial) -> dict[str, Any]:
    return {"model": model, "trial": trial.to_dict()}


def _audited_development_data(
    primary: PrimaryExperimentConfig,
    neural: NeuralConfig,
) -> dict[str, Any]:
    preprocessing = json.loads(primary.preprocessing_audit_path.read_text(encoding="utf-8"))
    targets = preprocessing.get("folds", {}).get(str(neural.development_fold), {}).get(
        "targets"
    )
    if targets is None:
        raise PipelineInvariantError("Preprocessing audit lacks development-fold targets")
    split = pd.read_csv(primary.split_path, usecols=[f"role_fold_{neural.development_fold}"])
    counts = split.iloc[:, 0].value_counts().to_dict()
    if set(counts) != {"train", "validation", "test"}:
        raise PipelineInvariantError("Development fold roles are incomplete")
    return {
        "training_pair_count": int(counts["train"]),
        "validation_pair_count": int(counts["validation"]),
        "fit_pair_ids_sha256": str(targets["fit_pair_ids_sha256"]),
        "prefix_length": neural.primary_prefix_length,
        "channels": list(neural.channels),
        "class_order": list(targets["classes"]),
        "class_weights": list(targets["class_weights"]),
        "test_views_materialized": False,
    }


def _validate_trial_directory(
    path: Path,
    *,
    model: str,
    trial: NeuralTrial,
    neural: NeuralConfig,
    run: RunSpec,
    expected_data: dict[str, Any],
    input_hashes: dict[str, str],
    revision: str,
) -> dict[str, Any]:
    manifest_path = path / "trial.json"
    history_path = path / "training_history.csv"
    if not manifest_path.is_file() or not history_path.is_file():
        raise PipelineInvariantError(f"Incomplete neural tuning trial directory: {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("status") != "complete":
        raise PipelineInvariantError(f"Neural tuning trial is not complete: {path}")
    if manifest.get("package_version") != __version__:
        raise PipelineInvariantError("Neural tuning trial package version mismatch")
    if manifest.get("identity") != _trial_identity(model, trial):
        raise PipelineInvariantError("Neural tuning trial identity mismatch")
    if manifest.get("input_hashes") != input_hashes:
        raise PipelineInvariantError("Neural tuning trial input hashes are stale")
    if manifest.get("git", {}).get("revision") != revision:
        raise PipelineInvariantError("Neural tuning trial was produced by another revision")
    if manifest.get("git", {}).get("dirty") is not False:
        raise PipelineInvariantError("Neural tuning trial was not produced from a clean tree")
    if manifest.get("development_run") != run.to_dict():
        raise PipelineInvariantError("Neural tuning development run mismatch")
    if manifest.get("topology") != neural.topologies[model]:
        raise PipelineInvariantError("Neural tuning topology mismatch")
    if manifest.get("optimizer") != neural.optimizer:
        raise PipelineInvariantError("Neural tuning optimizer policy mismatch")
    if manifest.get("training_policy") != neural.training:
        raise PipelineInvariantError("Neural tuning training policy mismatch")
    if manifest.get("data") != expected_data:
        raise PipelineInvariantError("Neural tuning data contract mismatch")
    artifacts = manifest.get("artifacts", {})
    if set(artifacts) != {"training_history.csv"}:
        raise PipelineInvariantError("Neural tuning artifact inventory mismatch")
    if artifacts.get("training_history.csv", {}).get(
        "sha256"
    ) != sha256_file(history_path):
        raise PipelineInvariantError("Neural tuning history hash mismatch")
    history = pd.read_csv(history_path, float_precision="round_trip")
    if tuple(history.columns) != HISTORY_COLUMNS or history.empty:
        raise PipelineInvariantError("Neural tuning history schema is invalid")
    if not np.isfinite(history.to_numpy(dtype=float)).all():
        raise PipelineInvariantError("Neural tuning history contains non-finite values")
    result = manifest.get("result", {})
    if set(result) != {
        "best_epoch",
        "best_validation_macro_f1",
        "validation_loss_at_best_epoch",
        "epochs_completed",
        "parameter_count",
        "device",
    }:
        raise PipelineInvariantError("Neural tuning result schema mismatch")
    if int(result["epochs_completed"]) != len(history):
        raise PipelineInvariantError("Neural tuning epoch count disagrees with history")
    expected_epochs = np.arange(1, len(history) + 1)
    if not np.array_equal(history["epoch"].to_numpy(dtype=int), expected_epochs):
        raise PipelineInvariantError("Neural tuning history epochs are not consecutive")
    best_epoch = int(result.get("best_epoch", 0))
    if best_epoch not in set(history["epoch"].astype(int)):
        raise PipelineInvariantError("Neural tuning best epoch is absent from history")
    best_row = history.loc[history["epoch"].astype(int) == best_epoch].iloc[0]
    if not np.isclose(
        float(best_row["validation_macro_f1"]),
        float(result.get("best_validation_macro_f1", np.nan)),
        rtol=1e-12,
        atol=1e-15,
    ):
        raise PipelineInvariantError("Neural tuning best score disagrees with history")
    first_best_epoch = int(history.loc[history["validation_macro_f1"].idxmax(), "epoch"])
    if best_epoch != first_best_epoch:
        raise PipelineInvariantError("Neural tuning did not retain the first best macro-F1 epoch")
    if not np.isclose(
        float(best_row["validation_loss"]),
        float(result["validation_loss_at_best_epoch"]),
        rtol=1e-12,
        atol=1e-15,
    ):
        raise PipelineInvariantError("Neural tuning best validation loss disagrees with history")
    network = build_neural_model(
        model,
        feature_count=len(neural.channels),
        class_count=len(expected_data["class_order"]),
        width=trial.width,
        dropout=trial.dropout,
        maximum_length=neural.maximum_prefix_length,
        topology=neural.topologies[model],
    )
    if int(result["parameter_count"]) != trainable_parameter_count(network):
        raise PipelineInvariantError("Neural tuning parameter count mismatch")
    if str(result["device"]) not in {"cpu", "cuda", "mps"}:
        raise PipelineInvariantError("Neural tuning device value is invalid")
    return manifest


def _publish_trial(
    model_root: Path,
    *,
    model_name: str,
    trial: NeuralTrial,
    neural: NeuralConfig,
    development: PreparedNeuralDevelopment,
    run: RunSpec,
    input_hashes: dict[str, str],
    expected_data: dict[str, Any],
    provenance: dict[str, Any],
    device_name: str,
) -> dict[str, Any]:
    target = model_root / f"trial_{trial.trial_id:02d}"
    if target.exists():
        return _validate_trial_directory(
            target,
            model=model_name,
            trial=trial,
            neural=neural,
            run=run,
            expected_data=expected_data,
            input_hashes=input_hashes,
            revision=str(provenance["revision"]),
        )

    seed_neural_execution(neural.development_seed)
    network = build_neural_model(
        model_name,
        feature_count=len(neural.channels),
        class_count=len(development.state.classes),
        width=trial.width,
        dropout=trial.dropout,
        maximum_length=neural.maximum_prefix_length,
        topology=neural.topologies[model_name],
    )
    result = train_neural_model(
        network,
        development.training,
        development.validation,
        development.state,
        learning_rate=trial.learning_rate,
        batch_size=trial.batch_size,
        seed=neural.development_seed,
        optimizer_policy=neural.optimizer,
        training_policy=neural.training,
        device_name=device_name,
    )
    model_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".trial_{trial.trial_id:02d}-", dir=model_root))
    try:
        history_path = staging / "training_history.csv"
        result.history.to_csv(history_path, index=False)
        manifest = {
            "schema_version": 1,
            "status": "complete",
            "created_utc": datetime.now(UTC).isoformat(),
            "package_version": __version__,
            "git": provenance,
            "environment": _environment_versions(),
            "identity": _trial_identity(model_name, trial),
            "development_run": run.to_dict(),
            "input_hashes": input_hashes,
            "topology": neural.topologies[model_name],
            "optimizer": neural.optimizer,
            "training_policy": neural.training,
            "data": expected_data,
            "result": {
                "best_epoch": result.best_epoch,
                "best_validation_macro_f1": result.best_validation_macro_f1,
                "validation_loss_at_best_epoch": result.validation_loss_at_best_epoch,
                "epochs_completed": result.epochs_completed,
                "parameter_count": result.parameter_count,
                "device": result.device,
            },
            "artifacts": {
                "training_history.csv": {
                    "path": "training_history.csv",
                    "sha256": sha256_file(history_path),
                }
            },
        }
        _write_json(staging / "trial.json", manifest)
        _validate_trial_directory(
            staging,
            model=model_name,
            trial=trial,
            neural=neural,
            run=run,
            expected_data=expected_data,
            input_hashes=input_hashes,
            revision=str(provenance["revision"]),
        )
        staging.replace(target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def _select_trial(manifests: list[dict[str, Any]]) -> dict[str, Any]:
    return min(
        manifests,
        key=lambda manifest: (
            -float(manifest["result"]["best_validation_macro_f1"]),
            int(manifest["identity"]["trial"]["id"]),
        ),
    )


def _selection_payload(
    model_name: str,
    manifests: list[dict[str, Any]],
    *,
    neural: NeuralConfig,
    input_hashes: dict[str, str],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": "complete" if len(manifests) == neural.maximum_trials else "partial",
        "model": model_name,
        "selection_metric": neural.selection_metric,
        "tie_break": "lowest_trial_id",
        "expected_trial_count": neural.maximum_trials,
        "completed_trial_ids": sorted(
            int(manifest["identity"]["trial"]["id"]) for manifest in manifests
        ),
        "input_hashes": input_hashes,
        "git": provenance,
    }
    if manifests:
        environments = {
            json.dumps(manifest["environment"], sort_keys=True)
            for manifest in manifests
        }
        devices = {str(manifest["result"]["device"]) for manifest in manifests}
        if len(environments) != 1 or len(devices) != 1:
            raise PipelineInvariantError(
                "Neural tuning trials mix execution environments or devices"
            )
        payload["execution_environment"] = dict(manifests[0]["environment"])
        payload["execution_device"] = next(iter(devices))
    if len(manifests) == neural.maximum_trials:
        selected = _select_trial(manifests)
        payload["selected_trial"] = selected["identity"]["trial"]
        payload["selected_result"] = selected["result"]
    return payload


def load_selected_neural_configuration(
    primary: PrimaryExperimentConfig,
    neural: NeuralConfig,
    *,
    model_name: str,
) -> SelectedNeuralConfiguration:
    """Recompute and validate one complete tuning selection under the current revision."""
    run = _development_run(primary, neural, model_name)
    validate_contract_audit(primary, run)
    provenance = _validate_clean_provenance(primary)
    input_hashes = _tuning_inputs(primary, neural)
    expected_data = _audited_development_data(primary, neural)
    model_root = neural.tuning_output_root / model_name
    manifests = [
        _validate_trial_directory(
            model_root / f"trial_{trial.trial_id:02d}",
            model=model_name,
            trial=trial,
            neural=neural,
            run=run,
            expected_data=expected_data,
            input_hashes=input_hashes,
            revision=str(provenance["revision"]),
        )
        for trial in neural.trials
    ]
    expected = _selection_payload(
        model_name,
        manifests,
        neural=neural,
        input_hashes=input_hashes,
        provenance=provenance,
    )
    if expected["status"] != "complete":
        raise PipelineInvariantError("Neural tuning selection is incomplete")
    selected_path = model_root / "selected.json"
    tuning_manifest_path = model_root / "tuning_manifest.json"
    if not selected_path.is_file() or not tuning_manifest_path.is_file():
        raise PipelineInvariantError("Neural tuning selection artifacts are missing")
    selected_payload = json.loads(selected_path.read_text(encoding="utf-8"))
    tuning_payload = json.loads(tuning_manifest_path.read_text(encoding="utf-8"))
    if selected_payload != expected or tuning_payload != expected:
        raise PipelineInvariantError("Neural tuning selection artifacts disagree with trials")
    selected_trial_id = int(expected["selected_trial"]["id"])
    selected_trial = next(
        trial for trial in neural.trials if trial.trial_id == selected_trial_id
    )
    return SelectedNeuralConfiguration(
        model=model_name,
        trial=selected_trial,
        result=dict(expected["selected_result"]),
        selected_path=selected_path,
        selected_sha256=sha256_file(selected_path),
        tuning_manifest_sha256=sha256_file(tuning_manifest_path),
        tuning_revision=str(provenance["revision"]),
        tuning_environment=dict(expected["execution_environment"]),
        tuning_device=str(expected["execution_device"]),
    )


def tune_neural_model(
    primary: PrimaryExperimentConfig,
    neural: NeuralConfig,
    *,
    model_name: str,
    device_name: str = "auto",
    trial_ids: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    """Run or resume frozen development trials without materializing held-out views."""
    run = _development_run(primary, neural, model_name)
    validate_contract_audit(primary, run)
    provenance = _validate_clean_provenance(primary)
    input_hashes = _tuning_inputs(primary, neural)
    expected_data = _audited_development_data(primary, neural)
    selected_ids = (
        tuple(trial.trial_id for trial in neural.trials)
        if trial_ids is None
        else tuple(trial_ids)
    )
    if not selected_ids or len(set(selected_ids)) != len(selected_ids):
        raise PipelineInvariantError("Requested neural tuning trial IDs are empty or duplicated")
    unknown = set(selected_ids) - {trial.trial_id for trial in neural.trials}
    if unknown:
        raise PipelineInvariantError(f"Unknown neural tuning trial IDs: {sorted(unknown)}")

    model_root = neural.tuning_output_root / model_name
    missing_requested = [
        trial
        for trial in neural.trials
        if trial.trial_id in selected_ids
        and not (model_root / f"trial_{trial.trial_id:02d}").exists()
    ]
    development = None
    if missing_requested:
        development = prepare_neural_development(
            primary.canonical_path,
            primary.split_path,
            run,
            prefix_length=neural.primary_prefix_length,
            channels=neural.channels,
        )
        observed_data = {
            "training_pair_count": len(development.training.pair_ids),
            "validation_pair_count": len(development.validation.pair_ids),
            "fit_pair_ids_sha256": development.state.fit_pair_ids_sha256,
            "prefix_length": neural.primary_prefix_length,
            "channels": list(neural.channels),
            "class_order": list(development.state.classes),
            "class_weights": development.state.class_weights.tolist(),
            "test_views_materialized": False,
        }
        if observed_data != expected_data:
            raise PipelineInvariantError("Materialized development data differs from audit")
    for trial in neural.trials:
        if trial.trial_id not in selected_ids:
            continue
        if development is None:
            _validate_trial_directory(
                model_root / f"trial_{trial.trial_id:02d}",
                model=model_name,
                trial=trial,
                neural=neural,
                run=run,
                expected_data=expected_data,
                input_hashes=input_hashes,
                revision=str(provenance["revision"]),
            )
        else:
            _publish_trial(
                model_root,
                model_name=model_name,
                trial=trial,
                neural=neural,
                development=development,
                run=run,
                input_hashes=input_hashes,
                expected_data=expected_data,
                provenance=provenance,
                device_name=device_name,
            )

    completed: list[dict[str, Any]] = []
    for trial in neural.trials:
        trial_path = model_root / f"trial_{trial.trial_id:02d}"
        if trial_path.exists():
            completed.append(
                _validate_trial_directory(
                    trial_path,
                    model=model_name,
                    trial=trial,
                    neural=neural,
                    run=run,
                    expected_data=expected_data,
                    input_hashes=input_hashes,
                    revision=str(provenance["revision"]),
                )
            )
    payload = _selection_payload(
        model_name,
        completed,
        neural=neural,
        input_hashes=input_hashes,
        provenance=provenance,
    )
    if len(completed) == neural.maximum_trials:
        selected_path = model_root / "selected.json"
        if selected_path.exists():
            existing = json.loads(selected_path.read_text(encoding="utf-8"))
            if existing != payload:
                raise PipelineInvariantError("Existing neural selection disagrees with trials")
        else:
            _atomic_json(selected_path, payload)
    _atomic_json(model_root / "tuning_manifest.json", payload)
    return payload
