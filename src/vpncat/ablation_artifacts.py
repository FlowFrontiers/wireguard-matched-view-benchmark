from __future__ import annotations

import importlib.metadata
import json
import platform
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from vpncat import __version__
from vpncat.ablations import (
    AblationConfig,
    AblationRunSpec,
    validate_ablation_contract,
)
from vpncat.artifacts import HISTORY_COLUMNS
from vpncat.errors import PipelineInvariantError
from vpncat.folds import FoldIndex
from vpncat.hashing import sha256_file
from vpncat.metrics import (
    PREDICTION_COLUMNS,
    compute_metrics,
    metrics_long_frame,
    validate_predictions,
)
from vpncat.preprocessing import FoldTargetState, pair_id_digest
from vpncat.provenance import git_provenance

ABLATION_RUN_ARTIFACTS = (
    "split_manifest.csv",
    "metrics.json",
    "metrics_long.csv",
    "predictions.parquet",
    "training_history.csv",
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _environment() -> dict[str, str]:
    packages = ("numpy", "pandas", "pyarrow", "scikit-learn", "torch")
    versions = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        **{name: importlib.metadata.version(name) for name in packages},
    }
    import torch

    versions.update(
        {
            "cuda_runtime": str(torch.version.cuda),
            "cudnn": str(torch.backends.cudnn.version()),
            "mps_built": str(torch.backends.mps.is_built()),
            "mps_available": str(torch.backends.mps.is_available()),
        }
    )
    if torch.cuda.is_available():
        versions["cuda_device"] = torch.cuda.get_device_name(torch.cuda.current_device())
    return versions


def ablation_base_input_hashes(config: AblationConfig) -> dict[str, str]:
    validate_ablation_contract(config)
    contract = _load_json(config.contract_audit_path)
    return {
        **contract["input_hashes"],
        "ablation_contract_audit": sha256_file(config.contract_audit_path),
    }


def validate_ablation_run_contract(
    config: AblationConfig,
    run: AblationRunSpec,
) -> dict[str, str]:
    return validate_ablation_run_contracts(config, (run,))


def validate_ablation_run_contracts(
    config: AblationConfig,
    runs: tuple[AblationRunSpec, ...],
) -> dict[str, str]:
    if not runs:
        raise PipelineInvariantError("No ablation runs were supplied")
    base_hashes = ablation_base_input_hashes(config)
    audited_rows = _load_json(config.contract_audit_path).get("runs", [])
    for run in runs:
        matches = [row for row in audited_rows if row.get("run_id") == run.run_id]
        if len(matches) != 1:
            raise PipelineInvariantError("Ablation run is absent or duplicated in contract")
        row = matches[0]
        for key, value in run.to_dict().items():
            if row.get(key) != value:
                raise PipelineInvariantError(f"Ablation contract field differs: {key}")
        expected_path = run.relative_output_dir.as_posix()
        if run.is_primary_reference:
            expected_path = row.get("primary_reference", {}).get("relative_output_dir")
        if row.get("artifact_relative_output_dir") != expected_path:
            raise PipelineInvariantError("Ablation contract artifact path differs")
    return base_hashes


def ablation_input_hashes(
    base_hashes: dict[str, str],
    selected: Any,
) -> dict[str, str]:
    return {
        **base_hashes,
        "neural_tuning_selection": selected.selected_sha256,
        "neural_tuning_manifest": selected.tuning_manifest_sha256,
    }


def _pair_contract(fold: FoldIndex) -> dict[str, Any]:
    return {
        "training_pair_count": len(fold.train_positions),
        "training_pair_ids_sha256": pair_id_digest(fold.pair_ids_for("train")),
        "validation_pair_count": len(fold.validation_positions),
        "validation_pair_ids_sha256": pair_id_digest(fold.pair_ids_for("validation")),
        "test_pair_count": len(fold.test_positions),
        "test_pair_ids_sha256": pair_id_digest(fold.pair_ids_for("test")),
    }


def _validate_history(
    history: pd.DataFrame,
    *,
    training_outcome: dict[str, Any],
) -> pd.DataFrame:
    if tuple(history.columns) != HISTORY_COLUMNS:
        raise PipelineInvariantError("Ablation training-history schema differs")
    if history.empty or history["epoch"].duplicated().any():
        raise PipelineInvariantError("Ablation training history is empty or duplicates epochs")
    numeric = history.loc[:, HISTORY_COLUMNS].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise PipelineInvariantError("Ablation training history contains non-finite values")
    if not np.array_equal(
        numeric["epoch"].to_numpy(dtype=np.int64),
        np.arange(1, len(numeric) + 1, dtype=np.int64),
    ):
        raise PipelineInvariantError("Ablation training epochs are not consecutive")
    best_epoch = int(training_outcome.get("best_epoch", 0))
    best_rows = numeric.loc[numeric["epoch"] == best_epoch]
    if len(best_rows) != 1:
        raise PipelineInvariantError("Ablation best epoch is absent from history")
    best = best_rows.iloc[0]
    expected_best = int(numeric.loc[numeric["validation_macro_f1"].idxmax(), "epoch"])
    if (
        best_epoch != expected_best
        or int(training_outcome.get("epochs_completed", 0)) != len(numeric)
        or not np.isclose(
            float(best["validation_macro_f1"]),
            float(training_outcome.get("best_validation_macro_f1", -1.0)),
            rtol=1e-12,
            atol=1e-15,
        )
        or not np.isclose(
            float(best["validation_loss"]),
            float(training_outcome.get("validation_loss_at_best_epoch", -1.0)),
            rtol=1e-12,
            atol=1e-15,
        )
        or (numeric["validation_macro_f1"] < 0.0).any()
        or (numeric["validation_macro_f1"] > 1.0).any()
        or (numeric["learning_rate"] <= 0.0).any()
    ):
        raise PipelineInvariantError("Ablation training outcome disagrees with history")
    return numeric


def _validate_model_hyperparameters(
    config: AblationConfig,
    run: AblationRunSpec,
    state: FoldTargetState,
    model_parameters: dict[str, Any],
    *,
    input_hashes: dict[str, str],
) -> None:
    selection_path = config.neural.tuning_output_root / run.model / "selected.json"
    tuning_manifest_path = (
        config.neural.tuning_output_root / run.model / "tuning_manifest.json"
    )
    if not selection_path.is_file() or not tuning_manifest_path.is_file():
        raise PipelineInvariantError("Completed ablation tuning artifacts are unavailable")
    if (
        sha256_file(selection_path) != input_hashes.get("neural_tuning_selection")
        or sha256_file(tuning_manifest_path)
        != input_hashes.get("neural_tuning_manifest")
    ):
        raise PipelineInvariantError("Completed ablation tuning hashes differ")
    selection = _load_json(selection_path)
    recorded_selection = model_parameters.get("selection", {})
    if (
        model_parameters.get("selected_trial") != selection.get("selected_trial")
        or recorded_selection
        != {
            "metric": config.neural.selection_metric,
            "development_fold": config.neural.development_fold,
            "development_train_domain": config.neural.development_train_domain,
            "tuning_revision": selection.get("git", {}).get("revision"),
            "tuning_environment": selection.get("execution_environment"),
            "tuning_device": selection.get("execution_device"),
            "selected_result": selection.get("selected_result"),
        }
    ):
        raise PipelineInvariantError("Completed ablation tuning selection differs")
    expected_policy = {
        "topology": config.neural.topologies[run.model],
        "optimizer": config.neural.optimizer,
        "training_policy": config.neural.training,
        "observation": {
            "prefix_length": run.prefix_length,
            "channels": list(run.channels),
        },
    }
    if any(model_parameters.get(key) != value for key, value in expected_policy.items()):
        raise PipelineInvariantError("Completed ablation model policy differs")
    trial = selection.get("selected_trial", {})
    try:
        from vpncat.models.neural import build_neural_model, trainable_parameter_count

        network = build_neural_model(
            run.model,
            feature_count=len(run.channels),
            class_count=len(state.classes),
            width=int(trial["width"]),
            dropout=float(trial["dropout"]),
            maximum_length=config.neural.maximum_prefix_length,
            topology=config.neural.topologies[run.model],
        )
        expected_count = trainable_parameter_count(network)
        recorded_count = int(model_parameters.get("parameter_count", -1))
    except (KeyError, TypeError, ValueError) as error:
        raise PipelineInvariantError("Completed ablation parameters are invalid") from error
    if recorded_count != expected_count:
        raise PipelineInvariantError("Completed ablation parameter count differs")
    if run.channels == config.neural.channels:
        tuned_count = int(selection.get("selected_result", {}).get("parameter_count", -1))
        if recorded_count != tuned_count:
            raise PipelineInvariantError("All-channel ablation differs from tuned architecture")


def _manifest(
    config: AblationConfig,
    run: AblationRunSpec,
    fold: FoldIndex,
    state: FoldTargetState,
    *,
    model_hyperparameters: dict[str, Any],
    input_hashes: dict[str, str],
) -> dict[str, Any]:
    if run.is_primary_reference:
        raise PipelineInvariantError("Primary references cannot publish ablation artifacts")
    if not input_hashes or any(
        len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        for digest in input_hashes.values()
    ):
        raise PipelineInvariantError("Ablation input hashes must be SHA-256 digests")
    expected_fit_hash = pair_id_digest(fold.pair_ids_for("train"))
    if (
        state.fold != run.fold
        or state.fit_pair_count != len(fold.train_positions)
        or state.fit_pair_ids_sha256 != expected_fit_hash
    ):
        raise PipelineInvariantError("Ablation target state is fitted on wrong pairs")
    return {
        "schema_version": 1,
        "status": "staging",
        "created_utc": datetime.now(UTC).isoformat(),
        "package_version": __version__,
        "git": git_provenance(config.project_root),
        "environment": _environment(),
        "run": run.to_dict(),
        "configuration": {
            "model_hyperparameters": model_hyperparameters,
            "observation": {
                "prefix_length": run.prefix_length,
                "channels": list(run.channels),
            },
            "augmentation": False,
        },
        "data": _pair_contract(fold),
        "preprocessing": {
            "train_domain": run.train_domain,
            "feature_transform": "fit-free",
            "fixed_transforms": ["direction-remap", "log1p-size", "log1p-iat"],
            "state": state.to_dict(),
        },
        "input_hashes": input_hashes,
        "split_manifest_sha256": input_hashes["split_manifest"],
        "class_order": list(state.classes),
    }


def write_completed_ablation_run(
    config: AblationConfig,
    run: AblationRunSpec,
    fold: FoldIndex,
    state: FoldTargetState,
    predictions: pd.DataFrame,
    *,
    model_hyperparameters: dict[str, Any],
    training_history: pd.DataFrame,
    input_hashes: dict[str, str],
) -> Path:
    """Atomically publish one trained ablation without overwriting output."""
    manifest = _manifest(
        config,
        run,
        fold,
        state,
        model_hyperparameters=model_hyperparameters,
        input_hashes=input_hashes,
    )
    history = _validate_history(
        training_history,
        training_outcome=model_hyperparameters["training_outcome"],
    )
    classes = tuple(state.classes)
    metrics = compute_metrics(predictions, run=run, fold=fold, classes=classes)
    metrics_long = metrics_long_frame(metrics, run=run)
    ordered_predictions = predictions.loc[:, PREDICTION_COLUMNS].sort_values(
        ["test_domain", "pair_id"], ignore_index=True
    )
    target = config.output_root / run.relative_output_dir
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite existing ablation run: {target}")
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    try:
        shutil.copy2(config.primary.split_path, staging / "split_manifest.csv")
        ordered_predictions.to_parquet(staging / "predictions.parquet", index=False)
        _write_json(staging / "metrics.json", {"run": run.to_dict(), "metrics": metrics})
        metrics_long.to_csv(staging / "metrics_long.csv", index=False)
        history.to_csv(staging / "training_history.csv", index=False)
        manifest["artifacts"] = {
            name: {"path": name, "sha256": sha256_file(staging / name)}
            for name in ABLATION_RUN_ARTIFACTS
        }
        manifest["status"] = "complete"
        _write_json(staging / "run.json", manifest)
        validate_completed_ablation_run(
            staging,
            config=config,
            run=run,
            fold=fold,
            state=state,
            expected_input_hashes=input_hashes,
        )
        staging.replace(target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target


def validate_completed_ablation_run(
    run_dir: Path,
    *,
    config: AblationConfig,
    run: AblationRunSpec,
    fold: FoldIndex,
    state: FoldTargetState,
    expected_input_hashes: dict[str, str],
    expected_git_revision: str | None = None,
) -> dict[str, Any]:
    if run.is_primary_reference:
        raise PipelineInvariantError("Primary references are not ablation outputs")
    manifest = _load_json(run_dir / "run.json")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("status") != "complete"
        or manifest.get("run") != run.to_dict()
        or manifest.get("package_version") != __version__
        or manifest.get("input_hashes") != expected_input_hashes
    ):
        raise PipelineInvariantError("Completed ablation manifest identity is invalid")
    if expected_git_revision is not None:
        provenance = manifest.get("git", {})
        if (
            provenance.get("revision") != expected_git_revision
            or provenance.get("dirty") is not False
        ):
            raise PipelineInvariantError("Completed ablation Git revision differs")
    expected_configuration = {
        "observation": {
            "prefix_length": run.prefix_length,
            "channels": list(run.channels),
        },
        "augmentation": False,
    }
    configuration = manifest.get("configuration", {})
    if any(configuration.get(key) != value for key, value in expected_configuration.items()):
        raise PipelineInvariantError("Completed ablation observation policy differs")
    if manifest.get("data") != _pair_contract(fold):
        raise PipelineInvariantError("Completed ablation pair contract differs")
    if manifest.get("class_order") != list(state.classes):
        raise PipelineInvariantError("Completed ablation class order differs")
    preprocessing = manifest.get("preprocessing", {})
    if (
        preprocessing.get("train_domain") != run.train_domain
        or preprocessing.get("state") != state.to_dict()
    ):
        raise PipelineInvariantError("Completed ablation preprocessing state differs")
    if sha256_file(run_dir / "split_manifest.csv") != manifest.get(
        "split_manifest_sha256"
    ):
        raise PipelineInvariantError("Completed ablation split hash differs")
    artifacts = manifest.get("artifacts", {})
    if set(artifacts) != set(ABLATION_RUN_ARTIFACTS):
        raise PipelineInvariantError("Completed ablation artifact inventory differs")
    physical = {path.name for path in run_dir.iterdir() if path.is_file()}
    if physical != set(ABLATION_RUN_ARTIFACTS) | {"run.json"}:
        raise PipelineInvariantError("Completed ablation directory has unexpected files")
    for name, metadata in artifacts.items():
        if sha256_file(run_dir / name) != metadata.get("sha256"):
            raise PipelineInvariantError(f"Completed ablation artifact hash mismatch: {name}")
    predictions = pd.read_parquet(run_dir / "predictions.parquet")
    validate_predictions(predictions, run=run, fold=fold, classes=state.classes)
    recomputed = compute_metrics(predictions, run=run, fold=fold, classes=state.classes)
    if _load_json(run_dir / "metrics.json") != {"run": run.to_dict(), "metrics": recomputed}:
        raise PipelineInvariantError("Completed ablation metrics disagree with predictions")
    expected_long = metrics_long_frame(recomputed, run=run)
    observed_long = pd.read_csv(run_dir / "metrics_long.csv")
    try:
        pd.testing.assert_frame_equal(
            observed_long,
            expected_long,
            check_exact=False,
            rtol=1e-12,
            atol=1e-15,
        )
    except AssertionError as error:
        raise PipelineInvariantError("Completed ablation long metrics disagree") from error
    model_parameters = configuration.get("model_hyperparameters", {})
    _validate_model_hyperparameters(
        config,
        run,
        state,
        model_parameters,
        input_hashes=expected_input_hashes,
    )
    _validate_history(
        pd.read_csv(run_dir / "training_history.csv", float_precision="round_trip"),
        training_outcome=model_parameters.get("training_outcome", {}),
    )
    return {
        "run_id": run.run_id,
        "prediction_rows": len(predictions),
        "test_pairs": len(fold.test_positions),
        "test_domains": list(run.test_domains),
        "status": "valid",
    }
