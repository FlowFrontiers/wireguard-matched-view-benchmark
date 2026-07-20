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
from vpncat.dann import (
    DANN_HISTORY_COLUMNS,
    DANNConfig,
    DANNRunSpec,
    validate_dann_contract,
)
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

DANN_RUN_ARTIFACTS = (
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


def dann_base_input_hashes(config: DANNConfig) -> dict[str, str]:
    validate_dann_contract(config)
    contract = _load_json(config.contract_audit_path)
    return {
        **contract["input_hashes"],
        "dann_contract_audit": sha256_file(config.contract_audit_path),
    }


def validate_dann_run_contract(config: DANNConfig, run: DANNRunSpec) -> dict[str, str]:
    return validate_dann_run_contracts(config, (run,))


def validate_dann_run_contracts(
    config: DANNConfig,
    runs: tuple[DANNRunSpec, ...],
) -> dict[str, str]:
    if not runs:
        raise PipelineInvariantError("No DANN runs were supplied for contract validation")
    base_hashes = dann_base_input_hashes(config)
    audit = _load_json(config.contract_audit_path)
    audited_rows = audit.get("runs", [])
    for run in runs:
        matches = [row for row in audited_rows if row.get("run_id") == run.run_id]
        if len(matches) != 1:
            raise PipelineInvariantError("DANN run is absent or duplicated in contract audit")
        row = matches[0]
        for key, value in run.to_dict().items():
            if row.get(key) != value:
                raise PipelineInvariantError(f"DANN contract run field differs: {key}")
        if row.get("relative_output_dir") != run.relative_output_dir.as_posix():
            raise PipelineInvariantError("DANN contract output path differs")
    return base_hashes


def _pair_contract(fold: FoldIndex) -> dict[str, Any]:
    return {
        "source_training_pair_count": len(fold.train_positions),
        "source_training_pair_ids_sha256": pair_id_digest(fold.pair_ids_for("train")),
        "adaptation_pair_count": len(fold.train_positions),
        "adaptation_pair_ids_sha256": pair_id_digest(fold.pair_ids_for("train")),
        "adaptation_labels_exposed": False,
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
    if tuple(history.columns) != DANN_HISTORY_COLUMNS:
        raise PipelineInvariantError("DANN training-history schema differs")
    if history.empty or history["epoch"].duplicated().any():
        raise PipelineInvariantError("DANN training history is empty or duplicates epochs")
    numeric = history.loc[:, DANN_HISTORY_COLUMNS].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise PipelineInvariantError("DANN training history contains non-finite values")
    if not np.array_equal(
        numeric["epoch"].to_numpy(dtype=np.int64),
        np.arange(1, len(numeric) + 1, dtype=np.int64),
    ):
        raise PipelineInvariantError("DANN training epochs are not consecutive from one")
    if not numeric["grl_coefficient_start"].is_monotonic_increasing or not numeric[
        "grl_coefficient_end"
    ].is_monotonic_increasing:
        raise PipelineInvariantError("DANN gradient-reversal history is not monotonic")
    if float(numeric["grl_coefficient_start"].iloc[0]) != 0.0:
        raise PipelineInvariantError("DANN gradient-reversal history does not start at zero")
    coefficients = numeric.loc[
        :, ["grl_coefficient_start", "grl_coefficient_end"]
    ].to_numpy()
    losses = numeric.loc[
        :,
        [
            "train_classification_loss",
            "train_domain_loss",
            "train_total_loss",
            "validation_loss",
        ],
    ].to_numpy()
    if (
        (coefficients < 0.0).any()
        or (coefficients > 1.0).any()
        or (coefficients[:, 0] > coefficients[:, 1]).any()
    ):
        raise PipelineInvariantError("DANN gradient-reversal history is outside [0, 1]")
    if (losses < 0.0).any():
        raise PipelineInvariantError("DANN training history contains negative losses")
    best_epoch = int(training_outcome.get("best_epoch", 0))
    best_rows = numeric.loc[numeric["epoch"] == best_epoch]
    if len(best_rows) != 1:
        raise PipelineInvariantError("DANN best epoch is absent from training history")
    best_row = best_rows.iloc[0]
    expected_best_epoch = int(
        numeric.loc[numeric["validation_macro_f1"].idxmax(), "epoch"]
    )
    if (
        best_epoch != expected_best_epoch
        or (numeric["validation_macro_f1"] < 0.0).any()
        or (numeric["validation_macro_f1"] > 1.0).any()
        or (numeric["learning_rate"] <= 0.0).any()
        or not np.isclose(
            float(best_row["validation_macro_f1"]),
            float(training_outcome.get("best_validation_macro_f1", -1.0)),
            rtol=1e-12,
            atol=1e-15,
        )
        or not np.isclose(
            float(best_row["validation_loss"]),
            float(training_outcome.get("validation_loss_at_best_epoch", -1.0)),
            rtol=1e-12,
            atol=1e-15,
        )
        or int(training_outcome.get("epochs_completed", 0)) != len(numeric)
        or float(best_row["validation_macro_f1"])
        != float(numeric["validation_macro_f1"].max())
    ):
        raise PipelineInvariantError("DANN training outcome disagrees with history")
    return numeric


def _validate_model_hyperparameters(
    config: DANNConfig,
    model_parameters: dict[str, Any],
    *,
    input_hashes: dict[str, str],
) -> None:
    selection_path = config.neural.tuning_output_root / "cnn1d" / "selected.json"
    tuning_manifest_path = (
        config.neural.tuning_output_root / "cnn1d" / "tuning_manifest.json"
    )
    if not selection_path.is_file() or not tuning_manifest_path.is_file():
        raise PipelineInvariantError("Completed DANN tuning artifacts are unavailable")
    if (
        sha256_file(selection_path) != input_hashes.get("neural_tuning_selection")
        or sha256_file(tuning_manifest_path)
        != input_hashes.get("neural_tuning_manifest")
    ):
        raise PipelineInvariantError("Completed DANN tuning hashes are incompatible")
    selection = _load_json(selection_path)
    if model_parameters.get("selected_trial") != selection.get("selected_trial"):
        raise PipelineInvariantError("Completed DANN selected trial is incompatible")
    recorded_selection = model_parameters.get("selection", {})
    if recorded_selection.get("selected_result") != selection.get("selected_result"):
        raise PipelineInvariantError("Completed DANN selected result is incompatible")
    try:
        selected_parameters = int(
            selection.get("selected_result", {}).get("parameter_count", -1)
        )
        total_parameters = int(model_parameters.get("parameter_count", -1))
    except (TypeError, ValueError) as error:
        raise PipelineInvariantError("Completed DANN parameter counts are invalid") from error
    if model_parameters.get("backbone_parameter_count") != selected_parameters:
        raise PipelineInvariantError("Completed DANN backbone parameter count is incompatible")
    expected = {
        "backbone_topology": config.neural.topologies["cnn1d"],
        "domain_head": config.domain_head,
        "optimizer": config.neural.optimizer,
        "training_policy": config.neural.training,
        "domain_loss_weight": config.domain_loss_weight,
        "gradient_reversal": config.gradient_reversal,
    }
    if any(model_parameters.get(key) != value for key, value in expected.items()):
        raise PipelineInvariantError("Completed DANN model policy is incompatible")
    if total_parameters <= selected_parameters:
        raise PipelineInvariantError("Completed DANN domain head adds no parameters")


def _manifest(
    config: DANNConfig,
    run: DANNRunSpec,
    fold: FoldIndex,
    state: FoldTargetState,
    *,
    model_hyperparameters: dict[str, Any],
    input_hashes: dict[str, str],
) -> dict[str, Any]:
    if not input_hashes or any(
        len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        for digest in input_hashes.values()
    ):
        raise PipelineInvariantError("DANN input hashes must all be SHA-256 digests")
    expected_fit_hash = pair_id_digest(fold.pair_ids_for("train"))
    if (
        state.fold != run.fold
        or state.fit_pair_count != len(fold.train_positions)
        or state.fit_pair_ids_sha256 != expected_fit_hash
    ):
        raise PipelineInvariantError("DANN target state is not fitted on source training pairs")
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
            "source_domain": config.source_domain,
            "adaptation_domain": config.adaptation_domain,
            "domain_loss_weight": config.domain_loss_weight,
            "gradient_reversal": config.gradient_reversal,
            "domain_head": config.domain_head,
            "augmentation": False,
        },
        "data": _pair_contract(fold),
        "preprocessing": {
            "train_domain": run.source_domain,
            "feature_transform": "fit-free",
            "fixed_transforms": ["direction-remap", "log1p-size", "log1p-iat"],
            "state": state.to_dict(),
        },
        "input_hashes": input_hashes,
        "split_manifest_sha256": input_hashes["split_manifest"],
        "class_order": list(state.classes),
    }


def write_completed_dann_run(
    config: DANNConfig,
    run: DANNRunSpec,
    fold: FoldIndex,
    state: FoldTargetState,
    predictions: pd.DataFrame,
    *,
    model_hyperparameters: dict[str, Any],
    training_history: pd.DataFrame,
    input_hashes: dict[str, str],
) -> Path:
    """Atomically publish one complete DANN run without overwriting output."""
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
        raise FileExistsError(f"Refusing to overwrite existing DANN run: {target}")
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    try:
        shutil.copy2(config.primary.split_path, staging / "split_manifest.csv")
        ordered_predictions.to_parquet(staging / "predictions.parquet", index=False)
        _write_json(staging / "metrics.json", {"run": run.to_dict(), "metrics": metrics})
        metrics_long.to_csv(staging / "metrics_long.csv", index=False)
        history.to_csv(staging / "training_history.csv", index=False)
        manifest["artifacts"] = {
            name: {"path": name, "sha256": sha256_file(staging / name)}
            for name in DANN_RUN_ARTIFACTS
        }
        manifest["status"] = "complete"
        _write_json(staging / "run.json", manifest)
        validate_completed_dann_run(
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


def validate_completed_dann_run(
    run_dir: Path,
    *,
    config: DANNConfig,
    run: DANNRunSpec,
    fold: FoldIndex,
    state: FoldTargetState,
    expected_input_hashes: dict[str, str],
    expected_git_revision: str | None = None,
) -> dict[str, Any]:
    manifest = _load_json(run_dir / "run.json")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("status") != "complete"
        or manifest.get("run") != run.to_dict()
        or manifest.get("package_version") != __version__
    ):
        raise PipelineInvariantError("Completed DANN manifest identity is invalid")
    if manifest.get("input_hashes") != expected_input_hashes:
        raise PipelineInvariantError("Completed DANN input hashes are incompatible")
    if expected_git_revision is not None:
        provenance = manifest.get("git", {})
        if (
            provenance.get("revision") != expected_git_revision
            or provenance.get("dirty") is not False
        ):
            raise PipelineInvariantError("Completed DANN Git revision is incompatible")
    expected_configuration = {
        "source_domain": config.source_domain,
        "adaptation_domain": config.adaptation_domain,
        "domain_loss_weight": config.domain_loss_weight,
        "gradient_reversal": config.gradient_reversal,
        "domain_head": config.domain_head,
        "augmentation": False,
    }
    configuration = manifest.get("configuration", {})
    if any(configuration.get(key) != value for key, value in expected_configuration.items()):
        raise PipelineInvariantError("Completed DANN policy differs from configuration")
    if manifest.get("data") != _pair_contract(fold):
        raise PipelineInvariantError("Completed DANN pair contract is invalid")
    if manifest.get("class_order") != list(state.classes):
        raise PipelineInvariantError("Completed DANN class order is invalid")
    preprocessing = manifest.get("preprocessing", {})
    if preprocessing.get("train_domain") != run.source_domain or preprocessing.get(
        "state"
    ) != state.to_dict():
        raise PipelineInvariantError("Completed DANN preprocessing state is invalid")
    if sha256_file(run_dir / "split_manifest.csv") != manifest.get(
        "split_manifest_sha256"
    ):
        raise PipelineInvariantError("Completed DANN split hash is invalid")
    artifacts = manifest.get("artifacts", {})
    if set(artifacts) != set(DANN_RUN_ARTIFACTS):
        raise PipelineInvariantError("Completed DANN artifact inventory is invalid")
    physical = {path.name for path in run_dir.iterdir() if path.is_file()}
    if physical != set(DANN_RUN_ARTIFACTS) | {"run.json"}:
        raise PipelineInvariantError("Completed DANN directory contains unexpected files")
    for name, metadata in artifacts.items():
        if sha256_file(run_dir / name) != metadata.get("sha256"):
            raise PipelineInvariantError(f"Completed DANN artifact hash mismatch: {name}")
    predictions = pd.read_parquet(run_dir / "predictions.parquet")
    validate_predictions(predictions, run=run, fold=fold, classes=state.classes)
    recomputed = compute_metrics(predictions, run=run, fold=fold, classes=state.classes)
    if _load_json(run_dir / "metrics.json") != {"run": run.to_dict(), "metrics": recomputed}:
        raise PipelineInvariantError("Completed DANN metrics disagree with predictions")
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
        raise PipelineInvariantError("Completed DANN long metrics disagree") from error
    model_parameters = configuration.get("model_hyperparameters", {})
    _validate_model_hyperparameters(
        config,
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
