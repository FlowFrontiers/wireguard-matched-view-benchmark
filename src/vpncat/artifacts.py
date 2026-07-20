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
from vpncat.errors import PipelineInvariantError
from vpncat.experiment import PrimaryExperimentConfig, RunSpec, enumerate_primary_runs
from vpncat.folds import FoldIndex
from vpncat.hashing import sha256_file
from vpncat.metrics import (
    PREDICTION_COLUMNS,
    compute_metrics,
    metrics_long_frame,
    validate_predictions,
)
from vpncat.preprocessing import (
    FoldPreprocessingState,
    FoldTargetState,
    pair_id_digest,
)
from vpncat.provenance import git_provenance

RUN_ARTIFACTS = (
    "split_manifest.csv",
    "metrics.json",
    "metrics_long.csv",
    "predictions.parquet",
)
HISTORY_COLUMNS = (
    "epoch",
    "train_loss",
    "validation_loss",
    "validation_macro_f1",
    "learning_rate",
)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _environment_versions(*, include_neural_runtime: bool) -> dict[str, str]:
    required = ("numpy", "pandas", "pyarrow", "scikit-learn")
    optional = ("torch", "xgboost")
    versions = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        **{name: importlib.metadata.version(name) for name in required},
    }
    for name in optional:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    if include_neural_runtime and versions["torch"] != "not-installed":
        import torch

        versions["cuda_runtime"] = str(torch.version.cuda)
        versions["cudnn"] = str(torch.backends.cudnn.version())
        versions["mps_built"] = str(torch.backends.mps.is_built())
        versions["mps_available"] = str(torch.backends.mps.is_available())
        if torch.cuda.is_available():
            versions["cuda_device"] = torch.cuda.get_device_name(torch.cuda.current_device())
    return versions


def verify_input_chain(config: PrimaryExperimentConfig) -> dict[str, str]:
    paths = {
        "primary_config": config.config_path,
        "canonical": config.canonical_path,
        "split_manifest": config.split_path,
        "dataset_manifest": config.dataset_manifest_path,
        "feature_audit": config.feature_audit_path,
        "preprocessing_audit": config.preprocessing_audit_path,
    }
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    dataset_manifest = _load_json(config.dataset_manifest_path)
    canonical_expected = dataset_manifest.get("artifacts", {}).get(
        "canonical_pairs", {}
    ).get("sha256")
    split_expected = dataset_manifest.get("artifacts", {}).get(
        "split_manifest", {}
    ).get("sha256")
    if hashes["canonical"] != canonical_expected or hashes["split_manifest"] != split_expected:
        raise PipelineInvariantError("Primary inputs disagree with the dataset manifest")
    feature_audit = _load_json(config.feature_audit_path)
    preprocessing_audit = _load_json(config.preprocessing_audit_path)
    if feature_audit.get("status") != "valid" or preprocessing_audit.get("status") != "valid":
        raise PipelineInvariantError("Feature and preprocessing audits must be valid")
    if feature_audit.get("canonical", {}).get("sha256") != hashes["canonical"]:
        raise PipelineInvariantError("Feature audit pins a different canonical dataset")
    audit_inputs = preprocessing_audit.get("inputs", {})
    if audit_inputs.get("canonical_sha256") != hashes["canonical"]:
        raise PipelineInvariantError("Preprocessing audit pins a different canonical dataset")
    if audit_inputs.get("split_sha256") != hashes["split_manifest"]:
        raise PipelineInvariantError("Preprocessing audit pins a different split manifest")
    return hashes


def bind_preprocessing_state(
    run: RunSpec,
    fold: FoldIndex,
    state: FoldPreprocessingState | FoldTargetState,
    *,
    preprocessing_audit: dict[str, Any],
) -> dict[str, Any]:
    """Bind a run to audited training-only state and reject domain mix-ups."""
    expected_hash = pair_id_digest(fold.pair_ids_for("train"))
    if state.fold != run.fold or state.fit_pair_count != len(fold.train_positions):
        raise PipelineInvariantError("Preprocessing state fold or training count mismatch")
    if state.fit_pair_ids_sha256 != expected_hash:
        raise PipelineInvariantError("Preprocessing state was not fitted on run training pairs")

    if run.representation in {"matched_flow_stats", "prefix_stats"}:
        if not isinstance(state, FoldPreprocessingState):
            raise PipelineInvariantError("Statistical runs require a fitted median state")
        if state.train_domain != run.train_domain:
            raise PipelineInvariantError(
                "Run train_domain disagrees with fitted preprocessing train_domain"
            )
        if state.representation != run.representation:
            raise PipelineInvariantError("Run and preprocessing representations disagree")
        audited = preprocessing_audit.get("fitted_states", {}).get(
            run.representation, {}
        ).get(str(run.fold), {}).get(run.train_domain)
        if audited is None:
            raise PipelineInvariantError("Preprocessing audit has no matching fitted state")
        state_payload = state.to_dict()
        for key in (
            "fold",
            "train_domain",
            "representation",
            "feature_names",
            "medians",
            "classes",
            "class_weights",
            "fit_pair_count",
            "fit_pair_ids_sha256",
        ):
            if state_payload[key] != audited.get(key):
                raise PipelineInvariantError(
                    f"Applied preprocessing state differs from audit field {key}"
                )
        return {
            "train_domain": run.train_domain,
            "feature_transform": "training-median",
            "state": state_payload,
        }

    if not isinstance(state, FoldTargetState):
        raise PipelineInvariantError("Fit-free SPLT runs require a fold target state")
    audited = preprocessing_audit.get("folds", {}).get(str(run.fold), {}).get("targets")
    if audited is None or state.to_dict() != audited:
        raise PipelineInvariantError("Applied target state differs from preprocessing audit")
    return {
        "train_domain": run.train_domain,
        "feature_transform": "fit-free",
        "fixed_transforms": ["direction-remap", "log1p-size", "log1p-iat"],
        "state": state.to_dict(),
    }


def build_run_manifest(
    config: PrimaryExperimentConfig,
    run: RunSpec,
    fold: FoldIndex,
    state: FoldPreprocessingState | FoldTargetState,
    *,
    model_hyperparameters: dict[str, Any],
    additional_input_hashes: dict[str, str] | None = None,
) -> dict[str, Any]:
    if run.run_id not in {candidate.run_id for candidate in enumerate_primary_runs(config)}:
        raise PipelineInvariantError("Run is not part of the frozen primary matrix")
    input_hashes = verify_input_chain(config)
    for name, digest in (additional_input_hashes or {}).items():
        if name in input_hashes:
            raise PipelineInvariantError(f"Additional input hash collides with {name}")
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise PipelineInvariantError(f"Additional input hash is not SHA-256: {name}")
        input_hashes[name] = digest
    preprocessing_audit = _load_json(config.preprocessing_audit_path)
    preprocessing = bind_preprocessing_state(
        run,
        fold,
        state,
        preprocessing_audit=preprocessing_audit,
    )
    return {
        "schema_version": 1,
        "status": "staging",
        "created_utc": datetime.now(UTC).isoformat(),
        "package_version": __version__,
        "git": git_provenance(config.project_root),
        "environment": _environment_versions(
            include_neural_runtime=run.family == "neural"
        ),
        "run": run.to_dict(),
        "configuration": {
            "model_hyperparameters": model_hyperparameters,
            "augmentation": False,
        },
        "preprocessing": preprocessing,
        "input_hashes": input_hashes,
        "split_manifest_sha256": input_hashes["split_manifest"],
        "class_order": list(state.classes),
    }


def _validate_history(history: pd.DataFrame | None, *, family: str) -> pd.DataFrame | None:
    if family == "classical":
        if history is not None:
            raise PipelineInvariantError("Classical runs must not write training history")
        return None
    if history is None:
        raise PipelineInvariantError("Neural runs require training_history.csv")
    if tuple(history.columns) != HISTORY_COLUMNS:
        raise PipelineInvariantError("Training-history schema differs from the frozen contract")
    if history.empty or history["epoch"].duplicated().any():
        raise PipelineInvariantError("Training history is empty or contains duplicate epochs")
    numeric = history.loc[:, HISTORY_COLUMNS].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise PipelineInvariantError("Training history contains non-finite values")
    return numeric


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_completed_run(
    config: PrimaryExperimentConfig,
    run: RunSpec,
    fold: FoldIndex,
    state: FoldPreprocessingState | FoldTargetState,
    predictions: pd.DataFrame,
    *,
    model_hyperparameters: dict[str, Any],
    training_history: pd.DataFrame | None = None,
    additional_input_hashes: dict[str, str] | None = None,
) -> Path:
    """Atomically publish one complete trained-model run; never overwrite a run."""
    manifest = build_run_manifest(
        config,
        run,
        fold,
        state,
        model_hyperparameters=model_hyperparameters,
        additional_input_hashes=additional_input_hashes,
    )
    classes = tuple(manifest["class_order"])
    metrics = compute_metrics(predictions, run=run, fold=fold, classes=classes)
    metrics_long = metrics_long_frame(metrics, run=run)
    history = _validate_history(training_history, family=run.family)
    ordered_predictions = predictions.loc[:, PREDICTION_COLUMNS].sort_values(
        ["test_domain", "pair_id"], ignore_index=True
    )

    target = config.output_root / run.relative_output_dir
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite existing run directory: {target}")
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    try:
        shutil.copy2(config.split_path, staging / "split_manifest.csv")
        ordered_predictions.to_parquet(staging / "predictions.parquet", index=False)
        _write_json(
            staging / "metrics.json",
            {"run": run.to_dict(), "metrics": metrics},
        )
        metrics_long.to_csv(staging / "metrics_long.csv", index=False)
        if history is not None:
            history.to_csv(staging / "training_history.csv", index=False)

        artifact_names = list(RUN_ARTIFACTS)
        if history is not None:
            artifact_names.append("training_history.csv")
        manifest["artifacts"] = {
            name: {"path": name, "sha256": sha256_file(staging / name)}
            for name in artifact_names
        }
        manifest["status"] = "complete"
        _write_json(staging / "run.json", manifest)
        validate_completed_run(staging, run=run, fold=fold, classes=classes)
        staging.replace(target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target


def validate_completed_run(
    run_dir: Path,
    *,
    run: RunSpec,
    fold: FoldIndex,
    classes: tuple[str, ...],
    expected_input_hashes: dict[str, str] | None = None,
    expected_git_revision: str | None = None,
) -> dict[str, Any]:
    manifest = _load_json(run_dir / "run.json")
    if manifest.get("status") != "complete" or manifest.get("run") != run.to_dict():
        raise PipelineInvariantError("Completed run manifest identity or status is invalid")
    if manifest.get("package_version") != __version__:
        raise PipelineInvariantError("Completed run package version is incompatible")
    if expected_input_hashes is not None and manifest.get("input_hashes") != expected_input_hashes:
        raise PipelineInvariantError("Completed run input hashes are incompatible")
    if expected_git_revision is not None:
        provenance = manifest.get("git", {})
        if (
            provenance.get("revision") != expected_git_revision
            or provenance.get("dirty") is not False
        ):
            raise PipelineInvariantError("Completed run Git revision is incompatible")
    preprocessing_domain = manifest.get("preprocessing", {}).get("train_domain")
    if preprocessing_domain != run.train_domain:
        raise PipelineInvariantError("Run manifest preprocessing domain mismatch")
    expected_names = set(RUN_ARTIFACTS)
    if run.family == "neural":
        expected_names.add("training_history.csv")
    artifacts = manifest.get("artifacts", {})
    if set(artifacts) != expected_names:
        raise PipelineInvariantError("Completed run artifact inventory is invalid")
    physical_files = {path.name for path in run_dir.iterdir() if path.is_file()}
    if physical_files != expected_names | {"run.json"}:
        raise PipelineInvariantError("Completed run directory contains unexpected files")
    for name, metadata in artifacts.items():
        path = run_dir / name
        if not path.is_file() or sha256_file(path) != metadata.get("sha256"):
            raise PipelineInvariantError(f"Completed run artifact hash mismatch: {name}")
    if sha256_file(run_dir / "split_manifest.csv") != manifest.get(
        "split_manifest_sha256"
    ):
        raise PipelineInvariantError("Run-local split manifest hash mismatch")

    predictions = pd.read_parquet(run_dir / "predictions.parquet")
    validate_predictions(predictions, run=run, fold=fold, classes=classes)
    recomputed = compute_metrics(predictions, run=run, fold=fold, classes=classes)
    recorded = _load_json(run_dir / "metrics.json")
    if recorded != {"run": run.to_dict(), "metrics": recomputed}:
        raise PipelineInvariantError("Recorded metrics disagree with predictions")
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
        raise PipelineInvariantError(
            "Long-form metrics disagree with predictions"
        ) from error
    history = None
    if run.family == "neural":
        history = pd.read_csv(
            run_dir / "training_history.csv", float_precision="round_trip"
        )
    _validate_history(history, family=run.family)
    return {
        "run_id": run.run_id,
        "prediction_rows": len(predictions),
        "test_pairs": len(fold.test_positions),
        "test_domains": list(run.test_domains),
        "status": "valid",
    }
