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
from vpncat.artifacts import HISTORY_COLUMNS
from vpncat.cross_session import (
    CrossSessionRunSpec,
    enumerate_cross_session_runs,
    validate_cross_session_contract,
)
from vpncat.cross_session_index import CrossSessionIndex
from vpncat.cross_session_metrics import (
    CROSS_SESSION_PREDICTION_COLUMNS,
    compute_cross_session_metrics,
    cross_session_metrics_long_frame,
    validate_cross_session_predictions,
)
from vpncat.cross_session_preprocessing import (
    CrossSessionPreprocessingState,
    CrossSessionTargetState,
)
from vpncat.cross_session_preprocessing_audit import CrossSessionPreprocessingConfig
from vpncat.errors import PipelineInvariantError
from vpncat.hashing import sha256_file
from vpncat.preprocessing import pair_id_digest
from vpncat.provenance import git_provenance

CROSS_SESSION_RUN_ARTIFACTS = (
    "split_manifest.csv",
    "metrics.json",
    "metrics_long.csv",
    "predictions.parquet",
)
CrossSessionState = CrossSessionPreprocessingState | CrossSessionTargetState


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _environment_versions(*, include_neural_runtime: bool) -> dict[str, str]:
    packages = ("numpy", "pandas", "pyarrow", "scikit-learn", "xgboost", "torch")
    versions = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    if include_neural_runtime and versions["torch"] != "not-installed":
        import torch

        versions["cuda_runtime"] = str(torch.version.cuda)
        versions["cudnn"] = str(torch.backends.cudnn.version())
        versions["mps_built"] = str(torch.backends.mps.is_built())
        versions["mps_available"] = str(torch.backends.mps.is_available())
        if torch.cuda.is_available():
            versions["cuda_device"] = torch.cuda.get_device_name(
                torch.cuda.current_device()
            )
    return versions


def verify_cross_session_input_chain(
    config: CrossSessionPreprocessingConfig,
) -> dict[str, str]:
    cross = config.cross_session
    validate_cross_session_contract(cross)
    preprocessing_audit = _load_json(config.audit_output)
    expected_preprocessing_hashes = {
        "preprocessing_config": sha256_file(config.config_path),
        "cross_session_config": sha256_file(cross.config_path),
        "canonical": sha256_file(cross.primary.canonical_path),
        "cross_session_split": sha256_file(cross.split_path),
        "cross_session_contract_audit": sha256_file(cross.contract_audit_path),
    }
    if (
        preprocessing_audit.get("status") != "valid"
        or preprocessing_audit.get("package_version") != __version__
        or preprocessing_audit.get("input_hashes") != expected_preprocessing_hashes
    ):
        raise PipelineInvariantError("Cross-session preprocessing audit is stale")
    contract_audit = _load_json(cross.contract_audit_path)
    hashes = {
        **contract_audit["input_hashes"],
        "cross_session_split": expected_preprocessing_hashes["cross_session_split"],
        "cross_session_contract_audit": expected_preprocessing_hashes[
            "cross_session_contract_audit"
        ],
        "cross_session_preprocessing_config": expected_preprocessing_hashes[
            "preprocessing_config"
        ],
        "cross_session_preprocessing_audit": sha256_file(config.audit_output),
    }
    if len(set(hashes)) != len(hashes):
        raise PipelineInvariantError("Cross-session input hash names collide")
    return hashes


def validate_cross_session_run_contract(
    config: CrossSessionPreprocessingConfig,
    run: CrossSessionRunSpec,
) -> dict[str, Any]:
    _, rows = validate_cross_session_run_contracts(config, (run,))
    return rows[run.run_id]


def validate_cross_session_run_contracts(
    config: CrossSessionPreprocessingConfig,
    runs: tuple[CrossSessionRunSpec, ...],
) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    input_hashes = verify_cross_session_input_chain(config)
    audit = _load_json(config.cross_session.contract_audit_path)
    audited_rows = audit.get("runs", [])
    rows: dict[str, dict[str, Any]] = {}
    for run in runs:
        matches = [row for row in audited_rows if row.get("run_id") == run.run_id]
        if len(matches) != 1:
            raise PipelineInvariantError(
                "Cross-session run is absent or duplicated in audit"
            )
        row = matches[0]
        if row.get("relative_output_dir") != run.relative_output_dir.as_posix():
            raise PipelineInvariantError("Cross-session audited output path is invalid")
        rows[run.run_id] = row
    return input_hashes, rows


def bind_cross_session_state(
    config: CrossSessionPreprocessingConfig,
    run: CrossSessionRunSpec,
    index: CrossSessionIndex,
    state: CrossSessionState,
) -> dict[str, Any]:
    expected_digest = pair_id_digest(index.pair_ids_for("train"))
    if (
        state.train_session != run.train_session
        or state.fit_pair_count != len(index.train_positions)
        or state.fit_pair_ids_sha256 != expected_digest
    ):
        raise PipelineInvariantError("Cross-session state does not bind source training pairs")
    audit = _load_json(config.audit_output)
    if run.representation in {"matched_flow_stats", "prefix_stats"}:
        if not isinstance(state, CrossSessionPreprocessingState):
            raise PipelineInvariantError("Cross-session statistical run lacks median state")
        if state.train_domain != run.train_domain or state.representation != run.representation:
            raise PipelineInvariantError("Cross-session fitted state identity is invalid")
        audited = audit.get("fitted_states", {}).get(run.representation, {}).get(
            str(run.train_session)
        )
        if audited is None:
            raise PipelineInvariantError("Cross-session fitted state is absent from audit")
        payload = state.to_dict()
        for key in (
            "train_session",
            "train_domain",
            "representation",
            "feature_names",
            "medians",
            "classes",
            "class_weights",
            "fit_pair_count",
            "fit_pair_ids_sha256",
        ):
            if payload[key] != audited.get(key):
                raise PipelineInvariantError(
                    f"Cross-session state differs from audited field {key}"
                )
        return {
            "train_session": run.train_session,
            "train_domain": run.train_domain,
            "feature_transform": "training-median",
            "state": payload,
        }

    if not isinstance(state, CrossSessionTargetState):
        raise PipelineInvariantError("Cross-session SPLT run lacks target state")
    audited = audit.get("directions", {}).get(str(run.train_session), {}).get("targets")
    if audited is None or state.to_dict() != audited:
        raise PipelineInvariantError("Cross-session target state differs from audit")
    return {
        "train_session": run.train_session,
        "train_domain": run.train_domain,
        "feature_transform": "fit-free",
        "fixed_transforms": ["direction-remap", "log1p-size", "log1p-iat"],
        "state": state.to_dict(),
    }


def _build_manifest(
    config: CrossSessionPreprocessingConfig,
    run: CrossSessionRunSpec,
    index: CrossSessionIndex,
    state: CrossSessionState,
    *,
    model_hyperparameters: dict[str, Any],
    additional_input_hashes: dict[str, str] | None = None,
) -> dict[str, Any]:
    if run.run_id not in {
        candidate.run_id for candidate in enumerate_cross_session_runs(config.cross_session)
    }:
        raise PipelineInvariantError("Run is not part of the cross-session base matrix")
    validate_cross_session_run_contract(config, run)
    input_hashes = verify_cross_session_input_chain(config)
    for name, digest in (additional_input_hashes or {}).items():
        if name in input_hashes:
            raise PipelineInvariantError(f"Cross-session input hash collides: {name}")
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise PipelineInvariantError(
                f"Cross-session additional input is not SHA-256: {name}"
            )
        input_hashes[name] = digest
    preprocessing = bind_cross_session_state(config, run, index, state)
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
        "split_manifest_sha256": input_hashes["cross_session_split"],
        "class_order": list(state.classes),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _validate_training_history(
    history: pd.DataFrame | None,
    *,
    family: str,
) -> pd.DataFrame | None:
    if family == "classical":
        if history is not None:
            raise PipelineInvariantError("Classical cross-session run has training history")
        return None
    if history is None:
        raise PipelineInvariantError("Neural cross-session run lacks training history")
    if tuple(history.columns) != HISTORY_COLUMNS:
        raise PipelineInvariantError("Cross-session training-history schema is invalid")
    if history.empty or history["epoch"].duplicated().any():
        raise PipelineInvariantError("Cross-session training history is empty or duplicated")
    numeric = history.loc[:, HISTORY_COLUMNS].apply(pd.to_numeric, errors="coerce")
    if not pd.notna(numeric).all().all():
        raise PipelineInvariantError("Cross-session training history contains missing values")
    values = numeric.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise PipelineInvariantError("Cross-session training history is not finite")
    return numeric


def write_completed_cross_session_run(
    config: CrossSessionPreprocessingConfig,
    run: CrossSessionRunSpec,
    index: CrossSessionIndex,
    state: CrossSessionState,
    predictions: pd.DataFrame,
    *,
    model_hyperparameters: dict[str, Any],
    training_history: pd.DataFrame | None = None,
    additional_input_hashes: dict[str, str] | None = None,
) -> Path:
    manifest = _build_manifest(
        config,
        run,
        index,
        state,
        model_hyperparameters=model_hyperparameters,
        additional_input_hashes=additional_input_hashes,
    )
    classes = tuple(manifest["class_order"])
    metrics = compute_cross_session_metrics(
        predictions, run=run, index=index, classes=classes
    )
    metrics_long = cross_session_metrics_long_frame(metrics, run=run)
    history = _validate_training_history(training_history, family=run.family)
    ordered = predictions.loc[:, CROSS_SESSION_PREDICTION_COLUMNS].sort_values(
        ["test_domain", "pair_id"], ignore_index=True
    )

    target = config.cross_session.output_root / run.relative_output_dir
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite cross-session run: {target}")
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    try:
        shutil.copy2(config.cross_session.split_path, staging / "split_manifest.csv")
        ordered.to_parquet(staging / "predictions.parquet", index=False)
        _write_json(staging / "metrics.json", {"run": run.to_dict(), "metrics": metrics})
        metrics_long.to_csv(staging / "metrics_long.csv", index=False)
        if history is not None:
            history.to_csv(staging / "training_history.csv", index=False)
        artifact_names = list(CROSS_SESSION_RUN_ARTIFACTS)
        if history is not None:
            artifact_names.append("training_history.csv")
        manifest["artifacts"] = {
            name: {"path": name, "sha256": sha256_file(staging / name)}
            for name in artifact_names
        }
        manifest["status"] = "complete"
        _write_json(staging / "run.json", manifest)
        validate_completed_cross_session_run(
            staging,
            config=config,
            run=run,
            index=index,
            state=state,
            classes=classes,
            expected_input_hashes=manifest["input_hashes"],
        )
        staging.replace(target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target


def validate_completed_cross_session_run(
    run_dir: Path,
    *,
    config: CrossSessionPreprocessingConfig,
    run: CrossSessionRunSpec,
    index: CrossSessionIndex,
    state: CrossSessionState,
    classes: tuple[str, ...],
    expected_input_hashes: dict[str, str] | None = None,
    verified_base_hashes: dict[str, str] | None = None,
    expected_git_revision: str | None = None,
) -> dict[str, Any]:
    manifest = _load_json(run_dir / "run.json")
    if manifest.get("status") != "complete" or manifest.get("run") != run.to_dict():
        raise PipelineInvariantError("Cross-session run manifest identity is invalid")
    if manifest.get("package_version") != __version__:
        raise PipelineInvariantError("Cross-session package version is incompatible")
    base_hashes = (
        verify_cross_session_input_chain(config)
        if verified_base_hashes is None
        else verified_base_hashes
    )
    if expected_input_hashes is None:
        if run.family == "neural":
            raise PipelineInvariantError(
                "Neural cross-session validation requires selection input hashes"
            )
        expected_input_hashes = base_hashes
    if any(expected_input_hashes.get(name) != digest for name, digest in base_hashes.items()):
        raise PipelineInvariantError("Cross-session expected inputs omit the base chain")
    if manifest.get("input_hashes") != expected_input_hashes:
        raise PipelineInvariantError("Cross-session run inputs are incompatible")
    if expected_git_revision is not None:
        provenance = manifest.get("git", {})
        if (
            provenance.get("revision") != expected_git_revision
            or provenance.get("dirty") is not False
        ):
            raise PipelineInvariantError("Cross-session Git revision is incompatible")
    expected_preprocessing = bind_cross_session_state(config, run, index, state)
    if manifest.get("preprocessing") != expected_preprocessing:
        raise PipelineInvariantError("Cross-session preprocessing payload is invalid")
    if manifest.get("class_order") != list(classes) or tuple(state.classes) != classes:
        raise PipelineInvariantError("Cross-session class order is invalid")
    artifacts = manifest.get("artifacts", {})
    expected_artifacts = set(CROSS_SESSION_RUN_ARTIFACTS)
    if run.family == "neural":
        expected_artifacts.add("training_history.csv")
    if set(artifacts) != expected_artifacts:
        raise PipelineInvariantError("Cross-session artifact inventory is invalid")
    physical = {path.name for path in run_dir.iterdir() if path.is_file()}
    if physical != expected_artifacts | {"run.json"}:
        raise PipelineInvariantError("Cross-session run directory contains unexpected files")
    for name, metadata in artifacts.items():
        path = run_dir / name
        if not path.is_file() or sha256_file(path) != metadata.get("sha256"):
            raise PipelineInvariantError(f"Cross-session artifact hash mismatch: {name}")
    if sha256_file(run_dir / "split_manifest.csv") != manifest.get(
        "split_manifest_sha256"
    ):
        raise PipelineInvariantError("Cross-session local split hash is invalid")

    predictions = pd.read_parquet(run_dir / "predictions.parquet")
    validate_cross_session_predictions(
        predictions, run=run, index=index, classes=classes
    )
    recomputed = compute_cross_session_metrics(
        predictions, run=run, index=index, classes=classes
    )
    if _load_json(run_dir / "metrics.json") != {
        "run": run.to_dict(),
        "metrics": recomputed,
    }:
        raise PipelineInvariantError("Cross-session metrics disagree with predictions")
    expected_long = cross_session_metrics_long_frame(recomputed, run=run)
    observed_long = pd.read_csv(run_dir / "metrics_long.csv")
    try:
        pd.testing.assert_frame_equal(
            observed_long, expected_long, check_exact=False, rtol=1e-12, atol=1e-15
        )
    except AssertionError as error:
        raise PipelineInvariantError("Cross-session long metrics are invalid") from error
    history = None
    if run.family == "neural":
        history = pd.read_csv(
            run_dir / "training_history.csv", float_precision="round_trip"
        )
    _validate_training_history(history, family=run.family)
    return {
        "run_id": run.run_id,
        "prediction_rows": len(predictions),
        "test_pairs": len(index.test_positions),
        "test_domains": list(run.test_domains),
        "status": "valid",
    }
