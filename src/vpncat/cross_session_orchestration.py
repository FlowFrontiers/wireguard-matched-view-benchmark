from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from vpncat.cross_session import (
    CrossSessionRunSpec,
    enumerate_cross_session_runs,
)
from vpncat.cross_session_artifacts import (
    CrossSessionState,
    validate_completed_cross_session_run,
    validate_cross_session_run_contracts,
)
from vpncat.cross_session_index import (
    CrossSessionIndex,
    materialize_cross_session_index,
)
from vpncat.cross_session_preprocessing import (
    CrossSessionPreprocessingState,
    CrossSessionTargetState,
)
from vpncat.cross_session_preprocessing_audit import CrossSessionPreprocessingConfig
from vpncat.cross_session_runner import run_cross_session_classical
from vpncat.errors import PipelineInvariantError
from vpncat.hashing import sha256_file
from vpncat.provenance import git_provenance

if TYPE_CHECKING:
    from vpncat.neural_config import NeuralConfig
    from vpncat.neural_tuning import SelectedNeuralConfiguration


@dataclass(frozen=True)
class CrossSessionRunFilters:
    families: tuple[str, ...] = ()
    models: tuple[str, ...] = ()
    representations: tuple[str, ...] = ()
    train_sessions: tuple[int, ...] = ()
    seeds: tuple[int, ...] = ()
    run_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, list[str | int]]:
        return {
            "families": list(self.families),
            "models": list(self.models),
            "representations": list(self.representations),
            "train_sessions": list(self.train_sessions),
            "seeds": list(self.seeds),
            "run_ids": list(self.run_ids),
        }


def select_cross_session_runs(
    config: CrossSessionPreprocessingConfig,
    filters: CrossSessionRunFilters,
) -> tuple[CrossSessionRunSpec, ...]:
    runs = enumerate_cross_session_runs(config.cross_session)
    checks = (
        (set(filters.families), {run.family for run in runs}, "families"),
        (set(filters.models), {run.model for run in runs}, "models"),
        (
            set(filters.representations),
            {run.representation for run in runs},
            "representations",
        ),
        (
            {int(session) for session in filters.train_sessions},
            set(config.cross_session.sessions),
            "training sessions",
        ),
        ({int(seed) for seed in filters.seeds}, {42, 43, 44}, "seeds"),
        (set(filters.run_ids), {run.run_id for run in runs}, "run IDs"),
    )
    for requested, known, label in checks:
        unknown = requested - known
        if unknown:
            raise PipelineInvariantError(
                f"Unknown cross-session {label}: {sorted(unknown)}"
            )
    selected = tuple(
        run
        for run in runs
        if (not filters.families or run.family in filters.families)
        and (not filters.models or run.model in filters.models)
        and (
            not filters.representations
            or run.representation in filters.representations
        )
        and (
            not filters.train_sessions or run.train_session in filters.train_sessions
        )
        and (not filters.seeds or run.seed in filters.seeds)
        and (not filters.run_ids or run.run_id in filters.run_ids)
    )
    if not selected:
        raise PipelineInvariantError("Cross-session filters select no runs")
    return selected


def _clean_revision(config: CrossSessionPreprocessingConfig) -> str:
    provenance = git_provenance(config.project_root)
    if not provenance.get("status_available") or provenance.get("dirty"):
        raise PipelineInvariantError(
            "Cross-session orchestration requires a clean Git revision"
        )
    revision = str(provenance.get("revision", ""))
    if not revision:
        raise PipelineInvariantError("Cross-session Git revision is unavailable")
    return revision


def _load_indexes(
    config: CrossSessionPreprocessingConfig,
) -> dict[int, CrossSessionIndex]:
    metadata = pq.read_table(
        config.cross_session.primary.canonical_path,
        columns=["pair_id", "session", "application_category"],
    ).to_pandas()
    split = pd.read_csv(config.cross_session.split_path)
    return {
        session: materialize_cross_session_index(
            metadata,
            split,
            train_session=session,
        )
        for session in config.cross_session.sessions
    }


def _audited_state(
    config: CrossSessionPreprocessingConfig,
    run: CrossSessionRunSpec,
) -> CrossSessionState:
    audit = json.loads(config.audit_output.read_text(encoding="utf-8"))
    if run.representation in {"matched_flow_stats", "prefix_stats"}:
        payload = audit.get("fitted_states", {}).get(run.representation, {}).get(
            str(run.train_session)
        )
        if payload is None:
            raise PipelineInvariantError("Cross-session fitted state is absent")
        return CrossSessionPreprocessingState(
            train_session=int(payload["train_session"]),
            train_domain=str(payload["train_domain"]),
            representation=str(payload["representation"]),
            feature_names=tuple(str(value) for value in payload["feature_names"]),
            medians=np.asarray(payload["medians"], dtype=np.float64),
            classes=tuple(str(value) for value in payload["classes"]),
            class_weights=np.asarray(payload["class_weights"], dtype=np.float64),
            fit_pair_count=int(payload["fit_pair_count"]),
            fit_pair_ids_sha256=str(payload["fit_pair_ids_sha256"]),
        )
    payload = audit.get("directions", {}).get(str(run.train_session), {}).get(
        "targets"
    )
    if payload is None:
        raise PipelineInvariantError("Cross-session target state is absent")
    return CrossSessionTargetState(
        train_session=int(payload["train_session"]),
        classes=tuple(str(value) for value in payload["classes"]),
        class_weights=np.asarray(payload["class_weights"], dtype=np.float64),
        fit_pair_count=int(payload["fit_pair_count"]),
        fit_pair_ids_sha256=str(payload["fit_pair_ids_sha256"]),
    )


def _neural_input_hashes(
    base_hashes: dict[str, str],
    neural: NeuralConfig,
    selected: SelectedNeuralConfiguration,
) -> dict[str, str]:
    return {
        **base_hashes,
        "neural_config": sha256_file(neural.config_path),
        "neural_tuning_selection": selected.selected_sha256,
        "neural_tuning_manifest": selected.tuning_manifest_sha256,
    }


def _load_neural_selection(
    config: CrossSessionPreprocessingConfig,
    neural: NeuralConfig,
    *,
    model_name: str,
) -> SelectedNeuralConfiguration:
    from vpncat.neural_tuning import load_selected_neural_configuration

    return load_selected_neural_configuration(
        config.cross_session.primary,
        neural,
        model_name=model_name,
    )


def _run_neural(
    config: CrossSessionPreprocessingConfig,
    neural: NeuralConfig,
    run: CrossSessionRunSpec,
    *,
    device_name: str,
    selected: SelectedNeuralConfiguration,
) -> Path:
    from vpncat.cross_session_neural_runner import run_cross_session_neural

    return run_cross_session_neural(
        config,
        neural,
        run,
        device_name=device_name,
        selected=selected,
    )


def _run_xgboost_isolated(
    config: CrossSessionPreprocessingConfig,
    run: CrossSessionRunSpec,
) -> Path:
    command = [
        sys.executable,
        "-m",
        "vpncat.isolated_cross_session_classical",
        "--config",
        str(config.config_path),
        "--artifact-dir",
        str(config.cross_session.primary.canonical_path.parent),
        "--output-root",
        str(config.cross_session.output_root),
        "--experiment-id",
        run.experiment_id,
        "--train-session",
        str(run.train_session),
        "--seed",
        str(run.seed),
    ]
    completed = subprocess.run(command, check=False, text=True, capture_output=True)
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "no output"
        raise PipelineInvariantError(
            f"Isolated cross-session XGBoost failed with exit "
            f"{completed.returncode}: {detail}"
        )
    output = config.cross_session.output_root / run.relative_output_dir
    if not output.is_dir():
        raise PipelineInvariantError(
            "Isolated cross-session XGBoost did not publish output"
        )
    return output


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def run_cross_session_matrix(
    config: CrossSessionPreprocessingConfig,
    neural: NeuralConfig,
    *,
    filters: CrossSessionRunFilters | None = None,
    execute: bool = False,
    maximum_pending_runs: int | None = None,
    device_name: str = "auto",
    report_path: Path | None = None,
) -> dict[str, Any]:
    if maximum_pending_runs is not None and maximum_pending_runs < 1:
        raise PipelineInvariantError("maximum_pending_runs must be positive")
    filters = CrossSessionRunFilters() if filters is None else filters
    revision = _clean_revision(config)
    selected_runs = select_cross_session_runs(config, filters)
    base_hashes, _ = validate_cross_session_run_contracts(config, selected_runs)
    indexes = _load_indexes(config)
    pending_candidates = [
        run
        for run in selected_runs
        if not (config.cross_session.output_root / run.relative_output_dir).exists()
    ]
    execution_candidates = (
        pending_candidates[:maximum_pending_runs]
        if maximum_pending_runs
        else pending_candidates
    )

    existing_neural_models = {
        run.model
        for run in selected_runs
        if run.family == "neural"
        and (config.cross_session.output_root / run.relative_output_dir).exists()
    }
    pending_neural_models = {
        run.model
        for run in execution_candidates
        if execute and run.family == "neural"
    }
    required_neural_models = existing_neural_models | pending_neural_models
    selected_neural: dict[str, Any] = {
        model: _load_neural_selection(config, neural, model_name=model)
        for model in sorted(required_neural_models)
    }
    environments = {
        (
            selected.tuning_device,
            json.dumps(selected.tuning_environment, sort_keys=True),
        )
        for selected in selected_neural.values()
    }
    if len(environments) > 1:
        raise PipelineInvariantError(
            "Cross-session neural selections use different tuning environments"
        )

    rows: list[dict[str, Any]] = []
    pending: list[CrossSessionRunSpec] = []
    for run in selected_runs:
        output = config.cross_session.output_root / run.relative_output_dir
        if not output.exists():
            pending.append(run)
            rows.append(
                {
                    "run_id": run.run_id,
                    "family": run.family,
                    "model": run.model,
                    "train_session": run.train_session,
                    "test_session": run.test_session,
                    "seed": run.seed,
                    "output": str(output),
                    "status": "pending",
                }
            )
            continue
        state = _audited_state(config, run)
        expected_hashes = base_hashes
        if run.family == "neural":
            expected_hashes = _neural_input_hashes(
                base_hashes,
                neural,
                selected_neural[run.model],
            )
        try:
            validation = validate_completed_cross_session_run(
                output,
                config=config,
                run=run,
                index=indexes[run.train_session],
                state=state,
                classes=state.classes,
                expected_input_hashes=expected_hashes,
                verified_base_hashes=base_hashes,
                expected_git_revision=revision,
            )
        except Exception as error:
            raise PipelineInvariantError(
                f"Existing cross-session output is incompatible: {run.run_id}"
            ) from error
        rows.append(
            {
                "run_id": run.run_id,
                "family": run.family,
                "model": run.model,
                "train_session": run.train_session,
                "test_session": run.test_session,
                "seed": run.seed,
                "output": str(output),
                "status": "complete",
                "prediction_rows": validation["prediction_rows"],
            }
        )

    execution_set = execution_candidates
    executed_ids: set[str] = set()
    if execute:
        for run in execution_set:
            if run.model == "xgboost":
                output = _run_xgboost_isolated(config, run)
                expected_hashes = base_hashes
            elif run.family == "classical":
                output = run_cross_session_classical(config, run)
                expected_hashes = base_hashes
            else:
                selected = selected_neural[run.model]
                output = _run_neural(
                    config,
                    neural,
                    run,
                    device_name=device_name,
                    selected=selected,
                )
                expected_hashes = _neural_input_hashes(
                    base_hashes, neural, selected
                )
            state = _audited_state(config, run)
            validate_completed_cross_session_run(
                output,
                config=config,
                run=run,
                index=indexes[run.train_session],
                state=state,
                classes=state.classes,
                expected_input_hashes=expected_hashes,
                verified_base_hashes=base_hashes,
                expected_git_revision=revision,
            )
            executed_ids.add(run.run_id)

    for row in rows:
        if row["run_id"] in executed_ids:
            row["status"] = "executed"
    counts = {
        status: sum(row["status"] == status for row in rows)
        for status in ("complete", "executed", "pending")
    }
    payload = {
        "schema_version": 1,
        "created_utc": datetime.now(UTC).isoformat(),
        "git_revision": revision,
        "execute": execute,
        "filters": filters.to_dict(),
        "selected_run_count": len(selected_runs),
        "counts": counts,
        "runs": rows,
        "status": "complete" if counts["pending"] == 0 else "partial",
    }
    if report_path is not None:
        _write_report(report_path, payload)
    return payload
