from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
import pyarrow.parquet as pq

from vpncat.artifacts import validate_completed_run, verify_input_chain
from vpncat.errors import PipelineInvariantError
from vpncat.experiment import (
    DOMAINS,
    MODEL_FAMILIES,
    PrimaryExperimentConfig,
    RunSpec,
    enumerate_primary_runs,
)
from vpncat.folds import FoldIndex, materialize_fold_index
from vpncat.hashing import sha256_file
from vpncat.preprocessing import fit_fold_targets
from vpncat.primary_runner import run_primary_classical, validate_contract_audit
from vpncat.provenance import git_provenance
from vpncat.splits import validate_split_manifest

if TYPE_CHECKING:
    from vpncat.neural_config import NeuralConfig
    from vpncat.neural_tuning import SelectedNeuralConfiguration


@dataclass(frozen=True)
class PrimaryRunFilters:
    families: tuple[str, ...] = ()
    models: tuple[str, ...] = ()
    representations: tuple[str, ...] = ()
    folds: tuple[int, ...] = ()
    train_domains: tuple[str, ...] = ()
    seeds: tuple[int, ...] = ()
    run_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, list[str | int]]:
        return {
            "families": list(self.families),
            "models": list(self.models),
            "representations": list(self.representations),
            "folds": list(self.folds),
            "train_domains": list(self.train_domains),
            "seeds": list(self.seeds),
            "run_ids": list(self.run_ids),
        }


def select_primary_runs(
    config: PrimaryExperimentConfig,
    filters: PrimaryRunFilters,
) -> tuple[RunSpec, ...]:
    """Select a deterministic subset without changing frozen run identities."""
    all_runs = enumerate_primary_runs(config)
    known_families = set(MODEL_FAMILIES.values())
    known_models = set(MODEL_FAMILIES)
    known_representations = {run.representation for run in all_runs}
    known_run_ids = {run.run_id for run in all_runs}
    checks = (
        (set(filters.families), known_families, "families"),
        (set(filters.models), known_models, "models"),
        (set(filters.representations), known_representations, "representations"),
        (set(filters.folds), set(config.folds), "folds"),
        (set(filters.train_domains), set(DOMAINS), "training domains"),
        ({int(seed) for seed in filters.seeds}, {42, 43, 44}, "seeds"),
        (set(filters.run_ids), known_run_ids, "run IDs"),
    )
    for requested, known, label in checks:
        unknown = requested - known
        if unknown:
            raise PipelineInvariantError(f"Unknown primary {label}: {sorted(unknown)}")

    selected = tuple(
        run
        for run in all_runs
        if (not filters.families or run.family in filters.families)
        and (not filters.models or run.model in filters.models)
        and (not filters.representations or run.representation in filters.representations)
        and (not filters.folds or run.fold in filters.folds)
        and (not filters.train_domains or run.train_domain in filters.train_domains)
        and (not filters.seeds or run.seed in filters.seeds)
        and (not filters.run_ids or run.run_id in filters.run_ids)
    )
    if not selected:
        raise PipelineInvariantError("Primary filters select no runs")
    return selected


def _load_fold_indexes(config: PrimaryExperimentConfig) -> dict[int, FoldIndex]:
    metadata = pq.read_table(
        config.canonical_path,
        columns=["pair_id", "session", "application_category"],
    ).to_pandas()
    split = pd.read_csv(config.split_path)
    validate_split_manifest(split, folds=len(config.folds))
    return {
        fold: materialize_fold_index(metadata, split, fold=fold)
        for fold in config.folds
    }


def _clean_revision(config: PrimaryExperimentConfig) -> str:
    provenance = git_provenance(config.project_root)
    if not provenance.get("status_available") or provenance.get("dirty"):
        raise PipelineInvariantError("Primary matrix orchestration requires a clean Git revision")
    revision = str(provenance.get("revision", ""))
    if not revision:
        raise PipelineInvariantError("Primary matrix Git revision is unavailable")
    return revision


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


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _run_xgboost_isolated(
    config: PrimaryExperimentConfig,
    run: RunSpec,
) -> Path:
    """Run XGBoost outside processes that may load PyTorch's OpenMP runtime."""
    command = [
        sys.executable,
        "-m",
        "vpncat.isolated_classical",
        "--config",
        str(config.config_path),
        "--artifact-dir",
        str(config.canonical_path.parent),
        "--output-root",
        str(config.output_root),
        "--experiment-id",
        run.experiment_id,
        "--fold",
        str(run.fold),
        "--train-domain",
        run.train_domain,
        "--seed",
        str(run.seed),
    ]
    completed = subprocess.run(command, check=False, text=True, capture_output=True)
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "no diagnostic output"
        raise PipelineInvariantError(
            f"Isolated XGBoost run failed with exit {completed.returncode}: {detail}"
        )
    output = config.output_root / run.relative_output_dir
    if not output.is_dir():
        raise PipelineInvariantError("Isolated XGBoost run did not publish its output")
    return output


def _load_neural_selection(
    primary: PrimaryExperimentConfig,
    neural: NeuralConfig,
    *,
    model_name: str,
) -> SelectedNeuralConfiguration:
    from vpncat.neural_tuning import load_selected_neural_configuration

    return load_selected_neural_configuration(
        primary,
        neural,
        model_name=model_name,
    )


def _run_neural(
    primary: PrimaryExperimentConfig,
    neural: NeuralConfig,
    run: RunSpec,
    *,
    device_name: str,
    selected: SelectedNeuralConfiguration,
) -> Path:
    from vpncat.neural_runner import run_primary_neural

    return run_primary_neural(
        primary,
        neural,
        run,
        device_name=device_name,
        selected=selected,
    )


def run_primary_matrix(
    primary: PrimaryExperimentConfig,
    neural: NeuralConfig,
    *,
    filters: PrimaryRunFilters | None = None,
    execute: bool = False,
    maximum_pending_runs: int | None = None,
    device_name: str = "auto",
    report_path: Path | None = None,
) -> dict[str, Any]:
    """Preflight, resume, and optionally execute a filtered primary run subset."""
    if maximum_pending_runs is not None and maximum_pending_runs < 1:
        raise PipelineInvariantError("maximum_pending_runs must be positive")
    filters = PrimaryRunFilters() if filters is None else filters
    revision = _clean_revision(primary)
    selected_runs = select_primary_runs(primary, filters)
    for run in selected_runs:
        validate_contract_audit(primary, run)
    folds = _load_fold_indexes(primary)
    base_hashes = verify_input_chain(primary)

    existing_neural_models = {
        run.model
        for run in selected_runs
        if run.family == "neural" and (primary.output_root / run.relative_output_dir).exists()
    }
    pending_neural_models = {
        run.model
        for run in selected_runs
        if execute
        and run.family == "neural"
        and not (primary.output_root / run.relative_output_dir).exists()
    }
    selected_neural: dict[str, Any] = {}
    required_neural_models = existing_neural_models | pending_neural_models
    if required_neural_models:
        selected_neural = {
            model: _load_neural_selection(primary, neural, model_name=model)
            for model in sorted(required_neural_models)
        }
    tuning_environments = {
        (
            selected.tuning_device,
            json.dumps(selected.tuning_environment, sort_keys=True),
        )
        for selected in selected_neural.values()
    }
    if len(tuning_environments) > 1:
        raise PipelineInvariantError(
            "Selected neural models were tuned in different environments"
        )

    rows: list[dict[str, Any]] = []
    pending: list[RunSpec] = []
    for run in selected_runs:
        output = primary.output_root / run.relative_output_dir
        if not output.exists():
            pending.append(run)
            rows.append(
                {
                    "run_id": run.run_id,
                    "family": run.family,
                    "model": run.model,
                    "fold": run.fold,
                    "train_domain": run.train_domain,
                    "seed": run.seed,
                    "output": str(output),
                    "status": "pending",
                }
            )
            continue
        expected_hashes = base_hashes
        if run.family == "neural":
            expected_hashes = _neural_input_hashes(
                base_hashes,
                neural,
                selected_neural[run.model],
            )
        fold = folds[run.fold]
        try:
            validation = validate_completed_run(
                output,
                run=run,
                fold=fold,
                classes=fit_fold_targets(fold).classes,
                expected_input_hashes=expected_hashes,
                expected_git_revision=revision,
            )
        except Exception as error:
            raise PipelineInvariantError(
                f"Existing primary output is incompatible: {run.run_id}"
            ) from error
        rows.append(
            {
                "run_id": run.run_id,
                "family": run.family,
                "model": run.model,
                "fold": run.fold,
                "train_domain": run.train_domain,
                "seed": run.seed,
                "output": str(output),
                "status": "complete",
                "prediction_rows": validation["prediction_rows"],
            }
        )

    execution_set = pending[:maximum_pending_runs] if maximum_pending_runs else pending
    executed_ids: set[str] = set()
    if execute:
        for run in execution_set:
            if run.model == "xgboost":
                output = _run_xgboost_isolated(primary, run)
                expected_hashes = base_hashes
            elif run.family == "classical":
                output = run_primary_classical(primary, run)
                expected_hashes = base_hashes
            else:
                selected = selected_neural[run.model]
                output = _run_neural(
                    primary,
                    neural,
                    run,
                    device_name=device_name,
                    selected=selected,
                )
                expected_hashes = _neural_input_hashes(base_hashes, neural, selected)
            fold = folds[run.fold]
            validate_completed_run(
                output,
                run=run,
                fold=fold,
                classes=fit_fold_targets(fold).classes,
                expected_input_hashes=expected_hashes,
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
