from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from vpncat.ablation_artifacts import (
    ablation_input_hashes,
    validate_ablation_run_contracts,
    validate_completed_ablation_run,
)
from vpncat.ablations import (
    AblationConfig,
    AblationRunSpec,
    enumerate_ablation_runs,
    primary_reference_run,
)
from vpncat.artifacts import validate_completed_run, verify_input_chain
from vpncat.errors import PipelineInvariantError
from vpncat.folds import FoldIndex, materialize_fold_index
from vpncat.hashing import sha256_file
from vpncat.preprocessing import fit_fold_targets
from vpncat.provenance import git_provenance


@dataclass(frozen=True)
class AblationRunFilters:
    models: tuple[str, ...] = ()
    observations: tuple[str, ...] = ()
    folds: tuple[int, ...] = ()
    run_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, list[str | int]]:
        return {
            "models": list(self.models),
            "observations": list(self.observations),
            "folds": list(self.folds),
            "run_ids": list(self.run_ids),
        }


def select_ablation_runs(
    config: AblationConfig,
    filters: AblationRunFilters,
) -> tuple[AblationRunSpec, ...]:
    all_runs = enumerate_ablation_runs(config)
    unknown_models = set(filters.models) - set(config.models)
    unknown_observations = set(filters.observations) - {
        run.observation_id for run in all_runs
    }
    unknown_folds = set(filters.folds) - set(config.folds)
    unknown_ids = set(filters.run_ids) - {run.run_id for run in all_runs}
    if unknown_models or unknown_observations or unknown_folds or unknown_ids:
        raise PipelineInvariantError("Ablation filters contain unknown values")
    selected = tuple(
        run
        for run in all_runs
        if (not filters.models or run.model in filters.models)
        and (not filters.observations or run.observation_id in filters.observations)
        and (not filters.folds or run.fold in filters.folds)
        and (not filters.run_ids or run.run_id in filters.run_ids)
    )
    if not selected:
        raise PipelineInvariantError("Ablation filters select no runs")
    return selected


def _clean_revision(config: AblationConfig) -> str:
    provenance = git_provenance(config.project_root)
    if not provenance.get("status_available") or provenance.get("dirty"):
        raise PipelineInvariantError("Ablation matrix requires a clean Git revision")
    return str(provenance["revision"])


def _folds(config: AblationConfig) -> dict[int, FoldIndex]:
    metadata = pq.read_table(
        config.primary.canonical_path,
        columns=["pair_id", "session", "application_category"],
    ).to_pandas()
    split = pd.read_csv(config.primary.split_path)
    return {
        fold: materialize_fold_index(metadata, split, fold=fold)
        for fold in config.folds
    }


def _load_selection(config: AblationConfig, model: str) -> Any:
    from vpncat.neural_tuning import load_selected_neural_configuration

    return load_selected_neural_configuration(
        config.primary,
        config.neural,
        model_name=model,
    )


def _primary_input_hashes(config: AblationConfig, selected: Any) -> dict[str, str]:
    return {
        **verify_input_chain(config.primary),
        "neural_config": sha256_file(config.neural.config_path),
        "neural_tuning_selection": selected.selected_sha256,
        "neural_tuning_manifest": selected.tuning_manifest_sha256,
    }


def _execute(
    config: AblationConfig,
    run: AblationRunSpec,
    *,
    device_name: str,
    selected: Any,
) -> Path:
    from vpncat.ablation_runner import run_ablation

    return run_ablation(config, run, device_name=device_name, selected=selected)


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def run_ablation_matrix(
    config: AblationConfig,
    *,
    filters: AblationRunFilters | None = None,
    execute: bool = False,
    maximum_pending_runs: int | None = None,
    device_name: str = "auto",
    report_path: Path | None = None,
) -> dict[str, Any]:
    if maximum_pending_runs is not None and maximum_pending_runs < 1:
        raise PipelineInvariantError("maximum_pending_runs must be positive")
    filters = AblationRunFilters() if filters is None else filters
    revision = _clean_revision(config)
    selected_runs = select_ablation_runs(config, filters)
    base_hashes = validate_ablation_run_contracts(config, selected_runs)
    folds = _folds(config)
    executable = tuple(run for run in selected_runs if not run.is_primary_reference)
    pending_candidates = [
        run
        for run in executable
        if not (config.output_root / run.relative_output_dir).exists()
    ]
    execution_set = (
        pending_candidates[:maximum_pending_runs]
        if maximum_pending_runs is not None
        else pending_candidates
    )
    models_requiring_selection = {
        run.model
        for run in selected_runs
        if (
            execute and run in execution_set
        )
        or (
            run.is_primary_reference
            and (
                config.primary.output_root
                / primary_reference_run(config, run).relative_output_dir
            ).exists()
        )
        or (
            not run.is_primary_reference
            and (config.output_root / run.relative_output_dir).exists()
        )
    }
    selections = {
        model: _load_selection(config, model)
        for model in sorted(models_requiring_selection)
    }

    rows: list[dict[str, Any]] = []
    for run in selected_runs:
        fold = folds[run.fold]
        state = fit_fold_targets(fold)
        common = {
            "run_id": run.run_id,
            "model": run.model,
            "observation_id": run.observation_id,
            "prefix_length": run.prefix_length,
            "channels": list(run.channels),
            "fold": run.fold,
            "seed": run.seed,
            "prediction_rows": len(fold.test_positions) * len(run.test_domains),
        }
        if run.is_primary_reference:
            reference = primary_reference_run(config, run)
            output = config.primary.output_root / reference.relative_output_dir
            if not output.exists():
                rows.append(
                    {
                        **common,
                        "artifact_source": "primary",
                        "referenced_run_id": reference.run_id,
                        "output": str(output),
                        "status": "reference_pending",
                    }
                )
                continue
            selected = selections.get(run.model)
            if selected is None:
                raise PipelineInvariantError("Selection unavailable for primary reference")
            try:
                validation = validate_completed_run(
                    output,
                    run=reference,
                    fold=fold,
                    classes=state.classes,
                    expected_input_hashes=_primary_input_hashes(config, selected),
                    expected_git_revision=revision,
                )
            except Exception as error:
                raise PipelineInvariantError(
                    f"Primary ablation reference is incompatible: {reference.run_id}"
                ) from error
            rows.append(
                {
                    **common,
                    "artifact_source": "primary",
                    "referenced_run_id": reference.run_id,
                    "output": str(output),
                    "status": "reference_complete",
                    "prediction_rows": validation["prediction_rows"],
                }
            )
            continue
        output = config.output_root / run.relative_output_dir
        if not output.exists():
            rows.append(
                {
                    **common,
                    "artifact_source": "ablation",
                    "referenced_run_id": None,
                    "output": str(output),
                    "status": "pending",
                }
            )
            continue
        selected = selections.get(run.model)
        if selected is None:
            raise PipelineInvariantError("Selection unavailable for existing ablation")
        try:
            validation = validate_completed_ablation_run(
                output,
                config=config,
                run=run,
                fold=fold,
                state=state,
                expected_input_hashes=ablation_input_hashes(base_hashes, selected),
                expected_git_revision=revision,
            )
        except Exception as error:
            raise PipelineInvariantError(
                f"Existing ablation output is incompatible: {run.run_id}"
            ) from error
        rows.append(
            {
                **common,
                "artifact_source": "ablation",
                "referenced_run_id": None,
                "output": str(output),
                "status": "complete",
                "prediction_rows": validation["prediction_rows"],
            }
        )

    executed_ids: set[str] = set()
    if execute:
        for run in execution_set:
            selected = selections.get(run.model)
            if selected is None:
                raise PipelineInvariantError("Ablation execution requires tuning selection")
            output = _execute(
                config,
                run,
                device_name=device_name,
                selected=selected,
            )
            fold = folds[run.fold]
            validate_completed_ablation_run(
                output,
                config=config,
                run=run,
                fold=fold,
                state=fit_fold_targets(fold),
                expected_input_hashes=ablation_input_hashes(base_hashes, selected),
                expected_git_revision=revision,
            )
            executed_ids.add(run.run_id)
    for row in rows:
        if row["run_id"] in executed_ids:
            row["status"] = "executed"

    statuses = (
        "complete",
        "executed",
        "pending",
        "reference_complete",
        "reference_pending",
    )
    counts = {status: sum(row["status"] == status for row in rows) for status in statuses}
    payload = {
        "schema_version": 1,
        "created_utc": datetime.now(UTC).isoformat(),
        "git_revision": revision,
        "protocol": config.protocol,
        "kind": config.kind,
        "execute": execute,
        "filters": filters.to_dict(),
        "selected_cell_count": len(selected_runs),
        "selected_training_count": len(executable),
        "selected_reference_count": len(selected_runs) - len(executable),
        "counts": counts,
        "runs": rows,
        "status": (
            "complete"
            if counts["pending"] == 0 and counts["reference_pending"] == 0
            else "partial"
        ),
    }
    if report_path is not None:
        _write_report(report_path, payload)
    return payload
