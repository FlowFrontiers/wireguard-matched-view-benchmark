from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from vpncat.dann import DANNConfig, DANNRunSpec, enumerate_dann_runs
from vpncat.dann_artifacts import (
    validate_completed_dann_run,
    validate_dann_run_contracts,
)
from vpncat.errors import PipelineInvariantError
from vpncat.folds import FoldIndex, materialize_fold_index
from vpncat.preprocessing import fit_fold_targets
from vpncat.provenance import git_provenance


@dataclass(frozen=True)
class DANNRunFilters:
    folds: tuple[int, ...] = ()
    seeds: tuple[int, ...] = ()

    def to_dict(self) -> dict[str, list[int]]:
        return {"folds": list(self.folds), "seeds": list(self.seeds)}


def select_dann_runs(
    config: DANNConfig,
    filters: DANNRunFilters,
) -> tuple[DANNRunSpec, ...]:
    unknown_folds = set(filters.folds) - set(config.folds)
    unknown_seeds = set(filters.seeds) - set(config.seeds)
    if unknown_folds or unknown_seeds:
        raise PipelineInvariantError("DANN filters contain unknown folds or seeds")
    selected = tuple(
        run
        for run in enumerate_dann_runs(config)
        if (not filters.folds or run.fold in filters.folds)
        and (not filters.seeds or run.seed in filters.seeds)
    )
    if not selected:
        raise PipelineInvariantError("DANN filters select no runs")
    return selected


def _clean_revision(config: DANNConfig) -> str:
    provenance = git_provenance(config.project_root)
    if not provenance.get("status_available") or provenance.get("dirty"):
        raise PipelineInvariantError("DANN matrix requires a clean Git revision")
    return str(provenance["revision"])


def _folds(config: DANNConfig) -> dict[int, FoldIndex]:
    metadata = pq.read_table(
        config.primary.canonical_path,
        columns=["pair_id", "session", "application_category"],
    ).to_pandas()
    split = pd.read_csv(config.primary.split_path)
    return {
        fold: materialize_fold_index(metadata, split, fold=fold)
        for fold in config.folds
    }


def _load_selection(config: DANNConfig) -> Any:
    from vpncat.neural_tuning import load_selected_neural_configuration

    return load_selected_neural_configuration(
        config.primary,
        config.neural,
        model_name="cnn1d",
    )


def _input_hashes(base: dict[str, str], selected: Any) -> dict[str, str]:
    return {
        **base,
        "neural_tuning_selection": selected.selected_sha256,
        "neural_tuning_manifest": selected.tuning_manifest_sha256,
    }


def _execute(
    config: DANNConfig,
    run: DANNRunSpec,
    *,
    device_name: str,
    selected: Any,
) -> Path:
    from vpncat.dann_runner import run_dann

    return run_dann(config, run, device_name=device_name, selected=selected)


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def run_dann_matrix(
    config: DANNConfig,
    *,
    filters: DANNRunFilters | None = None,
    execute: bool = False,
    maximum_pending_runs: int | None = None,
    device_name: str = "auto",
    report_path: Path | None = None,
) -> dict[str, Any]:
    if maximum_pending_runs is not None and maximum_pending_runs < 1:
        raise PipelineInvariantError("maximum_pending_runs must be positive")
    filters = DANNRunFilters() if filters is None else filters
    revision = _clean_revision(config)
    selected_runs = select_dann_runs(config, filters)
    base_hashes = validate_dann_run_contracts(config, selected_runs)
    folds = _folds(config)
    pending_candidates = [
        run
        for run in selected_runs
        if not (config.output_root / run.relative_output_dir).exists()
    ]
    execution_set = (
        pending_candidates[:maximum_pending_runs]
        if maximum_pending_runs
        else pending_candidates
    )
    has_existing = len(pending_candidates) != len(selected_runs)
    selected = _load_selection(config) if has_existing or (execute and execution_set) else None
    expected_hashes = _input_hashes(base_hashes, selected) if selected is not None else None

    rows: list[dict[str, Any]] = []
    pending: list[DANNRunSpec] = []
    for run in selected_runs:
        output = config.output_root / run.relative_output_dir
        if not output.exists():
            pending.append(run)
            rows.append(
                {
                    "run_id": run.run_id,
                    "fold": run.fold,
                    "seed": run.seed,
                    "source_domain": run.source_domain,
                    "adaptation_domain": run.adaptation_domain,
                    "output": str(output),
                    "status": "pending",
                    "prediction_rows": len(folds[run.fold].test_positions)
                    * len(run.test_domains),
                }
            )
            continue
        if expected_hashes is None:
            raise PipelineInvariantError("DANN selection is unavailable for existing output")
        fold = folds[run.fold]
        state = fit_fold_targets(fold)
        try:
            validation = validate_completed_dann_run(
                output,
                config=config,
                run=run,
                fold=fold,
                state=state,
                expected_input_hashes=expected_hashes,
                expected_git_revision=revision,
            )
        except Exception as error:
            raise PipelineInvariantError(
                f"Existing DANN output is incompatible: {run.run_id}"
            ) from error
        rows.append(
            {
                "run_id": run.run_id,
                "fold": run.fold,
                "seed": run.seed,
                "source_domain": run.source_domain,
                "adaptation_domain": run.adaptation_domain,
                "output": str(output),
                "status": "complete",
                "prediction_rows": validation["prediction_rows"],
            }
        )

    executed_ids: set[str] = set()
    if execute:
        if selected is None or expected_hashes is None:
            raise PipelineInvariantError("DANN execution requires a tuning selection")
        for run in execution_set:
            output = _execute(
                config,
                run,
                device_name=device_name,
                selected=selected,
            )
            fold = folds[run.fold]
            state = fit_fold_targets(fold)
            validate_completed_dann_run(
                output,
                config=config,
                run=run,
                fold=fold,
                state=state,
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
