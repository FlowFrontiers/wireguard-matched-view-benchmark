from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from vpncat import __version__
from vpncat.artifacts import verify_input_chain
from vpncat.errors import PipelineInvariantError
from vpncat.experiment import PrimaryExperimentConfig, enumerate_primary_runs
from vpncat.folds import materialize_fold_index
from vpncat.preprocessing import pair_id_digest
from vpncat.provenance import git_provenance
from vpncat.splits import validate_split_manifest


def audit_experiment_contract(config: PrimaryExperimentConfig) -> dict[str, Any]:
    """Audit the frozen primary matrix and its preprocessing-state bindings."""
    input_hashes = verify_input_chain(config)
    metadata = pq.read_table(
        config.canonical_path,
        columns=["pair_id", "session", "application_category"],
    ).to_pandas()
    split = pd.read_csv(config.split_path)
    validate_split_manifest(split, folds=len(config.folds))
    folds = {
        fold: materialize_fold_index(metadata, split, fold=fold)
        for fold in config.folds
    }
    preprocessing_audit = json.loads(
        config.preprocessing_audit_path.read_text(encoding="utf-8")
    )
    runs = enumerate_primary_runs(config)
    family_counts = Counter(run.family for run in runs)
    domain_counts = Counter(run.train_domain for run in runs)
    representation_model_counts = Counter(
        (run.representation, run.model) for run in runs
    )
    if family_counts != {"classical": 60, "neural": 90}:
        raise PipelineInvariantError(f"Primary family counts are invalid: {family_counts}")
    if domain_counts != {"inner": 75, "outer": 75}:
        raise PipelineInvariantError(f"Primary domain counts are invalid: {domain_counts}")
    if set(representation_model_counts.values()) != {10, 30}:
        raise PipelineInvariantError("Primary per-configuration run counts are invalid")

    run_rows: list[dict[str, Any]] = []
    for run in runs:
        fold = folds[run.fold]
        expected_fit_hash = pair_id_digest(fold.pair_ids_for("train"))
        if run.representation in {"matched_flow_stats", "prefix_stats"}:
            audited_state = preprocessing_audit.get("fitted_states", {}).get(
                run.representation, {}
            ).get(str(run.fold), {}).get(run.train_domain)
        else:
            audited_state = preprocessing_audit.get("folds", {}).get(
                str(run.fold), {}
            ).get("targets")
        if audited_state is None:
            raise PipelineInvariantError(
                f"No audited preprocessing state exists for {run.run_id}"
            )
        if audited_state.get("fit_pair_ids_sha256") != expected_fit_hash:
            raise PipelineInvariantError(
                f"Audited preprocessing pair hash differs for {run.run_id}"
            )
        if audited_state.get("fit_pair_count") != len(fold.train_positions):
            raise PipelineInvariantError(
                f"Audited preprocessing pair count differs for {run.run_id}"
            )
        if run.representation in {"matched_flow_stats", "prefix_stats"} and (
            audited_state.get("train_domain") != run.train_domain
        ):
            raise PipelineInvariantError(
                f"Audited preprocessing domain differs for {run.run_id}"
            )
        run_rows.append(
            {
                **run.to_dict(),
                "relative_output_dir": run.relative_output_dir.as_posix(),
                "fit_pair_count": len(fold.train_positions),
                "fit_pair_ids_sha256": expected_fit_hash,
                "test_pair_count": len(fold.test_positions),
                "prediction_rows": len(fold.test_positions) * len(run.test_domains),
            }
        )

    payload = {
        "audit_version": 1,
        "created_utc": datetime.now(UTC).isoformat(),
        "package_version": __version__,
        "git": git_provenance(config.project_root),
        "input_hashes": input_hashes,
        "matrix": {
            "training_runs": len(runs),
            "prediction_groups": sum(len(run.test_domains) for run in runs),
            "family_counts": dict(sorted(family_counts.items())),
            "train_domain_counts": dict(sorted(domain_counts.items())),
            "configuration_counts": {
                f"{representation}__{model}": count
                for (representation, model), count in sorted(
                    representation_model_counts.items()
                )
            },
        },
        "runs": run_rows,
        "status": "valid",
    }
    config.contract_audit_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = config.contract_audit_path.with_suffix(
        config.contract_audit_path.suffix + ".tmp"
    )
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(config.contract_audit_path)
    return payload
