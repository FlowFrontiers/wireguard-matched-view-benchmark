from __future__ import annotations

import json
import shutil
import tempfile
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from vpncat import __version__
from vpncat.analysis import AnalysisConfig, validate_analysis_contract
from vpncat.analysis_groups import AnalysisGroup, ensemble_partition, enumerate_analysis_groups
from vpncat.analysis_statistics import compute_analysis_metrics, paired_bootstrap_intervals
from vpncat.errors import PipelineInvariantError
from vpncat.hashing import sha256_file
from vpncat.provenance import git_provenance


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _output_roots(config: AnalysisConfig) -> dict[str, Path]:
    return {
        "primary": config.primary.output_root,
        "cross_session": config.cross_session.output_root,
        "dann": config.dann.output_root,
        "ablation_prefix": config.ablation_prefix.output_root,
        "ablation_channels": config.ablation_channels.output_root,
    }


def _validate_campaign(config: AnalysisConfig) -> str:
    provenance = git_provenance(config.project_root)
    if not provenance.get("status_available") or provenance.get("dirty"):
        raise PipelineInvariantError("Result aggregation requires a clean Git revision")
    from vpncat.ablation_orchestration import run_ablation_matrix
    from vpncat.cross_session_orchestration import run_cross_session_matrix
    from vpncat.cross_session_preprocessing_audit import (
        load_cross_session_preprocessing_config,
    )
    from vpncat.dann_orchestration import run_dann_matrix
    from vpncat.orchestration import run_primary_matrix

    artifact_dir = config.primary.canonical_path.parent
    cross_preprocessing = load_cross_session_preprocessing_config(
        config.project_root / "configs" / "cross_session_preprocessing.yaml",
        artifact_dir=artifact_dir,
        output_root=config.cross_session.output_root,
    )
    reports = {
        "primary": run_primary_matrix(config.primary, config.dann.neural),
        "cross_session": run_cross_session_matrix(cross_preprocessing, config.dann.neural),
        "dann": run_dann_matrix(config.dann),
        "ablation_prefix": run_ablation_matrix(config.ablation_prefix),
        "ablation_channels": run_ablation_matrix(config.ablation_channels),
    }
    if any(report.get("status") != "complete" for report in reports.values()):
        pending = {name: report.get("counts") for name, report in reports.items()}
        raise PipelineInvariantError(f"Analysis campaign is incomplete: {pending}")
    return str(provenance["revision"])


def _inventory(config: AnalysisConfig) -> dict[str, dict[str, Any]]:
    contract = _load_json(config.contract_audit_path)
    return {row["artifact_id"]: row for row in contract["physical_artifacts"]}


def _load_group_inputs(
    config: AnalysisConfig,
    group: AnalysisGroup,
    inventory: dict[str, dict[str, Any]],
) -> tuple[pd.DataFrame, tuple[str, ...], dict[int, list[pd.DataFrame]]]:
    roots = _output_roots(config)
    by_partition: dict[
        tuple[Any, ...], list[tuple[dict[str, Any], pd.DataFrame, tuple[str, ...]]]
    ] = defaultdict(list)
    raw_by_seed: dict[int, list[pd.DataFrame]] = defaultdict(list)
    classes: tuple[str, ...] | None = None
    for artifact_id in group.artifact_ids:
        row = inventory[artifact_id]
        run = row["run"]
        run_dir = roots[row["output_group"]] / row["relative_output_dir"]
        manifest = _load_json(run_dir / "run.json")
        observed_classes = tuple(str(value) for value in manifest.get("class_order", []))
        if classes is None:
            classes = observed_classes
        elif classes != observed_classes:
            raise PipelineInvariantError(f"Group class orders differ: {group.group_id}")
        predictions = pd.read_parquet(run_dir / "predictions.parquet")
        partition = (
            (int(run["train_session"]), int(run["test_session"]))
            if row["protocol"] == "cross_session"
            else (int(run["fold"]),)
        )
        by_partition[partition].append((row, predictions, observed_classes))
        raw_by_seed[int(run["seed"])].append(predictions)
    if classes is None:
        raise PipelineInvariantError("Analysis group has no class order")
    ensembled = [ensemble_partition(group, inputs) for _, inputs in sorted(by_partition.items())]
    combined = pd.concat(ensembled, ignore_index=True)
    if len(combined) != 452_562 or combined.duplicated(["pair_id", "test_domain"]).any():
        raise PipelineInvariantError(f"Analysis group OOF coverage differs: {group.group_id}")
    return combined, classes, raw_by_seed


def aggregate_results(config: AnalysisConfig) -> Path:
    """Validate the complete campaign and atomically publish analysis artifacts."""
    validate_analysis_contract(config)
    target = config.output_root
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite analysis output: {target}")
    revision = _validate_campaign(config)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    predictions_root = staging / "predictions"
    predictions_root.mkdir()
    inventory = _inventory(config)
    groups = enumerate_analysis_groups(config)
    metric_rows = []
    seed_rows = []
    bootstrap_rows = []
    aliases = []
    try:
        for group in groups:
            frame, classes, raw_by_seed = _load_group_inputs(config, group, inventory)
            output_name = f"{group.group_id}.parquet"
            frame.to_parquet(predictions_root / output_name, index=False)
            metrics = compute_analysis_metrics(frame, classes)
            intervals = paired_bootstrap_intervals(
                frame,
                classes,
                metrics=config.primary_metrics,
                replicates=int(config.bootstrap["replicates"]),
                confidence_level=float(config.bootstrap["confidence_level"]),
                seed=int(config.bootstrap["seed"]),
            )
            for logical_id in group.logical_group_ids:
                aliases.append(
                    {
                        "logical_group_id": logical_id,
                        "physical_group_id": group.group_id,
                        "prediction_file": f"predictions/{output_name}",
                    }
                )
                for domain, values in metrics.items():
                    for metric, value in values.items():
                        metric_rows.append(
                            {
                                "logical_group_id": logical_id,
                                "physical_group_id": group.group_id,
                                "test_domain": domain,
                                "metric": metric,
                                "value": value,
                            }
                        )
                for row in intervals:
                    bootstrap_rows.append(
                        {"logical_group_id": logical_id, "physical_group_id": group.group_id, **row}
                    )
            if group.seed_policy == "mean_probabilities":
                for seed, frames in sorted(raw_by_seed.items()):
                    raw = pd.concat(frames, ignore_index=True)
                    seed_metrics = compute_analysis_metrics(raw, classes)
                    for domain, values in seed_metrics.items():
                        for metric, value in values.items():
                            seed_rows.append(
                                {
                                    "physical_group_id": group.group_id,
                                    "seed": seed,
                                    "test_domain": domain,
                                    "metric": metric,
                                    "value": value,
                                }
                            )
        metrics_frame = pd.DataFrame(metric_rows)
        seed_frame = pd.DataFrame(seed_rows)
        dispersion = (
            seed_frame.groupby(["physical_group_id", "test_domain", "metric"])["value"]
            .agg(seed_count="count", mean="mean", std="std", minimum="min", maximum="max")
            .reset_index()
        )
        pd.DataFrame(aliases).to_csv(staging / "logical_aliases.csv", index=False)
        metrics_frame.to_csv(staging / "metrics_summary.csv", index=False)
        seed_frame.to_csv(staging / "seed_metrics.csv", index=False)
        dispersion.to_csv(staging / "seed_dispersion.csv", index=False)
        pd.DataFrame(bootstrap_rows).to_csv(staging / "bootstrap_intervals.csv", index=False)
        files = sorted(path for path in staging.rglob("*") if path.is_file())
        manifest = {
            "schema_version": 1,
            "status": "complete",
            "created_utc": datetime.now(UTC).isoformat(),
            "package_version": __version__,
            "git": {"revision": revision, "dirty": False},
            "analysis_contract_sha256": sha256_file(config.contract_audit_path),
            "counts": {
                "physical_groups": len(groups),
                "logical_groups": len(aliases),
                "alias_rows": len(aliases),
                "prediction_rows": sum(
                    len(pd.read_parquet(path, columns=["pair_id"]))
                    for path in files
                    if path.suffix == ".parquet"
                ),
                "metric_rows": len(metrics_frame),
                "seed_metric_rows": len(seed_frame),
                "bootstrap_rows": len(bootstrap_rows),
            },
            "artifacts": {
                path.relative_to(staging).as_posix(): sha256_file(path) for path in files
            },
        }
        (staging / "analysis.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        validate_analysis_output(
            staging,
            config=config,
            expected_revision=revision,
        )
        staging.replace(target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target


def validate_analysis_output(
    output: Path,
    *,
    config: AnalysisConfig,
    expected_revision: str | None = None,
) -> dict[str, Any]:
    manifest = _load_json(output / "analysis.json")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("status") != "complete"
        or manifest.get("package_version") != __version__
        or manifest.get("analysis_contract_sha256") != sha256_file(config.contract_audit_path)
    ):
        raise PipelineInvariantError("Analysis output manifest identity differs")
    if expected_revision is not None and manifest.get("git") != {
        "revision": expected_revision,
        "dirty": False,
    }:
        raise PipelineInvariantError("Analysis output revision differs")
    artifacts = manifest.get("artifacts", {})
    physical = {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file() and path.name != "analysis.json"
    }
    if physical != set(artifacts):
        raise PipelineInvariantError("Analysis output artifact inventory differs")
    for relative, digest in artifacts.items():
        if sha256_file(output / relative) != digest:
            raise PipelineInvariantError(f"Analysis output hash differs: {relative}")
    counts = manifest.get("counts", {})
    expected_counts = {
        "physical_groups": 44,
        "logical_groups": 46,
        "alias_rows": 46,
        "prediction_rows": 19_912_728,
        "metric_rows": 460,
        "seed_metric_rows": 300,
        "bootstrap_rows": 92,
    }
    if counts != expected_counts:
        raise PipelineInvariantError("Analysis output row arithmetic differs")
    aliases = pd.read_csv(output / "logical_aliases.csv")
    metrics = pd.read_csv(output / "metrics_summary.csv")
    seeds = pd.read_csv(output / "seed_metrics.csv")
    dispersion = pd.read_csv(output / "seed_dispersion.csv")
    bootstrap = pd.read_csv(output / "bootstrap_intervals.csv")
    if (
        len(aliases) != 46
        or aliases["logical_group_id"].duplicated().any()
        or len(metrics) != 460
        or len(seeds) != 300
        or len(dispersion) != 100
        or len(bootstrap) != 92
    ):
        raise PipelineInvariantError("Analysis output table dimensions differ")
    recomputed: dict[str, dict[str, dict[str, float]]] = {}
    for row in aliases.itertuples(index=False):
        physical_id = str(row.physical_group_id)
        if physical_id not in recomputed:
            predictions = pd.read_parquet(output / str(row.prediction_file))
            classes = tuple(sorted(predictions["true_label"].astype(str).unique()))
            recomputed[physical_id] = compute_analysis_metrics(predictions, classes)
        expected = recomputed[physical_id]
        selected = metrics.loc[metrics["logical_group_id"].eq(row.logical_group_id)]
        observed = {
            (str(item.test_domain), str(item.metric)): float(item.value)
            for item in selected.itertuples(index=False)
        }
        expected_flat = {
            (domain, metric): value
            for domain, values in expected.items()
            for metric, value in values.items()
        }
        if set(observed) != set(expected_flat) or any(
            not np.isclose(observed[key], expected_flat[key], rtol=1e-12, atol=1e-15)
            for key in observed
        ):
            raise PipelineInvariantError("Analysis point metrics disagree with predictions")
        intervals = bootstrap.loc[
            bootstrap["logical_group_id"].eq(row.logical_group_id)
        ]
        for interval in intervals.itertuples(index=False):
            metric = str(interval.metric)
            values = (
                (float(interval.inner_estimate), expected["inner"][metric]),
                (float(interval.outer_estimate), expected["outer"][metric]),
                (
                    float(interval.gap_estimate),
                    expected["outer"][metric] - expected["inner"][metric],
                ),
            )
            if any(
                not np.isclose(observed_value, expected_value, rtol=1e-12, atol=1e-15)
                for observed_value, expected_value in values
            ):
                raise PipelineInvariantError(
                    "Analysis bootstrap estimates disagree with predictions"
                )
    observed_dispersion = (
        seeds.groupby(["physical_group_id", "test_domain", "metric"])["value"]
        .agg(seed_count="count", mean="mean", std="std", minimum="min", maximum="max")
        .reset_index()
    )
    try:
        pd.testing.assert_frame_equal(
            dispersion.sort_values(list(dispersion.columns[:3])).reset_index(drop=True),
            observed_dispersion.sort_values(list(observed_dispersion.columns[:3])).reset_index(
                drop=True
            ),
            check_exact=False,
            rtol=1e-12,
            atol=1e-15,
        )
    except AssertionError as error:
        raise PipelineInvariantError("Analysis seed dispersion differs") from error
    return {"status": "valid", "counts": expected_counts}
