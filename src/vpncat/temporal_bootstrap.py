from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from vpncat.config import DatasetConfig
from vpncat.errors import PipelineInvariantError
from vpncat.hashing import sha256_file

PRIMARY_METRICS = ("balanced_accuracy", "macro_f1")
METHOD_COMPARISONS = (
    (
        "dann_minus_plain_cnn1d",
        "dann__sequential_splt__dann_cnn1d",
        "primary__sequential_splt__cnn1d__train_inner",
    ),
    (
        "dann_minus_supervised_outer_cnn1d",
        "dann__sequential_splt__dann_cnn1d",
        "primary__sequential_splt__cnn1d__train_outer",
    ),
    (
        "cnn1d_minus_lstm",
        "primary__sequential_splt__cnn1d__train_inner",
        "primary__sequential_splt__lstm__train_inner",
    ),
    (
        "cnn1d_minus_transformer",
        "primary__sequential_splt__cnn1d__train_inner",
        "primary__sequential_splt__transformer__train_inner",
    ),
    (
        "xgboost_matched_minus_flattened",
        "primary__matched_flow_stats__xgboost__train_inner",
        "primary__flattened_splt__xgboost__train_inner",
    ),
    (
        "cnn1d_minus_xgboost_matched",
        "primary__sequential_splt__cnn1d__train_inner",
        "primary__matched_flow_stats__xgboost__train_inner",
    ),
    (
        "xgboost_prefix_minus_full",
        "primary__prefix_stats__xgboost__train_inner",
        "primary__matched_flow_stats__xgboost__train_inner",
    ),
)


def _pair_times(
    config: DatasetConfig,
    canonical_path: Path,
) -> pd.DataFrame:
    canonical = pd.read_parquet(
        canonical_path,
        columns=("pair_id", "session", "source_id", "source_flow_id"),
    )
    if canonical["pair_id"].duplicated().any():
        raise PipelineInvariantError("Canonical pair IDs are not unique")
    parts: list[pd.DataFrame] = []
    for session, flow_path in sorted(config.flow_files.items()):
        flows = pd.read_parquet(
            flow_path,
            columns=("id", "flow_id", "bidirectional_first_seen_ms"),
        )
        if flows["flow_id"].duplicated().any():
            raise PipelineInvariantError(f"Session {session} flow_id is not unique")
        flows.insert(0, "session", session)
        parts.append(flows)
    source = pd.concat(parts, ignore_index=True)
    joined = canonical.merge(
        source,
        left_on=["session", "source_flow_id"],
        right_on=["session", "flow_id"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if not joined["_merge"].eq("both").all():
        raise PipelineInvariantError("Canonical pair timestamps are incomplete")
    if not joined["source_id"].eq(joined["id"]).all():
        raise PipelineInvariantError("Canonical and released flow IDs disagree")
    if joined["bidirectional_first_seen_ms"].isna().any():
        raise PipelineInvariantError("Flow start timestamps are incomplete")
    return joined.loc[:, ["pair_id", "session", "bidirectional_first_seen_ms"]]


def build_time_clusters(
    pair_times: pd.DataFrame,
    *,
    block_hours: int,
) -> tuple[pd.DataFrame, dict[int, np.ndarray]]:
    if block_hours < 1:
        raise PipelineInvariantError("block_hours must be positive")
    width_ms = block_hours * 60 * 60 * 1000
    result_parts: list[pd.DataFrame] = []
    cluster_indices: dict[int, np.ndarray] = {}
    next_cluster = 0
    for session, frame in pair_times.groupby("session", sort=True):
        starts = frame["bidirectional_first_seen_ms"].astype(np.int64)
        local_block = ((starts - int(starts.min())) // width_ms).astype(np.int64)
        populated = np.sort(local_block.unique())
        local_to_global = {
            int(block): next_cluster + offset for offset, block in enumerate(populated)
        }
        global_cluster = local_block.map(local_to_global).astype(np.int64)
        selected = frame.loc[:, ["pair_id", "session"]].copy()
        selected["local_block"] = local_block.to_numpy()
        selected["cluster_index"] = global_cluster.to_numpy()
        result_parts.append(selected)
        indices = np.arange(next_cluster, next_cluster + len(populated), dtype=np.int64)
        cluster_indices[int(session)] = indices
        next_cluster += len(populated)
    result = pd.concat(result_parts, ignore_index=True)
    if result["pair_id"].duplicated().any() or len(result) != len(pair_times):
        raise PipelineInvariantError("Time-cluster assignment is not one-to-one")
    return result, cluster_indices


def session_preserving_multiplicities(
    cluster_indices: dict[int, np.ndarray],
    *,
    replicates: int,
    seed: int,
) -> np.ndarray:
    if replicates < 1:
        raise PipelineInvariantError("replicates must be positive")
    cluster_count = sum(len(indices) for indices in cluster_indices.values())
    multiplicities = np.zeros((replicates, cluster_count), dtype=np.int16)
    rng = np.random.default_rng(seed)
    for indices in cluster_indices.values():
        draws = rng.integers(0, len(indices), size=(replicates, len(indices)))
        for replicate in range(replicates):
            multiplicities[replicate, indices] = np.bincount(
                draws[replicate], minlength=len(indices)
            )
    return multiplicities


def _metric_from_confusions(confusion: np.ndarray, metric: str) -> np.ndarray:
    if confusion.ndim != 3 or confusion.shape[1] != confusion.shape[2]:
        raise PipelineInvariantError("Confusion tensor must have shape (samples, C, C)")
    true_positive = np.diagonal(confusion, axis1=1, axis2=2)
    true_support = confusion.sum(axis=2)
    if metric == "balanced_accuracy":
        values = np.divide(
            true_positive,
            true_support,
            out=np.zeros_like(true_positive, dtype=np.float64),
            where=true_support > 0,
        )
        return values.mean(axis=1)
    if metric == "macro_f1":
        predicted_support = confusion.sum(axis=1)
        denominator = true_support + predicted_support
        values = np.divide(
            2 * true_positive,
            denominator,
            out=np.zeros_like(true_positive, dtype=np.float64),
            where=denominator > 0,
        )
        return values.mean(axis=1)
    raise PipelineInvariantError(f"Unsupported temporal-bootstrap metric: {metric}")


def _cluster_confusions(
    frame: pd.DataFrame,
    clusters: pd.DataFrame,
    *,
    prediction_column: str,
    classes: tuple[str, ...],
) -> np.ndarray:
    selected = frame.loc[:, ["pair_id", "true_label", prediction_column]].merge(
        clusters.loc[:, ["pair_id", "cluster_index"]],
        on="pair_id",
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if not selected["_merge"].eq("both").all() or len(selected) != len(clusters):
        raise PipelineInvariantError("Predictions and time clusters are not pair-complete")
    class_index = {label: index for index, label in enumerate(classes)}
    true_index = selected["true_label"].astype(str).map(class_index)
    prediction_index = selected[prediction_column].astype(str).map(class_index)
    if true_index.isna().any() or prediction_index.isna().any():
        raise PipelineInvariantError("Prediction labels differ from the frozen class set")
    class_count = len(classes)
    cluster_count = len(clusters["cluster_index"].unique())
    codes = (
        selected["cluster_index"].to_numpy(dtype=np.int64) * class_count**2
        + true_index.to_numpy(dtype=np.int64) * class_count
        + prediction_index.to_numpy(dtype=np.int64)
    )
    return np.bincount(
        codes, minlength=cluster_count * class_count**2
    ).reshape(cluster_count, class_count, class_count)


def cluster_bootstrap_difference(
    frame: pd.DataFrame,
    clusters: pd.DataFrame,
    multiplicities: np.ndarray,
    *,
    a_column: str,
    b_column: str,
    classes: tuple[str, ...],
    metrics: tuple[str, ...] = PRIMARY_METRICS,
    confidence_level: float = 0.95,
) -> list[dict[str, float | str]]:
    """Return A-minus-B intervals under paired time-cluster resampling."""
    if not 0.0 < confidence_level < 1.0:
        raise PipelineInvariantError("confidence_level must be between zero and one")
    a_counts = _cluster_confusions(
        frame, clusters, prediction_column=a_column, classes=classes
    )
    b_counts = _cluster_confusions(
        frame, clusters, prediction_column=b_column, classes=classes
    )
    if multiplicities.shape[1] != len(a_counts):
        raise PipelineInvariantError("Bootstrap multiplicities disagree with cluster count")
    class_count = len(classes)
    a_samples = (multiplicities @ a_counts.reshape(len(a_counts), -1)).reshape(
        len(multiplicities), class_count, class_count
    )
    b_samples = (multiplicities @ b_counts.reshape(len(b_counts), -1)).reshape(
        len(multiplicities), class_count, class_count
    )
    a_point = a_counts.sum(axis=0, keepdims=True)
    b_point = b_counts.sum(axis=0, keepdims=True)
    alpha = (1.0 - confidence_level) / 2.0
    rows: list[dict[str, float | str]] = []
    for metric in metrics:
        a_values = _metric_from_confusions(a_samples, metric)
        b_values = _metric_from_confusions(b_samples, metric)
        differences = a_values - b_values
        rows.append(
            {
                "metric": metric,
                "a_estimate": float(_metric_from_confusions(a_point, metric)[0]),
                "b_estimate": float(_metric_from_confusions(b_point, metric)[0]),
                "delta_estimate": float(
                    _metric_from_confusions(a_point, metric)[0]
                    - _metric_from_confusions(b_point, metric)[0]
                ),
                "delta_ci_low": float(np.quantile(differences, alpha)),
                "delta_ci_high": float(np.quantile(differences, 1.0 - alpha)),
            }
        )
    return rows


def _load_analysis_index(analysis_root: Path) -> dict[str, Any]:
    path = analysis_root / "analysis.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or manifest.get("schema_version") != 1:
        raise PipelineInvariantError("Source analysis manifest is not complete")
    return manifest


def _prediction_path(
    analysis_root: Path,
    analysis_manifest: dict[str, Any],
    group_id: str,
) -> Path:
    relative = f"predictions/{group_id}.parquet"
    path = analysis_root / relative
    expected = analysis_manifest.get("artifacts", {}).get(relative)
    if not path.exists() or not expected or sha256_file(path) != expected:
        raise PipelineInvariantError(f"Prediction artifact is missing or changed: {group_id}")
    return path


def _load_paired_prediction(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(
        path, columns=("pair_id", "test_domain", "true_label", "prediction")
    )
    if frame.duplicated(["pair_id", "test_domain"]).any():
        raise PipelineInvariantError(f"Predictions duplicate pair/domain rows: {path.name}")
    views = {
        domain: frame.loc[frame["test_domain"].eq(domain)]
        .sort_values("pair_id")
        .reset_index(drop=True)
        for domain in ("inner", "outer")
    }
    if not np.array_equal(views["inner"]["pair_id"], views["outer"]["pair_id"]):
        raise PipelineInvariantError(f"Prediction views are not paired: {path.name}")
    if not np.array_equal(views["inner"]["true_label"], views["outer"]["true_label"]):
        raise PipelineInvariantError(f"Prediction labels differ by view: {path.name}")
    return pd.DataFrame(
        {
            "pair_id": views["inner"]["pair_id"],
            "true_label": views["inner"]["true_label"],
            "inner_prediction": views["inner"]["prediction"],
            "outer_prediction": views["outer"]["prediction"],
        }
    )


def _load_outer_prediction(path: Path, *, column: str) -> pd.DataFrame:
    frame = pd.read_parquet(
        path, columns=("pair_id", "test_domain", "true_label", "prediction")
    )
    selected = (
        frame.loc[frame["test_domain"].eq("outer"), ["pair_id", "true_label", "prediction"]]
        .sort_values("pair_id")
        .reset_index(drop=True)
        .rename(columns={"prediction": column})
    )
    if selected["pair_id"].duplicated().any():
        raise PipelineInvariantError(f"Outer predictions duplicate pair IDs: {path.name}")
    return selected


def _assert_close(observed: float, expected: float, *, context: str) -> None:
    if not np.isclose(observed, expected, rtol=1e-12, atol=1e-15):
        raise PipelineInvariantError(
            f"Temporal-bootstrap point estimate disagrees with paper evidence: {context}"
        )


def run_temporal_bootstrap(
    config: DatasetConfig,
    *,
    canonical_path: Path,
    dataset_manifest_path: Path,
    analysis_root: Path,
    paper_analysis_root: Path,
    output_dir: Path,
    block_hours: tuple[int, ...] = (1, 2),
    replicates: int = 1_000,
    confidence_level: float = 0.95,
    seed: int = 42,
    force: bool = False,
) -> dict[str, Any]:
    """Re-estimate load-bearing paired intervals using temporal clusters."""
    if not block_hours or any(hours < 1 for hours in block_hours):
        raise PipelineInvariantError("block_hours must contain positive integers")
    if len(set(block_hours)) != len(block_hours):
        raise PipelineInvariantError("block_hours must not contain duplicates")
    output_dir = output_dir.expanduser().resolve()
    intervals_path = output_dir / "temporal_block_intervals.csv"
    manifest_path = output_dir / "temporal_bootstrap_manifest.json"
    if not force and (intervals_path.exists() or manifest_path.exists()):
        raise FileExistsError("Refusing to overwrite temporal-bootstrap outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    classes = tuple(dataset_manifest["counts"]["retained_classes"])
    analysis_manifest = _load_analysis_index(analysis_root)
    primary_reference = pd.read_csv(paper_analysis_root / "primary_transfer_gaps.csv")
    method_reference = pd.read_csv(paper_analysis_root / "pairwise_method_differences.csv")
    pair_times = _pair_times(config, canonical_path)

    primary_groups = sorted(
        Path(relative).stem
        for relative in analysis_manifest["artifacts"]
        if relative.startswith("predictions/primary__")
        and relative.endswith("__train_inner.parquet")
    )
    if len(primary_groups) != 9:
        raise PipelineInvariantError("Expected nine primary source-only analysis groups")

    prediction_groups = set(primary_groups)
    for _, left_id, right_id in METHOD_COMPARISONS:
        prediction_groups.update((left_id, right_id))
    prediction_paths = {
        group_id: _prediction_path(analysis_root, analysis_manifest, group_id)
        for group_id in sorted(prediction_groups)
    }

    rows: list[dict[str, Any]] = []
    cluster_summary: dict[str, dict[str, int]] = {}
    for hours in block_hours:
        clusters, by_session = build_time_clusters(pair_times, block_hours=hours)
        multiplicities = session_preserving_multiplicities(
            by_session, replicates=replicates, seed=seed
        )
        cluster_summary[str(hours)] = {
            str(session): len(indices) for session, indices in by_session.items()
        }

        for group_id in primary_groups:
            frame = _load_paired_prediction(prediction_paths[group_id])
            intervals = cluster_bootstrap_difference(
                frame,
                clusters,
                multiplicities,
                a_column="outer_prediction",
                b_column="inner_prediction",
                classes=classes,
                confidence_level=confidence_level,
            )
            for interval in intervals:
                reference = primary_reference.loc[
                    primary_reference["logical_group_id"].eq(group_id)
                    & primary_reference["metric"].eq(interval["metric"])
                ]
                if len(reference) != 1:
                    raise PipelineInvariantError(f"Missing primary paper reference: {group_id}")
                ref = reference.iloc[0]
                _assert_close(
                    float(interval["a_estimate"]),
                    float(ref["target_estimate"]),
                    context=f"{group_id}/{interval['metric']}/outer",
                )
                _assert_close(
                    float(interval["b_estimate"]),
                    float(ref["source_estimate"]),
                    context=f"{group_id}/{interval['metric']}/inner",
                )
                rows.append(
                    {
                        "comparison_type": "view_gap",
                        "comparison_id": group_id,
                        "a_label": "outer",
                        "b_label": "inner",
                        "block_hours": hours,
                        "clusters": multiplicities.shape[1],
                        "replicates": replicates,
                        **interval,
                        "pair_delta_ci_low": float(ref["delta_ci_low"]),
                        "pair_delta_ci_high": float(ref["delta_ci_high"]),
                    }
                )

        for comparison_id, left_id, right_id in METHOD_COMPARISONS:
            left = _load_outer_prediction(prediction_paths[left_id], column="left_prediction")
            right = _load_outer_prediction(prediction_paths[right_id], column="right_prediction")
            try:
                frame = left.merge(
                    right,
                    on=("pair_id", "true_label"),
                    how="inner",
                    validate="one_to_one",
                )
            except pd.errors.MergeError as error:
                raise PipelineInvariantError(
                    f"Method predictions do not pair: {comparison_id}"
                ) from error
            if len(frame) != len(pair_times):
                raise PipelineInvariantError(
                    f"Method comparison is not pair-complete: {comparison_id}"
                )
            intervals = cluster_bootstrap_difference(
                frame,
                clusters,
                multiplicities,
                a_column="left_prediction",
                b_column="right_prediction",
                classes=classes,
                confidence_level=confidence_level,
            )
            for interval in intervals:
                reference = method_reference.loc[
                    method_reference["comparison_id"].eq(comparison_id)
                    & method_reference["metric"].eq(interval["metric"])
                ]
                if len(reference) != 1:
                    raise PipelineInvariantError(
                        f"Missing method paper reference: {comparison_id}"
                    )
                ref = reference.iloc[0]
                _assert_close(
                    float(interval["a_estimate"]),
                    float(ref["left_estimate"]),
                    context=f"{comparison_id}/{interval['metric']}/left",
                )
                _assert_close(
                    float(interval["b_estimate"]),
                    float(ref["right_estimate"]),
                    context=f"{comparison_id}/{interval['metric']}/right",
                )
                rows.append(
                    {
                        "comparison_type": "method_difference",
                        "comparison_id": comparison_id,
                        "a_label": left_id,
                        "b_label": right_id,
                        "block_hours": hours,
                        "clusters": multiplicities.shape[1],
                        "replicates": replicates,
                        **interval,
                        "pair_delta_ci_low": float(ref["delta_ci_low"]),
                        "pair_delta_ci_high": float(ref["delta_ci_high"]),
                    }
                )

    result = pd.DataFrame(rows).sort_values(
        ["comparison_type", "comparison_id", "metric", "block_hours"],
        ignore_index=True,
    )
    result.to_csv(intervals_path, index=False, float_format="%.15g")
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "policy": {
            "resampling_unit": "nonoverlapping_time_block",
            "flow_assignment": "bidirectional_first_seen_ms",
            "sessions_resampled_separately": True,
            "block_hours": list(block_hours),
            "replicates": replicates,
            "confidence_level": confidence_level,
            "seed": seed,
            "conditioning": "fitted prediction ensembles",
        },
        "cluster_counts": cluster_summary,
        "counts": {
            "retained_pairs": len(pair_times),
            "primary_groups": len(primary_groups),
            "method_comparisons": len(METHOD_COMPARISONS),
            "interval_rows": len(result),
        },
        "inputs": {
            "canonical_pairs": sha256_file(canonical_path),
            "dataset_manifest": sha256_file(dataset_manifest_path),
            "analysis_manifest": sha256_file(analysis_root / "analysis.json"),
            "primary_paper_reference": sha256_file(
                paper_analysis_root / "primary_transfer_gaps.csv"
            ),
            "method_paper_reference": sha256_file(
                paper_analysis_root / "pairwise_method_differences.csv"
            ),
            "prediction_artifacts": {
                group_id: sha256_file(path)
                for group_id, path in sorted(prediction_paths.items())
            },
            "flow_artifacts": {
                str(session): sha256_file(path)
                for session, path in sorted(config.flow_files.items())
            },
        },
        "outputs": {
            "temporal_block_intervals.csv": {
                "rows": len(result),
                "sha256": sha256_file(intervals_path),
            }
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest
