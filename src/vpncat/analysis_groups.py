from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from vpncat.analysis import AnalysisConfig, validate_analysis_contract
from vpncat.errors import PipelineInvariantError


@dataclass(frozen=True)
class AnalysisGroup:
    group_id: str
    protocol: str
    experiment_id: str
    representation: str
    model: str
    family: str
    train_domain: str
    seed_policy: str
    artifact_ids: tuple[str, ...]
    logical_group_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "protocol": self.protocol,
            "experiment_id": self.experiment_id,
            "representation": self.representation,
            "model": self.model,
            "family": self.family,
            "train_domain": self.train_domain,
            "seed_policy": self.seed_policy,
            "artifact_ids": list(self.artifact_ids),
            "logical_group_ids": list(self.logical_group_ids),
        }


def _group_key(row: dict[str, Any]) -> tuple[str, ...]:
    run = row["run"]
    protocol = row["protocol"]
    if protocol == "primary":
        return (protocol, run["experiment_id"], run["train_domain"])
    return (protocol, run["experiment_id"])


def _group_id(key: tuple[str, ...]) -> str:
    if key[0] == "primary":
        return f"primary__{key[1]}__train_{key[2]}"
    return f"{key[0]}__{key[1]}"


def _partition_key(protocol: str, run: dict[str, Any]) -> tuple[Any, ...]:
    if protocol == "cross_session":
        return (int(run["train_session"]), int(run["test_session"]))
    return (int(run["fold"]),)


def _expected_partitions(protocol: str) -> int:
    return 2 if protocol == "cross_session" else 5


def _validate_group_inputs(group: AnalysisGroup, rows: list[dict[str, Any]]) -> None:
    by_partition: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_partition[_partition_key(row["protocol"], row["run"])].append(row)
    if len(by_partition) != _expected_partitions(group.protocol):
        raise PipelineInvariantError(f"Analysis group partition coverage differs: {group.group_id}")
    expected_seeds = {42, 43, 44} if group.seed_policy == "mean_probabilities" else {42}
    for partition_rows in by_partition.values():
        seeds = {int(row["run"]["seed"]) for row in partition_rows}
        if seeds != expected_seeds or len(partition_rows) != len(expected_seeds):
            raise PipelineInvariantError(f"Analysis group seed coverage differs: {group.group_id}")
    expected_count = _expected_partitions(group.protocol) * len(expected_seeds)
    if len(rows) != expected_count:
        raise PipelineInvariantError(f"Analysis group artifact count differs: {group.group_id}")


def enumerate_analysis_groups(config: AnalysisConfig) -> tuple[AnalysisGroup, ...]:
    validate_analysis_contract(config)
    contract = json.loads(config.contract_audit_path.read_text(encoding="utf-8"))
    return enumerate_groups_from_contract(contract)


def enumerate_groups_from_contract(
    contract: dict[str, Any],
) -> tuple[AnalysisGroup, ...]:
    artifacts = list(contract["physical_artifacts"])
    references = list(contract["logical_references"])
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in artifacts:
        grouped[_group_key(row)].append(row)

    groups: list[AnalysisGroup] = []
    for key, rows in sorted(grouped.items()):
        first = rows[0]["run"]
        policy = (
            "mean_probabilities"
            if first["family"] == "neural" and len({row["run"]["seed"] for row in rows}) == 3
            else "single_seed"
        )
        group = AnalysisGroup(
            group_id=_group_id(key),
            protocol=key[0],
            experiment_id=str(first["experiment_id"]),
            representation=str(first["representation"]),
            model=str(first["model"]),
            family=str(first["family"]),
            train_domain=str(first.get("train_domain", first.get("source_domain", "inner"))),
            seed_policy=policy,
            artifact_ids=tuple(sorted(row["artifact_id"] for row in rows)),
            logical_group_ids=(_group_id(key),),
        )
        _validate_group_inputs(group, rows)
        groups.append(group)

    physical_by_id = {row["artifact_id"]: row for row in artifacts}
    references_by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in references:
        references_by_model[str(row["logical_run"]["model"])].append(row)
    for model, model_references in sorted(references_by_model.items()):
        artifact_ids = tuple(
            sorted({row["physical_artifact_id"] for row in model_references})
        )
        rows = [physical_by_id[artifact_id] for artifact_id in artifact_ids]
        logical_ids = tuple(
            sorted(
                {
                    f"{row['logical_protocol']}__{row['logical_run']['experiment_id']}"
                    for row in model_references
                }
            )
        )
        first = rows[0]["run"]
        group = AnalysisGroup(
            group_id=f"ablation_anchor__{model}",
            protocol="ablation_anchor",
            experiment_id=f"n050_all__{model}",
            representation="sequential_splt",
            model=model,
            family="neural",
            train_domain="inner",
            seed_policy="single_seed",
            artifact_ids=artifact_ids,
            logical_group_ids=logical_ids,
        )
        _validate_group_inputs(group, rows)
        groups.append(group)

    logical_ids = [logical for group in groups for logical in group.logical_group_ids]
    if (
        len(groups) != 44
        or len(logical_ids) != 46
        or len(set(logical_ids)) != 46
        or sum(group.seed_policy == "mean_probabilities" for group in groups) != 10
        or sum(group.protocol == "ablation_anchor" for group in groups) != 2
    ):
        raise PipelineInvariantError("Analysis group matrix differs from the freeze")
    return tuple(sorted(groups, key=lambda group: group.group_id))


def _probability_matrix(frame: pd.DataFrame, class_count: int) -> np.ndarray:
    try:
        values = np.asarray(frame["class_probabilities"].tolist(), dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise PipelineInvariantError("Analysis probabilities are not a numeric matrix") from error
    if values.shape != (len(frame), class_count) or not np.isfinite(values).all():
        raise PipelineInvariantError("Analysis probability shape or finiteness differs")
    if (values < 0).any() or (values > 1).any() or not np.allclose(
        values.sum(axis=1), 1.0, rtol=1e-6, atol=1e-8
    ):
        raise PipelineInvariantError("Analysis probabilities violate the simplex")
    return values


def ensemble_partition(
    group: AnalysisGroup,
    inputs: list[tuple[dict[str, Any], pd.DataFrame, tuple[str, ...]]],
) -> pd.DataFrame:
    """Average one partition's seeds after proving row and class alignment."""
    if not inputs:
        raise PipelineInvariantError("Analysis partition has no prediction inputs")
    seeds = tuple(sorted(int(row["run"]["seed"]) for row, _, _ in inputs))
    expected = (42, 43, 44) if group.seed_policy == "mean_probabilities" else (42,)
    if seeds != expected:
        raise PipelineInvariantError("Analysis partition seed set differs")
    ordered: list[tuple[dict[str, Any], pd.DataFrame, tuple[str, ...]]] = []
    for row, frame, classes in inputs:
        selected = frame.sort_values(["test_domain", "pair_id"], ignore_index=True)
        ordered.append((row, selected, classes))
    classes = ordered[0][2]
    if len(classes) < 2 or any(item[2] != classes for item in ordered):
        raise PipelineInvariantError("Analysis partition class orders differ")
    identity_columns = (
        "pair_id",
        "session",
        "train_domain",
        "test_domain",
        "true_label",
    )
    baseline = ordered[0][1]
    for _, frame, _ in ordered[1:]:
        try:
            pd.testing.assert_frame_equal(
                baseline.loc[:, identity_columns],
                frame.loc[:, identity_columns],
                check_dtype=False,
                check_exact=True,
            )
        except AssertionError as error:
            raise PipelineInvariantError("Analysis seed prediction identities differ") from error
    matrices = []
    for _, frame, _ in ordered:
        matrix = _probability_matrix(frame, len(classes))
        expected_predictions = np.asarray(classes, dtype=object)[np.argmax(matrix, axis=1)]
        if not np.array_equal(
            expected_predictions.astype(str), frame["prediction"].astype(str).to_numpy()
        ):
            raise PipelineInvariantError("Analysis input predictions disagree with argmax")
        matrices.append(matrix)
    probabilities = np.mean(matrices, axis=0)
    predictions = np.asarray(classes, dtype=object)[np.argmax(probabilities, axis=1)]
    first_run = ordered[0][0]["run"]
    result = baseline.loc[:, identity_columns].copy()
    result.insert(0, "analysis_group_id", group.group_id)
    result["source_protocol"] = group.protocol
    result["experiment_id"] = group.experiment_id
    result["representation"] = group.representation
    result["model"] = group.model
    result["family"] = group.family
    result["partition_fold"] = first_run.get("fold")
    result["train_session"] = first_run.get("train_session")
    result["test_session"] = first_run.get("test_session")
    result["seed_count"] = len(seeds)
    result["seeds"] = [list(seeds)] * len(result)
    result["prediction"] = predictions.astype(str)
    result["class_probabilities"] = probabilities.tolist()
    if result.duplicated(["pair_id", "test_domain"]).any():
        raise PipelineInvariantError("Analysis partition contains duplicate paired predictions")
    return result
