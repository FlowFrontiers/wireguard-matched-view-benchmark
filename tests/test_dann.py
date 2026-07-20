from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import vpncat.dann as dann
from vpncat.dann import (
    build_dann_contract,
    enumerate_dann_runs,
    load_dann_config,
    validate_dann_contract,
)
from vpncat.dann_data import prepare_dann_run
from vpncat.errors import PipelineInvariantError


def _base_config():
    return load_dann_config(Path(__file__).parents[1] / "configs" / "dann.yaml")


def _write_inputs(tmp_path: Path, *, row_count: int = 20):
    labels = ["A", "B"] * (row_count // 2)
    frame = pd.DataFrame(
        {
            "pair_id": [f"pair:{index}" for index in range(row_count)],
            "session": [1, 2] * (row_count // 2),
            "application_category": labels,
        }
    )
    for domain, offset in (("inner", 0.0), ("outer", 100.0)):
        frame[f"{domain}_direction"] = [np.asarray([0, 1, 0])] * row_count
        frame[f"{domain}_size"] = [
            np.asarray([100.0 + index + offset, 200.0, 50.0])
            for index in range(row_count)
        ]
        frame[f"{domain}_iat_ms"] = [
            np.asarray([0.0, 1.0 + index + offset, 2.0])
            for index in range(row_count)
        ]
    canonical = tmp_path / "canonical.parquet"
    frame.to_parquet(canonical, index=False)
    split = frame.loc[:, ["pair_id", "session", "application_category"]].copy()
    roles = ["train"] * 12 + ["validation"] * 4 + ["test"] * 4
    for fold in range(1, 6):
        split[f"role_fold_{fold}"] = roles
    split_path = tmp_path / "split.csv"
    split.to_csv(split_path, index=False)
    config = _base_config()
    primary = replace(
        config.primary,
        canonical_path=canonical,
        split_path=split_path,
    )
    return replace(
        config,
        primary=primary,
        contract_audit_path=tmp_path / "dann_contract.json",
        output_root=tmp_path / "outputs",
    )


def test_dann_configuration_and_matrix_are_frozen() -> None:
    config = _base_config()
    runs = enumerate_dann_runs(config)
    assert len(runs) == 15
    assert {run.fold for run in runs} == {1, 2, 3, 4, 5}
    assert {run.seed for run in runs} == {42, 43, 44}
    assert {run.source_domain for run in runs} == {"inner"}
    assert {run.adaptation_domain for run in runs} == {"outer"}
    assert config.gradient_reversal == {
        "schedule": "logistic",
        "gamma": 10.0,
        "start": 0.0,
        "end": 1.0,
    }


def test_dann_materialization_exposes_only_unlabeled_training_outer_view(
    tmp_path: Path,
) -> None:
    config = _write_inputs(tmp_path)
    run = enumerate_dann_runs(config)[0]
    prepared = prepare_dann_run(config, run)
    assert prepared.source_training.pair_ids == prepared.adaptation_training.pair_ids
    np.testing.assert_array_equal(
        prepared.source_training.positions,
        prepared.adaptation_training.positions,
    )
    assert not hasattr(prepared.adaptation_training, "targets")
    assert len(prepared.source_training.pair_ids) == 12
    assert len(prepared.source_validation.pair_ids) == 4
    assert len(prepared.tests["inner"].pair_ids) == 4
    assert prepared.tests["inner"].pair_ids == prepared.tests["outer"].pair_ids
    forbidden = set(prepared.source_validation.pair_ids) | set(
        prepared.tests["outer"].pair_ids
    )
    assert set(prepared.adaptation_training.pair_ids).isdisjoint(forbidden)
    assert np.all(
        prepared.adaptation_training.values[~prepared.adaptation_training.mask] == 0
    )
    assert not np.array_equal(
        prepared.source_training.values,
        prepared.adaptation_training.values,
    )


def test_dann_adaptation_training_is_immune_to_forbidden_outer_poison(
    tmp_path: Path,
) -> None:
    config = _write_inputs(tmp_path)
    run = enumerate_dann_runs(config)[0]
    baseline = prepare_dann_run(config, run)
    frame = pd.read_parquet(config.primary.canonical_path)
    outer_columns = ["outer_direction", "outer_size", "outer_iat_ms"]
    for row in range(12, len(frame)):
        frame.at[row, "outer_direction"] = np.asarray([1, 1, 1])
        frame.at[row, "outer_size"] = np.asarray([1e9, 1e9, 1e9])
        frame.at[row, "outer_iat_ms"] = np.asarray([1e9, 1e9, 1e9])
    assert set(outer_columns) <= set(frame.columns)
    poisoned = tmp_path / "poisoned.parquet"
    frame.to_parquet(poisoned, index=False)
    observed = prepare_dann_run(
        replace(config, primary=replace(config.primary, canonical_path=poisoned)),
        run,
    )
    np.testing.assert_array_equal(
        baseline.adaptation_training.values,
        observed.adaptation_training.values,
    )
    np.testing.assert_array_equal(
        baseline.source_training.values,
        observed.source_training.values,
    )


def test_dann_contract_is_deterministic_and_refuses_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _write_inputs(tmp_path)
    hashes = {
        "canonical": "a" * 64,
        "split_manifest": "b" * 64,
        "dann_config": "c" * 64,
    }
    monkeypatch.setattr(dann, "_input_hashes", lambda _config: hashes)
    payload = build_dann_contract(config)
    assert payload["matrix"] == {"training_runs": 15, "prediction_groups": 30}
    assert all(
        summary["adaptation_labels_exposed"] is False
        for summary in payload["folds"].values()
    )
    assert validate_dann_contract(config)["training_runs"] == 15
    with pytest.raises(FileExistsError, match="overwrite"):
        build_dann_contract(config)
    observed = json.loads(config.contract_audit_path.read_text(encoding="utf-8"))
    observed["runs"][0]["seed"] = 999
    config.contract_audit_path.write_text(
        json.dumps(observed, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(PipelineInvariantError, match="stale"):
        validate_dann_contract(config)


@pytest.mark.parametrize("field", ["protocol", "matrix"])
def test_dann_contract_rejects_summary_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    config = _write_inputs(tmp_path)
    monkeypatch.setattr(dann, "_input_hashes", lambda _config: {"input": "a" * 64})
    build_dann_contract(config)
    observed = json.loads(config.contract_audit_path.read_text(encoding="utf-8"))
    observed[field] = {}
    config.contract_audit_path.write_text(
        json.dumps(observed, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(PipelineInvariantError, match="stale"):
        validate_dann_contract(config)
