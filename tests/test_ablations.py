from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import vpncat.ablations as ablations
from vpncat.ablation_data import prepare_ablation_run
from vpncat.ablations import (
    build_ablation_contract,
    enumerate_ablation_runs,
    load_ablation_config,
    primary_reference_run,
    validate_ablation_contract,
)
from vpncat.errors import PipelineInvariantError
from vpncat.neural_data import prepare_neural_run


def _config_path(name: str) -> Path:
    return Path(__file__).parents[1] / "configs" / name


def _write_inputs(tmp_path: Path, config_name: str):
    row_count = 24
    labels = ["A", "B"] * (row_count // 2)
    frame = pd.DataFrame(
        {
            "pair_id": [f"pair:{index}" for index in range(row_count)],
            "session": [1, 2] * (row_count // 2),
            "application_category": labels,
        }
    )
    for domain, offset in (("inner", 0.0), ("outer", 100.0)):
        frame[f"{domain}_direction"] = [
            np.asarray([index % 2, (index + 1) % 2] * 6, dtype=np.int8)
            for index in range(row_count)
        ]
        frame[f"{domain}_size"] = [
            np.arange(12, dtype=np.float32) + index + offset + 1
            for index in range(row_count)
        ]
        frame[f"{domain}_iat_ms"] = [
            np.arange(12, dtype=np.float32) / 10 + offset
            for _ in range(row_count)
        ]
    canonical = tmp_path / "canonical.parquet"
    frame.to_parquet(canonical, index=False)
    split = frame.loc[:, ["pair_id", "session", "application_category"]].copy()
    roles = ["train"] * 16 + ["validation"] * 4 + ["test"] * 4
    for fold in range(1, 6):
        split[f"role_fold_{fold}"] = roles
    split_path = tmp_path / "split.csv"
    split.to_csv(split_path, index=False)
    base = load_ablation_config(_config_path(config_name))
    primary_contract = tmp_path / "primary_contract.json"
    primary = replace(
        base.primary,
        canonical_path=canonical,
        split_path=split_path,
        contract_audit_path=primary_contract,
    )
    config = replace(
        base,
        primary=primary,
        contract_audit_path=tmp_path / f"{base.protocol}.json",
        output_root=tmp_path / "outputs",
    )
    reference_rows = []
    for run in enumerate_ablation_runs(config):
        if run.is_primary_reference:
            reference = primary_reference_run(config, run)
            reference_rows.append(
                {
                    "run_id": reference.run_id,
                    "relative_output_dir": reference.relative_output_dir.as_posix(),
                }
            )
    primary_contract.write_text(
        json.dumps({"status": "valid", "runs": reference_rows}) + "\n",
        encoding="utf-8",
    )
    return config


def test_ablation_matrices_freeze_training_and_reference_counts() -> None:
    prefix = load_ablation_config(_config_path("ablation_prefix.yaml"))
    channels = load_ablation_config(_config_path("ablation_channels.yaml"))
    prefix_runs = enumerate_ablation_runs(prefix)
    channel_runs = enumerate_ablation_runs(channels)
    assert len(prefix_runs) == 40
    assert sum(not run.is_primary_reference for run in prefix_runs) == 30
    assert len(channel_runs) == 50
    assert sum(not run.is_primary_reference for run in channel_runs) == 40
    assert {run.seed for run in (*prefix_runs, *channel_runs)} == {42}
    assert {run.train_domain for run in (*prefix_runs, *channel_runs)} == {"inner"}
    prefix_references = {
        primary_reference_run(prefix, run).run_id
        for run in prefix_runs
        if run.is_primary_reference
    }
    channel_references = {
        primary_reference_run(channels, run).run_id
        for run in channel_runs
        if run.is_primary_reference
    }
    assert prefix_references == channel_references
    assert len(prefix_references) == 10


def test_prefix_materialization_is_an_exact_slice_of_n80(tmp_path: Path) -> None:
    config = _write_inputs(tmp_path, "ablation_prefix.yaml")
    runs = enumerate_ablation_runs(config)
    n10 = next(
        run
        for run in runs
        if run.model == "cnn1d" and run.fold == 1 and run.prefix_length == 10
    )
    n80 = next(
        run
        for run in runs
        if run.model == "cnn1d" and run.fold == 1 and run.prefix_length == 80
    )
    short = prepare_ablation_run(config, n10)
    long = prepare_ablation_run(config, n80)
    for short_subset, long_subset in (
        (short.training, long.training),
        (short.validation, long.validation),
        (short.tests["inner"], long.tests["inner"]),
        (short.tests["outer"], long.tests["outer"]),
    ):
        np.testing.assert_array_equal(short_subset.values, long_subset.values[:, :10])
        np.testing.assert_array_equal(short_subset.mask, long_subset.mask[:, :10])
        assert short_subset.pair_ids == long_subset.pair_ids


def test_channel_materialization_is_exact_projection_of_all_channels(
    tmp_path: Path,
) -> None:
    config = _write_inputs(tmp_path, "ablation_channels.yaml")
    runs = enumerate_ablation_runs(config)
    size_timing = next(
        run
        for run in runs
        if run.model == "transformer"
        and run.fold == 1
        and run.observation_id == "size_timing"
    )
    reference = next(
        run
        for run in runs
        if run.model == "transformer" and run.fold == 1 and run.is_primary_reference
    )
    observed = prepare_ablation_run(config, size_timing)
    all_channels = prepare_neural_run(
        config.primary.canonical_path,
        config.primary.split_path,
        reference,
        prefix_length=50,
        channels=("direction", "size", "iat_ms"),
    )
    for projected, complete in (
        (observed.training, all_channels.training),
        (observed.validation, all_channels.validation),
        (observed.tests["inner"], all_channels.tests["inner"]),
        (observed.tests["outer"], all_channels.tests["outer"]),
    ):
        np.testing.assert_array_equal(projected.values, complete.values[:, :, 1:])
        np.testing.assert_array_equal(projected.mask, complete.mask)


def test_primary_reference_cannot_be_materialized_for_retraining(tmp_path: Path) -> None:
    config = _write_inputs(tmp_path, "ablation_prefix.yaml")
    reference = next(
        run for run in enumerate_ablation_runs(config) if run.is_primary_reference
    )
    with pytest.raises(PipelineInvariantError, match="must not be retrained"):
        prepare_ablation_run(config, reference)


def test_ablation_materialization_rejects_observation_drift(tmp_path: Path) -> None:
    config = _write_inputs(tmp_path, "ablation_channels.yaml")
    run = next(
        run
        for run in enumerate_ablation_runs(config)
        if not run.is_primary_reference
    )
    with pytest.raises(PipelineInvariantError, match="frozen matrix"):
        prepare_ablation_run(config, replace(run, channels=("iat_ms",)))


def test_ablation_training_is_immune_to_outer_training_and_validation_poison(
    tmp_path: Path,
) -> None:
    config = _write_inputs(tmp_path, "ablation_prefix.yaml")
    run = next(
        run
        for run in enumerate_ablation_runs(config)
        if run.model == "cnn1d" and run.fold == 1 and run.prefix_length == 20
    )
    baseline = prepare_ablation_run(config, run)
    frame = pd.read_parquet(config.primary.canonical_path)
    for row in range(20):
        frame.at[row, "outer_size"] = np.full(12, 1e9, dtype=np.float32)
        frame.at[row, "outer_iat_ms"] = np.full(12, 1e9, dtype=np.float32)
    poisoned_path = tmp_path / "poisoned.parquet"
    frame.to_parquet(poisoned_path, index=False)
    observed = prepare_ablation_run(
        replace(config, primary=replace(config.primary, canonical_path=poisoned_path)),
        run,
    )
    assert baseline.state.to_dict() == observed.state.to_dict()
    np.testing.assert_array_equal(baseline.training.values, observed.training.values)
    np.testing.assert_array_equal(baseline.validation.values, observed.validation.values)


@pytest.mark.parametrize(
    ("config_name", "expected"),
    [("ablation_prefix.yaml", 40), ("ablation_channels.yaml", 50)],
)
def test_ablation_contract_builds_validates_and_rejects_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    config_name: str,
    expected: int,
) -> None:
    config = _write_inputs(tmp_path, config_name)
    hashes = {
        "canonical": "a" * 64,
        "split_manifest": "b" * 64,
        "ablation_config": "c" * 64,
    }
    monkeypatch.setattr(ablations, "_input_hashes", lambda _config: hashes)
    payload = build_ablation_contract(config)
    assert payload["matrix"]["cells"] == expected
    assert payload["matrix"]["primary_references"] == 10
    assert validate_ablation_contract(config)["status"] == "valid"
    with pytest.raises(FileExistsError, match="overwrite"):
        build_ablation_contract(config)
    observed = json.loads(config.contract_audit_path.read_text(encoding="utf-8"))
    observed["protocol"]["augmentation"] = True
    config.contract_audit_path.write_text(
        json.dumps(observed, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(PipelineInvariantError, match="stale"):
        validate_ablation_contract(config)
