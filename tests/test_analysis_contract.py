from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

import vpncat.analysis as analysis
from vpncat.ablations import enumerate_ablation_runs, primary_reference_run
from vpncat.analysis import (
    _validate_analysis_config,
    build_analysis_contract,
    enumerate_analysis_inventory,
    load_analysis_config,
    validate_analysis_contract,
)
from vpncat.cross_session import enumerate_cross_session_runs
from vpncat.dann import enumerate_dann_runs
from vpncat.errors import PipelineInvariantError
from vpncat.experiment import enumerate_primary_runs


def _config_path() -> Path:
    return Path(__file__).parents[1] / "configs" / "analysis.yaml"


def _config():
    return load_analysis_config(_config_path())


def _audit_rows(runs, *, ablation: bool = False):
    rows = []
    for run in runs:
        row = {**run.to_dict(), "prediction_rows": 2}
        if ablation:
            row["artifact_relative_output_dir"] = run.relative_output_dir.as_posix()
            row["primary_reference"] = None
        else:
            row["relative_output_dir"] = run.relative_output_dir.as_posix()
        rows.append(row)
    return rows


def _synthetic_audits(config):
    prefix_rows = _audit_rows(
        enumerate_ablation_runs(config.ablation_prefix), ablation=True
    )
    channel_rows = _audit_rows(
        enumerate_ablation_runs(config.ablation_channels), ablation=True
    )
    for ablation_config, rows in (
        (config.ablation_prefix, prefix_rows),
        (config.ablation_channels, channel_rows),
    ):
        by_id = {row["run_id"]: row for row in rows}
        for run in enumerate_ablation_runs(ablation_config):
            if not run.is_primary_reference:
                continue
            reference = primary_reference_run(ablation_config, run)
            by_id[run.run_id]["primary_reference"] = {
                "run_id": reference.run_id,
                "relative_output_dir": reference.relative_output_dir.as_posix(),
            }
            by_id[run.run_id]["artifact_relative_output_dir"] = (
                reference.relative_output_dir.as_posix()
            )
    return {
        "primary": {"runs": _audit_rows(enumerate_primary_runs(config.primary))},
        "cross_session": {
            "runs": _audit_rows(enumerate_cross_session_runs(config.cross_session))
        },
        "dann": {"runs": _audit_rows(enumerate_dann_runs(config.dann))},
        "ablation_prefix": {"runs": prefix_rows},
        "ablation_channels": {"runs": channel_rows},
    }


def test_analysis_policy_is_frozen() -> None:
    config = _config()
    assert config.metrics == (
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
        "macro_ovr_average_precision",
    )
    assert config.bootstrap == {
        "resampling_unit": "pair_id",
        "paired_views": True,
        "replicates": 1000,
        "confidence_level": 0.95,
        "seed": 42,
    }
    with pytest.raises(PipelineInvariantError, match="bootstrap policy"):
        _validate_analysis_config(
            replace(config, bootstrap={**config.bootstrap, "replicates": 999}),
            {"protocol": "analysis"},
        )


def test_inventory_deduplicates_ablation_references(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    monkeypatch.setattr(
        analysis, "_validated_audits", lambda *_: _synthetic_audits(config)
    )
    artifacts, references = enumerate_analysis_inventory(config)
    assert len(artifacts) == 265
    assert len(references) == 20
    assert len({row["physical_artifact_id"] for row in references}) == 10
    assert {row["protocol"] for row in artifacts} == {
        "primary",
        "cross_session",
        "dann",
        "ablation_prefix",
        "ablation_channels",
    }
    physical_ids = {row["artifact_id"] for row in artifacts}
    assert all(row["physical_artifact_id"] in physical_ids for row in references)
    assert all(
        row["physical_artifact_id"].startswith("primary:") for row in references
    )


def test_contract_builds_validates_and_rejects_policy_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = replace(_config(), contract_audit_path=tmp_path / "analysis.json")
    artifacts = [
        {
            "artifact_id": f"primary:run-{index}",
            "protocol": "primary",
            "expected_prediction_rows": 2,
        }
        for index in range(265)
    ]
    references = [
        {
            "logical_artifact_id": f"ablation:reference-{index}",
            "physical_artifact_id": f"primary:run-{index % 10}",
            "expected_prediction_rows": 2,
        }
        for index in range(20)
    ]
    monkeypatch.setattr(
        analysis, "enumerate_analysis_inventory", lambda *_: (artifacts, references)
    )
    monkeypatch.setattr(analysis, "_input_hashes", lambda *_: {"input": "a" * 64})
    monkeypatch.setattr(
        analysis,
        "git_provenance",
        lambda *_: {"status_available": True, "dirty": False, "revision": "clean"},
    )
    payload = build_analysis_contract(config)
    assert payload["matrix"]["physical_artifacts"] == 265
    assert payload["matrix"]["logical_references"] == 20
    assert validate_analysis_contract(config)["status"] == "valid"
    with pytest.raises(FileExistsError, match="overwrite"):
        build_analysis_contract(config)
    tampered = json.loads(config.contract_audit_path.read_text(encoding="utf-8"))
    tampered["policy"]["bootstrap"]["replicates"] = 999
    config.contract_audit_path.write_text(
        json.dumps(tampered, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(PipelineInvariantError, match="stale"):
        validate_analysis_contract(config)
