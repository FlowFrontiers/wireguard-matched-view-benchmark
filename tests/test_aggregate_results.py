from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import vpncat.ablation_orchestration as ablation_orchestration
import vpncat.aggregate_results as aggregate
import vpncat.cross_session_orchestration as cross_orchestration
import vpncat.cross_session_preprocessing_audit as cross_preprocessing
import vpncat.dann_orchestration as dann_orchestration
import vpncat.orchestration as primary_orchestration
from vpncat.errors import PipelineInvariantError


def _config(tmp_path: Path):
    primary = SimpleNamespace(
        canonical_path=tmp_path / "artifacts" / "canonical.parquet",
        output_root=tmp_path / "primary",
    )
    return SimpleNamespace(
        project_root=tmp_path,
        output_root=tmp_path / "analysis",
        primary=primary,
        cross_session=SimpleNamespace(output_root=tmp_path / "cross"),
        dann=SimpleNamespace(neural=object()),
        ablation_prefix=object(),
        ablation_channels=object(),
    )


def test_aggregation_refuses_existing_output_before_campaign_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    config.output_root.mkdir()
    monkeypatch.setattr(aggregate, "validate_analysis_contract", lambda *_: None)
    monkeypatch.setattr(
        aggregate,
        "_validate_campaign",
        lambda *_: pytest.fail("campaign validation must not run before overwrite refusal"),
    )
    with pytest.raises(FileExistsError, match="overwrite"):
        aggregate.aggregate_results(config)


def test_campaign_validation_rejects_any_pending_protocol(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    monkeypatch.setattr(
        aggregate,
        "git_provenance",
        lambda *_: {"status_available": True, "dirty": False, "revision": "clean"},
    )
    monkeypatch.setattr(
        cross_preprocessing,
        "load_cross_session_preprocessing_config",
        lambda *_args, **_kwargs: object(),
    )
    complete = {"status": "complete", "counts": {"pending": 0}}
    monkeypatch.setattr(primary_orchestration, "run_primary_matrix", lambda *_: complete)
    monkeypatch.setattr(cross_orchestration, "run_cross_session_matrix", lambda *_: complete)
    monkeypatch.setattr(dann_orchestration, "run_dann_matrix", lambda *_: complete)
    monkeypatch.setattr(ablation_orchestration, "run_ablation_matrix", lambda *_: complete)
    monkeypatch.setattr(
        cross_orchestration,
        "run_cross_session_matrix",
        lambda *_: {"status": "partial", "counts": {"pending": 1}},
    )
    with pytest.raises(PipelineInvariantError, match="incomplete"):
        aggregate._validate_campaign(config)
