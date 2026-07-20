from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import vpncat.ablation_orchestration as orchestration
from vpncat.ablation_orchestration import (
    AblationRunFilters,
    run_ablation_matrix,
    select_ablation_runs,
)
from vpncat.ablations import enumerate_ablation_runs, load_ablation_config
from vpncat.errors import PipelineInvariantError
from vpncat.folds import FoldIndex


def _project_root() -> Path:
    return Path(__file__).parents[1]


def _fold(number: int) -> FoldIndex:
    return FoldIndex(
        fold=number,
        pair_ids=("train:a", "train:b", "validation:a", "validation:b", "test:a", "test:b"),
        sessions=np.asarray([1, 2, 1, 2, 1, 2], dtype=np.int16),
        labels=("A", "B", "A", "B", "A", "B"),
        roles=("train", "train", "validation", "validation", "test", "test"),
        train_positions=np.asarray([0, 1], dtype=np.int64),
        validation_positions=np.asarray([2, 3], dtype=np.int64),
        test_positions=np.asarray([4, 5], dtype=np.int64),
    )


def _config(tmp_path: Path, name: str):
    base = load_ablation_config(_project_root() / "configs" / name)
    primary = replace(base.primary, output_root=tmp_path / "primary")
    return replace(base, primary=primary, output_root=tmp_path / "ablation")


def _patch_preflight(monkeypatch: pytest.MonkeyPatch, config) -> None:
    monkeypatch.setattr(orchestration, "_clean_revision", lambda *_: "revision")
    monkeypatch.setattr(
        orchestration,
        "validate_ablation_run_contracts",
        lambda *_: {"contract": "a" * 64},
    )
    monkeypatch.setattr(
        orchestration,
        "_folds",
        lambda *_: {number: _fold(number) for number in config.folds},
    )


@pytest.mark.parametrize(
    ("config_name", "cells", "training", "references"),
    [
        ("ablation_prefix.yaml", 40, 30, 10),
        ("ablation_channels.yaml", 50, 40, 10),
    ],
)
def test_preflight_counts_cells_without_loading_tuning_or_torch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    config_name: str,
    cells: int,
    training: int,
    references: int,
) -> None:
    config = _config(tmp_path, config_name)
    _patch_preflight(monkeypatch, config)
    monkeypatch.setattr(
        orchestration,
        "_load_selection",
        lambda *_: pytest.fail("preflight must not load tuning selections"),
    )
    report = run_ablation_matrix(config)
    assert report["selected_cell_count"] == cells
    assert report["selected_training_count"] == training
    assert report["selected_reference_count"] == references
    assert report["counts"] == {
        "complete": 0,
        "executed": 0,
        "pending": training,
        "reference_complete": 0,
        "reference_pending": references,
    }
    assert report["status"] == "partial"


def test_filters_are_exact_and_reject_unknown_values(tmp_path: Path) -> None:
    config = _config(tmp_path, "ablation_prefix.yaml")
    selected = select_ablation_runs(
        config,
        AblationRunFilters(models=("cnn1d",), observations=("n010",), folds=(1,)),
    )
    assert len(selected) == 1
    assert selected[0].model == "cnn1d" and selected[0].prefix_length == 10
    with pytest.raises(PipelineInvariantError, match="unknown"):
        select_ablation_runs(config, AblationRunFilters(observations=("n999",)))


def test_controller_executes_only_pending_training_cells(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, "ablation_prefix.yaml")
    _patch_preflight(monkeypatch, config)
    selected = SimpleNamespace(
        model="cnn1d",
        selected_sha256="b" * 64,
        tuning_manifest_sha256="c" * 64,
    )
    monkeypatch.setattr(orchestration, "_load_selection", lambda *_: selected)

    def fake_execute(config, run, **_kwargs):
        output = config.output_root / run.relative_output_dir
        output.mkdir(parents=True)
        return output

    monkeypatch.setattr(orchestration, "_execute", fake_execute)
    monkeypatch.setattr(
        orchestration,
        "validate_completed_ablation_run",
        lambda *_args, **_kwargs: {"prediction_rows": 4},
    )
    report = run_ablation_matrix(
        config,
        filters=AblationRunFilters(models=("cnn1d",), observations=("n010",), folds=(1,)),
        execute=True,
        maximum_pending_runs=1,
        device_name="cpu",
    )
    assert report["counts"]["executed"] == 1
    assert report["counts"]["pending"] == 0
    assert report["selected_reference_count"] == 0
    assert report["status"] == "complete"


def test_existing_output_is_revalidated_and_incompatibility_is_fatal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, "ablation_channels.yaml")
    _patch_preflight(monkeypatch, config)
    run = next(
        run
        for run in enumerate_ablation_runs(config)
        if run.model == "transformer"
        and run.observation_id == "size_timing"
        and run.fold == 1
    )
    (config.output_root / run.relative_output_dir).mkdir(parents=True)
    selected = SimpleNamespace(
        model="transformer",
        selected_sha256="b" * 64,
        tuning_manifest_sha256="c" * 64,
    )
    monkeypatch.setattr(orchestration, "_load_selection", lambda *_: selected)
    monkeypatch.setattr(
        orchestration,
        "validate_completed_ablation_run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(PipelineInvariantError("tampered")),
    )
    with pytest.raises(PipelineInvariantError, match="Existing ablation output"):
        run_ablation_matrix(config, filters=AblationRunFilters(run_ids=(run.run_id,)))


def test_reference_only_execution_never_calls_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, "ablation_prefix.yaml")
    _patch_preflight(monkeypatch, config)
    monkeypatch.setattr(
        orchestration,
        "_load_selection",
        lambda *_: pytest.fail("missing reference requires no tuning selection"),
    )
    monkeypatch.setattr(
        orchestration,
        "_execute",
        lambda *_args, **_kwargs: pytest.fail("primary reference must never execute"),
    )
    report = run_ablation_matrix(
        config,
        filters=AblationRunFilters(observations=("n050",)),
        execute=True,
    )
    assert report["selected_training_count"] == 0
    assert report["counts"]["reference_pending"] == 10
