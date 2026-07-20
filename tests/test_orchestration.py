from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import vpncat.orchestration as orchestration
from vpncat.errors import PipelineInvariantError
from vpncat.experiment import load_primary_experiment_config
from vpncat.folds import FoldIndex
from vpncat.neural_config import load_neural_config
from vpncat.neural_tuning import SelectedNeuralConfiguration
from vpncat.orchestration import (
    PrimaryRunFilters,
    run_primary_matrix,
    select_primary_runs,
)


def _configs(tmp_path: Path):
    project_root = Path(__file__).parents[1]
    primary = load_primary_experiment_config(project_root / "configs" / "primary.yaml")
    primary = replace(primary, output_root=tmp_path / "primary")
    neural = load_neural_config(project_root / "configs" / "neural.yaml")
    return primary, neural


def _fold() -> FoldIndex:
    pair_ids = tuple(f"pair:{index}" for index in range(8))
    return FoldIndex(
        fold=1,
        pair_ids=pair_ids,
        sessions=np.ones(8, dtype=np.int16),
        labels=("A", "B", "A", "B", "A", "B", "A", "B"),
        roles=("train", "train", "train", "train", "validation", "validation", "test", "test"),
        train_positions=np.asarray([0, 1, 2, 3], dtype=np.int64),
        validation_positions=np.asarray([4, 5], dtype=np.int64),
        test_positions=np.asarray([6, 7], dtype=np.int64),
    )


def _classical_filter() -> PrimaryRunFilters:
    return PrimaryRunFilters(
        models=("random_forest",),
        representations=("matched_flow_stats",),
        folds=(1,),
        train_domains=("inner",),
        seeds=(42,),
    )


def _patch_common(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(orchestration, "_clean_revision", lambda _config: "revision")
    monkeypatch.setattr(orchestration, "_load_fold_indexes", lambda _config: {1: _fold()})
    monkeypatch.setattr(orchestration, "verify_input_chain", lambda _config: {"base": "a" * 64})
    monkeypatch.setattr(orchestration, "validate_contract_audit", lambda *_args: None)


def test_primary_filters_are_deterministic_and_reject_unknown_values(tmp_path: Path) -> None:
    primary, _ = _configs(tmp_path)
    selected = select_primary_runs(primary, _classical_filter())
    assert len(selected) == 1
    assert selected[0].experiment_id == "matched_flow_stats__random_forest"
    with pytest.raises(PipelineInvariantError, match="Unknown primary models"):
        select_primary_runs(primary, PrimaryRunFilters(models=("unknown",)))


def test_primary_matrix_executes_pending_run_and_revalidates_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    primary, neural = _configs(tmp_path)
    _patch_common(monkeypatch)
    validations: list[dict[str, object]] = []

    def fake_runner(config, run):
        output = config.output_root / run.relative_output_dir
        output.mkdir(parents=True)
        return output

    def fake_validate(_output, **kwargs):
        validations.append(kwargs)
        return {"prediction_rows": 4}

    monkeypatch.setattr(orchestration, "run_primary_classical", fake_runner)
    monkeypatch.setattr(orchestration, "validate_completed_run", fake_validate)
    report = run_primary_matrix(
        primary,
        neural,
        filters=_classical_filter(),
        execute=True,
        maximum_pending_runs=1,
    )

    assert report["counts"] == {"complete": 0, "executed": 1, "pending": 0}
    assert report["status"] == "complete"
    assert len(validations) == 1
    assert validations[0]["expected_input_hashes"] == {"base": "a" * 64}
    assert validations[0]["expected_git_revision"] == "revision"


def test_incompatible_existing_run_stops_before_any_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    primary, neural = _configs(tmp_path)
    _patch_common(monkeypatch)
    runs = select_primary_runs(primary, _classical_filter())
    (primary.output_root / runs[0].relative_output_dir).mkdir(parents=True)
    executed = False

    def invalid(*_args, **_kwargs):
        raise PipelineInvariantError("tampered")

    def forbidden(*_args, **_kwargs):
        nonlocal executed
        executed = True

    monkeypatch.setattr(orchestration, "validate_completed_run", invalid)
    monkeypatch.setattr(orchestration, "run_primary_classical", forbidden)
    with pytest.raises(PipelineInvariantError, match="Existing primary output"):
        run_primary_matrix(
            primary,
            neural,
            filters=_classical_filter(),
            execute=True,
        )
    assert executed is False


def test_neural_selection_is_loaded_once_and_bound_to_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    primary, neural = _configs(tmp_path)
    _patch_common(monkeypatch)
    trial = neural.trials[0]
    selected = SelectedNeuralConfiguration(
        model="cnn1d",
        trial=trial,
        result={"parameter_count": 1},
        selected_path=tmp_path / "selected.json",
        selected_sha256="b" * 64,
        tuning_manifest_sha256="c" * 64,
        tuning_revision="revision",
        tuning_environment={"torch": "test"},
        tuning_device="mps",
    )
    loads = 0
    observed_selected = None
    observed_hashes = None

    def fake_load(*_args, **_kwargs):
        nonlocal loads
        loads += 1
        return selected

    def fake_runner(config, _neural, run, *, device_name, selected):
        nonlocal observed_selected
        assert device_name == "mps"
        observed_selected = selected
        output = config.output_root / run.relative_output_dir
        output.mkdir(parents=True)
        return output

    def fake_validate(_output, **kwargs):
        nonlocal observed_hashes
        observed_hashes = kwargs["expected_input_hashes"]
        return {"prediction_rows": 4}

    monkeypatch.setattr(orchestration, "_load_neural_selection", fake_load)
    monkeypatch.setattr(orchestration, "_run_neural", fake_runner)
    monkeypatch.setattr(orchestration, "validate_completed_run", fake_validate)
    report = run_primary_matrix(
        primary,
        neural,
        filters=PrimaryRunFilters(
            models=("cnn1d",),
            folds=(1,),
            train_domains=("inner",),
            seeds=(42,),
        ),
        execute=True,
        device_name="mps",
    )

    assert report["counts"]["executed"] == 1
    assert loads == 1
    assert observed_selected is selected
    assert observed_hashes["neural_tuning_selection"] == "b" * 64
    assert observed_hashes["neural_tuning_manifest"] == "c" * 64


def test_xgboost_execution_uses_isolated_subprocess_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    primary, neural = _configs(tmp_path)
    _patch_common(monkeypatch)
    isolated_runs = []

    def fake_isolated(config, run):
        isolated_runs.append(run.run_id)
        output = config.output_root / run.relative_output_dir
        output.mkdir(parents=True)
        return output

    monkeypatch.setattr(orchestration, "_run_xgboost_isolated", fake_isolated)
    monkeypatch.setattr(
        orchestration,
        "validate_completed_run",
        lambda *_args, **_kwargs: {"prediction_rows": 4},
    )
    report = run_primary_matrix(
        primary,
        neural,
        filters=PrimaryRunFilters(
            models=("xgboost",),
            representations=("matched_flow_stats",),
            folds=(1,),
            train_domains=("inner",),
            seeds=(42,),
        ),
        execute=True,
    )

    assert report["counts"]["executed"] == 1
    assert isolated_runs == [report["runs"][0]["run_id"]]
