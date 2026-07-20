from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import vpncat.cross_session_orchestration as orchestration
from vpncat.cross_session_index import CrossSessionIndex
from vpncat.cross_session_preprocessing import CrossSessionTargetState
from vpncat.cross_session_preprocessing_audit import (
    load_cross_session_preprocessing_config,
)
from vpncat.errors import PipelineInvariantError
from vpncat.neural_config import load_neural_config
from vpncat.neural_tuning import SelectedNeuralConfiguration
from vpncat.preprocessing import pair_id_digest


def _configs(tmp_path: Path):
    project_root = Path(__file__).parents[1]
    config = load_cross_session_preprocessing_config(
        project_root / "configs" / "cross_session_preprocessing.yaml",
        output_root=tmp_path / "cross-session",
    )
    neural = load_neural_config(project_root / "configs" / "neural.yaml")
    return config, neural


def _index(train_session: int) -> CrossSessionIndex:
    test_session = 3 - train_session
    pair_ids = tuple(f"pair:{index}" for index in range(8))
    return CrossSessionIndex(
        train_session=train_session,
        test_session=test_session,
        pair_ids=pair_ids,
        sessions=np.asarray(
            [train_session] * 6 + [test_session] * 2,
            dtype=np.int16,
        ),
        labels=("A", "B", "A", "B", "A", "B", "A", "B"),
        roles=(
            "train",
            "train",
            "train",
            "train",
            "validation",
            "validation",
            "test",
            "test",
        ),
        train_positions=np.asarray([0, 1, 2, 3], dtype=np.int64),
        validation_positions=np.asarray([4, 5], dtype=np.int64),
        test_positions=np.asarray([6, 7], dtype=np.int64),
    )


def _state(train_session: int) -> CrossSessionTargetState:
    index = _index(train_session)
    return CrossSessionTargetState(
        train_session=train_session,
        classes=("A", "B"),
        class_weights=np.asarray([1.0, 1.0]),
        fit_pair_count=4,
        fit_pair_ids_sha256=pair_id_digest(index.pair_ids_for("train")),
    )


def _classical_filter(*, model: str = "random_forest"):
    return orchestration.CrossSessionRunFilters(
        models=(model,),
        representations=("matched_flow_stats",),
        train_sessions=(1,),
        seeds=(42,),
    )


def _patch_common(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(orchestration, "_clean_revision", lambda _config: "revision")
    monkeypatch.setattr(
        orchestration,
        "validate_cross_session_run_contracts",
        lambda _config, _runs: ({"base": "a" * 64}, {}),
    )
    monkeypatch.setattr(
        orchestration,
        "_load_indexes",
        lambda _config: {1: _index(1), 2: _index(2)},
    )
    monkeypatch.setattr(
        orchestration,
        "_audited_state",
        lambda _config, run: _state(run.train_session),
    )


def test_cross_session_filters_cover_frozen_matrix_and_reject_unknown(
    tmp_path: Path,
) -> None:
    config, _ = _configs(tmp_path)
    all_runs = orchestration.select_cross_session_runs(
        config, orchestration.CrossSessionRunFilters()
    )
    assert len(all_runs) == 30
    assert sum(run.family == "classical" for run in all_runs) == 12
    assert sum(run.family == "neural" for run in all_runs) == 18
    with pytest.raises(PipelineInvariantError, match="Unknown cross-session models"):
        orchestration.select_cross_session_runs(
            config,
            orchestration.CrossSessionRunFilters(models=("unknown",)),
        )


def test_plan_only_preflight_does_not_require_neural_selections(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, neural = _configs(tmp_path)
    _patch_common(monkeypatch)

    def forbidden(*args, **kwargs):
        raise AssertionError("plan-only preflight must not load tuning")

    monkeypatch.setattr(orchestration, "_load_neural_selection", forbidden)
    report = orchestration.run_cross_session_matrix(config, neural)
    assert report["selected_run_count"] == 30
    assert report["counts"] == {"complete": 0, "executed": 0, "pending": 30}
    assert report["status"] == "partial"


def test_cross_session_matrix_executes_and_revalidates_one_classical_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, neural = _configs(tmp_path)
    _patch_common(monkeypatch)
    validations: list[dict[str, object]] = []

    def fake_run(current, run):
        output = current.cross_session.output_root / run.relative_output_dir
        output.mkdir(parents=True)
        return output

    def fake_validate(_output, **kwargs):
        validations.append(kwargs)
        return {"prediction_rows": 4}

    monkeypatch.setattr(orchestration, "run_cross_session_classical", fake_run)
    monkeypatch.setattr(
        orchestration, "validate_completed_cross_session_run", fake_validate
    )
    report = orchestration.run_cross_session_matrix(
        config,
        neural,
        filters=_classical_filter(),
        execute=True,
        maximum_pending_runs=1,
    )
    assert report["counts"] == {"complete": 0, "executed": 1, "pending": 0}
    assert len(validations) == 1
    assert validations[0]["expected_input_hashes"] == {"base": "a" * 64}
    assert validations[0]["expected_git_revision"] == "revision"


def test_bounded_full_matrix_classical_smoke_does_not_load_neural_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, neural = _configs(tmp_path)
    _patch_common(monkeypatch)

    def forbidden(*args, **kwargs):
        raise AssertionError("a pending but unexecuted neural run must not load tuning")

    def fake_run(current, run):
        output = current.cross_session.output_root / run.relative_output_dir
        output.mkdir(parents=True)
        return output

    monkeypatch.setattr(orchestration, "_load_neural_selection", forbidden)
    monkeypatch.setattr(orchestration, "run_cross_session_classical", fake_run)
    monkeypatch.setattr(
        orchestration,
        "validate_completed_cross_session_run",
        lambda *_args, **_kwargs: {"prediction_rows": 4},
    )
    report = orchestration.run_cross_session_matrix(
        config,
        neural,
        execute=True,
        maximum_pending_runs=1,
    )
    assert report["counts"] == {"complete": 0, "executed": 1, "pending": 29}


def test_incompatible_existing_cross_session_run_blocks_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, neural = _configs(tmp_path)
    _patch_common(monkeypatch)
    run = orchestration.select_cross_session_runs(config, _classical_filter())[0]
    (config.cross_session.output_root / run.relative_output_dir).mkdir(parents=True)
    executed = False

    def invalid(*args, **kwargs):
        raise PipelineInvariantError("tampered")

    def forbidden(*args, **kwargs):
        nonlocal executed
        executed = True

    monkeypatch.setattr(
        orchestration, "validate_completed_cross_session_run", invalid
    )
    monkeypatch.setattr(orchestration, "run_cross_session_classical", forbidden)
    with pytest.raises(PipelineInvariantError, match="Existing cross-session output"):
        orchestration.run_cross_session_matrix(
            config,
            neural,
            filters=_classical_filter(),
            execute=True,
        )
    assert executed is False


def test_cross_session_neural_selection_loaded_once_and_hash_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, neural = _configs(tmp_path)
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
        tuning_device="cpu",
    )
    loads = 0
    validations: list[dict[str, object]] = []

    def load(*args, **kwargs):
        nonlocal loads
        loads += 1
        return selected

    def execute(current, _neural, run, **kwargs):
        assert kwargs["selected"] is selected
        output = current.cross_session.output_root / run.relative_output_dir
        output.mkdir(parents=True)
        return output

    def validate(_output, **kwargs):
        validations.append(kwargs)
        return {"prediction_rows": 4}

    monkeypatch.setattr(orchestration, "_load_neural_selection", load)
    monkeypatch.setattr(orchestration, "_run_neural", execute)
    monkeypatch.setattr(
        orchestration, "validate_completed_cross_session_run", validate
    )
    report = orchestration.run_cross_session_matrix(
        config,
        neural,
        filters=orchestration.CrossSessionRunFilters(
            models=("cnn1d",),
            train_sessions=(1,),
        ),
        execute=True,
        maximum_pending_runs=2,
        device_name="cpu",
    )
    assert loads == 1
    assert report["counts"] == {"complete": 0, "executed": 2, "pending": 1}
    assert all(
        row["expected_input_hashes"]["neural_tuning_selection"] == "b" * 64
        for row in validations
    )


def test_cross_session_xgboost_dispatches_to_isolated_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, neural = _configs(tmp_path)
    _patch_common(monkeypatch)
    dispatched: list[str] = []

    def isolated(current, run):
        dispatched.append(run.run_id)
        output = current.cross_session.output_root / run.relative_output_dir
        output.mkdir(parents=True)
        return output

    monkeypatch.setattr(orchestration, "_run_xgboost_isolated", isolated)
    monkeypatch.setattr(
        orchestration,
        "validate_completed_cross_session_run",
        lambda *_args, **_kwargs: {"prediction_rows": 4},
    )
    report = orchestration.run_cross_session_matrix(
        config,
        neural,
        filters=_classical_filter(model="xgboost"),
        execute=True,
    )
    assert dispatched == [report["runs"][0]["run_id"]]
