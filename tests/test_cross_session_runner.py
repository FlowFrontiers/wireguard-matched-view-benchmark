from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from vpncat.cross_session import CrossSessionRunSpec
from vpncat.cross_session_artifacts import (
    validate_completed_cross_session_run,
    write_completed_cross_session_run,
)
from vpncat.cross_session_data import PreparedCrossSessionClassical
from vpncat.cross_session_index import CrossSessionIndex
from vpncat.cross_session_metrics import (
    compute_cross_session_metrics,
    validate_cross_session_predictions,
)
from vpncat.cross_session_preprocessing import CrossSessionTargetState
from vpncat.cross_session_runner import build_cross_session_prediction_frame
from vpncat.errors import PipelineInvariantError
from vpncat.hashing import sha256_file
from vpncat.preprocessing import pair_id_digest


def _run() -> CrossSessionRunSpec:
    return CrossSessionRunSpec(
        protocol="cross_session",
        experiment_id="flattened_splt__random_forest",
        representation="flattened_splt",
        model="random_forest",
        family="classical",
        seed=42,
        train_session=1,
        test_session=2,
        train_domain="inner",
        test_domains=("inner", "outer"),
    )


def _prepared() -> PreparedCrossSessionClassical:
    pair_ids = tuple(f"pair:{index}" for index in range(8))
    labels = ("A", "B", "A", "B", "A", "B", "A", "B")
    index = CrossSessionIndex(
        train_session=1,
        test_session=2,
        pair_ids=pair_ids,
        sessions=np.asarray([1, 1, 1, 1, 1, 1, 2, 2], dtype=np.int16),
        labels=labels,
        roles=("train", "train", "train", "train", "validation", "validation", "test", "test"),
        train_positions=np.asarray([0, 1, 2, 3], dtype=np.int64),
        validation_positions=np.asarray([4, 5], dtype=np.int64),
        test_positions=np.asarray([6, 7], dtype=np.int64),
    )
    state = CrossSessionTargetState(
        train_session=1,
        classes=("A", "B"),
        class_weights=np.asarray([1.0, 1.0]),
        fit_pair_count=4,
        fit_pair_ids_sha256=pair_id_digest(index.pair_ids_for("train")),
    )
    return PreparedCrossSessionClassical(
        index=index,
        state=state,
        feature_names=("feature",),
        training_values=np.asarray([[0.0], [1.0], [2.0], [3.0]]),
        training_targets=np.asarray([0, 1, 0, 1]),
        training_labels=np.asarray(["A", "B", "A", "B"]),
        validation_values=np.asarray([[4.0], [5.0]]),
        validation_targets=np.asarray([0, 1]),
        test_values={"inner": np.asarray([[6.0], [7.0]]), "outer": np.asarray([[8.0], [9.0]])},
    )


def _predictions() -> tuple[PreparedCrossSessionClassical, pd.DataFrame]:
    prepared = _prepared()
    probabilities = {
        "inner": np.asarray([[0.9, 0.1], [0.2, 0.8]]),
        "outer": np.asarray([[0.7, 0.3], [0.4, 0.6]]),
    }
    return prepared, build_cross_session_prediction_frame(
        _run(), prepared, probabilities
    )


def test_cross_session_predictions_and_metrics_are_paired() -> None:
    prepared, predictions = _predictions()
    classes = prepared.state.classes
    probabilities = validate_cross_session_predictions(
        predictions, run=_run(), index=prepared.index, classes=classes
    )
    assert probabilities.shape == (4, 2)
    assert set(predictions["session"]) == {2}
    inner = predictions.loc[predictions["test_domain"] == "inner", "pair_id"].tolist()
    outer = predictions.loc[predictions["test_domain"] == "outer", "pair_id"].tolist()
    assert inner == outer == ["pair:6", "pair:7"]
    metrics = compute_cross_session_metrics(
        predictions, run=_run(), index=prepared.index, classes=classes
    )
    assert all(
        value == pytest.approx(1.0)
        for domain in metrics.values()
        for value in domain.values()
    )


def test_cross_session_prediction_guards_reject_corruption() -> None:
    prepared, predictions = _predictions()
    duplicated = pd.concat([predictions, predictions.iloc[[0]]], ignore_index=True)
    with pytest.raises(PipelineInvariantError, match="duplicate"):
        validate_cross_session_predictions(
            duplicated,
            run=_run(),
            index=prepared.index,
            classes=prepared.state.classes,
        )
    wrong_session = predictions.copy()
    wrong_session.loc[0, "session"] = 1
    with pytest.raises(PipelineInvariantError, match="sessions"):
        validate_cross_session_predictions(
            wrong_session,
            run=_run(),
            index=prepared.index,
            classes=prepared.state.classes,
        )
    wrong_argmax = predictions.copy()
    wrong_argmax.loc[0, "prediction"] = "B"
    with pytest.raises(PipelineInvariantError, match="argmax"):
        validate_cross_session_predictions(
            wrong_argmax,
            run=_run(),
            index=prepared.index,
            classes=prepared.state.classes,
        )


def test_cross_session_atomic_publication_and_tamper_detection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared, predictions = _predictions()
    split = tmp_path / "split.csv"
    split.write_text("pair_id,role\npair:6,test\n", encoding="utf-8")
    output_root = tmp_path / "outputs"
    cross = SimpleNamespace(output_root=output_root, split_path=split)
    config = SimpleNamespace(cross_session=cross)
    input_hashes = {"cross_session_split": sha256_file(split)}
    manifest = {
        "schema_version": 1,
        "status": "staging",
        "package_version": "0.1.0",
        "git": {"revision": "test", "dirty": False},
        "run": _run().to_dict(),
        "preprocessing": {"train_domain": "inner", "train_session": 1},
        "input_hashes": input_hashes,
        "split_manifest_sha256": input_hashes["cross_session_split"],
        "class_order": list(prepared.state.classes),
    }
    monkeypatch.setattr(
        "vpncat.cross_session_artifacts._build_manifest",
        lambda *args, **kwargs: manifest.copy(),
    )
    monkeypatch.setattr(
        "vpncat.cross_session_artifacts.verify_cross_session_input_chain",
        lambda *args, **kwargs: input_hashes,
    )
    monkeypatch.setattr(
        "vpncat.cross_session_artifacts.bind_cross_session_state",
        lambda *args, **kwargs: {
            "train_domain": "inner",
            "train_session": 1,
        },
    )
    target = write_completed_cross_session_run(
        config,
        _run(),
        prepared.index,
        prepared.state,
        predictions,
        model_hyperparameters={"n_estimators": 10},
    )
    assert target.is_dir()
    validation = validate_completed_cross_session_run(
        target,
        config=config,
        run=_run(),
        index=prepared.index,
        state=prepared.state,
        classes=prepared.state.classes,
        expected_git_revision="test",
    )
    assert validation["prediction_rows"] == 4
    with pytest.raises(FileExistsError, match="overwrite"):
        write_completed_cross_session_run(
            config,
            _run(),
            prepared.index,
            prepared.state,
            predictions,
            model_hyperparameters={"n_estimators": 10},
        )
    run_json_path = target / "run.json"
    original_manifest = json.loads(run_json_path.read_text(encoding="utf-8"))
    poisoned_manifest = dict(original_manifest)
    poisoned_manifest["preprocessing"] = {
        **poisoned_manifest["preprocessing"],
        "train_session": 2,
    }
    run_json_path.write_text(
        json.dumps(poisoned_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(PipelineInvariantError, match="preprocessing payload"):
        validate_completed_cross_session_run(
            target,
            config=config,
            run=_run(),
            index=prepared.index,
            state=prepared.state,
            classes=prepared.state.classes,
        )
    run_json_path.write_text(
        json.dumps(original_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    metrics_path = target / "metrics_long.csv"
    metrics_path.write_text(metrics_path.read_text() + "tamper", encoding="utf-8")
    with pytest.raises(PipelineInvariantError, match="hash mismatch"):
        validate_completed_cross_session_run(
            target,
            config=config,
            run=_run(),
            index=prepared.index,
            state=prepared.state,
            classes=prepared.state.classes,
        )


def test_cross_session_runner_refuses_existing_output_before_preparation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vpncat.cross_session_runner import run_cross_session_classical

    target = tmp_path / _run().relative_output_dir
    target.mkdir(parents=True)
    config = SimpleNamespace(
        cross_session=SimpleNamespace(output_root=tmp_path),
    )
    prepared = False

    def should_not_prepare(*args, **kwargs):
        nonlocal prepared
        prepared = True
        raise AssertionError("preparation must not run")

    monkeypatch.setattr(
        "vpncat.cross_session_runner.prepare_cross_session_classical",
        should_not_prepare,
    )
    with pytest.raises(FileExistsError, match="overwrite"):
        run_cross_session_classical(config, _run())
    assert prepared is False
