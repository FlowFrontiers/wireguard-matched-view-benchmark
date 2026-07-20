from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vpncat.errors import PipelineInvariantError
from vpncat.experiment import (
    RunSpec,
    load_primary_experiment_config,
    select_primary_run,
)
from vpncat.folds import FoldIndex
from vpncat.hashing import sha256_file
from vpncat.models.neural import build_neural_model, trainable_parameter_count
from vpncat.neural_config import load_neural_config
from vpncat.neural_data import NeuralSubset, PreparedNeuralRun
from vpncat.neural_runner import build_neural_prediction_frame, run_primary_neural
from vpncat.neural_training import TrainingResult
from vpncat.neural_tuning import SelectedNeuralConfiguration
from vpncat.preprocessing import FoldTargetState


def _run() -> RunSpec:
    return RunSpec(
        protocol="primary",
        experiment_id="sequential_splt__cnn1d",
        representation="sequential_splt",
        model="cnn1d",
        family="neural",
        fold=1,
        seed=42,
        train_domain="inner",
        test_domains=("inner", "outer"),
    )


def _subset(pair_ids: tuple[str, ...], positions: np.ndarray) -> NeuralSubset:
    return NeuralSubset(
        pair_ids=pair_ids,
        positions=positions,
        values=np.zeros((len(pair_ids), 4, 3), dtype=np.float32),
        mask=np.ones((len(pair_ids), 4), dtype=bool),
        targets=np.asarray([0, 1], dtype=np.int64),
    )


def _prepared() -> PreparedNeuralRun:
    pair_ids = ("train:0", "validation:0", "test:0", "test:1")
    fold = FoldIndex(
        fold=1,
        pair_ids=pair_ids,
        sessions=np.asarray([1, 1, 1, 2], dtype=np.int16),
        labels=("A", "B", "A", "B"),
        roles=("train", "validation", "test", "test"),
        train_positions=np.asarray([0], dtype=np.int64),
        validation_positions=np.asarray([1], dtype=np.int64),
        test_positions=np.asarray([2, 3], dtype=np.int64),
    )
    state = FoldTargetState(
        fold=1,
        classes=("A", "B"),
        class_weights=np.asarray([1.0, 1.0]),
        fit_pair_count=1,
        fit_pair_ids_sha256="synthetic",
    )
    test_pair_ids = ("test:0", "test:1")
    positions = np.asarray([2, 3], dtype=np.int64)
    training = _subset(("train:0", "train:1"), np.asarray([0, 0]))
    validation = _subset(("validation:0", "validation:1"), np.asarray([1, 1]))
    return PreparedNeuralRun(
        fold=fold,
        state=state,
        channels=("direction", "size", "iat_ms"),
        training=training,
        validation=validation,
        tests={
            "inner": _subset(test_pair_ids, positions),
            "outer": _subset(test_pair_ids, positions),
        },
    )


def test_prediction_frame_preserves_paired_test_identity_across_views() -> None:
    run = _run()
    probabilities = {
        "inner": np.asarray([[0.8, 0.2], [0.1, 0.9]]),
        "outer": np.asarray([[0.4, 0.6], [0.7, 0.3]]),
    }
    frame = build_neural_prediction_frame(run, _prepared(), probabilities)

    assert len(frame) == 4
    assert frame.groupby("test_domain")["pair_id"].apply(tuple).to_dict() == {
        "inner": ("test:0", "test:1"),
        "outer": ("test:0", "test:1"),
    }
    assert frame["train_domain"].unique().tolist() == ["inner"]
    assert frame["prediction"].tolist() == ["A", "B", "B", "A"]


def test_prediction_frame_rejects_wrong_probability_shape() -> None:
    with pytest.raises(PipelineInvariantError, match="probability shape"):
        build_neural_prediction_frame(
            _run(),
            _prepared(),
            {
                "inner": np.ones((2, 2)),
                "outer": np.ones((2, 3)),
            },
        )


def test_primary_runner_binds_selection_and_uses_one_model_for_both_views(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root = Path(__file__).parents[1]
    primary = load_primary_experiment_config(project_root / "configs" / "primary.yaml")
    neural = load_neural_config(project_root / "configs" / "neural.yaml")
    run = select_primary_run(
        primary,
        experiment_id="sequential_splt__cnn1d",
        fold=1,
        train_domain="inner",
        seed=42,
    )
    prepared = _prepared()
    trial = neural.trials[0]
    expected_parameters = trainable_parameter_count(
        build_neural_model(
            "cnn1d",
            feature_count=3,
            class_count=2,
            width=trial.width,
            dropout=trial.dropout,
            maximum_length=neural.maximum_prefix_length,
            topology=neural.topologies["cnn1d"],
        )
    )
    selected = SelectedNeuralConfiguration(
        model="cnn1d",
        trial=trial,
        result={"parameter_count": expected_parameters},
        selected_path=tmp_path / "selected.json",
        selected_sha256="b" * 64,
        tuning_manifest_sha256="c" * 64,
        tuning_revision="revision",
        tuning_environment={"torch": "test"},
        tuning_device="cpu",
    )
    calls: dict[str, object] = {"prediction_models": []}

    monkeypatch.setattr("vpncat.neural_runner.validate_contract_audit", lambda *_: None)
    monkeypatch.setattr(
        "vpncat.neural_runner.load_selected_neural_configuration",
        lambda *_args, **_kwargs: selected,
    )
    monkeypatch.setattr(
        "vpncat.neural_runner.prepare_neural_run",
        lambda *_args, **_kwargs: prepared,
    )

    def fake_train(model, *_args, **_kwargs):
        calls["trained_model"] = model
        history = pd.DataFrame(
            [[1, 1.0, 0.9, 0.5, trial.learning_rate]],
            columns=(
                "epoch",
                "train_loss",
                "validation_loss",
                "validation_macro_f1",
                "learning_rate",
            ),
        )
        return TrainingResult(
            model=model,
            history=history,
            best_epoch=1,
            best_validation_macro_f1=0.5,
            validation_loss_at_best_epoch=0.9,
            epochs_completed=1,
            parameter_count=expected_parameters,
            device="cpu",
        )

    def fake_predict(model, *_args, **_kwargs):
        calls["prediction_models"].append(model)
        return np.asarray([[0.8, 0.2], [0.1, 0.9]])

    def fake_write(*args, **kwargs):
        calls["write_args"] = args
        calls["write_kwargs"] = kwargs
        return tmp_path / "completed-run"

    monkeypatch.setattr("vpncat.neural_runner.train_neural_model", fake_train)
    monkeypatch.setattr("vpncat.neural_runner.predict_neural_probabilities", fake_predict)
    monkeypatch.setattr("vpncat.neural_runner.write_completed_run", fake_write)

    output = run_primary_neural(primary, neural, run, device_name="cpu")

    assert output == tmp_path / "completed-run"
    assert calls["prediction_models"] == [calls["trained_model"]] * 2
    write_kwargs = calls["write_kwargs"]
    assert write_kwargs["additional_input_hashes"] == {
        "neural_config": sha256_file(neural.config_path),
        "neural_tuning_selection": "b" * 64,
        "neural_tuning_manifest": "c" * 64,
    }
    assert write_kwargs["training_history"].iloc[0]["epoch"] == 1
