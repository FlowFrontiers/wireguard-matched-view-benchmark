from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from vpncat.cross_session import CrossSessionRunSpec, select_cross_session_run
from vpncat.cross_session_artifacts import (
    validate_completed_cross_session_run,
    write_completed_cross_session_run,
)
from vpncat.cross_session_data import PreparedCrossSessionNeural
from vpncat.cross_session_index import CrossSessionIndex
from vpncat.cross_session_neural_runner import (
    build_cross_session_neural_prediction_frame,
    run_cross_session_neural,
)
from vpncat.cross_session_preprocessing import CrossSessionTargetState
from vpncat.cross_session_preprocessing_audit import (
    load_cross_session_preprocessing_config,
)
from vpncat.errors import PipelineInvariantError
from vpncat.hashing import sha256_file
from vpncat.models.neural import build_neural_model, trainable_parameter_count
from vpncat.neural_config import load_neural_config
from vpncat.neural_data import NeuralSubset
from vpncat.neural_training import TrainingResult
from vpncat.neural_tuning import SelectedNeuralConfiguration
from vpncat.preprocessing import pair_id_digest


def _run(*, model: str = "cnn1d") -> CrossSessionRunSpec:
    return CrossSessionRunSpec(
        protocol="cross_session",
        experiment_id=f"sequential_splt__{model}",
        representation="sequential_splt",
        model=model,
        family="neural",
        seed=42,
        train_session=1,
        test_session=2,
        train_domain="inner",
        test_domains=("inner", "outer"),
    )


def _subset(
    pair_ids: tuple[str, ...],
    positions: tuple[int, ...],
    targets: tuple[int, ...],
) -> NeuralSubset:
    count = len(positions)
    values = np.zeros((count, 5, 3), dtype=np.float32)
    values[:, :3, :] = 1.0
    mask = np.zeros((count, 5), dtype=bool)
    mask[:, :3] = True
    return NeuralSubset(
        pair_ids=pair_ids,
        positions=np.asarray(positions, dtype=np.int64),
        values=values,
        mask=mask,
        targets=np.asarray(targets, dtype=np.int64),
    )


def _prepared() -> PreparedCrossSessionNeural:
    pair_ids = tuple(f"pair:{index}" for index in range(8))
    labels = ("A", "B", "A", "B", "A", "B", "A", "B")
    index = CrossSessionIndex(
        train_session=1,
        test_session=2,
        pair_ids=pair_ids,
        sessions=np.asarray([1, 1, 1, 1, 1, 1, 2, 2], dtype=np.int16),
        labels=labels,
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
    state = CrossSessionTargetState(
        train_session=1,
        classes=("A", "B"),
        class_weights=np.asarray([1.0, 1.0]),
        fit_pair_count=4,
        fit_pair_ids_sha256=pair_id_digest(index.pair_ids_for("train")),
    )
    tests = {
        domain: _subset(("pair:6", "pair:7"), (6, 7), (0, 1))
        for domain in ("inner", "outer")
    }
    return PreparedCrossSessionNeural(
        index=index,
        state=state,
        channels=("direction", "size", "iat_ms"),
        training=_subset(("pair:0", "pair:1", "pair:2", "pair:3"), (0, 1, 2, 3), (0, 1, 0, 1)),
        validation=_subset(("pair:4", "pair:5"), (4, 5), (0, 1)),
        tests=tests,
    )


def test_cross_session_neural_predictions_preserve_paired_target_views() -> None:
    prepared = _prepared()
    predictions = build_cross_session_neural_prediction_frame(
        _run(),
        prepared,
        {
            "inner": np.asarray([[0.9, 0.1], [0.2, 0.8]]),
            "outer": np.asarray([[0.7, 0.3], [0.4, 0.6]]),
        },
    )
    assert len(predictions) == 4
    assert set(predictions["session"]) == {2}
    assert predictions.loc[
        predictions["test_domain"] == "inner", "pair_id"
    ].tolist() == predictions.loc[
        predictions["test_domain"] == "outer", "pair_id"
    ].tolist()


def test_cross_session_neural_runner_reuses_primary_selection_and_one_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root = Path(__file__).parents[1]
    config = load_cross_session_preprocessing_config(
        project_root / "configs" / "cross_session_preprocessing.yaml",
        output_root=tmp_path / "outputs",
    )
    neural = load_neural_config(project_root / "configs" / "neural.yaml")
    run = select_cross_session_run(
        config.cross_session,
        experiment_id="sequential_splt__cnn1d",
        train_session=1,
        seed=42,
    )
    prepared = _prepared()
    trial = neural.trials[0]
    parameter_count = trainable_parameter_count(
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
        result={"parameter_count": parameter_count},
        selected_path=tmp_path / "selected.json",
        selected_sha256="b" * 64,
        tuning_manifest_sha256="c" * 64,
        tuning_revision="revision",
        tuning_environment={"torch": "test"},
        tuning_device="cpu",
    )
    calls: dict[str, object] = {"prediction_models": []}
    monkeypatch.setattr(
        "vpncat.cross_session_neural_runner.validate_cross_session_run_contract",
        lambda *_: None,
    )
    monkeypatch.setattr(
        "vpncat.cross_session_neural_runner.prepare_cross_session_neural",
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
            parameter_count=parameter_count,
            device="cpu",
        )

    def fake_predict(model, *_args, **_kwargs):
        calls["prediction_models"].append(model)
        return np.asarray([[0.8, 0.2], [0.1, 0.9]])

    def fake_write(*args, **kwargs):
        calls["write_args"] = args
        calls["write_kwargs"] = kwargs
        return tmp_path / "completed"

    monkeypatch.setattr(
        "vpncat.cross_session_neural_runner.train_neural_model", fake_train
    )
    monkeypatch.setattr(
        "vpncat.cross_session_neural_runner.predict_neural_probabilities", fake_predict
    )
    monkeypatch.setattr(
        "vpncat.cross_session_neural_runner.write_completed_cross_session_run",
        fake_write,
    )
    output = run_cross_session_neural(
        config,
        neural,
        run,
        device_name="cpu",
        selected=selected,
    )
    assert output == tmp_path / "completed"
    assert calls["prediction_models"] == [calls["trained_model"]] * 2
    write_kwargs = calls["write_kwargs"]
    assert write_kwargs["additional_input_hashes"] == {
        "neural_config": sha256_file(neural.config_path),
        "neural_tuning_selection": "b" * 64,
        "neural_tuning_manifest": "c" * 64,
    }
    assert write_kwargs["training_history"].iloc[0]["epoch"] == 1


def test_cross_session_neural_runner_refuses_existing_output_first(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = _run()
    target = tmp_path / run.relative_output_dir
    target.mkdir(parents=True)
    config = SimpleNamespace(cross_session=SimpleNamespace(output_root=tmp_path))
    selection_loaded = False

    def should_not_load(*args, **kwargs):
        nonlocal selection_loaded
        selection_loaded = True
        raise AssertionError("selection must not load")

    monkeypatch.setattr(
        "vpncat.cross_session_neural_runner.load_selected_neural_configuration",
        should_not_load,
    )
    with pytest.raises(FileExistsError, match="overwrite"):
        run_cross_session_neural(
            config,
            SimpleNamespace(topologies={"cnn1d": {}}),
            run,
        )
    assert selection_loaded is False


def test_cross_session_neural_publication_requires_history_and_selection_hashes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _prepared()
    run = _run()
    predictions = build_cross_session_neural_prediction_frame(
        run,
        prepared,
        {
            "inner": np.asarray([[0.9, 0.1], [0.2, 0.8]]),
            "outer": np.asarray([[0.7, 0.3], [0.4, 0.6]]),
        },
    )
    split = tmp_path / "split.csv"
    split.write_text("pair_id,role\npair:6,test\n", encoding="utf-8")
    cross = SimpleNamespace(output_root=tmp_path / "outputs", split_path=split)
    config = SimpleNamespace(cross_session=cross)
    base_hashes = {"cross_session_split": sha256_file(split)}
    selection_hashes = {
        "neural_config": "a" * 64,
        "neural_tuning_selection": "b" * 64,
        "neural_tuning_manifest": "c" * 64,
    }
    all_hashes = {**base_hashes, **selection_hashes}
    manifest = {
        "schema_version": 1,
        "status": "staging",
        "package_version": "0.1.0",
        "git": {"revision": "test", "dirty": False},
        "run": run.to_dict(),
        "preprocessing": {"train_domain": "inner", "train_session": 1},
        "input_hashes": all_hashes,
        "split_manifest_sha256": base_hashes["cross_session_split"],
        "class_order": list(prepared.state.classes),
    }
    monkeypatch.setattr(
        "vpncat.cross_session_artifacts._build_manifest",
        lambda *args, **kwargs: manifest.copy(),
    )
    monkeypatch.setattr(
        "vpncat.cross_session_artifacts.verify_cross_session_input_chain",
        lambda *args, **kwargs: base_hashes,
    )
    monkeypatch.setattr(
        "vpncat.cross_session_artifacts.bind_cross_session_state",
        lambda *args, **kwargs: {
            "train_domain": "inner",
            "train_session": 1,
        },
    )
    history = pd.DataFrame(
        [[1, 1.0, 0.9, 0.5, 0.001]],
        columns=(
            "epoch",
            "train_loss",
            "validation_loss",
            "validation_macro_f1",
            "learning_rate",
        ),
    )
    target = write_completed_cross_session_run(
        config,
        run,
        prepared.index,
        prepared.state,
        predictions,
        model_hyperparameters={"parameter_count": 10},
        training_history=history,
        additional_input_hashes=selection_hashes,
    )
    assert (target / "training_history.csv").is_file()
    validation = validate_completed_cross_session_run(
        target,
        config=config,
        run=run,
        index=prepared.index,
        state=prepared.state,
        classes=prepared.state.classes,
        expected_input_hashes=all_hashes,
    )
    assert validation["prediction_rows"] == 4
    with pytest.raises(PipelineInvariantError, match="requires selection input hashes"):
        validate_completed_cross_session_run(
            target,
            config=config,
            run=run,
            index=prepared.index,
            state=prepared.state,
            classes=prepared.state.classes,
        )
    with pytest.raises(PipelineInvariantError, match="lacks training history"):
        write_completed_cross_session_run(
            SimpleNamespace(
                cross_session=SimpleNamespace(
                    output_root=tmp_path / "other",
                    split_path=split,
                )
            ),
            run,
            prepared.index,
            prepared.state,
            predictions,
            model_hyperparameters={"parameter_count": 10},
            additional_input_hashes=selection_hashes,
        )
