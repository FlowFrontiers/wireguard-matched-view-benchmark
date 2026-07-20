from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import vpncat.ablation_artifacts as artifacts
from vpncat.ablation_artifacts import validate_completed_ablation_run
from vpncat.ablation_runner import build_ablation_prediction_frame, run_ablation
from vpncat.ablations import enumerate_ablation_runs, load_ablation_config
from vpncat.errors import PipelineInvariantError
from vpncat.folds import FoldIndex
from vpncat.hashing import sha256_file
from vpncat.models.neural import build_neural_model, trainable_parameter_count
from vpncat.neural_config import load_neural_config
from vpncat.neural_data import NeuralSubset, PreparedNeuralRun
from vpncat.neural_training import TrainingResult
from vpncat.neural_tuning import SelectedNeuralConfiguration
from vpncat.preprocessing import FoldTargetState, pair_id_digest


def _project_root() -> Path:
    return Path(__file__).parents[1]


def _fold() -> FoldIndex:
    pair_ids = ("train:a", "train:b", "validation:a", "validation:b", "test:a", "test:b")
    return FoldIndex(
        fold=1,
        pair_ids=pair_ids,
        sessions=np.asarray([1, 2, 1, 2, 1, 2], dtype=np.int16),
        labels=("A", "B", "A", "B", "A", "B"),
        roles=("train", "train", "validation", "validation", "test", "test"),
        train_positions=np.asarray([0, 1], dtype=np.int64),
        validation_positions=np.asarray([2, 3], dtype=np.int64),
        test_positions=np.asarray([4, 5], dtype=np.int64),
    )


def _subset(
    fold: FoldIndex,
    positions: np.ndarray,
    *,
    length: int,
    feature_count: int,
) -> NeuralSubset:
    positions = np.asarray(positions, dtype=np.int64)
    labels = np.asarray(fold.labels, dtype=object)[positions]
    return NeuralSubset(
        pair_ids=tuple(fold.pair_ids[position] for position in positions),
        positions=positions,
        values=np.ones((len(positions), length, feature_count), dtype=np.float32),
        mask=np.ones((len(positions), length), dtype=bool),
        targets=np.asarray([0 if label == "A" else 1 for label in labels], dtype=np.int64),
    )


def _prepared(run) -> PreparedNeuralRun:
    fold = _fold()
    state = FoldTargetState(
        fold=1,
        classes=("A", "B"),
        class_weights=np.asarray([1.0, 1.0]),
        fit_pair_count=2,
        fit_pair_ids_sha256=pair_id_digest(fold.pair_ids_for("train")),
    )
    feature_count = len(run.channels)
    return PreparedNeuralRun(
        fold=fold,
        state=state,
        channels=run.channels,
        training=_subset(
            fold, fold.train_positions, length=run.prefix_length, feature_count=feature_count
        ),
        validation=_subset(
            fold,
            fold.validation_positions,
            length=run.prefix_length,
            feature_count=feature_count,
        ),
        tests={
            domain: _subset(
                fold,
                fold.test_positions,
                length=run.prefix_length,
                feature_count=feature_count,
            )
            for domain in run.test_domains
        },
    )


def _selected(model: str, class_count: int, feature_count: int) -> SelectedNeuralConfiguration:
    neural = load_neural_config(_project_root() / "configs" / "neural.yaml")
    trial = neural.trials[0]
    network = build_neural_model(
        model,
        feature_count=feature_count,
        class_count=class_count,
        width=trial.width,
        dropout=trial.dropout,
        maximum_length=neural.maximum_prefix_length,
        topology=neural.topologies[model],
    )
    return SelectedNeuralConfiguration(
        model=model,
        trial=trial,
        result={"parameter_count": trainable_parameter_count(network)},
        selected_path=Path("selected.json"),
        selected_sha256="b" * 64,
        tuning_manifest_sha256="c" * 64,
        tuning_revision="revision",
        tuning_environment={"torch": "test"},
        tuning_device="cpu",
    )


def test_prediction_frame_preserves_paired_views() -> None:
    config = load_ablation_config(_project_root() / "configs" / "ablation_prefix.yaml")
    run = next(
        run
        for run in enumerate_ablation_runs(config)
        if run.model == "cnn1d" and run.fold == 1 and run.prefix_length == 10
    )
    prepared = _prepared(run)
    probabilities = {
        "inner": np.asarray([[0.8, 0.2], [0.1, 0.9]]),
        "outer": np.asarray([[0.4, 0.6], [0.7, 0.3]]),
    }
    frame = build_ablation_prediction_frame(run, prepared, probabilities)
    assert frame.groupby("test_domain")["pair_id"].apply(tuple).to_dict() == {
        "inner": ("test:a", "test:b"),
        "outer": ("test:a", "test:b"),
    }
    assert frame["prediction"].tolist() == ["A", "B", "B", "A"]


def test_runner_uses_selected_policy_and_one_model_for_both_views(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = load_ablation_config(_project_root() / "configs" / "ablation_channels.yaml")
    config = replace(base, output_root=tmp_path / "outputs")
    run = next(
        run
        for run in enumerate_ablation_runs(config)
        if run.model == "cnn1d" and run.fold == 1 and run.observation_id == "direction"
    )
    prepared = _prepared(run)
    # Tuning used all three channels; reduced-channel parameter counts differ by design.
    selected = _selected("cnn1d", class_count=2, feature_count=3)
    calls: dict[str, object] = {"prediction_models": []}
    monkeypatch.setattr(
        "vpncat.ablation_runner.git_provenance",
        lambda *_: {"status_available": True, "dirty": False, "revision": "clean"},
    )
    monkeypatch.setattr(
        "vpncat.ablation_runner.validate_ablation_run_contract", lambda *_: {"x": "a" * 64}
    )
    monkeypatch.setattr("vpncat.ablation_runner.prepare_ablation_run", lambda *_: prepared)

    def fake_train(model, *_args, **_kwargs):
        calls["trained_model"] = model
        history = pd.DataFrame(
            [[1, 1.0, 0.9, 0.5, selected.trial.learning_rate]],
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
            parameter_count=trainable_parameter_count(model),
            device="cpu",
        )

    def fake_predict(model, *_args, **_kwargs):
        calls["prediction_models"].append(model)
        return np.asarray([[0.8, 0.2], [0.1, 0.9]])

    def fake_write(*args, **kwargs):
        calls["write_kwargs"] = kwargs
        return tmp_path / "completed"

    monkeypatch.setattr("vpncat.ablation_runner.train_neural_model", fake_train)
    monkeypatch.setattr("vpncat.ablation_runner.predict_neural_probabilities", fake_predict)
    monkeypatch.setattr("vpncat.ablation_runner.write_completed_ablation_run", fake_write)
    output = run_ablation(config, run, device_name="cpu", selected=selected)
    assert output == tmp_path / "completed"
    assert calls["prediction_models"] == [calls["trained_model"]] * 2
    parameters = calls["write_kwargs"]["model_hyperparameters"]
    assert parameters["observation"] == {"prefix_length": 50, "channels": ["direction"]}
    assert parameters["parameter_count"] != selected.result["parameter_count"]


def test_runner_rejects_primary_reference_before_training() -> None:
    config = load_ablation_config(_project_root() / "configs" / "ablation_prefix.yaml")
    reference = next(run for run in enumerate_ablation_runs(config) if run.is_primary_reference)
    with pytest.raises(PipelineInvariantError, match="incompatible"):
        run_ablation(config, reference)


def test_artifact_validator_recomputes_metrics_after_hash_repair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = load_ablation_config(_project_root() / "configs" / "ablation_channels.yaml")
    split_path = tmp_path / "split.csv"
    split_path.write_text("pair_id\n", encoding="utf-8")
    primary = replace(base.primary, split_path=split_path)
    config = replace(base, primary=primary, output_root=tmp_path / "outputs")
    run = next(
        run
        for run in enumerate_ablation_runs(config)
        if run.model == "cnn1d" and run.fold == 1 and run.observation_id == "direction"
    )
    prepared = _prepared(run)
    probabilities = {
        domain: np.asarray([[0.8, 0.2], [0.1, 0.9]]) for domain in run.test_domains
    }
    predictions = build_ablation_prediction_frame(run, prepared, probabilities)
    round_trip_sensitive = 0.09417734788764953
    history = pd.DataFrame(
        [[1, 1.0, round_trip_sensitive, round_trip_sensitive, 0.001]],
        columns=("epoch", "train_loss", "validation_loss", "validation_macro_f1", "learning_rate"),
    )
    model_parameters = {
        "training_outcome": {
            "best_epoch": 1,
            "best_validation_macro_f1": round_trip_sensitive,
            "validation_loss_at_best_epoch": round_trip_sensitive,
            "epochs_completed": 1,
        }
    }
    input_hashes = {"split_manifest": sha256_file(split_path), "contract": "a" * 64}
    monkeypatch.setattr(
        artifacts,
        "_validate_model_hyperparameters",
        lambda *_args, **_kwargs: None,
    )
    output = artifacts.write_completed_ablation_run(
        config,
        run,
        prepared.fold,
        prepared.state,
        predictions,
        model_hyperparameters=model_parameters,
        training_history=history,
        input_hashes=input_hashes,
    )
    history_path = output / "training_history.csv"
    tampered_history = pd.read_csv(history_path, float_precision="round_trip")
    tampered_history.loc[0, "validation_loss"] += 1e-4
    tampered_history.to_csv(history_path, index=False)
    manifest_path = output / "run.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["training_history.csv"]["sha256"] = sha256_file(history_path)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(PipelineInvariantError, match="outcome disagrees"):
        validate_completed_ablation_run(
            output,
            config=config,
            run=run,
            fold=prepared.fold,
            state=prepared.state,
            expected_input_hashes=input_hashes,
        )

    history.to_csv(history_path, index=False)
    manifest["artifacts"]["training_history.csv"]["sha256"] = sha256_file(history_path)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    tampered = pd.read_parquet(output / "predictions.parquet")
    tampered.at[0, "class_probabilities"] = np.asarray([0.4, 0.6])
    tampered.at[0, "prediction"] = "B"
    tampered.to_parquet(output / "predictions.parquet", index=False)
    manifest["artifacts"]["predictions.parquet"]["sha256"] = sha256_file(
        output / "predictions.parquet"
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(PipelineInvariantError, match="metrics disagree"):
        validate_completed_ablation_run(
            output,
            config=config,
            run=run,
            fold=prepared.fold,
            state=prepared.state,
            expected_input_hashes=input_hashes,
        )
