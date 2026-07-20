from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vpncat.dann import enumerate_dann_runs, load_dann_config
from vpncat.dann_artifacts import (
    validate_completed_dann_run,
    write_completed_dann_run,
)
from vpncat.dann_data import PreparedDANNRun, UnlabeledNeuralSubset
from vpncat.dann_runner import build_dann_prediction_frame, run_dann
from vpncat.dann_training import DANNTrainingResult
from vpncat.errors import PipelineInvariantError
from vpncat.folds import FoldIndex
from vpncat.hashing import sha256_file
from vpncat.models.neural import build_neural_model, trainable_parameter_count
from vpncat.neural_data import NeuralSubset
from vpncat.neural_tuning import SelectedNeuralConfiguration
from vpncat.preprocessing import fit_fold_targets


def _config(tmp_path: Path):
    base = load_dann_config(Path(__file__).parents[1] / "configs" / "dann.yaml")
    split_path = tmp_path / "split.csv"
    split_path.write_text("pair_id,role_fold_1\n", encoding="utf-8")
    primary = replace(base.primary, split_path=split_path)
    neural = replace(base.neural, tuning_output_root=tmp_path / "tuning")
    return replace(
        base,
        primary=primary,
        neural=neural,
        output_root=tmp_path / "outputs",
    )


def _fold(fold: int = 1) -> FoldIndex:
    pair_ids = tuple(f"pair:{index}" for index in range(12))
    roles = ("train",) * 6 + ("validation",) * 2 + ("test",) * 4
    return FoldIndex(
        fold=fold,
        pair_ids=pair_ids,
        sessions=np.asarray([1, 2] * 6, dtype=np.int16),
        labels=tuple("A" if index % 2 == 0 else "B" for index in range(12)),
        roles=roles,
        train_positions=np.arange(6, dtype=np.int64),
        validation_positions=np.arange(6, 8, dtype=np.int64),
        test_positions=np.arange(8, 12, dtype=np.int64),
    )


def _subset(fold: FoldIndex, positions: np.ndarray, *, offset: float) -> NeuralSubset:
    rng = np.random.default_rng(42 + int(offset))
    values = rng.normal(size=(len(positions), 5, 3)).astype(np.float32)
    mask = np.ones((len(positions), 5), dtype=bool)
    targets = np.asarray(
        [0 if fold.labels[position] == "A" else 1 for position in positions],
        dtype=np.int64,
    )
    return NeuralSubset(
        pair_ids=tuple(fold.pair_ids[position] for position in positions),
        positions=positions.copy(),
        values=values,
        mask=mask,
        targets=targets,
    )


def _prepared() -> PreparedDANNRun:
    fold = _fold()
    source = _subset(fold, fold.train_positions, offset=0.0)
    target_values = source.values.copy() + 0.25
    adaptation = UnlabeledNeuralSubset(
        pair_ids=source.pair_ids,
        positions=source.positions.copy(),
        values=target_values,
        mask=source.mask.copy(),
    )
    return PreparedDANNRun(
        fold=fold,
        state=fit_fold_targets(fold),
        channels=("direction", "size", "iat_ms"),
        source_training=source,
        adaptation_training=adaptation,
        source_validation=_subset(fold, fold.validation_positions, offset=1.0),
        tests={
            "inner": _subset(fold, fold.test_positions, offset=2.0),
            "outer": _subset(fold, fold.test_positions, offset=3.0),
        },
    )


def _history() -> pd.DataFrame:
    round_trip_sensitive = 0.09417734788764953
    return pd.DataFrame(
        [
            [1, 1.0, 0.7, 1.7, 0.9, 0.05, 0.001, 0.0, 0.4],
            [
                2,
                0.8,
                0.6,
                1.4,
                round_trip_sensitive,
                round_trip_sensitive,
                0.001,
                0.5,
                0.8,
            ],
        ],
        columns=(
            "epoch",
            "train_classification_loss",
            "train_domain_loss",
            "train_total_loss",
            "validation_loss",
            "validation_macro_f1",
            "learning_rate",
            "grl_coefficient_start",
            "grl_coefficient_end",
        ),
    )


def _selection_artifacts(config) -> tuple[dict[str, object], dict[str, str]]:
    root = config.neural.tuning_output_root / "cnn1d"
    root.mkdir(parents=True)
    payload = {
        "selected_trial": config.neural.trials[0].to_dict(),
        "selected_result": {"parameter_count": 8},
    }
    selected_path = root / "selected.json"
    manifest_path = root / "tuning_manifest.json"
    selected_path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    manifest_path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return payload, {
        "neural_tuning_selection": sha256_file(selected_path),
        "neural_tuning_manifest": sha256_file(manifest_path),
    }


def _hyperparameters(config, selection: dict[str, object]) -> dict[str, object]:
    return {
        "selected_trial": selection["selected_trial"],
        "backbone_topology": config.neural.topologies["cnn1d"],
        "domain_head": config.domain_head,
        "optimizer": config.neural.optimizer,
        "training_policy": config.neural.training,
        "domain_loss_weight": config.domain_loss_weight,
        "gradient_reversal": config.gradient_reversal,
        "parameter_count": 10,
        "backbone_parameter_count": 8,
        "training_outcome": {
            "best_epoch": 2,
            "best_validation_macro_f1": 0.09417734788764953,
            "validation_loss_at_best_epoch": 0.09417734788764953,
            "epochs_completed": 2,
            "device": "cpu",
        },
        "selection": {"selected_result": selection["selected_result"]},
    }


def _probabilities(prepared: PreparedDANNRun) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for domain, subset in prepared.tests.items():
        probabilities = np.full((len(subset.pair_ids), 2), 0.1, dtype=np.float64)
        probabilities[np.arange(len(subset.pair_ids)), subset.targets] = 0.9
        result[domain] = probabilities
    return result


def test_dann_artifacts_publish_atomically_and_recompute_metrics(tmp_path: Path) -> None:
    config = _config(tmp_path)
    run = next(run for run in enumerate_dann_runs(config) if run.fold == 1 and run.seed == 42)
    prepared = _prepared()
    predictions = build_dann_prediction_frame(run, prepared, _probabilities(prepared))
    selection, tuning_hashes = _selection_artifacts(config)
    input_hashes = {
        "split_manifest": sha256_file(config.primary.split_path),
        **tuning_hashes,
    }
    output = write_completed_dann_run(
        config,
        run,
        prepared.fold,
        prepared.state,
        predictions,
        model_hyperparameters=_hyperparameters(config, selection),
        training_history=_history(),
        input_hashes=input_hashes,
    )
    report = validate_completed_dann_run(
        output,
        config=config,
        run=run,
        fold=prepared.fold,
        state=prepared.state,
        expected_input_hashes=input_hashes,
    )
    assert report["prediction_rows"] == 8
    manifest_path = output / "run.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["data"]["adaptation_labels_exposed"] is False
    assert manifest["data"]["source_training_pair_ids_sha256"] == manifest["data"][
        "adaptation_pair_ids_sha256"
    ]
    with pytest.raises(FileExistsError, match="overwrite"):
        write_completed_dann_run(
            config,
            run,
            prepared.fold,
            prepared.state,
            predictions,
            model_hyperparameters=_hyperparameters(config, selection),
            training_history=_history(),
            input_hashes=input_hashes,
        )

    history_path = output / "training_history.csv"
    original_history = history_path.read_bytes()
    tampered_history = pd.read_csv(history_path, float_precision="round_trip")
    tampered_history.loc[1, "validation_macro_f1"] += 1e-4
    tampered_history.to_csv(history_path, index=False)
    manifest["artifacts"]["training_history.csv"]["sha256"] = sha256_file(history_path)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(PipelineInvariantError, match="outcome disagrees"):
        validate_completed_dann_run(
            output,
            config=config,
            run=run,
            fold=prepared.fold,
            state=prepared.state,
            expected_input_hashes=input_hashes,
        )
    history_path.write_bytes(original_history)
    manifest["artifacts"]["training_history.csv"]["sha256"] = sha256_file(history_path)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    metrics_path = output / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["metrics"]["outer"]["macro_f1"] = 0.0
    metrics_path.write_text(json.dumps(metrics, sort_keys=True) + "\n", encoding="utf-8")
    manifest["artifacts"]["metrics.json"]["sha256"] = sha256_file(metrics_path)
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(PipelineInvariantError, match="metrics disagree"):
        validate_completed_dann_run(
            output,
            config=config,
            run=run,
            fold=prepared.fold,
            state=prepared.state,
            expected_input_hashes=input_hashes,
        )


def test_dann_runner_binds_tuned_backbone_and_publishes_both_views(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    run = next(run for run in enumerate_dann_runs(config) if run.fold == 1 and run.seed == 42)
    prepared = _prepared()
    trial = config.neural.trials[0]
    backbone = build_neural_model(
        "cnn1d",
        feature_count=3,
        class_count=2,
        width=trial.width,
        dropout=trial.dropout,
        maximum_length=config.neural.maximum_prefix_length,
        topology=config.neural.topologies["cnn1d"],
    )
    selected = SelectedNeuralConfiguration(
        model="cnn1d",
        trial=trial,
        result={"parameter_count": trainable_parameter_count(backbone)},
        selected_path=tmp_path / "selected.json",
        selected_sha256="b" * 64,
        tuning_manifest_sha256="c" * 64,
        tuning_revision="revision",
        tuning_environment={"torch": "test"},
        tuning_device="cpu",
    )
    calls: dict[str, object] = {"prediction_models": []}
    monkeypatch.setattr(
        "vpncat.dann_runner.git_provenance",
        lambda *_: {"status_available": True, "dirty": False, "revision": "revision"},
    )
    monkeypatch.setattr(
        "vpncat.dann_runner.validate_dann_run_contract",
        lambda *_: {"split_manifest": "a" * 64},
    )
    monkeypatch.setattr("vpncat.dann_runner.prepare_dann_run", lambda *_: prepared)

    def fake_train(model, training, adaptation, validation, *_args, **_kwargs):
        assert training.pair_ids == adaptation.pair_ids
        assert not hasattr(adaptation, "targets")
        assert set(training.pair_ids).isdisjoint(validation.pair_ids)
        calls["trained_model"] = model
        return DANNTrainingResult(
            model=model,
            history=_history(),
            best_epoch=2,
            best_validation_macro_f1=0.09417734788764953,
            validation_loss_at_best_epoch=0.09417734788764953,
            epochs_completed=2,
            parameter_count=trainable_parameter_count(model),
            backbone_parameter_count=trainable_parameter_count(model.backbone),
            device="cpu",
        )

    def fake_predict(model, subset, **_kwargs):
        calls["prediction_models"].append(model)
        probabilities = np.full((len(subset.pair_ids), 2), 0.1)
        probabilities[np.arange(len(subset.pair_ids)), subset.targets] = 0.9
        return probabilities

    def fake_write(*args, **kwargs):
        calls["write_args"] = args
        calls["write_kwargs"] = kwargs
        return tmp_path / "completed"

    monkeypatch.setattr("vpncat.dann_runner.train_dann_model", fake_train)
    monkeypatch.setattr("vpncat.dann_runner.predict_dann_probabilities", fake_predict)
    monkeypatch.setattr("vpncat.dann_runner.write_completed_dann_run", fake_write)
    output = run_dann(config, run, device_name="cpu", selected=selected)
    assert output == tmp_path / "completed"
    assert calls["prediction_models"] == [calls["trained_model"]] * 2
    write_kwargs = calls["write_kwargs"]
    assert write_kwargs["input_hashes"]["neural_tuning_selection"] == "b" * 64
    assert write_kwargs["input_hashes"]["neural_tuning_manifest"] == "c" * 64
    assert write_kwargs["model_hyperparameters"]["backbone_parameter_count"] == int(
        selected.result["parameter_count"]
    )
    predictions = calls["write_args"][4]
    assert set(predictions["test_domain"]) == {"inner", "outer"}
    assert set(predictions["train_domain"]) == {"inner"}
