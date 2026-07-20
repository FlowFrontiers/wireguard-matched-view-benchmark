from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vpncat.artifacts import (
    HISTORY_COLUMNS,
    bind_preprocessing_state,
    validate_completed_run,
    write_completed_run,
)
from vpncat.errors import PipelineInvariantError
from vpncat.experiment import enumerate_primary_runs, load_primary_experiment_config
from vpncat.folds import materialize_fold_index
from vpncat.hashing import sha256_file
from vpncat.preprocessing import FoldPreprocessingState, pair_id_digest


def _fold() -> tuple[pd.DataFrame, pd.DataFrame]:
    metadata = pd.DataFrame(
        {
            "pair_id": [f"s1:{index}" for index in range(8)],
            "session": [1] * 8,
            "application_category": ["A", "A", "B", "B", "A", "B", "A", "B"],
        }
    )
    split = metadata.copy()
    split["role_fold_1"] = [
        "train",
        "train",
        "train",
        "train",
        "validation",
        "validation",
        "test",
        "test",
    ]
    return metadata, split


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _contract(tmp_path: Path):
    source_config = Path(__file__).parents[1] / "configs" / "primary.yaml"
    base = load_primary_experiment_config(source_config)
    metadata, split = _fold()
    split_path = tmp_path / "split_manifest.csv"
    split.to_csv(split_path, index=False)
    canonical_path = tmp_path / "canonical.parquet"
    canonical_path.write_bytes(b"synthetic canonical identity")
    canonical_hash = sha256_file(canonical_path)
    split_hash = sha256_file(split_path)

    dataset_manifest_path = tmp_path / "dataset_manifest.json"
    _write_json(
        dataset_manifest_path,
        {
            "artifacts": {
                "canonical_pairs": {"sha256": canonical_hash},
                "split_manifest": {"sha256": split_hash},
            }
        },
    )
    feature_audit_path = tmp_path / "feature_audit.json"
    _write_json(
        feature_audit_path,
        {"status": "valid", "canonical": {"sha256": canonical_hash}},
    )

    fold = materialize_fold_index(metadata, split, fold=1)
    fit_hash = pair_id_digest(fold.pair_ids_for("train"))
    state = FoldPreprocessingState(
        fold=1,
        train_domain="inner",
        representation="matched_flow_stats",
        feature_names=("f1", "f2"),
        medians=np.asarray([1.5, 2.5]),
        classes=("A", "B"),
        class_weights=np.asarray([1.0, 1.0]),
        fit_pair_count=4,
        fit_pair_ids_sha256=fit_hash,
    )
    target_state = state.target_state()
    preprocessing_audit_path = tmp_path / "preprocessing_audit.json"
    _write_json(
        preprocessing_audit_path,
        {
            "status": "valid",
            "inputs": {
                "canonical_sha256": canonical_hash,
                "split_sha256": split_hash,
            },
            "fitted_states": {
                "matched_flow_stats": {"1": {"inner": state.to_dict()}}
            },
            "folds": {"1": {"targets": target_state.to_dict()}},
        },
    )
    config = replace(
        base,
        project_root=tmp_path,
        canonical_path=canonical_path,
        split_path=split_path,
        dataset_manifest_path=dataset_manifest_path,
        feature_audit_path=feature_audit_path,
        preprocessing_audit_path=preprocessing_audit_path,
        contract_audit_path=tmp_path / "contract_audit.json",
        output_root=tmp_path / "outputs",
    )
    run = next(
        item
        for item in enumerate_primary_runs(config)
        if item.experiment_id == "matched_flow_stats__random_forest"
        and item.fold == 1
        and item.train_domain == "inner"
    )
    return config, fold, run, state


def _predictions(run, fold) -> pd.DataFrame:
    rows = []
    for domain in run.test_domains:
        for position in fold.test_positions:
            true_label = fold.labels[position]
            probabilities = [0.9, 0.1] if true_label == "A" else [0.1, 0.9]
            rows.append(
                {
                    "run_id": run.run_id,
                    "protocol": run.protocol,
                    "representation": run.representation,
                    "model": run.model,
                    "pair_id": fold.pair_ids[position],
                    "session": int(fold.sessions[position]),
                    "train_domain": run.train_domain,
                    "test_domain": domain,
                    "fold": run.fold,
                    "seed": run.seed,
                    "true_label": true_label,
                    "prediction": true_label,
                    "class_probabilities": probabilities,
                }
            )
    return pd.DataFrame(rows)


def test_primary_matrix_contains_150_unique_training_runs() -> None:
    config = load_primary_experiment_config(
        Path(__file__).parents[1] / "configs" / "primary.yaml"
    )
    runs = enumerate_primary_runs(config)
    assert len(runs) == 150
    assert len({run.run_id for run in runs}) == 150
    assert sum(run.family == "classical" for run in runs) == 60
    assert sum(run.family == "neural" for run in runs) == 90
    assert sum(len(run.test_domains) for run in runs) == 300


def test_artifact_writer_is_atomic_complete_and_non_overwriting(tmp_path: Path) -> None:
    config, fold, run, state = _contract(tmp_path)
    run_dir = write_completed_run(
        config,
        run,
        fold,
        state,
        _predictions(run, fold),
        model_hyperparameters={"n_estimators": 500},
        additional_input_hashes={"model_selection": "a" * 64},
    )
    report = validate_completed_run(run_dir, run=run, fold=fold, classes=state.classes)

    assert report["status"] == "valid"
    assert report["prediction_rows"] == 4
    manifest = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert manifest["run"]["train_domain"] == "inner"
    assert manifest["preprocessing"]["train_domain"] == "inner"
    assert manifest["input_hashes"]["model_selection"] == "a" * 64
    assert set(manifest["artifacts"]) == {
        "metrics.json",
        "metrics_long.csv",
        "predictions.parquet",
        "split_manifest.csv",
    }
    with pytest.raises(FileExistsError, match="overwrite"):
        write_completed_run(
            config,
            run,
            fold,
            state,
            _predictions(run, fold),
            model_hyperparameters={"n_estimators": 500},
            additional_input_hashes={"model_selection": "a" * 64},
        )


def test_artifact_writer_rejects_invalid_additional_input_hash(tmp_path: Path) -> None:
    config, fold, run, state = _contract(tmp_path)
    with pytest.raises(PipelineInvariantError, match="SHA-256"):
        write_completed_run(
            config,
            run,
            fold,
            state,
            _predictions(run, fold),
            model_hyperparameters={"n_estimators": 500},
            additional_input_hashes={"model_selection": "not-a-sha256"},
        )


def test_contract_rejects_preprocessing_domain_mixup(tmp_path: Path) -> None:
    config, fold, run, state = _contract(tmp_path)
    wrong_domain = replace(state, train_domain="outer")
    audit = json.loads(config.preprocessing_audit_path.read_text(encoding="utf-8"))
    with pytest.raises(PipelineInvariantError, match="train_domain"):
        bind_preprocessing_state(run, fold, wrong_domain, preprocessing_audit=audit)


def test_neural_run_requires_history_and_records_fit_free_domain(tmp_path: Path) -> None:
    config, fold, _, statistical_state = _contract(tmp_path)
    run = next(
        item
        for item in enumerate_primary_runs(config)
        if item.experiment_id == "sequential_splt__cnn1d"
        and item.fold == 1
        and item.train_domain == "inner"
        and item.seed == 42
    )
    target_state = statistical_state.target_state()
    with pytest.raises(PipelineInvariantError, match="training_history"):
        write_completed_run(
            config,
            run,
            fold,
            target_state,
            _predictions(run, fold),
            model_hyperparameters={"width": 64},
        )

    history = pd.DataFrame(
        [[1, 1.0, 1.1, 0.5, 0.001], [2, 0.8, 0.9, 0.6, 0.001]],
        columns=HISTORY_COLUMNS,
    )
    run_dir = write_completed_run(
        config,
        run,
        fold,
        target_state,
        _predictions(run, fold),
        model_hyperparameters={"width": 64},
        training_history=history,
    )
    manifest = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert manifest["preprocessing"]["feature_transform"] == "fit-free"
    assert manifest["preprocessing"]["train_domain"] == run.train_domain
    assert "training_history.csv" in manifest["artifacts"]

    history_path = run_dir / "training_history.csv"
    tampered = pd.read_csv(history_path, float_precision="round_trip")
    tampered.loc[0, "validation_loss"] = np.inf
    tampered.to_csv(history_path, index=False)
    manifest["artifacts"]["training_history.csv"]["sha256"] = sha256_file(history_path)
    (run_dir / "run.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(PipelineInvariantError, match="non-finite"):
        validate_completed_run(
            run_dir,
            run=run,
            fold=fold,
            classes=target_state.classes,
        )


def test_prediction_contract_rejects_missing_test_domain_rows(tmp_path: Path) -> None:
    config, fold, run, state = _contract(tmp_path)
    incomplete = _predictions(run, fold).iloc[:-1].copy()
    with pytest.raises(PipelineInvariantError, match="exactly cover"):
        write_completed_run(
            config,
            run,
            fold,
            state,
            incomplete,
            model_hyperparameters={"n_estimators": 500},
        )
    assert not config.output_root.exists() or not any(config.output_root.rglob("run.json"))


def test_completed_run_rejects_stale_inputs_revision_and_extra_files(tmp_path: Path) -> None:
    config, fold, run, state = _contract(tmp_path)
    run_dir = write_completed_run(
        config,
        run,
        fold,
        state,
        _predictions(run, fold),
        model_hyperparameters={"n_estimators": 500},
    )
    manifest = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    with pytest.raises(PipelineInvariantError, match="input hashes"):
        validate_completed_run(
            run_dir,
            run=run,
            fold=fold,
            classes=state.classes,
            expected_input_hashes={"canonical": "stale"},
        )
    with pytest.raises(PipelineInvariantError, match="Git revision"):
        validate_completed_run(
            run_dir,
            run=run,
            fold=fold,
            classes=state.classes,
            expected_git_revision="another-revision",
        )
    (run_dir / "untracked.txt").write_text("unexpected", encoding="utf-8")
    with pytest.raises(PipelineInvariantError, match="unexpected files"):
        validate_completed_run(
            run_dir,
            run=run,
            fold=fold,
            classes=state.classes,
            expected_input_hashes=manifest["input_hashes"],
        )
