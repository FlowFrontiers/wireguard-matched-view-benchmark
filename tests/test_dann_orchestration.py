from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from vpncat.dann import load_dann_config
from vpncat.dann_orchestration import DANNRunFilters, run_dann_matrix
from vpncat.folds import FoldIndex


def _config(tmp_path: Path):
    base = load_dann_config(Path(__file__).parents[1] / "configs" / "dann.yaml")
    return replace(base, output_root=tmp_path / "outputs")


def _fold(number: int) -> FoldIndex:
    return FoldIndex(
        fold=number,
        pair_ids=("a", "b", "c", "d", "e", "f"),
        sessions=np.ones(6, dtype=np.int16),
        labels=("A", "B", "A", "B", "A", "B"),
        roles=("train", "train", "validation", "validation", "test", "test"),
        train_positions=np.asarray([0, 1]),
        validation_positions=np.asarray([2, 3]),
        test_positions=np.asarray([4, 5]),
    )


def _common_patches(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "vpncat.dann_orchestration.git_provenance",
        lambda *_: {"status_available": True, "dirty": False, "revision": "revision"},
    )
    monkeypatch.setattr(
        "vpncat.dann_orchestration.validate_dann_run_contracts",
        lambda *_: {"base": "a" * 64},
    )
    monkeypatch.setattr(
        "vpncat.dann_orchestration._folds",
        lambda config: {fold: _fold(fold) for fold in config.folds},
    )


def test_dann_plan_enumerates_fifteen_without_loading_tuning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    _common_patches(monkeypatch)

    def forbidden_selection(_config):
        raise AssertionError("plan-only pending matrix must not load tuning")

    monkeypatch.setattr("vpncat.dann_orchestration._load_selection", forbidden_selection)
    report = run_dann_matrix(config)
    assert report["selected_run_count"] == 15
    assert report["counts"] == {"complete": 0, "executed": 0, "pending": 15}
    assert report["status"] == "partial"


def test_dann_controller_executes_bounded_run_and_revalidates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    _common_patches(monkeypatch)
    selected = SimpleNamespace(
        selected_sha256="b" * 64,
        tuning_manifest_sha256="c" * 64,
    )
    monkeypatch.setattr("vpncat.dann_orchestration._load_selection", lambda *_: selected)
    executed: list[str] = []

    def fake_execute(config, run, **_kwargs):
        executed.append(run.run_id)
        return config.output_root / run.relative_output_dir

    validations: list[str] = []

    def fake_validate(_path, *, run, **_kwargs):
        validations.append(run.run_id)
        return {"prediction_rows": 4}

    monkeypatch.setattr("vpncat.dann_orchestration._execute", fake_execute)
    monkeypatch.setattr(
        "vpncat.dann_orchestration.validate_completed_dann_run", fake_validate
    )
    report = run_dann_matrix(
        config,
        filters=DANNRunFilters(folds=(1,), seeds=(42,)),
        execute=True,
        maximum_pending_runs=1,
        device_name="cpu",
    )
    assert len(executed) == 1
    assert validations == executed
    assert report["counts"] == {"complete": 0, "executed": 1, "pending": 0}
    assert report["status"] == "complete"


def test_importing_dann_orchestration_does_not_import_torch() -> None:
    root = Path(__file__).parents[1]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(root / "src")
    command = (
        "import sys; import vpncat.dann_orchestration; "
        "raise SystemExit(1 if 'torch' in sys.modules else 0)"
    )
    completed = subprocess.run(
        [sys.executable, "-c", command],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
