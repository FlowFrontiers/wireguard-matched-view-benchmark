from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib.axes
import pandas as pd
import pytest

from vpncat.errors import PipelineInvariantError
from vpncat.hashing import sha256_file
from vpncat.paper_analysis import _paper_figures, load_paper_analysis_config
from vpncat.paper_bundle import render_presentation, validate_bundle_quick


def _bundle_root() -> Path:
    return Path("artifacts/f82a743/paper_analysis")


def test_committed_paper_bundle_quick_validation_is_deterministic() -> None:
    config = load_paper_analysis_config(Path("configs/paper_analysis.yaml"))
    report = validate_bundle_quick(_bundle_root(), config=config, rerender=True)
    assert report["status"] == "valid"
    assert report["mode"] == "quick"
    assert report["presentation_files"] == 19
    macros = (_bundle_root() / "latex" / "results_macros.tex").read_text(encoding="utf-8")
    assert "\\newcommand{\\XGBFlatInnerMacroFOne}{0.962}" in macros
    assert "\\newcommand{\\XGBFlatOuterMacroFOne}{0.651}" in macros
    assert "\\newcommand{\\CNNOuterBalancedAccuracy}{0.958}" in macros
    assert "\\newcommand{\\DANNMacroFOneGain}{0.009}" in macros


def test_quick_validation_rejects_evidence_tampering(tmp_path: Path) -> None:
    copied = tmp_path / "paper_analysis"
    shutil.copytree(_bundle_root(), copied)
    path = copied / "cross_session_by_direction.csv"
    frame = pd.read_csv(path)
    frame.loc[0, "outer_estimate"] += 0.02
    frame.to_csv(path, index=False)
    config = load_paper_analysis_config(Path("configs/paper_analysis.yaml"))
    with pytest.raises(PipelineInvariantError, match="Evidence hash differs"):
        validate_bundle_quick(copied, config=config, rerender=False)


def test_render_captures_git_provenance_before_creating_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    copied = tmp_path / "paper_analysis"
    shutil.copytree(_bundle_root(), copied)
    observed_staging: list[list[Path]] = []

    def capture_provenance(_project_root: Path) -> dict[str, object]:
        observed_staging.append(
            list(copied.parent.glob(f".{copied.name}.presentation-*"))
        )
        return {"revision": "a" * 40, "dirty": False, "status_available": True}

    monkeypatch.setattr("vpncat.paper_bundle.git_provenance", capture_provenance)
    config = load_paper_analysis_config(Path("configs/paper_analysis.yaml"))
    render_presentation(copied, config=config)

    assert observed_staging == [[]]
    manifest = json.loads((copied / "presentation_manifest.json").read_text(encoding="utf-8"))
    assert manifest["generation_git"] == {
        "revision": "a" * 40,
        "dirty": False,
        "status_available": True,
    }


def test_confusion_validation_rejects_hash_repaired_count_tampering(tmp_path: Path) -> None:
    copied = tmp_path / "paper_analysis"
    shutil.copytree(_bundle_root(), copied)
    path = copied / "selected_confusions.csv"
    frame = pd.read_csv(path)
    frame.loc[0, "count"] += 1
    frame.to_csv(path, index=False)

    manifest_path = copied / "evidence_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    evidence_hash = sha256_file(path)
    manifest["evidence_artifacts"]["selected_confusions.csv"] = evidence_hash

    receipt_path = copied / "confusion_extension_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["evidence_artifact"]["selected_confusions.csv"] = evidence_hash
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest["confusion_extension_receipt_sha256"] = sha256_file(receipt_path)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    config = load_paper_analysis_config(Path("configs/paper_analysis.yaml"))
    with pytest.raises(PipelineInvariantError, match="normalization differs"):
        validate_bundle_quick(copied, config=config, rerender=False)


def test_transfer_gap_figure_contains_confidence_interval_whiskers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _bundle_root()
    frames = {
        name: pd.read_csv(root / name)
        for name in (
            "primary_transfer_gaps.csv",
            "cross_session_by_direction.csv",
            "prefix_ablation.csv",
            "channel_ablation.csv",
            "per_class_metrics.csv",
            "seed_dispersion.csv",
            "selected_confusions.csv",
        )
    }
    observed_errorbars = []
    original_barh = matplotlib.axes.Axes.barh

    def capture_barh(self, *args, **kwargs):
        container = original_barh(self, *args, **kwargs)
        if kwargs.get("xerr") is not None:
            observed_errorbars.append(container.errorbar)
        return container

    monkeypatch.setattr(matplotlib.axes.Axes, "barh", capture_barh)
    _paper_figures(
        tmp_path,
        frames["primary_transfer_gaps.csv"],
        frames["cross_session_by_direction.csv"],
        frames["prefix_ablation.csv"],
        frames["channel_ablation.csv"],
        frames["per_class_metrics.csv"],
        frames["seed_dispersion.csv"],
        frames["selected_confusions.csv"],
    )

    assert len(observed_errorbars) == 1
    assert observed_errorbars[0] is not None
    assert observed_errorbars[0].has_xerr
