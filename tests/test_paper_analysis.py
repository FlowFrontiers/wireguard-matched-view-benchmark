from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from vpncat.errors import PipelineInvariantError
from vpncat.paper_analysis import _generation_provenance, load_paper_analysis_config


def test_paper_analysis_config_freezes_campaign_and_bootstrap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    configs = project / "configs"
    configs.mkdir(parents=True)
    source = Path("configs/paper_analysis.yaml").read_text(encoding="utf-8")
    config_path = configs / "paper_analysis.yaml"
    config_path.write_text(source, encoding="utf-8")
    monkeypatch.setattr("vpncat.paper_analysis.load_analysis_config", lambda *_: object())
    config = load_paper_analysis_config(config_path)
    assert config.expected_campaign_revision.startswith("f82a743")
    assert config.bootstrap == {
        "resampling_unit": "pair_id",
        "replicates": 1000,
        "confidence_level": 0.95,
        "seed": 42,
    }


def test_paper_analysis_config_rejects_bootstrap_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "paper_analysis.yaml"
    raw = yaml.safe_load(Path("configs/paper_analysis.yaml").read_text(encoding="utf-8"))
    raw["paper_analysis"]["bootstrap"]["replicates"] = 999
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    monkeypatch.setattr("vpncat.paper_analysis.load_analysis_config", lambda *_: object())
    with pytest.raises(PipelineInvariantError, match="differs from the freeze"):
        load_paper_analysis_config(config_path)


def test_generation_provenance_requires_clean_tree_unless_preview_is_explicit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dirty = {"status_available": True, "revision": "a" * 40, "dirty": True}
    monkeypatch.setattr("vpncat.paper_analysis.git_provenance", lambda *_: dirty)
    with pytest.raises(PipelineInvariantError, match="clean committed revision"):
        _generation_provenance(tmp_path, allow_dirty=False)
    assert _generation_provenance(tmp_path, allow_dirty=True) == dirty
