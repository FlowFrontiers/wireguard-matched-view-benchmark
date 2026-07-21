from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pytest

from vpncat.errors import PipelineInvariantError
from vpncat.paper_analysis import load_paper_analysis_config
from vpncat.paper_bundle import extend_diagnostic_evidence, validate_bundle_quick
from vpncat.paper_diagnostics import (
    DANN_CNN1D,
    ENCAPSULATION_DIAGNOSTICS,
    PER_CLASS_DIAGNOSTICS,
    SEED_PAIR_DIAGNOSTICS,
    build_encapsulation_diagnostics,
    build_per_class_diagnostics,
    build_seed_pair_diagnostics,
    render_encapsulation_figures,
    validate_per_class_diagnostics,
    validate_seed_pair_diagnostics,
)

CANONICAL_PATH = Path("data/processed/canonical_pairs.parquet")
DATASET_MANIFEST_PATH = Path("data/processed/dataset_manifest.json")
REQUIRES_CANONICAL_DATA = pytest.mark.skipif(
    not CANONICAL_PATH.exists() or not DATASET_MANIFEST_PATH.exists(),
    reason="requires the rebuilt canonical dataset",
)


@REQUIRES_CANONICAL_DATA
def test_encapsulation_diagnostics_match_frozen_canonical_data(tmp_path: Path) -> None:
    frame = build_encapsulation_diagnostics(
        CANONICAL_PATH,
        DATASET_MANIFEST_PATH,
    )
    values = frame.set_index("metric")["value"]
    assert len(frame) == 26
    assert values["weighted_mean_padding_bytes"] == pytest.approx(4.775397632462357)
    assert values["eligible_flows_with_outer_order_inversion"] == pytest.approx(
        25972 / 226421
    )
    assert values["median_absolute_mean_iat_change_ms"] == pytest.approx(
        0.01612232349543774
    )
    render_encapsulation_figures(tmp_path, frame)
    assert {
        path.name for path in (tmp_path / "figures").glob("encapsulation_*.pdf")
    } == {
        "encapsulation_ordering.pdf",
        "encapsulation_padding.pdf",
        "encapsulation_timing.pdf",
    }


def test_per_class_diagnostics_reproduce_existing_evidence() -> None:
    source = pd.read_csv(
        "artifacts/f82a743/paper_analysis/per_class_metrics.csv",
        float_precision="round_trip",
    )
    frame = build_per_class_diagnostics(source)
    correlations = frame.loc[
        frame["diagnostic"].eq("support_vs_absolute_transfer_change_spearman"),
        "value",
    ]
    dann = frame.loc[
        frame["diagnostic"].eq("dann_minus_source_outer_f1")
        & frame["logical_group_id"].eq(DANN_CNN1D)
    ].set_index("class_name")
    assert len(frame) == 23
    assert correlations.min() == pytest.approx(-0.7494505494505495)
    assert correlations.max() == pytest.approx(-0.45494505494505494)
    assert int(dann["value"].gt(0).sum()) == 9
    assert dann.loc["Database", "value"] == pytest.approx(0.066512, abs=1e-6)
    assert dann.loc["System", "value"] == pytest.approx(-0.022676, abs=1e-6)


def test_per_class_validation_rejects_hash_repaired_content_change() -> None:
    source = pd.read_csv(
        "artifacts/f82a743/paper_analysis/per_class_metrics.csv",
        float_precision="round_trip",
    )
    frame = build_per_class_diagnostics(source)
    frame.loc[frame["diagnostic"].eq("dann_minus_source_outer_f1"), "value"] += 0.01
    with pytest.raises(PipelineInvariantError, match="differ from source evidence"):
        validate_per_class_diagnostics(frame, source)


def test_seed_pair_diagnostics_reproduce_frozen_campaign_source() -> None:
    source = pd.read_csv(
        "artifacts/f82a743/paper_analysis/seed_metrics.csv",
        float_precision="round_trip",
    )
    frame = build_seed_pair_diagnostics(source)
    assert len(frame) == 3
    assert frame["adapted_minus_source"].tolist() == pytest.approx(
        [0.0499210660878665, 0.0050629248062219, 0.0198887898807182]
    )
    assert frame["adapted_minus_source"].gt(0).all()


def test_seed_pair_validation_rejects_hash_repaired_content_change() -> None:
    source = pd.read_csv(
        "artifacts/f82a743/paper_analysis/seed_metrics.csv",
        float_precision="round_trip",
    )
    frame = build_seed_pair_diagnostics(source)
    frame.loc[0, "adapted_minus_source"] += 0.01
    with pytest.raises(PipelineInvariantError, match="invariants differ"):
        validate_seed_pair_diagnostics(frame, source)


def test_diagnostic_file_names_remain_stable() -> None:
    assert ENCAPSULATION_DIAGNOSTICS == "encapsulation_diagnostics.csv"
    assert PER_CLASS_DIAGNOSTICS == "per_class_diagnostics.csv"
    assert SEED_PAIR_DIAGNOSTICS == "seed_pair_diagnostics.csv"


@REQUIRES_CANONICAL_DATA
def test_extension_restamp_validates_before_publishing(tmp_path: Path) -> None:
    copied = tmp_path / "paper_analysis"
    shutil.copytree("artifacts/f82a743/paper_analysis", copied)
    config = load_paper_analysis_config(
        Path("configs/paper_analysis.yaml"), output_root=copied
    )
    extend_diagnostic_evidence(
        config,
        canonical_path=CANONICAL_PATH,
        dataset_manifest_path=DATASET_MANIFEST_PATH,
        seed_metrics_path=Path(
            "artifacts/f82a743/paper_analysis/seed_metrics.csv"
        ),
        allow_dirty=True,
        force=True,
    )
    assert validate_bundle_quick(copied, config=config, rerender=True)["status"] == "valid"
