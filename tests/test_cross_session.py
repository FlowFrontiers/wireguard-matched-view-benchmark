from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from vpncat.cross_session import (
    build_cross_session_contract,
    build_cross_session_split,
    enumerate_cross_session_runs,
    load_cross_session_config,
    validate_cross_session_contract,
)
from vpncat.errors import PipelineInvariantError
from vpncat.hashing import sha256_file


def _metadata() -> pd.DataFrame:
    rows = []
    for session, count in ((1, 20), (2, 10)):
        for label in ("A", "B", "C"):
            for index in range(count):
                rows.append(
                    {
                        "pair_id": f"s{session}:{label}:{index:03d}",
                        "session": session,
                        "application_category": label,
                    }
                )
    return pd.DataFrame(rows)


def _config(tmp_path: Path):
    project_root = Path(__file__).parents[1]
    base = load_cross_session_config(project_root / "configs" / "cross_session.yaml")
    canonical_path = tmp_path / "canonical.parquet"
    _metadata().to_parquet(canonical_path, index=False)
    canonical_hash = sha256_file(canonical_path)
    dataset_manifest_path = tmp_path / "dataset_manifest.json"
    dataset_manifest_path.write_text(
        json.dumps(
            {"artifacts": {"canonical_pairs": {"sha256": canonical_hash}}}
        ),
        encoding="utf-8",
    )
    feature_audit_path = tmp_path / "feature_audit.json"
    feature_audit_path.write_text(
        json.dumps({"status": "valid", "canonical": {"sha256": canonical_hash}}),
        encoding="utf-8",
    )
    primary = replace(
        base.primary,
        canonical_path=canonical_path,
        dataset_manifest_path=dataset_manifest_path,
        feature_audit_path=feature_audit_path,
    )
    return replace(
        base,
        primary=primary,
        split_path=tmp_path / "cross_session_split_manifest.csv",
        contract_audit_path=tmp_path / "cross_session_contract_audit.json",
        output_root=tmp_path / "outputs",
    )


def test_cross_session_split_is_input_order_independent_and_exactly_stratified(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    metadata = _metadata()
    metadata["session"] = metadata["session"].astype("int8")
    first = build_cross_session_split(metadata, config)
    shuffled = build_cross_session_split(
        metadata.sample(frac=1, random_state=99).reset_index(drop=True),
        config,
    )
    pd.testing.assert_frame_equal(first, shuffled, check_exact=True)

    for train_session, expected_validation in ((1, 6), (2, 3)):
        column = f"role_train_session_{train_session}"
        source = first["session"] == train_session
        target = ~source
        assert (first.loc[target, column] == "test").all()
        assert int((first[column] == "validation").sum()) == expected_validation
        validation_counts = first.loc[
            first[column] == "validation", "application_category"
        ].value_counts()
        assert validation_counts.to_dict() == {
            "A": expected_validation // 3,
            "B": expected_validation // 3,
            "C": expected_validation // 3,
        }


def test_cross_session_base_matrix_contains_30_unique_inner_trained_runs(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    runs = enumerate_cross_session_runs(config)
    assert len(runs) == 30
    assert len({run.run_id for run in runs}) == 30
    assert sum(run.family == "classical" for run in runs) == 12
    assert sum(run.family == "neural" for run in runs) == 18
    assert {run.train_domain for run in runs} == {"inner"}
    assert {run.test_domains for run in runs} == {("inner", "outer")}
    assert {(run.train_session, run.test_session) for run in runs} == {(1, 2), (2, 1)}


def test_cross_session_contract_build_validate_and_tamper_detection(tmp_path: Path) -> None:
    config = _config(tmp_path)
    report = build_cross_session_contract(config)
    validation = validate_cross_session_contract(config)

    assert report["matrix"] == {
        "training_runs": 30,
        "prediction_groups": 60,
        "family_counts": {"classical": 12, "neural": 18},
        "outer_trained_references": "deferred_until_primary_selection",
    }
    assert validation["status"] == "valid"
    assert validation["training_runs"] == 30
    split = pd.read_csv(config.split_path)
    split.loc[0, "role_train_session_1"] = "validation"
    split.to_csv(config.split_path, index=False)
    with pytest.raises(PipelineInvariantError, match="deterministic build"):
        validate_cross_session_contract(config)


def test_cross_session_contract_refuses_overwrite(tmp_path: Path) -> None:
    config = _config(tmp_path)
    build_cross_session_contract(config)
    with pytest.raises(FileExistsError, match="overwrite"):
        build_cross_session_contract(config)
