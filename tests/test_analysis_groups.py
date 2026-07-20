from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vpncat.ablations import enumerate_ablation_runs, primary_reference_run
from vpncat.analysis import load_analysis_config
from vpncat.analysis_groups import (
    AnalysisGroup,
    ensemble_partition,
    enumerate_groups_from_contract,
)
from vpncat.cross_session import enumerate_cross_session_runs
from vpncat.dann import enumerate_dann_runs
from vpncat.errors import PipelineInvariantError
from vpncat.experiment import enumerate_primary_runs


def _config():
    return load_analysis_config(Path(__file__).parents[1] / "configs" / "analysis.yaml")


def _artifact(run, protocol: str) -> dict:
    return {
        "artifact_id": f"{protocol}:{run.run_id}",
        "protocol": protocol,
        "output_group": protocol,
        "relative_output_dir": run.relative_output_dir.as_posix(),
        "expected_prediction_rows": 2,
        "run": run.to_dict(),
    }


def _contract() -> dict:
    config = _config()
    artifacts = [
        *(_artifact(run, "primary") for run in enumerate_primary_runs(config.primary)),
        *(
            _artifact(run, "cross_session")
            for run in enumerate_cross_session_runs(config.cross_session)
        ),
        *(_artifact(run, "dann") for run in enumerate_dann_runs(config.dann)),
    ]
    references = []
    for ablation_config, protocol in (
        (config.ablation_prefix, "ablation_prefix"),
        (config.ablation_channels, "ablation_channels"),
    ):
        for run in enumerate_ablation_runs(ablation_config):
            if run.is_primary_reference:
                reference = primary_reference_run(ablation_config, run)
                references.append(
                    {
                        "logical_protocol": protocol,
                        "logical_run": run.to_dict(),
                        "physical_artifact_id": f"primary:{reference.run_id}",
                    }
                )
            else:
                artifacts.append(_artifact(run, protocol))
    return {"physical_artifacts": artifacts, "logical_references": references}


def test_group_planner_freezes_seed_and_reference_semantics() -> None:
    groups = enumerate_groups_from_contract(_contract())
    assert len(groups) == 44
    assert sum(len(group.logical_group_ids) for group in groups) == 46
    assert sum(group.seed_policy == "mean_probabilities" for group in groups) == 10
    anchors = [group for group in groups if group.protocol == "ablation_anchor"]
    assert len(anchors) == 2
    assert {group.model for group in anchors} == {"cnn1d", "transformer"}
    assert all(len(group.artifact_ids) == 5 for group in anchors)
    assert all(len(group.logical_group_ids) == 2 for group in anchors)
    assert {
        logical for group in anchors for logical in group.logical_group_ids
    } == {
        "ablation_prefix__n050__cnn1d",
        "ablation_prefix__n050__transformer",
        "ablation_channels__all__cnn1d",
        "ablation_channels__all__transformer",
    }


def _group() -> AnalysisGroup:
    return AnalysisGroup(
        group_id="primary__sequential_splt__cnn1d__train_inner",
        protocol="primary",
        experiment_id="sequential_splt__cnn1d",
        representation="sequential_splt",
        model="cnn1d",
        family="neural",
        train_domain="inner",
        seed_policy="mean_probabilities",
        artifact_ids=("seed42", "seed43", "seed44"),
        logical_group_ids=("primary__sequential_splt__cnn1d__train_inner",),
    )


def _prediction(seed: int, probabilities: list[list[float]]) -> pd.DataFrame:
    predictions = np.asarray(("A", "B"), dtype=object)[
        np.argmax(np.asarray(probabilities), axis=1)
    ]
    return pd.DataFrame(
        {
            "run_id": [f"run-{seed}"] * 4,
            "protocol": ["primary"] * 4,
            "representation": ["sequential_splt"] * 4,
            "model": ["cnn1d"] * 4,
            "pair_id": ["a", "b", "a", "b"],
            "session": [1, 2, 1, 2],
            "train_domain": ["inner"] * 4,
            "test_domain": ["inner", "inner", "outer", "outer"],
            "fold": [1] * 4,
            "seed": [seed] * 4,
            "true_label": ["A", "B", "A", "B"],
            "prediction": predictions.astype(str),
            "class_probabilities": probabilities,
        }
    )


def test_ensemble_averages_probabilities_before_argmax() -> None:
    probabilities = {
        42: [[0.9, 0.1], [0.2, 0.8], [0.9, 0.1], [0.2, 0.8]],
        43: [[0.6, 0.4], [0.4, 0.6], [0.6, 0.4], [0.4, 0.6]],
        44: [[0.3, 0.7], [0.6, 0.4], [0.3, 0.7], [0.6, 0.4]],
    }
    inputs = [
        (
            {"run": {"seed": seed, "fold": 1}},
            _prediction(seed, values),
            ("A", "B"),
        )
        for seed, values in probabilities.items()
    ]
    result = ensemble_partition(_group(), inputs)
    matrix = np.asarray(result["class_probabilities"].tolist())
    np.testing.assert_allclose(
        matrix,
        np.mean(np.asarray(list(probabilities.values())), axis=0),
    )
    assert result["prediction"].tolist() == ["A", "B", "A", "B"]
    assert result["seed_count"].unique().tolist() == [3]
    assert result["seeds"].iloc[0] == [42, 43, 44]


def test_ensemble_rejects_seed_identity_misalignment() -> None:
    inputs = [
        (
            {"run": {"seed": seed, "fold": 1}},
            _prediction(seed, [[0.8, 0.2], [0.2, 0.8]] * 2),
            ("A", "B"),
        )
        for seed in (42, 43, 44)
    ]
    inputs[1][1].at[0, "pair_id"] = "wrong"
    with pytest.raises(PipelineInvariantError, match="identities differ"):
        ensemble_partition(_group(), inputs)
