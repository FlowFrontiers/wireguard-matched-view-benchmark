from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from vpncat.errors import PipelineInvariantError
from vpncat.hashing import sha256_file
from vpncat.provenance import git_provenance

COPIED_FILES = (
    "endpoint_overlap.csv",
    "excluded_categories.csv",
    "temporal_block_intervals.csv",
    "source_collection_structure_manifest.json",
    "source_temporal_bootstrap_manifest.json",
)


def _single_row(frame: pd.DataFrame, **selectors: object) -> pd.Series:
    selected = frame
    for column, value in selectors.items():
        selected = selected.loc[selected[column].eq(value)]
    if len(selected) != 1:
        raise PipelineInvariantError(f"Expected one row for selectors: {selectors}")
    return selected.iloc[0]


def _interval(
    temporal: pd.DataFrame,
    comparison_id: str,
    *,
    metric: str = "macro_f1",
    block_hours: int = 2,
) -> tuple[float, float]:
    row = _single_row(
        temporal,
        comparison_id=comparison_id,
        metric=metric,
        block_hours=block_hours,
    )
    return float(row["delta_ci_low"]), float(row["delta_ci_high"])


def build_robustness_macros(
    endpoint_overlap: pd.DataFrame,
    excluded: pd.DataFrame,
    temporal: pd.DataFrame,
) -> str:
    primary = endpoint_overlap.loc[
        endpoint_overlap["protocol"].eq("primary")
        & endpoint_overlap["endpoint_key"].eq("remote_ip")
    ]
    cross_session = endpoint_overlap.loc[
        endpoint_overlap["protocol"].eq("cross_session")
        & endpoint_overlap["endpoint_key"].eq("remote_ip")
    ]
    if len(primary) != 5 or len(cross_session) != 2:
        raise PipelineInvariantError("Endpoint-overlap rows differ from the protocol contract")
    if len(excluded) != 6 or int(excluded["eligible_flow_count"].sum()) != 140:
        raise PipelineInvariantError("Excluded-category totals differ from the canonical manifest")

    two_hour_gaps = temporal.loc[
        temporal["comparison_type"].eq("view_gap")
        & temporal["metric"].eq("macro_f1")
        & temporal["block_hours"].eq(2)
    ]
    if len(two_hour_gaps) != 9 or not two_hour_gaps["delta_ci_high"].lt(0).all():
        raise PipelineInvariantError(
            "Two-hour primary gap result differs from the frozen extension"
        )

    cnn_gap = _interval(
        temporal, "primary__sequential_splt__cnn1d__train_inner"
    )
    cnn_lstm = _interval(temporal, "cnn1d_minus_lstm")
    cnn_transformer = _interval(temporal, "cnn1d_minus_transformer")
    dann_plain = _interval(temporal, "dann_minus_plain_cnn1d")
    dann_supervised = _interval(temporal, "dann_minus_supervised_outer_cnn1d")

    def decimal(value: float) -> str:
        return f"{value:.3f}"

    def percent(value: float) -> str:
        return f"{100 * value:.1f}\\%"

    lines = [
        "% Generated from the robustness evidence extension; do not edit.",
        f"\\newcommand{{\\ExcludedCategoryCount}}{{{len(excluded)}}}",
        "\\newcommand{\\ExcludedBelowSupportFlowCount}"
        f"{{{int(excluded['eligible_flow_count'].sum()):,}}}",
        f"\\newcommand{{\\PrimaryRemoteIPOverlapLow}}{{{percent(float(primary['target_seen_fraction'].min()))}}}",
        f"\\newcommand{{\\PrimaryRemoteIPOverlapHigh}}{{{percent(float(primary['target_seen_fraction'].max()))}}}",
        f"\\newcommand{{\\CrossSessionRemoteIPOverlapLow}}{{{percent(float(cross_session['target_seen_fraction'].min()))}}}",
        f"\\newcommand{{\\CrossSessionRemoteIPOverlapHigh}}{{{percent(float(cross_session['target_seen_fraction'].max()))}}}",
        f"\\newcommand{{\\TemporalPrimaryGapCount}}{{{len(two_hour_gaps)}}}",
        f"\\newcommand{{\\CNNTemporalGapLow}}{{{decimal(cnn_gap[0])}}}",
        f"\\newcommand{{\\CNNTemporalGapHigh}}{{{decimal(cnn_gap[1])}}}",
        f"\\newcommand{{\\CNNLSTMTemporalDifferenceLow}}{{{decimal(cnn_lstm[0])}}}",
        f"\\newcommand{{\\CNNLSTMTemporalDifferenceHigh}}{{{decimal(cnn_lstm[1])}}}",
        f"\\newcommand{{\\CNNTransformerTemporalDifferenceLow}}{{{decimal(cnn_transformer[0])}}}",
        f"\\newcommand{{\\CNNTransformerTemporalDifferenceHigh}}{{{decimal(cnn_transformer[1])}}}",
        f"\\newcommand{{\\DANNTemporalGainLow}}{{{decimal(dann_plain[0])}}}",
        f"\\newcommand{{\\DANNTemporalGainHigh}}{{{decimal(dann_plain[1])}}}",
        f"\\newcommand{{\\DANNTemporalVsSupervisedLow}}{{{decimal(dann_supervised[0])}}}",
        f"\\newcommand{{\\DANNTemporalVsSupervisedHigh}}{{{decimal(dann_supervised[1])}}}",
    ]
    return "\n".join(lines) + "\n"


def _validate_source_manifest(path: Path, expected_files: tuple[str, ...]) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise PipelineInvariantError(f"Unsupported source manifest: {path}")
    outputs = manifest.get("outputs", {})
    for filename in expected_files:
        record = outputs.get(filename)
        source = path.parent / filename
        if not record or not source.exists() or sha256_file(source) != record.get("sha256"):
            raise PipelineInvariantError(f"Source extraction changed: {source}")
    return manifest


def validate_robustness_extension(output_dir: Path) -> dict[str, Any]:
    receipt_path = output_dir / "robustness_extension_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("status") != "complete" or receipt.get("schema_version") != 1:
        raise PipelineInvariantError("Robustness extension receipt is incomplete")
    for filename, expected in receipt.get("evidence_artifacts", {}).items():
        path = output_dir / filename
        if not path.exists() or sha256_file(path) != expected:
            raise PipelineInvariantError(f"Robustness extension artifact changed: {filename}")
    endpoint = pd.read_csv(output_dir / "endpoint_overlap.csv")
    excluded = pd.read_csv(output_dir / "excluded_categories.csv")
    temporal = pd.read_csv(output_dir / "temporal_block_intervals.csv")
    expected_macros = build_robustness_macros(endpoint, excluded, temporal)
    if (output_dir / "robustness_macros.tex").read_text(encoding="utf-8") != expected_macros:
        raise PipelineInvariantError("Robustness macros disagree with extension evidence")
    return receipt


def publish_robustness_extension(
    *,
    collection_dir: Path,
    temporal_dir: Path,
    output_dir: Path,
    project_root: Path,
    force: bool = False,
) -> dict[str, Any]:
    collection_manifest_path = collection_dir / "collection_structure_manifest.json"
    temporal_manifest_path = temporal_dir / "temporal_bootstrap_manifest.json"
    _validate_source_manifest(
        collection_manifest_path, ("endpoint_overlap.csv", "excluded_categories.csv")
    )
    _validate_source_manifest(
        temporal_manifest_path, ("temporal_block_intervals.csv",)
    )
    provenance = git_provenance(project_root)
    if provenance.get("dirty"):
        raise PipelineInvariantError("Publish the robustness extension from a clean revision")
    if output_dir.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite robustness extension: {output_dir}")

    with tempfile.TemporaryDirectory(dir=output_dir.parent) as temporary:
        staging = Path(temporary) / output_dir.name
        staging.mkdir()
        shutil.copy2(collection_dir / "endpoint_overlap.csv", staging)
        shutil.copy2(collection_dir / "excluded_categories.csv", staging)
        shutil.copy2(temporal_dir / "temporal_block_intervals.csv", staging)
        shutil.copy2(
            collection_manifest_path,
            staging / "source_collection_structure_manifest.json",
        )
        shutil.copy2(
            temporal_manifest_path,
            staging / "source_temporal_bootstrap_manifest.json",
        )
        endpoint = pd.read_csv(staging / "endpoint_overlap.csv")
        excluded = pd.read_csv(staging / "excluded_categories.csv")
        temporal = pd.read_csv(staging / "temporal_block_intervals.csv")
        (staging / "robustness_macros.tex").write_text(
            build_robustness_macros(endpoint, excluded, temporal), encoding="utf-8"
        )
        artifact_names = (*COPIED_FILES, "robustness_macros.tex")
        receipt = {
            "schema_version": 1,
            "status": "complete",
            "generation_git": provenance,
            "source_artifacts": {
                "collection_structure_manifest.json": sha256_file(collection_manifest_path),
                "temporal_bootstrap_manifest.json": sha256_file(temporal_manifest_path),
            },
            "counts": {
                "endpoint_overlap.csv": len(endpoint),
                "excluded_categories.csv": len(excluded),
                "temporal_block_intervals.csv": len(temporal),
            },
            "evidence_artifacts": {
                filename: sha256_file(staging / filename) for filename in artifact_names
            },
            "scope": (
                "Postprocessing only: endpoint recurrence, support filtering, and "
                "time-cluster resampling of frozen predictions; no model fitting or inference"
            ),
        }
        (staging / "robustness_extension_receipt.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        validate_robustness_extension(staging)
        if output_dir.exists():
            shutil.rmtree(output_dir)
        staging.replace(output_dir)
    return validate_robustness_extension(output_dir)
