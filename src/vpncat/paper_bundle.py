from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from vpncat import __version__
from vpncat.errors import PipelineInvariantError
from vpncat.hashing import sha256_file
from vpncat.paper_analysis import (
    EXPECTED_COUNTS,
    SELECTED_CONFUSION_GROUPS,
    PaperAnalysisConfig,
    _generation_environment,
    _generation_provenance,
    _latex_tables,
    _paper_figures,
    build_paper_analysis,
)
from vpncat.paper_diagnostics import (
    DIAGNOSTIC_COUNTS,
    DIAGNOSTICS_EXTENSION_RECEIPT,
    ENCAPSULATION_DIAGNOSTICS,
    EXPECTED_CANONICAL_SHA256,
    EXPECTED_SEED_METRICS_SHA256,
    PER_CLASS_DIAGNOSTICS,
    SEED_METRICS_SOURCE,
    SEED_PAIR_DIAGNOSTICS,
    build_encapsulation_diagnostics,
    build_per_class_diagnostics,
    build_seed_pair_diagnostics,
    render_encapsulation_figures,
    validate_encapsulation_diagnostics,
    validate_per_class_diagnostics,
    validate_seed_pair_diagnostics,
)
from vpncat.provenance import git_provenance

EVIDENCE_FILES = tuple(EXPECTED_COUNTS)
VALIDATION_RECEIPT = "full_validation_receipt.json"
CONFUSION_EXTENSION_RECEIPT = "confusion_extension_receipt.json"
PRESENTATION_DIRS = ("figures", "latex", "tables")
PRESENTATION_FILES = (
    "figures/channel_ablation.pdf",
    "figures/class_distribution.pdf",
    "figures/cross_session_directions.pdf",
    "figures/encapsulation_ordering.pdf",
    "figures/encapsulation_padding.pdf",
    "figures/encapsulation_timing.pdf",
    "figures/per_class_f1.pdf",
    "figures/per_class_model_heatmap.pdf",
    "figures/per_class_transfer_change.pdf",
    "figures/prefix_ablation.pdf",
    "figures/seed_dispersion.pdf",
    "figures/selected_confusions.pdf",
    "figures/transfer_direction_asymmetry.pdf",
    "figures/transfer_gaps.pdf",
    "latex/results_macros.tex",
    "tables/cross_session.tex",
    "tables/dann_comparison.tex",
    "tables/primary_2x2.tex",
    "tables/primary_forward.tex",
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _evidence_frames(root: Path) -> dict[str, pd.DataFrame]:
    return {name: pd.read_csv(root / name, float_precision="round_trip") for name in EVIDENCE_FILES}


def _diagnostic_frames(root: Path) -> dict[str, pd.DataFrame]:
    return {
        name: pd.read_csv(root / name, float_precision="round_trip")
        for name in DIAGNOSTIC_COUNTS
    }


def _validate_selected_confusions(frame: pd.DataFrame, classes: list[str]) -> None:
    if set(frame["logical_group_id"]) != set(SELECTED_CONFUSION_GROUPS):
        raise PipelineInvariantError("Selected confusion group inventory differs")
    expected_cells = len(classes) ** 2
    for group_id in SELECTED_CONFUSION_GROUPS:
        selected = frame.loc[frame["logical_group_id"].eq(group_id)]
        if (
            len(selected) != expected_cells
            or set(selected["true_label"]) != set(classes)
            or set(selected["predicted_label"]) != set(classes)
            or selected[["true_label", "predicted_label"]].duplicated().any()
            or (selected["count"] < 0).any()
        ):
            raise PipelineInvariantError("Selected confusion cells differ")
        for _, rows in selected.groupby("true_label", sort=False):
            expected_fractions = rows["count"] / rows["support"]
            if (
                rows["support"].nunique() != 1
                or int(rows["count"].sum()) != int(rows["support"].iloc[0])
                or not (rows["row_fraction"] - expected_fractions).abs().le(1e-12).all()
                or abs(float(rows["row_fraction"].sum()) - 1.0) > 1e-12
            ):
                raise PipelineInvariantError("Selected confusion normalization differs")


def freeze_validated_evidence(
    config: PaperAnalysisConfig,
    *,
    validated_bundle: Path,
    output_root: Path,
) -> Path:
    """Adopt CSVs from a previously full-validated combined paper bundle."""
    validated_bundle = validated_bundle.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite paper bundle: {output_root}")
    receipt_path = validated_bundle / "analysis_manifest.json"
    receipt = _load_json(receipt_path)
    if (
        receipt.get("schema_version") != 1
        or receipt.get("status") != "complete"
        or receipt.get("campaign_revision") != config.expected_campaign_revision
        or receipt.get("campaign_archive_sha256") != config.expected_campaign_archive_sha256
        or receipt.get("source_analysis_manifest_sha256")
        != config.expected_analysis_manifest_sha256
        or receipt.get("analysis_contract_sha256") != config.expected_analysis_contract_sha256
        or receipt.get("bootstrap") != config.bootstrap
        or receipt.get("counts") != EXPECTED_COUNTS
    ):
        raise PipelineInvariantError("Validated source bundle identity differs")
    artifacts = receipt.get("artifacts", {})
    for name in EVIDENCE_FILES:
        if artifacts.get(name) != sha256_file(validated_bundle / name):
            raise PipelineInvariantError(f"Validated evidence hash differs: {name}")

    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.evidence-", dir=output_root.parent)
    )
    try:
        for name in EVIDENCE_FILES:
            shutil.copyfile(validated_bundle / name, staging / name)
        shutil.copyfile(receipt_path, staging / VALIDATION_RECEIPT)
        frames = _evidence_frames(staging)
        counts = {name: len(frame) for name, frame in frames.items()}
        if counts != EXPECTED_COUNTS:
            raise PipelineInvariantError("Frozen evidence row counts differ")
        manifest = {
            "schema_version": 1,
            "status": "complete",
            "created_utc": datetime.now(UTC).isoformat(),
            "package_version": __version__,
            "campaign_revision": config.expected_campaign_revision,
            "campaign_archive_sha256": config.expected_campaign_archive_sha256,
            "source_analysis_manifest_sha256": config.expected_analysis_manifest_sha256,
            "analysis_contract_sha256": config.expected_analysis_contract_sha256,
            "bootstrap": config.bootstrap,
            "class_order": receipt.get("class_order"),
            "counts": counts,
            "source_prediction_hashes": receipt.get("source_prediction_hashes"),
            "full_validation_receipt_sha256": sha256_file(receipt_path),
            "evidence_schemas": {name: list(frame.columns) for name, frame in frames.items()},
            "evidence_artifacts": {name: sha256_file(staging / name) for name in EVIDENCE_FILES},
        }
        _write_json(staging / "evidence_manifest.json", manifest)
        _validate_evidence_core(staging, config=config)
        staging.replace(output_root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output_root


def _validate_evidence_core(root: Path, *, config: PaperAnalysisConfig) -> dict[str, Any]:
    manifest = _load_json(root / "evidence_manifest.json")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("status") != "complete"
        or manifest.get("campaign_revision") != config.expected_campaign_revision
        or manifest.get("campaign_archive_sha256") != config.expected_campaign_archive_sha256
        or manifest.get("source_analysis_manifest_sha256")
        != config.expected_analysis_manifest_sha256
        or manifest.get("analysis_contract_sha256") != config.expected_analysis_contract_sha256
        or manifest.get("bootstrap") != config.bootstrap
        or manifest.get("counts") != EXPECTED_COUNTS
    ):
        raise PipelineInvariantError("Evidence manifest identity differs")
    frames = _evidence_frames(root)
    receipt_path = root / VALIDATION_RECEIPT
    if not receipt_path.is_file() or sha256_file(receipt_path) != manifest.get(
        "full_validation_receipt_sha256"
    ):
        raise PipelineInvariantError("Full-validation receipt hash differs")
    receipt = _load_json(receipt_path)
    _validate_selected_confusions(
        frames["selected_confusions.csv"], list(manifest.get("class_order", []))
    )
    for name, frame in frames.items():
        if len(frame) != EXPECTED_COUNTS[name]:
            raise PipelineInvariantError(f"Evidence row count differs: {name}")
        if list(frame.columns) != manifest.get("evidence_schemas", {}).get(name):
            raise PipelineInvariantError(f"Evidence schema differs: {name}")
        if sha256_file(root / name) != manifest.get("evidence_artifacts", {}).get(name):
            raise PipelineInvariantError(f"Evidence hash differs: {name}")
        if receipt.get("artifacts", {}).get(name) == manifest["evidence_artifacts"][name]:
            continue
        if name != "selected_confusions.csv":
            raise PipelineInvariantError(f"Validation receipt evidence differs: {name}")
        extension_path = root / CONFUSION_EXTENSION_RECEIPT
        if not extension_path.is_file() or sha256_file(extension_path) != manifest.get(
            "confusion_extension_receipt_sha256"
        ):
            raise PipelineInvariantError("Confusion extension receipt differs")
        extension = _load_json(extension_path)
        source_predictions = extension.get("source_predictions", {})
        expected_sources = {
            f"predictions/{group_id}.parquet": manifest["source_prediction_hashes"].get(
                f"predictions/{group_id}.parquet"
            )
            for group_id in SELECTED_CONFUSION_GROUPS
        }
        if (
            extension.get("schema_version") != 1
            or extension.get("status") != "complete"
            or extension.get("evidence_artifact")
            != {name: manifest["evidence_artifacts"][name]}
            or extension.get("row_count") != EXPECTED_COUNTS[name]
            or extension.get("class_order") != manifest.get("class_order")
            or source_predictions != expected_sources
        ):
            raise PipelineInvariantError("Confusion extension identity differs")
    if len(manifest.get("source_prediction_hashes", {})) != 44:
        raise PipelineInvariantError("Evidence source-prediction inventory differs")
    return {"status": "valid", "counts": EXPECTED_COUNTS}


def _validate_diagnostic_extension(root: Path, manifest: dict[str, Any]) -> None:
    receipt_path = root / DIAGNOSTICS_EXTENSION_RECEIPT
    if not receipt_path.is_file() or sha256_file(receipt_path) != manifest.get(
        "diagnostics_extension_receipt_sha256"
    ):
        raise PipelineInvariantError("Diagnostics extension receipt differs")
    receipt = _load_json(receipt_path)
    frames = _diagnostic_frames(root)
    observed_hashes = {name: sha256_file(root / name) for name in DIAGNOSTIC_COUNTS}
    if (
        receipt.get("schema_version") != 1
        or receipt.get("status") != "complete"
        or receipt.get("counts") != DIAGNOSTIC_COUNTS
        or receipt.get("evidence_artifacts") != observed_hashes
        or receipt.get("source_artifacts", {}).get("canonical_pairs.parquet")
        != EXPECTED_CANONICAL_SHA256
        or receipt.get("source_artifacts", {}).get("per_class_metrics.csv")
        != sha256_file(root / "per_class_metrics.csv")
        or receipt.get("source_artifacts", {}).get("seed_metrics.csv")
        != EXPECTED_SEED_METRICS_SHA256
        or not (root / SEED_METRICS_SOURCE).is_file()
        or sha256_file(root / SEED_METRICS_SOURCE) != EXPECTED_SEED_METRICS_SHA256
        or manifest.get("diagnostic_extension_counts") != DIAGNOSTIC_COUNTS
    ):
        raise PipelineInvariantError("Diagnostics extension identity differs")
    for name, frame in frames.items():
        if (
            len(frame) != DIAGNOSTIC_COUNTS[name]
            or manifest.get("evidence_artifacts", {}).get(name) != observed_hashes[name]
            or manifest.get("evidence_schemas", {}).get(name) != list(frame.columns)
        ):
            raise PipelineInvariantError(f"Diagnostic evidence differs: {name}")
    validate_encapsulation_diagnostics(frames[ENCAPSULATION_DIAGNOSTICS])
    validate_per_class_diagnostics(
        frames[PER_CLASS_DIAGNOSTICS],
        pd.read_csv(root / "per_class_metrics.csv", float_precision="round_trip"),
    )
    validate_seed_pair_diagnostics(frames[SEED_PAIR_DIAGNOSTICS])


def validate_evidence_quick(root: Path, *, config: PaperAnalysisConfig) -> dict[str, Any]:
    report = _validate_evidence_core(root, config=config)
    manifest = _load_json(root / "evidence_manifest.json")
    _validate_diagnostic_extension(root, manifest)
    return {**report, "diagnostic_counts": DIAGNOSTIC_COUNTS}


def extend_diagnostic_evidence(
    config: PaperAnalysisConfig,
    *,
    canonical_path: Path,
    dataset_manifest_path: Path,
    seed_metrics_path: Path,
    allow_dirty: bool = False,
    force: bool = False,
) -> Path:
    root = config.output_root.expanduser().resolve()
    canonical_path = canonical_path.expanduser().resolve()
    dataset_manifest_path = dataset_manifest_path.expanduser().resolve()
    seed_metrics_path = seed_metrics_path.expanduser().resolve()
    _validate_evidence_core(root, config=config)
    extension_paths = [
        *(root / name for name in DIAGNOSTIC_COUNTS),
        root / SEED_METRICS_SOURCE,
        root / DIAGNOSTICS_EXTENSION_RECEIPT,
    ]
    if any(path.exists() for path in extension_paths) and not force:
        raise FileExistsError("Refusing to overwrite diagnostic evidence without --force")

    provenance = _generation_provenance(config.project_root, allow_dirty=allow_dirty)
    encapsulation = build_encapsulation_diagnostics(canonical_path, dataset_manifest_path)
    per_class_source = pd.read_csv(
        root / "per_class_metrics.csv", float_precision="round_trip"
    )
    per_class = build_per_class_diagnostics(per_class_source)
    if sha256_file(seed_metrics_path) != EXPECTED_SEED_METRICS_SHA256:
        raise PipelineInvariantError("Seed metrics differ from the frozen campaign source")
    seed_pairs = build_seed_pair_diagnostics(
        pd.read_csv(seed_metrics_path, float_precision="round_trip")
    )

    staging = Path(tempfile.mkdtemp(prefix=f".{root.name}.diagnostics-", dir=root.parent))
    try:
        outputs = {
            ENCAPSULATION_DIAGNOSTICS: encapsulation,
            PER_CLASS_DIAGNOSTICS: per_class,
            SEED_PAIR_DIAGNOSTICS: seed_pairs,
        }
        for name, frame in outputs.items():
            frame.to_csv(staging / name, index=False, lineterminator="\n")
        evidence_hashes = {name: sha256_file(staging / name) for name in outputs}
        receipt = {
            "schema_version": 1,
            "status": "complete",
            "created_utc": datetime.now(UTC).isoformat(),
            "generation_git": provenance,
            "source_artifacts": {
                "canonical_pairs.parquet": sha256_file(canonical_path),
                "dataset_manifest.json": sha256_file(dataset_manifest_path),
                "per_class_metrics.csv": sha256_file(root / "per_class_metrics.csv"),
                "seed_metrics.csv": sha256_file(seed_metrics_path),
            },
            "counts": DIAGNOSTIC_COUNTS,
            "evidence_artifacts": evidence_hashes,
        }
        _write_json(staging / DIAGNOSTICS_EXTENSION_RECEIPT, receipt)
        manifest = _load_json(root / "evidence_manifest.json")
        manifest["diagnostic_extension_counts"] = DIAGNOSTIC_COUNTS
        manifest["diagnostics_extension_receipt_sha256"] = sha256_file(
            staging / DIAGNOSTICS_EXTENSION_RECEIPT
        )
        manifest["evidence_artifacts"].update(evidence_hashes)
        manifest["evidence_schemas"].update(
            {name: list(frame.columns) for name, frame in outputs.items()}
        )
        _write_json(staging / "evidence_manifest.json", manifest)
        shutil.copyfile(
            root / "per_class_metrics.csv", staging / "per_class_metrics.csv"
        )
        shutil.copyfile(seed_metrics_path, staging / SEED_METRICS_SOURCE)
        _validate_diagnostic_extension(staging, manifest)

        for name in (
            *DIAGNOSTIC_COUNTS,
            SEED_METRICS_SOURCE,
            DIAGNOSTICS_EXTENSION_RECEIPT,
        ):
            (staging / name).replace(root / name)
        (staging / "evidence_manifest.json").replace(root / "evidence_manifest.json")
        validate_evidence_quick(root, config=config)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return render_presentation(root, config=config)


def _render_macros(
    output: Path,
    *,
    primary: pd.DataFrame,
    pairwise: pd.DataFrame,
    dann: pd.DataFrame,
    encapsulation: pd.DataFrame,
    per_class_diagnostics: pd.DataFrame,
    seed_pair_diagnostics: pd.DataFrame,
    campaign_revision: str,
) -> None:
    latex = output / "latex"
    latex.mkdir()

    def primary_value(group_id: str, domain: str, metric: str) -> float:
        selected = primary.loc[
            primary["logical_group_id"].eq(group_id)
            & primary["test_domain"].eq(domain)
            & primary["metric"].eq(metric),
            "estimate",
        ]
        if len(selected) != 1:
            raise PipelineInvariantError("Presentation primary metric is not unique")
        return float(selected.iloc[0])

    def dann_value(configuration: str, metric: str) -> float:
        selected = dann.loc[
            dann["row_type"].eq("estimate")
            & dann["configuration"].eq(configuration)
            & dann["metric"].eq(metric),
            "estimate",
        ]
        if len(selected) != 1:
            raise PipelineInvariantError("Presentation DANN metric is not unique")
        return float(selected.iloc[0])

    gain = pairwise.loc[
        pairwise["comparison_id"].eq("dann_minus_plain_cnn1d") & pairwise["metric"].eq("macro_f1")
    ]
    if len(gain) != 1:
        raise PipelineInvariantError("Presentation DANN gain is not unique")
    gain_row = gain.iloc[0]
    encapsulation_summary = encapsulation.set_index("metric")["value"]
    correlations = per_class_diagnostics.loc[
        per_class_diagnostics["diagnostic"].eq(
            "support_vs_absolute_transfer_change_spearman"
        ),
        "value",
    ]
    dann_classes = per_class_diagnostics.loc[
        per_class_diagnostics["diagnostic"].eq("dann_minus_source_outer_f1")
    ].set_index("class_name")
    seed_gains = seed_pair_diagnostics["adapted_minus_source"]
    iat_within_one_percent = encapsulation.loc[
        encapsulation["metric"].eq("flow_mean_iat_relative_change_within")
        & encapsulation["x"].eq(0.01),
        "value",
    ]
    if len(iat_within_one_percent) != 1:
        raise PipelineInvariantError("One-percent IAT diagnostic is not unique")
    ordering_flow_row = encapsulation.loc[
        encapsulation["metric"].eq("eligible_flows_with_outer_order_inversion")
    ]
    if len(ordering_flow_row) != 1:
        raise PipelineInvariantError("Flow-order diagnostic is not unique")
    valid_iat_row = encapsulation.loc[
        encapsulation["metric"].eq("median_absolute_mean_iat_change_ms")
    ]
    if len(valid_iat_row) != 1:
        raise PipelineInvariantError("Valid-IAT diagnostic is not unique")
    plain = "primary__sequential_splt__cnn1d__train_inner"
    flattened_xgb = "primary__flattened_splt__xgboost__train_inner"
    values = {
        "CampaignRevision": campaign_revision[:7],
        "MatchedPairCount": "226,281",
        "XGBFlatInnerMacroFOne": f"{primary_value(flattened_xgb, 'inner', 'macro_f1'):.3f}",
        "XGBFlatOuterMacroFOne": f"{primary_value(flattened_xgb, 'outer', 'macro_f1'):.3f}",
        "CNNOuterBalancedAccuracy": f"{primary_value(plain, 'outer', 'balanced_accuracy'):.3f}",
        "CNNOuterMacroFOne": f"{primary_value(plain, 'outer', 'macro_f1'):.3f}",
        "DANNOuterBalancedAccuracy": f"{dann_value('dann', 'balanced_accuracy'):.3f}",
        "DANNOuterMacroFOne": f"{dann_value('dann', 'macro_f1'):.3f}",
        "DANNMacroFOneGain": f"{gain_row['delta_estimate']:.3f}",
        "DANNMacroFOneGainLow": f"{gain_row['delta_ci_low']:.3f}",
        "DANNMacroFOneGainHigh": f"{gain_row['delta_ci_high']:.3f}",
        "MeanVariablePaddingBytes": (
            f"{encapsulation_summary['weighted_mean_padding_bytes']:.1f}"
        ),
        "EligibleMatchedFlowCount": f"{int(ordering_flow_row['denominator'].item()):,}",
        "ValidMeanIATFlowCount": f"{int(valid_iat_row['count'].item()):,}",
        "OuterOrderInversionFlowPercent": (
            f"{100.0 * encapsulation_summary['eligible_flows_with_outer_order_inversion']:.1f}\\%"
        ),
        "MeanIATWithinOnePercent": (
            f"{100.0 * iat_within_one_percent.item():.1f}\\%"
        ),
        "SupportLossCorrelationLow": f"{correlations.min():.2f}",
        "SupportLossCorrelationHigh": f"{correlations.max():.2f}",
        "SourceOnlyConfigurationCount": str(len(correlations)),
        "DANNImprovedCategoryCount": str(int(dann_classes["value"].gt(0).sum())),
        "ApplicationCategoryCount": str(len(dann_classes)),
        "DANNCategoryCount": str(len(dann_classes)),
        "DANNPositiveSeedCount": str(int(seed_gains.gt(0).sum())),
        "DANNSeedCount": str(len(seed_gains)),
        "DANNMinimumSeedMacroFOneGain": f"{seed_gains.min():.3f}",
        "DANNMaximumSeedMacroFOneGain": f"{seed_gains.max():.3f}",
        "DANNDatabaseFOneGain": f"{dann_classes.loc['Database', 'value']:.3f}",
        "DANNConnCheckFOneGain": f"{dann_classes.loc['ConnCheck', 'value']:.3f}",
        "DANNSystemFOneGain": f"{dann_classes.loc['System', 'value']:.3f}",
        "DANNCollaborativeFOneGain": (
            f"{dann_classes.loc['Collaborative', 'value']:.3f}"
        ),
    }
    lines = ["% Generated; do not edit."] + [
        f"\\newcommand{{\\{name}}}{{{value}}}" for name, value in values.items()
    ]
    (latex / "results_macros.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _materialize_presentation(evidence_root: Path, output: Path, *, campaign_revision: str) -> None:
    frames = _evidence_frames(evidence_root)
    diagnostics = _diagnostic_frames(evidence_root)
    _latex_tables(
        output,
        frames["primary_2x2_metrics.csv"],
        frames["cross_session_by_direction.csv"],
        frames["dann_comparison.csv"],
    )
    _render_macros(
        output,
        primary=frames["primary_2x2_metrics.csv"],
        pairwise=frames["pairwise_method_differences.csv"],
        dann=frames["dann_comparison.csv"],
        encapsulation=diagnostics[ENCAPSULATION_DIAGNOSTICS],
        per_class_diagnostics=diagnostics[PER_CLASS_DIAGNOSTICS],
        seed_pair_diagnostics=diagnostics[SEED_PAIR_DIAGNOSTICS],
        campaign_revision=campaign_revision,
    )
    _paper_figures(
        output,
        frames["primary_transfer_gaps.csv"],
        frames["cross_session_by_direction.csv"],
        frames["prefix_ablation.csv"],
        frames["channel_ablation.csv"],
        frames["per_class_metrics.csv"],
        frames["seed_dispersion.csv"],
        frames["selected_confusions.csv"],
    )
    render_encapsulation_figures(output, diagnostics[ENCAPSULATION_DIAGNOSTICS])


def render_presentation(root: Path, *, config: PaperAnalysisConfig) -> Path:
    root = root.expanduser().resolve()
    validate_evidence_quick(root, config=config)
    generation_git = git_provenance(config.project_root)
    staging = Path(tempfile.mkdtemp(prefix=f".{root.name}.presentation-", dir=root.parent))
    try:
        _materialize_presentation(
            root, staging, campaign_revision=config.expected_campaign_revision
        )
        observed = {
            path.relative_to(staging).as_posix(): sha256_file(path)
            for path in sorted(staging.rglob("*"))
            if path.is_file()
        }
        if set(observed) != set(PRESENTATION_FILES):
            raise PipelineInvariantError("Presentation artifact inventory differs")
        manifest = {
            "schema_version": 1,
            "status": "complete",
            "created_utc": datetime.now(UTC).isoformat(),
            "package_version": __version__,
            "generation_git": generation_git,
            "generation_environment": _generation_environment(),
            "evidence_manifest_sha256": sha256_file(root / "evidence_manifest.json"),
            "presentation_artifacts": observed,
        }
        for directory in PRESENTATION_DIRS:
            destination = root / directory
            backup = root / f".{directory}.presentation-backup"
            if backup.exists() and not destination.exists():
                backup.replace(destination)
            if backup.exists():
                raise PipelineInvariantError(
                    f"Stale presentation backup requires inspection: {backup}"
                )
            if destination.exists():
                destination.replace(backup)
            try:
                (staging / directory).replace(destination)
            except Exception:
                if backup.exists() and not destination.exists():
                    backup.replace(destination)
                raise
            shutil.rmtree(backup, ignore_errors=True)
        staged_manifest = staging / "presentation_manifest.json"
        _write_json(staged_manifest, manifest)
        staged_manifest.replace(root / "presentation_manifest.json")
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return root


def validate_bundle_quick(
    root: Path,
    *,
    config: PaperAnalysisConfig,
    rerender: bool = True,
) -> dict[str, Any]:
    root = root.expanduser().resolve()
    evidence = validate_evidence_quick(root, config=config)
    manifest = _load_json(root / "presentation_manifest.json")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("status") != "complete"
        or manifest.get("evidence_manifest_sha256") != sha256_file(root / "evidence_manifest.json")
    ):
        raise PipelineInvariantError("Presentation manifest identity differs")
    observed_files = {
        path.relative_to(root).as_posix()
        for directory in PRESENTATION_DIRS
        for path in (root / directory).rglob("*")
        if path.is_file()
    }
    expected = manifest.get("presentation_artifacts", {})
    if observed_files != set(expected):
        raise PipelineInvariantError("Presentation file inventory differs")
    for relative, digest in expected.items():
        if sha256_file(root / relative) != digest:
            raise PipelineInvariantError(f"Presentation hash differs: {relative}")
    if rerender:
        with tempfile.TemporaryDirectory(prefix="vpncat-paper-quick-") as temporary:
            rebuilt = Path(temporary)
            _materialize_presentation(
                root,
                rebuilt,
                campaign_revision=config.expected_campaign_revision,
            )
            for relative in PRESENTATION_FILES:
                if sha256_file(root / relative) != sha256_file(rebuilt / relative):
                    raise PipelineInvariantError(f"Presentation differs from evidence: {relative}")
    return {
        "status": "valid",
        "mode": "quick",
        "evidence_counts": evidence["counts"],
        "presentation_files": len(PRESENTATION_FILES),
    }


def build_bundle_from_predictions(
    config: PaperAnalysisConfig, *, allow_dirty: bool = False
) -> Path:
    """Run the expensive prediction-derived evidence build.

    Canonical diagnostics and presentation outputs are added by the subsequent
    paper-diagnostics command because they have separate source artifacts.
    """
    if config.output_root.exists():
        raise FileExistsError(f"Refusing to overwrite paper bundle: {config.output_root}")
    with tempfile.TemporaryDirectory(prefix="vpncat-paper-full-") as temporary:
        combined_root = Path(temporary) / "combined"
        combined_config = replace(config, output_root=combined_root)
        build_paper_analysis(combined_config, allow_dirty=allow_dirty)
        freeze_validated_evidence(
            config,
            validated_bundle=combined_root,
            output_root=config.output_root,
        )
    return config.output_root


def validate_bundle_full(
    root: Path,
    *,
    config: PaperAnalysisConfig,
    allow_dirty: bool = False,
) -> dict[str, Any]:
    """Release-only recomputation of frozen evidence from source predictions."""
    quick = validate_bundle_quick(root, config=config, rerender=True)
    with tempfile.TemporaryDirectory(prefix="vpncat-paper-full-check-") as temporary:
        combined_root = Path(temporary) / "combined"
        combined_config = replace(config, output_root=combined_root)
        build_paper_analysis(combined_config, allow_dirty=allow_dirty)
        for name in EVIDENCE_FILES:
            if sha256_file(root / name) != sha256_file(combined_root / name):
                raise PipelineInvariantError(
                    f"Frozen evidence differs from source predictions: {name}"
                )
    return {**quick, "mode": "full"}
