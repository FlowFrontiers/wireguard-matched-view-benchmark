from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from vpncat import __version__
from vpncat.aggregate_results import validate_analysis_output
from vpncat.analysis import AnalysisConfig, load_analysis_config
from vpncat.analysis_statistics import paired_bootstrap_intervals
from vpncat.errors import PipelineInvariantError
from vpncat.hashing import sha256_file
from vpncat.paper_statistics import PRIMARY_METRICS, paired_method_intervals, per_class_metrics
from vpncat.provenance import git_provenance

EXPECTED_COUNTS = {
    "claim_ledger.csv": 12,
    "primary_2x2_metrics.csv": 72,
    "primary_transfer_gaps.csv": 36,
    "pairwise_method_differences.csv": 16,
    "cross_session_by_direction.csv": 36,
    "dann_comparison.csv": 10,
    "prefix_ablation.csv": 32,
    "channel_ablation.csv": 40,
    "per_class_metrics.csv": 1540,
    "seed_dispersion.csv": 100,
    "selected_confusions.csv": 588,
}

REPRESENTATION_LABELS = {
    "flattened_splt": "Flattened-SPLT-50",
    "matched_flow_stats": "MatchedFlowStats",
    "prefix_stats": "PrefixStats-50",
    "sequential_splt": "Sequential-SPLT-50",
}
MODEL_LABELS = {
    "random_forest": "RF",
    "xgboost": "XGBoost",
    "cnn1d": "CNN1D",
    "lstm": "LSTM",
    "transformer": "Transformer",
    "dann_cnn1d": "DANN-CNN1D",
}


@dataclass(frozen=True)
class PaperAnalysisConfig:
    config_path: Path
    project_root: Path
    analysis: AnalysisConfig
    source_analysis_root: Path
    output_root: Path
    expected_campaign_revision: str
    expected_campaign_archive_sha256: str
    expected_analysis_manifest_sha256: str
    expected_analysis_contract_sha256: str
    bootstrap: dict[str, Any]


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def load_paper_analysis_config(
    path: Path,
    *,
    source_analysis_root: Path | None = None,
    output_root: Path | None = None,
) -> PaperAnalysisConfig:
    path = path.expanduser().resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")).get("paper_analysis", {})
    root = path.parent.parent
    analysis = load_analysis_config(_resolve(root, raw.get("analysis_config_path", "")))
    config = PaperAnalysisConfig(
        config_path=path,
        project_root=root,
        analysis=analysis,
        source_analysis_root=_resolve(
            root,
            source_analysis_root
            if source_analysis_root is not None
            else raw.get("source_analysis_root", ""),
        ),
        output_root=_resolve(
            root, output_root if output_root is not None else raw.get("output_root", "")
        ),
        expected_campaign_revision=str(raw.get("expected_campaign_revision", "")),
        expected_campaign_archive_sha256=str(raw.get("expected_campaign_archive_sha256", "")),
        expected_analysis_manifest_sha256=str(raw.get("expected_analysis_manifest_sha256", "")),
        expected_analysis_contract_sha256=str(raw.get("expected_analysis_contract_sha256", "")),
        bootstrap=dict(raw.get("bootstrap", {})),
    )
    expected_bootstrap = {
        "resampling_unit": "pair_id",
        "replicates": 1000,
        "confidence_level": 0.95,
        "seed": 42,
    }
    if (
        raw.get("protocol") != "paper_analysis"
        or config.bootstrap != expected_bootstrap
        or len(config.expected_campaign_revision) != 40
        or any(
            len(value) != 64
            for value in (
                config.expected_campaign_archive_sha256,
                config.expected_analysis_manifest_sha256,
                config.expected_analysis_contract_sha256,
            )
        )
    ):
        raise PipelineInvariantError("Paper-analysis configuration differs from the freeze")
    return config


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _generation_provenance(project_root: Path, *, allow_dirty: bool) -> dict[str, Any]:
    provenance = git_provenance(project_root)
    if (
        not provenance.get("status_available")
        or provenance.get("revision") in {None, "", "UNBORN"}
        or (provenance.get("dirty") and not allow_dirty)
    ):
        raise PipelineInvariantError(
            "Paper analysis requires a clean committed revision; "
            "use --allow-dirty only for audit previews"
        )
    return provenance


def _generation_environment() -> dict[str, str]:
    return {
        package: version(package)
        for package in ("matplotlib", "numpy", "pandas", "pyarrow", "scikit-learn")
    }


def _validate_source(config: PaperAnalysisConfig) -> dict[str, Any]:
    manifest_path = config.source_analysis_root / "analysis.json"
    if sha256_file(manifest_path) != config.expected_analysis_manifest_sha256:
        raise PipelineInvariantError("Source analysis manifest differs from the frozen campaign")
    manifest = _load_json(manifest_path)
    if (
        manifest.get("git") != {"revision": config.expected_campaign_revision, "dirty": False}
        or manifest.get("analysis_contract_sha256") != config.expected_analysis_contract_sha256
    ):
        raise PipelineInvariantError("Source analysis identity differs from the frozen campaign")
    validate_analysis_output(
        config.source_analysis_root,
        config=config.analysis,
        expected_revision=config.expected_campaign_revision,
    )
    return manifest


class _Evidence:
    def __init__(self, config: PaperAnalysisConfig) -> None:
        self.config = config
        self.aliases = pd.read_csv(config.source_analysis_root / "logical_aliases.csv")
        self.metrics = pd.read_csv(config.source_analysis_root / "metrics_summary.csv")
        self.bootstrap = pd.read_csv(config.source_analysis_root / "bootstrap_intervals.csv")
        self._paths = {
            str(row.logical_group_id): config.source_analysis_root / str(row.prediction_file)
            for row in self.aliases.itertuples(index=False)
        }
        self._frames: dict[str, pd.DataFrame] = {}
        self._classes: tuple[str, ...] | None = None

    def frame(self, group_id: str) -> pd.DataFrame:
        if group_id not in self._paths:
            raise PipelineInvariantError(f"Unknown paper-analysis group: {group_id}")
        if group_id not in self._frames:
            frame = pd.read_parquet(self._paths[group_id])
            classes = tuple(sorted(frame["true_label"].astype(str).unique()))
            if self._classes is None:
                self._classes = classes
            elif self._classes != classes:
                raise PipelineInvariantError("Paper-analysis class orders differ")
            self._frames[group_id] = frame
        return self._frames[group_id]

    @property
    def classes(self) -> tuple[str, ...]:
        if self._classes is None:
            self.frame("primary__sequential_splt__cnn1d__train_inner")
        assert self._classes is not None
        return self._classes

    def metric(self, group_id: str, domain: str, metric: str) -> float:
        selected = self.metrics.loc[
            self.metrics["logical_group_id"].eq(group_id)
            & self.metrics["test_domain"].eq(domain)
            & self.metrics["metric"].eq(metric),
            "value",
        ]
        if len(selected) != 1:
            raise PipelineInvariantError("Paper metric lookup is not unique")
        return float(selected.iloc[0])

    def interval(self, group_id: str, metric: str) -> pd.Series:
        selected = self.bootstrap.loc[
            self.bootstrap["logical_group_id"].eq(group_id) & self.bootstrap["metric"].eq(metric)
        ]
        if len(selected) != 1:
            raise PipelineInvariantError("Paper interval lookup is not unique")
        return selected.iloc[0]


def _primary_identity(group_id: str) -> tuple[str, str, str]:
    parts = group_id.split("__")
    if len(parts) != 4 or parts[0] != "primary" or not parts[3].startswith("train_"):
        raise PipelineInvariantError(f"Invalid primary logical ID: {group_id}")
    return parts[1], parts[2], parts[3].removeprefix("train_")


def _primary_tables(evidence: _Evidence) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    gaps = []
    primary_ids = sorted(
        group_id for group_id in evidence._paths if group_id.startswith("primary__")
    )
    for group_id in primary_ids:
        representation, model, train_domain = _primary_identity(group_id)
        for test_domain in ("inner", "outer"):
            for metric in PRIMARY_METRICS:
                interval = evidence.interval(group_id, metric)
                rows.append(
                    {
                        "logical_group_id": group_id,
                        "representation": representation,
                        "model": model,
                        "train_domain": train_domain,
                        "test_domain": test_domain,
                        "metric": metric,
                        "estimate": float(interval[f"{test_domain}_estimate"]),
                        "ci_low": float(interval[f"{test_domain}_ci_low"]),
                        "ci_high": float(interval[f"{test_domain}_ci_high"]),
                    }
                )
        for metric in PRIMARY_METRICS:
            interval = evidence.interval(group_id, metric)
            if train_domain == "inner":
                source_domain, target_domain, sign = "inner", "outer", 1.0
            else:
                source_domain, target_domain, sign = "outer", "inner", -1.0
            low = float(interval["gap_ci_low"])
            high = float(interval["gap_ci_high"])
            gaps.append(
                {
                    "logical_group_id": group_id,
                    "representation": representation,
                    "model": model,
                    "transfer_direction": f"{source_domain}_to_{target_domain}",
                    "metric": metric,
                    "source_estimate": evidence.metric(group_id, source_domain, metric),
                    "target_estimate": evidence.metric(group_id, target_domain, metric),
                    "delta_estimate": sign * float(interval["gap_estimate"]),
                    "delta_ci_low": low if sign > 0 else -high,
                    "delta_ci_high": high if sign > 0 else -low,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(gaps)


PAIRWISE_COMPARISONS = (
    (
        "dann_minus_plain_cnn1d",
        "dann__sequential_splt__dann_cnn1d",
        "primary__sequential_splt__cnn1d__train_inner",
        "outer",
    ),
    (
        "dann_minus_supervised_outer_cnn1d",
        "dann__sequential_splt__dann_cnn1d",
        "primary__sequential_splt__cnn1d__train_outer",
        "outer",
    ),
    (
        "cnn1d_minus_lstm",
        "primary__sequential_splt__cnn1d__train_inner",
        "primary__sequential_splt__lstm__train_inner",
        "outer",
    ),
    (
        "cnn1d_minus_transformer",
        "primary__sequential_splt__cnn1d__train_inner",
        "primary__sequential_splt__transformer__train_inner",
        "outer",
    ),
    (
        "xgboost_matched_minus_flattened",
        "primary__matched_flow_stats__xgboost__train_inner",
        "primary__flattened_splt__xgboost__train_inner",
        "outer",
    ),
    (
        "cnn1d_minus_xgboost_matched",
        "primary__sequential_splt__cnn1d__train_inner",
        "primary__matched_flow_stats__xgboost__train_inner",
        "outer",
    ),
    (
        "xgboost_prefix_minus_full",
        "primary__prefix_stats__xgboost__train_inner",
        "primary__matched_flow_stats__xgboost__train_inner",
        "outer",
    ),
    (
        "cnn1d_size_timing_minus_all",
        "ablation_channels__size_timing__cnn1d",
        "ablation_channels__all__cnn1d",
        "outer",
    ),
)


def _pairwise_table(evidence: _Evidence, config: PaperAnalysisConfig) -> pd.DataFrame:
    rows = []
    for comparison_id, left_id, right_id, domain in PAIRWISE_COMPARISONS:
        intervals = paired_method_intervals(
            evidence.frame(left_id),
            evidence.frame(right_id),
            evidence.classes,
            domain=domain,
            replicates=int(config.bootstrap["replicates"]),
            confidence_level=float(config.bootstrap["confidence_level"]),
            seed=int(config.bootstrap["seed"]),
        )
        for row in intervals:
            rows.append(
                {
                    "comparison_id": comparison_id,
                    "left_group_id": left_id,
                    "right_group_id": right_id,
                    **row,
                }
            )
    return pd.DataFrame(rows)


def _cross_session_table(evidence: _Evidence, config: PaperAnalysisConfig) -> pd.DataFrame:
    rows = []
    ids = sorted(group_id for group_id in evidence._paths if group_id.startswith("cross_session__"))
    for group_id in ids:
        _, representation, model = group_id.split("__")
        frame = evidence.frame(group_id)
        for train_session, test_session in ((1, 2), (2, 1)):
            selected = frame.loc[
                frame["train_session"].eq(train_session) & frame["test_session"].eq(test_session)
            ].copy()
            intervals = paired_bootstrap_intervals(
                selected,
                evidence.classes,
                metrics=PRIMARY_METRICS,
                replicates=int(config.bootstrap["replicates"]),
                confidence_level=float(config.bootstrap["confidence_level"]),
                seed=int(config.bootstrap["seed"]),
            )
            for interval in intervals:
                rows.append(
                    {
                        "logical_group_id": group_id,
                        "representation": representation,
                        "model": model,
                        "train_session": train_session,
                        "test_session": test_session,
                        **interval,
                    }
                )
    return pd.DataFrame(rows)


def _dann_table(evidence: _Evidence, pairwise: pd.DataFrame) -> pd.DataFrame:
    configurations = (
        ("plain_inner_trained", "primary__sequential_splt__cnn1d__train_inner"),
        ("dann", "dann__sequential_splt__dann_cnn1d"),
        ("supervised_outer_trained", "primary__sequential_splt__cnn1d__train_outer"),
    )
    rows = []
    for configuration, group_id in configurations:
        for metric in PRIMARY_METRICS:
            interval = evidence.interval(group_id, metric)
            rows.append(
                {
                    "row_type": "estimate",
                    "configuration": configuration,
                    "comparison_id": "",
                    "metric": metric,
                    "estimate": float(interval["outer_estimate"]),
                    "ci_low": float(interval["outer_ci_low"]),
                    "ci_high": float(interval["outer_ci_high"]),
                }
            )
    for comparison_id in (
        "dann_minus_plain_cnn1d",
        "dann_minus_supervised_outer_cnn1d",
    ):
        for row in pairwise.loc[pairwise["comparison_id"].eq(comparison_id)].itertuples(
            index=False
        ):
            rows.append(
                {
                    "row_type": "difference",
                    "configuration": "",
                    "comparison_id": comparison_id,
                    "metric": row.metric,
                    "estimate": row.delta_estimate,
                    "ci_low": row.delta_ci_low,
                    "ci_high": row.delta_ci_high,
                }
            )
    return pd.DataFrame(rows)


def _ablation_tables(evidence: _Evidence) -> tuple[pd.DataFrame, pd.DataFrame]:
    prefix_rows = []
    channel_rows = []
    for group_id in sorted(evidence._paths):
        if group_id.startswith("ablation_prefix__"):
            _, condition, model = group_id.split("__")
            prefix_length = int(condition.removeprefix("n"))
            target = prefix_rows
            condition_values = {"prefix_length": prefix_length}
        elif group_id.startswith("ablation_channels__"):
            _, channels, model = group_id.split("__")
            target = channel_rows
            condition_values = {"channels": channels}
        else:
            continue
        for metric in PRIMARY_METRICS:
            interval = evidence.interval(group_id, metric)
            for domain in ("inner", "outer"):
                target.append(
                    {
                        "logical_group_id": group_id,
                        "model": model,
                        **condition_values,
                        "test_domain": domain,
                        "metric": metric,
                        "estimate": float(interval[f"{domain}_estimate"]),
                        "ci_low": float(interval[f"{domain}_ci_low"]),
                        "ci_high": float(interval[f"{domain}_ci_high"]),
                        "seed_policy": "seed_42_per_fold",
                    }
                )
    return pd.DataFrame(prefix_rows), pd.DataFrame(channel_rows)


def _per_class_table(evidence: _Evidence) -> pd.DataFrame:
    rows = []
    for group_id in sorted(evidence._paths):
        frame = evidence.frame(group_id)
        if group_id.startswith("cross_session__"):
            partitions = (
                ("s1_to_s2", frame.loc[frame["train_session"].eq(1)]),
                ("s2_to_s1", frame.loc[frame["train_session"].eq(2)]),
            )
        else:
            partitions = (("combined", frame),)
        for partition, selected in partitions:
            for domain in ("inner", "outer"):
                metrics = per_class_metrics(selected, evidence.classes, domain=domain)
                metrics.insert(0, "test_domain", domain)
                metrics.insert(0, "partition", partition)
                metrics.insert(0, "logical_group_id", group_id)
                rows.extend(metrics.to_dict("records"))
    return pd.DataFrame(rows)


SELECTED_CONFUSION_GROUPS = (
    "primary__sequential_splt__cnn1d__train_inner",
    "primary__matched_flow_stats__xgboost__train_inner",
    "primary__flattened_splt__xgboost__train_inner",
)


def _selected_confusion_table_from_frames(
    frames: dict[str, pd.DataFrame], classes: tuple[str, ...]
) -> pd.DataFrame:
    rows = []
    for group_id in SELECTED_CONFUSION_GROUPS:
        if group_id not in frames:
            raise PipelineInvariantError(f"Missing selected confusion group: {group_id}")
        frame = frames[group_id]
        selected = frame.loc[frame["test_domain"].eq("outer")]
        if len(selected) != 226_281 or selected["pair_id"].duplicated().any():
            raise PipelineInvariantError(
                "Selected confusion predictions do not cover each pair once"
            )
        matrix = pd.crosstab(selected["true_label"], selected["prediction"]).reindex(
            index=classes, columns=classes, fill_value=0
        )
        for true_label in classes:
            support = int(matrix.loc[true_label].sum())
            if support <= 0:
                raise PipelineInvariantError("Selected confusion class has zero support")
            for predicted_label in classes:
                count = int(matrix.loc[true_label, predicted_label])
                rows.append(
                    {
                        "logical_group_id": group_id,
                        "true_label": true_label,
                        "predicted_label": predicted_label,
                        "support": support,
                        "count": count,
                        "row_fraction": count / support,
                    }
                )
    return pd.DataFrame(rows)


def _selected_confusion_table(evidence: _Evidence) -> pd.DataFrame:
    return _selected_confusion_table_from_frames(
        {group_id: evidence.frame(group_id) for group_id in SELECTED_CONFUSION_GROUPS},
        evidence.classes,
    )


def _claim_ledger() -> pd.DataFrame:
    rows = (
        (
            "C01",
            "design",
            "Primary folds are disjoint by physical pair_id.",
            "source analysis contract",
            "strong",
        ),
        (
            "C02",
            "design",
            "Each trained model is evaluated on both views of the same held-out pairs.",
            "primary_2x2_metrics.csv",
            "strong",
        ),
        (
            "C03",
            "result",
            "Transfer robustness depends jointly on representation and model family.",
            "primary_transfer_gaps.csv",
            "empirical",
        ),
        (
            "C04",
            "result",
            "CNN1D has the smallest absolute forward-transfer loss among evaluated configurations.",
            "primary_transfer_gaps.csv",
            "empirical",
        ),
        (
            "C05",
            "result",
            "Flattened SPLT with classical models transfers less robustly than "
            "the evaluated sequence-aware configurations.",
            "primary_transfer_gaps.csv",
            "empirical",
        ),
        (
            "C06",
            "result",
            "Cross-session performance differs by session direction.",
            "cross_session_by_direction.csv",
            "empirical",
        ),
        (
            "C07",
            "result",
            "DANN modestly improves outer-view CNN1D performance over source-only training.",
            "dann_comparison.csv",
            "empirical",
        ),
        (
            "C08",
            "result",
            "DANN remains below the supervised outer-trained CNN1D reference.",
            "dann_comparison.csv",
            "empirical",
        ),
        (
            "C09",
            "result",
            "Short packet prefixes retain substantial transferable signal.",
            "prefix_ablation.csv",
            "cautious",
        ),
        (
            "C10",
            "result",
            "Size and timing retain strong signal without direction in this benchmark.",
            "channel_ablation.csv",
            "cautious",
        ),
        (
            "C11",
            "scope",
            "Ablations use one frozen seed per fold and are sensitivity analyses.",
            "prefix_ablation.csv; channel_ablation.csv",
            "strong",
        ),
        (
            "C12",
            "scope",
            "Results are bounded to this WireGuard dataset, site, endpoint, and two sessions.",
            "dataset publication; paper limitations",
            "strong",
        ),
    )
    return pd.DataFrame(
        rows, columns=("claim_id", "claim_type", "approved_claim", "evidence", "language")
    )


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, float_format="%.15g")


def _fmt_ci(estimate: float, low: float, high: float) -> str:
    return f"{estimate:.3f} [{low:.3f}, {high:.3f}]"


def _latex_tables(
    output: Path,
    primary: pd.DataFrame,
    cross_session: pd.DataFrame,
    dann: pd.DataFrame,
) -> None:
    tables = output / "tables"
    tables.mkdir()
    forward = primary.loc[primary["train_domain"].eq("inner")]
    lines = [
        "% Generated; do not edit.",
        "\\begin{tabular}{llcc}",
        "\\toprule",
        "Representation & Model & Bal. Acc. & Macro F1 \\\\",
        "\\midrule",
    ]
    for (representation, model), group in forward.groupby(["representation", "model"], sort=True):
        values = {}
        for row in group.loc[group["test_domain"].eq("outer")].itertuples(index=False):
            values[row.metric] = _fmt_ci(row.estimate, row.ci_low, row.ci_high)
        line = (
            f"{REPRESENTATION_LABELS[representation]} & {MODEL_LABELS[model]} & "
            f"{values['balanced_accuracy']} & {values['macro_f1']} \\\\"
        )
        lines.append(line)
    lines += ["\\bottomrule", "\\end{tabular}"]
    (tables / "primary_forward.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    lines = [
        "% Generated; do not edit.",
        "\\begin{tabular}{llcccc}",
        "\\toprule",
        "Representation & Model & I$\\to$I & I$\\to$O & O$\\to$I & O$\\to$O \\\\",
        "\\midrule",
    ]
    for (representation, model), group in primary.loc[primary["metric"].eq("macro_f1")].groupby(
        ["representation", "model"], sort=True
    ):
        values = {
            (row.train_domain, row.test_domain): row.estimate
            for row in group.itertuples(index=False)
        }
        line = (
            f"{REPRESENTATION_LABELS[representation]} & {MODEL_LABELS[model]} & "
            f"{values[('inner', 'inner')]:.3f} & {values[('inner', 'outer')]:.3f} & "
            f"{values[('outer', 'inner')]:.3f} & {values[('outer', 'outer')]:.3f} \\\\"
        )
        lines.append(line)
    lines += ["\\bottomrule", "\\end{tabular}"]
    (tables / "primary_2x2.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    lines = [
        "% Generated; do not edit.",
        "\\begin{tabular}{llcccc}",
        "\\toprule",
        "Representation & Model & S1$\\to$S2 I & S1$\\to$S2 O & S2$\\to$S1 I & S2$\\to$S1 O \\\\",
        "\\midrule",
    ]
    subset = cross_session.loc[cross_session["metric"].eq("macro_f1")]
    for (representation, model), group in subset.groupby(["representation", "model"], sort=True):
        values = {}
        for row in group.itertuples(index=False):
            values[(row.train_session, "inner")] = row.inner_estimate
            values[(row.train_session, "outer")] = row.outer_estimate
        line = (
            f"{REPRESENTATION_LABELS[representation]} & {MODEL_LABELS[model]} & "
            f"{values[(1, 'inner')]:.3f} & {values[(1, 'outer')]:.3f} & "
            f"{values[(2, 'inner')]:.3f} & {values[(2, 'outer')]:.3f} \\\\"
        )
        lines.append(line)
    lines += ["\\bottomrule", "\\end{tabular}"]
    (tables / "cross_session.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    lines = [
        "% Generated; do not edit.",
        "\\begin{tabular}{lcc}",
        "\\toprule",
        "Configuration & Bal. Acc. & Macro F1 \\\\",
        "\\midrule",
    ]
    labels = {
        "plain_inner_trained": "Source-only CNN1D",
        "dann": "DANN-CNN1D",
        "supervised_outer_trained": "Supervised outer CNN1D",
    }
    for configuration in labels:
        group = dann.loc[dann["row_type"].eq("estimate") & dann["configuration"].eq(configuration)]
        values = {
            row.metric: _fmt_ci(row.estimate, row.ci_low, row.ci_high)
            for row in group.itertuples(index=False)
        }
        lines.append(
            f"{labels[configuration]} & {values['balanced_accuracy']} & {values['macro_f1']} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    (tables / "dann_comparison.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _latex_macros(output: Path, evidence: _Evidence, pairwise: pd.DataFrame) -> None:
    latex = output / "latex"
    latex.mkdir()
    lookup = {(row.comparison_id, row.metric): row for row in pairwise.itertuples(index=False)}
    plain = "primary__sequential_splt__cnn1d__train_inner"
    dann = "dann__sequential_splt__dann_cnn1d"
    flattened_xgb = "primary__flattened_splt__xgboost__train_inner"
    dann_gain = lookup[("dann_minus_plain_cnn1d", "macro_f1")]
    values = {
        "CampaignRevision": evidence.config.expected_campaign_revision[:7],
        "MatchedPairCount": "226,281",
        "XGBFlatInnerMacroFOne": f"{evidence.metric(flattened_xgb, 'inner', 'macro_f1'):.3f}",
        "XGBFlatOuterMacroFOne": f"{evidence.metric(flattened_xgb, 'outer', 'macro_f1'):.3f}",
        "CNNOuterBalancedAccuracy": f"{evidence.metric(plain, 'outer', 'balanced_accuracy'):.3f}",
        "CNNOuterMacroFOne": f"{evidence.metric(plain, 'outer', 'macro_f1'):.3f}",
        "DANNOuterBalancedAccuracy": f"{evidence.metric(dann, 'outer', 'balanced_accuracy'):.3f}",
        "DANNOuterMacroFOne": f"{evidence.metric(dann, 'outer', 'macro_f1'):.3f}",
        "DANNMacroFOneGain": f"{dann_gain.delta_estimate:.3f}",
        "DANNMacroFOneGainLow": f"{dann_gain.delta_ci_low:.3f}",
        "DANNMacroFOneGainHigh": f"{dann_gain.delta_ci_high:.3f}",
    }
    lines = ["% Generated; do not edit."] + [
        f"\\newcommand{{\\{name}}}{{{value}}}" for name, value in values.items()
    ]
    (latex / "results_macros.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _paper_figures(
    output: Path,
    primary_gaps: pd.DataFrame,
    cross_session: pd.DataFrame,
    prefix: pd.DataFrame,
    channels: pd.DataFrame,
    per_class: pd.DataFrame,
    seed_dispersion: pd.DataFrame,
    selected_confusions: pd.DataFrame,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
        }
    )
    figures = output / "figures"
    figures.mkdir()
    metadata = {
        "Creator": "vpncat paper analysis",
        "Producer": "vpncat",
        "CreationDate": None,
        "ModDate": None,
    }

    def save(fig: Any, name: str, *, tight: bool = True) -> None:
        if tight:
            fig.tight_layout()
        fig.savefig(figures / f"{name}.pdf", bbox_inches="tight", metadata=metadata)
        plt.close(fig)

    forward = primary_gaps.loc[
        primary_gaps["transfer_direction"].eq("inner_to_outer")
        & primary_gaps["metric"].eq("macro_f1")
    ].copy()
    forward["label"] = forward.apply(
        lambda row: f"{MODEL_LABELS[row.model]} / {REPRESENTATION_LABELS[row.representation]}",
        axis=1,
    )
    forward = forward.sort_values("delta_estimate")
    fig, ax = plt.subplots(figsize=(6.8, 3.3))
    lower_error = forward["delta_estimate"] - forward["delta_ci_low"]
    upper_error = forward["delta_ci_high"] - forward["delta_estimate"]
    ax.barh(
        forward["label"],
        forward["delta_estimate"],
        color="#176B87",
        xerr=np.vstack((lower_error, upper_error)),
        error_kw={
            "ecolor": "#111111",
            "elinewidth": 0.8,
            "capsize": 2.0,
            "capthick": 0.8,
        },
    )
    ax.axvline(0, color="black", linewidth=0.7)
    ax.set_xlabel("Outer minus inner macro F1")
    save(fig, "transfer_gaps")

    direction_rows = primary_gaps.loc[primary_gaps["metric"].eq("macro_f1")].copy()
    direction_rows["label"] = direction_rows.apply(
        lambda row: f"{MODEL_LABELS[row.model]} / {REPRESENTATION_LABELS[row.representation]}",
        axis=1,
    )
    labels = (
        direction_rows.loc[direction_rows["transfer_direction"].eq("inner_to_outer")]
        .sort_values("delta_estimate")["label"]
        .tolist()
    )
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    for offset, direction, color, marker, legend in (
        (-0.13, "inner_to_outer", "#176B87", "o", "Inner to outer"),
        (0.13, "outer_to_inner", "#D95F02", "s", "Outer to inner"),
    ):
        values = direction_rows.loc[
            direction_rows["transfer_direction"].eq(direction)
        ].set_index("label").loc[labels]
        estimate = values["delta_estimate"].to_numpy()
        errors = np.vstack(
            (
                estimate - values["delta_ci_low"].to_numpy(),
                values["delta_ci_high"].to_numpy() - estimate,
            )
        )
        ax.errorbar(
            estimate,
            y + offset,
            xerr=errors,
            fmt=marker,
            color=color,
            capsize=2.0,
            elinewidth=0.8,
            markersize=4.0,
            label=legend,
        )
    ax.axvline(0, color="black", linewidth=0.7)
    ax.set_yticks(y, labels)
    ax.set_xlabel("Target-view minus source-view macro F1")
    ax.legend(
        frameon=False,
        ncols=2,
        loc="lower right",
        bbox_to_anchor=(1.0, 1.01),
        borderaxespad=0,
    )
    save(fig, "transfer_direction_asymmetry")

    subset = cross_session.loc[cross_session["metric"].eq("macro_f1")].copy()
    subset["label"] = subset.apply(
        lambda row: f"{MODEL_LABELS[row.model]} / {REPRESENTATION_LABELS[row.representation]}",
        axis=1,
    )
    labels = sorted(subset["label"].unique())
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7.2, 3.7))
    for offset, train_session, color in ((-0.16, 1, "#176B87"), (0.16, 2, "#D95F02")):
        values = (
            subset.loc[subset["train_session"].eq(train_session)].set_index("label").loc[labels]
        )
        ax.barh(
            y + offset,
            values["outer_estimate"],
            height=0.28,
            color=color,
            label=f"S{train_session} to S{3 - train_session}",
        )
    ax.set_yticks(y, labels)
    ax.set_xlabel("Outer-view macro F1")
    ax.legend(
        frameon=False,
        ncols=2,
        loc="lower right",
        bbox_to_anchor=(1.0, 1.01),
        borderaxespad=0,
    )
    save(fig, "cross_session_directions")

    fig, ax = plt.subplots(figsize=(5.8, 3.2))
    selected = prefix.loc[prefix["test_domain"].eq("outer") & prefix["metric"].eq("macro_f1")]
    for model, color, marker in (("cnn1d", "#176B87", "o"), ("transformer", "#D95F02", "s")):
        values = selected.loc[selected["model"].eq(model)].sort_values("prefix_length")
        ax.plot(
            values["prefix_length"],
            values["estimate"],
            marker=marker,
            color=color,
            label=MODEL_LABELS[model],
        )
    ax.set_xlabel("Packet prefix length")
    ax.set_ylabel("Outer-view macro F1")
    ax.legend(frameon=False)
    save(fig, "prefix_ablation")

    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    selected = channels.loc[channels["test_domain"].eq("outer") & channels["metric"].eq("macro_f1")]
    order = ["direction", "direction_size", "direction_timing", "size_timing", "all"]
    x = np.arange(len(order))
    for offset, model, color in ((-0.18, "cnn1d", "#176B87"), (0.18, "transformer", "#D95F02")):
        values = selected.loc[selected["model"].eq(model)].set_index("channels").loc[order]
        ax.bar(x + offset, values["estimate"], width=0.34, color=color, label=MODEL_LABELS[model])
    ax.set_xticks(x, [value.replace("_", "+") for value in order], rotation=20, ha="right")
    ax.set_ylabel("Outer-view macro F1")
    ax.legend(frameon=False)
    save(fig, "channel_ablation")

    key = per_class.loc[
        per_class["logical_group_id"].eq("primary__sequential_splt__cnn1d__train_inner")
        & per_class["partition"].eq("combined")
    ]
    inner = key.loc[key["test_domain"].eq("inner")].set_index("class_name")
    outer = key.loc[key["test_domain"].eq("outer")].set_index("class_name")
    classes = inner.sort_values("support", ascending=True).index
    y = np.arange(len(classes))
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(inner.loc[classes, "f1"], y, "o", color="#176B87", label="Inner")
    ax.plot(outer.loc[classes, "f1"], y, "s", color="#D95F02", label="Outer")
    for index, class_name in enumerate(classes):
        ax.plot(
            [inner.loc[class_name, "f1"], outer.loc[class_name, "f1"]],
            [index, index],
            color="#BBBBBB",
            linewidth=0.8,
            zorder=0,
        )
    ax.set_yticks(y, classes)
    ax.set_xlabel("Per-class F1")
    ax.legend(frameon=False)
    save(fig, "per_class_f1")

    support = inner.loc[classes, "support"]
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.barh(classes, support, color="#176B87")
    ax.set_xscale("log")
    ax.set_xlabel("Held-out flow pairs (log scale)")
    save(fig, "class_distribution")

    diagnostic_groups = (
        ("CNN1D / Sequential-SPLT-50", "primary__sequential_splt__cnn1d__train_inner"),
        ("LSTM / Sequential-SPLT-50", "primary__sequential_splt__lstm__train_inner"),
        (
            "Transformer / Sequential-SPLT-50",
            "primary__sequential_splt__transformer__train_inner",
        ),
        ("XGBoost / MatchedFlowStats", "primary__matched_flow_stats__xgboost__train_inner"),
        ("XGBoost / PrefixStats-50", "primary__prefix_stats__xgboost__train_inner"),
        ("XGBoost / Flattened-SPLT-50", "primary__flattened_splt__xgboost__train_inner"),
    )
    class_order = inner.sort_values("support", ascending=False).index.tolist()
    outer_rows = []
    change_rows = []
    row_labels = []
    for label, group_id in diagnostic_groups:
        selected = per_class.loc[
            per_class["logical_group_id"].eq(group_id)
            & per_class["partition"].eq("combined")
        ]
        selected_inner = selected.loc[selected["test_domain"].eq("inner")].set_index(
            "class_name"
        )
        selected_outer = selected.loc[selected["test_domain"].eq("outer")].set_index(
            "class_name"
        )
        outer_rows.append(selected_outer.loc[class_order, "f1"].to_numpy())
        change_rows.append(
            (
                selected_outer.loc[class_order, "f1"]
                - selected_inner.loc[class_order, "f1"]
            ).to_numpy()
        )
        row_labels.append(label)

    fig, ax = plt.subplots(figsize=(9.0, 3.5))
    image = ax.imshow(np.asarray(outer_rows), aspect="auto", vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(np.arange(len(class_order)), class_order, rotation=40, ha="right")
    ax.set_yticks(np.arange(len(row_labels)), row_labels)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    colorbar.set_label("Outer-view F1")
    save(fig, "per_class_model_heatmap")

    changes = np.asarray(change_rows)
    limit = max(0.05, float(np.abs(changes).max()))
    fig, ax = plt.subplots(figsize=(9.0, 3.5))
    image = ax.imshow(
        changes,
        aspect="auto",
        vmin=-limit,
        vmax=limit,
        cmap="RdBu",
    )
    ax.set_xticks(np.arange(len(class_order)), class_order, rotation=40, ha="right")
    ax.set_yticks(np.arange(len(row_labels)), row_labels)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    colorbar.set_label("Outer minus inner per-class F1")
    save(fig, "per_class_transfer_change")

    seed_groups = (
        ("CNN1D", "primary__sequential_splt__cnn1d__train_inner"),
        ("LSTM", "primary__sequential_splt__lstm__train_inner"),
        ("Transformer", "primary__sequential_splt__transformer__train_inner"),
        ("DANN-CNN1D", "dann__sequential_splt__dann_cnn1d"),
    )
    selected = seed_dispersion.loc[
        seed_dispersion["test_domain"].eq("outer")
        & seed_dispersion["metric"].eq("macro_f1")
    ].set_index("physical_group_id")
    labels = [label for label, _ in seed_groups]
    values = selected.loc[[group_id for _, group_id in seed_groups]]
    means = values["mean"].to_numpy()
    ranges = np.vstack(
        (
            means - values["minimum"].to_numpy(),
            values["maximum"].to_numpy() - means,
        )
    )
    fig, ax = plt.subplots(figsize=(5.8, 2.8))
    ax.errorbar(
        means,
        np.arange(len(labels)),
        xerr=ranges,
        fmt="o",
        color="#176B87",
        capsize=3.0,
        elinewidth=1.0,
    )
    ax.set_yticks(np.arange(len(labels)), labels)
    ax.set_xlabel("Per-seed outer-view macro F1 (mean and range)")
    save(fig, "seed_dispersion")

    from matplotlib.colors import LogNorm

    confusion_labels = {
        "primary__sequential_splt__cnn1d__train_inner": "CNN1D / Sequential-SPLT-50",
        "primary__matched_flow_stats__xgboost__train_inner": "XGBoost / MatchedFlowStats",
        "primary__flattened_splt__xgboost__train_inner": "XGBoost / Flattened-SPLT-50",
    }
    fig, axes = plt.subplots(
        1, 3, figsize=(12.0, 4.1), sharex=True, sharey=True, constrained_layout=True
    )
    image = None
    for ax, group_id in zip(axes, SELECTED_CONFUSION_GROUPS, strict=True):
        selected = selected_confusions.loc[
            selected_confusions["logical_group_id"].eq(group_id)
        ]
        matrix = selected.pivot(
            index="true_label", columns="predicted_label", values="row_fraction"
        ).loc[class_order, class_order]
        values = matrix.to_numpy()
        masked = np.ma.masked_where(values <= 0, values)
        image = ax.imshow(masked, cmap="Blues", norm=LogNorm(vmin=1e-3, vmax=1.0))
        ax.set_title(confusion_labels[group_id])
        ax.set_xticks(np.arange(len(class_order)), class_order, rotation=55, ha="right")
        ax.set_yticks(np.arange(len(class_order)), class_order)
        ax.set_xlabel("Predicted category")
    axes[0].set_ylabel("True category")
    assert image is not None
    colorbar = fig.colorbar(image, ax=axes, fraction=0.02, pad=0.02)
    colorbar.set_label("Fraction of true category (log scale)")
    save(fig, "selected_confusions", tight=False)


def _materialize_bundle(
    config: PaperAnalysisConfig,
    output: Path,
    *,
    source_already_validated: bool = False,
) -> dict[str, Any]:
    source_manifest = (
        _load_json(config.source_analysis_root / "analysis.json")
        if source_already_validated
        else _validate_source(config)
    )
    evidence = _Evidence(config)
    primary, gaps = _primary_tables(evidence)
    pairwise = _pairwise_table(evidence, config)
    cross_session = _cross_session_table(evidence, config)
    dann = _dann_table(evidence, pairwise)
    prefix, channels = _ablation_tables(evidence)
    per_class = _per_class_table(evidence)
    outputs = {
        "claim_ledger.csv": _claim_ledger(),
        "primary_2x2_metrics.csv": primary,
        "primary_transfer_gaps.csv": gaps,
        "pairwise_method_differences.csv": pairwise,
        "cross_session_by_direction.csv": cross_session,
        "dann_comparison.csv": dann,
        "prefix_ablation.csv": prefix,
        "channel_ablation.csv": channels,
        "per_class_metrics.csv": per_class,
        "seed_dispersion.csv": pd.read_csv(config.source_analysis_root / "seed_dispersion.csv"),
        "selected_confusions.csv": _selected_confusion_table(evidence),
    }
    for name, frame in outputs.items():
        _write_csv(frame, output / name)
    _latex_tables(output, primary, cross_session, dann)
    _latex_macros(output, evidence, pairwise)
    _paper_figures(
        output,
        gaps,
        cross_session,
        prefix,
        channels,
        per_class,
        outputs["seed_dispersion.csv"],
        outputs["selected_confusions.csv"],
    )
    return {
        "source_manifest": source_manifest,
        "counts": {name: len(frame) for name, frame in outputs.items()},
        "classes": list(evidence.classes),
    }


def build_paper_analysis(config: PaperAnalysisConfig, *, allow_dirty: bool = False) -> Path:
    if config.output_root.exists():
        raise FileExistsError(f"Refusing to overwrite paper analysis: {config.output_root}")
    provenance = _generation_provenance(config.project_root, allow_dirty=allow_dirty)
    config.output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{config.output_root.name}.staging-", dir=config.output_root.parent
        )
    )
    try:
        result = _materialize_bundle(config, staging)
        files = sorted(path for path in staging.rglob("*") if path.is_file())
        manifest = {
            "schema_version": 1,
            "status": "complete",
            "created_utc": datetime.now(UTC).isoformat(),
            "package_version": __version__,
            "generation_git": provenance,
            "generation_environment": _generation_environment(),
            "campaign_revision": config.expected_campaign_revision,
            "campaign_archive_sha256": config.expected_campaign_archive_sha256,
            "source_analysis_manifest_sha256": config.expected_analysis_manifest_sha256,
            "analysis_contract_sha256": config.expected_analysis_contract_sha256,
            "bootstrap": config.bootstrap,
            "class_order": result["classes"],
            "counts": result["counts"],
            "source_prediction_hashes": {
                relative: digest
                for relative, digest in result["source_manifest"]["artifacts"].items()
                if relative.startswith("predictions/")
            },
            "artifacts": {
                path.relative_to(staging).as_posix(): sha256_file(path) for path in files
            },
        }
        (staging / "analysis_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        validate_paper_analysis(
            staging,
            config=config,
            recompute=False,
            source_already_validated=True,
            allow_dirty_generation=allow_dirty,
        )
        staging.replace(config.output_root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return config.output_root


def validate_paper_analysis(
    output: Path,
    *,
    config: PaperAnalysisConfig,
    recompute: bool = True,
    source_already_validated: bool = False,
    allow_dirty_generation: bool = False,
) -> dict[str, Any]:
    manifest = _load_json(output / "analysis_manifest.json")
    generation_git = manifest.get("generation_git", {})
    if (
        manifest.get("schema_version") != 1
        or manifest.get("status") != "complete"
        or manifest.get("package_version") != __version__
        or manifest.get("campaign_revision") != config.expected_campaign_revision
        or manifest.get("campaign_archive_sha256") != config.expected_campaign_archive_sha256
        or manifest.get("source_analysis_manifest_sha256")
        != config.expected_analysis_manifest_sha256
        or manifest.get("analysis_contract_sha256") != config.expected_analysis_contract_sha256
        or manifest.get("bootstrap") != config.bootstrap
        or manifest.get("counts") != EXPECTED_COUNTS
        or not generation_git.get("status_available")
        or generation_git.get("revision") in {None, "", "UNBORN"}
        or (generation_git.get("dirty") and not allow_dirty_generation)
        or manifest.get("generation_environment") != _generation_environment()
    ):
        raise PipelineInvariantError("Paper-analysis manifest identity or counts differ")
    expected_files = set(manifest.get("artifacts", {}))
    physical_files = {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file() and path.name != "analysis_manifest.json"
    }
    if physical_files != expected_files:
        raise PipelineInvariantError("Paper-analysis artifact inventory differs")
    for relative, digest in manifest["artifacts"].items():
        if sha256_file(output / relative) != digest:
            raise PipelineInvariantError(f"Paper-analysis artifact hash differs: {relative}")
    if not source_already_validated:
        _validate_source(config)
    source_manifest = _load_json(config.source_analysis_root / "analysis.json")
    expected_prediction_hashes = {
        relative: digest
        for relative, digest in source_manifest["artifacts"].items()
        if relative.startswith("predictions/")
    }
    if manifest.get("source_prediction_hashes") != expected_prediction_hashes:
        raise PipelineInvariantError("Paper-analysis source prediction hashes differ")
    if recompute:
        with tempfile.TemporaryDirectory(prefix="vpncat-paper-validate-") as temporary:
            rebuilt = Path(temporary)
            result = _materialize_bundle(config, rebuilt, source_already_validated=True)
            if result["counts"] != EXPECTED_COUNTS:
                raise PipelineInvariantError("Recomputed paper-analysis counts differ")
            for relative in sorted(expected_files):
                if sha256_file(output / relative) != sha256_file(rebuilt / relative):
                    raise PipelineInvariantError(
                        f"Paper-analysis content differs from predictions: {relative}"
                    )
    return {"status": "valid", "counts": EXPECTED_COUNTS}
