from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from vpncat.errors import PipelineInvariantError
from vpncat.hashing import sha256_file

ENCAPSULATION_DIAGNOSTICS = "encapsulation_diagnostics.csv"
PER_CLASS_DIAGNOSTICS = "per_class_diagnostics.csv"
SEED_PAIR_DIAGNOSTICS = "seed_pair_diagnostics.csv"
SEED_METRICS_SOURCE = "seed_metrics.csv"
DIAGNOSTICS_EXTENSION_RECEIPT = "diagnostics_extension_receipt.json"
DIAGNOSTIC_COUNTS = {
    ENCAPSULATION_DIAGNOSTICS: 26,
    PER_CLASS_DIAGNOSTICS: 23,
    SEED_PAIR_DIAGNOSTICS: 3,
}
EXPECTED_CANONICAL_SHA256 = "ca51e8447717224737a86aa24fc5ac889f39b3cb404fbc89fdec58db8cd83e1e"
EXPECTED_SEED_METRICS_SHA256 = "d3b868cc39b3068b59ec0a60aabce2a75346dd3aca0febaa2599fb3e195dbd9f"

PLAIN_CNN1D = "primary__sequential_splt__cnn1d__train_inner"
DANN_CNN1D = "dann__sequential_splt__dann_cnn1d"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _rank_correlation(left: pd.Series, right: pd.Series) -> float:
    ranked_left = left.rank(method="average").to_numpy(dtype=float)
    ranked_right = right.rank(method="average").to_numpy(dtype=float)
    return float(np.corrcoef(ranked_left, ranked_right)[0, 1])


def build_encapsulation_diagnostics(
    canonical_path: Path, dataset_manifest_path: Path
) -> pd.DataFrame:
    manifest = _load_json(dataset_manifest_path)
    canonical_artifact = manifest.get("artifacts", {}).get("canonical_pairs", {})
    if (
        canonical_artifact.get("sha256") != EXPECTED_CANONICAL_SHA256
        or sha256_file(canonical_path) != EXPECTED_CANONICAL_SHA256
    ):
        raise PipelineInvariantError("Canonical dataset hash differs from its manifest")

    columns = (
        "inner_packets",
        "outer_packets",
        "inner_bytes",
        "outer_bytes",
        "inner_mean_iat_ms",
        "outer_mean_iat_ms",
    )
    frame = pd.read_parquet(canonical_path, columns=list(columns))
    if (
        len(frame) != manifest.get("counts", {}).get("retained_rows")
        or not frame["inner_packets"].equals(frame["outer_packets"])
        or (frame["inner_packets"] <= 0).any()
    ):
        raise PipelineInvariantError("Canonical matched-pair inventory differs")

    packet_count = frame["inner_packets"].astype(np.int64)
    padding_bytes = frame["outer_bytes"] - frame["inner_bytes"]
    mean_padding = padding_bytes / packet_count
    if (padding_bytes < 0).any() or (mean_padding > 15).any():
        raise PipelineInvariantError("Outer padded sizes violate the WireGuard padding contract")

    rows: list[dict[str, Any]] = []

    def add(
        section: str,
        metric: str,
        *,
        x: float = np.nan,
        value: float,
        count: int = 0,
        denominator: int = 0,
        unit: str,
    ) -> None:
        rows.append(
            {
                "section": section,
                "metric": metric,
                "x": x,
                "value": value,
                "count": count,
                "denominator": denominator,
                "unit": unit,
            }
        )

    total_packets = int(packet_count.sum())
    total_padding = float(padding_bytes.sum())
    add(
        "summary",
        "weighted_mean_padding_bytes",
        value=total_padding / total_packets,
        count=int(round(total_padding)),
        denominator=total_packets,
        unit="bytes_per_packet",
    )
    add(
        "summary",
        "total_padded_byte_increase_fraction",
        value=total_padding / float(frame["inner_bytes"].sum()),
        count=int(round(total_padding)),
        denominator=int(round(float(frame["inner_bytes"].sum()))),
        unit="fraction",
    )

    padding_edges = np.arange(0.0, 17.0, 1.0)
    padding_counts, _ = np.histogram(mean_padding.to_numpy(dtype=float), bins=padding_edges)
    for index, count in enumerate(padding_counts):
        add(
            "padding_distribution",
            "flow_mean_padding_bin",
            x=index + 0.5,
            value=int(count) / len(frame),
            count=int(count),
            denominator=len(frame),
            unit="fraction",
        )

    counts = manifest.get("counts", {})
    eligible_flows = int(counts.get("eligible_rows", 0))
    packet_rows = int(counts.get("packet_match_rows", 0))
    reordered_flows = int(counts.get("outer_reordered_flows", 0))
    reordered_pairs = int(counts.get("outer_reordered_adjacent_pairs", 0))
    adjacent_pairs = packet_rows - eligible_flows
    if min(eligible_flows, packet_rows, adjacent_pairs) <= 0:
        raise PipelineInvariantError("Dataset manifest lacks ordering-audit counts")
    add(
        "ordering",
        "eligible_flows_with_outer_order_inversion",
        value=reordered_flows / eligible_flows,
        count=reordered_flows,
        denominator=eligible_flows,
        unit="fraction",
    )
    add(
        "ordering",
        "adjacent_pairs_with_outer_order_inversion",
        value=reordered_pairs / adjacent_pairs,
        count=reordered_pairs,
        denominator=adjacent_pairs,
        unit="fraction",
    )

    valid_iat = (
        frame["inner_packets"].gt(1)
        & frame["inner_mean_iat_ms"].gt(0)
        & np.isfinite(frame["outer_mean_iat_ms"])
    )
    inner_iat = frame.loc[valid_iat, "inner_mean_iat_ms"].astype(float)
    outer_iat = frame.loc[valid_iat, "outer_mean_iat_ms"].astype(float)
    absolute_change = (outer_iat - inner_iat).abs()
    relative_change = (outer_iat / inner_iat - 1.0).abs()
    add(
        "summary",
        "median_absolute_mean_iat_change_ms",
        value=float(absolute_change.median()),
        count=len(absolute_change),
        denominator=len(frame),
        unit="milliseconds",
    )
    add(
        "summary",
        "p95_absolute_mean_iat_change_ms",
        value=float(absolute_change.quantile(0.95)),
        count=len(absolute_change),
        denominator=len(frame),
        unit="milliseconds",
    )
    for threshold in (0.001, 0.01, 0.05, 0.10):
        within = int(relative_change.le(threshold).sum())
        add(
            "iat_distribution",
            "flow_mean_iat_relative_change_within",
            x=threshold,
            value=within / len(relative_change),
            count=within,
            denominator=len(relative_change),
            unit="fraction",
        )

    result = pd.DataFrame(rows)
    validate_encapsulation_diagnostics(result)
    return result


def _derive_per_class_diagnostics(per_class: pd.DataFrame) -> pd.DataFrame:
    source_groups = sorted(
        group_id
        for group_id in per_class["logical_group_id"].astype(str).unique()
        if group_id.startswith("primary__") and group_id.endswith("__train_inner")
    )
    if len(source_groups) != 9:
        raise PipelineInvariantError("Primary source-only per-class inventory differs")

    rows: list[dict[str, Any]] = []
    for group_id in source_groups:
        selected = per_class.loc[
            per_class["logical_group_id"].eq(group_id)
            & per_class["partition"].eq("combined")
        ]
        inner = selected.loc[selected["test_domain"].eq("inner")].set_index("class_name")
        outer = selected.loc[selected["test_domain"].eq("outer")].set_index("class_name")
        if len(inner) != 14 or not inner.index.equals(outer.index):
            raise PipelineInvariantError("Primary per-class identities differ between views")
        absolute_change = (outer["f1"] - inner["f1"]).abs()
        rows.append(
            {
                "diagnostic": "support_vs_absolute_transfer_change_spearman",
                "logical_group_id": group_id,
                "class_name": np.nan,
                "support": int(inner["support"].sum()),
                "source_f1": np.nan,
                "comparison_f1": np.nan,
                "value": _rank_correlation(inner["support"], absolute_change),
            }
        )

    def outer(group_id: str) -> pd.DataFrame:
        selected = per_class.loc[
            per_class["logical_group_id"].eq(group_id)
            & per_class["partition"].eq("combined")
            & per_class["test_domain"].eq("outer")
        ].set_index("class_name")
        if len(selected) != 14:
            raise PipelineInvariantError(f"Per-class inventory differs: {group_id}")
        return selected

    plain = outer(PLAIN_CNN1D)
    adapted = outer(DANN_CNN1D)
    if not plain.index.equals(adapted.index) or not plain["support"].equals(adapted["support"]):
        raise PipelineInvariantError("DANN and source-only per-class identities differ")
    for class_name in plain.index:
        rows.append(
            {
                "diagnostic": "dann_minus_source_outer_f1",
                "logical_group_id": DANN_CNN1D,
                "class_name": class_name,
                "support": int(plain.loc[class_name, "support"]),
                "source_f1": float(plain.loc[class_name, "f1"]),
                "comparison_f1": float(adapted.loc[class_name, "f1"]),
                "value": float(adapted.loc[class_name, "f1"] - plain.loc[class_name, "f1"]),
            }
        )
    return pd.DataFrame(rows)


def build_per_class_diagnostics(per_class: pd.DataFrame) -> pd.DataFrame:
    result = _derive_per_class_diagnostics(per_class)
    validate_per_class_diagnostics(result, per_class)
    return result


def _derive_seed_pair_diagnostics(seed_metrics: pd.DataFrame) -> pd.DataFrame:
    identity = seed_metrics["physical_group_id"].astype(str)

    def select(group_id: str) -> pd.Series:
        selected = seed_metrics.loc[
            identity.eq(group_id)
            & seed_metrics["test_domain"].eq("outer")
            & seed_metrics["metric"].eq("macro_f1")
        ].set_index("seed")["value"]
        if list(selected.index.sort_values()) != [42, 43, 44]:
            raise PipelineInvariantError(f"Seed-metric inventory differs: {group_id}")
        return selected.sort_index()

    source = select(PLAIN_CNN1D)
    adapted = select(DANN_CNN1D)
    result = pd.DataFrame(
        {
            "seed": source.index.astype(int),
            "source_group_id": PLAIN_CNN1D,
            "adapted_group_id": DANN_CNN1D,
            "test_domain": "outer",
            "metric": "macro_f1",
            "source_estimate": source.to_numpy(dtype=float),
            "adapted_estimate": adapted.to_numpy(dtype=float),
        }
    )
    result["adapted_minus_source"] = (
        result["adapted_estimate"] - result["source_estimate"]
    )
    return result


def build_seed_pair_diagnostics(seed_metrics: pd.DataFrame) -> pd.DataFrame:
    result = _derive_seed_pair_diagnostics(seed_metrics)
    validate_seed_pair_diagnostics(result, seed_metrics)
    return result


def validate_encapsulation_diagnostics(frame: pd.DataFrame) -> None:
    expected_columns = [
        "section",
        "metric",
        "x",
        "value",
        "count",
        "denominator",
        "unit",
    ]
    if list(frame.columns) != expected_columns or len(frame) != DIAGNOSTIC_COUNTS[
        ENCAPSULATION_DIAGNOSTICS
    ]:
        raise PipelineInvariantError("Encapsulation diagnostic schema or count differs")
    padding = frame.loc[frame["section"].eq("padding_distribution")]
    iat = frame.loc[frame["section"].eq("iat_distribution")]
    ordering = frame.loc[frame["section"].eq("ordering")]
    if (
        len(padding) != 16
        or int(padding["count"].sum()) != int(padding["denominator"].iloc[0])
        or not padding["denominator"].nunique() == 1
        or len(iat) != 4
        or not iat.sort_values("x")["value"].is_monotonic_increasing
        or len(ordering) != 2
        or not np.allclose(
            ordering["value"],
            ordering["count"] / ordering["denominator"],
            rtol=0,
            atol=1e-15,
        )
    ):
        raise PipelineInvariantError("Encapsulation diagnostic invariants differ")


def validate_per_class_diagnostics(
    frame: pd.DataFrame, per_class: pd.DataFrame
) -> None:
    expected = _derive_per_class_diagnostics(per_class)
    if len(frame) != DIAGNOSTIC_COUNTS[PER_CLASS_DIAGNOSTICS]:
        raise PipelineInvariantError("Per-class diagnostic row count differs")
    try:
        pd.testing.assert_frame_equal(
            frame.reset_index(drop=True),
            expected.reset_index(drop=True),
            check_exact=False,
            rtol=1e-12,
            atol=1e-15,
        )
    except AssertionError as error:
        raise PipelineInvariantError("Per-class diagnostics differ from source evidence") from error


def validate_seed_pair_diagnostics(
    frame: pd.DataFrame, seed_metrics: pd.DataFrame | None = None
) -> None:
    expected_columns = [
        "seed",
        "source_group_id",
        "adapted_group_id",
        "test_domain",
        "metric",
        "source_estimate",
        "adapted_estimate",
        "adapted_minus_source",
    ]
    if (
        list(frame.columns) != expected_columns
        or len(frame) != DIAGNOSTIC_COUNTS[SEED_PAIR_DIAGNOSTICS]
        or frame["seed"].astype(int).tolist() != [42, 43, 44]
        or set(frame["source_group_id"]) != {PLAIN_CNN1D}
        or set(frame["adapted_group_id"]) != {DANN_CNN1D}
        or set(frame["test_domain"]) != {"outer"}
        or set(frame["metric"]) != {"macro_f1"}
        or not np.allclose(
            frame["adapted_minus_source"],
            frame["adapted_estimate"] - frame["source_estimate"],
            rtol=1e-12,
            atol=1e-15,
        )
        or not frame["adapted_minus_source"].gt(0).all()
    ):
        raise PipelineInvariantError("Seed-pair diagnostic invariants differ")
    if seed_metrics is not None:
        expected = _derive_seed_pair_diagnostics(seed_metrics)
        try:
            pd.testing.assert_frame_equal(
                frame.reset_index(drop=True),
                expected.reset_index(drop=True),
                check_exact=False,
                rtol=1e-12,
                atol=1e-15,
            )
        except AssertionError as error:
            raise PipelineInvariantError(
                "Seed-pair diagnostics differ from source evidence"
            ) from error


def render_encapsulation_figures(output: Path, frame: pd.DataFrame) -> None:
    validate_encapsulation_diagnostics(frame)
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
    padding = frame.loc[frame["section"].eq("padding_distribution")].sort_values("x")
    ordering = frame.loc[frame["section"].eq("ordering")]
    iat = frame.loc[frame["section"].eq("iat_distribution")].sort_values("x")

    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    metadata = {
        "Creator": "vpncat paper diagnostics",
        "Producer": "vpncat",
        "CreationDate": None,
        "ModDate": None,
    }

    padding_figure, padding_axis = plt.subplots(figsize=(3.35, 2.65))
    padding_axis.bar(
        padding["x"],
        100.0 * padding["value"],
        width=0.9,
        color="#176B87",
    )
    padding_axis.set_xlabel("Mean variable padding per flow (bytes)")
    padding_axis.set_ylabel("Retained flows (%)")
    padding_axis.set_xticks((0.5, 5.5, 10.5, 15.5), ("0", "5", "10", "15"))
    padding_figure.tight_layout()
    padding_figure.savefig(
        figures / "encapsulation_padding.pdf",
        bbox_inches="tight",
        metadata=metadata,
    )
    plt.close(padding_figure)

    order_values = [
        float(
            ordering.loc[
                ordering["metric"].eq("eligible_flows_with_outer_order_inversion"),
                "value",
            ].item()
        ),
        float(
            ordering.loc[
                ordering["metric"].eq("adjacent_pairs_with_outer_order_inversion"),
                "value",
            ].item()
        ),
    ]
    ordering_figure, ordering_axis = plt.subplots(figsize=(3.35, 2.65))
    bars = ordering_axis.bar(
        ("Flows affected", "Adjacent pairs"),
        100.0 * np.asarray(order_values),
        color=("#D95F02", "#E69F00"),
    )
    ordering_axis.set_ylabel("Order inversions (%)")
    ordering_axis.tick_params(axis="x", rotation=12)
    ordering_axis.set_ylim(0, 100.0 * max(order_values) * 1.18)
    for bar, value in zip(bars, order_values, strict=True):
        ordering_axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.35,
            f"{100.0 * value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=7,
        )
    ordering_figure.tight_layout()
    ordering_figure.savefig(
        figures / "encapsulation_ordering.pdf",
        bbox_inches="tight",
        metadata=metadata,
    )
    plt.close(ordering_figure)

    timing_figure, timing_axis = plt.subplots(figsize=(3.35, 2.65))
    timing_axis.plot(
        100.0 * iat["x"],
        100.0 * iat["value"],
        marker="o",
        color="#176B87",
    )
    timing_axis.set_xscale("log")
    timing_axis.set_xticks((0.1, 1.0, 5.0, 10.0), ("0.1", "1", "5", "10"))
    timing_axis.set_xlabel("Mean-IAT relative-change threshold (%)")
    timing_axis.set_ylabel("Flows within threshold (%)")
    timing_axis.set_ylim(70, 101)
    timing_axis.grid(axis="y", color="#DDDDDD", linewidth=0.5)
    timing_figure.tight_layout()
    timing_figure.savefig(
        figures / "encapsulation_timing.pdf",
        bbox_inches="tight",
        metadata=metadata,
    )
    plt.close(timing_figure)
