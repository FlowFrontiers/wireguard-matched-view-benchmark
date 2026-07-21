from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from vpncat.config import DatasetConfig
from vpncat.errors import PipelineInvariantError
from vpncat.hashing import sha256_file

FLOW_COLUMNS = (
    "id",
    "flow_id",
    "src_ip",
    "dst_ip",
    "src_port",
    "dst_port",
    "requested_server_name",
    "application_confidence",
    "application_is_guessed",
    "bidirectional_first_seen_ms",
    "bidirectional_last_seen_ms",
    "application_category_name",
)

OUTPUT_FILENAMES = (
    "endpoint_overlap.csv",
    "endpoint_concentration.csv",
    "label_confidence.csv",
    "temporal_block_summary.csv",
    "temporal_category_coverage.csv",
    "excluded_categories.csv",
)


def _read_inputs(
    config: DatasetConfig,
    canonical_path: Path,
    *,
    local_ip: str,
) -> pd.DataFrame:
    canonical = pd.read_parquet(
        canonical_path,
        columns=(
            "pair_id",
            "session",
            "source_id",
            "source_flow_id",
            "application_category",
        ),
    )
    if canonical["pair_id"].duplicated().any():
        raise PipelineInvariantError("Canonical pair IDs are not unique")

    parts: list[pd.DataFrame] = []
    for session, flow_path in sorted(config.flow_files.items()):
        if not flow_path.exists():
            raise FileNotFoundError(flow_path)
        flows = pd.read_parquet(flow_path, columns=FLOW_COLUMNS)
        if flows["flow_id"].duplicated().any():
            raise PipelineInvariantError(f"Session {session} flow_id is not unique")
        flows.insert(0, "session", session)
        parts.append(flows)
    source = pd.concat(parts, ignore_index=True)

    joined = canonical.merge(
        source,
        left_on=["session", "source_flow_id"],
        right_on=["session", "flow_id"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if not joined["_merge"].eq("both").all():
        missing = int(joined["_merge"].ne("both").sum())
        raise PipelineInvariantError(f"Canonical-to-flow join left {missing} unmatched pairs")
    joined = joined.drop(columns="_merge")
    if not joined["source_id"].eq(joined["id"]).all():
        raise PipelineInvariantError("source_id and released flow id disagree")
    if not joined["application_category"].eq(joined["application_category_name"]).all():
        raise PipelineInvariantError("Canonical and released application categories disagree")

    src_local = joined["src_ip"].astype(str).eq(local_ip)
    dst_local = joined["dst_ip"].astype(str).eq(local_ip)
    valid_orientation = src_local ^ dst_local
    if not valid_orientation.all():
        invalid = int((~valid_orientation).sum())
        raise PipelineInvariantError(
            f"Exactly one endpoint must equal local_ip for every flow; invalid={invalid}"
        )
    joined["remote_ip"] = np.where(src_local, joined["dst_ip"], joined["src_ip"])
    joined["remote_port"] = np.where(src_local, joined["dst_port"], joined["src_port"])
    joined["remote_ip"] = joined["remote_ip"].astype(str)
    joined["remote_port"] = joined["remote_port"].astype(np.int64)
    joined["remote_ip_port"] = (
        joined["remote_ip"] + "|" + joined["remote_port"].astype(str)
    )
    joined["requested_server_name"] = (
        joined["requested_server_name"].fillna("").astype(str).str.strip()
    )
    return joined


def _overlap_row(
    frame: pd.DataFrame,
    *,
    protocol: str,
    split: str,
    source_ids: set[str],
    target_ids: set[str],
    endpoint_column: str,
) -> dict[str, Any]:
    source = frame.loc[frame["pair_id"].isin(source_ids), endpoint_column]
    target = frame.loc[frame["pair_id"].isin(target_ids), endpoint_column]
    if len(source) != len(source_ids) or len(target) != len(target_ids):
        raise PipelineInvariantError(f"{protocol}/{split} pair IDs do not resolve exactly")
    source_values = set(source.astype(str))
    seen = target.astype(str).isin(source_values)
    return {
        "protocol": protocol,
        "split": split,
        "endpoint_key": endpoint_column,
        "source_pairs": len(source),
        "target_pairs": len(target),
        "source_unique_endpoints": len(source_values),
        "target_unique_endpoints": int(target.nunique()),
        "target_seen_pairs": int(seen.sum()),
        "target_seen_fraction": float(seen.mean()),
    }


def endpoint_overlap(
    frame: pd.DataFrame,
    primary_split_path: Path,
    cross_session_split_path: Path,
) -> pd.DataFrame:
    primary = pd.read_csv(primary_split_path)
    cross_session = pd.read_csv(cross_session_split_path)
    rows: list[dict[str, Any]] = []
    endpoint_columns = ("remote_ip", "remote_ip_port")

    role_columns = sorted(
        column for column in primary if column.startswith("role_fold_")
    )
    if not role_columns:
        raise PipelineInvariantError("Primary split has no role_fold columns")
    for role_column in role_columns:
        fold = role_column.removeprefix("role_fold_")
        source_ids = set(primary.loc[primary[role_column].eq("train"), "pair_id"])
        target_ids = set(primary.loc[primary[role_column].eq("test"), "pair_id"])
        for endpoint_column in endpoint_columns:
            rows.append(
                _overlap_row(
                    frame,
                    protocol="primary",
                    split=f"fold_{fold}",
                    source_ids=source_ids,
                    target_ids=target_ids,
                    endpoint_column=endpoint_column,
                )
            )

    for train_session in sorted(frame["session"].unique()):
        role_column = f"role_train_session_{train_session}"
        if role_column not in cross_session:
            raise PipelineInvariantError(f"Cross-session split lacks {role_column}")
        source_ids = set(
            cross_session.loc[cross_session[role_column].eq("train"), "pair_id"]
        )
        target_ids = set(
            cross_session.loc[cross_session[role_column].eq("test"), "pair_id"]
        )
        target_sessions = frame.loc[frame["pair_id"].isin(target_ids), "session"].unique()
        if len(target_sessions) != 1:
            raise PipelineInvariantError("Cross-session target IDs span multiple sessions")
        split = f"s{train_session}_to_s{int(target_sessions[0])}"
        for endpoint_column in endpoint_columns:
            rows.append(
                _overlap_row(
                    frame,
                    protocol="cross_session",
                    split=split,
                    source_ids=source_ids,
                    target_ids=target_ids,
                    endpoint_column=endpoint_column,
                )
            )
    return pd.DataFrame(rows).sort_values(
        ["protocol", "split", "endpoint_key"], ignore_index=True
    )


def endpoint_concentration(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for category, group in frame.groupby("application_category", sort=True):
        ip_counts = group["remote_ip"].value_counts()
        rows.append(
            {
                "application_category": category,
                "flow_count": len(group),
                "unique_remote_ips": int(group["remote_ip"].nunique()),
                "unique_remote_ip_ports": int(group["remote_ip_port"].nunique()),
                "top_1_remote_ip_share": float(ip_counts.head(1).sum() / len(group)),
                "top_3_remote_ip_share": float(ip_counts.head(3).sum() / len(group)),
                "nonempty_sni_count": int(group["requested_server_name"].ne("").sum()),
                "nonempty_sni_share": float(group["requested_server_name"].ne("").mean()),
                "unique_nonempty_sni": int(
                    group.loc[
                        group["requested_server_name"].ne(""), "requested_server_name"
                    ].nunique()
                ),
            }
        )
    return pd.DataFrame(rows)


def label_confidence(frame: pd.DataFrame) -> pd.DataFrame:
    result = (
        frame.groupby(
            ["application_confidence", "application_is_guessed"],
            dropna=False,
            observed=True,
        )
        .size()
        .rename("flow_count")
        .reset_index()
    )
    result["share"] = result["flow_count"] / len(frame)
    return result.sort_values(
        ["application_confidence", "application_is_guessed"], ignore_index=True
    )


def temporal_blocks(
    frame: pd.DataFrame,
    *,
    block_hours: tuple[int, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries: list[dict[str, Any]] = []
    category_rows: list[dict[str, Any]] = []
    for session, session_frame in frame.groupby("session", sort=True):
        start = session_frame["bidirectional_first_seen_ms"].astype(np.int64)
        end = session_frame["bidirectional_last_seen_ms"].astype(np.int64)
        if start.isna().any() or end.isna().any():
            raise PipelineInvariantError("Flow timestamps must be complete")
        session_origin = int(start.min())
        for hours in block_hours:
            width_ms = int(hours * 60 * 60 * 1000)
            block_id = ((start - session_origin) // width_ms).astype(np.int64)
            counts = block_id.value_counts().sort_index()
            expected_blocks = int(block_id.max()) + 1
            category_coverage = (
                session_frame.assign(_block_id=block_id)
                .groupby("application_category", observed=True)["_block_id"]
                .nunique()
            )
            summaries.append(
                {
                    "session": int(session),
                    "block_hours": hours,
                    "session_start_ms": session_origin,
                    "first_seen_span_hours": float((start.max() - start.min()) / 3_600_000),
                    "maximum_flow_duration_hours": float(((end - start).max()) / 3_600_000),
                    "populated_blocks": len(counts),
                    "expected_contiguous_blocks": expected_blocks,
                    "empty_blocks": expected_blocks - len(counts),
                    "pairs_p10": float(counts.quantile(0.10)),
                    "pairs_median": float(counts.quantile(0.50)),
                    "pairs_p90": float(counts.quantile(0.90)),
                    "minimum_category_block_presence": int(category_coverage.min()),
                    "median_category_block_presence": float(category_coverage.median()),
                    "maximum_category_block_presence": int(category_coverage.max()),
                }
            )
            for category, present_blocks in category_coverage.items():
                category_rows.append(
                    {
                        "session": int(session),
                        "block_hours": hours,
                        "application_category": category,
                        "blocks_present": int(present_blocks),
                        "populated_blocks": len(counts),
                        "block_presence_fraction": float(present_blocks / len(counts)),
                    }
                )
    return pd.DataFrame(summaries), pd.DataFrame(category_rows)


def excluded_categories(dataset_manifest_path: Path) -> pd.DataFrame:
    manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    counts = manifest["counts"]
    support = counts["class_support_after_eligibility_before_class_filtering"]
    retained = set(counts["retained_classes"])
    rows = [
        {"application_category": category, "eligible_flow_count": int(count)}
        for category, count in sorted(support.items())
        if category not in retained
    ]
    result = pd.DataFrame(rows)
    if int(result["eligible_flow_count"].sum()) != int(counts["excluded_below_class_support"]):
        raise PipelineInvariantError("Excluded-category counts disagree with dataset manifest")
    return result


def extract_collection_structure(
    config: DatasetConfig,
    *,
    canonical_path: Path,
    primary_split_path: Path,
    cross_session_split_path: Path,
    dataset_manifest_path: Path,
    output_dir: Path,
    local_ip: str,
    block_hours: tuple[int, ...] = (1, 2),
    force: bool = False,
) -> dict[str, Any]:
    """Extract compact collection-structure summaries without changing evidence."""
    if any(hours < 1 for hours in block_hours) or len(set(block_hours)) != len(block_hours):
        raise PipelineInvariantError("block_hours must be unique positive integers")
    output_dir = output_dir.expanduser().resolve()
    manifest_path = output_dir / "collection_structure_manifest.json"
    outputs = tuple(output_dir / name for name in OUTPUT_FILENAMES) + (manifest_path,)
    existing = [path for path in outputs if path.exists()]
    if existing and not force:
        raise FileExistsError(
            "Refusing to overwrite existing extraction outputs: "
            + ", ".join(map(str, existing))
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = _read_inputs(config, canonical_path, local_ip=local_ip)
    extracted = {
        "endpoint_overlap.csv": endpoint_overlap(
            frame, primary_split_path, cross_session_split_path
        ),
        "endpoint_concentration.csv": endpoint_concentration(frame),
        "label_confidence.csv": label_confidence(frame),
        "excluded_categories.csv": excluded_categories(dataset_manifest_path),
    }
    block_summary, category_coverage = temporal_blocks(frame, block_hours=block_hours)
    extracted["temporal_block_summary.csv"] = block_summary
    extracted["temporal_category_coverage.csv"] = category_coverage
    for filename, table in extracted.items():
        table.to_csv(output_dir / filename, index=False, float_format="%.12g")

    input_paths = {
        "canonical_pairs": canonical_path,
        "primary_split": primary_split_path,
        "cross_session_split": cross_session_split_path,
        "dataset_manifest": dataset_manifest_path,
    }
    for session, path in sorted(config.flow_files.items()):
        input_paths[f"session_{session}_flows"] = path
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "local_inner_ip": local_ip,
        "block_hours": list(block_hours),
        "inputs": {
            key: {"sha256": sha256_file(path)}
            for key, path in input_paths.items()
        },
        "counts": {
            "retained_pairs": len(frame),
            "sessions": {
                str(session): int(count)
                for session, count in frame["session"].value_counts().sort_index().items()
            },
            "unique_remote_ips": int(frame["remote_ip"].nunique()),
            "unique_remote_ip_ports": int(frame["remote_ip_port"].nunique()),
            "nonempty_sni": int(frame["requested_server_name"].ne("").sum()),
            "nonempty_sni_share": float(frame["requested_server_name"].ne("").mean()),
            "confidence_nonmissing": int(frame["application_confidence"].notna().sum()),
            "guessed_flag_nonmissing": int(frame["application_is_guessed"].notna().sum()),
            "excluded_categories": len(extracted["excluded_categories.csv"]),
            "excluded_below_support": int(
                extracted["excluded_categories.csv"]["eligible_flow_count"].sum()
            ),
        },
        "outputs": {
            filename: {
                "rows": len(table),
                "sha256": sha256_file(output_dir / filename),
            }
            for filename, table in sorted(extracted.items())
        },
        "notes": {
            "endpoint_orientation": (
                "remote endpoint is the non-local side, not unconditionally dst_ip"
            ),
            "confidence_codes": (
                "raw released nDPI codes; semantic names are intentionally not inferred"
            ),
            "overlap_source": (
                "only role=train pairs are endpoint-exposure sources; validation is excluded"
            ),
            "time_block_assignment": "flow assigned by bidirectional_first_seen_ms",
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest
