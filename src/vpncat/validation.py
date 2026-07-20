from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from vpncat.errors import PipelineInvariantError
from vpncat.features import build_prefix_stats
from vpncat.hashing import sha256_file
from vpncat.schema import (
    ABSOLUTE_TIMESTAMP_NAMES,
    CANONICAL_COLUMNS,
    CANONICAL_STATS,
    STAT_COLUMNS,
)
from vpncat.splits import validate_split_manifest


def validate_dataset_artifacts(
    canonical_path: Path,
    split_path: Path,
    *,
    folds: int,
    minimum_class_support: int,
    maximum_prefix_length: int,
    manifest_path: Path | None = None,
    assignment_audit_path: Path | None = None,
) -> dict[str, Any]:
    """Validate canonical data and fixed split artifacts without loading all sequences at once."""
    parquet = pq.ParquetFile(canonical_path)
    columns = parquet.schema_arrow.names
    missing = set(CANONICAL_COLUMNS) - set(columns)
    if missing:
        raise PipelineInvariantError(f"Canonical dataset is missing columns: {sorted(missing)}")
    banned = [name for name in columns if any(token in name for token in ABSOLUTE_TIMESTAMP_NAMES)]
    if banned:
        raise PipelineInvariantError(f"Absolute timestamp columns are prohibited: {banned}")

    pair_ids: set[str] = set()
    class_support: Counter[str] = Counter()
    session_support: Counter[int] = Counter()
    metadata_parts: list[pd.DataFrame] = []
    row_count = 0
    scan_columns = [
        "pair_id",
        "session",
        "application_category",
        "inner_direction",
        "inner_size",
        "inner_iat_ms",
        "outer_direction",
        "outer_size",
        "outer_iat_ms",
        "inner_length",
        "outer_length",
        *STAT_COLUMNS,
    ]
    for batch in parquet.iter_batches(columns=scan_columns, batch_size=10_000):
        frame = batch.to_pandas()
        duplicate_in_batch = frame["pair_id"].duplicated().any()
        overlap = pair_ids.intersection(frame["pair_id"])
        if duplicate_in_batch or overlap:
            raise PipelineInvariantError("Canonical dataset contains duplicate pair_id values")
        pair_ids.update(frame["pair_id"])
        class_support.update(frame["application_category"])
        session_support.update(int(value) for value in frame["session"])
        metadata_parts.append(
            frame.loc[:, ["pair_id", "session", "application_category"]].copy()
        )
        row_count += len(frame)

        statistics = frame.loc[:, STAT_COLUMNS].apply(pd.to_numeric, errors="coerce")
        statistic_values = statistics.to_numpy(dtype=np.float64)
        if np.isinf(statistic_values).any():
            raise PipelineInvariantError("Canonical statistics contain infinite values")
        finite_values = statistic_values[np.isfinite(statistic_values)]
        if (finite_values < 0).any():
            raise PipelineInvariantError("Canonical statistics contain negative values")
        for sentinel in (1e9, 2e9):
            if np.isclose(finite_values, sentinel, rtol=0.0, atol=1e-6).any():
                raise PipelineInvariantError(
                    f"Canonical statistics contain suspect sentinel value {sentinel:g}"
                )

        for domain in ("inner", "outer"):
            duration_seconds = statistics[f"{domain}_duration_ms"] / 1000.0
            valid_duration = duration_seconds.where(duration_seconds > 0)
            expected_rates = {
                "packet_rate": statistics[f"{domain}_packets"] / valid_duration,
                "byte_rate": statistics[f"{domain}_bytes"] / valid_duration,
            }
            for rate_name, expected in expected_rates.items():
                observed = statistics[f"{domain}_{rate_name}"]
                agrees = np.isclose(
                    observed.to_numpy(dtype=np.float64),
                    expected.to_numpy(dtype=np.float64),
                    rtol=1e-10,
                    atol=1e-10,
                    equal_nan=True,
                )
                if not agrees.all():
                    raise PipelineInvariantError(
                        f"{domain}_{rate_name} is inconsistent with count/duration"
                    )

        for domain in ("inner", "outer"):
            lengths = frame[f"{domain}_length"].astype(int)
            if not lengths.between(1, maximum_prefix_length).all():
                raise PipelineInvariantError(
                    f"{domain} sequence lengths are outside allowed bounds"
                )
            for channel in ("direction", "size", "iat_ms"):
                observed = frame[f"{domain}_{channel}"].map(len)
                if not observed.equals(lengths):
                    raise PipelineInvariantError(
                        f"{domain}_{channel} lengths disagree with {domain}_length"
                    )
                for sequence in frame[f"{domain}_{channel}"]:
                    values = np.asarray(sequence, dtype=np.float64)
                    if not np.isfinite(values).all():
                        raise PipelineInvariantError(
                            f"{domain}_{channel} contains non-finite values"
                        )
                    if channel == "direction":
                        if not np.isin(values, (0, 1)).all():
                            raise PipelineInvariantError(
                                f"{domain}_direction contains values outside {{0, 1}}"
                            )
                    elif (values < 0).any():
                        raise PipelineInvariantError(
                            f"{domain}_{channel} contains negative values"
                        )

        if not frame["inner_length"].astype(int).equals(
            frame["outer_length"].astype(int)
        ):
            raise PipelineInvariantError("Matched-view sequence lengths differ")
        if not np.array_equal(
            statistics["inner_packets"].to_numpy(),
            statistics["outer_packets"].to_numpy(),
        ):
            raise PipelineInvariantError("Matched-view full packet counts differ")
        expected_lengths = np.minimum(
            statistics["inner_packets"].to_numpy(dtype=int), maximum_prefix_length
        )
        if not np.array_equal(expected_lengths, frame["inner_length"].to_numpy(dtype=int)):
            raise PipelineInvariantError("Sequence lengths disagree with matched packet counts")

        complete = statistics["inner_packets"].le(maximum_prefix_length)
        if complete.any():
            complete_frame = frame.loc[complete]
            for domain in ("inner", "outer"):
                reconstructed = build_prefix_stats(
                    complete_frame,
                    domain=domain,
                    prefix_length=maximum_prefix_length,
                ).values
                observed = statistics.loc[
                    complete, [f"{domain}_{name}" for name in CANONICAL_STATS]
                ].to_numpy(dtype=float)
                agrees = np.isclose(
                    observed,
                    reconstructed,
                    rtol=1e-10,
                    atol=1e-8,
                    equal_nan=True,
                )
                if not agrees.all():
                    raise PipelineInvariantError(
                        f"{domain} complete-flow statistics disagree with PrefixStats conventions"
                    )

    too_small = {
        label: count
        for label, count in class_support.items()
        if count < minimum_class_support
    }
    if too_small:
        raise PipelineInvariantError(f"Retained classes violate minimum support: {too_small}")

    split = pd.read_csv(split_path)
    validate_split_manifest(split, folds=folds)
    split_ids = set(split["pair_id"])
    if split_ids != pair_ids:
        missing_from_split = len(pair_ids - split_ids)
        unknown_in_split = len(split_ids - pair_ids)
        detail = (
            f"missing_from_split={missing_from_split}, "
            f"unknown_in_split={unknown_in_split}"
        )
        raise PipelineInvariantError(f"Canonical/split pair coverage differs: {detail}")

    canonical_metadata = pd.concat(metadata_parts, ignore_index=True).set_index("pair_id")
    split_metadata = split.set_index("pair_id").loc[canonical_metadata.index]
    if not split_metadata["session"].astype(int).equals(
        canonical_metadata["session"].astype(int)
    ):
        raise PipelineInvariantError("Session labels disagree between canonical and split data")
    if not split_metadata["application_category"].astype(str).equals(
        canonical_metadata["application_category"].astype(str)
    ):
        raise PipelineInvariantError("Class labels disagree between canonical and split data")

    audit: pd.DataFrame | None = None
    if assignment_audit_path is not None:
        audit = pd.read_parquet(assignment_audit_path)
        required_audit = {
            "session",
            "source_id",
            "flow_id",
            "released_matched_packets",
            "reproduced_matched_packets",
            "counts_equal",
            "directional_bytes_fidelity_checked",
            "directional_bytes_fidelity_equal",
        }
        missing_audit = required_audit - set(audit.columns)
        if missing_audit:
            raise PipelineInvariantError(
                f"Assignment audit is missing columns: {sorted(missing_audit)}"
            )
        equal = (
            audit["released_matched_packets"].astype(int)
            == audit["reproduced_matched_packets"].astype(int)
        )
        if not equal.all() or not audit["counts_equal"].astype(bool).all():
            raise PipelineInvariantError("Assignment audit contains count mismatches")
        if int(audit["reproduced_matched_packets"].sum()) <= 0:
            raise PipelineInvariantError("Assignment audit contains no packet assignments")
        checked_bytes = audit["directional_bytes_fidelity_checked"].astype(bool)
        equal_bytes = audit["directional_bytes_fidelity_equal"].astype(bool)
        if not equal_bytes.loc[checked_bytes].all():
            raise PipelineInvariantError(
                "Assignment audit contains directional-byte fidelity failures"
            )

    if manifest_path is not None:
        with manifest_path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        artifact_paths = {
            "canonical_pairs": canonical_path,
            "split_manifest": split_path,
        }
        if assignment_audit_path is not None:
            artifact_paths["assignment_audit"] = assignment_audit_path
        for artifact_name, artifact_path in artifact_paths.items():
            expected = manifest.get("artifacts", {}).get(artifact_name, {}).get("sha256")
            if not expected:
                raise PipelineInvariantError(
                    f"Dataset manifest has no SHA-256 for {artifact_name}"
                )
            observed = sha256_file(artifact_path)
            if observed != expected:
                detail = f"expected {expected}, observed {observed}"
                raise PipelineInvariantError(
                    f"SHA-256 mismatch for {artifact_name}: {detail}"
                )
        if audit is not None:
            invariant_counts = manifest.get("invariants", {})
            if invariant_counts.get("assignment_count_equal_flows") != len(audit):
                raise PipelineInvariantError(
                    "Manifest assignment invariant count is inconsistent"
                )
            if invariant_counts.get("assigned_packet_rows_conserved") != int(
                audit["reproduced_matched_packets"].sum()
            ):
                raise PipelineInvariantError(
                    "Manifest packet-conservation count is inconsistent"
                )
            if invariant_counts.get("prefix_convention_equal_flows") != int(
                audit["released_matched_packets"].gt(0).sum()
            ):
                raise PipelineInvariantError(
                    "Manifest prefix-convention count is inconsistent"
                )
            if invariant_counts.get("endpoint_orientation_equal_packets") != int(
                audit["reproduced_matched_packets"].sum()
            ):
                raise PipelineInvariantError(
                    "Manifest endpoint-orientation count is inconsistent"
                )
            if invariant_counts.get("directional_bytes_fidelity_equal_flows") != int(
                (checked_bytes & equal_bytes).sum()
            ):
                raise PipelineInvariantError(
                    "Manifest directional-byte fidelity count is inconsistent"
                )

    return {
        "rows": row_count,
        "classes": len(class_support),
        "sessions": dict(sorted(session_support.items())),
        "folds": folds,
        "assignment_rows": len(audit) if audit is not None else None,
        "assigned_packets": (
            int(audit["reproduced_matched_packets"].sum())
            if audit is not None
            else None
        ),
        "endpoint_orientation_checked": (
            int(audit["reproduced_matched_packets"].sum())
            if audit is not None
            else None
        ),
        "endpoint_orientation_rate": 1.0 if audit is not None else None,
        "directional_bytes_fidelity_checked": (
            int(audit["directional_bytes_fidelity_checked"].sum())
            if audit is not None
            else None
        ),
        "directional_bytes_fidelity_rate": (
            float(
                audit.loc[
                    audit["directional_bytes_fidelity_checked"],
                    "directional_bytes_fidelity_equal",
                ].mean()
            )
            if audit is not None
            else None
        ),
        "status": "valid",
    }
