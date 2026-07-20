from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from vpncat import __version__
from vpncat.config import FeatureConfig
from vpncat.features import (
    build_flattened_splt,
    build_matched_flow_stats,
    build_prefix_stats,
    build_sequential_splt,
)
from vpncat.hashing import sha256_file
from vpncat.provenance import git_provenance
from vpncat.schema import CANONICAL_COLUMNS


def audit_features(config: FeatureConfig, *, batch_size: int = 10_000) -> dict[str, Any]:
    """Exercise every primary representation over the complete canonical dataset."""
    parquet = pq.ParquetFile(config.canonical_path)
    row_count = 0
    missing_values = {
        domain: {"matched_flow_stats": 0, "prefix_stats": 0}
        for domain in ("inner", "outer")
    }
    for batch in parquet.iter_batches(columns=list(CANONICAL_COLUMNS), batch_size=batch_size):
        frame = batch.to_pandas()
        row_count += len(frame)
        for domain in ("inner", "outer"):
            full = build_matched_flow_stats(frame, domain=domain)
            prefix = build_prefix_stats(
                frame,
                domain=domain,
                prefix_length=config.primary_prefix_length,
            )
            sequential = build_sequential_splt(
                frame,
                domain=domain,
                prefix_length=config.primary_prefix_length,
                channels=config.sequence_channels,
                log_transform_magnitudes=config.log_transform_magnitudes,
            )
            flattened = build_flattened_splt(
                frame,
                domain=domain,
                prefix_length=config.primary_prefix_length,
                channels=config.sequence_channels,
                log_transform_magnitudes=config.log_transform_magnitudes,
            )
            if not np.array_equal(flattened.values, sequential.values.reshape(len(frame), -1)):
                raise ValueError("Flattened SPLT differs from the sequential tensor")
            if not np.array_equal(flattened.mask, sequential.mask):
                raise ValueError("Flattened and sequential SPLT masks disagree")
            missing_values[domain]["matched_flow_stats"] += int(np.isnan(full.values).sum())
            missing_values[domain]["prefix_stats"] += int(np.isnan(prefix.values).sum())

    payload = {
        "created_utc": datetime.now(UTC).isoformat(),
        "package_version": __version__,
        "git": git_provenance(config.project_root),
        "canonical": {
            "path": str(config.canonical_path),
            "sha256": sha256_file(config.canonical_path),
            "rows": row_count,
        },
        "configuration": {
            "primary_prefix_length": config.primary_prefix_length,
            "available_prefix_lengths": list(config.available_prefix_lengths),
            "sequence_channels": list(config.sequence_channels),
            "log_transform_magnitudes": config.log_transform_magnitudes,
            "direction_encoding": {"raw_0": -1, "raw_1": 1, "padding": 0},
            "flattened_layout": "timestep-major",
        },
        "representations": {
            "matched_flow_stats_features": 21,
            "prefix_stats_features": 21,
            "sequential_splt_shape": [config.primary_prefix_length, len(config.sequence_channels)],
            "flattened_splt_features": (
                config.primary_prefix_length * len(config.sequence_channels)
            ),
        },
        "missing_values": missing_values,
        "status": "valid",
    }
    config.audit_output.parent.mkdir(parents=True, exist_ok=True)
    temporary = config.audit_output.with_suffix(config.audit_output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(config.audit_output)
    return payload
