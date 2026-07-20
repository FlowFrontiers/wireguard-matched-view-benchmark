from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from vpncat.errors import PipelineInvariantError


@dataclass(frozen=True)
class DatasetConfig:
    project_root: Path
    input_root: Path
    flow_files: dict[int, Path]
    packet_match_files: dict[int, Path]
    output_dir: Path
    minimum_class_support: int
    maximum_prefix_length: int
    packet_batch_size: int
    aggregation_partitions: int
    assignment_padding_ms: float
    folds: int
    validation_fraction: float
    random_seed: int


@dataclass(frozen=True)
class FeatureConfig:
    project_root: Path
    canonical_path: Path
    audit_output: Path
    primary_prefix_length: int
    available_prefix_lengths: tuple[int, ...]
    sequence_channels: tuple[str, ...]
    log_transform_magnitudes: bool


@dataclass(frozen=True)
class PreprocessingConfig:
    project_root: Path
    canonical_path: Path
    split_path: Path
    dataset_manifest_path: Path
    audit_output: Path
    folds: int
    prefix_length: int
    statistical_representations: tuple[str, ...]
    domains: tuple[str, ...]


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def load_dataset_config(
    config_path: Path,
    *,
    input_root: Path | None = None,
    output_dir: Path | None = None,
) -> DatasetConfig:
    """Load and validate dataset/split configuration from YAML."""
    config_path = config_path.expanduser().resolve()
    with config_path.open(encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle)

    project_root = config_path.parent.parent
    dataset = raw.get("dataset", {})
    splits = raw.get("splits", {})

    configured_input = input_root if input_root is not None else dataset.get("input_root")
    configured_output = output_dir if output_dir is not None else dataset.get("output_dir")
    if configured_input is None or configured_output is None:
        raise PipelineInvariantError("dataset.input_root and dataset.output_dir are required")

    root = _resolve(project_root, configured_input)
    sessions = dataset.get("sessions", {})
    flow_files = {
        int(key): root / str(value["flows"]) for key, value in sessions.items()
    }
    packet_match_files = {
        int(key): root / str(value["packet_matches"])
        for key, value in sessions.items()
    }
    if len(flow_files) < 2 or flow_files.keys() != packet_match_files.keys():
        raise PipelineInvariantError("At least two session files are required")

    config = DatasetConfig(
        project_root=project_root,
        input_root=root,
        flow_files=flow_files,
        packet_match_files=packet_match_files,
        output_dir=_resolve(project_root, configured_output),
        minimum_class_support=int(dataset.get("minimum_class_support", 200)),
        maximum_prefix_length=int(dataset.get("maximum_prefix_length", 80)),
        packet_batch_size=int(dataset.get("packet_batch_size", 500_000)),
        aggregation_partitions=int(dataset.get("aggregation_partitions", 64)),
        assignment_padding_ms=float(dataset.get("assignment_padding_ms", 2_000.0)),
        folds=int(splits.get("folds", 5)),
        validation_fraction=float(splits.get("validation_fraction", 0.10)),
        random_seed=int(splits.get("random_seed", 42)),
    )
    _validate_config(config)
    return config


def _validate_config(config: DatasetConfig) -> None:
    if config.minimum_class_support < 1:
        raise PipelineInvariantError("minimum_class_support must be positive")
    if config.maximum_prefix_length < 1:
        raise PipelineInvariantError("maximum_prefix_length must be positive")
    if config.packet_batch_size < 1:
        raise PipelineInvariantError("packet_batch_size must be positive")
    if config.aggregation_partitions < 1:
        raise PipelineInvariantError("aggregation_partitions must be positive")
    if config.assignment_padding_ms < 0:
        raise PipelineInvariantError("assignment_padding_ms must be nonnegative")
    if config.folds < 2:
        raise PipelineInvariantError("folds must be at least 2")
    if not 0.0 < config.validation_fraction < 1.0:
        raise PipelineInvariantError("validation_fraction must be between 0 and 1")


def load_feature_config(
    config_path: Path,
    *,
    canonical_path: Path | None = None,
    audit_output: Path | None = None,
) -> FeatureConfig:
    """Load the deterministic representation configuration."""
    config_path = config_path.expanduser().resolve()
    with config_path.open(encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle)
    project_root = config_path.parent.parent
    features = raw.get("features", {})
    configured_canonical = canonical_path or features.get("canonical_path")
    configured_audit = audit_output or features.get("audit_output")
    if configured_canonical is None or configured_audit is None:
        raise PipelineInvariantError(
            "features.canonical_path and features.audit_output are required"
        )

    config = FeatureConfig(
        project_root=project_root,
        canonical_path=_resolve(project_root, configured_canonical),
        audit_output=_resolve(project_root, configured_audit),
        primary_prefix_length=int(features.get("primary_prefix_length", 50)),
        available_prefix_lengths=tuple(
            int(value) for value in features.get("available_prefix_lengths", [50])
        ),
        sequence_channels=tuple(
            str(value)
            for value in features.get(
                "sequence_channels", ["direction", "size", "iat_ms"]
            )
        ),
        log_transform_magnitudes=bool(
            features.get("log_transform_magnitudes", True)
        ),
    )
    valid_channels = {"direction", "size", "iat_ms"}
    if config.primary_prefix_length < 1:
        raise PipelineInvariantError("primary_prefix_length must be positive")
    if config.primary_prefix_length not in config.available_prefix_lengths:
        raise PipelineInvariantError(
            "primary_prefix_length must appear in available_prefix_lengths"
        )
    if not config.available_prefix_lengths or min(config.available_prefix_lengths) < 1:
        raise PipelineInvariantError("available_prefix_lengths must be positive")
    if not config.sequence_channels or not set(config.sequence_channels) <= valid_channels:
        raise PipelineInvariantError("sequence_channels contains unsupported channels")
    return config


def load_preprocessing_config(
    config_path: Path,
    *,
    canonical_path: Path | None = None,
    split_path: Path | None = None,
    dataset_manifest_path: Path | None = None,
    audit_output: Path | None = None,
) -> PreprocessingConfig:
    """Load the fold-safe preprocessing audit configuration."""
    config_path = config_path.expanduser().resolve()
    with config_path.open(encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle)
    project_root = config_path.parent.parent
    preprocessing = raw.get("preprocessing", {})
    configured_canonical = canonical_path or preprocessing.get("canonical_path")
    configured_split = split_path or preprocessing.get("split_path")
    configured_manifest = dataset_manifest_path or preprocessing.get(
        "dataset_manifest_path"
    )
    configured_audit = audit_output or preprocessing.get("audit_output")
    if any(
        value is None
        for value in (
            configured_canonical,
            configured_split,
            configured_manifest,
            configured_audit,
        )
    ):
        raise PipelineInvariantError(
            "preprocessing canonical, split, manifest, and audit paths are required"
        )
    config = PreprocessingConfig(
        project_root=project_root,
        canonical_path=_resolve(project_root, configured_canonical),
        split_path=_resolve(project_root, configured_split),
        dataset_manifest_path=_resolve(project_root, configured_manifest),
        audit_output=_resolve(project_root, configured_audit),
        folds=int(preprocessing.get("folds", 5)),
        prefix_length=int(preprocessing.get("prefix_length", 50)),
        statistical_representations=tuple(
            str(value)
            for value in preprocessing.get(
                "statistical_representations",
                ["matched_flow_stats", "prefix_stats"],
            )
        ),
        domains=tuple(str(value) for value in preprocessing.get("domains", ["inner", "outer"])),
    )
    if config.folds < 2:
        raise PipelineInvariantError("preprocessing.folds must be at least 2")
    if config.prefix_length < 1:
        raise PipelineInvariantError("preprocessing.prefix_length must be positive")
    if len(config.statistical_representations) != 2 or set(
        config.statistical_representations
    ) != {"matched_flow_stats", "prefix_stats"}:
        raise PipelineInvariantError(
            "M3 must audit matched_flow_stats and prefix_stats"
        )
    if len(config.domains) != 2 or set(config.domains) != {"inner", "outer"}:
        raise PipelineInvariantError("M3 must audit both inner and outer domains")
    return config
