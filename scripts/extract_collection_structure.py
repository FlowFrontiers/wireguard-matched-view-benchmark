from __future__ import annotations

import argparse
import json
from pathlib import Path

from vpncat.collection_structure import extract_collection_structure
from vpncat.config import load_dataset_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract endpoint, label, and temporal collection summaries"
    )
    parser.add_argument("--dataset-config", type=Path, default=Path("configs/dataset.yaml"))
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument(
        "--canonical-path", type=Path, default=Path("data/processed/canonical_pairs.parquet")
    )
    parser.add_argument(
        "--primary-split", type=Path, default=Path("data/processed/split_manifest.csv")
    )
    parser.add_argument(
        "--cross-session-split",
        type=Path,
        default=Path("data/processed/cross_session_split_manifest.csv"),
    )
    parser.add_argument(
        "--dataset-manifest", type=Path, default=Path("data/processed/dataset_manifest.json")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/collection_structure"))
    parser.add_argument("--local-ip", default="10.14.0.2")
    parser.add_argument("--block-hours", type=int, nargs="+", default=(1, 2))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = load_dataset_config(args.dataset_config, input_root=args.input_root)
    report = extract_collection_structure(
        config,
        canonical_path=args.canonical_path,
        primary_split_path=args.primary_split,
        cross_session_split_path=args.cross_session_split,
        dataset_manifest_path=args.dataset_manifest,
        output_dir=args.output_dir,
        local_ip=args.local_ip,
        block_hours=tuple(args.block_hours),
        force=args.force,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
