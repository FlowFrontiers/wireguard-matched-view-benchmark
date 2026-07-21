from __future__ import annotations

import argparse
import json
from pathlib import Path

from vpncat.config import load_dataset_config
from vpncat.temporal_bootstrap import run_temporal_bootstrap


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-estimate paired intervals with session-preserving time blocks"
    )
    parser.add_argument("--dataset-config", type=Path, default=Path("configs/dataset.yaml"))
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument(
        "--canonical-path", type=Path, default=Path("data/processed/canonical_pairs.parquet")
    )
    parser.add_argument(
        "--dataset-manifest", type=Path, default=Path("data/processed/dataset_manifest.json")
    )
    parser.add_argument("--analysis-root", type=Path, default=Path("outputs/analysis"))
    parser.add_argument(
        "--paper-analysis-root", type=Path, default=Path("artifacts/f82a743/paper_analysis")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/temporal_bootstrap")
    )
    parser.add_argument("--block-hours", type=int, nargs="+", default=(1, 2))
    parser.add_argument("--replicates", type=int, default=1_000)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = load_dataset_config(args.dataset_config, input_root=args.input_root)
    report = run_temporal_bootstrap(
        config,
        canonical_path=args.canonical_path,
        dataset_manifest_path=args.dataset_manifest,
        analysis_root=args.analysis_root,
        paper_analysis_root=args.paper_analysis_root,
        output_dir=args.output_dir,
        block_hours=tuple(args.block_hours),
        replicates=args.replicates,
        confidence_level=args.confidence_level,
        seed=args.seed,
        force=args.force,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
