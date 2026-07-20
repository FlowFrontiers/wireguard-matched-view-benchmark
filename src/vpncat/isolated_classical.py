from __future__ import annotations

import argparse
from pathlib import Path

from vpncat.experiment import load_primary_experiment_config, select_primary_run
from vpncat.primary_runner import run_primary_classical


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute one isolated classical run")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--train-domain", choices=("inner", "outer"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    config = load_primary_experiment_config(
        args.config,
        artifact_dir=args.artifact_dir,
        output_root=args.output_root,
    )
    run = select_primary_run(
        config,
        experiment_id=args.experiment_id,
        fold=args.fold,
        train_domain=args.train_domain,
        seed=args.seed,
    )
    run_primary_classical(config, run)


if __name__ == "__main__":
    main()
