from __future__ import annotations

import argparse
from pathlib import Path

from vpncat.cross_session import select_cross_session_run
from vpncat.cross_session_preprocessing_audit import (
    load_cross_session_preprocessing_config,
)
from vpncat.cross_session_runner import run_cross_session_classical


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Execute one isolated cross-session classical run"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--train-session", type=int, choices=(1, 2), required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    config = load_cross_session_preprocessing_config(
        args.config,
        artifact_dir=args.artifact_dir,
        output_root=args.output_root,
    )
    run = select_cross_session_run(
        config.cross_session,
        experiment_id=args.experiment_id,
        train_session=args.train_session,
        seed=args.seed,
    )
    run_cross_session_classical(config, run)


if __name__ == "__main__":
    main()
