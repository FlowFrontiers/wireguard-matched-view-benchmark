from __future__ import annotations

import argparse
import json
from pathlib import Path

from vpncat.robustness_extension import publish_robustness_extension


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Publish compact collection-structure and temporal-bootstrap evidence"
    )
    parser.add_argument(
        "--collection-dir", type=Path, default=Path("outputs/collection_structure")
    )
    parser.add_argument(
        "--temporal-dir", type=Path, default=Path("outputs/temporal_bootstrap")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/f82a743/paper_analysis/robustness_extension"),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parent.parent
    report = publish_robustness_extension(
        collection_dir=args.collection_dir.resolve(),
        temporal_dir=args.temporal_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        project_root=project_root,
        force=args.force,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
