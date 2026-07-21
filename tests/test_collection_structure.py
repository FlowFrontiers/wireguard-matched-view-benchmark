from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from vpncat.collection_structure import extract_collection_structure
from vpncat.config import load_dataset_config


def _write_fixture(root: Path) -> tuple[Path, Path]:
    input_root = root / "raw"
    processed = root / "processed"
    for session in (1, 2):
        (input_root / f"session{session}").mkdir(parents=True)
    processed.mkdir()

    rows = {
        1: [
            (0, "10.0.0.1", "1.1.1.1", 50000, 443, "a.example", 6, 0, 0, 1000, "Web"),
            (1, "2.2.2.2", "10.0.0.1", 53, 50001, "", 1, 1, 3_600_000, 3_601_000, "Network"),
            (2, "10.0.0.1", "3.3.3.3", 50002, 443, "", 0, 1, 7_200_000, 7_201_000, "Web"),
        ],
        2: [
            (0, "10.0.0.1", "1.1.1.1", 50003, 443, "a.example", 6, 0, 0, 1000, "Web"),
            (1, "4.4.4.4", "10.0.0.1", 53, 50004, "", 7, 0, 3_600_000, 3_601_000, "Network"),
        ],
    }
    canonical_rows: list[dict[str, object]] = []
    for session, session_rows in rows.items():
        frame = pd.DataFrame(
            session_rows,
            columns=(
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
            ),
        )
        frame.insert(0, "id", frame["flow_id"])
        frame.to_parquet(
            input_root / f"session{session}" / f"session{session}_flows.parquet",
            index=False,
        )
        (input_root / f"session{session}" / f"session{session}_packet_matches.parquet").write_bytes(
            b"unused"
        )
        for row in frame.itertuples(index=False):
            canonical_rows.append(
                {
                    "pair_id": f"s{session}:{row.id}",
                    "session": session,
                    "source_id": row.id,
                    "source_flow_id": row.flow_id,
                    "application_category": row.application_category_name,
                }
            )
    pd.DataFrame(canonical_rows).to_parquet(processed / "canonical_pairs.parquet", index=False)

    pairs = pd.DataFrame(canonical_rows)[["pair_id", "session", "application_category"]]
    pairs["role_fold_1"] = ["train", "test", "validation", "test", "train"]
    pairs["role_fold_2"] = ["test", "train", "train", "train", "test"]
    pairs.to_csv(processed / "split_manifest.csv", index=False)
    pairs["role_train_session_1"] = ["train", "validation", "train", "test", "test"]
    pairs["role_train_session_2"] = ["test", "test", "test", "train", "validation"]
    pairs.to_csv(processed / "cross_session_split_manifest.csv", index=False)

    manifest = {
        "counts": {
            "class_support_after_eligibility_before_class_filtering": {
                "Advertisement": 2,
                "Network": 2,
                "Web": 3,
            },
            "excluded_below_class_support": 2,
            "retained_classes": ["Network", "Web"],
        }
    }
    (processed / "dataset_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    config = {
        "dataset": {
            "input_root": str(input_root),
            "output_dir": str(processed),
            "sessions": {
                1: {
                    "flows": "session1/session1_flows.parquet",
                    "packet_matches": "session1/session1_packet_matches.parquet",
                },
                2: {
                    "flows": "session2/session2_flows.parquet",
                    "packet_matches": "session2/session2_packet_matches.parquet",
                },
            },
            "minimum_class_support": 2,
            "maximum_prefix_length": 80,
        },
        "splits": {"folds": 2, "validation_fraction": 0.1, "random_seed": 42},
    }
    config_path = root / "configs" / "dataset.yaml"
    config_path.parent.mkdir()
    import yaml

    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path, processed


def test_extracts_remote_orientation_and_protocol_train_overlap(tmp_path: Path) -> None:
    config_path, processed = _write_fixture(tmp_path)
    config = load_dataset_config(config_path)
    output = tmp_path / "output"
    report = extract_collection_structure(
        config,
        canonical_path=processed / "canonical_pairs.parquet",
        primary_split_path=processed / "split_manifest.csv",
        cross_session_split_path=processed / "cross_session_split_manifest.csv",
        dataset_manifest_path=processed / "dataset_manifest.json",
        output_dir=output,
        local_ip="10.0.0.1",
        block_hours=(1, 2),
    )

    assert report["counts"]["retained_pairs"] == 5
    assert report["counts"]["unique_remote_ips"] == 4
    assert report["counts"]["excluded_below_support"] == 2
    overlap = pd.read_csv(output / "endpoint_overlap.csv")
    row = overlap.loc[
        overlap["protocol"].eq("cross_session")
        & overlap["split"].eq("s1_to_s2")
        & overlap["endpoint_key"].eq("remote_ip")
    ].iloc[0]
    assert row["source_pairs"] == 2
    assert row["target_pairs"] == 2
    assert row["target_seen_pairs"] == 1
    assert row["target_seen_fraction"] == 0.5

    confidence = pd.read_csv(output / "label_confidence.csv")
    assert confidence["flow_count"].sum() == 5
    temporal = pd.read_csv(output / "temporal_block_summary.csv")
    assert set(temporal["block_hours"]) == {1, 2}
