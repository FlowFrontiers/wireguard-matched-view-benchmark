from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from vpncat.config import (
    load_dataset_config,
    load_feature_config,
    load_preprocessing_config,
)
from vpncat.contract_audit import audit_experiment_contract
from vpncat.cross_session import (
    build_cross_session_contract,
    load_cross_session_config,
    select_cross_session_run,
    validate_cross_session_contract,
)
from vpncat.cross_session_preprocessing_audit import (
    audit_cross_session_preprocessing,
    load_cross_session_preprocessing_config,
)
from vpncat.data import (
    ASSIGNMENT_AUDIT_FILENAME,
    CANONICAL_FILENAME,
    MANIFEST_FILENAME,
    SPLIT_FILENAME,
    build_canonical_dataset,
)
from vpncat.experiment import load_primary_experiment_config, select_primary_run
from vpncat.feature_audit import audit_features
from vpncat.preprocessing_audit import audit_preprocessing
from vpncat.primary_runner import run_primary_classical
from vpncat.validation import validate_dataset_artifacts

NEURAL_MODELS = ("cnn1d", "lstm", "transformer")


def _common_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=Path, default=Path("configs/dataset.yaml"))
    parser.add_argument("--input-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser


def build_main(argv: Sequence[str] | None = None) -> None:
    parser = _common_parser("Build canonical paired WireGuard benchmark artifacts")
    parser.add_argument("--force", action="store_true", help="Replace canonical artifacts")
    args = parser.parse_args(argv)
    config = load_dataset_config(
        args.config,
        input_root=args.input_root,
        output_dir=args.output_dir,
    )
    manifest = build_canonical_dataset(config, force=args.force)
    print(json.dumps(manifest, indent=2, sort_keys=True))


def validate_main(argv: Sequence[str] | None = None) -> None:
    parser = _common_parser("Validate canonical paired WireGuard benchmark artifacts")
    args = parser.parse_args(argv)
    config = load_dataset_config(
        args.config,
        input_root=args.input_root,
        output_dir=args.output_dir,
    )
    report = validate_dataset_artifacts(
        config.output_dir / CANONICAL_FILENAME,
        config.output_dir / SPLIT_FILENAME,
        folds=config.folds,
        minimum_class_support=config.minimum_class_support,
        maximum_prefix_length=config.maximum_prefix_length,
        manifest_path=config.output_dir / MANIFEST_FILENAME,
        assignment_audit_path=config.output_dir / ASSIGNMENT_AUDIT_FILENAME,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


def audit_features_main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Audit all primary feature representations")
    parser.add_argument("--config", type=Path, default=Path("configs/features.yaml"))
    parser.add_argument("--canonical-path", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch-size", type=int, default=10_000)
    args = parser.parse_args(argv)
    config = load_feature_config(
        args.config,
        canonical_path=args.canonical_path,
        audit_output=args.output,
    )
    report = audit_features(config, batch_size=args.batch_size)
    print(json.dumps(report, indent=2, sort_keys=True))


def audit_preprocessing_main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Audit fold-safe fitted preprocessing")
    parser.add_argument("--config", type=Path, default=Path("configs/preprocessing.yaml"))
    parser.add_argument("--canonical-path", type=Path)
    parser.add_argument("--split-path", type=Path)
    parser.add_argument("--dataset-manifest-path", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    config = load_preprocessing_config(
        args.config,
        canonical_path=args.canonical_path,
        split_path=args.split_path,
        dataset_manifest_path=args.dataset_manifest_path,
        audit_output=args.output,
    )
    report = audit_preprocessing(config)
    print(json.dumps(report, indent=2, sort_keys=True))


def audit_experiment_contract_main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Audit the frozen primary run contract")
    parser.add_argument("--config", type=Path, default=Path("configs/primary.yaml"))
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    config = load_primary_experiment_config(
        args.config,
        artifact_dir=args.artifact_dir,
        output_root=args.output_root,
        contract_audit_path=args.output,
    )
    report = audit_experiment_contract(config)
    print(json.dumps(report, indent=2, sort_keys=True))


def run_primary_classical_main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run one frozen primary classical experiment")
    parser.add_argument("--config", type=Path, default=Path("configs/primary.yaml"))
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--fold", required=True, type=int)
    parser.add_argument("--train-domain", required=True, choices=("inner", "outer"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
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
    output = run_primary_classical(config, run)
    print(json.dumps({"run_id": run.run_id, "output": str(output)}, indent=2))


def tune_neural_main(argv: Sequence[str] | None = None) -> None:
    from vpncat.neural_config import load_neural_config
    from vpncat.neural_tuning import tune_neural_model

    parser = argparse.ArgumentParser(description="Run or resume frozen neural tuning trials")
    parser.add_argument("--primary-config", type=Path, default=Path("configs/primary.yaml"))
    parser.add_argument("--neural-config", type=Path, default=Path("configs/neural.yaml"))
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--tuning-output-root", type=Path)
    parser.add_argument("--model", required=True, choices=NEURAL_MODELS)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--trial-id", action="append", type=int)
    args = parser.parse_args(argv)
    primary = load_primary_experiment_config(
        args.primary_config,
        artifact_dir=args.artifact_dir,
    )
    neural = load_neural_config(
        args.neural_config,
        tuning_output_root=args.tuning_output_root,
    )
    report = tune_neural_model(
        primary,
        neural,
        model_name=args.model,
        device_name=args.device,
        trial_ids=None if args.trial_id is None else tuple(args.trial_id),
    )
    print(json.dumps(report, indent=2, sort_keys=True))


def run_primary_neural_main(argv: Sequence[str] | None = None) -> None:
    from vpncat.neural_config import load_neural_config
    from vpncat.neural_runner import run_primary_neural

    parser = argparse.ArgumentParser(description="Run one selected primary neural experiment")
    parser.add_argument("--primary-config", type=Path, default=Path("configs/primary.yaml"))
    parser.add_argument("--neural-config", type=Path, default=Path("configs/neural.yaml"))
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--tuning-output-root", type=Path)
    parser.add_argument("--model", required=True, choices=NEURAL_MODELS)
    parser.add_argument("--fold", required=True, type=int)
    parser.add_argument("--train-domain", required=True, choices=("inner", "outer"))
    parser.add_argument("--seed", required=True, type=int, choices=(42, 43, 44))
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    args = parser.parse_args(argv)
    primary = load_primary_experiment_config(
        args.primary_config,
        artifact_dir=args.artifact_dir,
        output_root=args.output_root,
    )
    neural = load_neural_config(
        args.neural_config,
        tuning_output_root=args.tuning_output_root,
    )
    run = select_primary_run(
        primary,
        experiment_id=f"sequential_splt__{args.model}",
        fold=args.fold,
        train_domain=args.train_domain,
        seed=args.seed,
    )
    output = run_primary_neural(primary, neural, run, device_name=args.device)
    print(json.dumps({"run_id": run.run_id, "output": str(output)}, indent=2))


def primary_matrix_main(argv: Sequence[str] | None = None) -> None:
    from vpncat.neural_config import load_neural_config
    from vpncat.orchestration import PrimaryRunFilters, run_primary_matrix

    parser = argparse.ArgumentParser(
        description="Plan, validate, or resume a filtered primary experiment matrix"
    )
    parser.add_argument("--primary-config", type=Path, default=Path("configs/primary.yaml"))
    parser.add_argument("--neural-config", type=Path, default=Path("configs/neural.yaml"))
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--tuning-output-root", type=Path)
    parser.add_argument("--family", action="append", choices=("classical", "neural"))
    parser.add_argument(
        "--model",
        action="append",
        choices=("random_forest", "xgboost", *NEURAL_MODELS),
    )
    parser.add_argument(
        "--representation",
        action="append",
        choices=(
            "matched_flow_stats",
            "prefix_stats",
            "flattened_splt",
            "sequential_splt",
        ),
    )
    parser.add_argument("--fold", action="append", type=int, choices=(1, 2, 3, 4, 5))
    parser.add_argument("--train-domain", action="append", choices=("inner", "outer"))
    parser.add_argument("--seed", action="append", type=int, choices=(42, 43, 44))
    parser.add_argument("--run-id", action="append")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--maximum-pending-runs", type=int)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    primary = load_primary_experiment_config(
        args.primary_config,
        artifact_dir=args.artifact_dir,
        output_root=args.output_root,
    )
    neural = load_neural_config(
        args.neural_config,
        tuning_output_root=args.tuning_output_root,
    )
    report = run_primary_matrix(
        primary,
        neural,
        filters=PrimaryRunFilters(
            families=tuple(args.family or ()),
            models=tuple(args.model or ()),
            representations=tuple(args.representation or ()),
            folds=tuple(args.fold or ()),
            train_domains=tuple(args.train_domain or ()),
            seeds=tuple(args.seed or ()),
            run_ids=tuple(args.run_id or ()),
        ),
        execute=args.execute,
        maximum_pending_runs=args.maximum_pending_runs,
        device_name=args.device,
        report_path=args.report,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


def cross_session_contract_main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build or validate the frozen cross-session split and run contract"
    )
    parser.add_argument("--config", type=Path, default=Path("configs/cross_session.yaml"))
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--split-output", type=Path)
    parser.add_argument("--audit-output", type=Path)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    config = load_cross_session_config(
        args.config,
        artifact_dir=args.artifact_dir,
        output_root=args.output_root,
        split_path=args.split_output,
        contract_audit_path=args.audit_output,
    )
    if args.validate:
        if args.force:
            parser.error("--force cannot be combined with --validate")
        report = validate_cross_session_contract(config)
    else:
        report = build_cross_session_contract(config, force=args.force)
    print(json.dumps(report, indent=2, sort_keys=True))


def cross_session_preprocessing_main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Audit cross-session preprocessing")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/cross_session_preprocessing.yaml"),
    )
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--split-path", type=Path)
    parser.add_argument("--contract-audit-path", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    config = load_cross_session_preprocessing_config(
        args.config,
        artifact_dir=args.artifact_dir,
        split_path=args.split_path,
        contract_audit_path=args.contract_audit_path,
        audit_output=args.output,
    )
    report = audit_cross_session_preprocessing(config)
    print(json.dumps(report, indent=2, sort_keys=True))


def run_cross_session_classical_main(argv: Sequence[str] | None = None) -> None:
    from vpncat.cross_session_runner import run_cross_session_classical

    parser = argparse.ArgumentParser(description="Run one cross-session classical experiment")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/cross_session_preprocessing.yaml"),
    )
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--train-session", required=True, type=int, choices=(1, 2))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
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
    output = run_cross_session_classical(config, run)
    print(json.dumps({"run_id": run.run_id, "output": str(output)}, indent=2))


def run_cross_session_neural_main(argv: Sequence[str] | None = None) -> None:
    from vpncat.cross_session_neural_runner import run_cross_session_neural
    from vpncat.neural_config import load_neural_config

    parser = argparse.ArgumentParser(description="Run one cross-session neural experiment")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/cross_session_preprocessing.yaml"),
    )
    parser.add_argument("--neural-config", type=Path, default=Path("configs/neural.yaml"))
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--tuning-output-root", type=Path)
    parser.add_argument("--model", required=True, choices=NEURAL_MODELS)
    parser.add_argument("--train-session", required=True, type=int, choices=(1, 2))
    parser.add_argument("--seed", required=True, type=int, choices=(42, 43, 44))
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    args = parser.parse_args(argv)
    config = load_cross_session_preprocessing_config(
        args.config,
        artifact_dir=args.artifact_dir,
        output_root=args.output_root,
    )
    neural = load_neural_config(
        args.neural_config,
        tuning_output_root=args.tuning_output_root,
    )
    run = select_cross_session_run(
        config.cross_session,
        experiment_id=f"sequential_splt__{args.model}",
        train_session=args.train_session,
        seed=args.seed,
    )
    output = run_cross_session_neural(
        config,
        neural,
        run,
        device_name=args.device,
    )
    print(json.dumps({"run_id": run.run_id, "output": str(output)}, indent=2))


def cross_session_matrix_main(argv: Sequence[str] | None = None) -> None:
    from vpncat.cross_session_orchestration import (
        CrossSessionRunFilters,
        run_cross_session_matrix,
    )
    from vpncat.neural_config import load_neural_config

    parser = argparse.ArgumentParser(
        description="Plan, validate, or resume the cross-session experiment matrix"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/cross_session_preprocessing.yaml"),
    )
    parser.add_argument("--neural-config", type=Path, default=Path("configs/neural.yaml"))
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--tuning-output-root", type=Path)
    parser.add_argument("--family", action="append", choices=("classical", "neural"))
    parser.add_argument(
        "--model",
        action="append",
        choices=("random_forest", "xgboost", *NEURAL_MODELS),
    )
    parser.add_argument(
        "--representation",
        action="append",
        choices=(
            "matched_flow_stats",
            "prefix_stats",
            "flattened_splt",
            "sequential_splt",
        ),
    )
    parser.add_argument("--train-session", action="append", type=int, choices=(1, 2))
    parser.add_argument("--seed", action="append", type=int, choices=(42, 43, 44))
    parser.add_argument("--run-id", action="append")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--maximum-pending-runs", type=int)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    config = load_cross_session_preprocessing_config(
        args.config,
        artifact_dir=args.artifact_dir,
        output_root=args.output_root,
    )
    neural = load_neural_config(
        args.neural_config,
        tuning_output_root=args.tuning_output_root,
    )
    report = run_cross_session_matrix(
        config,
        neural,
        filters=CrossSessionRunFilters(
            families=tuple(args.family or ()),
            models=tuple(args.model or ()),
            representations=tuple(args.representation or ()),
            train_sessions=tuple(args.train_session or ()),
            seeds=tuple(args.seed or ()),
            run_ids=tuple(args.run_id or ()),
        ),
        execute=args.execute,
        maximum_pending_runs=args.maximum_pending_runs,
        device_name=args.device,
        report_path=args.report,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


def dann_contract_main(argv: Sequence[str] | None = None) -> None:
    from vpncat.dann import (
        build_dann_contract,
        load_dann_config,
        validate_dann_contract,
    )

    parser = argparse.ArgumentParser(
        description="Build or validate the frozen DANN experiment contract"
    )
    parser.add_argument("--config", type=Path, default=Path("configs/dann.yaml"))
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--tuning-output-root", type=Path)
    parser.add_argument("--audit-output", type=Path)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    config = load_dann_config(
        args.config,
        artifact_dir=args.artifact_dir,
        output_root=args.output_root,
        tuning_output_root=args.tuning_output_root,
        contract_audit_path=args.audit_output,
    )
    if args.validate:
        if args.force:
            parser.error("--force cannot be combined with --validate")
        report = validate_dann_contract(config)
    else:
        report = build_dann_contract(config, force=args.force)
    print(json.dumps(report, indent=2, sort_keys=True))


def run_dann_main(argv: Sequence[str] | None = None) -> None:
    from vpncat.dann import enumerate_dann_runs, load_dann_config
    from vpncat.dann_runner import run_dann

    parser = argparse.ArgumentParser(description="Run one frozen DANN experiment")
    parser.add_argument("--config", type=Path, default=Path("configs/dann.yaml"))
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--tuning-output-root", type=Path)
    parser.add_argument("--audit-output", type=Path)
    parser.add_argument("--fold", required=True, type=int, choices=(1, 2, 3, 4, 5))
    parser.add_argument("--seed", required=True, type=int, choices=(42, 43, 44))
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    args = parser.parse_args(argv)
    config = load_dann_config(
        args.config,
        artifact_dir=args.artifact_dir,
        output_root=args.output_root,
        tuning_output_root=args.tuning_output_root,
        contract_audit_path=args.audit_output,
    )
    run = next(
        candidate
        for candidate in enumerate_dann_runs(config)
        if candidate.fold == args.fold and candidate.seed == args.seed
    )
    output = run_dann(config, run, device_name=args.device)
    print(json.dumps({"run_id": run.run_id, "output": str(output)}, indent=2))


def dann_matrix_main(argv: Sequence[str] | None = None) -> None:
    from vpncat.dann import load_dann_config
    from vpncat.dann_orchestration import DANNRunFilters, run_dann_matrix

    parser = argparse.ArgumentParser(
        description="Plan, validate, or resume the frozen DANN matrix"
    )
    parser.add_argument("--config", type=Path, default=Path("configs/dann.yaml"))
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--tuning-output-root", type=Path)
    parser.add_argument("--audit-output", type=Path)
    parser.add_argument("--fold", action="append", type=int, choices=(1, 2, 3, 4, 5))
    parser.add_argument("--seed", action="append", type=int, choices=(42, 43, 44))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--maximum-pending-runs", type=int)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    config = load_dann_config(
        args.config,
        artifact_dir=args.artifact_dir,
        output_root=args.output_root,
        tuning_output_root=args.tuning_output_root,
        contract_audit_path=args.audit_output,
    )
    report = run_dann_matrix(
        config,
        filters=DANNRunFilters(
            folds=tuple(args.fold or ()),
            seeds=tuple(args.seed or ()),
        ),
        execute=args.execute,
        maximum_pending_runs=args.maximum_pending_runs,
        device_name=args.device,
        report_path=args.report,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


def ablation_contract_main(argv: Sequence[str] | None = None) -> None:
    from vpncat.ablations import (
        build_ablation_contract,
        load_ablation_config,
        validate_ablation_contract,
    )

    parser = argparse.ArgumentParser(
        description="Build or validate one frozen representation-ablation contract"
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--tuning-output-root", type=Path)
    parser.add_argument("--audit-output", type=Path)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    config = load_ablation_config(
        args.config,
        artifact_dir=args.artifact_dir,
        output_root=args.output_root,
        tuning_output_root=args.tuning_output_root,
        contract_audit_path=args.audit_output,
    )
    if args.validate:
        if args.force:
            parser.error("--force cannot be combined with --validate")
        report = validate_ablation_contract(config)
    else:
        report = build_ablation_contract(config, force=args.force)
    print(json.dumps(report, indent=2, sort_keys=True))


def run_ablation_main(argv: Sequence[str] | None = None) -> None:
    from vpncat.ablation_runner import run_ablation
    from vpncat.ablations import enumerate_ablation_runs, load_ablation_config

    parser = argparse.ArgumentParser(description="Run one frozen ablation experiment")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--tuning-output-root", type=Path)
    parser.add_argument("--audit-output", type=Path)
    parser.add_argument("--model", required=True, choices=("cnn1d", "transformer"))
    parser.add_argument("--observation", required=True)
    parser.add_argument("--fold", required=True, type=int, choices=(1, 2, 3, 4, 5))
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    args = parser.parse_args(argv)
    config = load_ablation_config(
        args.config,
        artifact_dir=args.artifact_dir,
        output_root=args.output_root,
        tuning_output_root=args.tuning_output_root,
        contract_audit_path=args.audit_output,
    )
    matches = [
        run
        for run in enumerate_ablation_runs(config)
        if run.model == args.model
        and run.observation_id == args.observation
        and run.fold == args.fold
    ]
    if len(matches) != 1:
        parser.error("model, observation, and fold do not select exactly one run")
    output = run_ablation(config, matches[0], device_name=args.device)
    print(json.dumps({"run_id": matches[0].run_id, "output": str(output)}, indent=2))


def ablation_matrix_main(argv: Sequence[str] | None = None) -> None:
    from vpncat.ablation_orchestration import AblationRunFilters, run_ablation_matrix
    from vpncat.ablations import load_ablation_config

    parser = argparse.ArgumentParser(
        description="Plan, validate, or resume one frozen ablation matrix"
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--tuning-output-root", type=Path)
    parser.add_argument("--audit-output", type=Path)
    parser.add_argument("--model", action="append", choices=("cnn1d", "transformer"))
    parser.add_argument("--observation", action="append")
    parser.add_argument("--fold", action="append", type=int, choices=(1, 2, 3, 4, 5))
    parser.add_argument("--run-id", action="append")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--maximum-pending-runs", type=int)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    config = load_ablation_config(
        args.config,
        artifact_dir=args.artifact_dir,
        output_root=args.output_root,
        tuning_output_root=args.tuning_output_root,
        contract_audit_path=args.audit_output,
    )
    report = run_ablation_matrix(
        config,
        filters=AblationRunFilters(
            models=tuple(args.model or ()),
            observations=tuple(args.observation or ()),
            folds=tuple(args.fold or ()),
            run_ids=tuple(args.run_id or ()),
        ),
        execute=args.execute,
        maximum_pending_runs=args.maximum_pending_runs,
        device_name=args.device,
        report_path=args.report,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


def analysis_contract_main(argv: Sequence[str] | None = None) -> None:
    from vpncat.analysis import (
        build_analysis_contract,
        load_analysis_config,
        validate_analysis_contract,
    )

    parser = argparse.ArgumentParser(
        description="Build or validate the frozen cross-protocol analysis inventory"
    )
    parser.add_argument("--config", type=Path, default=Path("configs/analysis.yaml"))
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--primary-output-root", type=Path)
    parser.add_argument("--cross-session-output-root", type=Path)
    parser.add_argument("--dann-output-root", type=Path)
    parser.add_argument("--ablation-prefix-output-root", type=Path)
    parser.add_argument("--ablation-channels-output-root", type=Path)
    parser.add_argument("--tuning-output-root", type=Path)
    parser.add_argument("--audit-output", type=Path)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    config = load_analysis_config(
        args.config,
        artifact_dir=args.artifact_dir,
        output_root=args.output_root,
        primary_output_root=args.primary_output_root,
        cross_session_output_root=args.cross_session_output_root,
        dann_output_root=args.dann_output_root,
        ablation_prefix_output_root=args.ablation_prefix_output_root,
        ablation_channels_output_root=args.ablation_channels_output_root,
        tuning_output_root=args.tuning_output_root,
        contract_audit_path=args.audit_output,
    )
    if args.validate:
        if args.force:
            parser.error("--force cannot be combined with --validate")
        report = validate_analysis_contract(config)
    else:
        report = build_analysis_contract(config, force=args.force)
    print(json.dumps(report, indent=2, sort_keys=True))


def aggregate_results_main(argv: Sequence[str] | None = None) -> None:
    from vpncat.aggregate_results import aggregate_results
    from vpncat.analysis import load_analysis_config

    parser = argparse.ArgumentParser(
        description="Validate and aggregate the complete frozen experiment campaign"
    )
    parser.add_argument("--config", type=Path, default=Path("configs/analysis.yaml"))
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--primary-output-root", type=Path)
    parser.add_argument("--cross-session-output-root", type=Path)
    parser.add_argument("--dann-output-root", type=Path)
    parser.add_argument("--ablation-prefix-output-root", type=Path)
    parser.add_argument("--ablation-channels-output-root", type=Path)
    parser.add_argument("--tuning-output-root", type=Path)
    parser.add_argument("--audit-output", type=Path)
    args = parser.parse_args(argv)
    config = load_analysis_config(
        args.config,
        artifact_dir=args.artifact_dir,
        output_root=args.output_root,
        primary_output_root=args.primary_output_root,
        cross_session_output_root=args.cross_session_output_root,
        dann_output_root=args.dann_output_root,
        ablation_prefix_output_root=args.ablation_prefix_output_root,
        ablation_channels_output_root=args.ablation_channels_output_root,
        tuning_output_root=args.tuning_output_root,
        contract_audit_path=args.audit_output,
    )
    output = aggregate_results(config)
    print(json.dumps({"status": "complete", "output": str(output)}, indent=2))
