from __future__ import annotations

from vpncat.ablations import AblationConfig, AblationRunSpec, enumerate_ablation_runs
from vpncat.errors import PipelineInvariantError
from vpncat.neural_data import PreparedNeuralRun, prepare_neural_run


def prepare_ablation_run(
    config: AblationConfig,
    run: AblationRunSpec,
) -> PreparedNeuralRun:
    """Materialize one executable ablation using the frozen primary split."""
    if run.is_primary_reference:
        raise PipelineInvariantError("Primary reference ablations must not be retrained")
    matches = [
        candidate
        for candidate in enumerate_ablation_runs(config)
        if candidate.run_id == run.run_id
    ]
    if len(matches) != 1 or matches[0].to_dict() != run.to_dict():
        raise PipelineInvariantError("Ablation run differs from the frozen matrix")
    if (
        run.protocol != config.protocol
        or run.model not in config.models
        or run.fold not in config.folds
        or run.seed != config.seed
        or run.train_domain != config.train_domain
    ):
        raise PipelineInvariantError("Ablation run is incompatible with data contract")
    return prepare_neural_run(
        config.primary.canonical_path,
        config.primary.split_path,
        run,
        prefix_length=run.prefix_length,
        channels=run.channels,
    )
