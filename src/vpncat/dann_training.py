from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from vpncat.dann import DANN_HISTORY_COLUMNS
from vpncat.dann_data import UnlabeledNeuralSubset
from vpncat.errors import PipelineInvariantError
from vpncat.models.dann import DANNClassifier, logistic_grl_coefficient
from vpncat.models.neural import trainable_parameter_count
from vpncat.neural_data import NeuralSubset
from vpncat.neural_training import resolve_device, seed_neural_execution
from vpncat.preprocessing import FoldTargetState


@dataclass(frozen=True)
class DANNTrainingResult:
    model: DANNClassifier
    history: pd.DataFrame
    best_epoch: int
    best_validation_macro_f1: float
    validation_loss_at_best_epoch: float
    epochs_completed: int
    parameter_count: int
    backbone_parameter_count: int
    device: str


def build_paired_dann_loader(
    source: NeuralSubset,
    adaptation: UnlabeledNeuralSubset,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    workers: int,
) -> DataLoader:
    """Build one loader whose shuffle keeps matched source/target rows together."""
    if batch_size < 1 or workers != 0:
        raise PipelineInvariantError("Frozen DANN loaders require positive batches and 0 workers")
    if source.pair_ids != adaptation.pair_ids or not np.array_equal(
        source.positions, adaptation.positions
    ):
        raise PipelineInvariantError("DANN source/adaptation pairing or order differs")
    if source.values.shape != adaptation.values.shape or source.mask.shape != adaptation.mask.shape:
        raise PipelineInvariantError("DANN source/adaptation tensor shapes differ")
    dataset = TensorDataset(
        torch.from_numpy(source.values),
        torch.from_numpy(source.mask),
        torch.from_numpy(source.targets),
        torch.from_numpy(adaptation.values),
        torch.from_numpy(adaptation.mask),
        torch.arange(len(source.pair_ids), dtype=torch.int64),
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        drop_last=False,
        generator=generator,
        pin_memory=False,
    )


def _labeled_loader(
    subset: NeuralSubset,
    *,
    batch_size: int,
    seed: int,
    workers: int,
) -> DataLoader:
    if batch_size < 1 or workers != 0:
        raise PipelineInvariantError("Frozen DANN loaders require positive batches and 0 workers")
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        TensorDataset(
            torch.from_numpy(subset.values),
            torch.from_numpy(subset.mask),
            torch.from_numpy(subset.targets),
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        drop_last=False,
        generator=generator,
        pin_memory=False,
    )


def _evaluate_source(
    model: DANNClassifier,
    loader: DataLoader,
    criterion: nn.CrossEntropyLoss,
    *,
    device: torch.device,
    class_count: int,
    class_weights: Tensor,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_weight = 0.0
    targets: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for values, mask, batch_targets in loader:
            values = values.to(device=device, dtype=torch.float32)
            mask = mask.to(device=device, dtype=torch.bool)
            batch_targets = batch_targets.to(device=device, dtype=torch.int64)
            logits = model.classify(values, mask)
            losses = criterion(logits, batch_targets)
            total_loss += float(losses.sum().cpu())
            total_weight += float(class_weights[batch_targets].sum().cpu())
            targets.append(batch_targets.cpu().numpy())
            predictions.append(torch.argmax(logits, dim=1).cpu().numpy())
    if total_weight <= 0.0:
        raise PipelineInvariantError("DANN source validation loader is empty")
    macro_f1 = f1_score(
        np.concatenate(targets),
        np.concatenate(predictions),
        labels=np.arange(class_count),
        average="macro",
        zero_division=0,
    )
    return total_loss / total_weight, float(macro_f1)


def _validate_training_contract(
    training: NeuralSubset,
    adaptation: UnlabeledNeuralSubset,
    validation: NeuralSubset,
    *,
    learning_rate: float,
    domain_loss_weight: float,
    gradient_reversal: dict[str, Any],
    optimizer_policy: dict[str, Any],
    training_policy: dict[str, Any],
) -> None:
    if set(training.pair_ids) & set(validation.pair_ids):
        raise PipelineInvariantError("DANN trainer received overlapping train/validation pairs")
    if training.pair_ids != adaptation.pair_ids or not np.array_equal(
        training.positions, adaptation.positions
    ):
        raise PipelineInvariantError("DANN trainer received mispaired adaptation data")
    if len(set(training.pair_ids)) != len(training.pair_ids):
        raise PipelineInvariantError("DANN training pair identities are not unique")
    if learning_rate <= 0.0 or domain_loss_weight != 1.0:
        raise PipelineInvariantError("DANN learning rate or domain-loss weight differs")
    if optimizer_policy.get("name") != "adamw":
        raise PipelineInvariantError("DANN trainer requires AdamW")
    if gradient_reversal != {
        "schedule": "logistic",
        "gamma": 10.0,
        "start": 0.0,
        "end": 1.0,
    }:
        raise PipelineInvariantError("DANN gradient-reversal policy differs from the freeze")
    if (
        training_policy.get("mixed_precision") is not False
        or training_policy.get("deterministic_algorithms") is not True
        or training_policy.get("training_shuffle") is not True
        or training_policy.get("drop_last") is not False
    ):
        raise PipelineInvariantError("DANN determinism or loader policy differs from the freeze")


def train_dann_model(
    model: DANNClassifier,
    training: NeuralSubset,
    adaptation: UnlabeledNeuralSubset,
    validation: NeuralSubset,
    state: FoldTargetState,
    *,
    learning_rate: float,
    batch_size: int,
    seed: int,
    domain_loss_weight: float,
    gradient_reversal: dict[str, Any],
    optimizer_policy: dict[str, Any],
    training_policy: dict[str, Any],
    device_name: str = "auto",
) -> DANNTrainingResult:
    """Train DANN and select checkpoints using source validation macro F1 only."""
    _validate_training_contract(
        training,
        adaptation,
        validation,
        learning_rate=learning_rate,
        domain_loss_weight=domain_loss_weight,
        gradient_reversal=gradient_reversal,
        optimizer_policy=optimizer_policy,
        training_policy=training_policy,
    )
    seed_neural_execution(seed)
    device = resolve_device(device_name)
    model = model.to(device)
    workers = int(training_policy["data_loader_workers"])
    paired_loader = build_paired_dann_loader(
        training,
        adaptation,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        workers=workers,
    )
    validation_loader = _labeled_loader(
        validation,
        batch_size=batch_size,
        seed=seed,
        workers=workers,
    )
    class_weights = torch.as_tensor(state.class_weights, dtype=torch.float32, device=device)
    classification_criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        reduction="none",
    )
    domain_criterion = nn.BCEWithLogitsLoss(reduction="none")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=tuple(float(value) for value in optimizer_policy["betas"]),
        eps=float(optimizer_policy["epsilon"]),
        amsgrad=bool(optimizer_policy["amsgrad"]),
        weight_decay=float(optimizer_policy["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(training_policy["scheduler_factor"]),
        patience=int(training_policy["scheduler_patience"]),
        threshold=float(training_policy["scheduler_threshold"]),
        threshold_mode=str(training_policy["scheduler_threshold_mode"]),
        cooldown=int(training_policy["scheduler_cooldown"]),
        min_lr=float(training_policy["minimum_learning_rate"]),
        eps=float(training_policy["scheduler_epsilon"]),
    )

    maximum_epochs = int(training_policy["maximum_epochs"])
    maximum_steps = maximum_epochs * len(paired_loader)
    if maximum_steps < 1:
        raise PipelineInvariantError("DANN paired training loader is empty")
    denominator = max(maximum_steps - 1, 1)
    best_score = -np.inf
    best_loss = np.inf
    best_epoch = 0
    best_state: dict[str, Tensor] | None = None
    stale_epochs = 0
    global_step = 0
    history_rows: list[list[float | int]] = []
    stopping_patience = int(training_policy["early_stopping_patience"])
    stopping_delta = float(training_policy["early_stopping_min_delta"])
    clip_norm = float(optimizer_policy["gradient_clip_norm"])

    for epoch in range(1, maximum_epochs + 1):
        model.train()
        classification_sum = 0.0
        classification_weight = 0.0
        domain_sum = 0.0
        domain_count = 0
        objective_sum = 0.0
        objective_steps = 0
        coefficients: list[float] = []
        learning_rate_used = float(optimizer.param_groups[0]["lr"])
        for source_values, source_mask, targets, target_values, target_mask, _ in paired_loader:
            source_values = source_values.to(device=device, dtype=torch.float32)
            source_mask = source_mask.to(device=device, dtype=torch.bool)
            targets = targets.to(device=device, dtype=torch.int64)
            target_values = target_values.to(device=device, dtype=torch.float32)
            target_mask = target_mask.to(device=device, dtype=torch.bool)
            coefficient = logistic_grl_coefficient(
                global_step / denominator,
                gamma=float(gradient_reversal["gamma"]),
                start=float(gradient_reversal["start"]),
                end=float(gradient_reversal["end"]),
            )
            coefficients.append(coefficient)
            values = torch.cat((source_values, target_values), dim=0)
            mask = torch.cat((source_mask, target_mask), dim=0)
            optimizer.zero_grad(set_to_none=True)
            class_logits, domain_logits = model(
                values,
                mask,
                grl_coefficient=coefficient,
            )
            source_count = len(targets)
            classification_losses = classification_criterion(
                class_logits[:source_count], targets
            )
            batch_classification_weight = class_weights[targets].sum()
            classification_loss = classification_losses.sum() / batch_classification_weight
            domain_targets = torch.cat(
                (
                    torch.zeros(source_count, device=device),
                    torch.ones(source_count, device=device),
                )
            )
            domain_losses = domain_criterion(domain_logits, domain_targets)
            domain_loss = domain_losses.mean()
            total_loss = classification_loss + domain_loss_weight * domain_loss
            total_loss.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=clip_norm,
                error_if_nonfinite=True,
            )
            optimizer.step()
            classification_sum += float(classification_losses.sum().detach().cpu())
            classification_weight += float(batch_classification_weight.detach().cpu())
            domain_sum += float(domain_losses.sum().detach().cpu())
            domain_count += int(domain_losses.numel())
            objective_sum += float(total_loss.detach().cpu())
            objective_steps += 1
            global_step += 1

        validation_loss, validation_macro_f1 = _evaluate_source(
            model,
            validation_loader,
            classification_criterion,
            device=device,
            class_count=len(state.classes),
            class_weights=class_weights,
        )
        classification_loss_epoch = classification_sum / classification_weight
        domain_loss_epoch = domain_sum / domain_count
        history_rows.append(
            [
                epoch,
                classification_loss_epoch,
                domain_loss_epoch,
                objective_sum / objective_steps,
                validation_loss,
                validation_macro_f1,
                learning_rate_used,
                coefficients[0],
                coefficients[-1],
            ]
        )
        if validation_macro_f1 > best_score + stopping_delta:
            best_score = validation_macro_f1
            best_loss = validation_loss
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
        scheduler.step(validation_loss)
        if stale_epochs >= stopping_patience:
            break

    if best_state is None or best_epoch == 0:
        raise PipelineInvariantError("DANN training did not produce a best checkpoint")
    model.load_state_dict(best_state)
    history = pd.DataFrame(history_rows, columns=DANN_HISTORY_COLUMNS)
    if not np.isfinite(history.loc[:, DANN_HISTORY_COLUMNS].to_numpy(dtype=float)).all():
        raise PipelineInvariantError("DANN training history contains non-finite values")
    return DANNTrainingResult(
        model=model,
        history=history,
        best_epoch=best_epoch,
        best_validation_macro_f1=float(best_score),
        validation_loss_at_best_epoch=float(best_loss),
        epochs_completed=len(history),
        parameter_count=trainable_parameter_count(model),
        backbone_parameter_count=trainable_parameter_count(model.backbone),
        device=str(device),
    )


def predict_dann_probabilities(
    model: DANNClassifier,
    subset: NeuralSubset,
    *,
    batch_size: int,
    class_count: int,
    device_name: str,
) -> np.ndarray:
    """Predict classification probabilities without executing the domain head."""
    device = resolve_device(device_name)
    model = model.to(device)
    loader = _labeled_loader(subset, batch_size=batch_size, seed=0, workers=0)
    model.eval()
    batches: list[np.ndarray] = []
    with torch.no_grad():
        for values, mask, _ in loader:
            values = values.to(device=device, dtype=torch.float32)
            mask = mask.to(device=device, dtype=torch.bool)
            probabilities = torch.softmax(model.classify(values, mask), dim=1)
            batches.append(probabilities.cpu().numpy().astype(np.float64))
    if not batches:
        raise PipelineInvariantError("DANN prediction subset is empty")
    probabilities = np.concatenate(batches, axis=0)
    if probabilities.shape != (len(subset.pair_ids), class_count):
        raise PipelineInvariantError("DANN probability matrix shape is invalid")
    if not np.isfinite(probabilities).all() or not np.allclose(
        probabilities.sum(axis=1), 1.0, rtol=1e-6, atol=1e-8
    ):
        raise PipelineInvariantError("DANN probabilities are invalid")
    return probabilities
