from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from vpncat.artifacts import HISTORY_COLUMNS
from vpncat.errors import PipelineInvariantError
from vpncat.models.neural import trainable_parameter_count
from vpncat.neural_data import NeuralSubset
from vpncat.preprocessing import FoldTargetState


@dataclass(frozen=True)
class TrainingResult:
    model: nn.Module
    history: pd.DataFrame
    best_epoch: int
    best_validation_macro_f1: float
    validation_loss_at_best_epoch: float
    epochs_completed: int
    parameter_count: int
    device: str


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise PipelineInvariantError("CUDA was requested but is not available")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise PipelineInvariantError("MPS was requested but is not available")
    if requested not in {"cpu", "cuda", "mps"}:
        raise PipelineInvariantError(f"Unsupported neural device: {requested}")
    return torch.device(requested)


def seed_neural_execution(seed: int) -> None:
    if seed < 0:
        raise PipelineInvariantError("Neural seed must be nonnegative")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _loader(
    subset: NeuralSubset,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    workers: int,
) -> DataLoader:
    if batch_size < 1 or workers != 0:
        raise PipelineInvariantError("Frozen neural loaders require positive batches and 0 workers")
    dataset = TensorDataset(
        torch.from_numpy(subset.values),
        torch.from_numpy(subset.mask),
        torch.from_numpy(subset.targets),
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


def _evaluate(
    model: nn.Module,
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
            logits = model(values, mask)
            losses = criterion(logits, batch_targets)
            total_loss += float(losses.sum().detach().cpu())
            total_weight += float(class_weights[batch_targets].sum().detach().cpu())
            targets.append(batch_targets.detach().cpu().numpy())
            predictions.append(torch.argmax(logits, dim=1).detach().cpu().numpy())
    if total_weight <= 0:
        raise PipelineInvariantError("Neural validation loader is empty")
    y_true = np.concatenate(targets)
    y_pred = np.concatenate(predictions)
    macro_f1 = f1_score(
        y_true,
        y_pred,
        labels=np.arange(class_count),
        average="macro",
        zero_division=0,
    )
    return total_loss / total_weight, float(macro_f1)


def train_neural_model(
    model: nn.Module,
    training: NeuralSubset,
    validation: NeuralSubset,
    state: FoldTargetState,
    *,
    learning_rate: float,
    batch_size: int,
    seed: int,
    optimizer_policy: dict[str, Any],
    training_policy: dict[str, Any],
    device_name: str = "auto",
) -> TrainingResult:
    """Train from source train/validation subsets and restore the best macro-F1 epoch."""
    if set(training.pair_ids) & set(validation.pair_ids):
        raise PipelineInvariantError("Neural trainer received overlapping train/validation pairs")
    if learning_rate <= 0:
        raise PipelineInvariantError("Neural learning rate must be positive")
    if optimizer_policy.get("name") != "adamw":
        raise PipelineInvariantError("Neural trainer requires AdamW")
    if training_policy.get("mixed_precision") is not False:
        raise PipelineInvariantError("Mixed precision is prohibited by the neural freeze")
    if (
        training_policy.get("deterministic_algorithms") is not True
        or training_policy.get("training_shuffle") is not True
        or training_policy.get("drop_last") is not False
    ):
        raise PipelineInvariantError("Neural determinism or loader policy differs from the freeze")

    seed_neural_execution(seed)
    device = resolve_device(device_name)
    model = model.to(device)
    workers = int(training_policy["data_loader_workers"])
    training_loader = _loader(
        training,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        workers=workers,
    )
    validation_loader = _loader(
        validation,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
        workers=workers,
    )
    class_weights = torch.as_tensor(
        state.class_weights,
        dtype=torch.float32,
        device=device,
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights, reduction="none")
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

    best_score = -np.inf
    best_loss = np.inf
    best_epoch = 0
    best_state: dict[str, Tensor] | None = None
    stale_epochs = 0
    history_rows: list[list[float | int]] = []
    maximum_epochs = int(training_policy["maximum_epochs"])
    stopping_patience = int(training_policy["early_stopping_patience"])
    stopping_delta = float(training_policy["early_stopping_min_delta"])
    clip_norm = float(optimizer_policy["gradient_clip_norm"])

    for epoch in range(1, maximum_epochs + 1):
        model.train()
        training_loss = 0.0
        training_weight = 0.0
        learning_rate_used = float(optimizer.param_groups[0]["lr"])
        for values, mask, targets in training_loader:
            values = values.to(device=device, dtype=torch.float32)
            mask = mask.to(device=device, dtype=torch.bool)
            targets = targets.to(device=device, dtype=torch.int64)
            optimizer.zero_grad(set_to_none=True)
            logits = model(values, mask)
            losses = criterion(logits, targets)
            batch_weight = class_weights[targets].sum()
            (losses.sum() / batch_weight).backward()
            nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=clip_norm,
                error_if_nonfinite=True,
            )
            optimizer.step()
            training_loss += float(losses.sum().detach().cpu())
            training_weight += float(batch_weight.detach().cpu())

        validation_loss, validation_macro_f1 = _evaluate(
            model,
            validation_loader,
            criterion,
            device=device,
            class_count=len(state.classes),
            class_weights=class_weights,
        )
        history_rows.append(
            [
                epoch,
                training_loss / training_weight,
                validation_loss,
                validation_macro_f1,
                learning_rate_used,
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
        raise PipelineInvariantError("Neural training did not produce a best checkpoint")
    model.load_state_dict(best_state)
    history = pd.DataFrame(history_rows, columns=HISTORY_COLUMNS)
    if not np.isfinite(history.loc[:, HISTORY_COLUMNS].to_numpy(dtype=float)).all():
        raise PipelineInvariantError("Neural training history contains non-finite values")
    return TrainingResult(
        model=model,
        history=history,
        best_epoch=best_epoch,
        best_validation_macro_f1=float(best_score),
        validation_loss_at_best_epoch=float(best_loss),
        epochs_completed=len(history),
        parameter_count=trainable_parameter_count(model),
        device=str(device),
    )


def predict_neural_probabilities(
    model: nn.Module,
    subset: NeuralSubset,
    *,
    batch_size: int,
    class_count: int,
    device_name: str,
) -> np.ndarray:
    """Predict one subset in canonical order and apply softmax exactly once."""
    device = resolve_device(device_name)
    model = model.to(device)
    loader = _loader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        seed=0,
        workers=0,
    )
    model.eval()
    batches: list[np.ndarray] = []
    with torch.no_grad():
        for values, mask, _ in loader:
            values = values.to(device=device, dtype=torch.float32)
            mask = mask.to(device=device, dtype=torch.bool)
            logits = model(values, mask)
            probabilities = torch.softmax(logits, dim=1)
            batches.append(probabilities.detach().cpu().numpy().astype(np.float64))
    if not batches:
        raise PipelineInvariantError("Neural prediction subset is empty")
    probabilities = np.concatenate(batches, axis=0)
    if probabilities.shape != (len(subset.pair_ids), class_count):
        raise PipelineInvariantError("Neural probability matrix shape is invalid")
    if not np.isfinite(probabilities).all():
        raise PipelineInvariantError("Neural probabilities contain non-finite values")
    if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=1e-6, atol=1e-8):
        raise PipelineInvariantError("Neural probabilities do not sum to one")
    return probabilities
