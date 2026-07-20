from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn

from vpncat.errors import PipelineInvariantError
from vpncat.models.neural import CNN1DClassifier, build_neural_model

FROZEN_DOMAIN_HEAD = {
    "hidden_layers": 1,
    "hidden_width": "backbone_width",
    "activation": "gelu",
    "dropout": "selected_backbone_dropout",
}


class _GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, values: Tensor, coefficient: float) -> Tensor:
        ctx.coefficient = coefficient
        return values.view_as(values)

    @staticmethod
    def backward(ctx: Any, gradient: Tensor) -> tuple[Tensor, None]:
        return -ctx.coefficient * gradient, None


def gradient_reverse(values: Tensor, coefficient: float) -> Tensor:
    if not math.isfinite(coefficient) or not 0.0 <= coefficient <= 1.0:
        raise PipelineInvariantError("Gradient-reversal coefficient must be in [0, 1]")
    return _GradientReversal.apply(values, coefficient)


def logistic_grl_coefficient(
    progress: float,
    *,
    gamma: float,
    start: float,
    end: float,
) -> float:
    """Return the standard DANN logistic schedule at normalized progress.

    The configured end value is the asymptotic scale; with gamma 10, progress 1
    yields approximately 0.99991 rather than mathematically reaching 1.
    """
    if (
        not all(math.isfinite(value) for value in (progress, gamma, start, end))
        or not 0.0 <= progress <= 1.0
        or gamma <= 0.0
        or not 0.0 <= start < end <= 1.0
    ):
        raise PipelineInvariantError("Gradient-reversal schedule arguments are invalid")
    ramp = 2.0 / (1.0 + math.exp(-gamma * progress)) - 1.0
    return start + (end - start) * ramp


class DANNClassifier(nn.Module):
    """Frozen CNN1D classifier with a gradient-reversed binary domain head."""

    def __init__(
        self,
        backbone: CNN1DClassifier,
        *,
        width: int,
        dropout: float,
        domain_head: dict[str, Any],
    ) -> None:
        super().__init__()
        if domain_head != FROZEN_DOMAIN_HEAD:
            raise PipelineInvariantError("DANN domain head differs from the freeze")
        if width != backbone.backbone_width or dropout != backbone.dropout_probability:
            raise PipelineInvariantError("DANN domain head does not match the backbone selection")
        self.backbone = backbone
        self.domain_head = nn.Sequential(
            nn.Linear(backbone.embedding_width, width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width, 1),
        )

    def encode(self, values: Tensor, mask: Tensor) -> Tensor:
        return self.backbone.encode(values, mask)

    def classify_features(self, features: Tensor) -> Tensor:
        return self.backbone.classifier(features)

    def classify(self, values: Tensor, mask: Tensor) -> Tensor:
        return self.classify_features(self.encode(values, mask))

    def forward(
        self,
        values: Tensor,
        mask: Tensor,
        *,
        grl_coefficient: float,
    ) -> tuple[Tensor, Tensor]:
        features = self.encode(values, mask)
        class_logits = self.classify_features(features)
        domain_logits = self.domain_head(
            gradient_reverse(features, grl_coefficient)
        ).squeeze(1)
        return class_logits, domain_logits


def build_dann_model(
    *,
    feature_count: int,
    class_count: int,
    width: int,
    dropout: float,
    maximum_length: int,
    topology: dict[str, Any],
    domain_head: dict[str, Any],
) -> DANNClassifier:
    backbone = build_neural_model(
        "cnn1d",
        feature_count=feature_count,
        class_count=class_count,
        width=width,
        dropout=dropout,
        maximum_length=maximum_length,
        topology=topology,
    )
    if not isinstance(backbone, CNN1DClassifier):
        raise PipelineInvariantError("DANN requires the frozen CNN1D backbone")
    return DANNClassifier(
        backbone,
        width=width,
        dropout=dropout,
        domain_head=domain_head,
    )
