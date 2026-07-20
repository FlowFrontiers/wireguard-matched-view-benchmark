from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from vpncat.errors import PipelineInvariantError
from vpncat.neural_config import FROZEN_TOPOLOGIES


def _validate_inputs(values: Tensor, mask: Tensor, *, feature_count: int) -> Tensor:
    if values.ndim != 3 or values.shape[2] != feature_count:
        raise PipelineInvariantError("Neural input tensor has an invalid shape")
    if mask.shape != values.shape[:2]:
        raise PipelineInvariantError("Neural mask shape disagrees with input tensor")
    mask = mask.to(dtype=torch.bool, device=values.device)
    if not torch.all(mask.any(dim=1)):
        raise PipelineInvariantError("Every neural sequence must contain an observed packet")
    if not torch.isfinite(values).all():
        raise PipelineInvariantError("Neural input tensor contains non-finite values")
    return mask


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    weights = mask.unsqueeze(-1).to(dtype=values.dtype)
    return (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def _mask_channels(values: Tensor, mask: Tensor) -> Tensor:
    return values * mask.unsqueeze(1).to(dtype=values.dtype)


class CNN1DClassifier(nn.Module):
    """Pure multi-scale convolutional baseline with mask-safe residual processing."""

    def __init__(
        self,
        feature_count: int,
        class_count: int,
        *,
        width: int,
        dropout: float,
        kernels: tuple[int, ...] = (3, 7, 11),
    ) -> None:
        super().__init__()
        if not kernels or any(kernel % 2 == 0 for kernel in kernels):
            raise PipelineInvariantError("CNN1D kernels must be non-empty and odd")
        self.feature_count = feature_count
        self.backbone_width = width
        self.dropout_probability = float(dropout)
        self.embedding_width = width * 2
        self.branches = nn.ModuleList(
            nn.Conv1d(feature_count, width, kernel, padding=kernel // 2)
            for kernel in kernels
        )
        channels = width * len(kernels)
        self.branch_norm = nn.LayerNorm(channels)
        self.residual_conv = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.residual_norm = nn.LayerNorm(channels)
        self.final_conv = nn.Conv1d(channels, self.embedding_width, kernel_size=3, padding=1)
        self.final_norm = nn.LayerNorm(self.embedding_width)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(self.embedding_width, width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width, class_count),
        )

    @staticmethod
    def _normalize(values: Tensor, normalization: nn.LayerNorm) -> Tensor:
        return normalization(values.transpose(1, 2)).transpose(1, 2)

    def encode(self, values: Tensor, mask: Tensor) -> Tensor:
        """Return the mask-safe pooled representation used by the classifier."""
        mask = _validate_inputs(values, mask, feature_count=self.feature_count)
        values = values * mask.unsqueeze(-1).to(dtype=values.dtype)
        channels = values.transpose(1, 2)
        channels = torch.cat([branch(channels) for branch in self.branches], dim=1)
        channels = self.activation(self._normalize(channels, self.branch_norm))
        channels = _mask_channels(channels, mask)

        residual = self.residual_conv(channels)
        residual = self.activation(self._normalize(residual, self.residual_norm))
        residual = _mask_channels(self.dropout(residual), mask)
        channels = _mask_channels(self.activation(channels + residual), mask)

        channels = self.final_conv(channels)
        channels = self.activation(self._normalize(channels, self.final_norm))
        channels = _mask_channels(channels, mask).transpose(1, 2)
        return _masked_mean(channels, mask)

    def forward(self, values: Tensor, mask: Tensor) -> Tensor:
        return self.classifier(self.encode(values, mask))


class LSTMClassifier(nn.Module):
    """Two-layer unidirectional recurrent baseline without an attention block."""

    def __init__(
        self,
        feature_count: int,
        class_count: int,
        *,
        width: int,
        dropout: float,
        layers: int = 2,
    ) -> None:
        super().__init__()
        self.feature_count = feature_count
        self.lstm = nn.LSTM(
            input_size=feature_count,
            hidden_size=width,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
            bidirectional=False,
        )
        self.normalization = nn.LayerNorm(width)
        self.classifier = nn.Sequential(
            nn.Linear(width, width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width, class_count),
        )

    def forward(self, values: Tensor, mask: Tensor) -> Tensor:
        mask = _validate_inputs(values, mask, feature_count=self.feature_count)
        values = values * mask.unsqueeze(-1).to(dtype=values.dtype)
        lengths = mask.sum(dim=1).to(dtype=torch.int64).cpu()
        packed = pack_padded_sequence(
            values,
            lengths,
            batch_first=True,
            enforce_sorted=False,
        )
        packed_output, _ = self.lstm(packed)
        output, _ = pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=values.shape[1],
        )
        output = self.normalization(output)
        return self.classifier(_masked_mean(output, mask))


class TransformerClassifier(nn.Module):
    """Masked Transformer encoder baseline with learnable packet positions."""

    def __init__(
        self,
        feature_count: int,
        class_count: int,
        *,
        width: int,
        dropout: float,
        maximum_length: int,
        layers: int = 2,
        attention_heads: int = 4,
        feedforward_multiplier: int = 4,
    ) -> None:
        super().__init__()
        if width % attention_heads:
            raise PipelineInvariantError("Transformer width must be divisible by head count")
        self.feature_count = feature_count
        self.maximum_length = maximum_length
        self.input_projection = nn.Linear(feature_count, width)
        self.position_embedding = nn.Parameter(torch.empty(maximum_length, width))
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=attention_heads,
            dim_feedforward=width * feedforward_multiplier,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=layers,
            norm=nn.LayerNorm(width),
            enable_nested_tensor=False,
        )
        self.classifier = nn.Sequential(
            nn.Linear(width, width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width, class_count),
        )

    def forward(self, values: Tensor, mask: Tensor) -> Tensor:
        mask = _validate_inputs(values, mask, feature_count=self.feature_count)
        if values.shape[1] > self.maximum_length:
            raise PipelineInvariantError("Transformer input exceeds frozen maximum length")
        values = values * mask.unsqueeze(-1).to(dtype=values.dtype)
        encoded = self.input_projection(values)
        encoded = encoded + self.position_embedding[: values.shape[1]].unsqueeze(0)
        encoded = self.encoder(encoded, src_key_padding_mask=~mask)
        return self.classifier(_masked_mean(encoded, mask))


def build_neural_model(
    model_name: str,
    *,
    feature_count: int,
    class_count: int,
    width: int,
    dropout: float,
    maximum_length: int,
    topology: dict[str, Any],
) -> nn.Module:
    if topology != FROZEN_TOPOLOGIES.get(model_name):
        raise PipelineInvariantError(f"{model_name} topology differs from the freeze")
    if model_name == "cnn1d":
        return CNN1DClassifier(
            feature_count,
            class_count,
            width=width,
            dropout=dropout,
            kernels=tuple(topology["kernels"]),
        )
    if model_name == "lstm":
        return LSTMClassifier(
            feature_count,
            class_count,
            width=width,
            dropout=dropout,
            layers=int(topology["layers"]),
        )
    if model_name == "transformer":
        return TransformerClassifier(
            feature_count,
            class_count,
            width=width,
            dropout=dropout,
            maximum_length=maximum_length,
            layers=int(topology["layers"]),
            attention_heads=int(topology["attention_heads"]),
            feedforward_multiplier=int(topology["feedforward_multiplier"]),
        )
    raise PipelineInvariantError(f"Unsupported neural model: {model_name}")


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
