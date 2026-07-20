from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vpncat.errors import PipelineInvariantError


@dataclass(frozen=True)
class SPLTSequence:
    direction: list[int]
    size: list[float]
    iat_ms: list[float]

    @property
    def length(self) -> int:
        return len(self.size)


def _parse_numeric_array(value: object, *, field: str) -> np.ndarray:
    if not isinstance(value, str) or not value.strip():
        raise PipelineInvariantError(f"{field} is empty or not a string")
    normalized = value.strip().removeprefix("[").removesuffix("]").replace(",", " ")
    try:
        array = np.fromstring(normalized, sep=" ", dtype=np.float64)
    except ValueError as error:
        raise PipelineInvariantError(f"{field} contains malformed values") from error
    if array.size == 0:
        raise PipelineInvariantError(f"{field} contains no numeric values")
    if array.size != len(normalized.split()) or not np.isfinite(array).all():
        raise PipelineInvariantError(f"{field} contains malformed or non-finite values")
    return array


def parse_splt_triplet(
    direction_value: object,
    size_value: object,
    iat_value: object,
    *,
    maximum_length: int,
    context: str,
) -> SPLTSequence:
    """Parse one SPLT triplet, remove -1 padding, and truncate deterministically."""
    direction = _parse_numeric_array(direction_value, field=f"{context}.direction")
    size = _parse_numeric_array(size_value, field=f"{context}.size")
    iat = _parse_numeric_array(iat_value, field=f"{context}.iat_ms")

    if not (len(direction) == len(size) == len(iat)):
        raise PipelineInvariantError(
            f"{context} SPLT arrays have unequal lengths: "
            f"{len(direction)}, {len(size)}, {len(iat)}"
        )

    padding_positions = np.flatnonzero(size == -1)
    valid_length = int(padding_positions[0]) if padding_positions.size else len(size)
    if valid_length == 0:
        raise PipelineInvariantError(f"{context} contains no valid packets")

    if padding_positions.size:
        if not np.all(size[valid_length:] == -1):
            raise PipelineInvariantError(f"{context}.size contains data after padding begins")
        if not np.all(direction[valid_length:] == -1):
            raise PipelineInvariantError(f"{context}.direction padding is inconsistent")
        if not np.all(iat[valid_length:] == -1):
            raise PipelineInvariantError(f"{context}.iat_ms padding is inconsistent")

    limit = min(valid_length, maximum_length)
    valid_direction = direction[:limit]
    valid_size = size[:limit]
    valid_iat = iat[:limit]

    if not np.all(np.isin(valid_direction, (0, 1))):
        raise PipelineInvariantError(f"{context}.direction contains values outside {{0, 1}}")
    if np.any(valid_size < 0) or np.any(valid_iat < 0):
        raise PipelineInvariantError(f"{context} contains negative non-padding values")

    return SPLTSequence(
        direction=valid_direction.astype(np.int8).tolist(),
        size=valid_size.astype(np.float32).tolist(),
        iat_ms=valid_iat.astype(np.float32).tolist(),
    )
