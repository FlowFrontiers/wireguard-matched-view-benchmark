import pytest

from vpncat.errors import PipelineInvariantError
from vpncat.sequences import parse_splt_triplet


def test_parse_splt_removes_padding_and_truncates() -> None:
    sequence = parse_splt_triplet(
        "[0, 1, 0, -1, -1]",
        "[100, 200, 300, -1, -1]",
        "[0, 5, 7, -1, -1]",
        maximum_length=2,
        context="test",
    )

    assert sequence.direction == [0, 1]
    assert sequence.size == [100.0, 200.0]
    assert sequence.iat_ms == [0.0, 5.0]
    assert sequence.length == 2


def test_parse_splt_rejects_inconsistent_padding() -> None:
    with pytest.raises(PipelineInvariantError, match="padding is inconsistent"):
        parse_splt_triplet(
            "[0, 1, 0]",
            "[100, -1, -1]",
            "[0, -1, -1]",
            maximum_length=50,
            context="test",
        )


def test_parse_splt_rejects_invalid_direction() -> None:
    with pytest.raises(PipelineInvariantError, match="outside"):
        parse_splt_triplet(
            "[0, 2]",
            "[100, 200]",
            "[0, 1]",
            maximum_length=50,
            context="test",
        )


def test_parse_splt_rejects_malformed_values() -> None:
    with pytest.raises(PipelineInvariantError, match="malformed"):
        parse_splt_triplet(
            "[0, 1]",
            "[100, broken]",
            "[0, 1]",
            maximum_length=50,
            context="test",
        )
