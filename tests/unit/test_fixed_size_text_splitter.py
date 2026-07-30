import pytest
from app.domain.exceptions.chunk import InvalidChunkConfigError
from app.infrastructure.chunking.fixed_size_text_splitter import FixedSizeTextSplitter


def test_split_empty_text_returns_empty_list() -> None:
    splitter = FixedSizeTextSplitter(chunk_size=10, chunk_overlap=2)

    assert splitter.split("") == []


def test_short_text_becomes_a_single_chunk() -> None:
    splitter = FixedSizeTextSplitter(chunk_size=100, chunk_overlap=10)

    assert splitter.split("short text") == ["short text"]


def test_text_exactly_chunk_size_becomes_a_single_chunk() -> None:
    splitter = FixedSizeTextSplitter(chunk_size=10, chunk_overlap=2)
    text = "0123456789"

    assert splitter.split(text) == [text]


def test_overlap_is_correctly_reflected() -> None:
    splitter = FixedSizeTextSplitter(chunk_size=10, chunk_overlap=3)
    text = "".join(str(i % 10) for i in range(25))

    chunks = splitter.split(text)

    assert chunks == [
        "0123456789",
        "7890123456",
        "4567890123",
        "1234",
    ]
    # Each chunk (except the last) shares its trailing chunk_overlap
    # characters with the following chunk's leading characters.
    for first, second in zip(chunks, chunks[1:], strict=False):
        assert first[-3:] == second[:3]


def test_last_chunk_reaches_end_of_text_without_gaps() -> None:
    splitter = FixedSizeTextSplitter(chunk_size=10, chunk_overlap=3)
    text = "".join(str(i % 10) for i in range(25))

    chunks = splitter.split(text)

    assert "".join(chunks[-1]) == text[-len(chunks[-1]) :]
    assert chunks[-1][-1] == text[-1]


def test_chunk_size_must_be_positive() -> None:
    with pytest.raises(InvalidChunkConfigError):
        FixedSizeTextSplitter(chunk_size=0, chunk_overlap=0)


def test_chunk_overlap_must_not_be_negative() -> None:
    with pytest.raises(InvalidChunkConfigError):
        FixedSizeTextSplitter(chunk_size=10, chunk_overlap=-1)


def test_chunk_overlap_must_be_smaller_than_chunk_size() -> None:
    with pytest.raises(InvalidChunkConfigError):
        FixedSizeTextSplitter(chunk_size=10, chunk_overlap=10)
