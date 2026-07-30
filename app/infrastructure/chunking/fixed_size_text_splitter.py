"""Character-count based implementation of the TextSplitter port."""

from app.domain.exceptions.chunk import InvalidChunkConfigError


class FixedSizeTextSplitter:
    """Splits text into fixed-size, overlapping character windows."""

    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        self._validate(chunk_size, chunk_overlap)
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def split(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self._chunk_size:
            return [text]

        chunks: list[str] = []
        step = self._chunk_size - self._chunk_overlap
        start = 0
        while start < len(text):
            end = min(start + self._chunk_size, len(text))
            chunks.append(text[start:end])
            if end == len(text):
                break
            start += step
        return chunks

    def _validate(self, chunk_size: int, chunk_overlap: int) -> None:
        if chunk_size <= 0:
            raise InvalidChunkConfigError("chunk_size must be a positive integer")
        if chunk_overlap < 0:
            raise InvalidChunkConfigError("chunk_overlap must not be negative")
        if chunk_overlap >= chunk_size:
            raise InvalidChunkConfigError("chunk_overlap must be smaller than chunk_size")
