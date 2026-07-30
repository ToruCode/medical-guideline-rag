"""Abstract interface for splitting raw text into chunks."""

from typing import Protocol


class TextSplitter(Protocol):
    """Splits a single string of text into an ordered list of fragments."""

    def split(self, text: str) -> list[str]: ...
