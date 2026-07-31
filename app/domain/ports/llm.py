"""Abstract interface for generating a text completion from a prompt."""

from typing import Protocol


class Llm(Protocol):
    """Generates a text completion for a single, fully-composed prompt."""

    def generate(self, prompt: str) -> str: ...
