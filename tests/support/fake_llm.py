"""Deterministic test-only Llm implementations.

Never call a real model or perform network access; used to test
Llm-dependent code without any external API or credentials.
"""


class FakeLlm:
    def __init__(self, answer: str = "fake answer") -> None:
        self._answer = answer
        self.received_prompt: str | None = None

    def generate(self, prompt: str) -> str:
        self.received_prompt = prompt
        return self._answer


class RaisingLlm:
    def __init__(self, error: Exception) -> None:
        self._error = error

    def generate(self, prompt: str) -> str:
        raise self._error
