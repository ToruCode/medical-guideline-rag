"""Deterministic, dependency-free Llm implementations.

Never call a real model or perform network access, and require no API
key. Used both by tests and by the FastAPI app's default dependency
wiring (app/api/dependencies.py) as a stand-in until a real LLM adapter
is implemented; see docs/adr/0009-generation-strategy.md.
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
