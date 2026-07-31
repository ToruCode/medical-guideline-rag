"""Unit tests for OpenAiLlm.

openai.OpenAI is monkeypatched with a fake client, so these tests make
no real network requests and need no real API key.
"""

from typing import Any

import pytest
from app.infrastructure.llm.openai_llm import OpenAiLlm


class _FakeMessage:
    def __init__(self, content: str | None) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str | None) -> None:
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content: str | None) -> None:
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    def __init__(self, content: str | None) -> None:
        self._content = content
        self.received_kwargs: dict[str, Any] | None = None

    def create(self, **kwargs: Any) -> _FakeResponse:
        self.received_kwargs = kwargs
        return _FakeResponse(self._content)


class _FakeChat:
    def __init__(self, completions: _FakeCompletions) -> None:
        self.completions = completions


class FakeOpenAiClient:
    def __init__(self, api_key: str, timeout: float, content: str = "fake completion") -> None:
        self.api_key = api_key
        self.timeout = timeout
        self._completions = _FakeCompletions(content)
        self.chat = _FakeChat(self._completions)

    @property
    def received_create_kwargs(self) -> dict[str, Any] | None:
        return self._completions.received_kwargs


def test_generate_sends_prompt_as_single_user_message(monkeypatch: pytest.MonkeyPatch) -> None:
    created_clients: list[FakeOpenAiClient] = []

    def fake_openai_constructor(api_key: str, timeout: float) -> FakeOpenAiClient:
        client = FakeOpenAiClient(api_key=api_key, timeout=timeout)
        created_clients.append(client)
        return client

    monkeypatch.setattr("openai.OpenAI", fake_openai_constructor)

    llm = OpenAiLlm(api_key="sk-test", model="gpt-4o-mini", timeout=10.0)
    result = llm.generate("do-not-leak-this-prompt")

    assert result == "fake completion"
    assert len(created_clients) == 1
    assert created_clients[0].api_key == "sk-test"
    assert created_clients[0].timeout == 10.0
    assert created_clients[0].received_create_kwargs == {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "do-not-leak-this-prompt"}],
    }


def test_generate_returns_empty_string_when_content_is_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_openai_constructor(api_key: str, timeout: float) -> FakeOpenAiClient:
        return FakeOpenAiClient(api_key=api_key, timeout=timeout, content=None)

    monkeypatch.setattr("openai.OpenAI", fake_openai_constructor)

    llm = OpenAiLlm(api_key="sk-test", model="gpt-4o-mini", timeout=10.0)
    result = llm.generate("a prompt")

    assert result == ""


def test_generate_propagates_client_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    class RaisingCompletions:
        def create(self, **kwargs: Any) -> None:
            raise RuntimeError("openai api unavailable")

    class RaisingClient:
        def __init__(self, api_key: str, timeout: float) -> None:
            self.chat = _FakeChat(RaisingCompletions())  # type: ignore[arg-type]

    monkeypatch.setattr("openai.OpenAI", RaisingClient)

    llm = OpenAiLlm(api_key="sk-test", model="gpt-4o-mini", timeout=10.0)

    with pytest.raises(RuntimeError):
        llm.generate("a prompt")
