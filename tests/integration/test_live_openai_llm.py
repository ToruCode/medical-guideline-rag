"""Opt-in test against the real OpenAI API.

Skipped by default: requires a real, billable API key. Set
MEDICAL_RAG_LLM_API_KEY to a real key to run it, e.g. after changing
OpenAiLlm, to catch real integration issues (response shape, auth,
network) that mocked unit tests (tests/unit/test_openai_llm.py) cannot.
"""

import os

import pytest
from app.infrastructure.llm.openai_llm import OpenAiLlm

_API_KEY = os.environ.get("MEDICAL_RAG_LLM_API_KEY")

pytestmark = pytest.mark.skipif(
    not _API_KEY, reason="requires a real OpenAI API key in MEDICAL_RAG_LLM_API_KEY to run"
)


def test_generate_returns_a_real_completion() -> None:
    assert _API_KEY is not None
    llm = OpenAiLlm(
        api_key=_API_KEY,
        model=os.environ.get("MEDICAL_RAG_LLM_MODEL_NAME", "gpt-4o-mini"),
        timeout=30.0,
    )

    answer = llm.generate("Reply with exactly the single word: pong")

    assert isinstance(answer, str)
    assert answer.strip() != ""
