"""Unit tests for QuestionApiClient.

httpx.post is monkeypatched with fakes, so these tests make no real
network requests and need no running API server.
"""

from typing import Any

import httpx
import pytest
from app.ui.api_client import (
    ApiConnectionError,
    ApiRequestError,
    CitationView,
    QuestionApiClient,
)


def _citation_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "document_id": "doc-1",
        "source_name": "sample.pdf",
        "title": "Guideline",
        "page_number": 3,
        "chunk_index": 0,
        "score": 0.87,
        "text_preview": "passage preview",
    }
    payload.update(overrides)
    return payload


def test_ask_question_returns_parsed_result_on_success(monkeypatch: pytest.MonkeyPatch) -> None:
    received_kwargs: dict[str, Any] = {}

    def fake_post(url: str, **kwargs: Any) -> httpx.Response:
        received_kwargs["url"] = url
        received_kwargs["json"] = kwargs["json"]
        return httpx.Response(
            status_code=200,
            json={
                "answer": "the answer",
                "citations": [_citation_payload()],
                "is_insufficient_evidence": False,
            },
        )

    monkeypatch.setattr(httpx, "post", fake_post)

    client = QuestionApiClient(base_url="http://127.0.0.1:8000", api_v1_prefix="/api/v1")
    result = client.ask_question("a question", top_k=5)

    assert received_kwargs["url"] == "http://127.0.0.1:8000/api/v1/questions/ask"
    assert received_kwargs["json"] == {"question": "a question", "top_k": 5}
    assert result.answer == "the answer"
    assert result.is_insufficient_evidence is False
    assert result.citations == [CitationView(**_citation_payload())]


def test_ask_question_strips_trailing_slash_from_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    received_urls: list[str] = []

    def fake_post(url: str, **kwargs: Any) -> httpx.Response:
        received_urls.append(url)
        return httpx.Response(
            status_code=200,
            json={"answer": "a", "citations": [], "is_insufficient_evidence": False},
        )

    monkeypatch.setattr(httpx, "post", fake_post)

    client = QuestionApiClient(base_url="http://127.0.0.1:8000/", api_v1_prefix="/api/v1")
    client.ask_question("q", top_k=1)

    assert received_urls == ["http://127.0.0.1:8000/api/v1/questions/ask"]


def test_ask_question_raises_api_connection_error_on_request_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_post(url: str, **kwargs: Any) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(httpx, "post", fake_post)

    client = QuestionApiClient(base_url="http://127.0.0.1:8000", api_v1_prefix="/api/v1")

    with pytest.raises(ApiConnectionError):
        client.ask_question("a question", top_k=5)


def test_ask_question_raises_api_request_error_with_detail_on_4xx(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_post(url: str, **kwargs: Any) -> httpx.Response:
        return httpx.Response(status_code=400, json={"detail": "question must not be empty."})

    monkeypatch.setattr(httpx, "post", fake_post)

    client = QuestionApiClient(base_url="http://127.0.0.1:8000", api_v1_prefix="/api/v1")

    with pytest.raises(ApiRequestError) as exc_info:
        client.ask_question("", top_k=5)

    assert exc_info.value.status_code == 400
    assert "question must not be empty" in str(exc_info.value)


def test_ask_question_raises_api_request_error_on_5xx_without_json_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_post(url: str, **kwargs: Any) -> httpx.Response:
        return httpx.Response(status_code=502, content=b"not json")

    monkeypatch.setattr(httpx, "post", fake_post)

    client = QuestionApiClient(base_url="http://127.0.0.1:8000", api_v1_prefix="/api/v1")

    with pytest.raises(ApiRequestError) as exc_info:
        client.ask_question("a question", top_k=5)

    assert exc_info.value.status_code == 502
