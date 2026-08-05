"""Unit tests for scripts/verify_live_rag.py.

Uses httpx.MockTransport (built into httpx, no extra dependency) so
these tests never make a real network call and never require a running
API server or an API key.
"""

from pathlib import Path

import httpx
import pytest
from scripts.verify_live_rag import _post_ask, _post_index, _resolve_report_path


def _make_client(handler: httpx.MockTransport) -> httpx.Client:
    return httpx.Client(base_url="http://testserver", transport=handler)


def test_resolve_report_path_default_uses_data_eval_results() -> None:
    path = _resolve_report_path("__default__")

    assert path.parent == Path("data/eval/results")
    assert path.name.startswith("live_verification_")
    assert path.suffix == ".json"


def test_resolve_report_path_explicit_value_is_used_as_is() -> None:
    path = _resolve_report_path("custom/report.json")

    assert path == Path("custom/report.json")


def test_post_index_success_reports_status_timing_and_response(tmp_path: Path) -> None:
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake content")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/documents/index"
        return httpx.Response(
            201,
            json={
                "document_id": "abc123",
                "source_name": "sample.pdf",
                "page_count": 3,
                "chunk_count": 5,
                "indexed_count": 5,
            },
        )

    with _make_client(httpx.MockTransport(handler)) as client:
        result = _post_index(client, pdf_path)

    assert result["endpoint"] == "POST /api/v1/documents/index"
    assert result["http_status"] == 201
    assert isinstance(result["elapsed_seconds"], float)
    assert result["response"]["chunk_count"] == 5


def test_post_ask_success_extracts_citations_and_answer_char_count() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/questions/ask"
        assert request.headers["content-type"].startswith("application/json")
        return httpx.Response(
            200,
            json={
                "answer": "テスト回答です。",
                "citations": [
                    {
                        "document_id": "abc123",
                        "source_name": "sample.pdf",
                        "title": None,
                        "page_number": 1,
                        "chunk_index": 0,
                        "score": 0.91,
                        "text_preview": "...",
                    },
                    {
                        "document_id": "abc123",
                        "source_name": "sample.pdf",
                        "title": None,
                        "page_number": 2,
                        "chunk_index": 1,
                        "score": 0.87,
                        "text_preview": "...",
                    },
                ],
                "is_insufficient_evidence": False,
            },
        )

    with _make_client(httpx.MockTransport(handler)) as client:
        result = _post_ask(client, "サンプルの質問", top_k=5)

    assert result["http_status"] == 200
    assert result["retrieved_chunk_count"] == 2
    assert result["answer_char_count"] == len("テスト回答です。")
    assert result["is_insufficient_evidence"] is False
    assert result["cited_pages"] == [1, 2]
    assert result["citations"] == [
        {"page_number": 1, "chunk_index": 0, "score": 0.91},
        {"page_number": 2, "chunk_index": 1, "score": 0.87},
    ]
    assert result["answer"] == "テスト回答です。"


def test_post_ask_error_status_preserves_response_body_without_answer_fields() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"detail": "There was an error parsing the body"})

    with _make_client(httpx.MockTransport(handler)) as client:
        result = _post_ask(client, "サンプルの質問", top_k=5)

    assert result["http_status"] == 400
    assert result["response"] == {"detail": "There was an error parsing the body"}
    assert "answer_char_count" not in result
    assert "citations" not in result


def test_post_ask_non_json_response_falls_back_to_raw_text() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            500, text="Internal Server Error", headers={"content-type": "text/plain"}
        )

    with _make_client(httpx.MockTransport(handler)) as client:
        result = _post_ask(client, "サンプルの質問", top_k=5)

    assert result["http_status"] == 500
    assert result["response"] == {"raw_text": "Internal Server Error"}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
