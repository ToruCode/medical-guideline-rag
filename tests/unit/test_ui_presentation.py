"""Unit tests for the Streamlit UI's presentation-mapping functions.

No Streamlit import and no network call - these test pure functions only.
"""

from app.ui.api_client import ApiConnectionError, ApiRequestError, CitationView
from app.ui.presentation import (
    CONNECTION_ERROR_MESSAGE,
    EMPTY_QUESTION_MESSAGE,
    GENERATION_ERROR_MESSAGE,
    SERVER_ERROR_MESSAGE,
    UNEXPECTED_ERROR_MESSAGE,
    VALIDATION_ERROR_MESSAGE,
    citation_label,
    describe_error,
    validate_question,
)


def test_validate_question_rejects_empty_string() -> None:
    assert validate_question("") == EMPTY_QUESTION_MESSAGE


def test_validate_question_rejects_whitespace_only() -> None:
    assert validate_question("   \n\t  ") == EMPTY_QUESTION_MESSAGE


def test_validate_question_accepts_non_empty_question() -> None:
    assert validate_question("糖尿病の治療方針は？") is None


def test_citation_label_prefers_title_over_source_name() -> None:
    citation = CitationView(
        document_id="doc-1",
        source_name="sample.pdf",
        title="Diabetes Guideline",
        page_number=12,
        chunk_index=2,
        score=0.912,
        text_preview="passage preview",
    )

    label = citation_label(citation)

    assert "Diabetes Guideline" in label
    assert "sample.pdf" not in label
    assert "12" in label
    assert "2" in label
    assert "0.912" in label


def test_citation_label_falls_back_to_source_name_when_title_is_none() -> None:
    citation = CitationView(
        document_id="doc-1",
        source_name="sample.pdf",
        title=None,
        page_number=1,
        chunk_index=0,
        score=0.5,
        text_preview="passage preview",
    )

    assert "sample.pdf" in citation_label(citation)


def test_citation_label_excludes_text_preview() -> None:
    citation = CitationView(
        document_id="doc-1",
        source_name="sample.pdf",
        title=None,
        page_number=1,
        chunk_index=0,
        score=0.5,
        text_preview="do-not-leak-this-guideline-passage",
    )

    assert "do-not-leak-this-guideline-passage" not in citation_label(citation)


def test_describe_error_maps_connection_error() -> None:
    assert describe_error(ApiConnectionError("boom")) == CONNECTION_ERROR_MESSAGE


def test_describe_error_maps_400_to_validation_message() -> None:
    assert describe_error(ApiRequestError(400, "detail")) == VALIDATION_ERROR_MESSAGE


def test_describe_error_maps_502_to_generation_message() -> None:
    assert describe_error(ApiRequestError(502, "detail")) == GENERATION_ERROR_MESSAGE


def test_describe_error_maps_other_5xx_to_server_error_message() -> None:
    assert describe_error(ApiRequestError(500, "detail")) == SERVER_ERROR_MESSAGE


def test_describe_error_never_leaks_underlying_exception_message() -> None:
    message = describe_error(ApiRequestError(400, "do-not-leak-this-detail sk-secret"))

    assert "do-not-leak-this-detail" not in message
    assert "sk-secret" not in message


def test_describe_error_falls_back_to_unexpected_message_for_unknown_error() -> None:
    from app.ui.api_client import ApiClientError

    class _OtherApiClientError(ApiClientError):
        pass

    assert describe_error(_OtherApiClientError("boom")) == UNEXPECTED_ERROR_MESSAGE
