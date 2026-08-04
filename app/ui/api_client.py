"""HTTP client the Streamlit UI uses to reach the existing FastAPI question-
answering endpoint.

The UI runs as a separate process from the FastAPI server and never imports
Application/Domain/Infrastructure code or calls OpenAI directly - it only
speaks HTTP to whichever already-running API process a guideline document
was indexed into (app.api.v1.endpoints.questions.ask_question, which
composes the existing AskQuestionService). See
docs/adr/0024-streamlit-demo-ui.md for why: InMemoryVectorStore is
process-wide, so a Streamlit process constructing its own AskQuestionService
in-process would see an empty, unrelated vector store.
"""

from dataclasses import dataclass

import httpx


class ApiClientError(Exception):
    """Base class for UI-facing API client failures."""


class ApiConnectionError(ApiClientError):
    """Raised when the API server could not be reached at all."""


class ApiRequestError(ApiClientError):
    """Raised when the API responded with an error status code (4xx/5xx)."""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True, slots=True)
class CitationView:
    """One citation, mirroring app.schemas.question.CitationSchema."""

    document_id: str
    source_name: str
    title: str | None
    page_number: int
    chunk_index: int
    score: float
    text_preview: str


@dataclass(frozen=True, slots=True)
class AskQuestionResult:
    """Mirrors app.schemas.question.AskQuestionResponse."""

    answer: str
    citations: list[CitationView]
    is_insufficient_evidence: bool


def _extract_detail(response: httpx.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return f"HTTP {response.status_code}"
    detail = body.get("detail")
    return str(detail) if detail else f"HTTP {response.status_code}"


class QuestionApiClient:
    """Thin wrapper around POST {base_url}{api_v1_prefix}/questions/ask."""

    def __init__(self, base_url: str, api_v1_prefix: str, *, timeout: float = 30.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_v1_prefix = api_v1_prefix
        self._timeout = timeout

    def ask_question(self, question: str, top_k: int) -> AskQuestionResult:
        url = f"{self._base_url}{self._api_v1_prefix}/questions/ask"
        try:
            response = httpx.post(
                url, json={"question": question, "top_k": top_k}, timeout=self._timeout
            )
        except httpx.RequestError as exc:
            raise ApiConnectionError(
                f"Could not reach the API server: {type(exc).__name__}"
            ) from exc

        if response.status_code >= 400:
            raise ApiRequestError(response.status_code, _extract_detail(response))

        payload = response.json()
        return AskQuestionResult(
            answer=payload["answer"],
            citations=[CitationView(**item) for item in payload["citations"]],
            is_insufficient_evidence=payload["is_insufficient_evidence"],
        )
