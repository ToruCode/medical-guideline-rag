"""Pure presentation-mapping functions for the Streamlit UI.

Kept free of any Streamlit import so this module's logic (validation,
citation formatting, error-message mapping) can be unit tested directly,
without running a Streamlit app or a network call.
"""

from app.ui.api_client import ApiClientError, ApiConnectionError, ApiRequestError, CitationView

DEFAULT_TOP_K = 5

MEDICAL_DISCLAIMER = (
    "本アプリケーションは技術デモンストレーションです。表示される回答は個別の診断や"
    "治療方針の決定を目的としたものではありません。実際の臨床判断にあたっては、"
    "必ず元のガイドラインおよび最新の臨床情報をご確認ください。"
)

INSUFFICIENT_EVIDENCE_NOTICE = (
    "この質問に対して、十分な根拠となるガイドラインの記載が見つかりませんでした。"
    "元のガイドラインおよび最新の臨床情報をご確認ください。"
)

EMPTY_QUESTION_MESSAGE = "質問を入力してください。"
CONNECTION_ERROR_MESSAGE = (
    "APIサーバーに接続できませんでした。サーバーが起動しているか確認してください。"
)
VALIDATION_ERROR_MESSAGE = "入力内容に誤りがあります。質問を確認してください。"
GENERATION_ERROR_MESSAGE = "回答の生成に失敗しました。しばらくしてから再度お試しください。"
SERVER_ERROR_MESSAGE = "サーバーでエラーが発生しました。しばらくしてから再度お試しください。"
UNEXPECTED_ERROR_MESSAGE = "予期しないエラーが発生しました。"


def validate_question(question: str) -> str | None:
    """Returns a Japanese error message if question is empty/whitespace-only, else None."""
    if not question.strip():
        return EMPTY_QUESTION_MESSAGE
    return None


def citation_label(citation: CitationView) -> str:
    """Formats one citation's safe metadata into a single display line.

    Deliberately excludes text_preview (shown separately by the caller) so
    this function's output never grows unbounded with guideline text.
    """
    source = citation.title or citation.source_name
    return (
        f"{source}｜{citation.page_number}ページ｜チャンク{citation.chunk_index}"
        f"｜スコア{citation.score:.3f}"
    )


def describe_error(exc: ApiClientError) -> str:
    """Maps an ApiClientError to a safe, user-friendly Japanese message.

    Never includes the exception's own message, request/response detail, or
    any part of a stack trace - only a fixed message chosen by error
    category, so an API key, prompt, or internal path can never leak into
    the UI regardless of what the underlying exception happens to contain.
    """
    if isinstance(exc, ApiConnectionError):
        return CONNECTION_ERROR_MESSAGE
    if isinstance(exc, ApiRequestError):
        if exc.status_code == 400:
            return VALIDATION_ERROR_MESSAGE
        if exc.status_code == 502:
            return GENERATION_ERROR_MESSAGE
        return SERVER_ERROR_MESSAGE
    return UNEXPECTED_ERROR_MESSAGE
