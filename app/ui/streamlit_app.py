"""Streamlit demo UI for the medical-guideline question-answering flow.

Presentation only, per docs/adr/0024-streamlit-demo-ui.md: all validation,
formatting, and error-message mapping live in app.ui.presentation (tested
without Streamlit or a network call), and all communication with the
question-answering flow goes through app.ui.api_client.QuestionApiClient,
which calls the existing POST /api/v1/questions/ask endpoint over HTTP. This
module never imports Application/Domain/Infrastructure code or an LLM SDK
directly.

Run with: uv run streamlit run app/ui/streamlit_app.py
(requires a separately running API server, e.g. `make dev`, with any
guideline documents already indexed via POST /documents/index.)
"""

import streamlit as st

from app.core.config import get_settings
from app.ui.api_client import ApiClientError, QuestionApiClient
from app.ui.presentation import (
    DEFAULT_TOP_K,
    INSUFFICIENT_EVIDENCE_NOTICE,
    MEDICAL_DISCLAIMER,
    citation_label,
    describe_error,
    validate_question,
)

st.set_page_config(page_title="Medical Guideline RAG Demo", page_icon="📋")


@st.cache_resource
def _get_api_client() -> QuestionApiClient:
    settings = get_settings()
    return QuestionApiClient(
        base_url=settings.ui_api_base_url, api_v1_prefix=settings.api_v1_prefix
    )


def main() -> None:
    st.title("Medical Guideline RAG デモ")
    st.warning(MEDICAL_DISCLAIMER)

    question = st.text_area("ガイドラインに関する質問を入力してください", height=120)
    submitted = st.button("質問する")

    if not submitted:
        return

    validation_error = validate_question(question)
    if validation_error is not None:
        st.error(validation_error)
        return

    client = _get_api_client()
    with st.spinner("回答を生成しています..."):
        try:
            result = client.ask_question(question, top_k=DEFAULT_TOP_K)
        except ApiClientError as exc:
            st.error(describe_error(exc))
            return

    if result.is_insufficient_evidence:
        st.warning(INSUFFICIENT_EVIDENCE_NOTICE)
    else:
        st.subheader("回答")
        st.write(result.answer)

    if result.citations:
        with st.expander(f"根拠となった引用（{len(result.citations)}件）"):
            for citation in result.citations:
                st.markdown(f"**{citation_label(citation)}**")
                st.caption(citation.text_preview)


main()
