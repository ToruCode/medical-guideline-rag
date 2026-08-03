"""Opt-in answer-quality-and-citation-consistency measurement against
the real OpenAI API (Issue #10, requirement 5: "optional live
evaluation").

Skipped unless both RUN_SLOW_TESTS=1 (a real sentence-transformers
model is downloaded, matching
tests/integration/test_retrieval_evaluation.py) and a real
MEDICAL_RAG_LLM_API_KEY are set (matching
tests/integration/test_live_openai_llm.py) - this test is billable and
requires network access, unlike the default, deterministic
FakeLlm-based coverage in tests/unit/test_answer_quality_core.py.

There is no pass/fail threshold on answer_point_coverage: a real LLM's
answer wording cannot be guaranteed to reproduce
ANSWER_EVALUATION_CASES' exact expected_answer_points substrings, and
answer_point_coverage is a lexical approximation, not a semantic one
(docs/adr/0023-answer-quality-and-citation-consistency-evaluation.md).
This test's real assertions are the ones with a deterministic ground
truth: retrieval-driven citation precision/recall against
expected_page_numbers, and the citations_consistent structural
invariant. Coverage is only printed for manual inspection.
"""

import os
from pathlib import Path

import pytest
from app.application.services.chunk_document import ChunkDocumentService
from app.application.services.embed_chunks import EmbedChunksService
from app.application.services.generate_answer import GenerateAnswerService
from app.application.services.index_chunks import IndexChunksService
from app.application.services.index_document import IndexDocumentService
from app.application.services.load_document import LoadDocumentService
from app.application.services.retrieve_chunks import RetrieveChunksService
from app.application.services.search_chunks import SearchChunksService
from app.infrastructure.chunking.fixed_size_text_splitter import FixedSizeTextSplitter
from app.infrastructure.embedding.sentence_transformer_embedder import (
    SentenceTransformerEmbedder,
    load_sentence_transformer_model,
)
from app.infrastructure.llm.openai_llm import OpenAiLlm
from app.infrastructure.pdf.pypdf_loader import PypdfLoader
from app.infrastructure.vector_store.in_memory_vector_store import InMemoryVectorStore
from scripts.answer_quality_core import citations_are_subset_of_retrieved
from tests.support.evaluation.metrics import (
    answer_point_coverage,
    citation_precision,
    citation_recall,
    mean,
)
from tests.support.evaluation.qa_dataset import ANSWER_EVALUATION_CASES, SAMPLE_PAGES
from tests.support.pdf_factory import build_pdf

_API_KEY = os.environ.get("MEDICAL_RAG_LLM_API_KEY")

pytestmark = pytest.mark.skipif(
    not (os.environ.get("RUN_SLOW_TESTS") and _API_KEY),
    reason=(
        "downloads a real sentence-transformers model and calls the real OpenAI "
        "API; set RUN_SLOW_TESTS=1 and a real MEDICAL_RAG_LLM_API_KEY to run"
    ),
)

TOP_K = 3
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def test_live_answer_quality_and_citation_consistency(tmp_path: Path) -> None:
    assert _API_KEY is not None
    model = load_sentence_transformer_model("intfloat/multilingual-e5-base")
    passage_embedder = SentenceTransformerEmbedder(model, prefix="passage: ")
    query_embedder = SentenceTransformerEmbedder(model, prefix="query: ")
    vector_store = InMemoryVectorStore()

    index_document = IndexDocumentService(
        load_document=LoadDocumentService(PypdfLoader()),
        chunk_document=ChunkDocumentService(
            FixedSizeTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        ),
        embed_chunks=EmbedChunksService(passage_embedder),
        index_chunks=IndexChunksService(vector_store),
    )
    retrieve_chunks = RetrieveChunksService(
        embedder=query_embedder, search_chunks=SearchChunksService(vector_store)
    )
    llm = OpenAiLlm(api_key=_API_KEY, model="gpt-4o-mini", timeout=30.0)
    generate_answer = GenerateAnswerService(llm)

    pdf_path = build_pdf(
        tmp_path / "answer_quality_eval_guideline.pdf",
        SAMPLE_PAGES,
        title="Answer Quality Evaluation Sample",
    )
    index_document.execute(pdf_path)

    precision_scores = []
    recall_scores = []
    coverage_scores = []
    consistency_violations = 0
    report_lines = []
    for case in ANSWER_EVALUATION_CASES:
        search_results = retrieve_chunks.execute(case.question, top_k=TOP_K)
        result = generate_answer.execute(case.question, search_results)

        expected_pages = set(case.expected_page_numbers)
        cited_pages = {c.embedded_chunk.chunk.page_number for c in result.citations}
        case_precision = citation_precision(cited_pages, expected_pages)
        case_recall = citation_recall(cited_pages, expected_pages)
        case_coverage = answer_point_coverage(result.answer, case.expected_answer_points)
        is_consistent = citations_are_subset_of_retrieved(result.citations, search_results)

        if case_precision is not None:
            precision_scores.append(case_precision)
        recall_scores.append(case_recall)
        if case_coverage is not None:
            coverage_scores.append(case_coverage)
        if not is_consistent:
            consistency_violations += 1

        report_lines.append(
            f"  precision={case_precision} recall={case_recall:.2f} "
            f"coverage={case_coverage} consistent={is_consistent} "
            f"expected_pages={sorted(expected_pages)} cited_pages={sorted(cited_pages)}"
        )

    report = "\n".join(
        [
            f"Answer-quality evaluation over {len(ANSWER_EVALUATION_CASES)} cases (real OpenAI):",
            f"  Citation recall (mean) = {mean(recall_scores):.3f}",
            f"  Citation consistency violations = {consistency_violations} (expected: 0)",
            *report_lines,
        ]
    )
    print(report)

    # Deterministic, retrieval-driven assertions only - see module docstring
    # for why answer_point_coverage is printed but not asserted on.
    assert consistency_violations == 0, report
    assert mean(recall_scores) >= 0.8, report
