"""Opt-in retrieval-quality gate: Recall@k and MRR against a real,
downloaded sentence-transformers model.

Skipped by default, for the same reason as
tests/integration/test_live_sentence_transformer_embedder.py: set
RUN_SLOW_TESTS=1 to run it. No OpenAI API key is needed, since this
evaluates retrieval only (RetrieveChunksService), never generation -
see docs/requirements.md's "Separate retrieval quality from generation
quality" and docs/adr/0013-retrieval-evaluation.md.

The sample PDF (tests/support/evaluation/qa_dataset.py's SAMPLE_PAGES)
is generated at test-run time via tests/support/pdf_factory.build_pdf;
no PDF is committed to the repository, and no real guideline or
patient content is used.

MIN_RECALL_AT_3 / MIN_MRR are provisional thresholds calibrated
against the current EVALUATION_CASES only - not fixed production SLAs.
Recalibrate them (rerun once, inspect the printed per-case report, and
adjust) whenever EVALUATION_CASES grows or its content changes
meaningfully.
"""

import os
from pathlib import Path

import pytest
from app.application.services.chunk_document import ChunkDocumentService
from app.application.services.embed_chunks import EmbedChunksService
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
from app.infrastructure.pdf.pypdf_loader import PypdfLoader
from app.infrastructure.vector_store.in_memory_vector_store import InMemoryVectorStore
from tests.support.evaluation.metrics import mean, recall_at_k, reciprocal_rank
from tests.support.evaluation.qa_dataset import EVALUATION_CASES, SAMPLE_PAGES
from tests.support.pdf_factory import build_pdf

pytestmark = pytest.mark.skipif(
    not os.environ.get("RUN_SLOW_TESTS"),
    reason="downloads a real sentence-transformers model; set RUN_SLOW_TESTS=1 to run",
)

TOP_K = 3
# Provisional, calibrated against the current 8-case EVALUATION_CASES
# dataset (tests/support/evaluation/qa_dataset.py) - see module docstring.
MIN_RECALL_AT_3 = 0.8
MIN_MRR = 0.7

# Matches Settings' defaults (app/core/config.py), hardcoded rather than
# read from Settings so this gate's "one page = one chunk" assumption
# does not silently change with a developer's local .env.
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def test_retrieval_recall_and_mrr_meet_thresholds(tmp_path: Path) -> None:
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

    pdf_path = build_pdf(
        tmp_path / "retrieval_eval_guideline.pdf", SAMPLE_PAGES, title="Retrieval Evaluation Sample"
    )
    index_result = index_document.execute(pdf_path)
    assert index_result.page_count == len(SAMPLE_PAGES)
    assert index_result.chunk_count == len(SAMPLE_PAGES), (
        "expected one chunk per sample page; a sample sentence may have grown "
        "past CHUNK_SIZE, invalidating the page-number-as-chunk-id assumption"
    )

    recall_scores = []
    reciprocal_rank_scores = []
    report_lines = []
    for case in EVALUATION_CASES:
        results = retrieve_chunks.execute(case.question, top_k=TOP_K)
        ranked_ids = [str(result.embedded_chunk.chunk.page_number) for result in results]
        relevant_ids = {str(page_number) for page_number in case.expected_page_numbers}

        case_recall = recall_at_k(ranked_ids, relevant_ids, k=TOP_K)
        case_reciprocal_rank = reciprocal_rank(ranked_ids, relevant_ids)
        recall_scores.append(case_recall)
        reciprocal_rank_scores.append(case_reciprocal_rank)
        report_lines.append(
            f"  recall@{TOP_K}={case_recall:.2f} rr={case_reciprocal_rank:.2f} "
            f"expected={sorted(relevant_ids)} got={ranked_ids}"
        )

    mean_recall = mean(recall_scores)
    mean_rr = mean(reciprocal_rank_scores)
    report = "\n".join(
        [
            f"Retrieval evaluation over {len(EVALUATION_CASES)} cases:",
            f"  Recall@{TOP_K} = {mean_recall:.3f} (threshold {MIN_RECALL_AT_3})",
            f"  MRR       = {mean_rr:.3f} (threshold {MIN_MRR})",
            *report_lines,
        ]
    )
    print(report)

    assert mean_recall >= MIN_RECALL_AT_3, report
    assert mean_rr >= MIN_MRR, report
