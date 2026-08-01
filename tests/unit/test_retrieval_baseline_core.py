import json
from pathlib import Path

import pytest
from app.application.services.retrieve_chunks import RetrieveChunksService
from app.application.services.search_chunks import SearchChunksService
from app.domain.models.chunk import Chunk
from app.domain.models.embedding import EmbeddedChunk
from app.infrastructure.vector_store.in_memory_vector_store import InMemoryVectorStore
from scripts.retrieval_baseline_core import (
    TEXT_PREVIEW_MAX_CHARS,
    Aggregate,
    CaseResult,
    ConfigurationRun,
    DatasetCase,
    DatasetDocument,
    RankedChunkResult,
    RunConfig,
    evaluate_case,
    evaluate_configuration,
    load_dataset,
    location_key,
    print_case_report,
    print_verbose_case_results,
    summarize,
    truncate_text,
    write_local_report,
)
from tests.support.pdf_factory import build_pdf


class _FakeVectors(list):
    def tolist(self) -> list[list[float]]:
        return list(self)


class _FakeSentenceTransformerModel:
    """Minimal stand-in for sentence_transformers.SentenceTransformer's
    encode() interface, so evaluate_configuration() can be unit-tested
    without downloading a real model. Deterministic, text-length-based
    (like FakeEmbedder) - not semantically meaningful, only used to
    verify evaluate_configuration()'s own wiring/aggregation logic.
    """

    def encode(self, texts: list[str], convert_to_numpy: bool = True) -> _FakeVectors:
        del convert_to_numpy
        return _FakeVectors([[float(len(text) % 5), float((len(text) + 1) % 5)] for text in texts])


class _QueryAwareEmbedder:
    def __init__(self, vectors_by_text: dict[str, list[float]]) -> None:
        self._vectors_by_text = vectors_by_text

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vectors_by_text[text] for text in texts]


def _make_chunk(page_number: int, chunk_index: int, text: str) -> Chunk:
    return Chunk(
        document_id="doc-1",
        source_name="guideline.pdf",
        source_path="/tmp/guideline.pdf",
        page_number=page_number,
        chunk_index=chunk_index,
        text=text,
        title="Guideline",
    )


def _build_retrieve_chunks_service(
    vectors_by_text: dict[str, list[float]], chunks: list[Chunk]
) -> RetrieveChunksService:
    store = InMemoryVectorStore()
    store.upsert(
        [EmbeddedChunk(chunk=chunk, vector=vectors_by_text[chunk.text]) for chunk in chunks]
    )
    embedder = _QueryAwareEmbedder(vectors_by_text)
    return RetrieveChunksService(embedder=embedder, search_chunks=SearchChunksService(store))


def test_location_key_without_chunk_index_returns_page_only() -> None:
    assert location_key(12, None) == "12"


def test_location_key_with_chunk_index_returns_composite_key() -> None:
    assert location_key(12, 0) == "12:0"


def test_load_dataset_parses_page_granularity(tmp_path: Path) -> None:
    dataset_path = tmp_path / "qa.json"
    dataset_path.write_text(
        json.dumps(
            {
                "document": {"source_path": "data/raw/example.pdf", "label": "Guideline A"},
                "cases": [{"question": "q1", "granularity": "page", "expected": [1, 2]}],
            }
        ),
        encoding="utf-8",
    )

    document, cases = load_dataset(dataset_path)

    assert document.source_path == Path("data/raw/example.pdf")
    assert document.label == "Guideline A"
    assert cases == [
        DatasetCase(question="q1", granularity="page", expected_locations=[(1, None), (2, None)])
    ]


def test_load_dataset_parses_chunk_granularity(tmp_path: Path) -> None:
    dataset_path = tmp_path / "qa.json"
    dataset_path.write_text(
        json.dumps(
            {
                "document": {"source_path": "data/raw/example.pdf"},
                "cases": [{"question": "q1", "granularity": "chunk", "expected": [[3, 1]]}],
            }
        ),
        encoding="utf-8",
    )

    document, cases = load_dataset(dataset_path)

    assert document.label == "Guideline A"  # default
    assert cases == [DatasetCase(question="q1", granularity="chunk", expected_locations=[(3, 1)])]


def test_load_dataset_rejects_unknown_granularity(tmp_path: Path) -> None:
    dataset_path = tmp_path / "qa.json"
    dataset_path.write_text(
        json.dumps(
            {
                "document": {"source_path": "data/raw/example.pdf"},
                "cases": [{"question": "q1", "granularity": "paragraph", "expected": [1]}],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unknown granularity"):
        load_dataset(dataset_path)


def test_load_dataset_rejects_empty_cases(tmp_path: Path) -> None:
    dataset_path = tmp_path / "qa.json"
    dataset_path.write_text(
        json.dumps({"document": {"source_path": "data/raw/example.pdf"}, "cases": []}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="no evaluation cases"):
        load_dataset(dataset_path)


def test_evaluate_case_page_granularity_counts_any_chunk_from_expected_page() -> None:
    close_chunk = _make_chunk(1, 0, "close passage")
    far_chunk = _make_chunk(2, 0, "far passage")
    vectors = {
        "a query": [1.0, 0.0],
        "close passage": [1.0, 0.0],
        "far passage": [0.0, 1.0],
    }
    service = _build_retrieve_chunks_service(vectors, [close_chunk, far_chunk])
    case = DatasetCase(question="a query", granularity="page", expected_locations=[(1, None)])

    result = evaluate_case(case, service, top_k=5)

    assert result.recall_at_1 == 1.0
    assert result.reciprocal_rank == 1.0
    assert result.ranked_locations[0] == "1"


def test_evaluate_case_chunk_granularity_requires_exact_chunk_index() -> None:
    chunk_a = _make_chunk(1, 0, "chunk a")
    chunk_b = _make_chunk(1, 1, "chunk b")
    vectors = {"q": [1.0, 0.0], "chunk a": [1.0, 0.0], "chunk b": [0.9, 0.1]}
    service = _build_retrieve_chunks_service(vectors, [chunk_a, chunk_b])
    # Only chunk_index 1 (chunk_b) is correct, even though chunk_a shares the same page.
    case = DatasetCase(question="q", granularity="chunk", expected_locations=[(1, 1)])

    result = evaluate_case(case, service, top_k=5)

    assert result.recall_at_1 == 0.0
    assert result.recall_at_3 == 1.0
    assert result.reciprocal_rank == 0.5


def test_evaluate_case_populates_ranked_results_with_rank_page_chunk_score_and_preview() -> None:
    chunk_a = _make_chunk(1, 0, "chunk a text")
    chunk_b = _make_chunk(2, 0, "chunk b text")
    vectors = {"q": [1.0, 0.0], "chunk a text": [1.0, 0.0], "chunk b text": [0.9, 0.1]}
    service = _build_retrieve_chunks_service(vectors, [chunk_a, chunk_b])
    case = DatasetCase(question="q", granularity="page", expected_locations=[(1, None)])

    result = evaluate_case(case, service, top_k=5)

    assert [r.rank for r in result.ranked_results] == [1, 2]
    assert result.ranked_results[0].page_number == 1
    assert result.ranked_results[0].chunk_index == 0
    assert result.ranked_results[0].text_preview == "chunk a text"
    assert isinstance(result.ranked_results[0].score, float)
    assert result.ranked_results[1].page_number == 2


def test_summarize_averages_all_case_scores() -> None:
    results = [
        CaseResult(
            question="q1",
            expected=[(1, None)],
            ranked_locations=["1"],
            recall_at_1=1.0,
            recall_at_3=1.0,
            recall_at_5=1.0,
            reciprocal_rank=1.0,
        ),
        CaseResult(
            question="q2",
            expected=[(2, None)],
            ranked_locations=["3"],
            recall_at_1=0.0,
            recall_at_3=0.0,
            recall_at_5=0.0,
            reciprocal_rank=0.0,
        ),
    ]

    aggregate = summarize(results)

    assert aggregate.recall_at_1 == pytest.approx(0.5)
    assert aggregate.recall_at_3 == pytest.approx(0.5)
    assert aggregate.recall_at_5 == pytest.approx(0.5)
    assert aggregate.mrr == pytest.approx(0.5)


def test_evaluate_configuration_populates_config_and_indexes_document(tmp_path: Path) -> None:
    pdf_path = build_pdf(tmp_path / "sample.pdf", ["short page one", "short page two"])
    document = DatasetDocument(source_path=pdf_path, label="Guideline A")
    cases = [
        DatasetCase(question="short page one", granularity="page", expected_locations=[(1, None)])
    ]
    model = _FakeSentenceTransformerModel()

    run = evaluate_configuration(
        document,
        cases,
        model,  # type: ignore[arg-type]
        chunk_size=1000,
        chunk_overlap=200,
        top_k=5,
        embedding_model_name="fake-model",
    )

    assert run.config.chunk_size == 1000
    assert run.config.chunk_overlap == 200
    assert run.config.top_k == 5
    assert run.config.embedding_model_name == "fake-model"
    assert run.config.case_count == 1
    assert run.config.indexed_page_count == 2
    assert run.config.indexed_chunk_count == 2
    assert len(run.case_results) == 1


def test_evaluate_configuration_chunk_size_controls_splitting(tmp_path: Path) -> None:
    long_text = "word " * 300  # ~1500 characters
    pdf_path = build_pdf(tmp_path / "long.pdf", [long_text])
    document = DatasetDocument(source_path=pdf_path, label="Guideline A")
    cases = [DatasetCase(question="word", granularity="page", expected_locations=[(1, None)])]
    model = _FakeSentenceTransformerModel()

    small_chunks = evaluate_configuration(
        document,
        cases,
        model,  # type: ignore[arg-type]
        chunk_size=300,
        chunk_overlap=100,
        top_k=5,
        embedding_model_name="fake-model",
    )
    large_chunks = evaluate_configuration(
        document,
        cases,
        model,  # type: ignore[arg-type]
        chunk_size=2000,
        chunk_overlap=100,
        top_k=5,
        embedding_model_name="fake-model",
    )

    assert large_chunks.config.indexed_chunk_count == 1
    assert small_chunks.config.indexed_chunk_count > large_chunks.config.indexed_chunk_count


def test_evaluate_configuration_uses_a_fresh_vector_store_per_call(tmp_path: Path) -> None:
    first_pdf = build_pdf(tmp_path / "first.pdf", ["alpha"])
    second_pdf = build_pdf(tmp_path / "second.pdf", ["beta", "gamma", "delta"])
    cases = [DatasetCase(question="alpha", granularity="page", expected_locations=[(1, None)])]
    model = _FakeSentenceTransformerModel()

    first_run = evaluate_configuration(
        DatasetDocument(source_path=first_pdf, label="A"),
        cases,
        model,  # type: ignore[arg-type]
        chunk_size=1000,
        chunk_overlap=200,
        top_k=5,
        embedding_model_name="fake-model",
    )
    second_run = evaluate_configuration(
        DatasetDocument(source_path=second_pdf, label="B"),
        cases,
        model,  # type: ignore[arg-type]
        chunk_size=1000,
        chunk_overlap=200,
        top_k=5,
        embedding_model_name="fake-model",
    )

    assert first_run.config.indexed_page_count == 1
    assert first_run.config.indexed_chunk_count == 1
    assert second_run.config.indexed_page_count == 3
    assert second_run.config.indexed_chunk_count == 3


def test_truncate_text_returns_unchanged_when_at_or_under_max_chars() -> None:
    exact = "a" * TEXT_PREVIEW_MAX_CHARS
    short = "short text"

    assert truncate_text(exact) == exact
    assert truncate_text(short) == short


def test_truncate_text_truncates_and_appends_ellipsis_when_over_max_chars() -> None:
    long_text = "a" * (TEXT_PREVIEW_MAX_CHARS + 1)

    preview = truncate_text(long_text)

    assert preview == "a" * TEXT_PREVIEW_MAX_CHARS + "..."
    assert preview.endswith("...")


def _make_case_result_with_ranked_results() -> CaseResult:
    return CaseResult(
        question="dialysis water bacteria limit?",
        expected=[(3, 0)],
        ranked_locations=["3:0", "1:0"],
        recall_at_1=1.0,
        recall_at_3=1.0,
        recall_at_5=1.0,
        reciprocal_rank=1.0,
        ranked_results=[
            RankedChunkResult(
                rank=1, page_number=3, chunk_index=0, score=0.8123, text_preview="water text"
            ),
            RankedChunkResult(
                rank=2, page_number=1, chunk_index=0, score=0.7941, text_preview="other text"
            ),
        ],
    )


def test_print_verbose_case_results_prints_rank_page_chunk_score_and_text_preview(
    capsys: pytest.CaptureFixture[str],
) -> None:
    case_result = _make_case_result_with_ranked_results()

    print_verbose_case_results([case_result])

    output = capsys.readouterr().out
    assert "Q: dialysis water bacteria limit?" in output
    assert "rank=1 page=3 chunk=0 score=0.8123" in output
    assert "text_preview: water text" in output
    assert "rank=2 page=1 chunk=0 score=0.7941" in output
    assert "text_preview: other text" in output


def test_print_case_report_does_not_show_text_preview_body(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """print_case_report() is the only per-question output shown outside
    --verbose, so it must never include retrieved chunk body text -
    only print_verbose_case_results() (called when --verbose is passed)
    may do that.
    """
    case_result = _make_case_result_with_ranked_results()

    print_case_report([case_result])

    output = capsys.readouterr().out
    assert "text_preview" not in output
    assert "water text" not in output


def test_write_local_report_includes_text_preview_in_saved_json(tmp_path: Path) -> None:
    case_result = _make_case_result_with_ranked_results()
    config = RunConfig(
        chunk_size=500,
        chunk_overlap=200,
        top_k=5,
        embedding_model_name="intfloat/multilingual-e5-base",
        query_prefix="query: ",
        passage_prefix="passage: ",
        case_count=1,
        indexed_page_count=10,
        indexed_chunk_count=42,
        measured_at="2026-08-02",
    )
    aggregate = Aggregate(recall_at_1=1.0, recall_at_3=1.0, recall_at_5=1.0, mrr=1.0)
    run = ConfigurationRun(config=config, case_results=[case_result], aggregate=aggregate)
    document = DatasetDocument(source_path=Path("data/raw/example.pdf"), label="Guideline A")
    report_path = tmp_path / "report.json"

    write_local_report(report_path, document, [run])

    saved = json.loads(report_path.read_text(encoding="utf-8"))
    ranked_results = saved["runs"][0]["cases"][0]["ranked_results"]
    assert ranked_results == [
        {
            "rank": 1,
            "page_number": 3,
            "chunk_index": 0,
            "score": 0.8123,
            "text_preview": "water text",
        },
        {
            "rank": 2,
            "page_number": 1,
            "chunk_index": 0,
            "score": 0.7941,
            "text_preview": "other text",
        },
    ]
