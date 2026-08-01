"""Shared library for real-data retrieval evaluation tooling.

Not a CLI entry point - imported by scripts/evaluate_retrieval_baseline.py
(Issue #18: measure one configuration against a local, real dataset) and
scripts/compare_chunk_sizes.py (Issue #19: measure several chunk_size
values against the same dataset). Both compose Application-layer
services directly (no API layer, no LLM - retrieval only, per
docs/requirements.md's "separate retrieval quality from generation
quality") and report only aggregate/local output; see
docs/adr/0014-real-data-retrieval-baseline.md and
docs/adr/0015-chunk-size-comparison.md.

Never commit a dataset file, the PDF it points to, or any
--save-report output; data/eval/ is gitignored for exactly this
reason. See docs/evaluation-dataset-format.md.
"""

import json
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any

from app.application.services.chunk_document import ChunkDocumentService
from app.application.services.embed_chunks import EmbedChunksService
from app.application.services.index_chunks import IndexChunksService
from app.application.services.index_document import IndexDocumentService
from app.application.services.load_document import LoadDocumentService
from app.application.services.retrieve_chunks import RetrieveChunksService
from app.application.services.search_chunks import SearchChunksService
from app.infrastructure.chunking.fixed_size_text_splitter import FixedSizeTextSplitter
from app.infrastructure.embedding.sentence_transformer_embedder import SentenceTransformerEmbedder
from app.infrastructure.pdf.pypdf_loader import PypdfLoader
from app.infrastructure.vector_store.in_memory_vector_store import InMemoryVectorStore
from tests.support.evaluation.metrics import mean, recall_at_k, reciprocal_rank

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "

# Max length of a retrieved chunk's text_preview (RankedChunkResult) before
# truncation, for --verbose terminal output and --save-report JSON. Deliberately
# separate from CITATION_TEXT_PREVIEW_MAX_CHARS (app/core/constants.py): this
# tooling is for debugging retrieval quality locally, not an API response, so it
# is implemented independently of app/api/v1/endpoints/questions.py rather than
# importing from the API layer.
TEXT_PREVIEW_MAX_CHARS = 300


def truncate_text(text: str, max_chars: int = TEXT_PREVIEW_MAX_CHARS) -> str:
    """Truncates text to at most max_chars, appending "..." when cut.

    The returned string's own length can exceed max_chars by the length of
    the "..." marker; callers relying on a hard cap should account for that.
    """
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


@dataclass(frozen=True, slots=True)
class DatasetDocument:
    source_path: Path
    label: str


@dataclass(frozen=True, slots=True)
class DatasetCase:
    question: str
    granularity: str  # "page" or "chunk"
    expected_locations: list[tuple[int, int | None]]  # (page_number, chunk_index)


@dataclass(frozen=True, slots=True)
class RankedChunkResult:
    """One retrieved chunk for a single question, for --verbose terminal
    output and --save-report JSON only - not used in Recall/MRR computation
    (see CaseResult.ranked_locations for that).
    """

    rank: int
    page_number: int
    chunk_index: int
    score: float
    text_preview: str


@dataclass(frozen=True, slots=True)
class CaseResult:
    question: str
    expected: list[tuple[int, int | None]]
    ranked_locations: list[str]
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    reciprocal_rank: float
    ranked_results: list[RankedChunkResult] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class Aggregate:
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    mrr: float


@dataclass(frozen=True, slots=True)
class RunConfig:
    chunk_size: int
    chunk_overlap: int
    top_k: int
    embedding_model_name: str
    query_prefix: str
    passage_prefix: str
    case_count: int
    indexed_page_count: int
    indexed_chunk_count: int
    measured_at: str


@dataclass(frozen=True, slots=True)
class ConfigurationRun:
    """The full result of evaluate_configuration() for one chunk_size/
    chunk_overlap/top_k/embedding_model_name combination.
    """

    config: RunConfig
    case_results: list[CaseResult]
    aggregate: Aggregate


def location_key(page_number: int, chunk_index: int | None) -> str:
    if chunk_index is None:
        return str(page_number)
    return f"{page_number}:{chunk_index}"


def _parse_case(case_raw: dict[str, Any]) -> DatasetCase:
    granularity = case_raw["granularity"]
    if granularity not in ("page", "chunk"):
        raise ValueError(f"Unknown granularity: {granularity!r} (must be 'page' or 'chunk')")

    expected_raw = case_raw["expected"]
    expected_locations: list[tuple[int, int | None]]
    if granularity == "page":
        expected_locations = [(int(page), None) for page in expected_raw]
    else:
        expected_locations = [(int(page), int(chunk_index)) for page, chunk_index in expected_raw]

    return DatasetCase(
        question=case_raw["question"],
        granularity=granularity,
        expected_locations=expected_locations,
    )


def load_dataset(path: Path) -> tuple[DatasetDocument, list[DatasetCase]]:
    raw = json.loads(path.read_text(encoding="utf-8"))

    document_raw = raw["document"]
    document = DatasetDocument(
        source_path=Path(document_raw["source_path"]),
        label=document_raw.get("label", "Guideline A"),
    )

    cases = [_parse_case(case_raw) for case_raw in raw["cases"]]
    if not cases:
        raise ValueError(f"{path} contains no evaluation cases")
    return document, cases


def evaluate_case(
    case: DatasetCase, retrieve_chunks: RetrieveChunksService, top_k: int
) -> CaseResult:
    results = retrieve_chunks.execute(case.question, top_k=top_k)
    if case.granularity == "chunk":
        ranked_ids = [
            location_key(
                result.embedded_chunk.chunk.page_number, result.embedded_chunk.chunk.chunk_index
            )
            for result in results
        ]
    else:
        ranked_ids = [
            location_key(result.embedded_chunk.chunk.page_number, None) for result in results
        ]
    relevant_ids = {location_key(page, idx) for page, idx in case.expected_locations}
    ranked_results = [
        RankedChunkResult(
            rank=rank,
            page_number=result.embedded_chunk.chunk.page_number,
            chunk_index=result.embedded_chunk.chunk.chunk_index,
            score=result.score,
            text_preview=truncate_text(result.embedded_chunk.chunk.text),
        )
        for rank, result in enumerate(results, start=1)
    ]

    return CaseResult(
        question=case.question,
        expected=case.expected_locations,
        ranked_locations=ranked_ids,
        recall_at_1=recall_at_k(ranked_ids, relevant_ids, k=1),
        recall_at_3=recall_at_k(ranked_ids, relevant_ids, k=3),
        recall_at_5=recall_at_k(ranked_ids, relevant_ids, k=5),
        reciprocal_rank=reciprocal_rank(ranked_ids, relevant_ids),
        ranked_results=ranked_results,
    )


def summarize(case_results: list[CaseResult]) -> Aggregate:
    return Aggregate(
        recall_at_1=mean([c.recall_at_1 for c in case_results]),
        recall_at_3=mean([c.recall_at_3 for c in case_results]),
        recall_at_5=mean([c.recall_at_5 for c in case_results]),
        mrr=mean([c.reciprocal_rank for c in case_results]),
    )


def evaluate_configuration(
    document: DatasetDocument,
    cases: list[DatasetCase],
    model: "SentenceTransformer",
    *,
    chunk_size: int,
    chunk_overlap: int,
    top_k: int,
    embedding_model_name: str,
) -> ConfigurationRun:
    """Indexes document.source_path at the given chunk_size/chunk_overlap
    using model (already loaded - loading it is the expensive part, so
    callers comparing several configurations load it once and pass it
    to every evaluate_configuration() call), then runs every case in
    cases through RetrieveChunksService at the given top_k.

    Builds a fresh InMemoryVectorStore each call, so results from one
    configuration never leak into another.
    """
    passage_embedder = SentenceTransformerEmbedder(model, prefix=PASSAGE_PREFIX)
    query_embedder = SentenceTransformerEmbedder(model, prefix=QUERY_PREFIX)
    vector_store = InMemoryVectorStore()

    index_document = IndexDocumentService(
        load_document=LoadDocumentService(PypdfLoader()),
        chunk_document=ChunkDocumentService(
            FixedSizeTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        ),
        embed_chunks=EmbedChunksService(passage_embedder),
        index_chunks=IndexChunksService(vector_store),
    )
    retrieve_chunks = RetrieveChunksService(
        embedder=query_embedder, search_chunks=SearchChunksService(vector_store)
    )

    index_result = index_document.execute(document.source_path)

    case_results = [evaluate_case(case, retrieve_chunks, top_k) for case in cases]
    config = RunConfig(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        top_k=top_k,
        embedding_model_name=embedding_model_name,
        query_prefix=QUERY_PREFIX,
        passage_prefix=PASSAGE_PREFIX,
        case_count=len(cases),
        indexed_page_count=index_result.page_count,
        indexed_chunk_count=index_result.chunk_count,
        measured_at=date.today().isoformat(),
    )
    return ConfigurationRun(
        config=config, case_results=case_results, aggregate=summarize(case_results)
    )


def print_case_report(case_results: list[CaseResult]) -> None:
    print("\nPer-question results (local only - never paste this section anywhere committed):")
    for index, case_result in enumerate(case_results, start=1):
        expected_keys = sorted(location_key(page, idx) for page, idx in case_result.expected)
        print(
            f"  [{index:>2}] rr={case_result.reciprocal_rank:.2f} "
            f"recall@1/3/5={case_result.recall_at_1:.0f}/{case_result.recall_at_3:.0f}/"
            f"{case_result.recall_at_5:.0f} "
            f"expected={expected_keys} got={case_result.ranked_locations}"
        )
        print(f"        Q: {case_result.question}")


def print_verbose_case_results(case_results: list[CaseResult]) -> None:
    """Prints each question's retrieved chunks (rank/page/chunk/score and a
    text_preview), for --verbose only (local use only - never paste this
    section anywhere committed; it may contain guideline-derived content).
    """
    print(
        "\nPer-question retrieved chunks (local only - never paste this "
        "section anywhere committed):"
    )
    for index, case_result in enumerate(case_results, start=1):
        print(f"\n[{index}] Q: {case_result.question}")
        for ranked in case_result.ranked_results:
            print(
                f"  rank={ranked.rank} page={ranked.page_number} "
                f"chunk={ranked.chunk_index} score={ranked.score:.4f}"
            )
            print(f"  text_preview: {ranked.text_preview}")


def print_aggregate_report(aggregate: Aggregate, config: RunConfig) -> None:
    print("\nAggregate:")
    print(
        f"  Indexed {config.indexed_page_count} pages into "
        f"{config.indexed_chunk_count} chunks "
        f"(chunk_size={config.chunk_size}, chunk_overlap={config.chunk_overlap})."
    )
    print(f"  Recall@1 = {aggregate.recall_at_1:.3f}")
    print(f"  Recall@3 = {aggregate.recall_at_3:.3f}")
    print(f"  Recall@5 = {aggregate.recall_at_5:.3f}")
    print(f"  MRR@{config.top_k}  = {aggregate.mrr:.3f}")
    print(f"  (over {config.case_count} questions)")


def markdown_snippet(label: str, aggregate: Aggregate, config: RunConfig) -> str:
    return (
        f"## Baseline ({config.measured_at})\n\n"
        f"- Document: {label}\n"
        f"- Cases: {config.case_count}\n"
        f"- chunk_size={config.chunk_size}, chunk_overlap={config.chunk_overlap}, "
        f"top_k={config.top_k}\n"
        f"- Embedding: sentence_transformers / {config.embedding_model_name} "
        f'(query prefix: "{config.query_prefix}", passage prefix: "{config.passage_prefix}")\n'
        f"- Recall@1: {aggregate.recall_at_1:.2f}\n"
        f"- Recall@3: {aggregate.recall_at_3:.2f}\n"
        f"- Recall@5: {aggregate.recall_at_5:.2f}\n"
        f"- MRR@{config.top_k}: {aggregate.mrr:.2f}\n"
    )


def print_markdown_snippet(label: str, aggregate: Aggregate, config: RunConfig) -> None:
    print("\n" + "=" * 70)
    print("Markdown snippet for docs/baseline-retrieval-evaluation.md")
    print("(review before pasting - confirm nothing identifying leaks):")
    print("=" * 70)
    print(markdown_snippet(label, aggregate, config))


def resolve_report_path(save_report_arg: str, dataset_path: Path, name_suffix: str = "") -> Path:
    if save_report_arg != "__default__":
        return Path(save_report_arg)
    today = date.today().isoformat()
    return Path("data/eval/results") / f"{dataset_path.stem}{name_suffix}_{today}.json"


def write_local_report(path: Path, document: DatasetDocument, runs: list[ConfigurationRun]) -> None:
    """Writes one or more ConfigurationRuns (one per evaluated
    configuration) to a single local JSON report. Used with a
    one-element list by evaluate_retrieval_baseline.py and a
    multi-element list (one per chunk_size candidate) by
    compare_chunk_sizes.py.
    """
    payload = {
        "document_label": document.label,
        "runs": [
            {
                "config": asdict(run.config),
                "aggregate": asdict(run.aggregate),
                "cases": [
                    {
                        "question": case_result.question,
                        "expected": sorted(
                            location_key(page, idx) for page, idx in case_result.expected
                        ),
                        "ranked_locations": case_result.ranked_locations,
                        "recall_at_1": case_result.recall_at_1,
                        "recall_at_3": case_result.recall_at_3,
                        "recall_at_5": case_result.recall_at_5,
                        "reciprocal_rank": case_result.reciprocal_rank,
                        "ranked_results": [asdict(ranked) for ranked in case_result.ranked_results],
                    }
                    for case_result in run.case_results
                ],
            }
            for run in runs
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved detailed local report to {path}")
    print("Reminder: this file may contain guideline-derived content - never commit it.")
