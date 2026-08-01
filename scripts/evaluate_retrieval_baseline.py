"""Manual retrieval-quality baseline measurement against a real, local
evaluation dataset (Recall@1/3/5, MRR).

Not a pytest test: unlike tests/integration/test_retrieval_evaluation.py
(Issue #17's synthetic, committed, CI-reproducible dataset), this reads
a real guideline PDF and a real question/expected-page dataset that
exist only on the machine running it - never committed, never
reproducible by another developer or CI. There is no pass/fail
threshold here; this issue (#18) is a baseline measurement only, not a
quality gate. See docs/adr/0014-real-data-retrieval-baseline.md.

Requires MEDICAL_RAG_EMBEDDING_PROVIDER=sentence_transformers (in
.env or the environment): FakeEmbedder is not semantically meaningful,
so a baseline measured with it would not reflect real retrieval
quality. Uses the real sentence-transformers model to index the real
PDF, then runs every dataset question through RetrieveChunksService
exactly as POST /questions/ask would.

Dataset format: see docs/evaluation-dataset-format.md. Never commit a
dataset file, the PDF it points to, or this script's --save-report
output; data/eval/ is gitignored for exactly this reason.

Usage (from the repo root):

    uv run python -m scripts.evaluate_retrieval_baseline \\
        --dataset data/eval/my_guideline_qa.json --save-report

Prints a per-question breakdown (local use only), an aggregate
summary, and a ready-to-review Markdown snippet for
docs/baseline-retrieval-evaluation.md (aggregate numbers and
measurement configuration only - review it for anything identifying
before pasting).
"""

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

from app.application.services.chunk_document import ChunkDocumentService
from app.application.services.embed_chunks import EmbedChunksService
from app.application.services.index_chunks import IndexChunksService
from app.application.services.index_document import IndexDocumentService
from app.application.services.load_document import LoadDocumentService
from app.application.services.retrieve_chunks import RetrieveChunksService
from app.application.services.search_chunks import SearchChunksService
from app.core.config import get_settings
from app.infrastructure.chunking.fixed_size_text_splitter import FixedSizeTextSplitter
from app.infrastructure.embedding.sentence_transformer_embedder import (
    SentenceTransformerEmbedder,
    load_sentence_transformer_model,
)
from app.infrastructure.pdf.pypdf_loader import PypdfLoader
from app.infrastructure.vector_store.in_memory_vector_store import InMemoryVectorStore
from tests.support.evaluation.metrics import mean, recall_at_k, reciprocal_rank

QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "


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
class CaseResult:
    question: str
    expected: list[tuple[int, int | None]]
    ranked_locations: list[str]
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    reciprocal_rank: float


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
    measured_at: str


def _location_key(page_number: int, chunk_index: int | None) -> str:
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


def _load_dataset(path: Path) -> tuple[DatasetDocument, list[DatasetCase]]:
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


def _evaluate_case(
    case: DatasetCase, retrieve_chunks: RetrieveChunksService, top_k: int
) -> CaseResult:
    results = retrieve_chunks.execute(case.question, top_k=top_k)
    if case.granularity == "chunk":
        ranked_ids = [
            _location_key(
                result.embedded_chunk.chunk.page_number, result.embedded_chunk.chunk.chunk_index
            )
            for result in results
        ]
    else:
        ranked_ids = [
            _location_key(result.embedded_chunk.chunk.page_number, None) for result in results
        ]
    relevant_ids = {_location_key(page, idx) for page, idx in case.expected_locations}

    return CaseResult(
        question=case.question,
        expected=case.expected_locations,
        ranked_locations=ranked_ids,
        recall_at_1=recall_at_k(ranked_ids, relevant_ids, k=1),
        recall_at_3=recall_at_k(ranked_ids, relevant_ids, k=3),
        recall_at_5=recall_at_k(ranked_ids, relevant_ids, k=5),
        reciprocal_rank=reciprocal_rank(ranked_ids, relevant_ids),
    )


def _aggregate(case_results: list[CaseResult]) -> Aggregate:
    return Aggregate(
        recall_at_1=mean([c.recall_at_1 for c in case_results]),
        recall_at_3=mean([c.recall_at_3 for c in case_results]),
        recall_at_5=mean([c.recall_at_5 for c in case_results]),
        mrr=mean([c.reciprocal_rank for c in case_results]),
    )


def _print_case_report(case_results: list[CaseResult]) -> None:
    print("\nPer-question results (local only - never paste this section anywhere committed):")
    for index, case_result in enumerate(case_results, start=1):
        expected_keys = sorted(_location_key(page, idx) for page, idx in case_result.expected)
        print(
            f"  [{index:>2}] rr={case_result.reciprocal_rank:.2f} "
            f"recall@1/3/5={case_result.recall_at_1:.0f}/{case_result.recall_at_3:.0f}/"
            f"{case_result.recall_at_5:.0f} "
            f"expected={expected_keys} got={case_result.ranked_locations}"
        )
        print(f"        Q: {case_result.question}")


def _print_aggregate_report(aggregate: Aggregate, config: RunConfig) -> None:
    print("\nAggregate:")
    print(f"  Recall@1 = {aggregate.recall_at_1:.3f}")
    print(f"  Recall@3 = {aggregate.recall_at_3:.3f}")
    print(f"  Recall@5 = {aggregate.recall_at_5:.3f}")
    print(f"  MRR@{config.top_k}  = {aggregate.mrr:.3f}")
    print(f"  (over {config.case_count} questions)")


def _markdown_snippet(label: str, aggregate: Aggregate, config: RunConfig) -> str:
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


def _print_markdown_snippet(label: str, aggregate: Aggregate, config: RunConfig) -> None:
    print("\n" + "=" * 70)
    print("Markdown snippet for docs/baseline-retrieval-evaluation.md")
    print("(review before pasting - confirm nothing identifying leaks):")
    print("=" * 70)
    print(_markdown_snippet(label, aggregate, config))


def _resolve_report_path(save_report_arg: str, dataset_path: Path) -> Path:
    if save_report_arg != "__default__":
        return Path(save_report_arg)
    today = date.today().isoformat()
    return Path("data/eval/results") / f"{dataset_path.stem}_{today}.json"


def _write_local_report(
    path: Path,
    document: DatasetDocument,
    case_results: list[CaseResult],
    aggregate: Aggregate,
    config: RunConfig,
) -> None:
    payload = {
        "document_label": document.label,
        "config": asdict(config),
        "aggregate": asdict(aggregate),
        "cases": [
            {
                "question": case_result.question,
                "expected": sorted(_location_key(page, idx) for page, idx in case_result.expected),
                "ranked_locations": case_result.ranked_locations,
                "recall_at_1": case_result.recall_at_1,
                "recall_at_3": case_result.recall_at_3,
                "recall_at_5": case_result.recall_at_5,
                "reciprocal_rank": case_result.reciprocal_rank,
            }
            for case_result in case_results
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved detailed local report to {path}")
    print("Reminder: this file may contain guideline-derived content - never commit it.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure a retrieval baseline (Recall@1/3/5, MRR) against a local, "
            "real evaluation dataset. See docs/evaluation-dataset-format.md."
        )
    )
    parser.add_argument(
        "--dataset", required=True, help="Path to a local dataset JSON file (never committed)."
    )
    parser.add_argument(
        "--top-k", type=int, default=5, help="Retrieval depth; must be >= 5 (default: 5)."
    )
    parser.add_argument(
        "--save-report",
        nargs="?",
        const="__default__",
        default=None,
        metavar="PATH",
        help=(
            "Save detailed per-question results locally (never commit this file). "
            "Defaults to data/eval/results/<dataset-name>_<date>.json when given "
            "without a value."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    dataset_path = Path(args.dataset)
    if args.top_k < 5:
        raise SystemExit("--top-k must be at least 5 (Recall@5 requires it).")

    settings = get_settings()
    if settings.embedding_provider != "sentence_transformers":
        raise SystemExit(
            "MEDICAL_RAG_EMBEDDING_PROVIDER must be 'sentence_transformers' for a "
            "meaningful baseline measurement (FakeEmbedder is not semantically "
            "meaningful - its vectors are derived from text length only)."
        )

    document, cases = _load_dataset(dataset_path)

    model = load_sentence_transformer_model(settings.embedding_model_name)
    passage_embedder = SentenceTransformerEmbedder(model, prefix=PASSAGE_PREFIX)
    query_embedder = SentenceTransformerEmbedder(model, prefix=QUERY_PREFIX)
    vector_store = InMemoryVectorStore()

    index_document = IndexDocumentService(
        load_document=LoadDocumentService(PypdfLoader()),
        chunk_document=ChunkDocumentService(
            FixedSizeTextSplitter(
                chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
            )
        ),
        embed_chunks=EmbedChunksService(passage_embedder),
        index_chunks=IndexChunksService(vector_store),
    )
    retrieve_chunks = RetrieveChunksService(
        embedder=query_embedder, search_chunks=SearchChunksService(vector_store)
    )

    index_result = index_document.execute(document.source_path)
    print(
        f"Indexed {index_result.page_count} pages into {index_result.chunk_count} chunks "
        f"(chunk_size={settings.chunk_size}, chunk_overlap={settings.chunk_overlap})."
    )

    case_results = [_evaluate_case(case, retrieve_chunks, args.top_k) for case in cases]
    aggregate = _aggregate(case_results)
    config = RunConfig(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        top_k=args.top_k,
        embedding_model_name=settings.embedding_model_name,
        query_prefix=QUERY_PREFIX,
        passage_prefix=PASSAGE_PREFIX,
        case_count=len(cases),
        measured_at=date.today().isoformat(),
    )

    _print_case_report(case_results)
    _print_aggregate_report(aggregate, config)
    _print_markdown_snippet(document.label, aggregate, config)

    if args.save_report is not None:
        report_path = _resolve_report_path(args.save_report, dataset_path)
        _write_local_report(report_path, document, case_results, aggregate, config)


if __name__ == "__main__":
    main()
