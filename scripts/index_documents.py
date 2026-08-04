"""CLI to index local PDF guideline documents into the persistent Qdrant
vector store (MEDICAL_RAG_VECTOR_STORE_PROVIDER=qdrant).

Never run this against files that must not be committed - PDFs under
data/raw/ and the resulting vector_store_path directory are both
gitignored; see CLAUDE.md's data/copyright rules and
docs/adr/0026-persistent-vector-store.md.

Usage (from the repo root):

    # Index every *.pdf under data/raw/ (default --input-dir), reusing
    # the existing persistent index (chunk_id-based upsert - re-indexing
    # the same file updates its entries in place, it does not duplicate
    # them).
    uv run python -m scripts.index_documents

    # Index specific files instead of scanning --input-dir.
    uv run python -m scripts.index_documents data/raw/guideline_a.pdf

    # Discard the existing persistent index first, then index from
    # scratch - e.g. after changing embedding_provider/embedding_model_name,
    # since old and new vectors would otherwise have incompatible
    # dimensions.
    uv run python -m scripts.index_documents --rebuild
"""

import argparse
import sys
from pathlib import Path

from app.application.services.chunk_document import ChunkDocumentService
from app.application.services.embed_chunks import EmbedChunksService
from app.application.services.index_chunks import IndexChunksService
from app.application.services.index_document import IndexDocumentService
from app.application.services.load_document import LoadDocumentService
from app.core.config import Settings, get_settings
from app.domain.exceptions.chunk import InvalidChunkConfigError
from app.domain.exceptions.document import DocumentLoadError
from app.domain.exceptions.embedding import EmbeddingError
from app.domain.exceptions.vector_store import VectorStoreError
from app.domain.ports.embedder import Embedder
from app.domain.ports.pdf_loader import PdfLoader
from app.infrastructure.chunking.fixed_size_text_splitter import FixedSizeTextSplitter
from app.infrastructure.embedding.fake_embedder import FakeEmbedder
from app.infrastructure.embedding.sentence_transformer_embedder import (
    SentenceTransformerEmbedder,
    load_sentence_transformer_model,
)
from app.infrastructure.pdf.pymupdf_loader import PyMuPdfLoader
from app.infrastructure.pdf.pypdf_loader import PypdfLoader
from app.infrastructure.vector_store.qdrant_vector_store import QdrantVectorStore

DEFAULT_INPUT_DIR = "data/raw"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Index local PDF guideline documents into the persistent Qdrant "
            "vector store (MEDICAL_RAG_VECTOR_STORE_PROVIDER=qdrant)."
        )
    )
    parser.add_argument(
        "pdf_paths",
        nargs="*",
        help=(
            f"PDF files to index. Defaults to every *.pdf under --input-dir "
            f"({DEFAULT_INPUT_DIR}) when omitted."
        ),
    )
    parser.add_argument(
        "--input-dir",
        default=DEFAULT_INPUT_DIR,
        help=(
            "Directory scanned for *.pdf when no PDF_PATHS are given "
            f"(default: {DEFAULT_INPUT_DIR})."
        ),
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help=(
            "Discard the existing persistent index before indexing (e.g. after "
            "changing embedding_provider/embedding_model_name). Without this "
            "flag, the existing index is reused."
        ),
    )
    return parser.parse_args()


def _resolve_pdf_paths(args: argparse.Namespace) -> list[Path]:
    if args.pdf_paths:
        return [Path(p) for p in args.pdf_paths]
    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {input_dir}")
    return sorted(input_dir.glob("*.pdf"))


def _build_pdf_loader(settings: Settings) -> PdfLoader:
    if settings.pdf_extractor == "pymupdf":
        return PyMuPdfLoader()
    if settings.pdf_extractor == "pypdf":
        return PypdfLoader()
    raise ValueError(f"Unknown pdf_extractor: {settings.pdf_extractor!r}")


def _build_passage_embedder(settings: Settings) -> Embedder:
    if settings.embedding_provider == "fake":
        return FakeEmbedder()
    if settings.embedding_provider == "sentence_transformers":
        model = load_sentence_transformer_model(settings.embedding_model_name)
        return SentenceTransformerEmbedder(model, prefix="passage: ")
    raise ValueError(f"Unknown embedding_provider: {settings.embedding_provider!r}")


def _build_index_document_service(
    settings: Settings, vector_store: QdrantVectorStore
) -> IndexDocumentService:
    return IndexDocumentService(
        load_document=LoadDocumentService(_build_pdf_loader(settings)),
        chunk_document=ChunkDocumentService(
            FixedSizeTextSplitter(
                chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
            )
        ),
        embed_chunks=EmbedChunksService(_build_passage_embedder(settings)),
        index_chunks=IndexChunksService(vector_store),
    )


def main() -> None:
    args = _parse_args()
    settings = get_settings()
    if settings.vector_store_provider != "qdrant":
        raise SystemExit(
            "MEDICAL_RAG_VECTOR_STORE_PROVIDER must be 'qdrant' to use this script "
            f"(got {settings.vector_store_provider!r}). Indexing into the 'memory' "
            "provider from a one-shot CLI process would be discarded immediately "
            "when the process exits."
        )

    pdf_paths = _resolve_pdf_paths(args)
    if not pdf_paths:
        raise SystemExit("No PDF files found to index.")

    vector_store = QdrantVectorStore(
        path=settings.vector_store_path, collection_name=settings.vector_store_collection_name
    )
    if args.rebuild:
        print(f"Rebuilding collection {settings.vector_store_collection_name!r}...")
        vector_store.rebuild()

    service = _build_index_document_service(settings, vector_store)

    failures: list[str] = []
    for pdf_path in pdf_paths:
        try:
            result = service.execute(pdf_path)
        except (
            DocumentLoadError,
            InvalidChunkConfigError,
            EmbeddingError,
            VectorStoreError,
        ) as exc:
            print(f"FAILED  {pdf_path}: {exc}", file=sys.stderr)
            failures.append(str(pdf_path))
            continue
        print(
            f"OK      {pdf_path}: document_id={result.document_id} "
            f"pages={result.page_count} chunks={result.chunk_count} "
            f"indexed={result.indexed_count}"
        )

    vector_store.close()

    if failures:
        raise SystemExit(f"{len(failures)} of {len(pdf_paths)} file(s) failed to index.")


if __name__ == "__main__":
    main()
