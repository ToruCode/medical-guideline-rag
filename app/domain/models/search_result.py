"""Search result domain model."""

from dataclasses import dataclass

from app.domain.models.embedding import EmbeddedChunk


@dataclass(frozen=True, slots=True)
class SearchResult:
    """An EmbeddedChunk matched by a similarity search, with its score.

    score follows a single convention regardless of which VectorStore
    implementation produced it: a higher value always means a closer
    match. score is intentionally not clamped to [0.0, 1.0], since
    cosine similarity ranges over [-1.0, 1.0] and other metrics (e.g.
    dot product) are unbounded. Translating a concrete backend's native
    distance metric (e.g. Qdrant/pgvector distance, where lower is
    closer) into this "higher is more similar" convention is the
    responsibility of the infrastructure-layer VectorStore
    implementation, not the caller.
    """

    embedded_chunk: EmbeddedChunk
    score: float
