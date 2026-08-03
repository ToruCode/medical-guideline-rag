"""Reranker abstraction for Issue #25's Cross Encoder reranking
comparison.

Reranker is a narrow Protocol specifically so the Cross Encoder model
used here can later be swapped for a different reranking strategy - an
API-based reranker (e.g. Cohere Rerank), BAAI/bge-reranker, etc. -
without changing scripts/reranking_core.py's calling code, mirroring
scripts/hybrid_scorer.py's ScoreFuser design from Issue #24.

CrossEncoderReranker.score() returns the model's raw output score, used
directly for ranking - this issue does not blend it with Dense/BM25
scores (see docs/adr/0020-cross-encoder-reranker-comparison.md for why).
"""

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder


class Reranker(Protocol):
    """Scores each of texts against query; higher means a better match.

    Implementations must return a list the same length as texts, in
    the same order (texts[i]'s score at result[i]).
    """

    def score(self, query: str, texts: list[str]) -> list[float]: ...


def resolve_device(device: str) -> str:
    """Resolves "auto" to "cuda" if a CUDA device is available, else
    "cpu". "cpu"/"cuda" pass through unchanged - this project's default
    CLI device is "cpu" (Windows+CPU reproducibility takes priority;
    see docs/adr/0020-cross-encoder-reranker-comparison.md), so "auto"
    is only relevant when a caller explicitly opts into GPU-if-available.
    """
    if device != "auto":
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def load_cross_encoder(model_name: str, device: str) -> "CrossEncoder":
    """Downloads (on first use) and loads a sentence-transformers
    CrossEncoder model. A free function (not called at import time),
    mirroring load_sentence_transformer_model
    (app/infrastructure/embedding/sentence_transformer_embedder.py):
    loading a real Cross Encoder model is expensive (network access,
    seconds of load time), so callers using FakeReranker in tests never
    pay that cost.
    """
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name, device=resolve_device(device))


class CrossEncoderReranker:
    """Reranker backed by an already-loaded sentence-transformers
    CrossEncoder model (loaded once - see load_cross_encoder() - and
    reused across every question, never reloaded per query).

    model is typed Any rather than CrossEncoder: CrossEncoder.predict's
    real declared signature is a heavily overloaded, multi-modal (text/
    image/audio/video) type that mypy cannot match structurally against
    a plain predict(pairs, batch_size) Protocol, and CrossEncoderReranker
    only ever calls predict() with a list[tuple[str, str]] - a usage
    sentence-transformers documents and supports at runtime regardless
    of what the type stubs express. Any also lets
    tests/unit/test_reranker.py exercise this class against a
    lightweight stub object without importing sentence_transformers.
    """

    def __init__(self, model: Any, batch_size: int = 8) -> None:
        self._model = model
        self._batch_size = batch_size

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []

        pairs = [(query, text) for text in texts]
        scores = self._model.predict(pairs, batch_size=self._batch_size)
        return [float(score) for score in scores]


class FakeReranker:
    """Deterministic, dependency-free Reranker for unit tests - never
    loads a real model or performs network access.

    scores_by_text maps chunk text -> score, so a test can construct
    any desired ranking outcome (e.g. to force a rank improvement or
    regression relative to Hybrid's own ordering). Text not present in
    scores_by_text scores 0.0. When scores_by_text is omitted, scores by
    text length instead (longer text ranks higher) - a simple default
    that still varies across distinct inputs.
    """

    def __init__(self, scores_by_text: dict[str, float] | None = None) -> None:
        self._scores_by_text = scores_by_text

    def score(self, query: str, texts: list[str]) -> list[float]:
        if self._scores_by_text is not None:
            return [self._scores_by_text.get(text, 0.0) for text in texts]
        return [float(len(text)) for text in texts]
