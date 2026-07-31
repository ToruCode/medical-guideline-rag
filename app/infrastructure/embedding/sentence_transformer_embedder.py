"""sentence-transformers-based implementation of the Embedder port.

Loading sentence_transformers (and its transitive torch/transformers
dependencies) is expensive: importing it, and constructing a
SentenceTransformer, can take seconds and hundreds of MB of memory.
load_sentence_transformer_model is a free function (not called at
import time by this module) so the app only pays that cost when the
"sentence_transformers" provider is actually selected; see
app/api/dependencies.py, which lru_cache's the loaded model once and
shares it between the query- and passage-prefixed Embedder instances
below, and docs/adr/0011-real-embedding-and-llm-adapters.md.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


def load_sentence_transformer_model(model_name: str) -> "SentenceTransformer":
    """Downloads (on first use) and loads a sentence-transformers model."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


class SentenceTransformerEmbedder:
    """Embeds text using an already-loaded sentence-transformers model.

    prefix is prepended to every input text before embedding. Models in
    the intfloat/multilingual-e5-* family are asymmetric: they require a
    "query: " prefix for search queries and a "passage: " prefix for
    indexed text to produce good retrieval quality (see
    docs/adr/0011-real-embedding-and-llm-adapters.md). Pass "" (the
    default) for symmetric models that don't use this convention.
    """

    def __init__(self, model: "SentenceTransformer", prefix: str = "") -> None:
        self._model = model
        self._prefix = prefix

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        prefixed_texts = [f"{self._prefix}{text}" for text in texts]
        vectors = self._model.encode(prefixed_texts, convert_to_numpy=True)
        result: list[list[float]] = vectors.tolist()
        return result
