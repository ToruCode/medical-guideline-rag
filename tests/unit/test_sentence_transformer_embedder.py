"""Unit tests for SentenceTransformerEmbedder.

Never loads a real model: the underlying SentenceTransformer is either
a hand-written fake object or (for load_sentence_transformer_model) a
monkeypatched stand-in for the sentence_transformers.SentenceTransformer
class, so these tests need no network access and no model download.
"""

import numpy as np
import pytest
from app.infrastructure.embedding.sentence_transformer_embedder import (
    SentenceTransformerEmbedder,
    load_sentence_transformer_model,
)


class FakeSentenceTransformerModel:
    def __init__(self) -> None:
        self.received_texts: list[str] | None = None

    def encode(self, texts: list[str], convert_to_numpy: bool = True) -> np.ndarray:
        self.received_texts = texts
        return np.array([[float(len(text)), 0.0] for text in texts])


def test_embed_prepends_prefix_to_every_text() -> None:
    model = FakeSentenceTransformerModel()
    embedder = SentenceTransformerEmbedder(model, prefix="query: ")  # type: ignore[arg-type]

    embedder.embed(["hello", "world"])

    assert model.received_texts == ["query: hello", "query: world"]


def test_embed_with_no_prefix_passes_texts_unchanged() -> None:
    model = FakeSentenceTransformerModel()
    embedder = SentenceTransformerEmbedder(model)  # type: ignore[arg-type]

    embedder.embed(["hello"])

    assert model.received_texts == ["hello"]


def test_embed_converts_numpy_array_to_list_of_lists() -> None:
    model = FakeSentenceTransformerModel()
    embedder = SentenceTransformerEmbedder(model)  # type: ignore[arg-type]

    vectors = embedder.embed(["ab", "abcd"])

    assert vectors == [[2.0, 0.0], [4.0, 0.0]]
    assert all(isinstance(vector, list) for vector in vectors)


def test_embed_with_empty_texts_returns_empty_list_without_calling_model() -> None:
    model = FakeSentenceTransformerModel()
    embedder = SentenceTransformerEmbedder(model)  # type: ignore[arg-type]

    result = embedder.embed([])

    assert result == []
    assert model.received_texts is None


def test_load_sentence_transformer_model_constructs_with_model_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    constructed_with: list[str] = []

    class FakeSentenceTransformerClass:
        def __init__(self, model_name: str) -> None:
            constructed_with.append(model_name)

    monkeypatch.setattr("sentence_transformers.SentenceTransformer", FakeSentenceTransformerClass)

    load_sentence_transformer_model("intfloat/multilingual-e5-base")

    assert constructed_with == ["intfloat/multilingual-e5-base"]
