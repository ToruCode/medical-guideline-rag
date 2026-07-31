"""Opt-in test against a real, downloaded sentence-transformers model.

Skipped by default: downloading and loading a real model can take a
long time on first run and requires network access. Set RUN_SLOW_TESTS=1
to run it, e.g. after changing SentenceTransformerEmbedder or the
configured model name, to catch real integration issues that mocked
unit tests (tests/unit/test_sentence_transformer_embedder.py) cannot.
"""

import os

import pytest
from app.infrastructure.embedding.sentence_transformer_embedder import (
    SentenceTransformerEmbedder,
    load_sentence_transformer_model,
)

pytestmark = pytest.mark.skipif(
    not os.environ.get("RUN_SLOW_TESTS"),
    reason="downloads a real sentence-transformers model; set RUN_SLOW_TESTS=1 to run",
)


def test_real_model_embeds_texts_into_distinct_same_length_vectors() -> None:
    model = load_sentence_transformer_model("intfloat/multilingual-e5-base")
    embedder = SentenceTransformerEmbedder(model, prefix="passage: ")

    vectors = embedder.embed(
        ["dosage guidance for adults", "unrelated administrative scheduling text"]
    )

    assert len(vectors) == 2
    assert len(vectors[0]) == len(vectors[1])
    assert vectors[0] != vectors[1]


def test_real_model_query_and_passage_prefixes_both_produce_vectors() -> None:
    model = load_sentence_transformer_model("intfloat/multilingual-e5-base")
    passage_embedder = SentenceTransformerEmbedder(model, prefix="passage: ")
    query_embedder = SentenceTransformerEmbedder(model, prefix="query: ")

    passage_vectors = passage_embedder.embed(["dosage guidance for adults"])
    query_vectors = query_embedder.embed(["what is the dosage guidance"])

    assert len(passage_vectors[0]) == len(query_vectors[0])
