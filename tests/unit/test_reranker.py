"""Unit tests for scripts/reranker.py.

CrossEncoderReranker is tested against a stub model object (recording
calls, returning canned scores) - no real sentence-transformers
CrossEncoder is loaded or downloaded. See
tests/integration/test_live_cross_encoder_reranker.py for the
real-model smoke test (skipped unless RUN_SLOW_TESTS=1).
"""

import pytest
from scripts.reranker import CrossEncoderReranker, FakeReranker, resolve_device


class _StubCrossEncoderModel:
    """Records every predict() call's arguments and returns canned scores."""

    def __init__(self, scores: list[float]) -> None:
        self._scores = scores
        self.predict_calls: list[tuple[list[tuple[str, str]], int]] = []

    def predict(self, pairs: list[tuple[str, str]], batch_size: int = 32) -> list[float]:
        self.predict_calls.append((pairs, batch_size))
        return self._scores


# --- CrossEncoderReranker ---


def test_cross_encoder_reranker_passes_query_text_pairs() -> None:
    model = _StubCrossEncoderModel(scores=[0.1, 0.9])
    reranker = CrossEncoderReranker(model, batch_size=8)

    scores = reranker.score("what is the dosage?", ["text about dosage", "unrelated text"])

    assert scores == [0.1, 0.9]
    assert len(model.predict_calls) == 1
    pairs, batch_size = model.predict_calls[0]
    assert pairs == [
        ("what is the dosage?", "text about dosage"),
        ("what is the dosage?", "unrelated text"),
    ]
    assert batch_size == 8


def test_cross_encoder_reranker_returns_plain_floats() -> None:
    model = _StubCrossEncoderModel(scores=[0.5])
    reranker = CrossEncoderReranker(model)

    scores = reranker.score("q", ["one text"])

    assert scores == [0.5]
    assert all(isinstance(score, float) for score in scores)


def test_cross_encoder_reranker_handles_empty_texts_without_calling_model() -> None:
    model = _StubCrossEncoderModel(scores=[])
    reranker = CrossEncoderReranker(model)

    scores = reranker.score("q", [])

    assert scores == []
    assert model.predict_calls == []


# --- FakeReranker ---


def test_fake_reranker_uses_provided_scores_by_text() -> None:
    reranker = FakeReranker(scores_by_text={"a": 1.0, "b": 2.0})

    scores = reranker.score("q", ["a", "b"])

    assert scores == [1.0, 2.0]


def test_fake_reranker_defaults_missing_text_to_zero() -> None:
    reranker = FakeReranker(scores_by_text={"a": 1.0})

    scores = reranker.score("q", ["a", "unknown"])

    assert scores == [1.0, 0.0]


def test_fake_reranker_default_scores_by_text_length() -> None:
    reranker = FakeReranker()

    scores = reranker.score("q", ["short", "a much longer piece of text"])

    assert scores[0] < scores[1]


def test_fake_reranker_handles_empty_texts() -> None:
    reranker = FakeReranker()

    assert reranker.score("q", []) == []


# --- resolve_device ---


def test_resolve_device_passes_through_cpu() -> None:
    assert resolve_device("cpu") == "cpu"


def test_resolve_device_passes_through_cuda() -> None:
    assert resolve_device("cuda") == "cuda"


def test_resolve_device_auto_falls_back_to_cpu_without_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert resolve_device("auto") == "cpu"
