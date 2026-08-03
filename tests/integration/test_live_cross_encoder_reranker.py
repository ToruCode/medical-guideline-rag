"""Opt-in test against a real, downloaded Cross Encoder reranker model.

Skipped by default: downloading and loading a real model can take a
long time on first run and requires network access. Set RUN_SLOW_TESTS=1
to run it, e.g. after changing CrossEncoderReranker or the configured
reranker model name, to catch real integration issues that mocked unit
tests (tests/unit/test_reranker.py) cannot.
"""

import os

import pytest
from scripts.reranker import CrossEncoderReranker, load_cross_encoder

pytestmark = pytest.mark.skipif(
    not os.environ.get("RUN_SLOW_TESTS"),
    reason="downloads a real Cross Encoder model; set RUN_SLOW_TESTS=1 to run",
)


def test_real_cross_encoder_scores_relevant_text_higher() -> None:
    model = load_cross_encoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1", device="cpu")
    reranker = CrossEncoderReranker(model, batch_size=8)

    scores = reranker.score(
        "What is the adult dosage of Medicamentum X?",
        [
            "Adults should take 500 mg of Medicamentum X twice daily with food.",
            "Unrelated administrative scheduling text about clinic opening hours.",
        ],
    )

    assert len(scores) == 2
    assert scores[0] > scores[1]
