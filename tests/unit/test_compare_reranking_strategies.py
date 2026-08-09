import argparse

import pytest
from scripts.compare_reranking_strategies import _parse_args, _parse_strategies
from scripts.reranking_core import ComparisonRow, markdown_comparison_table


def test_parse_strategies_splits_names() -> None:
    assert _parse_strategies("dense,hybrid,hybrid_rerank") == ["dense", "hybrid", "hybrid_rerank"]


def test_parse_strategies_strips_whitespace() -> None:
    assert _parse_strategies(" dense , hybrid_rerank ") == ["dense", "hybrid_rerank"]


def test_parse_strategies_rejects_empty_string() -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_strategies("")


def test_parse_strategies_rejects_unknown_name() -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="Unknown strategy"):
        _parse_strategies("dense,rrf")


def test_default_args(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sys.argv", ["compare_reranking_strategies.py", "--dataset", "data/eval/x.json"]
    )

    args = _parse_args()

    assert args.strategies == ["dense", "hybrid", "hybrid_rerank"]
    assert args.top_k == 5
    assert args.dense_candidate_k == 20
    assert args.bm25_candidate_k == 20
    assert args.reranker_candidate_k == 10
    assert args.alpha == 0.7
    assert args.reranker_model_name == "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    assert args.device == "cpu"
    assert args.batch_size == 8


def test_device_rejects_invalid_choice(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "compare_reranking_strategies.py",
            "--dataset",
            "data/eval/x.json",
            "--device",
            "tpu",
        ],
    )

    with pytest.raises(SystemExit):
        _parse_args()


def _make_row(strategy: str, alpha: float | None, reranker_model_name: str | None) -> ComparisonRow:
    return ComparisonRow(
        strategy=strategy,
        alpha=alpha,
        reranker_model_name=reranker_model_name,
        recall_at_1=0.583,
        recall_at_3=0.833,
        recall_at_5=0.917,
        mrr=0.697,
        avg_latency_ms=12.3,
    )


def test_markdown_comparison_table_includes_one_row_per_strategy() -> None:
    rows = [
        _make_row("dense", alpha=None, reranker_model_name=None),
        _make_row("hybrid", alpha=0.7, reranker_model_name=None),
        _make_row(
            "hybrid_rerank",
            alpha=0.7,
            reranker_model_name="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
        ),
    ]

    table = markdown_comparison_table("Guideline A", case_count=12, top_k=5, rows=rows)

    assert "Document: Guideline A" in table
    assert "top_k=5" in table
    assert "| dense |  | none | 0.58 | 0.83 | 0.92 | 0.70 | 12.3 |" in table
    assert "| hybrid | 0.7 | none | 0.58 | 0.83 | 0.92 | 0.70 | 12.3 |" in table
    assert (
        "| hybrid_rerank | 0.7 | cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 | "
        "0.58 | 0.83 | 0.92 | 0.70 | 12.3 |" in table
    )


def test_markdown_comparison_table_never_includes_question_text() -> None:
    rows = [_make_row("dense", alpha=None, reranker_model_name=None)]

    table = markdown_comparison_table("Guideline A", case_count=12, top_k=5, rows=rows)

    assert "case_results" not in table
    assert "text_preview" not in table
