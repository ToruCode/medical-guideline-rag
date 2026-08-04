"""Developer-only Streamlit dashboard for inspecting local answer-quality
evaluation reports (Issue #15).

Presentation only, mirroring app/ui/streamlit_app.py's convention (Issue
#13): all report loading lives in scripts.evaluation_report_loader, and
all filtering/comparison logic lives in scripts.evaluation_dashboard_core
and scripts.answer_quality_core - this module only renders their output.
It never re-runs evaluation (no Embedder/Llm/VectorStore is constructed
here) and never imports Application/Domain/Infrastructure code.

Reports live only under a local, gitignored directory
(data/eval/results/ by default - see docs/evaluation-dataset-format.md)
and may contain guideline-derived content and generated answer text; this
tool is for local development use only and must never be deployed
alongside the production API.

Run with: uv run streamlit run app/ui/evaluation_dashboard.py
"""

from pathlib import Path

import streamlit as st
from scripts.answer_quality_core import AnswerCaseResult, AnswerConfigurationRun, failure_reasons
from scripts.evaluation_dashboard_core import (
    available_categories,
    available_difficulties,
    compare_aggregates,
    filter_case_results,
)
from scripts.evaluation_report_loader import (
    DEFAULT_REPORTS_DIR,
    EvaluationReport,
    load_answer_quality_reports,
)

st.set_page_config(page_title="Evaluation Dashboard", page_icon="📊")


def _run_label(report: EvaluationReport, run: AnswerConfigurationRun, index: int) -> str:
    llm_label = run.config.llm_provider
    if run.config.llm_model_name:
        llm_label += f"/{run.config.llm_model_name}"
    return f"{report.path.name} [run {index}] ({llm_label}, {run.config.measured_at})"


def _case_row(case: AnswerCaseResult) -> dict[str, object]:
    return {
        "question": case.question,
        "category": case.category or "-",
        "difficulty": case.difficulty or "-",
        "citation_precision": case.citation_precision,
        "citation_recall": case.citation_recall,
        "answer_point_coverage": case.answer_point_coverage,
        "insufficient_evidence_correct": case.insufficient_evidence_correct,
        "citations_consistent": case.citations_consistent,
        "latency_seconds": round(case.latency_seconds, 3),
    }


def _render_aggregate(run: AnswerConfigurationRun) -> None:
    aggregate = run.aggregate
    columns = st.columns(3)
    precision = aggregate.mean_citation_precision
    recall = aggregate.mean_citation_recall
    columns[0].metric("Citation precision", "n/a" if precision is None else f"{precision:.3f}")
    columns[1].metric("Citation recall", "n/a" if recall is None else f"{recall:.3f}")
    columns[2].metric(
        "Answer point coverage",
        "n/a"
        if aggregate.mean_answer_point_coverage is None
        else f"{aggregate.mean_answer_point_coverage:.3f}",
    )
    columns = st.columns(3)
    columns[0].metric(
        "Insufficient evidence accuracy", f"{aggregate.insufficient_evidence_accuracy:.3f}"
    )
    columns[1].metric("Average latency (s)", f"{aggregate.mean_latency_seconds:.3f}")
    columns[2].metric("Citation consistency violations", aggregate.citation_consistency_violations)


def _select_report_and_run(
    reports: list[EvaluationReport], *, key_prefix: str
) -> tuple[EvaluationReport, AnswerConfigurationRun] | None:
    report_index = st.selectbox(
        "レポート",
        options=range(len(reports)),
        format_func=lambda i: reports[i].path.name,
        key=f"{key_prefix}_report",
    )
    report = reports[report_index]
    if not report.runs:
        st.warning("このレポートには run が含まれていません。")
        return None
    run_index = 0
    if len(report.runs) > 1:
        run_index = st.selectbox(
            "run",
            options=range(len(report.runs)),
            format_func=lambda i: _run_label(report, report.runs[i], i),
            key=f"{key_prefix}_run",
        )
    return report, report.runs[run_index]


def _render_single_report_view(reports: list[EvaluationReport]) -> None:
    selection = _select_report_and_run(reports, key_prefix="single")
    if selection is None:
        return
    _report, run = selection

    st.subheader("全体指標")
    _render_aggregate(run)

    st.subheader("質問ごとの結果")
    failures_only = st.checkbox("失敗ケースのみ表示", value=False)
    categories = available_categories(run.case_results)
    difficulties = available_difficulties(run.case_results)
    category = st.selectbox("category", options=["(すべて)"] + categories)
    difficulty = st.selectbox("difficulty", options=["(すべて)"] + difficulties)
    question_contains = st.text_input("質問文で検索", value="")

    filtered = filter_case_results(
        run.case_results,
        failures_only=failures_only,
        category=None if category == "(すべて)" else category,
        difficulty=None if difficulty == "(すべて)" else difficulty,
        question_contains=question_contains or None,
    )
    st.caption(f"{len(filtered)} / {len(run.case_results)} 件を表示")
    st.dataframe([_case_row(case) for case in filtered], use_container_width=True)

    st.subheader("失敗分析")
    failing_cases = [case for case in filtered if failure_reasons(case)]
    if not failing_cases:
        st.success("表示中のケースに失敗はありません。")
    else:
        for case in failing_cases:
            with st.expander(f"{case.question}"):
                st.write("失敗理由: " + ", ".join(failure_reasons(case)))
                st.caption("回答プレビュー（ローカル専用、絶対にコミットしないこと）:")
                st.text(case.answer_preview)


def _render_comparison_view(reports: list[EvaluationReport]) -> None:
    if len(reports) < 2:
        st.info("比較するには2件以上のレポートが必要です。")
        return

    left, right = st.columns(2)
    with left:
        st.markdown("**Run A**")
        selection_a = _select_report_and_run(reports, key_prefix="compare_a")
    with right:
        st.markdown("**Run B**")
        selection_b = _select_report_and_run(reports, key_prefix="compare_b")

    if selection_a is None or selection_b is None:
        return
    _, run_a = selection_a
    _, run_b = selection_b

    comparisons = compare_aggregates(run_a.aggregate, run_b.aggregate)
    status_labels = {
        "improved": "🟢 改善",
        "degraded": "🔴 悪化",
        "unchanged": "⚪ 変化なし",
        "unavailable": "- データなし",
    }
    st.dataframe(
        [
            {
                "指標": comparison.name,
                "A": "n/a" if comparison.value_a is None else round(comparison.value_a, 3),
                "B": "n/a" if comparison.value_b is None else round(comparison.value_b, 3),
                "差分": "n/a" if comparison.delta is None else round(comparison.delta, 3),
                "判定": status_labels[comparison.status],
            }
            for comparison in comparisons
        ],
        use_container_width=True,
    )


def main() -> None:
    st.title("評価ダッシュボード")
    st.caption(
        "開発者向けのローカル専用ツールです。実運用のAPIとは独立しており、"
        "本番トラフィックやAPIキーは一切扱いません。"
    )

    reports_dir_input = st.sidebar.text_input(
        "レポートディレクトリ", value=str(DEFAULT_REPORTS_DIR)
    )
    reports = load_answer_quality_reports(Path(reports_dir_input))

    if not reports:
        st.info(
            f"'{reports_dir_input}' に回答品質評価レポート（Issue #10形式）が見つかりません。"
            " `uv run python -m scripts.evaluate_answer_quality --save-report` で作成してください。"
        )
        return

    tab_single, tab_compare = st.tabs(["レポート表示", "比較モード"])
    with tab_single:
        _render_single_report_view(reports)
    with tab_compare:
        _render_comparison_view(reports)


main()
