"""Self-authored question / expected-page dataset for retrieval evaluation.

SAMPLE_PAGES is fictional placeholder content (a made-up drug name, no
real guideline or patient content), the same spirit as
tests/integration/test_live_rag_e2e.py's sample sentence, per
CLAUDE.md's data/copyright rules. It is built into a PDF at test-run
time via tests/support/pdf_factory.build_pdf and never committed as a
file.
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    """One evaluation question and the 1-based page number(s) that answer it.

    expected_page_numbers matches Chunk.page_number. Every SAMPLE_PAGES
    entry is far shorter than the default chunk_size (1000 chars), so
    each page becomes exactly one chunk (chunk_index 0); a page number
    alone is therefore a sufficient, stable ground-truth identifier for
    this dataset, without needing a full chunk_id.
    """

    question: str
    expected_page_numbers: list[int]


SAMPLE_PAGES: list[str] = [
    "Adults should take 500 mg of Medicamentum X twice daily with food.",
    "Pediatric dosing of Medicamentum X is weight-based; consult a "
    "specialist before prescribing to children under 12.",
    "Common side effects of Medicamentum X include mild nausea and headache.",
    "Medicamentum X is contraindicated in patients with severe renal impairment.",
    "Medicamentum X should be stored below 25 degrees Celsius, away from direct sunlight.",
    "Missed doses of Medicamentum X should be taken as soon as remembered, unless it "
    "is almost time for the next scheduled dose.",
    "Medicamentum X may interact with anticoagulant medications; review a patient's "
    "full medication list before prescribing.",
    "Overdose of Medicamentum X requires immediate medical attention; symptoms include "
    "severe dizziness and an irregular heartbeat.",
]

# Kept in sync 1:1 with SAMPLE_PAGES: page N (1-based) is SAMPLE_PAGES[N - 1].
EVALUATION_CASES: list[EvaluationCase] = [
    EvaluationCase("What is the adult dosage of Medicamentum X?", [1]),
    EvaluationCase("How should Medicamentum X be dosed in children?", [2]),
    EvaluationCase("What are the common side effects of Medicamentum X?", [3]),
    EvaluationCase("When is Medicamentum X contraindicated?", [4]),
    EvaluationCase("How should Medicamentum X be stored?", [5]),
    EvaluationCase("What should a patient do if they miss a dose of Medicamentum X?", [6]),
    EvaluationCase("Does Medicamentum X interact with other medications?", [7]),
    EvaluationCase("What are the symptoms of a Medicamentum X overdose?", [8]),
]


@dataclass(frozen=True, slots=True)
class AnswerEvaluationCase:
    """One evaluation question plus the ground truth needed to measure
    answer quality and citation consistency (Issue #10), not just
    retrieval (contrast with EvaluationCase above).

    expected_answer_points are short, lexical substrings a correct
    answer should contain - deliberately not full sentences, since
    answer_point_coverage (tests/support/evaluation/metrics.py) is a
    case-insensitive substring match, not a semantic one; see
    docs/adr/0023-answer-quality-and-citation-consistency-evaluation.md.
    """

    question: str
    expected_page_numbers: list[int]
    expected_answer_points: list[str]


# Kept in sync 1:1 with SAMPLE_PAGES, like EVALUATION_CASES. Every
# expected_answer_points entry is a substring that appears verbatim
# (case-insensitive) in the corresponding SAMPLE_PAGES sentence, so a
# FakeLlm can be scripted to "answer correctly" by simply repeating
# that sentence - see tests/integration/test_live_answer_quality_evaluation.py
# for where a *real* LLM's answer is measured against these same points.
ANSWER_EVALUATION_CASES: list[AnswerEvaluationCase] = [
    AnswerEvaluationCase(
        "What is the adult dosage of Medicamentum X?", [1], ["500 mg", "twice daily"]
    ),
    AnswerEvaluationCase("How should Medicamentum X be dosed in children?", [2], ["weight-based"]),
    AnswerEvaluationCase(
        "What are the common side effects of Medicamentum X?", [3], ["nausea", "headache"]
    ),
    AnswerEvaluationCase("When is Medicamentum X contraindicated?", [4], ["renal impairment"]),
    AnswerEvaluationCase("How should Medicamentum X be stored?", [5], ["25 degrees"]),
    AnswerEvaluationCase(
        "What should a patient do if they miss a dose of Medicamentum X?",
        [6],
        ["as soon as remembered"],
    ),
    AnswerEvaluationCase(
        "Does Medicamentum X interact with other medications?", [7], ["anticoagulant"]
    ),
    AnswerEvaluationCase(
        "What are the symptoms of a Medicamentum X overdose?",
        [8],
        ["dizziness", "irregular heartbeat"],
    ),
]
