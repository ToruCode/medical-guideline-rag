import json
from pathlib import Path

from scripts.evaluation_common import DatasetCase, load_dataset


def _write_dataset(tmp_path: Path, cases: list[dict]) -> Path:
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(
        json.dumps(
            {"document": {"source_path": "data/raw/example.pdf"}, "cases": cases},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return dataset_path


def test_load_dataset_defaults_answer_quality_fields_when_absent(tmp_path: Path) -> None:
    dataset_path = _write_dataset(
        tmp_path, [{"question": "q1", "granularity": "page", "expected": [1]}]
    )

    _, cases = load_dataset(dataset_path)

    assert cases == [
        DatasetCase(
            question="q1",
            granularity="page",
            expected_locations=[(1, None)],
            expected_answer_points=[],
            expected_insufficient_evidence=False,
        )
    ]


def test_load_dataset_parses_expected_answer_points(tmp_path: Path) -> None:
    dataset_path = _write_dataset(
        tmp_path,
        [
            {
                "question": "q1",
                "granularity": "page",
                "expected": [1],
                "expected_answer_points": ["500 mg", "twice daily"],
            }
        ],
    )

    _, cases = load_dataset(dataset_path)

    assert cases[0].expected_answer_points == ["500 mg", "twice daily"]


def test_load_dataset_parses_expected_insufficient_evidence(tmp_path: Path) -> None:
    dataset_path = _write_dataset(
        tmp_path,
        [
            {
                "question": "q1",
                "granularity": "page",
                "expected": [1],
                "expected_insufficient_evidence": True,
            }
        ],
    )

    _, cases = load_dataset(dataset_path)

    assert cases[0].expected_insufficient_evidence is True


def test_load_dataset_parses_category_and_difficulty(tmp_path: Path) -> None:
    dataset_path = _write_dataset(
        tmp_path,
        [
            {
                "question": "q1",
                "granularity": "page",
                "expected": [1],
                "category": "dosage",
                "difficulty": "easy",
            }
        ],
    )

    _, cases = load_dataset(dataset_path)

    assert cases[0].category == "dosage"
    assert cases[0].difficulty == "easy"


def test_load_dataset_defaults_category_and_difficulty_to_none_when_absent(
    tmp_path: Path,
) -> None:
    dataset_path = _write_dataset(
        tmp_path, [{"question": "q1", "granularity": "page", "expected": [1]}]
    )

    _, cases = load_dataset(dataset_path)

    assert cases[0].category is None
    assert cases[0].difficulty is None
