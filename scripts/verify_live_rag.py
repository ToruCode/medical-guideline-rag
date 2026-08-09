"""Live HTTP verification of an already-running FastAPI app: index a PDF,
ask one or more questions, and optionally save the full request/response
detail (HTTP status, timing, citations, scores) as a local JSON report.

Unlike scripts/evaluate_answer_quality.py and friends, this makes real
HTTP requests against a server the caller already started (see the
"Demo" section in README.md, e.g. `make dev`), rather than calling
Application services in-process. That is the point: it exercises the
same multipart-upload and JSON-body encoding path a real client (curl,
PowerShell, a browser) would use. It uses httpx (already a project
dependency, used by FastAPI's own TestClient) rather than shelling out
to curl, since httpx encodes UTF-8 multipart filenames and JSON bodies
correctly regardless of the host OS's locale/codepage - the mojibake
and "error parsing the body" failures curl/PowerShell can hit with
non-ASCII content on Windows do not occur here.

Never commits its own --save-report output (data/eval/results/ is
gitignored, like every other scripts/*.py --save-report here); see
docs/examples/live_verification_sample.json for a fictional, committed
example of the output shape.

Usage (from the repo root, with the API already running):

    uv run python -m scripts.verify_live_rag \\
        --pdf data/raw/some_guideline.pdf \\
        --question "質問文その1" \\
        --question "質問文その2" \\
        --save-report
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
from app.core.config import get_settings

DEFAULT_BASE_URL = "http://127.0.0.1:8000"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify an already-running FastAPI app end to end over real HTTP: "
            "index a PDF, ask one or more questions, and optionally save a JSON report."
        )
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"Base URL of the already-running API (default: {DEFAULT_BASE_URL}).",
    )
    parser.add_argument("--pdf", required=True, help="Path to a PDF to index.")
    parser.add_argument(
        "--question",
        action="append",
        required=True,
        dest="questions",
        help="Question to ask (repeat --question for multiple questions).",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Retrieval depth (default: 5).")
    parser.add_argument(
        "--save-report",
        nargs="?",
        const="__default__",
        default=None,
        metavar="PATH",
        help=(
            "Save a JSON report locally (never commit this file). Defaults to "
            "data/eval/results/live_verification_<date>.json when given without a value."
        ),
    )
    return parser.parse_args()


def _resolve_report_path(save_report_arg: str) -> Path:
    if save_report_arg != "__default__":
        return Path(save_report_arg)
    today = datetime.now().date().isoformat()
    return Path("data/eval/results") / f"live_verification_{today}.json"


def _parse_json_body(response: httpx.Response) -> dict[str, Any]:
    try:
        body: dict[str, Any] = response.json()
    except ValueError:
        body = {"raw_text": response.text}
    return body


def _post_index(client: httpx.Client, pdf_path: Path) -> dict[str, Any]:
    start = time.perf_counter()
    with pdf_path.open("rb") as f:
        response = client.post(
            "/api/v1/documents/index",
            files={"file": (pdf_path.name, f, "application/pdf")},
        )
    elapsed = time.perf_counter() - start

    return {
        "endpoint": "POST /api/v1/documents/index",
        "http_status": response.status_code,
        "elapsed_seconds": round(elapsed, 3),
        "response": _parse_json_body(response),
    }


def _post_ask(client: httpx.Client, question: str, top_k: int) -> dict[str, Any]:
    start = time.perf_counter()
    response = client.post(
        "/api/v1/questions/ask",
        json={"question": question, "top_k": top_k},
    )
    elapsed = time.perf_counter() - start
    body = _parse_json_body(response)

    result: dict[str, Any] = {
        "endpoint": "POST /api/v1/questions/ask",
        "http_status": response.status_code,
        "elapsed_seconds": round(elapsed, 3),
        "question": question,
        "top_k": top_k,
    }

    if response.status_code != 200:
        result["response"] = body
        return result

    citations = body.get("citations", [])
    answer = body.get("answer", "")
    result.update(
        {
            "retrieved_chunk_count": len(citations),
            "answer_char_count": len(answer),
            "is_insufficient_evidence": body.get("is_insufficient_evidence"),
            "citations": [
                {
                    "page_number": c.get("page_number"),
                    "chunk_index": c.get("chunk_index"),
                    "score": c.get("score"),
                }
                for c in citations
            ],
            "cited_pages": [c.get("page_number") for c in citations],
            "answer": answer,
        }
    )
    return result


def main() -> None:
    args = _parse_args()
    pdf_path = Path(args.pdf)
    if not pdf_path.is_file():
        raise SystemExit(f"PDF not found: {pdf_path}")

    settings = get_settings()
    environment = {
        "embedding_provider": settings.embedding_provider,
        "embedding_model_name": settings.embedding_model_name,
        "llm_provider": settings.llm_provider,
        "llm_model_name": settings.llm_model_name,
        "vector_store_provider": settings.vector_store_provider,
    }

    try:
        with httpx.Client(base_url=args.base_url, timeout=120.0) as client:
            index_result = _post_index(client, pdf_path)
            print(
                f"{index_result['endpoint']}: {index_result['http_status']} "
                f"({index_result['elapsed_seconds']}s)"
            )

            ask_results = [_post_ask(client, question, args.top_k) for question in args.questions]
            for question, ask_result in zip(args.questions, ask_results, strict=True):
                print(
                    f"{ask_result['endpoint']}: {ask_result['http_status']} "
                    f"({ask_result['elapsed_seconds']}s) - {question!r}"
                )
    except httpx.RequestError as exc:
        raise SystemExit(
            f"Could not reach {args.base_url}: {exc}. Is the API running? "
            "(e.g. `make dev` / `uv run uvicorn app.main:app`)"
        ) from exc

    if args.save_report is None:
        return

    report = {
        "document_label": pdf_path.stem,
        "verified_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "environment": environment,
        "requests": [index_result, *ask_results],
    }
    report_path = _resolve_report_path(args.save_report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved verification report to {report_path}")
    print(
        "Reminder: this file may contain guideline-derived content and "
        "generated answer text - never commit it."
    )


if __name__ == "__main__":
    main()
