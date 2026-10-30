"""Run the automated evaluation harness and write JSON + markdown results."""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import typer

from src.chains.rag_chain import SupportPearlzAgent
from src.config import PROJECT_ROOT, get_settings
from src.utils.cache_setup import setup_langchain_cache
from src.utils.logging_setup import setup_logging

app = typer.Typer(add_completion=False)
logger = logging.getLogger(__name__)

EVAL_DIR = PROJECT_ROOT / "evaluation"
DEFAULT_QUESTIONS = EVAL_DIR / "test_questions.json"
RESULTS_DIR = EVAL_DIR / "results"


def _score_case(case: dict[str, Any], answer: str, answered: bool, sources: list[dict]) -> dict[str, Any]:
    """Heuristic auto-score (manual review still expected for the report)."""
    text = (answer or "").lower()
    must_refuse = bool(case.get("must_refuse"))
    key = case.get("expected_key_fact")
    expected_source = case.get("ground_truth_source")

    refusal_ok = (not answered) or any(
        p in text
        for p in (
            "could not find",
            "does not cover",
            "not in the",
            "documentation does not",
            "cannot confirm",
            "out of scope",
            "only answer",
            "human agent",
            "contact support",
        )
    )

    if must_refuse:
        correctness = 1.0 if refusal_ok and not (
            key and key.lower() in text and answered
        ) else 0.0
        # Adversarial false-premise case V02 is not must_refuse
        groundedness = 1.0 if refusal_ok else 0.0
        citation_accuracy = 1.0  # refusals should not fabricate citations
        if sources and must_refuse:
            citation_accuracy = 0.5
        verdict = "pass" if correctness >= 1.0 else "fail"
        fault = None if verdict == "pass" else "prompt_or_gate"
        return {
            "correctness": correctness,
            "groundedness": groundedness,
            "citation_accuracy": citation_accuracy,
            "verdict": verdict,
            "fault_stage": fault,
        }

    fact_ok = True if not key else key.lower() in text
    source_names = {s.get("source") for s in sources}
    cite_ok = True
    if expected_source:
        cite_ok = expected_source in source_names or any(
            expected_source.split(".")[0] in (s or "") for s in source_names
        )

    # Ambiguous: accept listing multiple options / clarifying
    if case.get("category") == "ambiguous":
        fact_ok = answered and (
            "?" in answer
            or "depend" in text
            or "which" in text
            or "sku" in text
            or "14" in text
            or "filter" in text
        )

    correctness = 1.0 if (answered and fact_ok) else (0.5 if fact_ok else 0.0)
    groundedness = 1.0 if (sources and answered) or (not answered and refusal_ok) else 0.5
    citation_accuracy = 1.0 if cite_ok else 0.0

    if correctness >= 1.0 and citation_accuracy >= 1.0:
        verdict = "pass"
        fault = None
    elif correctness >= 0.5:
        verdict = "partial"
        fault = "citation" if not cite_ok else "generation"
    else:
        verdict = "fail"
        fault = "retrieval" if not sources else "generation"

    return {
        "correctness": correctness,
        "groundedness": groundedness,
        "citation_accuracy": citation_accuracy,
        "verdict": verdict,
        "fault_stage": fault,
    }


@app.command()
def main(
    questions_path: Path = typer.Option(DEFAULT_QUESTIONS, "--questions"),
    limit: int | None = typer.Option(None, "--limit", help="Run only first N cases"),
) -> None:
    settings = get_settings()
    setup_logging(level=settings.log_level, log_file=settings.log_file)
    setup_langchain_cache(settings)

    cases: list[dict[str, Any]] = json.loads(questions_path.read_text(encoding="utf-8"))
    if limit:
        cases = cases[:limit]

    agent = SupportPearlzAgent(settings=settings)
    sessions: dict[str, SupportPearlzAgent] = {}
    rows: list[dict[str, Any]] = []

    for case in cases:
        cid = case["id"]
        session_key = case.get("session")
        if session_key:
            if session_key not in sessions:
                sessions[session_key] = SupportPearlzAgent(settings=settings)
            runner = sessions[session_key]
        else:
            agent.reset()
            runner = agent

        started = time.perf_counter()
        try:
            result = runner.ask(case["question"])
            error = None
        except Exception as exc:  # noqa: BLE001 — eval must continue
            logger.exception("Eval case %s failed", cid)
            result = None
            error = str(exc)

        latency_ms = int((time.perf_counter() - started) * 1000)
        if result is None:
            scores = {
                "correctness": 0.0,
                "groundedness": 0.0,
                "citation_accuracy": 0.0,
                "verdict": "fail",
                "fault_stage": "api_or_runtime",
            }
            answer = ""
            answered = False
            sources: list[dict] = []
            confidence = "none"
            flags: list[str] = ["error"]
        else:
            answer = result.answer
            answered = result.answered
            sources = [s.model_dump() for s in result.sources]
            confidence = result.confidence.value
            flags = list(result.flags)
            scores = _score_case(case, answer, answered, sources)

        row = {
            "id": cid,
            "category": case.get("category"),
            "question": case["question"],
            "answer": answer,
            "answered": answered,
            "confidence": confidence,
            "sources": sources,
            "flags": flags,
            "latency_ms": latency_ms,
            "error": error,
            **scores,
            "expected_behaviour": case.get("expected_behaviour"),
            "ground_truth_source": case.get("ground_truth_source"),
        }
        rows.append(row)
        typer.echo(
            f"{cid} [{case.get('category')}] {scores['verdict']} "
            f"corr={scores['correctness']} cite={scores['citation_accuracy']}"
        )

    n = max(len(rows), 1)
    unanswerable = [r for r in rows if r["category"] == "unanswerable"]
    refusal_rate = (
        sum(1 for r in unanswerable if r["verdict"] == "pass") / max(len(unanswerable), 1)
    )
    # Hallucination proxy: unanswerable answered with a confident invented fact
    hallucination_count = sum(
        1
        for r in rows
        if r.get("must_refuse") is True  # not on row; use category
        or (
            r["category"] in {"unanswerable", "adversarial"}
            and r["answered"]
            and r["verdict"] == "fail"
        )
    )
    # Fix hallucination using category
    hallucination_count = sum(
        1
        for r in rows
        if r["category"] in {"unanswerable"}
        and r["answered"]
        and r["verdict"] == "fail"
    )

    aggregate = {
        "n": len(rows),
        "mean_correctness": sum(r["correctness"] for r in rows) / n,
        "mean_groundedness": sum(r["groundedness"] for r in rows) / n,
        "mean_citation_accuracy": sum(r["citation_accuracy"] for r in rows) / n,
        "pass_rate": sum(1 for r in rows if r["verdict"] == "pass") / n,
        "refusal_rate_unanswerable": refusal_rate,
        "hallucination_count_proxy": hallucination_count,
        "by_category": {},
    }
    for cat in sorted({r["category"] for r in rows}):
        subset = [r for r in rows if r["category"] == cat]
        aggregate["by_category"][cat] = {
            "n": len(subset),
            "pass_rate": sum(1 for r in subset if r["verdict"] == "pass") / max(len(subset), 1),
        }

    failures = [r for r in rows if r["verdict"] != "pass"]
    failure_analysis = [
        {
            "id": r["id"],
            "verdict": r["verdict"],
            "fault_stage": r.get("fault_stage"),
            "note": r.get("expected_behaviour"),
        }
        for r in failures
    ]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_json = RESULTS_DIR / f"eval_{stamp}.json"
    out_md = RESULTS_DIR / f"eval_{stamp}.md"
    latest_json = RESULTS_DIR / "latest.json"
    latest_md = RESULTS_DIR / "latest.md"

    payload = {
        "generated_at": stamp,
        "aggregate": aggregate,
        "failure_analysis": failure_analysis,
        "cases": rows,
        "scoring_note": (
            "Auto-scored heuristics for CI/smoke. Manual rubric scoring still required "
            "for the formal technical report."
        ),
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    latest_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md_lines = [
        f"# SupportPearlz evaluation ({stamp})",
        "",
        "## Aggregate",
        f"- cases: {aggregate['n']}",
        f"- mean correctness: {aggregate['mean_correctness']:.2f}",
        f"- mean groundedness: {aggregate['mean_groundedness']:.2f}",
        f"- mean citation accuracy: {aggregate['mean_citation_accuracy']:.2f}",
        f"- pass rate: {aggregate['pass_rate']:.2f}",
        f"- refusal rate (unanswerable): {aggregate['refusal_rate_unanswerable']:.2f}",
        f"- hallucination proxy count: {aggregate['hallucination_count_proxy']}",
        "",
        "## Results table",
        "| id | category | verdict | corr | ground | cite | fault |",
        "|----|----------|---------|------|--------|------|-------|",
    ]
    for r in rows:
        md_lines.append(
            f"| {r['id']} | {r['category']} | {r['verdict']} | {r['correctness']} | "
            f"{r['groundedness']} | {r['citation_accuracy']} | {r.get('fault_stage') or '-'} |"
        )
    md_lines.extend(["", "## Failure analysis"])
    if not failure_analysis:
        md_lines.append("_No failures under auto-score._")
    else:
        for f in failure_analysis:
            md_lines.append(
                f"- **{f['id']}** ({f['verdict']}): stage=`{f['fault_stage']}` — {f['note']}"
            )

    md_text = "\n".join(md_lines) + "\n"
    out_md.write_text(md_text, encoding="utf-8")
    latest_md.write_text(md_text, encoding="utf-8")

    typer.echo(f"Wrote {out_json}")
    typer.echo(f"Wrote {out_md}")
    typer.echo(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    app()
