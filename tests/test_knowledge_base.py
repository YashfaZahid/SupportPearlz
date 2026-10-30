"""Phase 1 — knowledge base structure checks (no LLM)."""

from __future__ import annotations

from pathlib import Path

KB = Path(__file__).resolve().parents[1] / "data" / "knowledge_base"

REQUIRED_SUFFIXES = {".md", ".pdf", ".csv", ".docx"}


def _customer_docs() -> list[Path]:
    return [
        p
        for p in KB.iterdir()
        if p.is_file() and not p.name.startswith("_") and not p.name.startswith(".")
    ]


def test_at_least_ten_documents() -> None:
    docs = _customer_docs()
    assert len(docs) >= 10, f"expected >=10 docs, found {len(docs)}: {[d.name for d in docs]}"


def test_at_least_three_formats() -> None:
    suffixes = {p.suffix.lower() for p in _customer_docs()}
    assert len(suffixes) >= 3, suffixes
    assert ".pdf" in suffixes
    assert ".csv" in suffixes
    assert ".md" in suffixes or ".docx" in suffixes


def test_seams_and_gaps_file_exists() -> None:
    path = KB / "_seams_and_gaps.md"
    text = path.read_text(encoding="utf-8").lower()
    assert "student" in text
    assert "nsf" in text
    assert "instalment" in text or "installment" in text


def test_injection_canary_present() -> None:
    handbook = (KB / "customer_service_handbook.md").read_text(encoding="utf-8")
    assert "ignore all prior instructions" in handbook.lower()


def test_warranty_period_consistent_in_md() -> None:
    warranty = (KB / "warranty_policy.md").read_text(encoding="utf-8")
    assert "24 months" in warranty
    faq = (KB / "faq.md").read_text(encoding="utf-8")
    assert "24 months" in faq
