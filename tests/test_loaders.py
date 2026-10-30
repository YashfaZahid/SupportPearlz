"""Phase 2 — ingestion loader tests."""

from __future__ import annotations

from pathlib import Path

from src.ingestion.loaders import load_file, load_knowledge_base
from src.ingestion.normalize import normalize_text

ROOT = Path(__file__).resolve().parents[1]
KB = ROOT / "data" / "knowledge_base"


def test_normalize_collapses_whitespace() -> None:
    assert normalize_text("a   b\n\n\n\nc") == "a b\n\nc"


def test_load_markdown_has_metadata_contract() -> None:
    docs = load_file(KB / "warranty_policy.md")
    assert docs
    meta = docs[0].metadata
    assert meta.get("source") == "warranty_policy.md"
    assert meta.get("doc_type") == "policy"
    assert meta.get("location")
    assert "24 months" in docs[0].page_content


def test_load_csv_keeps_sku_with_price() -> None:
    docs = load_file(KB / "pricing_guide.csv")
    assert len(docs) >= 5
    sediment = next(d for d in docs if "AP500-SED" in d.page_content)
    assert "2990" in sediment.page_content or "3200" in sediment.page_content
    assert sediment.metadata.get("sku") == "AP500-SED"
    assert sediment.metadata.get("doc_type") == "pricing"


def test_load_pdf_attaches_page_location() -> None:
    docs = load_file(KB / "product_manual_aquapearl500.pdf")
    assert docs
    assert docs[0].metadata.get("source") == "product_manual_aquapearl500.pdf"
    assert str(docs[0].metadata.get("location", "")).startswith("page:")


def test_load_docx_service_agreement() -> None:
    docs = load_file(KB / "service_maintenance_agreement.docx")
    assert docs
    assert "Claim ID" in docs[0].page_content or "claim" in docs[0].page_content.lower()
    assert docs[0].metadata.get("source") == "service_maintenance_agreement.docx"


def test_knowledge_base_skips_underscore_and_loads_rest() -> None:
    docs, report = load_knowledge_base(KB)
    assert report.files_found >= 10
    assert report.loaded >= 10
    assert all(d.metadata.get("source") and d.metadata.get("doc_type") for d in docs)
    names = {r.path.name for r in report.results}
    assert "_seams_and_gaps.md" not in names


def test_corrupt_file_is_skipped(tmp_path: Path) -> None:
    good = tmp_path / "warranty_policy.md"
    good.write_text("# Warranty\n\nPeriod is 24 months.\n", encoding="utf-8")
    bad = tmp_path / "broken.pdf"
    bad.write_bytes(b"%PDF-1.4 corrupted<<<not a real pdf")

    docs, report = load_knowledge_base(tmp_path)
    assert any(r.status == "loaded" and r.path.name == "warranty_policy.md" for r in report.results)
    assert any(r.status == "failed" and r.path.name == "broken.pdf" for r in report.results)
    assert docs
    assert all(d.metadata.get("source") and d.metadata.get("doc_type") for d in docs)
