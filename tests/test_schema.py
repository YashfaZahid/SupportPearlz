"""Phase 6 — schema and citation unit tests."""

from __future__ import annotations

from langchain_core.documents import Document

from src.chains.citations import validate_and_dedupe_citations
from src.chains.schemas import Confidence, GroundedResponse, SourceCitation


def test_refusal_factory() -> None:
    r = GroundedResponse.refusal("not in docs")
    assert r.answered is False
    assert r.confidence == Confidence.none
    assert r.sources == []


def test_schema_roundtrip() -> None:
    raw = {
        "answer": "Warranty is 24 months.",
        "sources": [{"label": "S1", "source": "warranty_policy.md", "location": "section:1"}],
        "confidence": "high",
        "answered": True,
    }
    obj = GroundedResponse.model_validate(raw)
    assert obj.answered is True
    assert obj.sources[0].label == "S1"


def test_citation_validator_drops_unknown_label() -> None:
    supplied = {
        "S1": Document(
            page_content="Warranty is 24 months from invoice date.",
            metadata={"source": "warranty_policy.md", "location": "section:1", "doc_type": "policy"},
        )
    }
    response = GroundedResponse(
        answer="24 months",
        sources=[
            SourceCitation(label="S1", source="warranty_policy.md", location=""),
            SourceCitation(label="S5", source="fake.md", location=""),
        ],
        confidence=Confidence.high,
        answered=True,
    )
    cleaned = validate_and_dedupe_citations(response, supplied)
    assert len(cleaned.sources) == 1
    assert cleaned.sources[0].label == "S1"
    assert "invalid_citations_dropped" in cleaned.flags


def test_citation_dedupe_same_source_location() -> None:
    supplied = {
        "S1": Document(
            page_content="a",
            metadata={"source": "warranty_policy.md", "location": "section:1", "doc_type": "policy"},
        ),
        "S2": Document(
            page_content="b",
            metadata={"source": "warranty_policy.md", "location": "section:1", "doc_type": "policy"},
        ),
    }
    response = GroundedResponse(
        answer="x",
        sources=[
            SourceCitation(label="S1", source="warranty_policy.md", location="section:1"),
            SourceCitation(label="S2", source="warranty_policy.md", location="section:1"),
        ],
        confidence=Confidence.high,
        answered=True,
    )
    cleaned = validate_and_dedupe_citations(response, supplied)
    assert len(cleaned.sources) == 1
