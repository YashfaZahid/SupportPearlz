"""Phase 6 — live RAG smoke tests (requires index + API key)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.chains.rag_chain import SupportPearlzAgent
from src.chains.schemas import Confidence
from src.config import clear_settings_cache, get_settings
from src.retrieval.vector_store import index_exists

ROOT = Path(__file__).resolve().parents[1]


def _ready() -> bool:
    clear_settings_cache()
    if not (ROOT / ".env").exists() and not os.getenv("OPENAI_API_KEY"):
        return False
    try:
        settings = get_settings()
    except Exception:
        return False
    return index_exists(settings=settings)


pytestmark = pytest.mark.skipif(not _ready(), reason="Requires .env + persisted index")


@pytest.fixture(scope="module")
def agent() -> SupportPearlzAgent:
    clear_settings_cache()
    return SupportPearlzAgent()


def test_factual_warranty_answer(agent: SupportPearlzAgent) -> None:
    agent.reset()
    result = agent.ask("How long is the warranty on the AquaPearl 500 Pro?")
    assert result.answered is True
    assert "24" in result.answer
    assert result.confidence in {Confidence.high, Confidence.partial}
    assert result.sources


def test_unanswerable_student_discount(agent: SupportPearlzAgent) -> None:
    agent.reset()
    result = agent.ask("Do you offer student discounts?")
    assert result.answered is False or "not" in result.answer.lower()
    assert result.confidence in {Confidence.none, Confidence.partial}


def test_follow_up_condensation(agent: SupportPearlzAgent) -> None:
    agent.reset()
    first = agent.ask("What's the return window?")
    assert first.answered is True
    second = agent.ask("Does that apply if I opened it?")
    assert second.answered is True
    # Should talk about opened returns / restocking rather than ignore history
    blob = second.answer.lower()
    assert any(w in blob for w in ("open", "restock", "return", "14", "packaging"))
