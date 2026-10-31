"""Bonus feature unit tests: tools routing and access metadata."""

from __future__ import annotations

from src.chains.tools import detect_tool_intent, extract_order_id, extract_sku, get_order_status
from src.ingestion.loaders import infer_access_level


def test_infer_access_level() -> None:
    assert infer_access_level("customer_service_handbook.md") == "internal"
    assert infer_access_level("warranty_policy.md") == "public"


def test_detect_order_intent() -> None:
    assert detect_tool_intent("Where is order 88213?") == "order"
    assert detect_tool_intent("Is AP500-SED in stock?") == "stock"
    assert detect_tool_intent("How long is the warranty?") == "none"


def test_extractors() -> None:
    assert extract_order_id("status for 88213 please") == "88213"
    assert extract_sku("stock of AP500-SED") == "AP500-SED"


def test_mock_order_tool() -> None:
    text = get_order_status.invoke({"order_id": "88213"})
    assert "In transit" in text
