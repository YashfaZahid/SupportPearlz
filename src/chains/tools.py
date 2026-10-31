"""C7 mocked tools for order status / stock (function-calling style routing)."""

from __future__ import annotations

import re
from typing import Literal

from langchain_core.tools import tool

# Deterministic mock catalogue — not live commerce data
_MOCK_ORDERS = {
    "88213": {
        "status": "In transit",
        "eta": "3 working days",
        "sku": "AP500-UNIT",
        "city": "Lahore",
    },
    "10001": {
        "status": "Delivered",
        "eta": "Delivered 2026-01-12",
        "sku": "AERO-MINI",
        "city": "Karachi",
    },
}

_MOCK_STOCK = {
    "AP500-SED": "In stock (warehouse KHI)",
    "AP500-CAR": "In stock (warehouse LHE)",
    "AP500-ROM": "Low stock — 4 units",
    "AP500-UNIT": "In stock",
    "AERO-MINI": "In stock",
}


@tool
def get_order_status(order_id: str) -> str:
    """Look up a Pearlz order status by numeric order id (mocked)."""
    key = (order_id or "").strip()
    if not key:
        return "Missing order_id. Ask the customer for their order number."
    row = _MOCK_ORDERS.get(key)
    if not row:
        return f"No mock order found for id {key}. Escalate to support@pearlzhome.example."
    return (
        f"Order {key}: status={row['status']}; ETA/info={row['eta']}; "
        f"SKU={row['sku']}; city={row['city']}."
    )


@tool
def check_stock(sku: str) -> str:
    """Check warehouse stock for a product or filter SKU (mocked)."""
    key = (sku or "").strip().upper()
    if not key:
        return "Missing SKU. Ask which product or filter SKU they mean."
    if key not in _MOCK_STOCK:
        return f"No stock record for SKU {key} in the mock catalogue."
    return f"SKU {key}: {_MOCK_STOCK[key]}"


TOOLS = [get_order_status, check_stock]


def detect_tool_intent(question: str) -> Literal["order", "stock", "none"]:
    q = question.lower()
    if re.search(r"\border\b|\border\s*#?\s*\d+|\btracking\b|\bshipment status\b", q):
        return "order"
    if re.search(r"\bin stock\b|\bavailability\b|\bstock (of|for|on)\b|\bdo you have\b", q):
        return "stock"
    return "none"


def extract_order_id(question: str) -> str | None:
    m = re.search(r"\b(\d{4,6})\b", question)
    return m.group(1) if m else None


def extract_sku(question: str) -> str | None:
    m = re.search(r"\b(AP\d{3}-[A-Z]+|AERO-[A-Z]+|SVC-[A-Z]+)\b", question, re.I)
    return m.group(1).upper() if m else None
