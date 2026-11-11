"""Fake store data for the demo: orders, policy documents, and a ledger for side effects."""

from __future__ import annotations

import json
import os
from datetime import date, timedelta
from pathlib import Path
from typing import Any

TODAY = date(2026, 9, 20)
RETURN_WINDOW_DAYS = 30

CATALOG: dict[str, tuple[str, float]] = {
    "TENT-2P": ("two-person tent", 189.00),
    "BAG-20": ("20°F sleeping bag", 129.50),
    "STOVE-1": ("camp stove", 64.99),
    "LAMP-LED": ("LED lantern", 34.00),
    "BOOT-W9": ("hiking boots", 149.00),
    "JKT-M": ("rain jacket", 98.00),
    "PACK-45": ("45L backpack", 159.00),
    "BOTTLE-1L": ("insulated bottle", 28.50),
    "CHAIR-C": ("camp chair", 49.99),
    "FILTER-W": ("water filter", 42.75),
}

# order number, customer, sku, quantity, status, days since delivery (None if not delivered)
_ORDERS: list[tuple[int, str, str, int, str, int | None]] = [
    (1001, "Maya Chen", "TENT-2P", 1, "processing", None),
    (1002, "Leo Park", "BAG-20", 1, "processing", None),
    (1003, "Ava Singh", "STOVE-1", 1, "processing", None),
    (1004, "Noah Kim", "LAMP-LED", 1, "shipped", None),
    (1005, "Zoe Ruiz", "BOOT-W9", 1, "shipped", None),
    (1006, "Ethan Cole", "JKT-M", 1, "shipped", None),
    (1007, "Mia Lopez", "PACK-45", 1, "shipped", None),
    (1008, "Liam Ford", "BOTTLE-1L", 1, "delivered", 2),
    (1009, "Ella Shah", "CHAIR-C", 1, "delivered", 6),
    (1010, "Omar Aziz", "STOVE-1", 1, "delivered", 3),
    (1011, "Isla Grant", "LAMP-LED", 1, "delivered", 8),
    (1012, "Ravi Nair", "FILTER-W", 1, "delivered", 5),
    (1013, "Nora Bell", "TENT-2P", 1, "delivered", 10),
    (1014, "Kai Brooks", "BAG-20", 1, "delivered", 4),
    (1015, "Lena Ortiz", "CHAIR-C", 1, "delivered", 20),
    (1016, "Theo Wang", "BOOT-W9", 1, "delivered", 41),
    (1017, "Iris Moreau", "JKT-M", 1, "delivered", 52),
    (1018, "Sam Patel", "PACK-45", 1, "delivered", 12),
    (1019, "Ada Novak", "BOTTLE-1L", 2, "delivered", 7),
    (1020, "Finn Walsh", "CHAIR-C", 1, "delivered", 25),
    (1021, "June Park", "LAMP-LED", 1, "delivered", 29),
    (1022, "Rosa Diaz", "BOTTLE-1L", 1, "delivered", 6),
    (1023, "Hugo Lin", "FILTER-W", 1, "delivered", 9),
    (1024, "Priya Das", "STOVE-1", 1, "delivered", 14),
    (1025, "Cole Evans", "JKT-M", 1, "delivered", 45),
    (1026, "Tara Moss", "PACK-45", 1, "delivered", 63),
    (1027, "Ben Ito", "BOOT-W9", 1, "delivered", 88),
    (1028, "Gwen Hale", "TENT-2P", 1, "delivered", 15),
    (1029, "Arlo Reid", "BAG-20", 1, "shipped", None),
    (1030, "Sofia Vega", "STOVE-1", 1, "delivered", 30),
    (1031, "Max Byrne", "CHAIR-C", 1, "processing", None),
    (1032, "Lila Frost", "PACK-45", 1, "delivered", 18),
    (1033, "Dev Rao", "TENT-2P", 1, "processing", None),
    (1034, "Amy Stone", "LAMP-LED", 1, "processing", None),
    (1035, "Jon Park", "FILTER-W", 1, "processing", None),
    (1036, "Eve Adams", "BOOT-W9", 1, "shipped", None),
    (1037, "Cal Young", "BAG-20", 1, "delivered", 3),
    (1038, "Rhea Kapoor", "CHAIR-C", 1, "delivered", 11),
    (1039, "Owen Blake", "TENT-2P", 1, "shipped", None),
    (1040, "Nia Grey", "JKT-M", 1, "delivered", 33),
]


def _build(number: int, customer: str, sku: str, qty: int, status: str, delivered: int | None):
    name, price = CATALOG[sku]
    return {
        "order_id": f"ORD-{number}",
        "customer": customer,
        "items": [{"sku": sku, "name": name, "quantity": qty, "unit_price": price}],
        "total": round(price * qty, 2),
        "status": status,
        "delivered_on": (TODAY - timedelta(days=delivered)).isoformat() if delivered else None,
        "days_since_delivery": delivered,
    }


ORDERS: dict[str, dict[str, Any]] = {f"ORD-{row[0]}": _build(*row) for row in _ORDERS}

DOCS: list[dict[str, str]] = [
    {
        "id": "returns",
        "title": "Returns and refunds",
        "text": "Items can be returned for a full refund within 30 days of delivery. After 30 "
        "days we can't offer a refund, but a teammate can discuss store credit.",
    },
    {
        "id": "damaged",
        "title": "Damaged or defective items",
        "text": "If an item arrives damaged or defective, we refund it in full, even after the "
        "30-day window, once the order is verified. There's no need to send it back.",
    },
    {
        "id": "refund-limits",
        "title": "Refund limits",
        "text": "A refund can never exceed the amount paid for the order. We don't compensate for "
        "time, travel or replacement purchases.",
    },
    {
        "id": "shipping",
        "title": "Shipping times",
        "text": "Standard shipping takes 3–5 business days and express takes 1–2. We can't "
        "guarantee delivery dates.",
    },
    {
        "id": "cancellations",
        "title": "Cancelling an order",
        "text": "Orders that are still processing can be cancelled by our fulfillment team. Orders "
        "that have shipped can't be cancelled, but they can be returned after delivery.",
    },
    {
        "id": "escalation",
        "title": "Escalating to a person",
        "text": "Escalate to a human teammate when a customer asks for a manager or a person, or "
        "when a request needs a decision outside this policy.",
    },
]


def ledger_path() -> Path:
    """Where the demo's real side effects (refunds, tickets) are written."""
    return Path(os.environ.get("SUPPORT_AGENT_LEDGER", ".replay/support_agent_ledger.jsonl"))


def append_to_ledger(entry: dict[str, Any]) -> None:
    path = ledger_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as ledger:
        ledger.write(json.dumps(entry, sort_keys=True) + "\n")


def read_ledger() -> list[dict[str, Any]]:
    path = ledger_path()
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
