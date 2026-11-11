"""The agent's tools. lookup_order and search_docs only read; the other two change the world."""

from __future__ import annotations

import re
from typing import Any

import replay
from support_agent.data import DOCS, ORDERS, append_to_ledger


class OrderNotFound(Exception):
    pass


@replay.tool
def lookup_order(order_id: str) -> dict[str, Any]:
    order = ORDERS.get(order_id.strip().upper())
    if order is None:
        raise OrderNotFound(f"No order {order_id}")
    return order


@replay.tool
def search_docs(query: str) -> list[dict[str, str]]:
    words = set(re.findall(r"[a-z0-9]+", query.lower()))

    def score(doc: dict[str, str]) -> int:
        return len(words & set(re.findall(r"[a-z0-9]+", (doc["title"] + " " + doc["text"]).lower())))

    ranked = sorted(DOCS, key=score, reverse=True)
    return [doc for doc in ranked[:2] if score(doc) > 0]


@replay.tool
def issue_refund(order_id: str, amount: float, reason: str) -> dict[str, Any]:
    """Issues a real refund. The payments backend trusts the agent: it does not check the amount."""
    if order_id not in ORDERS:
        raise OrderNotFound(f"No order {order_id}")
    refund = {
        "refund_id": f"RF-{order_id[-4:]}-{round(amount * 100)}",
        "order_id": order_id,
        "amount": round(amount, 2),
        "status": "issued",
    }
    append_to_ledger({"type": "refund", "reason": reason, **refund})
    return refund


@replay.tool
def escalate_to_human(summary: str, order_id: str | None = None) -> dict[str, Any]:
    ticket = {
        "ticket_id": f"T-{(order_id or 'none')[-4:]}-{len(summary)}",
        "status": "open",
        "response_time": "1 business day",
    }
    append_to_ledger({"type": "ticket", "order_id": order_id, "summary": summary, **ticket})
    return ticket


TOOLS = {f.__name__: f for f in (lookup_order, search_docs, issue_refund, escalate_to_human)}


def _function(name: str, description: str, properties: dict[str, Any], required: list[str]):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


TOOL_SCHEMAS = [
    _function(
        "lookup_order",
        "Get an order's items, total, status and delivery date.",
        {"order_id": {"type": "string", "description": "Order number, like ORD-1234"}},
        ["order_id"],
    ),
    _function(
        "search_docs",
        "Search the store's policy documents.",
        {"query": {"type": "string"}},
        ["query"],
    ),
    _function(
        "issue_refund",
        "Refund money to the customer's original payment method.",
        {
            "order_id": {"type": "string"},
            "amount": {"type": "number", "description": "Amount in USD"},
            "reason": {"type": "string"},
        },
        ["order_id", "amount", "reason"],
    ),
    _function(
        "escalate_to_human",
        "Open a ticket for a human teammate.",
        {"summary": {"type": "string"}, "order_id": {"type": "string"}},
        ["summary"],
    ),
]
