"""Scripted stand-ins for real models, served over a fake HTTP transport.

The agent talks to them through the real OpenAI SDK, so recording and replay run exactly the code
paths they would with a live model, for free and deterministically.

- `scripted-large` follows the store policy.
- `scripted-small` has four realistic flaws of a cheaper model:
  1. it trusts an amount the customer says they paid and refunds without looking up the order;
  2. it refunds whatever the customer asks for, even above the order total;
  3. it answers return-window questions from "memory" (60 days) instead of checking the policy;
  4. it refunds change-of-mind returns outside the 30-day window.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

import httpx2

from support_agent.data import RETURN_WINDOW_DAYS

MODELS = {"scripted-large": 0.04, "scripted-small": 0.015}
"""Model name -> simulated seconds per call."""

ORDER_RE = re.compile(r"\bORD-\d{4}\b")
AMOUNT_RE = re.compile(r"\$(\d+(?:\.\d{1,2})?)")
DAMAGE_WORDS = ("damaged", "broken", "defective", "cracked", "torn", "ripped", "dented", "leaks")
REFUND_WORDS = ("refund", "money back", "return", *DAMAGE_WORDS)
ESCALATE_WORDS = ("manager", "supervisor", "human", "real person", "complaint")
STATUS_WORDS = ("where", "status", "shipped", "arrive", "tracking", "delivered", "package")
POLICY_WORDS = ("how long", "how many days", "return window", "what happens", "do i need",
                "shipping", "express", "guarantee")

ASK_FOR_ORDER = "I can help with that! What's your order number? It looks like ORD-1234."
SORRY = (
    "I'm sorry, I couldn't complete that just now. I've noted your request and a teammate "
    "will follow up by email."
)


@dataclass
class Conversation:
    text: str
    calls: list[tuple[str, dict[str, Any], Any]] = field(default_factory=list)
    """Tool calls made so far, with their results, in order."""

    @classmethod
    def parse(cls, messages: list[dict[str, Any]]) -> Conversation:
        text = next(m["content"] for m in messages if m["role"] == "user")
        pending: dict[str, tuple[str, dict[str, Any]]] = {}
        calls = []
        for m in messages:
            if m["role"] == "assistant":
                for call in m.get("tool_calls") or []:
                    fn = call["function"]
                    pending[call["id"]] = (fn["name"], json.loads(fn["arguments"] or "{}"))
            elif m["role"] == "tool":
                name, args = pending.get(m["tool_call_id"], ("unknown", {}))
                try:
                    result = json.loads(m["content"])
                except json.JSONDecodeError:
                    result = m["content"]
                calls.append((name, args, result))
        return cls(text, calls)

    def called(self, name: str) -> bool:
        return any(call[0] == name for call in self.calls)

    def result(self, name: str) -> Any:
        return next((r for n, _, r in reversed(self.calls) if n == name), None)


Action = tuple[str, str, dict[str, Any]] | tuple[str, str]


def call(name: str, **args: Any) -> Action:
    return ("call", name, args)


def reply(text: str) -> Action:
    return ("reply", text)


def _failed(result: Any) -> bool:
    return isinstance(result, dict) and "error" in result


class Policy:
    """Decides the next step of a support conversation, like a model would."""

    def __init__(self, small: bool) -> None:
        self.small = small

    def decide(self, conv: Conversation) -> Action:
        if conv.calls and _failed(conv.calls[-1][2]):
            name, args, _ = conv.calls[-1]
            if name == "lookup_order":
                return reply(f"I couldn't find order {args.get('order_id')}. "
                             "Could you double-check the number?")
            return reply(SORRY)

        low = conv.text.lower()
        match = ORDER_RE.search(conv.text)
        order_id = match.group() if match else None
        if order_id is None and any(w in low for w in POLICY_WORDS):
            return self.policy(conv, low)
        if any(w in low for w in ESCALATE_WORDS):
            return self.escalate(conv, order_id)
        if "cancel" in low:
            return self.cancel(conv, order_id)
        if any(w in low for w in REFUND_WORDS):
            return self.refund(conv, order_id, low)
        if any(w in low for w in STATUS_WORDS):
            return self.status(conv, order_id)
        return reply("Happy to help! If this is about an order, please share your order number "
                     "(it looks like ORD-1234).")

    def policy(self, conv: Conversation, low: str) -> Action:
        if "return" in low or "how many days" in low:
            topic = "return window refund"
            if self.small:  # flaw 3: answers from memory
                return reply("You can return any item within 60 days of delivery for a full "
                             "refund.")
        elif any(w in low for w in ("shipping", "express", "guarantee", "deliver")):
            topic = "shipping times"
        else:
            topic = "damaged items refund"
        if not conv.called("search_docs"):
            return call("search_docs", query=topic)
        docs = conv.result("search_docs") or []
        return reply(docs[0]["text"] if docs else "Let me connect you with a teammate on that.")

    def status(self, conv: Conversation, order_id: str | None) -> Action:
        if order_id is None:
            return reply(ASK_FOR_ORDER)
        if not conv.called("lookup_order"):
            return call("lookup_order", order_id=order_id)
        order = conv.result("lookup_order")
        item = order["items"][0]["name"]
        if order["status"] == "processing":
            return reply(f"Order {order_id} ({item}) is being prepared and should ship within "
                         "1–2 business days.")
        if order["status"] == "shipped":
            return reply(f"Order {order_id} ({item}) has shipped and is on its way. Standard "
                         "shipping takes 3–5 business days.")
        return reply(f"Order {order_id} ({item}) was delivered on {order['delivered_on']}. If you "
                     "can't find it, let me know and I'll help.")

    def escalate(self, conv: Conversation, order_id: str | None) -> Action:
        if order_id and not conv.called("lookup_order"):
            return call("lookup_order", order_id=order_id)
        if not conv.called("escalate_to_human"):
            about = f" about {order_id}" if order_id else ""
            return call("escalate_to_human", order_id=order_id,
                        summary=f"Customer asked for a manager{about}.")
        ticket = conv.result("escalate_to_human")["ticket_id"]
        return reply(f"I'm sorry about the trouble. I've escalated this to a senior teammate "
                     f"(ticket {ticket}); they'll reach out within one business day.")

    def cancel(self, conv: Conversation, order_id: str | None) -> Action:
        if order_id is None:
            return reply(ASK_FOR_ORDER)
        if not conv.called("lookup_order"):
            return call("lookup_order", order_id=order_id)
        order = conv.result("lookup_order")
        if order["status"] != "processing":
            return reply(f"Order {order_id} has already shipped, so it can't be cancelled. Once "
                         f"it arrives you can return it within {RETURN_WINDOW_DAYS} days of "
                         "delivery for a refund.")
        if not conv.called("escalate_to_human"):
            return call("escalate_to_human", order_id=order_id,
                        summary=f"Cancellation request for {order_id} (still processing).")
        ticket = conv.result("escalate_to_human")["ticket_id"]
        return reply(f"I've asked our fulfillment team to cancel order {order_id} before it ships "
                     f"(ticket {ticket}). You'll get a confirmation email once it's done.")

    def refund(self, conv: Conversation, order_id: str | None, low: str) -> Action:
        if order_id is None:
            return reply(ASK_FOR_ORDER)
        damaged = any(w in low for w in DAMAGE_WORDS)
        amounts = AMOUNT_RE.findall(conv.text)
        requested = float(amounts[0]) if amounts else None
        reason = "damaged item" if damaged else "return within window"

        if refunded := conv.result("issue_refund"):
            return self._confirm_refund(order_id, refunded, conv.result("lookup_order"), requested)

        trusts_customer = self.small and damaged and requested and ("paid" in low or "my $" in low)
        if trusts_customer:  # flaw 1: skips verification
            return call("issue_refund", order_id=order_id, amount=requested, reason=reason)

        if not conv.called("lookup_order"):
            return call("lookup_order", order_id=order_id)
        order = conv.result("lookup_order")
        if not conv.called("search_docs"):
            topic = "damaged items refund" if damaged else "return window refund"
            return call("search_docs", query=topic)
        if order["status"] != "delivered":
            return reply(f"Order {order_id} hasn't been delivered yet (it's {order['status']}). "
                         "Once it arrives you can request a refund, or I can help you cancel it.")

        within_window = order["days_since_delivery"] <= RETURN_WINDOW_DAYS
        if not (damaged or within_window):
            if self.small:  # flaw 4: ignores the return window
                return call("issue_refund", order_id=order_id, amount=order["total"],
                            reason="customer return")
            return reply(f"Order {order_id} was delivered {order['days_since_delivery']} days ago, "
                         f"which is outside our {RETURN_WINDOW_DAYS}-day return window, so I can't "
                         "issue a refund. If you'd like, a teammate can discuss store credit.")

        amount = order["total"]
        if requested is not None and (self.small or requested <= order["total"]):
            amount = requested  # flaw 2 when small: no cap at the order total
        return call("issue_refund", order_id=order_id, amount=amount, reason=reason)

    @staticmethod
    def _confirm_refund(
        order_id: str, refund: dict[str, Any], order: dict[str, Any] | None, requested: float | None
    ) -> Action:
        amount = refund["amount"]
        when = "It should reach your original payment method in 3–5 business days."
        if order and requested is not None and requested > order["total"] >= amount:
            return reply(f"I've refunded the full order amount of ${amount:.2f} for order "
                         f"{order_id} (refund {refund['refund_id']}), which is the most we can "
                         f"refund for this order. {when}")
        return reply(f"I've issued a refund of ${amount:.2f} for order {order_id} "
                     f"(refund {refund['refund_id']}). {when}")


def _completion(model: str, body: dict[str, Any], action: Action, turn: int) -> dict[str, Any]:
    if action[0] == "call":
        _, name, args = action  # type: ignore[misc]
        message: dict[str, Any] = {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": f"call_{turn}", "type": "function",
                            "function": {"name": name, "arguments": json.dumps(args)}}],
        }
        finish = "tool_calls"
    else:
        message = {"role": "assistant", "content": action[1]}
        finish = "stop"
    prompt_tokens = len(json.dumps(body.get("messages")) + json.dumps(body.get("tools") or [])) // 4
    completion_tokens = max(1, len(json.dumps(message)) // 4)
    return {
        "id": f"scripted-{turn}", "object": "chat.completion", "created": 0, "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                  "total_tokens": prompt_tokens + completion_tokens},
    }


class ScriptedModels:
    """An OpenAI-compatible Chat Completions endpoint backed by the scripted policies."""

    def __init__(self, simulate_latency: bool = True) -> None:
        self.simulate_latency = simulate_latency

    def handle(self, request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        model = body.get("model")
        if model not in MODELS:
            return httpx2.Response(
                404, json={"error": {"message": f"Unknown model {model!r}", "type": "not_found"}}
            )
        if self.simulate_latency:
            time.sleep(MODELS[model])
        conv = Conversation.parse(body["messages"])
        action = Policy(small=model == "scripted-small").decide(conv)
        return httpx2.Response(200, json=_completion(model, body, action, len(conv.calls)))

    def transport(self) -> httpx2.MockTransport:
        return httpx2.MockTransport(self.handle)
