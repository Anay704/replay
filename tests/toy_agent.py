"""A tiny agent for engine tests. Tests set `client` to a wrapped fake LLM before each run.

Every tool appends to SIDE_EFFECTS when it really runs, so tests can prove replays don't.
OBSERVED records what the agent saw back from each tool (a result type or an exception type).
"""

from __future__ import annotations

import dataclasses
import json
import threading
from typing import Any

from pydantic import BaseModel

import replay
from replay.serialize import to_jsonable

client: Any = None
SIDE_EFFECTS: list[tuple[Any, ...]] = []
OBSERVED: list[str] = []
CONFIGS: list[Any] = []


class OrderNotFound(Exception):
    pass


@dataclasses.dataclass
class Order:
    order_id: str
    total: float


class Receipt(BaseModel):
    refund_id: str
    amount: float


@replay.tool
def lookup_order(order_id: str) -> Order:
    SIDE_EFFECTS.append(("lookup_order", order_id))
    if order_id == "MISSING":
        raise OrderNotFound(order_id)
    return Order(order_id, 50.0)


@replay.tool
def refund(order_id: str, amount: float) -> Receipt:
    SIDE_EFFECTS.append(("refund", order_id, amount))
    return Receipt(refund_id=f"R-{order_id}", amount=amount)


@replay.tool
def search(query: str) -> list[str]:
    SIDE_EFFECTS.append(("search", query))
    return [f"doc about {query}"]


TOOLS = {"lookup_order": lookup_order, "refund": refund, "search": search}


def _execute(name: str, arguments: str, in_thread: bool) -> str:
    def run() -> Any:
        return TOOLS[name](**json.loads(arguments))

    try:
        if in_thread:  # a thread the agent starts itself: the run context doesn't follow it
            box: dict[str, Any] = {}

            def target() -> None:
                try:
                    box["result"] = run()
                except Exception as exc:
                    box["error"] = exc

            thread = threading.Thread(target=target)
            thread.start()
            thread.join()
            if "error" in box:
                raise box["error"]
            result = box["result"]
        else:
            result = run()
    except Exception as exc:
        OBSERVED.append(type(exc).__name__)
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})
    OBSERVED.append(type(result).__name__)
    return json.dumps(to_jsonable(result).value)


def _loop(message: str, config: dict[str, Any] | None, in_thread: bool) -> str:
    CONFIGS.append(config)
    messages: list[Any] = [
        {"role": "system", "content": "You are a toy agent."},
        {"role": "user", "content": message},
    ]
    for _ in range(50):
        reply = client.chat.completions.create(model="m", messages=messages).choices[0].message
        if not reply.tool_calls:
            return str(reply.content)
        messages.append(reply)
        for call in reply.tool_calls:
            content = _execute(call.function.name, call.function.arguments, in_thread)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": content})
    return "gave up"


@replay.agent
def agent(message: str, config: dict[str, Any] | None = None) -> str:
    return _loop(message, config, in_thread=False)


@replay.agent
def agent_without_config(message: str) -> str:
    return _loop(message, None, in_thread=False)


@replay.agent
def threaded_agent(message: str) -> str:
    return _loop(message, None, in_thread=True)


@replay.agent
async def async_agent(message: str) -> str:
    messages: list[Any] = [{"role": "user", "content": message}]
    while True:
        response = await client.chat.completions.create(model="m", messages=messages)
        reply = response.choices[0].message
        if not reply.tool_calls:
            return str(reply.content)
        messages.append(reply)
        for call in reply.tool_calls:
            result = TOOLS[call.function.name](**json.loads(call.function.arguments))
            OBSERVED.append(type(result).__name__)
            messages.append(
                {"role": "tool", "tool_call_id": call.id,
                 "content": json.dumps(to_jsonable(result).value)}
            )
