from __future__ import annotations

import asyncio
import dataclasses
from typing import Any

import pytest

import replay
from replay.schema import ToolStep


class OrderNotFound(Exception):
    pass


@dataclasses.dataclass
class Order:
    order_id: str
    total: float


@replay.tool
def lookup_order(order_id: str, include_items: bool = False) -> dict[str, Any]:
    if order_id == "missing":
        raise OrderNotFound(order_id)
    return {"order_id": order_id, "total": 49.99, "items": [] if include_items else None}


@replay.tool(name="search_docs")
def search(query: str, *tags: str, **filters: Any) -> list[str]:
    return [query, *tags, *filters]


@replay.tool
def get_order(order_id: str) -> Order:
    return Order(order_id, 10.0)


@replay.tool
def open_file(path: str) -> object:
    return object()


@replay.tool
async def async_lookup(order_id: str) -> dict[str, str]:
    await asyncio.sleep(0)
    return {"order_id": order_id}


class Tools:
    @replay.tool
    def refund(self, order_id: str, amount: float) -> str:
        return f"refunded {amount} on {order_id}"


def tool_steps(sink: replay.MemorySink) -> list[ToolStep]:
    [trace] = sink.traces
    return trace.tool_steps()


def test_positional_and_keyword_calls_record_the_same_args(sink: replay.MemorySink) -> None:
    @replay.agent
    def run() -> None:
        lookup_order("A1")
        lookup_order(order_id="A1")
        lookup_order("A1", include_items=True)

    run()
    args = [s.args for s in tool_steps(sink)]
    assert args == [
        {"order_id": "A1", "include_items": False},
        {"order_id": "A1", "include_items": False},
        {"order_id": "A1", "include_items": True},
    ]
    step = tool_steps(sink)[0]
    assert step.name == "lookup_order"
    assert step.result == {"order_id": "A1", "total": 49.99, "items": None}
    assert step.resolution == "executed"
    assert step.latency_ms >= 0


def test_variadic_arguments_and_custom_name(sink: replay.MemorySink) -> None:
    @replay.agent
    def run() -> None:
        search("refunds", "policy", lang="en")

    run()
    [step] = tool_steps(sink)
    assert step.name == "search_docs"
    assert step.args == {"query": "refunds", "tags": ["policy"], "lang": "en"}


def test_methods_drop_self(sink: replay.MemorySink) -> None:
    @replay.agent
    def run() -> str:
        return Tools().refund("A1", amount=5.0)

    assert run() == "refunded 5.0 on A1"
    assert tool_steps(sink)[0].args == {"order_id": "A1", "amount": 5.0}


def test_exceptions_are_recorded_and_reraised(sink: replay.MemorySink) -> None:
    @replay.agent
    def run() -> str:
        try:
            lookup_order("missing")
        except OrderNotFound:
            return "not found"
        return "found"

    assert run() == "not found"
    [step] = tool_steps(sink)
    assert step.error is not None
    assert step.error.type == "tests.test_tools.OrderNotFound"
    assert step.error.args == ["missing"]
    assert step.result is None


def test_typed_and_unserializable_results(sink: replay.MemorySink) -> None:
    @replay.agent
    def run() -> None:
        get_order("A1")
        open_file("/tmp/x")

    run()
    typed, opaque = tool_steps(sink)
    assert typed.result == {"order_id": "A1", "total": 10.0}
    assert typed.result_type == "dataclass:tests.test_tools.Order"
    assert opaque.replayable is False
    assert opaque.result_type == "repr"
    assert any("open_file" in w for w in sink.traces[0].warnings)


def test_async_tool(sink: replay.MemorySink) -> None:
    @replay.agent
    async def run() -> None:
        await asyncio.gather(async_lookup("A1"), async_lookup("A2"))

    asyncio.run(run())
    steps = tool_steps(sink)
    assert sorted(s.args["order_id"] for s in steps) == ["A1", "A2"]
    assert [s.index for s in steps] == [0, 1]


def test_tools_outside_a_run_just_run(sink: replay.MemorySink) -> None:
    assert lookup_order("A1")["order_id"] == "A1"
    assert sink.traces == []


def test_bad_arguments_raise_the_normal_type_error(sink: replay.MemorySink) -> None:
    @replay.agent
    def run() -> None:
        lookup_order()  # type: ignore[call-arg]

    with pytest.raises(TypeError):
        run()
    assert sink.traces[0].tool_steps() == []
