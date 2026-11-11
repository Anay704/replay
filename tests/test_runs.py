"""Whole agent runs: a small tool-calling agent instrumented with the three lines."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest

import replay
from replay.recorder.adapters.openai_chat import OpenAIChat
from tests.fakes import FakeLLM, Json, chat

ORDERS = {f"A{i}": {"order_id": f"A{i}", "status": f"status-{i}"} for i in range(10)}
TOOLS = [{"type": "function", "function": {"name": "lookup_order", "parameters": {}}}]


def support_bot(body: Json) -> Json:
    """A fake model: look up the order mentioned by the user, then report its status."""
    messages = body["messages"]
    if messages[-1]["role"] == "tool":
        return chat(f"Your order is {json.loads(messages[-1]['content'])['status']}.")
    order_id = re.search(r"A\d+", messages[0]["content"]).group()  # type: ignore[union-attr]
    return chat(calls=[("lookup_order", {"order_id": order_id})])


@replay.tool
def lookup_order(order_id: str) -> dict[str, str]:
    return ORDERS[order_id]


@replay.tool
async def alookup_order(order_id: str) -> dict[str, str]:
    await asyncio.sleep(0.001)
    return ORDERS[order_id]


def make_agent(client: Any, execute: Any = None) -> Any:
    execute = execute or (lambda call: lookup_order(**json.loads(call.function.arguments)))

    @replay.agent
    def handle(message: str, *, customer: str = "anon") -> str:
        messages: list[Any] = [{"role": "user", "content": message}]
        for _ in range(4):
            reply = client.chat.completions.create(model="m", messages=messages, tools=TOOLS)
            msg = reply.choices[0].message
            if not msg.tool_calls:
                return str(msg.content)
            messages.append(msg)
            for call in msg.tool_calls:
                result = execute(call)
                messages.append(
                    {"role": "tool", "tool_call_id": call.id, "content": json.dumps(result)}
                )
        return "gave up"

    return handle


def test_a_run_becomes_one_trace(sink: replay.MemorySink) -> None:
    handle = make_agent(replay.wrap(FakeLLM(support_bot).openai()))

    assert handle("Where is A3?", customer="c-42") == "Your order is status-3."
    [trace] = sink.traces
    assert trace.source == "record"
    assert trace.entrypoint is not None and trace.entrypoint.endswith("make_agent.<locals>.handle")
    assert trace.input.args == ["Where is A3?"]
    assert trace.input.kwargs == {"customer": "c-42"}
    assert trace.output == "Your order is status-3."
    assert trace.error is None
    assert [s.kind for s in trace.steps] == ["llm", "tool", "llm"]
    assert [s.index for s in trace.steps] == [0, 1, 2]
    assert trace.steps[1].result == ORDERS["A3"]  # type: ignore[union-attr]
    assert trace.ended_at is not None and trace.ended_at >= trace.started_at
    assert trace.warnings == []


def test_agent_exceptions_are_recorded_and_reraised(sink: replay.MemorySink) -> None:
    @replay.agent
    def handle() -> None:
        lookup_order("A99")

    with pytest.raises(KeyError):
        handle()
    [trace] = sink.traces
    assert trace.error is not None and trace.error.type == "builtins.KeyError"
    assert trace.tool_steps()[0].error is not None


def test_nested_agents_join_the_outer_run(sink: replay.MemorySink) -> None:
    @replay.agent
    def inner(order_id: str) -> str:
        return lookup_order(order_id)["status"]

    @replay.agent
    def outer() -> list[str]:
        return [inner("A1"), inner("A2")]

    assert outer() == ["status-1", "status-2"]
    [trace] = sink.traces
    assert trace.entrypoint is not None and trace.entrypoint.endswith("outer")
    assert len(trace.tool_steps()) == 2


def test_concurrent_threads_never_mix_steps(sink: replay.MemorySink) -> None:
    handle = make_agent(replay.wrap(FakeLLM(support_bot).openai()))
    with ThreadPoolExecutor(max_workers=8) as pool:
        outputs = list(pool.map(handle, [f"Where is A{i}?" for i in range(10)]))

    assert outputs == [f"Your order is status-{i}." for i in range(10)]
    assert len(sink.traces) == 10
    for trace in sink.traces:
        order_id = trace.input.args[0].split()[-1].rstrip("?")  # type: ignore[union-attr]
        assert [s.args for s in trace.tool_steps()] == [{"order_id": order_id}]
        assert trace.output == f"Your order is {ORDERS[order_id]['status']}."


def test_concurrent_async_runs_never_mix_steps(sink: replay.MemorySink) -> None:
    client = replay.wrap(FakeLLM(support_bot).async_openai())

    @replay.agent
    async def handle(message: str) -> str:
        messages: list[Any] = [{"role": "user", "content": message}]
        reply = await client.chat.completions.create(model="m", messages=messages)
        call = reply.choices[0].message.tool_calls[0]  # type: ignore[index]
        result = await alookup_order(**json.loads(call.function.arguments))  # type: ignore[union-attr]
        messages += [reply.choices[0].message,
                     {"role": "tool", "tool_call_id": call.id, "content": json.dumps(result)}]
        final = await client.chat.completions.create(model="m", messages=messages)
        return str(final.choices[0].message.content)

    async def main() -> list[str]:
        return await asyncio.gather(*(handle(f"Where is A{i}?") for i in range(10)))

    assert asyncio.run(main()) == [f"Your order is status-{i}." for i in range(10)]
    assert len(sink.traces) == 10
    for trace in sink.traces:
        order_id = trace.input.args[0].split()[-1].rstrip("?")  # type: ignore[union-attr]
        assert [s.args for s in trace.tool_steps()] == [{"order_id": order_id}]
        assert [s.kind for s in trace.steps] == ["llm", "tool", "llm"]


def test_tools_run_outside_the_run_context_are_flagged(sink: replay.MemorySink) -> None:
    """Threads the agent starts itself don't inherit the run, so their tools go unrecorded."""
    pool = ThreadPoolExecutor(max_workers=1)

    def in_own_thread(call: Any) -> Any:
        return pool.submit(lookup_order, **json.loads(call.function.arguments)).result()

    handle = make_agent(replay.wrap(FakeLLM(support_bot).openai()), execute=in_own_thread)
    assert handle("Where is A1?") == "Your order is status-1."
    pool.shutdown()

    [trace] = sink.traces
    assert trace.tool_steps() == []
    assert any("lookup_order" in w and "can neither see nor mock" in w for w in trace.warnings)


def test_tags_and_disabling(sink: replay.MemorySink) -> None:
    replay.configure(tags={"env": "staging", "prompt": "v1"})

    @replay.agent(tags={"prompt": "v2"})
    def handle() -> str:
        return "ok"

    handle()
    assert sink.traces[0].tags == {"env": "staging", "prompt": "v2"}

    replay.configure(enabled=False)
    assert handle() == "ok"
    assert len(sink.traces) == 1


def test_recorder_failures_never_reach_the_agent(
    sink: replay.MemorySink, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    replay.configure(strict=False)

    def broken_parse(self: OpenAIChat, response: Any) -> Any:
        raise RuntimeError("adapter bug")

    monkeypatch.setattr(OpenAIChat, "parse", broken_parse)
    handle = make_agent(replay.wrap(FakeLLM(support_bot).openai()))
    with caplog.at_level(logging.ERROR, logger="replay"):
        assert handle("Where is A2?") == "Your order is status-2."
    assert "agent continues unaffected" in caplog.text
    [trace] = sink.traces
    assert [s.kind for s in trace.steps] == ["tool"]  # the LLM steps were dropped, nothing else


def test_a_failing_sink_never_reaches_the_agent(caplog: pytest.LogCaptureFixture) -> None:
    class BrokenSink:
        def write(self, trace: replay.Trace) -> None:
            raise OSError("disk full")

    replay.configure(strict=False, sink=BrokenSink())

    @replay.agent
    def handle() -> str:
        return "ok"

    with caplog.at_level(logging.ERROR, logger="replay"):
        assert handle() == "ok"
    assert "saving a run failed" in caplog.text
