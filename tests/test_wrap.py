from __future__ import annotations

import asyncio
from typing import Any

import anthropic
import openai
import pytest

import replay
from replay.schema import LLMStep
from tests.fakes import FakeLLM, chat, chat_stream, error, message, responses

LOOKUP = ("lookup_order", {"order_id": "A1"})
USER = [{"role": "user", "content": "Where is order A1?"}]


def only_llm_step(sink: replay.MemorySink) -> LLMStep:
    [trace] = sink.traces
    [step] = trace.llm_steps()
    return step


def test_openai_chat_sync(sink: replay.MemorySink) -> None:
    fake = FakeLLM([chat(calls=[LOOKUP], cached_tokens=4, reasoning_tokens=2)])
    client = replay.wrap(fake.openai())

    @replay.agent
    def run() -> str:
        response = client.chat.completions.create(model="gpt-x", messages=USER, temperature=0)
        return response.choices[0].message.tool_calls[0].function.name  # type: ignore[union-attr]

    assert run() == "lookup_order"
    step = only_llm_step(sink)
    assert (step.provider, step.api, step.model) == ("openai", "chat.completions", "gpt-x")
    assert step.request == {"model": "gpt-x", "messages": USER, "temperature": 0}
    assert [(c.name, c.args, c.id) for c in step.tool_calls] == [
        ("lookup_order", {"order_id": "A1"}, "call_0")
    ]
    assert step.text is None
    assert step.stop_reason == "tool_calls"
    assert step.usage.model_dump() == {
        "input_tokens": 10, "output_tokens": 5, "cache_read_tokens": 4,
        "cache_write_tokens": 0, "reasoning_tokens": 2,
    }
    assert step.response is not None and step.response["id"] == "chatcmpl-1"
    assert step.error is None


def test_openai_chat_async(sink: replay.MemorySink) -> None:
    fake = FakeLLM([chat("It shipped yesterday.")])
    client = replay.wrap(fake.async_openai())

    @replay.agent
    async def run() -> str | None:
        response = await client.chat.completions.create(model="gpt-x", messages=USER)
        return response.choices[0].message.content

    assert asyncio.run(run()) == "It shipped yesterday."
    step = only_llm_step(sink)
    assert step.text == "It shipped yesterday."
    assert step.stop_reason == "stop"


def test_sdk_objects_passed_back_as_messages_are_serialized(sink: replay.MemorySink) -> None:
    fake = FakeLLM([chat(calls=[LOOKUP]), chat("Done.")])
    client = replay.wrap(fake.openai())

    @replay.agent
    def run() -> None:
        first = client.chat.completions.create(model="m", messages=USER)
        history: list[Any] = [*USER, first.choices[0].message]
        client.chat.completions.create(model="m", messages=history)

    run()
    second = sink.traces[0].llm_steps()[1]
    assistant = second.request["messages"][1]  # type: ignore[index]
    assert assistant["role"] == "assistant"  # type: ignore[index]
    assert assistant["tool_calls"][0]["function"]["name"] == "lookup_order"  # type: ignore[index]


@pytest.mark.parametrize("use_async", [False, True])
def test_openai_responses(sink: replay.MemorySink, use_async: bool) -> None:
    fake = FakeLLM([responses("Checking.", calls=[LOOKUP], reasoning_tokens=3)])

    @replay.agent
    async def run_async() -> str:
        client = replay.wrap(fake.async_openai())
        response = await client.responses.create(model="gpt-x", input="hi", instructions="Be kind")
        return response.output_text

    @replay.agent
    def run_sync() -> str:
        client = replay.wrap(fake.openai())
        return client.responses.create(model="gpt-x", input="hi", instructions="Be kind").output_text

    output = asyncio.run(run_async()) if use_async else run_sync()
    assert output == "Checking."
    step = only_llm_step(sink)
    assert (step.api, step.text, step.stop_reason) == ("responses", "Checking.", "completed")
    assert step.request["instructions"] == "Be kind"
    assert [(c.name, c.args, c.id) for c in step.tool_calls] == [
        ("lookup_order", {"order_id": "A1"}, "call_0")
    ]
    assert step.usage.reasoning_tokens == 3


@pytest.mark.parametrize("use_async", [False, True])
def test_anthropic_messages(sink: replay.MemorySink, use_async: bool) -> None:
    fake = FakeLLM([message("Let me check.", calls=[LOOKUP], cache_read=3, cache_write=2)])
    kwargs: dict[str, Any] = {
        "model": "claude-x", "max_tokens": 100, "system": "Be kind", "messages": USER
    }

    @replay.agent
    async def run_async() -> None:
        await replay.wrap(fake.async_anthropic()).messages.create(**kwargs)

    @replay.agent
    def run_sync() -> None:
        replay.wrap(fake.anthropic()).messages.create(**kwargs)

    asyncio.run(run_async()) if use_async else run_sync()
    step = only_llm_step(sink)
    assert (step.provider, step.api, step.model) == ("anthropic", "messages", "claude-x")
    assert step.request["system"] == "Be kind"
    assert step.text == "Let me check."
    assert step.stop_reason == "tool_use"
    assert [(c.name, c.args, c.id) for c in step.tool_calls] == [
        ("lookup_order", {"order_id": "A1"}, "toolu_0")
    ]
    # input_tokens is normalized to the full prompt size, cached parts included.
    assert step.usage.input_tokens == 15
    assert (step.usage.cache_read_tokens, step.usage.cache_write_tokens) == (3, 2)


def test_api_errors_are_recorded_and_reraised(sink: replay.MemorySink) -> None:
    client = replay.wrap(FakeLLM([error(400, "context too long")]).openai())

    @replay.agent
    def run() -> None:
        client.chat.completions.create(model="m", messages=USER)

    with pytest.raises(openai.BadRequestError):
        run()
    step = only_llm_step(sink)
    assert step.error is not None and step.error.type == "openai.BadRequestError"
    assert step.response is None
    assert sink.traces[0].error is not None


def test_streaming_passes_through_unrecorded(sink: replay.MemorySink) -> None:
    client = replay.wrap(FakeLLM([chat_stream("hi")]).openai())

    @replay.agent
    def run() -> str:
        stream = client.chat.completions.create(model="m", messages=USER, stream=True)
        return "".join(chunk.choices[0].delta.content or "" for chunk in stream)

    assert run() == "hi"
    [trace] = sink.traces
    assert trace.llm_steps() == []
    assert any("streaming" in w for w in trace.warnings)


def test_calls_outside_a_run_pass_through(sink: replay.MemorySink) -> None:
    client = replay.wrap(FakeLLM([chat("hi")]).openai())
    assert client.chat.completions.create(model="m", messages=USER).choices[0].message.content
    assert sink.traces == []


def test_wrap_is_in_place_and_idempotent(sink: replay.MemorySink) -> None:
    raw = FakeLLM([chat("one")]).openai()
    client = replay.wrap(replay.wrap(raw))
    assert client is raw
    assert isinstance(client, openai.OpenAI)

    @replay.agent
    def run() -> None:
        client.chat.completions.create(model="m", messages=USER)

    run()
    assert len(sink.traces[0].llm_steps()) == 1


def test_with_options_copies_are_wrapped(sink: replay.MemorySink) -> None:
    client = replay.wrap(FakeLLM([chat("one"), message("two")]).openai())
    claude = replay.wrap(FakeLLM([message("two")]).anthropic())

    @replay.agent
    def run() -> None:
        client.with_options(timeout=5).chat.completions.create(model="m", messages=USER)
        claude.with_options(max_retries=1).messages.create(
            model="c", max_tokens=5, messages=USER
        )

    run()
    assert [s.api for s in sink.traces[0].llm_steps()] == ["chat.completions", "messages"]


def test_unsupported_clients_are_rejected() -> None:
    with pytest.raises(TypeError, match="Anthropic and OpenAI"):
        replay.wrap(object())


def test_transport_kwargs_are_not_recorded(sink: replay.MemorySink) -> None:
    client = replay.wrap(FakeLLM([message("ok")]).anthropic())

    @replay.agent
    def run() -> None:
        client.messages.create(
            model="c", max_tokens=5, messages=USER, extra_headers={"x-secret": "s"}, timeout=3
        )

    run()
    assert set(only_llm_step(sink).request) == {"model", "max_tokens", "messages"}


def test_anthropic_client_type_is_preserved() -> None:
    client = replay.wrap(FakeLLM([]).async_anthropic())
    assert isinstance(client, anthropic.AsyncAnthropic)
