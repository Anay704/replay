"""Fake provider endpoints for offline tests.

Responses are served through the real SDKs via an httpx2 MockTransport, so the SDKs parse real
provider-format JSON and the recorder sees exactly the objects it would see in production.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from typing import Any

import anthropic
import httpx2
import openai

Json = dict[str, Any]
Responder = Callable[[Json], Json | httpx2.Response]


class FakeLLM:
    """Serves queued responses in order, or computes each response from the request body."""

    def __init__(self, responses: Sequence[Json | httpx2.Response] | Responder) -> None:
        self._queue = None if callable(responses) else list(responses)
        self._respond = responses if callable(responses) else None
        self.requests: list[Json] = []

    def handle(self, request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        self.requests.append(body)
        item = self._respond(body) if self._respond else self._queue.pop(0)  # type: ignore[union-attr]
        return item if isinstance(item, httpx2.Response) else httpx2.Response(200, json=item)

    def openai(self) -> openai.OpenAI:
        return openai.OpenAI(
            api_key="test",
            base_url="https://fake.test/v1",
            max_retries=0,
            http_client=httpx2.Client(transport=httpx2.MockTransport(self.handle)),
        )

    def async_openai(self) -> openai.AsyncOpenAI:
        return openai.AsyncOpenAI(
            api_key="test",
            base_url="https://fake.test/v1",
            max_retries=0,
            http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(self.handle)),
        )

    def anthropic(self) -> anthropic.Anthropic:
        return anthropic.Anthropic(
            api_key="test",
            base_url="https://fake.test",
            max_retries=0,
            http_client=httpx2.Client(transport=httpx2.MockTransport(self.handle)),
        )

    def async_anthropic(self) -> anthropic.AsyncAnthropic:
        return anthropic.AsyncAnthropic(
            api_key="test",
            base_url="https://fake.test",
            max_retries=0,
            http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(self.handle)),
        )


Calls = Sequence[tuple[str, Json]]


def chat(
    content: str | None = None,
    calls: Calls = (),
    *,
    model: str = "test-model",
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
    cached_tokens: int = 0,
    reasoning_tokens: int = 0,
) -> Json:
    """An OpenAI Chat Completions response."""
    tool_calls = [
        {"id": f"call_{i}", "type": "function",
         "function": {"name": name, "arguments": json.dumps(args)}}
        for i, (name, args) in enumerate(calls)
    ]
    message: Json = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "id": "chatcmpl-1", "object": "chat.completion", "created": 1, "model": model,
        "choices": [{"index": 0, "message": message,
                     "finish_reason": "tool_calls" if tool_calls else "stop"}],
        "usage": {
            "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "prompt_tokens_details": {"cached_tokens": cached_tokens},
            "completion_tokens_details": {"reasoning_tokens": reasoning_tokens},
        },
    }


def chat_stream(text: str) -> httpx2.Response:
    """A streamed Chat Completions response (server-sent events)."""
    chunk = {"id": "c1", "object": "chat.completion.chunk", "created": 1, "model": "test-model",
             "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}]}
    body = f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n"
    return httpx2.Response(200, headers={"content-type": "text/event-stream"}, content=body)


def responses(
    text: str | None = None,
    calls: Calls = (),
    *,
    model: str = "test-model",
    input_tokens: int = 10,
    output_tokens: int = 5,
    reasoning_tokens: int = 0,
) -> Json:
    """An OpenAI Responses API response."""
    output: list[Json] = [
        {"type": "function_call", "id": f"fc_{i}", "call_id": f"call_{i}", "name": name,
         "arguments": json.dumps(args), "status": "completed"}
        for i, (name, args) in enumerate(calls)
    ]
    if text is not None:
        output.append({"type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
                       "content": [{"type": "output_text", "text": text, "annotations": []}]})
    return {
        "id": "resp_1", "object": "response", "created_at": 1, "model": model,
        "status": "completed", "output": output, "parallel_tool_calls": True,
        "tool_choice": "auto", "tools": [],
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens,
                  "total_tokens": input_tokens + output_tokens,
                  "input_tokens_details": {"cached_tokens": 0},
                  "output_tokens_details": {"reasoning_tokens": reasoning_tokens}},
    }


def message(
    text: str | None = None,
    calls: Calls = (),
    *,
    model: str = "claude-test",
    input_tokens: int = 10,
    output_tokens: int = 5,
    cache_read: int = 0,
    cache_write: int = 0,
) -> Json:
    """An Anthropic Messages response."""
    content: list[Json] = []
    if text is not None:
        content.append({"type": "text", "text": text})
    content += [{"type": "tool_use", "id": f"toolu_{i}", "name": name, "input": args}
                for i, (name, args) in enumerate(calls)]
    return {
        "id": "msg_1", "type": "message", "role": "assistant", "model": model,
        "content": content, "stop_reason": "tool_use" if calls else "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens,
                  "cache_read_input_tokens": cache_read, "cache_creation_input_tokens": cache_write},
    }


def error(status: int, message: str = "bad request") -> httpx2.Response:
    return httpx2.Response(status, json={"error": {"message": message, "type": "invalid_request"}})
