"""`replay.wrap(client)`: instrument an Anthropic or OpenAI client in place."""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any, TypeVar

from replay.recorder.adapters import Adapter
from replay.recorder.adapters.anthropic_messages import AnthropicMessages
from replay.recorder.adapters.openai_chat import OpenAIChat
from replay.recorder.adapters.openai_responses import OpenAIResponses
from replay.recorder.context import current_run

C = TypeVar("C")

_MARK = "__replay_wrapped__"

# (attribute path to the resource, methods to instrument, adapter) per SDK.
_TARGETS: dict[str, list[tuple[str, tuple[str, ...], Adapter]]] = {
    "anthropic": [("messages", ("create", "parse"), AnthropicMessages())],
    "openai": [
        ("chat.completions", ("create", "parse"), OpenAIChat()),
        ("responses", ("create", "parse"), OpenAIResponses()),
    ],
}


def wrap(client: C) -> C:
    """Record every LLM call made through `client` while an agent run is active.

    Supports `Anthropic`, `AsyncAnthropic`, `OpenAI` and `AsyncOpenAI` (including subclasses such
    as Azure or Bedrock clients, and OpenAI-compatible providers via `base_url`). The client is
    instrumented in place and returned, so `client = replay.wrap(OpenAI())` is all it takes.
    Calls made outside a run pass straight through. Wrapping twice is harmless.
    """
    sdk = _sdk_of(client)
    is_async = any(cls.__name__ == "AsyncAPIClient" for cls in type(client).__mro__)
    for path, methods, adapter in _TARGETS[sdk]:
        resource: Any = client
        for attr in path.split("."):
            resource = getattr(resource, attr)
        for method in methods:
            original = getattr(resource, method, None)
            if original is None or getattr(original, _MARK, False):
                continue
            make = _async_method if is_async else _sync_method
            setattr(resource, method, make(original, adapter))
    _rewrap_copies(client)
    return client


def _sdk_of(client: object) -> str:
    for cls in type(client).__mro__:
        root = cls.__module__.split(".")[0]
        if root in _TARGETS:
            return root
    raise TypeError(
        f"replay.wrap() supports Anthropic and OpenAI SDK clients, not {type(client).__name__}."
    )


def _sync_method(original: Callable[..., Any], adapter: Adapter) -> Callable[..., Any]:
    @functools.wraps(original)
    def method(*args: Any, **kwargs: Any) -> Any:
        run = current_run()
        if run is None:
            return original(*args, **kwargs)
        if kwargs.get("stream"):
            run.warn(f"A streaming {adapter.api} call was not recorded; streaming is unsupported.")
            return original(*args, **kwargs)
        return run.invoke_llm(adapter, original, args, kwargs)

    setattr(method, _MARK, True)
    return method


def _async_method(original: Callable[..., Any], adapter: Adapter) -> Callable[..., Any]:
    @functools.wraps(original)
    async def method(*args: Any, **kwargs: Any) -> Any:
        run = current_run()
        if run is None:
            return await original(*args, **kwargs)
        if kwargs.get("stream"):
            run.warn(f"A streaming {adapter.api} call was not recorded; streaming is unsupported.")
            return await original(*args, **kwargs)
        return await run.ainvoke_llm(adapter, original, args, kwargs)

    setattr(method, _MARK, True)
    return method


def _rewrap_copies(client: object) -> None:
    """`client.with_options(...)` / `client.copy(...)` build a fresh client; wrap that too."""
    for name in ("with_options", "copy"):
        original = getattr(client, name, None)
        if original is None or getattr(original, _MARK, False):
            continue

        def rewrapped(*args: Any, __original: Callable[..., Any] = original, **kwargs: Any) -> Any:
            return wrap(__original(*args, **kwargs))

        setattr(rewrapped, _MARK, True)
        setattr(client, name, rewrapped)
