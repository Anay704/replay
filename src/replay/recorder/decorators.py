"""`@replay.agent` marks one agent run; `@replay.tool` marks a tool the agent calls."""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable, Mapping
from typing import Any, ParamSpec, TypeVar, overload

from replay.recorder.context import activate, current_run, end_run, guard, start_run

P = ParamSpec("P")
R = TypeVar("R")


@overload
def agent(fn: Callable[P, R], /) -> Callable[P, R]: ...
@overload
def agent(
    *, tags: Mapping[str, str] | None = None
) -> Callable[[Callable[P, R]], Callable[P, R]]: ...
def agent(
    fn: Callable[P, R] | None = None, /, *, tags: Mapping[str, str] | None = None
) -> Callable[P, R] | Callable[[Callable[P, R]], Callable[P, R]]:
    """Record every call of the decorated function as one trace.

    The function should be the agent's entry point: its arguments are the run's input and its
    return value is the run's output. Calls made while another run is active join that run.
    """
    if fn is None:
        return lambda f: agent(f, tags=tags)
    return _wrap_agent(fn, dict(tags or {}))


def _wrap_agent(fn: Callable[P, R], tags: dict[str, str]) -> Callable[P, R]:
    entrypoint = f"{fn.__module__}:{fn.__qualname__}"

    if inspect.iscoroutinefunction(fn):

        @functools.wraps(fn)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> Any:
            run = None if current_run() is not None else start_run(entrypoint, args, kwargs, tags)
            if run is None:
                return await fn(*args, **kwargs)
            with activate(run):
                try:
                    output = await fn(*args, **kwargs)
                except Exception as exc:
                    end_run(run, error=exc)
                    raise
            end_run(run, output=output)
            return output

        return async_wrapper  # type: ignore[return-value]

    @functools.wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        run = None if current_run() is not None else start_run(entrypoint, args, kwargs, tags)
        if run is None:
            return fn(*args, **kwargs)
        with activate(run):
            try:
                output = fn(*args, **kwargs)
            except Exception as exc:
                end_run(run, error=exc)
                raise
        end_run(run, output=output)
        return output

    return wrapper


@overload
def tool(fn: Callable[P, R], /) -> Callable[P, R]: ...
@overload
def tool(*, name: str | None = None) -> Callable[[Callable[P, R]], Callable[P, R]]: ...
def tool(
    fn: Callable[P, R] | None = None, /, *, name: str | None = None
) -> Callable[P, R] | Callable[[Callable[P, R]], Callable[P, R]]:
    """Record calls to the decorated function as tool steps (and, during replay, mock them).

    `name` defaults to the function name and should match the tool name the model sees.
    """
    if fn is None:
        return lambda f: tool(f, name=name)
    return _wrap_tool(fn, name or fn.__name__)


def _wrap_tool(fn: Callable[P, R], tool_name: str) -> Callable[P, R]:
    signature = inspect.signature(fn)

    if inspect.iscoroutinefunction(fn):

        @functools.wraps(fn)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> Any:
            run = current_run()
            bound = bind_arguments(signature, args, kwargs) if run is not None else None
            if run is None or bound is None:
                return await fn(*args, **kwargs)
            return await run.ainvoke_tool(tool_name, bound, lambda: fn(*args, **kwargs))

        return async_wrapper  # type: ignore[return-value]

    @functools.wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        run = current_run()
        bound = bind_arguments(signature, args, kwargs) if run is not None else None
        if run is None or bound is None:
            return fn(*args, **kwargs)
        return run.invoke_tool(tool_name, bound, lambda: fn(*args, **kwargs))

    return wrapper


def bind_arguments(
    signature: inspect.Signature, args: tuple[Any, ...], kwargs: Mapping[str, Any]
) -> dict[str, Any] | None:
    """Name every argument, so `f("A1")` and `f(order_id="A1")` record identically.

    Defaults are filled in, `self`/`cls` is dropped, and **kwargs are flattened. Returns None if
    the arguments don't fit the signature (the call itself will then raise TypeError).
    """
    with guard("binding tool arguments"):
        try:
            bound = signature.bind(*args, **kwargs)
        except TypeError:
            return None
        bound.apply_defaults()
        params = list(signature.parameters.values())
        skip = params[0].name if params and params[0].name in {"self", "cls"} else None
        named: dict[str, Any] = {}
        for key, value in bound.arguments.items():
            kind = signature.parameters[key].kind
            if key == skip:
                continue
            if kind is inspect.Parameter.VAR_KEYWORD:
                named.update(value)
            elif kind is inspect.Parameter.VAR_POSITIONAL:
                named[key] = list(value)
            else:
                named[key] = value
        return named
    return None
