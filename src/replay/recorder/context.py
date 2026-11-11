"""The run currently being recorded, and recorder-wide settings.

The current run lives in a ContextVar, so concurrent runs in threads or asyncio tasks never see
each other's steps. Recording fails open: any error inside Replay is logged and the agent carries on.
"""

from __future__ import annotations

import logging
import threading
import uuid
from collections.abc import Awaitable, Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime
from time import perf_counter
from typing import TYPE_CHECKING, Any, TypeVar

from replay.recorder.sinks import MemorySink, Sink
from replay.schema import JsonValue, RunInput, Step, ToolStep, Trace
from replay.serialize import error_info, to_jsonable

if TYPE_CHECKING:
    from replay.recorder.adapters import Adapter

log = logging.getLogger("replay")
T = TypeVar("T")


@dataclass
class Settings:
    enabled: bool = True
    strict: bool = False
    """Raise recorder errors instead of logging them. For Replay's own tests."""
    sink: Sink = field(default_factory=MemorySink)
    tags: dict[str, str] = field(default_factory=dict)


settings = Settings()


def configure(
    *,
    enabled: bool | None = None,
    sink: Sink | None = None,
    tags: dict[str, str] | None = None,
    strict: bool | None = None,
) -> None:
    """Change recorder settings. Arguments left as None keep their current value."""
    if enabled is not None:
        settings.enabled = enabled
    if sink is not None:
        settings.sink = sink
    if tags is not None:
        settings.tags = dict(tags)
    if strict is not None:
        settings.strict = strict


@contextmanager
def guard(action: str) -> Iterator[None]:
    """Swallow and log errors raised by Replay itself, so recording never breaks the agent."""
    try:
        yield
    except Exception:
        if settings.strict:
            raise
        log.exception("replay: %s failed; the agent continues unaffected", action)


def now() -> datetime:
    return datetime.now(UTC)


class Run:
    """One agent run being recorded.

    The replay engine subclasses this and overrides the `invoke_*` hooks to serve recorded tool
    results and apply configuration overrides; decorators and client wrappers only call the hooks.
    """

    def __init__(self, trace: Trace) -> None:
        self.trace = trace
        self._lock = threading.Lock()
        self._next_index = 0

    def reserve_index(self) -> int:
        """Steps are numbered in the order they start (parallel calls may finish out of order)."""
        with self._lock:
            index = self._next_index
            self._next_index += 1
            return index

    def add_step(self, step: Step) -> None:
        with self._lock:
            self.trace.steps.append(step)

    def warn(self, message: str) -> None:
        with self._lock:
            if message not in self.trace.warnings:
                self.trace.warnings.append(message)
        log.warning("replay: %s", message)

    # Tool calls ----------------------------------------------------------------------------

    def invoke_tool(self, name: str, args: dict[str, Any], call: Callable[[], T]) -> T:
        index, started_at, t0 = self.reserve_index(), now(), perf_counter()
        try:
            result = call()
        except Exception as exc:
            self._record_tool(index, started_at, t0, name, args, error=exc)
            raise
        self._record_tool(index, started_at, t0, name, args, result=result)
        return result

    async def ainvoke_tool(
        self, name: str, args: dict[str, Any], call: Callable[[], Awaitable[T]]
    ) -> T:
        index, started_at, t0 = self.reserve_index(), now(), perf_counter()
        try:
            result = await call()
        except Exception as exc:
            self._record_tool(index, started_at, t0, name, args, error=exc)
            raise
        self._record_tool(index, started_at, t0, name, args, result=result)
        return result

    def _record_tool(
        self,
        index: int,
        started_at: datetime,
        t0: float,
        name: str,
        args: dict[str, Any],
        *,
        result: object = None,
        error: BaseException | None = None,
    ) -> None:
        latency_ms = (perf_counter() - t0) * 1000
        with guard(f"recording tool call {name!r}"):
            json_args, args_ok = serialize_args(args)
            serialized = to_jsonable(result)
            replayable = args_ok and (error is not None or serialized.replayable)
            if not replayable:
                self.warn(
                    f"Tool {name!r} has arguments or a result that are not JSON-serializable; "
                    "the step was stored as its repr and cannot be replayed faithfully."
                )
            self.add_step(
                ToolStep(
                    index=index,
                    started_at=started_at,
                    latency_ms=latency_ms,
                    name=name,
                    args=json_args,
                    result=None if error is not None else serialized.value,
                    result_type="json" if error is not None else serialized.type_tag,
                    replayable=replayable,
                    error=error_info(error) if error is not None else None,
                )
            )

    # LLM calls -----------------------------------------------------------------------------

    def invoke_llm(
        self, adapter: Adapter, create: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> Any:
        index, started_at, t0 = self.reserve_index(), now(), perf_counter()
        try:
            response = create(*args, **kwargs)
        except Exception as exc:
            self._record_llm(adapter, index, started_at, t0, kwargs, error=exc)
            raise
        self._record_llm(adapter, index, started_at, t0, kwargs, response=response)
        return response

    async def ainvoke_llm(
        self,
        adapter: Adapter,
        create: Callable[..., Awaitable[Any]],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        index, started_at, t0 = self.reserve_index(), now(), perf_counter()
        try:
            response = await create(*args, **kwargs)
        except Exception as exc:
            self._record_llm(adapter, index, started_at, t0, kwargs, error=exc)
            raise
        self._record_llm(adapter, index, started_at, t0, kwargs, response=response)
        return response

    def _record_llm(
        self,
        adapter: Adapter,
        index: int,
        started_at: datetime,
        t0: float,
        kwargs: dict[str, Any],
        *,
        response: Any = None,
        error: BaseException | None = None,
    ) -> None:
        latency_ms = (perf_counter() - t0) * 1000
        with guard(f"recording a {adapter.api} call"):
            self.add_step(
                adapter.build_step(
                    index=index,
                    started_at=started_at,
                    latency_ms=latency_ms,
                    kwargs=kwargs,
                    response=response,
                    error=error,
                )
            )

    # Finishing -----------------------------------------------------------------------------

    def finish(self, *, output: object = None, error: BaseException | None = None) -> Trace:
        trace = self.trace
        trace.ended_at = now()
        trace.steps.sort(key=lambda step: step.index)
        if error is not None:
            trace.error = error_info(error)
        else:
            serialized = to_jsonable(output)
            trace.output = serialized.value
            if not serialized.replayable:
                self.warn("The agent's return value is not JSON-serializable; stored as its repr.")
        self._check_uninstrumented_tools()
        return trace

    def _check_uninstrumented_tools(self) -> None:
        requested = {call.name for step in self.trace.llm_steps() for call in step.tool_calls}
        executed = {step.name for step in self.trace.tool_steps()}
        for name in sorted(requested - executed):
            self.warn(
                f"The model requested tool {name!r} but no @replay.tool named {name!r} ran in "
                "this run. If the agent executes it, Replay can neither see nor mock it."
            )


_current: ContextVar[Run | None] = ContextVar("replay_current_run", default=None)


def current_run() -> Run | None:
    return _current.get()


def serialize_args(args: Mapping[str, Any]) -> tuple[dict[str, JsonValue], bool]:
    out: dict[str, JsonValue] = {}
    ok = True
    for key, value in args.items():
        serialized = to_jsonable(value)
        out[key] = serialized.value
        ok = ok and serialized.replayable
    return out, ok


def start_run(
    entrypoint: str, args: tuple[Any, ...], kwargs: dict[str, Any], tags: Mapping[str, str]
) -> Run | None:
    """Begin recording a run, or return None if recording is off or fails to start."""
    if not settings.enabled:
        return None
    with guard("starting a run"):
        json_args = [to_jsonable(a) for a in args]
        json_kwargs, kwargs_ok = serialize_args(kwargs)
        trace = Trace(
            run_id=uuid.uuid4().hex,
            entrypoint=entrypoint,
            input=RunInput(args=[a.value for a in json_args], kwargs=json_kwargs),
            tags={**settings.tags, **tags},
            started_at=now(),
        )
        run = Run(trace)
        if not (kwargs_ok and all(a.replayable for a in json_args)):
            run.warn("The agent's input is not JSON-serializable; this run cannot be replayed.")
        return run
    return None


def end_run(run: Run, *, output: object = None, error: BaseException | None = None) -> None:
    with guard("saving a run"):
        settings.sink.write(run.finish(output=output, error=error))


@contextmanager
def activate(run: Run) -> Iterator[Run]:
    token = _current.set(run)
    try:
        yield run
    finally:
        _current.reset(token)
