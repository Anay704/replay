"""Replay recorded traces against a new configuration."""

from __future__ import annotations

import asyncio
import copy
import importlib
import inspect
import sys
import uuid
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from time import perf_counter
from typing import Any

from replay.engine.config import ReplayConfig
from replay.engine.netguard import network_guard
from replay.engine.run import ReplayRun
from replay.errors import ReplayAborted, ReplayConfigError
from replay.limits import Limits, RateLimiter
from replay.recorder.context import activate, now, replay_mode
from replay.recorder.wrap import llm_hosts
from replay.schema import RunResult, Session, Trace
from replay.store import TraceStore


def replay_traces(
    baselines: Sequence[Trace],
    config: ReplayConfig,
    *,
    store: TraceStore | None = None,
    limiter: RateLimiter | None = None,
    extra_hosts: Iterable[str] = (),
    on_result: Callable[[RunResult], None] | None = None,
) -> Session:
    """Re-run each baseline's input through its entry point under `config`.

    Tool calls are answered from each baseline's recording; LLM calls go live. Returns the session
    (and saves it to `store` if given). Results come back in the same order as `baselines`.
    """
    started_at, t0 = now(), perf_counter()
    session_id = uuid.uuid4().hex[:12]
    limits = config.limits
    limiter = limiter or RateLimiter(
        Limits(limits.requests_per_minute, limits.tokens_per_minute, limits.max_retries)
    )
    for path in reversed(config.pythonpath):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    entrypoints = {spec: load_entrypoint(spec) for spec in _entrypoint_specs(baselines, config)}
    replayer = _Replayer(config, session_id, limiter, config.overrides.prompt(), on_result)

    results: list[RunResult | None] = [None] * len(baselines)
    sync_jobs: list[tuple[int, Trace, str, Callable[..., Any]]] = []
    async_jobs: list[tuple[int, Trace, str, Callable[..., Any]]] = []
    for i, baseline in enumerate(baselines):
        spec = config.entrypoint or str(baseline.entrypoint)
        fn = entrypoints[spec]
        (async_jobs if inspect.iscoroutinefunction(fn) else sync_jobs).append((i, baseline, spec, fn))

    allowed = set(config.network.allow_hosts) | set(extra_hosts)
    with ExitStack() as stack:
        stack.enter_context(replay_mode())
        if config.network.guard:
            stack.enter_context(network_guard(lambda: llm_hosts() | allowed))
        if sync_jobs:
            with ThreadPoolExecutor(max_workers=limits.concurrency) as pool:
                done = pool.map(lambda job: replayer.replay(*job[1:]), sync_jobs)
                for (i, *_), result in zip(sync_jobs, done, strict=True):
                    results[i] = result
        if async_jobs:
            for (i, *_), result in zip(async_jobs, asyncio.run(replayer.replay_all(async_jobs)), strict=True):
                results[i] = result

    session = Session(
        session_id=session_id,
        created_at=started_at,
        duration_ms=(perf_counter() - t0) * 1000,
        config=config.model_dump(mode="json"),
        results=[r for r in results if r is not None],
    )
    if store is not None:
        store.put_session(session)
    return session


def _entrypoint_specs(baselines: Sequence[Trace], config: ReplayConfig) -> set[str]:
    if config.entrypoint:
        return {config.entrypoint}
    missing = [b.run_id for b in baselines if not b.entrypoint]
    if missing:
        raise ReplayConfigError(
            f"Traces {', '.join(missing[:3])} have no recorded entry point; set `entrypoint` in "
            "the config."
        )
    return {str(b.entrypoint) for b in baselines}


def load_entrypoint(spec: str) -> Callable[..., Any]:
    """Import "package.module:function"."""
    module_name, sep, attr = spec.partition(":")
    if not sep or not module_name or not attr:
        raise ReplayConfigError(f"Entry point {spec!r} must look like 'package.module:function'.")
    if "<locals>" in attr:
        raise ReplayConfigError(
            f"Entry point {spec!r} was defined inside a function, so it can't be imported. Set "
            "`entrypoint` in the config to an importable function."
        )
    try:
        obj: Any = importlib.import_module(module_name)
    except ImportError as exc:
        raise ReplayConfigError(
            f"Can't import {module_name!r} for entry point {spec!r}: {exc}. Is its directory "
            "listed in `pythonpath`?"
        ) from exc
    for part in attr.split("."):
        try:
            obj = getattr(obj, part)
        except AttributeError as exc:
            raise ReplayConfigError(f"Entry point {spec!r}: {module_name} has no {attr}.") from exc
    if not callable(obj):
        raise ReplayConfigError(f"Entry point {spec!r} is not callable.")
    return obj  # type: ignore[no-any-return]


def accepts_config(fn: Callable[..., Any]) -> bool:
    params = inspect.signature(fn).parameters.values()
    return any(p.name == "config" or p.kind is inspect.Parameter.VAR_KEYWORD for p in params)


class _Replayer:
    def __init__(
        self,
        config: ReplayConfig,
        session_id: str,
        limiter: RateLimiter,
        system_prompt: str | None,
        on_result: Callable[[RunResult], None] | None,
    ) -> None:
        self.config = config
        self.session_id = session_id
        self.limiter = limiter
        self.system_prompt = system_prompt
        self.on_result = on_result

    def _prepare(
        self, baseline: Trace, spec: str, fn: Callable[..., Any]
    ) -> tuple[ReplayRun, list[Any], dict[str, Any]] | RunResult:
        if not baseline.input.replayable:
            return RunResult(
                baseline=baseline,
                replay_error="The recorded input isn't JSON-serializable, so it can't be re-run.",
            )
        trace = Trace(
            run_id=uuid.uuid4().hex,
            source="replay",
            replay_of=baseline.run_id,
            entrypoint=spec,
            input=baseline.input.model_copy(deep=True),
            tags={**baseline.tags, "replay_session": self.session_id},
            started_at=now(),
        )
        run = ReplayRun(
            trace,
            baseline=baseline,
            config=self.config,
            limiter=self.limiter,
            system_prompt=self.system_prompt,
        )
        args = copy.deepcopy(baseline.input.args)
        kwargs = copy.deepcopy(baseline.input.kwargs)
        if self.config.agent_config is not None and accepts_config(fn):
            kwargs["config"] = copy.deepcopy(self.config.agent_config)
        return run, args, kwargs

    def _finish(
        self, baseline: Trace, run: ReplayRun, output: object, error: BaseException | None, t0: float
    ) -> RunResult:
        result = RunResult(
            baseline=baseline,
            candidate=run.finish(output=output, error=error),
            duration_ms=(perf_counter() - t0) * 1000,
        )
        if self.on_result is not None:
            self.on_result(result)
        return result

    def replay(self, baseline: Trace, spec: str, fn: Callable[..., Any]) -> RunResult:
        t0 = perf_counter()
        prepared = self._prepare(baseline, spec, fn)
        if isinstance(prepared, RunResult):
            return prepared
        run, args, kwargs = prepared
        output: object = None
        error: BaseException | None = None
        with activate(run):
            try:
                output = fn(*args, **kwargs)
            except (Exception, ReplayAborted) as exc:
                error = exc
        return self._finish(baseline, run, output, error, t0)

    async def areplay(self, baseline: Trace, spec: str, fn: Callable[..., Any]) -> RunResult:
        t0 = perf_counter()
        prepared = self._prepare(baseline, spec, fn)
        if isinstance(prepared, RunResult):
            return prepared
        run, args, kwargs = prepared
        output: object = None
        error: BaseException | None = None
        with activate(run):
            try:
                output = await fn(*args, **kwargs)
            except (Exception, ReplayAborted) as exc:
                error = exc
        return self._finish(baseline, run, output, error, t0)

    async def replay_all(
        self, jobs: list[tuple[int, Trace, str, Callable[..., Any]]]
    ) -> list[RunResult]:
        gate = asyncio.Semaphore(self.config.limits.concurrency)

        async def one(baseline: Trace, spec: str, fn: Callable[..., Any]) -> RunResult:
            async with gate:
                return await self.areplay(baseline, spec, fn)

        return list(await asyncio.gather(*(one(b, s, f) for _, b, s, f in jobs)))
