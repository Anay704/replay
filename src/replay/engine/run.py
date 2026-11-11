"""A replayed run: tool calls are answered from the recording, LLM calls go live with overrides."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal, TypeVar

from pydantic import JsonValue

from replay.engine.config import ReplayConfig
from replay.engine.matcher import Match, NearMiss, ToolMatcher
from replay.errors import ReplayAborted, ReplayBlockedToolError
from replay.limits import RateLimiter
from replay.recorder.context import Run, now, serialize_args
from replay.schema import MatchInfo, ToolStep, Trace
from replay.serialize import error_info, rebuild, rebuild_error

if TYPE_CHECKING:
    from replay.recorder.adapters import Adapter

T = TypeVar("T")
Decision = Match | Literal["live"] | ReplayBlockedToolError


class ReplayRun(Run):
    fail_closed = True
    owns_retries = True

    def __init__(
        self,
        trace: Trace,
        *,
        baseline: Trace,
        config: ReplayConfig,
        limiter: RateLimiter,
        system_prompt: str | None = None,
    ) -> None:
        super().__init__(trace)
        self.baseline = baseline
        self.config = config
        self.matcher = ToolMatcher(baseline, config.tools)
        self.limiter = limiter
        self.system_prompt = system_prompt
        self._llm_calls = 0

    # Tools ---------------------------------------------------------------------------------

    def invoke_tool(
        self, name: str, args: dict[str, Any], call: Callable[[], T], resolution: Any = None
    ) -> T:
        decision, json_args = self._decide(name, args)
        if decision == "live":
            return super().invoke_tool(name, args, call, resolution="live")
        return self._serve_or_block(name, json_args, decision)  # type: ignore[return-value]

    async def ainvoke_tool(
        self,
        name: str,
        args: dict[str, Any],
        call: Callable[[], Awaitable[T]],
        resolution: Any = None,
    ) -> T:
        decision, json_args = self._decide(name, args)
        if decision == "live":
            return await super().ainvoke_tool(name, args, call, resolution="live")
        return self._serve_or_block(name, json_args, decision)  # type: ignore[return-value]

    def _decide(self, name: str, args: dict[str, Any]) -> tuple[Decision, dict[str, JsonValue]]:
        json_args, serializable = serialize_args(args)
        found = self.matcher.match(name, json_args) if serializable else None
        if isinstance(found, Match):
            return found, json_args
        if name in self.config.tools.allow_live:
            return "live", json_args
        return self._blocked(name, found), json_args

    def _blocked(self, name: str, near: NearMiss | None) -> ReplayBlockedToolError:
        if near is None:
            why = "the recorded run never called it with serializable arguments"
        else:
            why = f"the closest recorded call is only {near.confidence:.0%} similar"
        return ReplayBlockedToolError(
            f"Tool {name!r} was blocked during replay: no recorded call matches ({why}), and it "
            "isn't listed in tools.allow_live, so it can't run for real."
        )

    def _serve_or_block(
        self, name: str, json_args: dict[str, JsonValue], decision: Decision
    ) -> object:
        index, started_at = self.reserve_index(), now()
        if isinstance(decision, Match):
            return self._serve(index, started_at, name, json_args, decision)
        assert isinstance(decision, ReplayBlockedToolError)
        near = self.matcher.match(name, json_args)
        self.add_step(
            ToolStep(
                index=index,
                started_at=started_at,
                latency_ms=0.0,
                name=name,
                args=json_args,
                resolution="blocked",
                error=error_info(decision),
                match=_match_info(near) if isinstance(near, NearMiss) else None,
            )
        )
        if self.config.tools.on_blocked == "abort":
            raise ReplayAborted(str(decision))
        raise decision

    def _serve(
        self,
        index: int,
        started_at: datetime,
        name: str,
        json_args: dict[str, JsonValue],
        match: Match,
    ) -> object:
        recorded = match.step
        self.add_step(
            ToolStep(
                index=index,
                started_at=started_at,
                # Mocked calls count at their recorded latency, so mocking doesn't look faster.
                latency_ms=recorded.latency_ms,
                name=name,
                args=json_args,
                result=recorded.result,
                result_type=recorded.result_type,
                error=recorded.error,
                resolution=match.tier,
                match=MatchInfo(
                    baseline_index=recorded.index,
                    confidence=match.confidence,
                    arg_diff=match.arg_diff,
                    reused=match.reused,
                ),
            )
        )
        if recorded.error is not None:
            raise rebuild_error(recorded.error)
        return rebuild(recorded.result, recorded.result_type)

    # LLM calls -----------------------------------------------------------------------------

    def invoke_llm(
        self,
        adapter: Adapter,
        create: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        limiter: RateLimiter | None = None,
    ) -> Any:
        self._count_llm_call()
        overridden = self._override(adapter, kwargs)
        return super().invoke_llm(adapter, create, args, overridden, limiter or self.limiter)

    async def ainvoke_llm(
        self,
        adapter: Adapter,
        create: Callable[..., Awaitable[Any]],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        limiter: RateLimiter | None = None,
    ) -> Any:
        self._count_llm_call()
        overridden = self._override(adapter, kwargs)
        return await super().ainvoke_llm(adapter, create, args, overridden, limiter or self.limiter)

    def _count_llm_call(self) -> None:
        with self._lock:
            self._llm_calls += 1
            count = self._llm_calls
        limit = self.config.limits.max_llm_calls_per_run
        if count > limit:
            raise ReplayAborted(
                f"The run made more than {limit} LLM calls (limits.max_llm_calls_per_run) and was "
                "stopped as a likely loop."
            )

    def _override(self, adapter: Adapter, kwargs: dict[str, Any]) -> dict[str, Any]:
        overrides = self.config.overrides
        out = {**kwargs, **overrides.params}
        if overrides.model is not None and "model" in out:
            out["model"] = overrides.model_for(str(out["model"]))
        if self.system_prompt is not None:
            out = adapter.with_system_prompt(out, self.system_prompt)
        return out


def _match_info(near: NearMiss) -> MatchInfo:
    return MatchInfo(
        baseline_index=near.step.index, confidence=near.confidence, arg_diff=near.arg_diff
    )
