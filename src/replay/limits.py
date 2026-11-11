"""Client-side rate limiting: pace LLM calls per model and retry rate-limited ones.

Free and low-tier API plans cap requests and tokens per minute. The limiter keeps a token bucket
per model for requests and for tokens (the token cost of a call is estimated up front and
corrected from real usage afterwards), and retries 429/overloaded errors with exponential backoff,
honoring the provider's retry-after header. Time spent waiting is reported separately from the
call's own latency, so latency comparisons measure the model rather than the limiter.
"""

from __future__ import annotations

import asyncio
import json
import random
import threading
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

T = TypeVar("T")

RETRYABLE_STATUS = {429, 503, 529}
"""Rate limited (429), service unavailable (503), and Anthropic's overloaded (529)."""


@dataclass(frozen=True)
class Limits:
    requests_per_minute: float | None = None
    tokens_per_minute: float | None = None
    max_retries: int = 6
    max_backoff_s: float = 60.0


@dataclass
class CallStats:
    attempts: int = 0
    waited_ms: float = 0.0
    """Time spent pacing and backing off, excluded from latency."""
    latency_ms: float = 0.0
    """Duration of the final attempt only."""


@dataclass
class _Bucket:
    capacity: float
    level: float
    updated: float

    def refill(self, now: float) -> None:
        rate = self.capacity / 60.0
        self.level = min(self.capacity, self.level + (now - self.updated) * rate)
        self.updated = now

    def wait_for(self, amount: float) -> float:
        """Seconds until `amount` is available. An amount above capacity only needs a full bucket."""
        needed = min(amount, self.capacity)
        return max(0.0, (needed - self.level) * 60.0 / self.capacity)


@dataclass
class _ModelState:
    requests: _Bucket | None
    tokens: _Bucket | None


@dataclass
class RateLimiter:
    limits: Limits = field(default_factory=Limits)
    clock: Callable[[], float] = time.monotonic
    sleep: Callable[[float], None] = time.sleep
    async_sleep: Callable[[float], Awaitable[None]] = asyncio.sleep
    jitter: Callable[[], float] = random.random

    def __post_init__(self) -> None:
        self._lock = threading.Lock()
        self._models: dict[str, _ModelState] = {}

    # Pacing ------------------------------------------------------------------------------

    def _state(self, model: str, now: float) -> _ModelState:
        if model not in self._models:
            rpm, tpm = self.limits.requests_per_minute, self.limits.tokens_per_minute
            self._models[model] = _ModelState(
                requests=_Bucket(rpm, rpm, now) if rpm else None,
                tokens=_Bucket(tpm, tpm, now) if tpm else None,
            )
        return self._models[model]

    def _try_take(self, model: str, tokens: int) -> float:
        """Take capacity if available (returning 0), else return how long to wait."""
        with self._lock:
            now = self.clock()
            state = self._state(model, now)
            buckets = [(b, n) for b, n in ((state.requests, 1), (state.tokens, tokens)) if b]
            for bucket, _ in buckets:
                bucket.refill(now)
            wait = max((bucket.wait_for(n) for bucket, n in buckets), default=0.0)
            if wait > 0:
                return wait
            for bucket, n in buckets:
                bucket.level -= min(n, bucket.capacity)
            return 0.0

    def settle(self, model: str, estimated: int, actual: int) -> None:
        """Correct the token bucket once the real usage of a call is known."""
        with self._lock:
            state = self._models.get(model)
            if state and state.tokens:
                bucket = state.tokens
                bucket.level = min(bucket.capacity, bucket.level + estimated - actual)

    # Retries -----------------------------------------------------------------------------

    def _backoff(self, exc: BaseException, attempt: int) -> float | None:
        """Seconds to wait before retrying `exc`, or None if it should not be retried."""
        if attempt > self.limits.max_retries or not is_retryable(exc):
            return None
        hinted = retry_after_seconds(exc)
        base = hinted if hinted is not None else min(2.0 ** (attempt - 1), self.limits.max_backoff_s)
        return min(base + self.jitter() * 0.25 * base, self.limits.max_backoff_s)

    def call(
        self,
        model: str,
        estimated_tokens: int,
        fn: Callable[[], T],
        actual_tokens: Callable[[T], int | None] | None = None,
    ) -> tuple[T, CallStats]:
        stats = CallStats()
        while True:
            while (wait := self._try_take(model, estimated_tokens)) > 0:
                self.sleep(wait)
                stats.waited_ms += wait * 1000
            stats.attempts += 1
            started = time.perf_counter()
            try:
                result = fn()
            except Exception as exc:
                delay = self._backoff(exc, stats.attempts)
                if delay is None:
                    raise
                self.sleep(delay)
                stats.waited_ms += delay * 1000
                continue
            stats.latency_ms = (time.perf_counter() - started) * 1000
            self._settle_from(model, estimated_tokens, result, actual_tokens)
            return result, stats

    async def acall(
        self,
        model: str,
        estimated_tokens: int,
        fn: Callable[[], Awaitable[T]],
        actual_tokens: Callable[[T], int | None] | None = None,
    ) -> tuple[T, CallStats]:
        stats = CallStats()
        while True:
            while (wait := self._try_take(model, estimated_tokens)) > 0:
                await self.async_sleep(wait)
                stats.waited_ms += wait * 1000
            stats.attempts += 1
            started = time.perf_counter()
            try:
                result = await fn()
            except Exception as exc:
                delay = self._backoff(exc, stats.attempts)
                if delay is None:
                    raise
                await self.async_sleep(delay)
                stats.waited_ms += delay * 1000
                continue
            stats.latency_ms = (time.perf_counter() - started) * 1000
            self._settle_from(model, estimated_tokens, result, actual_tokens)
            return result, stats

    def _settle_from(
        self,
        model: str,
        estimated: int,
        result: T,
        actual_tokens: Callable[[T], int | None] | None,
    ) -> None:
        if actual_tokens is None:
            return
        try:
            actual = actual_tokens(result)
        except Exception:
            return
        if actual is not None:
            self.settle(model, estimated, actual)


def is_retryable(exc: BaseException) -> bool:
    return getattr(exc, "status_code", None) in RETRYABLE_STATUS


def retry_after_seconds(exc: BaseException) -> float | None:
    """Read `retry-after-ms` or `retry-after` (seconds) from an SDK error's HTTP response."""
    headers = getattr(getattr(exc, "response", None), "headers", None)
    if headers is None:
        return None
    for name, scale in (("retry-after-ms", 0.001), ("retry-after", 1.0)):
        raw = headers.get(name)
        if raw is None:
            continue
        try:
            return max(0.0, float(raw) * scale)
        except ValueError:
            continue
    return None


# Rough size of a model's reply, used until the real usage is known.
_EXPECTED_OUTPUT_TOKENS = 300


def estimate_tokens(kwargs: dict[str, Any]) -> int:
    """A cheap upper-ish estimate of a request's token cost: ~4 characters per token."""
    parts = [kwargs.get(key) for key in ("messages", "input", "system", "instructions", "tools")]
    chars = len(json.dumps([p for p in parts if p is not None], default=str))
    return chars // 4 + _EXPECTED_OUTPUT_TOKENS
