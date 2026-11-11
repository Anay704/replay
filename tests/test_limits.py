from __future__ import annotations

import asyncio
from typing import Any

import httpx2
import openai
import pytest

import replay
from replay.limits import Limits, RateLimiter, estimate_tokens, retry_after_seconds
from tests.fakes import FakeLLM, chat


class FakeTime:
    """A clock that only moves when the limiter sleeps."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(round(seconds, 3))
        self.now += seconds

    async def async_sleep(self, seconds: float) -> None:
        self.sleep(seconds)


def limiter(t: FakeTime, **limits: Any) -> RateLimiter:
    return RateLimiter(
        Limits(**limits),
        clock=lambda: t.now,
        sleep=t.sleep,
        async_sleep=t.async_sleep,
        jitter=lambda: 0.0,
    )


class RateLimited(Exception):
    def __init__(self, retry_after: str | None = None, status_code: int = 429) -> None:
        super().__init__("rate limited")
        self.status_code = status_code
        headers = {"retry-after": retry_after} if retry_after else {}
        self.response = httpx2.Response(status_code, headers=headers)


def test_requests_per_minute() -> None:
    t = FakeTime()
    lim = limiter(t, requests_per_minute=2)
    for _ in range(3):
        lim.call("m", 0, lambda: None)
    assert t.sleeps == [30.0]  # the third request waits for one request's worth of refill


def test_tokens_per_minute_and_settling() -> None:
    t = FakeTime()
    lim = limiter(t, tokens_per_minute=1000)
    lim.call("m", 400, lambda: None)
    lim.call("m", 400, lambda: None)
    lim.call("m", 400, lambda: None)  # needs 200 more tokens: 200 / (1000/60) = 12s
    assert t.sleeps == [12.0]

    t2 = FakeTime()
    lim2 = limiter(t2, tokens_per_minute=1000)
    for _ in range(3):  # each call really used 100 tokens, so the estimates are refunded
        lim2.call("m", 400, lambda: "ok", actual_tokens=lambda _: 100)
    assert t2.sleeps == []


def test_oversized_requests_wait_for_a_full_bucket_instead_of_forever() -> None:
    t = FakeTime()
    lim = limiter(t, tokens_per_minute=1000)
    lim.call("m", 5000, lambda: None)
    lim.call("m", 5000, lambda: None)
    assert t.sleeps == [60.0]


def test_models_are_paced_independently() -> None:
    t = FakeTime()
    lim = limiter(t, requests_per_minute=1)
    lim.call("big", 0, lambda: None)
    lim.call("small", 0, lambda: None)
    assert t.sleeps == []


def test_rate_limited_calls_are_retried_honoring_retry_after() -> None:
    t = FakeTime()
    lim = limiter(t)
    errors = [RateLimited("2"), RateLimited("2.5")]

    def flaky() -> str:
        if errors:
            raise errors.pop(0)
        return "ok"

    result, stats = lim.call("m", 0, flaky)
    assert result == "ok"
    assert stats.attempts == 3
    assert t.sleeps == [2.0, 2.5]
    assert stats.waited_ms == pytest.approx(4500)


def test_backoff_without_a_hint_is_exponential() -> None:
    t = FakeTime()
    lim = limiter(t, max_retries=3)
    with pytest.raises(RateLimited):
        lim.call("m", 0, lambda: (_ for _ in ()).throw(RateLimited(status_code=529)))
    assert t.sleeps == [1.0, 2.0, 4.0]


def test_other_errors_are_not_retried() -> None:
    t = FakeTime()
    lim = limiter(t)
    with pytest.raises(ValueError):
        lim.call("m", 0, lambda: (_ for _ in ()).throw(ValueError("bad input")))
    assert t.sleeps == []


def test_async_calls() -> None:
    t = FakeTime()
    lim = limiter(t, requests_per_minute=1)
    errors = [RateLimited("1")]

    async def flaky() -> str:
        if errors:
            raise errors.pop(0)
        return "ok"

    async def main() -> list[str]:
        return [(await lim.acall("m", 0, flaky))[0] for _ in range(2)]

    assert asyncio.run(main()) == ["ok", "ok"]
    # 1s backoff, then the retry and the second call each need a fresh request slot.
    assert t.sleeps == [1.0, 59.0, 60.0]


def test_retry_after_parsing() -> None:
    assert retry_after_seconds(RateLimited("3")) == 3.0
    assert retry_after_seconds(RateLimited("soon")) is None
    assert retry_after_seconds(ValueError()) is None


def test_estimate_tokens_counts_the_prompt() -> None:
    small = estimate_tokens({"messages": [{"role": "user", "content": "hi"}]})
    big = estimate_tokens({"messages": [{"role": "user", "content": "hi " * 1000}]})
    assert big - small == pytest.approx(3000 // 4, abs=5)


def test_wrapped_client_uses_the_limiter_and_records_waits(sink: replay.MemorySink) -> None:
    """A real SDK 429 (with retry-after) is retried by the limiter and kept out of latency."""
    t = FakeTime()
    too_many = httpx2.Response(
        429, headers={"retry-after": "3"}, json={"error": {"message": "slow down"}}
    )
    client = replay.wrap(FakeLLM([too_many, chat("ok")]).openai(), limiter=limiter(t))

    @replay.agent
    def run() -> str:
        return str(client.chat.completions.create(model="m", messages=[]).choices[0].message.content)

    assert run() == "ok"
    [step] = sink.traces[0].llm_steps()
    assert (step.attempts, step.rate_limit_wait_ms) == (2, 3000.0)
    assert step.latency_ms < 1000
    assert t.sleeps == [3.0]


def test_wrapped_client_gives_up_after_max_retries(sink: replay.MemorySink) -> None:
    t = FakeTime()
    too_many = [httpx2.Response(429, json={"error": {"message": "slow"}}) for _ in range(3)]
    client = replay.wrap(FakeLLM(too_many).openai(), limiter=limiter(t, max_retries=2))

    @replay.agent
    def run() -> None:
        client.chat.completions.create(model="m", messages=[])

    with pytest.raises(openai.RateLimitError):
        run()
    [step] = sink.traces[0].llm_steps()
    assert step.error is not None and step.error.type == "openai.RateLimitError"
