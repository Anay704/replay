from __future__ import annotations

import asyncio
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx2
import pytest

import replay
from replay.engine import ReplayConfig, load_config, replay_traces
from replay.engine.runner import load_entrypoint
from replay.errors import ReplayConfigError
from replay.limits import Limits, RateLimiter
from replay.recorder.adapters.anthropic_messages import AnthropicMessages
from replay.recorder.adapters.openai_chat import OpenAIChat
from replay.recorder.adapters.openai_responses import OpenAIResponses
from replay.schema import Session, ToolStep, Trace
from replay.store import TraceStore
from tests import toy_agent
from tests.fakes import FakeLLM, chat


@pytest.fixture(autouse=True)
def reset_toy_agent() -> Iterator[None]:
    for log in (toy_agent.SIDE_EFFECTS, toy_agent.OBSERVED, toy_agent.CONFIGS):
        log.clear()
    yield
    toy_agent.client = None


def calls(*names_and_args: tuple[str, dict[str, Any]]) -> dict[str, Any]:
    return chat(calls=list(names_and_args))


def record(entry: Any, responses: list[Any], sink: replay.MemorySink, message: str = "hi") -> Trace:
    """Run the toy agent for real against scripted model responses, and return its trace."""
    toy_agent.client = replay.wrap(FakeLLM(responses).openai())
    entry(message)
    trace = sink.traces[-1]
    toy_agent.SIDE_EFFECTS.clear()
    toy_agent.OBSERVED.clear()
    toy_agent.CONFIGS.clear()
    return trace


def replay_with(
    baselines: list[Trace], responses: list[Any], config: ReplayConfig | None = None, **kwargs: Any
) -> Session:
    fake = FakeLLM(responses)
    toy_agent.client = replay.wrap(fake.openai())
    kwargs.setdefault("limiter", RateLimiter(Limits(), sleep=lambda s: None))
    return replay_traces(baselines, config or ReplayConfig(), **kwargs)


def candidate_tools(session: Session) -> list[ToolStep]:
    [result] = session.results
    assert result.candidate is not None
    return result.candidate.tool_steps()


LOOKUP = ("lookup_order", {"order_id": "A1"})
REFUND = ("refund", {"order_id": "A1", "amount": 50.0})
SCRIPT = [calls(LOOKUP), calls(REFUND), chat("Refunded.")]


def test_mocked_tools_serve_recorded_results_without_running(sink: replay.MemorySink) -> None:
    baseline = record(toy_agent.agent, SCRIPT, sink)
    session = replay_with([baseline], SCRIPT)

    assert toy_agent.SIDE_EFFECTS == []  # nothing really ran
    assert toy_agent.OBSERVED == ["Order", "Receipt"]  # recorded values rebuilt to their types
    steps = candidate_tools(session)
    assert [(s.name, s.resolution) for s in steps] == [("lookup_order", "exact"), ("refund", "exact")]
    assert [s.latency_ms for s in steps] == [s.latency_ms for s in baseline.tool_steps()]
    assert steps[1].result == {"refund_id": "R-A1", "amount": 50.0}
    [result] = session.results
    assert result.candidate is not None
    assert (result.candidate.source, result.candidate.replay_of) == ("replay", baseline.run_id)
    assert result.candidate.output == "Refunded."
    assert result.candidate.tags["replay_session"] == session.session_id


def test_recorded_errors_are_replayed_as_the_original_exception(sink: replay.MemorySink) -> None:
    script = [calls(("lookup_order", {"order_id": "MISSING"})), chat("Not found.")]
    baseline = record(toy_agent.agent, script, sink)
    session = replay_with([baseline], script)
    assert toy_agent.OBSERVED == ["OrderNotFound"]
    assert toy_agent.SIDE_EFFECTS == []
    [step] = candidate_tools(session)
    assert step.error is not None and step.error.type == "tests.toy_agent.OrderNotFound"


def test_unmatched_side_effect_tools_are_blocked(sink: replay.MemorySink) -> None:
    baseline = record(toy_agent.agent, SCRIPT, sink)
    over_refund = ("refund", {"order_id": "A1", "amount": 500.0})
    session = replay_with([baseline], [calls(LOOKUP), calls(over_refund), chat("Sorry.")])

    assert toy_agent.SIDE_EFFECTS == []
    assert toy_agent.OBSERVED == ["Order", "ReplayBlockedToolError"]
    blocked = candidate_tools(session)[1]
    assert blocked.resolution == "blocked"
    assert blocked.error is not None and "isn't listed in tools.allow_live" in blocked.error.message
    assert blocked.match is not None and blocked.match.confidence < 0.8
    assert blocked.match.arg_diff == {"amount": {"recorded": 50.0, "replayed": 500.0}}
    assert session.results[0].candidate.output == "Sorry."  # type: ignore[union-attr]


def test_abort_mode_stops_the_run(sink: replay.MemorySink) -> None:
    baseline = record(toy_agent.agent, SCRIPT, sink)
    config = ReplayConfig.model_validate({"tools": {"on_blocked": "abort"}})
    over_refund = ("refund", {"order_id": "A1", "amount": 500.0})
    session = replay_with([baseline], [calls(LOOKUP), calls(over_refund), chat("x")], config)
    candidate = session.results[0].candidate
    assert candidate is not None and candidate.error is not None
    assert candidate.error.type == "replay.errors.ReplayAborted"
    assert candidate.output is None
    assert toy_agent.SIDE_EFFECTS == []


def test_allowed_read_only_tools_run_live(sink: replay.MemorySink) -> None:
    baseline = record(toy_agent.agent, [calls(("search", {"query": "refunds"})), chat("ok")], sink)
    config = ReplayConfig.model_validate({"tools": {"allow_live": ["search"]}})
    session = replay_with(
        [baseline], [calls(("search", {"query": "shipping times"})), chat("ok")], config
    )
    assert toy_agent.SIDE_EFFECTS == [("search", "shipping times")]
    [step] = candidate_tools(session)
    assert (step.resolution, step.result) == ("live", ["doc about shipping times"])


def test_tools_called_outside_the_run_context_are_blocked(sink: replay.MemorySink) -> None:
    baseline = record(toy_agent.threaded_agent, SCRIPT, sink)
    assert baseline.tool_steps() == []  # the threads lost the context while recording, too
    assert any("can neither see nor mock" in w for w in baseline.warnings)

    replay_with([baseline], SCRIPT)
    assert toy_agent.SIDE_EFFECTS == []
    assert toy_agent.OBSERVED == ["ReplayBlockedToolError", "ReplayBlockedToolError"]


def test_overrides_reach_the_model(sink: replay.MemorySink) -> None:
    baseline = record(toy_agent.agent, [chat("hello")], sink)
    config = ReplayConfig.model_validate(
        {"overrides": {"model": "m-small", "params": {"temperature": 0}, "system_prompt": "NEW"}}
    )
    fake = FakeLLM([chat("hi")])
    toy_agent.client = replay.wrap(fake.openai())
    session = replay_traces([baseline], config)

    [sent] = fake.requests
    assert (sent["model"], sent["temperature"]) == ("m-small", 0)
    assert sent["messages"][0] == {"role": "system", "content": "NEW"}
    step = session.results[0].candidate.llm_steps()[0]  # type: ignore[union-attr]
    assert step.model == "m-small" and step.request["temperature"] == 0

    mapped = ReplayConfig.model_validate({"overrides": {"model": {"m": "m-mapped", "x": "y"}}})
    fake = FakeLLM([chat("hi")])
    toy_agent.client = replay.wrap(fake.openai())
    replay_traces([baseline], mapped)
    assert fake.requests[0]["model"] == "m-mapped"


def test_system_prompt_overrides_per_api() -> None:
    assert AnthropicMessages().with_system_prompt({"system": "a"}, "b") == {"system": "b"}
    assert OpenAIResponses().with_system_prompt({"instructions": "a"}, "b") == {"instructions": "b"}
    chat_api = OpenAIChat()
    original = [{"role": "developer", "content": "a"}, {"role": "user", "content": "q"}]
    replaced = chat_api.with_system_prompt({"messages": original}, "b")["messages"]
    assert replaced == [{"role": "developer", "content": "b"}, {"role": "user", "content": "q"}]
    assert original[0]["content"] == "a"  # the agent's own list is never mutated
    added = chat_api.with_system_prompt({"messages": [{"role": "user", "content": "q"}]}, "b")
    assert added["messages"][0] == {"role": "system", "content": "b"}


def test_runaway_loops_are_stopped(sink: replay.MemorySink) -> None:
    search = ("search", {"query": "refunds"})
    baseline = record(toy_agent.agent, [calls(search), chat("ok")], sink)
    config = ReplayConfig.model_validate(
        {"tools": {"allow_live": ["search"]}, "limits": {"max_llm_calls_per_run": 3}}
    )
    session = replay_with([baseline], lambda body: calls(search), config)  # type: ignore[arg-type]
    candidate = session.results[0].candidate
    assert candidate is not None and candidate.error is not None
    assert "more than 3 LLM calls" in candidate.error.message
    assert len(candidate.llm_steps()) == 3


def test_agent_config_is_passed_only_to_entry_points_that_accept_it(
    sink: replay.MemorySink,
) -> None:
    with_config = record(toy_agent.agent, [chat("a")], sink)
    without = record(toy_agent.agent_without_config, [chat("b")], sink)
    config = ReplayConfig.model_validate({"agent_config": {"prompt_version": "v2"}})
    session = replay_with([with_config, without], [chat("a"), chat("b")], config)
    assert toy_agent.CONFIGS == [{"prompt_version": "v2"}, None]
    assert all(r.candidate and r.candidate.error is None for r in session.results)


def test_async_entry_points(sink: replay.MemorySink) -> None:
    toy_agent.client = replay.wrap(FakeLLM(SCRIPT).async_openai())
    asyncio.run(toy_agent.async_agent("hi"))
    baseline = sink.traces[-1]
    toy_agent.SIDE_EFFECTS.clear()
    toy_agent.OBSERVED.clear()

    toy_agent.client = replay.wrap(FakeLLM(SCRIPT).async_openai())
    session = replay_traces([baseline], ReplayConfig())
    assert toy_agent.SIDE_EFFECTS == []
    assert toy_agent.OBSERVED == ["Order", "Receipt"]
    assert [s.resolution for s in candidate_tools(session)] == ["exact", "exact"]


def test_replay_owns_retries_and_keeps_waits_out_of_latency(sink: replay.MemorySink) -> None:
    baseline = record(toy_agent.agent, [chat("ok")], sink)
    too_many = httpx2.Response(
        429, headers={"retry-after-ms": "5"}, json={"error": {"message": "slow"}}
    )
    fake = FakeLLM([too_many, too_many, too_many, chat("ok")])
    raw = fake.openai().with_options(max_retries=2)  # the SDK would retry twice on its own
    toy_agent.client = replay.wrap(raw)
    session = replay_traces(
        [baseline], ReplayConfig(), limiter=RateLimiter(Limits(), sleep=lambda s: None)
    )
    step = session.results[0].candidate.llm_steps()[0]  # type: ignore[union-attr]
    assert step.attempts == 4  # every attempt went through Replay, none through the SDK
    assert len(fake.requests) == 4
    assert step.rate_limit_wait_ms > 0


def test_unreplayable_inputs_are_reported(sink: replay.MemorySink) -> None:
    baseline = record(toy_agent.agent, [chat("ok")], sink)
    baseline.input.replayable = False
    session = replay_with([baseline], [])
    [result] = session.results
    assert result.candidate is None
    assert result.replay_error is not None and "can't be re-run" in result.replay_error


def test_results_keep_baseline_order(sink: replay.MemorySink) -> None:
    baselines = [record(toy_agent.agent, [chat(f"answer {i}")], sink, f"q{i}") for i in range(6)]

    def respond(body: dict[str, Any]) -> dict[str, Any]:
        return chat(f"answer {body['messages'][-1]['content'][1:]}")

    config = ReplayConfig.model_validate({"limits": {"concurrency": 3}})
    session = replay_with(baselines, respond, config)  # type: ignore[arg-type]
    assert [r.baseline.run_id for r in session.results] == [b.run_id for b in baselines]
    assert [r.candidate.output for r in session.results] == [f"answer {i}" for i in range(6)]  # type: ignore[union-attr]


def test_sessions_are_stored(sink: replay.MemorySink, tmp_path: Path) -> None:
    baseline = record(toy_agent.agent, SCRIPT, sink)
    store = TraceStore(tmp_path / "db.sqlite")
    session = replay_with([baseline], SCRIPT, store=store)
    assert store.get_session(session.session_id) == session
    assert store.get_session() == session  # latest
    [(session_id, _, runs)] = store.list_sessions()
    assert (session_id, runs) == (session.session_id, 1)
    with pytest.raises(KeyError):
        store.get_session("nope")


def test_entry_point_errors() -> None:
    with pytest.raises(ReplayConfigError, match="must look like"):
        load_entrypoint("tests.toy_agent")
    with pytest.raises(ReplayConfigError, match="Can't import"):
        load_entrypoint("no_such_module:run")
    with pytest.raises(ReplayConfigError, match="has no"):
        load_entrypoint("tests.toy_agent:missing")
    with pytest.raises(ReplayConfigError, match="inside a function"):
        load_entrypoint("tests.test_runs:make_agent.<locals>.handle")
    assert load_entrypoint("tests.toy_agent:agent") is toy_agent.agent


def test_load_config(tmp_path: Path) -> None:
    (tmp_path / "prompts").mkdir()
    (tmp_path / "prompts" / "v2.txt").write_text("Be brief.")
    (tmp_path / "replay.yaml").write_text(
        "entrypoint: agent:run\n"
        "pythonpath: [src]\n"
        "overrides:\n  model: small\n  system_prompt_file: prompts/v2.txt\n"
        "tools:\n  allow_live: [search]\n  volatile_fields: {'*': [request_id]}\n"
    )
    config = load_config(tmp_path / "replay.yaml")
    assert config.pythonpath == [(tmp_path / "src").resolve()]
    assert config.overrides.prompt() == "Be brief."
    assert config.tools.volatile_for("search") == {"request_id"}

    (tmp_path / "typo.yaml").write_text("tools:\n  alow_live: [search]\n")
    with pytest.raises(ReplayConfigError, match="alow_live"):
        load_config(tmp_path / "typo.yaml")
    (tmp_path / "both.yaml").write_text("overrides: {system_prompt: a, system_prompt_file: b}\n")
    with pytest.raises(ReplayConfigError, match="not both"):
        load_config(tmp_path / "both.yaml")
    with pytest.raises(ReplayConfigError, match="Can't read"):
        load_config(tmp_path / "missing.yaml")
