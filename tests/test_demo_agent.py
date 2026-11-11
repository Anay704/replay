"""The demo support agent on the scripted models: correct under scripted-large, and wrong in
exactly the four designed ways under scripted-small."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import openai
import pytest

import replay
from replay.store import read_traces
from support_agent import generate_traces
from support_agent.agent import handle
from support_agent.data import ORDERS, read_ledger
from support_agent.scenarios import FLAWED_CATEGORIES, SCENARIOS


@pytest.fixture(autouse=True)
def demo_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    ledger = tmp_path / "ledger.jsonl"
    monkeypatch.setenv("SUPPORT_AGENT_LEDGER", str(ledger))
    monkeypatch.setenv("SUPPORT_AGENT_SCRIPTED_LATENCY", "0")
    monkeypatch.setenv("SUPPORT_AGENT_BACKEND", "scripted")
    monkeypatch.setenv("SUPPORT_AGENT_MODEL", "scripted-large")
    yield ledger


def record_all(model: str) -> dict[str, replay.Trace]:
    traces = generate_traces.record(SCENARIOS, backend="scripted", model=model, concurrency=8)
    return {t.tags["scenario"]: t for t in traces}


def tool_names(trace: replay.Trace) -> list[str]:
    return [s.name for s in trace.tool_steps()]


def refunds(trace: replay.Trace) -> list[dict[str, object]]:
    return [s.args for s in trace.tool_steps() if s.name == "issue_refund"]


def by_category(traces: dict[str, replay.Trace], category: str) -> list[replay.Trace]:
    return [traces[s.id] for s in SCENARIOS if s.category == category]


def test_fifty_scenarios_across_the_categories() -> None:
    assert len(SCENARIOS) == 50
    assert len({s.message for s in SCENARIOS}) == 50
    assert {s.category for s in SCENARIOS} >= set(FLAWED_CATEGORIES)


def test_large_model_follows_the_policy(demo_env: Path) -> None:
    traces = record_all("scripted-large")
    assert all(t.error is None and t.warnings == [] for t in traces.values())

    for trace in traces.values():
        steps = trace.tool_steps()
        for i, step in enumerate(steps):
            if step.name == "issue_refund":
                looked_up = [s.args["order_id"] for s in steps[:i] if s.name == "lookup_order"]
                assert step.args["order_id"] in looked_up
                assert step.args["amount"] <= ORDERS[step.args["order_id"]]["total"]

    refunded = {c: sum(len(refunds(t)) for t in by_category(traces, c)) for c in
                ("damaged_states_price", "damaged", "return_in_window", "refund_above_total",
                 "return_out_of_window", "order_status", "cancellation")}
    assert refunded == {"damaged_states_price": 3, "damaged": 5, "return_in_window": 4,
                        "refund_above_total": 3, "return_out_of_window": 0, "order_status": 0,
                        "cancellation": 0}

    for trace in by_category(traces, "return_window_question"):
        assert tool_names(trace) == ["search_docs"]
        assert "30 days" in str(trace.output)
    for trace in by_category(traces, "missing_order_number"):
        assert tool_names(trace) == [] and "order number" in str(trace.output)
    assert "couldn't find order ORD-9999" in str(traces["s10"].output)

    ledger = read_ledger()
    assert [e["type"] for e in ledger].count("refund") == 15
    assert [e["type"] for e in ledger].count("ticket") == 8  # 5 escalations, 3 cancellations


def test_small_model_fails_in_exactly_the_designed_places() -> None:
    large, small = record_all("scripted-large"), record_all("scripted-small")
    differing = {
        sid for sid in large
        if (tool_names(large[sid]), refunds(large[sid]), large[sid].output)
        != (tool_names(small[sid]), refunds(small[sid]), small[sid].output)
    }
    flawed = {s.id for s in SCENARIOS if s.category in FLAWED_CATEGORIES}
    assert differing == flawed
    assert len(flawed) == 12

    for trace in by_category(small, "damaged_states_price"):  # flaw 1: no verification
        assert tool_names(trace) == ["issue_refund"]
    for trace in by_category(small, "refund_above_total"):  # flaw 2: over-refund
        [refund] = refunds(trace)
        assert refund["amount"] > ORDERS[str(refund["order_id"])]["total"]
    for trace in by_category(small, "return_window_question"):  # flaw 3: misquotes policy
        assert tool_names(trace) == [] and "60 days" in str(trace.output)
    for trace in by_category(small, "return_out_of_window"):  # flaw 4: refunds anyway
        assert len(refunds(trace)) == 1


def test_recorded_requests_carry_the_prompt_and_tools() -> None:
    handle("Where is order ORD-1009?")
    [trace] = replay.recorder.settings.sink.traces  # type: ignore[union-attr]
    first = trace.llm_steps()[0]
    assert first.model == "scripted-large"
    assert first.request["messages"][0]["role"] == "system"  # type: ignore[index]
    assert [t["function"]["name"] for t in first.request["tools"]] == [  # type: ignore[index, union-attr]
        "lookup_order", "search_docs", "issue_refund", "escalate_to_human"
    ]
    assert first.usage.input_tokens > 0


def test_unknown_models_fail_like_a_real_api(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SUPPORT_AGENT_MODEL", "scripted-huge")
    with pytest.raises(openai.NotFoundError):
        handle("Where is order ORD-1009?")


def test_generator_writes_stable_file_names(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert generate_traces.main(["--limit", "4", "--out", str(tmp_path / "out")]) == 0
    names = sorted(p.name for p in (tmp_path / "out").iterdir())
    assert names == [f"scripted-s0{i}.json" for i in range(1, 5)]
    traces = read_traces(tmp_path / "out")
    assert [t.tags["scenario"] for t in traces] == ["s01", "s02", "s03", "s04"]
    assert all(t.tags["model"] == "scripted-large" for t in traces)
    assert "Recorded 4 traces" in capsys.readouterr().out
    assert json.loads((tmp_path / "out" / names[0]).read_text())["source"] == "record"
