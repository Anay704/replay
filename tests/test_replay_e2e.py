"""End to end on the demo: replay the committed golden set (recorded on scripted-large) under
scripted-small and under an unchanged config, with the demo's real side-effect ledger watched."""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from replay.engine import load_config, replay_traces
from replay.schema import RunResult, Session
from replay.store import read_traces
from support_agent.scripted import SORRY

ROOT = Path(__file__).parent.parent
GOLDEN = ROOT / "examples/support_agent/traces/scripted"
CONFIGS = ROOT / "examples/support_agent/configs"


@pytest.fixture(autouse=True)
def ledger(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    path = tmp_path / "ledger.jsonl"
    monkeypatch.setenv("SUPPORT_AGENT_LEDGER", str(path))
    monkeypatch.setenv("SUPPORT_AGENT_BACKEND", "scripted")
    monkeypatch.delenv("SUPPORT_AGENT_MODEL", raising=False)
    yield path
    assert not path.exists(), "a replay performed a real side effect"


def run(config_name: str) -> Session:
    return replay_traces(read_traces(GOLDEN), load_config(CONFIGS / config_name))


def by_scenario(session: Session) -> dict[str, RunResult]:
    return {r.baseline.tags["scenario"]: r for r in session.results}


def tools(result: RunResult) -> list[tuple[str, str]]:
    assert result.candidate is not None
    return [(s.name, s.resolution) for s in result.candidate.tool_steps()]


def test_golden_set_is_the_50_scripted_scenarios() -> None:
    traces = read_traces(GOLDEN)
    assert len(traces) == 50
    assert {t.tags["model"] for t in traces} == {"scripted-large"}
    assert all(t.error is None and t.warnings == [] for t in traces)


def test_small_model_replay_diverges_exactly_where_designed() -> None:
    results = by_scenario(run("scripted-small.yaml"))
    assert len(results) == 50
    assert all(r.candidate is not None and r.candidate.error is None for r in results.values())

    # Flaw 1: refunds straight away; the identical refund is served from the recording.
    for sid in ("s19", "s20", "s21"):
        assert tools(results[sid]) == [("issue_refund", "exact")]
    # Flaw 2: over-refunds. Close amounts match by similarity, far ones are blocked.
    assert tools(results["s31"])[-1] == ("issue_refund", "blocked")  # $80 on a $28.50 order
    assert tools(results["s32"])[-1] == ("issue_refund", "similar")  # $100 on $42.75
    assert tools(results["s33"])[-1] == ("issue_refund", "similar")  # $120 on $64.99
    similar = results["s33"].candidate.tool_steps()[-1]  # type: ignore[union-attr]
    assert similar.match is not None and similar.match.confidence == pytest.approx(0.847, abs=0.001)
    assert similar.match.arg_diff == {"amount": {"recorded": 64.99, "replayed": 120.0}}
    # Flaw 3: answers from memory, no tools at all.
    for sid in ("s11", "s12", "s13"):
        assert tools(results[sid]) == []
        assert "60 days" in str(results[sid].candidate.output)  # type: ignore[union-attr]
    # Flaw 4: tries to refund outside the window; blocked, and the agent apologizes.
    for sid in ("s34", "s35", "s36"):
        assert tools(results[sid]) == [
            ("lookup_order", "exact"), ("search_docs", "exact"), ("issue_refund", "blocked")
        ]
        assert results[sid].candidate.output == SORRY  # type: ignore[union-attr]

    flawed = {f"s{n}" for n in (11, 12, 13, 19, 20, 21, 31, 32, 33, 34, 35, 36)}
    for sid, result in results.items():
        if sid in flawed:
            continue
        assert result.candidate is not None
        assert result.candidate.output == result.baseline.output, sid
        assert all(resolution == "exact" for _, resolution in tools(result)), sid


def test_unchanged_config_reproduces_every_baseline() -> None:
    for result in run("scripted-control.yaml").results:
        assert result.candidate is not None
        assert result.candidate.output == result.baseline.output
        assert [s.args for s in result.candidate.tool_steps()] == [
            s.args for s in result.baseline.tool_steps()
        ]
        assert all(resolution == "exact" for _, resolution in tools(result))


def test_replaying_fifty_traces_is_fast() -> None:
    """Success criterion: 50 traces in under 2 minutes (scripted models include simulated latency)."""
    started = time.perf_counter()
    session = run("scripted-small.yaml")
    elapsed = time.perf_counter() - started
    assert len(session.results) == 50
    assert elapsed < 120
    print(f"\nReplayed 50 traces in {elapsed:.2f}s")
