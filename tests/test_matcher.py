from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from replay.engine.config import ToolsConfig
from replay.engine.matcher import (
    Match,
    NearMiss,
    ToolMatcher,
    field_similarity,
    is_identifier,
    normalize,
    similarity,
)
from replay.schema import ToolStep, Trace

T0 = datetime(2026, 9, 25, tzinfo=UTC)


def baseline(*calls: tuple[str, dict[str, Any], Any], replayable: bool = True) -> Trace:
    steps = [
        ToolStep(
            index=i, started_at=T0, latency_ms=5.0, name=name, args=args, result=result,
            replayable=replayable,
        )
        for i, (name, args, result) in enumerate(calls)
    ]
    return Trace(run_id="b", started_at=T0, steps=steps)


REFUND = {"order_id": "ORD-1024", "amount": 64.99, "reason": "damaged item"}


@pytest.mark.parametrize(
    ("replayed", "tier"),
    [
        ({"order_id": "ORD-1024", "amount": 64.99, "reason": "damaged item"}, "exact"),
        ({"reason": "damaged item", "amount": 64.99, "order_id": "ORD-1024"}, "exact"),  # key order
        ({"order_id": "ORD-1024", "amount": 64.99, "reason": "  damaged   item "}, "normalized"),
        ({"order_id": "ORD-1024", "amount": 64.99, "reason": "damaged item", "note": None},
         "normalized"),  # null means absent
        ({"order_id": "ORD-1024", "amount": 60.0, "reason": "damaged item"}, "similar"),
        ({"order_id": "ORD-1024", "amount": 64.99, "reason": "item arrived damaged"}, "similar"),
    ],
)
def test_tiers(replayed: dict[str, Any], tier: str) -> None:
    matcher = ToolMatcher(baseline(("issue_refund", REFUND, {"refund_id": "R1"})), ToolsConfig())
    found = matcher.match("issue_refund", replayed)
    assert isinstance(found, Match)
    assert found.tier == tier
    assert found.step.result == {"refund_id": "R1"}
    assert (found.confidence == 1.0) == (tier != "similar")


@pytest.mark.parametrize(
    "replayed",
    [
        {"order_id": "ORD-1025", "amount": 64.99, "reason": "damaged item"},  # different order
        {"order_id": "ord-1024", "amount": 64.99, "reason": "damaged item"},  # ids are exact
        {"order_id": "ORD-1024", "amount": 200.0, "reason": "damaged item"},  # too far off
        {"order_id": "ORD-1024"},  # most fields missing
    ],
)
def test_near_misses(replayed: dict[str, Any]) -> None:
    matcher = ToolMatcher(baseline(("issue_refund", REFUND, "ok")), ToolsConfig())
    found = matcher.match("issue_refund", replayed)
    assert isinstance(found, NearMiss)
    assert found.confidence < 0.8


def test_near_miss_reports_the_differences() -> None:
    matcher = ToolMatcher(baseline(("issue_refund", REFUND, "ok")), ToolsConfig())
    found = matcher.match("issue_refund", {**REFUND, "amount": 200.0})
    assert isinstance(found, NearMiss)
    assert found.arg_diff == {"amount": {"recorded": 64.99, "replayed": 200.0}}
    assert found.confidence == pytest.approx((1 + 64.99 / 200 + 1) / 3, abs=0.001)


def test_other_tools_and_missing_calls_are_never_candidates() -> None:
    matcher = ToolMatcher(baseline(("lookup_order", {"order_id": "A1"}, "x")), ToolsConfig())
    assert matcher.match("issue_refund", {"order_id": "A1"}) is None


def test_type_exact_comparison() -> None:
    matcher = ToolMatcher(baseline(("t", {"n": 2, "flag": True}, "x")), ToolsConfig())
    exact = matcher.match("t", {"n": 2, "flag": True})
    assert isinstance(exact, Match) and exact.tier == "exact"
    assert isinstance(m := matcher.match("t", {"n": 2.0, "flag": True}), Match)
    assert m.tier == "normalized"
    # true is never "similar" to 1 or to false
    assert not isinstance(matcher.match("t", {"n": 2, "flag": 1}), Match)
    assert not isinstance(matcher.match("t", {"n": 2, "flag": False}), Match)


def test_volatile_fields() -> None:
    recorded = {"order_id": "A1", "request_id": "r-1", "meta": {"ts": "09:00", "v": 1}}
    replayed = {"order_id": "A1", "request_id": "r-2", "meta": {"ts": "10:30", "v": 1}}
    step = ("lookup_order", recorded, "x")
    assert not isinstance(ToolMatcher(baseline(step), ToolsConfig()).match("lookup_order", replayed), Match)

    config = ToolsConfig(volatile_fields={"*": ["request_id"], "lookup_order": ["meta.ts"]})
    found = ToolMatcher(baseline(step), config).match("lookup_order", replayed)
    assert isinstance(found, Match) and found.tier == "normalized"
    assert found.arg_diff == {}


def test_duplicate_calls_are_served_in_recorded_order_then_reused() -> None:
    matcher = ToolMatcher(
        baseline(
            ("get_status", {"id": "J1"}, "queued"),
            ("get_status", {"id": "J1"}, "running"),
            ("get_status", {"id": "J1"}, "done"),
        ),
        ToolsConfig(),
    )
    served = [matcher.match("get_status", {"id": "J1"}) for _ in range(4)]
    assert [m.step.result for m in served if isinstance(m, Match)] == [
        "queued", "running", "done", "done"
    ]
    assert [m.reused for m in served if isinstance(m, Match)] == [False, False, False, True]


def test_similarity_ties_prefer_unused_calls() -> None:
    matcher = ToolMatcher(
        baseline(("search", {"q": "refund policy"}, "first"), ("search", {"q": "refund policy"}, "second")),
        ToolsConfig(),
    )
    results = [matcher.match("search", {"q": "the refund policy"}) for _ in range(2)]
    assert [m.step.result for m in results if isinstance(m, Match)] == ["first", "second"]


def test_per_tool_tier_limits_and_threshold() -> None:
    step = ("issue_refund", REFUND, "ok")
    exact_only = ToolsConfig(match={"issue_refund": "exact"})
    loose = {**REFUND, "reason": " damaged item"}
    assert isinstance(ToolMatcher(baseline(step), exact_only).match("issue_refund", loose), NearMiss)

    close = {**REFUND, "amount": 60.0}
    strict = ToolsConfig(similarity_threshold=0.99)
    assert isinstance(ToolMatcher(baseline(step), strict).match("issue_refund", close), NearMiss)


def test_unreplayable_recordings_are_never_served() -> None:
    matcher = ToolMatcher(baseline(("t", {"a": 1}, "<obj>"), replayable=False), ToolsConfig())
    assert matcher.match("t", {"a": 1}) is None


@pytest.mark.parametrize(
    ("value", "expected"),
    [("ORD-1042", True), ("a@b.com", True), ("123e4567-e89b-12d3-a456-426614174000", True),
     ("damaged", False), ("two words 1", False), ("A1", True)],
)
def test_is_identifier(value: str, expected: bool) -> None:
    assert is_identifier(value) is expected


def test_field_similarity() -> None:
    assert field_similarity("Damaged", "damaged") == 1.0  # labels ignore case
    assert field_similarity("damaged", "broken") == 0.0
    assert field_similarity(50, 100) == 0.5
    assert field_similarity(0, 0.0) == 1.0
    assert field_similarity("ORD-1", "ORD-2") is None
    assert field_similarity(True, False) is None
    assert field_similarity("refund policy for items", "items refund policy") == pytest.approx(
        2 * 3 / 7
    )


def test_normalize_and_similarity() -> None:
    assert normalize({"a": 2.0, "b": " x  y ", "c": None, "d": [1.0, {"e": None}]}) == {
        "a": 2, "b": "x y", "d": [1, {}]
    }
    assert similarity({"a": 1}, {"a": 1}) == 1.0
    assert similarity({"a": 1, "b": "x"}, {"a": 1}) == 0.5  # b is missing on one side
    assert similarity({}, {}) == 1.0
