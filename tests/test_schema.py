from __future__ import annotations

import dataclasses
from datetime import UTC, datetime

import pytest
from pydantic import BaseModel, ValidationError

from replay.schema import SCHEMA_VERSION, LLMStep, ToolCallRequest, ToolStep, Trace, Usage
from replay.serialize import error_info, to_jsonable

T0 = datetime(2026, 9, 25, 12, 0, tzinfo=UTC)


def sample_trace() -> Trace:
    return Trace(
        run_id="r1",
        entrypoint="support_agent.agent:run",
        started_at=T0,
        ended_at=T0,
        output="done",
        steps=[
            LLMStep(
                index=0, started_at=T0, latency_ms=12.5, provider="openai",
                api="chat.completions", model="m", request={"messages": []},
                tool_calls=[ToolCallRequest(id="c1", name="lookup_order", args={"order_id": "A1"})],
                usage=Usage(input_tokens=10, output_tokens=5),
            ),
            ToolStep(index=1, started_at=T0, latency_ms=1.0, name="lookup_order",
                     args={"order_id": "A1"}, result={"total": 49.99}),
        ],
    )


def test_trace_round_trips_through_json() -> None:
    trace = sample_trace()
    restored = Trace.model_validate_json(trace.model_dump_json())
    assert restored == trace
    assert restored.schema_version == SCHEMA_VERSION
    assert isinstance(restored.steps[0], LLMStep)
    assert isinstance(restored.steps[1], ToolStep)
    assert [s.kind for s in restored.steps] == ["llm", "tool"]


def test_unknown_fields_are_rejected() -> None:
    data = sample_trace().model_dump(mode="json")
    data["steps"][1]["surprise"] = 1
    with pytest.raises(ValidationError):
        Trace.model_validate(data)


def test_step_helpers() -> None:
    trace = sample_trace()
    assert [s.model for s in trace.llm_steps()] == ["m"]
    assert [s.name for s in trace.tool_steps()] == ["lookup_order"]


class Order(BaseModel):
    order_id: str
    total: float


@dataclasses.dataclass
class Receipt:
    refund_id: str
    amount: float


def test_to_jsonable_remembers_types() -> None:
    assert to_jsonable({"a": (1, 2)}) == ({"a": [1, 2]}, "json", True)
    assert to_jsonable(T0).value == "2026-09-25T12:00:00Z"
    model = to_jsonable(Order(order_id="A1", total=5.0))
    assert model.value == {"order_id": "A1", "total": 5.0}
    assert model.type_tag == "pydantic:tests.test_schema.Order"
    assert to_jsonable(Receipt("R1", 2.5)).type_tag == "dataclass:tests.test_schema.Receipt"


def test_to_jsonable_falls_back_to_repr() -> None:
    class Opaque:
        def __repr__(self) -> str:
            return "<opaque>"

    assert to_jsonable(Opaque()) == ("<opaque>", "repr", False)


def test_error_info() -> None:
    info = error_info(KeyError("A9"))
    assert info.type == "builtins.KeyError"
    assert info.args == ["A9"]
    assert error_info(ValueError(object())).args == []
