"""Provider adapters: translate SDK requests and responses into provider-neutral LLM steps.

Each adapter knows one API (Anthropic Messages, OpenAI Chat Completions, OpenAI Responses).
Nothing outside this package looks at provider-specific response shapes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, ClassVar

from pydantic import JsonValue

from replay.schema import Api, LLMStep, Provider, ToolCallRequest, Usage
from replay.serialize import error_info, to_jsonable

# Request kwargs that carry transport settings (and possibly credentials), not model input.
_TRANSPORT_KWARGS = {"extra_headers", "extra_query", "extra_body", "timeout"}


@dataclass
class Parsed:
    text: str | None = None
    tool_calls: list[ToolCallRequest] = field(default_factory=list)
    stop_reason: str | None = None
    usage: Usage = field(default_factory=Usage)


class Adapter:
    provider: ClassVar[Provider]
    api: ClassVar[Api]

    def parse(self, response: Any) -> Parsed:
        raise NotImplementedError

    def build_step(
        self,
        *,
        index: int,
        started_at: datetime,
        latency_ms: float,
        kwargs: dict[str, Any],
        response: Any = None,
        error: BaseException | None = None,
    ) -> LLMStep:
        request = request_json(kwargs)
        model = str(kwargs.get("model") or getattr(response, "model", "") or "")
        if response is None:
            return LLMStep(
                index=index,
                started_at=started_at,
                latency_ms=latency_ms,
                provider=self.provider,
                api=self.api,
                model=model,
                request=request,
                error=error_info(error) if error is not None else None,
            )
        parsed = self.parse(response)
        return LLMStep(
            index=index,
            started_at=started_at,
            latency_ms=latency_ms,
            provider=self.provider,
            api=self.api,
            model=model,
            request=request,
            response=response_json(response),
            text=parsed.text,
            tool_calls=parsed.tool_calls,
            stop_reason=parsed.stop_reason,
            usage=parsed.usage,
        )


def _is_omitted(value: object) -> bool:
    # The SDKs mark unset parameters with sentinel objects (Omit / NotGiven).
    return type(value).__name__ in {"Omit", "NotGiven"}


def request_json(kwargs: dict[str, Any]) -> dict[str, JsonValue]:
    return {
        key: to_jsonable(value).value
        for key, value in kwargs.items()
        if key not in _TRANSPORT_KWARGS and not _is_omitted(value)
    }


def response_json(response: Any) -> dict[str, JsonValue] | None:
    data = to_jsonable(response).value
    return data if isinstance(data, dict) else {"value": data}


def parse_tool_args(raw: object) -> dict[str, JsonValue]:
    """Tool arguments arrive as a dict (Anthropic) or a JSON string (OpenAI)."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw) if raw.strip() else {}
        except json.JSONDecodeError:
            return {"_raw": raw}
    data = to_jsonable(raw).value
    return data if isinstance(data, dict) else {"_value": data}
