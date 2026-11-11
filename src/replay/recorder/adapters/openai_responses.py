from __future__ import annotations

from typing import Any

from replay.recorder.adapters import Adapter, Parsed, parse_tool_args
from replay.schema import ToolCallRequest, Usage


class OpenAIResponses(Adapter):
    provider = "openai"
    api = "responses"

    def parse(self, response: Any) -> Parsed:
        texts: list[str] = []
        calls: list[ToolCallRequest] = []
        for item in response.output or []:
            if item.type == "function_call":
                calls.append(
                    ToolCallRequest(
                        id=item.call_id, name=item.name, args=parse_tool_args(item.arguments)
                    )
                )
            elif item.type == "message":
                texts.extend(part.text for part in item.content if part.type == "output_text")
        usage = Usage()
        if response.usage is not None:
            u = response.usage
            usage = Usage(
                input_tokens=u.input_tokens,
                output_tokens=u.output_tokens,
                cache_read_tokens=u.input_tokens_details.cached_tokens or 0,
                reasoning_tokens=u.output_tokens_details.reasoning_tokens or 0,
            )
        stop_reason = response.status
        if response.incomplete_details is not None and response.incomplete_details.reason:
            stop_reason = response.incomplete_details.reason
        return Parsed(
            text="".join(texts) or None,
            tool_calls=calls,
            stop_reason=stop_reason,
            usage=usage,
        )
