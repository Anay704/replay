from __future__ import annotations

from typing import Any

from replay.recorder.adapters import Adapter, Parsed, parse_tool_args
from replay.schema import ToolCallRequest, Usage


class AnthropicMessages(Adapter):
    provider = "anthropic"
    api = "messages"

    def parse(self, response: Any) -> Parsed:
        texts: list[str] = []
        calls: list[ToolCallRequest] = []
        for block in response.content or []:
            kind = getattr(block, "type", None)
            if kind == "text":
                texts.append(block.text)
            elif kind == "tool_use":
                calls.append(
                    ToolCallRequest(id=block.id, name=block.name, args=parse_tool_args(block.input))
                )
        usage = Usage()
        if response.usage is not None:
            u = response.usage
            cache_read = u.cache_read_input_tokens or 0
            cache_write = u.cache_creation_input_tokens or 0
            usage = Usage(
                input_tokens=u.input_tokens + cache_read + cache_write,
                output_tokens=u.output_tokens,
                cache_read_tokens=cache_read,
                cache_write_tokens=cache_write,
            )
        return Parsed(
            text="".join(texts) or None,
            tool_calls=calls,
            stop_reason=response.stop_reason,
            usage=usage,
        )
