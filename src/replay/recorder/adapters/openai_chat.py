from __future__ import annotations

from typing import Any

from replay.recorder.adapters import Adapter, Parsed, parse_tool_args
from replay.schema import ToolCallRequest, Usage


class OpenAIChat(Adapter):
    provider = "openai"
    api = "chat.completions"

    def parse(self, response: Any) -> Parsed:
        # Only the first choice is recorded; agents almost always use n=1.
        choice = response.choices[0] if response.choices else None
        text: str | None = None
        calls: list[ToolCallRequest] = []
        if choice is not None:
            text = choice.message.content
            for call in choice.message.tool_calls or []:
                if call.type == "function":
                    calls.append(
                        ToolCallRequest(
                            id=call.id,
                            name=call.function.name,
                            args=parse_tool_args(call.function.arguments),
                        )
                    )
        usage = Usage()
        if response.usage is not None:
            u = response.usage
            prompt_details = u.prompt_tokens_details
            completion_details = u.completion_tokens_details
            usage = Usage(
                input_tokens=u.prompt_tokens,
                output_tokens=u.completion_tokens,
                cache_read_tokens=(prompt_details.cached_tokens or 0) if prompt_details else 0,
                reasoning_tokens=(
                    (completion_details.reasoning_tokens or 0) if completion_details else 0
                ),
            )
        return Parsed(
            text=text,
            tool_calls=calls,
            stop_reason=choice.finish_reason if choice is not None else None,
            usage=usage,
        )
