"""The trace format: the contract every other module reads and writes.

A trace is one agent run: its input, its ordered steps (LLM calls and tool calls), and its output.
Recorded runs and replayed runs use the same format, so comparing runs is always trace vs trace.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue

SCHEMA_VERSION = 1

Provider = Literal["anthropic", "openai"]
Api = Literal["messages", "chat.completions", "responses"]
Resolution = Literal["executed", "exact", "normalized", "similar", "live", "blocked"]
"""How a tool step got its result: "executed" when recording; the rest when replaying."""


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ErrorInfo(_Model):
    type: str
    """Fully qualified exception class, e.g. "support_agent.tools.OrderNotFound"."""
    message: str
    args: list[JsonValue] = Field(default_factory=list)
    """Exception args when they are JSON-serializable, so replay can rebuild the exception."""


class Usage(_Model):
    """Token usage, normalized across providers.

    `input_tokens` is the total prompt size including cached tokens; `cache_read_tokens` and
    `cache_write_tokens` are the parts of it that were read from or written to a prompt cache.
    `reasoning_tokens` is the part of `output_tokens` spent on reasoning, when reported.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0


class ToolCallRequest(_Model):
    """A tool call the model asked for (not necessarily executed)."""

    id: str | None = None
    name: str
    args: dict[str, JsonValue]


class MatchInfo(_Model):
    """How a replayed tool call was matched to a recorded one."""

    baseline_index: int
    confidence: float
    arg_diff: dict[str, JsonValue] = Field(default_factory=dict)
    reused: bool = False


class _StepBase(_Model):
    index: int
    started_at: datetime
    latency_ms: float
    error: ErrorInfo | None = None


class LLMStep(_StepBase):
    kind: Literal["llm"] = "llm"
    provider: Provider
    api: Api
    model: str
    request: dict[str, JsonValue]
    response: dict[str, JsonValue] | None = None
    text: str | None = None
    tool_calls: list[ToolCallRequest] = Field(default_factory=list)
    stop_reason: str | None = None
    usage: Usage = Field(default_factory=Usage)
    attempts: int = 1
    rate_limit_wait_ms: float = 0.0
    """Time spent waiting on rate limits during replay; excluded from latency_ms."""


class ToolStep(_StepBase):
    kind: Literal["tool"] = "tool"
    name: str
    args: dict[str, JsonValue]
    result: JsonValue = None
    result_type: str = "json"
    """How to rebuild the Python value: "json", "pydantic:<module>.<Class>", "dataclass:<...>"."""
    replayable: bool = True
    """False when the result could not be stored faithfully (it was stored as its repr)."""
    resolution: Resolution = "executed"
    match: MatchInfo | None = None


Step = Annotated[LLMStep | ToolStep, Field(discriminator="kind")]


class RunInput(_Model):
    args: list[JsonValue] = Field(default_factory=list)
    kwargs: dict[str, JsonValue] = Field(default_factory=dict)


class Trace(_Model):
    schema_version: int = SCHEMA_VERSION
    run_id: str
    source: Literal["record", "replay"] = "record"
    replay_of: str | None = None
    entrypoint: str | None = None
    """Import path of the agent entry point, "package.module:function"."""
    input: RunInput = Field(default_factory=RunInput)
    output: JsonValue = None
    error: ErrorInfo | None = None
    tags: dict[str, str] = Field(default_factory=dict)
    started_at: datetime
    ended_at: datetime | None = None
    steps: list[Step] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    def llm_steps(self) -> list[LLMStep]:
        return [s for s in self.steps if isinstance(s, LLMStep)]

    def tool_steps(self) -> list[ToolStep]:
        return [s for s in self.steps if isinstance(s, ToolStep)]


class SchemaVersionError(ValueError):
    pass


Migration = Callable[[dict[str, Any]], dict[str, Any]]

MIGRATIONS: dict[int, Migration] = {}
"""MIGRATIONS[n] upgrades a trace dict from schema version n to n + 1."""


def load_trace(
    data: dict[str, Any],
    *,
    current: int = SCHEMA_VERSION,
    migrations: Mapping[int, Migration] = MIGRATIONS,
) -> Trace:
    """Validate a stored trace, upgrading it from an older schema version if needed."""
    version = data.get("schema_version", 1)
    if not isinstance(version, int) or version < 1:
        raise SchemaVersionError(f"Trace has an invalid schema_version: {version!r}")
    if version > current:
        raise SchemaVersionError(
            f"Trace {data.get('run_id', '?')} uses schema version {version}, but this version of "
            f"Replay only understands versions up to {current}. Upgrade Replay to read it."
        )
    for step in range(version, current):
        data = migrations[step](data)
    return Trace.model_validate(data)
