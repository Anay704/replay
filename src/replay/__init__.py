"""Replay: regression testing for AI agents through record and replay.

    import replay
    client = replay.wrap(OpenAI())

    @replay.agent
    def handle_ticket(message: str) -> str: ...

    @replay.tool
    def lookup_order(order_id: str) -> dict: ...
"""

from replay.recorder import MemorySink, Sink, agent, configure, current_run, tool, wrap
from replay.schema import LLMStep, ToolStep, Trace

__version__ = "0.1.0"

__all__ = [
    "LLMStep",
    "MemorySink",
    "Sink",
    "ToolStep",
    "Trace",
    "__version__",
    "agent",
    "configure",
    "current_run",
    "tool",
    "wrap",
]
