"""Where finished traces go."""

from __future__ import annotations

import threading
from typing import Protocol

from replay.schema import Trace


class Sink(Protocol):
    def write(self, trace: Trace) -> None: ...


class MemorySink:
    """Keeps traces in a list. Useful in tests and notebooks."""

    def __init__(self) -> None:
        self.traces: list[Trace] = []
        self._lock = threading.Lock()

    def write(self, trace: Trace) -> None:
        with self._lock:
            self.traces.append(trace)

    def clear(self) -> None:
        with self._lock:
            self.traces.clear()
