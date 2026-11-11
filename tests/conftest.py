from __future__ import annotations

from collections.abc import Iterator

import pytest

import replay
from replay.recorder import settings


@pytest.fixture(autouse=True)
def sink() -> Iterator[replay.MemorySink]:
    """Every test records into a fresh in-memory sink, with recorder errors raised, not logged."""
    saved = (settings.enabled, settings.strict, settings.sink, dict(settings.tags))
    memory = replay.MemorySink()
    replay.configure(enabled=True, strict=True, sink=memory, tags={})
    yield memory
    settings.enabled, settings.strict, settings.sink, settings.tags = saved
