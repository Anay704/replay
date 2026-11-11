"""Match a replayed tool call to a recorded one: exact, then normalized, then similar.

Only tool steps from the same baseline trace with the same tool name are candidates. See
docs/DESIGN.md (section 8.3) for the reasoning behind each rule.
"""

from __future__ import annotations

import json
import re
import threading
import unicodedata
from dataclasses import dataclass
from typing import Any

from pydantic import JsonValue

from replay.engine.config import MatchTier, ToolsConfig
from replay.schema import ToolStep, Trace

_TIERS: tuple[MatchTier, ...] = ("exact", "normalized", "similar")
_UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)
_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_WORDS = re.compile(r"\w+")


@dataclass(frozen=True)
class Match:
    step: ToolStep
    """The recorded step whose result is served."""
    tier: MatchTier
    confidence: float
    reused: bool
    arg_diff: dict[str, JsonValue]


@dataclass(frozen=True)
class NearMiss:
    """The closest recorded call when nothing matched, for the report."""

    step: ToolStep
    confidence: float
    arg_diff: dict[str, JsonValue]


class ToolMatcher:
    """Serves recorded tool results for one baseline trace. Thread-safe."""

    def __init__(self, baseline: Trace, config: ToolsConfig) -> None:
        self._steps = [s for s in baseline.tool_steps() if s.replayable]
        self._config = config
        self._used: set[int] = set()
        self._lock = threading.Lock()

    def match(self, name: str, args: dict[str, JsonValue]) -> Match | NearMiss | None:
        """The recorded call answering this one, the closest miss, or None if no candidates."""
        candidates = [s for s in self._steps if s.name == name]
        if not candidates:
            return None
        loosest = _TIERS.index(self._config.match.get(name, "similar"))
        volatile = self._config.volatile_for(name)

        exact = [s for s in candidates if canonical(s.args) == canonical(args)]
        if exact:
            return self._take(exact, "exact", 1.0, args, volatile)

        wanted = normalize(args, volatile)
        normalized = [s for s in candidates if canonical(normalize(s.args, volatile)) == canonical(wanted)]
        if normalized and loosest >= 1:
            return self._take(normalized, "normalized", 1.0, args, volatile)

        scored = [
            (score, s)
            for s in candidates
            if (score := similarity(wanted, normalize(s.args, volatile))) is not None
        ]
        if not scored:
            best_miss = candidates[0]
            return NearMiss(best_miss, 0.0, arg_diff(best_miss.args, args, volatile))
        top = max(score for score, _ in scored)
        best = [s for score, s in scored if score == top]
        if loosest >= 2 and top >= self._config.similarity_threshold:
            return self._take(best, "similar", round(top, 3), args, volatile)
        return NearMiss(best[0], round(top, 3), arg_diff(best[0].args, args, volatile))

    def _take(
        self,
        steps: list[ToolStep],
        tier: MatchTier,
        confidence: float,
        args: dict[str, JsonValue],
        volatile: frozenset[str],
    ) -> Match:
        """Serve recorded calls in recorded order; once all are used, reuse the last one."""
        with self._lock:
            unused = [s for s in steps if s.index not in self._used]
            step = unused[0] if unused else steps[-1]
            self._used.add(step.index)
        return Match(step, tier, confidence, not unused, arg_diff(step.args, args, volatile))


def canonical(value: object) -> str:
    """Type-exact comparison key: 1, 1.0 and True all differ."""
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def normalize(value: Any, volatile: frozenset[str] = frozenset(), path: str = "") -> Any:
    """Drop volatile and null fields, collapse whitespace, and make 2.0 equal 2."""
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            here = f"{path}.{key}" if path else key
            if item is None or key in volatile or here in volatile:
                continue
            out[key] = normalize(item, volatile, here)
        return out
    if isinstance(value, list):
        return [normalize(item, volatile, path) for item in value]
    if isinstance(value, bool):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        return " ".join(unicodedata.normalize("NFC", value).split())
    return value


def flatten(value: Any, path: str = "") -> dict[str, Any]:
    """{"a": {"b": [1]}} -> {"a.b[0]": 1}. Empty containers are kept as leaves."""
    if isinstance(value, dict) and value:
        out: dict[str, Any] = {}
        for key, item in value.items():
            out.update(flatten(item, f"{path}.{key}" if path else str(key)))
        return out
    if isinstance(value, list) and value:
        out = {}
        for i, item in enumerate(value):
            out.update(flatten(item, f"{path}[{i}]"))
        return out
    return {path: value}


def is_identifier(value: str) -> bool:
    """Values like ORD-1042, UUIDs and emails: close is not good enough."""
    if any(ch.isspace() for ch in value):
        return False
    return any(ch.isdigit() for ch in value) or bool(_UUID.match(value) or _EMAIL.match(value))


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def field_similarity(a: Any, b: Any) -> float | None:
    """How alike two leaf values are, 0 to 1, or None if they must match and don't."""
    if canonical(a) == canonical(b):
        return 1.0
    if isinstance(a, bool) or isinstance(b, bool):
        return None
    if _is_number(a) and _is_number(b):
        largest = max(abs(a), abs(b))
        return max(0.0, 1.0 - abs(a - b) / largest) if largest else 1.0
    if isinstance(a, str) and isinstance(b, str):
        if is_identifier(a) or is_identifier(b):
            return None
        if not any(ch.isspace() for ch in a + b):  # short labels like "damaged"
            return 1.0 if a.casefold() == b.casefold() else 0.0
        words_a, words_b = set(_WORDS.findall(a.lower())), set(_WORDS.findall(b.lower()))
        if not words_a and not words_b:
            return 1.0
        return 2 * len(words_a & words_b) / (len(words_a) + len(words_b))
    return 0.0


def similarity(a: Any, b: Any) -> float | None:
    """Average field similarity over the union of fields; None on any must-match mismatch."""
    flat_a, flat_b = flatten(a), flatten(b)
    paths = flat_a.keys() | flat_b.keys()
    if not paths:
        return 1.0
    total = 0.0
    for path in paths:
        if path not in flat_a or path not in flat_b:
            continue  # present on one side only: scores 0
        score = field_similarity(flat_a[path], flat_b[path])
        if score is None:
            return None
        total += score
    return total / len(paths)


def arg_diff(
    recorded: dict[str, JsonValue], replayed: dict[str, JsonValue], volatile: frozenset[str]
) -> dict[str, JsonValue]:
    """Fields that differ between the recorded and replayed arguments."""
    flat_rec, flat_new = flatten(recorded), flatten(replayed)
    diff: dict[str, JsonValue] = {}
    for path in sorted(flat_rec.keys() | flat_new.keys()):
        if path.split(".")[-1].split("[")[0] in volatile or path in volatile:
            continue
        before, after = flat_rec.get(path), flat_new.get(path)
        if canonical(before) != canonical(after):
            diff[path] = {"recorded": before, "replayed": after}
    return diff
