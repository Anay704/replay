"""Trace storage: a local SQLite database, plus JSON files for sharing and golden sets.

Each trace is stored whole as a JSON document, with a few indexed columns for listing and
filtering. JSON export writes one pretty-printed file per trace with sorted keys, so a golden set
checked into a repo diffs cleanly and re-exporting the same traces is byte-identical.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from collections.abc import Iterable, Iterator, Mapping
from contextlib import closing, contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from replay.schema import Trace, load_trace

DEFAULT_DB = ".replay/replay.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS traces (
    run_id          TEXT PRIMARY KEY,
    schema_version  INTEGER NOT NULL,
    source          TEXT NOT NULL,
    replay_of       TEXT,
    entrypoint      TEXT,
    started_at      TEXT NOT NULL,
    tags            TEXT NOT NULL,
    body            TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS traces_by_time ON traces (started_at);
CREATE INDEX IF NOT EXISTS traces_by_source ON traces (source, entrypoint);
"""


class TraceStore:
    """SQLite-backed trace storage. Safe to use from several threads and processes."""

    def __init__(self, path: str | os.PathLike[str] = DEFAULT_DB) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(_SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        # A short-lived connection per operation keeps this thread- and process-safe; WAL mode
        # lets readers and a writer work concurrently, and busy_timeout waits out other writers.
        with closing(sqlite3.connect(self.path, timeout=30)) as conn, conn:
            conn.execute("PRAGMA busy_timeout=30000")
            yield conn

    def put(self, trace: Trace) -> None:
        """Insert a trace, replacing any stored trace with the same run_id."""
        self.put_many([trace])

    def put_many(self, traces: Iterable[Trace]) -> int:
        rows = [
            (
                t.run_id,
                t.schema_version,
                t.source,
                t.replay_of,
                t.entrypoint,
                t.started_at.isoformat(),
                json.dumps(t.tags, sort_keys=True),
                t.model_dump_json(),
            )
            for t in traces
        ]
        with self._connect() as conn:
            conn.executemany("INSERT OR REPLACE INTO traces VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows)
        return len(rows)

    def write(self, trace: Trace) -> None:
        """Sink interface: lets the recorder write straight into the store."""
        self.put(trace)

    def get(self, run_id: str) -> Trace:
        with self._connect() as conn:
            row = conn.execute("SELECT body FROM traces WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(f"No trace with run_id {run_id!r} in {self.path}")
        return load_trace(json.loads(row[0]))

    def list(
        self,
        *,
        source: str | None = None,
        entrypoint: str | None = None,
        tags: Mapping[str, str] | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int | None = None,
    ) -> list[Trace]:
        """Traces matching every given filter, oldest first."""
        where: list[str] = []
        params: list[Any] = []
        if source is not None:
            where.append("source = ?")
            params.append(source)
        if entrypoint is not None:
            where.append("entrypoint = ?")
            params.append(entrypoint)
        for key, value in (tags or {}).items():
            where.append("json_extract(tags, ?) = ?")
            params += [f'$."{key}"', value]
        if since is not None:
            where.append("started_at >= ?")
            params.append(since.isoformat())
        if until is not None:
            where.append("started_at < ?")
            params.append(until.isoformat())
        sql = "SELECT body FROM traces"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY started_at, run_id"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [load_trace(json.loads(body)) for (body,) in rows]

    def count(self) -> int:
        with self._connect() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM traces").fetchone()[0])


_default_stores: dict[Path, TraceStore] = {}
_default_lock = threading.Lock()


def default_store() -> TraceStore:
    """The store recordings go to unless configured otherwise: $REPLAY_DB or .replay/replay.db."""
    path = Path(os.environ.get("REPLAY_DB") or DEFAULT_DB).resolve()
    with _default_lock:
        if path not in _default_stores:
            _default_stores[path] = TraceStore(path)
        return _default_stores[path]


# JSON files ----------------------------------------------------------------------------------


def trace_to_json(trace: Trace) -> str:
    data = trace.model_dump(mode="json")
    return json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def export_traces(traces: Iterable[Trace], out_dir: str | os.PathLike[str]) -> list[Path]:
    """Write one `<run_id>.json` file per trace. Returns the paths written."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for trace in traces:
        path = out / f"{trace.run_id}.json"
        path.write_text(trace_to_json(trace), encoding="utf-8")
        paths.append(path)
    return paths


def read_traces(path: str | os.PathLike[str]) -> list[Trace]:
    """Read a single trace file, or every `*.json` file in a directory (sorted by name)."""
    source = Path(path)
    files = sorted(source.glob("*.json")) if source.is_dir() else [source]
    traces = []
    for file in files:
        try:
            data = json.loads(file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{file} is not valid JSON: {exc}") from exc
        traces.append(load_trace(data))
    return traces


def import_traces(path: str | os.PathLike[str], store: TraceStore) -> int:
    """Load trace files into the store. Traces with an existing run_id are replaced."""
    return store.put_many(read_traces(path))
