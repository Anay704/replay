from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

import replay
from replay.recorder import settings
from replay.recorder.context import Settings
from replay.schema import SCHEMA_VERSION, SchemaVersionError, ToolStep, Trace, load_trace
from replay.store import TraceStore, export_traces, import_traces, read_traces, trace_to_json

T0 = datetime(2026, 9, 25, 12, 0, tzinfo=UTC)


def make_trace(i: int, *, source: str = "record", tags: dict[str, str] | None = None) -> Trace:
    return Trace(
        run_id=f"run-{i:03d}",
        source=source,  # type: ignore[arg-type]
        entrypoint="agent:run",
        started_at=T0 + timedelta(minutes=i),
        ended_at=T0 + timedelta(minutes=i, seconds=3),
        input={"args": [f"message {i}"], "kwargs": {}},  # type: ignore[arg-type]
        output=f"answer {i} — ünïcode",
        tags=tags or {},
        steps=[
            ToolStep(
                index=0,
                started_at=T0,
                latency_ms=1.5,
                name="lookup_order",
                args={"order_id": f"A{i}", "include": ["items"]},
                result={"total": 10.5 + i},
            )
        ],
    )


@pytest.fixture
def store(tmp_path: Path) -> TraceStore:
    return TraceStore(tmp_path / "replay.db")


def test_put_and_get_round_trip(store: TraceStore) -> None:
    trace = make_trace(1)
    store.put(trace)
    assert store.get("run-001") == trace
    assert store.count() == 1
    with pytest.raises(KeyError, match="run-999"):
        store.get("run-999")


def test_putting_the_same_run_id_replaces_it(store: TraceStore) -> None:
    store.put(make_trace(1))
    store.put(make_trace(1).model_copy(update={"output": "new"}))
    assert store.count() == 1
    assert store.get("run-001").output == "new"


def test_list_filters(store: TraceStore) -> None:
    store.put_many(
        [
            make_trace(1, tags={"env": "prod", "prompt v": "1"}),
            make_trace(2, tags={"env": "staging"}),
            make_trace(3, tags={"env": "prod", "prompt v": "2"}),
            make_trace(4, source="replay"),
        ]
    )

    def ids(traces: list[Trace]) -> list[str]:
        return [t.run_id for t in traces]

    assert ids(store.list()) == ["run-001", "run-002", "run-003", "run-004"]
    assert ids(store.list(source="replay")) == ["run-004"]
    assert ids(store.list(tags={"env": "prod"})) == ["run-001", "run-003"]
    assert ids(store.list(tags={"env": "prod", "prompt v": "2"})) == ["run-003"]
    assert ids(store.list(since=T0 + timedelta(minutes=2))) == ["run-002", "run-003", "run-004"]
    assert ids(store.list(until=T0 + timedelta(minutes=2))) == ["run-001"]
    assert ids(store.list(entrypoint="other:run")) == []
    assert ids(store.list(limit=2)) == ["run-001", "run-002"]


def test_json_export_import_round_trip(store: TraceStore, tmp_path: Path) -> None:
    traces = [make_trace(i) for i in range(3)]
    paths = export_traces(traces, tmp_path / "golden")
    assert [p.name for p in paths] == ["run-000.json", "run-001.json", "run-002.json"]
    assert read_traces(tmp_path / "golden") == traces
    assert read_traces(paths[1]) == [traces[1]]

    assert import_traces(tmp_path / "golden", store) == 3
    assert store.list() == traces


def test_export_is_byte_stable_and_readable(tmp_path: Path) -> None:
    trace = make_trace(7)
    first = export_traces([trace], tmp_path / "a")[0].read_bytes()
    reloaded = read_traces(tmp_path / "a")[0]
    second = export_traces([reloaded], tmp_path / "b")[0].read_bytes()
    assert first == second
    text = first.decode()
    assert "ünïcode" in text  # not \\u-escaped, so diffs stay readable
    assert text.startswith('{\n  "ended_at"')  # sorted keys, indented
    assert text.endswith("}\n")


def test_invalid_json_file_names_the_file(tmp_path: Path) -> None:
    bad = tmp_path / "broken.json"
    bad.write_text("{not json")
    with pytest.raises(ValueError, match="broken.json"):
        read_traces(tmp_path)


def test_future_schema_versions_are_rejected(store: TraceStore, tmp_path: Path) -> None:
    data = json.loads(trace_to_json(make_trace(1)))
    data["schema_version"] = SCHEMA_VERSION + 1
    (tmp_path / "future.json").write_text(json.dumps(data))
    with pytest.raises(SchemaVersionError, match="Upgrade Replay"):
        read_traces(tmp_path / "future.json")


def test_older_versions_are_migrated() -> None:
    legacy = json.loads(trace_to_json(make_trace(1)))
    legacy["answer"] = legacy.pop("output")  # pretend version 1 called the output "answer"

    def upgrade(data: dict[str, Any]) -> dict[str, Any]:
        data = dict(data)
        data["output"] = data.pop("answer")
        data["schema_version"] = 2
        return data

    trace = load_trace(legacy, current=2, migrations={1: upgrade})
    assert trace.output == make_trace(1).output
    assert trace.schema_version == 2


def test_concurrent_writers_in_threads(store: TraceStore) -> None:
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(store.put, [make_trace(i) for i in range(40)]))
    assert store.count() == 40


def test_concurrent_writers_in_processes(tmp_path: Path) -> None:
    db = tmp_path / "shared.db"
    TraceStore(db)
    script = textwrap.dedent(
        """
        import sys
        from tests.test_store import make_trace
        from replay.store import TraceStore
        store = TraceStore(sys.argv[1])
        offset = int(sys.argv[2])
        for i in range(25):
            store.put(make_trace(offset + i))
        """
    )
    procs = [
        subprocess.Popen([sys.executable, "-c", script, str(db), str(offset)], cwd=Path.cwd())
        for offset in (0, 100)
    ]
    assert [p.wait(timeout=60) for p in procs] == [0, 0]
    assert TraceStore(db).count() == 50


# The recorder writing to the store ------------------------------------------------------------


@replay.agent
def greet(name: str) -> str:
    return f"hi {name}"


def test_recorder_writes_to_the_default_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("REPLAY_DB", str(tmp_path / "rec.db"))
    settings.sink = None  # the default: the store at $REPLAY_DB
    greet("ana")
    greet("bo")
    [first, second] = TraceStore(tmp_path / "rec.db").list()
    assert (first.output, second.output) == ("hi ana", "hi bo")
    assert first.entrypoint == "tests.test_store:greet"


def test_sample_rate(sink: replay.MemorySink) -> None:
    replay.configure(sample_rate=0.0)
    for _ in range(20):
        greet("x")
    assert sink.traces == []
    replay.configure(sample_rate=1.0)
    greet("x")
    assert len(sink.traces) == 1
    with pytest.raises(ValueError):
        replay.configure(sample_rate=1.5)


@pytest.mark.parametrize(
    ("env", "enabled", "rate"),
    [
        ({}, True, 1.0),
        ({"REPLAY_RECORD": "0"}, False, 1.0),
        ({"REPLAY_RECORD": "off", "REPLAY_SAMPLE_RATE": "0.25"}, False, 0.25),
        ({"REPLAY_SAMPLE_RATE": "0.1"}, True, 0.1),
        ({"REPLAY_SAMPLE_RATE": "lots"}, True, 1.0),
        ({"REPLAY_SAMPLE_RATE": "2"}, True, 1.0),
    ],
)
def test_settings_from_env(
    monkeypatch: pytest.MonkeyPatch, env: dict[str, str], enabled: bool, rate: float
) -> None:
    monkeypatch.delenv("REPLAY_RECORD", raising=False)
    monkeypatch.delenv("REPLAY_SAMPLE_RATE", raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    loaded = Settings.from_env()
    assert (loaded.enabled, loaded.sample_rate) == (enabled, rate)
