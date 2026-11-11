"""Record the demo scenarios as traces, the way production traffic would be recorded.

    uv run python -m support_agent.generate_traces                      # scripted, 50 traces
    uv run python -m support_agent.generate_traces --backend groq --limit 20

Run from the repository root with `examples` on the path (the uv project does this for you via
PYTHONPATH in the commands above, or set PYTHONPATH=examples).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import replay
from replay.store import export_traces
from support_agent.agent import DEFAULT_MODELS, handle
from support_agent.scenarios import SCENARIOS, Scenario

HERE = Path(__file__).parent


def record(
    scenarios: list[Scenario], *, backend: str, model: str, concurrency: int
) -> list[replay.Trace]:
    """Run every scenario through the agent and return one trace per scenario, in order."""
    os.environ["SUPPORT_AGENT_BACKEND"] = backend
    os.environ["SUPPORT_AGENT_MODEL"] = model
    sink = replay.MemorySink()
    replay.configure(enabled=True, sample_rate=1.0, sink=sink)
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        list(pool.map(lambda s: handle(s.message), scenarios))

    by_message = {t.input.args[0]: t for t in sink.traces}
    traces = []
    for scenario in scenarios:
        trace = by_message[scenario.message]
        trace.run_id = f"{backend}-{scenario.id}"
        trace.tags.update(
            {"scenario": scenario.id, "category": scenario.category, "model": model}
        )
        traces.append(trace)
    return traces


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--backend", choices=["scripted", "groq"], default="scripted")
    parser.add_argument("--model", help="defaults to scripted-large / openai/gpt-oss-120b")
    parser.add_argument("--limit", type=int, help="record only the first N scenarios")
    parser.add_argument("--concurrency", type=int, help="parallel runs (default 8, groq 2)")
    parser.add_argument("--out", type=Path, help="output directory for the trace files")
    args = parser.parse_args(argv)

    model = args.model or DEFAULT_MODELS[args.backend]
    out = args.out or HERE / "traces" / args.backend
    concurrency = args.concurrency or (8 if args.backend == "scripted" else 2)
    scenarios = SCENARIOS[: args.limit] if args.limit else SCENARIOS

    started = time.perf_counter()
    traces = record(scenarios, backend=args.backend, model=model, concurrency=concurrency)
    export_traces(traces, out)
    elapsed = time.perf_counter() - started

    failed = [t.run_id for t in traces if t.error]
    tokens = sum(s.usage.input_tokens + s.usage.output_tokens for t in traces for s in t.llm_steps())
    print(
        f"Recorded {len(traces)} traces with {model} in {elapsed:.1f}s "
        f"({tokens:,} tokens) -> {out}"
    )
    if failed:
        print(f"{len(failed)} runs raised errors: {', '.join(failed)}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
