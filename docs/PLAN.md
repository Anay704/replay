# Replay: milestone plan

*Status: for approval, 2026-09-25. See [DESIGN.md](DESIGN.md) for what each piece does and why.*

## How I'll work

- **One milestone at a time.** A milestone is done only when all of these are true:
  - its tests pass (`uv run pytest`);
  - lint and type checks are clean (`ruff`, `mypy`);
  - it's committed as small, logical commits (no co-author trailers).
- **All tests run offline.** They use fake HTTP transports that serve provider-format responses through the real `openai` and `anthropic` SDKs. They cost $0 and give the same result every run.
- **Checkpoints:**
  - I stop after **M4** (replay engine) for your review.
  - I stop again before anything is pushed to GitHub, or any PR is opened or commented on.
- **Budget: $0.** No paid API calls. Your Anthropic key is not used anywhere.

## Setup

- **Location:** `~/Desktop/claude/replay`, as its own git repo.
- **Python:** 3.12 via uv (`requires-python >= 3.11`).
- **Virtualenv:** lives outside iCloud, at `~/.venvs/replay`, with `.venv` as a symlink to it. iCloud hides `.pth` files, which breaks editable installs (the same problem hit agent-arena).
- **Layout:**

```
replay/
  src/replay/
    __init__.py            public API: wrap, agent, tool, assertion, fail
    schema.py
    recorder/              context.py, decorators.py, wrap.py
      adapters/            anthropic.py, openai_chat.py, openai_responses.py
    store.py
    engine/                config.py, runner.py, matcher.py, policy.py, limits.py, netguard.py
    compare/               structural.py, assertions.py, judge.py, metrics.py, pricing.py, verdict.py
    report/                render.py, templates/
    cli.py
  examples/support_agent/  data.py, tools.py, agent.py, scripted.py, scenarios.py,
                           generate_traces.py, assertions.yaml, assertions.py, configs/, traces/
  tests/
  docs/
  .github/workflows/replay.yml
```

## M1. Trace schema and recorder

**Build**
- Project scaffold: uv, src layout, ruff, mypy, pytest.
- `schema.py`: Trace, LLMStep, ToolStep, with `schema_version = 1` and lossless JSON round trips.
- Recorder:
  - the run context (contextvars);
  - `@replay.agent` and `@replay.tool`, for sync and async functions;
  - `replay.wrap()` for Anthropic `messages.create`, OpenAI `chat.completions.create` and OpenAI `responses.create`, each on sync and async clients. `with_options()` returns a wrapped client too.
- Fail-open error handling everywhere in the recorder.
- An in-memory sink for traces. The real store arrives in M2.

**Tests**
- Each API with sync and async clients records the model, request, requested tool calls, text, usage and latency correctly.
- The tool decorator:
  - binds positional and keyword args to parameter names;
  - captures results and exceptions;
  - handles Pydantic and dataclass results.
- Concurrent runs in threads and in asyncio tasks never mix steps.
- A nested agent joins the outer run, and calls made outside any run pass through.
- A forced recorder failure never reaches agent code.
- The uninstrumented-tool warning fires.

**Done when:** a toy agent with the three added lines produces a correct trace on all three APIs.

## M2. Storage

**Build**
- SQLite store:
  - WAL mode;
  - `traces` and `sessions` tables;
  - listing and filtering by tag, date, source and entry point.
- The recorder writes each trace when its run ends. `REPLAY_RECORD` and `REPLAY_SAMPLE_RATE` are honored.
- JSON export and import: one file per trace, with stable formatting.
- Schema version check, plus a migration hook.

**Tests**
- Round trips through the database and through JSON are lossless.
- Concurrent writers work, both threads and two processes.
- Traces with a future schema version are rejected with a clear message.
- Export output is byte-identical across repeated exports.

## M3. Demo agent and trace generator

**Build**
- Fake orders and policy documents.
- The four tools. The real `issue_refund` writes to a ledger file.
- The Chat Completions agent loop.
- Scripted "large" and "small" models served through a fake transport.
- The 50 scenarios.
- `generate_traces.py`: scripted by default, or `--live groq` with pacing.
- The rate limiter (`engine/limits.py`), built here because the live generator needs it.
- A committed golden set of 50 scripted traces.

**Tests**
- Under scripted-large, the agent handles every scenario correctly.
- The generator emits 50 valid traces, deterministically.
- The ledger is written in record mode.
- The limiter paces requests and tokens, and honors `retry-after` on 429s. These tests use a fake clock.

**Done when:** generating the 50 scripted traces works offline in seconds.

## M4. Replay engine and tool mocking (checkpoint)

**Build**
- Config model and YAML loading.
- The runner:
  - thread pool or asyncio;
  - a context per run;
  - importing the entry point and injecting `config=`;
  - the per-run LLM call limit.
- Overrides applied in the wrapper: model, params, system prompt.
- The three-tier matcher.
- The safety policy:
  - `allow_live`;
  - block by default;
  - `raise` or `abort` on blocked calls;
  - fail-closed without context;
  - uninstrumented-tool detection.
- Reconstruction of recorded errors and result types.
- The network guard.
- Replay-mode retries, with wait time kept out of latency.
- Storing candidate traces and the session.

**Tests**
- Table-driven matcher tests:
  - each tier;
  - the hard rules for identifiers and booleans;
  - duplicate calls served in recorded order, and reuse of the last result;
  - volatile fields;
  - threshold edges.
- Policy and fail-closed tests.
- A network-guard test.
- End to end: record the 50 scripted-large traces, then replay under scripted-small. The candidates show the expected mocked, blocked and diverging steps, and the ledger is unchanged.

**Measured:** wall time for replaying the 50 scripted traces. The target is under 2 minutes.

**Then I stop for your review.**

## M5. Diff and scoring

**Build**
- Structural diff: alignment and the materiality rules.
- Assertions: the YAML built-ins, plus Python assertions with `RunView` helpers.
- The judge:
  - talks to any OpenAI-compatible endpoint;
  - uses the rubric;
  - swaps answer order;
  - skips identical answers;
  - parses JSON, with one retry.
- Pricing table, including Groq list prices, plus cost and latency metrics.
- The verdict rules and the session summary.

**Tests**
- Every diff case.
- Every assertion type in each of its states: passed, failed and pre-existing.
- The judge, against faked responses:
  - agreement;
  - disagreement, which becomes uncertain;
  - garbled JSON;
  - the identical-answer skip.
- Verdict precedence.
- **End-to-end regression test:** record, replay, and detect the four regression kinds with exact expected counts. Replaying under the unchanged scripted model must report zero regressions.

## M6. CLI

**Build**
- The commands: `init`, `run`, `report`, `export`, `import`, `list`.
- Exit codes 0, 1, 2 and 3.
- `fail_on` thresholds.
- A summary table printed to the console.

**Tests**
- Typer's `CliRunner` for each command.
- Each exit code.
- A full end-to-end run through the CLI in a temp directory.

## M7. HTML report

**Build**
- A Jinja2 template with:
  - the summary;
  - the list of regressions;
  - a side-by-side aligned view that highlights the first divergence;
  - argument diffs field by field;
  - collapsed step details.
- The Markdown PR summary.

**Tests**
- Every number in the report matches the session summary.
- Every regressed run appears in the report.
- The first divergence is marked.
- Hostile content such as `<script>` is escaped.

**Checked visually** in the browser pane, at desktop and phone widths.

## M8. GitHub Action, live demo, README

**Build**
- `.github/workflows/replay.yml`. It:
  - runs in scripted mode;
  - posts a PR comment and updates the same comment on later pushes;
  - uploads the HTML report as an artifact;
  - runs a test job.
- The live run on Groq:
  1. a 2-trace smoke test;
  2. record about 20 baselines on gpt-oss-120b;
  3. the noise-floor replay;
  4. the candidate replay on gpt-oss-20b, judged.

  Measured numbers are saved in `docs/results/`.
- The README:
  - a 60-second quickstart;
  - a GIF-friendly walkthrough;
  - the measured results for both modes;
  - the limitations.

**Then I stop and ask** before:
- pushing to github.com/Anay704/replay;
- opening one test PR so the Action can be seen commenting.

## What I need from you

1. **Now:** approval, or changes, on DESIGN.md and this plan, including the decisions listed below.
2. **By M8:** a free Groq API key. I can't create accounts, so you'd sign up at console.groq.com yourself and add `GROQ_API_KEY=...` to `replay/.env`, which is gitignored. Nothing before M8 needs it.
3. **At M8:** your OK before anything is pushed or posted to GitHub.

## Decisions to confirm

| Decision | Proposed | Why |
|---|---|---|
| "Replay 50 traces in under 2 minutes" | Measured in scripted mode. The live run's time is reported separately. | On Groq's free tier, 8K tokens per minute per model means about 9 minutes for 20 traces, whatever Replay does. |
| Size of the live golden set | About 20 traces. The scripted set has 50. | This keeps recording, the noise-floor replay, the candidate replay and judging within one day of free quota. |
| "At most 3 lines to add recording" | Three lines, plus one decorator per tool. | Replay has to know which functions are tools. A decorator is the least intrusive way to say so. |
| Changed side effects | Counted as regressions by default (`strict_side_effects`). | In a CI gate, a false alarm is cheap and a missed wrong refund is expensive. |
| Judge | `qwen/qwen3.8-27b` on Groq, a different family from the gpt-oss models under test. Off in scripted mode. | This reduces the judge's bias toward its own model's style, and it has its own quota. |

## Risks

- **gpt-oss-20b may not regress much live.** I'll report whatever happens. If it produces fewer than 3 kinds of regression, I'd propose a second realistic change, such as the system prompt trimmed to save tokens, and ask before running it.
- **Free-tier models and limits change.** The smoke test checks them before the live run, and model IDs live in config.
- **qwen3.8-27b as a judge is unproven.** I'll check it on a few known pairs first. The fallback is another free Groq model from a family other than gpt-oss.
