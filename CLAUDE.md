# replay

Moved from the workspace `~/Desktop/claude/CLAUDE.md` so it loads only when working in this project.

`uv sync`, then `uv run pytest`, `uv run ruff check .`, `uv run mypy` (strict). `.venv` is a symlink to `~/.venvs/replay` (iCloud hides `.pth` files). Tests are fully offline: fake provider endpoints go through the real SDKs via an `httpx2.MockTransport`. Budget is $0: no paid API calls; live runs use Groq's free tier only. Design and milestones are in `docs/`.
