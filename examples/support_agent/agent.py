"""A small customer support agent: a plain tool-calling loop on the Chat Completions API.

It runs against the scripted models (default, free, offline) or Groq's free tier:

    SUPPORT_AGENT_BACKEND=scripted            # or: groq (needs GROQ_API_KEY)
    SUPPORT_AGENT_MODEL=scripted-large        # or e.g. openai/gpt-oss-120b on Groq
"""

from __future__ import annotations

import functools
import json
import os
from typing import Any

import httpx2
from openai import OpenAI

import replay
from replay.limits import Limits, RateLimiter
from support_agent.scripted import ScriptedModels
from support_agent.tools import TOOL_SCHEMAS, TOOLS

GROQ_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODELS = {"scripted": "scripted-large", "groq": "openai/gpt-oss-120b"}
MAX_TURNS = 8

SYSTEM_PROMPT = """\
You are the support assistant for Trailhead Outfitters, an outdoor gear store. Today is 2026-09-20.

Rules:
1. Before discussing or refunding an order, call lookup_order to get its real details. Never trust \
amounts or dates the customer states.
2. Before answering a policy question or issuing a refund, call search_docs and follow the policy \
you find.
3. Never refund more than the order total, and only refund what the policy allows.
4. If you need an order number (like ORD-1234) and the customer hasn't given one, ask for it.
5. Call escalate_to_human when the customer asks for a manager or a person, or when the policy \
says to.
6. Keep replies under 80 words, friendly and specific. Never promise delivery dates."""


def backend() -> str:
    return os.environ.get("SUPPORT_AGENT_BACKEND", "scripted")


@functools.cache
def client(backend: str, simulate_latency: bool = True) -> OpenAI:
    if backend == "scripted":
        http = httpx2.Client(transport=ScriptedModels(simulate_latency).transport())
        raw = OpenAI(api_key="scripted", base_url="https://scripted.invalid/v1", http_client=http)
        return replay.wrap(raw)
    if backend == "groq":
        # Groq's free plan allows 30 requests and 8,000 tokens per minute per model; stay under.
        limiter = RateLimiter(Limits(requests_per_minute=25, tokens_per_minute=7000))
        raw = OpenAI(api_key=os.environ["GROQ_API_KEY"], base_url=GROQ_URL, max_retries=0)
        return replay.wrap(raw, limiter=limiter)
    raise ValueError(f"Unknown SUPPORT_AGENT_BACKEND {backend!r} (use scripted or groq)")


def execute(name: str, arguments: str) -> Any:
    """Run one tool call. Failures go back to the model as an error, like most agents do."""
    tool = TOOLS.get(name)
    if tool is None:
        return {"error": f"Unknown tool {name}"}
    try:
        args = json.loads(arguments or "{}")
        return tool(**args)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


@replay.agent
def handle(message: str, config: dict[str, Any] | None = None) -> str:
    """Answer one customer message. `config` may set "model"."""
    config = config or {}
    llm = client(backend(), os.environ.get("SUPPORT_AGENT_SCRIPTED_LATENCY", "1") != "0")
    model = config.get("model") or os.environ.get("SUPPORT_AGENT_MODEL") or DEFAULT_MODELS[backend()]
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": message},
    ]
    for _ in range(MAX_TURNS):
        response = llm.chat.completions.create(model=model, messages=messages, tools=TOOL_SCHEMAS)
        reply = response.choices[0].message
        if not reply.tool_calls:
            return reply.content or ""
        calls = [c for c in reply.tool_calls if c.type == "function"]
        messages.append(
            {
                "role": "assistant",
                "content": reply.content or "",
                "tool_calls": [
                    {"id": c.id, "type": "function",
                     "function": {"name": c.function.name, "arguments": c.function.arguments}}
                    for c in calls
                ],
            }
        )
        for c in calls:
            result = execute(c.function.name, c.function.arguments)
            messages.append({"role": "tool", "tool_call_id": c.id, "content": json.dumps(result)})
    return "Sorry, I couldn't finish this one. A teammate will follow up by email."
