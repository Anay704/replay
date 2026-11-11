"""The replay configuration file (replay.yaml)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError, model_validator

from replay.errors import ReplayConfigError

MatchTier = Literal["exact", "normalized", "similar"]


class _Config(BaseModel):
    # Unknown keys are errors, so a typo in replay.yaml fails loudly instead of being ignored.
    model_config = ConfigDict(extra="forbid")


class Overrides(_Config):
    """Changes applied inside replay.wrap() to every LLM call, without touching agent code."""

    model: str | dict[str, str] | None = None
    """A model for every call, or a mapping {recorded model: replacement}."""
    params: dict[str, JsonValue] = Field(default_factory=dict)
    """Extra request parameters, e.g. {temperature: 0}."""
    system_prompt: str | None = None
    system_prompt_file: Path | None = None

    @model_validator(mode="after")
    def _one_prompt_source(self) -> Overrides:
        if self.system_prompt is not None and self.system_prompt_file is not None:
            raise ValueError("set system_prompt or system_prompt_file, not both")
        return self

    def prompt(self) -> str | None:
        if self.system_prompt_file is not None:
            return self.system_prompt_file.read_text(encoding="utf-8")
        return self.system_prompt

    def model_for(self, recorded: str) -> str:
        if isinstance(self.model, str):
            return self.model
        if isinstance(self.model, dict):
            return self.model.get(recorded, recorded)
        return recorded


class ToolsConfig(_Config):
    allow_live: list[str] = Field(default_factory=list)
    """Read-only tools that may run for real when no recorded call matches."""
    volatile_fields: dict[str, list[str]] = Field(default_factory=dict)
    """Argument fields ignored when matching, per tool name or "*" for all tools."""
    similarity_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    match: dict[str, MatchTier] = Field(default_factory=dict)
    """The loosest match tier allowed per tool (default: similar)."""
    on_blocked: Literal["raise", "abort"] = "raise"

    def volatile_for(self, tool: str) -> frozenset[str]:
        return frozenset(self.volatile_fields.get("*", []) + self.volatile_fields.get(tool, []))


class LimitsConfig(_Config):
    concurrency: int = Field(default=4, ge=1)
    requests_per_minute: float | None = Field(default=None, gt=0)
    tokens_per_minute: float | None = Field(default=None, gt=0)
    max_retries: int = Field(default=6, ge=0)
    max_llm_calls_per_run: int = Field(default=25, ge=1)


class NetworkConfig(_Config):
    guard: bool = True
    """Block outbound connections except to LLM hosts, the judge, and allow_hosts."""
    allow_hosts: list[str] = Field(default_factory=list)


class ReplayConfig(_Config):
    entrypoint: str | None = None
    """"package.module:function". Defaults to the entry point recorded in each trace."""
    pythonpath: list[Path] = Field(default_factory=list)
    """Directories added to sys.path before importing the entry point."""
    overrides: Overrides = Field(default_factory=Overrides)
    agent_config: dict[str, JsonValue] | None = None
    """Passed to the entry point as `config=` if it accepts that argument."""
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    limits: LimitsConfig = Field(default_factory=LimitsConfig)
    network: NetworkConfig = Field(default_factory=NetworkConfig)


def load_config(path: str | os.PathLike[str]) -> ReplayConfig:
    """Load a YAML config. Relative paths inside it resolve against the file's directory."""
    file = Path(path)
    try:
        data = yaml.safe_load(file.read_text(encoding="utf-8")) or {}
    except OSError as exc:
        raise ReplayConfigError(f"Can't read config {file}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ReplayConfigError(f"{file} is not valid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise ReplayConfigError(f"{file} must contain a YAML mapping")
    try:
        config = ReplayConfig.model_validate(data)
    except ValidationError as exc:
        raise ReplayConfigError(f"Invalid config {file}:\n{exc}") from exc
    return resolve_paths(config, file.parent)


def resolve_paths(config: ReplayConfig, base: Path) -> ReplayConfig:
    config.pythonpath = [p if p.is_absolute() else (base / p).resolve() for p in config.pythonpath]
    prompt_file = config.overrides.system_prompt_file
    if prompt_file is not None and not prompt_file.is_absolute():
        config.overrides.system_prompt_file = (base / prompt_file).resolve()
    return config
