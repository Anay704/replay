"""Exceptions Replay raises into agent code during a replay."""

from __future__ import annotations


class ReplayBlockedToolError(Exception):
    """A tool call was blocked during replay: nothing recorded matches it and it may not run live.

    It is an ordinary Exception, so the agent handles it like any failed tool call.
    """


class ReplayAborted(BaseException):
    """Stops a replayed run immediately (on_blocked: abort, or a runaway loop).

    Derives from BaseException so an agent's `except Exception` cannot swallow it.
    """


class ReplayedToolError(Exception):
    """A recorded tool error whose original exception class could not be rebuilt."""

    def __init__(self, original_type: str, message: str) -> None:
        super().__init__(f"{original_type}: {message}")
        self.original_type = original_type


class ReplayNetworkBlocked(ConnectionRefusedError):
    """An outbound connection to a host that isn't allowed during replay."""


class ReplayConfigError(ValueError):
    """The replay configuration is invalid or its entry point can't be loaded."""
