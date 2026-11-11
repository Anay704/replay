"""Turning arbitrary Python values into JSON for traces."""

from __future__ import annotations

import dataclasses
from typing import NamedTuple

from pydantic import BaseModel, JsonValue
from pydantic_core import PydanticSerializationError, to_jsonable_python

from replay.schema import ErrorInfo


class Serialized(NamedTuple):
    value: JsonValue
    type_tag: str
    """"json", "pydantic:<module>.<Class>", "dataclass:<module>.<Class>" or "repr"."""
    replayable: bool


def _qualname(cls: type) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"


def to_jsonable(value: object) -> Serialized:
    """Convert a value to JSON, remembering its type so replay can rebuild it.

    Values that cannot be represented as JSON are stored as their repr and marked not replayable.
    """
    try:
        data: JsonValue = to_jsonable_python(value)
    except (PydanticSerializationError, TypeError, ValueError):
        return Serialized(repr(value), "repr", False)
    if isinstance(value, BaseModel):
        return Serialized(data, f"pydantic:{_qualname(type(value))}", True)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return Serialized(data, f"dataclass:{_qualname(type(value))}", True)
    return Serialized(data, "json", True)


def error_info(exc: BaseException) -> ErrorInfo:
    args = to_jsonable(list(exc.args))
    return ErrorInfo(
        type=_qualname(type(exc)),
        message=str(exc),
        args=args.value if args.replayable and isinstance(args.value, list) else [],
    )
