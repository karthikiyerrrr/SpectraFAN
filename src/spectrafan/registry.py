"""Minimal name-keyed registry for pluggable builders.

Adding a model architecture or skip-transform variant becomes a decorator on a
new module rather than an edit to the training engine.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self, name: str) -> None:
        self._name = name
        self._entries: dict[str, Callable[..., T]] = {}

    def register(self, key: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
        def deco(fn: Callable[..., T]) -> Callable[..., T]:
            if key in self._entries:
                raise ValueError(f"{self._name} key already registered: {key!r}")
            self._entries[key] = fn
            return fn

        return deco

    def build(self, key: str, *args: object, **kwargs: object) -> T:
        if key not in self._entries:
            raise ValueError(f"unknown {self._name}: {key!r} (registered: {sorted(self._entries)})")
        return self._entries[key](*args, **kwargs)

    def keys(self) -> list[str]:
        return sorted(self._entries)
