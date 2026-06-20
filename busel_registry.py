"""
🛸 busel — Plug-in Extension Points
Sovereign type-based registry for swappable components (attention, optimizer,
activation, patcher). When a new paper drops a new attention mechanism or
optimizer, register it with a single decorator and it becomes available
throughout the project without code changes to call sites.

Example:
    from busel_registry import register, get, list_registered

    @register("attention", "gdn2")
    class BulbaGDN2SeRoPEBlock(nn.Module):
        ...

    cls = get("attention", "gdn2")  # → <class 'BulbaGDN2SeRoPEBlock'>
    list_registered("attention")    # → ['gdn2', 'mla']
"""
from __future__ import annotations

import threading
from typing import Any, Callable, TypeVar

T = TypeVar("T")

_REGISTRY: dict[str, dict[str, type]] = {}
_LOCK = threading.Lock()


def register(kind: str, name: str, *, override: bool = False) -> Callable[[type[T]], type[T]]:
    """Class decorator that registers a class under (kind, name).

    Args:
        kind: Category of component ("attention", "optimizer", "activation",
            "patcher", "loss", "router", ...).
        name: Human-readable unique identifier within the kind.
        override: If True, allow replacing an already-registered class. Default
            False → raises KeyError on collision (catches accidental re-imports).

    Returns:
        The class itself, unmodified (so it stays usable as a normal class).

    Raises:
        KeyError: If (kind, name) already registered and override=False.
    """
    if not kind or not isinstance(kind, str):
        raise ValueError(f"register(): kind must be a non-empty string, got {kind!r}")
    if not name or not isinstance(name, str):
        raise ValueError(f"register(): name must be a non-empty string, got {name!r}")

    def decorator(cls: type[T]) -> type[T]:
        with _LOCK:
            bucket = _REGISTRY.setdefault(kind, {})
            if name in bucket and not override:
                existing = bucket[name]
                if existing is not cls:
                    raise KeyError(
                        f"Registry collision: {kind!r}.{name!r} already registered "
                        f"to {existing.__module__}.{existing.__qualname__} (refusing to "
                        f"overwrite with {cls.__module__}.{cls.__qualname__}; "
                        f"pass override=True to allow)"
                    )
            bucket[name] = cls
        try:
            cls.__busel_registry__ = {"kind": kind, "name": name}
        except (AttributeError, TypeError):
            pass
        return cls

    return decorator


def get(kind: str, name: str) -> type:
    """Retrieve a class registered under (kind, name).

    Raises:
        KeyError: If kind or name is unknown; message includes the available
            names for that kind to make typos obvious.
    """
    with _LOCK:
        bucket = _REGISTRY.get(kind, {})
        if name not in bucket:
            available = sorted(bucket.keys())
            raise KeyError(
                f"Unknown registry entry: {kind!r}.{name!r}. "
                f"Available {kind!r} entries: {available or '(none registered)'}"
            )
        return bucket[name]


def list_registered(kind: str) -> list[str]:
    """Return sorted list of registered names for a given kind.

    Returns an empty list if the kind is unknown (instead of raising) so callers
    can safely check whether a category has any implementations yet.
    """
    with _LOCK:
        return sorted(_REGISTRY.get(kind, {}).keys())


def is_registered(kind: str, name: str) -> bool:
    """Boolean check for (kind, name) without raising on misses."""
    with _LOCK:
        return name in _REGISTRY.get(kind, {})


def unregister(kind: str, name: str) -> None:
    """Remove a (kind, name) entry. Mostly useful in tests. No-op if absent."""
    with _LOCK:
        bucket = _REGISTRY.get(kind, {})
        bucket.pop(name, None)
        if not bucket:
            _REGISTRY.pop(kind, None)


def clear_registry() -> None:
    """Wipe the entire registry. Only intended for test isolation."""
    with _LOCK:
        _REGISTRY.clear()
