"""Shared HTTP utilities and rate limiting for tools."""
from __future__ import annotations

import functools
import os
import threading
import time
from typing import Any, Callable

import requests


DEFAULT_UA = os.getenv(
    "STRAW_HATS_USER_AGENT",
    "straw-hats-treasure-agent/0.1 (+https://github.com/anomalyco)",
)


def http_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(
        {
            "User-Agent": DEFAULT_UA,
            "Accept": "*/*",
            "Accept-Language": "en-US,en;q=0.9",
        }
    )
    return s


class RateLimiter:
    """Simple monotonic rate limiter: minimum interval between calls."""

    def __init__(self, min_interval_s: float):
        self.min_interval = min_interval_s
        self._lock = threading.Lock()
        self._last = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            delta = now - self._last
            if delta < self.min_interval:
                time.sleep(self.min_interval - delta)
            self._last = time.monotonic()


def ok(data: Any) -> dict:
    return {"ok": True, "data": data, "error": None}


def err(message: str) -> dict:
    return {"ok": False, "data": None, "error": message}


def truncate(text: str, limit: int = 8000) -> str:
    if text is None:
        return ""
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n\n[truncated {len(text) - limit} chars]"


def cached_tool(ttl_seconds: float | None = 7 * 24 * 3600) -> Callable:
    """Decorator: cache a tool function's ``{ok,data,error}`` result on success.

    Applied *under* ``@tool`` so the LangChain wrapper sees the cached callable.
    Honors runtime toggles in ``straw_hats.memory`` (cache_enabled, memory_enabled).
    Only caches when the result is a dict with ``ok=True`` — failures retry.
    """

    def deco(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            # Import lazily to avoid a tools <-> memory import cycle at import time.
            try:
                from .. import memory
            except Exception:
                return fn(*args, **kwargs)

            if not (memory.memory_enabled() and memory.cache_enabled()):
                return fn(*args, **kwargs)

            # Fold positional args under their param names for stable keys.
            try:
                import inspect

                sig = inspect.signature(fn)
                bound = sig.bind_partial(*args, **kwargs)
                bound.apply_defaults()
                call_args = dict(bound.arguments)
            except Exception:
                call_args = {"_args": list(args), **kwargs}

            cache = memory.get_cache()
            key = cache.make_key(fn.__name__, call_args)
            hit = cache.get(key)
            if hit is not None:
                return hit
            result = fn(*args, **kwargs)
            try:
                if isinstance(result, dict) and result.get("ok") is True:
                    cache.set(key, fn.__name__, call_args, result, ttl_seconds)
            except Exception:
                pass
            return result

        wrapper.__wrapped__ = fn  # type: ignore[attr-defined]
        return wrapper

    return deco
