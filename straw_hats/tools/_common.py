"""Shared HTTP utilities and rate limiting for tools."""
from __future__ import annotations

import os
import threading
import time
from typing import Any

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
