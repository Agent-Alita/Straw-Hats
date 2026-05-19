"""Pytest fixtures — isolate memory state in a per-test temp dir."""
from __future__ import annotations

import importlib
import os

import pytest


@pytest.fixture(autouse=True)
def _isolated_memory(tmp_path, monkeypatch):
    """Redirect memory dir to a per-test tmp dir and reset singletons."""
    monkeypatch.setenv("STRAW_HATS_MEMORY_DIR", str(tmp_path / ".straw_hats"))

    # Reset singletons / module-level state in straw_hats.memory.
    from straw_hats import memory

    importlib.reload(memory)
    yield
