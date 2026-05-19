"""Unit tests for the cross-session memory layers."""
from __future__ import annotations

import time

from straw_hats import memory


# --------------------------- session_id ---------------------------


def test_session_id_deterministic():
    a = memory.session_id("poem text", "https://reddit.com/foo")
    b = memory.session_id("poem text", "https://reddit.com/foo")
    assert a == b
    assert len(a) == 12


def test_session_id_normalizes_whitespace_and_case():
    a = memory.session_id("Hello World", "https://reddit.com/X")
    b = memory.session_id("  hello   world  ", "HTTPS://REDDIT.COM/X")
    assert a == b


def test_session_id_differs_on_real_change():
    a = memory.session_id("p1", "u1")
    b = memory.session_id("p2", "u1")
    c = memory.session_id("p1", "u2")
    assert len({a, b, c}) == 3


# --------------------------- tool cache ---------------------------


def test_tool_cache_roundtrip():
    c = memory.get_cache()
    key = c.make_key("web_search", {"query": "Coit Tower"})
    assert c.get(key) is None
    c.set(key, "web_search", {"query": "Coit Tower"}, {"ok": True, "data": 1}, ttl_seconds=60)
    hit = c.get(key)
    assert hit == {"ok": True, "data": 1}


def test_tool_cache_normalizes_args():
    c = memory.get_cache()
    k1 = c.make_key("web_search", {"query": "Coit Tower"})
    k2 = c.make_key("web_search", {"query": "  coit   tower "})
    assert k1 == k2


def test_tool_cache_ttl_expiry():
    c = memory.get_cache()
    key = c.make_key("t", {"x": 1})
    c.set(key, "t", {"x": 1}, {"ok": True, "data": "v"}, ttl_seconds=0.01)
    time.sleep(0.05)
    assert c.get(key) is None


def test_tool_cache_purge_expired():
    c = memory.get_cache()
    c.set("k1", "t", {}, {"ok": True}, ttl_seconds=0.01)
    c.set("k2", "t", {}, {"ok": True}, ttl_seconds=3600)
    time.sleep(0.05)
    n = c.purge_expired()
    assert n == 1
    assert c.get("k2") is not None


# --------------------------- fact store ---------------------------


def test_fact_store_add_and_search():
    s = memory.get_store()
    fid = s.add("Coit Tower commemorates volunteer firefighters", tags=["coit", "telegraph-hill"])
    assert fid > 0
    hits = s.search("coit tower", k=5)
    assert any(h["id"] == fid for h in hits)


def test_fact_store_tag_match():
    s = memory.get_store()
    s.add("Vaillancourt Fountain is at Embarcadero Plaza", tags=["fountain", "embarcadero"])
    hits = s.search("embarcadero", k=5)
    assert len(hits) >= 1


def test_fact_store_list_recent_order():
    s = memory.get_store()
    s.add("older")
    time.sleep(0.01)
    s.add("newer")
    rows = s.list_recent(limit=10)
    assert rows[0]["fact"] == "newer"


def test_remember_cap():
    sid = "abc123"
    for _ in range(memory.REMEMBER_CAP):
        memory.bump_remember_count(sid)
    assert memory.remember_count(sid) == memory.REMEMBER_CAP


# --------------------------- hunt archive ---------------------------


def test_hunt_archive_lifecycle():
    arch = memory.get_archive()
    sid = "deadbeef0001"
    arch.upsert_start(sid, "poem body", "https://r.example/x")

    # upsert is idempotent
    arch.upsert_start(sid, "poem body", "https://r.example/x")

    row = arch.get(sid)
    assert row is not None
    assert row["completed_at"] is None
    assert row["location_name"] is None

    class _Fake:
        location_name = "Coit Tower"
        confidence = 0.92

        def model_dump(self):
            return {"location_name": "Coit Tower", "confidence": 0.92}

    arch.complete(sid, _Fake(), "# Report\n\nbody")
    row = arch.get(sid)
    assert row["location_name"] == "Coit Tower"
    assert row["confidence"] == 0.92
    assert row["report_md"].startswith("# Report")

    rows = arch.list_recent(limit=10)
    assert any(r["session_id"] == sid for r in rows)

    found = arch.find_by_url("https://r.example/x")
    assert any(r["session_id"] == sid for r in found)

    assert arch.delete(sid) == 1
    assert arch.get(sid) is None


def test_archive_complete_with_no_answer():
    arch = memory.get_archive()
    sid = "nullansr0001"
    arch.upsert_start(sid, "p", "u")
    arch.complete(sid, None, "# Raw\n\nno parse")
    row = arch.get(sid)
    assert row["location_name"] is None
    assert row["answer_json"] is None
    assert "no parse" in row["report_md"]


def test_archive_resume_does_not_clobber_prior_answer():
    """A no-op resume (answer=None) must NOT overwrite an existing good row."""
    arch = memory.get_archive()
    sid = "resume000001"
    arch.upsert_start(sid, "p", "u")

    class _Good:
        location_name = "Coit Tower"
        confidence = 0.84

        def model_dump(self):
            return {"location_name": self.location_name, "confidence": self.confidence}

    arch.complete(sid, _Good(), "# Real report")
    first_completed = arch.get(sid)["completed_at"]

    time.sleep(0.01)
    # Simulate a resume that produced nothing new.
    arch.complete(sid, None, "# Treasure Hunt Report\n\n(no structured answer parsed)")

    row = arch.get(sid)
    assert row["location_name"] == "Coit Tower"
    assert row["confidence"] == 0.84
    assert row["report_md"] == "# Real report"
    assert row["completed_at"] >= first_completed


# --------------------------- toggles ---------------------------


def test_memory_disabled_short_circuits_cache_decorator():
    """When memory is disabled, cached_tool calls the inner function every time."""
    from straw_hats.tools._common import cached_tool

    calls = {"n": 0}

    @cached_tool(ttl_seconds=60)
    def inner(query: str) -> dict:
        calls["n"] += 1
        return {"ok": True, "data": calls["n"]}

    memory.set_memory_enabled(False)
    try:
        inner(query="x")
        inner(query="x")
        assert calls["n"] == 2  # not cached
    finally:
        memory.set_memory_enabled(True)


def test_cached_tool_caches_on_success():
    from straw_hats.tools._common import cached_tool

    calls = {"n": 0}

    @cached_tool(ttl_seconds=60)
    def inner(query: str) -> dict:
        calls["n"] += 1
        return {"ok": True, "data": calls["n"]}

    inner(query="hello")
    inner(query="HELLO")  # normalized -> same key
    inner(query=" hello ")
    assert calls["n"] == 1


def test_cached_tool_does_not_cache_failures():
    from straw_hats.tools._common import cached_tool

    calls = {"n": 0}

    @cached_tool(ttl_seconds=60)
    def inner(query: str) -> dict:
        calls["n"] += 1
        return {"ok": False, "data": None, "error": "boom"}

    inner(query="z")
    inner(query="z")
    assert calls["n"] == 2
