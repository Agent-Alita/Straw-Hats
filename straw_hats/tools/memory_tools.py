"""Tools that let the agent read/write long-term semantic memory."""
from __future__ import annotations

from langchain_core.tools import tool

from .. import memory
from ._common import err, ok


@tool
def recall(query: str, k: int = 5) -> dict:
    """Search prior hunts' remembered insights for facts matching ``query``.
    Returns up to ``k`` notes with their tags and creation time. Call this
    EARLY in a hunt with key phrases from the poem (e.g. "north beach plaque",
    "Coit Tower stairs") to reuse durable insights from previous runs.

    Args:
        query: free-form keyword search (matches fact text or tags).
        k: max results (1-20).
    """
    if not memory.memory_enabled():
        return ok({"query": query, "results": [], "note": "memory disabled"})
    try:
        k = max(1, min(20, int(k)))
    except Exception:
        k = 5
    try:
        results = memory.get_store().search(query or "", k=k)
    except Exception as e:  # noqa: BLE001
        return err(f"recall failed: {e}")
    return ok({"query": query, "results": results})


@tool
def remember(fact: str, tags: list[str] | None = None) -> dict:
    """Persist a durable, generalizable insight for future hunts. Use sparingly
    (cap ~5 per run): only for verified, reusable knowledge such as
    "In SF treasure poems, 'the cup that doesn't spill' refers to Vaillancourt Fountain."
    Do NOT use for run-specific scratch notes.

    Args:
        fact: a single-sentence insight worth keeping across sessions.
        tags: optional list of short keywords for retrieval (e.g. ["north-beach","plaque"]).
    """
    if not memory.memory_enabled():
        return ok({"stored": False, "reason": "memory disabled"})
    sid = memory.current_session()
    if sid is not None:
        n = memory.remember_count(sid)
        if n >= memory.REMEMBER_CAP:
            return ok({
                "stored": False,
                "reason": f"remember cap reached ({memory.REMEMBER_CAP} per run)",
            })
    if not fact or not fact.strip():
        return err("fact is empty")
    try:
        fact_id = memory.get_store().add(fact, tags or [], session_id_=sid)
    except Exception as e:  # noqa: BLE001
        return err(f"remember failed: {e}")
    if sid is not None:
        memory.bump_remember_count(sid)
    return ok({"stored": True, "id": fact_id, "fact": fact.strip(), "tags": tags or []})
