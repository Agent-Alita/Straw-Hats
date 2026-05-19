"""Cross-session memory for straw-hats.

Provides four memory layers, all backed by SQLite under ``./.straw_hats/``:

1. Conversation checkpointer (LangGraph ``SqliteSaver``) — resume in-flight hunts.
2. Tool-call result cache — keyed by (tool_name, normalized_args), TTL per tool.
3. Long-term semantic facts (lightweight keyword Store) — distilled notes across hunts.
4. Per-hunt artifact archive — final JSON + markdown report indexed by session_id.

The checkpointer lives in its own file (``checkpoints.sqlite``) so it owns its
schema; the other three share ``memory.sqlite``.

A session_id is ``sha256(poem + reddit_url)[:12]`` (auto), or user-supplied.
"""
from __future__ import annotations

import contextvars
import hashlib
import json
import os
import re
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_DEFAULT_DIR = ".straw_hats"


def memory_dir() -> Path:
    """Return the memory directory (creating it if missing).

    Overridable via ``STRAW_HATS_MEMORY_DIR`` env var (useful for tests).
    """
    base = os.getenv("STRAW_HATS_MEMORY_DIR", _DEFAULT_DIR)
    p = Path(base)
    p.mkdir(parents=True, exist_ok=True)
    return p


def checkpoint_db_path() -> Path:
    return memory_dir() / "checkpoints.sqlite"


def memory_db_path() -> Path:
    return memory_dir() / "memory.sqlite"


# ---------------------------------------------------------------------------
# Session identity
# ---------------------------------------------------------------------------

_WS_RE = re.compile(r"\s+")


def _norm(s: str) -> str:
    return _WS_RE.sub(" ", (s or "").strip().lower())


def session_id(poem: str, reddit_url: str) -> str:
    """Deterministic 12-char id derived from normalized poem + reddit URL."""
    h = hashlib.sha256()
    h.update(_norm(poem).encode("utf-8"))
    h.update(b"\x00")
    h.update(_norm(reddit_url).encode("utf-8"))
    return h.hexdigest()[:12]


# Active session id (set by run_agent so tools can tag stored facts).
_current_session: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "straw_hats_current_session", default=None
)


def set_current_session(sid: str | None) -> contextvars.Token:
    return _current_session.set(sid)


def reset_current_session(token: contextvars.Token) -> None:
    _current_session.reset(token)


def current_session() -> str | None:
    return _current_session.get()


# ---------------------------------------------------------------------------
# Checkpointer (LangGraph thread persistence)
# ---------------------------------------------------------------------------


@contextmanager
def get_checkpointer() -> Iterator[Any]:
    """Yield a ready-to-use ``SqliteSaver`` bound to the project checkpoint DB.

    Usage::

        with get_checkpointer() as cp:
            agent = build_agent(checkpointer=cp)
            ...
    """
    from langgraph.checkpoint.sqlite import SqliteSaver

    path = str(checkpoint_db_path())
    with SqliteSaver.from_conn_string(path) as saver:
        saver.setup()
        yield saver


def checkpoint_exists(session_id_: str) -> bool:
    """Return True if there is at least one persisted checkpoint for the thread."""
    with get_checkpointer() as cp:
        try:
            tup = cp.get_tuple({"configurable": {"thread_id": session_id_}})
        except Exception:
            return False
        return tup is not None


def delete_checkpoints(session_id_: str) -> int:
    """Delete all checkpoints for a thread. Returns 1 if any existed prior, else 0."""
    with get_checkpointer() as cp:
        try:
            existed = (
                cp.get_tuple({"configurable": {"thread_id": session_id_}}) is not None
            )
        except Exception:
            existed = False
        try:
            cp.delete_thread(session_id_)
        except Exception:
            return 0
    return 1 if existed else 0


# ---------------------------------------------------------------------------
# Shared sqlite for cache / store / archive
# ---------------------------------------------------------------------------

_DB_LOCK = threading.Lock()
_DB_INITIALIZED = False


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(memory_db_path(), check_same_thread=False, timeout=15.0)
    conn.row_factory = sqlite3.Row
    # WAL improves concurrency between read/write
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def _init_db() -> None:
    global _DB_INITIALIZED
    if _DB_INITIALIZED:
        return
    with _DB_LOCK:
        if _DB_INITIALIZED:
            return
        with _connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS schema_version (
                    component TEXT PRIMARY KEY,
                    version   INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS tool_cache (
                    key        TEXT PRIMARY KEY,
                    tool       TEXT NOT NULL,
                    args_json  TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    expires_at REAL
                );
                CREATE INDEX IF NOT EXISTS idx_tool_cache_tool ON tool_cache(tool);
                CREATE TABLE IF NOT EXISTS store (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    fact       TEXT NOT NULL,
                    tags       TEXT NOT NULL DEFAULT '',
                    session_id TEXT,
                    created_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_store_session ON store(session_id);
                CREATE TABLE IF NOT EXISTS hunts (
                    session_id    TEXT PRIMARY KEY,
                    poem          TEXT NOT NULL,
                    reddit_url    TEXT NOT NULL,
                    created_at    REAL NOT NULL,
                    completed_at  REAL,
                    location_name TEXT,
                    confidence    REAL,
                    answer_json   TEXT,
                    report_md     TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_hunts_reddit ON hunts(reddit_url);
                CREATE INDEX IF NOT EXISTS idx_hunts_created ON hunts(created_at);
                """
            )
            conn.execute(
                "INSERT OR IGNORE INTO schema_version(component, version) VALUES (?, ?)",
                ("memory", 1),
            )
        _DB_INITIALIZED = True


# ---------------------------------------------------------------------------
# Tool-call result cache
# ---------------------------------------------------------------------------


class ToolCache:
    """SQLite-backed cache of tool results keyed by (tool_name, normalized args)."""

    @staticmethod
    def _normalize_args(args: dict) -> dict:
        """Lowercase + strip string args so trivial variants share a cache slot."""
        out: dict = {}
        for k, v in sorted(args.items()):
            if isinstance(v, str):
                out[k] = _norm(v)
            else:
                out[k] = v
        return out

    @classmethod
    def make_key(cls, tool: str, args: dict) -> str:
        norm = cls._normalize_args(args or {})
        payload = json.dumps({"t": tool, "a": norm}, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(self, key: str) -> dict | None:
        _init_db()
        now = time.time()
        with _connect() as conn:
            row = conn.execute(
                "SELECT result_json, expires_at FROM tool_cache WHERE key = ?",
                (key,),
            ).fetchone()
        if not row:
            return None
        if row["expires_at"] is not None and row["expires_at"] < now:
            self.delete(key)
            return None
        try:
            return json.loads(row["result_json"])
        except Exception:
            return None

    def set(
        self,
        key: str,
        tool: str,
        args: dict,
        result: dict,
        ttl_seconds: float | None,
    ) -> None:
        _init_db()
        now = time.time()
        expires = (now + ttl_seconds) if ttl_seconds else None
        try:
            payload = json.dumps(result, default=str)
        except Exception:
            return
        with _connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO tool_cache
                    (key, tool, args_json, result_json, created_at, expires_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    key,
                    tool,
                    json.dumps(args, default=str, sort_keys=True),
                    payload,
                    now,
                    expires,
                ),
            )

    def delete(self, key: str) -> None:
        _init_db()
        with _connect() as conn:
            conn.execute("DELETE FROM tool_cache WHERE key = ?", (key,))

    def purge_expired(self) -> int:
        _init_db()
        now = time.time()
        with _connect() as conn:
            cur = conn.execute(
                "DELETE FROM tool_cache WHERE expires_at IS NOT NULL AND expires_at < ?",
                (now,),
            )
            return cur.rowcount

    def stats(self) -> dict:
        _init_db()
        with _connect() as conn:
            total = conn.execute("SELECT COUNT(*) AS n FROM tool_cache").fetchone()["n"]
            by_tool = conn.execute(
                "SELECT tool, COUNT(*) AS n FROM tool_cache GROUP BY tool ORDER BY n DESC"
            ).fetchall()
        return {"total": total, "by_tool": [dict(r) for r in by_tool]}


_CACHE_SINGLETON: ToolCache | None = None
_CACHE_ENABLED = True


def get_cache() -> ToolCache:
    global _CACHE_SINGLETON
    if _CACHE_SINGLETON is None:
        _CACHE_SINGLETON = ToolCache()
    return _CACHE_SINGLETON


def set_cache_enabled(enabled: bool) -> None:
    """Global runtime toggle (used by --no-cache)."""
    global _CACHE_ENABLED
    _CACHE_ENABLED = bool(enabled)


def cache_enabled() -> bool:
    return _CACHE_ENABLED


# ---------------------------------------------------------------------------
# Long-term fact store
# ---------------------------------------------------------------------------


class FactStore:
    """Tiny keyword-matched store of remembered facts."""

    def add(self, fact: str, tags: list[str] | None = None, session_id_: str | None = None) -> int:
        _init_db()
        if not fact or not fact.strip():
            return 0
        tags_s = ",".join(t.strip().lower() for t in (tags or []) if t and t.strip())
        with _connect() as conn:
            cur = conn.execute(
                "INSERT INTO store (fact, tags, session_id, created_at) VALUES (?, ?, ?, ?)",
                (fact.strip(), tags_s, session_id_, time.time()),
            )
            return cur.lastrowid or 0

    def search(self, query: str, k: int = 5) -> list[dict]:
        _init_db()
        q = _norm(query)
        with _connect() as conn:
            if not q:
                rows = conn.execute(
                    "SELECT id, fact, tags, session_id, created_at FROM store "
                    "ORDER BY created_at DESC LIMIT ?",
                    (k,),
                ).fetchall()
            else:
                # split into terms; AND across terms via LIKE on fact OR tags
                terms = [t for t in q.split() if t]
                if not terms:
                    return []
                where = " AND ".join(["(LOWER(fact) LIKE ? OR tags LIKE ?)"] * len(terms))
                params: list = []
                for t in terms:
                    like = f"%{t}%"
                    params.extend([like, like])
                params.append(k)
                rows = conn.execute(
                    f"SELECT id, fact, tags, session_id, created_at FROM store "
                    f"WHERE {where} ORDER BY created_at DESC LIMIT ?",
                    params,
                ).fetchall()
        return [dict(r) for r in rows]

    def list_recent(self, limit: int = 50) -> list[dict]:
        _init_db()
        with _connect() as conn:
            rows = conn.execute(
                "SELECT id, fact, tags, session_id, created_at FROM store "
                "ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def delete(self, fact_id: int) -> int:
        _init_db()
        with _connect() as conn:
            cur = conn.execute("DELETE FROM store WHERE id = ?", (fact_id,))
            return cur.rowcount


_STORE_SINGLETON: FactStore | None = None


def get_store() -> FactStore:
    global _STORE_SINGLETON
    if _STORE_SINGLETON is None:
        _STORE_SINGLETON = FactStore()
    return _STORE_SINGLETON


# Per-session 'remember' call counter (cap ~5 facts/run to discourage spam)
_REMEMBER_COUNTS: dict[str, int] = {}
_REMEMBER_LOCK = threading.Lock()
REMEMBER_CAP = 5


def remember_count(session_id_: str) -> int:
    with _REMEMBER_LOCK:
        return _REMEMBER_COUNTS.get(session_id_, 0)


def bump_remember_count(session_id_: str) -> int:
    with _REMEMBER_LOCK:
        n = _REMEMBER_COUNTS.get(session_id_, 0) + 1
        _REMEMBER_COUNTS[session_id_] = n
        return n


# ---------------------------------------------------------------------------
# Hunt archive
# ---------------------------------------------------------------------------


class HuntArchive:
    """Final-answer + report archive, one row per session_id."""

    def upsert_start(self, session_id_: str, poem: str, reddit_url: str) -> None:
        _init_db()
        now = time.time()
        with _connect() as conn:
            conn.execute(
                """
                INSERT INTO hunts (session_id, poem, reddit_url, created_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(session_id) DO NOTHING
                """,
                (session_id_, poem, reddit_url, now),
            )

    def complete(
        self,
        session_id_: str,
        answer: Any | None,
        report_md: str,
    ) -> None:
        """Persist the final answer + report.

        If ``answer`` is None AND a prior good answer already exists for this
        session, the existing row is left intact (only ``completed_at`` is
        refreshed). This prevents a no-op resume from clobbering a real result.
        """
        _init_db()
        now = time.time()
        location_name = None
        confidence = None
        answer_json = None
        if answer is not None:
            try:
                answer_json = json.dumps(answer.model_dump(), default=str)
                location_name = getattr(answer, "location_name", None)
                confidence = getattr(answer, "confidence", None)
            except Exception:
                answer_json = None

        with _connect() as conn:
            existing = conn.execute(
                "SELECT answer_json FROM hunts WHERE session_id = ?", (session_id_,)
            ).fetchone()
            had_prior_answer = bool(existing and existing["answer_json"])

            if answer is None and had_prior_answer:
                # Refresh completion timestamp only; preserve prior good result.
                conn.execute(
                    "UPDATE hunts SET completed_at = ? WHERE session_id = ?",
                    (now, session_id_),
                )
                return

            conn.execute(
                """
                UPDATE hunts SET
                    completed_at  = ?,
                    location_name = ?,
                    confidence    = ?,
                    answer_json   = ?,
                    report_md     = ?
                WHERE session_id = ?
                """,
                (now, location_name, confidence, answer_json, report_md, session_id_),
            )

    def get(self, session_id_: str) -> dict | None:
        _init_db()
        with _connect() as conn:
            row = conn.execute(
                "SELECT * FROM hunts WHERE session_id = ?", (session_id_,)
            ).fetchone()
        return dict(row) if row else None

    def list_recent(self, limit: int = 20) -> list[dict]:
        _init_db()
        with _connect() as conn:
            rows = conn.execute(
                "SELECT session_id, reddit_url, created_at, completed_at, "
                "location_name, confidence FROM hunts "
                "ORDER BY COALESCE(completed_at, created_at) DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def find_by_url(self, reddit_url: str) -> list[dict]:
        _init_db()
        with _connect() as conn:
            rows = conn.execute(
                "SELECT session_id, reddit_url, created_at, completed_at, "
                "location_name, confidence FROM hunts WHERE reddit_url = ? "
                "ORDER BY created_at DESC",
                (reddit_url,),
            ).fetchall()
        return [dict(r) for r in rows]

    def delete(self, session_id_: str) -> int:
        _init_db()
        with _connect() as conn:
            cur = conn.execute("DELETE FROM hunts WHERE session_id = ?", (session_id_,))
            return cur.rowcount


_ARCHIVE_SINGLETON: HuntArchive | None = None


def get_archive() -> HuntArchive:
    global _ARCHIVE_SINGLETON
    if _ARCHIVE_SINGLETON is None:
        _ARCHIVE_SINGLETON = HuntArchive()
    return _ARCHIVE_SINGLETON


# ---------------------------------------------------------------------------
# Global memory enable flag (for --no-memory)
# ---------------------------------------------------------------------------

_MEMORY_ENABLED = True


def set_memory_enabled(enabled: bool) -> None:
    global _MEMORY_ENABLED
    _MEMORY_ENABLED = bool(enabled)


def memory_enabled() -> bool:
    return _MEMORY_ENABLED
