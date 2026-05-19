"""Wikipedia search + page summary."""
from __future__ import annotations

from langchain_core.tools import tool

from ._common import DEFAULT_UA, cached_tool, err, http_session, ok, truncate


def _search(query: str, limit: int = 5) -> list[dict]:
    s = http_session()
    r = s.get(
        "https://en.wikipedia.org/w/api.php",
        params={
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json",
            "srlimit": limit,
        },
        timeout=20,
    )
    r.raise_for_status()
    hits = r.json().get("query", {}).get("search", []) or []
    return [{"title": h.get("title"), "snippet": h.get("snippet"), "pageid": h.get("pageid")} for h in hits]


def _page_extract(title: str) -> dict | None:
    try:
        import wikipediaapi

        wiki = wikipediaapi.Wikipedia(user_agent=DEFAULT_UA, language="en")
        page = wiki.page(title)
        if not page.exists():
            return None
        return {
            "title": page.title,
            "url": page.fullurl,
            "summary": truncate(page.summary, 4000),
            "text": truncate(page.text, 6000),
        }
    except Exception:
        # fallback to REST summary
        s = http_session()
        r = s.get(
            f"https://en.wikipedia.org/api/rest_v1/page/summary/{title.replace(' ', '_')}",
            timeout=20,
        )
        if r.status_code != 200:
            return None
        j = r.json()
        return {
            "title": j.get("title"),
            "url": (j.get("content_urls") or {}).get("desktop", {}).get("page"),
            "summary": truncate(j.get("extract", ""), 4000),
            "text": "",
        }


@tool
@cached_tool(ttl_seconds=7 * 24 * 3600)  # 7 days
def wikipedia_lookup(query: str, results: int = 3) -> dict:
    """Search Wikipedia and return the top page summaries. Use for verifying SF
    historical facts: landmarks, fires, ships, neighborhoods, parks, monuments.

    Args:
        query: search query, e.g. "Buena Vista Park San Francisco".
        results: number of top pages to fetch summaries for (1-5).
    """
    try:
        hits = _search(query, limit=max(1, min(5, int(results))))
    except Exception as e:  # noqa: BLE001
        return err(f"wikipedia search failed: {e}")

    pages = []
    for h in hits[: results]:
        ext = _page_extract(h["title"])
        if ext:
            pages.append(ext)

    return ok({"query": query, "hits": hits, "pages": pages})
