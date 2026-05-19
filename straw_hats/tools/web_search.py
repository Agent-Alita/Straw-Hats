"""Web search via Tavily."""
from __future__ import annotations

import os

from langchain_core.tools import tool

from ._common import err, ok


@tool
def web_search(query: str, max_results: int = 5) -> dict:
    """Search the web with Tavily for general research (history, landmarks, news,
    treasure-hunt clues). Returns a list of {title, url, snippet} results.

    Args:
        query: search query string.
        max_results: how many results to return (1-10).
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return err("TAVILY_API_KEY not set; web_search unavailable")

    try:
        from tavily import TavilyClient
    except ImportError:
        return err("tavily-python is not installed")

    try:
        client = TavilyClient(api_key=api_key)
        resp = client.search(
            query=query,
            max_results=max(1, min(10, int(max_results))),
            search_depth="advanced",
            include_answer=False,
        )
        results = [
            {
                "title": r.get("title"),
                "url": r.get("url"),
                "snippet": r.get("content"),
                "score": r.get("score"),
            }
            for r in (resp.get("results") or [])
        ]
        return ok({"query": query, "results": results})
    except Exception as e:  # noqa: BLE001
        return err(f"tavily error: {e}")
