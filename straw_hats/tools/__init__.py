"""Tool registry — exports the list of LangChain tools passed to the agent."""
from __future__ import annotations

from .web_search import web_search
from .http_fetch import fetch_url
from .reddit import reddit_thread
from .wikipedia import wikipedia_lookup
from .maps import geocode, reverse_geocode, nearby_search
# from .vision import analyze_image  # disabled


def all_tools() -> list:
    return [
        web_search,
        fetch_url,
        reddit_thread,
        wikipedia_lookup,
        geocode,
        reverse_geocode,
        nearby_search,
        # analyze_image,  # disabled
    ]


__all__ = [
    "web_search",
    "fetch_url",
    "reddit_thread",
    "wikipedia_lookup",
    "geocode",
    "reverse_geocode",
    "nearby_search",
    # "analyze_image",  # disabled
    "all_tools",
]
