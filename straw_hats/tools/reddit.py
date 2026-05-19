"""Reddit thread fetcher via anonymous .json endpoints."""
from __future__ import annotations

import re
from urllib.parse import urlparse

from langchain_core.tools import tool
from tenacity import retry, stop_after_attempt, wait_exponential

from ._common import RateLimiter, err, http_session, ok, truncate


_LIMITER = RateLimiter(min_interval_s=2.0)


def _normalize(url_or_id: str) -> str:
    s = url_or_id.strip()
    # bare comment id like "abc123" -> assume full url required
    if s.startswith("http://") or s.startswith("https://"):
        # rewrite old.reddit / np.reddit / www.reddit to public json
        p = urlparse(s)
        host = "www.reddit.com"
        path = p.path
        # strip trailing slash
        if path.endswith("/"):
            path = path[:-1]
        if not path.endswith(".json"):
            path = path + ".json"
        q = p.query
        url = f"https://{host}{path}"
        if q:
            url += f"?{q}"
        return url
    raise ValueError("reddit_thread expects a full reddit URL")


_URL_RE = re.compile(r"(https?://[^\s)\]]+)")


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10), reraise=True)
def _fetch_json(url: str) -> list | dict:
    _LIMITER.wait()
    s = http_session()
    r = s.get(url, timeout=25)
    if r.status_code == 429:
        raise RuntimeError("reddit rate limited (429)")
    r.raise_for_status()
    return r.json()


def _extract_links(text: str) -> list[str]:
    if not text:
        return []
    return list(dict.fromkeys(_URL_RE.findall(text)))


def _walk_comments(children: list, out: list, max_items: int) -> None:
    for c in children:
        if len(out) >= max_items:
            return
        kind = c.get("kind")
        data = c.get("data", {})
        if kind == "t1":
            body = data.get("body") or ""
            out.append(
                {
                    "author": data.get("author"),
                    "score": data.get("score", 0),
                    "body": truncate(body, 1200),
                    "links": _extract_links(body),
                    "permalink": "https://www.reddit.com" + (data.get("permalink") or ""),
                }
            )
            replies = data.get("replies")
            if isinstance(replies, dict):
                _walk_comments(replies.get("data", {}).get("children", []), out, max_items)


@tool
def reddit_thread(url: str, max_comments: int = 40) -> dict:
    """Fetch a Reddit thread anonymously. Returns the OP (title + body), top comments
    sorted by score, and all outbound links found in the discussion.

    Args:
        url: full reddit thread URL (https://www.reddit.com/r/.../comments/...).
        max_comments: maximum comments to return (sorted by score, top-first).
    """
    try:
        json_url = _normalize(url)
    except Exception as e:  # noqa: BLE001
        return err(str(e))

    try:
        payload = _fetch_json(json_url)
    except Exception as e:  # noqa: BLE001
        return err(f"reddit fetch failed: {e}")

    if not isinstance(payload, list) or len(payload) < 2:
        return err("unexpected reddit response shape")

    try:
        post = payload[0]["data"]["children"][0]["data"]
        post_body = post.get("selftext") or ""
        op = {
            "title": post.get("title"),
            "author": post.get("author"),
            "score": post.get("score", 0),
            "subreddit": post.get("subreddit"),
            "url": "https://www.reddit.com" + (post.get("permalink") or ""),
            "body": truncate(post_body, 4000),
            "links_in_post": _extract_links(post_body) + ([post.get("url_overridden_by_dest")] if post.get("url_overridden_by_dest") else []),
            "created_utc": post.get("created_utc"),
            "num_comments": post.get("num_comments"),
        }
    except Exception as e:  # noqa: BLE001
        return err(f"parse op failed: {e}")

    comments: list[dict] = []
    try:
        children = payload[1]["data"]["children"]
        _walk_comments(children, comments, max_items=max(max_comments, 5) * 2)
    except Exception:
        comments = []

    comments.sort(key=lambda c: c.get("score", 0), reverse=True)
    comments = comments[:max_comments]

    all_links: list[str] = list(dict.fromkeys(
        [u for u in op["links_in_post"] if u]
        + [u for c in comments for u in c.get("links", [])]
    ))

    return ok(
        {
            "op": op,
            "comments": comments,
            "outbound_links": all_links[:60],
        }
    )
