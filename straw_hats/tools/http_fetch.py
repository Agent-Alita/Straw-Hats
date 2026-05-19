"""HTTP fetch + readability extraction."""
from __future__ import annotations

from langchain_core.tools import tool
from tenacity import retry, stop_after_attempt, wait_exponential

from ._common import cached_tool, err, http_session, ok, truncate


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8), reraise=True)
def _get(url: str, timeout: int = 25) -> str:
    s = http_session()
    r = s.get(url, timeout=timeout, allow_redirects=True)
    r.raise_for_status()
    return r.text


def _extract(html: str, url: str) -> tuple[str, str | None]:
    # Try trafilatura first for high-quality readable extraction
    try:
        import trafilatura

        text = trafilatura.extract(
            html, url=url, include_links=True, include_images=False, favor_recall=True
        )
        if text and len(text.strip()) > 80:
            md = trafilatura.extract_metadata(html)
            title = md.title if md and md.title else None
            return text, title
    except Exception:
        pass

    # Fallback: BeautifulSoup textification
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript", "nav", "footer", "header"]):
            tag.decompose()
        title = soup.title.string.strip() if soup.title and soup.title.string else None
        text = "\n".join(line.strip() for line in soup.get_text("\n").splitlines() if line.strip())
        return text, title
    except Exception as e:  # noqa: BLE001
        return f"[extraction failed: {e}]", None


@tool
@cached_tool(ttl_seconds=7 * 24 * 3600)  # 7 days
def fetch_url(url: str) -> dict:
    """Fetch a web page and return its readable text content + title. Use for
    articles, blog posts, news, image-host pages, etc. Returns truncated text.

    Args:
        url: absolute URL to fetch (http/https).
    """
    if not url or not url.lower().startswith(("http://", "https://")):
        return err("url must start with http:// or https://")
    try:
        html = _get(url)
    except Exception as e:  # noqa: BLE001
        return err(f"fetch failed: {e}")
    text, title = _extract(html, url)
    return ok(
        {
            "url": url,
            "title": title,
            "text": truncate(text, 8000),
            "length": len(text or ""),
        }
    )
