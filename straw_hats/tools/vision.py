"""Image analysis via Claude vision (through TokenRouter)."""
from __future__ import annotations

import base64
import mimetypes

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from ._common import err, http_session, ok, truncate


def _download(url: str) -> tuple[bytes, str]:
    s = http_session()
    r = s.get(url, timeout=25, stream=True)
    r.raise_for_status()
    ctype = r.headers.get("Content-Type", "").split(";")[0].strip()
    if not ctype:
        ctype = mimetypes.guess_type(url)[0] or "image/jpeg"
    data = r.content
    if len(data) > 8 * 1024 * 1024:
        raise RuntimeError("image too large (>8MB)")
    return data, ctype


@tool
def analyze_image(image_url: str, question: str = "Describe this image in detail. List any text, signs, landmarks, street markings, or distinctive features visible.") -> dict:
    """Analyze an image at a URL with Claude vision. Use for photos/maps/screenshots
    posted in Reddit threads or linked elsewhere. Returns a detailed description.

    Args:
        image_url: URL of the image.
        question: what to ask about the image (default: full description).
    """
    if not image_url.lower().startswith(("http://", "https://")):
        return err("image_url must be http(s)")

    try:
        raw, ctype = _download(image_url)
    except Exception as e:  # noqa: BLE001
        return err(f"image download failed: {e}")

    # Lazy import to avoid hard dependency during tool listing
    try:
        from ..llm import get_vision_llm
    except Exception as e:  # noqa: BLE001
        return err(f"vision llm unavailable: {e}")

    b64 = base64.b64encode(raw).decode("ascii")
    data_url = f"data:{ctype};base64,{b64}"

    try:
        llm = get_vision_llm()
        msg = HumanMessage(
            content=[
                {"type": "text", "text": question},
                {"type": "image_url", "image_url": {"url": data_url}},
            ]
        )
        resp = llm.invoke([msg])
        text = resp.content if isinstance(resp.content, str) else str(resp.content)
        return ok({"image_url": image_url, "description": truncate(text, 4000)})
    except Exception as e:  # noqa: BLE001
        return err(f"vision call failed: {e}")
