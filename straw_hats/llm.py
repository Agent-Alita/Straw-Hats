"""TokenRouter-backed Claude Opus 4.7 client.

TokenRouter exposes an OpenAI-compatible API. We use langchain-openai's ChatOpenAI
pointed at the TokenRouter base URL. The model id is passed through verbatim.
"""
from __future__ import annotations

import os

from langchain_openai import ChatOpenAI


def get_llm(temperature: float = 0.2, max_tokens: int = 8192) -> ChatOpenAI:
    api_key = os.getenv("TOKENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "TOKENROUTER_API_KEY is not set. Copy .env.example to .env and fill it in."
        )
    base_url = os.getenv("TOKENROUTER_BASE_URL", "https://api.tokenrouter.ai/v1")
    model = os.getenv("TOKENROUTER_MODEL", "tokenrouter/anthropic/claude-opus-4.7")

    return ChatOpenAI(
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=120,
        max_retries=2,
    )


def get_vision_llm() -> ChatOpenAI:
    """Same client; OpenAI-style multimodal content blocks are used for images."""
    return get_llm(temperature=0.1, max_tokens=2048)
