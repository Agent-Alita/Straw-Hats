"""LangGraph ReAct agent assembly and run loop."""
from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.prebuilt import create_react_agent

from .llm import get_llm
from .prompts import INITIAL_USER_TEMPLATE, SYSTEM_PROMPT
from .schemas import FinalAnswer
from .tools import all_tools


_JSON_BLOCK = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)
_BARE_JSON = re.compile(r"(\{[\s\S]*\})")


def _truncate_for_display(text: str, n: int = 600) -> str:
    text = text or ""
    if len(text) <= n:
        return text
    return text[:n] + f" …[+{len(text) - n}]"


def build_agent(max_tokens: int = 8192):
    llm = get_llm(temperature=0.2, max_tokens=max_tokens)
    tools = all_tools()
    return create_react_agent(llm, tools, prompt=SystemMessage(content=SYSTEM_PROMPT))


def run_agent(
    poem: str,
    reddit_url: str,
    max_turns: int = 30,
    verbose: bool = True,
    console=None,
) -> dict[str, Any]:
    """Run the treasure-hunt agent and return {final_answer, raw_text, messages}."""
    agent = build_agent()
    user_msg = HumanMessage(content=INITIAL_USER_TEMPLATE.format(poem=poem, reddit_url=reddit_url))

    state: dict[str, Any] = {"messages": [user_msg]}
    config = {"recursion_limit": max(6, max_turns * 2)}

    if console is None and verbose:
        try:
            from rich.console import Console

            console = Console()
        except Exception:
            console = None

    final_text = ""
    all_messages: list = []

    # Stream events for visibility
    try:
        for event in agent.stream(state, config=config, stream_mode="updates"):
            for node_name, node_state in event.items():
                msgs = node_state.get("messages", []) if isinstance(node_state, dict) else []
                for m in msgs:
                    all_messages.append(m)
                    if isinstance(m, AIMessage):
                        # Tool calls
                        tool_calls = getattr(m, "tool_calls", None) or []
                        if tool_calls and console:
                            for tc in tool_calls:
                                name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", "?")
                                args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", {})
                                console.print(f"[bold cyan]→ tool[/bold cyan] [bold]{name}[/bold] {_truncate_for_display(json.dumps(args, default=str), 240)}")
                        # Text content
                        text = m.content if isinstance(m.content, str) else None
                        if text and text.strip():
                            final_text = text
                            if console:
                                console.print(f"[dim]assistant:[/dim] {_truncate_for_display(text, 800)}")
                    elif isinstance(m, ToolMessage):
                        if console:
                            content = m.content if isinstance(m.content, str) else json.dumps(m.content, default=str)
                            console.print(f"[green]← {m.name}[/green] {_truncate_for_display(content, 400)}")
    except Exception as e:
        if console:
            console.print(f"[red]agent stream error: {e}[/red]")
        raise

    parsed = _parse_final(final_text)
    return {
        "final_answer": parsed,
        "raw_text": final_text,
        "messages": all_messages,
    }


def _parse_final(text: str) -> FinalAnswer | None:
    if not text:
        return None
    candidates: list[str] = []
    m = _JSON_BLOCK.search(text)
    if m:
        candidates.append(m.group(1))
    # Fallback: first { ... } substring
    if not candidates:
        m2 = _BARE_JSON.search(text)
        if m2:
            candidates.append(m2.group(1))

    for raw in candidates:
        try:
            obj = json.loads(raw)
            return FinalAnswer.model_validate(obj)
        except Exception:
            # try a relaxed cleanup: strip trailing commas
            try:
                cleaned = re.sub(r",(\s*[}\]])", r"\1", raw)
                obj = json.loads(cleaned)
                return FinalAnswer.model_validate(obj)
            except Exception:
                continue
    return None
