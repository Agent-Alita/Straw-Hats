"""LangGraph ReAct agent assembly and run loop."""
from __future__ import annotations

import json
import re
from contextlib import nullcontext
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.prebuilt import create_react_agent

from . import memory as _memory
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


def build_agent(checkpointer=None, max_tokens: int = 8192):
    llm = get_llm(temperature=0.2, max_tokens=max_tokens)
    tools = all_tools()
    return create_react_agent(
        llm,
        tools,
        prompt=SystemMessage(content=SYSTEM_PROMPT),
        checkpointer=checkpointer,
    )


def run_agent(
    poem: str,
    reddit_url: str,
    max_turns: int = 30,
    verbose: bool = True,
    console=None,
    session_id: str | None = None,
    resume: bool = True,
    use_memory: bool = True,
) -> dict[str, Any]:
    """Run the treasure-hunt agent and return {final_answer, raw_text, messages,
    session_id, resumed}.

    Args:
        poem: poem text.
        reddit_url: full reddit thread URL.
        max_turns: caps recursion_limit (= max(6, max_turns*2)).
        verbose: stream tool calls + assistant text via ``console``.
        console: optional rich.Console for streaming output.
        session_id: override the auto-derived (poem+url) session id.
        resume: if True and a prior checkpoint exists for the session, continue
            from it; otherwise start a fresh hunt under the same thread id.
        use_memory: if False, disable ALL memory layers (checkpointer, cache,
            store, archive) for this run. Reproduces pre-memory behavior.
    """
    _memory.set_memory_enabled(use_memory)

    sid = session_id or _memory.session_id(poem, reddit_url)

    if console is None and verbose:
        try:
            from rich.console import Console

            console = Console()
        except Exception:
            console = None

    # Decide whether we'll resume an existing thread.
    resumed = False
    if use_memory and resume:
        try:
            resumed = _memory.checkpoint_exists(sid)
        except Exception:
            resumed = False

    if console:
        if resumed:
            console.print(f"[yellow]Resuming session[/yellow] [bold]{sid}[/bold] (prior checkpoint found)")
        else:
            console.print(f"[dim]Session id:[/dim] [bold]{sid}[/bold]")

    # Record this hunt in the archive (idempotent).
    if use_memory:
        try:
            _memory.get_archive().upsert_start(sid, poem, reddit_url)
        except Exception as e:
            if console:
                console.print(f"[red]archive upsert_start failed: {e}[/red]")

    # Set the current session contextvar so tools (remember/recall) can tag rows.
    session_token = _memory.set_current_session(sid)

    user_msg = HumanMessage(content=INITIAL_USER_TEMPLATE.format(poem=poem, reddit_url=reddit_url))
    config: dict[str, Any] = {"recursion_limit": max(6, max_turns * 2)}
    if use_memory:
        config["configurable"] = {"thread_id": sid}

    # On resume, pass None to continue from the saved checkpoint; otherwise
    # supply the initial state.
    stream_input: dict[str, Any] | None = None if resumed else {"messages": [user_msg]}

    final_text = ""
    all_messages: list = []

    cp_ctx = _memory.get_checkpointer() if use_memory else nullcontext(None)

    try:
        with cp_ctx as checkpointer:
            agent = build_agent(checkpointer=checkpointer)
            try:
                for event in agent.stream(stream_input, config=config, stream_mode="updates"):
                    for node_name, node_state in event.items():
                        msgs = node_state.get("messages", []) if isinstance(node_state, dict) else []
                        for m in msgs:
                            all_messages.append(m)
                            if isinstance(m, AIMessage):
                                tool_calls = getattr(m, "tool_calls", None) or []
                                if tool_calls and console:
                                    for tc in tool_calls:
                                        name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", "?")
                                        args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", {})
                                        console.print(
                                            f"[bold cyan]→ tool[/bold cyan] [bold]{name}[/bold] "
                                            f"{_truncate_for_display(json.dumps(args, default=str), 240)}"
                                        )
                                text = m.content if isinstance(m.content, str) else None
                                if text and text.strip():
                                    final_text = text
                                    if console:
                                        console.print(
                                            f"[dim]assistant:[/dim] {_truncate_for_display(text, 800)}"
                                        )
                            elif isinstance(m, ToolMessage):
                                if console:
                                    content = (
                                        m.content
                                        if isinstance(m.content, str)
                                        else json.dumps(m.content, default=str)
                                    )
                                    console.print(
                                        f"[green]← {m.name}[/green] {_truncate_for_display(content, 400)}"
                                    )
            except Exception as e:
                if console:
                    console.print(f"[red]agent stream error: {e}[/red]")
                raise
    finally:
        _memory.reset_current_session(session_token)

    parsed = _parse_final(final_text)
    return {
        "final_answer": parsed,
        "raw_text": final_text,
        "messages": all_messages,
        "session_id": sid,
        "resumed": resumed,
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
