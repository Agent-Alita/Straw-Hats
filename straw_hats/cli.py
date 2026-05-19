"""CLI entrypoint for straw-hats."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

app = typer.Typer(add_completion=False, help="straw-hats — San Francisco treasure hunting agent.")


def _render_report(answer, raw_text: str) -> str:
    if answer is None:
        return "# Treasure Hunt Report\n\n(no structured answer parsed)\n\n## Raw model output\n\n" + (raw_text or "")
    lines = [
        "# Treasure Hunt Report",
        "",
        "## Predicted Location",
        f"**{answer.location_name}**",
    ]
    if answer.address:
        lines.append(f"Address: {answer.address}")
    if answer.lat is not None and answer.lng is not None:
        lines.append(f"Coordinates: `{answer.lat:.5f}, {answer.lng:.5f}`")
        lines.append(
            f"Map: https://www.openstreetmap.org/?mlat={answer.lat}&mlon={answer.lng}#map=18/{answer.lat}/{answer.lng}"
        )
    lines += [
        f"Confidence: **{answer.confidence:.2f}**",
        "",
        "## Reasoning",
        answer.reasoning or "(no reasoning)",
        "",
        "## Clue Mapping",
    ]
    if answer.clue_mapping:
        lines.append("| Clue | Interpretation |")
        lines.append("|---|---|")
        for k, v in answer.clue_mapping.items():
            lines.append(f"| {k} | {v} |")
    else:
        lines.append("(none)")

    lines += ["", "## Candidates Considered"]
    if answer.candidates_considered:
        for i, c in enumerate(answer.candidates_considered, 1):
            coord = f" ({c.lat:.5f}, {c.lng:.5f})" if c.lat is not None and c.lng is not None else ""
            lines.append(f"{i}. **{c.name}**{coord}")
            if c.address:
                lines.append(f"   - Address: {c.address}")
            if c.clues_matched:
                lines.append(f"   - Matches: {', '.join(c.clues_matched)}")
            if c.notes:
                lines.append(f"   - Notes: {c.notes}")
    else:
        lines.append("(none)")

    lines += ["", "## Sources"]
    if answer.sources:
        for s in answer.sources:
            lines.append(f"- {s}")
    else:
        lines.append("(none)")

    return "\n".join(lines)


def _fmt_ts(ts: float | None) -> str:
    if not ts:
        return "-"
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


@app.command()
def hunt(
    poem: Optional[Path] = typer.Option(None, "--poem", help="Path to poem text file."),
    poem_text: Optional[str] = typer.Option(None, "--poem-text", help="Poem text inline."),
    reddit: str = typer.Option(..., "--reddit", help="Reddit thread URL."),
    out: Optional[Path] = typer.Option(None, "--out", help="Write markdown report to this file."),
    json_out: Optional[Path] = typer.Option(None, "--json", help="Write structured JSON answer to this file."),
    max_turns: int = typer.Option(30, "--max-turns", help="Max agent loop turns."),
    verbose: bool = typer.Option(True, "--verbose/--quiet", help="Stream tool calls and thoughts."),
    session: Optional[str] = typer.Option(None, "--session", help="Override the auto-derived session id."),
    no_resume: bool = typer.Option(False, "--no-resume", help="Start a fresh run even if a prior checkpoint exists."),
    no_cache: bool = typer.Option(False, "--no-cache", help="Bypass the tool-call result cache for this run."),
    no_memory: bool = typer.Option(False, "--no-memory", help="Disable ALL memory layers (checkpointer, cache, store, archive)."),
):
    """Hunt for the SF treasure given a poem and a Reddit discussion thread."""
    load_dotenv()
    console = Console()

    if poem and poem_text:
        console.print("[red]Provide either --poem or --poem-text, not both.[/red]")
        raise typer.Exit(2)
    if not poem and not poem_text:
        console.print("[red]Must provide --poem <file> or --poem-text \"...\".[/red]")
        raise typer.Exit(2)

    poem_str = poem_text if poem_text else poem.read_text(encoding="utf-8")
    if not poem_str.strip():
        console.print("[red]Poem is empty.[/red]")
        raise typer.Exit(2)

    # Apply memory toggles before anything else imports/uses them.
    from . import memory as _memory

    _memory.set_memory_enabled(not no_memory)
    _memory.set_cache_enabled(not no_cache)

    console.print(
        Panel.fit(
            f"[bold]straw-hats[/bold] hunting in San Francisco\n"
            f"Reddit: {reddit}\nPoem chars: {len(poem_str)}\n"
            f"Memory: {'OFF' if no_memory else 'ON'}  Cache: {'OFF' if (no_memory or no_cache) else 'ON'}",
            border_style="cyan",
        )
    )

    from .agent import run_agent

    result = run_agent(
        poem=poem_str,
        reddit_url=reddit,
        max_turns=max_turns,
        verbose=verbose,
        console=console,
        session_id=session,
        resume=not no_resume,
        use_memory=not no_memory,
    )

    answer = result["final_answer"]
    raw = result["raw_text"]
    sid = result["session_id"]

    report = _render_report(answer, raw)
    console.print()
    console.print(Markdown(report))

    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report, encoding="utf-8")
        console.print(f"\n[green]Wrote report:[/green] {out}")

    if json_out:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        payload = answer.model_dump() if answer else {"error": "no parsed answer", "raw_text": raw}
        json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        console.print(f"[green]Wrote JSON:[/green] {json_out}")

    # Archive the final result (best-effort, even when answer parse failed).
    if not no_memory:
        try:
            _memory.get_archive().complete(sid, answer, report)
            console.print(f"[dim]Archived hunt session:[/dim] [bold]{sid}[/bold]")
        except Exception as e:
            console.print(f"[red]archive complete failed: {e}[/red]")

    if answer is None:
        console.print("[yellow]Warning: could not parse a structured final answer.[/yellow]")
        sys.exit(3)


@app.command()
def history(limit: int = typer.Option(20, "--limit", "-n", help="Max rows to show.")):
    """List recent hunts from the archive."""
    load_dotenv()
    console = Console()
    from . import memory as _memory

    rows = _memory.get_archive().list_recent(limit=limit)
    if not rows:
        console.print("[dim]No hunts archived yet.[/dim]")
        return

    table = Table(title=f"Recent hunts (last {len(rows)})", show_lines=False)
    table.add_column("session", style="bold")
    table.add_column("finished")
    table.add_column("location")
    table.add_column("conf", justify="right")
    table.add_column("reddit")
    for r in rows:
        conf = "-" if r["confidence"] is None else f"{r['confidence']:.2f}"
        table.add_row(
            r["session_id"],
            _fmt_ts(r["completed_at"] or r["created_at"]),
            (r["location_name"] or "(unparsed)")[:48],
            conf,
            (r["reddit_url"] or "")[:60],
        )
    console.print(table)


@app.command()
def show(session_id: str = typer.Argument(..., help="Session id from `straw-hats history`.")):
    """Print an archived hunt's stored report + JSON."""
    load_dotenv()
    console = Console()
    from . import memory as _memory

    row = _memory.get_archive().get(session_id)
    if not row:
        console.print(f"[red]No archived hunt with session_id={session_id}[/red]")
        raise typer.Exit(1)

    header = (
        f"session: {row['session_id']}\n"
        f"reddit:  {row['reddit_url']}\n"
        f"created: {_fmt_ts(row['created_at'])}\n"
        f"done:    {_fmt_ts(row['completed_at'])}\n"
        f"loc:     {row['location_name'] or '(unparsed)'}\n"
        f"conf:    {row['confidence'] if row['confidence'] is not None else '-'}"
    )
    console.print(Panel(header, title="hunt", border_style="cyan"))

    if row["report_md"]:
        console.print(Markdown(row["report_md"]))
    else:
        console.print("[dim](no stored report)[/dim]")

    if row["answer_json"]:
        try:
            obj = json.loads(row["answer_json"])
            console.print(Panel(json.dumps(obj, indent=2), title="answer.json", border_style="green"))
        except Exception:
            pass


@app.command()
def forget(
    session_id: str = typer.Argument(..., help="Session id to delete."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt."),
):
    """Delete the checkpoint + archive row for a session.

    The tool-call cache and remembered facts are NOT touched.
    """
    load_dotenv()
    console = Console()
    from . import memory as _memory

    row = _memory.get_archive().get(session_id)
    if not row and not _memory.checkpoint_exists(session_id):
        console.print(f"[yellow]No data found for session_id={session_id}[/yellow]")
        raise typer.Exit(1)

    if not yes:
        loc = (row or {}).get("location_name") or "(no archive)"
        confirm = typer.confirm(f"Delete checkpoint + archive for {session_id} ({loc})?")
        if not confirm:
            console.print("aborted.")
            raise typer.Exit(0)

    deleted_cp = _memory.delete_checkpoints(session_id)
    deleted_arch = _memory.get_archive().delete(session_id)
    console.print(
        f"[green]forgot[/green] {session_id} (checkpoint={deleted_cp}, archive_rows={deleted_arch})"
    )


@app.command()
def facts(
    query: Optional[str] = typer.Option(None, "--query", "-q", help="Search facts by keyword."),
    limit: int = typer.Option(20, "--limit", "-n", help="Max rows to show."),
    delete: Optional[int] = typer.Option(None, "--delete", help="Delete a fact by its id."),
):
    """List or search remembered long-term facts."""
    load_dotenv()
    console = Console()
    from . import memory as _memory

    store = _memory.get_store()
    if delete is not None:
        n = store.delete(int(delete))
        console.print(f"[green]deleted[/green] fact id={delete} (rows={n})")
        return

    rows = store.search(query, k=limit) if query else store.list_recent(limit=limit)
    if not rows:
        console.print("[dim]No facts stored.[/dim]")
        return

    table = Table(title=f"facts ({len(rows)})", show_lines=False)
    table.add_column("id", justify="right")
    table.add_column("when")
    table.add_column("tags")
    table.add_column("fact")
    for r in rows:
        table.add_row(
            str(r["id"]),
            _fmt_ts(r["created_at"]),
            r["tags"] or "",
            r["fact"],
        )
    console.print(table)


@app.command()
def cache(
    stats: bool = typer.Option(False, "--stats", help="Show cache stats."),
    purge_expired: bool = typer.Option(False, "--purge-expired", help="Delete expired entries."),
    clear_all: bool = typer.Option(False, "--clear-all", help="Delete ALL cache entries."),
):
    """Inspect or maintain the tool-call cache."""
    load_dotenv()
    console = Console()
    from . import memory as _memory

    c = _memory.get_cache()
    if clear_all:
        # rebuild table by deleting all rows
        import sqlite3 as _sql

        with _sql.connect(_memory.memory_db_path()) as conn:
            cur = conn.execute("DELETE FROM tool_cache")
            console.print(f"[green]cleared[/green] {cur.rowcount} cache entries")
        return
    if purge_expired:
        n = c.purge_expired()
        console.print(f"[green]purged[/green] {n} expired entries")
        return

    if stats or True:  # default to stats
        s = c.stats()
        console.print(f"[bold]Total cached:[/bold] {s['total']}")
        if s["by_tool"]:
            table = Table()
            table.add_column("tool")
            table.add_column("rows", justify="right")
            for r in s["by_tool"]:
                table.add_row(r["tool"], str(r["n"]))
            console.print(table)


def main():
    app()


if __name__ == "__main__":
    main()
