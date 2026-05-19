"""CLI entrypoint for straw-hats."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

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


@app.command()
def hunt(
    poem: Optional[Path] = typer.Option(None, "--poem", help="Path to poem text file."),
    poem_text: Optional[str] = typer.Option(None, "--poem-text", help="Poem text inline."),
    reddit: str = typer.Option(..., "--reddit", help="Reddit thread URL."),
    out: Optional[Path] = typer.Option(None, "--out", help="Write markdown report to this file."),
    json_out: Optional[Path] = typer.Option(None, "--json", help="Write structured JSON answer to this file."),
    max_turns: int = typer.Option(30, "--max-turns", help="Max agent loop turns."),
    verbose: bool = typer.Option(True, "--verbose/--quiet", help="Stream tool calls and thoughts."),
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

    console.print(Panel.fit(f"[bold]straw-hats[/bold] hunting in San Francisco\nReddit: {reddit}\nPoem chars: {len(poem_str)}", border_style="cyan"))

    from .agent import run_agent

    result = run_agent(
        poem=poem_str,
        reddit_url=reddit,
        max_turns=max_turns,
        verbose=verbose,
        console=console,
    )

    answer = result["final_answer"]
    raw = result["raw_text"]

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

    if answer is None:
        console.print("[yellow]Warning: could not parse a structured final answer.[/yellow]")
        sys.exit(3)


def main():
    app()


if __name__ == "__main__":
    main()
