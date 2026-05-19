# straw-hats — San Francisco Treasure Hunting Agent

A LangGraph ReAct agent that reasons over a poem of clues and a Reddit discussion
thread to pinpoint a precise treasure location in San Francisco. Powered by
**Claude Opus 4.7** via the **TokenRouter** API.

## Capabilities

The agent has a tool belt for the kinds of research a human treasure hunter would do:

| Tool | Purpose |
|---|---|
| `web_search` | General web search (Tavily) for clues, news, historical refs |
| `fetch_url` | Open any URL and read its readable text content |
| `reddit_thread` | Crawl the discussion thread (OP + top comments + outbound links) |
| `wikipedia_lookup` | Verify SF historical facts (parks, ships, fires, landmarks) |
| `geocode` / `reverse_geocode` | Forward / reverse geocoding via OpenStreetMap (Nominatim), SF-biased |
| `nearby_search` | Find specific POIs (benches, plaques, statues, fountains, murals) near a point |
| `analyze_image` *(disabled)* | Vision tool, currently commented out in the registry |

The agent loop is a `langgraph.prebuilt.create_react_agent` so Claude picks tools
turn-by-turn until it emits a final structured answer.

## Output

For every run you get:

1. A streamed trace of tool calls and intermediate reasoning to stdout.
2. A structured `FinalAnswer` JSON:
   ```json
   {
     "location_name": "Buena Vista Park — east overlook bench",
     "address": "Buena Vista Ave E, San Francisco, CA",
     "lat": 37.7686,
     "lng": -122.4416,
     "confidence": 0.72,
     "reasoning": "...",
     "clue_mapping": { "line 1": "interpretation", "...": "..." },
     "candidates_considered": [ { "name": "...", "lat": 0, "lng": 0, "clues_matched": ["..."], "notes": "..." } ],
     "sources": [ "https://..." ]
   }
   ```
3. A human-readable markdown report.

## Install

This is a [uv](https://docs.astral.sh/uv/) project.

```bash
cd /home/neleac/Straw-Hats
uv sync                 # creates .venv and installs all deps from pyproject.toml + uv.lock
cp .env.example .env    # then edit .env: set TOKENROUTER_API_KEY and TAVILY_API_KEY
```

## Usage

```bash
# Poem from a file (preferred):
uv run straw-hats \
  --poem ./examples/poem.txt \
  --reddit "https://www.reddit.com/r/sanfrancisco/comments/XXXXXX/treasure_hunt/" \
  --out ./out/report.md \
  --json ./out/answer.json

# Or inline:
uv run straw-hats \
  --poem-text "Where the city's oldest eyes still watch the bay..." \
  --reddit "https://www.reddit.com/r/sanfrancisco/comments/XXXXXX/" \
  --max-turns 30

# Module form also works:
uv run python -m straw_hats.cli --poem examples/poem.txt --reddit "https://..."
```

CLI flags:

| Flag | Meaning |
|---|---|
| `--poem PATH` | Read poem from a UTF-8 text file |
| `--poem-text "..."` | Pass the poem inline |
| `--reddit URL` | Reddit thread URL (required) |
| `--out FILE` | Write markdown report (optional) |
| `--json FILE` | Write structured JSON answer (optional) |
| `--max-turns N` | Cap on agent tool-call turns (default 30) |
| `--quiet` | Suppress streaming trace |

## Configuration (`.env`)

```dotenv
TOKENROUTER_API_KEY=sk-...
TOKENROUTER_BASE_URL=https://api.tokenrouter.ai/v1
TOKENROUTER_MODEL=tokenrouter/anthropic/claude-opus-4.7
TAVILY_API_KEY=tvly-...
STRAW_HATS_USER_AGENT=straw-hats-treasure-agent/0.1 (contact: you@example.com)
```

If TokenRouter's base URL or shape differs from the OpenAI-compatible default,
override `TOKENROUTER_BASE_URL`. The client uses `langchain-openai`'s `ChatOpenAI`,
which speaks any OpenAI-shaped `/v1/chat/completions` endpoint.

## Architecture

```
straw_hats/
├── cli.py        # Typer CLI; loads .env, runs agent, renders report
├── agent.py      # LangGraph ReAct loop + streaming + final-answer parser
├── llm.py        # TokenRouter-backed ChatOpenAI factory
├── prompts.py    # System prompt + initial user message template
├── schemas.py    # Pydantic models (FinalAnswer, Candidate, ToolResult)
└── tools/
    ├── web_search.py    # Tavily
    ├── http_fetch.py    # requests + trafilatura + BeautifulSoup
    ├── reddit.py        # anonymous reddit.json crawler
    ├── wikipedia.py     # Wikipedia API + REST summary fallback
    ├── maps.py          # Nominatim geocode / reverse / POI (SF-biased, 1 req/sec)
    └── vision.py        # Claude vision via TokenRouter (disabled in registry)
```

## Notes & Limits

- **Nominatim** is rate-limited to 1 req/sec by usage policy. A token-bucket limiter
  enforces this and respects 429s with exponential backoff.
- **Reddit** anonymous JSON endpoint occasionally 429s; we retry with a 2 s minimum
  interval and a desktop User-Agent.
- **Agent turns** are capped (`--max-turns`, default 30) so cost stays bounded.
- The SF bounding box used for geocoding bias and final validation is
  `(37.70, -122.55) → (37.83, -122.35)`.
